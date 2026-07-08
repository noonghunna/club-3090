"""BYO Route-C apply-swap: download a curated-arch fine-tune's weights and emit
a serve-locally compose that CLONES the ``--profile-like`` sibling's real compose
(keeping its curated chat-template / parsers / MTP wiring) with ``--model``
re-pointed at the brought weights.

This is a DISTINCT action from the locked 6-stratum pull gate (``pull.run_pull``):
it never runs the stratum-5 eligibility hard-stop that (correctly) refuses to
*price* an un-fittable curated-hybrid arch. The fit is inherited from the sibling
compose, so there is nothing to price — the user's ``pull.sh --apply-swap`` (the
c3 [D] press on a Route-C fit-check) is the explicit opt-in.

Why clone the sibling's real compose and NOT ``generate_from_profile``: the
derived-vllm template drops the curated model's chat template, reasoning/tool
parsers, and MTP config — serving a Qwen3.6 fine-tune through it is broken. We
read the sibling's actual compose file and surgically override only ``--model``,
``--served-model-name``, the volume mount, and (per the brought checkpoint's MTP
head) ``--speculative-config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml  # available on the pull.sh path (generate_compose already parses YAML)

from scripts.lib.profiles import deriver as D
from scripts.lib.profiles import downloader as DL
from scripts.lib.profiles.einput import EInput

# The generated serve-locally compose is written ALONGSIDE the sibling compose
# (so the sibling's relative ../ mounts — caches, chat template — resolve) and
# gitignored via the `_brought-*.yml` pattern.
_BROUGHT_MODEL_MOUNT = "/brought-model"


def resolve_swap(slug: str, der: Any, root: Path) -> dict:
    """Route-C swap resolution — mirror of pull.sh ``_swap_path``'s Route-C
    branch: the uncurated arch maps to a curated hybrid/MoE sibling we serve.
    Returns ``{route, sibling_slug, quant_match, has_mtp_head}``. ``route`` is
    "C" only for a non-curated, non-generic-dense arch that resolves to a
    curated sibling; else "B" (GGUF fallback) or None."""
    from scripts.lib.profiles import pull as P  # _FAMILY_WEIGHT_FIELDS (RO)
    from scripts.lib import generate_compose as gc

    blank = {"route": None, "sibling_slug": None, "quant_match": None,
             "has_mtp_head": False}
    if der is None or getattr(der, "error", None) is not None:
        return blank
    if getattr(der, "tier1", None) is not None:
        return blank
    if bool(getattr(der, "generic_dense_eligible", False)):
        return blank

    arch = (der.profile or {}).get("arch")
    try:
        rt = gc._load_yaml(root, "scripts/lib/profiles/profile_runtime.yml")
        canon, row = gc.resolve_arch_from_config(rt, gc.load_arches(root), arch)
    except Exception:
        canon, row = None, None

    sibling, family = None, None
    if row is not None:
        family = row.get("family")
        slugs = (((rt.get("arch_model_xref") or {}).get(canon) or {})
                 .get("model_slugs") or [])
        if slugs:
            sibling = slugs[0]

    if sibling is not None and family in P._FAMILY_WEIGHT_FIELDS:
        quant_match = None
        try:
            from scripts.lib.profiles.compat import load_profiles
            model = load_profiles().models.get(sibling)
            if model is not None and getattr(model, "weights", None):
                first_slug = next(iter(model.weights))
                meta = model.weights[first_slug] or {}
                quant_match = meta.get("format") or first_slug
        except Exception:
            quant_match = None
        return {"route": "C", "sibling_slug": sibling, "quant_match": quant_match,
                "has_mtp_head": bool((der.profile or {}).get("has_mtp_head"))}

    return {"route": "B", "sibling_slug": None, "quant_match": "gguf",
            "has_mtp_head": False}


def _min_einput(slug: str, der: Any, hf_home: Path) -> EInput:
    """The minimal EInput ``downloader.download_model`` reads (slug / hf_home /
    der.profile._hf_api). Every other field is dummy — download_model never
    touches runtime / c2a / gpu topology."""
    return EInput(
        slug=slug, terminal="proceed", is_override_accepted=False, der=der,
        runtime={}, selected_files=[], hf_home=Path(hf_home), c2a=None,
        hardware_sm=0.0, visible_gpu_count=0, per_gpu_vram_mib=[],
        selected_gpu_indices=[], selected_gpu_vram_mib=[], topology_summary="",
        club3090_commit="", diagnostics={},
    )


def apply_command_overrides(
    cmd: list, *, model_path: str, served_name: str, drop_spec: bool,
) -> list:
    """Surgery on the vllm ``command`` list: re-point ``--model``, replace
    ``--served-model-name`` value(s) with the brought name, and drop
    ``--speculative-config`` + its value iff the brought checkpoint has no MTP
    head. ``--quantization`` is left UNTOUCHED — the Route-C precondition is that
    the brought weights match the sibling compose's quant.

    The served-name is emitted as ``${SERVED_NAME:-<brought>}`` so the ② Serve
    override editor can rename the endpoint at up-time (env interpolation) without
    a re-emit — same shape as the compose's ${MAX_MODEL_LEN}/${KV_CACHE_DTYPE}."""
    out: list = []
    i, n = 0, len(cmd)
    while i < n:
        tok = cmd[i]
        if tok == "--model":
            out += ["--model", model_path]
            i += 2
            continue
        if tok == "--served-model-name":
            out += ["--served-model-name", f"${{SERVED_NAME:-{served_name}}}"]
            i += 1
            while i < n and not str(cmd[i]).startswith("--"):
                i += 1  # drop the sibling's served-name value(s)
            continue
        if tok == "--speculative-config" and drop_spec:
            i += 2  # drop the flag AND its config value
            continue
        out.append(tok)
        i += 1
    return out


def _gate_spec_via_entrypoint(svc: dict) -> None:
    """Move ``--speculative-config <json>`` out of ``command`` and re-add it from
    an entrypoint gated on ``${SPEC:-on}`` (mirrors the shipped nvfp4 compose), so
    SPEC=off drops the MTP drafter at up-time.  No-op if the command carries no
    ``--speculative-config``.  Mutates ``svc`` in place."""
    cmd = list(svc.get("command") or [])
    spec_val = None
    out: list = []
    i, n = 0, len(cmd)
    while i < n:
        if cmd[i] == "--speculative-config":
            spec_val = cmd[i + 1] if i + 1 < n else None
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    if spec_val is None:
        return
    svc["command"] = out
    env_list = list(svc.get("environment") or [])
    if "SPEC" not in env_list:
        env_list.append("SPEC")
    svc["environment"] = env_list
    # `$$` = docker-compose-escaped `$` (evaluated by the container's bash at
    # runtime, NOT interpolated by compose).  The JSON spec value has no single
    # quotes, so single-quoting it for the bash array is safe.
    svc["entrypoint"] = [
        "bash", "-c",
        (
            "# SPEC=off drops the built-in MTP drafter (tight-RAM / no-spec path);\n"
            "# default on. Toggled by the c3 ② Serve override editor.\n"
            "SPEC_ARGS=()\n"
            'if [ "$${SPEC:-on}" != "off" ]; then\n'
            f"  SPEC_ARGS=(--speculative-config '{spec_val}')\n"
            "else\n"
            '  echo "[brought] SPEC=off — MTP drafter disabled (no spec-dec)" >&2\n'
            "fi\n"
            'exec vllm serve "$$@" "$${SPEC_ARGS[@]}"'
        ),
        "--",
    ]


def emit_swap_compose(
    root: Path, profile_like: str, weights_host_dir: Path, *,
    served_name: str, has_mtp_head: bool, brought_san: str,
) -> Path:
    """Clone the ``--profile-like`` sibling's compose, override --model → the
    mounted brought weights (+ served-name / spec-config), and write it
    ALONGSIDE the sibling compose (relative mounts resolve there). Returns the
    written path (``_brought-<san>.yml``, gitignored)."""
    from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY

    entry = COMPOSE_REGISTRY.get(profile_like)
    if entry is None:
        raise ValueError(
            f"--profile-like '{profile_like}' is not a COMPOSE_REGISTRY slug"
        )
    compose_rel = entry["compose_path"]
    compose_file = root / compose_rel
    doc = yaml.safe_load(compose_file.read_text(encoding="utf-8"))

    services = (doc or {}).get("services") or {}
    if not services:
        raise ValueError(f"sibling compose {compose_rel} has no services")
    svc_name, svc = next(iter(services.items()))

    svc["command"] = apply_command_overrides(
        list(svc.get("command") or []),
        model_path=_BROUGHT_MODEL_MOUNT, served_name=served_name,
        drop_spec=not has_mtp_head,
    )
    # SPEC on/off gating (only when the checkpoint HAS an MTP head — else there's
    # nothing to toggle): pull --speculative-config out of the command and gate it
    # behind ${SPEC:-on} via the SAME entrypoint pattern the nvfp4 compose ships,
    # so the ② Serve override editor can flip spec-decode at up-time (no re-emit).
    if has_mtp_head:
        _gate_spec_via_entrypoint(svc)
    vols = list(svc.get("volumes") or [])
    vols.append(f"{weights_host_dir}:{_BROUGHT_MODEL_MOUNT}:ro")
    svc["volumes"] = vols
    svc["container_name"] = f"vllm-brought-{brought_san}"

    out_path = compose_file.parent / f"_brought-{brought_san}.yml"
    header = (
        "# GENERATED serve-locally compose (BYO Route-C apply-swap) — a clone of\n"
        f"#   {compose_rel}\n"
        f"# with --model re-pointed at the brought weights ({_BROUGHT_MODEL_MOUNT}).\n"
        "# Do NOT hand-edit or commit; regenerate via: pull.sh <repo> "
        "--profile-like <slug> --apply-swap\n"
    )
    out_path.write_text(
        header + yaml.safe_dump(doc, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def apply_swap(
    root: Path, slug: str, profile_like: str, *, hf_home: Optional[str] = None,
    fetcher: Any = None, download_fn: Any = None, do_download: bool = True,
) -> dict:
    """Orchestrate: derive → resolve Route-C swap → download weights
    (SHA-verified, via ``download_model``) → emit the sibling-clone serve
    compose. Returns a result dict with ``ok`` + ``compose_path`` (the artifact
    c3 serves) or ``error``.

    ``do_download=False`` skips the fetch (tests / a pre-downloaded model) and
    points the mount at the (possibly not-yet-present) pull dir."""
    download_fn = download_fn or DL.download_model
    der = D.derive(slug, hf_home=hf_home)
    if der.error is not None:
        return {"ok": False, "error": f"derive: {der.error}"}

    plan = resolve_swap(slug, der, root)
    if plan["route"] != "C":
        return {"ok": False,
                "error": f"not a Route-C swap (route={plan['route']}); "
                         "--apply-swap only applies to a curated-arch fine-tune"}

    hf_home_p = D.resolve_hf_home(hf_home)
    weights_dir = DL.pull_dir(hf_home_p, slug)

    if do_download:
        ei = _min_einput(slug, der, hf_home_p)
        try:
            dl = download_fn(ei, fetcher=fetcher)
        except TypeError:
            dl = download_fn(ei)  # an injected mock may omit the fetcher kwarg
        if not getattr(dl, "ok", False):
            return {"ok": False,
                    "error": f"download failed: {getattr(dl, 'failure', '?')}",
                    "detail": getattr(dl, "detail", "")}
        if getattr(dl, "local_dir", ""):
            weights_dir = Path(dl.local_dir)

    served = slug.rsplit("/", 1)[-1]
    compose_path = emit_swap_compose(
        root, profile_like, weights_dir, served_name=served,
        has_mtp_head=plan["has_mtp_head"], brought_san=DL.sanitize_slug(slug),
    )
    return {
        "ok": True, "route": "C", "sibling_slug": plan["sibling_slug"],
        "served_model_name": served, "has_mtp_head": plan["has_mtp_head"],
        "weights_dir": str(weights_dir), "compose_path": str(compose_path),
    }
