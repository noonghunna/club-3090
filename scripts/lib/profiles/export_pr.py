#!/usr/bin/env python3
"""Community-loop completion: export a validated profiles-LOCAL model as a
ready-to-commit CORE PR bundle.

Reads a model that already lives in the gitignored LOCAL layer
(``scripts/lib/profiles-local/models.d/<id>.yml`` +
``scripts/lib/profiles-local/composes/<id>/…`` + its ``registry.local.json``
entries) and emits, into an OUTPUT directory (default ``/tmp``), the three
artifacts a community contributor needs to open a curated-catalog PR —
translated to the CORE layout per docs/ADDING_MODELS.md:

    <out>/models/<id>.yml                                    the ModelProfile
    <out>/models/<id>/<engine>/compose/<topology>/<quant>/<serving>.yml
    <out>/compose_registry.patch   unified diff adding the ``_entry(...)`` row(s)
                                   to COMPOSE_REGISTRY + the ``setup:`` block notes

The contributor copies ``models/…`` into their branch verbatim and applies the
patch; nothing in THIS repo is touched (the export writes only under ``--out``).

Refusals (exit 3, nothing written) fire when the local id lacks REQUIRED core
fields — the checks a maintainer would bounce the PR for: a real
display_name/family, a well-formed weights map, complete registry-entry kwargs,
a vLLM ``kvcalc_key``, and a compose carrying the mandatory ``Status:``
profile-header line.  ``--check`` runs exactly that completeness validation and
writes nothing; ``--dry-run`` prints the plan and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Repo-root default: this file lives at <root>/scripts/lib/profiles/export_pr.py.
_DEFAULT_ROOT = Path(__file__).resolve().parents[3]

_REG_REL = "scripts/lib/profiles/compose_registry.py"
_LOCAL_DIR_REL = "scripts/lib/profiles-local"
_LOCAL_MODELS_REL = f"{_LOCAL_DIR_REL}/models.d"
_LOCAL_COMPOSES_REL = f"{_LOCAL_DIR_REL}/composes"
_LOCAL_REGISTRY_REL = f"{_LOCAL_DIR_REL}/registry.local.json"
_LOCAL_SLUG_PREFIX = "local/"

# Filesystem engine dir under composes/<id>/ → the registry SLUG prefix
# (docs/ADDING_MODELS.md Step 3: llamacpp, NOT llama-cpp).
_ENGINE_SLUG_PREFIX = {
    "vllm": "vllm",
    "llama-cpp": "llamacpp",
    "ik-llama": "ik-llama",
    "beellama": "beellama",
}

_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

EXIT_COLLISION = 3   # refusal — missing/incomplete local id, nothing written
EXIT_INTERNAL = 2


class Refusal(Exception):
    """A bad-shape / incomplete-id refusal → exit 3, nothing written."""


def _info(msg: str) -> None:
    print(f"[export-pr] {msg}")


def _py_literal(v: Any) -> str:
    """JSON-sourced value → Python literal for the _entry(...) row (same type
    set as the registry kwargs carry)."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_py_literal(x) for x in v) + "]"
    return repr(v)


def _is_real_str(v: Any) -> bool:
    """A filled-in human value — not empty, not a leftover ``<...>`` scaffold
    placeholder."""
    return isinstance(v, str) and bool(v.strip()) and "<" not in v


def load_local_state(root: Path, mid: str) -> dict:
    """Read EVERYTHING the export needs from the LOCAL layer (disk truth, not
    the scaffold preview). Raises Refusal when the id is absent/broken."""
    profile_path = root / _LOCAL_MODELS_REL / f"{mid}.yml"
    if not profile_path.exists():
        raise Refusal(f"no LOCAL model profile: {profile_path} — nothing to export")
    try:
        profile_text = profile_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"unreadable: {profile_path}: {exc}")

    try:
        import yaml  # noqa: E402
    except Exception as exc:  # pragma: no cover - env guard
        raise Refusal(f"PyYAML unavailable, cannot validate the profile: {exc}")
    try:
        profile = yaml.safe_load(profile_text) or {}
    except Exception as exc:
        raise Refusal(f"{profile_path} does not parse as YAML: {exc}")
    if not isinstance(profile, dict):
        raise Refusal(f"{profile_path}: expected a YAML mapping")

    reg_path = root / _LOCAL_REGISTRY_REL
    raw: dict = {}
    if reg_path.exists():
        try:
            raw = json.loads(reg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refusal(f"{reg_path} is unreadable: {exc}")
        if not isinstance(raw, dict):
            raise Refusal(f"{reg_path}: expected a JSON object of {{slug: kwargs}}")
    entries = [
        {"slug": slug, "kwargs": (kwargs or {})}
        for slug, kwargs in raw.items()
        if isinstance(kwargs, dict) and kwargs.get("model") == mid
    ]
    if not entries:
        raise Refusal(
            f"no {reg_path.name} entries reference model {mid!r} — "
            "promote it into the LOCAL layer first (c3 ⑤ Promote)"
        )
    for e in entries:
        if not str(e["slug"]).startswith(_LOCAL_SLUG_PREFIX):
            raise Refusal(
                f"registry slug {e['slug']!r} lacks the {_LOCAL_SLUG_PREFIX!r} "
                "namespace — not a LOCAL-layer entry"
            )
    return {
        "model_id": mid,
        "profile_path": profile_path,
        "profile_text": profile_text,
        "profile": profile,
        "entries": entries,
    }


def completeness_gaps(root: Path, state: dict) -> list[str]:
    """Every reason a maintainer would bounce this PR — the REQUIRED core
    fields. Empty list == exportable."""
    gaps: list[str] = []
    mid = state["model_id"]
    p = state["profile"]

    # ── Profile-level required fields ────────────────────────────────────────
    for key in ("display_name", "family"):
        if not _is_real_str(p.get(key)):
            gaps.append(
                f"models/{mid}.yml: {key} must be a real value "
                f"(got {p.get(key)!r} — fill it before exporting)"
            )
    weights = p.get("weights")
    if not isinstance(weights, dict) or not weights:
        gaps.append(f"models/{mid}.yml: weights must be a non-empty map")
        weights = {}
    dwv = p.get("default_weight_variant")
    if dwv not in weights:
        gaps.append(
            f"models/{mid}.yml: default_weight_variant {dwv!r} is not a weights key"
        )
    for name, w in sorted(weights.items()):
        if not isinstance(w, dict):
            gaps.append(f"weights.{name}: not a mapping")
            continue
        if not _is_real_str(w.get("hf_repo")):
            gaps.append(f"weights.{name}: hf_repo must be a real Org/Repo")
        files = w.get("files")
        if not isinstance(files, list) or not files:
            gaps.append(
                f"weights.{name}: files must be a non-empty list "
                "(no files: = the WHOLE repo is fetched — ADDING_MODELS 'Two silent traps')"
            )
        fmt = str(w.get("format") or "")
        if fmt == "gguf" and not str(w.get("verify_glob") or "").strip():
            gaps.append(
                f"weights.{name}: GGUF entries REQUIRE verify_glob ('*.gguf') — "
                "the safetensors default matches nothing"
            )

    # ── Per-registry-entry required kwargs ───────────────────────────────────
    required_kwargs = (
        "model", "weights_variant", "workload", "engine", "kv_format", "tp",
        "max_ctx", "max_num_seqs", "mem_util", "compose_path", "default_port",
        "status",
    )
    for e in state["entries"]:
        slug, kw = e["slug"], e["kwargs"]
        for key in required_kwargs:
            if kw.get(key) in (None, ""):
                gaps.append(f"{slug}: registry kwarg {key!r} is required")
        cpath = str(kw.get("compose_path") or "")
        if not cpath:
            continue
        if not cpath.startswith(f"{_LOCAL_COMPOSES_REL}/"):
            gaps.append(
                f"{slug}: compose_path {cpath!r} does not live under "
                f"{_LOCAL_COMPOSES_REL}/"
            )
            continue
        comp_abs = root / cpath
        if not comp_abs.exists():
            gaps.append(f"{slug}: compose file missing on disk: {comp_abs}")
            continue
        try:
            comp_text = comp_abs.read_text(encoding="utf-8")
        except OSError as exc:
            gaps.append(f"{slug}: unreadable compose {comp_abs}: {exc}")
            continue
        if "# Profile (at-a-glance):" not in comp_text or "Status:" not in comp_text:
            gaps.append(
                f"{slug}: compose lacks the '# Profile (at-a-glance):' header "
                "with a 'Status:' line (test-compose-status-drift fails CI)"
            )
        # vLLM entries MUST carry a kvcalc_key (kv-calc projection); the
        # llama.cpp family legitimately defaults to "SKIP".
        ns = compose_slug_namespace(cpath)
        if ns == "vllm" and not str(kw.get("kvcalc_key") or "").strip():
            gaps.append(
                f"{slug}: vLLM entries require kvcalc_key \"<model>:<profile>\" "
                "(kv-calc projection; llama-family may omit it → SKIP)"
            )

    # ── Dry-wrap the TRANSLATED kwargs exactly like compose_registry will —
    # a kwarg missing from _entry's signature (e.g. `drafter`) or rejected by
    # its validation refuses HERE, before any bundle is written.
    if not gaps:
        for e in state["entries"]:
            t = translated_entry(state, e)
            sys.path.insert(0, str(root))
            try:
                from scripts.lib.profiles.compose_registry import (  # noqa: E402
                    _entry,
                )

                _entry(**t["kwargs"])
            except TypeError as exc:
                gaps.append(f"{e['slug']}: not valid _entry kwargs: {exc}")
            except ValueError as exc:
                gaps.append(f"{e['slug']}: rejected by _entry: {exc}")
            except Exception as exc:  # import failure of the registry itself
                gaps.append(
                    f"{e['slug']}: compose_registry.py did not import: {exc}"
                )
    return gaps


def compose_slug_namespace(compose_rel: str) -> str:
    """The registry slug prefix for a local compose path — derived from the
    filesystem engine dir (composes/<id>/<engine>/compose/...)."""
    parts = Path(compose_rel).parts
    try:
        i = parts.index("composes")
        fs_engine = parts[i + 2]
    except (ValueError, IndexError):
        return ""
    return _ENGINE_SLUG_PREFIX.get(fs_engine, "")


def core_compose_rel(mid: str, local_compose_rel: str) -> str:
    """LOCAL compose path → CORE layout path:
    scripts/lib/profiles-local/composes/<rest> → models/<rest>."""
    rest = local_compose_rel[len(_LOCAL_COMPOSES_REL):].lstrip("/")
    return f"models/{rest}"


def translated_entry(state: dict, entry: dict) -> dict:
    """One LOCAL registry entry → its CORE ``_entry(...)`` kwargs (pure copy —
    the local JSON is never mutated)."""
    mid = state["model_id"]
    kw = dict(entry["kwargs"])
    cpath = str(kw["compose_path"])
    kw["compose_path"] = core_compose_rel(mid, cpath)
    ns = compose_slug_namespace(cpath)
    if not kw.get("kvcalc_key"):
        kw["kvcalc_key"] = "SKIP"  # llama.cpp family default; vLLM was gated above
    core_slug = f"{ns}/{str(entry['slug'])[len(_LOCAL_SLUG_PREFIX):]}"
    return {"slug": core_slug, "kwargs": kw}


def render_entry_blocks(translated: list[dict], mid: str) -> list[str]:
    """The inserted source lines: setup-block notes + one _entry(...) row per
    translated entry (comments are valid inside the COMPOSE_REGISTRY literal)."""
    lines = [
        f"    # ── Community PR import: {mid} (exported from the LOCAL layer) ──",
        "    # Setup-block notes (scripts/lib/profiles/models/"
        f"{mid}.yml `setup:`):",
        "    #   * ships NO setup: block — the default dispatch policy applies",
        "    #     (primary fetch = default_weight_variant, no aliases/drafters).",
        "    #     Add one ONLY if the fetch policy differs (ADDING_MODELS.md "
        "Step 4c).",
        "    #   * NEW MODELS START AT status=\"incubating\" (hidden from "
        "`switch.sh --list`,",
        "    #     --force to launch) — promote up the enum as the FULL gate "
        "clears.",
        "    #   * vLLM rows: author calibration/<id>.yml after 4+ measured "
        "boots and",
        "    #     bump test-compose-registry-disk.sh's catalog count for every "
        "compose added.",
        "    #   * Gateway-reachable? Add services/litellm/config.yaml route → "
        ":<default_port>",
        "    #     (test-litellm-ports-resolve.sh gates the port).",
    ]
    for t in translated:
        lines.append(f'    "{t["slug"]}": _entry(')
        for k, v in t["kwargs"].items():
            lines.append(f"        {k}={_py_literal(v)},")
        lines.append("    ),")
    return lines


def render_registry_patch(root: Path, translated: list[dict], mid: str) -> str:
    """A git-applyable unified diff inserting the entry block(s) immediately
    BEFORE the closing brace of the COMPOSE_REGISTRY dict literal."""
    reg_abs = root / _REG_REL
    try:
        text = reg_abs.read_text(encoding="utf-8")
    except OSError as exc:
        raise Refusal(f"cannot read {_REG_REL}: {exc}")
    src = text.splitlines()
    start = next(
        (i for i, ln in enumerate(src) if ln.startswith("COMPOSE_REGISTRY = {")),
        None,
    )
    if start is None:
        raise Refusal(f"no COMPOSE_REGISTRY = {{ anchor in {_REG_REL}")
    close = next(
        (i for i in range(start + 1, len(src)) if src[i].startswith("}")), None
    )
    if close is None:
        raise Refusal(f"no closing brace for COMPOSE_REGISTRY in {_REG_REL}")
    added = render_entry_blocks(translated, mid)
    # Context BEFORE and AFTER the insertion point.  Trailing context is NOT
    # optional: git apply rejects a hunk that ends on added lines (a pure
    # append) — the closing-brace line is always there, so the hunk is
    # conventionally well-formed.
    n_before = min(3, close - start - 1)
    before = src[close - n_before:close]
    after = src[close:close + 3]
    old_len = n_before + len(after)
    new_len = old_len + len(added)
    ctx_start = close - n_before + 1  # 1-based
    hunks = [f"@@ -{ctx_start},{old_len} +{ctx_start},{new_len} @@"]
    hunks += [f" {ln}" for ln in before]
    hunks += [f"+{ln}" for ln in added]
    hunks += [f" {ln}" for ln in after]
    return (
        f"--- a/{_REG_REL}\n+++ b/{_REG_REL}\n" + "\n".join(hunks) + "\n"
    )


def plan_lines(out: Path, state: dict, translated: list[dict]) -> list[str]:
    mid = state["model_id"]
    lines = [
        f"plan (CORE PR bundle → {out}):",
        f"  write {out / 'models' / f'{mid}.yml'}",
    ]
    for t in translated:
        lines.append(f"  write {out / t['kwargs']['compose_path']}")
    lines.append(f"  write {out / 'compose_registry.patch'}")
    for t in translated:
        lines.append(f"  entry: {t['slug']}  port {t['kwargs'].get('default_port')}"
                     f"  status {t['kwargs'].get('status')}")
    lines.append("  NOTHING in this repo is touched — copy models/ + apply the "
                 "patch on your PR branch.")
    return lines


def export(root: Path, out: Path, state: dict) -> list[Path]:
    """Write the THREE bundle artifacts. Returns the written paths."""
    mid = state["model_id"]
    translated = [translated_entry(state, e) for e in state["entries"]]
    written: list[Path] = []

    def _write(rel: Path, content: str) -> Path:
        abs_ = out / rel
        try:
            abs_.parent.mkdir(parents=True, exist_ok=True)
            abs_.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise Refusal(f"could not write {abs_}: {exc}")
        written.append(abs_)
        _info(f"wrote {abs_}")
        return abs_

    _write(Path("models") / f"{mid}.yml", state["profile_text"])
    for t in translated:
        src = root / _LOCAL_COMPOSES_REL / t["kwargs"]["compose_path"][
            len("models/"):]
        _write(Path(t["kwargs"]["compose_path"]),
               src.read_text(encoding="utf-8"))
    _write(Path("compose_registry.patch"),
           render_registry_patch(root, translated, mid))
    return written


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="export_pr.py",
        description=(
            "Export a validated profiles-LOCAL model as a ready-to-commit "
            "CORE PR bundle (models/<id>.yml + core-layout compose + "
            "compose_registry.patch). Writes ONLY under --out."
        ),
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--spec-env", metavar="VAR",
                   help="read the export spec JSON ({model_id: …}) from this env var")
    g.add_argument("--spec-file", metavar="PATH",
                   help="read the export spec JSON from this file")
    ap.add_argument("--root", default=str(_DEFAULT_ROOT),
                    help="repo root (default: this checkout)")
    ap.add_argument("--out", default="/tmp",
                    help="bundle output directory (default: /tmp)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing")
    ap.add_argument("--check", action="store_true",
                    help="validate completeness only — never writes anything")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    try:
        if args.spec_env:
            raw_spec = os.environ.get(args.spec_env)
            if not raw_spec:
                raise Refusal(f"env var {args.spec_env} is empty/unset — no spec")
            spec = json.loads(raw_spec)
        else:
            spec = json.loads(Path(args.spec_file).read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise Refusal("spec is not a JSON object")
        mid = spec.get("model_id")
        if not isinstance(mid, str) or not _MODEL_ID_RE.match(mid or ""):
            raise Refusal(
                f"bad spec.model_id {mid!r} — expected [a-z0-9][a-z0-9._-]*"
            )
        state = load_local_state(root, mid)
        gaps = completeness_gaps(root, state)
        if gaps:
            print(
                f"[export-pr] REFUSED: {mid} is not PR-ready — "
                f"{len(gaps)} required core field(s) missing:",
                file=sys.stderr,
            )
            for gp in gaps:
                print(f"  - {gp}", file=sys.stderr)
            return EXIT_COLLISION
        translated = [translated_entry(state, e) for e in state["entries"]]
    except Refusal as exc:
        print(f"[export-pr] REFUSED: {exc}", file=sys.stderr)
        return EXIT_COLLISION
    except json.JSONDecodeError as exc:
        print(f"[export-pr] REFUSED: spec is not valid JSON: {exc}",
              file=sys.stderr)
        return EXIT_COLLISION

    for ln in plan_lines(out, state, translated):
        _info(ln)
    if args.check:
        _info(f"CHECK_OK {mid} — all required core fields present.")
        return 0
    if args.dry_run:
        _info("dry-run — nothing written.")
        return 0

    export(root, out, state)
    _info(f"bundle ready: copy {out}/models/ onto your PR branch, apply "
          f"{out / 'compose_registry.patch'}, then run the full suite "
          "(docs/ADDING_MODELS.md Steps 4b–7).")
    print(f"EXPORT_OK {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    try:
        raise SystemExit(main())
    except Refusal as exc:  # defensive: raised outside the validate window
        print(f"[export-pr] REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_COLLISION)
