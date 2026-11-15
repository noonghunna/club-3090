#!/usr/bin/env python3
"""catalog_rules — what a slug or an engine must meet to enter the curated catalog.

docs/ADDING_ENGINES.md and docs/ADDING_MODELS.md ("Which case is this?") are the
contract; this is the part of it a machine can check. One checker, two strengths:

  advisory  the local path: `catalog.sh register`, and `preflight-add-model.sh` on a
            local slug. It reports what would stop a contribution and never refuses,
            because the local layer exists to run things the catalog would never ship.
  blocking  `preflight-add-model.sh` on a curated slug (--strict), and
            `export_pr.py --check` (exit 3).

Every finding is pass, fail or UNCHECKED. A network rule that could not run (no
docker, a rate limit, an unreachable API) is unchecked, never a pass; with
--network, --strict treats an unchecked required rule as a failure.

Engine family and container discovery are decided by the bash helpers the tooling
itself uses (scripts/lib/engine-kind.sh, scripts/lib/club-containers.sh), so this
file keeps no copy of the prefix list or the engine-port set.

  catalog_rules.py --slug <slug>   [--network] [--strict] [--json] [--root DIR]
  catalog_rules.py --engine <id>   [--network] [--strict] [--json] [--root DIR]

Exit: 0 nothing blocking · 1 --strict and a required rule failed (or, with
--network, could not be checked) · 2 usage or load error.
"""

from __future__ import annotations

import argparse
import fnmatch
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

try:  # the catalog.sh path must not crash on a python3 without PyYAML
    import yaml
except ImportError:  # pragma: no cover - exercised by test-catalog-rules
    yaml = None

ROOT_DEFAULT = Path(__file__).resolve().parents[3]

DOC_NAMING = "docs/ADDING_ENGINES.md → CONTRACT — naming"
DOC_COMPOSES = "docs/ADDING_ENGINES.md → CONTRACT — composes on a new engine"
DOC_IMAGE = "docs/ADDING_ENGINES.md → CONTRACT — image provenance"
DOC_LINEAGE = "docs/ADDING_ENGINES.md → CONTRACT — engine lineage"
DOC_PROFILE = "docs/ADDING_ENGINES.md → CONTRACT — the engine profile"
DOC_CASES = "docs/ADDING_MODELS.md → Which case is this?"
DOC_HEADER = "AGENTS.md → Profile schema header"

REQUIRED, ADVISORY = "required", "advisory"

# Curated engines that predate a rule, kept here once (test-engine-lineage reads the
# lineage list too). They show as advisory, so they stay visible without blocking.
# Entries may only be removed, never added.
LINEAGE_NO_SOURCE_REF = {
    "vllm-lmcache": "LMCache's v0.4.7-cu129 tag was re-pushed and the image carries no revision label",
}
FILENAME_NOT_ID: dict[str, str] = {}   # was llama-cpp-local (llama-cpp-mainline.yml), renamed 2026-10-12
_NIGHTLY = "a nightly tag upstream purges; no live slug runs on it"
IMAGE_PIN_EXCEPTIONS = {
    "vllm-nightly-clean": _NIGHTLY,
    "vllm-nightly-dflash": _NIGHTLY,
    "vllm-nightly-full": _NIGHTLY,
    "vllm-nightly-mtp": _NIGHTLY,
}

# install.method values. `docker` is what catalog.sh register wrote before it
# learned `docker_image`; it still names an image, so it gets the pull check.
IMAGE_METHODS = {"docker_image", "docker"}
NO_IMAGE_METHODS = {"pip"}
PASS, FAIL, UNCHECKED = "pass", "fail", "unchecked"


@dataclass
class Finding:
    rule: str
    level: str   # required | advisory
    state: str   # pass | fail | unchecked
    message: str
    doc: str


def blocking(findings: list[Finding], network: bool) -> list[Finding]:
    """Findings that fail --strict: a required failure, and, when the network rules
    were asked for, a required rule that could not be checked."""
    return [f for f in findings if f.level == REQUIRED
            and (f.state == FAIL or (network and f.state == UNCHECKED))]


# ── compose reading ──────────────────────────────────────────────────────────

_DEFAULTED = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}$")


def _unwrap(value: Any) -> str:
    """`${VAR:-default}` → default, as registry-emit's container_name() reads it."""
    s = str(value or "").strip().strip("\"'")
    m = _DEFAULTED.match(s)
    return m.group(1) if m else s


def _service(text: str) -> dict:
    """The compose's first service, the one registry-emit reads `container_name` from."""
    data = yaml.safe_load(text) or {}
    services = data.get("services") or {}
    return next(iter(services.values()), None) or {}


def _internal_ports(service: dict) -> list[str]:
    ports = []
    for p in service.get("ports") or []:
        if isinstance(p, dict):
            if p.get("target") is not None:
                ports.append(_unwrap(p["target"]))
            continue
        ports.append(_unwrap(str(p).split("/")[0].rsplit(":", 1)[-1]))
    return [p for p in ports if p]


def _cache_mounts(service: dict) -> list[str]:
    vols = []
    for v in service.get("volumes") or []:
        src = v.get("source", "") if isinstance(v, dict) else str(v)
        if "CLUB3090_ENGINE_CACHE_DIR" in str(src):
            vols.append(str(src))
    return vols


# ── profiles and the tooling's own helpers ───────────────────────────────────

def _yaml_files(*dirs: Path):
    for d in dirs:
        if d.is_dir():
            yield from sorted(d.glob("*.yml"))


def _engine_index(root: Path) -> dict[str, tuple[dict, Path, bool]]:
    """id → (profile, path, is_core); core first, then the local layer, the loader's
    precedence (the first file to declare an id keeps it)."""
    core = root / "scripts/lib/profiles/engines"
    local = root / "scripts/lib/profiles-local/engines.d"
    index: dict[str, tuple[dict, Path, bool]] = {}
    for path in _yaml_files(core, local):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        index.setdefault(str(data.get("id")), (data, path, path.parent == core))
    return index


def _engine_profile(root: Path, engine_id: str,
                    index: Optional[dict] = None) -> tuple[Optional[dict], Optional[Path], bool]:
    """(profile, path, is_core) for the engine whose `id:` is engine_id."""
    index = index if index is not None else _engine_index(root)
    return index.get(engine_id, (None, None, False))


_SEP = "\x1f"   # not IFS whitespace, so an empty field survives `read`

_FACTS_SH = r"""
source "$1/scripts/lib/engine-kind.sh" || exit 3
source "$1/scripts/lib/club-containers.sh" || exit 3
echo "ports=${CLUB_ENGINE_PORTS_ANY:-}"
while IFS=$'\x1f' read -r eid cname image; do
  printf 'kinds=%s\x1f%s\x1f%s\n' "$(engine_kind_from_engine_id "$eid")" \
    "$(engine_kind_from_container "$cname")" "$(engine_kind_from_image "$image")"
done < "$2"
if [[ -s "$3" ]]; then
  club_engine_port_lines < "$3" | sed 's/^/kept=/'
fi
"""


def _tooling_facts_many(root: Path, rows: list[tuple[str, str, str, list[str]]]) -> list[dict]:
    """One bash run of the helpers the tooling uses, for every (engine id, container,
    image, internal ports) row: the engine family each name reads as, and the first
    port at which club_engine_port_lines would keep a container like this one (the
    discovery verify/bench/health rely on). Batched because the discovery filter
    reads the registry each time it is called (~1.5 s)."""
    probe = {}
    for eid, cname, image, ports in rows:
        for p in ports:
            probe[(cname, image, p)] = f"{cname}|0.0.0.0:1->{p}/tcp|{image}"
    with tempfile.TemporaryDirectory() as tmp:
        rows_file, lines_file = Path(tmp, "rows"), Path(tmp, "lines")
        rows_file.write_text("".join(f"{e}{_SEP}{c}{_SEP}{i}\n" for e, c, i, _ in rows), encoding="utf-8")
        lines_file.write_text("".join(f"{v}\n" for v in probe.values()), encoding="utf-8")
        env = dict(os.environ, ENGINE_KIND_ROOT=str(root), CLUB_ROOT=str(root),
                   PYTHONUTF8=os.environ.get("PYTHONUTF8", "1"))
        out = subprocess.run(["bash", "-c", _FACTS_SH, "_", str(root), str(rows_file), str(lines_file)],
                             capture_output=True, text=True, encoding="utf-8", env=env, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"could not run the engine-kind / club-containers helpers (rc {out.returncode})")
    ports_any, kinds, kept = "", [], set()
    for line in out.stdout.splitlines():
        key, _, val = line.partition("=")
        if key == "ports":
            ports_any = val
        elif key == "kinds":
            kinds.append(val.split(_SEP))
        elif key == "kept":
            kept.add(val)
    if len(kinds) != len(rows):
        raise RuntimeError(f"the engine-kind helpers answered {len(kinds)} of {len(rows)} rows")
    facts = []
    for (eid, cname, image, ports), (k_id, k_cont, k_img) in zip(rows, kinds):
        found = next((p for p in ports if probe[(cname, image, p)] in kept), "")
        facts.append({"id": k_id, "container": k_cont, "image": k_img,
                      "ports": ports_any, "discovered": found})
    return facts


LAUNCHERS = ("scripts/switch.sh", "scripts/launch.sh")


def launcher_image_envs(root: Path) -> dict[str, set[str]]:
    """launcher → the image env vars its export_variant_engine_pin() case arm accepts.
    An engine whose image_env is missing from either makes every launch of its
    slugs exit 2 with "unexpected engine pin export" (docs/ADDING_ENGINES.md)."""
    out: dict[str, set[str]] = {}
    for rel in LAUNCHERS:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except OSError:
            out[rel] = set()
            continue
        m = re.search(r"export_variant_engine_pin\(\)\s*\{(.*?)\n\}", text, re.S)
        envs: set[str] = set()
        for line in (m.group(1) if m else "").splitlines():
            arm = re.match(r"\s*([A-Z0-9_|]+_IMAGE[A-Z0-9_|]*)\)", line)
            if arm:
                envs.update(arm.group(1).split("|"))
        out[rel] = envs
    return out


# ── network: injectable so the tests never touch the network ────────────────

def _hf_model_info(repo: str, revision: Optional[str]) -> tuple[Optional[int], Optional[dict]]:
    path = f"/api/models/{urllib.parse.quote(repo, safe='/')}"
    if revision:
        path += f"/revision/{urllib.parse.quote(str(revision), safe='')}"
    try:
        with urllib.request.urlopen(f"https://huggingface.co{path}", timeout=20) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception:
        return None, None


def _image_inspect(spec: str) -> tuple[str, str]:
    """('ok'|'denied'|'error', detail) for an anonymous manifest lookup: an empty
    DOCKER_CONFIG means no saved registry login can help."""
    if not shutil.which("docker"):
        return "error", "docker is not installed here"
    with tempfile.TemporaryDirectory() as cfg:
        out = subprocess.run(["docker", "buildx", "imagetools", "inspect", spec],
                             capture_output=True, text=True, encoding="utf-8",
                             env=dict(os.environ, DOCKER_CONFIG=cfg), timeout=90, check=False)
    if out.returncode == 0:
        return "ok", ""
    err = (out.stderr or out.stdout).strip().splitlines()[-1:] or [""]
    if re.search(r"unauthori[sz]ed|denied|not found|manifest unknown|requested access", err[0], re.I):
        return "denied", err[0]
    return "error", err[0]


def _image_rule(spec: str, method: str, network: bool,
                inspect: Callable[[str], tuple[str, str]]) -> Finding:
    """Anything that names an image gets the pull check: a method that installs no
    image passes, and an install with no image to pull fails rather than passing."""
    if method in NO_IMAGE_METHODS:
        return Finding("image-anonymous", REQUIRED, PASS,
                       f"installed with {method}, not an image", DOC_IMAGE)
    if not spec:
        return Finding("image-anonymous", REQUIRED, FAIL,
                       "no image to pull: install.spec is empty and the compose names none", DOC_IMAGE)
    if method and method not in IMAGE_METHODS:
        return Finding("image-anonymous", REQUIRED, UNCHECKED,
                       f"not checked: install.method {method!r} is not one the catalog knows", DOC_IMAGE)
    if not network:
        return Finding("image-anonymous", REQUIRED, UNCHECKED,
                       "not checked: run with --network", DOC_IMAGE)
    status, detail = inspect(spec)
    if status == "ok":
        return Finding("image-anonymous", REQUIRED, PASS,
                       f"{spec} pulls without registry credentials", DOC_IMAGE)
    if status == "denied":
        return Finding("image-anonymous", REQUIRED, FAIL,
                       f"{spec} does not pull without credentials: {detail}", DOC_IMAGE)
    return Finding("image-anonymous", REQUIRED, UNCHECKED,
                   f"not checked: {detail or 'docker buildx imagetools inspect failed'}", DOC_IMAGE)


# ── the rules ────────────────────────────────────────────────────────────────

def _registry(root: Path) -> dict:
    sys.path.insert(0, str(root / "scripts/lib/profiles"))
    try:
        from compose_registry import get_registry
        return get_registry(str(root))
    finally:
        sys.path.pop(0)


def _slug_inputs(root: Path, entry: dict, engines: dict) -> dict:
    engine_id = str(entry.get("engine") or "")
    cpath = Path(str(entry.get("compose_path") or ""))
    text = (cpath if cpath.is_absolute() else root / cpath).read_text(encoding="utf-8")
    svc = _service(text)
    profile, _, _ = _engine_profile(root, engine_id, engines)
    install = (profile or {}).get("install") or {}
    return {"engine_id": engine_id, "text": text, "svc": svc,
            "container": _unwrap(svc.get("container_name")), "ports": _internal_ports(svc),
            "install": install, "image": str(install.get("spec") or _unwrap(svc.get("image")))}


def check_slugs(root: Path, entries: dict[str, dict], *, network: bool = False,
                hf: Callable = _hf_model_info, inspect: Callable = _image_inspect,
                registry: Optional[dict] = None) -> dict[str, list[Finding]]:
    """check_slug for many slugs with one run of the bash helpers."""
    root = Path(root)
    registry = registry if registry is not None else _registry(root)
    engines = _engine_index(root)
    inputs = {slug: _slug_inputs(root, e, engines) for slug, e in entries.items()}
    facts = _tooling_facts_many(root, [(i["engine_id"], i["container"], i["image"], i["ports"])
                                       for i in inputs.values()])
    return {slug: _slug_findings(root, slug, entries[slug], i, f, network=network, hf=hf,
                                 inspect=inspect, registry=registry)
            for (slug, i), f in zip(inputs.items(), facts)}


def check_slug(root: Path, slug: str, entry: Optional[dict] = None, *, network: bool = False,
               hf: Callable = _hf_model_info, inspect: Callable = _image_inspect,
               registry: Optional[dict] = None) -> list[Finding]:
    root = Path(root)
    registry = registry if registry is not None else _registry(root)
    entry = entry if entry is not None else registry.get(slug)
    if entry is None:
        raise LookupError(f"no registry entry for {slug!r} (core or local)")
    return check_slugs(root, {slug: entry}, network=network, hf=hf, inspect=inspect,
                       registry=registry)[slug]


def _slug_findings(root: Path, slug: str, entry: dict, inp: dict, facts: dict, *, network: bool,
                   hf: Callable, inspect: Callable, registry: dict) -> list[Finding]:
    out: list[Finding] = []
    engine_id, text, svc = inp["engine_id"], inp["text"], inp["svc"]
    container, ports, install = inp["container"], inp["ports"], inp["install"]

    kind = facts.get("id", "unknown")
    out.append(Finding("engine-family", REQUIRED, PASS if kind != "unknown" else FAIL,
                       f"engine {engine_id!r} resolves to {kind}" if kind != "unknown" else
                       f"engine {engine_id!r} resolves to no family: give the id its family prefix "
                       "or the profile a `type:`", DOC_NAMING))

    ckind = facts.get("container", "unknown")
    ok = ckind == kind and kind != "unknown"
    out.append(Finding("container-family", REQUIRED, PASS if ok else FAIL,
                       f"container {container!r} reads as {ckind}" if ok else
                       f"container {container!r} reads as {ckind}, the engine as {kind}: "
                       "start the container name with the engine family's prefix", DOC_NAMING))

    found = facts.get("discovered", "")
    out.append(Finding("discoverable", REQUIRED, PASS if found else FAIL,
                       f"found by the serving scripts on internal port {found}" if found else
                       f"internal port(s) {', '.join(ports) or 'none'} aren't ones the serving scripts "
                       f"look for ({facts.get('ports', '').replace('|', ', ')}): verify.sh, bench.sh and "
                       "health.sh would need URL= and CONTAINER=", DOC_NAMING))

    has_header = "# Profile (at-a-glance):" in text and re.search(r"(?m)^#\s+Status:", text)
    out.append(Finding("status-header", REQUIRED, PASS if has_header else FAIL,
                       "Profile header with a Status: line" if has_header else
                       "no '# Profile (at-a-glance):' header with a 'Status:' line", DOC_HEADER))

    caches = _cache_mounts(svc)
    if not caches:
        out.append(Finding("cache-ownership", REQUIRED, PASS, "no engine-cache mount", DOC_COMPOSES))
    else:
        user_ok = str(svc.get("user", "")).startswith("0:")
        out.append(Finding("cache-ownership", REQUIRED, PASS if user_ok else FAIL,
                           "cache mount runs with the host GID (user: \"0:...\")" if user_ok else
                           "cache mount without user: \"0:${DOCKER_GID:-1000}\": its files come out "
                           "root:root in ~/.cache/club-3090", DOC_COMPOSES))
        umask_ok = "umask 0002" in text
        out.append(Finding("cache-umask", ADVISORY, PASS if umask_ok else FAIL,
                           "entrypoint sets umask 0002" if umask_ok else
                           "no `umask 0002` in the compose: cache files won't be group-writable",
                           DOC_COMPOSES))

    restart = str(svc.get("restart", ""))
    out.append(Finding("restart-knob", ADVISORY, PASS if "CLUB3090_RESTART" in restart else FAIL,
                       "restart: ${CLUB3090_RESTART:-...}" if "CLUB3090_RESTART" in restart else
                       f"restart is {restart or 'unset'!r}, not ${{CLUB3090_RESTART:-unless-stopped}}",
                       DOC_COMPOSES))

    out.extend(_same_model_defaults(slug, entry, registry))
    out.extend(_effort_rule(root, entry, text))
    out.extend(_weights_rules(root, entry, network, hf))
    out.append(_image_rule(inp["image"], str(install.get("method") or ""), network, inspect))
    return out


def _same_model_defaults(slug: str, entry: dict, registry: dict) -> list[Finding]:
    """Case 2: the model's slugs on OTHER engines set the expectations. Advisory, and
    `None` means "the registry doesn't say", not "different"."""
    model, engine = entry.get("model"), entry.get("engine")
    others = {s: e for s, e in registry.items()
              if s != slug and e.get("model") == model and e.get("engine") != engine
              and e.get("status") != "deprecated"}
    if not others:
        return [Finding("same-model-defaults", ADVISORY, PASS,
                        "no slug of this model on another engine to compare with", DOC_CASES)]
    mine = entry.get("chat_template")
    theirs = sorted({str(e.get("chat_template")) for e in others.values() if e.get("chat_template")})
    if mine is None:
        return [Finding("same-model-defaults", ADVISORY, FAIL,
                        "the registry doesn't say which chat template this slug serves "
                        f"(the model's other engines say: {', '.join(theirs) or 'nothing'})", DOC_CASES)]
    if theirs and str(mine) not in theirs:
        return [Finding("same-model-defaults", ADVISORY, FAIL,
                        f"chat_template {mine!r}; the model's other engines use {', '.join(theirs)}",
                        DOC_CASES)]
    return [Finding("same-model-defaults", ADVISORY, PASS,
                    f"chat_template {mine!r} matches the model's other engines", DOC_CASES)]


@functools.lru_cache(maxsize=None)
def _load_profiles_cached(root_str: str):
    """Parse the profile tree once per root (callers prime sys.path first)."""
    from scripts.lib.profiles.compat import load_profiles
    return load_profiles(Path(root_str) / "scripts/lib/profiles")


def _effort_rule(root: Path, entry: dict, text: str) -> list[Finding]:
    """#1951: a thinking slug's model must declare its `effort:` block, so the
    gateway render and the omp overrides have one source for the ladder. The
    block is the model profile's `effort:` key — levels once a wire probe has
    pinned them, or an honest `status:` (unprobed / no-string-ladder /
    no-thinking) until then; the compose keeps only the default (principle 1).
    Same gate scripts/tests/test-effort-profiles.sh holds the curated catalog
    to; this covers the PR-bundle and cockpit paths that never run that test."""
    sp = entry.get("sampler_profiles")
    thinking = isinstance(sp, dict) and any("thinking" in str(k).lower() for k in sp)
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    knob = "REASONING_EFFORT" in body or "reasoning_effort" in body
    if not (thinking or knob):
        return [Finding("effort-block", REQUIRED, PASS,
                        "not an effort-aware slug (no thinking sampler profile, "
                        "no reasoning-effort knob in the compose)", DOC_CASES)]
    sys.path.insert(0, str(root))
    try:
        from scripts.lib.profiles.compat import load_profiles
        # Cached per root: the catalog sweep calls this once per slug (~200
        # loads of the same tree otherwise), and load_profiles parses every
        # model + engine YAML each time.
        profiles = _load_profiles_cached(str(root))
    finally:
        sys.path.pop(0)
    model = profiles.models.get(str(entry.get("model")))
    if model is None:
        return [Finding("effort-block", REQUIRED, UNCHECKED,
                        f"model {entry.get('model')!r} has no profile to check "
                        "the effort block against", DOC_CASES)]
    eff = getattr(model, "effort", None)
    if eff and (eff.get("levels") or eff.get("status")):
        shown = "/".join(eff["levels"]) if eff.get("levels") else eff.get("status")
        return [Finding("effort-block", REQUIRED, PASS,
                        f"model {entry.get('model')} declares effort {shown}", DOC_CASES)]
    why = "thinking sampler profile" if thinking else "reasoning-effort knob in the compose"
    return [Finding("effort-block", REQUIRED, FAIL,
                    f"{why}, but model {entry.get('model')} has no `effort:` block — "
                    "add the probed ladder or an honest status (docs/ADDING_MODELS.md, "
                    "#1951)", DOC_CASES)]


def _weights_rules(root: Path, entry: dict, network: bool, hf: Callable) -> list[Finding]:
    if not network:
        return [Finding("weights-download", REQUIRED, UNCHECKED, "not checked: run with --network", DOC_CASES)]
    sys.path.insert(0, str(root))
    try:
        from scripts.lib.profiles.compat import load_profiles
        profiles = load_profiles(root / "scripts/lib/profiles")
    finally:
        sys.path.pop(0)
    model = profiles.models.get(str(entry.get("model")))
    if model is None:
        return [Finding("weights-download", REQUIRED, FAIL,
                        f"model {entry.get('model')!r} has no profile", DOC_CASES)]
    names = [str(entry.get("weights_variant"))]
    for comp in entry.get("weights_companions") or ():
        names.append(str(comp).split(":", 1)[-1])
    out = []
    for name in names:
        w = model.weights.get(name)
        if not isinstance(w, dict) or not w.get("hf_repo"):
            out.append(Finding("weights-download", REQUIRED, FAIL,
                               f"weights {name!r}: no hf_repo for setup.sh to download", DOC_CASES))
            continue
        repo, rev = str(w["hf_repo"]), w.get("revision")
        status, info = hf(repo, rev)
        if status is None:
            out.append(Finding("weights-download", REQUIRED, UNCHECKED,
                               f"not checked: Hugging Face API unreachable for {repo}", DOC_CASES))
            continue
        if status in (401, 403):
            out.append(Finding("weights-download", REQUIRED, UNCHECKED,
                               f"not checked: {repo} needs a token to list ({status})", DOC_CASES))
            continue
        if status != 200 or not isinstance(info, dict):
            out.append(Finding("weights-download", REQUIRED, FAIL,
                               f"{repo}{'@' + str(rev) if rev else ''} does not resolve on Hugging Face "
                               f"(HTTP {status})", DOC_CASES))
            continue
        files = [s.get("rfilename", "") for s in info.get("siblings") or []]
        patterns = w.get("files") or ["*"]
        missing = [g for g in patterns if not any(fnmatch.fnmatch(f, g) for f in files)]
        fetched = [f for f in files if any(fnmatch.fnmatch(f, g) for g in patterns)]
        verify = w.get("verify_glob")
        gated = info.get("gated")
        if missing:
            out.append(Finding("weights-download", REQUIRED, FAIL,
                               f"{repo}: files: {', '.join(missing)} match nothing in the repo", DOC_CASES))
        elif verify and not any(fnmatch.fnmatch(f, str(verify)) for f in fetched):
            # setup.sh verifies the download against verify_glob and fails when
            # nothing matches, so a repo without the model files cannot pass here.
            out.append(Finding("weights-download", REQUIRED, FAIL,
                               f"{repo}: verify_glob {verify} matches none of the files setup.sh "
                               "would download", DOC_CASES))
        else:
            note = f" (gated: {gated}; users must accept its license)" if gated else ""
            out.append(Finding("weights-download", REQUIRED, PASS,
                               f"{repo}: every files: pattern matches"
                               f"{f' and verify_glob {verify} finds the model files' if verify else ''}{note}",
                               DOC_CASES))
    return out


def check_engine(root: Path, engine_id: str, *, network: bool = False,
                 inspect: Callable = _image_inspect) -> list[Finding]:
    root = Path(root)
    profile, path, is_core = _engine_profile(root, engine_id)
    if profile is None:
        raise LookupError(f"no engine profile with id {engine_id!r} (core or local)")
    out: list[Finding] = []
    kind = _tooling_facts_many(root, [(engine_id, "", "", [])])[0]["id"]
    out.append(Finding("engine-family", REQUIRED, PASS if kind != "unknown" else FAIL,
                       f"resolves to {kind}" if kind != "unknown" else
                       "resolves to no family: give the id its family prefix or the profile a `type:`",
                       DOC_NAMING))
    if is_core:
        same = path.stem == engine_id
        if not same and engine_id in FILENAME_NOT_ID:
            out.append(Finding("profile-filename", ADVISORY, FAIL,
                               f"known exception: {FILENAME_NOT_ID[engine_id]}", DOC_PROFILE))
        else:
            out.append(Finding("profile-filename", REQUIRED, PASS if same else FAIL,
                               f"{path.name} matches its id" if same else
                               f"{path.name} declares id {engine_id!r}: name the file {engine_id}.yml",
                               DOC_PROFILE))
    lin = profile.get("lineage")
    if not isinstance(lin, dict):
        out.append(Finding("engine-lineage", REQUIRED, FAIL, "no `lineage:` block", DOC_LINEAGE))
    else:
        rel = lin.get("relation")
        gaps = [k for k in ("source", "source_ref") if rel != "stock" and not lin.get(k)]
        if gaps == ["source_ref"] and engine_id in LINEAGE_NO_SOURCE_REF:
            out.append(Finding("engine-lineage", ADVISORY, FAIL,
                               f"known exception, no source_ref: {LINEAGE_NO_SOURCE_REF[engine_id]}",
                               DOC_LINEAGE))
        else:
            out.append(Finding("engine-lineage", REQUIRED, FAIL if gaps else PASS,
                               f"relation {rel}: missing {', '.join(gaps)}" if gaps else f"relation {rel}",
                               DOC_LINEAGE))
    install = profile.get("install") or {}
    out.append(_pin_rule(engine_id, profile, path.read_text(encoding="utf-8"), is_core))
    out.append(_image_rule(str(install.get("spec") or ""), str(install.get("method") or ""), network, inspect))
    return out


def _pin_rule(engine_id: str, profile: dict, text: str, is_core: bool) -> Finding:
    """How the image is pinned (CONTRACT — image provenance): a derivative, fork or
    new engine pins by digest; a stock or patched one may pin a release tag, never
    nightly-* or latest. The spec is one unquoted token on exactly one `spec:` line,
    because engine-pin-bump.sh and resolved_config.py parse it that way."""
    install = profile.get("install") or {}
    method, spec = str(install.get("method") or ""), str(install.get("spec") or "")
    if method in NO_IMAGE_METHODS:
        return Finding("image-pin", REQUIRED, PASS, f"installed with {method}", DOC_IMAGE)
    if not spec:
        return Finding("image-pin", REQUIRED, FAIL, "no install.spec", DOC_IMAGE)
    lines = re.findall(r"(?m)^\s*spec:\s*(.*?)\s*$", text)
    relation = str((profile.get("lineage") or {}).get("relation") or "")
    last = spec.rsplit("/", 1)[-1]
    tag = last.rsplit(":", 1)[-1] if ":" in last and "@" not in last else ""
    if len(lines) != 1 or lines[0] != spec:
        problem = (f"{len(lines)} `spec:` lines" if len(lines) != 1 else f"`spec: {lines[0]}` is quoted")
        msg = f"{problem}: write install.spec as one unquoted token on exactly one `spec:` line"
    elif "@sha256:" in spec:
        return Finding("image-pin", REQUIRED, PASS, "pinned by digest", DOC_IMAGE)
    elif relation in ("stock", "patched") and tag and tag != "latest" and not tag.startswith("nightly"):
        return Finding("image-pin", REQUIRED, PASS, f"{relation} engine on release tag {tag}", DOC_IMAGE)
    elif relation in ("stock", "patched"):
        msg = f"{spec}: pin a release tag or a digest, never {tag or 'no tag'}"
    else:
        msg = f"{spec}: a {relation or 'non-stock'} engine pins its image by digest (@sha256:…)"
    if is_core and engine_id in IMAGE_PIN_EXCEPTIONS:
        return Finding("image-pin", ADVISORY, FAIL,
                       f"known exception: {IMAGE_PIN_EXCEPTIONS[engine_id]}", DOC_IMAGE)
    return Finding("image-pin", REQUIRED, FAIL, msg, DOC_IMAGE)


# ── output ───────────────────────────────────────────────────────────────────

_MARK = {PASS: "✓", UNCHECKED: "?"}


def render(subject: str, findings: list[Finding], network: bool) -> str:
    def mark(f):
        if f.state == FAIL:
            return "✗" if f.level == REQUIRED else "!"
        return _MARK[f.state]
    counts = {
        "pass": sum(f.state == PASS for f in findings),
        "required failing": sum(f.state == FAIL and f.level == REQUIRED for f in findings),
        "advisory": sum(f.state == FAIL and f.level == ADVISORY for f in findings),
        "unchecked": sum(f.state == UNCHECKED for f in findings),
    }
    lines = [f"[catalog-rules] {subject}: " + " · ".join(f"{v} {k}" for k, v in counts.items())]
    for f in findings:
        tag = "" if f.state == PASS else f" [{f.level}]"
        lines.append(f"  {mark(f)} {f.rule}{tag}  {f.message}")
        if f.state != PASS:
            lines.append(f"      → {f.doc}")
    if not network and any(f.state == UNCHECKED for f in findings):
        lines.append("  (network rules not run: add --network before opening a PR)")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--engine")
    ap.add_argument("--network", action="store_true", help="also check the HF download and the anonymous image pull")
    ap.add_argument("--strict", action="store_true", help="exit 1 when a required rule fails")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    args = ap.parse_args(argv)
    if yaml is None:
        print("[catalog-rules] skipped: needs PyYAML (pip install pyyaml)", file=sys.stderr)
        return 2 if args.strict else 0
    root = Path(args.root).resolve()
    try:
        findings = (check_slug(root, args.slug, network=args.network) if args.slug
                    else check_engine(root, args.engine, network=args.network))
    except Exception as exc:  # a rule that cannot run is never a pass
        print(f"[catalog-rules] could not check {args.slug or args.engine}: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=1))
    else:
        print(render(args.slug or args.engine, findings, args.network))
    return 1 if args.strict and blocking(findings, args.network) else 0


if __name__ == "__main__":
    raise SystemExit(main())
