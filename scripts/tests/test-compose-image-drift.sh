#!/usr/bin/env bash
# Drift guard (Codex review, #254/#324): every vLLM compose whose
# `image: ${VLLM_IMAGE:-<literal>}` default is a FIXED docker image must match the
# install.spec of the engine its registry slug resolves to. This catches the
# "bump the engine spec but forget the compose literal" drift that would silently
# feed direct `docker compose` users a stale image (the launcher injects VLLM_IMAGE
# from the engine and is fine; the compose literal is the unguarded path).
#
# Skipped (by design):
#   - templated literals like `…:nightly-${VLLM_NIGHTLY_SHA}` — self-sync via the
#     launcher-injected var, so the literal is always in step with the engine.
#   - pip-method engines (vllm-pip-baseline) — no docker image to match.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$ROOT_DIR" <<'PY'
import sys, re, pathlib
ROOT = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
from scripts.lib.profiles.compat import load_profiles

# Slugs whose ENGINE is itself pending the #254 migration off the purged
# vllm-nightly-clean nightly (the engine repoint needs its own stock-v0.22.0
# boot + tool-call/MTP validation — out of the #324 leg). Their compose literal
# was pre-bumped to v0.22.0, so they read as "drifted" against the dead nightly
# spec until the engine is repointed. Exempted + reported (NOT silently skipped);
# delete each from this set when it migrates to vllm-stable, and the guard then
# covers it. Tracked in #254.
PENDING_254 = set()

profiles = load_profiles()
# ANY `<ENGINE>_IMAGE` var, not just VLLM_IMAGE. Until 2026-09-12 this regex was
# VLLM_IMAGE-only, so 53 fixed literals across LLAMACPP_IMAGE / SGLANG_IMAGE /
# IK_LLAMA_IMAGE / BEELLAMA_IMAGE / VLLM_OMNI_IMAGE had NO drift guard at all --
# more unguarded than guarded. Found when the llama.cpp mainline pin bump turned up
# qwen3.6-35b-a3b/.../morikomorizz-q6kp/mtp.yml still on server-cuda-b9570 while its
# engine had moved to b10236 five weeks earlier. Negative control for the widening: a
# planted stale LLAMACPP_IMAGE + SGLANG_IMAGE literal made the PRE-widening gate
# report "ok"; after widening it REDS on both.
# Safe to take the first match per compose: verified 0 composes carry >1 FIXED image
# literal, so there is no sidecar ambiguity. If that changes, compare the literal
# whose registry/repo matches the engine spec, not merely the first one found.
pat = re.compile(r'image:\s*\$\{([A-Z_]+_IMAGE):-([^}]+)\}')
drift, checked, pending, deprecated = [], 0, [], 0
foreign = []
by_var = {}
for slug, entry in COMPOSE_REGISTRY.items():
    if entry.get("status") == "deprecated":
        deprecated += 1
        continue  # on its way out — not drift-guarded
    if slug in PENDING_254:
        pending.append(slug)
        continue
    eng_id = entry.get("engine")
    eng = profiles.engines.get(eng_id) if eng_id else None
    if eng is None:
        continue
    install = getattr(eng, "install", None) or {}
    if install.get("method") != "docker_image":
        continue  # pip baseline etc. — no docker image to compare against
    spec = install.get("spec", "")
    cpath = entry.get("compose_path")
    if not cpath:
        continue
    p = ROOT / cpath
    if not p.exists():
        continue
    m = pat.search(p.read_text())
    if not m:
        continue
    img_var, literal = m.group(1), m.group(2).strip()
    if "${" in literal:
        continue  # templated (e.g. nightly-${VLLM_NIGHTLY_SHA}) — self-syncs
    # Compare only within the SAME repository. A literal from a DIFFERENT repo is a
    # deliberately different engine image, not a stale tag of this engine's image --
    # e.g. the ik-llama/ornith* slugs carry ghcr.io/ikawrakow/ik-llama-cpp while their
    # registry entry says engine=llama-cpp-local, because no ik-llama engine profile
    # exists yet. Flagging those would be a false positive whose only "fix" is to point
    # an ik-llama compose at ggml-org/llama.cpp. Reported below as a NOTE instead, so the
    # registry oddity stays visible rather than silently passing. (Precedent: the docs
    # slug gate had to be narrowed after a wider scope produced ~200 false positives.)
    def _repo(ref):
        base = ref.split("@", 1)[0]
        return base.rsplit(":", 1)[0] if ":" in base.rsplit("/", 1)[-1] else base
    if _repo(literal) != _repo(spec):
        foreign.append(f"  {slug}: compose image `{literal}` is from a different repo than "
                       f"engine `{eng_id}` spec `{spec}` — no engine profile for it? ({cpath})")
        continue
    checked += 1
    by_var[img_var] = by_var.get(img_var, 0) + 1
    if literal != spec:
        drift.append(
            f"  {slug}: compose default `{literal}` != engine `{eng_id}` install.spec "
            f"`{spec}`  ({cpath})"
        )

if drift:
    print("IMAGE DRIFT — a fixed compose `${<ENGINE>_IMAGE:-…}` default disagrees with the")
    print("engine its slug resolves to (direct `docker compose` would serve a stale image):")
    print("\n".join(drift))
    print("Fix: bump the compose literal to the engine install.spec (or vice-versa).")
    sys.exit(1)
if pending:
    print(f"NOTE: {len(pending)} slug(s) exempt pending #254 engine migration: {', '.join(sorted(pending))}")
print(
    ("NOTE: different-repo engine images (informational, not drift):\n" + "\n".join(foreign) + "\n" if foreign else "") +
    f"test-compose-image-drift: ok ({checked} fixed-image composes match their engine spec "
    f"[{', '.join(f'{k}={v}' for k, v in sorted(by_var.items()))}]; "
    f"{deprecated} deprecated skipped, {len(pending)} pending-#254 exempt)"
)
PY
