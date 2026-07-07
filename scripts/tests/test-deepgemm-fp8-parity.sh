#!/usr/bin/env bash
# test-deepgemm-fp8-parity — every fp8/NVFP4-weights vLLM compose MUST carry the
# `- VLLM_USE_DEEP_GEMM` pass-through so the launcher's consumer-Blackwell
# auto-disable (_deepgemm_env, sm {8.9, 12.0, 12.1}) actually reaches the
# container. Without the receiving line, docker-compose drops the injected
# VLLM_USE_DEEP_GEMM=0 and a 5090 / PRO 6000 / GB10 user hits the fp8-GEMM
# "recipe not found" boot crash (disc #571 / #580; NVFP4 follow-up #613).
#
# This guards the sibling-compose DRIFT class: dual-max carried the line
# (added in #580) but multi-max / dual-lmcache / diffusiongemma-dual — the
# same fp8-weights config — had silently drifted without it. NVFP4 ModelOpt
# composes also route their FP8 attention/linear layers through DeepGEMM. These composes
# are hand-maintained parallel copies (no extends/include), so nothing else
# keeps them in sync.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import re
import sys

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

LINE = re.compile(r"^\s*-\s*VLLM_USE_DEEP_GEMM\s*$", re.M)
missing = []
checked = 0
for slug, e in COMPOSE_REGISTRY.items():
    # Mirror the _deepgemm_env gate EXACTLY: fp8-FAMILY weights (startswith
    # "fp8" → "fp8" and "fp8-dynamic"/compressed-tensors, e.g. agents-a1)
    # plus ModelOpt NVFP4, whose FP8 attention/linear layers route through
    # DeepGEMM. INT4/AWQ/W8A8/bf16 never invoke DeepGEMM → correctly excluded.
    weights_variant = e.get("weights_variant") or ""
    if not (weights_variant.startswith("fp8") or weights_variant == "nvfp4"):
        continue
    if "vllm" not in (e.get("engine") or ""):
        continue
    checked += 1
    try:
        txt = open(e["compose_path"]).read()
    except OSError:
        missing.append(f"{slug}: compose unreadable ({e['compose_path']})")
        continue
    if not LINE.search(txt):
        missing.append(f"{slug}: missing `- VLLM_USE_DEEP_GEMM` ({e['compose_path']})")

if missing:
    print("test-deepgemm-fp8-parity: FAIL", file=sys.stderr)
    for m in missing:
        print(f"  ✗ {m}", file=sys.stderr)
    print("  fp8 weights on consumer Blackwell (sm_120/121) need the pass-through so the",
          file=sys.stderr)
    print("  launcher's VLLM_USE_DEEP_GEMM=0 reaches the container (disc #571 / #580).",
          file=sys.stderr)
    sys.exit(1)

print(f"  ✓ all {checked} fp8/NVFP4-weights vLLM composes carry the VLLM_USE_DEEP_GEMM pass-through")
PY

echo "test-deepgemm-fp8-parity: ok"
