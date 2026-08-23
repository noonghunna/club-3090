#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# (Same contract as test-kv-calc-fit.sh; guarded by test-locale-utf8.sh.)
export PYTHONUTF8="${PYTHONUTF8:-1}"

# test-kv-calc-llama.sh — fixture test for the kv-calc llama.cpp projection.
#
# Contract:
#   1. Synthetic llama fixture → projection matches the HAND-COMPUTED VRAM
#      exactly (weights_on_gpu + KV + compute buffer + runtime + drafter).
#   2. Verdict bands reuse the ±1.5 GB FIT_BAND vocabulary:
#      PASS → fits-clean, TIGHT → fits-constrained, FAIL → wont-fit.
#   3. solve_max_ctx_llama round-trips: the solved ctx still fits, one step
#      above does not.
#   4. --fit works end-to-end for a flipped llama slug (registry knobs:
#      CTX_SIZE / KV_TYPE / -ngl) with the SAME result shape as the vLLM path.
#   5. The vLLM pricing path is byte-unchanged (golden verdict + calibration
#      matrix still 22/22).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(sys.argv[1])
_p = ROOT / "tools" / "kv-calc.py"
spec = importlib.util.spec_from_file_location("kv_calc", _p)
kv = importlib.util.module_from_spec(spec)
sys.modules["kv_calc"] = kv
spec.loader.exec_module(kv)

failures: list[str] = []

def check(name, cond, detail=""):
    if cond:
        print(f"  ok: {name}")
    else:
        failures.append(f"{name} {detail}")
        print(f"  FAIL: {name} {detail}")

# ---------------------------------------------------------------------------
# 1. Synthetic fixture — hand-computed projection
# ---------------------------------------------------------------------------
# 8-layer dense GGUF, all layers on GPU (-ngl 99):
#   KV(32768, q4_0) = 32768 ctx × 8 layers × 4 kv_heads × 128 head_dim
#                     × 2 (K+V) × 0.5625 B/el = 150,994,944 B = 0.150994944 GB
#   weights 8.0 + KV 0.150994944 + compute 0.55×(512/512)×(4096/4096)
#   + runtime 0.45 + embedded-MTP head 0.30 = 9.450994944 GB
synth = {
    "model_id": "synth-8b",
    "model_family": "llamacpp-dense",
    "hidden_size": 4096,
    "num_hidden_layers": 8,
    "num_growing_layers": 8,
    "num_kv_heads": 4,
    "head_dim_attn": 128,
    "max_ctx_supported": 65536,
    "weights_gb": 8.0,
}
p = kv.predict_llamacpp(synth, kv_type="q4_0", max_ctx=32768, vram_gb=16,
                        drafter_gb=kv.LLAMACPP_MTP_HEAD_GB)
EXPECTED = 8.0 + 0.150994944 + 0.55 + 0.45 + 0.30
check("hand-computed total (all layers on GPU)",
      abs(p.total_gb - EXPECTED) < 1e-9, f"got {p.total_gb!r} want {EXPECTED!r}")
check("kv term exact", abs(p.kv_pool_requested_gb - 0.150994944) < 1e-9,
      f"got {p.kv_pool_requested_gb!r}")
check("verdict PASS on 16 GB", p.verdict == "PASS", p.verdict)

# Partial offload: -ngl 4 of 8 → half the weights AND half the growing layers.
p4 = kv.predict_llamacpp(synth, kv_type="q4_0", max_ctx=32768, n_gpu_layers=4,
                         vram_gb=16, drafter_gb=0.0)
EXP4 = 4.0 + 0.075497472 + 0.55 + 0.45
check("hand-computed total (ngl=4)", abs(p4.total_gb - EXP4) < 1e-9,
      f"got {p4.total_gb!r} want {EXP4!r}")

# KV bpe spot-checks against the GGUF block layouts.
check("q8_0 bpe ≈1.06", kv.LLAMA_KV_BYTES["q8_0"] == 34 / 32)
check("q4_0 bpe ≈0.57", kv.LLAMA_KV_BYTES["q4_0"] == 18 / 32)
check("f16 bpe = 2", kv.LLAMA_KV_BYTES["f16"] == 2.0)

# ---------------------------------------------------------------------------
# 2. Verdict bands — same vocabulary as the vLLM path
# ---------------------------------------------------------------------------
def verdict_of(vram):
    return kv._RAW_VERDICT_MAP[
        kv.predict_llamacpp(synth, kv_type="q4_0", max_ctx=32768,
                            vram_gb=vram, drafter_gb=kv.LLAMACPP_MTP_HEAD_GB).verdict]

check("total 9.451 on 11.0 GB → fits-clean", verdict_of(11.0) == "fits-clean")
check("total 9.451 on 10.0 GB → fits-constrained (within band)",
      verdict_of(10.0) == "fits-constrained")
check("total 9.451 on 9.0 GB → wont-fit", verdict_of(9.0) == "wont-fit")

# ---------------------------------------------------------------------------
# 3. solve_max_ctx_llama round-trip
# ---------------------------------------------------------------------------
best = kv.solve_max_ctx_llama(synth, kv_type="q4_0", vram_gb=9.6,
                              drafter_gb=kv.LLAMACPP_MTP_HEAD_GB)
at_best = kv.predict_llamacpp(synth, kv_type="q4_0", max_ctx=best, vram_gb=9.6,
                              drafter_gb=kv.LLAMACPP_MTP_HEAD_GB)
above = kv.predict_llamacpp(synth, kv_type="q4_0", max_ctx=best + 1024,
                            vram_gb=9.6, drafter_gb=kv.LLAMACPP_MTP_HEAD_GB)
check("solved ctx fits", at_best.verdict in ("PASS", "TIGHT"), at_best.verdict)
check("ctx + 1024 does not fit", above.verdict == "FAIL", above.verdict)
check("solver bound by VRAM, not the spec ceiling",
      best < synth["max_ctx_supported"], f"best={best}")
check("solved ctx within spec ceiling", best <= synth["max_ctx_supported"])

# ---------------------------------------------------------------------------
# 4. Registry --fit path (llama slug) — same shape as vLLM
# ---------------------------------------------------------------------------
res = kv.fit_verdict("llamacpp/default", "rtx3090", 24)
check("llamacpp/default priced", res.get("verdict") in
      ("fits-clean", "fits-constrained", "wont-fit"), json.dumps(res))
check("fit result shape identical to vLLM path",
      set(res) == {"verdict", "vram_est_gb", "band_gb", "max_ctx"}, sorted(res))
check("band is the documented ±1.5 GB", res.get("band_gb") == kv.FIT_BAND_GB)

# The registry knobs must actually drive the projection: recompute by hand
# from the entry (unsloth-q4km has size_gb: variable → bpw estimate
# 27 × 4.85 / 8 = 16.36875 GB; ctx 200000, q4_0, -ngl 99, MTP head).
entry = kv.COMPOSE_REGISTRY["llamacpp/default"]
model = kv.PROFILES.models[entry["model"]]
lspec = kv.build_llama_spec(model)
lspec["weights_gb"] = kv.llama_weights_gb(model, entry["weights_variant"], entry["model"])
hand = kv.predict_llamacpp(lspec, kv_type=entry["kv_format"],
                           max_ctx=entry["max_ctx"], vram_gb=24,
                           drafter_gb=kv.LLAMACPP_MTP_HEAD_GB)
check("--fit vram_est matches hand recomputation",
      abs(res["vram_est_gb"] - round(hand.total_gb, 4)) < 1e-6,
      f"{res['vram_est_gb']} vs {hand.total_gb}")

# SKIP semantics unchanged for entries the lead did NOT flip.
res_skip = kv.fit_verdict("beellama/carnice-v2-dual-q8-mtp", "rtx3090", 24)
check("unflipped (dual) llama slug stays unknown/SKIP",
      res_skip.get("verdict") == "unknown", json.dumps(res_skip))

# ---------------------------------------------------------------------------
# 5. vLLM path byte-unchanged (golden captured before the llama work landed)
# ---------------------------------------------------------------------------
golden = kv.fit_verdict("vllm/dual", "rtx3090", 24)
check("vllm/dual golden verdict unchanged",
      golden == {"verdict": "fits-clean", "vram_est_gb": 19.881,
                 "band_gb": 1.5, "max_ctx": 262144}, json.dumps(golden))

if failures:
    print(f"\ntest-kv-calc-llama.sh FAILED: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("python checks OK")
PY

# Calibration matrix must stay fully green — proves the vLLM math is untouched.
CAL_OUT="$(python3 tools/kv-calc.py --calibration 2>/dev/null)"
echo "$CAL_OUT" | tail -4
echo "$CAL_OUT" | grep -Eq "Overall: [0-9]+/[0-9]+ \(100%\)" || {
  echo "test-kv-calc-llama.sh FAILED: calibration matrix no longer 100%"
  exit 1
}

echo "test-kv-calc-llama.sh OK"
