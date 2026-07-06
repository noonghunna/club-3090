#!/usr/bin/env bash
set -euo pipefail

# test-kv-calc-fit.sh — fixture test for `tools/kv-calc.py --fit … --json`.
#
# Contract: `--fit <slug|model> --card <gpu> --json` emits
#   {"verdict": "fits-clean"|"fits-constrained"|"wont-fit",
#    "vram_est_gb": float, "band_gb": float, "max_ctx": int}
# or, when an input can't be resolved (unknown slug/model, non-vLLM compose,
# unrecognized card),
#   {"verdict": "unknown", "error": "..."}.
#
# This asserts the SHAPE of the new emit (valid JSON + expected top-level
# keys + verdict enum) and that the wrapper REUSES the existing pricing
# (the slug path and the bare-model→curated-default path agree). It does
# NOT re-derive the GB math — that is kv-calc's calibrated authority, with
# its own --calibration gate.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
KVCALC = ROOT / "tools" / "kv-calc.py"

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


def run_fit(*extra_args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, str(KVCALC), *extra_args],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


VERDICTS_OK = {"fits-clean", "fits-constrained", "wont-fit"}
TOP_KEYS_OK = {"verdict", "vram_est_gb", "band_gb", "max_ctx"}

# ---------------------------------------------------------------------------
# (a) a known registry slug on a known card → valid JSON, ok-shape verdict.
# ---------------------------------------------------------------------------
rc, out, err = run_fit("--fit", "vllm/dual", "--card", "rtx3090", "--json")
check(rc == 0, f"(a) vllm/dual rtx3090 exits 0 (got {rc}; stderr={err.strip()!r})")
try:
    d = json.loads(out)
    parsed = True
except Exception as exc:  # noqa: BLE001
    d, parsed = {}, False
    check(False, f"(a) output is valid JSON (got {exc}; out={out!r})")
if parsed:
    check(d.get("verdict") in VERDICTS_OK,
          f"(a) verdict is an ok-shape enum value (got {d.get('verdict')!r})")
    check(set(d.keys()) == TOP_KEYS_OK,
          f"(a) top-level keys == {sorted(TOP_KEYS_OK)} (got {sorted(d.keys())})")
    check(isinstance(d.get("vram_est_gb"), (int, float)) and d["vram_est_gb"] > 0,
          f"(a) vram_est_gb is a positive number (got {d.get('vram_est_gb')!r})")
    check(isinstance(d.get("band_gb"), (int, float)) and d["band_gb"] > 0,
          f"(a) band_gb is a positive number (got {d.get('band_gb')!r})")
    check(isinstance(d.get("max_ctx"), int) and not isinstance(d.get("max_ctx"), bool)
          and d["max_ctx"] > 0,
          f"(a) max_ctx is a positive int (got {d.get('max_ctx')!r})")

# ---------------------------------------------------------------------------
# (a') SEAM regression: the HYPHENATED hardware-profile id 'rtx-3090' — the
#      exact form switch.sh --explain (explain_detect_card) feeds — must
#      resolve to a real verdict, identical to the de-hyphenated 'rtx3090'.
#      (kv-calc's normalizer used to not strip hyphens → returned 'unknown'
#      RC=2 → switch.sh silently showed "fit unavailable" on the 3090 rig.)
# ---------------------------------------------------------------------------
rc_h, out_h, err_h = run_fit("--fit", "vllm/dual", "--card", "rtx-3090", "--json")
check(rc_h == 0, f"(a') hyphenated --card rtx-3090 exits 0 (got {rc_h}; stderr={err_h.strip()!r})")
try:
    dh = json.loads(out_h)
    check(dh.get("verdict") in VERDICTS_OK,
          f"(a') rtx-3090 yields a real verdict, not 'unknown' (got {dh.get('verdict')!r})")
    check(dh == json.loads(out),
          "(a') rtx-3090 prices identically to rtx3090 (hyphen-insensitive)")
except Exception as exc:  # noqa: BLE001
    check(False, f"(a') rtx-3090 output is valid JSON (got {exc}; out={out_h!r})")

# ---------------------------------------------------------------------------
# (b) a bare catalogued model resolves to its curated vLLM default and prices
#     identically to that default slug (the wrapper reuses one pricing path).
# ---------------------------------------------------------------------------
rc_m, out_m, _ = run_fit("--fit", "qwen3.6-27b", "--card", "rtx3090", "--json")
rc_s, out_s, _ = run_fit("--fit", "vllm/dual", "--card", "rtx3090", "--json")
check(rc_m == 0, f"(b) bare model qwen3.6-27b exits 0 (got {rc_m})")
try:
    check(json.loads(out_m) == json.loads(out_s),
          "(b) bare model == its curated default slug (vllm/dual) — single pricing path")
except Exception as exc:  # noqa: BLE001
    check(False, f"(b) bare-model/slug JSON both parse (got {exc})")

# ---------------------------------------------------------------------------
# (c) --card accepts a bare GB number (uncatalogued cards never block).
# ---------------------------------------------------------------------------
rc_n, out_n, _ = run_fit("--fit", "vllm/dual", "--card", "48", "--json")
check(rc_n == 0, f"(c) numeric --card exits 0 (got {rc_n})")
try:
    check(json.loads(out_n).get("verdict") in VERDICTS_OK,
          "(c) numeric --card yields an ok-shape verdict")
except Exception as exc:  # noqa: BLE001
    check(False, f"(c) numeric --card JSON parses (got {exc})")

# ---------------------------------------------------------------------------
# (d) unresolvable inputs degrade to {"verdict":"unknown","error":...},
#     non-zero exit, valid JSON — never a crash/traceback.
# ---------------------------------------------------------------------------
for label, fit_arg, card_arg in (
    ("unknown slug/model", "definitely-not-a-real-slug", "rtx3090"),
    ("non-vLLM slug (kvcalc SKIP)", "beellama/dflash", "rtx3090"),
    ("unrecognized card", "vllm/dual", "frobnicator-9000"),
):
    rc_u, out_u, err_u = run_fit("--fit", fit_arg, "--card", card_arg, "--json")
    check("Traceback" not in err_u,
          f"(d) {label}: no traceback on stderr (got {err_u.strip()!r})")
    try:
        du = json.loads(out_u)
        ok = (du.get("verdict") == "unknown" and isinstance(du.get("error"), str)
              and du["error"])
        check(ok, f"(d) {label}: emits {{verdict:unknown, error:str}} (got {du!r})")
    except Exception as exc:  # noqa: BLE001
        check(False, f"(d) {label}: output is valid JSON (got {exc}; out={out_u!r})")
    check(rc_u != 0, f"(d) {label}: exits non-zero (got {rc_u})")

# ---------------------------------------------------------------------------
# (e) --fit-all: ONE process emits a verdict for EVERY registry slug, shaped
#     {card, card_vram_gb, variants:{slug: <fit_verdict>}}. vLLM slugs get a
#     real verdict identical to the per-slug --fit; non-vLLM (kvcalc SKIP) get
#     {"verdict":"skip"}. This is the cockpit catalog's batched fit path —
#     one subprocess instead of N.
# ---------------------------------------------------------------------------
rc_a, out_a, err_a = run_fit("--fit-all", "--card", "rtx3090", "--json")
check(rc_a == 0, f"(e) --fit-all exits 0 (got {rc_a}; stderr={err_a.strip()!r})")
try:
    da = json.loads(out_a)
    parsed_a = True
except Exception as exc:  # noqa: BLE001
    da, parsed_a = {}, False
    check(False, f"(e) --fit-all output is valid JSON (got {exc}; out={out_a!r})")
if parsed_a:
    check(set(da.keys()) >= {"card", "card_vram_gb", "variants"},
          f"(e) top-level keys include card/card_vram_gb/variants (got {sorted(da.keys())})")
    variants = da.get("variants", {})
    check(isinstance(variants, dict) and len(variants) > 10,
          f"(e) variants is a non-trivial dict (got {len(variants) if isinstance(variants, dict) else variants!r})")
    check("vllm/dual" in variants and variants["vllm/dual"].get("verdict") in VERDICTS_OK,
          f"(e) vllm/dual has a real verdict in the batch (got {variants.get('vllm/dual')!r})")
    check(variants.get("vllm/dual") == json.loads(out),
          "(e) batch vllm/dual == per-slug --fit (one pricing path)")
    check(variants.get("beellama/dflash", {}).get("verdict") == "skip",
          f"(e) non-vLLM beellama/dflash → skip (got {variants.get('beellama/dflash')!r})")
    check(all(isinstance(v, dict) and isinstance(v.get("verdict"), str)
              for v in variants.values()),
          "(e) every variant value carries a verdict string")

# (f) required_sm gate — the arch floor surfaced in the fit verdict (the
# cockpit's hide/warn signal for hardware-incompatible slugs, e.g. NVFP4 on
# Ampere). Three contract points: below-floor card → incompatible-hw with
# required_sm/card_sm populated; at/above-floor card → a REAL priced verdict;
# bare-number --card carries no arch info → gate skipped (permissive).
rc_f, out_f, err_f = run_fit("--fit", "vllm/qwen-27b-dual-nvfp4", "--card", "rtx-3090", "--json")
check(rc_f == 0, f"(f) nvfp4 --card rtx-3090 exits 0 (got {rc_f})")
try:
    df = json.loads(out_f)
except Exception as exc:  # noqa: BLE001
    df = {}
    check(False, f"(f) nvfp4/3090 output is valid JSON (got {exc})")
check(df.get("verdict") == "incompatible-hw",
      f"(f) nvfp4 on rtx-3090 → incompatible-hw (got {df.get('verdict')!r})")
check(df.get("required_sm") == 9.0 and df.get("card_sm") == 8.6,
      f"(f) incompatible-hw carries required_sm=9.0/card_sm=8.6 (got {df!r})")
check("sm" in str(df.get("error", "")),
      f"(f) incompatible-hw error names the sm floor (got {df.get('error')!r})")

rc_g, out_g, _err_g = run_fit("--fit", "vllm/qwen-27b-dual-nvfp4", "--card", "rtx-5090", "--json")
dg = json.loads(out_g) if rc_g == 0 else {}
check(dg.get("verdict") in VERDICTS_OK and dg.get("verdict") != "incompatible-hw",
      f"(f) nvfp4 on rtx-5090 (sm 12.0) → real priced verdict (got {dg.get('verdict')!r})")

rc_h, out_h, _err_h = run_fit("--fit", "vllm/qwen-27b-dual-nvfp4", "--card", "64", "--json")
dh = json.loads(out_h) if rc_h == 0 else {}
check(dh.get("verdict") != "incompatible-hw",
      f"(f) bare-number --card 64 skips the sm gate — permissive (got {dh.get('verdict')!r})")

# batch parity: --fit-all --card rtx-3090 marks nvfp4 incompatible-hw too
rc_i, out_i, _err_i = run_fit("--fit-all", "--card", "rtx-3090", "--json")
di = json.loads(out_i) if rc_i == 0 else {"variants": {}}
check(di.get("variants", {}).get("vllm/qwen-27b-single-nvfp4", {}).get("verdict") == "incompatible-hw",
      "(f) --fit-all on rtx-3090 marks single-nvfp4 incompatible-hw")

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll --fit / --fit-all JSON shape assertions passed (a-f).")
PY

echo "test-kv-calc-fit.sh OK"
