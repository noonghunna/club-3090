#!/usr/bin/env bash
# test-act-format-drift — every slug's registry `act_format` must match the activation format its
# compose ACTUALLY serves by default.
#
# WHY THIS TEST EXISTS
# --------------------
# The c3 catalog "act" column reads the registry `act_format` facet; nothing tied that facet to the
# compose. On 2026-09-19 the twelve SGLang AutoRound slugs (sgl/qwen38-27b-* and sgl/thinkingcap38-27b-*
# fast/superfast tiers) flipped their W4A8 default to ON (`"${W4A8:-1}"` in the entrypoint), but the
# registry kept `act_format: 16bit` and each header kept a paragraph saying "OPT-IN … OFF by default".
# c3 showed 16bit for five days while the slugs served int8 activations. This guard derives the
# expected format from the compose itself, in the same four shapes the catalog uses:
#   1. a W4A8 gate — `{W4A8:-1}` / `{W4A8:-0}` (vLLM env line or SGLang entrypoint): 1 -> int8, 0 -> 16bit;
#   2. a hardcoded humming int8 input config (`- 'VLLM_HUMMING_INPUT_QUANT_CONFIG={"dtype":"int8"}'`): int8;
#   3. a bare `- VLLM_HUMMING_INPUT_QUANT_CONFIG` passthrough with no gate (opt-in, unset by default): 16bit;
#   4. no activation knob at all: 16bit.
# and, for a gate that defaults ON, refuses the retired "OPT-IN / OFF by default" header wording.
# It also fails any vllm-stable compose that still FORWARDS VLLM_MARLIN_INPUT_DTYPE: vLLM v0.34.0
# removed Marlin's 8-bit activations (#65542) and refuses to start if that variable is set at all,
# so a forwarded host value would stop the boot. (Until v0.33.0 shapes 2 and 3 were spelled with it.)
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

out="$(bash scripts/lib/registry-emit.sh --json 2>/dev/null | python3 -c '
import json, os, re, sys
STALE = ("W4A8 (int8 activations) — OPT-IN",
         "It is OFF by default because it is a PREFILL",
         "# --- W4A8 opt-in (patch from club-3090#1226")
# The W4A8 default is written ${W4A8:-N} (use-default) or, in auto-sizing entrypoints such as
# single-fast, : "$${W4A8:=N}" (assign-default, #1598). Both set the default; read both.
GATE = re.compile(r"\{W4A8:[-=]([01])\}")
assert GATE.findall("$${W4A8:-1}") == ["1"] and GATE.findall(": $${W4A8:=1} $${SPEC_N:=0}") == ["1"], "gate regex lost a form"
assert GATE.findall("$${W4A8:=0}") == ["0"] and not GATE.findall("$${W4A8}"), "gate regex reads a default that is not there"
HUM_INT8 = re.compile(r"(?m)^\s*-\s*\x27?VLLM_HUMMING_INPUT_QUANT_CONFIG=\{\"dtype\":\s*\"int8\"\}\x27?\s*$")
HUM_BARE = re.compile(r"(?m)^\s*-\s*VLLM_HUMMING_INPUT_QUANT_CONFIG\s*$")
MARLIN = re.compile(r"(?m)^\s*-\s*VLLM_MARLIN_INPUT_DTYPE\b")
assert HUM_INT8.search("      - \x27VLLM_HUMMING_INPUT_QUANT_CONFIG={\"dtype\":\"int8\"}\x27"), "humming int8 regex lost its shape"
assert HUM_BARE.search("      - VLLM_HUMMING_INPUT_QUANT_CONFIG") and not HUM_BARE.search("      - VLLM_HUMMING_INPUT_QUANT_CONFIG=x"), "bare regex"
assert MARLIN.search("      - VLLM_MARLIN_INPUT_DTYPE") and MARLIN.search("      - VLLM_MARLIN_INPUT_DTYPE=int8"), "marlin regex lost a form"
fails, n_gate, n = [], 0, 0
for v in json.load(sys.stdin)["variants"]:
    p, slug, act = v["compose_path"], v["slug"], v.get("act_format")
    if not os.path.isfile(p):
        continue
    n += 1
    s = open(p, encoding="utf-8").read()
    code = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
    if v.get("engine") == "vllm-stable" and v.get("status") != "deprecated" and MARLIN.search(code):
        fails.append(f"{slug}: {p} forwards VLLM_MARLIN_INPUT_DTYPE, which vLLM v0.34.0 refuses at startup")
    gate = set(GATE.findall(code))
    if len(gate) > 1:
        fails.append(f"{slug}: W4A8 default disagrees with itself inside {p} ({sorted(gate)})"); continue
    if gate:
        n_gate += 1
        dflt = gate.pop()
        want, why = ("int8", "W4A8 gate defaults ON") if dflt == "1" else ("16bit", "W4A8 gate defaults OFF")
        if dflt == "1":
            for phrase in STALE:
                if phrase in s:
                    fails.append(f"{slug}: {p} defaults W4A8 ON but still says {phrase!r}")
    elif HUM_INT8.search(code):
        want, why = "int8", "hardcoded humming int8 input config"
    elif HUM_BARE.search(code):
        want, why = "16bit", "bare VLLM_HUMMING_INPUT_QUANT_CONFIG passthrough (opt-in, unset by default)"
    else:
        want, why = "16bit", "no activation knob"
    if act != want:
        fails.append(f"{slug}: registry act_format={act!r} but {p} serves {want} by default ({why})")
for f in fails:
    print("FAIL " + f)
print(f"SUMMARY {n} {n_gate} {len(fails)}")
' 2>&1)"
summary="$(printf '%s\n' "$out" | command grep -E '^SUMMARY ' | tail -1)"
if [[ -z "$summary" ]]; then
  echo "✗ test-act-format-drift: could not evaluate the registry (registry-emit.sh --json / python failed)" >&2
  printf '%s\n' "$out" | tail -5 >&2
  exit 1
fi
read -r _ n n_gate nfail <<< "$summary"
printf '%s\n' "$out" | command grep -E '^FAIL ' | sed 's/^FAIL /✗ /' >&2
[[ "$n_gate" -ge 30 ]] || { echo "✗ only $n_gate composes carry a W4A8 gate — the gate regex no longer matches (expected >= 30)" >&2; nfail=$((nfail+1)); }
if [[ "$nfail" -gt 0 ]]; then
  echo "test-act-format-drift: $nfail failure(s) across $n slugs" >&2
  exit 1
fi
echo "test-act-format-drift: ok ($n slugs; registry act_format matches the compose default, $n_gate via a W4A8 gate)"
