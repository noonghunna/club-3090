#!/usr/bin/env bash
# CONTRACT — switch.sh --explain <slug> [--json].
#
# `switch.sh --explain <slug>` prints ONE slug's full story: its registry
# variant row joined with engine/model/hardware/drafter facts, the kv-calc fit
# verdict for the local card(s), and the measured BENCHMARKS.md row(s) if any.
# `--json` emits the same data as a structured object.
#
# This fixture asserts the SHAPE of the new emit (valid JSON, the expected
# top-level keys, the joined registry facts, a fit object, a benchmarks array)
# and exercises the assembly logic in isolation. The sibling kv-calc `--fit`
# contract is built in parallel; --explain degrades to a {"available": false}
# fit object when it isn't wired yet, so this test does NOT depend on it — its
# LIVE integration is asserted in the Guard phase. We pin a known slug from the
# shipped registry and force MODEL_DIR so the run is host-independent.
#
# Must NOT regress any existing switch.sh flag: we also confirm a few
# pre-existing invocations still behave (standalone --json still errors;
# --explain with no slug errors).
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Host-independent: pin MODEL_DIR and avoid leaning on whatever GPUs / .env pin
# this host carries. --explain is read-only (no container, no .env writes).
export MODEL_DIR="${MODEL_DIR:-/mnt/models/huggingface}"

SWITCH="$ROOT_DIR/scripts/switch.sh"

# A slug guaranteed present in the shipped registry, with a measured
# BENCHMARKS.md row, so we can assert the benchmarks join is non-empty.
SLUG="vllm/minimal"
SERVING="minimal.yml"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

assert_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" == *"$needle"* ]] || note "${msg}: output lacks '${needle}'"
}

# --- 1) --json emits a single well-formed JSON object on a CLEAN stdout -------
# (stderr carries the [switch] MODEL_DIR notice on the --json path so stdout
#  stays machine-parseable).
JSON_OUT="$(bash "$SWITCH" --explain "$SLUG" --json 2>/dev/null)" \
  || note "--explain $SLUG --json exited non-zero"

if ! printf '%s' "$JSON_OUT" | python3 -m json.tool >/dev/null 2>&1; then
  note "--explain --json stdout is not valid JSON"
  echo "----- stdout was: -----" >&2
  printf '%s\n' "$JSON_OUT" >&2
fi

# --- 2) Top-level + nested keys are present, and the registry facts joined ---
python3 - "$JSON_OUT" "$SLUG" "$SERVING" <<'PY' || note "JSON shape assertions failed"
import json
import sys

obj = json.loads(sys.argv[1])
slug, serving = sys.argv[2], sys.argv[3]

errs = []

# Top-level shape.
for k in ("slug", "registry", "card", "fit", "benchmarks"):
    if k not in obj:
        errs.append(f"missing top-level key {k!r}")

if obj.get("slug") != slug:
    errs.append(f"slug mismatch: {obj.get('slug')!r} != {slug!r}")

# Registry row — joined engine/model/hardware/drafter facts.
reg = obj.get("registry") or {}
for k in (
    "slug", "model", "engine", "topology", "weights_variant", "drafter",
    "kv_format", "tp", "max_ctx", "default_port", "status", "compose_path",
    "serving_file",
):
    if k not in reg:
        errs.append(f"registry missing key {k!r}")
if reg.get("serving_file") != serving:
    errs.append(f"serving_file {reg.get('serving_file')!r} != {serving!r}")
if reg.get("model") != "qwen3.6-27b":
    errs.append(f"model {reg.get('model')!r} != 'qwen3.6-27b'")

# Fit object — present, a dict, and (seam guard) carrying a REAL verdict.
fit = obj.get("fit")
if not isinstance(fit, dict):
    errs.append("fit is not a JSON object")
elif fit.get("verdict") not in ("fits-clean", "fits-constrained", "wont-fit"):
    # The sibling kv-calc --fit is live and the detected card
    # (explain_detect_card defaults to rtx-3090, which kv-calc catalogues) is
    # priceable, so a real verdict MUST surface — not the "unavailable" stub.
    # Catches the switch.sh<->kv-calc --card hyphenation seam.
    errs.append(f"fit verdict not surfaced (got {fit!r}); switch<->kv-calc --fit seam broken")

# Benchmarks — a JSON array; minimal.yml has measured rows, so it's non-empty
# and each entry has row/columns.
bench = obj.get("benchmarks")
if not isinstance(bench, list):
    errs.append("benchmarks is not a JSON array")
elif not bench:
    errs.append("benchmarks unexpectedly empty for a slug with measured rows")
else:
    for i, b in enumerate(bench):
        if "row" not in b or "columns" not in b:
            errs.append(f"benchmarks[{i}] missing row/columns")

if errs:
    for e in errs:
        print(f"  - {e}", file=sys.stderr)
    sys.exit(1)
PY

# --- 3) Human form is a readable block carrying the headline facts -----------
HUMAN_OUT="$(bash "$SWITCH" --explain "$SLUG" 2>/dev/null)" \
  || note "--explain $SLUG (human) exited non-zero"
assert_contains "$HUMAN_OUT" "$SLUG" "human block"
assert_contains "$HUMAN_OUT" "qwen3.6-27b" "human block (model)"
assert_contains "$HUMAN_OUT" "Config (registry):" "human block (config section)"
assert_contains "$HUMAN_OUT" "Fit verdict" "human block (fit section)"
assert_contains "$HUMAN_OUT" "BENCHMARKS.md" "human block (benchmarks section)"

# --- 4) Flag order independence: --json before the slug works too ------------
JSON_OUT2="$(bash "$SWITCH" --explain --json "$SLUG" 2>/dev/null)" \
  || note "--explain --json $SLUG (json-first order) exited non-zero"
printf '%s' "$JSON_OUT2" | python3 -m json.tool >/dev/null 2>&1 \
  || note "--explain --json <slug> (order swapped) is not valid JSON"

# --- 5) `<…>/default` token resolution: --explain vllm/default resolves ------
DEFAULT_OUT="$(bash "$SWITCH" --explain vllm/default --json 2>/dev/null)" \
  || note "--explain vllm/default --json exited non-zero"
printf '%s' "$DEFAULT_OUT" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); assert d.get("registry",{}).get("engine"), "no engine in resolved default"' \
  2>/dev/null || note "--explain vllm/default did not resolve to a concrete slug"

# --- 6) Error paths -----------------------------------------------------------
# Unknown slug → non-zero with a clear message.
if bash "$SWITCH" --explain bogus/nope >/dev/null 2>&1; then
  note "--explain on an unknown slug should exit non-zero"
fi
# --explain with no slug → non-zero.
if bash "$SWITCH" --explain >/dev/null 2>&1; then
  note "--explain with no <slug> should exit non-zero"
fi

# --- 7) Additivity guard: standalone --json (no --explain) still errors -------
# Pre-existing behavior — --json was an Unknown flag; it must remain so.
if bash "$SWITCH" --json >/dev/null 2>&1; then
  note "standalone --json (no --explain) should still be rejected"
fi

if [[ "$fail" -ne 0 ]]; then
  echo "[switch-explain] FAIL" >&2
  exit 1
fi
echo "[switch-explain] PASS: --explain ${SLUG} emits a well-formed story (registry + fit + benchmarks), human + --json, order-independent, default-token-resolving; error + additivity guards hold"
