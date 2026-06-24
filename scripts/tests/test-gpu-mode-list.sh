#!/usr/bin/env bash
# Serve-cockpit — gpu-mode --list-modes scene-catalog emitter.
#
# `gpu-mode --list-modes` prints a human scene catalog; `--list-modes --json`
# emits the machine-readable [{name,group,description,services,ports,gpus}]
# array the cockpit consumes. This test exercises the new emit WITHOUT touching
# GPUs/docker (the catalog is static data derived from the dispatch case) and
# asserts its SHAPE: valid JSON, the six required keys per row, the contract's
# three groups, services/ports as arrays, and that every dispatch keyword that
# should appear is present in the correct group. It also guards the strictly-
# additive contract: an unknown mode still falls through to usage().
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GPU_MODE="$ROOT_DIR/scripts/gpu-mode.sh"

fail=0
note() { echo "FAIL: $1" >&2; fail=1; }

assert_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" == *"$needle"* ]] || note "${msg}: output lacks '${needle}'"
}
assert_not_contains() {
  local hay="$1" needle="$2" msg="$3"
  [[ "$hay" != *"$needle"* ]] || note "${msg}: output unexpectedly contains '${needle}'"
}

# --- 1. --list-modes --json is valid JSON with the contract shape -----------
JSON_OUT="$(bash "$GPU_MODE" --list-modes --json 2>&1)"

JSON_OUT="$JSON_OUT" python3 - <<'PY' || note "JSON shape assertion failed"
import json, os, sys

raw = os.environ["JSON_OUT"]
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    print(f"  not valid JSON: {e}", file=sys.stderr)
    sys.exit(1)

ok = True
def bad(msg):
    global ok
    print(f"  {msg}", file=sys.stderr)
    ok = False

if not isinstance(data, list) or not data:
    bad("top-level is not a non-empty array")
    sys.exit(0 if ok else 1)

REQUIRED = {"name", "group", "description", "services", "ports", "gpus"}
GROUPS = {"models", "studio", "ops"}
# Every dispatch keyword the catalog must expose, with its contract group.
# (serving → models; chat → ops — browser-chat infra, no GPU model.  bigmodel +
#  diffusiongemma scenes were removed — bigmodel ≈ off, dgemma → catalog slug.  The
#  standalone comfyui scene was removed — comfyui runs via ai-studio.)
EXPECT = {
    "27b": "models", "gemma": "models", "deckard": "models",
    "ai-studio": "studio",
    "chat": "ops", "off": "ops", "power-cap": "ops", "prune": "ops", "prune-all": "ops",
}

seen = {}
for i, row in enumerate(data):
    if not isinstance(row, dict):
        bad(f"row {i} is not an object")
        continue
    missing = REQUIRED - set(row)
    if missing:
        bad(f"row {row.get('name', i)} missing keys: {sorted(missing)}")
    if row.get("group") not in GROUPS:
        bad(f"row {row.get('name', i)} has non-contract group {row.get('group')!r}")
    if not isinstance(row.get("services"), list):
        bad(f"row {row.get('name', i)} services is not an array")
    if not isinstance(row.get("ports"), list):
        bad(f"row {row.get('name', i)} ports is not an array")
    if not isinstance(row.get("description"), str) or not row["description"]:
        bad(f"row {row.get('name', i)} description empty/non-string")
    seen[row.get("name")] = row.get("group")

for name, grp in EXPECT.items():
    if name not in seen:
        bad(f"expected mode {name!r} absent from catalog")
    elif seen[name] != grp:
        bad(f"mode {name!r} in group {seen[name]!r}, expected {grp!r}")

# Spot-check that a known mode carries real service/port data.
chat = next((r for r in data if r["name"] == "chat"), None)
if chat is None or "litellm" not in chat["services"] or "8080" not in chat["ports"]:
    bad("chat row missing expected litellm service / 8080 port")

sys.exit(0 if ok else 1)
PY

# --- 2. plain --list-modes renders the grouped catalog ----------------------
PLAIN_OUT="$(bash "$GPU_MODE" --list-modes 2>&1)"
assert_contains "$PLAIN_OUT" "Scene Catalog" "plain render has catalog header"
assert_contains "$PLAIN_OUT" "[models]"      "plain render groups models"
assert_contains "$PLAIN_OUT" "[studio]"      "plain render groups studio"
assert_contains "$PLAIN_OUT" "[ops]"         "plain render groups ops"
assert_contains "$PLAIN_OUT" "deckard"       "plain render lists a models mode"

# --- 3. additive guard: unknown mode still prints usage ---------------------
USAGE_OUT="$(bash "$GPU_MODE" definitely-not-a-mode 2>&1)"
assert_contains     "$USAGE_OUT" "GPU Mode Switcher" "unknown mode falls through to usage"
assert_not_contains "$USAGE_OUT" "Scene Catalog"     "usage does not leak catalog output"

if [[ "$fail" -ne 0 ]]; then
  echo "[gpu-mode-list] FAIL" >&2
  exit 1
fi
echo "test-gpu-mode-list: ok"
