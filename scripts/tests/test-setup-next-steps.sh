#!/usr/bin/env bash
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
#
# setup.sh's "Next — launch it" hint (scripts/lib/setup_next_steps.py) names the slug
# `setup.sh <slug>` was given (#1741: `setup.sh strata/qwen3.8-flash-next-iq3s-dual`
# suggested a llama.cpp slug), and keeps the default pick for `setup.sh <model>`.
#
#   1. every registry slug, passed as the slug: SAMPLE_SLUG is that slug, and the hint
#      carries --force exactly when its status is not production/caveats
#   2. the #1741 case, spelled out: the Strata slug, --force, the incubating note
#   3. no slug and an empty slug print the same (setup.sh <model> passes "")
#   4. a slug of ANOTHER model is not picked: the model's default stays
#   5. setup.sh hands SETUP_SLUG to the module (the wiring this all depends on)
#
# Reads the registry only: no network, no GPU, no download.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONUTF8="${PYTHONUTF8:-1}"
MOD="${ROOT_DIR}/scripts/lib/setup_next_steps.py"

fail() { echo "ASSERTION FAILED: $*" >&2; exit 1; }
ok() { echo "  ok  $*"; }
assert() { [[ "$1" == "$2" ]] || fail "$3: got '$1', want '$2'"; }   # <got> <want> <what>

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
REG="${TMP}/registry.json"
bash "${ROOT_DIR}/scripts/lib/registry-emit.sh" --json > "$REG"
[[ -s "$REG" ]] || fail "registry-emit.sh --json printed nothing"

# <model> [slug] → the module's assignments, evaluated; prints KEY=value lines
picked() {
  ( SAMPLE_SLUG=""; SAMPLE_LAUNCH_HINT=""; NEXT_STEPS_NOTE=""
    eval "$(python3 "$MOD" "$1" "$ROOT_DIR" "$REG" "${2-}")"
    printf 'SLUG=%s\nHINT=%s\nNOTE=%s\n' "$SAMPLE_SLUG" "$SAMPLE_LAUNCH_HINT" "$NEXT_STEPS_NOTE" )
}
field() { awk -F= -v k="$2" '$1 == k { print substr($0, length(k) + 2); exit }' <<< "$1"; }

# --- 1. every slug picks itself, with --force exactly when non-functional ------------
mapfile -t ROWS < <(python3 -c '
import json, sys
for v in json.load(open(sys.argv[1], encoding="utf-8"))["variants"]:
    print(v["model"], v["slug"], v.get("status") or "")
' "$REG")
(( ${#ROWS[@]} > 0 )) || fail "the registry emit lists no variants"
for row in "${ROWS[@]}"; do
  read -r model slug status <<< "$row"
  out="$(picked "$model" "$slug")"
  [[ "$(field "$out" SLUG)" == "$slug" ]] || fail "setup.sh $slug: hint picked '$(field "$out" SLUG)'"
  case "$status" in production|caveats) force="" ;; *) force="--force " ;; esac
  assert "$(field "$out" HINT)" "bash scripts/switch.sh ${force}${slug}" "setup.sh $slug ($status) hint"
done
ok "every one of ${#ROWS[@]} slugs: the hint names that slug, --force exactly when non-functional"

# --- 2. the #1741 case -----------------------------------------------------------------
out="$(picked qwen3.8-flash-next strata/qwen3.8-flash-next-iq3s-dual)"
assert "$(field "$out" HINT)" "bash scripts/switch.sh --force strata/qwen3.8-flash-next-iq3s-dual" "#1741 hint"
command grep -q "status 'incubating'" <<< "$out" || fail "#1741: no incubating note in: $out"
ok "#1741: setup.sh strata/qwen3.8-flash-next-iq3s-dual → switch.sh --force strata/qwen3.8-flash-next-iq3s-dual"

# --- 3. no slug == empty slug, for every model -------------------------------------------
mapfile -t MODELS < <(python3 "${ROOT_DIR}/scripts/lib/profiles/weights.py" catalog --json \
  | python3 -c 'import json, sys; [print(m["id"]) for m in json.load(sys.stdin)["models"]]')
for m in "${MODELS[@]}"; do
  [[ "$(python3 "$MOD" "$m" "$ROOT_DIR" "$REG")" == "$(python3 "$MOD" "$m" "$ROOT_DIR" "$REG" "")" ]] \
    || fail "$m: an empty slug changed the pick"
done
ok "${#MODELS[@]} models: no slug and an empty slug print the same"

# --- 4. another model's slug is not picked ----------------------------------------------
default_pick="$(field "$(picked qwen3.8-flash-next)" SLUG)"
[[ -n "$default_pick" ]] || fail "qwen3.8-flash-next has no default pick"
cross="$(field "$(picked qwen3.8-flash-next strata-maya/glm53-flash-dual-maya-s)" SLUG)"
[[ "$cross" == "$default_pick" ]] || fail "a GLM slug changed the Flash-Next pick to '$cross'"
ok "a slug of another model leaves the default pick ($default_pick)"

# --- 5. setup.sh passes the slug through ------------------------------------------------
command grep -qF 'scripts/lib/setup_next_steps.py" "${MODEL_NAME}" "${ROOT_DIR}" "${_reg_tmp}" "${SETUP_SLUG}"' \
  "${ROOT_DIR}/scripts/setup.sh" || fail "setup.sh does not hand SETUP_SLUG to setup_next_steps.py"
ok "setup.sh hands SETUP_SLUG to setup_next_steps.py"

echo "PASS test-setup-next-steps"
