#!/usr/bin/env bash
#
# Guard: engine classification is resolved in ONE place, and every consumer
# agrees with it.
#
# Why this exists (club-3090#1282, @paulp83): `spec-sweep.sh` carried its own
# private two-valued classifier —
#     print("vllm" if eng.startswith("vllm") else "llamacpp")
# — so `sglang-stable` fell through to `llamacpp` and the sweep went hunting for
# a llama.cpp server. That was the SIXTH site of a defect we had closed at five
# the same morning (#1263). The fix for five sites was five more private arms;
# this test exists so the seventh site cannot happen quietly.
#
# ⚠️ Arms 1-2 must FAIL against the pre-fix tree. If they pass before the fix
# they are asserting the wrong thing.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="test-engine-kind-resolver"
FAIL=0
bad() { echo "FAIL: $1 — expected '$2', got '$3'" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

LIB="${ROOT}/scripts/lib/engine-kind.sh"

# --- 1: the canonical resolver exists and is the single source of truth ------
if [[ ! -f "$LIB" ]]; then
  bad "canonical resolver" "scripts/lib/engine-kind.sh to exist" "missing"
else
  # shellcheck source=lib/engine-kind.sh
  source "$LIB"
  for probe in \
    "engine_kind_from_engine_id sglang-stable:sglang" \
    "engine_kind_from_engine_id vllm-stable:vllm" \
    "engine_kind_from_engine_id llama-cpp-local:llamacpp" \
    "engine_kind_from_engine_id llamacpp-club3090-v1.6:llamacpp" \
    "engine_kind_from_engine_id beellama-local:llamacpp" \
    "engine_kind_from_engine_id totally-new-engine:unknown" \
    "engine_kind_from_container sglang-qwen38:sglang" \
    "engine_kind_from_container sgl-qwen38:sglang" \
    "engine_kind_from_container vllm-qwen36-27b:vllm" \
    "engine_kind_from_container llama-cpp-glm53:llamacpp" \
    "engine_kind_from_container ik-llama-qwen:llamacpp" \
    "engine_kind_from_container nothing-like-an-engine:unknown" \
    "engine_kind_from_image lmsysorg/sglang:v0.5.19:sglang" \
    "engine_kind_from_image vllm/vllm-openai:v0.29.0:vllm" \
    "engine_kind_from_image ghcr.io/ggml-org/llama.cpp:llamacpp" \
    "engine_kind_from_fingerprint sglang-0.5.19:sglang" \
    "engine_kind_from_fingerprint vllm-0.29.0-tp2:vllm" \
    "engine_kind_from_fingerprint b10920-4df29be4f:llamacpp" \
    "engine_kind_from_owned_by llamacpp:llamacpp" \
    "engine_kind_from_owned_by vllm:vllm" \
    "engine_kind_from_owned_by sglang:sglang" \
    "engine_kind_from_owned_by tabbyAPI:exllamav3" \
    "engine_kind_from_owned_by openai:unknown" \
  ; do
    want="${probe##*:}"; call="${probe%:*}"
    got="$($call 2>/dev/null || true)"
    [[ "$got" == "$want" ]] || bad "resolver: $call" "$want" "$got"
  done
  [[ $FAIL -eq 0 ]] && ok "canonical resolver maps every known engine id, container, image, fingerprint and owned_by"
fi

# --- 2: spec-sweep.sh must classify an SGLang slug as sglang (#1282) ---------
# Drive the real script against a dead URL so it prints its classification and
# exits before touching a server. The banner line is the observable.
#
# ⚠️⚠️ SWEEP_DRY=1 IS LOAD-BEARING, NOT DECORATION. The banner prints BEFORE the
# dry-run check, so without it the script sails past and does the real thing: for
# any non-llamacpp family that means `switch.sh --force <slug>` per arm, i.e. this
# "resolver" unit test BOOTS A 27B MODEL on the rig's GPUs. `timeout 120` does not
# save you — it TERMs spec-sweep only, and the switch.sh GRANDCHILD survives and
# keeps launching after the test has moved on. Observed 2026-09-18: a suite run
# left `sglang-qwen38-27b-max-dual` serving on :8145 with the test long gone, and
# the 120s timeout had been exceeded threefold. `| head -40` does not help either
# — nothing sends SIGPIPE while the child is quiet.
out="$(cd "$ROOT" && SWEEP_DRY=1 SWEEP_N="0 1" SLUG=sgl/qwen38-27b-dual-max \
        URL=http://127.0.0.1:9 timeout 120 bash scripts/spec-sweep.sh 2>&1 | head -40 || true)"
line="$(command grep -oE '\[spec-sweep\] engine=[a-z]+' <<<"$out" | head -1)"
case "$line" in
  *engine=sglang) ok "spec-sweep resolves an sglang slug to engine=sglang (#1282)" ;;
  "")             bad "spec-sweep classification" "a '[spec-sweep] engine=' banner" "no banner (script died earlier)" ;;
  *)              bad "spec-sweep classification" "engine=sglang" "${line##*engine=}" ;;
esac

# --- 3: controls — the other two kinds must NOT regress ---------------------
for pair in "vllm/minimal:vllm" "llamacpp/default:llamacpp"; do
  slug="${pair%:*}"; want="${pair##*:}"
  # SWEEP_DRY=1: same reason as arm 2 — without it the vllm control boots a model.
  out="$(cd "$ROOT" && SWEEP_DRY=1 SWEEP_N="0 1" SLUG="$slug" URL=http://127.0.0.1:9 \
          timeout 120 bash scripts/spec-sweep.sh 2>&1 | head -40 || true)"
  got="$(command grep -oE '\[spec-sweep\] engine=[a-z]+' <<<"$out" | head -1)"; got="${got##*engine=}"
  [[ "$got" == "$want" ]] || bad "spec-sweep control $slug" "$want" "${got:-<no banner>}"
done
[[ $FAIL -eq 0 ]] && ok "vllm and llamacpp slugs still classify correctly"

# --- 4: no script may re-implement the mapping privately --------------------
# This is the arm that stops a seventh site. A new private classifier is any
# vllm/llamacpp/sglang literal decision outside the lib and its own test.
# Scope: executable lines only (comments explaining the OLD code are fine), and
# only the non-test, non-lib scripts — scripts/tests/ legitimately asserts on
# engine strings, and engine-kind.sh IS the implementation.
priv="$(command grep -rnE 'startswith\("vllm"\)|"vllm" if ' \
          "${ROOT}/scripts" --include='*.sh' 2>/dev/null \
          | command grep -v '/scripts/tests/' \
          | command grep -v '/scripts/lib/engine-kind.sh' \
          | command grep -vE ':[0-9]+:[[:space:]]*#' || true)"
if [[ -n "$priv" ]]; then
  bad "private classifier re-implementation" "none outside scripts/lib/engine-kind.sh" "$priv"
else
  ok "no script re-implements the engine mapping privately"
fi

# --- 5: no script may hand-list CONTAINER NAME PREFIXES ---------------------
# The sibling defect to arm 4. Arm 4 polices "which engine is this?"; this one
# polices "is this container ours at all?" — a different question that was ALSO
# copy-pasted per script, so every new engine silently fell out of every copy.
#
# #281 already fixed this once, in switch.sh: teardown used a fixed
# `^(vllm-|llama-cpp-)` regex, missed beellama-/ik-llama-/sglang- containers and
# leaked their VRAM across switches. The other copies were never converted, so by
# 2026-09-18 report.sh's filter was missing BOTH sglang- and exl3, and health.sh
# was missing exl3 — a rig serving exl3 was told "no engine container running"
# over a healthy server. "Not found" reads exactly like "not there".
#
# The set is registry-derived in scripts/lib/club-containers.sh.
# ⚠️ TWO SHAPES. The first pass of this arm only knew the `--filter 'name=X-'`
# form and therefore missed soak-test.sh, which hand-listed the SAME set as a
# grep alternation `^(vllm-|llama-cpp-|...)`. A gate that knows one spelling of a
# copy-paste is a gate that certifies the other spelling as clean.
hand="$(command grep -rnE "name=(vllm|llama-cpp|beellama|ik-llama|sglang|tabbyapi)-|\\^\\((vllm|llama-cpp|ik-llama|sglang|beellama)-\\|" \
          "${ROOT}/scripts" --include='*.sh' 2>/dev/null \
          | command grep -v '/scripts/tests/' \
          | command grep -v '/scripts/lib/club-containers.sh' \
          | command grep -vE ':[0-9]+:[[:space:]]*#' \
          | command grep -vE "name=vllm-qwen36" || true)"
if [[ -n "$hand" ]]; then
  bad "hand-listed container prefixes" \
      "discovery via club_running_container / club_container_re" "$hand"
else
  ok "no script hand-lists container-name prefixes (registry-derived discovery)"
fi

if [[ $FAIL -ne 0 ]]; then echo "FAIL: $NAME" >&2; exit 1; fi
echo "PASS: $NAME (centralised engine-kind resolver, #1282)"
