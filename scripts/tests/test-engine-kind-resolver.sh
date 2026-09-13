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
  ; do
    want="${probe##*:}"; call="${probe%:*}"
    got="$($call 2>/dev/null || true)"
    [[ "$got" == "$want" ]] || bad "resolver: $call" "$want" "$got"
  done
  [[ $FAIL -eq 0 ]] && ok "canonical resolver maps every known engine id, container, image and fingerprint"
fi

# --- 2: spec-sweep.sh must classify an SGLang slug as sglang (#1282) ---------
# Drive the real script against a dead URL so it prints its classification and
# exits before touching a server. The banner line is the observable.
out="$(cd "$ROOT" && SWEEP_N="0 1" SLUG=sgl/qwen38-27b-dual-max \
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
  out="$(cd "$ROOT" && SWEEP_N="0 1" SLUG="$slug" URL=http://127.0.0.1:9 \
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

if [[ $FAIL -ne 0 ]]; then echo "FAIL: $NAME" >&2; exit 1; fi
echo "PASS: $NAME (centralised engine-kind resolver, #1282)"
