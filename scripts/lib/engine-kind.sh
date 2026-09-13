#!/usr/bin/env bash
#
# engine-kind.sh — THE single source of truth for "which engine family is this?"
#
# Sourceable, stdlib-only (no python3, no registry emit) except for the one
# slug-based entry point, which delegates to registry-lookup.sh.
#
# WHY THIS FILE EXISTS (club-3090#1282, @paulp83)
# ----------------------------------------------
# Engine classification used to be re-implemented in every script that needed
# it. `spec-sweep.sh` carried a private TWO-valued version —
#     print("vllm" if eng.startswith("vllm") else "llamacpp")
# — so `sglang-stable` fell through to `llamacpp` and the sweep hunted for a
# llama.cpp server that was never there. That was the SIXTH site of a defect
# closed at five the same morning (#1263), and the five-site fix was itself five
# more private arms. The rules now live here once; scripts keep their own
# evidence-gathering (they genuinely have different evidence available) and
# delegate only the DECISION.
#
# CONTRACT
# --------
# Every function prints exactly one of: vllm | llamacpp | sglang | unknown
# and returns 0. "unknown" is a value, not an error — callers must handle it
# rather than treating silence as a kind. Nothing here touches the network.
#
#   engine_kind_from_engine_id   <registry `engine:` value>   e.g. sglang-stable
#   engine_kind_from_container   <container name>             e.g. sglang-qwen38
#   engine_kind_from_image       <docker image ref>           e.g. lmsysorg/sglang:v0.5.19
#   engine_kind_from_fingerprint <system_fingerprint>         e.g. b10920-4df29be4f
#   engine_kind_from_slug        <catalog slug>               e.g. sgl/qwen38-27b-dual-max
#
# python3 here must not be locale-coerced (#779): a UTF-8 slug or engine id
# would otherwise raise UnicodeEncodeError under LC_ALL=C and the caller
# would silently read "unknown".
export PYTHONUTF8="${PYTHONUTF8:-1}"

# ⚠️ ADDING AN ENGINE: add its arms HERE and nowhere else. The inventory of
# everything else an engine touches is in the maintainer tracker; the guard
# `scripts/tests/test-engine-kind-resolver.sh` fails if a private classifier
# reappears in scripts/.

# Registry `engine:` id → family. Ids are prefixed by family everywhere in the
# catalog (vllm-stable, sglang-stable, llamacpp-club3090-v1.6), with two
# historical exceptions that are llama.cpp builds under their own names:
# `llama-cpp-local` (the mainline profile's id — note the FILE is
# llama-cpp-mainline.yml; it is the only profile whose basename != its id) and
# `beellama-local`.
engine_kind_from_engine_id() {
  case "${1:-}" in
    vllm*)                                   echo "vllm" ;;
    sglang*|sgl-*)                           echo "sglang" ;;
    llama-cpp*|llamacpp*|ik-llama*|beellama*) echo "llamacpp" ;;
    *)                                       echo "unknown" ;;
  esac
  return 0
}

# Container-name convention. This is a SECOND taxonomy that parallels the
# engine ids — club3090-env.sh, rebench-full.sh and the launchers all name
# containers by family prefix. `sgl-*` is accepted alongside `sglang-*` because
# hand-rolled community runs use it (#1261).
engine_kind_from_container() {
  case "${1:-}" in
    vllm-*)                                        echo "vllm" ;;
    sglang-*|sgl-*)                                echo "sglang" ;;
    llama-cpp-*|llamacpp-*|ik-llama-*|beellama-*)  echo "llamacpp" ;;
    *)                                             echo "unknown" ;;
  esac
  return 0
}

# Docker image reference. ⚠️ Order matters: llama.cpp and SGLang are tested
# before vLLM because a vendor image can carry the vllm string in another
# family's ref (lmcache/vllm-openai is vLLM, but the general shape is why the
# narrower matches go first).
engine_kind_from_image() {
  local ref="${1:-}"
  case "$ref" in
    *llama.cpp*|*llama-cpp*|*llamacpp*) echo "llamacpp" ;;
    *sglang*|*lmsysorg*)                echo "sglang" ;;
    *vllm*)                             echo "vllm" ;;
    *)                                  echo "unknown" ;;
  esac
  return 0
}

# OpenAI `system_fingerprint`. vLLM emits "vllm-0.29.0-tp2-…"; llama-server
# emits its build string "b10920[-hash]". ⚠️ SGLang does NOT emit an
# sglang-prefixed fingerprint in every version — the arm is kept for the
# versions that do, but callers must fall back to container/image evidence
# rather than concluding "not sglang" from a miss (#1261).
engine_kind_from_fingerprint() {
  case "${1:-}" in
    vllm-*)   echo "vllm" ;;
    sglang-*) echo "sglang" ;;
    b[0-9]*)  echo "llamacpp" ;;
    *)        echo "unknown" ;;
  esac
  return 0
}

# Catalog slug → registry `engine:` → family. The only entry point that needs
# the registry; prints "unknown" when the slug is absent or the registry cannot
# be consulted, so callers keep one code path.
engine_kind_from_slug() {
  local slug="${1:-}" root eng
  [[ -n "$slug" ]] || { echo "unknown"; return 0; }
  root="${ENGINE_KIND_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
  eng="$(python3 - "$slug" "$root" <<'PY' 2>/dev/null || true
import sys, os
sys.path.insert(0, os.path.join(sys.argv[2], "scripts/lib/profiles"))
try:
    from compose_registry import COMPOSE_REGISTRY
except Exception:
    raise SystemExit
e = COMPOSE_REGISTRY.get(sys.argv[1])
if e:
    print(e.get("engine") or "")
PY
)"
  [[ -n "$eng" ]] || { echo "unknown"; return 0; }
  engine_kind_from_engine_id "$eng"
  return 0
}
