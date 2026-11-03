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
# Every function prints exactly one of: vllm | llamacpp | sglang | exllamav3 | strata | qengine | unknown
# and returns 0. "unknown" is a value, not an error — callers must handle it
# rather than treating silence as a kind. Nothing here touches the network.
#
#   engine_kind_from_engine_id   <registry `engine:` value>   e.g. sglang-stable
#   engine_kind_from_container   <container name>             e.g. sglang-qwen38
#   engine_kind_from_image       <docker image ref>           e.g. lmsysorg/sglang:v0.5.19
#   engine_kind_from_fingerprint <system_fingerprint>         e.g. b10920-4df29be4f
#   engine_kind_from_owned_by    </v1/models owned_by>        e.g. llamacpp
#   engine_kind_from_build_info  </props build_info>          e.g. Strata ds4-0.1
#   engine_kind_from_slug        <catalog slug>               e.g. sgl/qwen38-27b-dual-max
#
# python3 here must not be locale-coerced (#779): a UTF-8 slug or engine id
# would otherwise raise UnicodeEncodeError under LC_ALL=C and the caller
# would silently read "unknown".
export PYTHONUTF8="${PYTHONUTF8:-1}"

# ⚠️ ADDING AN ENGINE: add its arms HERE and nowhere else. Everything else an
# engine touches is listed in docs/ADDING_ENGINES.md ("CONTRACT — wiring it into
# the tooling"). The guard `scripts/tests/test-engine-kind-resolver.sh` fails if
# a private classifier reappears in scripts/.

# Registry `engine:` id → family. Ids are prefixed by family everywhere in the
# catalog (vllm-stable, sglang-stable, llamacpp-club3090-v1.6), with two
# historical exceptions that are llama.cpp builds under their own names:
# `llama-cpp-local` (the mainline profile, engines/llama-cpp-local.yml) and
# `beellama-local`.
#
# An id no prefix arm recognises falls back to its engine profile's `type:`
# (what the engine behaves like). That covers a local engine registered with
# `catalog.sh register --engine my-build --engine-type llama.cpp`, and a
# derivative or fork named without its family prefix. The prefix arms run first,
# so the common path reads no files.
engine_kind_from_engine_id() {
  local kind
  case "${1:-}" in
    vllm*)                                   kind="vllm" ;;
    sglang*|sgl-*)                           kind="sglang" ;;
    exllamav3*|exl3*)                        kind="exllamav3" ;;
    strata*)                                 kind="strata" ;;
    qengine*)                                kind="qengine" ;;
    llama-cpp*|llamacpp*|ik-llama*|beellama*) kind="llamacpp" ;;
    *)                                       kind="unknown" ;;
  esac
  if [[ "$kind" == "unknown" && -n "${1:-}" ]]; then
    kind="$(_engine_kind_from_profile_type "$1")"
  fi
  echo "$kind"
  return 0
}

# The engine profile whose `id:` equals $1 → its `type:` → family. Core profiles
# first, then the local layer (profiles-local/engines.d), the loader's
# precedence. Matched on the `id:` line rather than the filename, so a misnamed
# profile still resolves (the profile-filename catalog rule flags one). `type` is not
# validated anywhere, so accept the spellings people write and map anything
# else to unknown.
_engine_kind_from_profile_type() {
  local want="$1" root dir f id type
  root="${ENGINE_KIND_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
  for dir in "$root/scripts/lib/profiles/engines" "$root/scripts/lib/profiles-local/engines.d"; do
    [[ -d "$dir" ]] || continue
    for f in "$dir"/*.yml; do
      [[ -f "$f" ]] || continue
      id="$(sed -nE "s/^id:[[:space:]]*[\"']?([^\"'#[:space:]]+)[\"']?.*$/\1/p" "$f" | head -n 1)"
      [[ "$id" == "$want" ]] || continue
      type="$(sed -nE "s/^type:[[:space:]]*[\"']?([^\"'#[:space:]]+)[\"']?.*$/\1/p" "$f" | head -n 1)"
      case "${type,,}" in
        vllm)                        echo "vllm" ;;
        sglang)                      echo "sglang" ;;
        exllamav3)                   echo "exllamav3" ;;
        strata)                      echo "strata" ;;
        qengine)                     echo "qengine" ;;
        llama.cpp|llama-cpp|llamacpp) echo "llamacpp" ;;
        *)                           echo "unknown" ;;
      esac
      return 0
    done
  done
  echo "unknown"
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
    tabbyapi-*|exl3-*|exllamav3-*)                 echo "exllamav3" ;;
    strata-*)                                      echo "strata" ;;
    qengine-*)                                     echo "qengine" ;;
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
    *strata*)                           echo "strata" ;;
    *qengine*)                          echo "qengine" ;;   # ghcr.io/bpx03/qengine
    *tabbyapi*|*exllamav3*)             echo "exllamav3" ;;
    *llama.cpp*|*llama-cpp*|*llamacpp*) echo "llamacpp" ;;
    *sglang*|*lmsysorg*)                echo "sglang" ;;
    *vllm*|*hyperqwen*)                 echo "vllm" ;;   # syv-ai/hyperqwen: a vLLM derivative whose image name omits "vllm"
    *)                                  echo "unknown" ;;
  esac
  return 0
}

# OpenAI `system_fingerprint`. vLLM emits "vllm-0.29.0-tp2-…"; llama-server
# emits its build string "b10920[-hash]". ⚠️ SGLang does NOT emit an
# sglang-prefixed fingerprint in every version — the arm is kept for the
# versions that do, but callers must fall back to container/image evidence
# rather than concluding "not sglang" from a miss (#1261). Strata sends a null
# fingerprint, so it has no arm here: its container name and image identify it.
# QEngine sends none either (its /v1/models owned_by does identify it).
engine_kind_from_fingerprint() {
  case "${1:-}" in
    vllm-*)   echo "vllm" ;;
    sglang-*) echo "sglang" ;;
    b[0-9]*)  echo "llamacpp" ;;
    *)        echo "unknown" ;;
  esac
  return 0
}

# OpenAI `/v1/models` → `data[0].owned_by`. A METADATA fingerprint: unlike
# `system_fingerprint` it costs no completion, so it is usable while the GPU
# is busy or before any request is sent. llama.cpp answers "llamacpp", SGLang
# "sglang", vLLM "vllm", tabbyAPI "tabbyAPI", QEngine "qengine". A proxy (LiteLLM) or cloud
# endpoint answers something else, and callers must treat that "unknown" as
# "cannot tell", not as any particular engine (#1383). Strata's /v1/models has no
# owned_by at all, so it has no arm here either: engine_kind_from_build_info names it.
engine_kind_from_owned_by() {
  case "${1,,}" in
    vllm*)                               echo "vllm" ;;
    sglang*)                             echo "sglang" ;;
    tabby*|exllama*)                     echo "exllamav3" ;;
    llamacpp*|llama.cpp*|llama-cpp*)     echo "llamacpp" ;;
    qengine*)                            echo "qengine" ;;
    *)                                   echo "unknown" ;;
  esac
  return 0
}

# GET /props → `build_info`. Strata's server, and Project Maya's (a Strata fork), answer
# "Strata <engine version>" ("Strata ds4-0.1" from Maya's DeepSeek engine), where llama.cpp answers
# its build string ("b10454-4df29be4f"). Their /v1/models has no owned_by and they set no
# system_fingerprint, so for a host-run Strata or Maya (CONTAINER=none) this is the only name the
# server gives itself; without it a /props answer read as llama.cpp (#1915). Only the Strata arm:
# a llama.cpp build string stays "unknown" here, and the callers' /props fallback decides it.
engine_kind_from_build_info() {
  case "${1,,}" in
    strata\ *|strata)                     echo "strata" ;;
    *)                                   echo "unknown" ;;
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
    from compose_registry import get_registry
    e = get_registry(sys.argv[2]).get(sys.argv[1])  # core + the local layer
except Exception:
    raise SystemExit
if e:
    print(e.get("engine") or "")
PY
)"
  [[ -n "$eng" ]] || { echo "unknown"; return 0; }
  engine_kind_from_engine_id "$eng"
  return 0
}
