#!/usr/bin/env bash
#
# Model-aware one-shot setup for club-3090.
#
#   bash scripts/setup.sh                # interactive model picker in a TTY
#   bash scripts/setup.sh <model-name>   # scripted/CI positional form
#   bash scripts/setup.sh <slug>         # everything one launch slug needs: its weights
#                                        # + its drafter / vision projector companions
#
# Supported model families are DERIVED from scripts/lib/profiles/models/*.yml
# (via weights.py `catalog --json`) — never hand-listed here. Exact repositories,
# files, and local subdirectories live in those same YAMLs.
#
# What it does (per supported model):
#   - downloads model weights into $MODEL_DIR with SHA256 verification against
#     the hub's published x-linked-etag (no-redirect resolve HEAD via
#     scripts/lib/profiles/hf_fetch.py). A file the hub publishes no hash for is
#     reported UNVERIFIED and EXCLUDED from the verified count — "verified"
#     never attaches to a size-only check (#857)
#   - downloads the always-required drafter (Gemma 4: MTP "assistant"; Qwen3.6:
#     no always-required drafter — DFlash is optional via WITH_DFLASH_DRAFT=1;
#     Qwen3.8: DFlash2 optional via WITH_DFLASH_DRAFT=1 — REQUIRED by the 12
#     super*/ultra* slugs, so set it when serving those)
#
# Env vars (optional):
#   MODEL_DIR           Where to place model weights. Default: <repo>/models-cache
#   WEIGHTS             'autoround' (default, vLLM INT4) or 'gguf' (llama.cpp /
#                       ik_llama). gguf fetches the Q4_K_M MTP GGUF + mmproj for
#                       the llamacpp/* + ik-llama/* composes (qwen3.6-27b only).
#                       Use this if you're serving via llama.cpp/ik_llama rather than vLLM.
#   SKIP_MODEL          Set to 1 to skip the model download step
#   HF_TOKEN            HF token (public models, usually unnecessary)
#   WITH_ASSISTANT_DRAFT  Set to 1 to ALSO download the model's MTP "assistant"
#                       drafter when its profile registers one (setup:
#                       assistant_draft). Default: 0 — EXCEPT where the profile
#                       marks the drafter always_draft, in which case it is
#                       fetched unconditionally and this flag is redundant.
#                       ⚠️ vllm/gemma-26ba4b-single REQUIRES it: the slug
#                       defaults to SPEC_N=4, so without the drafter on disk
#                       vLLM aborts on a missing config.json. Either set this
#                       flag, or boot the slug drafter-less with SPEC_N=0.
#   WITH_DFLASH_DRAFT   Set to 1 to ALSO download the model family's DFlash
#                       drafter when one is registered in profiles/models/*.yml.
#                       Default: 0.
#                       Note: draft model is still under training as of
#                       2026-04-26; bench numbers in DUAL_CARD.md were
#                       measured against that snapshot. AL improvements
#                       expected when z-lab tags training-complete.
#   WITH_VISION         Set to 1 to ALSO download the F16 mmproj vision projector
#                       when the model registers one (qwen3.8-27b: the companion of
#                       the llamacpp/qwen38-27b-single-iq4xs slug, which ships
#                       q4/262K/vision). Default: 0. c3's Download pulls it via the
#                       slug's weights_companions regardless.
#   WITH_PRISM_EAGLE3   Set to 1 to ALSO download the Prism EAGLE3 drafter when
#                       the model registers one (setup: prism_eagle3).
#                       Default: 0.
#   HF_DOWNLOAD_RETRIES Attempts per repo download before giving up. Default: 4.
#                       A transfer is resumable, so a retry costs only the
#                       reconnect; raise it on a flaky link.
#   HF_DOWNLOAD_RETRY_SLEEP
#                       Seconds before the first retry, doubling up to 120.
#                       Default: 8.
#   PREFLIGHT_DISK_GB   Required free space at MODEL_DIR. Default: derived from
#                       the size_gb of every key this run would actually fetch
#                       (already-present weights cost nothing) + headroom.
#
# Idempotent: safe to re-run — skips steps already done.

set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_READER="${ROOT_DIR}/scripts/lib/profiles/weights.py"

# ---------- Catalog derivation (registry-derived front door) ----------
# The profile YAMLs are the single source of truth. Everything that used to be
# a hand-written bash case here (usage list, model labels, the picker, the
# dispatch, the WEIGHTS= aliases, the "Supported:" error) is derived from
#   python3 "$WEIGHTS_READER" catalog --json
# so a new scripts/lib/profiles/models/<id>.yml gets its front door for free.
# Parity is guarded by scripts/tests/test-setup-registry-derived.sh — the drift
# guard that replaces these hand-maintained lists (#910-#914 defect class).
_CATALOG_JSON=""

catalog_json() {
  if [[ -z "${_CATALOG_JSON}" ]]; then
    command -v python3 >/dev/null 2>&1 || {
      echo "ERROR: python3 is required to read the profile catalog." >&2
      exit 1
    }
    _CATALOG_JSON="$(python3 "${WEIGHTS_READER}" catalog --json)" || {
      echo "ERROR: could not read the model catalog (scripts/lib/profiles/models/*.yml)." >&2
      echo "       Install python3-yaml/PyYAML if missing, or check the profiles." >&2
      exit 1
    }
  fi
  printf '%s' "${_CATALOG_JSON}"
}

# _catalog_py <python-expr> [arg] — evaluate EXPR with `data` bound to the
# parsed catalog JSON and sys.argv[2] = ARG. Scalars print verbatim; lists
# print newline-joined; None prints as empty.
_catalog_py() {
  python3 -c '
import json, sys
data = json.loads(sys.stdin.read())
val = eval(sys.argv[1], {"data": data, "sys": sys})
if val is None:
    print("")
elif isinstance(val, str):
    print(val)
else:
    print("\n".join(str(v) for v in val))
' "$1" "${2:-}" < <(catalog_json)
}

# read_both_models <array-name> — the `both` mode's model pair, derived from
# RECOMMENDED_DEFAULT_MODELS in compose_registry.py (first two entries) like
# everything else. Fewer than two entries => `both` has no pair and stays
# disabled (the shortlist is maintainer-owned; it shrank to one entry when the
# single-card qwen3.6-27b defaults were retired).
read_both_models() {
  local -n _out="$1"
  _out=()
  local _m
  while IFS= read -r _m; do
    [[ -n "${_m}" ]] && _out+=("${_m}")
  done < <(CLUB3090_PROFILES_DIR="${ROOT_DIR}/scripts/lib/profiles" python3 -c '
import os, sys
sys.path.insert(0, os.environ["CLUB3090_PROFILES_DIR"])
from compose_registry import RECOMMENDED_DEFAULT_MODELS
for _m in list(RECOMMENDED_DEFAULT_MODELS)[:2]:
    print(_m)
' 2>/dev/null)
}

usage() {
  echo "Usage: $0 <model-name>"
  echo "       $0 <slug>       # a launch slug's weights + its drafter / projector companions"
  echo "       $0              # interactive model picker in a TTY"
  echo ""
  echo "Run with no model name in a normal terminal to open the hardware-aware"
  echo "model picker. Use the positional form in scripts/CI to skip prompts."
  echo ""
  echo "Supported model names (derived from scripts/lib/profiles/models/*.yml):"
  local _id
  while IFS= read -r _id; do
    echo "  ${_id}"
  done < <(_catalog_py "[m['id'] for m in data['models']]")
  echo ""
  echo "Exact catalog entry fetch: WEIGHT_KEY=<registry-key> $0 <model-name>"
  echo "Slug fetch without its companions: WEIGHT_EXTRA_KEYS= $0 <slug>"
  echo ""
  # These are OPTIONAL downloads, and a slug that needs one fails at switch.sh
  # time with the engine's own error — which names neither the flag nor the way
  # out (club-3090#1304). A flag documented only in this file's header comment
  # is a flag you have to already know about to find, so they are listed here,
  # where a stuck user actually looks. Guarded by
  # scripts/tests/test-setup-optin-flags-documented.sh.
  echo "Optional extra downloads (set to 1):"
  echo "  WITH_ASSISTANT_DRAFT  MTP \"assistant\" drafter, when the model registers one."
  echo "                        REQUIRED by vllm/gemma-26ba4b-single — that slug defaults"
  echo "                        to SPEC_N=4, so without the drafter on disk vLLM aborts on"
  echo "                        a missing config.json."
  echo "  WITH_DFLASH_DRAFT     DFlash drafter, when the model family registers one."
  echo "                        REQUIRED by the qwen3.8-27b super*/ultra* slugs."
  echo "  WITH_VISION           F16 mmproj vision projector, when the model registers one."
  echo "  WITH_PRISM_EAGLE3     Prism EAGLE3 drafter, when the model registers one."
  echo ""
  echo "Missing drafter, and you would rather not re-download? Every vLLM compose"
  echo "honours SPEC_N=0 (alias SPEC=off) to boot with speculative decoding disabled:"
  echo "  SPEC_N=0 bash scripts/switch.sh <slug> --force"
}

model_label() {
  _catalog_py "next((m['display_name'] for m in data['models'] if m['id'] == sys.argv[2]), sys.argv[2])" "$1"
}


load_weight_recipe() {
  local key="$1"
  local env_lines
  command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 is required to read profile weight recipes." >&2
    exit 1
  }
  env_lines="$(python3 "${WEIGHTS_READER}" entry "$key" 2>/dev/null)" || {
    echo "ERROR: could not resolve weight recipe '${key}' from scripts/lib/profiles/models/*.yml." >&2
    echo "       Install python3-yaml/PyYAML if missing, or check the catalog key." >&2
    exit 1
  }
  eval "$env_lines"
  if [[ -n "${WEIGHT_MODEL:-}" && "${WEIGHT_MODEL}" != "${MODEL_NAME}" ]]; then
    echo "ERROR: weight recipe '${key}' belongs to ${WEIGHT_MODEL}, not ${MODEL_NAME}." >&2
    exit 1
  fi
  if [[ -z "${WEIGHT_REPO:-}" ]]; then
    echo "ERROR: weight recipe '${key}' has no direct download recipe." >&2
    [[ -n "${WEIGHT_MANUAL_NOTE:-}" ]] && echo "       ${WEIGHT_MANUAL_NOTE}" >&2
    exit 1
  fi
  MODEL_REPO="${WEIGHT_REPO}"
  MODEL_REVISION="${WEIGHT_REVISION:-}"
  MODEL_SUBDIR="${WEIGHT_SUBDIR}"
  GGUF_FILES="${WEIGHT_FILES}"
  VERIFY_GLOB="${WEIGHT_VERIFY_GLOB:-*.safetensors}"
  echo "[model]   ${WEIGHT_KEY} -> ${MODEL_REPO} ${WEIGHT_FILES} -> ${MODEL_SUBDIR}"
}

# _picker_hw_mark <model> <space-separated compose paths (repo-relative)>
# Hardware-fit mark for the picker. compose-meta's friendly table only knows
# the legacy headline pair; for every other catalog model, fall back to the
# model's registered composes (registry-derived) and fit-check those directly.
_picker_hw_mark() {
  local model="$1" paths="$2" status file=""
  status="$(compose_hw_model_status "$ROOT_DIR" "$model" 2>/dev/null || true)"
  if [[ "${status#*|}" == "unknown model:"* && -n "${paths}" ]]; then
    for file in ${paths}; do
      [[ -f "${ROOT_DIR}/${file}" ]] || continue
      status="$(compose_hw_compose_status "${ROOT_DIR}/${file}" 2>/dev/null || true)"
      if [[ "${status}" == ok\|* ]]; then
        printf 'ok|fits your rig'
        return
      fi
    done
    [[ -n "${file:-}" ]] && { printf '%s' "${status}"; return; }
    printf 'no|no registered compose'
    return
  fi
  printf '%s' "${status}"
}

model_picker_line() { # <idx> <model> <label> <size-text> <compose-paths>
  local idx="$1" model="$2" label="$3" size="$4" paths="$5" status mark reason
  status="$(_picker_hw_mark "$model" "$paths")"
  reason="${status#*|}"
  if [[ "$status" == ok\|* ]]; then
    mark="✓"
  else
    mark="✗"
  fi
  printf "  %s. %-14s (%s)  %s %s\n" "$idx" "$label" "$size" "$mark" "$reason"
}

pick_model_interactive() {
  # shellcheck source=lib/compose-meta.sh
  source "${ROOT_DIR}/scripts/lib/compose-meta.sh"
  # shellcheck source=lib/registry-lookup.sh
  source "${ROOT_DIR}/scripts/lib/registry-lookup.sh"
  REGISTRY_LOOKUP_ROOT="${ROOT_DIR}"

  # ⚠️ Every picker line renders in a $( ) subshell, and a memo set inside one
  # dies with it — so each line used to re-run weights.py (twice), nvidia-smi,
  # and the registry lookups paid a second full emit (#1382). Resolve each input
  # ONCE here, in the shell those subshells fork from, and they inherit it.
  compose_hw_detect_gpus >/dev/null 2>&1 || true   # primes _COMPOSE_HW_GPU_CACHE
  # One catalog read for every row: id, label, default-weights size.
  local -a _ids _labels _sizes _both=()
  local _row_id _row_label _row_size
  while IFS=$'\x1f' read -r _row_id _row_label _row_size; do
    [[ -n "${_row_id}" ]] || continue
    _ids+=("${_row_id}")
    _labels+=("${_row_label}")
    _sizes+=("${_row_size}")
  done < <(_catalog_py "[chr(31).join([m['id'], m.get('display_name') or m['id'], str(m.get('size_gb'))]) for m in data['models']]")
  read_both_models _both

  # Registry-derived compose paths per model, for the hw-fit fallback above.
  # ONE registry emit, into registry-lookup's per-process cache, so the
  # compose_hw_model_status lookups below reuse it instead of emitting again.
  local _reg_json=""
  registry_lookup_cache_path 2>/dev/null && _reg_json="${_registry_lookup_cache}"
  local -A _model_composes=()
  if [[ -n "${_reg_json}" ]]; then
    eval "$(python3 -c '
import json, shlex, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
acc = {}
order = []
for v in data.get("variants", []):
    m = v.get("model")
    p = v.get("compose_path") or ""
    if not m or not p:
        continue
    if m not in acc:
        acc[m] = []
        order.append(m)
    acc[m].append(p)
sep = " "
for m in order:
    print(f"_model_composes[{shlex.quote(m)}]={shlex.quote(sep.join(acc[m]))}")
' "${_reg_json}")"
  fi

  echo "[setup] Which model to download?" >&2
  echo "" >&2
  local _idx=0 _i _id _both_idx=""
  for _i in "${!_ids[@]}"; do
    _idx=$((_idx + 1))
    _id="${_ids[${_i}]}"
    # Label + size of the model's default weight variant, from the one catalog read above.
    model_picker_line "$_idx" "${_id}" "${_labels[${_i}]}" "~${_sizes[${_i}]} GB default weights" "${_model_composes[${_id}]:-}" >&2
  done
  # The registry was only needed to render the list; don't strand the cache
  # file for the rest of a (possibly hours-long) download run.
  registry_lookup_cleanup
  if ((${#_both[@]} >= 2)); then
    _idx=$((_idx + 1))
    _both_idx="${_idx}"
    echo "  ${_idx}. Both (${_both[0]} + ${_both[1]})   downloads both recommended defaults" >&2
  fi
  echo "" >&2
  local _max=$(( ${#_ids[@]} + (${#_both[@]} >= 2 ? 1 : 0) ))
  while true; do
    local pick
    read -rp "Choice [1-${_max}]: " pick
    if [[ "${pick}" =~ ^[0-9]+$ ]] && ((pick >= 1 && pick <= ${#_ids[@]})); then
      echo "${_ids[$((pick - 1))]}"
      return
    elif [[ -n "${_both_idx}" && "${pick}" == "${_both_idx}" ]]; then
      echo "both"
      return
    fi
    echo "  ! invalid — pick 1-${_max}" >&2
  done
}

# ---------- Model dispatch ----------
case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

# setup.sh takes a SINGLE positional (the model). It DOWNLOADS WEIGHTS — it does
# NOT select or boot a serving config. A stray second arg (commonly a launch slug
# like `vllm/int8`) used to be silently ignored, which let users believe it did
# something — see issue #250, where the slug was dropped and the real failure
# (a purged-nightly compose pin) got mis-attributed to it. Reject it loudly and
# point at the launch path. (`both` recurses with a single arg, so it's unaffected.)
if [[ $# -gt 1 ]]; then
  echo "ERROR: setup.sh takes a single model name; got extra argument(s): ${*:2}" >&2
  echo "       setup.sh only DOWNLOADS WEIGHTS for a model — e.g. bash scripts/setup.sh ${1}" >&2
  [[ "${2:-}" == */* ]] && echo "       To download everything a slug needs instead: bash scripts/setup.sh ${2}" >&2
  echo "       To LAUNCH a serving config (a slug such as 'vllm/gemma-31b-dual'), use:" >&2
  echo "         bash scripts/launch.sh --variant <slug>      # or: bash scripts/switch.sh <slug>" >&2
  echo "       See the slugs available for a model:  bash scripts/switch.sh --list" >&2
  exit 64
fi

MODEL_NAME="${1:-}"
if [[ -z "${MODEL_NAME}" ]]; then
  if [[ -t 0 && -t 1 ]]; then
    MODEL_NAME="$(pick_model_interactive)"
  else
    usage
    echo ""
    echo "(Interactive picker available in a TTY shell. Use the positional form in scripts/CI.)"
    exit 1
  fi
fi

# ---------- Slug form: everything one launch slug needs ----------
# `setup.sh <slug>` (any argument with a '/', e.g. llamacpp/qwen38-27b-single-iq4xs)
# resolves the slug through the registry: its model, its weights variant, and its
# `weights_companions` (a drafter or vision projector its compose mounts). Before
# this, only the serve-cockpit's Download action knew a slug's companions, so a CLI
# download could leave a slug that crash-loops on a missing drafter (#1247). These
# are the same keys the cockpit builds (services.py run_weights_download).
# Every declared companion is fetched: all drafters are default-ON, and every
# projector is mounted unconditionally except GLM's, which is ~1 GB next to
# 114-157 GB of weights. Explicit WEIGHT_KEY / WEIGHT_EXTRA_KEYS still win, so
# `WEIGHT_EXTRA_KEYS= bash scripts/setup.sh <slug>` skips the companions.
SETUP_SLUG=""
if [[ "${MODEL_NAME}" == */* ]]; then
  SETUP_SLUG="${MODEL_NAME}"
  _slug_rc=0
  _slug_env="$(CLUB3090_PROFILES_DIR="${ROOT_DIR}/scripts/lib/profiles" python3 -c '
import os, shlex, sys
sys.path.insert(0, os.environ["CLUB3090_PROFILES_DIR"])
from compose_registry import get_registry
e = get_registry().get(sys.argv[1])
if e is None:
    raise SystemExit(3)
model, variant = e["model"], e["weights_variant"]
comps = [c if ":" in c else f"{model}:{c}" for c in (e.get("weights_companions") or []) if c]
print("_slug_model=" + shlex.quote(model))
print("_slug_key=" + shlex.quote(model + ":" + variant))
print("_slug_companions=" + shlex.quote(" ".join(comps)))
print("_slug_status=" + shlex.quote(str(e.get("status") or "?")))
' "${SETUP_SLUG}")" || _slug_rc=$?
  if [[ "${_slug_rc}" == "3" ]]; then
    echo "ERROR: '${SETUP_SLUG}' is not a model name or a known launch slug." >&2
    echo "       Model names: $(_catalog_py "[', '.join(m['id'] for m in data['models'])]")" >&2
    echo "       Launch slugs: bash scripts/switch.sh --list" >&2
    exit 1
  elif [[ "${_slug_rc}" != "0" ]]; then
    echo "ERROR: could not read the launch-slug registry to resolve '${SETUP_SLUG}'." >&2
    echo "       Install python3-yaml/PyYAML if missing, or run: python3 scripts/lib/profiles/migrate_registry_to_yaml.py --check" >&2
    exit 1
  fi
  eval "${_slug_env}"
  MODEL_NAME="${_slug_model}"
  : "${WEIGHT_KEY:=${_slug_key}}"
  [[ -n "${WEIGHT_EXTRA_KEYS+x}" ]] || WEIGHT_EXTRA_KEYS="${_slug_companions}"
  echo "[slug]    ${SETUP_SLUG} (${_slug_status}) -> ${WEIGHT_KEY}${WEIGHT_EXTRA_KEYS:+ + ${WEIGHT_EXTRA_KEYS}}"
fi

declare -a BOTH_MODELS=()
read_both_models BOTH_MODELS
if [[ "${MODEL_NAME}" == "both" ]]; then
  if ((${#BOTH_MODELS[@]} < 2)); then
    echo "ERROR: 'both' needs at least two RECOMMENDED_DEFAULT_MODELS entries in" >&2
    echo "       scripts/lib/profiles/compose_registry.py (found: ${BOTH_MODELS[*]:-none})." >&2
    echo "       Download each model explicitly: bash scripts/setup.sh <model-name>" >&2
    exit 1
  fi
  # Resolve MODEL_DIR once in the parent by reusing the normal prompt below,
  # then recurse through the positional form for each model.
  SETUP_BOTH_MODE=1
  MODEL_NAME="${BOTH_MODELS[0]}"
else
  SETUP_BOTH_MODE=0
fi

# The profile YAMLs are the only source of download recipes. setup.sh only
# maps friendly setup knobs (MODEL_NAME, WEIGHTS, WITH_*) to profile keys —
# all of it derived from `weights.py catalog --json`; a model's optional
# `setup:` block in profiles/models/<id>.yml captures whatever differs from
# the defaults (primary = default_weight_variant, no aliases, no drafters).
ALWAYS_DRAFT_KEY=""
DFLASH_KEY=""
VISION_KEY=""
PRISM_EAGLE3_KEY=""
PRIMARY_WEIGHT_KEY=""
EXTRA_WEIGHT_KEYS=()
SETUP_ASSISTANT_DRAFT=""
SETUP_SUPPORTED_WEIGHTS="autoround"
declare -A SETUP_ALIAS=()
declare -A SETUP_ALIAS_EXTRAS=()

# Resolve MODEL_NAME's dispatch policy from the catalog.
_setup_found=0
eval "$(python3 - "${MODEL_NAME}" "$(catalog_json)" <<'PY'
import json, shlex, sys

model, raw = sys.argv[1], sys.argv[2]
data = json.loads(raw)
m = next((x for x in data["models"] if x["id"] == model), None)

def q(v):
    return shlex.quote(str(v if v is not None else ""))


if m is None:
    print("_setup_found=0")
    raise SystemExit(0)
print("_setup_found=1")
print(f"PRIMARY_WEIGHT_KEY={q(m['default_key'])}")
for var, key in (
    ("ALWAYS_DRAFT_KEY", "always_draft"),
    ("DFLASH_KEY", "dflash"),
    ("VISION_KEY", "vision"),
    ("PRISM_EAGLE3_KEY", "prism_eagle3"),
):
    print(f"{var}={q(model + ':' + m[key] if m.get(key) else '')}")
# Bare variant — setup.sh qualifies it with MODEL_NAME when
# WITH_ASSISTANT_DRAFT=1 overrides ALWAYS_DRAFT_KEY.
print(f"SETUP_ASSISTANT_DRAFT={q(m.get('assistant_draft') or '')}")
for alias, variant in (m.get("aliases") or {}).items():
    print(f"SETUP_ALIAS[{q(alias)}]={q(variant)}")
for alias, extras in (m.get("alias_extras") or {}).items():
    print(f"SETUP_ALIAS_EXTRAS[{q(alias)}]={q(' '.join(model + ':' + x for x in extras))}")
aliases = sorted((m.get("aliases") or {}))
print(f"SETUP_SUPPORTED_WEIGHTS={q(' '.join(['autoround'] + aliases))}")
PY
)"

# An unknown friendly model name is fatal ONLY when WEIGHT_KEY is NOT set.
# WEIGHT_KEY is the authoritative "exact catalog entry" fetch-now flow (used
# by preflight + the serve-cockpit Download action): the recipe fully
# specifies <model>:<variant>, and load_weight_recipe() validates that the
# key's model matches MODEL_NAME — so no per-family dispatch is needed for an
# arbitrary catalog entry. The supported list is DERIVED, never hand-written:
# a new model yml is accepted here automatically.
if ((_setup_found == 0)) && [[ -z "${WEIGHT_KEY:-}" ]]; then
  echo "ERROR: unsupported model '${MODEL_NAME}'."
  echo "Supported: $(_catalog_py "[', '.join(m['id'] for m in data['models'])]")"
  echo "(To add a new model, add scripts/lib/profiles/models/<id>.yml — setup.sh derives"
  echo " its front door from the catalog automatically. Or pass WEIGHT_KEY=<model>:<variant>"
  echo " to fetch an exact catalog entry directly.)"
  exit 1
fi

# ---------- Weights format / exact registry entry ----------
# WEIGHTS selects a common weight variant for the model family. WEIGHT_KEY is
# the exact catalog-entry path used by preflight's fetch-now flow.
WEIGHTS="${WEIGHTS:-autoround}"
GGUF_FILES=""
VERIFY_GLOB="*.safetensors"

if [[ -n "${WEIGHT_KEY:-}" ]]; then
  PRIMARY_WEIGHT_KEY="${WEIGHT_KEY}"
elif [[ "${WEIGHTS}" == "autoround" ]]; then
  # The default: PRIMARY_WEIGHT_KEY already resolved to the model's default key.
  :
elif [[ -n "${SETUP_ALIAS[${WEIGHTS}]:-}" ]]; then
  # WEIGHTS=<alias> -> the model's registered variant for that alias
  # (profiles/models/<id>.yml `setup:.weights_aliases`).
  PRIMARY_WEIGHT_KEY="${MODEL_NAME}:${SETUP_ALIAS[${WEIGHTS}]}"
  if [[ -n "${SETUP_ALIAS_EXTRAS[${WEIGHTS}]:-}" ]]; then
    read -ra _alias_extras <<< "${SETUP_ALIAS_EXTRAS[${WEIGHTS}]}"
    EXTRA_WEIGHT_KEYS+=("${_alias_extras[@]}")
  fi
else
  echo "ERROR: WEIGHTS='${WEIGHTS}' not recognized for ${MODEL_NAME} (supported: ${SETUP_SUPPORTED_WEIGHTS})." >&2
  echo "       Use WEIGHT_KEY=<model>:<variant> for exact catalog entries." >&2
  exit 1
fi

if [[ "${WITH_ASSISTANT_DRAFT:-0}" == "1" ]]; then
  if [[ -z "${SETUP_ASSISTANT_DRAFT:-}" ]]; then
    echo "ERROR: WITH_ASSISTANT_DRAFT=1 is not wired for ${MODEL_NAME}: no assistant_draft in its profiles/models/<id>.yml setup block." >&2
    exit 1
  fi
  ALWAYS_DRAFT_KEY="${MODEL_NAME}:${SETUP_ASSISTANT_DRAFT}"
fi

# ---------- Companion artifacts (cockpit Download, setup.sh <slug>) ----------
# A slug's `weights_companions` (a drafter / mmproj vision projector its compose
# mounts from a separate subdir) arrive as a space/comma-separated WEIGHT_EXTRA_KEYS
# list of fully-qualified <model>:<variant> keys: from the serve-cockpit Download
# action, or from the slug form above.  Fetch them ALONGSIDE the core so a
# downloaded slug actually serves — otherwise it reads "present" then fails to boot
# for the missing companion.  Each is a normal catalog entry pulled by the
# EXTRA_WEIGHT_KEYS loop below (after the SKIP_MODEL guard), with its own SHA verify.
# Queued HERE, before the dump, so SETUP_DUMP_KEYS shows exactly what would download.
_COMPANION_KEYS=()
if [[ -n "${WEIGHT_EXTRA_KEYS:-}" ]]; then
  read -ra _COMPANION_KEYS <<< "${WEIGHT_EXTRA_KEYS//,/ }"
  for _ck in "${_COMPANION_KEYS[@]}"; do
    [[ -n "${_ck}" ]] || continue
    # A companion can also be the primary or the model's always-fetched drafter
    # (GLM's dflash2-q4km is both): queue it once, or the disk estimate counts it twice.
    [[ "${_ck}" == "${PRIMARY_WEIGHT_KEY}" || "${_ck}" == "${ALWAYS_DRAFT_KEY:-}" ]] && continue
    [[ " ${EXTRA_WEIGHT_KEYS[*]:-} " == *" ${_ck} "* ]] && continue
    EXTRA_WEIGHT_KEYS+=("${_ck}")
  done
fi

# Debug / CI surface: print the resolved dispatch keys and exit before any
# preflight or download. Used by scripts/tests/test-setup-registry-derived.sh
# to assert setup.sh's derived keys agree with `weights.py catalog --json` for
# EVERY model and alias without exercising the fetch path.
if [[ "${SETUP_DUMP_KEYS:-0}" == "1" ]]; then
  echo "model=${MODEL_NAME}"
  echo "label=$(model_label "${MODEL_NAME}")"
  echo "primary=${PRIMARY_WEIGHT_KEY}"
  echo "always_draft=${ALWAYS_DRAFT_KEY}"
  echo "dflash=${DFLASH_KEY}"
  echo "vision=${VISION_KEY}"
  echo "prism_eagle3=${PRISM_EAGLE3_KEY}"
  echo "extras=${EXTRA_WEIGHT_KEYS[*]:-}"
  echo "slug=${SETUP_SLUG}"
  echo "companions=${WEIGHT_EXTRA_KEYS:-}"
  exit 0
fi

load_weight_recipe "${PRIMARY_WEIGHT_KEY}"

# Companions were queued above, before the SETUP_DUMP_KEYS surface.
[[ -n "${_COMPANION_KEYS[*]:-}" ]] && echo "[model]   + companion(s): ${_COMPANION_KEYS[*]}"

# ---------- MODEL_DIR resolution ----------
# Order of precedence:
#   1. MODEL_DIR already exported in the calling shell  → use as-is
#   2. .env at repo root sets MODEL_DIR                  → source it
#   3. Interactive prompt (only if stdin is a TTY)       → ask user
#   4. Silent fallback to <repo>/models-cache            → in-repo default
#
# The prompt only fires for fresh users on a TTY who haven't set anything.
# CI / scripted runs (no TTY) get the silent fallback, preserving prior behavior.

# Step 2: source repo-root .env if present (lets a saved choice persist)
if [[ -z "${MODEL_DIR:-}" && -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck source=/dev/null
  set -a; source "${ROOT_DIR}/.env"; set +a
fi

# Step 3: prompt if still unset + interactive
if [[ -z "${MODEL_DIR:-}" && -t 0 && -t 1 ]]; then
  echo ""
  echo "Where should I put model weights?"
  echo "  Models are large (Qwen3.6-27B AutoRound: ~14 GB; Gemma 4 31B: ~21 GB)."
  echo "  This dir lives outside the git tree — pick a location with sufficient free space."
  echo ""
  echo "  1) ${ROOT_DIR}/models-cache  (in-repo, default — pollutes git tree)"
  echo "  2) ${HOME}/models             (recommended for cross-rig — outside repo)"
  echo "  3) custom path"
  echo ""
  while true; do
    read -rp "Choice [1-3] (or set MODEL_DIR env var to skip): " pick
    case "${pick}" in
      1) MODEL_DIR="${ROOT_DIR}/models-cache"; break ;;
      2) MODEL_DIR="${HOME}/models"; break ;;
      3)
        read -rp "  Enter absolute path: " custom
        if [[ "${custom}" =~ ^/ ]]; then
          MODEL_DIR="${custom}"; break
        else
          echo "  ! must be an absolute path (start with /)" >&2
        fi
        ;;
      *) echo "  ! invalid — pick 1, 2, or 3" >&2 ;;
    esac
  done
  echo ""

  # Offer to persist the choice so future runs skip the prompt
  read -rp "Save MODEL_DIR=${MODEL_DIR} to .env so we skip this next time? [Y/n]: " save
  if [[ "${save:-y}" =~ ^[Yy]$ || -z "${save:-}" ]]; then
    if [[ -f "${ROOT_DIR}/.env" ]]; then
      # Update existing .env (replace MODEL_DIR= line if present, else append)
      if grep -qE "^MODEL_DIR=" "${ROOT_DIR}/.env"; then
        sed -i "s|^MODEL_DIR=.*|MODEL_DIR=${MODEL_DIR}|" "${ROOT_DIR}/.env"
      else
        echo "MODEL_DIR=${MODEL_DIR}" >> "${ROOT_DIR}/.env"
      fi
    else
      echo "MODEL_DIR=${MODEL_DIR}" > "${ROOT_DIR}/.env"
    fi
    echo "  → saved. (.env is gitignored.)"
  else
    echo "  → not saved. Set MODEL_DIR=... when re-running, or you'll get this prompt again."
  fi
  echo ""
fi

# Step 4: silent fallback (preserves prior behavior for non-TTY contexts)
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models-cache}"
if [[ "${SETUP_BOTH_MODE:-0}" == "1" ]]; then
  export MODEL_DIR
  echo "[setup] downloading the recommended defaults into ${MODEL_DIR}: ${BOTH_MODELS[*]}"
  echo ""
  for _both_model in "${BOTH_MODELS[@]}"; do
    bash "$0" "${_both_model}"
    echo ""
  done
  echo "[setup] ✓ All recommended models downloaded."
  echo "[setup] Next: bash scripts/launch.sh"
  exit 0
fi

cd "${ROOT_DIR}"

# ---------- Pre-flight checks ----------
# Catches the common "first-run failures": missing docker, no GPU visible,
# disk too small for the ~14 GB AutoRound int4 download. Fails fast with
# actionable hints rather than mid-download or first-boot crash.
# shellcheck source=preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

# Required disk — derived from what this run will actually FETCH.
#
# This used to be a hardcoded 25 GB (28 with a DFlash draft), sized when
# setup.sh served exactly one ~14 GB model. That constant is wrong in both
# directions once the catalog holds large weights, and the under-gate is the
# dangerous one:
#
#   too LOW  — DeepSeek-Flash IQ2 is 85 GiB and Q8 is 151 GiB. A user with
#              30 GB free PASSES the gate, then runs out mid-download. The
#              check gives false assurance exactly where it matters most.
#   too HIGH — with the weights already on disk nothing needs fetching, yet
#              the gate still demanded 25 GB and hard-exited (found by
#              actually running setup.sh, 2026-08-07).
#
# So: sum `size_gb` over the keys this run would download, counting only the
# ones NOT already present, and add headroom for download temp files. An
# explicit PREFLIGHT_DISK_GB still wins — it is the documented escape hatch.
_disk_need_gb() {
  local total=0 key subdir size present
  for key in "$@"; do
    [[ -n "$key" ]] || continue
    # Subshell: don't let one entry's WEIGHT_* leak into the next iteration.
    read -r size subdir < <(
      env_lines="$(python3 "${ROOT_DIR}/scripts/lib/profiles/weights.py" entry "$key" 2>/dev/null)" || exit 0
      eval "$env_lines"
      printf '%s %s\n' "${WEIGHT_SIZE_GB:-0}" "${WEIGHT_VERIFY_GLOB:-}|${WEIGHT_SUBDIR:-}"
    )
    [[ -n "${size:-}" ]] || continue
    local glob="${subdir%%|*}" dir="${subdir##*|}"
    # Already on disk -> costs nothing. Same presence test the verifier uses,
    # so "present" here means the same thing it means at verify time.
    present=0
    if [[ -n "$dir" && -d "${MODEL_DIR}/${dir}" ]]; then
      # shellcheck disable=SC2086  # glob must stay unquoted to expand
      compgen -G "${MODEL_DIR}/${dir}/${glob}" >/dev/null 2>&1 && present=1
    fi
    [[ "$present" == "1" ]] && continue
    # Some profiles declare a non-numeric size (qwen3.6-27b gguf emits the
    # literal `WEIGHT_SIZE_GB=variable`, #1131).  Bash arithmetic would read
    # that string as an unset variable name under `set -u` and kill the
    # preflight.  Skip it with a note -- the download reports the real size
    # when it runs -- instead of crashing.  (The note goes to stderr: stdout
    # of this function is command-substituted into PREFLIGHT_DISK_GB.)
    if [[ ! "$size" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "[preflight] WARN:  $key: non-numeric size '$size' -- excluded from the disk estimate; set PREFLIGHT_DISK_GB=<N> to override." >&2
      continue
    fi
    total=$(( total + ${size%%.*} ))
  done
  # Headroom for partial-download temp files, plus a floor so a fully-present
  # re-run still refuses to proceed on a volume with nothing left at all.
  if [[ "$total" -gt 0 ]]; then
    echo $(( total + 10 ))
  else
    echo 5
  fi
}

_DISK_KEYS=("${PRIMARY_WEIGHT_KEY:-}" "${ALWAYS_DRAFT_KEY:-}" "${EXTRA_WEIGHT_KEYS[@]}")
[[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]] && _DISK_KEYS+=("${DFLASH_KEY:-${MODEL_NAME}:dflash}")
# WITH_VISION=1 opts into the mmproj projector (disk-gate it AND queue the download).
[[ "${WITH_VISION:-0}" == "1" && -n "${VISION_KEY:-}" ]] && { _DISK_KEYS+=("${VISION_KEY}"); EXTRA_WEIGHT_KEYS+=("${VISION_KEY}"); }
PREFLIGHT_DISK_GB="${PREFLIGHT_DISK_GB:-$(_disk_need_gb "${_DISK_KEYS[@]}")}"

echo "[preflight] checking environment..."
# docker is soft-warn for setup.sh — this script only fetches models, no docker
# invocations until you actually `docker compose up` later. Hard-failing blocks
# non-docker container-runtime users (microk8s / podman / k8s / manual) from
# running setup at all (club-3090 disc #48). launch.sh keeps the hard check
# because it actually invokes docker.
preflight_docker || echo "[preflight] WARN:  docker unavailable — setup will continue (model fetch doesn't need docker), but you'll need a working container runtime before 'docker compose up'."
preflight_gpu 1  || exit 1
preflight_disk "${MODEL_DIR}" "${PREFLIGHT_DISK_GB}" || exit 1
preflight_hf_token  # soft-warn only; downloads will surface the hard failure
echo "[preflight] ok."
echo ""

# ---------- WSL2 detection — auto-configure .env for known WSL2 boot crash ----------
# WSL2 + driver 596.36 + vLLM nightly hit a `gptq_marlin_repack` boot crash
# with `cudaErrorNotReady`. Workaround is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`
# (PR #84). The compose default is `expandable_segments:True,max_split_size_mb:512`
# which works on bare-metal Linux but fails on WSL2 — so we auto-create a .env
# override here on detected WSL2 systems. Cross-rig validated by @timxx (issue #60),
# @easel, and others. Safe no-op on bare-metal (only runs when /proc/version
# contains "microsoft").
COMPOSE_DIR="${ROOT_DIR}/models/${MODEL_NAME}/vllm/compose"
if [[ -f /proc/version ]] && command grep -qi microsoft /proc/version 2>/dev/null; then
  ENV_FILE="${COMPOSE_DIR}/.env"
  if [[ -d "${COMPOSE_DIR}" ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
      cat > "${ENV_FILE}" <<'EOF'
# WSL2 boot-crash workaround — see PR #84 + issue #60.
# vLLM + WSL2 + driver 596.36 hit `gptq_marlin_repack` cudaErrorNotReady on boot
# with the default `expandable_segments:True`. This override fixes it.
# Auto-created by scripts/setup.sh on detected WSL2 systems. Safe to delete
# on bare-metal Linux (the compose default works there).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
EOF
      echo "[wsl2] detected WSL2 — created ${ENV_FILE} with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
      echo "[wsl2] this fixes the known gptq_marlin_repack boot crash on WSL2 + driver ≥596.36 (issue #60)."
    elif ! grep -q "expandable_segments:False" "${ENV_FILE}"; then
      echo "[wsl2] WARN: detected WSL2 but ${ENV_FILE} exists without the expandable_segments:False override."
      echo "[wsl2]       If vLLM fails to boot with cudaErrorNotReady, add:"
      echo "[wsl2]         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
      echo "[wsl2]       See PR #84 / issue #60 for context."
    else
      echo "[wsl2] detected WSL2 — ${ENV_FILE} already has the expandable_segments:False override. ✓"
    fi
  fi
fi

# ---------- Tool checks ----------
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required tool '$1' not found in PATH." >&2
    exit 1
  }
}
need curl
need sha256sum

echo "Setup root:   ${ROOT_DIR}"
echo "Model dir:    ${MODEL_DIR}"

# ---------- Model download ----------
if [[ "${SKIP_MODEL:-0}" == "1" ]]; then
  echo "[model]   SKIP_MODEL=1 — not downloading."
  exit 0
fi

# Ensure an HF download CLI ('hf', or legacy 'huggingface-cli') is available.
# If neither is present, offer a CONSENT-GATED, ISOLATED install (uv tool / pipx
# — lands on PATH, no changes to system Python). We deliberately NEVER run
# `pip --break-system-packages` or `sudo apt` on the user's behalf; those stay
# copy-paste the user owns. Returns 0 when a CLI is available (already or after
# an accepted install); prints the working manual options + returns 1 otherwise.
# Honors CLUB3090_ASSUME_YES=1 for non-interactive opt-in. Fixes the PEP 668
# `externally-managed-environment` wall on Ubuntu 24.04 / WSL, where the old
# bare `pip install` hint could not run.
ensure_hf_cli() {
  command -v hf >/dev/null 2>&1 && return 0
  command -v huggingface-cli >/dev/null 2>&1 && return 0

  # First ISOLATED installer available (never touches system Python).
  local installer="" cmd=""
  if command -v uv >/dev/null 2>&1; then
    installer="uv";   cmd="uv tool install --with hf_transfer huggingface-hub"
  elif command -v pipx >/dev/null 2>&1; then
    installer="pipx"; cmd="pipx install 'huggingface-hub[hf_transfer]'"
  fi

  if [[ -n "$installer" ]]; then
    local go=""
    if [[ "${CLUB3090_ASSUME_YES:-0}" == "1" ]]; then
      go=1
      echo "[model] hf CLI missing — CLUB3090_ASSUME_YES=1, installing via ${installer} (isolated)." >&2
    elif [[ -t 0 ]]; then
      echo "[model] The Hugging Face CLI ('hf') is required to download weights, but it's not installed." >&2
      echo "[model] I can install it for you, isolated (on your PATH, no changes to system Python):" >&2
      echo "[model]     ${cmd}" >&2
      local reply=""
      read -rp "[model] Install it now? [Y/n] " reply
      case "${reply}" in ""|[Yy]|[Yy][Ee][Ss]) go=1 ;; esac
    fi
    if [[ -n "$go" ]]; then
      echo "[model] Installing: ${cmd}" >&2
      if eval "$cmd" >&2; then
        # Freshly-installed console scripts land in ~/.local/bin (pipx + uv's
        # default tool bin) — make them visible to THIS run, no shell restart.
        export PATH="${HOME}/.local/bin:${PATH}"
        hash -r 2>/dev/null || true
        if command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1; then
          echo "[model] hf CLI installed and on PATH." >&2
          return 0
        fi
        echo "[model] Installed, but 'hf' isn't on PATH in this shell yet — restart your terminal (or run 'pipx ensurepath' / add ~/.local/bin to PATH) and re-run setup.sh." >&2
        return 1
      fi
      echo "[model] Automatic install did not complete — use a manual option below." >&2
    fi
  fi

  # No isolated installer, declined, or install failed → the working manual set.
  {
    echo "ERROR: the Hugging Face CLI ('hf') is required but not installed."
    echo "  Recommended — isolated, puts 'hf' on your PATH (works on Ubuntu 24.04 / WSL):"
    echo "    sudo apt install -y pipx && pipx install 'huggingface-hub[hf_transfer]' && pipx ensurepath"
    echo "    (then restart your shell and re-run this command)"
    echo "  Or, with uv:"
    echo "    uv tool install --with hf_transfer huggingface-hub"
    echo "  Quick override — modifies SYSTEM Python (only on a dedicated box / WSL):"
    echo "    pip install --break-system-packages 'huggingface-hub[hf_transfer]'"
  } >&2
  return 1
}

_hf_download_repo() {
  local repo="$1"
  local subdir="$2"
  local files="${3:-}"
  local revision="${4:-}"
  # Optional commit-SHA / tag pin (#319). Empty -> track HEAD (today's behavior).
  local rev_args=()
  [[ -n "$revision" ]] && rev_args=(--revision "$revision")
  local dest="${MODEL_DIR}/${subdir}"
  mkdir -p "${dest}"
  # Guarantee a download CLI (consent-gated isolated install if missing).
  ensure_hf_cli || exit 1

  local cli=""
  if command -v hf >/dev/null 2>&1; then
    cli="hf"
    echo "[model]   Using 'hf download' (hf_transfer if available) ..."
  elif command -v huggingface-cli >/dev/null 2>&1; then
    cli="huggingface-cli"
    echo "[model]   Using 'huggingface-cli download' ..."
  else
    # Unreachable: ensure_hf_cli returned 0 so one of the above resolves.
    echo "ERROR: hf CLI unexpectedly unavailable after ensure_hf_cli." >&2
    exit 1
  fi

  # Bounded retry with backoff (club-3090#1305). These transfers are 16-17 GB,
  # so the window for one transient stall is wide — and this ran ONCE, under
  # `set -euo pipefail`, so a single mid-transfer
  # `httpx.ConnectTimeout: _ssl.c:1015: The handshake operation timed out`
  # aborted the whole setup run with a ~40-line unhandled Python traceback,
  # against a hub that was otherwise answering in well under a second. The hub
  # client resumes from the `.incomplete` markers it has already written, so a
  # retry costs nothing but the reconnect. Env knobs exist mainly so the gate can
  # exercise this without sleeping.
  # Validate the knobs before any arithmetic touches them. Under `set -u` a
  # non-numeric value makes `(( attempt >= max ))` resolve it as a VARIABLE name
  # and die "foo: unbound variable" — a crash inside the code that exists to
  # handle crashes. Same reasoning as the composes' non-numeric SPEC_N check: a
  # typo gets a clear error, never a silent fallback.
  local max="${HF_DOWNLOAD_RETRIES:-4}"
  local backoff="${HF_DOWNLOAD_RETRY_SLEEP:-8}"
  case "${max}" in ""|*[!0-9]*|0)
    echo "ERROR: HF_DOWNLOAD_RETRIES='${max}' is not a positive integer (attempts per download)." >&2
    exit 1 ;;
  esac
  case "${backoff}" in ""|*[!0-9]*)
    echo "ERROR: HF_DOWNLOAD_RETRY_SLEEP='${backoff}' is not a non-negative integer (seconds)." >&2
    exit 1 ;;
  esac
  local attempt=1 rc=0
  while :; do
    rc=0
    # files is intentionally word-split: empty -> whole repo; non-empty -> selected files.
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 \
      "${cli}" download "$repo" ${files} "${rev_args[@]}" --local-dir "${dest}" || rc=$?
    if [[ "${rc}" -eq 0 ]]; then
      return 0
    fi
    if (( attempt >= max )); then
      break
    fi
    echo "[model]   attempt ${attempt}/${max} failed (rc=${rc}) — retrying in ${backoff}s." >&2
    echo "[model]   Nothing is lost: the transfer resumes from the partial files already on disk." >&2
    [[ "${backoff}" != "0" ]] && sleep "${backoff}"
    attempt=$(( attempt + 1 ))
    backoff=$(( backoff * 2 > 120 ? 120 : backoff * 2 ))
  done

  # Give up with an instruction, not a traceback. The recovery here is trivial —
  # re-run the identical command — and was previously undiscoverable from the
  # output, so a first-time user on a 17 GB download reasonably read it as
  # "this is broken" rather than "run it again".
  {
    echo ""
    echo "ERROR: downloading '${repo}' failed on all ${max} attempts (last rc=${rc})."
    echo "       Any traceback above comes from the hub client. The transfer itself is"
    echo "       RESUMABLE: re-run the SAME command and it continues from where it"
    echo "       stopped — completed files are skipped and partial ones are kept under"
    echo "         ${dest}/.cache/huggingface/download/*.incomplete"
    echo "       If it keeps failing, check that the hub is reachable:"
    echo "         curl -sS -o /dev/null -w 'http=%{http_code} total=%{time_total}s\\n' https://huggingface.co"
    echo "       Tune the retry with HF_DOWNLOAD_RETRIES (default 4) if your link is flaky."
  } >&2
  exit 1
}

# _hf_remote_meta <repo> <revision> <file> -> "<sha256> <size>" on stdout
#
# Delegates to scripts/lib/profiles/hf_fetch.py::resolve_head (#855) rather than
# re-implementing the lookup here. Two reasons that are not style preferences:
#
#   1. The HEAD must NOT follow redirects. The old `curl -sfI` did, and on any
#      Xet-backed repo the 302 lands on the CAS bridge, whose response carries
#      the CAS *blob* ETag and NO `x-linked-etag` — so the canonical sha256,
#      which lives on the FIRST hop, read as "not published" on every modern
#      repo. The SKIP was the normal outcome, not a rare edge (#857).
#   2. One implementation means the downloader and setup.sh cannot disagree
#      about what "verified" means. They already did, and the weaker surface
#      was the one printing the reassuring line.
#
# Prints "<sha256|-> <size|->" and returns 0 when the hub answered at all;
# returns non-zero only when the lookup itself failed. A literal "-" in the hash
# field means the hub published NO sha256 for this file — the caller MUST treat
# that as UNVERIFIED, never as a pass, however good the size looks.
_hf_remote_meta() {
  local repo="$1" revision="$2" file="$3"
  command -v python3 >/dev/null 2>&1 || return 1
  [[ -f "${ROOT_DIR}/scripts/lib/profiles/hf_fetch.py" ]] || return 1
  python3 - "$repo" "$revision" "$file" <<PY 2>/dev/null
import sys
sys.path.insert(0, "${ROOT_DIR}/scripts/lib/profiles")
import hf_fetch
m = hf_fetch.resolve_head(sys.argv[1], sys.argv[3], revision=sys.argv[2],
                          token=hf_fetch._token())
if m.sha256 is None and m.size is None:
    raise SystemExit(1)          # the hub told us nothing at all
print(f"{m.sha256 or '-'} {m.size if m.size is not None else '-'}")
PY
}

# Verify the downloaded artifacts against the hub's published hashes.
#
# #857 — the word "verified" only ever attaches to a HASH check. Before this,
# a file whose etag lookup came back empty printed `SKIP (no etag)`, was not
# counted as a failure, and was still included in the `N file(s) SHA-verified`
# total. Combined with the redirect-following lookup above, on a Xet-backed repo
# that meant the reassuring line was printed for files nothing had checked —
# the same failure shape as the historical "DONE (hash-verified)" incident.
#
# Now: hash-checkable files are verified; everything else is UNVERIFIED (size-
# only at best) and is EXCLUDED from the verified count, with the split stated.
_verify_downloaded_files() {
  local repo="$1"
  local subdir="$2"
  local verify_glob="$3"
  # Pin the etag lookup to the same revision we downloaded (#319). Empty -> main
  # HEAD; a stale pin would otherwise etag-check against a newer HEAD and FAIL.
  local revision="${4:-main}"
  local fail=0 verified=0 unverified=0 total=0 skipped=0
  local _mk _sz _mt _prev
  local f meta expected exp_size actual local_size

  echo "[verify]  Checking SHA256 of every ${verify_glob} against the hub's published"
  echo "          x-linked-etag (no-redirect resolve HEAD, rev: ${revision}) ..."
  cd "${MODEL_DIR}/${subdir}"
  for f in ${verify_glob}; do
    [[ -f "$f" ]] || continue
    total=$((total + 1))
    expected=""; exp_size=""
    if meta="$(_hf_remote_meta "$repo" "$revision" "$f")"; then
      read -r expected exp_size <<<"$meta"
      [[ "$expected" == "-" ]] && expected=""
      [[ "$exp_size" == "-" ]] && exp_size=""
    fi
    if [[ -z "$expected" ]]; then
      # No published hash -> this file CANNOT be verified. Size is the only
      # cross-check available and it is not a verification: a truncated-then-
      # padded or silently-corrupted file matches on size (the exact incident
      # this rule exists for). Say so, and keep it out of the verified count.
      local_size="$(stat -c '%s' "$f" 2>/dev/null || echo "?")"
      if [[ -n "$exp_size" && "$exp_size" == "$local_size" ]]; then
        printf "  %-50s UNVERIFIED (no published hash; size matches: %s bytes — NOT a verification)\n" \
          "$f" "$local_size"
      elif [[ -n "$exp_size" ]]; then
        printf "  %-50s FAIL  size mismatch  exp=%s  act=%s\n" "$f" "$exp_size" "$local_size"
        fail=$((fail + 1))
        continue
      else
        printf "  %-50s UNVERIFIED (hub published neither hash nor size)\n" "$f"
      fi
      unverified=$((unverified + 1))
      continue
    fi
    # ── marker-based skip ────────────────────────────────────────────────
    # Re-hashing every file on every run costs minutes of pure disk read on a
    # 157 GB model even when nothing was fetched. Skip ONLY when all four of
    # (expected-sha, size, mtime, revision) match what was recorded at the last
    # PASSING verify of THIS file.
    # ⚠️ Deliberately NOT a bare "verified" flag — the incident this function
    # exists for was a run that TRUSTED such a flag. The hub's expected sha is
    # re-fetched every run (above), so an upstream change moves it; any local
    # mutation moves size or mtime. Either way we fall through and re-hash.
    # FORCE_VERIFY=1 bypasses the skip entirely.
    _mk="${MODEL_DIR}/${subdir}/.setup-verified.tsv"
    _sz="$(stat -c '%s' "$f" 2>/dev/null || echo '?')"
    _mt="$(stat -c '%Y' "$f" 2>/dev/null || echo '?')"
    if [[ "${FORCE_VERIFY:-0}" != "1" && -f "$_mk" ]]; then
      _prev="$(awk -F'\t' -v n="$f" '$5==n {print $1"|"$2"|"$3"|"$4}' "$_mk" 2>/dev/null | tail -1)"
      if [[ -n "$_prev" && "$_prev" == "${expected}|${_sz}|${_mt}|${revision}" ]]; then
        printf "  %-50s SKIP  (unchanged since last verify)\n" "$f"
        skipped=$((skipped + 1))
        continue
      fi
    fi
    actual="$(sha256sum "$f" | awk '{print $1}')"
    if [[ "$expected" == "$actual" ]]; then
      printf "  %-50s OK\n" "$f"
      verified=$((verified + 1))
      # Record ONLY on a pass, and only for a file we actually hashed.
      printf '%s\t%s\t%s\t%s\t%s\n' "$expected" "$_sz" "$_mt" "$revision" "$f" \
        >> "$_mk" 2>/dev/null || true
    else
      printf "  %-50s FAIL  exp=%.12s  act=%.12s\n" "$f" "$expected" "$actual"
      fail=$((fail + 1))
    fi
  done
  cd "${ROOT_DIR}"

  if [[ "$fail" != "0" ]]; then
    echo "[verify]  ${fail} file(s) failed their integrity check." >&2
    echo "          Delete ${MODEL_DIR}/${subdir} and re-run setup.sh." >&2
    exit 1
  fi
  if [[ "$total" == "0" ]]; then
    echo "[verify]  No ${verify_glob} found in ${MODEL_DIR}/${subdir} — download may have failed." >&2
    exit 1
  fi
  # The count in this line is HASH-VERIFIED FILES ONLY. Anything the hub does
  # not publish a hash for is reported separately and loudly — a reader must
  # never be able to read "N file(s) SHA-verified" and conclude N == total when
  # it does not.
  echo "[done]    ${verified}/${total} file(s) SHA-verified in ${subdir}."
  if [[ "$skipped" != "0" ]]; then
    echo "[verify]  ${skipped}/${total} file(s) SKIPPED — expected-sha + size + mtime +"
    echo "            revision all unchanged since their last PASSING verify, so they"
    echo "            were NOT re-hashed this run. This is a skip, not a verification."
    echo "            Force a full re-hash with: FORCE_VERIFY=1"
  fi
  if [[ "$unverified" != "0" ]]; then
    echo "[verify]  ⚠ ${unverified}/${total} file(s) UNVERIFIED — the hub published no sha256 for them,"
    echo "            so nothing checked their contents (a size match is not a verification)."
    echo "            Re-fetch through the resilience ladder for a hash-gated pull:"
    echo "              python3 scripts/lib/profiles/hf_fetch.py ${repo} \\"
    echo "                --local-dir ${MODEL_DIR}/${subdir} --verify-in-place"
  fi
}

download_weight_key() {
  local key="$1"
  load_weight_recipe "$key"
  echo "[model]   Downloading ${WEIGHT_LABEL:-$key} ..."
  _hf_download_repo "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_FILES" "${WEIGHT_REVISION:-}"
  _verify_downloaded_files "$WEIGHT_REPO" "$WEIGHT_SUBDIR" "$WEIGHT_VERIFY_GLOB" "${WEIGHT_REVISION:-main}"
}

# #634 — honour the PRIMARY recipe's verify_glob (set by load_weight_recipe →
# line 100, e.g. "*.gguf" for GGUF keys), NOT a re-hardcoded *.safetensors, or
# every GGUF primary fetch (WEIGHTS=gguf/iq4ks) fails verify despite a good
# download.  An explicit VERIFY_GLOB_OVERRIDE still wins.
VERIFY_GLOB="${VERIFY_GLOB_OVERRIDE:-${VERIFY_GLOB}}"
_hf_download_repo "${MODEL_REPO}" "${MODEL_SUBDIR}" "${GGUF_FILES}" "${MODEL_REVISION:-}"
_verify_downloaded_files "${MODEL_REPO}" "${MODEL_SUBDIR}" "${VERIFY_GLOB}" "${MODEL_REVISION:-main}"

for extra_key in "${EXTRA_WEIGHT_KEYS[@]}"; do
  download_weight_key "$extra_key"
done

echo ""
echo ""

# ---------- Optional / companion draft models ----------
if [[ -n "${ALWAYS_DRAFT_KEY:-}" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  echo "[draft]   downloading required companion drafter ${ALWAYS_DRAFT_KEY} ..."
  download_weight_key "${ALWAYS_DRAFT_KEY}"
  echo ""
fi

if [[ "${WITH_DFLASH_DRAFT:-0}" == "1" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  if [[ -z "${DFLASH_KEY:-}" ]]; then
    echo "ERROR: WITH_DFLASH_DRAFT=1 is not wired for ${MODEL_NAME}." >&2
    exit 1
  fi
  echo "[dflash]  WITH_DFLASH_DRAFT=1 — downloading ${DFLASH_KEY} ..."
  download_weight_key "${DFLASH_KEY}"
  # setup.dflash is ONE key, but the engines need DIFFERENT drafters. For
  # qwen3.8-27b it resolves to `dflash2` (syvai W4A16) — correct for the vLLM
  # super*/ultra* tiers. The SGLang sgl/…-super* slugs need the UNQUANTIZED z-lab
  # release, and pointing them at the quantized one does NOT fail loudly: it drafts
  # garbage at accept len ~1.03 vs ~5.3 (decode ~38 vs ~171 tok/s) with no error and
  # a passing test suite (sglang#39087). Surfaced by a user hitting the
  # missing-drafter path on 2x 5090 (club-3090#1251).
  case "${DFLASH_KEY}" in
    qwen3.8-27b:dflash2)
      echo ""
      echo "[dflash]  NOTE: that is the vLLM drafter. To serve an SGLang"
      echo "[dflash]        sgl/qwen38-27b-*-super* slug, ALSO fetch the z-lab BF16 one:"
      echo "[dflash]          WEIGHT_KEY=qwen3.8-27b:dflash2-zlab bash scripts/setup.sh qwen3.8-27b"
      echo "[dflash]        (~3.85 GB. The quantized drafter silently collapses"
      echo "[dflash]         acceptance on SGLang - sglang#39087.)"
      ;;
  esac
  echo ""
else
  echo "[dflash]  Skipping DFlash draft model. Set WITH_DFLASH_DRAFT=1 to fetch it when a matching compose requires it."
fi

echo ""

if [[ "${WITH_PRISM_EAGLE3:-0}" == "1" ]] && [[ "${SKIP_MODEL:-0}" != "1" ]]; then
  if [[ -z "${PRISM_EAGLE3_KEY:-}" ]]; then
    echo "ERROR: WITH_PRISM_EAGLE3=1 is not wired for ${MODEL_NAME}: no prism_eagle3 in its profiles/models/<id>.yml setup block." >&2
    exit 1
  fi
  download_weight_key "${PRISM_EAGLE3_KEY}"
  echo ""
fi

# Note: vllm#40361 Marlin pad-sub-tile-n patched files are vendored in-repo
# at models/qwen3.6-27b/vllm/patches/vllm-marlin-pad/. Dual-card composes
# mount them via repo-relative paths — no host filesystem dependency, no
# clone needed. (Previous design required cloning a fork to /opt/ai/engines/vllm/primary/;
# refactored 2026-05-03 to vendor the two files in-repo, fixing #37.)


# Per-model "next steps" — different composes / served-model-name / port between
# models. Derived from the registry (`scripts/lib/registry-emit.sh --json`),
# replacing the hand-written per-model case whose arms new models silently
# missed (#914 class).
#
# Slug pick (deterministic): the model's curated default slug when it is
# FUNCTIONAL (status production/caveats — mirror of
# compose_registry.FUNCTIONAL_STATUSES), else the highest-ranked functional
# slug, else the highest-ranked slug overall. Non-functional pick ⇒ the launch
# hint carries --force, exactly like switch.sh's gate.
#
# ⚠️ Initialise every SAMPLE_* here regardless of derivation success. If the
# registry emit fails these must be EMPTY, never wrong — and `set -u` must
# never kill the script AFTER a fully successful download (DeepSeek-Flash hit
# exactly that; found by running setup.sh for real, 2026-08-07).
SAMPLE_CONTAINER=""
SAMPLE_COMPOSE_PATH=""
SAMPLE_PORT=""
SAMPLE_MODEL_NAME=""
SAMPLE_LAUNCH_HINT=""
NEXT_STEPS_NOTE=""
SAMPLE_SLUG=""
SAMPLE_STATUS=""
SETUP_MODEL_DISPLAY="$(model_label "${MODEL_NAME}")"
_reg_tmp="$(mktemp)"
# Don't blind-swallow a broken emit silently: an empty file just degrades the
# SAMPLE_* block to generic hints (documented above), never a wrong hint.
bash "${ROOT_DIR}/scripts/lib/registry-emit.sh" --json >"${_reg_tmp}" 2>/dev/null || true
if [[ -s "${_reg_tmp}" ]]; then
  eval "$(python3 - "${MODEL_NAME}" "${ROOT_DIR}" "${_reg_tmp}" <<'PY'
import json, os, re, shlex, sys

model, root, reg_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(reg_path, encoding="utf-8") as _fh:
    data = json.load(_fh)

RANK = {
    "production": 5,
    "caveats": 4,
    "preview": 3,
    "experimental": 2,
    "incubating": 1,
    "upstream-gated": 1,
    "deprecated": 0,
}
FUNCTIONAL = {"production", "caveats"}

variants = [v for v in data.get("variants", []) if v.get("model") == model]
defaults = [d["slug"] for d in data.get("defaults", []) if d.get("model") == model]


def q(v):
    return shlex.quote(str(v if v is not None else ""))


pick = None
for slug in defaults:
    c = [v for v in variants if v.get("slug") == slug]
    if c and c[0].get("status") in FUNCTIONAL:
        pick = c[0]
        break
if pick is None:
    pool = [v for v in variants if v.get("status") in FUNCTIONAL] or variants
    if pool:
        # sorted() is stable: ties keep registry order.
        pick = sorted(pool, key=lambda v: RANK.get(v.get("status", ""), 0), reverse=True)[0]
if pick is None:
    raise SystemExit(0)

slug = pick.get("slug", "")
status = pick.get("status", "")
print(f"SAMPLE_SLUG={q(slug)}")
print(f"SAMPLE_STATUS={q(status)}")
print(f"SAMPLE_CONTAINER={q(pick.get('container'))}")
print(f"SAMPLE_PORT={q(pick.get('port'))}")
print(f"SAMPLE_COMPOSE_PATH={q(pick.get('compose_path'))}")

# served_name is a first-class registry fact now (registry-emit --json); the
# direct compose grep below stays only as the fallback for a null field.
served = pick.get("served_name") or ""
if not served:
    try:
        with open(os.path.join(root, pick.get("compose_path") or ""), encoding="utf-8") as fh:
            m = re.search(r"--served-model-name\s+(?:-\s+)?(\S+)", fh.read())
        if m:
            served = m.group(1)
    except OSError:
        pass
print(f"SAMPLE_MODEL_NAME={q(served)}")

force = status not in FUNCTIONAL
print(f"SAMPLE_LAUNCH_HINT={q('bash scripts/switch.sh ' + ('--force ' if force else '') + slug)}")
if force:
    note = f"status '{status}' — launch needs --force (non-functional by default)."
    if status == "incubating":
        note += " Incubating slugs are hidden from 'switch.sh --list'; reveal with --list --all."
else:
    sibs = sorted(
        {
            v["slug"]
            for v in variants
            if v.get("status") in FUNCTIONAL and v.get("slug") != slug
        }
    )
    if sibs:
        lines = ["Functional variants:"]
        lines += [f"  bash scripts/switch.sh {s}" for s in sibs[:6]]
        if len(sibs) > 6:
            lines.append(f"  ... and {len(sibs) - 6} more: bash scripts/switch.sh --list")
        note = "\n".join(lines)
    else:
        note = ""
print(f"NEXT_STEPS_NOTE={q(note)}")
PY
)"
fi
rm -f "${_reg_tmp}"

echo "[setup] ✓ ${SETUP_MODEL_DISPLAY} downloaded."
echo "[setup] Next: bash scripts/launch.sh"
echo ""
# SAMPLE_LAUNCH_HINT is registry-derived (see the derivation above) and always
# names a real slug — with --force when the slug's status is non-functional.
# Only a failed registry emit leaves it empty; that fallback points at
# switch.sh's list rather than printing a models/<model>/vllm/compose path
# that may not exist (the #914 class).
if [[ -n "${SAMPLE_LAUNCH_HINT:-}" ]]; then
  echo "Next — launch it:"
  echo "  ${SAMPLE_LAUNCH_HINT}"
  echo "  docker logs -f ${SAMPLE_CONTAINER}"
else
  echo "Next — launch it:"
  echo "  bash scripts/switch.sh --list        # pick a slug for ${MODEL_NAME}"
  echo "  bash scripts/switch.sh <slug>"
fi
echo ""
echo "${NEXT_STEPS_NOTE}"
echo ""
if [[ -n "${SAMPLE_PORT:-}" && -n "${SAMPLE_MODEL_NAME:-}" ]]; then
  echo "Sanity test (after 'Application startup complete'):"
  echo "  curl -sf http://localhost:${SAMPLE_PORT}/v1/chat/completions \\"
  echo "    -H 'Content-Type: application/json' \\"
  echo "    -d '{\"model\":\"${SAMPLE_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France?\"}],\"max_tokens\":200}'"
fi
