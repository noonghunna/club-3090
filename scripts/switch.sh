#!/usr/bin/env bash
#
# Switch between club-3090 compose variants.
#
# Brings down whatever's currently running, brings up the new variant,
# and (optionally) waits for the server to report ready on /v1/models.
# Stateless — re-run any time you want a different config.
#
# Usage:
#   bash scripts/switch.sh <variant>            # switch + tail until ready
#   bash scripts/switch.sh <variant> --no-wait  # switch and return immediately
#   bash scripts/switch.sh --force <variant>    # skip hardware/free-VRAM preflight
#   bash scripts/switch.sh --list               # actionable variants on THIS machine (deprecated hidden) + defaults
#   bash scripts/switch.sh --list --all         # every variant — all GPU counts + deprecated
#   bash scripts/switch.sh --list-all           # alias for --list --all
#   bash scripts/switch.sh --defaults           # just the per-model defaults view
#   bash scripts/switch.sh --down               # just bring down whatever's up
#   bash scripts/switch.sh --set-default <slug>  # pin <slug> as YOUR default for its model (.env)
#   bash scripts/switch.sh --clear-default <model>  # remove your pinned default for <model>
#
# `<…>/default` tokens auto-resolve to a concrete slug (design §13.1):
#   <engine>/default        e.g. vllm/default — the maintainer's recommended
#                           config for that engine on the detected topology.
#   <engine>/<topo>/default e.g. vllm/dual/default — force the topology.
#   <model>/default         e.g. qwen3.6-27b/default — YOUR preferred config:
#                           your `.env` pin if set, else the curated pick
#                           (ENGINE_PREFERENCE walk) for the detected topology.
#
# Variant names are derived from the compose registry (the single source of
# truth); `bash scripts/switch.sh --list` is authoritative. A representative
# subset (engine/file, file is the docker-compose.<file>.yml stem):
#
#   Single-card vLLM:
#     vllm/default            48K + TQ3 + MTP + vision + tools (recommended)
#     vllm/long-vision        198K + TQ3 + vision (cliff-safe; Cliff 2 single-prompt >50K still applies)
#     vllm/long-text          180K + TQ3 + MTP + text-only (Balanced MTP — 60K single-prompt closed via v7.69 + #35975)
#     vllm/long-text-no-mtp   200K + TQ3 + no MTP + text-only (Max-context — same Cliff 2 closure, more KV pool, slower decode)
#     vllm/bounded-thinking   180K + TQ3 + structured-CoT FSM in reasoning (recommended grammar: DeepSeek scratchpad — 87.4% combined HE+/LCB v6)
#     vllm/tools-text         75K + fp8 + MTP + text-only (IDE agents — Cline / Cursor)
#     vllm/minimal            32K + fp8 (no Genesis, no spec-decode, simplest)
#
#   Dual-card vLLM (TP=2):
#     vllm/dual             262K + fp8 + 2 streams + vision (Qwen dual default)
#     vllm/dual4            262K + fp8 + 4 streams + vision (4× 3090 PCIe baseline)
#     vllm/dual4-dflash     262K + FP16 + DFlash N=5 + 2 streams + vision (4× 3090 code)
#     (NVLink is auto-detected at boot by every dual compose — no separate
#      nvlink-* variant. Force it with NVLINK_MODE=force_on if auto-detect misses.)
#     vllm/gemma-int8       Gemma-4-31B dual default — 262K + INT8 KV + vision (v0.21.0 + #40391 overlay)
#     vllm/gemma-mtp        Gemma-4-31B stable fallback — 32K + bf16 KV + vision (stock v0.22.0, no overlay)
#     (other Qwen dual variants — dflash / tq3 / bf16 / int8 — were deprecated
#      2026-05-31; see `switch.sh --list --all`.)
#
#   Single-card llama.cpp:
#     llamacpp/default      alias for llamacpp/mtp (Q4_K_M MTP, no vision)
#     llamacpp/mtp          Q4_K_M MTP + 200K (max-safe @ -ub 512; 131K @ -ub 1024 faster prefill) + q4_0 KV (fast ~60 TPS code; no vision; cliff-immune)
#     llamacpp/bounded-thinking Q4_K_M MTP + 200K + reasoning on + per-request GBNF grammar
#     llamacpp/mtp-vision   Q4_K_M MTP + 150K @ 1M-px + q4_0 KV + mmproj (multimodal; 4M-px = override, lower ctx)
#   Single-card ik_llama (IQ4_KS — ~0.5-0.8 GB leaner; best for VRAM-tight / WSL):
#     ik-llama/iq4ks-mtp         IQ4_KS MTP + 200K + q4_0 KV (own image: ikawrakow/ik-llama-cpp)
#     ik-llama/iq4ks-mtp-vision  IQ4_KS MTP + 160K @ 1M-px + q4_0 KV + mmproj (multimodal; 4M-px = override, lower ctx)
#
# Env overrides (rarely needed):
#   COMPOSE_BIN     Default: "docker compose" (set to e.g. "podman compose" if needed)
#   CLUB3090_GPU    Single-card GPU index override, e.g. "1" on a hetero rig
#   FORCE           Set to 1 to skip hardware/free-VRAM preflight
#   READY_URL       Default: http://localhost:8020/v1/models
#   READY_TIMEOUT   Default: 600 (seconds — longer for cold cudagraph capture)

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_BIN="${COMPOSE_BIN:-docker compose}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
LAUNCH_PROFILE="${LAUNCH_PROFILE:-${ROOT_DIR}/scripts/lib/profiles/launch_compat.py}"

# Load .env if present, so PORT / MODEL_DIR / etc. flow through to docker
# compose AND to the ready-URL probe below.
#
# Precedence matches docker compose (and launch.sh): a variable already set in
# the shell environment WINS over the .env file — so `export MODEL_DIR=…` is no
# longer clobbered by a stale .env entry (#425). We parse line-by-line instead
# of `source` (a) to honour that precedence per-variable and (b) to tolerate
# CRLF line endings from Windows editors (#187). Values are taken literally
# (no shell expansion), matching docker compose's own .env semantics.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  while IFS= read -r _env_line || [[ -n "$_env_line" ]]; do
    _env_line="${_env_line#"${_env_line%%[![:space:]]*}"}"   # strip leading whitespace
    _env_line="${_env_line%$'\r'}"                           # strip trailing CR (CRLF .env)
    [[ -z "$_env_line" || "$_env_line" == '#'* ]] && continue
    _env_line="${_env_line#export }"
    _env_key="${_env_line%%=*}"
    [[ "$_env_key" == "$_env_line" || -z "$_env_key" ]] && continue   # no '=' on the line
    [[ -n "${!_env_key+x}" ]] && continue                    # already set in env → shell wins
    _env_val="${_env_line#*=}"
    _env_val="${_env_val#\"}"; _env_val="${_env_val%\"}"     # strip surrounding double quotes
    _env_val="${_env_val#\'}"; _env_val="${_env_val%\'}"     # strip surrounding single quotes
    export "${_env_key}=${_env_val}"
  done < "${ROOT_DIR}/.env"
  unset _env_line _env_key _env_val
fi

# Surface the resolved MODEL_DIR + its source so the precedence is unambiguous
# (the exact confusion behind #425 / #187). Unset → the compose's built-in
# default applies; preflight_compose_deps notes that case.
if [[ -n "${MODEL_DIR:-}" ]]; then
  echo "[switch] MODEL_DIR=${MODEL_DIR}"
fi

# Variant tables are DERIVED from the single source of truth
# (scripts/lib/profiles/compose_registry.py COMPOSE_REGISTRY).
declare -A VARIANT_DEFAULT_PORT=()
declare -A VARIANTS=()
declare -A VARIANT_STATUS=()
declare -A VARIANT_STATUS_NOTE=()
# shellcheck source=lib/registry-emit.sh
source "${ROOT_DIR}/scripts/lib/registry-emit.sh"
derive_switch_variant_tables "${ROOT_DIR}"

# Container name patterns we'll bring down — covers all current composes
# AND any vllm/llama-cpp container we don't formally know about (catches
# locally-built variants and one-off `docker run` instances that would
# otherwise pin GPU memory invisibly to switch.sh).
RUNNING_PATTERN="^(vllm-|llama-cpp-)"


PRIMARY_MODEL="${PRIMARY_MODEL:-qwen3.6-27b}"

switch_topology_from_gpus() {
  local selector="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}" count=0
  if [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]; then
    IFS=',' read -ra _switch_gpu_tokens <<< "$selector"
    local token
    for token in "${_switch_gpu_tokens[@]}"; do
      token="${token//[[:space:]]/}"
      [[ -n "$token" ]] && count=$((count + 1))
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
  else
    count=1
  fi
  case "$count" in
    0|1) printf 'single' ;;
    2) printf 'dual' ;;
    4) printf 'multi4' ;;
    *) printf 'multi%s' "$count" ;;
  esac
}

resolve_default_variant() {
  # Resolves a `<…>/default` token to a concrete slug. Three forms (design
  # §13.1):
  #   <engine>/<topology>/default  → engine-recommendation, explicit topology
  #   <X>/default                  → dispatch on X: engine name → engine
  #                                   recommendation; model-id → the user's
  #                                   model default (.env pin ‖ curated walk)
  #   anything else                → passthrough (already a concrete slug)
  local variant="$1" engine topology target
  if [[ "$variant" =~ ^([^/]+)/(single|dual|multi[0-9]+)/default$ ]]; then
    engine="${BASH_REMATCH[1]}"
    topology="${BASH_REMATCH[2]}"
    if ! target="$(registry_default_target "$ROOT_DIR" "$PRIMARY_MODEL" "$engine" "$topology")"; then
      echo "ERROR: cannot resolve default variant '${variant}' for primary model ${PRIMARY_MODEL}." >&2
      exit 1
    fi
    printf '%s' "$target"
    return 0
  elif [[ "$variant" =~ ^([^/]+)/default$ ]]; then
    topology="$(switch_topology_from_gpus)"
    if ! target="$(x_default_dispatch "$ROOT_DIR" "$variant" "$topology" "$PRIMARY_MODEL")"; then
      echo "ERROR: cannot resolve default variant '${variant}'." >&2
      exit 1
    fi
    printf '%s' "$target"
    return 0
  fi
  printf '%s' "$variant"
}

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# --- PR-B: user-pinnable model defaults (.env) -------------------------------
ENV_FILE="${ROOT_DIR}/.env"

# Derive (model, pin-key) from a slug, or fail with a message. Echoes
# "<model>\t<pin-key>".
slug_model_and_pinkey() {
  local slug="$1" out
  if ! out="$(python3 - "$ROOT_DIR" "$slug" <<'PY_SLUGINFO'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import model_of_slug, model_default_pin_key  # noqa: E402
slug = sys.argv[2]
model = model_of_slug(slug)
if not model:
    print(f"unknown slug {slug!r} — run: scripts/switch.sh --list", file=sys.stderr)
    raise SystemExit(1)
print(f"{model}\t{model_default_pin_key(model)}")
PY_SLUGINFO
)"; then
    return 1
  fi
  printf '%s' "$out"
}

# Write KEY=VALUE into .env, replacing any existing line for KEY (round-trips
# with --clear-default). Preserves all other lines + ordering.
env_set_key() {
  local key="$1" value="$2" tmp
  tmp="$(mktemp)"
  if [[ -f "$ENV_FILE" ]]; then
    # Drop any existing assignment for KEY (with or without `export`).
    grep -vE "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" > "$tmp" || true
  fi
  printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mv "$tmp" "$ENV_FILE"
}

# Remove any assignment for KEY from .env (no-op if .env or the key is absent).
env_clear_key() {
  local key="$1" tmp
  [[ -f "$ENV_FILE" ]] || return 0
  tmp="$(mktemp)"
  grep -vE "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" > "$tmp" || true
  mv "$tmp" "$ENV_FILE"
}

set_default() {
  local slug="$1" info model key
  if [[ -z "${VARIANTS[$slug]:-}" ]]; then
    echo "[switch] ERROR: '${slug}' is not a known variant — can't pin it." >&2
    echo "[switch]        Run: bash scripts/switch.sh --list" >&2
    exit 1
  fi
  if ! info="$(slug_model_and_pinkey "$slug")"; then
    exit 1
  fi
  IFS=$'\t' read -r model key <<< "$info"
  env_set_key "$key" "$slug"
  echo "[switch] pinned '${slug}' as your default for ${model} (${key} in .env)."
  echo "[switch] bare 'launch.sh' / '${model%%/*}…' resolves there now; clear it with:"
  echo "[switch]   bash scripts/switch.sh --clear-default ${model}"
  exit 0
}

clear_default() {
  local model="$1" key
  key="$(python3 - "$ROOT_DIR" "$model" <<'PY_CLEARKEY'
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import model_default_pin_key  # noqa: E402
print(model_default_pin_key(sys.argv[2]))
PY_CLEARKEY
)"
  if [[ -f "$ENV_FILE" ]] && grep -qE "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE"; then
    env_clear_key "$key"
    echo "[switch] cleared your pinned default for ${model} (removed ${key} from .env)."
  else
    echo "[switch] no pinned default set for ${model} (${key} not in .env) — nothing to clear."
  fi
  exit 0
}

# Map a registry status word to the marker shown in --list and to launch
# gating. `production` → unmarked; `caveats` → "(caveats)"; the (NA) set
# (experimental/preview/upstream-gated/deprecated) → "(NA: <word>)".
status_marker() {
  case "$1" in
    production|"") printf '' ;;
    caveats)       printf '(caveats)' ;;
    *)             printf '(NA: %s)' "$1" ;;
  esac
}

# Map a topology word (as `switch_topology_from_gpus` emits it, or as a
# compose file's first path segment carries it) to a numeric rank, so we can
# compare "can this machine run that slug?". single=1, dual=2, multi*=3+.
# Unknown → 9 (sorts last; never filtered out by accident). Echoes the rank.
topology_rank() {
  case "$1" in
    single)  printf '1' ;;
    dual)    printf '2' ;;
    multi*)  printf '3' ;;
    *)       printf '9' ;;
  esac
}

# Is GPU detection RELIABLE for the hardware filter? True iff we have a
# concrete signal: an explicit selector (CUDA/NVIDIA_VISIBLE_DEVICES naming
# specific GPUs) OR nvidia-smi present AND reporting ≥1 GPU. Without either we
# can't trust the count — switch_topology_from_gpus falls back to "single" in
# that case, but for the --list filter we must FAIL OPEN (show all) rather than
# hide dual/multi based on a guess. Returns 0 (reliable) / 1 (unknown).
list_gpu_detect_reliable() {
  local selector="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  if [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' ')"
    [[ "${n:-0}" -ge 1 ]] && return 0
  fi
  return 1
}

# Human GPU-count label for a detected topology word, for the filter note
# (e.g. "single" → "1", "dual" → "2", "multi4" → "4", "multi6" → "6").
topology_gpu_label() {
  case "$1" in
    single)  printf '1' ;;
    dual)    printf '2' ;;
    multi*)  printf '%s' "${1#multi}" ;;
    *)       printf '?' ;;
  esac
}

list_variants() {
  # Grouped by model · topology so each slug's binding is visible at a glance.
  # VARIANTS stores "<engine>|<dir>|<file>" where
  #   dir  = models/<model>/<engine>/compose      → model is dir field 2
  #   file = <topology>/<quant>/<serving>.yml      → topology/quant/serving
  # (the registry emitter splits compose_path on "/compose/", so dir stops at
  #  /compose and the topology+quant live in file). Engine is the slug prefix.
  # The trailing column is the health marker derived from the registry status.
  #
  # Hardware filter (PR-C): a dual/multi slug can't run on a 1-GPU box, so by
  # default we hide slugs whose topology rank exceeds what this machine can run
  # (detected via switch_topology_from_gpus, which reads CUDA/NVIDIA_VISIBLE_-
  # DEVICES then nvidia-smi). `--list --all` (LIST_ALL=1) shows everything for
  # discoverability. Fail-open: if detection is unavailable we show ALL rather
  # than hide based on a failed probe.
  local show_all="${LIST_ALL:-0}" detected_topo max_rank
  detected_topo="$(switch_topology_from_gpus 2>/dev/null || true)"
  if [[ -z "$detected_topo" ]] || ! list_gpu_detect_reliable; then
    # Detection unavailable / count unknown → fail-open, show everything. We do
    # NOT hide dual/multi off the back of switch_topology_from_gpus's "single"
    # fallback when there's no real signal (no selector, no nvidia-smi).
    show_all=1
    max_rank=9
  else
    max_rank="$(topology_rank "$detected_topo")"
  fi

  echo "Available variants — grouped by model · topology (right cols: <quant>/<serving>.yml · max-ctx + health):"
  echo "  Health: bare max-ctx = production · (caveats, <ctx>) = works w/ documented limits · (NA: …, <ctx>) = needs --force"
  echo "  Context: a single value = registry matches the compose default · 'A/B' = validated(registry)/compose-default mismatch"

  # Counts: split into VISIBLE vs HIDDEN by the hardware filter, so the header
  # reflects what's actually shown (+ how many were hidden). Health split is
  # over the VISIBLE set; the by-topology hidden tally drives the note.
  local _prod=0 _cav=0 _na=0 _hidden=0 _dep_hidden=0
  declare -A _seen_models=() _hidden_by_topo=()
  for v in "${!VARIANTS[@]}"; do
    IFS='|' read -r _e _d _f <<< "${VARIANTS[$v]}"
    IFS=/ read -ra _ds <<< "$_d"
    IFS=/ read -ra _fs <<< "$_f"
    local _vtopo="${_fs[0]:-unknown}" _vrank
    _vrank="$(topology_rank "$_vtopo")"
    # Hide deprecated by default (tombstoned / flagged for removal); --all reveals.
    if [[ "$show_all" != "1" && "${VARIANT_STATUS[$v]:-production}" == "deprecated" ]]; then
      _dep_hidden=$((_dep_hidden + 1))
      continue
    fi
    if [[ "$show_all" != "1" && "$_vrank" -gt "$max_rank" ]]; then
      _hidden=$((_hidden + 1))
      _hidden_by_topo["$_vtopo"]=$(( ${_hidden_by_topo["$_vtopo"]:-0} + 1 ))
      continue
    fi
    _seen_models["${_ds[1]:-?}"]=1
    case "${VARIANT_STATUS[$v]:-production}" in
      production) _prod=$((_prod + 1)) ;;
      caveats)    _cav=$((_cav + 1)) ;;
      *)          _na=$((_na + 1)) ;;
    esac
  done
  local _visible=$(( _prod + _cav + _na ))
  # List the hidden topologies in rank order so both the header tally and the
  # filter note name exactly what's missing (e.g. "dual/multi4"). Pure bash —
  # no external grep/sort dependency.
  local _topo_list="" _t
  local _hidden_topos
  _hidden_topos="$(
    for _t in "${!_hidden_by_topo[@]}"; do
      printf '%s\t%s\n' "$(topology_rank "$_t")" "$_t"
    done | sort -k1,1n -k2,2 | cut -f2
  )"
  while IFS= read -r _t; do
    [[ -n "$_t" ]] || continue
    _topo_list="${_topo_list:+$_topo_list/}$_t"
  done <<< "$_hidden_topos"
  local _hidden_note=""
  if [[ "$_hidden" -gt 0 ]]; then
    _hidden_note="  (+${_hidden} ${_topo_list} hidden — --all)"
  fi
  local _dep_note=""
  if [[ "$_dep_hidden" -gt 0 ]]; then
    _dep_note="  (+${_dep_hidden} deprecated hidden — --all)"
  fi
  echo "  Models: ${#_seen_models[@]} · variants: ${_visible} (${_prod} production · ${_cav} caveats · ${_na} NA)${_hidden_note}${_dep_note}"

  {
    for v in "${!VARIANTS[@]}"; do
      IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
      IFS=/ read -ra dseg <<< "$dir"    # dseg[1] = model
      IFS=/ read -ra fseg <<< "$file"   # fseg[0]=topology fseg[1]=quant fseg[2]=serving
      topo="${fseg[0]:-unknown}"
      rank="$(topology_rank "$topo")"
      if [[ "$show_all" != "1" && "${VARIANT_STATUS[$v]:-production}" == "deprecated" ]]; then
        continue
      fi
      if [[ "$show_all" != "1" && "$rank" -gt "$max_rank" ]]; then
        continue
      fi
      marker="$(status_marker "${VARIANT_STATUS[$v]:-production}")"
      printf '%s\t%d\t%s\t%s\t%s/%s\t%s\t%s\n' \
        "${dseg[1]:-?}" "$rank" "$topo" "$v" "${fseg[1]:-?}" "${fseg[2]:-${file}}" "$marker" "${VARIANT_CTX[$v]:-}"
    done
  } | sort -t$'\t' -k1,1 -k2,2n -k4,4 | awk -F'\t' '
    { rows[NR] = $0; cnt[$1]++ }
    END {
      for (i = 1; i <= NR; i++) {
        split(rows[i], f, "\t")
        if (f[1] != m) { printf "\n%s  (%d variants)\n", f[1], cnt[f[1]]; m = f[1]; t = "" }
        tl = (f[3] == t ? "" : f[3]); t = f[3]
        ann = f[6]; ctx = f[7]
        if (ctx != "") {
          if (ann == "") ann = ctx                  # production: bare max-ctx (stays "unmarked")
          else sub(/\)$/, ", " ctx ")", ann)         # caveats / NA: fold ctx into the paren
        }
        printf "  %-8s %-34s %-36s %s\n", tl, f[4], f[5], ann
      }
    }
  '
  # Don't silently hide: one-line note when (and only when) the filter dropped
  # something. No note under --all or when nothing was hidden.
  if [[ "$_hidden" -gt 0 ]]; then
    local _gpu_label
    _gpu_label="$(topology_gpu_label "$detected_topo")"
    echo
    echo "(showing ${detected_topo}-GPU configs for this ${_gpu_label}-GPU machine — use --list --all for ${_topo_list})"
  fi
  echo
  show_defaults_view
  echo
  echo "Switch to one:  bash scripts/switch.sh <variant>"
  echo "Or via wizard:  bash scripts/launch.sh   (or: launch.sh --variant <variant>)"
  exit 0
}

# Discoverability (design §7): per model, what `<model>/default` resolves to on
# the DETECTED topology, marked user-pin vs curated, with a hint to pin. Shared
# between `--list` (appended) and `--defaults` (standalone). Reads the .env pin
# straight from the loaded environment (callers load .env above).
show_defaults_view() {
  local topology
  topology="$(switch_topology_from_gpus)"
  echo "Defaults — what \`<model>/default\` resolves to on this rig (${topology}):"
  echo "  (pin = your .env pin · curated = ENGINE_PREFERENCE walk · — = none for this topology)"
  local models model pin_key pin_value resolved source note
  models="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_set; print('\n'.join(sorted(model_set())))")"
  while IFS= read -r model; do
    [[ -n "$model" ]] || continue
    pin_key="$(python3 -c "import sys; sys.path.insert(0,'$ROOT_DIR'); from scripts.lib.profiles.compose_registry import model_default_pin_key; print(model_default_pin_key('$model'))")"
    pin_value="${!pin_key:-}"
    note=""
    if resolved="$(model_default_target "$ROOT_DIR" "$model" "$topology" 2>/dev/null)"; then
      if [[ -n "$pin_value" && "$resolved" == "$pin_value" ]]; then
        source="pin"
      elif [[ -n "$pin_value" ]]; then
        source="curated"
        note="  (your pin ${pin_value} was ignored — invalid/mismatched; see warnings)"
      else
        source="curated"
      fi
      printf '  %-18s %-32s [%s]%s\n' "$model" "$resolved" "$source" "$note"
    else
      printf '  %-18s %-32s [%s]\n' "$model" "—" "pick explicitly"
    fi
  done <<< "$models"
  echo "  Pin your own:    bash scripts/switch.sh --set-default <slug>"
  echo "  Clear a pin:     bash scripts/switch.sh --clear-default <model>"
}

defaults_view_standalone() {
  show_defaults_view
  exit 0
}

down_running() {
  local running
  running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E "$RUNNING_PATTERN" || true)
  if [[ -z "$running" ]]; then
    echo "[switch] no club-3090 container running"
    return
  fi
  for c in $running; do
    echo "[switch] bringing down: ${c}"
    # find the compose dir from the container's labels — fallback to direct stop
    local lbl_dir lbl_file
    lbl_dir=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.working_dir"}}' "$c" 2>/dev/null || true)
    lbl_file=$(docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.config_files"}}' "$c" 2>/dev/null || true)
    if [[ -n "$lbl_dir" && -n "$lbl_file" ]]; then
      (cd "$lbl_dir" && ${COMPOSE_BIN} -f "$lbl_file" down) || docker stop "$c" >/dev/null
    else
      docker stop "$c" >/dev/null
    fi
  done
}

gpu_preflight() {
  # Catch the "switch.sh said no club-3090 container running but GPU is
  # still pinned at 22 GiB and the new container OOMs at boot" failure
  # mode. down_running() only catches docker containers we manage; this
  # function catches anything else (out-of-band vllm/ollama/training
  # processes, exited containers that didn't release GPU memory cleanly,
  # etc.). Skip with FORCE=1 if you know what you're doing.
  if [[ "${FORCE:-0}" == "1" ]]; then
    echo "[switch] FORCE=1 — skipping GPU pre-flight"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  # Free MiB per GPU. Tolerate small overhead (driver, X server) — abort
  # if any selected GPU has <80% of its total memory free.
  local mem_query
  mem_query=$(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null) || return
  local selector="${NVIDIA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
  local selector_specific=0
  if [[ -n "$selector" && "$selector" != "all" && "$selector" != "void" ]]; then
    selector_specific=1
  fi
  local bad=0
  while IFS=',' read -r idx free total; do
    free=$(echo "$free" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    idx=$(echo "$idx" | tr -d ' ')
    [[ -z "$free" || -z "$total" ]] && continue
    if [[ "$selector_specific" -eq 1 && ",${selector}," != *",${idx},"* ]]; then
      continue
    fi
    # Require ≥80% free. Compose default gpu-memory-utilization is 0.92.
    local need=$(( total * 80 / 100 ))
    if [[ "$free" -lt "$need" ]]; then
      if [[ "$bad" -eq 0 ]]; then
        echo "[switch] ERROR: GPU memory pre-flight failed." >&2
        echo "[switch]        Something is still pinning GPU memory after down_running()." >&2
        echo "[switch]        Per-GPU state (free / total MiB; need ≥80% free):" >&2
      fi
      echo "[switch]          GPU $idx: $free / $total MiB free  (need ≥ $need)" >&2
      bad=1
    fi
  done <<< "$mem_query"

  if [[ "$bad" -eq 1 ]]; then
    echo "[switch]" >&2
    echo "[switch]        Holding processes:" >&2
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$apps" ]]; then
      while IFS= read -r line; do
        echo "[switch]          $line" >&2
      done <<< "$apps"
    else
      echo "[switch]          (nvidia-smi shows no compute apps — likely a zombie process or driver state)" >&2
    fi
    echo "[switch]" >&2
    echo "[switch]        Common fixes:" >&2
    echo "[switch]          docker ps -a | grep -E 'vllm|llama'       # find stopped containers" >&2
    echo "[switch]          docker rm \$(docker ps -aq --filter status=exited)" >&2
    echo "[switch]          fuser -v /dev/nvidia*                     # find host process holding the device" >&2
    echo "[switch]" >&2
    echo "[switch]        Override (skip this check):  FORCE=1 bash scripts/switch.sh ${VARIANT}" >&2
    exit 1
  fi
}

export_variant_engine_pin() {
  local variant="$1" output line key value
  [[ "$variant" == vllm/* ]] || return 0
  if ! output="$(python3 "$LAUNCH_PROFILE" resolve-variant-pin --variant "$variant" --format shell 2>&1)"; then
    echo "$output" >&2
    exit 2
  fi
  while IFS='=' read -r key value; do
    [[ -n "$key" ]] || continue
    case "$key" in
      VLLM_NIGHTLY_SHA) export VLLM_NIGHTLY_SHA="$value" ;;
      VLLM_IMAGE) export VLLM_IMAGE="$value" ;;
      *) echo "[switch] ERROR: unexpected engine pin export: $key" >&2; exit 2 ;;
    esac
  done <<< "$output"
  if [[ -n "${VLLM_IMAGE:-}" ]]; then
    if [[ -n "${VLLM_NIGHTLY_SHA:-}" ]]; then
      echo "[switch] vLLM image override: ${VLLM_IMAGE} (profile nightly SHA ${VLLM_NIGHTLY_SHA})"
    else
      echo "[switch] vLLM image: ${VLLM_IMAGE}"
    fi
  else
    echo "[switch] vLLM nightly SHA: ${VLLM_NIGHTLY_SHA:-unset}"
  fi
}

status_gate() {
  # Lifecycle gate (PR-A health flag). production → launch silently;
  # caveats → launch with a one-line notice; the (NA) set
  # (experimental/preview/upstream-gated/deprecated) → warn + require --force.
  local v="$1" status note
  status="${VARIANT_STATUS[$v]:-production}"
  note="${VARIANT_STATUS_NOTE[$v]:-}"
  case "$status" in
    production)
      ;;
    caveats)
      echo "[switch] NOTE: '${v}' is ⚠️ production-with-caveats.${note:+  ${note}}"
      ;;
    *)
      if [[ "${FORCE:-0}" != "1" ]]; then
        echo "[switch] ERROR: '${v}' is (NA: ${status}) — not a reliable config.${note:+  ${note}}" >&2
        echo "[switch]        It is surfaced for visibility, but won't launch without an explicit override." >&2
        echo "[switch]        Re-run with --force if you know what you're doing:" >&2
        echo "[switch]          bash scripts/switch.sh --force ${v}" >&2
        exit 1
      fi
      echo "[switch] WARNING: forcing (NA: ${status}) variant '${v}'.${note:+  ${note}}"
      ;;
  esac
}

up_variant() {
  local v="$1"
  if [[ -z "${VARIANTS[$v]:-}" ]]; then
    echo "ERROR: unknown variant '${v}'." >&2
    echo "Run: bash scripts/switch.sh --list" >&2
    exit 1
  fi
  status_gate "$v"
  IFS='|' read -r eng dir file <<< "${VARIANTS[$v]}"
  local full_dir="${ROOT_DIR}/${dir}"
  if [[ ! -f "${full_dir}/${file}" ]]; then
    echo "ERROR: compose file missing at ${full_dir}/${file}" >&2
    exit 1
  fi

  # Pre-up sanity:
  #  - genesis_pin: warn if on-disk Genesis tree differs from GENESIS_PIN in setup.sh
  #  - repo_drift: warn if local HEAD is behind origin/master
  #  - compose_deps: HARD error if compose mounts a model dir that doesn't exist on host
  #    (catches the "you didn't WITH_DFLASH_DRAFT=1 then tried dual-dflash-noviz" case;
  #     see club-3090#37 — this is the canonical fix raphael / snoby asked for)
  #  - kv_format_hint: soft warn if VRAM class needs --kv-cache-dtype override (#47)
  if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
    # shellcheck source=preflight.sh
    source "${ROOT_DIR}/scripts/preflight.sh"
    preflight_genesis_pin "${ROOT_DIR}" || true
    preflight_repo_drift "${ROOT_DIR}" || true
    preflight_compose_deps "${full_dir}/${file}" || exit 1
    if [[ "$eng" == "vllm" ]]; then
      preflight_compose_hardware "${full_dir}/${file}" "$v" "${FORCE:-0}" || exit 1
    fi
    preflight_kv_format_hint "${full_dir}/${file}" || true
  fi
  gpu_preflight

  echo "[switch] bringing up: ${v}  (${dir}/${file})"
  export_variant_engine_pin "$v"
  (cd "${full_dir}" && ${COMPOSE_BIN} -f "${file}" up -d)
}

resolve_ready_url() {
  # Precedence: $READY_URL (full override) → $PORT (port only, host=localhost)
  # → per-variant default port from VARIANT_DEFAULT_PORT.
  local variant="$1"
  if [[ -n "${READY_URL:-}" ]]; then
    return 0
  fi
  local port="${PORT:-${VARIANT_DEFAULT_PORT[$variant]:-8020}}"
  READY_URL="http://localhost:${port}/v1/models"
}

wait_ready() {
  # Find the container we just brought up so we can detect crashes mid-boot
  # AND surface stage progress markers from its logs while we wait.
  local container
  container=$(docker ps --format '{{.Names}}' 2>/dev/null \
    | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b|vllm-gemma-4-31b)' | head -1)

  if [[ -z "$container" ]]; then
    # Compose started but no container is up — almost always a syntax error
    # or env-var issue caught before vLLM even started.
    echo "[switch] ERROR: no container running after 'compose up' — boot failed before vLLM started." >&2
    echo "[switch]        Run 'docker compose -f <file> logs' for the compose-level error." >&2
    exit 1
  fi

  echo "[switch] waiting for ${READY_URL} (container=${container}, timeout ${READY_TIMEOUT}s)..."
  local elapsed=0 step=4 last_marker=""
  until curl -sf -o /dev/null --max-time 3 "${READY_URL}"; do
    # CRASH DETECTION: if the container died, dump tail and exit fast — don't
    # silently burn through the full timeout on a dead server.
    local state
    state=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo missing)
    if [[ "$state" != "true" ]]; then
      local exit_code
      exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$container" 2>/dev/null || echo "?")
      echo "[switch] ERROR: container '${container}' is no longer running (state=${state}, exit=${exit_code})." >&2
      echo "[switch]        Last 30 log lines:" >&2
      docker logs --tail 30 "$container" 2>&1 | sed 's/^/[switch]   | /' >&2
      echo "[switch]        Full logs:  docker logs ${container}" >&2
      exit 1
    fi

    sleep $step
    elapsed=$((elapsed + step))

    # PROGRESS SIGNAL: surface boot-stage markers so users see WHAT vLLM is
    # doing, not just that it's "still waiting". The grep is selective — one
    # line per phase transition, not raw log streaming.
    local marker
    marker=$(docker logs --tail 50 "$container" 2>&1 | grep -oE \
      'Genesis Results: .* applied|Resolved architecture: \w+|Loading weights|Compilation finished|Memory profiling|Capturing CUDA graphs|Application startup complete' \
      | tail -1 || true)
    if [[ -n "$marker" && "$marker" != "$last_marker" ]]; then
      echo "[switch]   ${elapsed}s — ${marker}"
      last_marker="$marker"
    elif [[ $((elapsed % 30)) -eq 0 ]]; then
      echo "[switch]   ${elapsed}s elapsed, still waiting..."
    fi

    if [[ $elapsed -ge $READY_TIMEOUT ]]; then
      echo "[switch] timeout — server not ready after ${READY_TIMEOUT}s" >&2
      echo "[switch] tail logs:  docker logs --tail 100 ${container}" >&2
      exit 1
    fi
  done
  echo "[switch] ✓ ready (${elapsed}s)"
}

# --- arg parsing ---
WAIT=1
FORCE="${FORCE:-0}"
VARIANT=""
LIST_REQUESTED=0
LIST_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage ;;
    # --list is deferred (not run inline) so `--list --all` works in either
    # order; --all toggles the hardware filter off. --list-all is the sibling.
    --list) LIST_REQUESTED=1 ;;
    --all) LIST_ALL=1 ;;
    --list-all) LIST_REQUESTED=1; LIST_ALL=1 ;;
    --defaults) defaults_view_standalone ;;
    --set-default)
      [[ -n "${2:-}" ]] || { echo "ERROR: --set-default needs a <slug> (e.g. vllm/dual)." >&2; exit 1; }
      set_default "$2"
      ;;
    --clear-default)
      [[ -n "${2:-}" ]] || { echo "ERROR: --clear-default needs a <model> (e.g. qwen3.6-27b)." >&2; exit 1; }
      clear_default "$2"
      ;;
    --down) down_running; exit 0 ;;
    --no-wait) WAIT=0 ;;
    --force) FORCE=1 ;;
    --*) echo "Unknown flag: $1"; exit 1 ;;
    *)
      if [[ -n "$VARIANT" ]]; then
        echo "ERROR: multiple variants supplied: '${VARIANT}' and '$1'" >&2
        exit 1
      fi
      VARIANT="$1"
      ;;
  esac
  shift
done

# --list (possibly with --all) is a terminal action — run it once, after the
# whole arg vector is parsed, so order doesn't matter.
if [[ "$LIST_REQUESTED" -eq 1 ]]; then
  list_variants   # exits
fi
# --all only makes sense alongside --list / --list-all.
if [[ "$LIST_ALL" -eq 1 ]]; then
  echo "ERROR: --all only applies to --list (try: bash scripts/switch.sh --list --all)." >&2
  exit 1
fi

[[ -n "$VARIANT" ]] || usage
VARIANT="$(resolve_default_variant "$VARIANT")"

resolve_ready_url "${VARIANT}"
down_running
up_variant "${VARIANT}"
[[ $WAIT -eq 1 ]] && wait_ready
echo "[switch] done. Try:  curl -s ${READY_URL%/v1/models}/v1/models | jq ."
