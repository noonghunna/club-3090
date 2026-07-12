#!/usr/bin/env bash
#
# Quality-test wrapper around `benchlocal-cli` — measures behavioral quality
# (tool-call correctness, instruction-following, structured output, etc) of
# the running compose. Sits in the test pipeline between bench.sh and
# soak-test.sh:
#
#   verify.sh         — fast smoke ("does it serve")
#   verify-full.sh    — functional ("does everything work")
#   verify-stress.sh  — boundary ("does it survive stress")
#   bench.sh          — throughput ("what's the TPS")
#   quality-test.sh   — behavioral ("does it produce useful output")  ← THIS
#   soak-test.sh      — stability ("does it stay healthy over time")
#
# Catches what the operational tests miss: a compose can pass all 5 layers
# of operational testing and still ship with degraded tool-call accuracy or
# instruction-follow drift from quantization or Genesis env-var changes.
#
# Reference: docs/QUALITY_TEST.md
#
# Prereq: benchlocal-cli installed (see "Install" below) + a running compose.

set -euo pipefail

# ---- usage / help ------------------------------------------------------------

usage() {
  cat <<'EOF'
quality-test.sh — behavioral quality bench against a running compose

USAGE
  bash scripts/quality-test.sh [MODE | --pack PACK_ID] [OPTIONS]

MODES
  --quick    2 packs:  toolcall-15, instructfollow-15
             ~5-10 min, no Docker required
  --medium   5 packs:  + structoutput-15, dataextract-15, reasonmath-15  (DEFAULT)
             ~15-25 min, no Docker required
  --full     8 packs:  + bugfind-15, hermesagent-20, cli-40
             ~25-40 min, requires Docker (auto-starts sandbox containers)
  --reasoning
             Reasoning suite: humaneval-plus-30, lcb-v6-30, gpqa-diamond
             metadata gate, gsm-symbolic-30. Separate from --full; code
             packs require Docker.

  --pack PACK_ID   Run a single pack (overrides mode flag).
                   Available IDs:
                     toolcall-15  instructfollow-15  structoutput-15
                     dataextract-15  reasonmath-15
                     bugfind-15  cli-40  hermesagent-20  (require Docker)
  --scenario P/S   Run one pack-qualified scenario, e.g. cli-40/CLI-31 (repeatable;
                   intersects with --pack / mode when one is given, else the pack
                   set is derived from the selection).
  --scenarios-file F
                   Newline-delimited PACK_ID/SCENARIO_ID selections (# comments OK).
                   Curated probe sets live in scripts/scenario-sets/.
                   ⚠ Selection results are PARTIAL — labeled in the JSON
                   (selection + catalog_scenario_count) and never a /150 claim;
                   history ingestion and rescore refuse them without --allow-partial.
  --incremental    Journal each scored scenario to <save-json>.partial.jsonl
                   (fsynced) so an interrupted run is inspectable and resumable.
  --resume PATH    Resume a result.json or .partial.jsonl: restores the original
                   pack-set/selection/thinking/sampling/timeout config and runs
                   only the missing scenario/repeat arms. Mutually exclusive with
                   mode/--pack/--scenario/thinking/sampling/timeout flags.
  --allow-partial  Permit a partial (selection) result into --history-file / rescore.
                     humaneval-plus-30  lcb-v6-30  gsm-symbolic-30
                     gpqa-diamond  (gated metadata-only until access approved)

OPTIONS
  -h, --help       Show this help and exit
  --list-packs     List available packs and exit
  --no-sandboxed   On --full, skip the Docker sandbox packs (= --medium scope)
  --sandboxed-only Run only the 3 sandbox packs (bugfind-15, cli-40, hermesagent-20).
                   Skips the deterministic packs — useful when iterating on
                   sandbox verifiers without paying the deterministic-pack cost.

OPTIONS (extra)
  --model NAME     Pin the served-model-name. When set (here or via MODEL env),
                   we use YOUR value and never override it from /v1/models —
                   required for llama-swap / multi-model endpoints where
                   /v1/models returns the first (often wrong) registered model.
  --timeout-per-case N
                   Pass through to benchlocal-cli as --timeout-per-case N
                   (seconds). When NOT set, benchlocal-cli uses per-pack
                   metadata defaults (60s for the deterministic packs, 300s
                   for cli-40 / hermesagent-20, 1800s for aider-polyglot-30;
                   see benchlocal-cli #41). Set this only to override.
  --sandbox-log-dir DIR
                   Capture each sandboxed pack's container log to
                   DIR/sandbox-<pack_id>.log before teardown (forwarded to
                   benchlocal-cli). Without it, sandbox logs are lost on
                   container cleanup. Also settable via SANDBOX_LOG_DIR env.
  --progress / --no-progress
                   Toggle benchlocal-cli's per-scenario `[N/M]` live progress
                   to stderr. **Default ON** — long quality runs (--full,
                   --reasoning, --pack aider-polyglot-30, --pack cli-40) go
                   dark for 10-60 min without it, with no signal whether
                   anything is wrong mid-run. Pass --no-progress for CI / when
                   stderr volume matters. Also settable via PROGRESS=0/1 env.
  --sampling-from-server
                   Inherit sampling from the serving config instead of using
                   the pack's default temp=0. Omits sampling params from
                   requests so the server applies its own defaults (llama.cpp
                   --temp, vLLM --override-generation-config). Reads back via
                   GET /props and records the values. Tags the run as
                   non-canonical. Also settable via SAMPLING_FROM_SERVER=1 env.
  --enable-thinking
                   Forward to benchlocal-cli --enable-thinking so reasoning
                   models are evaluated with request-level thinking enabled
                   for every pack (overrides each pack's default_thinking).
                   Also settable via ENABLE_THINKING=1 env.
  --no-thinking
                   Forward to benchlocal-cli --no-thinking — force thinking
                   OFF for every pack, ignoring per-pack default_thinking
                   (the packs that default thinking-on: instructfollow-15,
                   reasonmath-15, bugfind-15, hermesagent-20 — plus the
                   --reasoning suite). Mutually exclusive with --enable-thinking.
                   Use for a clean all-off arm of a reasoning A/B. Also
                   settable via NO_THINKING=1 env.
  --thinking-max-tokens N
                   Forward to benchlocal-cli --thinking-max-tokens N. The
                   budget applies only to packs whose thinking gate resolves on
                   (pack default or --enable-thinking). Also settable via
                   THINKING_MAX_TOKENS env.
  --max-tokens N   Forward to benchlocal-cli --max-tokens N — overrides the
                   per-pack completion budget (~1024 default) for BOTH arms.
                   Use when a verbose model truncates the deterministic packs
                   (finish_reason=length) before emitting its final answer; the
                   thinking arm still uses --thinking-max-tokens if that is set.
                   Also settable via MAX_TOKENS env.

ENV VARS
  URL              Endpoint base URL (default: auto-detected via preflight,
                   falls back to http://localhost:8020)
  MODEL            Served model name. If set (env or --model), it's respected
                   verbatim — no /v1/models override. If UNSET, auto-detected
                   from /v1/models (fixes the wrong-name → HTTP 404 footgun on
                   single-model composes). --model and MODEL are equivalent.
  TIMEOUT_PER_CASE Per-scenario HTTP timeout override in seconds. UNSET means
                   benchlocal-cli's per-pack metadata default applies (60s for
                   the deterministic packs, 300s for cli-40 / hermesagent-20,
                   1800s for aider-polyglot-30; see benchlocal-cli #41).
                   --timeout-per-case is equivalent.
  ENABLE_THINKING Set to 1 to send request-level enable_thinking=true via
                   benchlocal-cli --enable-thinking. Default: 0.
  NO_THINKING     Set to 1 to force thinking off for every pack via
                   benchlocal-cli --no-thinking. Mutually exclusive with
                   ENABLE_THINKING. Default: 0.
  THINKING_MAX_TOKENS
                   Optional thinking budget passed through to benchlocal-cli.
                   Applies only to packs whose thinking gate resolves on.
  MAX_TOKENS       Optional completion budget passed through to benchlocal-cli
                   (--max-tokens) for BOTH arms — overrides the per-pack ~1024
                   default. Raise for verbose models that self-truncate the
                   deterministic packs. --max-tokens is equivalent.

EXAMPLES
  bash scripts/quality-test.sh                          # --medium against running compose
  bash scripts/quality-test.sh --quick                  # quicker, 2 packs only
  bash scripts/quality-test.sh --full                   # everything, needs Docker
  bash scripts/quality-test.sh --reasoning              # HE+/LCB/GSM/GPQA reasoning suite
  bash scripts/quality-test.sh --pack toolcall-15       # just the tool-call pack
  bash scripts/quality-test.sh --pack aider-polyglot-30 --timeout-per-case 3600
  URL=http://localhost:8030 bash scripts/quality-test.sh # against a different port

INSTALL benchlocal-cli (one-time)
  pip install git+https://github.com/noonghunna/benchlocal-cli.git
  # OR for development from source:
  pip install -e /path/to/benchlocal-cli

OUTPUT
  - Markdown table to stdout (paste-ready for BENCHMARKS quality rows)
  - JSON blob to results/quality/quality-<timestamp>.json (full detail)
  - Compact one-liner for the compose `Quality:` profile field

EOF
}

# ---- preamble ----------------------------------------------------------------

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint
fi

URL="${URL:-http://localhost:8020}"
# Track whether the user explicitly set MODEL (via env or the --model flag).
# If they did, we respect it and do NOT clobber it with the /v1/models
# auto-detect below — critical for llama-swap / multi-model endpoints where
# /v1/models returns the first (often wrong) registered model. Auto-detect
# only kicks in when the user left MODEL unset.
MODEL_EXPLICIT=0
[[ -n "${MODEL:-}" ]] && MODEL_EXPLICIT=1
MODEL="${MODEL:-qwen3.6-27b}"

# Track whether the user explicitly set TIMEOUT_PER_CASE (via env or
# --timeout-per-case flag). When unset, we DON'T pass --timeout-per-case to
# benchlocal-cli, so it uses per-pack metadata defaults (benchlocal-cli #41:
# 60s deterministic, 300s cli-40/hermes, 1800s aider). Passing 60 by default
# would have defeated those pack-aware budgets — the wrapper would override
# every agentic pack back to 60s.
# --progress is default-on so long quality runs surface per-scenario `[N/M]`
# lines to stderr instead of going dark for 30+ minutes. The buffered-stderr
# trap was painful enough to warrant making it default. Use --no-progress
# (or PROGRESS=0) to suppress for CI / log-volume-sensitive contexts.
PROGRESS="${PROGRESS:-1}"

TIMEOUT_PER_CASE_SET=0
if [[ -n "${TIMEOUT_PER_CASE:-}" ]]; then
  TIMEOUT_PER_CASE_SET=1
fi

# ---- arg parsing -------------------------------------------------------------

MODE="--medium"   # default
PACK=""
NO_SANDBOX=0
SANDBOXED_ONLY=0
LIST_PACKS=0
SANDBOX_LOG_DIR="${SANDBOX_LOG_DIR:-}"
SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
NO_THINKING="${NO_THINKING:-0}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-}"
MAX_TOKENS="${MAX_TOKENS:-}"
# #252: passthroughs to benchlocal-cli for the quality-baseline corpus —
# --repeat (n>=3 aggregate), --previous-result (diff vs a baseline), and a
# --save-json override (write the run to an explicit path, e.g. a baseline file).
REPEAT=""
PREVIOUS_RESULT=""
SAVE_JSON_OVERRIDE=""
# benchlocal #84/#85: scenario-level selection + incremental/resume passthroughs.
SCENARIOS=()
SCENARIOS_FILE=""
INCREMENTAL=0
RESUME=""
ALLOW_PARTIAL=0
MODE_EXPLICIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--medium|--full|--reasoning)
      MODE="$1"
      MODE_EXPLICIT=1
      shift
      ;;
    --pack)
      PACK="${2:-}"
      if [[ -z "$PACK" ]]; then
        echo "✗ --pack requires a pack id" >&2
        exit 2
      fi
      shift 2
      ;;
    --scenario)
      if [[ -z "${2:-}" ]]; then
        echo "✗ --scenario requires PACK_ID/SCENARIO_ID (e.g. cli-40/CLI-31)" >&2
        exit 2
      fi
      SCENARIOS+=("$2")
      shift 2
      ;;
    --scenarios-file)
      if [[ -z "${2:-}" || ! -f "${2:-}" ]]; then
        echo "✗ --scenarios-file requires an existing file (see scripts/scenario-sets/)" >&2
        exit 2
      fi
      SCENARIOS_FILE="$2"
      shift 2
      ;;
    --incremental)
      INCREMENTAL=1
      shift
      ;;
    --resume)
      if [[ -z "${2:-}" || ! -e "${2:-}" ]]; then
        echo "✗ --resume requires an existing results.json or .partial.jsonl" >&2
        exit 2
      fi
      RESUME="$2"
      shift 2
      ;;
    --allow-partial)
      ALLOW_PARTIAL=1
      shift
      ;;
    --model)
      MODEL="${2:-}"
      if [[ -z "$MODEL" ]]; then
        echo "✗ --model requires a served-model-name" >&2
        exit 2
      fi
      MODEL_EXPLICIT=1
      shift 2
      ;;
    --no-sandboxed)
      NO_SANDBOX=1
      shift
      ;;
    --sandboxed-only)
      SANDBOXED_ONLY=1
      shift
      ;;
    --list-packs)
      LIST_PACKS=1
      shift
      ;;
    --sandbox-log-dir)
      SANDBOX_LOG_DIR="${2:-}"
      if [[ -z "$SANDBOX_LOG_DIR" ]]; then
        echo "✗ --sandbox-log-dir requires a directory" >&2
        exit 2
      fi
      shift 2
      ;;
    --timeout-per-case)
      TIMEOUT_PER_CASE="${2:-}"
      if [[ -z "$TIMEOUT_PER_CASE" ]] || ! [[ "$TIMEOUT_PER_CASE" =~ ^[0-9]+$ ]]; then
        echo "✗ --timeout-per-case requires a positive integer (seconds)" >&2
        exit 2
      fi
      TIMEOUT_PER_CASE_SET=1
      shift 2
      ;;
    --sampling-from-server)
      SAMPLING_FROM_SERVER=1
      shift
      ;;
    --enable-thinking)
      ENABLE_THINKING=1
      shift
      ;;
    --no-thinking)
      NO_THINKING=1
      shift
      ;;
    --thinking-max-tokens)
      THINKING_MAX_TOKENS="${2:-}"
      if [[ -z "$THINKING_MAX_TOKENS" ]] || ! [[ "$THINKING_MAX_TOKENS" =~ ^[0-9]+$ ]]; then
        echo "✗ --thinking-max-tokens requires a positive integer" >&2
        exit 2
      fi
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:-}"
      if [[ -z "$MAX_TOKENS" ]] || ! [[ "$MAX_TOKENS" =~ ^[0-9]+$ ]]; then
        echo "✗ --max-tokens requires a positive integer" >&2
        exit 2
      fi
      shift 2
      ;;
    --repeat)
      REPEAT="${2:-}"
      if [[ -z "$REPEAT" ]] || ! [[ "$REPEAT" =~ ^[0-9]+$ ]]; then
        echo "✗ --repeat requires a positive integer" >&2
        exit 2
      fi
      shift 2
      ;;
    --previous-result)
      PREVIOUS_RESULT="${2:-}"
      if [[ -z "$PREVIOUS_RESULT" ]]; then
        echo "✗ --previous-result requires a path to a saved RunResult JSON" >&2
        exit 2
      fi
      shift 2
      ;;
    --save-json)
      SAVE_JSON_OVERRIDE="${2:-}"
      if [[ -z "$SAVE_JSON_OVERRIDE" ]]; then
        echo "✗ --save-json requires a path" >&2
        exit 2
      fi
      shift 2
      ;;
    --progress)
      PROGRESS=1
      shift
      ;;
    --no-progress)
      PROGRESS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "✗ unknown argument: $1" >&2
      echo "  run 'bash scripts/quality-test.sh --help' for usage." >&2
      exit 2
      ;;
  esac
done

# ---- prerequisite checks -----------------------------------------------------

if [[ "$ENABLE_THINKING" == "1" && "$NO_THINKING" == "1" ]]; then
  echo "✗ --enable-thinking and --no-thinking are mutually exclusive (force thinking on OR off, not both)" >&2
  exit 2
fi

# --resume restores pack-set/selection/thinking/sampling/timeout from the saved
# run — passing any of those alongside it would silently fork the config, so refuse.
if [[ -n "$RESUME" ]]; then
  _resume_conflicts=()
  # (if-form, not `[[ ]] &&` — a false condition would trip set -e)
  if [[ "$MODE_EXPLICIT" == "1" ]]; then _resume_conflicts+=("$MODE"); fi
  if [[ -n "$PACK" ]]; then _resume_conflicts+=(--pack); fi
  if [[ ${#SCENARIOS[@]} -gt 0 ]]; then _resume_conflicts+=(--scenario); fi
  if [[ -n "$SCENARIOS_FILE" ]]; then _resume_conflicts+=(--scenarios-file); fi
  if [[ "$SANDBOXED_ONLY" == "1" ]]; then _resume_conflicts+=(--sandboxed-only); fi
  if [[ -n "$REPEAT" ]]; then _resume_conflicts+=(--repeat); fi
  if [[ -n "$PREVIOUS_RESULT" ]]; then _resume_conflicts+=(--previous-result); fi
  if [[ "$ENABLE_THINKING" == "1" ]]; then _resume_conflicts+=(--enable-thinking); fi
  if [[ "$NO_THINKING" == "1" ]]; then _resume_conflicts+=(--no-thinking); fi
  if [[ -n "$THINKING_MAX_TOKENS" ]]; then _resume_conflicts+=(--thinking-max-tokens); fi
  if [[ -n "$MAX_TOKENS" ]]; then _resume_conflicts+=(--max-tokens); fi
  if [[ "$SAMPLING_FROM_SERVER" == "1" ]]; then _resume_conflicts+=(--sampling-from-server); fi
  if [[ ${#_resume_conflicts[@]} -gt 0 ]]; then
    echo "✗ --resume restores the original run configuration; drop: ${_resume_conflicts[*]}" >&2
    exit 2
  fi
fi

if ! command -v benchlocal-cli >/dev/null 2>&1; then
  cat >&2 <<EOF
✗ benchlocal-cli not found on \$PATH

Install it (one-time):
  pip install git+https://github.com/noonghunna/benchlocal-cli.git

Or from a local checkout:
  pip install -e /path/to/benchlocal-cli

See docs/QUALITY_TEST.md for full setup.
EOF
  exit 127
fi

if [[ "$LIST_PACKS" == "1" ]]; then
  benchlocal-cli list
  exit 0
fi

if ! curl -sf -m 5 "${URL}/v1/models" >/dev/null 2>&1; then
  echo "✗ endpoint ${URL}/v1/models not responding" >&2
  echo "  bring up a compose first: bash scripts/launch.sh" >&2
  exit 1
fi

# Resolve the served model id from /v1/models. Behaviour depends on whether
# the user explicitly set MODEL:
#   - MODEL unset  → trust the endpoint, auto-detect (fixes the common "wrong
#                    model name → HTTP 404" footgun on single-model composes).
#   - MODEL set    → respect the user's value, do NOT override. Only warn if
#                    the endpoint disagrees. This is the llama-swap / multi-model
#                    case: /v1/models returns the first registered model (often
#                    the wrong one), and clobbering the user's choice routes the
#                    whole run at the wrong model (see disc #152, @ampersandru).
DETECTED_MODEL=$(curl -sf -m 5 "${URL}/v1/models" 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null \
  || echo "")
if [[ -n "$DETECTED_MODEL" && "$DETECTED_MODEL" != "$MODEL" ]]; then
  if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    echo "[quality-test] NOTE: endpoint /v1/models reports '${DETECTED_MODEL}', but you set MODEL='${MODEL}' — using YOUR value." >&2
    echo "[quality-test]   (Expected on llama-swap/multi-model endpoints. Leave MODEL unset to auto-detect the first served model instead.)" >&2
  else
    echo "[quality-test] model id auto-detected from endpoint: ${DETECTED_MODEL} (set MODEL=... or --model to pin it)" >&2
    MODEL="$DETECTED_MODEL"
  fi
fi

# hermesagent-20 runs its agent inside a Docker sandbox container. Localhost-style
# URLs (localhost/127.x/[::1]) inside the container resolve to the container itself,
# not the host's vLLM. Auto-set BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1 so benchlocal-cli
# (a) adds --add-host=host.docker.internal:host-gateway to the sandbox container, and
# (b) rewrites the model endpoint URL to use host.docker.internal:<port> for the
# hermes-agent's outbound API calls. Skip if already set (user override) or if URL
# already uses host.docker.internal / a non-loopback host (real LAN IP, k8s service).
if [[ -z "${BENCHLOCAL_HERMES_RESOLVE_LOCALHOST:-}" ]] \
   && [[ "$URL" =~ ^https?://(localhost|127\.|\[::1\]) ]]; then
  export BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1
  echo "[quality-test] localhost URL detected — auto-set BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1 for hermes sandbox endpoint rewrite" >&2
fi

server_reasoning_on() {
  if curl -sf -m 3 "${URL}/props" 2>/dev/null | python3 -c '
import json, sys
try:
    obj = json.load(sys.stdin)
except Exception:
    sys.exit(1)

def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            lk = str(k).lower()
            if lk in {"reasoning", "enable_reasoning"}:
                if v is True or str(v).lower() in {"1", "true", "on", "yes"}:
                    return True
            if walk(v):
                return True
    elif isinstance(x, list):
        return any(walk(v) for v in x)
    return False
sys.exit(0 if walk(obj) else 1)
' >/dev/null 2>&1; then
    return 0
  fi
  if [[ -n "${CONTAINER:-}" && "${CONTAINER:-}" != "none" ]] \
     && command -v docker >/dev/null 2>&1 \
     && docker inspect "$CONTAINER" >/dev/null 2>&1; then
    docker inspect "$CONTAINER" 2>/dev/null \
      | grep -Eq -- '(--reasoning[= ]+on|"--reasoning"[[:space:]]*,[[:space:]]*"on")' && return 0
  fi
  return 1
}

if [[ "$ENABLE_THINKING" != "1" && "$NO_THINKING" != "1" ]] && server_reasoning_on; then
  echo "[quality-test] WARN: server appears to have reasoning enabled, but --enable-thinking is not forced. Pack defaults still apply; use --enable-thinking or ENABLE_THINKING=1 to force thinking on for every pack (or --no-thinking / NO_THINKING=1 to force it off)." >&2
fi

# ---- sandbox-image preflight (--full / --sandboxed-only) --------------------
# The sandboxed packs (BugFind / CLI / Hermes) need pre-built Docker images that
# are NOT auto-pulled. benchlocal-cli's own mid-run hint points at a relative
# `tools/build-sandboxes.sh` that only exists inside a benchlocal-cli *checkout*
# (not a pip install, and not here) — so surface the correct steps UP FRONT
# instead of letting users discover a dead path mid-run (club-3090 #492).
# Does a scenario selection touch the Docker-sandboxed packs? (drives the
# same image preflight that --full gets — cli-40 probes are the primary use)
SELECTION_HAS_SANDBOX=0
if [[ ${#SCENARIOS[@]} -gt 0 || -n "$SCENARIOS_FILE" ]]; then
  _sel_lines="$(printf '%s\n' ${SCENARIOS[@]+"${SCENARIOS[@]}"})"
  if [[ -n "$SCENARIOS_FILE" ]]; then
    _sel_lines+=$'\n'"$(command grep -vE '^[[:space:]]*(#|$)' "$SCENARIOS_FILE" 2>/dev/null || true)"
  fi
  if command grep -qE '^(bugfind-15|cli-40|hermesagent-20)/' <<<"$_sel_lines"; then
    SELECTION_HAS_SANDBOX=1
  fi
fi

if { [[ -z "$PACK" ]] && { [[ "$MODE" == "--full" && "$NO_SANDBOX" != "1" ]] || [[ "$SANDBOXED_ONLY" == "1" ]]; }; } || [[ "$SELECTION_HAS_SANDBOX" == "1" && "$NO_SANDBOX" != "1" ]]; then
  _sb_missing=()
  _sb_stale=()
  if ! command -v docker >/dev/null 2>&1; then
    _sb_missing=("Docker not found on PATH")
  else
    # Install time of the benchlocal-cli console script — rewritten on every
    # (re)install, so it's a portable "CLI last updated" timestamp that works
    # for pip-from-git AND editable-checkout installs alike.
    _bl_bin="$(command -v benchlocal-cli || true)"
    _bl_mtime=0
    [[ -n "$_bl_bin" ]] && _bl_mtime="$(stat -c %Y "$_bl_bin" 2>/dev/null || echo 0)"
    # Editable-checkout installs (maintainer rigs): `git pull` updates the code
    # WITHOUT rewriting the console script, so the script mtime under-reports
    # "CLI last updated". Take max(script mtime, checkout last-commit time).
    if [[ -n "$_bl_bin" ]]; then
      _bl_py="$(head -1 "$_bl_bin" 2>/dev/null | sed 's/^#!//')"
      _bl_src_dir="$([[ -x "$_bl_py" ]] && "$_bl_py" -c 'import importlib.metadata as m, json
try:
    d = json.loads(m.distribution("benchlocal-cli").read_text("direct_url.json") or "{}")
    u = d.get("url", "")
    print(u[7:] if (d.get("dir_info") or {}).get("editable") and u.startswith("file://") else "")
except Exception:
    pass' 2>/dev/null)"
      if [[ -n "$_bl_src_dir" && -d "$_bl_src_dir" ]]; then
        _bl_commit_ts="$(git -C "$_bl_src_dir" log -1 --format=%ct 2>/dev/null || echo 0)"
        [[ "$_bl_commit_ts" -gt "$_bl_mtime" ]] && _bl_mtime="$_bl_commit_ts"
      fi
    fi
    for _img in benchlocal-sandbox-bugfind benchlocal-sandbox-cli benchlocal-sandbox-hermes; do
      if ! docker image inspect "${_img}:latest" >/dev/null 2>&1; then
        _sb_missing+=("${_img}:latest")
        continue
      fi
      # Staleness heuristic: the image was built BEFORE the currently-installed
      # benchlocal-cli. If that update touched sandbox sources (verifiers,
      # harness, deps), results run against the OLD behavior — the exact
      # incident class where user rigs kept scoring on pre-fix sandboxes until
      # told to rebuild manually. Heuristic (an unrelated reinstall also trips
      # it), hence WARN not abort.
      _img_created="$(docker image inspect "${_img}:latest" --format '{{.Created}}' 2>/dev/null || true)"
      if [[ -n "$_img_created" && "$_bl_mtime" -gt 0 ]]; then
        _img_epoch="$(date -d "$_img_created" +%s 2>/dev/null || echo 0)"
        if [[ "$_img_epoch" -gt 0 && "$_img_epoch" -lt "$_bl_mtime" ]]; then
          _sb_stale+=("${_img}:latest (built $(date -d "@${_img_epoch}" '+%F %H:%M') < CLI updated $(date -d "@${_bl_mtime}" '+%F %H:%M'))")
        fi
      fi
    done
  fi
  if [[ ${#_sb_missing[@]} -gt 0 ]]; then
    if [[ "$SANDBOXED_ONLY" == "1" ]]; then
      # --sandboxed-only with nothing to run in is a guaranteed-useless run:
      # refuse up front instead of warn-and-skip-everything (#492 follow-up).
      echo "✗ --sandboxed-only requested but the sandbox prerequisites are missing: ${_sb_missing[*]}" >&2
      echo "  The sandbox packs need pre-built Docker images that aren't auto-pulled. Build them once" >&2
      echo "  from a benchlocal-cli CHECKOUT (the build tooling isn't in the pip package):" >&2
      echo "    git clone https://github.com/noonghunna/benchlocal-cli" >&2
      echo "    bash benchlocal-cli/tools/build-sandboxes.sh        # ~30 GB free; prune if tight" >&2
      echo "  then re-run. For a no-Docker run instead:  bash scripts/quality-test.sh --medium" >&2
      exit 1
    fi
    echo "[quality-test] ⚠  sandbox packs (BugFind / CLI / Hermes) will be SKIPPED — not available: ${_sb_missing[*]}" >&2
    echo "               They need pre-built Docker images that aren't auto-pulled. Build them once from a" >&2
    echo "               benchlocal-cli CHECKOUT (the build tooling isn't in the pip package):" >&2
    echo "                 git clone https://github.com/noonghunna/benchlocal-cli" >&2
    echo "                 bash benchlocal-cli/tools/build-sandboxes.sh        # ~30 GB free; prune if tight" >&2
    echo "               then re-run --full. For a clean no-Docker run now:  bash scripts/quality-test.sh --medium" >&2
    echo "               (Continuing with the deterministic packs.)" >&2
    echo >&2
  fi
  if [[ ${#_sb_stale[@]} -gt 0 ]]; then
    echo "[quality-test] ⚠  sandbox image(s) OLDER than your installed benchlocal-cli:" >&2
    for _s in "${_sb_stale[@]}"; do echo "                 - ${_s}" >&2; done
    echo "               If that benchlocal-cli update changed sandbox sources (verifiers/harness)," >&2
    echo "               your scores will reflect the OLD sandbox behavior. Rebuild to be safe:" >&2
    echo "                 bash <benchlocal-cli-checkout>/tools/build-sandboxes.sh" >&2
    echo "               (Heuristic — an unrelated CLI reinstall also trips this. Continuing.)" >&2
    echo >&2
  fi
fi

# ---- run benchlocal-cli ------------------------------------------------------

RESULTS_DIR="${ROOT_DIR}/results/quality"
mkdir -p "$RESULTS_DIR"
TS=$(date +%Y-%m-%dT%H-%M-%S)
JSON_OUT="${RESULTS_DIR}/quality-${TS}.json"
# #252: --save-json overrides the default per-run path (e.g. write to a baseline file).
if [[ -n "$SAVE_JSON_OVERRIDE" ]]; then
  JSON_OUT="$SAVE_JSON_OVERRIDE"
  mkdir -p "$(dirname "$JSON_OUT")"
fi

if [[ "$TIMEOUT_PER_CASE_SET" == "1" ]]; then
  TIMEOUT_DISPLAY="${TIMEOUT_PER_CASE}s"
else
  TIMEOUT_DISPLAY="pack-default (60s deterministic / 300s cli-40+hermes / 1800s aider)"
fi
if [[ -n "$RESUME" ]]; then
  echo "[quality-test] RESUME ${RESUME}  endpoint=${URL}  model=${MODEL} (config restored from the saved run)"
elif [[ ${#SCENARIOS[@]} -gt 0 || -n "$SCENARIOS_FILE" ]]; then
  _sel_n=${#SCENARIOS[@]}
  if [[ -n "$SCENARIOS_FILE" ]]; then
    _sel_n=$(( _sel_n + $(command grep -cvE '^[[:space:]]*(#|$)' "$SCENARIOS_FILE" 2>/dev/null || echo 0) ))
  fi
  echo "[quality-test] SELECTION (${_sel_n} scenarios${SCENARIOS_FILE:+, file=$SCENARIOS_FILE})${PACK:+ ∩ pack=$PACK}  endpoint=${URL}  model=${MODEL}  timeout=${TIMEOUT_DISPLAY}"
  echo "[quality-test] ⚠ partial-selection result — never a canonical pack total (needs --allow-partial for history/rescore)"
elif [[ -n "$PACK" ]]; then
  echo "[quality-test] pack=${PACK}  endpoint=${URL}  model=${MODEL}  timeout=${TIMEOUT_DISPLAY}"
else
  echo "[quality-test] mode=${MODE}  endpoint=${URL}  model=${MODEL}  timeout=${TIMEOUT_DISPLAY}"
fi
echo "[quality-test] results JSON → ${JSON_OUT}"
echo

# Build CLI args
CLI_ARGS=(
  run
  --endpoint "${URL}"
  --model "${MODEL}"
  --output markdown
  --save-json "${JSON_OUT}"
)
if [[ "$TIMEOUT_PER_CASE_SET" == "1" ]]; then
  CLI_ARGS+=(--timeout-per-case "${TIMEOUT_PER_CASE}")
fi
if [[ -n "$REPEAT" ]]; then
  CLI_ARGS+=(--repeat "$REPEAT")
fi
if [[ -n "$PREVIOUS_RESULT" ]]; then
  CLI_ARGS+=(--previous-result "$PREVIOUS_RESULT")
fi
if [[ "$PROGRESS" == "1" ]]; then
  CLI_ARGS+=(--progress)
fi
if [[ -n "$RESUME" ]]; then
  # resume restores pack-set/selection/thinking/sampling/timeout from the journal
  CLI_ARGS+=(--resume "$RESUME")
elif [[ "$SANDBOXED_ONLY" == "1" ]]; then
  CLI_ARGS+=(--sandboxed-only)
elif [[ -n "$PACK" ]]; then
  CLI_ARGS+=(--pack "$PACK")
elif [[ ${#SCENARIOS[@]} -gt 0 || -n "$SCENARIOS_FILE" ]]; then
  : # bare selection: benchlocal derives the pack set from it (mode "custom")
else
  CLI_ARGS+=("$MODE")
fi
if [[ -z "$RESUME" ]]; then
  for _sc in ${SCENARIOS[@]+"${SCENARIOS[@]}"}; do
    CLI_ARGS+=(--scenario "$_sc")
  done
  if [[ -n "$SCENARIOS_FILE" ]]; then
    CLI_ARGS+=(--scenarios-file "$SCENARIOS_FILE")
  fi
fi
if [[ "$INCREMENTAL" == "1" ]]; then
  CLI_ARGS+=(--incremental)
fi
if [[ "$ALLOW_PARTIAL" == "1" ]]; then
  CLI_ARGS+=(--allow-partial)
fi
if [[ "$NO_SANDBOX" == "1" && "$SANDBOXED_ONLY" != "1" ]]; then
  CLI_ARGS+=(--no-sandboxed-packs)
fi
# Capture each sandboxed pack's container log before teardown (else it's lost).
if [[ -n "$SANDBOX_LOG_DIR" ]]; then
  mkdir -p "$SANDBOX_LOG_DIR"
  CLI_ARGS+=(--sandbox-log-dir "$SANDBOX_LOG_DIR")
  echo "[quality-test] sandbox logs → ${SANDBOX_LOG_DIR}/sandbox-<pack>.log"
fi
if [[ "$SAMPLING_FROM_SERVER" == "1" ]]; then
  CLI_ARGS+=(--sampling-from-server)
  echo "[quality-test] sampling: inherited from server (non-canonical)"
fi
if [[ "$ENABLE_THINKING" == "1" ]]; then
  CLI_ARGS+=(--enable-thinking)
  echo "[quality-test] thinking: enabled for every pack (non-canonical)"
fi
if [[ "$NO_THINKING" == "1" ]]; then
  CLI_ARGS+=(--no-thinking)
  echo "[quality-test] thinking: disabled for every pack, ignoring per-pack defaults (non-canonical)"
fi
if [[ -n "$THINKING_MAX_TOKENS" ]]; then
  CLI_ARGS+=(--thinking-max-tokens "$THINKING_MAX_TOKENS")
  echo "[quality-test] thinking max tokens: $THINKING_MAX_TOKENS (applies to thinking-enabled packs)"
fi
if [[ -n "$MAX_TOKENS" ]]; then
  CLI_ARGS+=(--max-tokens "$MAX_TOKENS")
  echo "[quality-test] max tokens: $MAX_TOKENS (overrides the per-pack completion budget for both arms)"
fi

# Run; capture exit code so we can also try to emit the compact one-liner
benchlocal-cli "${CLI_ARGS[@]}" || RC=$?
RC="${RC:-0}"

# ---- emit compact one-liner suitable for compose Quality: schema field -------

if [[ -f "$JSON_OUT" ]]; then
  echo
  echo "=========================================================================="
  echo "Quality: line for compose schema field (paste into compose YAML header):"
  echo "=========================================================================="

  python3 - "$JSON_OUT" "${PACK:-$MODE}" <<'PYEOF'
import json, sys, datetime
path, mode = sys.argv[1], sys.argv[2]
with open(path) as f:
    d = json.load(f)

date = datetime.date.today().isoformat()
mode_short = mode.lstrip("-")
parts = []
for p in d.get("packs", []):
    if p.get("status") == "stubbed" and p.get("total", 0) == 0:
        continue
    pid = p["pack_id"]
    pa = p["passed"]
    pt = p["total"]
    pct = round(100 * p["score"]) if pt else 0
    parts.append(f"{pid} {pa}/{pt} ({pct}%)")
suffix = f" (--{mode_short}, {date})"
if parts:
    print("Quality:   " + " · ".join(parts) + suffix)
else:
    print("Quality:   (no scoreable packs ran)")
PYEOF
fi

# ---- pointer: where to read failure reasons --------------------------------
if [[ -f "$JSON_OUT" ]]; then
  echo
  echo "Failure reasons: see the 'Failure breakdown:' above (failure_mode + detail per failed scenario)."
  echo "Dig deeper — full trace / older run / filter / diff:"
  echo "  benchlocal-cli inspect ${JSON_OUT} --failed                 # all failures + reason"
  echo "  benchlocal-cli inspect ${JSON_OUT} --scenario <ID> --full   # full prompt/response/verifier trace"
  echo "  benchlocal-cli inspect ${JSON_OUT} --mode timeout           # filter by failure type"
fi

echo
exit "$RC"
