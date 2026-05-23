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

  --pack PACK_ID   Run a single pack (overrides mode flag).
                   Available IDs:
                     toolcall-15  instructfollow-15  structoutput-15
                     dataextract-15  reasonmath-15
                     bugfind-15  cli-40  hermesagent-20  (require Docker)

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
                   (seconds). Default: 60. For aider-polyglot-30 on low-power
                   single-card rigs, bump to 3600+ to avoid mid-batch kills.
  --sandbox-log-dir DIR
                   Capture each sandboxed pack's container log to
                   DIR/sandbox-<pack_id>.log before teardown (forwarded to
                   benchlocal-cli). Without it, sandbox logs are lost on
                   container cleanup. Also settable via SANDBOX_LOG_DIR env.
  --sampling-from-server
                   Inherit sampling from the serving config instead of using
                   the pack's default temp=0. Omits sampling params from
                   requests so the server applies its own defaults (llama.cpp
                   --temp, vLLM --override-generation-config). Reads back via
                   GET /props and records the values. Tags the run as
                   non-canonical. Also settable via SAMPLING_FROM_SERVER=1 env.
  --enable-thinking
                   Forward to benchlocal-cli --enable-thinking so reasoning
                   models are evaluated with request-level thinking enabled.
                   Also settable via ENABLE_THINKING=1 env.
  --thinking-max-tokens N
                   Forward to benchlocal-cli --thinking-max-tokens N when
                   --enable-thinking / ENABLE_THINKING=1 is active. Also
                   settable via THINKING_MAX_TOKENS env.

ENV VARS
  URL              Endpoint base URL (default: auto-detected via preflight,
                   falls back to http://localhost:8020)
  MODEL            Served model name. If set (env or --model), it's respected
                   verbatim — no /v1/models override. If UNSET, auto-detected
                   from /v1/models (fixes the wrong-name → HTTP 404 footgun on
                   single-model composes). --model and MODEL are equivalent.
  TIMEOUT_PER_CASE Per-scenario HTTP timeout in seconds (default: 60).
                   --timeout-per-case overrides this when both are set.
  ENABLE_THINKING Set to 1 to send request-level enable_thinking=true via
                   benchlocal-cli --enable-thinking. Default: 0.
  THINKING_MAX_TOKENS
                   Optional thinking budget passed through to benchlocal-cli
                   --thinking-max-tokens when thinking is enabled.

EXAMPLES
  bash scripts/quality-test.sh                          # --medium against running compose
  bash scripts/quality-test.sh --quick                  # quicker, 2 packs only
  bash scripts/quality-test.sh --full                   # everything, needs Docker
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
MODEL="${MODEL:-qwen3.6-27b-autoround}"
TIMEOUT_PER_CASE="${TIMEOUT_PER_CASE:-60}"

# ---- arg parsing -------------------------------------------------------------

MODE="--medium"   # default
PACK=""
NO_SANDBOX=0
SANDBOXED_ONLY=0
LIST_PACKS=0
SANDBOX_LOG_DIR="${SANDBOX_LOG_DIR:-}"
SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick|--medium|--full)
      MODE="$1"
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
    --thinking-max-tokens)
      THINKING_MAX_TOKENS="${2:-}"
      if [[ -z "$THINKING_MAX_TOKENS" ]] || ! [[ "$THINKING_MAX_TOKENS" =~ ^[0-9]+$ ]]; then
        echo "✗ --thinking-max-tokens requires a positive integer" >&2
        exit 2
      fi
      shift 2
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

if [[ "$ENABLE_THINKING" != "1" ]] && server_reasoning_on; then
  echo "[quality-test] WARN: server appears to have reasoning enabled, but requests will send enable_thinking=false. Use --enable-thinking or ENABLE_THINKING=1 for reasoning-on evals." >&2
fi

# ---- run benchlocal-cli ------------------------------------------------------

RESULTS_DIR="${ROOT_DIR}/results/quality"
mkdir -p "$RESULTS_DIR"
TS=$(date +%Y-%m-%dT%H-%M-%S)
JSON_OUT="${RESULTS_DIR}/quality-${TS}.json"

if [[ -n "$PACK" ]]; then
  echo "[quality-test] pack=${PACK}  endpoint=${URL}  model=${MODEL}  timeout=${TIMEOUT_PER_CASE}s"
else
  echo "[quality-test] mode=${MODE}  endpoint=${URL}  model=${MODEL}  timeout=${TIMEOUT_PER_CASE}s"
fi
echo "[quality-test] results JSON → ${JSON_OUT}"
echo

# Build CLI args
CLI_ARGS=(
  run
  --endpoint "${URL}"
  --model "${MODEL}"
  --timeout-per-case "${TIMEOUT_PER_CASE}"
  --output markdown
  --save-json "${JSON_OUT}"
)
if [[ "$SANDBOXED_ONLY" == "1" ]]; then
  CLI_ARGS+=(--sandboxed-only)
elif [[ -n "$PACK" ]]; then
  CLI_ARGS+=(--pack "$PACK")
else
  CLI_ARGS+=("$MODE")
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
  echo "[quality-test] thinking: enabled (non-canonical)"
  if [[ -n "$THINKING_MAX_TOKENS" ]]; then
    CLI_ARGS+=(--thinking-max-tokens "$THINKING_MAX_TOKENS")
    echo "[quality-test] thinking max tokens: $THINKING_MAX_TOKENS"
  fi
elif [[ -n "$THINKING_MAX_TOKENS" ]]; then
  echo "[quality-test] NOTE: THINKING_MAX_TOKENS is set but thinking is off; ignoring it. Set ENABLE_THINKING=1 or --enable-thinking." >&2
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

echo
exit "$RC"
