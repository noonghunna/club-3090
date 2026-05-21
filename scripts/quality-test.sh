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

ENV VARS
  URL              Endpoint base URL (default: auto-detected via preflight,
                   falls back to http://localhost:8020)
  MODEL            Served model name. If set (env or --model), it's respected
                   verbatim — no /v1/models override. If UNSET, auto-detected
                   from /v1/models (fixes the wrong-name → HTTP 404 footgun on
                   single-model composes). --model and MODEL are equivalent.
  TIMEOUT_PER_CASE Per-scenario HTTP timeout in seconds (default: 60).
                   --timeout-per-case overrides this when both are set.

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
    --timeout-per-case)
      TIMEOUT_PER_CASE="${2:-}"
      if [[ -z "$TIMEOUT_PER_CASE" ]] || ! [[ "$TIMEOUT_PER_CASE" =~ ^[0-9]+$ ]]; then
        echo "✗ --timeout-per-case requires a positive integer (seconds)" >&2
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
