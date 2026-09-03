#!/usr/bin/env bash
# quality-baseline.sh (#252) — diff a fresh quality 8-pack run against the
# curated baseline for a (registry-slug, thinking-mode), or capture/refresh that
# baseline. Thin wrapper over quality-test.sh.
#
# Baselines live in results/baselines/<slug>__<mode>.json (committed — the
# trusted n>=3 aggregates). Canonical = no-thinking (temp-0 reproducible);
# non-canonical = enable-thinking. A baseline is an n>=3 aggregate (--repeat) so
# run-to-run noise (~+-5-7 / 150 on the 8-pack) isn't flagged as a regression.
#
# Usage:
#   # diff a fresh run vs the canonical (no-thinking) baseline for a slug
#   bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual
#   # ...vs the thinking-on baseline
#   bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual --mode enable-thinking
#   # capture / refresh a baseline (n=3 aggregate) — needs a live endpoint
#   bash scripts/quality-baseline.sh --slug vllm/qwen-35b-a3b-dual --capture
#   # print the resolved quality-test.sh command without running
#   bash scripts/quality-baseline.sh --slug X --dry-run
#
# Endpoint/model are inherited by quality-test.sh (auto-detect, or MODEL=/URL=).
# Any unrecognized args pass through to quality-test.sh (e.g.
# --sampling-from-server, --thinking-max-tokens, --model NAME).
#
# Flags:
#   --slug <registry-slug>   REQUIRED. e.g. vllm/qwen-35b-a3b-dual ('/' -> '-' in the filename)
#   --mode no-thinking|enable-thinking   default: no-thinking (canonical)
#   --capture                write/refresh the baseline instead of diffing against it
#   --repeat N               runs per scenario (default 3; applies to both capture and diff)
#   --dry-run                print the command, don't run
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="${ROOT_DIR}/results/baselines"

SLUG=""
MODE="no-thinking"
CAPTURE=0
DRY=0
REPEAT="${REPEAT:-3}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --slug)    SLUG="${2:-}"; shift 2 ;;
    --mode)    MODE="${2:-}"; shift 2 ;;
    --capture) CAPTURE=1; shift ;;
    --repeat)  REPEAT="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)         PASSTHROUGH+=("$1"); shift ;;
  esac
done

if [[ -z "$SLUG" ]]; then
  echo "✗ --slug <registry-slug> is required (e.g. vllm/qwen-35b-a3b-dual)" >&2; exit 2
fi
case "$MODE" in
  no-thinking)     MODE_FLAG="--no-thinking" ;;
  enable-thinking) MODE_FLAG="--enable-thinking" ;;
  *) echo "✗ --mode must be no-thinking | enable-thinking (got: '${MODE}')" >&2; exit 2 ;;
esac
if ! [[ "$REPEAT" =~ ^[0-9]+$ ]] || [[ "$REPEAT" -lt 1 ]]; then
  echo "✗ --repeat must be a positive integer (got: '${REPEAT}')" >&2; exit 2
fi

# registry slugs contain '/'; make the baseline filename safe.
SLUG_SAFE="${SLUG//\//-}"
BASELINE_FILE="${BASELINE_DIR}/${SLUG_SAFE}__${MODE}.json"
QT="${ROOT_DIR}/scripts/quality-test.sh"

CMD=(bash "$QT" --full "$MODE_FLAG" --repeat "$REPEAT")
if [[ "$CAPTURE" == "1" ]]; then
  CMD+=(--save-json "$BASELINE_FILE")
  echo "[quality-baseline] CAPTURE → ${BASELINE_FILE}  (n=${REPEAT}, mode=${MODE})"
else
  if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "✗ no baseline for slug='${SLUG}' mode='${MODE}':" >&2
    echo "    ${BASELINE_FILE}" >&2
    echo "  Capture one first:  bash scripts/quality-baseline.sh --slug '${SLUG}' --mode ${MODE} --capture" >&2
    exit 1
  fi
  CMD+=(--previous-result "$BASELINE_FILE")
  echo "[quality-baseline] DIFF vs ${BASELINE_FILE}  (mode=${MODE})"
fi
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && CMD+=("${PASSTHROUGH[@]}")

if [[ "$DRY" == "1" ]]; then
  printf '[quality-baseline] (dry-run) would run:\n  %s\n' "${CMD[*]}"
  exit 0
fi
exec "${CMD[@]}"
