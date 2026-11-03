#!/usr/bin/env bash
# compaction-probe.sh — does throughput degrade AFTER a context compaction at depth?
#
# The reported shape (#1052, @xtj7): "generating around 140k tokens and when I
# compact I suddenly see the live TPS drop to around 25 and stay [there]".
#
# Nothing we ship measured that. `bench-agentic.sh` grows a conversation
# MONOTONICALLY and never compacts; `gdn-mtp-apc-repro.py` likewise. Compaction is
# a distinct event: a long history is replaced by a short summary, which
# invalidates most of the prefix cache and forces a fresh prefill against
# already-warm GDN/SSM state — and it is a PEAK-MEMORY moment, because the new
# sequence prefills while the old blocks are still held.
#
# Shape:
#   1 GROW     inflate context to FILL_FRAC x max_ctx
#   2 BEFORE   N short generations at depth  -> baseline
#   3 COMPACT  history -> summary + last turn
#   4 AFTER    N short generations           -> comparison
#
# ⚠️ FILL_FRAC defaults to 0.90, NOT 0.95. verify-stress measured this class of
# model fillable to 91% of n_ctx with only 669 MB free, under its own 1024 MB
# margin threshold. Compaction needs headroom for the new prefill on top of that,
# so a 0.95 default would measure an OOM rather than the thing being asked about.
# Raise it deliberately (FILL_FRAC=0.95) to probe the edge; the preflight warns.
#
# ⚠️ TPS comes from vLLM's OWN counters as well as client wall time. Client-side
# timing on a reasoning model is easy to get wrong — that is exactly what #1096
# turned out to be (reasoning deltas charged to prefill) — so the engine figure is
# the citable one. Requests are greedy + seeded so reps are comparable.
#
# Env:
#   URL         endpoint. Default: the running engine container's port, found as bench.sh finds it
#               (preflight's autodetect; PREFLIGHT_NO_AUTODETECT=1 turns it off), else the curated
#               default port
#   MODEL       served model id. Default: auto-detected from /v1/models
#   FILL_FRAC   fraction of max_ctx to fill before compacting. Default: 0.90
#   REPS        generations per phase. Default: 5
#   MAX_CTX     override the model's advertised context
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
[[ -f scripts/lib/registry.sh ]] && source scripts/lib/registry.sh 2>/dev/null || true
_DEFAULT_PORT=""
if declare -F registry_lookup_default_port >/dev/null 2>&1; then
  _DEFAULT_PORT="$(registry_lookup_default_port qwen3.6-27b 2>/dev/null || true)"
fi
# The server that is running, as bench.sh finds it; the default port only when nothing is. This
# used to be the default port whatever was up.
if [[ -z "${URL:-}" ]]; then
  # shellcheck source=preflight.sh
  source scripts/preflight.sh
  preflight_autodetect_endpoint || true
fi
URL="${URL:-http://localhost:${_DEFAULT_PORT:-8020}}"
# The engine's API key (Strata 0.2+): the URL is the only input — club_container_for_url
# answers which container publishes it, engine-auth.sh which key (no per-script rules).
# shellcheck source=lib/engine-auth.sh
source scripts/lib/engine-auth.sh
club_engine_auth_enable_for_url "$URL"
FILL_FRAC="${FILL_FRAC:-0.90}"
REPS="${REPS:-5}"

# served-model.sh: TabbyAPI lists every model folder in /v1/models, so the first entry can be the wrong one (#1360)
# shellcheck source=lib/served-model.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/served-model.sh"
MODEL="${MODEL:-$(club_served_model_id "$URL")}"
if [[ -z "$MODEL" ]]; then
  echo "[compaction] no model at ${URL} — is the engine up? (set MODEL= to override)" >&2
  exit 2
fi

echo "[compaction] url=${URL} model=${MODEL} fill_frac=${FILL_FRAC} reps=${REPS}"
exec python3 scripts/lib/compaction_probe.py \
  --url "$URL" --model "$MODEL" --fill-frac "$FILL_FRAC" --reps "$REPS" \
  ${MAX_CTX:+--max-ctx "$MAX_CTX"}
