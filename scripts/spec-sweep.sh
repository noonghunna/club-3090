#!/usr/bin/env bash
# spec-sweep.sh — n-sweep for speculative-decoding drafters: measure decode TPS
# + draft acceptance across num_speculative_tokens values and report the curve
# + sweet spot. The tuning sibling of concurrency-probe.sh (streams knob) and
# power-cap-sweep.sh (watts knob) — this sweeps the DRAFT-DEPTH knob.
#
# Why a script: this sweep has been hand-rolled repeatedly (Deckard MTP n=2
# sweet spot, Tess MTP n-sweep 2026-07-09, EAGLE3 n-sweep 2026-07-11, the
# gemma dual-awq n=4-vs-8 and DFlash n=5/7/8 rows) — same shape every time:
# per-n arms, warm + measured gens, TPS + acceptance, pick the knee. Draft
# depth is the one spec-dec knob every drafter shares, and the sweet spot is
# model+quant+hardware-specific (per-position acceptance falls off a cliff on
# weakly-aligned heads — n past the cliff SLOWS decode).
#
# TWO ENGINE PATHS:
#   llama.cpp family — FAST PATH, no reboots. llama-server accepts per-request
#     `"speculative": {"n_max": N}` and reports `timings.draft_n` /
#     `timings.draft_n_accepted` / `timings.predicted_per_second` in the
#     response, so the whole curve runs against ONE live server (~2 min).
#     A capability probe guards this: if the server ignores the field
#     (timings.draft_n missing or identical across differing n), the sweep
#     REFUSES rather than emit a fake flat curve — reboot-per-n via
#     `MTP_DRAFT_N_MAX=<n> bash scripts/switch.sh <slug>` is the fallback.
#   vLLM — reboot per n (vLLM has no per-request draft-depth knob):
#     `SPEC_N_MAX=<n> switch.sh <slug>` per arm; `SPEC=off` for the n=0
#     baseline arm. Slower (~5-8 min/arm, boot-dominated).
#
# Usage:
#   SLUG=llamacpp/tess-dual-mtp SWEEP_N="0 1 2 3 4" bash scripts/spec-sweep.sh
#   URL=http://localhost:8020 SWEEP_N="0 1 2 4" bash scripts/spec-sweep.sh   # llama.cpp fast path vs a running server (no slug needed)
#   SLUG=vllm/dual SWEEP_N="0 2 3 4" bash scripts/spec-sweep.sh              # vLLM: reboots per arm
#
# Env:
#   SWEEP_N     (required) space-separated draft depths; 0 = spec-OFF baseline
#   SLUG        registry slug — required for vLLM (reboots) and for the
#               llama.cpp reboot fallback; optional when URL points at a
#               running llama.cpp server (fast path only)
#   URL         endpoint (default: derived from the slug's default_port,
#               else http://localhost:8020)
#   MODEL       served model name (default: first id from /v1/models)
#   GENS        measured gens per arm (default 3)  ·  GEN_TOKENS (default 800)
#   TEMP        sampling temp for measured gens (default 0.6)
#   SWEEP_DRY   1 = print the plan, boot/measure nothing
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_N="${SWEEP_N:-}"
SLUG="${SLUG:-}"
URL="${URL:-}"
MODEL="${MODEL:-}"
GENS="${GENS:-3}"
GEN_TOKENS="${GEN_TOKENS:-800}"
TEMP="${TEMP:-0.6}"
SWEEP_DRY="${SWEEP_DRY:-0}"
PROMPT='Write a detailed 800-word essay on the history and impact of the printing press.'

[[ -n "$SWEEP_N" ]] || { echo "SWEEP_N is required (e.g. SWEEP_N=\"0 1 2 4\") — 0 = spec-OFF baseline arm." >&2; exit 2; }

# --- engine resolution (registry when SLUG given; else llama.cpp fast path) --
ENGINE_FAMILY="llamacpp"
if [[ -n "$SLUG" ]]; then
  ENGINE_FAMILY="$(python3 - "$SLUG" <<'PY'
import sys
sys.path.insert(0, "scripts/lib/profiles")
from compose_registry import COMPOSE_REGISTRY
e = COMPOSE_REGISTRY.get(sys.argv[1])
if e is None:
    print("unknown"); raise SystemExit
eng = e["engine"]
print("vllm" if eng.startswith("vllm") else "llamacpp")
PY
)"
  [[ "$ENGINE_FAMILY" != "unknown" ]] || { echo "unknown SLUG '$SLUG' (not in compose_registry)" >&2; exit 2; }
  if [[ -z "$URL" ]]; then
    PORT="$(python3 - "$SLUG" <<'PY'
import sys
sys.path.insert(0, "scripts/lib/profiles")
from compose_registry import COMPOSE_REGISTRY
print(COMPOSE_REGISTRY[sys.argv[1]]["default_port"])
PY
)"
    URL="http://localhost:${PORT}"
  fi
elif [[ -z "$URL" ]]; then
  URL="http://localhost:8020"
fi
[[ "$ENGINE_FAMILY" == "vllm" && -z "$SLUG" ]] && { echo "vLLM sweeps need SLUG (reboot per arm)." >&2; exit 2; }

echo "[spec-sweep] engine=$ENGINE_FAMILY url=$URL n in { $SWEEP_N } gens=$GENS x ${GEN_TOKENS}tok temp=$TEMP"

if [[ "$SWEEP_DRY" == "1" ]]; then
  for n in $SWEEP_N; do
    if [[ "$ENGINE_FAMILY" == "vllm" ]]; then
      echo "[sweep:dry] would: $( [[ "$n" == "0" ]] && echo "SPEC=off" || echo "SPEC_N_MAX=$n" ) switch.sh $SLUG -> measure n=$n"
    else
      echo "[sweep:dry] would: per-request speculative.n_max=$n against $URL (no reboot)"
    fi
  done
  echo "[spec-sweep] sweet spot: (dry run)"
  exit 0
fi

# --- shared measurement -------------------------------------------------------
_detect_model() {
  [[ -n "$MODEL" ]] && return 0
  MODEL="$(curl -s -m 5 "$URL/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
  [[ -n "$MODEL" ]] || { echo "no server at $URL (and MODEL unset)" >&2; exit 1; }
}

# gen <n_or_empty> <max_tokens> <temp> -> raw JSON response.
# llama.cpp: embeds per-request speculative.n_max (n=0 disables drafting).
_gen() {
  local n="$1" mt="$2" tp="$3" spec=""
  if [[ "$ENGINE_FAMILY" == "llamacpp" && -n "$n" ]]; then
    spec=",\"speculative\":{\"n_max\":$n}"
  fi
  curl -s -m 300 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$mt,\"temperature\":$tp$spec}"
}

# _timings <json> -> "tps draft_n draft_acc" (llama.cpp timings block; vLLM lacks it -> client wall handled by caller)
_timings() {
  python3 -c '
import json, sys
d = json.loads(sys.stdin.read() or "{}")
t = d.get("timings") or {}
print(round(t.get("predicted_per_second") or 0, 2),
      t.get("draft_n") if t.get("draft_n") is not None else -1,
      t.get("draft_n_accepted") if t.get("draft_n_accepted") is not None else -1)'
}

RESULTS=()  # "n|tps|accept"

_measure_llamacpp_arm() {
  local n="$1" tps_list=() dn_tot=0 da_tot=0
  _gen "$n" 300 "$TEMP" >/dev/null  # warm
  for _ in $(seq 1 "$GENS"); do
    read -r tps dn da < <(_gen "$n" "$GEN_TOKENS" "$TEMP" | _timings)
    tps_list+=("$tps")
    [[ "$dn" != "-1" ]] && dn_tot=$((dn_tot + dn)) && da_tot=$((da_tot + da))
  done
  local med acc="—"
  med="$(printf '%s\n' "${tps_list[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
  if [[ "$n" != "0" && $dn_tot -gt 0 ]]; then
    acc="$(awk -v a="$da_tot" -v d="$dn_tot" 'BEGIN{printf "%.2f", a/d}')"
  fi
  RESULTS+=("$n|$med|$acc")
  printf "  n=%-3s decode=%-8s accept=%s  (tps: %s)\n" "$n" "$med" "$acc" "${tps_list[*]}"
}

_measure_vllm_arm() {
  local n="$1" tps_list=()
  _gen "" 300 "$TEMP" >/dev/null  # warm
  for _ in $(seq 1 "$GENS"); do
    local t0 toks dt
    t0=$(date +%s.%N)
    toks="$(_gen "" "$GEN_TOKENS" "$TEMP" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("usage",{}).get("completion_tokens",0))' 2>/dev/null || echo 0)"
    dt="$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{print a-b}')"
    tps_list+=("$(awk -v t="$toks" -v d="$dt" 'BEGIN{printf "%.2f", (d>0)?t/d:0}')")
  done
  local med acc="—"
  med="$(printf '%s\n' "${tps_list[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
  if [[ "$n" != "0" ]]; then
    local cn
    cn="$(docker ps --format '{{.Names}}' | grep -m1 -E 'vllm' || true)"
    [[ -n "$cn" ]] && acc="$(docker logs "$cn" 2>&1 | grep 'SpecDecoding metrics' | tail -1 | grep -oE 'acceptance rate: [0-9.]+%' | tail -1 | grep -oE '[0-9.]+' || echo '—')"
    [[ "$acc" != "—" && -n "$acc" ]] && acc="$(awk -v p="$acc" 'BEGIN{printf "%.2f", p/100}')"
  fi
  RESULTS+=("$n|$med|$acc")
  printf "  n=%-3s decode=%-8s accept=%s  (wall tps: %s)\n" "$n" "$med" "${acc:-—}" "${tps_list[*]}"
}

# --- llama.cpp fast path: capability probe then per-request arms --------------
if [[ "$ENGINE_FAMILY" == "llamacpp" ]]; then
  curl -sf -m 5 "$URL/v1/models" >/dev/null || {
    if [[ -n "$SLUG" ]]; then
      echo "[spec-sweep] no server at $URL — booting $SLUG once…"
      bash scripts/switch.sh "$SLUG" >/dev/null
      for _ in $(seq 1 90); do curl -sf -m 3 "$URL/v1/models" >/dev/null 2>&1 && break; sleep 2; done
      curl -sf -m 5 "$URL/v1/models" >/dev/null || { echo "boot failed" >&2; exit 1; }
    else
      echo "no server at $URL and no SLUG to boot" >&2; exit 1
    fi
  }
  _detect_model
  # capability probe: draft_n must be REPORTED and must DIFFER between n=1 and
  # a larger n — otherwise the server is ignoring per-request speculative and a
  # "sweep" would be a flat lie. (Requires the server to have a drafter loaded:
  # a spec-less boot always reports draft_n=-1 -> refuse with a hint.)
  read -r _ p1 _ < <(_gen 1 64 0 | _timings)
  read -r _ p4 _ < <(_gen 4 64 0 | _timings)
  if [[ "$p1" == "-1" || "$p4" == "-1" ]]; then
    echo "[spec-sweep] REFUSING: server does not report timings.draft_n — either no drafter is loaded on this serve, or this build predates per-request speculative. Fallback: reboot per arm (MTP_DRAFT_N_MAX=<n> bash scripts/switch.sh <slug>)." >&2
    exit 3
  fi
  echo "[spec-sweep] per-request speculative honored (probe draft_n: n1=$p1 n4=$p4) — fast path, no reboots"
  for n in $SWEEP_N; do _measure_llamacpp_arm "$n"; done
else
  # --- vLLM: reboot per arm ----------------------------------------------------
  for n in $SWEEP_N; do
    if [[ "$n" == "0" ]]; then
      echo "[spec-sweep] boot $SLUG SPEC=off…"
      SPEC=off bash scripts/switch.sh "$SLUG" >/dev/null
    else
      echo "[spec-sweep] boot $SLUG SPEC_N_MAX=$n…"
      SPEC_N_MAX="$n" bash scripts/switch.sh "$SLUG" >/dev/null
    fi
    ready=0
    for _ in $(seq 1 240); do curl -sf -m 3 "$URL/v1/models" >/dev/null 2>&1 && { ready=1; break; }; sleep 3; done
    [[ "$ready" == "1" ]] || { echo "  n=$n: boot not ready — skipping"; continue; }
    _detect_model
    _measure_vllm_arm "$n"
  done
fi

# --- summary + sweet spot ------------------------------------------------------
echo ""
echo "=== spec-sweep summary (model=$MODEL) ==="
printf "  %-5s %-10s %s\n" "n" "decode" "accept"
best_n=""; best_tps=0
for r in "${RESULTS[@]}"; do
  IFS='|' read -r n tps acc <<<"$r"
  printf "  %-5s %-10s %s\n" "$n" "$tps" "$acc"
  awk -v t="$tps" -v b="$best_tps" 'BEGIN{exit !(t>b)}' && { best_tps="$tps"; best_n="$n"; }
done
echo "  sweet spot: n=$best_n ($best_tps tok/s)$( [[ "$best_n" == "0" ]] && echo ' — the DRAFTER IS NET-NEGATIVE on this serve (spec-off wins)' )"
# machine-readable
for r in "${RESULTS[@]}"; do IFS='|' read -r n tps acc <<<"$r"; echo "RESULT n=$n tps=$tps accept=$acc"; done
