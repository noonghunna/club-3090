#!/usr/bin/env bash
# concurrency-probe.sh — the #246 Phase 2 measurement: how many concurrent
# streams a card's KV pool sustains cleanly, and at what per-stream throughput.
#
# soak-test.sh is SINGLE-STREAM by design (no concurrency stress) and only ramps
# to ~22-25K accumulated tokens; this is its concurrent, model-max sibling. It
# is the measurement behind an envelopes.yml `max_num_seqs` row — the tool that
# upgrades a `computed` row to `validated` (see
# /opt/ai/docs/phase2-soak-validation-protocol.md).
#
# TWO ROW CLASSES, TWO MODES (this is the whole point):
#   • pool-ceiling rows (e.g. 5090): the value IS the kv-calc ceiling, so this is
#     a FIT + STABILITY test — boot at N, drive N streams at the row's target_ctx,
#     require all-complete / 0 silent / 0-growth VRAM / >=98% TPS retention.
#     Use:  VALIDATE=1  (fills the served max-model-len; gates fit + retention)
#   • bandwidth-cap rows (PRO 6000 / Spark): the value sits far BELOW the pool
#     ceiling, so a fit test proves nothing (N=8 trivially fits 96 GB). Sweep N
#     and find the THROUGHPUT KNEE — the largest N whose per-stream decode TPS
#     stays above the floor.
#     Use:  SWEEP="4 8 12 16" SLUG=vllm/minimal TPS_FLOOR=15
#
# Usage:
#   MAX_NUM_SEQS=4 bash scripts/switch.sh vllm/dual   # boot at the candidate
#   bash scripts/concurrency-probe.sh                 # plain fit check (server's N)
#   VALIDATE=1 bash scripts/concurrency-probe.sh      # validation-grade fit @ target_ctx
#   SWEEP="4 8 12 16" SLUG=vllm/minimal TPS_FLOOR=15 bash scripts/concurrency-probe.sh
#
# Env: URL (default http://localhost:8010) · MODEL (auto) · CONTAINER (auto for
#   VRAM) · CONCURRENCY (default: served max-num-seqs, else 2) · ROUNDS (5) ·
#   PROMPT_TOKENS (16000) · GEN_TOKENS (256) · VRAM_GROWTH_MB (200) ·
#   REQ_TIMEOUT (600).
#   Validation knobs: VALIDATE (0) · TARGET_CTX (auto from --max-model-len) ·
#   TPS_FLOOR (0 = report-only) · RETENTION_MIN (0.98) ·
#   SWEEP ("" = single-N) · SLUG (required for SWEEP) · SWEEP_DRY (0) ·
#   BOOT_TIMEOUT (360).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

URL="${URL:-http://localhost:8010}"
ROUNDS="${ROUNDS:-5}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16000}"
GEN_TOKENS="${GEN_TOKENS:-256}"
VRAM_GROWTH_MB="${VRAM_GROWTH_MB:-200}"
REQ_TIMEOUT="${REQ_TIMEOUT:-600}"
TPS_FLOOR="${TPS_FLOOR:-0}"           # 0 = report-only; >0 = gate per-stream decode TPS
RETENTION_MIN="${RETENTION_MIN:-0.98}"
VALIDATE="${VALIDATE:-0}"             # 1 = fill target_ctx + gate fit + retention
TARGET_CTX="${TARGET_CTX:-}"          # override; else auto from container --max-model-len
SWEEP="${SWEEP:-}"                    # e.g. "4 8 12 16" -> reboot+probe per N, find knee
SLUG="${SLUG:-}"                      # required for SWEEP (reboot target)
SWEEP_DRY="${SWEEP_DRY:-0}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-360}"

MODEL="${MODEL:-$(curl -s -m 5 "${URL}/v1/models" 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3.6-27b)}"
# best-effort container for VRAM + cmd introspection (name heuristic)
CONTAINER="${CONTAINER:-$(docker ps --format '{{.Names}}' 2>/dev/null | grep -m1 -E 'vllm-(qwen|gemma)' || true)}"

_container_cmd() { docker inspect "$CONTAINER" --format '{{join .Config.Cmd " "}}' 2>/dev/null || true; }
_served_seqs()   { _container_cmd | grep -oE 'max-num-seqs [0-9]+'  | grep -oE '[0-9]+' | head -1; }
_served_ctx()    { _container_cmd | grep -oE 'max-model-len [0-9]+' | grep -oE '[0-9]+' | head -1; }

# CONCURRENCY defaults to the served max-num-seqs (the thing we're validating).
if [[ -z "${CONCURRENCY:-}" ]]; then
  CONCURRENCY="$(_served_seqs || true)"; CONCURRENCY="${CONCURRENCY:-2}"
fi

# VALIDATE preset: fill each stream to the served target context (N full-context
# sessions is what the row's no-preemption ceiling claims), and run a touch
# longer so the retention signal is meaningful.
if [[ "$VALIDATE" == "1" && -z "$SWEEP" ]]; then
  ctx="${TARGET_CTX:-$(_served_ctx || true)}"; ctx="${ctx:-32768}"
  headroom=$(( GEN_TOKENS + 512 ))
  if (( ctx > headroom )); then PROMPT_TOKENS=$(( ctx - headroom )); else PROMPT_TOKENS="$ctx"; fi
  ROUNDS="${ROUNDS_OVERRIDE:-6}"
  echo "[concurrency-probe] VALIDATE: filling to target_ctx=${ctx} (prompt=${PROMPT_TOKENS}tok), rounds=${ROUNDS}"
fi

# --- the probe core, parameterised by N (one python invocation per call) ------
# Emits the per-round table, a human verdict, and a machine-readable RESULT line.
# Exit 0 iff PASS: fit-clean AND (per-stream TPS >= floor, if floor set) AND
# (retention >= RETENTION_MIN, if VALIDATE).
run_probe() {
  local N="$1"
  URL="$URL" MODEL="$MODEL" CONTAINER="$CONTAINER" CONCURRENCY="$N" \
  ROUNDS="$ROUNDS" PROMPT_TOKENS="$PROMPT_TOKENS" GEN_TOKENS="$GEN_TOKENS" \
  VRAM_GROWTH_MB="$VRAM_GROWTH_MB" REQ_TIMEOUT="$REQ_TIMEOUT" \
  TPS_FLOOR="$TPS_FLOOR" RETENTION_MIN="$RETENTION_MIN" VALIDATE="$VALIDATE" \
  python3 - <<'PY'
import concurrent.futures as cf
import json, os, statistics, subprocess, time, urllib.request

URL=os.environ["URL"]; MODEL=os.environ["MODEL"]; N=int(os.environ["CONCURRENCY"])
ROUNDS=int(os.environ["ROUNDS"]); PTOK=int(os.environ["PROMPT_TOKENS"])
GTOK=int(os.environ["GEN_TOKENS"]); GROWTH=int(os.environ["VRAM_GROWTH_MB"])
REQ_TIMEOUT=float(os.environ["REQ_TIMEOUT"]); TPS_FLOOR=float(os.environ["TPS_FLOOR"])
RETENTION_MIN=float(os.environ["RETENTION_MIN"]); VALIDATE=os.environ["VALIDATE"]=="1"
CONTAINER=os.environ.get("CONTAINER") or ""

BLOCK=("This section describes the history of computing in detail. Transistors "
       "were invented in 1947 at Bell Labs. The integrated circuit came a decade "
       "later. Microprocessors emerged in the 1970s and changed the world. ")
def prompt(stream, rnd):  # ~0.23 tok/char; unique salt per stream -> no prefix-cache free ride
    reps=int(PTOK/(len(BLOCK)*0.23))+1
    return f"[probe s{stream} r{rnd}] "+BLOCK*reps+"\nWrite a detailed multi-paragraph summary."

def vram_used_mb():
    try:
        out=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,timeout=10).stdout
        return sum(int(x) for x in out.split())
    except Exception:
        return -1

def one(stream, rnd):
    # Streamed so we can separate TTFT (prefill) from decode and report a real
    # per-stream DECODE tok/s — the signal the bandwidth-knee sweep needs, and
    # the only honest throughput number at deep context (where prefill dominates
    # wall time). completion_tokens comes from the include_usage final chunk.
    body=json.dumps({"model":MODEL,"max_tokens":GTOK,"temperature":0.0,"stream":True,
        "stream_options":{"include_usage":True},
        "messages":[{"role":"user","content":prompt(stream,rnd)}]}).encode()
    req=urllib.request.Request(URL+"/v1/chat/completions",data=body,
        headers={"Content-Type":"application/json"})
    t0=time.time(); t_first=None; t_last=None; toks=0; chunks=0
    try:
        resp=urllib.request.urlopen(req,timeout=REQ_TIMEOUT)
        for raw in resp:
            line=raw.decode("utf-8","ignore").strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            try: ch=json.loads(data)
            except Exception: continue
            u=ch.get("usage")
            if u and u.get("completion_tokens"): toks=u["completion_tokens"]
            ch_choices=ch.get("choices") or []
            if ch_choices:
                d=ch_choices[0].get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now=time.time()
                    if t_first is None: t_first=now
                    t_last=now; chunks+=1
        dt=time.time()-t0
        if not toks: toks=chunks  # fall back to chunk count if usage absent
        decode_dt=(t_last-t_first) if (t_first and t_last and t_last>t_first) else 0.0
        tps=(toks/decode_dt) if (toks>1 and decode_dt>0) else 0.0
        ttft=(t_first-t0) if t_first else None
        return {"ok":toks>0,"toks":toks,"silent":toks==0,"err":None,"dt":dt,"ttft":ttft,"tps":tps}
    except Exception as e:
        return {"ok":False,"toks":0,"silent":False,"err":str(e)[:80],"dt":time.time()-t0,"ttft":None,"tps":0.0}

print(f"\n{'round':>5} {'done':>7} {'silent':>7} {'errors':>7} {'vram_MB':>8} {'agg_t/s':>8} {'per-strm':>9}")
vram0=vram_used_mb()
vram_by_round=[]; mtps_by_round=[]; bad=0
for rnd in range(1,ROUNDS+1):
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        res=list(ex.map(lambda s: one(s,rnd), range(N)))
    wall=time.time()-t0
    done=sum(1 for r in res if r["ok"]); silent=sum(1 for r in res if r["silent"])
    errs=sum(1 for r in res if r["err"])
    v=vram_used_mb(); vram_by_round.append(v)
    agg=sum(r["toks"] for r in res)/wall if wall else 0
    tps_ok=[r["tps"] for r in res if r["ok"] and r["tps"]>0]
    mtps=statistics.median(tps_ok) if tps_ok else 0.0
    mtps_by_round.append(mtps)
    print(f"{rnd:>5} {done:>4}/{N:<2} {silent:>7} {errs:>7} {v:>8} {agg:>8.1f} {mtps:>9.1f}")
    if done<N or silent or errs: bad+=1

# VRAM: leak = post-warm growth (round 2 baseline), NOT the expected cold->warm fill.
warm_i=1 if ROUNDS>=3 else 0
warm=vram_by_round[warm_i]
pool_fill=warm-vram0 if vram0>=0 else -1
leak=vram_by_round[-1]-warm if vram0>=0 else -1
vram_peak=max(vram_by_round) if vram_by_round else -1
# per-stream decode TPS: steady-state = last round; retention = late vs early.
# Round 1 is cudagraph/cold warmup (systematically slow) — anchoring retention
# on it would flatter the ratio, so drop it once there are >=4 rounds.
report_tps=mtps_by_round[-1] if mtps_by_round else 0.0
warm_tps=mtps_by_round[1:] if ROUNDS>=4 else mtps_by_round
if len(warm_tps)>=3 and warm_tps[0]>0:
    early=statistics.median(warm_tps[:2]); late=statistics.median(warm_tps[-2:])
    retention=(late/early) if early>0 else 0.0
else:
    retention=1.0  # too few post-warmup rounds to judge

clean_fit=(bad==0) and (0<=leak<=GROWTH)
floor_ok=(report_tps>=TPS_FLOOR) if TPS_FLOOR>0 else True
retention_ok=(retention>=RETENTION_MIN) if ROUNDS>=3 else True
PASS=clean_fit and floor_ok and (retention_ok if VALIDATE else True)

print(f"\n=== verdict (N={N}) ===")
print(f"  VRAM: cold {vram0} -> warm {warm} MB (pool fill {pool_fill} MB, expected) "
      f"-> final {vram_by_round[-1]} MB (post-warm growth {leak} MB / {GROWTH})  peak {vram_peak} MB")
print(f"  per-stream decode: {report_tps:.1f} tok/s (steady) · retention {retention*100:.1f}% "
      f"(min {RETENTION_MIN*100:.0f}%)" + (f" · floor {TPS_FLOOR:.0f}" if TPS_FLOOR>0 else " · floor off"))
flags=[]
if not clean_fit: flags.append("fit")
if TPS_FLOOR>0 and not floor_ok: flags.append("tps-floor")
if VALIDATE and not retention_ok: flags.append("retention")
print(f"  concurrency {N} @ ~{PTOK} tok: {'PASS — sustained clean' if PASS else 'FAIL — '+','.join(flags)}")
if PASS:
    print(f"  envelope row: max_num_seqs: {N}  validated: {{ concurrency_soak: "
          f"'{N} @ ~{PTOK//1000}K, {report_tps:.0f} tok/s/stream, {leak} MB post-warm', "
          f"vram_peak_gb: {vram_peak/1024:.1f} }}")
# machine-readable line for SWEEP parsing
print(f"RESULT N={N} clean={int(clean_fit)} pass={int(PASS)} mps_tps={report_tps:.2f} "
      f"retention={retention:.3f} leak={leak} vram_peak={vram_peak} floor_ok={int(floor_ok)}")
raise SystemExit(0 if PASS else 1)
PY
}

# --- SWEEP: reboot per N, probe, find the throughput knee ----------------------
if [[ -n "$SWEEP" ]]; then
  if [[ -z "$SLUG" ]]; then
    echo "SWEEP needs SLUG=<compose slug> — vLLM can't hot-change max-num-seqs, so each N is a reboot." >&2
    exit 2
  fi
  echo "[sweep] slug=$SLUG N in { $SWEEP } · floor=${TPS_FLOOR} tok/s/stream · reboots the server per N"
  knee=""; knee_tps=""
  for N in $SWEEP; do
    if [[ "$SWEEP_DRY" == "1" ]]; then
      echo "[sweep:dry] would: MAX_NUM_SEQS=$N switch.sh $SLUG  ->  wait ready  ->  probe N=$N"
      continue
    fi
    echo "[sweep] boot $SLUG @ MAX_NUM_SEQS=$N ..."
    if ! MAX_NUM_SEQS="$N" bash "$ROOT_DIR/scripts/switch.sh" "$SLUG" >/dev/null 2>&1; then
      echo "[sweep] N=$N: boot FAILED — skipping"; continue
    fi
    ready=0
    for _ in $(seq 1 $(( BOOT_TIMEOUT / 2 )) ); do
      if curl -s -m 3 "${URL}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
      sleep 2
    done
    if [[ "$ready" != "1" ]]; then echo "[sweep] N=$N: not ready in ${BOOT_TIMEOUT}s — skipping"; continue; fi
    out="$(run_probe "$N" || true)"; echo "$out"
    line="$(printf '%s\n' "$out" | grep -m1 '^RESULT ' || true)"
    clean="$(sed -n 's/.* clean=\([0-9]*\).*/\1/p' <<<"$line")"
    mps="$(sed -n 's/.* mps_tps=\([0-9.]*\).*/\1/p' <<<"$line")"
    # knee = largest N that is fit-clean AND (floor off OR per-stream TPS >= floor)
    above="$(awk -v t="$mps" -v f="$TPS_FLOOR" 'BEGIN{print (f<=0 || t>=f)?1:0}')"
    if [[ "$clean" == "1" && "$above" == "1" ]]; then knee="$N"; knee_tps="$mps"; fi
  done
  echo ""
  echo "=== sweep knee ==="
  if [[ -n "$knee" ]]; then
    echo "  largest clean N at/above floor: N=$knee (${knee_tps} tok/s/stream)"
    echo "  -> validate the envelope row at max_num_seqs: $knee"
  else
    echo "  no N met the bar — lower the sweep range or the target_ctx, or check the floor."
  fi
  exit 0
fi

# --- single-N mode -------------------------------------------------------------
echo "[concurrency-probe] URL=$URL model=$MODEL N=$CONCURRENCY rounds=$ROUNDS prompt=${PROMPT_TOKENS}tok gen=${GEN_TOKENS}" \
     "$( [[ "$TPS_FLOOR" != "0" ]] && echo "floor=${TPS_FLOOR}" )"
run_probe "$CONCURRENCY"
