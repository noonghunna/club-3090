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

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
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

# Argument validation BEFORE any environment probing: slot detection (#818) is
# also a FATAL exit 2, so probing first masks this message on rigs where no
# server answers — the test suite caught exactly that on a serverless box.
if [[ -n "$SWEEP" && -z "$SLUG" ]]; then
  echo "SWEEP needs SLUG=<compose slug> — vLLM can't hot-change max-num-seqs, so each N is a reboot." >&2
  exit 2
fi

# Remember whether the caller pinned MODEL — SWEEP re-resolves after each boot
# and must not clobber an explicit pin.
MODEL_PINNED="${MODEL:+1}"
MODEL="${MODEL:-$(curl -s -m 5 "${URL}/v1/models" 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3.6-27b)}"
# best-effort container for VRAM + cmd introspection (name heuristic)
CONTAINER="${CONTAINER:-$(docker ps --format '{{.Names}}' 2>/dev/null | grep -m1 -E 'vllm-(qwen|gemma)' || true)}"

_container_cmd() { docker inspect "$CONTAINER" --format '{{join .Config.Cmd " "}}' 2>/dev/null || true; }
_served_seqs()   { _container_cmd | grep -oE 'max-num-seqs [0-9]+'  | grep -oE '[0-9]+' | head -1; }
_served_np()     { _container_cmd | grep -oE '\-np +[0-9]+'         | grep -oE '[0-9]+' | head -1; }
_served_ctx()    { _container_cmd | grep -oE 'max-model-len [0-9]+' | grep -oE '[0-9]+' | head -1; }
# llama.cpp-family servers report the slot count as total_slots on /props.
_props_slots()   { curl -s -m 3 "${URL}/props" 2>/dev/null \
  | python3 -c 'import json,sys; v=json.load(sys.stdin).get("total_slots",""); print(v if isinstance(v,int) else "")' 2>/dev/null; }

# CONCURRENCY defaults to the served slot count (the thing we're validating).
# Detection order: vLLM container flag -> llama.cpp container flag -> /props.
# A failed detection is FATAL (#818): the old silent CONCURRENCY=2 fallback
# measured queue-wait as concurrency against 1-slot servers and mislabeled arms.
# SWEEP mode skips detection entirely: each arm passes CONCURRENCY=$N into
# run_probe after its own boot, and probing the PRE-sweep server (or a
# serverless box, in SWEEP_DRY) would FATAL on a value nothing consumes.
if [[ -n "$SWEEP" ]]; then
  : # per-arm CONCURRENCY comes from the sweep loop
elif [[ -z "${CONCURRENCY:-}" ]]; then
  _conc_src="container max-num-seqs"; CONCURRENCY="$(_served_seqs || true)"
  if [[ -z "$CONCURRENCY" ]]; then _conc_src="container -np";          CONCURRENCY="$(_served_np || true)"; fi
  if [[ -z "$CONCURRENCY" ]]; then _conc_src="server /props total_slots"; CONCURRENCY="$(_props_slots || true)"; fi
  if [[ -z "$CONCURRENCY" ]]; then
    echo "[concurrency-probe] FATAL: cannot detect the served slot count" \
         "(container cmd and ${URL}/props both failed) — pass CONCURRENCY=N explicitly" >&2
    exit 2
  fi
  echo "[concurrency-probe] CONCURRENCY=$CONCURRENCY (detected: $_conc_src)"
else
  echo "[concurrency-probe] CONCURRENCY=$CONCURRENCY (explicit)"
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
import json, os, re, statistics, subprocess, time, urllib.request

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

def pct(xs, q):
    # p50 alone describes NO ACTUAL USER once requests queue: with Running:9 /
    # Waiting:55 the admitted streams see a short TTFT and the queued ones wait
    # several batches, so the median lands between the two populations. The tail
    # is also why `aggregate` (divided by the round WALL, set by the slowest
    # stream) comes in below N x median per-stream.
    if not xs: return 0.0
    ss=sorted(xs)
    return ss[min(len(ss)-1, int(q*len(ss)))]

def engine_stats():
    # vLLM logs "Running: N reqs, Waiting: M reqs" and a prefix-cache hit rate.
    # Running is the concurrency the scheduler ACTUALLY admits, which can be far
    # below max_num_seqs when --max-num-batched-tokens caps the per-step budget:
    # a sweep rung labelled N=64 may only ever run ~10 wide. Reporting the label
    # without this reads as a concurrency measurement when it is a queue-depth one.
    if not CONTAINER: return (None, None, None)
    try:
        out=subprocess.run(["docker","logs","--tail","400",CONTAINER],
                           capture_output=True,text=True,timeout=20,
                           encoding="utf-8",errors="replace")
        txt=(out.stdout or "")+(out.stderr or "")
    except Exception:
        return (None, None, None)
    run=wait=hit=None
    for m in re.finditer(r"Running:\s*(\d+)\s*reqs?,\s*Waiting:\s*(\d+)\s*reqs?", txt):
        run, wait = int(m.group(1)), int(m.group(2))
    for m in re.finditer(r"[Pp]refix cache hit rate:\s*([0-9.]+)", txt):
        hit = float(m.group(1))
    return (run, wait, hit)

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

print(f"\n{'round':>5} {'done':>7} {'silent':>7} {'errors':>7} {'vram_MB':>8} {'agg_t/s':>8} {'per-strm':>9} {'ttft_ms':>8} {'pf_t/s':>7}"
      f" {'ttft_p95':>9} {'tps_p05':>8} {'run/wait':>10} {'pfxhit':>6}")
vram0=vram_used_mb()
vram_by_round=[]; mtps_by_round=[]; agg_by_round=[]; ttft_by_round=[]; pf_by_round=[]; bad=0; err_rounds=0
ttft_p95_by_round=[]; tps_p05_by_round=[]; run_by_round=[]; wait_by_round=[]; hit_by_round=[]
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
    mtps_by_round.append(mtps); agg_by_round.append(agg)
    # concurrent prefill: median TTFT across streams; prefill rate derived as
    # PTOK/ttft (same convention as bench.sh's prefill probe: prompt_tokens/TTFT).
    ttfts=[r["ttft"] for r in res if r["ok"] and r["ttft"]]
    ttft_med=statistics.median(ttfts) if ttfts else 0.0
    pf=(PTOK/ttft_med) if ttft_med>0 else 0.0
    ttft_by_round.append(ttft_med); pf_by_round.append(pf)
    # TAIL, not just p50 — see pct() for why the median misleads under queueing.
    ttft_p95=pct(ttfts,0.95); tps_p05=pct(tps_ok,0.05)
    ttft_p95_by_round.append(ttft_p95); tps_p05_by_round.append(tps_p05)
    run,wait,hit=engine_stats()
    run_by_round.append(run); wait_by_round.append(wait); hit_by_round.append(hit)
    print(f"{rnd:>5} {done:>4}/{N:<2} {silent:>7} {errs:>7} {v:>8} {agg:>8.1f} {mtps:>9.1f} {ttft_med*1000:>8.0f} {pf:>7.0f}"
          f" {ttft_p95*1000:>9.0f} {tps_p05:>8.1f} {('-' if run is None else run):>4}/{('-' if wait is None else wait):<5}"
          f" {('-' if hit is None else f'{hit:.1f}'):>6}")
    if done<N or silent or errs: bad+=1
    if errs: err_rounds+=1

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
# aggregate = summed completion tokens / wall per round (what "total rig
# throughput at N agents" means); steady-state = last round, like per-stream.
report_agg=agg_by_round[-1] if agg_by_round else 0.0
# An aggregate computed over a round that had errors is not a throughput number:
# failed streams contribute toks=0 while their prefill still counts in the wall,
# so the figure is (survivors' tokens)/(full round wall) and reads 5-10x low. It
# has been quoted out of context as a measured regression before. Report NaN so a
# broken arm cannot be mistaken for a slow one — clean= and the FAIL verdict below
# already carry the diagnosis.
if err_rounds:
    report_agg=float("nan")
    report_tps=float("nan")
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
print(f"  per-stream decode: {report_tps:.1f} tok/s (steady) · aggregate {report_agg:.1f} tok/s "
      f"({N} streams) · retention {retention*100:.1f}% "
      f"(min {RETENTION_MIN*100:.0f}%)" + (f" · floor {TPS_FLOOR:.0f}" if TPS_FLOOR>0 else " · floor off"))
steady_ttft=ttft_by_round[-1] if ttft_by_round else 0.0
steady_pf=pf_by_round[-1] if pf_by_round else 0.0
print(f"  concurrent prefill: steady TTFT {steady_ttft*1000:.0f} ms (median of {N} streams) "
      f"· ~{steady_pf:.0f} tok/s/stream derived @ {PTOK}-tok prompts")
s_p95=ttft_p95_by_round[-1] if ttft_p95_by_round else 0.0
s_p05=tps_p05_by_round[-1] if tps_p05_by_round else 0.0
s_run=run_by_round[-1] if run_by_round else None
s_wait=wait_by_round[-1] if wait_by_round else None
s_hit=hit_by_round[-1] if hit_by_round else None
print(f"  tail: TTFT p95 {s_p95*1000:.0f} ms (vs p50 {steady_ttft*1000:.0f}) "
      f"· slowest-decile stream {s_p05:.1f} tok/s (vs median {report_tps:.1f})")
if s_run is not None:
    admitted = "" if s_run>=N else f"  ** engine admitted {s_run} of {N} requested — the rest QUEUED **"
    print(f"  engine: running {s_run} · waiting {s_wait}"
          + (f" · prefix-cache hit {s_hit:.1f}%" if s_hit is not None else "")
          + admitted)
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
      f"agg_tps={report_agg:.2f} retention={retention:.3f} leak={leak} vram_peak={vram_peak} floor_ok={int(floor_ok)} "
      f"ttft_ms={steady_ttft*1000:.0f} pf_tps={steady_pf:.1f} "
      f"ttft_p95_ms={s_p95*1000:.0f} tps_p05={s_p05:.2f} "
      f"running={'-' if s_run is None else s_run} waiting={'-' if s_wait is None else s_wait} "
      f"prefix_hit={'-' if s_hit is None else f'{s_hit:.1f}'}")
raise SystemExit(0 if PASS else 1)
PY
}

# --- SWEEP: reboot per N, probe, find the throughput knee ----------------------
if [[ -n "$SWEEP" ]]; then
  # SLUG presence already validated up top, before environment probing.
  echo "[sweep] slug=$SLUG N in { $SWEEP } · floor=${TPS_FLOOR} tok/s/stream · reboots the server per N"
  knee=""; knee_tps=""; knee_agg=""; sweep_rows=""
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
    # Re-resolve the served model AFTER the boot. The top-of-script autodetect
    # ran BEFORE the sweep booted anything on $URL, so it silently fell back to
    # the default — which 404s every request against any other model (the whole
    # arm reads as errors=N). Skip when the caller pinned MODEL explicitly.
    if [[ -z "$MODEL_PINNED" ]]; then
      det="$(curl -s -m 5 "${URL}/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
      if [[ -n "$det" && "$det" != "$MODEL" ]]; then
        echo "[sweep] served model: $det (re-resolved post-boot; was: $MODEL)"
        MODEL="$det"
      fi
    fi
    out="$(run_probe "$N" || true)"; echo "$out"
    line="$(printf '%s\n' "$out" | grep -m1 '^RESULT ' || true)"
    clean="$(sed -n 's/.* clean=\([0-9]*\).*/\1/p' <<<"$line")"
    mps="$(sed -n 's/.* mps_tps=\([0-9.]*\).*/\1/p' <<<"$line")"
    agg="$(sed -n 's/.* agg_tps=\([0-9.]*\).*/\1/p' <<<"$line")"
    sweep_rows="${sweep_rows:-}
  N=$N  per-stream=${mps:-?} tok/s  aggregate=${agg:-?} tok/s  clean=${clean:-?}"
    # knee = largest N that is fit-clean AND (floor off OR per-stream TPS >= floor)
    above="$(awk -v t="$mps" -v f="$TPS_FLOOR" 'BEGIN{print (f<=0 || t>=f)?1:0}')"
    if [[ "$clean" == "1" && "$above" == "1" ]]; then knee="$N"; knee_tps="$mps"; knee_agg="$agg"; fi
  done
  echo ""
  echo "=== sweep summary (per-stream vs aggregate) ==="
  echo "${sweep_rows:-  (no completed rows)}"
  echo ""
  echo "=== sweep knee ==="
  if [[ -n "$knee" ]]; then
    echo "  largest clean N at/above floor: N=$knee (${knee_tps} tok/s/stream · ${knee_agg:-?} tok/s aggregate)"
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
