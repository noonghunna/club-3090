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
#   bash scripts/concurrency-probe.sh --sweep         # live N×ctx matrix + card
#   SWEEP="4 8 12 16" SLUG=vllm/minimal TPS_FLOOR=15 bash scripts/concurrency-probe.sh
#
# Env: URL (default http://localhost:8010) · MODEL (auto) · CONTAINER (auto for
#   VRAM) · CONCURRENCY (default: served max-num-seqs, else 2) · ROUNDS (5;
#   --sweep defaults to 3) · PROMPT_TOKENS (16000) · GEN_TOKENS (256) ·
#   VRAM_GROWTH_MB (200) · REQ_TIMEOUT (600).
#   Validation knobs: VALIDATE (0) · TARGET_CTX (auto from --max-model-len) ·
#   TPS_FLOOR (0 = report-only) · RETENTION_MIN (0.98) ·
#   SWEEP ("" = single-N) · SLUG (required for SWEEP) · SWEEP_DRY (0) ·
#   BOOT_TIMEOUT (360).
#   --sweep knobs: CTX_SWEEP (1k 4k 8k 16k 32k) · N_LIST (1 2 4 8 16 32) ·
#   KV_TOKENS (0 = auto from vLLM logs) · N_MAX · CTX_MAX · WALL_BUDGET (15m) ·
#   EARLY_STOP (1) · CACHE (shared) · SHARE_FRAC (0.75).
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
PROBE_PY="$ROOT_DIR/scripts/lib/concurrency_probe.py"

usage() {
  cat <<'EOF'
concurrency-probe.sh — concurrent-stream fit + throughput matrix

  bash scripts/concurrency-probe.sh              fit check at the served N
  VALIDATE=1 bash scripts/concurrency-probe.sh   fill target_ctx + retention
  bash scripts/concurrency-probe.sh --sweep      live N×ctx matrix + card
  bash scripts/concurrency-probe.sh --sweep --dry
  SWEEP="4 8 16" SLUG=<slug> bash scripts/concurrency-probe.sh
                                                 reboot-per-N envelope knee

--sweep flags (all optional; clipped to the live server):
  --url URL          default http://localhost:8010
  --ctx 1k,4k,8k,16k,32k
  --n 1,2,4,8,16,32
  --budget 15m       stop the matrix and print a partial card
  --dry              print the clipped grid, do not probe

Does not reboot. Measures the compose that is already serving.
After --sweep, prints a compose recommendation (MAX_NUM_SEQS / MAX_MODEL_LEN)
for keep-full-ctx vs max-aggregate (short prompts).
EOF
}

MATRIX=0
N_LIST_EXPLICIT=0
URL="${URL:-http://localhost:8010}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16000}"
GEN_TOKENS="${GEN_TOKENS:-256}"
VRAM_GROWTH_MB="${VRAM_GROWTH_MB:-200}"
REQ_TIMEOUT="${REQ_TIMEOUT:-600}"
TPS_FLOOR="${TPS_FLOOR:-0}"
RETENTION_MIN="${RETENTION_MIN:-0.98}"
VALIDATE="${VALIDATE:-0}"
TARGET_CTX="${TARGET_CTX:-}"
SWEEP="${SWEEP:-}"
SLUG="${SLUG:-}"
SWEEP_DRY="${SWEEP_DRY:-0}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-360}"
CTX_SWEEP="${CTX_SWEEP:-}"
N_LIST="${N_LIST:-}"
KV_TOKENS="${KV_TOKENS:-}"
N_MAX="${N_MAX:-}"
CTX_MAX="${CTX_MAX:-}"
WALL_BUDGET="${WALL_BUDGET:-}"
EARLY_STOP="${EARLY_STOP:-1}"
SHARE_FRAC="${SHARE_FRAC:-0.75}"
UNIQUE_MIN="${UNIQUE_MIN:-256}"
# CACHE / ROUNDS default after we know the mode — --sweep wants 3 / shared.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep) MATRIX=1; shift ;;
    --validate) VALIDATE=1; shift ;;
    --dry) SWEEP_DRY=1; shift ;;
    --url)
      [[ $# -ge 2 ]] || { echo "--url needs a value" >&2; exit 2; }
      URL="$2"; shift 2 ;;
    --ctx)
      [[ $# -ge 2 ]] || { echo "--ctx needs a value" >&2; exit 2; }
      CTX_SWEEP="$2"; shift 2 ;;
    --n)
      [[ $# -ge 2 ]] || { echo "--n needs a value" >&2; exit 2; }
      N_LIST="$2"; N_LIST_EXPLICIT=1; shift 2 ;;
    --budget)
      [[ $# -ge 2 ]] || { echo "--budget needs a value" >&2; exit 2; }
      WALL_BUDGET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "unknown argument: $1 (try --help)" >&2
      exit 2 ;;
  esac
done

if [[ "$MATRIX" == "1" && "$VALIDATE" == "1" ]]; then
  echo "[concurrency-probe] --sweep and --validate / VALIDATE=1 cannot combine" \
       "(VALIDATE is cache-cold fit at target_ctx; --sweep is the live matrix)" >&2
  exit 2
fi
if [[ "$MATRIX" == "1" && -n "$SWEEP" ]]; then
  echo "[concurrency-probe] --sweep is the live matrix; SWEEP=+SLUG is the reboot knee. Pick one." >&2
  exit 2
fi

if [[ "$MATRIX" == "1" ]]; then
  ROUNDS="${ROUNDS:-3}"
  CACHE="${CACHE:-shared}"
  WALL_BUDGET="${WALL_BUDGET:-15m}"
  CTX_SWEEP="${CTX_SWEEP:-1k 4k 8k 16k 32k}"
  N_LIST="${N_LIST:-1 2 4 8 16 32}"
else
  ROUNDS="${ROUNDS:-5}"
  CACHE="${CACHE:-cold}"
  WALL_BUDGET="${WALL_BUDGET:-0}"
fi

parse_duration() {
  local s="${1:-0}"
  if [[ -z "$s" || "$s" == "0" ]]; then echo 0; return; fi
  if [[ "$s" =~ ^[0-9]+$ ]]; then echo "$s"; return; fi
  if [[ "$s" =~ ^([0-9]+)s$ ]]; then echo "${BASH_REMATCH[1]}"; return; fi
  if [[ "$s" =~ ^([0-9]+)m(in)?$ ]]; then echo $((BASH_REMATCH[1] * 60)); return; fi
  echo "[concurrency-probe] bad --budget / WALL_BUDGET '$s' (use 900, 900s, 15m)" >&2
  exit 2
}
WALL_SECS="$(parse_duration "$WALL_BUDGET")"

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
CONTAINER="${CONTAINER:-$(docker ps --format '{{.Names}}' 2>/dev/null | command grep -m1 -E 'vllm-(qwen|gemma)' || true)}"

_container_cmd() { docker inspect "$CONTAINER" --format '{{join .Config.Cmd " "}}' 2>/dev/null || true; }
_served_seqs()   { _container_cmd | command grep -oE 'max-num-seqs [0-9]+'  | command grep -oE '[0-9]+' | head -1; }
_served_np()     { _container_cmd | command grep -oE '\-np +[0-9]+'         | command grep -oE '[0-9]+' | head -1; }
_served_ctx()    { _container_cmd | command grep -oE 'max-model-len [0-9]+' | command grep -oE '[0-9]+' | head -1; }
# llama.cpp-family servers report the slot count as total_slots on /props.
_props_slots()   { curl -s -m 3 "${URL}/props" 2>/dev/null \
  | python3 -c 'import json,sys; v=json.load(sys.stdin).get("total_slots",""); print(v if isinstance(v,int) else "")' 2>/dev/null; }

_detect_slots() {
  local n
  n="$(_served_seqs || true)"; [[ -n "$n" ]] && { echo "$n"; return; }
  n="$(_served_np || true)";   [[ -n "$n" ]] && { echo "$n"; return; }
  n="$(_props_slots || true)"; [[ -n "$n" ]] && { echo "$n"; return; }
  echo ""
}

_gpu_fp() {
  local names n first
  names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | sed 's/^NVIDIA //' || true)"
  [[ -z "$names" ]] && { echo "? GPU"; return; }
  n="$(printf '%s\n' "$names" | wc -l | tr -d ' ')"
  first="$(printf '%s\n' "$names" | head -1)"
  if [[ "$n" == "1" ]]; then echo "1× ${first}"
  else echo "${n}× ${first}"
  fi
}

_spec_fp() {
  local cmd
  cmd="$(_container_cmd)"
  if [[ -z "$cmd" ]]; then echo "spec ?"; return; fi
  if printf '%s' "$cmd" | command grep -qiE 'dflash|spec-dflash'; then echo "DFlash"; return; fi
  if printf '%s' "$cmd" | command grep -qiE 'num-speculative|speculative-config|--speculative'; then echo "MTP"; return; fi
  if printf '%s' "$cmd" | command grep -qiE 'ngram'; then echo "ngram"; return; fi
  echo "spec off"
}

# CONCURRENCY defaults to the served slot count (the thing we're validating).
# Detection order: vLLM container flag -> llama.cpp container flag -> /props.
# A failed detection is FATAL (#818): the old silent CONCURRENCY=2 fallback
# measured queue-wait as concurrency against 1-slot servers and mislabeled arms.
# SWEEP / --sweep skip detection here: each arm passes CONCURRENCY=$N into
# run_probe, and probing the PRE-sweep server (or a serverless box, in DRY)
# would FATAL on a value nothing consumes.
if [[ -n "$SWEEP" || "$MATRIX" == "1" ]]; then
  : # per-arm CONCURRENCY comes from the sweep / matrix loop
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
if [[ "$VALIDATE" == "1" && -z "$SWEEP" && "$MATRIX" != "1" ]]; then
  CACHE=cold
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
  CACHE="$CACHE" SHARE_FRAC="$SHARE_FRAC" UNIQUE_MIN="$UNIQUE_MIN" \
  python3 "$PROBE_PY"
}

# --- --sweep: live N×ctx matrix, no reboot ------------------------------------
if [[ "$MATRIX" == "1" ]]; then
  slots="$(_detect_slots || true)"
  slots_src="undetected"
  if [[ -n "$(_served_seqs || true)" ]]; then slots_src="container max-num-seqs"
  elif [[ -n "$(_served_np || true)" ]]; then slots_src="container -np"
  elif [[ -n "$(_props_slots || true)" ]]; then slots_src="server /props total_slots"
  fi
  max_len="$(_served_ctx || true)"
  if [[ -z "$KV_TOKENS" || "$KV_TOKENS" == "0" ]]; then
    KV_TOKENS="$(CONTAINER="$CONTAINER" python3 "$PROBE_PY" --detect-kv || true)"
  fi

  if [[ "$SWEEP_DRY" != "1" && -z "$slots" && "$N_LIST_EXPLICIT" != "1" ]]; then
    echo "[concurrency-probe] FATAL: --sweep cannot detect the served slot count" \
         "(container cmd and ${URL}/props both failed) — pass --n 1,2,4 or CONCURRENCY=N" >&2
    exit 2
  fi

  export CTX_SWEEP N_LIST GEN_TOKENS KV_TOKENS N_MAX CTX_MAX
  export SERVED_SLOTS="${slots:-}" SERVED_MAX_LEN="${max_len:-}"
  cache_label="$CACHE"
  if [[ "$CACHE" == "shared" ]]; then
    cache_label="shared $(awk -v f="$SHARE_FRAC" 'BEGIN{printf "%.0f", f*100}')%"
  fi
  header="[sweep] live  slots=${slots:-?} (${slots_src:-undetected})  max-len=${max_len:-?}  KV=${KV_TOKENS:-?}  cache=${cache_label}  rounds=${ROUNDS}  budget=${WALL_BUDGET}  floor=${TPS_FLOOR}"
  PLAN_HEADER="$header" python3 "$PROBE_PY" --plan

  if [[ "$SWEEP_DRY" == "1" ]]; then
    exit 0
  fi

  if [[ "${slots:-1}" == "1" && "$N_LIST_EXPLICIT" != "1" ]]; then
    echo "[sweep] server is 1-slot — raise MAX_NUM_SEQS / -np and rerun for a real sweep" >&2
  fi

  t0_matrix=$(date +%s)
  budget_hit=0
  declare -A ROW_DEAD=()
  cells_jsonl="$(mktemp /tmp/cprobe-cells.XXXXXX)"
  trap 'rm -f "$cells_jsonl"' EXIT

  emit_row() {
    python3 - "$cells_jsonl" <<'PY'
import json, sys
path = sys.argv[1]
row = json.loads(sys.stdin.read())
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
  }

  while IFS=$'\t' read -r action ctx n reason; do
    [[ -z "${action:-}" ]] && continue
    if [[ "$budget_hit" == "1" ]]; then
      echo "[sweep] ${ctx} tok  N=${n}: skip (budget)"
      printf '%s\n' "{\"ctx\":$ctx,\"n\":$n,\"skip\":\"budget\"}" | emit_row
      continue
    fi
    if [[ "$action" == "skip" ]]; then
      echo "[sweep] ${ctx} tok  N=${n}: skip (${reason})"
      printf '%s\n' "$(python3 -c 'import json,sys; print(json.dumps({"ctx":int(sys.argv[1]),"n":int(sys.argv[2]),"skip":sys.argv[3]}))' "$ctx" "$n" "${reason:-clipped}")" | emit_row
      continue
    fi
    if [[ "$EARLY_STOP" == "1" && -n "${ROW_DEAD[$ctx]:-}" ]]; then
      echo "[sweep] ${ctx} tok  N=${n}: skip (early-stop — N=${ROW_DEAD[$ctx]} failed)"
      printf '%s\n' "$(python3 -c 'import json,sys; print(json.dumps({"ctx":int(sys.argv[1]),"n":int(sys.argv[2]),"skip":"early-stop"}))' "$ctx" "$n")" | emit_row
      continue
    fi
    echo
    echo "───────────────── ${ctx} tok × N=${n}  (cache=${CACHE}) ─────────────────"
    PROMPT_TOKENS="$ctx"
    set +e
    out="$(run_probe "$n")"
    rc=$?
    set -e
    printf '%s\n' "$out"
    line="$(printf '%s\n' "$out" | command grep -m1 '^RESULT ' || true)"
    python3 - "$cells_jsonl" "$line" "$ctx" "$n" "$rc" <<'PY'
import json, os, sys
path, line, ctx, n, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
fields = {}
s = line.strip()
if s.startswith("RESULT "):
    s = s[7:]
for part in s.split():
    if "=" in part:
        k, v = part.split("=", 1)
        fields[k] = v

def fnum(k):
    v = fields.get(k)
    if v in (None, "", "-", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

row = {
    "ctx": ctx,
    "n": n,
    "skip": None,
    "clean": int(fields["clean"]) if fields.get("clean", "").isdigit() else 0,
    "pass": int(fields["pass"]) if fields.get("pass", "").isdigit() else int(rc == 0),
    "strm": fnum("mps_tps"),
    "agg": fnum("agg_tps"),
    "ttft_s": (fnum("ttft_ms") / 1000.0) if fnum("ttft_ms") is not None else None,
    "vram_gb": (fnum("vram_peak") / 1024.0) if fnum("vram_peak") not in (None, -1.0) else None,
    "running": fields.get("running", "-"),
    "waiting": fields.get("waiting", "-"),
    "result": line,
}
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
    fail=0
    clean="$(sed -n 's/.* clean=\([0-9]*\).*/\1/p' <<<"$line")"
    running="$(sed -n 's/.* running=\([^ ]*\).*/\1/p' <<<"$line")"
    [[ "$rc" != "0" || "$clean" != "1" ]] && fail=1
    if [[ "$running" =~ ^[0-9]+$ && "$running" -lt "$n" ]]; then
      fail=1
      echo "[sweep] admitted ${running}/${n} — treating as fail for early-stop"
    fi
    if [[ "$fail" == "1" && "$EARLY_STOP" == "1" ]]; then
      ROW_DEAD[$ctx]="$n"
    fi
    now=$(date +%s)
    if [[ "$WALL_SECS" -gt 0 && $((now - t0_matrix)) -ge "$WALL_SECS" ]]; then
      echo "[sweep] WALL_BUDGET=${WALL_BUDGET} reached — remaining cells skipped"
      budget_hit=1
    fi
  done < <(python3 "$PROBE_PY" --plan-tsv)

  rec_json="$(mktemp /tmp/cprobe-rec.XXXXXX)"
  if [[ "$slots_src" == "container -np" ]]; then engine="llamacpp"
  elif [[ "$slots_src" == "container max-num-seqs" ]]; then engine="vllm"
  else engine=""
  fi
  MODEL="$MODEL" SLUG="${SLUG:-}" SPEC="$(_spec_fp)" GPUS="$(_gpu_fp)" \
  KV_TOKENS="${KV_TOKENS:-}" SLOTS="${slots:-}" GEN_TOKENS="$GEN_TOKENS" \
  CACHE_LABEL="$cache_label" ENGINE="$engine" SERVED_MAX_LEN="${max_len:-}" \
  PROBE_LIBDIR="$ROOT_DIR/scripts/lib" \
  python3 - "$cells_jsonl" "$rec_json" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ.get("PROBE_LIBDIR") or "")
from concurrency_probe import recommend_compose
rows = []
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
kv = os.environ.get("KV_TOKENS") or None
slots = os.environ.get("SLOTS") or None
served = os.environ.get("SERVED_MAX_LEN") or None
rec = {
    "model": os.environ.get("MODEL") or "?",
    "slug": os.environ.get("SLUG") or "",
    "spec": os.environ.get("SPEC") or "spec ?",
    "gpus": os.environ.get("GPUS") or "? GPU",
    "kv_tokens": int(kv) if kv and str(kv).isdigit() else None,
    "slots": int(slots) if slots and str(slots).isdigit() else None,
    "served_max_len": int(served) if served and str(served).isdigit() else None,
    "engine": os.environ.get("ENGINE") or "",
    "gen_tokens": int(os.environ.get("GEN_TOKENS") or 256),
    "cache": os.environ.get("CACHE_LABEL") or "cold",
    "command": "bash scripts/concurrency-probe.sh --sweep",
    "rows": rows,
}
rec["recommend"] = recommend_compose(rec)
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=2)
    fh.write("\n")
PY

  echo
  python3 "$PROBE_PY" --card < "$rec_json"

  out_dir="$ROOT_DIR/results"
  mkdir -p "$out_dir"
  stamp="$(date +%Y%m%d-%H%M%S)"
  safe_model="$(printf '%s' "$MODEL" | tr '/ :' '___')"
  base="$out_dir/concurrency-${stamp}-${safe_model}"
  cp "$rec_json" "${base}.json"
  python3 "$PROBE_PY" --card < "$rec_json" > "${base}.md"
  echo
  echo "[sweep] wrote ${base}.md  +  ${base}.json"
  rm -f "$rec_json"
  exit 0
fi

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
    line="$(printf '%s\n' "$out" | command grep -m1 '^RESULT ' || true)"
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
