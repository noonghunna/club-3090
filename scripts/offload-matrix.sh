#!/usr/bin/env bash
# offload-matrix.sh — serving matrix for CPU expert-offloaded MoE models.
#
# Sweeps up to SEVEN dimensions and emits an append-only TSV:
#   concurrency (N) x draft depth (n-max) x drafter x offloaded layers x ctx
#   x expert-cache MiB x cache admission policy
# plus REPS (repeat each arm; medians) and external bandwidth telemetry.
#
# ---------------------------------------------------------------------------
# ⚠ ENGINE SCOPE — llama.cpp AND ITS FORKS ONLY. NOT vLLM, SGLang, TensorRT-LLM,
#   MLC, ExLlama or anything else.
#
#   This is not merely a flag-syntax limitation; the tool measures a mechanism
#   that only llama.cpp implements this way:
#     * CLI/flags: -m -ngl -ot -ub -ct -sm -ts -np -t --rope-scaling
#     * log scraping: "n_layer =", "n_expert =", "slot print_timing", "eval time",
#       "n_ctx_seq", "draft acceptance", "[moe-cache] ... slots=/hits=/evictions="
#     * endpoints: /health, /v1/models (llama-server)
#     * `--moe-cache` + GGML_CUDA_MOE_CACHE_* (fork-only; auto-detected)
#
#   Why it cannot be ported by swapping flags: llama.cpp offload COMPUTES the
#   offloaded experts ON THE CPU (activations cross PCIe; bound by RAM bandwidth
#   and core count). vLLM's CPU offload instead streams expert WEIGHTS to the GPU
#   and computes there (bound by PCIe bandwidth). The dimensions here — offloaded
#   layer count, a VRAM cache of hot CPU-resident experts, its admission policy —
#   have no counterpart in that design, so the numbers would not be comparable
#   even if the flags were translated. Benchmark those engines with their own tools.
#
# REQUIREMENTS
#   * llama-server binary (LLAMA_SERVER=, or on PATH)
#   * a GGUF MoE model                (MODEL= — REQUIRED)
#   * nvidia-smi for GPU telemetry    (optional; columns become "-" without it)
#   * The expert-cache dimensions (SWEEP_CACHE / SWEEP_ADMIT) and the pool/hit
#     columns need a build with the VRAM MoE expert cache (`--moe-cache`). The
#     script AUTO-DETECTS it and degrades gracefully on mainline llama.cpp.
#
# QUICK START
#   MODEL=/path/model.gguf bash offload-matrix.sh
#   MODEL=... SWEEP_N="1 4 8" SWEEP_NMAX="0 3" REPS=3 bash offload-matrix.sh
#   MODEL=... SWEEP_OFFLOAD="19 28" SWEEP_CTX="65536 262144" bash offload-matrix.sh
#   MODEL=... SWEEP_DRAFTER="ngram-mod suffix" SWEEP_NMAX="0 3" bash offload-matrix.sh
#
# Everything rig-specific is an env var with a sane default or is auto-detected:
# thread count (nproc/2), GPU count and tensor split, total layer count (probed
# from the model), KV type, rope args, and the offload layer list (generated, not
# hardcoded). No paths, core counts or model quirks are baked in.
# ---------------------------------------------------------------------------
set -uo pipefail

# Python decodes reads/stdout/argv with the LOCALE codec; UTF-8 mode (PEP 540) overrides
# that in one move. Defaulted, not forced, so a user setting PYTHONUTF8=0 keeps control.
export PYTHONUTF8="${PYTHONUTF8:-1}"

die() { echo "ERROR: $*" >&2; exit 2; }

# ---- inherit from a RUNNING llama-server (when MODEL is unset) --------------
# The richest source is the live process's own argv: it carries -m, -ot, -t, -ub,
# -ct, -ngl, -ts and rope flags. /v1/models only gives the served id (which for
# llama-server happens to be the model path, but nothing else).
# Explicit env always wins; INHERIT=0 disables.
INHERIT="${INHERIT:-1}"
_running_pid="$(pgrep -x llama-server 2>/dev/null | head -1 || true)"
if [[ "$INHERIT" == 1 && -n "$_running_pid" ]]; then
  _argv="$(tr '\0' ' ' < "/proc/$_running_pid/cmdline" 2>/dev/null || true)"
  _get() { printf '%s' "$_argv" | command grep -oE -- "(^| )$1 [^ ]+" | tail -1 | awk '{print $2}'; }
  _got=""
  if [[ -z "${MODEL:-}" ]]; then
    _m="$(_get -m)"; [[ -n "$_m" ]] && { MODEL="$_m"; _got="$_got model"; }
  fi
  [[ -z "${THREADS+x}"  ]] && { _v="$(_get -t)";    [[ -n "$_v" ]] && { THREADS="$_v";  _got="$_got threads"; }; }
  [[ -z "${UBATCH+x}"   ]] && { _v="$(_get -ub)";   [[ -n "$_v" ]] && { UBATCH="$_v";   _got="$_got ubatch"; }; }
  [[ -z "${KV_TYPE+x}"  ]] && { _v="$(_get -ct)";   [[ -n "$_v" ]] && { KV_TYPE="$_v";  _got="$_got kv"; }; }
  [[ -z "${NGL+x}"      ]] && { _v="$(_get -ngl)";  [[ -n "$_v" ]] && { NGL="$_v";      _got="$_got ngl"; }; }
  if [[ -z "${ROPE_ARGS+x}" ]]; then
    _rs="$(printf '%s' "$_argv" | command grep -oE -- '--rope-scaling [^ ]+ --rope-scale [^ ]+ --yarn-orig-ctx [^ ]+' | tail -1)"
    [[ -n "$_rs" ]] && { ROPE_ARGS="$_rs"; _got="$_got rope"; }
  fi
  # the live -ot tells us how many layers are currently offloaded — a natural
  # anchor for SWEEP_OFFLOAD candidates
  _live_ot="$(printf '%s' "$_argv" | command grep -oE -- "-ot [^ ]+" | tail -1 | awk '{print $2}')"
  if [[ -n "$_live_ot" ]]; then
    LIVE_OFFLOAD="$(printf '%s' "$_live_ot" | command grep -oE '[0-9]+\|' | wc -l)"
    (( LIVE_OFFLOAD > 0 )) && LIVE_OFFLOAD=$(( LIVE_OFFLOAD + 1 ))   # last id has no trailing |
    _got="$_got offload(~${LIVE_OFFLOAD}L)"
  fi
  [[ -n "$_got" ]] && { echo "inherited from running llama-server (pid $_running_pid):$_got"; \
                        echo "  (explicit env overrides any of these; INHERIT=0 disables)"; }
fi

# ---- required / discovered -------------------------------------------------
MODEL="${MODEL:-}"
[[ -n "$MODEL" ]] || die "MODEL=/path/to/model.gguf is required (or start a llama-server first and let INHERIT pick it up)"
[[ -f "$MODEL" ]]  || die "MODEL not found: $MODEL"
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
[[ -n "$LLAMA_SERVER" && -x "$LLAMA_SERVER" ]] || die "llama-server not found; set LLAMA_SERVER=/path/to/llama-server"
DRAFTER_GGUF="${DRAFTER_GGUF:-}"          # only needed if SWEEP_DRAFTER includes a model-based drafter
OUT_DIR="${OUT_DIR:-$PWD/offload-matrix-out}"; mkdir -p "$OUT_DIR"
TSV="${TSV:-$OUT_DIR/offload-matrix-results.tsv}"
PORT="${PORT:-8099}"
HOST="${HOST:-127.0.0.1}"

# ---- auto-detected hardware ------------------------------------------------
NPROC="$(nproc 2>/dev/null || echo 8)"
THREADS="${THREADS:-$(( NPROC / 2 ))}"; (( THREADS < 1 )) && THREADS=1
if command -v nvidia-smi >/dev/null 2>&1; then
  NGPU="${NGPU:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}"
else
  NGPU="${NGPU:-0}"
fi
# even tensor split across detected GPUs unless the caller overrides SPLIT_ARGS
if [[ -z "${SPLIT_ARGS+x}" ]]; then
  if (( NGPU > 1 )); then
    ts="1"; for ((g=1; g<NGPU; g++)); do ts="$ts,1"; done
    SPLIT_ARGS="-sm layer -ts $ts"
  else
    SPLIT_ARGS=""
  fi
fi

# ---- model / serving knobs (all overridable) -------------------------------
NGL="${NGL:-99}"
UBATCH="${UBATCH:-2048}"
KV_TYPE="${KV_TYPE:-f16}"                 # f16 is mainline-safe; set q8_0 / turbo4 / etc. per your build
ROPE_ARGS="${ROPE_ARGS:-}"                # e.g. "--rope-scaling yarn --rope-scale 32 --yarn-orig-ctx 8192"
EXTRA_ARGS="${EXTRA_ARGS:-}"              # anything else you want on every boot
EXPERT_TENSOR_RE="${EXPERT_TENSOR_RE:-ffn_(gate|up|down)_exps}"   # tensor-name group to offload
LAYERS_TOTAL="${LAYERS_TOTAL:-}"          # auto-probed if empty
FIRST_MOE_LAYER="${FIRST_MOE_LAYER:-1}"   # many MoE models keep layer 0 dense

# ---- DIMENSION DEPENDENCY ORDER --------------------------------------------
# The dimensions are NOT independent. Measure upstream ones first; a downstream
# result measured on an unsettled base may have to be thrown away.
#
#   1. offload layers   BASE. Everything sits on it. Changing it changes hit rate,
#                       which changes miss-streaming, which changes the acceptance
#                       a drafter must clear to pay for itself.
#   2. cache MiB +      Tunes the base. Admission policy alone was worth +5.4% WITH
#      admission        speculation on one rig — rank drafters at the wrong
#                       admission and you rank them handicapped.
#   3. ctx              Bounded by whatever layout+cache leave free, so it cannot
#                       be chosen before them.
#   4. concurrency N    Find the knee on the settled+tuned base, spec OFF.
#   5. drafter + n-max  LAST. Only meaningful at the N where speculation is still
#                       viable, on a base that is not moving under it.
#
# PLAN=1 prints the staged invocation sequence for this rig and exits.
# ---------------------------------------------------------------------------

# ---- PRESETS ---------------------------------------------------------------
# Every sweep var has a default, but the bare default is a ONE-ARM smoke test.
# A preset answers a specific question with a sensible matrix. Explicit SWEEP_*
# env vars always win over the preset.
#   PRESET=smoke        1 arm — prove the harness works on this rig (default)
#   PRESET=spec         does speculation help me?      N=1, n-max 0/3
#   PRESET=drafters     which drafter is best?         N=1, all model-free types
#   PRESET=depth        best draft depth?              N=1, n-max 0/2/3/4/5
#   PRESET=concurrency  how many agents fit?           N=1/2/4/6/8, no spec
#   PRESET=offload      how many layers to offload?    needs OFFLOAD_SET
#   PRESET=cache        pool size + admission policy   needs a --moe-cache build
#   PRESET=full         everything (SLOW — check the printed arm count first)
PRESET="${PRESET:-smoke}"
OFFLOAD_SET="${OFFLOAD_SET:-}"        # e.g. "19 28 33" — used by offload/full presets
case "$PRESET" in
  smoke)      : ;;
  spec)       : "${SWEEP_N:=1}" "${SWEEP_NMAX:=0 3}" "${REPS:=2}" ;;
  drafters)   : "${SWEEP_N:=1}" "${SWEEP_NMAX:=0 3}" "${REPS:=2}" \
                "${SWEEP_DRAFTER:=ngram-mod ngram-map-k suffix copyspec}" ;;
  depth)      : "${SWEEP_N:=1}" "${SWEEP_NMAX:=0 2 3 4 5}" "${REPS:=2}" ;;
  concurrency): "${SWEEP_N:=1 2 4 6 8}" "${SWEEP_NMAX:=0}" "${REPS:=1}" ;;
  offload)    : "${SWEEP_N:=1}" "${SWEEP_NMAX:=0}" "${REPS:=2}" \
                "${SWEEP_OFFLOAD:=${OFFLOAD_SET}}" ;;
  cache)      : "${SWEEP_N:=1}" "${SWEEP_NMAX:=0 3}" "${REPS:=1}" \
                "${SWEEP_CACHE:=2048 4096 8192}" "${SWEEP_ADMIT:=1/64 3/16}" ;;
  full)       : "${SWEEP_N:=1 4 8}" "${SWEEP_NMAX:=0 3}" "${REPS:=2}" \
                "${SWEEP_DRAFTER:=ngram-mod suffix}" "${SWEEP_OFFLOAD:=${OFFLOAD_SET}}" ;;
  *)          die "unknown PRESET='$PRESET' (smoke|spec|drafters|depth|concurrency|offload|cache|full)" ;;
esac
[[ "$PRESET" == offload || "$PRESET" == full ]] && [[ -z "${SWEEP_OFFLOAD:-}" ]] && \
  die "PRESET=$PRESET needs OFFLOAD_SET=\"19 28\" (layer counts to compare)"

# ---- sweep dimensions ------------------------------------------------------
SWEEP_OFFLOAD="${SWEEP_OFFLOAD:-}"        # counts of layers to offload, e.g. "19 28"; empty = one arm, no offload
SWEEP_N="${SWEEP_N:-1}"
SWEEP_NMAX="${SWEEP_NMAX:-0}"             # 0 = paired no-spec baseline; keep it present
SWEEP_DRAFTER="${SWEEP_DRAFTER:-ngram-mod}"
SWEEP_CTX="${SWEEP_CTX:-32768}"
SWEEP_CACHE="${SWEEP_CACHE:-0}"           # per-device --moe-cache MiB; 0 = don't pass the flag
SWEEP_ADMIT="${SWEEP_ADMIT:-default}"     # "ADMIT/THROTTLE" pairs, or "default" to leave env unset
REPS="${REPS:-1}"
ROUNDS="${ROUNDS:-4}"
GEN="${GEN:-900}"
RESERVE_MB="${RESERVE_MB:-}"              # expert-cache reserve; unset = engine default
PROMPT_FILE="${PROMPT_FILE:-}"            # override the built-in shapes with one static prompt
SWEEP_PCACHE="${SWEEP_PCACHE:-on}"        # on | off — llama-server prompt (prefix) caching.
                                          # ⚠ The engine defaults this ON and the driver sends the SAME
                                          # prompt each round, so measured rounds reuse the prefix and SKIP
                                          # prefill: 3115ms/2073tok on the first request vs 120ms/1tok
                                          # after, worth ~14% on agg. Realistic for agentic multi-turn,
                                          # optimistic for distinct-prompt fleets — so it is passed
                                          # explicitly and recorded per arm, never left to a default.
SWEEP_SHAPE="${SWEEP_SHAPE:-copy}"        # copy | novel | code  (workload FLIPS the sign of
                                          # spec results, so this is a first-class dimension)
HETERO="${HETERO:-1}"                     # 1 = each slot gets a DIFFERENT prompt instance.
                                          # NOTE: only the `copy` shape can vary per slot (seeded
                                          # doc). `novel`/`code` are byte-identical to bench.sh's
                                          # canonical prompts, so at N>1 those slots share a prompt
                                          # (only the [agent N] salt differs) — which flatters
                                          # multi-slot numbers via expert-union overlap. Prefer
                                          # `copy` for N>1 work; use novel/code for N=1 comparability.
                                          # Near-identical prompts across slots inflate
                                          # expert-union overlap and flatter multi-slot numbers.
AVG_EXPERT_KIB="${AVG_EXPERT_KIB:-}"      # for derived RAM read; auto-read from pool lines if available
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1200}"

# ---- capability detection --------------------------------------------------
# NOTE: do NOT pipe --help into `grep -q` here. grep -q exits on first match,
# SIGPIPEs llama-server (141), and `set -o pipefail` then propagates that as a
# pipeline failure — so a build that DOES support --moe-cache is detected as not
# supporting it. Capture the text first, match in-shell.
HAS_MOE_CACHE=0
_HELP="$("$LLAMA_SERVER" --help 2>&1 || true)"
case "$_HELP" in *--moe-cache*) HAS_MOE_CACHE=1 ;; esac
# engine sanity: refuse anything that is not llama.cpp-family. Every flag we emit
# and every log line we scrape is llama.cpp-specific (see ENGINE SCOPE above).
case "$_HELP" in
  *--n-gpu-layers*|*--override-tensor*) : ;;
  *) die "LLAMA_SERVER does not look like a llama.cpp server (no --n-gpu-layers /
       --override-tensor in --help). This tool is llama.cpp-and-forks ONLY: it emits
       llama.cpp flags and scrapes llama.cpp log lines. vLLM / SGLang / TensorRT-LLM
       measure a different CPU-offload mechanism entirely (weights streamed to GPU
       vs experts computed on CPU) — use their own benchmarking tools." ;;
esac
# same trap: match KV type support without a pipe
case "$_HELP" in *"$KV_TYPE"*) : ;; *) echo "NOTE: KV_TYPE='$KV_TYPE' not listed in --help; boot may reject it." ;; esac
if (( ! HAS_MOE_CACHE )) && [[ "$SWEEP_CACHE" != "0" || "$SWEEP_ADMIT" != "default" ]]; then
  echo "NOTE: this llama-server has no --moe-cache; ignoring SWEEP_CACHE/SWEEP_ADMIT and"
  echo "      leaving pool/hit/eviction columns empty."
  SWEEP_CACHE=0; SWEEP_ADMIT=default
fi

# ---- probe total layer count once (cheap: -ngl 0, tiny ctx, killed at ready) --
probe_layers() {
  local plog="$OUT_DIR/.probe.log"
  # -lv 4 is REQUIRED: the `print_info:` block (which carries n_layer / n_expert)
  # is above the default verbosity 3, so without it the probe waits out its whole
  # timeout for a line that never prints.
  "$LLAMA_SERVER" -m "$MODEL" --host "$HOST" --port "$PORT" -ngl 0 -c 256 -np 1 \
    $ROPE_ARGS -lv 4 > "$plog" 2>&1 &
  local pp=$! i
  for i in $(seq 1 300); do
    kill -0 "$pp" 2>/dev/null || break
    command grep -qE 'n_layer +=' "$plog" && break
    # also stop once the model is up: if n_layer has not appeared by then it is
    # not going to, so fail fast instead of burning the full timeout.
    command grep -q 'model loaded' "$plog" && break
    sleep 1
  done
  kill "$pp" 2>/dev/null; wait "$pp" 2>/dev/null
  command grep -oE 'n_layer +=[ ]*[0-9]+' "$plog" | head -1 | command grep -oE '[0-9]+'
}
probe_experts() {   # 0 / empty => dense model => offloading "_exps" matches nothing
  command grep -oE 'n_expert +=[ ]*[0-9]+' "$OUT_DIR/.probe.log" 2>/dev/null | head -1 | command grep -oE '[0-9]+'
}
if [[ -z "$LAYERS_TOTAL" && -n "$SWEEP_OFFLOAD" ]]; then
  echo "probing model layer count ..."
  LAYERS_TOTAL="$(probe_layers)"
  [[ -n "$LAYERS_TOTAL" ]] || die "could not probe n_layer; set LAYERS_TOTAL=<n> explicitly"
  N_EXPERT="$(probe_experts)"
  echo "  n_layer = $LAYERS_TOTAL  n_expert = ${N_EXPERT:-unknown}"
fi
N_EXPERT="${N_EXPERT:-}"

# Build an INTERLEAVED offload list of $1 layers from [FIRST_MOE_LAYER, LAYERS_TOTAL-1].
# Interleaved beats contiguous on multi-GPU (contiguous clusters resident experts on
# one card and cannot be rebalanced, since KV follows layer assignment). Adjacent
# offloaded layers are avoided where the count allows, so GPU work can overlap CPU work.
gen_offload_re() {
  local want="$1"
  python3 - "$want" "$FIRST_MOE_LAYER" "$LAYERS_TOTAL" "$EXPERT_TENSOR_RE" <<'PY'
import sys
want, lo, total, tre = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pool = list(range(lo, total))
if want >= len(pool):
    sel = pool
else:
    # prefer a stride that avoids adjacency; fall back to even spacing when the
    # requested count is too dense for any gap to exist
    stride = len(pool) / want
    sel = sorted({pool[min(len(pool)-1, int(round(i*stride)))] for i in range(want)})
    while len(sel) < want:                        # spacing collisions -> backfill
        for c in pool:
            if c not in sel:
                sel.append(c); break
        sel = sorted(set(sel))
print(f"blk\\.({'|'.join(map(str, sel))})\\.{tre}\\.weight=CPU")
print(len(sel), file=sys.stderr)
PY
}

HDR=$'arm\trep\tshape\tpcache\toffload\tlayers_total\tN\tctx_req\tctx_slot\tdrafter\tnmax\tmax_batch\tcache_mb\tadmit\tthrottle\tagg\tstrm\tttft_ms\taccept\tpool_slots\thits_pct\tmisses\tevicts\trxpci\ttxpci\tsm_pct\tmemctl_pct\tcpu_pct\tram_rd_mbps\tvram_peak\tvram_idle\tvram_post\tleak\tbypass\terrors\tstatus'
[[ -f "$TSV" ]] || printf '%s\n' "$HDR" > "$TSV"
NCOL=$(awk -F'\t' '{print NF}' <<<"$HDR")
# Emit a row padded to the header width, with $status always last. Early-exit rows
# (BOOT_FAIL/TIMEOUT) used to be hand-built with a fixed dash run that omitted `shape`
# and fell 4 columns short of the header -- every boot failure wrote SHIFTED columns,
# and a sweep that ladders context until OOM produces those routinely. The header is
# now the only place column count is defined.
emit_row() {   # $1=status, $2.. = leading values in header order starting at `arm`
  local status="$1"; shift
  local -a f=("$@")
  if (( ${#f[@]} > NCOL - 1 )); then
    echo "  !! emit_row: ${#f[@]} values but header holds $((NCOL-1)) + status -- TSV would be corrupt" >&2
    return 1
  fi
  while (( ${#f[@]} < NCOL - 1 )); do f+=("-"); done
  local IFS=$'\t'; printf '%s\t%s\n' "${f[*]}" "$status" >> "$TSV"
}

cpu_snap() { awk '/^cpu /{i=$5;t=0;for(j=2;j<=NF;j++)t+=$j;print t,i}' /proc/stat; }

# depth flag is PER DRAFTER TYPE. Note --draft-max does NOT clamp the ngram family
# in forks that expose size-m/n-max (their defaults can be 48-64 tokens, which will
# blow past any expert-cache batch clamp and silently disable the cache).
drafter_flags() {
  local d="$1" depth="$2"
  case "$d" in
    ngram-mod)     echo "--spec-type ngram-mod --spec-ngram-mod-n-max $depth" ;;
    ngram-simple)  echo "--spec-type ngram-simple --spec-ngram-simple-size-m $depth" ;;
    ngram-map-k)   echo "--spec-type ngram-map-k --spec-ngram-map-k-size-m $depth" ;;
    ngram-map-k4v) echo "--spec-type ngram-map-k4v --spec-ngram-map-k4v-size-m $depth" ;;
    ngram-cache)   echo "--spec-type ngram-cache --draft-max $depth" ;;
    suffix|copyspec|recycle) echo "--spec-type $d --draft-max $depth" ;;
    draft-model)   [[ -n "$DRAFTER_GGUF" ]] || { echo UNKNOWN_DRAFTER; return; }
                   echo "-md $DRAFTER_GGUF --draft-max $depth" ;;
    *)             echo "UNKNOWN_DRAFTER" ;;
  esac
}

run_arm() {
  set +u
  local off="$1" n="$2" nmax="$3" cache="$4" admit_pair="$5" ctx="$6" drafter="$7" rep="$8" shape="${9:-copy}" pcache="${10:-on}"
  local admit throttle
  if [[ "$admit_pair" == default ]]; then admit=""; throttle=""
  else admit="${admit_pair%%/*}"; throttle="${admit_pair##*/}"; fi
  local ot="" nlay="$off"
  if [[ -n "$off" && "$off" != 0 ]]; then
    ot="$(gen_offload_re "$off" 2>/dev/null)"
    [[ -n "$ot" ]] || { echo "  !! could not build offload regex"; return 1; }
  fi
  local dtag="$drafter"; (( nmax == 0 )) && dtag="none"
  local arm="off${off:-0}-n${n}-c$((ctx/1024))K-${shape}-${dtag}d${nmax}-p${cache}-a${admit_pair//\//x}-pc${pcache}"
  local log="$OUT_DIR/om-$arm-r$rep.log"
  # RULE: the expert-cache batch clamp must be >= n-max + 1 (a sequence's verify
  # block). Whether concurrency also enters it depends on whether the engine packs
  # multiple sequences into one ubatch — on hybrid/recurrent memory with
  # kv_unified=false it does not. We size for the safe upper bound.
  # MAX_BATCH sizing. MEASURED 2026-07-30, and it corrects an earlier published rule
  # of ours ("MAX_BATCH >= n-max+1, N does NOT enter it"):
  #
  #     N=1 n-max=0/3  clamp 4 -> clean
  #     N=2 n-max=0    clamp 4 -> clean
  #     N=2 n-max=3    clamp 4 -> BYPASS, packed batch 5
  #     N=4 n-max=0    clamp 4 -> BYPASS, packed batch 6   <- no speculation at all
  #     N=4 n-max=3    clamp 4 -> BYPASS, packed batch 6
  #
  # The packed decode batch grows with n-max AND N jointly. Worse, those observed
  # sizes are LOWER BOUNDS: the engine warns once per session on the FIRST breach,
  # never reporting the largest, so no formula can be fitted from them.
  # Since the engine ceiling is 8 and sitting at it has no measured cost, take the
  # ceiling whenever concurrency is in play and treat n-max+1 purely as a floor.
  # Guessing tighter buys nothing and risks a silently uncached arm.
  local mb=$(( nmax + 1 ))
  (( n > 1 )) && mb=8
  (( mb < 4 )) && mb=4; (( mb > 8 )) && mb=8
  # R03: the engine clamps MAX_BATCH to [1,8]. At n-max >= 8 the clamp CANNOT satisfy
  # mb >= nmax+1, so the expert cache would refuse every decode and the arm would
  # silently measure the no-cache path. Refuse before burning a boot on it.
  if (( HAS_MOE_CACHE )) && [[ "$cache" != 0 ]] && (( mb < nmax + 1 )); then
    echo "### $arm rep$rep -- REFUSED"
    echo "  !! MAX_BATCH clamps to $mb but n-max=$nmax needs >= $((nmax+1)). The expert cache"
    echo "     would be bypassed on every decode and this arm would measure the no-cache"
    echo "     path while looking healthy. Use n-max <= 7, or SWEEP_CACHE=0 to opt out."
    emit_row MB_CLAMP_CONFLICT "$arm" "$rep" "$shape" "${off:-0}" "${LAYERS_TOTAL:--}" "$n" \
             "-" "$dtag" "$nmax" "$mb" "$cache" "${admit:--}" "${throttle:--}"
    return 1
  fi
  local extra=()
  if (( nmax > 0 )); then
    local df; df="$(drafter_flags "$drafter" "$nmax")"
    [[ "$df" == UNKNOWN_DRAFTER ]] && { echo "  !! unusable drafter '$drafter' (model-based drafters need DRAFTER_GGUF)"; return 1; }
    read -r -a extra <<< "$df"
  fi
  local cache_args=()
  (( HAS_MOE_CACHE )) && [[ "$cache" != 0 ]] && cache_args=(--moe-cache "$cache")
  # Passed EXPLICITLY either way, so the arm's log records the intent rather than
  # inheriting whatever the engine happens to default to.
  local pc_args=(); [[ "$pcache" == off ]] && pc_args=(--no-cache-prompt) || pc_args=(--cache-prompt)

  # R01: if the port already answers, a foreign server would satisfy our readiness
  # probe while our own boot bind-fails -- every arm then measures the SAME untouched
  # config and reports OK. Never measure a server we did not start.
  if curl -sf -m 2 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "### $arm rep$rep -- REFUSED"
    echo "  !! something is ALREADY serving on $HOST:$PORT. This arm would measure that"
    echo "     server, not one it booted. Stop it, or set PORT= to a free port."
    emit_row PORT_BUSY "$arm" "$rep" "$shape" "${off:-0}" "${LAYERS_TOTAL:--}" "$n" \
             "-" "$dtag" "$nmax" "$mb" "$cache" "${admit:--}" "${throttle:--}"
    return 1
  fi
  echo "### $arm rep$rep  offload=${off:-0}/${LAYERS_TOTAL:-?} N=$n ctx=$ctx depth=$nmax($dtag) pc=$pcache clamp=$mb pool=${cache}MiB admit=${admit_pair}  $(date +%T)"

  # DIAGNOSTIC HEADER. Truncates the log (the server below APPENDS), so a re-run into
  # an existing OUT_DIR can never leave stale text for the cache-stat greps to parse.
  # Records the fully-resolved invocation: any single arm is reproducible by hand from
  # its log alone, without re-deriving auto-detected values (threads, split, layers).
  {
    echo "# ===== offload-matrix arm: $arm  rep=$rep  $(date -Is) ====="
    echo "# shape=$shape hetero=$HETERO drafter=$dtag n-max=$nmax cache-batch-clamp=$mb"
    echo "# pool=${cache}MiB admit=${admit_pair} reserve_mb=${RESERVE_MB:-<engine default>}"
    echo "# autodetected: threads=$THREADS gpus=$NGPU layers_total=${LAYERS_TOTAL:-?} has_moe_cache=$HAS_MOE_CACHE"
    echo "# env: GGML_CUDA_MOE_CACHE_ADMIT_AFTER=${admit:-<unset>} THROTTLE=${throttle:-<unset>} MAX_BATCH=${mb}"
    echo "# cmd: $LLAMA_SERVER -m $MODEL --host $HOST --port $PORT -np $n -c $ctx $ROPE_ARGS \\"
    echo "#        -ngl $NGL $SPLIT_ARGS -fa on ${ot:+-ot \"$ot\"} -t $THREADS -ub $UBATCH -ct $KV_TYPE \\"
    echo "#        ${cache_args[*]} ${pc_args[*]} ${extra[*]} $EXTRA_ARGS -lv 4"
    echo "# =========================================================="
  } > "$log"

  ( [[ -n "$admit" ]]      && export GGML_CUDA_MOE_CACHE_ADMIT_AFTER="$admit"
    [[ -n "$throttle" ]]   && export GGML_CUDA_MOE_CACHE_THROTTLE="$throttle"
    [[ -n "$RESERVE_MB" ]] && export GGML_CUDA_MOE_CACHE_RESERVE_MB="$RESERVE_MB"
    (( HAS_MOE_CACHE ))    && export GGML_CUDA_MOE_CACHE_MAX_BATCH="$mb" GGML_CUDA_MOE_CACHE_STATS=1000
    exec "$LLAMA_SERVER" -m "$MODEL" --host "$HOST" --port "$PORT" \
      -np "$n" -c "$ctx" $ROPE_ARGS -ngl "$NGL" $SPLIT_ARGS -fa on \
      ${ot:+-ot "$ot"} -t "$THREADS" -ub "$UBATCH" -ct "$KV_TYPE" \
      "${cache_args[@]}" "${pc_args[@]}" "${extra[@]}" $EXTRA_ARGS -lv 4 ) >> "$log" 2>&1 &
  local pid=$! ok=0 i
  # leading identity columns, shared by the failure rows below (padded by emit_row)
  local -a idcols=("$arm" "$rep" "$shape" "$pcache" "${off:-0}" "${LAYERS_TOTAL:--}" "$n" "$ctx" \
                   "${ctx_slot:--}" "$dtag" "$nmax" "$mb" "$cache" "${admit:--}" "${throttle:--}")
  for i in $(seq 1 "$BOOT_TIMEOUT"); do
    kill -0 "$pid" 2>/dev/null || { echo "  BOOT FAILED (see $log)"; emit_row BOOT_FAIL "${idcols[@]}"; return 1; }
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  (( ok )) || { echo "  BOOT TIMEOUT"; kill "$pid" 2>/dev/null; emit_row TIMEOUT "${idcols[@]}"; return 1; }

  local ctx_slot; ctx_slot=$(command grep -oE 'n_ctx_seq *= *[0-9]+|n_ctx_per_seq *= *[0-9]+' "$log" | head -1 | command grep -oE '[0-9]+')
  # PHASE 1 (before): idle baseline — VRAM after load but before any request, and
  # a short idle dmon sample so live figures can be read against a floor.
  local vram_idle="-" dmon_idle="$OUT_DIR/.idle-$arm-r$rep"
  if (( NGPU > 0 )); then
    vram_idle=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc)
    timeout 4 nvidia-smi dmon -s ut -d 1 -c 3 > "$dmon_idle" 2>/dev/null || true
  fi
  # cache counters BEFORE the measured window, so hits/evictions can be reported as
  # a DELTA (steady state) rather than lifetime-cumulative (which includes the cold
  # fill storm and understates the hit rate).
  local h0 m0 e0
  read -r h0 m0 < <(command grep -oE 'hits=[0-9]+/[0-9]+' "$log" | tail -"${NGPU:-2}" \
    | command grep -oE '[0-9]+/[0-9]+' | awk -F/ '{h+=$1; t+=$2} END{print h+0, (t-h)+0}')
  e0=$(command grep -oE 'evictions=[0-9]+' "$log" | tail -"${NGPU:-2}" | command grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null)
  h0="${h0:-0}"; m0="${m0:-0}"; e0="${e0:-0}"

  # PHASE 2 (during): telemetry windowed to the MEASURED requests only
  local dmon="$OUT_DIR/.dmon-$arm-r$rep" dpid="" vpid=""
  if (( NGPU > 0 )); then
    nvidia-smi dmon -s ut -d 1 > "$dmon" 2>/dev/null & dpid=$!
    ( vp=0; while kill -0 "$pid" 2>/dev/null; do
        v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null)
        [[ -n "$v" && "$v" -gt "$vp" ]] && { vp=$v; echo "$vp" > "$OUT_DIR/.vp-$arm-r$rep"; }; sleep 2
      done ) & vpid=$!
  fi
  read -r c_tot0 c_idle0 < <(cpu_snap); local t0; t0=$(date +%s.%N)

  N="$n" ROUNDS="$ROUNDS" GEN="$GEN" PORT="$PORT" HOST="$HOST" PROMPT_FILE="$PROMPT_FILE" \
  SHAPE="$shape" HETERO="$HETERO" \
  python3 - > "$OUT_DIR/.drv-$arm-r$rep" <<'PY'
import json,os,time,threading,urllib.request,statistics as st
H=os.environ["HOST"]; P=os.environ["PORT"]; N=int(os.environ["N"])
ROUNDS=int(os.environ["ROUNDS"]); GEN=int(os.environ["GEN"]); PF=os.environ.get("PROMPT_FILE","")
SHAPE=os.environ.get("SHAPE","").strip() or "copy"; HETERO=os.environ.get("HETERO","1")=="1"
def doc(seed):
    return "\n".join(f'  {{"id": {i}, "name": "svc-{seed}-{i}", "port": {8000+i}, '
                      f'"replicas": {(i+seed)%5+1}, "region": "eu-west-{(i+seed)%3+1}", '
                      f'"enabled": {"true" if (i+seed)%2 else "false"}}},' for i in range(40))
def shape_prompt(s):
    # HETERO: give each slot a DIFFERENT instance. Near-identical prompts across
    # slots make concurrent slots route to nearly the same experts, which flatters
    # every multi-slot number (union overlap is unrealistically high).
    seed = s if HETERO else 0
    if SHAPE == "copy":     # output mostly reproduced from prompt -> where spec pays
        return ("Here is a service configuration document:\n\n{\n \"services\": [\n"
                + doc(seed) + "\n ]\n}\n\nReproduce the ENTIRE document exactly as given, "
                "changing only every \"region\" value to \"us-east-1\". Output the full JSON "
                "and nothing else.")
    if SHAPE == "code":
        # BYTE-IDENTICAL to bench.sh's canonical PROMPT_CODE, so matrix rows can be
        # laid directly against BENCHMARKS rows. Single algorithm on purpose.
        return "Write a Python implementation of quicksort with comments explaining each step."
    if SHAPE == "novel":
        # BYTE-IDENTICAL to bench.sh's canonical PROMPT_NARR. Single topic on purpose.
        return "Write a detailed 800-word essay explaining transformer attention."
    raise SystemExit(f"unknown SHAPE={SHAPE}")
if PF and os.path.exists(PF):
    _static=open(PF).read()
    def mk(s): return (f"[agent {s}] " if HETERO else "") + _static
else:
    def mk(s): return shape_prompt(s)
aggs=[]; ttfts=[]; errs=[]
def one(s,out):
    t0=time.time(); first=None; toks=0
    body=json.dumps({"model":"x","messages":[{"role":"user","content":mk(s)}],
                     "max_tokens":GEN,"temperature":0.0,"stream":True}).encode()
    req=urllib.request.Request(f"http://{H}:{P}/v1/chat/completions",body,{"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=1800) as r:
            for raw in r:
                ln=raw.decode("utf-8","ignore").strip()
                if not ln.startswith("data: ") or ln.endswith("[DONE]"): continue
                try: d=json.loads(ln[6:])
                except Exception: continue
                c=d.get("choices",[{}])[0].get("delta",{}).get("content")
                if c:
                    if first is None: first=time.time()-t0
                    toks+=1
        out.append((toks,first))
    except Exception as e:
        out.append((0,None)); errs.append(f"{type(e).__name__}: {e}"[:160])
for rnd in range(ROUNDS):
    out=[]; w0=time.time()
    ts=[threading.Thread(target=one,args=(s,out)) for s in range(N)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall=time.time()-w0
    if rnd>0:
        tot=sum(t for t,_ in out)
        if wall: aggs.append(tot/wall)
        f=[x for _,x in out if x]
        if f: ttfts.append(st.median(f))
print(f"{st.median(aggs):.2f}" if aggs else "-")
print(f"{st.median(ttfts)*1000:.0f}" if ttfts else "-")
print(len(errs))                                   # line 3: request error count
print(errs[0] if errs else "-")                    # line 4: first error, for triage
print(f"shape={SHAPE} hetero={HETERO} prompt[0:120]={mk(0)[:120]!r}")   # line 5: what was actually sent
PY
  local t1; t1=$(date +%s.%N); read -r c_tot1 c_idle1 < <(cpu_snap)
  [[ -n "$dpid" ]] && kill "$dpid" 2>/dev/null; [[ -n "$vpid" ]] && kill "$vpid" 2>/dev/null
  local agg ttft elapsed cpu_pct rx tx smp memc
  agg=$(sed -n 1p "$OUT_DIR/.drv-$arm-r$rep"); ttft=$(sed -n 2p "$OUT_DIR/.drv-$arm-r$rep")
  local req_err req_err_msg prompt_dbg
  req_err=$(sed -n 3p "$OUT_DIR/.drv-$arm-r$rep"); req_err_msg=$(sed -n 4p "$OUT_DIR/.drv-$arm-r$rep")
  prompt_dbg=$(sed -n 5p "$OUT_DIR/.drv-$arm-r$rep")
  { echo "# driver: $prompt_dbg"; echo "# driver: request_errors=${req_err:-?} first=${req_err_msg:--}"; } >> "$log"
  elapsed=$(python3 -c "print(max(0.001,$t1-$t0))")
  cpu_pct=$(python3 -c "dt=$c_tot1-$c_tot0; di=$c_idle1-$c_idle0; print(f'{(1-di/dt)*100:.1f}' if dt>0 else '-')")
  if [[ -s "$dmon" ]]; then
    read -r rx tx smp memc < <(awk '!/^#/ && NF>=9 {s+=$2;m+=$3;r+=$8;t+=$9;k++} END{if(k)printf "%.0f %.0f %.1f %.1f",r/k,t/k,s/k,m/k; else print "- - - -"}' "$dmon")
  else rx=-; tx=-; smp=-; memc=-; fi
  rm -f "$dmon" "$OUT_DIR/.drv-$arm-r$rep"

  # PHASE 3 (after): post-run VRAM for a leak check against the idle floor
  local vram_post="-" leak="-"
  if (( NGPU > 0 )); then
    vram_post=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc)
    [[ "$vram_idle" != "-" && -n "$vram_post" ]] && leak=$(( vram_post - vram_idle ))
  fi
  local strm acc pool hits ev miss byp vpeak ramrd aexp
  strm=$(command grep -E 'eval time' "$log" | command grep -v 'prompt eval' \
    | command grep -oE '[0-9.]+ tokens per second' | command grep -oE '^[0-9.]+' | tail -$(( n*2 )) \
    | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%.2f",(NR%2)?a[int((NR+1)/2)]:(a[NR/2]+a[NR/2+1])/2; else print "-"}')
  acc=$(command grep -oE 'draft acceptance = [0-9.]+' "$log" | tail -1 | command grep -oE '[0-9.]+')
  pool=$(command grep -oE 'slots=[0-9]+' "$log" | command grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null)
  # DELTAS across the measured window (not lifetime): excludes the cold fill storm
  local h1 m1 e1
  read -r h1 m1 < <(command grep -oE 'hits=[0-9]+/[0-9]+' "$log" | tail -"${NGPU:-2}" \
    | command grep -oE '[0-9]+/[0-9]+' | awk -F/ '{h+=$1; t+=$2} END{print h+0, (t-h)+0}')
  e1=$(command grep -oE 'evictions=[0-9]+' "$log" | tail -"${NGPU:-2}" | command grep -oE '[0-9]+' | paste -sd+ | bc 2>/dev/null)
  h1="${h1:-0}"; m1="${m1:-0}"; e1="${e1:-0}"
  miss=$(( m1 - m0 )); (( miss < 0 )) && miss=0
  ev=$(( ${e1:-0} - ${e0:-0} )); (( ev < 0 )) && ev=0
  local dh=$(( h1 - h0 )); (( dh < 0 )) && dh=0
  if (( dh + miss > 0 )); then hits=$(python3 -c "print(f'{$dh*100.0/($dh+$miss):.1f}%')"); else hits="-"; fi
  # derived RAM read on the miss path; expert size read from the pool lines when present
  aexp="$AVG_EXPERT_KIB"
  [[ -z "$aexp" ]] && aexp=$(command grep -oE 'expert=[0-9]+ KiB' "$log" | command grep -oE '[0-9]+' | awk '{s+=$1;n++} END{if(n)printf "%.0f",s/n}')
  if [[ -n "$aexp" && -n "$miss" && "$miss" != 0 ]]; then
    ramrd=$(python3 -c "print(f'{$miss*$aexp/1024/$elapsed:.0f}')" 2>/dev/null)
  else ramrd="-"; fi
  byp=$(command grep -c 'bypassing cache' "$log")
  # CACHE-NOT-ALLOCATED detection. Distinct from bypass: bypass means the cache exists
  # but a decode declined it; THIS means the pool was never created at all, so the arm
  # silently measures the no-cache path while every other column looks healthy.
  # Found live 2026-07-30: the engine's DEFAULT reserve (3072 MiB) left no budget at
  # 262K ctx, logging "CUDA0 has no cache budget after 3072 MiB reserve" -- a string the
  # bypass grep does not match. Both arms reported OK and ran ~12-22% slow.
  # Count DEVICES THAT ACTUALLY GOT A POOL, not the presence of the "enabled:" line.
  # A per-device budget failure still prints "enabled:" while one GPU runs with no
  # pool at all — measured 2026-07-30: CUDA0 got no budget, CUDA1 allocated 69 slots,
  # and "enabled:" printed regardless. Keying on that line reports OK for a run that
  # is half-uncached.
  local cache_enabled=0 reserve_seen="" pooled_devs=0
  pooled_devs=$(command grep -oE 'CUDA[0-9]+ pool\[' "$log" 2>/dev/null \
                | awk '{print $1}' | sort -u | command grep -c '' )
  (( pooled_devs > 0 )) && cache_enabled=1
  reserve_seen=$(command grep -oE 'reserve=[0-9]+ MiB|after [0-9]+ MiB reserve' "$log" \
                 | command grep -oE '[0-9]+' | tail -1)
  local nobudget; nobudget=$(command grep -c 'no cache budget' "$log")
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  vpeak=$(cat "$OUT_DIR/.vp-$arm-r$rep" 2>/dev/null); rm -f "$OUT_DIR/.vp-$arm-r$rep"

  local status=OK
  (( byp > 0 )) && status="INVALID_BYPASS"
  # A requested cache that never allocated invalidates the arm for any cache-dimension
  # conclusion, so it must never read OK.
  if (( HAS_MOE_CACHE )) && [[ "$cache" != 0 ]] && (( cache_enabled == 0 )); then
    status="CACHE_DISABLED"
    echo "  !! expert cache was REQUESTED (--moe-cache $cache) but NO DEVICE ALLOCATED A POOL."
    if (( nobudget > 0 )); then
      echo "     reason: no budget after ${reserve_seen:-?} MiB reserve. Lower the reserve"
      echo "     (RESERVE_MB=1536) or reduce ctx/pool so the pool fits."
    else
      echo "     no '[moe-cache] CUDAn pool[' lines in $log -- check the build and flags."
    fi
  elif (( HAS_MOE_CACHE )) && [[ "$cache" != 0 ]] && (( NGPU > 0 )) && (( pooled_devs < NGPU )); then
    # PARTIAL: some devices cached, others not. Worse than either extreme, because
    # the arm looks healthy and the uncached device silently drags the whole number.
    status="CACHE_PARTIAL"
    echo "  !! expert cache allocated on only $pooled_devs of $NGPU devices."
    echo "     The uncached device(s) run at no-cache speed and this arm's throughput"
    echo "     is a blend of two regimes — not comparable with a fully-cached arm."
    (( nobudget > 0 )) && echo "     reason: no budget after ${reserve_seen:-?} MiB reserve on at least one device."
  fi
  # zero throughput means every request failed — never let that land as OK (the
  # first live run reported agg=0.00 with status OK because the prompt builder
  # rejected an empty SHAPE and the driver exited)
  case "${agg:-}" in ''|'-'|0|0.0|0.00) status="NO_TOKENS" ;; esac
  [[ -n "${req_err:-}" && "${req_err:-0}" != 0 ]] && status="REQ_ERRORS"
  # Routed through emit_row so the column count comes from $HDR alone. This used to be
  # a hand-written printf with 32 specifiers for 33 arguments -- bash then RE-RAN the
  # format string for the surplus arg, so every arm wrote a short row PLUS a junk row
  # containing just the status. Never hand-count columns again.
  emit_row "$status" \
    "$arm" "$rep" "$shape" "$pcache" "${off:-0}" "${LAYERS_TOTAL:--}" "$n" "$ctx" "${ctx_slot:--}" "$dtag" "$nmax" "$mb" "$cache" \
    "${admit:--}" "${throttle:--}" "${agg:--}" "${strm:--}" "${ttft:--}" "${acc:--}" "${pool:--}" \
    "${hits:--}" "${miss:--}" "${ev:--}" "${rx:--}" "${tx:--}" "${smp:--}" "${memc:--}" "${cpu_pct:--}" \
    "${ramrd:--}" "${vpeak:--}" "${vram_idle:--}" "${vram_post:--}" "${leak:--}" "$byp" "${req_err:--}"
  echo "  agg=$agg strm=$strm ttft=${ttft}ms acc=${acc:--} | pcie rx=$rx tx=$tx MB/s sm=${smp}% memctl=${memc}% cpu=${cpu_pct}% ram_rd=${ramrd}MB/s | pool=${pool:--} hits=${hits:--} bypass=$byp req_err=${req_err:-?} [$status]"
  [[ "${req_err:-0}" != 0 ]] && echo "    first request error: ${req_err_msg:--}   (full context: $log)"
  sleep 5; echo
}

echo "offload-matrix"
echo "  model=$MODEL"
echo "  server=$LLAMA_SERVER  moe-cache=$( ((HAS_MOE_CACHE)) && echo yes || echo no)"
echo "  gpus=$NGPU  threads=$THREADS (nproc=$NPROC)  split='${SPLIT_ARGS:-none}'  kv=$KV_TYPE"
echo "  offload=[${SWEEP_OFFLOAD:-none}]/${LAYERS_TOTAL:-?}  N=[$SWEEP_N]  ctx=[$SWEEP_CTX]"
echo "  drafter=[$SWEEP_DRAFTER]  n-max=[$SWEEP_NMAX]  cache=[$SWEEP_CACHE]  admit=[$SWEEP_ADMIT]"
echo "  shape=[$SWEEP_SHAPE]  hetero=$HETERO   (workload shape flips the sign of spec results)"
echo "  preset=$PRESET  reps=$REPS rounds=$ROUNDS  out=$OUT_DIR"
# Pre-empt the silent killer: with the cache requested but no explicit reserve, the
# engine default (3072 MiB in this fork) can consume the whole budget at high ctx and
# the pool is never created -- the sweep then measures the no-cache path at every arm.
if (( HAS_MOE_CACHE )) && [[ -z "$RESERVE_MB" ]] && [[ "$SWEEP_CACHE" != 0 ]]; then
  echo "  note: RESERVE_MB unset -> engine default reserve. At high ctx this can leave NO"
  echo "        budget, so the expert cache is never allocated (arms report CACHE_DISABLED)."
  echo "        If that happens, re-run with RESERVE_MB=1536."
fi
echo "note: n-max=0 arms are the paired no-spec baselines; they run first in each group so"
echo "      every spec row has a contemporaneous control at identical settings."

# --- CONDITIONAL APPLICABILITY ------------------------------------------------
# A dimension is only measurable if its enabling precondition holds. Skipping it
# beats emitting empty columns that read like data.
_off_active=0
[[ -n "${SWEEP_OFFLOAD:-}" ]] && for _o in $SWEEP_OFFLOAD; do [ "$_o" != 0 ] && _off_active=1; done

# (a0) The dense guard below needs n_expert, which is only probed when LAYERS_TOTAL was
# empty. Setting LAYERS_TOTAL (which the probe's own error message suggests) skips the
# probe and DISARMS the guard -- so say so rather than let it look like a clean pass.
if (( _off_active )) && [[ -z "$N_EXPERT" ]]; then
  echo "note: expert count unknown (layer probe skipped), so the dense-model guard below"
  echo "      cannot run. If this model is DENSE, -ot matches nothing and every offload"
  echo "      column would be a silent no-op. Set N_EXPERT=<n> to arm the guard."
fi

# (a) DENSE model: the "_exps" offload regex matches nothing -> -ot is a silent no-op
if (( _off_active )) && [[ -n "$N_EXPERT" && "$N_EXPERT" == 0 ]]; then
  echo
  echo "⚠  This model reports n_expert = 0 (DENSE). The offload regex targets"
  echo "   '$EXPERT_TENSOR_RE', which matches no tensor here, so -ot would be a"
  echo "   SILENT no-op and every offload/cache column would be meaningless."
  echo "   Set EXPERT_TENSOR_RE to a tensor group this model actually has, or drop"
  echo "   SWEEP_OFFLOAD. Refusing to run a sweep that cannot measure its subject."
  exit 2
fi

# (b) no offload -> nothing for the expert cache to cache
if (( ! _off_active )) && { [[ "$SWEEP_CACHE" != "0" ]] || [[ "$SWEEP_ADMIT" != "default" ]]; }; then
  echo
  echo "ℹ  No layers are offloaded, so the expert cache has nothing to cache."
  echo "   Disabling the cache dimensions (they are downstream of offload):"
  echo "     SWEEP_CACHE=$SWEEP_CACHE -> 0 ; SWEEP_ADMIT=$SWEEP_ADMIT -> default"
  SWEEP_CACHE=0; SWEEP_ADMIT=default
fi

# (c) no GPU -> VRAM cache + PCIe telemetry are moot
if (( NGPU == 0 )); then
  echo
  echo "ℹ  No GPU detected: PCIe/SM/VRAM columns will be '-', and a VRAM expert"
  echo "   cache cannot apply. Offload sweeps are meaningless with everything already"
  echo "   on CPU — consider SWEEP_OFFLOAD='' and treating this as a CPU baseline."
fi

# --- conflict guard: this tool BOOTS its own servers per arm --------------------
if [[ -n "${_running_pid:-}" && "${PLAN:-0}" != 1 ]]; then
  echo
  echo "⚠  A llama-server is already running (pid $_running_pid)."
  echo "   Every arm boots its OWN server, so they would compete for VRAM and each"
  echo "   arm would either OOM or be measured under contention — the numbers would"
  echo "   be meaningless. Stop it first:   kill $_running_pid"
  echo "   (config was inherited from it above, so re-running after the kill keeps"
  echo "   the same settings. Set ALLOW_CONCURRENT_SERVER=1 to override.)"
  [[ "${ALLOW_CONCURRENT_SERVER:-0}" == 1 ]] || exit 2
fi

# --- ARM BUDGET ---------------------------------------------------------------
# The sweep is a cartesian product over NINE dimensions. A careless list turns
# minutes into days: three values on three dimensions is 27 arms, and every arm
# is a full server boot. This counts the planned work BEFORE anything boots and
# refuses to start past the budget, so a typo costs a message instead of a night.
#
# Counted in BOOTS (arms x REPS), because a boot is the unit of rig time.
MAX_BOOTS="${MAX_BOOTS:-50}"
_n() { local c=0 x; for x in $1; do c=$((c+1)); done; (( c )) || c=1; echo "$c"; }
_arms=1
for _d in "${SWEEP_OFFLOAD:-0}" "$SWEEP_CTX" "$SWEEP_N" "$SWEEP_CACHE" "$SWEEP_ADMIT" \
          "$SWEEP_SHAPE" "$SWEEP_PCACHE"; do _arms=$(( _arms * $(_n "$_d") )); done
# n-max: each 0 is one arm (no drafter); each non-zero fans out over the drafters
_spec=0; _base=0
for _d in $SWEEP_NMAX; do if [[ "$_d" == 0 ]]; then _base=$((_base+1)); else _spec=$((_spec+1)); fi; done
_arms=$(( _arms * ( _base + _spec * $(_n "$SWEEP_DRAFTER") ) ))
_boots=$(( _arms * REPS ))
echo "  -> planned: $_arms arms x REPS=$REPS = $_boots boots (budget MAX_BOOTS=$MAX_BOOTS)"
if (( _boots > MAX_BOOTS )); then
  echo
  echo "REFUSING: $_boots boots exceeds MAX_BOOTS=$MAX_BOOTS."
  echo "   Nine sweepable dimensions multiply, so this is usually a wider list than"
  echo "   intended rather than a deliberate long run. Either narrow the sweep (the"
  echo "   docs recommend staging it — run PLAN=1 to see the order), or raise the"
  echo "   budget deliberately:  MAX_BOOTS=$(( _boots > 200 ? _boots : 200 )) $0"
  echo "   At roughly 1-2 min per boot that is about $(( _boots * 90 / 60 )) minutes of rig time."
  exit 2
fi

# --- PLAN mode: print the dependency-ordered sequence and exit -----------------
if [[ "${PLAN:-0}" == 1 ]]; then
  cat <<PLANEOF

Recommended staged sequence (dimension dependency order).
Run each stage, read the table, pick the winner, then feed it into the next.

  STAGE 1 — offload layers (BASE)
    MODEL=$MODEL PRESET=offload OFFLOAD_SET="<candidates>" bash \$0
    -> pick the layer count with the best throughput. Note: more offload buys a
       bigger cache and a better hit rate, and can STILL be slower — a cache hit
       is not free, so minimise offloaded layers subject to fitting.

  STAGE 2 — cache size + admission (tunes the base)
    MODEL=$MODEL PRESET=cache SWEEP_OFFLOAD="<winner>" bash \$0
    -> watch pool/hits/evicts in --cache view, and check VRAM peak for headroom
       you left unused (an under-provisioned pool inflates every later cost).

  STAGE 3 — context per slot
    MODEL=$MODEL SWEEP_OFFLOAD="<winner>" SWEEP_CACHE="<winner>" \\
      SWEEP_ADMIT="<winner>" SWEEP_CTX="<candidates>" SWEEP_NMAX=0 bash \$0

  STAGE 4 — concurrency knee (spec OFF)
    MODEL=$MODEL PRESET=concurrency SWEEP_OFFLOAD="<winner>" \\
      SWEEP_CACHE="<winner>" SWEEP_ADMIT="<winner>" SWEEP_CTX="<winner>" bash \$0
    -> aggregate usually peaks then falls; TTFT often disqualifies a level
       before throughput does, so read both columns.

  STAGE 5 — drafter + depth (LAST, at the viable N only)
    MODEL=$MODEL PRESET=drafters SWEEP_N="<viable N>" SWEEP_OFFLOAD="<winner>" \\
      SWEEP_CACHE="<winner>" SWEEP_ADMIT="<winner>" SWEEP_CTX="<winner>" bash \$0
    then PRESET=depth to tune n-max on the winning drafter.

All stages append to the same TSV, so the final table shows every arm together.
PLANEOF
  exit 0
fi

# --- guards: every sweep var is OPTIONAL, but two combinations mislead silently ---
case " $SWEEP_NMAX " in
  *" 0 "*) ;;
  *) echo
     echo "⚠  SWEEP_NMAX does not include 0, so NO no-spec baseline will be measured."
     echo "   Spec rows will show 'no-spec agg = -' and carry no delta. Speculation gains"
     echo "   are only meaningful against a contemporaneous control (cross-run baselines"
     echo "   have been wrong by 5+ tok/s in practice). Add 0: SWEEP_NMAX=\"0 $SWEEP_NMAX\""
     ;;
esac

_n_arms=0
for _r in $(seq 1 "$REPS"); do for _o in ${SWEEP_OFFLOAD:-0}; do for _x in $SWEEP_CTX; do
  for _n in $SWEEP_N; do for _c in $SWEEP_CACHE; do for _a in $SWEEP_ADMIT; do
    for _sh in $SWEEP_SHAPE; do for _d in $SWEEP_NMAX; do
      if [ "$_d" = 0 ]; then _n_arms=$((_n_arms+1)); else
        for _dr in $SWEEP_DRAFTER; do _n_arms=$((_n_arms+1)); done; fi
    done; done; done; done; done; done; done; done
echo "  -> $_n_arms arm(s) to run (one server boot each)"
if [ "$_n_arms" -le 1 ]; then
  echo
  echo "ℹ  Only one arm: this is a SMOKE TEST of the harness, not a comparison."
  echo "   A useful first real sweep looks like:"
  echo "     MODEL=$MODEL SWEEP_OFFLOAD=\"<n>\" SWEEP_N=\"1 4\" SWEEP_NMAX=\"0 3\" REPS=2 \\"
  echo "       bash \$0"
  echo "   (SWEEP_OFFLOAD is the point of this tool — without it nothing is offloaded.)"
fi

# dependency-order violation: measuring downstream dims on an unsettled base
_spec_swept=0; case " $SWEEP_NMAX " in *" "[1-9]*) _spec_swept=1 ;; esac
_multi() { [ "$(echo $1 | wc -w)" -gt 1 ]; }
if (( _spec_swept )); then
  _msgs=""
  [[ -z "${SWEEP_OFFLOAD:-}" ]] && _msgs="$_msgs\n   - offload is UNSET (nothing offloaded) — settle it first (PRESET=offload)"
  _multi "$SWEEP_OFFLOAD" && _msgs="$_msgs\n   - offload is still being swept alongside spec: the base is moving under the"$'\n'"     drafter comparison, so a drafter win may not survive a layout change"
  { _multi "$SWEEP_CACHE" || _multi "$SWEEP_ADMIT"; } && _msgs="$_msgs\n   - cache/admission still being swept alongside spec — admission alone moved"$'\n'"     speculation by +5.4% on one rig; tune it first (PRESET=cache)"
  if [[ -n "$_msgs" ]]; then
    echo
    echo "⚠  DEPENDENCY ORDER: spec dimensions are downstream of the base config.$(printf '%b' "$_msgs")"
    echo "   Not fatal — the arms still run and the TSV records everything — but a"
    echo "   result measured on an unsettled base may need re-running. PLAN=1 shows the order."
  fi
fi
echo
for r in $(seq 1 "$REPS"); do
 for off in ${SWEEP_OFFLOAD:-0}; do
  for x in $SWEEP_CTX; do
   for n in $SWEEP_N; do
    for c in $SWEEP_CACHE; do
     for a in $SWEEP_ADMIT; do
      for sh in $SWEEP_SHAPE; do
       for pc in $SWEEP_PCACHE; do
        for d in $SWEEP_NMAX; do
         if (( d == 0 )); then run_arm "$off" "$n" 0 "$c" "$a" "$x" none "$r" "$sh" "$pc"
         else for dr in $SWEEP_DRAFTER; do run_arm "$off" "$n" "$d" "$c" "$a" "$x" "$dr" "$r" "$sh" "$pc"; done
         fi
        done
       done
      done
     done
    done
   done
  done
 done
done
echo "### OFFLOAD MATRIX DONE $(date +%T)"
echo
RENDER="$(dirname "$0")/offload-matrix-render.py"
[[ -f "$RENDER" ]] && python3 "$RENDER" "$TSV" || echo "(renderer not found next to script; run offload-matrix-render.py $TSV)"
