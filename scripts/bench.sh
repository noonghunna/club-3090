#!/usr/bin/env bash
#
# Canonical bench against the running vLLM service.
#   - Runs both the canonical narrative AND code prompts in one invocation.
#     This matches the README's narrative/code TPS pairing.
#   - 3 warmup + N measured runs per prompt (default 5 narrative + 5 code).
#   - per-run: wall time, TTFT (via streaming), completion tokens,
#     wall_TPS (= comp / wall), decode_TPS (= comp / (wall - TTFT))
#   - per-prompt summary: mean / std / CV for both TPS metrics + mean TTFT
#     + prompt-processing throughput (`PP tok/s`)
#   - shows MTP SpecDecoding metrics from docker logs at the end
#
# ⚠️⚠️ NEVER COMPARE A COLD BOOT AGAINST A HEAT-SOAKED ONE.
#   The 3 warmups above settle the ENGINE (cudagraphs, caches). They do NOT settle
#   the HARDWARE. A rig can take ~1 hour to reach thermal steady state, and a cold
#   card measures materially faster on decode — @juslex reported ~4% on 2x3090
#   (#922: rack air 29.6 -> 35.6 C over ~55 min, cores 80/82 -> 82/84 C). No
#   throttling was involved: power sat pinned at the cap the whole time, so the
#   CAP binds, not temperature — which is exactly why it does not show up as a
#   clock drop you would notice.
#
#   That is larger than most deltas worth reporting. Their first read put one cold
#   boot against three soaked ones and inflated a decode delta; it cost a day.
#   Shubham independently hit the same shape on ik_llama (whichever arm ran SECOND
#   was ~1% faster regardless of which arm it was).
#
#   ⇒ For any A/B: pre-warm both arms, or interleave A/B/A/B and compare like with
#     like. A single sequential pair — the thing this script makes easiest — is the
#     one design that cannot distinguish the patch from the position.
#   ⇒ Prefill is far more robust (CV <=0.1% within a boot, <=1% across boots);
#     decode is where this bites.
#
# Why two TPS metrics:
#   - wall_TPS  = "user-perceived speed" (includes prefill cost)
#   - decode_TPS = "model decode rate" (excludes prefill)
#   For long prompts the two can differ a lot. For short prompts they
#   converge. Reporting both keeps comparisons honest across configs.
#
# Why narrative + code:
#   MTP acceptance varies wildly by prompt structure. Code (repetitive,
#   token-predictable) accepts at ~80% per position; prose (semantically
#   rich) at ~50%. Reporting only one half is misleading. README claims
#   like "66 / 85 TPS" pair them; bench should too.
#
# Prereq: stack is running and reports "Application startup complete".
#
# Env vars:
#   URL                Endpoint. Default: registry-derived for qwen3.6-27b
#                      (curated DEFAULTS walk; currently :8020)
#   MODEL              Served model name. Default: auto-detected from
#                      /v1/models, else qwen3.6-27b
#   CONTAINER          Container for log scraping. Default: vllm-qwen36-27b
#   RUNS               Measured runs per prompt. Default: 5
#   WARMUPS            Warm-up runs (shared across both). Default: 3
#   PROMPT_NARR        Override narrative prompt
#   PROMPT_CODE        Override code prompt
#   MAX_TOKENS_NARR    Default: 1000
#   MAX_TOKENS_CODE    Default: 800
#   ONLY               Set to "narr" or "code" to skip the other. Default: both
#   QUIET              Set to 1 to skip per-run lines (just print summary)
#   PP                 Set to 1 to add the long-prompt PP fallback probe.
#                      llama.cpp containers enable this automatically.
#                      NOTE (#817): the `PP tok/s` line scraped from the ENGINE
#                      LOG is a ~10 s window average, and on the canonical ~15-tok
#                      prompts it is not a prefill rate at all — it rendered as
#                      `mean=2.00 CV=55.9%` and three community reports pasted it
#                      as data. It now prints only when it could be a rate
#                      (prompt >= 1000 tok, mean >= 50 tok/s, CV <= 100%) and is
#                      labelled `(engine log, windowed — indicative only)` when it
#                      does; otherwise it says n/a WITH THE REASON and points at
#                      the client-side `prefill tok/s`, which is trustworthy at
#                      CV 0.3-1.5%. Thresholds live in scripts/lib/capture.sh and
#                      are overridable via CAP_PP_MIN_PROMPT_TOKS /
#                      CAP_PP_MIN_TPS / CAP_PP_MAX_CV.
#   PP_FALLBACK_TOKENS Approximate filler-token target for PP=1. Default: 10000
#   PP_MAX_TOKENS      Completion cap for the PP fallback request. Default: 16
#   PREFILL_PROBE      Set to 0 to skip the canonical two-depth prefill/TTFT
#                      probe (catalog-baselines 2c). Default: 1. Each request
#                      uses a FRESH salted haystack — composes serve
#                      enable_prefix_caching, so identical prompts re-measure
#                      the CACHE HIT, not prefill. Depths exceeding the served
#                      context are skipped with a note.
#   PREFILL_DEPTHS     CSV of prompt-token depths. Default: 10000,90000 (the
#                      shallow/deep warm anchors; 90K sits inside the DeltaNet
#                      degradation regime and pairs with the verify-stress
#                      ladder's ~94K rung for anchor calibration).
#   PREFILL_RUNS       Measured runs per depth. Default: 3
#   ENABLE_THINKING    Set to 1 to send chat_template_kwargs.enable_thinking=true
#                      in bench requests. Default: 0.
#   FORCE_TOKENS       Force EXACTLY this many output tokens per run (sets
#                      max_tokens + min_tokens + ignore_eos), overriding
#                      MAX_TOKENS_NARR/CODE. Use to bench at a fixed / larger output
#                      size. Essential for DIFFUSION LMs: they self-terminate early
#                      (~1-2K words), so raising the cap alone won't lengthen the
#                      generation — you must force the length to measure sustained
#                      throughput. vLLM-oriented (ignore_eos/min_tokens). 0 = off
#                      (model decides length). Default: 0.
#   DECODE_GRANULARITY How this model emits tokens, which decides whether
#                      `decode_TPS` means anything at all (#809).
#                        auto   (default) classify from the measured runs
#                        token  autoregressive — decode_TPS is a decode rate
#                        canvas block diffusion (dLLM): the model denoises a whole
#                               canvas in parallel and the SSE endpoint emits ~one
#                               chunk per canvas, so TTFT == wall on any response
#                               that fits in one block and the decode window is
#                               zero-width. decode_TPS is NOT a decode rate for
#                               this class; the headline number is wall_TPS.
#   BENCH_DEGEN_WINDOW_FRAC
#                      A run whose decode window is below this fraction of its own
#                      wall time has no measurable decode rate and prints n/a
#                      rather than a number. Default: 0.05 — an AR decode window
#                      is ~the whole wall, so this sits ~20x away from any AR run
#                      and cannot fire on one.
#
# Capture knobs (the always-on capture layer — scripts/lib/capture.sh):
#   SERVER_LOG         Path to a BARE-METAL server log. CONTAINER=none runs used to
#                      report "log scrape unavailable" and every engine-side number
#                      had to be hand-extracted; point this at the file the server
#                      writes and the engine-side TPS, expert-cache and drafter
#                      captures all light up. Default: unset (docker log scrape,
#                      or nothing).
#   SERVER_PID         Pin the serving process for /proc inspection (argv fingerprint,
#                      moe-cache env, RSS, swap). Default: auto (`pgrep -x`).
#                      SERVER_PID=none disables /proc inspection entirely.
#   ENDPOINT           chat (default) drives /v1/chat/completions with the model's
#                      chat template applied — the historical bench behaviour, kept
#                      as the default so numbers stay comparable. ENDPOINT=completion
#                      drives raw /v1/completions with NO template, for base models
#                      and for isolating template overhead.
#   STREAM_CALIB       1 = run a ~3 s STREAM-triad RAM-bandwidth ceiling calibration
#                      on the idle box before the bench, so the derived miss-path
#                      demand can be stated as a fraction of what this host delivers.
#                      Default: 0 (opt-in — it costs 3 s and needs numpy).
#                      NOTE: the CONTENTION probe (co-running STREAM during decode)
#                      is deliberately NOT here — it perturbs the run by construction
#                      and belongs in a separate opt-in arm.
#   CAPTURE            0 = skip the whole capture layer (the pre-capture output
#                      shape, for a harness that parses it). Default: 1.
#   SHORT_EOS_FRAC     A run whose completion is below this fraction of max_tokens
#                      counts as a short-EOS run and is excluded from "n-usable".
#                      Chat-tuned models EOS early and silently degenerate a 5-run
#                      shape to n=1. Default: 0.25.
#
# Card rendering (docs/BENCH_CARD.md — default OFF; unset CARD leaves output
# byte-identical to a run without it):
#   CARD=snapshot      Render Template 1 (SNAPSHOT) as a fenced markdown block at
#                      the end of the run, filled from this run's own captures.
#                      Human judgement (verdict, "quiet box", config slug) stays an
#                      explicit <fill: …> placeholder — the renderer never guesses.
#   CARD=ab            Render Template 2 (A/B) against a previously saved run.
#     BASELINE=<path>    REQUIRED for CARD=ab: a saved bench.sh log (arm A). Parsing
#                        is strict — a log missing its fingerprint or summary blocks
#                        fails with "baseline unparseable: <why>" instead of half a
#                        delta. The run's own numbers are printed either way.
#     KNOB="<text>"      Names the thing under test. Fingerprint differences matching
#                        it are ATTRIBUTED to the knob; anything else marks the card
#                        CONFOUNDED at the top.
#     NOISE_BAND=<pct>   Override the band. Default: 2x the worst decode CV across
#                        both arms — and the card states which was used.
#     EXPECTATION="<t>"  Fills the pre-registered-expectation line (default "none").
#     BOOT_POLICY="<t>"  Fills the boot-policy line.
#   CARD_RECORD_OUT=<path>
#                      Debug seam: also write the in-process card record, so it can
#                      be diffed against the one parsed back out of the run's log.
#
#   Usage:
#     CARD=snapshot bash scripts/bench.sh | tee run-A.log
#     CARD=ab BASELINE=run-A.log KNOB=moe-cache bash scripts/bench.sh
#
# Presets (#832 — `--help` prints the same text):
#   --quick / QUICK=1  WARMUPS=1 RUNS=1 ONLY=narr PREFILL_PROBE=0 QUIET=1
#                      A preset over existing knobs; no new measurement code. For
#                      A/B sweeps where the canonical protocol dominates wall-clock
#                      and only a direction is needed. NOT CANONICAL: no CV, no code
#                      shape, no prefill anchor, not a BENCHMARKS.md row, and CARD
#                      rendering is refused. An explicitly-set env var wins over the
#                      preset and the banner says which.
#                      ⚠ It removes WITHIN-boot samples only. Any comparison that
#                      requires a reboot still needs >=2 BOOTS per arm — pool
#                      allocation on the CPU-offload MoE path swings ~20% between
#                      boots of a byte-identical config.
#
# Usage:
#   bash scripts/bench.sh
#   bash scripts/bench.sh --quick             # directional A/B arm (non-canonical)
#   ONLY=code bash scripts/bench.sh
#   PP=1 bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh
#   FORCE_TOKENS=4000 bash scripts/bench.sh   # fixed 4000-tok output (diffusion / sustained-TPS)
#   SERVER_LOG=/var/log/llama-server.log bash scripts/bench.sh   # bare metal
#   STREAM_CALIB=1 bash scripts/bench.sh      # + RAM-bandwidth ceiling

set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

# ===========================================================================
# CLI arguments (#832)
# ===========================================================================
# Everything else about this script is env-driven. `--quick` is a PRESET over
# those same knobs and adds NO measurement code — the point of the issue was
# discoverability, not capability: composing `WARMUPS=1 RUNS=1 ONLY=narr
# PREFILL_PROBE=0 QUIET=1` by hand requires reading ~40 lines of the env block
# and knowing that PREFILL_PROBE defaults to 1 and silently adds two long-prompt
# requests (10K and 90K), the 90K of which is plausibly the single largest term
# in a "quick" run.
#
# Parsed HERE, before the env defaults below, so "was this knob set explicitly?"
# is still answerable — and before preflight.sh is sourced, since a sourced
# script inherits our positional parameters.
QUICK="${QUICK:-0}"
# The documented expansion. This array IS the contract: --help prints it, the
# banner prints it, and the suite asserts the run behaves as exactly this set.
QUICK_PRESET=(WARMUPS=1 RUNS=1 ONLY=narr PREFILL_PROBE=0 QUIET=1)

bench_usage() {
  cat <<USAGE
Usage: bash scripts/bench.sh [--quick] [--help]

bench.sh is env-driven; see the header block of this file for the full knob list.

  --quick    ${QUICK_PRESET[*]}
             1 warmup + 1 measured run, narrative only, no prefill probe.
             Directional signal for A/B sweeps. Equivalent to QUICK=1.

             NOT CANONICAL. It gives up:
               * the 4 other measured runs, so there is NO CV and no way to tell
                 a real delta from noise within the arm;
               * the code shape (MTP/drafter acceptance differs sharply between
                 prose and code — one half is not the pair);
               * both prefill depths, so no prefill/TTFT anchor at all.
             Single-run TPS is NOT comparable across boots, and the output is
             not a BENCHMARKS.md row. CARD rendering is refused under --quick.

             --quick removes WITHIN-boot samples. It does NOT remove the need
             for >=2 BOOTS per arm on any comparison that requires a reboot
             (-ub, -b, KV type, offload layout). Pool allocation on the
             CPU-offload MoE path swings ~20% between boots of a byte-identical
             config (4949 vs 3942 slots) and throughput tracked it (-12%); that
             flipped "22L is the best no-spec config" from first to last once
             the arm was confirmed at 2 reps.

  -h, --help Print this and exit.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)   QUICK=1 ;;
    -h|--help) bench_usage; exit 0 ;;
    --)        shift; break ;;
    *) echo "ERROR: unknown argument '$1'. bench.sh is env-driven — see --help." >&2; exit 2 ;;
  esac
  shift
done

QUICK_APPLIED=(); QUICK_KEPT=()
if [[ "$QUICK" == "1" ]]; then
  for _kv in "${QUICK_PRESET[@]}"; do
    _k="${_kv%%=*}"; _v="${_kv#*=}"
    # An explicit env var wins over the preset, and the banner says so — silently
    # overriding what the operator typed is how a "quick" run turns into a
    # different protocol than either of them intended.
    if [[ -n "${!_k+x}" ]]; then
      QUICK_KEPT+=("${_k}=${!_k} (preset wanted ${_v})")
    else
      printf -v "$_k" '%s' "$_v"
      export "${_k?}"
      QUICK_APPLIED+=("$_kv")
    fi
  done
  if [[ -n "${CARD:-}" ]]; then
    echo "ERROR: --quick and CARD=${CARD} are incompatible." >&2
    echo "  A --quick run is n=1, so it has no CV — and the A/B card's noise band" >&2
    echo "  DEFAULTS to 2x the worst decode CV across both arms, which is then zero." >&2
    echo "  Every delta would render as significant. Render cards from canonical runs." >&2
    exit 2
  fi
fi

# Auto-detect running container + port (URL/CONTAINER env vars still win).
# See scripts/preflight.sh::preflight_autodetect_endpoint.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "${ROOT_DIR}/scripts/preflight.sh"
  preflight_autodetect_endpoint || true
fi
# The measurement heredoc shells out to the lib for the #817 PP plausibility gate,
# so it needs the path even when the capture layer itself is off (CAPTURE=0).
export BENCH_CAPTURE_LIB="${ROOT_DIR}/scripts/lib/capture.sh"

# --- per-rig #249 record: self-tee stdout so we can emit a corpus record at the
# end (c3's per-rig "TPS (rig)" column reads results/measurement-records/*.jsonl).
# BENCH_RECORD=0 skips it. resolve-serving maps the running container -> registry
# slug; a bare-metal / unmatched / --quick run just skips cleanly (no error).
BENCH_RECORD="${BENCH_RECORD:-1}"
_BENCH_REC_LOG=""
if [[ "${BENCH_RECORD}" == "1" ]] && command -v python3 >/dev/null 2>&1; then
  _BENCH_REC_LOG="$(mktemp 2>/dev/null || echo "/tmp/bench-rec.$$.log")"
  exec > >(tee -a "${_BENCH_REC_LOG}")
fi
# Interconnect state (#805). The SAME lib report.sh sources — the two surfaces
# must never disagree about whether P2P was engaged for a given run, and a
# second copy of the classifier here is exactly how they would drift.
if [[ -f "${ROOT_DIR}/scripts/lib/p2p-state.sh" ]]; then
  # shellcheck source=lib/p2p-state.sh
  source "${ROOT_DIR}/scripts/lib/p2p-state.sh"
fi
# Default endpoint follows the registry's curated DEFAULTS walk for qwen3.6-27b
# instead of a hand-maintained :8020/:8010 literal that drifts from the catalog.
# The trailing literal is only a last resort when the registry can't be consulted.
_DEFAULT_ENDPOINT_PORT=""
if [[ -f "${ROOT_DIR}/scripts/lib/registry-lookup.sh" ]]; then
  # shellcheck source=lib/registry-lookup.sh
  source "${ROOT_DIR}/scripts/lib/registry-lookup.sh"
  REGISTRY_LOOKUP_ROOT="${ROOT_DIR}"
  _DEFAULT_ENDPOINT_PORT="$(registry_lookup_default_port qwen3.6-27b 2>/dev/null || true)"
fi
URL="${URL:-http://localhost:${_DEFAULT_ENDPOINT_PORT:-8020}}"
# Resolve the served model from /v1/models when MODEL is unset (#372). The qwen
# literal below is only a last resort if detection no-ops (endpoint unreachable).
declare -F preflight_autodetect_model >/dev/null && preflight_autodetect_model
MODEL="${MODEL:-qwen3.6-27b}"
if [[ -z "${CONTAINER:-}" && -f "${ROOT_DIR}/scripts/lib/registry-lookup.sh" ]]; then
  # The old literal default 'vllm-qwen36-27b' matches NO registry container, so
  # the docker-inspect/exec consumers below silently no-op'd on an undetected
  # endpoint. Default to the MODEL's curated-default slug container instead
  # (qwen3.6-27b → vllm/minimal → vllm-qwen36-27b-minimal); the dead literal
  # stays only as a last-resort fallback when the registry can't be consulted.
  # shellcheck source=lib/registry-lookup.sh
  source "${ROOT_DIR}/scripts/lib/registry-lookup.sh"
  REGISTRY_LOOKUP_ROOT="${ROOT_DIR}"
  CONTAINER="$(registry_lookup_default_container "$MODEL" 2>/dev/null || true)"
fi
CONTAINER="${CONTAINER:-vllm-qwen36-27b}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS_NARR="${MAX_TOKENS_NARR:-1000}"
MAX_TOKENS_CODE="${MAX_TOKENS_CODE:-800}"
PROMPT_NARR="${PROMPT_NARR:-Write a detailed 800-word essay explaining transformer attention.}"
PROMPT_CODE="${PROMPT_CODE:-Write a Python implementation of quicksort with comments explaining each step.}"
ONLY="${ONLY:-both}"
QUIET="${QUIET:-0}"
PP="${PP:-0}"
PP_FALLBACK_TOKENS="${PP_FALLBACK_TOKENS:-10000}"
PP_MAX_TOKENS="${PP_MAX_TOKENS:-16}"
PREFILL_PROBE="${PREFILL_PROBE:-1}"
PREFILL_DEPTHS="${PREFILL_DEPTHS:-10000,90000}"
# Measured runs per prefill depth. SCALAR (all depths) or CSV aligned with
# PREFILL_DEPTHS. Default "3,1" — the depths differ ~13x in cost and the deep leg
# dominates total bench wall-clock:
#   10K  ~26 s/run,    measured CV 0.6%  -> 3 runs cost 78 s, buy a real CV
#   90K  ~5.5 min/run, measured CV 2.3%  -> 3 runs cost ~16.5 min, most of the bench
# A CSV shorter than the depth list reuses its LAST value; a malformed entry falls
# back to 1 rather than aborting a long run.
# ⚠️ n=1 gives no CV for that depth — state it when quoting a single-run number.
PREFILL_RUNS="${PREFILL_RUNS:-3,1}"
# The probe knobs reach the python heredoc via the environment (the argv tuple
# is full); export them here.
export PREFILL_PROBE PREFILL_DEPTHS PREFILL_RUNS
ENABLE_THINKING="${ENABLE_THINKING:-0}"
FORCE_TOKENS="${FORCE_TOKENS:-0}"
# --- decode-granularity knobs (#809) ----------------------------------------
# Precedence: an explicit env var (the operator meant it) > the SERVED MODEL's
# own declaration > auto-detection from the measured runs.
#
# The middle rung is the point: a model that declares `decode_granularity:
# canvas` in its profile YAML has that injected into the serving container by
# switch.sh/launch.sh, so the class travels WITH the server instead of living in
# an operator's memory. Auto-detection still works and still fires as the
# backstop — but it is a majority vote over a measured shape, and the model's
# own profile is the authority.
if [[ -z "${DECODE_GRANULARITY:-}" ]] && [[ "${CONTAINER:-}" != "none" ]] \
   && command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  _served_gran="$(docker exec "$CONTAINER" printenv DECODE_GRANULARITY 2>/dev/null | tr -d '\r' || true)"
  if [[ "$_served_gran" == "canvas" || "$_served_gran" == "token" ]]; then
    DECODE_GRANULARITY="$_served_gran"
    echo "[bench] decode granularity: ${DECODE_GRANULARITY} (declared by the served model's profile)"
  fi
fi
DECODE_GRANULARITY="${DECODE_GRANULARITY:-auto}"
case "$DECODE_GRANULARITY" in
  auto|token|canvas) : ;;
  *) echo "ERROR: DECODE_GRANULARITY must be 'auto', 'token' or 'canvas' (got '$DECODE_GRANULARITY')" >&2; exit 1 ;;
esac
BENCH_DEGEN_WINDOW_FRAC="${BENCH_DEGEN_WINDOW_FRAC:-0.05}"
export DECODE_GRANULARITY BENCH_DEGEN_WINDOW_FRAC
# --- capture layer knobs (see the header block) -----------------------------
CAPTURE="${CAPTURE:-1}"
ENDPOINT="${ENDPOINT:-chat}"
STREAM_CALIB="${STREAM_CALIB:-0}"
SHORT_EOS_FRAC="${SHORT_EOS_FRAC:-0.25}"
case "$ENDPOINT" in
  chat|completion) : ;;
  *) echo "ERROR: ENDPOINT must be 'chat' or 'completion' (got '$ENDPOINT')" >&2; exit 1 ;;
esac

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need curl
need python3

ENGINE_KIND="${ENGINE_KIND:-unknown}"
if [[ "$ENGINE_KIND" == "unknown" && "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  container_image="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER}" 2>/dev/null || true)"
  container_name="$(docker inspect --format '{{.Name}}' "${CONTAINER}" 2>/dev/null || true)"
  if [[ "${container_image} ${container_name}" == *"llama.cpp"* || "${container_image} ${container_name}" == *"llama-cpp"* ]]; then
    ENGINE_KIND="llamacpp"
  elif [[ "${container_image} ${container_name}" == *"vllm"* ]]; then
    ENGINE_KIND="vllm"
  fi
fi

PP_MODE="log"
if [[ "$PP" == "1" || "$ENGINE_KIND" == "llamacpp" ]]; then
  PP_MODE="fallback"
fi

# Which request field turns reasoning off is model-specific, and an unrecognised
# one is silently ignored — the model then reasons at full effort and the run
# measures reasoning-heavy generation while reporting itself as thinking-off.
# See preflight.sh::preflight_detect_thinking_control.
if declare -F preflight_detect_thinking_control >/dev/null; then
  preflight_detect_thinking_control
else
  THINK_CONTROL="enable_thinking"
  THINK_OFF_KW='{"enable_thinking": false}'; THINK_ON_KW='{"enable_thinking": true}'
  THINK_OFF_EFFORT=''; THINK_ON_EFFORT=''
fi
if [[ "$ENABLE_THINKING" == "1" ]]; then
  export BENCH_THINK_KW="$THINK_ON_KW"  BENCH_THINK_EFFORT="$THINK_ON_EFFORT"
  echo "[bench] thinking: enabled (${THINK_CONTROL} → ${THINK_ON_KW})" >&2
else
  export BENCH_THINK_KW="$THINK_OFF_KW" BENCH_THINK_EFFORT="$THINK_OFF_EFFORT"
fi

if [[ "${BENCH_MOCK:-0}" == "1" ]]; then
  if [[ "$PP_MODE" == "fallback" ]]; then
    cat <<'EOF'

========== PROMPT-PROCESSING (fallback target=10000 prompt tokens, max_tokens=16) ==========
=== measured (1) ===
  run-1      wall=  3.20s  ttft= 2500ms  prompt_toks=  9876  PP_tok/s=3950.40

=== summary [prompt-processing] (n=1) ===
  PP tok/s       mean=3950.40   std=  0.00   CV= 0.0%   min=3950.40   max=3950.40
  TTFT          mean=  2500ms  std=    0ms  min=2500ms  max=2500ms
EOF
  else
    cat <<'EOF'

========== NARRATIVE (prompt=61 chars, max_tokens=1000) ==========
=== measured (1) ===
  run-1      wall=  4.20s  ttft=   120ms  toks=1000  wall_TPS=238.10  decode_TPS=245.10

=== summary [narrative] (n=1) ===
  wall_TPS       mean= 238.10   std=  0.00   CV= 0.0%   min=238.10   max=238.10
  decode_TPS     mean= 245.10   std=  0.00   CV= 0.0%   min=245.10   max=245.10
  TTFT          mean=   120ms  std=    0ms  min=120ms  max=120ms
  PP tok/s       mean=2843.21   std=  0.00   CV= 0.0%   min=2843.21   max=2843.21
EOF
  fi
  exit 0
fi

if ! curl -sf "${URL}/v1/models" >/dev/null; then
  echo "ERROR: service not reachable at ${URL}/v1/models" >&2
  echo "  Start with: cd compose && docker compose up -d" >&2
  exit 1
fi

# --- --quick banner (#832) --------------------------------------------------
# Deliberately AFTER the BENCH_MOCK branch, so mock output stays byte-identical
# for the harnesses that parse it. On stdout, so it lands in a tee'd log next to
# the numbers it qualifies — a caveat that only exists in the operator's terminal
# is a caveat that does not survive being pasted into an issue.
quick_banner() {
  echo "=============================================================================="
  echo "⚠ QUICK MODE — NOT CANONICAL. Directional signal only."
  if (( ${#QUICK_APPLIED[@]} )); then
    echo "  preset applied : ${QUICK_APPLIED[*]}"
  else
    echo "  preset applied : (none — every knob was already set explicitly)"
  fi
  if (( ${#QUICK_KEPT[@]} )); then
    echo "  your env kept  : ${QUICK_KEPT[*]}"
  fi
  echo "  gives up       : 4 measured runs (so NO CV — a single-run delta cannot be"
  echo "                   told from noise), the code shape, both prefill depths."
  echo "  ⚠⚠ --quick removes WITHIN-boot samples. It does NOT remove the need for"
  echo "     >=2 BOOTS per arm on any reboot-requiring comparison (-ub, -b, KV type,"
  echo "     offload layout). Pool allocation on the CPU-offload MoE path swings ~20%"
  echo "     between boots of a byte-identical config (4949 vs 3942 slots) and"
  echo "     throughput tracked it (-12%) — that flipped a same-day \"22L is the best"
  echo "     no-spec config\" from first to last once the arm was confirmed at 2 reps."
  echo "  Single-run TPS is NOT comparable across boots. This output is NOT a"
  echo "  BENCHMARKS.md row — do not paste it into one."
  echo "=============================================================================="
}
if [[ "$QUICK" == "1" ]]; then quick_banner; fi

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
  echo "[bench] WARN: server appears to have reasoning enabled, but bench requests send thinking-off. Use ENABLE_THINKING=1 for reasoning-on TPS." >&2
fi
# server_reasoning_on() is blind twice over: it needs a docker CONTAINER (so a
# bare-metal llama-server is skipped entirely) and it looks for an explicit
# `--reasoning on` flag, which a model that reasons BY DEFAULT never needs. Both
# were true on Inkling-Small 2026-08-12 — the run measured reasoning-on
# generation, reported itself thinking-off, and this guard never fired. The
# detected control closes that gap: no switch means reasoning cannot be turned
# off at all, whatever the flags say.
if [[ "$ENABLE_THINKING" != "1" && "${THINK_CONTROL:-}" == none* ]]; then
  echo "[bench] WARN: no reasoning off-switch detected for this model — a thinking-off run is NOT possible and these numbers include reasoning tokens. Set VERIFY_THINK_OFF='{\"<key>\": <value>}' if the template uses a switch this harness doesn't know." >&2
fi

# ===========================================================================
# CAPTURE LAYER — everything from here to the end of the run is ADDITIVE.
#
# A bench harness's failure mode is a PLAUSIBLE WRONG NUMBER, not a crash. The
# captures below exist because each has already produced one: a decode scrape
# that printed 0.00 next to a healthy TTFT; a cumulative cache hit rate that
# said a bigger pool was worse; a drafter reporting 0.992 acceptance that fired
# on 5 of ~20 requests. Shared with the sweep via scripts/lib/capture.sh.
#
# CONDITIONALITY CONTRACT:
#   ALWAYS        KV cache type · config fingerprint (argv + /props + moe-cache
#                 env + --moe-cache cap) · system RAM + swap · PCIe
#   IF OFFLOAD    expert-cache pool census / hits / evictions / health counters
#   IF OFFLOAD
#     AND CACHE   marginal (not cumulative) hit rate · derived RAM-read demand
# Every capture degrades to ONE `capture: <thing> unavailable (<why>)` line.
# ===========================================================================
CAP_ENABLED=0
CAP_T0=""; CAP_T1=""
CAP_DMON_PID=""; CAP_VRAM_PID=""; CAP_LINK_PID=""
CAP_VRAM_IDLE=""; CAP_CPU0=""
CAP_ARGV=""; CAP_OFFLOAD=""; CAP_CACHE_WANTED=0
CAP_PCIE_PCT=""; CAP_RAM_DEMAND=""; CAP_RAM_WINDOW=""; CAP_RAM_SUMMARY=""
CAP_WORK=""
if [[ "$CAPTURE" == "1" && -f "${ROOT_DIR}/scripts/lib/capture.sh" ]]; then
  # shellcheck source=lib/capture.sh
  source "${ROOT_DIR}/scripts/lib/capture.sh"
  CAP_ENABLED=1
  CAP_WORK="$(mktemp -d)"
  trap '[[ -n "${CAP_WORK:-}" ]] && rm -rf "$CAP_WORK"' EXIT
  cap_init "$CAP_WORK" || true

  CAP_ARGV="$(cap_proc_argv || true)"

  echo ""
  echo "========== CONFIG FINGERPRINT =========="
  echo "  endpoint       : ${URL}  (mode=${ENDPOINT})"
  echo "  model          : ${MODEL}"
  if [[ -n "${CAP_PID:-}" ]]; then
    echo "  serving pid    : ${CAP_PID}$([[ -n "$CAP_ARGV" ]] || echo '  (argv unreadable — different uid?)')"
  else
    echo "  serving pid    : unknown (set SERVER_PID=<pid> for argv/RSS/swap capture)"
  fi
  echo "  log source     : ${CAP_LOG_SOURCE:-none}${SERVER_LOG:+ (${SERVER_LOG})}"
  _v="$(cap_props_get "default_generation_settings.n_ctx" || true)"
  _s="$(cap_props_get "total_slots" || true)"
  [[ -n "$_v" ]] && echo "  served ctx     : ${_v}${_s:+  (slots=${_s})}"
  _v="$(cap_props_get "model_ftype" || true)"
  [[ -n "$_v" ]] && echo "  weights ftype  : ${_v}"
  _v="$(cap_props_get "default_generation_settings.params.speculative.types" || true)"
  [[ -n "$_v" ]] && echo "  drafter        : ${_v}"
  # ALWAYS-ON: KV cache type. Offload or not, a bench without its KV type is a rumor.
  # The notice is emitted HERE, not inside cap_kv_type: that function returns its
  # value on stdout, so a notice printed there gets captured by this $(...) and
  # rendered inside the label — seen live as
  #   "KV cache type  : capture: KV cache type unavailable (…)".
  if _v="$(cap_kv_type "$CAP_ARGV")"; then
    echo "  KV cache type  : ${_v}"
  else
    cap_note_unavailable "KV cache type" "not in argv, /props or the boot log"
  fi
  # ALWAYS-ON: argv fingerprint — solves the fingerprint problem without operator input.
  if [[ -n "$CAP_ARGV" ]]; then
    _v="$(cap_argv_fingerprint "$CAP_ARGV" || true)"
    [[ -n "$_v" ]] && echo "  serving argv   : ${_v}"
  fi
  # ALWAYS-ON: the cache CONFIG, not just the resulting pool. The census
  # self-limits below the requested cap, so pool size alone can't identify an arm.
  _v="$(cap_moe_cache_config "$CAP_ARGV" || true)"
  [[ -n "$_v" ]] && echo "  moe-cache cfg  : ${_v}"
  case "$_v" in *"moe_cache_cap=<unset>"*) : ;; *) CAP_CACHE_WANTED=1 ;; esac
  # The GATE for every moe-cache capture below.
  CAP_OFFLOAD="$(cap_offload_detected "$CAP_ARGV" || true)"
  # One rendering, used for BOTH the printed line and the card record — the card
  # parser reads the printed line back, so a second phrasing here would make the
  # in-process record and the parsed one disagree (caught by the round-trip test).
  if [[ -n "$CAP_OFFLOAD" ]]; then
    CAP_OFFLOAD_DESC="DETECTED via ${CAP_OFFLOAD}"
  else
    CAP_OFFLOAD_DESC="not detected (moe-cache captures skipped)"
  fi
  echo "  CPU offload    : ${CAP_OFFLOAD_DESC}"

  # ALWAYS-ON (item 8): host RAM is the requirement launchers cannot detect, and
  # it is two lines from /proc. The swap leg is the free integrity item — pages of
  # the SERVING PROCESS in swap make every number in the run suspect.
  echo ""
  echo "========== SYSTEM RAM =========="
  if _v="$(cap_ram_status)"; then
    CAP_RAM_SUMMARY="$_v"
    echo "  ${_v}"
  else
    cap_note_unavailable "system RAM" "/proc/meminfo not readable"
  fi
  _v="$(cap_swap_verdict || true)"
  [[ -n "$_v" ]] && echo "  swap check     : ${_v}"

  if [[ "$STREAM_CALIB" == "1" ]]; then
    echo ""
    echo "========== RAM BANDWIDTH CEILING (STREAM triad, idle box) =========="
    _screc=0
    _sc="$(cap_stream_triad_gbps 3)" || _screc=$?
    if (( _screc == 3 )); then
      cap_note_unavailable "STREAM triad ceiling" \
        "numpy not installed — a pure-python loop would measure the interpreter, not RAM"
      _sc=""
    elif (( _screc != 0 )); then
      cap_note_unavailable "STREAM triad ceiling" "calibration failed (rc=${_screc})"
      _sc=""
    fi
    CAP_STREAM="${_sc%% *}"
    _scw="${_sc##* }"
    if [[ -n "$CAP_STREAM" ]]; then
      echo "  triad ceiling  : ${CAP_STREAM} GB/s  (${_scw} concurrent workers, ~3 s, measured on THIS host)"
      echo "                   Sustained STREAM-triad, not a spec sheet figure. A box that is"
      echo "                   already serving reports a LOWER ceiling than an idle one — which"
      echo "                   is the honest denominator for a run measured under that load."
    fi
  fi

  # --- start the windowed samplers ----------------------------------------
  CAP_DMON_PID="$(cap_dmon_start "$CAP_WORK/dmon")" \
    || { cap_note_unavailable "PCIe dmon" "nvidia-smi not in PATH"; CAP_DMON_PID=""; }
  CAP_LINK_PID="$(cap_pcie_link_sampler_start "$CAP_WORK/link" || true)"
  CAP_VRAM_PID="$(cap_vram_peak_start "$CAP_WORK/vpeak" || true)"
  CAP_VRAM_IDLE="$(cap_vram_used || true)"
  CAP_CPU0="$(cap_cpu_snap || true)"
  # Cache counters BEFORE the measured window, so the MARGINAL rate can be derived.
  if [[ -n "$CAP_OFFLOAD" ]]; then
    cap_refresh_log
    cap_moe_parse counters > "$CAP_WORK/counters0" 2>/dev/null || true
  fi
  # Line offset at the start of the measured window. Per-request scrapes
  # (acceptance, timings, bypass) count only what happened AFTER this point —
  # otherwise a log that already held a previous run's lines inflates every
  # count, and the drafter fire rate can exceed 100%.
  CAP_LOG_OFFSET=0
  [[ -n "$CAP_LOG" && -r "$CAP_LOG" ]] && CAP_LOG_OFFSET="$(command grep -c '' "$CAP_LOG" 2>/dev/null || echo 0)"
  export CAP_LINE_OFFSET="$CAP_LOG_OFFSET"
  # The prefill probe and PP fallback ask for ~16-token completions; a decode rate
  # measured over 16 tokens is start-up dominated. Exclude them from the decode
  # cross-check so it compares like with like.
  export CAP_MIN_DECODE_TOKENS="${CAP_MIN_DECODE_TOKENS:-32}"
  CAP_T0="$(date +%s)"
  export BENCH_PHASE_FILE="$CAP_WORK/phases" BENCH_SUMMARY_JSON="$CAP_WORK/summary.json"
  export BENCH_ENDPOINT="$ENDPOINT" BENCH_SHORT_EOS_FRAC="$SHORT_EOS_FRAC"
fi
export BENCH_ENDPOINT="${BENCH_ENDPOINT:-$ENDPOINT}"
export BENCH_SHORT_EOS_FRAC="${BENCH_SHORT_EOS_FRAC:-$SHORT_EOS_FRAC}"

python3 - "$URL" "$MODEL" "$WARMUPS" "$RUNS" "$QUIET" "$ONLY" \
            "$CONTAINER" "$PP_MODE" "$PP_FALLBACK_TOKENS" "$PP_MAX_TOKENS" \
            "$ENABLE_THINKING" "$FORCE_TOKENS" \
            "$PROMPT_NARR" "$MAX_TOKENS_NARR" \
            "$PROMPT_CODE" "$MAX_TOKENS_CODE" << 'PYEOF'
import json, os, random, re, shutil, string, subprocess, sys, time, urllib.request, statistics as s

(URL, MODEL, WARMUPS, RUNS, QUIET, ONLY,
 CONTAINER, PP_MODE, PP_FALLBACK_TOKENS, PP_MAX_TOKENS,
 ENABLE_THINKING, FORCE, PROMPT_NARR, MAX_NARR, PROMPT_CODE, MAX_CODE) = sys.argv[1:]
WARMUPS = int(WARMUPS); RUNS = int(RUNS); QUIET = int(QUIET) == 1
MAX_NARR = int(MAX_NARR); MAX_CODE = int(MAX_CODE)
PP_FALLBACK_TOKENS = int(PP_FALLBACK_TOKENS); PP_MAX_TOKENS = int(PP_MAX_TOKENS)
ENABLE_THINKING = ENABLE_THINKING == "1"
try:
    THINK_KW = json.loads(os.environ.get("BENCH_THINK_KW") or "")
except Exception:
    THINK_KW = {"enable_thinking": ENABLE_THINKING}
THINK_EFFORT = os.environ.get("BENCH_THINK_EFFORT") or ""
FORCE = int(FORCE)   # >0: force EXACTLY this many output tokens (max+min+ignore_eos)

# --- capture-layer plumbing (all optional; absent => historical behaviour) ---
# ENDPOINT=chat (default) keeps the historical /v1/chat/completions path with the
# model's template applied. ENDPOINT=completion drives raw /v1/completions with NO
# template — for base models, and for isolating template overhead on a chat model.
# --- canonical sampler (#962) -----------------------------------------------
# AGENTS.md defines the bench protocol as temperature=0.6, top_p=0.95, top_k=20.
# Historically only the first two were SENT, so top_k and min_p fell through to
# per-engine defaults — and those differ: llama.cpp applies top_k 40 / min_p 0.05,
# vLLM top_k off / min_p 0. A "canonical" protocol that resolves differently per
# engine cannot do the one job it exists for (cross-engine BENCHMARKS.md rows).
#
# All four are now sent explicitly. min_p is pinned to 0.0 — the protocol never
# mentions it, and 0.0 means "no min_p floor", which is the neutral reading.
# ⚠️ This CHANGES llama.cpp's effective sampling (top_k 40 -> 20, min_p 0.05 -> 0.0).
# Throughput is essentially sampler-insensitive at fixed max_tokens, so historical
# TPS rows stay comparable; anything QUALITY-shaped measured before this change was
# taken under different sampling and should not be diffed against post-change runs.
# Override per-run for sampler A/Bs (BENCH_TEMP / BENCH_TOP_P / BENCH_TOP_K /
# BENCH_MIN_P); the effective values are printed at startup so a run is never
# ambiguous about what it sampled with.
def _envf(name, default):
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return float(default)

SAMPLER = {
    "temperature": _envf("BENCH_TEMP", 0.6),
    "top_p": _envf("BENCH_TOP_P", 0.95),
    "top_k": int(_envf("BENCH_TOP_K", 20)),
    "min_p": _envf("BENCH_MIN_P", 0.0),
}

ENDPOINT_MODE = os.environ.get("BENCH_ENDPOINT", "chat")
PHASE_FILE = os.environ.get("BENCH_PHASE_FILE", "")
SUMMARY_JSON = os.environ.get("BENCH_SUMMARY_JSON", "")
try:
    SHORT_EOS_FRAC = float(os.environ.get("BENCH_SHORT_EOS_FRAC", "0.25"))
except ValueError:
    SHORT_EOS_FRAC = 0.25
SUMMARY = {"shapes": {}, "endpoint": ENDPOINT_MODE, "sampler": dict(SAMPLER)}


def progress(msg):
    """Item 6: one line per run, FLUSHED, on stderr.

    Section summaries buffer to section end (correct — a partial table is worse
    than none), but a prefill probe at depth can then sit silent for 10+ minutes
    with no way to tell it apart from a hang. stderr keeps stdout's shape intact
    for anything parsing it."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def announce_sampler():
    """Print the EFFECTIVE sampler once (#962).

    The bug this closes was invisible precisely because nothing echoed what the
    bench sampled with — the only way to find out was to read the engine's own
    boot log. A run should never be ambiguous about that after the fact.
    """
    progress("[bench] sampler: " + "  ".join(
        f"{k}={v}" for k, v in (
            ("temperature", SAMPLER["temperature"]), ("top_p", SAMPLER["top_p"]),
            ("top_k", SAMPLER["top_k"]), ("min_p", SAMPLER["min_p"]))))


def _log_lines():
    """Current line count of the server log, or 0 when there isn't one.

    This is the seam that makes PHASE-ATTRIBUTED counters possible: the engine's
    cumulative moe-cache counters carry no timestamps, but reading them truncated
    at two different line positions gives the delta across exactly that span."""
    path = os.environ.get("SERVER_LOG", "")
    if not path:
        return 0
    try:
        with open(path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def phase_start():
    return time.time(), _log_lines()


def mark_phase(label, start):
    """Record a measured window so PCIe samples and cache counters can be sliced
    by PHASE.

    Decode and prefill differ by ~13x in mean PCIe traffic on an offloaded MoE
    (measured), so a whole-run mean describes neither — and a miss count averaged
    over a run that was ~40% prefill understates decode-window RAM demand."""
    if not PHASE_FILE:
        return
    t0, l0 = start
    try:
        with open(PHASE_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{label} {t0:.0f} {time.time():.0f} {l0} {_log_lines()}\n")
    except OSError:
        pass


_TOKEN_RATIO = [None]


def token_ratio():
    """Item 7: measured tokens-per-word for THIS tokenizer.

    The haystack builders sized filler by WORD count, which is a per-tokenizer
    guess: a "40K" warm-up tokenized to 55K under DeepSeek's tokenizer (+37%).
    Prefer the server's own /tokenize; fall back to one tiny max_tokens=1 request
    and read usage.prompt_tokens; fall back to the word heuristic (ratio 1.0),
    which is exactly the historical behaviour."""
    if _TOKEN_RATIO[0] is not None:
        return _TOKEN_RATIO[0]
    sample = ("club3090 prompt processing calibration filler with stable token shape. "
              "This sentence is intentionally plain so tokenizer variance stays modest. ") * 20
    words = max(len(sample.split()), 1)
    ratio = None
    try:
        body = json.dumps({"content": sample}).encode()
        req = urllib.request.Request(f"{URL}/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            obj = json.loads(r.read().decode("utf-8", errors="ignore"))
        toks = obj.get("tokens", obj if isinstance(obj, list) else [])
        if isinstance(toks, list) and toks:
            ratio = len(toks) / words
    except Exception:
        ratio = None
    if ratio is None:
        try:
            _w, _t, _k, ptoks = run_once(sample, 1)
            if ptoks > words:            # a real count, not the word-split fallback
                ratio = ptoks / words
        except Exception:
            ratio = None
    if ratio is None or ratio <= 0:
        progress("[bench] capture: token-ratio probe unavailable (no /tokenize, probe failed) "
                 "— haystacks fall back to the word-count heuristic")
        ratio = 1.0
    else:
        progress(f"[bench] capture: tokenizer ratio {ratio:.2f} tok/word "
                 f"(haystacks sized by measured tokens, not words)")
    _TOKEN_RATIO[0] = ratio
    return ratio


def run_once(prompt, max_tokens):
    # FORCE>0: force EXACTLY FORCE output tokens (overrides the per-prompt cap) so the
    # model can't self-terminate early — required to measure sustained throughput at a
    # chosen output size on diffusion LMs (which stop ~1-2K words otherwise).
    mt = FORCE if FORCE > 0 else max_tokens
    req_body = {
        "model": MODEL,
        "max_tokens": mt,
        # Canonical sampler, sent in FULL so the effective values do not depend on
        # engine defaults (#962). See the SAMPLER block above.
        **SAMPLER,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ENDPOINT_MODE == "completion":
        # RAW completion: no chat template, no thinking kwargs (they are a chat
        # concept). Deliberately not the default — switching endpoints changes
        # the prompt the model actually sees, so numbers are NOT comparable
        # across modes.
        req_body["prompt"] = prompt
        path = "/v1/completions"
    else:
        req_body["messages"] = [{"role": "user", "content": prompt}]
        # The reasoning switch is resolved by the shell (preflight.sh
        # ::preflight_detect_thinking_control) and handed over as final JSON,
        # because WHICH key works is model-specific and an unrecognised one is
        # silently ignored — which would make a "thinking off" run silently
        # measure reasoning-heavy generation.
        req_body["chat_template_kwargs"] = THINK_KW
        if THINK_EFFORT:
            # The OpenAI-standard top-level parameter, sent alongside for
            # engines that implement it (llama.cpp currently does not forward it
            # into the template — see the preflight.sh block header).
            req_body["reasoning_effort"] = THINK_EFFORT
        path = "/v1/chat/completions"
    if FORCE > 0:
        req_body["min_tokens"] = FORCE
        req_body["ignore_eos"] = True
    body = json.dumps(req_body).encode()
    req = urllib.request.Request(f"{URL}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                # /v1/completions puts the text at choices[0].text, not in a delta
                content = (delta.get("content") or delta.get("reasoning_content")
                           or choices[0].get("text"))
                if content and ttft is None:
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    t_end = time.time()
    wall = t_end - t_send
    if ttft is None:
        ttft = wall
    if not prompt_tokens:
        prompt_tokens = max(1, len(prompt.split()))
    return wall, ttft, completion_tokens, prompt_tokens

# --- decode-window guard (#809) ---------------------------------------------
# `decode_TPS = toks / (wall - TTFT)` assumes token-by-token AR streaming. A
# canvas-granularity (block-diffusion) model denoises a whole canvas in parallel
# and the SSE endpoint emits roughly ONE chunk per completed canvas: any response
# that fits in a single canvas arrives as one chunk, TTFT == wall, and the decode
# window is zero-width. The old `max(wall - ttft, 1e-6)` then divided by an
# epsilon and printed `decode_TPS=690000000.00` as though it were a throughput
# (#822, real output from a 2x5090 diffusiongemma run).
#
# The guard is a RATIO, not an absolute threshold, and that is what makes it safe
# for AR models: a real AR decode window IS essentially the whole wall (TTFT is a
# few percent of it), so 5% sits ~20x away from any autoregressive run and cannot
# fire on one. When it does fire there is no decode rate to report — not a small
# one, not a large one — so the column prints n/a and the run is excluded from the
# decode statistics. wall_TPS is the honest number for such a run.
try:
    DEGEN_WINDOW_FRAC = float(os.environ.get("BENCH_DEGEN_WINDOW_FRAC", "0.05"))
except ValueError:
    DEGEN_WINDOW_FRAC = 0.05
GRANULARITY = (os.environ.get("DECODE_GRANULARITY", "auto") or "auto").strip().lower()
if GRANULARITY not in ("auto", "token", "canvas"):
    GRANULARITY = "auto"


def decode_window(wall, ttft):
    """(decode_seconds, degenerate?). Degenerate means UNMEASURABLE, not slow."""
    dt = wall - ttft
    return dt, (wall <= 0 or dt <= 0 or dt < DEGEN_WINDOW_FRAC * wall)


def classify_granularity(n_runs, degen_runs):
    """(granularity, why). Declared wins; otherwise classify from the runs."""
    if GRANULARITY in ("token", "canvas"):
        return GRANULARITY, f"declared: DECODE_GRANULARITY={GRANULARITY}"
    # A single degenerate run can be a fluke (a scheduler hiccup, an EOS on the
    # first chunk). A MAJORITY of them across a shape is a property of the model.
    if n_runs >= 2 and degen_runs * 2 >= n_runs:
        return "canvas", (f"auto-detected: {degen_runs}/{n_runs} runs emitted their whole "
                          f"completion inside one block, TTFT == wall")
    return "token", ""


def fmt(label, wall, ttft, toks):
    dt, degen = decode_window(wall, ttft)
    wtps = toks / wall if wall > 0 else 0
    if degen:
        # Never a number here. The historical divide-by-1e-6 is the defect (#809).
        dtps = None
        pct = (dt / wall * 100) if wall > 0 else 0.0
        dcol = (f"decode_TPS={'n/a':>6s}  (decode window {max(dt, 0.0):.3f}s = {pct:.1f}% of wall "
                f"— single-block emission, no measurable decode rate; quote wall_TPS)")
    else:
        dtps = toks / dt
        dcol = f"decode_TPS={dtps:6.2f}"
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  toks={toks:>4d}  wall_TPS={wtps:6.2f}  {dcol}"
    return wtps, dtps, ttft, line

# --- prompt-processing plausibility gate (#817) -----------------------------
# The policy and its three thresholds live in ONE place — capture.sh's
# cap_pp_plausible. Shelling out to it is deliberate: bench.sh has two print
# sites for this scrape, and a second copy of the rule in python is exactly how
# two callers of a shared scrape drift apart.
CAPTURE_LIB = os.environ.get("BENCH_CAPTURE_LIB", "")


def pp_pointer():
    if os.environ.get("PREFILL_PROBE", "1") == "1":
        return ("the trustworthy prefill number is the client-side `prefill tok/s` "
                "summary (prompt_tokens/TTFT, cache-busted)")
    return ("run with PREFILL_PROBE=1 (the default) or PP=1 for a client-side "
            "prefill measurement")


def pp_verdict(vals, prompt_tokens):
    """(plausible?, reason) for a scraped PP sample. Fails CLOSED."""
    if not vals:
        return False, "no values scraped"
    m = s.mean(vals)
    cv = (s.stdev(vals) / m * 100) if len(vals) > 1 and m > 0 else 0.0
    if not CAPTURE_LIB or not os.path.exists(CAPTURE_LIB):
        # Without the gate we cannot tell a measurement from an artifact, and the
        # artifact is the one that gets pasted into an issue. Refuse to print.
        return False, "plausibility gate unavailable (scripts/lib/capture.sh not found)"
    try:
        p = subprocess.run(
            ["bash", "-c", 'source "$1"; cap_pp_plausible "$2" "$3" "$4"', "_",
             CAPTURE_LIB, f"{m:.4f}", f"{cv:.4f}", str(int(prompt_tokens))],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=20, check=False)
    except Exception as e:
        return False, f"plausibility gate failed to run ({e})"
    return (p.returncode == 0), p.stdout.strip()


def fmt_pp(label, wall, ttft, prompt_tokens):
    pp = prompt_tokens / max(ttft, 1e-6)
    line = f"  {label:<10s} wall={wall:6.2f}s  ttft={ttft*1000:6.0f}ms  prompt_toks={prompt_tokens:>6d}  PP_tok/s={pp:7.2f}"
    return pp, ttft, line

def stats(name, xs, unit=""):
    m = s.mean(xs)
    sd = s.stdev(xs) if len(xs) > 1 else 0
    cv = (sd / m * 100) if m > 0 else 0
    return f"  {name:<14s} mean={m:7.2f}{unit}   std={sd:6.2f}   CV={cv:4.1f}%   min={min(xs):.2f}   max={max(xs):.2f}"

def scrape_server_log_prefill(n):
    """Item 4: the BARE-METAL half of the engine-side scrape.

    `CONTAINER=none` runs report "log scrape unavailable" and every engine-side
    number then has to be hand-extracted from a file that was sitting there the
    whole time. With SERVER_LOG set, read llama.cpp's own `prompt eval time`
    lines instead."""
    path = os.environ.get("SERVER_LOG", "")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    vals = [float(m.group(1)) for m in
            re.finditer(r"prompt eval time.*?([0-9]+(?:\.[0-9]+)?)\s+tokens per second", text)]
    return vals[-max(n, 1):]


def scrape_prompt_throughput(container, n):
    vals = scrape_server_log_prefill(n)
    if vals:
        return vals
    if not container or container == "none" or shutil.which("docker") is None:
        return []
    try:
        proc = subprocess.run(
            ["docker", "logs", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:
        return []
    vals = [
        float(m.group(1))
        for m in re.finditer(r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s", proc.stdout)
    ]
    return vals[-max(n, 1):]

def run_set(label, prompt, max_tokens):
    print(f"\n========== {label.upper()} (prompt={len(prompt)} chars, max_tokens={max_tokens}) ==========")
    print(f"=== warmups ({WARMUPS}) ===")
    for i in range(WARMUPS):
        try:
            w, t, k, _ = run_once(prompt, max_tokens)
            _, _, _, line = fmt(f"warm-{i+1}", w, t, k)
            if not QUIET:
                print(line)
        except Exception as e:
            print(f"  warm-{i+1}  FAIL: {e}")
    print(f"\n=== measured ({RUNS}) ===")
    walls, decodes, ttfts, toks_seen = [], [], [], []
    ptoks_seen = []          # prompt size per run — the #817 plausibility gate needs it
    degen_runs = 0           # runs with a zero-width decode window (#809)
    errors = 0
    mt = FORCE if FORCE > 0 else max_tokens
    phase_t0 = phase_start()
    for i in range(RUNS):
        try:
            w, t, k, ptk = run_once(prompt, max_tokens)
            wtps, dtps, ttft, line = fmt(f"run-{i+1}", w, t, k)
            if not QUIET:
                print(line)
            rate = f"{dtps:.2f} decode TPS" if dtps is not None else f"{wtps:.2f} wall TPS (no decode window)"
            progress(f"[bench] {label} run {i+1}/{RUNS}: {k} tok in {w:.1f}s "
                     f"({rate}, ttft {ttft*1000:.0f}ms)")
            walls.append(wtps); ttfts.append(ttft)
            # A degenerate run contributes NO decode value. Averaging in a
            # divide-by-epsilon is how `decode_TPS mean=2728143.37 CV=223.6%`
            # reached three community reports (#809).
            if dtps is None:
                degen_runs += 1
            else:
                decodes.append(dtps)
            toks_seen.append(k); ptoks_seen.append(ptk)
        except Exception as e:
            errors += 1
            print(f"  run-{i+1}  FAIL: {e}")
            progress(f"[bench] {label} run {i+1}/{RUNS}: FAIL: {e}")
    mark_phase("decode", phase_t0)
    # Item 2: a chat-tuned model answers briefly and EOSes. The per-shape stats
    # then silently describe whichever one or two runs happened to run long — a
    # 5-run shape degenerates to n=1 while still printing "n=5". Count the runs
    # that are long enough to be worth a decode rate and say so.
    floor = max(1, int(mt * SHORT_EOS_FRAC))
    usable = [k for k in toks_seen if k >= floor]
    def _cv(xs):
        if len(xs) < 2:
            return 0.0
        m = s.mean(xs)
        return (s.stdev(xs) / m * 100) if m > 0 else 0.0

    gran, gran_why = classify_granularity(len(walls), degen_runs)
    # Derived == "every run had a zero-width decode window", so the only rate
    # available is completion/wall, which IS wall_TPS by construction.
    derived = bool(walls) and not decodes
    SUMMARY["shapes"][label] = {
        "n": len(walls), "n_usable": len(usable), "errors": errors,
        "max_tokens": mt, "short_eos_floor": floor,
        "tokens": toks_seen,
        "wall_tps_mean": (s.mean(walls) if walls else 0.0),
        "wall_tps_cv": _cv(walls),
        "decode_tps_mean": (s.mean(decodes) if decodes else (s.mean(walls) if walls else 0.0)),
        "decode_tps_cv": (_cv(decodes) if decodes else _cv(walls)),
        "decode_degenerate": degen_runs,
        "decode_derived": derived,
        "decode_granularity": gran,
        "ttft_mean_ms": (s.mean(ttfts) * 1000 if ttfts else 0.0),
        "ttft_std_ms": (s.stdev(ttfts) * 1000 if len(ttfts) > 1 else 0.0),
    }
    if walls:
        print(f"\n=== summary [{label}] (n={len(walls)}) ===")
        print(stats("wall_TPS",   walls))
        if decodes:
            print(stats("decode_TPS", decodes))
        else:
            # #809 proposal item 1: DERIVE, don't zero — and never present the
            # derived value bare. Printing nothing here would also strand the
            # measurement-record producer, which keys off a `decode_TPS mean=`
            # line and refuses to write a record without one.
            print(stats("decode_TPS", walls))
            print(f"  ⚠ decode_TPS above is DERIVED, NOT MEASURED: all {len(walls)} run(s) had a "
                  f"zero-width decode")
            print("    window, so it is completion/wall = wall_TPS by construction. It INCLUDES "
                  "prefill and")
            print("    must never be quoted as a decode rate (#809).")
        if degen_runs and decodes:
            print(f"  decode-window  unmeasurable on {degen_runs}/{len(walls)} run(s) "
                  f"(decode window < {DEGEN_WINDOW_FRAC:.0%} of wall); "
                  f"decode_TPS above is over the {len(decodes)} measured run(s)")
        if gran == "canvas":
            print(f"  ⚠ CANVAS GRANULARITY ({gran_why})")
            print("    dLLM: block emission, decode window undefined. decode_TPS is NOT a decode "
                  "rate for")
            print("    this model class — the HEADLINE number is wall_TPS (#809).")
            if GRANULARITY == "auto":
                print("    Set DECODE_GRANULARITY=token if this model really is autoregressive.")
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std={s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0:5.0f}ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")
        print(f"  n-usable      {len(usable)}/{len(walls)}  (runs with >= {floor} tokens, "
              f"{SHORT_EOS_FRAC:.0%} of max_tokens={mt})")
        if len(usable) < len(walls):
            short = len(walls) - len(usable)
            print(f"  ⚠ WARN: {short}/{len(walls)} run(s) EOSed well below max_tokens "
                  f"(tokens={toks_seen}). Decode stats for this shape rest on "
                  f"{len(usable)} run(s), not {len(walls)} — treat the CV as unreliable.")
            print("           Fix: FORCE_TOKENS=<n> to force a fixed output length, or use a "
                  "prompt this model answers at length.")
            progress(f"[bench] WARN {label}: only {len(usable)}/{len(walls)} runs usable "
                     f"(short EOS) — see the summary block")
        if PP_MODE == "log":
            pp_vals = scrape_prompt_throughput(CONTAINER, len(walls))
            # #817: the canonical narrative/code prompts are ~15 tokens, and the
            # engine's windowed prompt-throughput average over them is not a
            # prefill rate. Print it only when it could be one.
            pp_prompt = int(s.mean(ptoks_seen)) if ptoks_seen else 0
            pp_ok, pp_why = pp_verdict(pp_vals, pp_prompt)
            if pp_vals and pp_ok:
                print(stats("PP tok/s", pp_vals) + "   (engine log, windowed — indicative only)")
            elif pp_vals:
                print(f"  PP tok/s       n/a (engine-log scrape suppressed: {pp_why})")
                print(f"                 {pp_pointer()}")
            else:
                print("  PP tok/s       n/a (engine log scrape unavailable; use PP=1 for the "
                      "long-prompt fallback, or SERVER_LOG=<path> on bare metal)")
        else:
            print("  PP tok/s       n/a (long-prompt fallback below)")

def long_prompt(target_tokens):
    filler = (
        "club3090 prompt processing calibration filler with stable token shape. "
        "This sentence is intentionally plain so tokenizer variance stays modest. "
    )
    # Item 7: size by MEASURED tokens, not words. The word heuristic is a
    # per-tokenizer guess and a "40K" haystack tokenized to 55K on DeepSeek.
    words_per_chunk = max(len(filler.split()), 1)
    chunks = max(1, int(target_tokens / (words_per_chunk * token_ratio())))
    return (
        "Read the following calibration text. Reply with one concise sentence summarizing its purpose.\n\n"
        + filler * chunks
    )

def run_pp_fallback():
    prompt = long_prompt(PP_FALLBACK_TOKENS)
    print(
        f"\n========== PROMPT-PROCESSING "
        f"(fallback target={PP_FALLBACK_TOKENS} prompt tokens, max_tokens={PP_MAX_TOKENS}) =========="
    )
    print("=== measured (1) ===")
    pp_vals, ttfts = [], []
    phase_t0 = phase_start()
    try:
        w, t, _k, prompt_tokens = run_once(prompt, PP_MAX_TOKENS)
        pp, ttft, line = fmt_pp("run-1", w, t, prompt_tokens)
        print(line)
        progress(f"[bench] prompt-processing run 1/1: {prompt_tokens} prompt tok, "
                 f"{pp:.1f} PP tok/s, ttft {ttft*1000:.0f}ms")
        pp_vals.append(pp); ttfts.append(ttft)
    except Exception as e:
        print(f"  run-1      FAIL: {e}")
        progress(f"[bench] prompt-processing run 1/1: FAIL: {e}")
    mark_phase("prefill", phase_t0)
    if pp_vals:
        print("\n=== summary [prompt-processing] (n=1) ===")
        print(stats("PP tok/s", pp_vals))
        print(f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  std=    0ms  min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms")

def prefill_prompt(target_tokens, salt):
    """Cache-busted long prompt (catalog-baselines 2c).

    Composes serve ``enable_prefix_caching=True`` — an identical prompt
    re-measures the PREFIX-CACHE HIT, not prefill.  vLLM's prefix cache is
    block-chained, so a unique FIRST line breaks the whole chain; the salt is
    also woven through the filler as belt-and-braces against block-aligned
    partial hits."""
    filler = (
        f"calibration {salt} segment with stable token shape for the prefill probe. "
        "This sentence is intentionally plain so tokenizer variance stays modest. "
    )
    # Item 7: divide by the MEASURED tok/word ratio so the warm-up lands on the
    # requested depth. The measured runs recalibrate from the warm-up's actual
    # count regardless — this stops the WARM-UP from overshooting (the "40K" that
    # tokenized to 55K), which is the run nothing corrects.
    words_per_chunk = max(len(filler.split()), 1)
    chunks = max(1, int(target_tokens / (words_per_chunk * token_ratio())))
    return (
        f"[probe {salt}] Read the following calibration text. Reply with the single word DONE.\n\n"
        + filler * chunks
    )

def run_prefill_probe():
    """Canonical two-depth prefill/TTFT probe (catalog-baselines 2c): warm
    engine, FRESH salted haystack per request, prefill tok/s = prompt_tokens/TTFT.
    The deep anchor pairs with the verify-stress ladder's nearest rung for
    anchor calibration (agreement certifies the ladder's whole depth curve).
    A depth the served context can't hold is SKIPPED with a note."""
    depths = [int(x) for x in os.environ.get("PREFILL_DEPTHS", "10000,90000").split(",") if x.strip()]
    _parts = [x.strip() for x in os.environ.get("PREFILL_RUNS", "3,1").split(",") if x.strip()]

    def _runs_for(idx):
        """Runs for depth #idx. CSV shorter than depths reuses the last value;
        a malformed entry degrades to 1 instead of killing a long run."""
        if not _parts:
            return 1
        tok = _parts[idx] if idx < len(_parts) else _parts[-1]
        try:
            return max(1, int(tok))
        except ValueError:
            return 1

    def salt():
        return "".join(random.choices(string.ascii_lowercase, k=8))

    for _di, target in enumerate(depths):
        n = _runs_for(_di)
        label = f"prefill-{target // 1000}k"
        print(
            f"\n========== {label.upper()} (target={target} prompt tokens, "
            f"max_tokens={PP_MAX_TOKENS}, cache-busted: fresh haystack per run) =========="
        )
        print("=== warmups (1) ===")
        # The warmup doubles as the TOKEN CALIBRATOR: the word-count heuristic
        # overshoots (salted filler tokenizes ~1.3 tok/word), so the measured
        # runs scale their request by the warmup's actual/target ratio — the
        # depth label stays honest to within a few percent.
        request_tokens = target
        try:
            w, t, _k, ptoks = run_once(prefill_prompt(target, salt()), PP_MAX_TOKENS)
            _pp, _t, line = fmt_pp("warm-1", w, t, ptoks)
            if not QUIET:
                print(line)
            if ptoks > 0:
                request_tokens = max(200, int(target * target / ptoks))
                print(f"  [calibrated: {ptoks} tok at request={target} → measured runs request {request_tokens}]")
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ("400", "maximum context", "max_model_len", "context length", "context window")):
                print(f"  SKIP: {target} tokens exceeds the served context ({msg[:100]})")
                continue
            print(f"  warm-1     FAIL: {e}")
        print(f"\n=== measured ({n}) ===")
        pps, ttfts, ptoks_seen = [], [], []
        phase_t0 = phase_start()
        for i in range(n):
            try:
                w, t, _k, ptoks = run_once(prefill_prompt(request_tokens, salt()), PP_MAX_TOKENS)
                pp, ttft, line = fmt_pp(f"run-{i+1}", w, t, ptoks)
                if not QUIET:
                    print(line)
                progress(f"[bench] {label} run {i+1}/{n}: {ptoks} prompt tok, "
                         f"{pp:.1f} prefill tok/s, ttft {ttft*1000:.0f}ms")
                pps.append(pp)
                ttfts.append(ttft)
                ptoks_seen.append(ptoks)
            except Exception as e:
                print(f"  run-{i+1}    FAIL: {e}")
                progress(f"[bench] {label} run {i+1}/{n}: FAIL: {e}")
        mark_phase("prefill", phase_t0)
        if pps:
            print(f"\n=== summary [{label}] (n={len(pps)}) ===")
            # prompt_tokens/TTFT = the CLIENT-OBSERVED rate (includes
            # tokenization + transfer + scheduling) — the user-truth number.
            print(stats("prefill tok/s", pps))
            print(
                f"  TTFT          mean={s.mean(ttfts)*1000:6.0f}ms  "
                f"std={(s.stdev(ttfts)*1000 if len(ttfts) > 1 else 0):5.0f}ms  "
                f"min={min(ttfts)*1000:.0f}ms  max={max(ttfts)*1000:.0f}ms"
            )
            # Engine-internal rate from the vLLM stats log — windowed (tokens
            # per 10s stats window), so treat as approximate; the gap between
            # this and the client-observed rate is tokenization/transfer/
            # scheduling overhead, not prefill compute.
            if PP_MODE == "log":
                vals = scrape_prompt_throughput(CONTAINER, len(pps) * 2)
                vals = [v for v in vals if v > 0][-len(pps):]
                # #817: the prompt IS long here, so the depth condition passes —
                # but this variant still renders `mean=1127.30 CV=171.9%` (min 13
                # / max 8044) when the sampled windows straddle idle and busy.
                # Same gate, same single definition.
                w_ok, w_why = pp_verdict(vals, int(s.mean(ptoks_seen)) if ptoks_seen else 0)
                if vals and w_ok:
                    print(stats("PP tok/s (engine log, windowed — indicative only)", vals))
                elif vals:
                    print(f"  PP tok/s (engine log, windowed) n/a (suppressed: {w_why})")
                    print("                 the `prefill tok/s` line above is the client-side "
                          "measurement for this depth")
            _pm = s.mean(pps)
            SUMMARY["shapes"][label] = {
                "n": len(pps), "n_usable": len(pps), "errors": 0,
                "prefill_tps_mean": _pm,
                "prefill_tps_cv": ((s.stdev(pps) / _pm * 100) if len(pps) > 1 and _pm > 0 else 0.0),
                "ttft_mean_ms": s.mean(ttfts) * 1000,
            }

announce_sampler()

if ONLY in ("both", "narr"):
    run_set("narrative", PROMPT_NARR, MAX_NARR)
if ONLY in ("both", "code"):
    run_set("code", PROMPT_CODE, MAX_CODE)
if PP_MODE == "fallback":
    run_pp_fallback()
if os.environ.get("PREFILL_PROBE", "1") == "1":
    run_prefill_probe()

# Hand the measured totals to the shell's capture layer. stdout keeps its exact
# historical shape; the cross-checks (item 1) read this instead of re-parsing it.
if SUMMARY_JSON:
    try:
        tot = sum(sum(v.get("tokens", [])) for v in SUMMARY["shapes"].values())
        SUMMARY["total_tokens"] = tot
        SUMMARY["total_errors"] = sum(v.get("errors", 0) for v in SUMMARY["shapes"].values())
        SUMMARY["token_ratio"] = _TOKEN_RATIO[0]
        tmp = SUMMARY_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(SUMMARY, fh)
        os.replace(tmp, SUMMARY_JSON)     # atomic; a failed encode leaves no half-file
    except Exception as e:
        sys.stderr.write(f"[bench] capture: summary hand-off failed ({e})\n")
PYEOF

# ===========================================================================
# CAPTURE REPORT — everything below is additive to the historical output.
# ===========================================================================
if (( CAP_ENABLED )); then
  CAP_T1="$(date +%s)"
  CAP_ELAPSED=$(( CAP_T1 - CAP_T0 )); (( CAP_ELAPSED < 1 )) && CAP_ELAPSED=1
  cap_stop_pid "$CAP_DMON_PID"; cap_stop_pid "$CAP_LINK_PID"; cap_stop_pid "$CAP_VRAM_PID"
  cap_refresh_log
  # A run that produced no phase markers (every request failed) still needs a
  # window, or the PCIe slice would silently be empty rather than reported empty.
  [[ -s "$CAP_WORK/phases" ]] || printf 'decode %s %s\n' "$CAP_T0" "$CAP_T1" > "$CAP_WORK/phases"

  # ---- PCIe: measured used + queried available (item 10) -------------------
  echo ""
  echo "========== CAPTURE: PCIe =========="
  if [[ ! -s "$CAP_WORK/dmon" ]]; then
    cap_note_unavailable "PCIe rx/tx sampling" \
      "nvidia-smi dmon produced no output (no GPU, or dmon not supported here)"
  else
    for _ph in decode prefill; do
      command grep -q "^${_ph} " "$CAP_WORK/phases" 2>/dev/null || continue
      if _out="$(cap_dmon_phase "$CAP_WORK/dmon" "$CAP_WORK/phases" "$_ph" 2>/dev/null)"; then
        printf '%s\n' "$_out" >> "$CAP_WORK/pcie.txt"
        while IFS= read -r _l; do [[ -n "$_l" ]] && echo "  ${_l}"; done <<< "$_out"
      else
        # Reported, never silently omitted: a phase shorter than the 1 Hz sample
        # period genuinely has no samples, and that is a fact about the run.
        echo "  ${_ph}: no dmon samples inside the measured windows (phase shorter than the 1 Hz sample period)"
      fi
    done
    echo "  note: decode and prefill PCIe traffic differ by ~100x on an offloaded MoE —"
    echo "        a whole-run mean describes neither. 1 Hz under-reads bursts, so the"
    echo "        PEAK is reported next to the mean, not instead of it."
  fi
  # Link capability, sampled UNDER LOAD: an idle link downshifts to Gen1 by
  # design, so an idle reading manufactures false alarms.
  if _out="$(cap_pcie_link_under_load "$CAP_WORK/link" 2>/dev/null)"; then
    printf '%s\n' "$_out" > "$CAP_WORK/link.txt"
    echo "  link (sampled under load, best observed):"
    while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_out"
    # Utilisation% against a STATED denominator — a bandwidth percentage with an
    # unstated denominator is not a number anyone can check.
    _g="$(printf '%s\n' "$_out" | command grep -oE 'gen=[0-9]+' | head -1 | cut -d= -f2 || true)"
    _w="$(printf '%s\n' "$_out" | command grep -oE 'width=[0-9]+' | head -1 | cut -d= -f2 || true)"
    _prac="$(cap_pcie_practical_mbps "${_g:-0}" "${_w:-0}" 2>/dev/null || true)"
    if [[ -n "$_prac" ]]; then
      _rx="$(cap_dmon_phase "$CAP_WORK/dmon" "$CAP_WORK/phases" decode 2>/dev/null \
             | command grep -m1 ' all ' | command grep -oE 'rx_peak=[0-9]+' | cut -d= -f2 || true)"
      echo "    denominator: ~${_prac} MB/s practical for a negotiated gen${_g} x${_w} link"
      echo "                 (per-lane raw x width x 0.85; NOT a theoretical peak)"
      if [[ -n "$_rx" ]]; then
        CAP_PCIE_PCT="$(awk -v a="$_rx" -v b="$_prac" 'BEGIN{ if (b > 0) printf "%.1f", a*100/b }')"
        awk -v a="$_rx" -v b="$_prac" 'BEGIN{
          if (b > 0) printf "    decode rx peak = %s MB/s = %.1f%% of that denominator\n", a, a*100/b }'
      fi
    fi
  else
    cap_note_unavailable "PCIe link state" "nvidia-smi could not report pcie.link.* under load"
  fi

  # ---- VRAM triplet + leak delta (item 11b) --------------------------------
  echo ""
  echo "========== CAPTURE: VRAM =========="
  CAP_VRAM_POST="$(cap_vram_used || true)"
  CAP_VRAM_PEAK="$(cat "$CAP_WORK/vpeak" 2>/dev/null || true)"
  if [[ -n "$CAP_VRAM_IDLE" ]]; then
    echo "  idle=${CAP_VRAM_IDLE} MiB  peak=${CAP_VRAM_PEAK:-n/a} MiB  post=${CAP_VRAM_POST:-n/a} MiB"
    if [[ -n "$CAP_VRAM_POST" ]]; then
      # A free soak-lite check on VRAM that did not come back. But the LABEL has to
      # match what the config can legitimately do: an expert cache allocates its
      # pool and graphs lazily, on first inference, so a single run with a cache
      # active grows VRAM BY DESIGN. Calling that a leak cries wolf on a healthy
      # run (measured: +1036 MiB with moe-cache active). Only a monotonic rise
      # across REPEATED runs is evidence of a leak there.
      _vd=$(( CAP_VRAM_POST - CAP_VRAM_IDLE ))
      if [[ -n "$CAP_OFFLOAD" ]] && (( CAP_CACHE_WANTED )); then
        echo "  growth delta (post - idle) = ${_vd} MiB"
        echo "    includes lazy expert-cache pool/graph allocation on first inference —"
        echo "    expected with a cache active. Only a MONOTONIC rise across REPEATED"
        echo "    runs indicates a leak."
      else
        echo "  leak delta (post - idle) = ${_vd} MiB"
      fi
    fi
    while IFS= read -r _l; do [[ -n "$_l" ]] && echo "  ${_l} MiB (post-run)"; done \
      < <(cap_vram_per_device 2>/dev/null || true)
  else
    cap_note_unavailable "VRAM triplet" "nvidia-smi not available"
  fi

  # ---- CPU (item 11d) ------------------------------------------------------
  CAP_CPU1="$(cap_cpu_snap || true)"
  if [[ -n "$CAP_CPU0" && -n "$CAP_CPU1" ]]; then
    CAP_CPU_PCT="$(cap_cpu_pct "$CAP_CPU0" "$CAP_CPU1" || true)"
    # cpu% is the missing half of offload bottleneck forensics: a low sm% next to
    # a high cpu% says the GPU is waiting on the host, not the other way round.
    [[ -n "$CAP_CPU_PCT" ]] && echo "  host CPU busy across the run: ${CAP_CPU_PCT}%"
  fi

  # ---- draft acceptance WITH fire rate (item 11a) --------------------------
  echo ""
  echo "========== CAPTURE: DRAFT ACCEPTANCE =========="
  # Requests in the SAME windowed log the acceptance lines came from. Deriving
  # the denominator from the summary JSON instead counts only measured runs while
  # the numerator counted warm-ups too — which produced a 142% fire rate.
  # Field-exact extraction. `grep -oE 'n=[0-9]+'` ALSO matches the tail of
  # "mean=100.00", which produced a multi-line value and silently skipped the
  # whole fire-rate block.
  CAP_REQS="$(cap_moe_parse timings 2>/dev/null \
              | awk '$1=="decode"{for(i=2;i<=NF;i++) if($i ~ /^n=/){split($i,a,"="); print a[2]; exit}}' || true)"
  if cap_log_available "draft acceptance"; then
    if _acc="$(cap_moe_parse acceptance 2>/dev/null)"; then
      _fired="$(printf '%s' "$_acc" | command grep -oE 'fired=[0-9]+' | cut -d= -f2)"
      _reqs="${CAP_REQS:-}"
      echo "  ${_acc}"
      if [[ -n "$_reqs" && "${_reqs:-0}" -gt 0 ]]; then
        awk -v f="${_fired:-0}" -v r="$_reqs" 'BEGIN{
          if (f > r) f = r
          printf "  fire rate: fired on %d of %d requests in the measured window (%.1f%%)\n", f, r, f*100/r
          if (r > 0 && f * 100 / r < 50)
            print "  ⚠ WARN: acceptance WITHOUT a fire rate is a deception. This drafter was\n" \
                  "          idle on most requests, so its acceptance describes a minority of\n" \
                  "          the run and does NOT predict the end-to-end speed-up."
        }'
      fi
    else
      echo "  no drafter output in the log (spec-dec off, or the engine does not log acceptance)"
    fi
  fi

  # ---- engine-side timings + cross-check (item 1b, item 4) -----------------
  echo ""
  echo "========== CAPTURE: ENGINE-SIDE TIMINGS =========="
  if cap_log_available "engine-side timings"; then
    if _tim="$(cap_moe_parse timings 2>/dev/null)"; then
      while IFS= read -r _l; do
        case "$_l" in
          "") ;;
          dropped_short_decode*)
            echo "  ${_l}  (short probe completions excluded from the decode cross-check)" ;;
          *)  echo "  ${_l}  (tok/s, from the server's own print_timing)" ;;
        esac
      done <<< "$_tim"
      _eng="$(printf '%s\n' "$_tim" | command grep -m1 '^decode ' | command grep -oE 'median=[0-9.]+' | cut -d= -f2 || true)"
      _cli="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
shapes = d.get("shapes", {}).values()
xs = [v.get("decode_tps_mean", 0) for v in shapes
      if v.get("decode_tps_mean") and not v.get("decode_derived")]
print(f"{sum(xs)/len(xs):.2f}" if xs else "")
' "$CAP_WORK/summary.json" 2>/dev/null || true)"
      if [[ -n "$_eng" && -n "$_cli" ]]; then
        _div="$(cap_divergence_pct "$_eng" "$_cli" || true)"
        if [[ -n "$_div" ]]; then
          echo "  cross-check: client-observed ${_cli} vs engine-side ${_eng} tok/s -> ${_div}% divergence"
          awk -v d="$_div" 'BEGIN{ if (d+0 > 5) print "  ⚠ WARN: >5% divergence. One of these two is not measuring what its label\n          says. State WHICH number a card quotes, and why." }'
        fi
      fi
    else
      echo "  no print_timing lines in the log (engine may not emit them at this verbosity)"
    fi
  fi

  # ---- moe-cache: GATED on CPU-offload detection (item 3) ------------------
  echo ""
  echo "========== CAPTURE: EXPERT CACHE (moe-cache) =========="
  if [[ -z "$CAP_OFFLOAD" ]]; then
    echo "  skipped: no CPU offload detected — nothing for an expert cache to cache."
  elif ! cap_log_available "moe-cache telemetry"; then
    : # the notice is already printed by cap_log_available
  else
    CAP_CACHE_OK=1
    if _cen="$(cap_moe_parse census 2>/dev/null)"; then
      echo "  pool census (per device — a device can hold MULTIPLE pools; slots/bytes are summed):"
      while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_cen"
    else
      _cen=""
    fi
    # #824: the '[moe-cache] enabled:' banner still prints when only SOME devices
    # got a pool. Ground truth is the count of devices WITH a pool vs the device
    # count — not the banner, and not the raw pool-line count (multi-pool devices
    # make that count pass while a device has nothing).
    if _pd="$(cap_moe_parse pooldev 2>/dev/null)"; then
      _devs="$(printf '%s' "$_pd" | command grep -oE 'devices=[0-9]+' | cut -d= -f2)"
      _enb="$(printf '%s' "$_pd" | command grep -oE 'enabled=[0-9]+' | cut -d= -f2)"
      _nob="$(printf '%s' "$_pd" | command grep -oE 'nobudget=[0-9]+' | cut -d= -f2)"
      echo "  allocation: ${_pd}  (GPUs on this host: ${CAP_NGPU})"
      if (( CAP_CACHE_WANTED )) && [[ "${_devs:-0}" -lt "${CAP_NGPU:-0}" ]]; then
        CAP_CACHE_OK=0
        echo "  ⚠ WARN: a cache was REQUESTED but only ${_devs:-0} of ${CAP_NGPU} devices hold a pool."
        if [[ "${_nob:-0}" != "0" ]]; then
          echo "          reason: a device had no budget after its reserve. Lower RESERVE_MB."
        elif [[ "${_enb:-0}" == "1" ]]; then
          echo "          the 'enabled:' banner still printed — this is the #824 half-cached state."
        fi
        echo "          This run measured a HALF-CACHED path; do not quote it for a cache conclusion."
      fi
    fi
    cap_moe_parse counters > "$CAP_WORK/counters1" 2>/dev/null || true
    if [[ -s "$CAP_WORK/counters1" ]]; then
      [[ -s "$CAP_WORK/counters0" ]] || : > "$CAP_WORK/counters0"
      if _mar="$(cap_marginal_rates "$CAP_WORK/counters0" "$CAP_WORK/counters1" 2>/dev/null)"; then
        # ⚠️ ABSENCE OF MEASUREMENT IS NOT A MEASURED ZERO. If lookups did not
        # ADVANCE across the bench window, this run produced no cache data and
        # the cumulative column is from some OTHER session -- print the cause,
        # not a number. Two real ways to land here, both seen in the wild
        # (club-3090 #1105/#1106):
        #   1. GGML_CUDA_MOE_CACHE_STATS unset -> engine `stats_every` is 0
        #      (moe-cache.cu:169) and the `stats_every > 0` gate (:2795) means
        #      NO `hits=` line is ever emitted. Not gated on log verbosity.
        #   2. A DEAD SESSION's line is still in the docker log (an earlier boot,
        #      or a crash-looped container). The parser takes the LAST `hits=`
        #      line, which can predate this boot entirely -- #1106 rendered
        #      `cumulative=0.0% admission=12` from an early session whose
        #      container had exited hours before the bench ran.
        # Rendering either as `cumulative=0.0% dhits=0 dlookups=0` reads as "the
        # cache did nothing", and that misread has now cost two community reports
        # and one of our own sessions.
        _dlk="$(printf '%s\n' "$_mar" | awk '{
          for (i = 1; i <= NF; i++) if ($i ~ /^dlookups=/) { split($i, a, "="); t += a[2] }
        } END { print t + 0 }')"
        if [[ "${_dlk:-0}" -eq 0 ]]; then
          # ⚠️ Deliberately does NOT set CAP_CACHE_OK=0. That flips the status to
          # CACHE_DISABLED (cap_status_classify: want==1 && got!=1), which ASSERTS
          # the cache is off when all we know is that we did not measure it — the
          # same absence-reported-as-data error this branch exists to prevent.
          echo "  hit rate: ⚠ NO MEASUREMENT IN THIS RUN — lookups did not advance across the bench window."
          echo "            The cache may be fine; this run simply did not measure it. Do NOT read"
          echo "            the numbers below as a hit rate of zero."
          # TWO independent conditions, BOTH required. Measured on a live 2-arm boot
          # 2026-08-26 (one variable): STATS=200 at default verbosity produced ZERO
          # `[moe-cache]` lines of any kind; STATS=200 + verbosity 4 produced 228
          # lines / 211 stats lines. MOE_CACHE_LOG is GGML_LOG_INFO, which the
          # server's verbosity threshold drops below 4 — so STATS alone is inert.
          _st="$(cap_proc_env GGML_CUDA_MOE_CACHE_STATS 2>/dev/null || true)"
          [[ -z "$_st" ]] && _st="$(printenv GGML_CUDA_MOE_CACHE_STATS 2>/dev/null || true)"
          _vb="$(cap_proc_env LLAMA_ARG_LOG_VERBOSITY 2>/dev/null || true)"
          [[ -z "$_vb" ]] && _vb="$(printenv LLAMA_ARG_LOG_VERBOSITY 2>/dev/null || true)"
          if [[ -z "$_st" || "$_st" == "0" ]] || [[ -z "$_vb" || "$_vb" -lt 4 ]] 2>/dev/null; then
            echo "            likely cause: cache telemetry needs BOTH knobs; this run had"
            echo "                          GGML_CUDA_MOE_CACHE_STATS=${_st:-<unset>} and LLAMA_ARG_LOG_VERBOSITY=${_vb:-<unset>}."
            echo "                          STATS<=0 emits no stats line (stats_every defaults to 0);"
            echo "                          verbosity <4 drops EVERY [moe-cache] line at the log filter."
            echo "            Fix: re-boot with MOE_STATS=200 AND LLAMA_ARG_LOG_VERBOSITY=4."
          else
            echo "            likely cause: the parsed counters predate this boot (stale session in the"
            echo "                          docker log), or the cache was never consulted."
            echo "            Fix: restart the container before benching so the log carries one session."
          fi
          echo "    raw (UNSCOPED — may belong to another session):"
        else
          echo "  hit rate, per device (MARGINAL = across the measured window only):"
        fi
        while IFS= read -r _l; do [[ -n "$_l" ]] && echo "    ${_l}"; done <<< "$_mar"
        echo "    note: the CUMULATIVE column embeds the cold-fill phase and is NOT"
        echo "          boot-comparable — a BIGGER pool shows a LOWER cumulative rate on"
        echo "          the same traffic. Compare boots on the MARGINAL column only."
        echo "    note: the per-device split is signal, not noise — a CUDA0/CUDA1 hit"
        echo "          asymmetry is routing entropy, and averaging it away hides it."
      fi
      echo "  health counters: $(cap_health_verdict "$CAP_WORK/counters1" || echo 'unreadable')"

      # ---- derived RAM-read demand (item 9a) — offload AND cache ----------
      if [[ -n "$_cen" ]]; then
        _exp="$(printf '%s\n' "$_cen" | command grep -oE 'expert_kib=[0-9]+' | cut -d= -f2 | head -1 || true)"
        _miss="$(printf '%s\n' "${_mar:-}" | awk '{
          for (i = 1; i <= NF; i++) {
            if ($i ~ /^dhits=/)    { split($i, a, "="); h += a[2] }
            if ($i ~ /^dlookups=/) { split($i, b, "="); t += b[2] }
          }
        } END { m = t - h; print (m > 0 ? m : 0) }' || true)"

        # DECODE-WINDOW attribution. Dividing the whole-run miss count by the whole
        # run wall dilutes the figure with the prefill phases, during which the
        # cache is not on the hot path — a live run spent ~40% of its wall in
        # prefill and reported ~14 GB/s where the decode-window figure is roughly
        # double. The engine's counters carry no timestamps, so the decode windows
        # are recovered by reading the cumulative counters truncated at the log
        # line positions python recorded at each phase boundary.
        _win_secs=0; _win_miss=0; _win_ok=0
        if [[ -s "$CAP_WORK/phases" ]]; then
          while read -r _lab _pt0 _pt1 _pl0 _pl1 _junk; do
            [[ "$_lab" == "decode" ]] || continue
            _win_secs=$(( _win_secs + _pt1 - _pt0 ))
            [[ -n "${_pl1:-}" && -n "${_pl0:-}" && "${_pl1:-0}" -gt "${_pl0:-0}" ]] || continue
            CAP_LINE_LIMIT="$_pl0" cap_moe_parse counters > "$CAP_WORK/pc0" 2>/dev/null || continue
            CAP_LINE_LIMIT="$_pl1" cap_moe_parse counters > "$CAP_WORK/pc1" 2>/dev/null || continue
            _wm="$(cap_marginal_rates "$CAP_WORK/pc0" "$CAP_WORK/pc1" 2>/dev/null | awk '{
              for (i = 1; i <= NF; i++) {
                if ($i ~ /^dhits=/)    { split($i, a, "="); h += a[2] }
                if ($i ~ /^dlookups=/) { split($i, b, "="); t += b[2] }
              }
            } END { m = t - h; print (m > 0 ? m : 0) }' || true)"
            [[ -n "$_wm" && "$_wm" -gt 0 ]] && { _win_miss=$(( _win_miss + _wm )); _win_ok=1; }
          done < "$CAP_WORK/phases"
        fi
        unset CAP_LINE_LIMIT

        _ram=""; _ram_scope=""; _ram_arith=""; _ram_caveat=""
        if [[ -n "$_exp" ]] && (( _win_ok )) && (( _win_secs > 0 )) && (( _win_miss > 0 )); then
          _ram="$(cap_ram_rd_mbps "$_win_miss" "$_exp" "$_win_secs" || true)"
          _ram_scope="(DECODE WINDOW)"
          _ram_arith="= ${_win_miss} misses x ${_exp} KiB / ${_win_secs}s of decode"
        elif [[ -n "$_exp" && -n "$_miss" && "${_miss:-0}" -gt 0 ]]; then
          _ram="$(cap_ram_rd_mbps "$_miss" "$_exp" "$CAP_ELAPSED" || true)"
          _ram_scope="(WHOLE RUN — diluted)"
          _ram_arith="= ${_miss} misses x ${_exp} KiB / ${CAP_ELAPSED}s"
          _ram_caveat="dilution"
        fi
        if [[ -n "$_ram" ]]; then
          CAP_RAM_DEMAND="$_ram"
          CAP_RAM_WINDOW="$([[ "$_ram_caveat" == "dilution" ]] && echo "whole-run (diluted)" || echo "decode window")"
          echo "  derived host-RAM read demand on the miss path: ~${_ram} MB/s  ${_ram_scope}"
          echo "    ${_ram_arith}"
          echo "    ⚠ DERIVED, NOT A PERF COUNTER. A lower bound on the miss path only"
          echo "      (ignores KV traffic, activations and prefetch). IMC/uncore counters"
          echo "      are root-gated on bare metal and absent in VM guests, so this is the"
          echo "      portable substitute — never quote it as measured bandwidth."
          if [[ "$_ram_caveat" == "dilution" ]]; then
            echo "    ⚠ Averaged over the FULL RUN INCLUDING non-decode phases (prefill, warm-up),"
            echo "      during which the cache is not on the hot path — the decode-window demand"
            echo "      is HIGHER than this. Set SERVER_LOG=<path> to get the decode-window figure."
          fi
          if [[ -n "${CAP_STREAM:-}" ]]; then
            awk -v d="$_ram" -v c="${CAP_STREAM}" 'BEGIN{
              if (c > 0) printf "    ~%.0f of ~%.0f GB/s STREAM-triad ceiling (%.0f%%)\n", d/1000, c, d/10/c }'
          else
            echo "    (STREAM_CALIB=1 adds a ~3 s triad ceiling so this can be stated as a fraction)"
          fi
        fi
      fi
    fi
    _byp="$(cap_moe_parse bypass 2>/dev/null || echo 0)"
    echo "  bypass counter: ${_byp:-0}$( [[ "${_byp:-0}" != "0" ]] && echo '  ⚠ a decode DECLINED the cache — this run does not measure the cached path' )"
  fi

  # ---- integrity panel (items 1, 3c, 8, 11c) -------------------------------
  echo ""
  echo "========== CAPTURE: INTEGRITY =========="
  CAP_TOKENS="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
print(d.get("total_tokens", 0))
' "$CAP_WORK/summary.json" 2>/dev/null || echo 0)"
  CAP_ERRS="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
print(d.get("total_errors", 0))
' "$CAP_WORK/summary.json" 2>/dev/null || echo 0)"
  # The status enum keys on A tps, and `_cli` only exists when an engine-side log
  # was available to cross-check against. With no log — CONTAINER=none and no
  # SERVER_LOG — a perfectly healthy run classified NO_TOKENS purely because the
  # cross-check had nothing to compare with. Same hole now reachable a second way:
  # a canvas-granularity run's decode mean is DERIVED and deliberately excluded
  # from the cross-check (#809), which empties `_cli` on a run that measured fine.
  # Fall back to the client-measured wall rate, which every successful run has.
  CAP_TPS="${_cli:-}"
  if [[ -z "$CAP_TPS" || "$CAP_TPS" == "0" ]]; then
    CAP_TPS="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(1)
xs = [v.get("wall_tps_mean", 0) for v in d.get("shapes", {}).values() if v.get("wall_tps_mean")]
print(f"{sum(xs)/len(xs):.2f}" if xs else "0")
' "$CAP_WORK/summary.json" 2>/dev/null || echo 0)"
  fi
  CAP_STATUS="$(cap_status_classify "${CAP_TOKENS:-0}" "${CAP_TPS:-0}" "${CAP_ERRS:-0}" \
                  "$CAP_CACHE_WANTED" "${CAP_CACHE_OK:-1}" "${_byp:-0}")"
  echo "  status: ${CAP_STATUS}"
  case "$CAP_STATUS" in
    NO_TOKENS)
      # The exact "plausible wrong number" class #824 catalogued: wall_TPS and
      # decode_TPS printed 0.00 while TTFT printed fine, and nothing warned.
      echo "  ⚠⚠ NO_TOKENS: the run measured ${CAP_TOKENS:-0} completion tokens at ${CAP_TPS:-0} TPS."
      echo "     A TPS of 0.00 next to a healthy-looking TTFT is a SCRAPE failure, not a"
      echo "     slow model. Do NOT quote any throughput number from this run."
      echo "     Check: did the model EOS immediately? did usage arrive in the stream?"
      echo "     (stream_options.include_usage), and does the endpoint support it?" ;;
    REQ_ERRORS)
      echo "  ⚠⚠ REQ_ERRORS: ${CAP_ERRS} request(s) failed. Every aggregate above is over the"
      echo "     survivors only." ;;
    CACHE_DISABLED)
      echo "  ⚠⚠ CACHE_DISABLED: a cache was requested but is not (fully) allocated — see the"
      echo "     expert-cache section. This run measured the no-cache or half-cached path." ;;
    INVALID_BYPASS)
      echo "  ⚠⚠ INVALID_BYPASS: ${_byp} decode(s) declined the cache." ;;
    OK)
      echo "  (no capture-level defect detected — this is NOT a statement about the numbers'"
      echo "   quality, only that the measurement machinery did its job)" ;;
  esac
  echo "  swap check   : $(cap_swap_verdict 2>/dev/null || echo 'unavailable')"
  if [[ -s "$CAP_WORK/counters1" ]]; then
    echo "  cache health : $(cap_health_verdict "$CAP_WORK/counters1" 2>/dev/null || echo 'unavailable')"
  fi
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
for name, v in d.get("shapes", {}).items():
    n, u = v.get("n", 0), v.get("n_usable", 0)
    floor = v.get("short_eos_floor", 0)
    if n and u < n:
        print(f"  ⚠ n-usable   : {name} {u}/{n} runs above the short-EOS floor "
              f"({floor} tok) - the CV for this shape is not trustworthy")
    dg = v.get("decode_degenerate", 0)
    if n and dg:
        gran = v.get("decode_granularity", "token")
        tag = ("canvas granularity - dLLM block emission, decode window undefined"
               if gran == "canvas" else "zero-width decode window")
        note = " decode_TPS is DERIVED (= wall_TPS)." if v.get("decode_derived") else ""
        print(f"  ⚠ decode-win : {name} {dg}/{n} run(s) unmeasurable - {tag}.{note}"
              f" Quote wall_TPS, not decode_TPS (#809)")
' "$CAP_WORK/summary.json" || true

  # ---- BENCH CARD (opt-in: CARD=snapshot | ab) ----------------------------
  # docs/BENCH_CARD.md's templates are correct and nobody fills them in by hand
  # under time pressure — which is when the awkward facts get left off. Everything
  # above is already captured; this pours it into the template.
  if [[ -n "${CARD:-}" && -f "${ROOT_DIR}/scripts/lib/card.sh" ]]; then
    # shellcheck source=lib/card.sh
    source "${ROOT_DIR}/scripts/lib/card.sh"
    CARD_REC="$CAP_WORK/card.kv"
    : > "$CARD_REC"
    card_kv "$CARD_REC" meta.date        "$(date +%F)"
    card_kv "$CARD_REC" meta.endpoint    "$URL"
    card_kv "$CARD_REC" meta.mode        "$ENDPOINT"
    card_kv "$CARD_REC" fp.model         "$MODEL"
    card_kv "$CARD_REC" fp.ctx           "$(cap_props_get 'default_generation_settings.n_ctx' || true)"
    card_kv "$CARD_REC" fp.slots         "$(cap_props_get 'total_slots' || true)"
    card_kv "$CARD_REC" fp.ftype         "$(cap_props_get 'model_ftype' || true)"
    card_kv "$CARD_REC" fp.drafter       "$(cap_props_get 'default_generation_settings.params.speculative.types' || true)"
    card_kv "$CARD_REC" fp.kv            "$(cap_kv_type "$CAP_ARGV" || true)"
    card_kv "$CARD_REC" fp.argv          "$(cap_argv_fingerprint "$CAP_ARGV" 2>/dev/null || true)"
    card_kv "$CARD_REC" fp.moe           "$(cap_moe_cache_config "$CAP_ARGV" || true)"
    card_kv "$CARD_REC" fp.offload       "${CAP_OFFLOAD_DESC:-}"
    card_kv "$CARD_REC" proto.warmups    "$WARMUPS"
    card_kv "$CARD_REC" proto.runs       "$RUNS"
    card_kv "$CARD_REC" proto.force_tokens "$FORCE_TOKENS"
    card_kv "$CARD_REC" proto.thinking   "$ENABLE_THINKING"
    # The Env line should name what was NOT default — that is what makes a run
    # reproducible, and it is the line most often reconstructed wrongly by hand.
    _envnote=""
    [[ "$RUNS" != "5" ]]            && _envnote+="RUNS=${RUNS} "
    [[ "$WARMUPS" != "3" ]]         && _envnote+="WARMUPS=${WARMUPS} "
    [[ "$FORCE_TOKENS" != "0" ]]    && _envnote+="FORCE_TOKENS=${FORCE_TOKENS} "
    [[ "$ENABLE_THINKING" == "1" ]] && _envnote+="ENABLE_THINKING=1 "
    [[ "$ENDPOINT" != "chat" ]]     && _envnote+="ENDPOINT=${ENDPOINT} "
    [[ "$ONLY" != "both" ]]         && _envnote+="ONLY=${ONLY} "
    [[ "$PREFILL_DEPTHS" != "10000,90000" ]] && _envnote+="PREFILL_DEPTHS=${PREFILL_DEPTHS} "
    [[ -n "${SERVER_LOG:-}" ]]      && _envnote+="SERVER_LOG=<set> "
    card_kv "$CARD_REC" proto.env "${_envnote:-}"
    card_kv "$CARD_REC" gpu.vram_idle    "${CAP_VRAM_IDLE:-}"
    card_kv "$CARD_REC" gpu.vram_peak    "${CAP_VRAM_PEAK:-}"
    card_kv "$CARD_REC" gpu.vram_post    "${CAP_VRAM_POST:-}"
    [[ -n "${CAP_VRAM_IDLE:-}" && -n "${CAP_VRAM_POST:-}" ]] \
      && card_kv "$CARD_REC" gpu.leak "$(( CAP_VRAM_POST - CAP_VRAM_IDLE ))"
    card_kv "$CARD_REC" integrity.status "${CAP_STATUS:-}"
    card_kv "$CARD_REC" integrity.divergence "${_div:-}"
    card_kv "$CARD_REC" integrity.stream_ceiling "${CAP_STREAM:-}"
    [[ -s "$CAP_WORK/counters1" ]] \
      && card_kv "$CARD_REC" integrity.health "$(cap_health_verdict "$CAP_WORK/counters1" || true)"
    card_kv "$CARD_REC" integrity.swap "$(cap_swap_verdict 2>/dev/null || true)"
    # --- offload / PCIe / RAM telemetry for the card's offload block ---------
    card_kv "$CARD_REC" sys.ram "${CAP_RAM_SUMMARY:-}"
    card_kv "$CARD_REC" ram.demand_mbps "${CAP_RAM_DEMAND:-}"
    card_kv "$CARD_REC" ram.window      "${CAP_RAM_WINDOW:-}"
    card_kv "$CARD_REC" ram.ceiling_gbps "${CAP_STREAM:-}"
    card_kv "$CARD_REC" pcie.link_pct   "${CAP_PCIE_PCT:-}"
    card_kv "$CARD_REC" pcie.practical_mbps "${_prac:-}"
    if [[ -s "$CAP_WORK/link.txt" ]]; then
      card_kv "$CARD_REC" pcie.link "$(command grep -m1 -oE 'GPU[0-9]+ gen=[0-9]+/[0-9]+ width=[0-9]+/[0-9]+' "$CAP_WORK/link.txt" || true)"
      command grep -q 'WARN: never reached' "$CAP_WORK/link.txt" \
        && card_kv "$CARD_REC" pcie.link_warn "link trained below its own maximum under load"
    fi
    if [[ -s "$CAP_WORK/pcie.txt" ]]; then
      while read -r _ph _dev _rest; do
        [[ "$_dev" == "all" || "$_dev" == GPU* ]] || continue
        _k="pcie.${_ph}.${_dev}"
        card_kv "$CARD_REC" "${_k}.rx_mean" "$(printf '%s' "$_rest" | command grep -oE 'rx_mean=[0-9]+' | cut -d= -f2 || true)"
        card_kv "$CARD_REC" "${_k}.rx_peak" "$(printf '%s' "$_rest" | command grep -oE 'rx_peak=[0-9]+' | cut -d= -f2 || true)"
      done < "$CAP_WORK/pcie.txt"
    fi
    # Per-device cache detail the offload table needs beyond hit rates.
    if [[ -s "$CAP_WORK/counters1" ]]; then
      while read -r _d _rest; do
        [[ -n "$_d" ]] || continue
        card_kv "$CARD_REC" "cache.${_d}.evictions" \
          "$(printf '%s' "$_rest" | command grep -oE 'devictions=[0-9]+' | cut -d= -f2 || true)"
      done < <(cap_marginal_rates "$CAP_WORK/counters0" "$CAP_WORK/counters1" 2>/dev/null || true)
      while read -r _d _rest; do
        [[ -n "$_d" ]] || continue
        card_kv "$CARD_REC" "cache.${_d}.skips" \
          "$(printf '%s' "$_rest" | command grep -oE 'skips=[0-9]+' | cut -d= -f2 || true)"
      done < "$CAP_WORK/counters1"
    fi
    while read -r _d _rest; do
      [[ -n "$_d" ]] || continue
      card_kv "$CARD_REC" "cache.${_d}.type" \
        "$(printf '%s' "$_rest" | command grep -oE 'type=[^ ]+' | cut -d= -f2 || true)"
    done < <(cap_moe_parse census 2>/dev/null || true)
    # Per-device cache rows stay PER DEVICE on the card (item 3d).
    if [[ -s "$CAP_WORK/counters1" ]]; then
      while read -r _d _rest; do
        [[ -n "$_d" ]] || continue
        card_kv "$CARD_REC" "cache.${_d}.marginal" \
          "$(printf '%s' "$_rest" | command grep -oE 'marginal=[0-9.]+' | cut -d= -f2 || true)"
        card_kv "$CARD_REC" "cache.${_d}.cumulative" \
          "$(printf '%s' "$_rest" | command grep -oE 'cumulative=[0-9.]+' | cut -d= -f2 || true)"
      done < <(cap_marginal_rates "$CAP_WORK/counters0" "$CAP_WORK/counters1" 2>/dev/null || true)
      while read -r _d _rest; do
        [[ -n "$_d" ]] || continue
        card_kv "$CARD_REC" "cache.${_d}.cum_start" \
          "$(printf '%s' "$_rest" | command grep -oE 'cumulative=[0-9.]+' | cut -d= -f2 || true)"
      done < <(cap_cumulative_rates "$CAP_WORK/counters0" 2>/dev/null || true)
    fi
    while read -r _d _rest; do
      [[ -n "$_d" ]] || continue
      card_kv "$CARD_REC" "cache.${_d}.slots" \
        "$(printf '%s' "$_rest" | command grep -oE 'slots=[0-9]+' | cut -d= -f2 || true)"
      card_kv "$CARD_REC" "cache.${_d}.total_mib" \
        "$(printf '%s' "$_rest" | command grep -oE 'total_mib=[0-9]+' | cut -d= -f2 || true)"
    done < <(cap_moe_parse census 2>/dev/null || true)
    card_kv_from_summary "$CARD_REC" "$CAP_WORK/summary.json" || true
    # Debug seam: lets the suite diff this in-process record against the one
    # card.sh parses back out of this run's own log. If those disagree, every
    # A/B silently compares two different measurements.
    [[ -n "${CARD_RECORD_OUT:-}" ]] && cp "$CARD_REC" "$CARD_RECORD_OUT"

    echo ""
    echo "========== BENCH CARD (CARD=${CARD}) =========="
    echo "Paste the block below into the issue / discussion. Placeholders marked"
    echo "<fill: …> are human judgement — the run cannot know them, so it will not guess."
    echo ""
    echo '```markdown'
    case "$CARD" in
      snapshot) card_render snapshot "$CARD_REC" || echo "(card render failed)" ;;
      ab)       card_render ab "$CARD_REC" "${BASELINE:-}" \
                  || echo "(A/B card not rendered — see the error above; the run's own numbers are intact)" ;;
      *)        echo "(unknown CARD=${CARD}; expected 'snapshot' or 'ab')" ;;
    esac
    echo '```'
  fi
fi

# GPU state
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "=== GPU state ==="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv,noheader
fi

# ===========================================================================
# Interconnect state — all three layers (#805)
# ===========================================================================
# report.sh has reported this since #789; bench.sh did not — so every
# BENCHMARKS.md Rig cell's field 4 (resolved custom-AR state) was transcribed by
# hand from a DIFFERENT run's report, and went stale silently. The three layers
# are genuinely independent and a rig can sit in any combination of them:
#
#   layer 1  driver P2P GRANT   — did the kernel module grant peer access at all
#                                 (`topo -p2p`)? A closed GeForce module refuses
#                                 it outright; the open modules can grant it.
#   layer 2  NCCL USE           — is the collective library configured to use
#                                 P2P (NCCL_P2P_*/NVLINK_MODE), or told not to?
#   layer 3  engine custom-AR   — did the ENGINE engage its own all-reduce
#                                 kernel? vLLM vetoes its custom AR at world>2
#                                 without NVLink even when layers 1-2 are ON
#                                 (#786) — P2P is still live via NCCL there, so
#                                 this is a third state, not "off".
#
# Degrades cleanly: host-mode / llama.cpp runs (CONTAINER=none) have no engine
# all-reduce at all, so layer 3 says so and layers 1-2 still report.
bench_interconnect_block() {
  declare -F p2p_gpu_count >/dev/null || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local ngpu cap flavor eng_text nccl_line l3 verdict
  ngpu="$(p2p_gpu_count 2>/dev/null || echo 0)"

  echo ""
  echo "=== Interconnect (three layers) ==="

  if [[ "${ngpu:-0}" -lt 2 ]]; then
    # Not silent: a BENCHMARKS row for a single-card run should SAY it was
    # single-card rather than leave the reader to infer it from an absent block.
    echo "  layer 1  driver P2P grant : n/a — ${ngpu:-0} GPU(s) visible (single-card run)"
    echo "  layer 2  NCCL use         : n/a — no multi-GPU communicator"
    echo "  layer 3  engine custom-AR : n/a — no multi-GPU communicator"
    return 0
  fi

  # ---- layer 1: driver grant -----------------------------------------------
  cap="$(p2p_host_capability "$ngpu" 2>/dev/null || echo none)"
  flavor="$(p2p_driver_flavor 2>/dev/null || echo unknown)"
  case "$cap" in
    nvlink)   echo "  layer 1  driver P2P grant : NVLink bridge present (topo -m reports NV#); kernel module: ${flavor}" ;;
    pcie_p2p) echo "  layer 1  driver P2P grant : GRANTED over PCIe — topo -p2p reports OK on every pair; kernel module: ${flavor}" ;;
    *)        echo "  layer 1  driver P2P grant : REFUSED — topo -p2p reports no all-pairs OK; kernel module: ${flavor}" ;;
  esac

  # ---- layer 2: NCCL use ---------------------------------------------------
  # Container env first (the serving process's RESOLVED value, post-entrypoint),
  # then the bare-metal server's /proc environ, then our own.
  nccl_line=""
  if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 \
     && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    nccl_line="$(docker exec "$CONTAINER" env 2>/dev/null \
      | command grep -E '^(NCCL_P2P|NVLINK_MODE|NCCL_CUMEM)=' | sort | paste -sd' ' - || true)"
  fi
  # cap_proc_env lives in capture.sh, which is only sourced when CAPTURE=1 —
  # a CAPTURE=0 run still gets layers 1 and 3, just not the /proc fallback.
  if [[ -z "$nccl_line" && -n "${SERVER_PID:-}" && "${SERVER_PID}" != "none" ]] \
     && declare -F cap_proc_env >/dev/null; then
    local v got
    for v in NCCL_P2P_DISABLE NCCL_P2P_LEVEL NVLINK_MODE NCCL_CUMEM_ENABLE; do
      got="$(cap_proc_env "$v" "$SERVER_PID" 2>/dev/null || true)"
      [[ -n "$got" ]] && nccl_line="${nccl_line}${nccl_line:+ }${v}=${got}"
    done
  fi
  if [[ -n "$nccl_line" ]]; then
    echo "  layer 2  NCCL use         : ${nccl_line}"
  else
    echo "  layer 2  NCCL use         : no NCCL_P2P*/NVLINK_MODE in the serving environment (engine default)"
  fi

  # ---- layer 3: engine custom-AR -------------------------------------------
  # The classifier wants the same text report.sh feeds it: the [nvlink] decision
  # trail + vLLM's own gate line + the resolved env.
  eng_text=""
  if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 \
     && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    eng_text="$(docker logs "$CONTAINER" 2>&1 | command grep -E '\[nvlink\]|Custom allreduce is disabled|disable_custom_all_reduce=True' | head -8 || true)"
  fi
  if [[ "$ENGINE_KIND" == "llamacpp" || "${CONTAINER:-}" == "none" ]]; then
    # llama.cpp/ik-llama split layers across cards with plain copies — there is
    # no custom all-reduce kernel to engage or veto. Saying "off" would read as
    # a misconfiguration; it is a property of the engine.
    l3="engine: ${ENGINE_KIND/llamacpp/llama.cpp} — custom-AR n/a"
  else
    case "$(printf '%s\n%s' "$eng_text" "$nccl_line" | p2p_classify_engagement 2>/dev/null || echo unknown)" in
      on)        l3="ENGAGED — engine reports its custom all-reduce ON" ;;
      nccl_only) l3="custom-AR OFF, P2P LIVE — the engine is not using its custom all-reduce; peer transfers still go via NCCL. Cause is either vLLM's own NVLink-only gate at world>2 (#786) or an operator-supplied --disable-custom-all-reduce (#922). Check the engine log to tell which; both are healthy states" ;;
      off)       l3="OFF — the serving container resolved to PCIe/no-P2P mode" ;;
      requested) l3="REQUESTED but UNVERIFIED — P2P forced on without a driver grant (#688)" ;;
      *)         l3="unknown — no [nvlink] boot line and no engine gate line in the log" ;;
    esac
  fi
  echo "  layer 3  engine custom-AR : ${l3}"

  # ---- the cross-referenced verdict (same matrix report.sh renders) ---------
  verdict="$(p2p_verdict "$ngpu" "$cap" \
    "$(printf '%s\n%s' "$eng_text" "$nccl_line" | p2p_classify_engagement 2>/dev/null || echo unknown)" 2>/dev/null || true)"
  [[ -n "$verdict" ]] && echo "  verdict  : ${verdict}"
  return 0
}
bench_interconnect_block || true

# MTP / spec-decode stats — only when running against a Docker container we own.
# Skipped silently in endpoint-first mode (CONTAINER=none).
if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1 \
   && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo ""
  echo "=== Last 3 SpecDecoding metrics ==="
  docker logs "${CONTAINER}" 2>&1 | grep "SpecDecoding metrics" | tail -3 || true
fi

# Repeated at the END on purpose (#832): a reader who tails the log, or who
# scrolls to the summary and copies it, never sees the opening banner.
if [[ "$QUICK" == "1" ]]; then
  echo ""
  echo "⚠ QUICK MODE — the numbers above are NOT canonical: n=1, no CV, narrative only,"
  echo "  no prefill anchor. Not a BENCHMARKS.md row. For any comparison that needed a"
  echo "  reboot, run >=2 BOOTS per arm — --quick cut the within-boot samples, not the"
  echo "  between-boot variance. Drop --quick for the canonical protocol."
fi

# --- emit the per-rig #249 record from THIS run's captured output -------------
# Skipped for --quick (not a canonical measurement). resolve-serving maps the
# served container -> registry slug + auto-detects the fingerprint; an unmatched
# or bare-metal run skips cleanly. Records are append-only history; c3 shows the
# newest per slug. Failure here never fails the bench.
# ⚠️ Skip under BENCH_MOCK: mock runs emit synthetic ~475 TPS output, and
# test-bench-capture.sh drives bench.sh under BENCH_MOCK many times — with a live
# container present, --resolve-serving would match and write a hollow mock record
# into the real per-rig corpus (they surfaced in c3 once tagged bench-measured).
if [[ -n "${_BENCH_REC_LOG:-}" && -f "${_BENCH_REC_LOG}" && "${QUICK:-0}" != "1" && -z "${BENCH_MOCK:-}" ]]; then
  sync 2>/dev/null || true
  sleep 0.4   # let the tee subprocess flush the summary block before we read it
  python3 "${ROOT_DIR}/scripts/lib/profiles/measurement_record.py" \
    --resolve-serving --serving-url "$URL" --result-class bench-measured \
    --bench-output "${_BENCH_REC_LOG}" >/dev/null 2>&1 || true
  rm -f "${_BENCH_REC_LOG}"
fi
