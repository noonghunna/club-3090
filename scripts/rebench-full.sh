#!/usr/bin/env bash
#
# rebench-full.sh — canonical rebench against the currently-running model
# (a fail-fast verify-full preflight + 5 measured steps). Built to eliminate
# the recurring mistakes from manual runs:
#
#   - Wrong cwd (`scripts/X.sh: No such file or directory`)
#   - Forgot `--save-json` on benchlocal-cli direct invocations
#   - Forgot `MODEL=` override → HTTP 404 from served-model-name mismatch
#   - Forgot `BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1` for localhost URLs
#   - Wrong port (8010 / 8011 / 8030 / 8032 — easy typo)
#   - No idempotent resume — every interrupt redoes the whole matrix
#
# Order matches docs/QUALITY_TEST.md "test pipeline":
#   0. verify-full.sh          — functional preflight, FAIL-FAST (~2 min)
#   1. bench.sh                — TPS narrative + code (~5 min)
#   2. verify-stress.sh        — long-context + boundary (~10-15 min)
#   3. 8-pack quality          — OPT-IN, skipped by default (#338). Enable with
#      --with-8pack-thinking[=off|on|both]:
#        =off  → quality-test.sh --full --no-thinking      (8 packs /150, reasoning OFF, ~45-60 min)
#        =on   → quality-test.sh --full --enable-thinking   (8 packs /150, reasoning ON,  ~60-90 min)
#        =both → both passes (the production-promotion gate)
#   4. soak-test.sh fresh-mode — stability over 50 turns (~15-20 min)
#
# DEFAULT (no --with-8pack-thinking) = fast structural gates only (verify +
# bench + stress + soak, ~35-45 min). The long 8-pack is opt-in: new-model
# promotion passes --with-8pack-thinking=both; a quick "does it boot / serve /
# recall / soak" re-check needs no flag.
#
# verify-full is a HARD GATE: if the endpoint isn't functional we abort before
# the run rather than benching a broken server. Bypass with --skip verify-full
# (e.g. you just ran it while tuning); --resume skips it too.
# NOTE: =off forces --no-thinking (all 8 packs think-OFF) for a clean
# with/without-reasoning A/B — not the pack-default mixed mode. The =on / =both
# leg only scores correctly if the server PARSES reasoning — boot the compose
# with --reasoning on (REASONING=on) so <think> goes to reasoning_content, not
# the graded answer.
#
# All artifacts land in results/rebench/<tag>/. Run twice on different models
# (e.g. one Qwen leg, one Gemma leg) to assemble a matched-config head-to-head.
#
# Usage:
#   bash scripts/rebench-full.sh                      # fast structural gates only (no 8-pack)
#   bash scripts/rebench-full.sh --with-8pack-thinking=both  # + full 8-pack off+on (promotion gate)
#   bash scripts/rebench-full.sh --with-8pack-thinking=off   # + 8-pack, reasoning OFF only
#   bash scripts/rebench-full.sh --with-8pack-thinking=on    # + 8-pack, reasoning ON only
#   bash scripts/rebench-full.sh --tag qwen-int8      # explicit tag
#   bash scripts/rebench-full.sh --skip soak          # skip phases (CSV)
#   bash scripts/rebench-full.sh --resume             # skip steps that have
#                                                       artifacts already
#
# Endpoint-first mode (for non-Docker engines: llama-swap, ramalama, host-
# build llama-server, or any OpenAI-compatible server):
#   bash scripts/rebench-full.sh \
#     --url http://192.168.1.50:8887 \
#     --model 'Qwen3.6-27B MTP ik_llama:instruct' \
#     --engine llama-cpp                              # vllm|llama-cpp|sglang|other
#
#   When --url is passed, autodetect is skipped and the chained scripts run
#   in host-only mode (CONTAINER=none) — no docker logs / docker inspect
#   scrapes, which are graceful no-ops when absent.
#
# Env overrides (rarely needed — preflight auto-detects when running our composes):
#   URL                 endpoint (default: auto-detect from running container)
#   MODEL               served-model-name (default: GET /v1/models)
#   TAG                 output-dir basename (default: ${MODEL}-YYYYMMDD-HHMM)
#   OUT_DIR             override the output directory
#   SOAK_SESSIONS       passed through to soak-test.sh (default: 10 —
#                       halved from the 20-session default to keep total
#                       runtime ~1.75-2 hr per leg; bump to 20 for the
#                       canonical stability matrix when validating new
#                       compose paths.)
#   SOAK_TURNS          passed through to soak-test.sh (default: 5)
#   SAMPLING_FROM_SERVER
#                       Set to 1 to inherit sampling from the serving config
#                       instead of the pack's default temp=0. Passed through
#                       to quality-test.sh --sampling-from-server. Useful when
#                       the compose encodes the model's recommended sampling
#                       (e.g. Qwopus temp=0.8). Tags runs as non-canonical.
#   ENABLE_THINKING
#                       Set to 1 to pass request-level enable_thinking=true
#                       through bench.sh. (The 8-pack quality passes are now
#                       controlled by --with-8pack-thinking — =off forces
#                       --no-thinking, =on forces --enable-thinking — not by
#                       this env var. #338.)
#   THINKING_MAX_TOKENS
#                       Optional thinking budget forwarded to the 8-pack
#                       reasoning-ON pass (--with-8pack-thinking=on|both).
#   MAX_TOKENS          Optional completion budget forwarded to BOTH 8-pack
#                       passes (off + on) as quality-test.sh --max-tokens —
#                       overrides the per-pack ~1024 default. Raise for verbose
#                       models that self-truncate the deterministic packs
#                       (finish_reason=length before the final answer).
#

set -euo pipefail

# --- canonical cwd ----------------------------------------------------------
# This is THE fix for the recurring `scripts/X.sh: No such file or directory`
# bug — always resolve to the repo root regardless of where the user invokes
# from.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- args -------------------------------------------------------------------
SKIP_CSV=""
RESUME=0
TAG_OVERRIDE=""
URL_FLAG=""
MODEL_FLAG=""
ENGINE_FLAG=""
WITH_8PACK=""   # #338: 8-pack quality opt-in — "" (omit)=skip | off | on | both
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)     SKIP_CSV="$2"; shift 2 ;;
    --tag)      TAG_OVERRIDE="$2"; shift 2 ;;
    --resume)   RESUME=1; shift ;;
    --url)      URL_FLAG="$2"; shift 2 ;;
    --model)    MODEL_FLAG="$2"; shift 2 ;;
    --engine)   ENGINE_FLAG="$2"; shift 2 ;;
    --with-8pack-thinking)    WITH_8PACK="off"; shift ;;
    --with-8pack-thinking=*)  WITH_8PACK="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,55p' "$0"
      exit 0
      ;;
    *)
      echo "✗ unknown arg: $1 (see --help)" >&2
      exit 2
      ;;
  esac
done

# --- endpoint-first mode ----------------------------------------------------
# When --url is passed, the user is targeting an OpenAI-compatible endpoint
# that may not be one of our pre-baked containers (llama-swap, ramalama,
# host-build llama-server, raw vLLM, etc). Skip container autodetect and
# tell the chained scripts to run in host mode (CONTAINER=none).
if [[ -n "$URL_FLAG" ]]; then
  export URL="$URL_FLAG"
  export PREFLIGHT_NO_AUTODETECT=1
  export CONTAINER="none"
  [[ -n "$MODEL_FLAG" ]] && export MODEL="$MODEL_FLAG"
  [[ -n "$ENGINE_FLAG" ]] && export ENGINE_KIND="$ENGINE_FLAG"
fi

skip_step() {
  IFS=',' read -ra SKIPS <<< "$SKIP_CSV"
  for s in "${SKIPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

# --- #338: 8-pack quality is opt-in via --with-8pack-thinking[=off|on|both] -
# Default (flag omitted) = skip both quality passes (fast structural gates only).
RUN_8PACK_OFF=0
RUN_8PACK_ON=0
case "${WITH_8PACK}" in
  "")    ;;                                  # omitted → skip the 8-pack entirely
  off)   RUN_8PACK_OFF=1 ;;
  on)    RUN_8PACK_ON=1 ;;
  both)  RUN_8PACK_OFF=1; RUN_8PACK_ON=1 ;;
  *) echo "✗ --with-8pack-thinking must be off|on|both (got: '${WITH_8PACK}')" >&2; exit 2 ;;
esac

# --- endpoint + model auto-detect ------------------------------------------
# Source preflight if available; it sets URL + CONTAINER from running compose.
# Skipped silently when PREFLIGHT_NO_AUTODETECT=1 (set above by --url).
if [[ -f "$ROOT_DIR/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "$ROOT_DIR/scripts/preflight.sh"
  preflight_autodetect_endpoint
fi
URL="${URL:-http://localhost:8010}"

if ! curl -sf -m 5 "$URL/v1/models" >/dev/null 2>&1; then
  echo "✗ endpoint $URL/v1/models not responding" >&2
  if [[ -n "$URL_FLAG" ]]; then
    echo "  --url '$URL_FLAG' not reachable; check the host/port + model server health." >&2
  else
    echo "  start a compose first: gpu-mode <mode>" >&2
    echo "  or pass an external endpoint:" >&2
    echo "    bash scripts/rebench-full.sh --url http://HOST:PORT --model NAME --engine vllm|llama-cpp|sglang|other" >&2
  fi
  exit 1
fi

# Resolve actual served model id — eliminates MODEL=qwen vs MODEL=gemma
# typos that produce HTTP 404 from served-model-name mismatch.
DETECTED_MODEL=$(curl -sf -m 5 "$URL/v1/models" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
  || echo "")
if [[ -n "$DETECTED_MODEL" && -z "${MODEL:-}" ]]; then
  MODEL="$DETECTED_MODEL"
fi
if [[ -z "${MODEL:-}" ]]; then
  echo "✗ could not detect served model. Set MODEL=<name> explicitly." >&2
  exit 1
fi

# --- output dir -------------------------------------------------------------
TAG="${TAG_OVERRIDE:-${TAG:-${MODEL//[^a-z0-9._-]/-}-$(date +%Y%m%d-%H%M)}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/rebench/$TAG}"
mkdir -p "$OUT_DIR"

# --- env that benchlocal-cli + quality-test need ---------------------------
# Auto-set the localhost resolve flag so hermes sandbox can reach host vLLM.
# Idempotent — don't overwrite an explicit user value.
if [[ -z "${BENCHLOCAL_HERMES_RESOLVE_LOCALHOST:-}" ]] \
   && [[ "$URL" =~ ^https?://(localhost|127\.|\[::1\]) ]]; then
  export BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1
fi

# --- preamble -----------------------------------------------------------------
echo "==============================================================="
echo " rebench-full.sh"
echo "==============================================================="
echo "  endpoint:    $URL"
echo "  model:       $MODEL"
echo "  out dir:     $OUT_DIR"
echo "  resume:      $RESUME"
echo "  skips:       ${SKIP_CSV:-(none)}"
echo "  8-pack:      ${WITH_8PACK:-(skipped — opt-in via --with-8pack-thinking=off|on|both)}"
echo "  hermes env:  BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=${BENCHLOCAL_HERMES_RESOLVE_LOCALHOST:-0}"
echo "  thinking:    ${ENABLE_THINKING:-0}${THINKING_MAX_TOKENS:+ (max_tokens=$THINKING_MAX_TOKENS)}"
echo "==============================================================="
date +"  started:     %Y-%m-%dT%H:%M:%SZ" -u
echo

# --- capture container snapshot (one-shot, used by rebench-report.py) ------
# Picks the first vllm-*/llama-cpp-*/sglang-* container — same heuristic
# preflight uses. Silently no-ops in endpoint-first mode (CONTAINER=none).
if [[ "${CONTAINER:-}" != "none" ]] && command -v docker >/dev/null 2>&1; then
  CONTAINER_NAME=$(docker ps --format '{{.Names}}' 2>/dev/null \
    | grep -E '^(vllm-|llama-cpp-|sglang-)' | head -1 || true)
  if [[ -n "$CONTAINER_NAME" ]]; then
    docker inspect "$CONTAINER_NAME" > "$OUT_DIR/container-config.json" 2>/dev/null || true
    # Boot log: capture lines that the report parser needs (KV pool size,
    # max concurrency, model load footprint, MTP detection). Trimmed to keep
    # the file small; full container log is still available via `docker logs`.
    docker logs "$CONTAINER_NAME" 2>&1 \
      | grep -E "GPU KV cache size|Maximum concurrency|Available KV cache memory|Model loading took|Detected MTP|kv_cache_dtype|num_speculative_tokens" \
      > "$OUT_DIR/vllm-boot.log" 2>/dev/null || true
  fi
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader > "$OUT_DIR/gpu-state-start.log" 2>/dev/null || true

# --- rig.txt: hostname, GPUs (nvidia-smi -L), per-card power cap ----------
{
  echo "hostname: $(hostname)"
  nvidia-smi -L 2>/dev/null || true
  cap_line=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "")
  if [[ -n "$cap_line" ]]; then
    echo "power_cap_w: ${cap_line%% *}"
  fi
} > "$OUT_DIR/rig.txt" 2>/dev/null || true

# --- timings.json: per-phase wall-clock --------------------------------------
TIMINGS_FILE="$OUT_DIR/timings.json"
echo "{}" > "$TIMINGS_FILE"
record_timing() {
  local phase="$1" secs="$2"
  python3 -c "
import json, sys
p = '$TIMINGS_FILE'
d = json.load(open(p))
d['$phase'] = $secs
json.dump(d, open(p, 'w'))
" 2>/dev/null || true
}

# --- helpers ----------------------------------------------------------------
have_artifact() { [[ -s "$1" ]]; }

run_step() {
  local name="$1" artifact="$2"
  shift 2
  if skip_step "$name"; then
    echo "[$name] skipped (--skip)"
    return 0
  fi
  if [[ "$RESUME" == "1" ]] && have_artifact "$artifact"; then
    echo "[$name] skipped (--resume found $artifact)"
    return 0
  fi
  echo "[$name] running…"
  local t0=$(date +%s)
  # tee: stream live to console (so benchlocal-cli [N/M] progress is visible)
  # AND capture to the per-step log. rc from PIPESTATUS[0], not tee's exit.
  "$@" 2>&1 | tee "$OUT_DIR/$name.log"
  local rc=${PIPESTATUS[0]} dt=$(( $(date +%s) - t0 ))
  record_timing "$name" "$dt"
  if [[ $rc -eq 0 ]]; then
    echo "[$name] ✓ ${dt}s — log: $OUT_DIR/$name.log"
  else
    echo "[$name] ✗ ${dt}s — failed (rc=$rc) — log: $OUT_DIR/$name.log" >&2
    return $rc
  fi
}

# copy the most recent quality.json from the shared results dir into our
# per-tag dir; that's where quality-test.sh writes its --save-json output.
snapshot_quality_json() {
  local target="$1"
  local src
  src="$(ls -t "$ROOT_DIR"/results/quality/quality-*.json 2>/dev/null | head -1)"
  if [[ -n "$src" ]]; then
    cp "$src" "$target"
  fi
}

# --- step 0: verify-full (fail-fast functional preflight) -------------------
# ~2 min smoke (boots / serves / tool-calls / streams / coherent output). If the
# endpoint isn't functional there's no point spending hours on bench → soak, so
# we ABORT here. Skipped cleanly by --skip verify-full or --resume (run_step
# returns 0 in both cases → no abort); only a real run-and-fail aborts.
if ! URL="$URL" MODEL="$MODEL" \
     run_step verify-full "$OUT_DIR/verify-full.log" \
       bash "$ROOT_DIR/scripts/verify-full.sh"; then
  echo "[rebench] ✗ verify-full failed — endpoint not functional; aborting before the multi-hour run." >&2
  echo "          Fix the server, or re-run with --skip verify-full to bypass the preflight." >&2
  exit 1
fi

# --- step 1: bench ----------------------------------------------------------
URL="$URL" MODEL="$MODEL" RUNS="${RUNS:-3}" WARMUPS="${WARMUPS:-1}" \
  ENABLE_THINKING="${ENABLE_THINKING:-0}" \
  run_step bench "$OUT_DIR/bench.log" \
    bash "$ROOT_DIR/scripts/bench.sh" || true

# --- step 2: verify-stress --------------------------------------------------
URL="$URL" MODEL="$MODEL" \
  run_step verify-stress "$OUT_DIR/verify-stress.log" \
    bash "$ROOT_DIR/scripts/verify-stress.sh" || true

# --- step 3: 8-pack quality, reasoning OFF (opt-in: --with-8pack-thinking=off|both) --
# Forces --no-thinking (all 8 packs think-OFF) for a clean with/without-reasoning
# A/B against step 4 — NOT the pack-default mixed mode. Skipped unless opted in. #338.
if [[ "$RUN_8PACK_OFF" == "1" ]]; then
  URL="$URL" MODEL="$MODEL" \
    SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-0}" \
    MAX_TOKENS="${MAX_TOKENS:-}" \
    NO_THINKING=1 \
    run_step quality-full "$OUT_DIR/quality-full.log" \
      bash "$ROOT_DIR/scripts/quality-test.sh" --full --no-thinking --sandbox-log-dir "$OUT_DIR"
  snapshot_quality_json "$OUT_DIR/quality-full.json"
else
  echo "[quality-full] skipped — 8-pack is opt-in (pass --with-8pack-thinking=off|both)"
fi

# --- step 4: 8-pack quality, reasoning ON (opt-in: --with-8pack-thinking=on|both) --
# enable_thinking is sent per-request via benchlocal --enable-thinking. Scores
# correctly only if the server PARSES reasoning (boot --reasoning on), else
# <think> leaks into the graded answer. Skipped unless opted in. #338.
if [[ "$RUN_8PACK_ON" == "1" ]]; then
  URL="$URL" MODEL="$MODEL" \
    SAMPLING_FROM_SERVER="${SAMPLING_FROM_SERVER:-0}" \
    THINKING_MAX_TOKENS="${THINKING_MAX_TOKENS:-}" \
    MAX_TOKENS="${MAX_TOKENS:-}" \
    ENABLE_THINKING=1 \
    run_step quality-thinking "$OUT_DIR/quality-full-thinking.log" \
      bash "$ROOT_DIR/scripts/quality-test.sh" --full --enable-thinking --sandbox-log-dir "$OUT_DIR"
  snapshot_quality_json "$OUT_DIR/quality-full-thinking.json"
else
  echo "[quality-thinking] skipped — 8-pack is opt-in (pass --with-8pack-thinking=on|both)"
fi

# --- step 5: soak-test ------------------------------------------------------
URL="$URL" MODEL="$MODEL" \
  SOAK_MODE="${SOAK_MODE:-fresh}" \
  SOAK_OUTPUT="$OUT_DIR/soak-artifacts" \
  SESSIONS="${SOAK_SESSIONS:-10}" \
  TURNS="${SOAK_TURNS:-5}" \
  run_step soak "$OUT_DIR/soak.log" \
    bash "$ROOT_DIR/scripts/soak-test.sh"

# --- final GPU state snapshot ----------------------------------------------
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader > "$OUT_DIR/gpu-state-end.log" 2>/dev/null || true

# --- synthesize REPORT.md ---------------------------------------------------
echo
echo "[report] synthesizing REPORT.md…"
if command -v python3 >/dev/null 2>&1 && [[ -f "$ROOT_DIR/scripts/rebench-report.py" ]]; then
  python3 "$ROOT_DIR/scripts/rebench-report.py" "$OUT_DIR" || \
    echo "[report] ⚠ rebench-report.py failed — raw artifacts still available." >&2
else
  echo "[report] ⚠ python3 or rebench-report.py missing — skipping REPORT.md." >&2
fi

# --- summary ----------------------------------------------------------------
echo
echo "==============================================================="
echo " rebench complete"
echo "==============================================================="
date +"  finished:    %Y-%m-%dT%H:%M:%SZ" -u
echo "  artifacts:   $OUT_DIR"
if [[ -f "$OUT_DIR/REPORT.md" ]]; then
  echo "  report:      $OUT_DIR/REPORT.md"
fi
echo
echo "Headline pulls (grep through the logs):"
echo "  TPS:           grep -E 'mean=|decode_TPS' $OUT_DIR/bench.log"
echo "  verify-stress: tail -5 $OUT_DIR/verify-stress.log"
[[ -f "$OUT_DIR/quality-full.log" ]] && \
  echo "  quality(off):  grep 'TOTAL' $OUT_DIR/quality-full.log"
[[ -f "$OUT_DIR/quality-full-thinking.log" ]] && \
  echo "  quality(on):   grep 'TOTAL' $OUT_DIR/quality-full-thinking.log"
[[ "$RUN_8PACK_OFF$RUN_8PACK_ON" == "00" ]] && \
  echo "  8-pack:        skipped (opt-in: re-run with --with-8pack-thinking=off|on|both)"
echo "  soak:          grep -E 'verdict|silent_empty|p50_decode' $OUT_DIR/soak.log"
echo
echo "To submit your numbers (review then PR):"
echo "  bash scripts/submit-bench.sh --tag $TAG"

# --- measurement record + baseline-induction prompt (catalog-baselines slice 2) ---
# Auto-emit the #249 measurement record into the per-rig corpus when the served
# container EXACT-matches a registry slug (identity semantics — a port/substring
# match is a shape guess and must not stamp another slug's record; see the c3
# detect layer). BYO/swap serves have no registry identity → skipped with a note.
if [[ -f "$OUT_DIR/bench.log" ]] && command -v python3 >/dev/null 2>&1; then
  OUT_DIR="$OUT_DIR" TAG="$TAG" python3 - <<'PY_RECORD' || true
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(Path.cwd()))
try:
    from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
    from scripts.lib.profiles.measurement_record import (
        build_record, parse_bench_output, write_record,
    )
except Exception as exc:  # pragma: no cover - env without the profiles tree
    print(f"  record:      skipped (profiles unavailable: {exc})")
    raise SystemExit(0)

out_dir = Path(os.environ["OUT_DIR"])
tag = os.environ["TAG"]

# EXACT container -> slug (never port/substring).
slug = None
try:
    names = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10,
    ).stdout.split()
    norm = {n.replace("_", "-") for n in names}
    for s, e in COMPOSE_REGISTRY.items():
        # container name = the compose's container_name default
        try:
            txt = Path(e["compose_path"]).read_text(errors="replace")
        except OSError:
            continue
        m = re.search(r'container_name:\s*"?(?:\$\{[^:}]*:-)?([A-Za-z0-9._-]+)\}?"?', txt)
        if m and m.group(1).replace("_", "-") in norm:
            slug = s
            break
except Exception:
    pass

if not slug:
    print("  record:      skipped — no running container exact-matches a registry slug")
    raise SystemExit(0)


def _quality(path: Path):
    try:
        q = json.loads(path.read_text())
        p = sum(int(x.get("passed") or 0) for x in q.get("packs") or [])
        t = sum(int(x.get("total") or 0) for x in q.get("packs") or [])
        return f"{p}/{t}" if t else None
    except (OSError, ValueError, TypeError):
        return None


# Fingerprint enrichment — the cohort keys the corpus joins/staleness rely on.
engine_pin = None
try:
    from scripts.lib.profiles.compat import load_profiles
    from scripts.lib.profiles.launch_compat import ProfileError, resolve_variant_pin

    exports = resolve_variant_pin(load_profiles(), slug)
    if "VLLM_NIGHTLY_SHA" not in exports:
        engine_pin = next(iter(exports.values()))
except Exception:
    pass
if not engine_pin:
    try:
        txt = Path(COMPOSE_REGISTRY[slug]["compose_path"]).read_text(errors="replace")
        m = re.search(r'^\s*image:\s*["\x27]?(?:\$\{[A-Z_0-9]+:-)?([^\s}"\x27]+)\}?', txt, re.M)
        engine_pin = m.group(1) if m else None
    except OSError:
        pass

hardware = None
power_cap = None
try:
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    ).stdout.strip().splitlines()
    if rows:
        name = rows[0].split(",")[0].strip()
        m = re.search(r"RTX\s*(\d{4})", name)
        hardware = f"rtx-{m.group(1)}" if m else None
        power_cap = round(float(rows[0].split(",")[1]))
except Exception:
    pass

try:
    metrics = parse_bench_output((out_dir / "bench.log").read_text(errors="replace"))
    soak = "not-run"
    soak_log = out_dir / "soak.log"
    if soak_log.is_file():
        soak = "pass" if re.search(r"verdict\s+PASS", soak_log.read_text(errors="replace")) else "fail"
    rec = build_record(
        tag=slug, bench_metrics=metrics,
        hardware=hardware, engine_pin=engine_pin, power_cap_w=power_cap,
        smoke_status="pass",   # rebench aborts at step 0 unless verify-full passed
        soak_status=soak,
    )
    q_off = _quality(out_dir / "quality-full.json")
    q_on = _quality(out_dir / "quality-full-thinking.json")
    if q_off:
        rec["measured_extensions"]["quality_8pk"] = q_off
    if q_on:
        rec["measured_extensions"]["quality_8pk_think_on"] = q_on
    rec["_rebench_tag"] = tag
    path = write_record(rec)
    print(f"  record:      corpus record appended -> {path}  (slug {slug})")
except Exception as exc:
    print(f"  record:      skipped ({exc})")
    raise SystemExit(0)

print()
print("To induct these numbers as the slug's SHIPPED baseline (review then PR):")
print(f"  bash scripts/catalog-baseline.sh {slug} --from-tag {tag}")
PY_RECORD
fi
