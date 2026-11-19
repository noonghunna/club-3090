#!/usr/bin/env bash
# test-report-qengine — report.sh reads a QEngine container the way it reads Strata's (#2103).
#
# #2103's `report.sh --full` on QEngine 1.3.6 showed the gaps:
#   - "Engine configuration (resolved)" said "engine kind `qengine` has no known startup dump to read",
#     although QEngine states its pool, host-RAM split, parallel mode and thinking budget at boot, and
#     the env allowlist withheld every FLASH_ / QE_ knob, so a docker-run QEngine showed no config;
#   - QEngine ran the vLLM probe set, whose greps find nothing in its log: no runtime summary, no tail;
#   - "First 200 lines of docker logs" on a 55-65 line boot was mostly per-request lines, which the
#     reporter had to trim to fit GitHub's issue size limit.
# Drives the REAL report.sh (the report-harness fixture, docker stubbed, logs from fixtures shaped
# like the 27B and Flash-Next boots) and asserts per section:
#   1. 27B: the resolved dump has its pool / per-card / host-RAM / parallel-mode lines and leaves out
#      its GPU layout lines and its raw `env:` line; the env block shows FLASH_ / QE_ knobs and
#      withholds a planted QE_API_KEY; the engine-state section has the acceptance length (the 1-step
#      warm-up left out), the latest split and the recent requests; the boot block stops at the graph
#      warm-up; the tail has the post-boot warning and no per-request line; the interconnect line
#      that the vLLM probe set prints for QEngine is still there;
#   2. Flash-Next: its [qflash]-prefixed per-card / slot / host-RAM lines, and its expert-cache line;
#   3. an idle server (boot only): each block says so instead of printing an empty value;
#   4. a boot that never reached its graph warm-up: the boot block shows it whole, the tail says so;
#   5. negative control: a vLLM container keeps "Full boot log (first 200 lines)" and gets no QEngine
#      section.
# ⚠️ Against the pre-#2103 tree this must FAIL (legs 1-4).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # #779

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0
ok() { PASS=$((PASS+1)); }
no() { echo "FAIL: $1" >&2; FAIL=$((FAIL+1)); }
has()   { if [[ "$2" == *"$3"* ]]; then ok; else no "$1: missing '$3'"; fi; }
hasnt() { if [[ "$2" != *"$3"* ]]; then ok; else no "$1: must NOT contain '$3'"; fi; }

# shellcheck source=fixtures/report-harness/report-env.sh
source "${ROOT_DIR}/scripts/tests/fixtures/report-harness/report-env.sh"
report_env_init "$ROOT_DIR"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; report_env_cleanup' EXIT

report_docker_stub <<'STUB'
#!/usr/bin/env bash
# docker for this test: inspect / ps / logs from fixtures, nothing else answers.
case "${1:-}" in
  info)    exit 0 ;;
  ps)      [[ "${*}" == *"-a"* ]] && exit 0
           [[ -n "${RC_CONTAINER:-}" ]] && echo "${RC_CONTAINER}"; exit 0 ;;
  logs)    [[ -n "${RC_LOGS:-}" && -f "${RC_LOGS}" ]] && cat "${RC_LOGS}"; exit 0 ;;
  inspect) [[ "${2:-}" == "${RC_CONTAINER:-}" && -f "${RC_INSPECT:-}" ]] || exit 1
           case "${4:-}" in
             '{{json .}}')         cat "${RC_INSPECT}" ;;
             '{{.Config.Image}}')  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["Config"]["Image"])' "${RC_INSPECT}" ;;
             *) exit 1 ;;
           esac ;;
  *)       exit 1 ;;
esac
STUB

run_report() {  # <container> <log file> <inspect json> -> REPORT_OUT
  REPORT_OUT="$(env PATH="${REPORT_FAKE_BIN}:$PATH" HOME="${REPORT_ENV_DIR}/home" REPORT_FULL_CALIBRATION=0 \
    RC_CONTAINER="$1" RC_LOGS="$2" RC_INSPECT="$3" \
    bash "${REPORT_FAKE_ROOT}/scripts/report.sh" --container "$1" 2>&1)"
}
# block <heading> — the text under one "## " / "### " heading, up to the next heading of the same
# level or higher (a "## " section keeps its own "### " sub-headings)
block() {
  printf '%s\n' "$REPORT_OUT" | awk -v h="$1" '
    BEGIN { lvl = length(h) - length(substr(h, index(h, " "))) }
    index($0, h) == 1 { on = 1; next }
    on && match($0, /^#+ /) && RLENGTH - 1 <= lvl { exit }
    on'
}
inspect_json() {  # <out> <image> — a docker-run container: no compose labels, env-only config
  python3 - "$1" "$2" <<'PY'
import json, sys
json.dump({"Config": {"Image": sys.argv[2], "Entrypoint": ["/usr/local/bin/qengine-entrypoint"], "Cmd": None,
                      "Env": ["CUDA_VISIBLE_DEVICES=0,1", "FLASH_MAX_SEQS=8", "QE_REASONING_EFFORT=low",
                              "QE_API_KEY=club3090-planted-qe-key", "PATH=/usr/bin"], "Labels": {}},
           "State": {"Status": "running"}}, open(sys.argv[1], "w", encoding="utf-8"))
PY
}
QE_IMG="ghcr.io/bpx03/qengine:1.3.6-vision"
inspect_json "$TMP/qe.json" "$QE_IMG"

# --- fixtures: the two boot shapes, lines as QEngine 1.3.x prints them ---------------------------------
cat > "$TMP/qe27.log" <<'LOG'
[thinking] default enable_thinking=true (a request's own setting wins)
==== qengine server start: QEngine 1.3.6, pid 1, build 444b2e4 (/app), log /root/.cache/qengine/logs/qwen3.8-27b.log
argv: server/app.py --host 0.0.0.0 --port 8080 --devices 0,1 --model /app/models/Qwen3.8-27B-INT4 --served-name qwen3.8-27b --think 1
env: CUDA_VISIBLE_DEVICES=0,1 QE_MAX_SEQS=32 QE_SENTINEL=club3090-envline-sentinel
QEngine 1.3.6 (build 444b2e4)
model qwen38-27b (qwen3_5), tokenizer /app/models/Qwen3.8-27B-INT4/tokenizer.json
sampling defaults, thinking (when the request has none): temperature 1.00, top_p 0.95, top_k 20, min_p 0.00, presence_penalty 0.00
GPU0: attn q[0,12) o[0,12) kv[0,2) | deltanet v[0,24) k[0,8) | mlp [0,8704) | vocab [0,124160)
P2P: decode all-reduce GPU to GPU (peer access between all 2 GPUs)
[comm] prefill transfers: copy engine (QE_COMM=auto, auto)
KV pool: 351232 tokens shared by 32 sequence slots (max context 262144)
GPU0: weights 7.17 GB, cache+act 12.92 GB
GPU1: weights 7.17 GB, cache+act 12.92 GB
host RAM plan: MemTotal 257.6 GB, MemAvailable 243.0 GB, reserve 20.6 GB (OS + desktop), server base 6.0 GB, engine buffers 7.9 GB; KV archive 34.4 GB + pinned prefixes 8.6 GB; reduced: nothing
context: requested 262144 tokens, KV pool 351232 tokens, served 262144 tokens per conversation
engine parallel mode: tp ()
thinking budget by effort: low 4096, medium 16384, xhigh 32768, default none
monitoring: http://0.0.0.0:9393/  (qetop, /api/stats, /metrics)
graph warm-up: 0.9s
archive pool: 42.9 GB pinned in 6.0 s
WARNING empty reply: slot 0, conversation 1 (session b4b27974): reasoning only, no text and no tool call; finish length
[chat] slot 0 prompt 41 (cached 0, prefill 790 tok/s) | gen 1 @ 1316.5 tok/s | length
[stats] prompt eval time =      51.51 ms /    41 tokens (    1.26 ms per token,   795.94 tokens per second)
[stats] SpecDecoding metrics: Mean acceptance length: 1.00, Accepted throughput: 0.00 tokens/s, Drafted throughput: 63.64 tokens/s, Accepted: 0 tokens, Drafted: 3 tokens, Avg Draft acceptance rate: 0.0% (MTP 0/3, lookup 0/0, 1 steps)
[chat] slot 1 prompt 46 (cached 0, prefill 801 tok/s) | gen 128 @ 110.2 tok/s | length
[stats]        eval time =    1161.30 ms /   128 tokens (    9.07 ms per token,   110.22 tokens per second)
[stats] SpecDecoding metrics: Mean acceptance length: 3.63, Accepted throughput: 79.22 tokens/s, Drafted throughput: 190.30 tokens/s, Accepted: 92 tokens, Drafted: 221 tokens, Avg Draft acceptance rate: 41.6% (MTP 92/221, lookup 0/0, 35 steps)
[stats] SpecDecoding metrics: Mean acceptance length: 3.41, Accepted throughput: 70.10 tokens/s, Drafted throughput: 180.00 tokens/s, Accepted: 80 tokens, Drafted: 210 tokens, Avg Draft acceptance rate: 38.1% (MTP 80/210, lookup 0/0, 33 steps)
[messages] slot 0 prompt 47 (cached 30, prefill 632 tok/s) | gen 6 @ 94.0 tok/s | stop
LOG
cat > "$TMP/qefn.log" <<'LOG'
==== qengine server start: QEngine 1.3.6, pid 1, build 444b2e4 (/app), log /root/.cache/qengine/logs/qwen3.8-flash-next.log
argv: server/app.py --host 0.0.0.0 --port 8080 --devices 0,1 --model /app/models/Qwen3.8-Flash-Next-QE-INT4 --mtp-k 5
model qwen38-flash-next (qwen4_exp), tokenizer /app/models/Qwen3.8-Flash-Next-QE-INT4/tokenizer.json
[comm] prefill transfers: P2P SM loads / stores, pairs 1<-0:pull 0<-1:pull (FLASH_COMM=auto, auto: P2P tax 1.21)
[qflash] GPU0: 23.81 GB weights+state, 15.46 GB cold experts in host memory, 0.41 GiB free
[qflash] GPU1: 23.06 GB weights+state, 16.22 GB cold experts in host memory, 0.41 GiB free
[qflash] host RAM plan: MemTotal 257.6 GB, MemAvailable 243.6 GB, reserve 20.6 GB (OS + desktop), server base 6.0 GB, engine buffers 1.3 GB; experts in host memory 31.7 GB (estimate 32.8 GB), n-gram table 51.8 GB pinned
[qflash] 8 sequence slots over a pool of 256 periods; archive: 48.0 GB host budget (pinning in the background), checkpoints every 8192 tokens
[qflash 115s] ready, context 262144
context: requested 262144 tokens, KV pool 262144 tokens, served 262144 tokens per conversation
engine parallel mode: tp (auto: peer access between the cards: TP)
graph warm-up: 0.0s
[chat] slot 0 prompt 54 (cached 0, prefill 331 tok/s) | gen 6 @ 134.2 tok/s | stop
[stats] SpecDecoding metrics: Mean acceptance length: 3.27, Accepted throughput: 51.77 tokens/s, Drafted throughput: 129.14 tokens/s, Accepted: 93 tokens, Drafted: 232 tokens, Avg Draft acceptance rate: 40.1% (MTP 45/66, lookup 48/166, 41 steps)
[qflash] expert cache: 2 decode steps, misses/step mean 597.50 p50 600 p90 600 max 600 (all cards), cold-group hit rate 42.7% decode (889/2084) 33.7% prefill (3599/10677); copies 77, evicted kv 26
LOG

# --- 1. the 27B ------------------------------------------------------------------------------------
run_report qengine-qwen38-27b-dual "$TMP/qe27.log" "$TMP/qe.json"
RES="$(block '## Engine configuration (resolved)')"
has   "27B resolved: the startup dump was read"        "$RES" "KV pool: 351232 tokens shared by 32 sequence slots"
has   "27B resolved: per-card split"                   "$RES" "GPU0: weights 7.17 GB, cache+act 12.92 GB"
has   "27B resolved: host RAM plan"                    "$RES" "host RAM plan: MemTotal 257.6 GB"
has   "27B resolved: served context"                   "$RES" "context: requested 262144 tokens, KV pool 351232 tokens"
has   "27B resolved: parallel mode"                    "$RES" "engine parallel mode: tp ()"
has   "27B resolved: thinking budget"                  "$RES" "thinking budget by effort: low 4096"
has   "27B resolved: argv"                             "$RES" "argv: server/app.py"
hasnt "27B resolved: no 'no known startup dump'"       "$RES" "no known startup dump"
hasnt "27B resolved: the GPU layout line stays out"    "$RES" "GPU0: attn q["
hasnt "27B resolved: the raw env: line stays out"      "$RES" "club3090-envline-sentinel"
has   "27B resolved: env block shows a FLASH_ knob"    "$RES" "FLASH_MAX_SEQS=8"
has   "27B resolved: env block shows a QE_ knob"       "$RES" "QE_REASONING_EFFORT=low"
hasnt "27B resolved: a planted QE_API_KEY is withheld" "$RES" "club3090-planted-qe-key"
ST="$(block '### Container engine state (QEngine)')"
has   "27B state: acceptance, the 1-step warm-up left out" "$ST" "mean acceptance length 3.52 over the last 2 multi-step line(s)"
has   "27B state: the latest split"                    "$ST" "latest split MTP 80/210, lookup 0/0"
has   "27B state: no expert cache, said so"            "$ST" "the 27B has no expert cache"
has   "27B state: the recent requests"                 "$ST" "[chat] slot 1 prompt 46"
has   "27B state: a /v1/messages request counts too"   "$ST" "[messages] slot 0 prompt 47"
BOOT="$(block "### Boot log (up to QEngine's graph warm-up)")"
has   "27B boot block: ends at the graph warm-up"      "$BOOT" "graph warm-up: 0.9s"
hasnt "27B boot block: no per-request line"            "$BOOT" "[chat] slot 0 prompt 41"
hasnt "27B boot block: no post-boot line"              "$BOOT" "archive pool: 42.9 GB"
TAIL="$(block '### Engine log since boot')"
has   "27B tail: the post-boot warning"                "$TAIL" "WARNING empty reply: slot 0"
has   "27B tail: the archive line"                     "$TAIL" "archive pool: 42.9 GB pinned"
hasnt "27B tail: no [stats] line"                      "$TAIL" "[stats] SpecDecoding"
hasnt "27B tail: no [chat] line"                       "$TAIL" "[chat] slot"
hasnt "27B tail: no [messages] line"                   "$TAIL" "[messages] slot"
hasnt "27B tail: no boot line"                         "$TAIL" "KV pool: 351232"
has   "27B: the interconnect line still runs for QEngine" "$REPORT_OUT" "QEngine runs its own collectives (no NCCL). Its log: decode all-reduce GPU to GPU"
hasnt "27B: no 'Full boot log (first 200 lines)'"      "$REPORT_OUT" "### Full boot log (first 200 lines)"

# --- 2. Flash-Next -----------------------------------------------------------------------------------
run_report qengine-qwen38-flash-next-dual "$TMP/qefn.log" "$TMP/qe.json"
RES="$(block '## Engine configuration (resolved)')"
has   "Flash-Next resolved: per-card split"            "$RES" "[qflash] GPU0: 23.81 GB weights+state, 15.46 GB cold experts in host memory"
has   "Flash-Next resolved: host RAM plan"             "$RES" "[qflash] host RAM plan: MemTotal 257.6 GB"
has   "Flash-Next resolved: slots and archive"         "$RES" "[qflash] 8 sequence slots over a pool of 256 periods"
has   "Flash-Next resolved: parallel mode"             "$RES" "engine parallel mode: tp (auto: peer access between the cards: TP)"
ST="$(block '### Container engine state (QEngine)')"
has   "Flash-Next state: acceptance and split"         "$ST" "mean acceptance length 3.27 over the last 1 multi-step line(s); latest split MTP 45/66, lookup 48/166"
has   "Flash-Next state: expert cache"                 "$ST" "cold-group hit rate 42.7% decode (889/2084) 33.7% prefill"

# --- 3. an idle server: the boot and nothing else -----------------------------------------------------
awk '{ print } /^graph warm-up:/ { exit }' "$TMP/qe27.log" > "$TMP/idle.log"
run_report qengine-qwen38-27b-dual "$TMP/idle.log" "$TMP/qe.json"
ST="$(block '### Container engine state (QEngine)')"
has   "idle: no acceptance, said so"                   "$ST" "no acceptance line in the last 200 log lines"
has   "idle: no requests, said so"                     "$ST" "Recent requests:** none in the last 200 log lines"
has   "idle: the tail says nothing came after the boot" "$REPORT_OUT" "Nothing logged after the boot besides per-request lines"

# --- 4. a boot that never reached its graph warm-up ---------------------------------------------------
head -12 "$TMP/qe27.log" > "$TMP/crash.log"
echo "RuntimeError: CUDA error: out of memory" >> "$TMP/crash.log"
run_report qengine-qwen38-27b-dual "$TMP/crash.log" "$TMP/qe.json"
BOOT="$(block "### Boot log (up to QEngine's graph warm-up)")"
has   "crash: the boot block shows the whole log"      "$BOOT" "RuntimeError: CUDA error: out of memory"
has   "crash: the tail points at the boot block"       "$REPORT_OUT" "the boot never reached its graph warm-up"

# --- 5. negative control: vLLM keeps its own boot block, no QEngine section ---------------------------
printf '%s\n' "INFO 11-19 09:00:00 [api_server.py:1] vLLM API server version 0.34.0" \
  "INFO 11-19 09:00:01 [core.py:1] Initializing a V1 LLM engine" > "$TMP/vllm.log"
inspect_json "$TMP/vllm.json" "vllm/vllm-openai:v0.34.0"
run_report vllm-qwen38-27b-dual "$TMP/vllm.log" "$TMP/vllm.json"
has   "vLLM: keeps the first-200-lines block"          "$REPORT_OUT" "### Full boot log (first 200 lines)"
hasnt "vLLM: no QEngine state section"                 "$REPORT_OUT" "Container engine state (QEngine)"
hasnt "vLLM: no QEngine boot block"                    "$REPORT_OUT" "graph warm-up"

echo "PASS: $PASS  FAIL: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
echo "test-report-qengine: ok (resolved dump + env knobs, engine state, boot block, tail, idle, crash, vLLM control)"
