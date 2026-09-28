#!/usr/bin/env bash
#
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-quality-thinking-budget — guards the #1383 contract for
# `quality-test.sh --thinking-budget N`:
#
#   1. OPT-IN. Without the flag the benchlocal argv is exactly what it was:
#      no --extra-body, no derived --thinking-max-tokens, no banner. Every
#      published BENCHMARKS row was measured unbounded; a default would
#      silently break comparability with all of them.
#   2. VERIFICATION IS THE FEATURE. The budget is only sent when the wrapper
#      has POSITIVE evidence it can take effect on the serving engine:
#        llama.cpp  the container's resolved --reasoning-budget == N. The
#                   shipped composes always emit the flag as
#                   `--reasoning-budget "${REASONING_BUDGET:--1}"`, so the flag
#                   being PRESENT proves nothing — only its VALUE does.
#        vLLM       --reasoning-parser set at boot (v0.29.0 rejects the
#                   per-request field otherwise — every scenario would 400).
#        SGLang     --enable-custom-logit-processor set at boot (v0.5.20
#                   rejects the per-request processor otherwise), read from
#                   /server_info first, docker second.
#      Positive evidence that it will NOT take effect is a hard refusal with
#      the fix instruction, and THINKING_BUDGET_UNVERIFIED=1 does NOT bypass
#      it. Only the no-evidence case (no container, no readback) honours that
#      explicit acknowledgement, and then labels the run.
#   3. PER PACK CLASS. hermesagent-20 / aider-polyglot-30 make their model
#      calls from an agent INSIDE the sandbox; a per-request budget never
#      reaches them, so on vLLM/SGLang a selection that includes them is
#      refused. On llama.cpp the boot flag governs them for free.
#   4. MATCHED CLIENT CAP. --thinking-max-tokens is derived as budget +
#      headroom (default 4096) unless the caller set one ABOVE the budget; a
#      cap at or below the budget is refused (no answer headroom).
#   5. A pass-through --extra-body is MERGED with the budget's, never allowed
#      to replace it (argparse last-wins would have dropped the budget).
#
# NEGATIVE CONTROLS — each of these must be RED for a wrapper that sends the
# budget without looking: C (llama.cpp flag present, value -1), D (value
# mismatch), E (flag absent), I (vLLM without parser), L (SGLang flag false),
# J (agentic pack on a per-request engine), O (no evidence at all). Every one
# asserts exit 2 AND that benchlocal-cli was never invoked.
#
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAILED=0

assert_contains() {
  local label="$1" haystack="$2" needle="$3"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED [$label]: expected to contain:" >&2
    echo "  $needle" >&2
    echo "--- got ---" >&2
    echo "$haystack" >&2
    FAILED=1
  fi
}

assert_not_contains() {
  local label="$1" haystack="$2" needle="$3"
  if [[ "$haystack" == *"$needle"* ]]; then
    echo "ASSERTION FAILED [$label]: expected NOT to contain:" >&2
    echo "  $needle" >&2
    echo "--- got ---" >&2
    echo "$haystack" >&2
    FAILED=1
  fi
}

assert_rc() {
  local label="$1" want="$2" got="$3"
  if [[ "$want" != "$got" ]]; then
    echo "ASSERTION FAILED [$label]: expected exit $want, got $got" >&2
    echo "--- output ---" >&2
    echo "$OUT" >&2
    FAILED=1
  fi
}

# benchlocal-cli must not have been invoked on a refusal: a refusal that still
# runs the eval is the silent partial this flag exists to prevent.
assert_not_invoked() {
  local label="$1"
  if [[ -s "$ARGS_LOG" ]]; then
    echo "ASSERTION FAILED [$label]: benchlocal-cli was invoked despite the refusal:" >&2
    cat "$ARGS_LOG" >&2
    FAILED=1
  fi
}

tmp_bin="$(mktemp -d)"
tmp_work="$(mktemp -d)"
ARGS_LOG="${tmp_work}/benchlocal-args.log"     # one arg per line
ARGS_JOINED="${tmp_work}/benchlocal-joined.log" # space-joined, for substring checks
before_list="$(mktemp)"
after_list="$(mktemp)"
find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$before_list" || true
cleanup() {
  find results/quality -maxdepth 1 -name 'quality-*.json' -print 2>/dev/null | sort > "$after_list" || true
  comm -13 "$before_list" "$after_list" | xargs -r rm -f
  rm -rf "$tmp_bin" "$tmp_work"
  rm -f "$before_list" "$after_list"
}
trap cleanup EXIT

# ---- mocks -------------------------------------------------------------------
# curl: /v1/models carries owned_by (the metadata engine fingerprint used when
# no container is known); /server_info is SGLang's readback; everything else
# fails like `curl -f` would on a 404.
cat > "${tmp_bin}/curl" <<'MOCK_CURL'
#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    */v1/models)
      printf '{"data":[{"id":"mock-model","owned_by":"%s"}]}' "${CURL_MOCK_OWNED_BY:-llamacpp}"
      exit 0 ;;
    */server_info|*/get_server_info)
      if [[ -n "${CURL_MOCK_SERVER_INFO:-}" ]]; then printf '%s' "$CURL_MOCK_SERVER_INFO"; exit 0; fi
      exit 22 ;;
    */props|*/get_model_info) exit 22 ;;
  esac
done
exit 0
MOCK_CURL
chmod +x "${tmp_bin}/curl"

# docker: `inspect <name>` returns the fixture named by DOCKER_MOCK_INSPECT;
# the --format probes the wrapper makes (RestartCount, Config.Image) answer
# from env; the sandbox-image preflight sees fresh images so --full does not
# degrade to "sandbox packs will be SKIPPED".
cat > "${tmp_bin}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
case "${1:-}" in
  inspect)
    shift; shift || true
    [[ -n "${DOCKER_MOCK_INSPECT:-}" && -f "${DOCKER_MOCK_INSPECT}" ]] || exit 1
    if [[ "${1:-}" == "--format" ]]; then
      case "${2:-}" in
        *RestartCount*) echo 0 ;;
        *Config.Image*) echo "${DOCKER_MOCK_IMAGE:-}" ;;
        *) echo "" ;;
      esac
      exit 0
    fi
    cat "${DOCKER_MOCK_INSPECT}"; exit 0 ;;
  image)
    if [[ "${2:-}" == "inspect" ]]; then
      case "${3:-}" in
        benchlocal-sandbox-*:latest)
          [[ "${4:-}" == "--format" ]] && date -u -d '+1 day' '+%Y-%m-%dT%H:%M:%S.000000000Z'
          exit 0 ;;
      esac
    fi
    exit 1 ;;
  ps) exit 0 ;;
esac
exit 1
MOCK_DOCKER
chmod +x "${tmp_bin}/docker"

cat > "${tmp_bin}/benchlocal-cli" <<'MOCK_BENCHLOCAL'
#!/usr/bin/env bash
json_out=""
argv=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-json) json_out="${2:-}"; shift 2 ;;
    list) echo 'toolcall-15'; exit 0 ;;
    --help) echo '--reasoning-effort'; exit 0 ;;
    *) shift ;;
  esac
done
printf '%s\n' "${argv[@]}" >> "${BENCHLOCAL_ARGS_LOG}"
printf '%s\n' "${argv[*]}" >> "${BENCHLOCAL_JOINED_LOG}"
if [[ -n "$json_out" ]]; then
  mkdir -p "$(dirname "$json_out")"
  printf '{"packs":[{"pack_id":"toolcall-15","status":"ok","passed":1,"total":1,"score":1.0}]}\n' > "$json_out"
fi
exit 0
MOCK_BENCHLOCAL
chmod +x "${tmp_bin}/benchlocal-cli"

# ---- docker-inspect fixtures ---------------------------------------------------
# mk_inspect <out> <entrypoint-json-array> <cmd-json-array> <env-json-array>
mk_inspect() {
  python3 - "$@" <<'PY'
import json, sys
out, entry, cmd, env = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4])
with open(out, "w") as fh:
    json.dump([{"Config": {"Entrypoint": entry, "Cmd": cmd, "Env": env, "Image": "mock/image"}}], fh)
PY
}

# The SHIPPED llama.cpp compose shape: a bash -c entrypoint that resolves a
# sampler ROW and appends `--reasoning-budget "${REASONING_BUDGET:--1}"` after
# "$@". A comment line mentioning the flag is included on purpose — it must be
# ignored. The value comes from Config.Env, exactly as the shell would read it.
LCPP_SCRIPT='# a comment mentioning --reasoning-budget 999 must be ignored\nROW=(--reasoning "on" --temp "0.6")\nROW+=(--reasoning-budget "${REASONING_BUDGET:--1}")\nROW+=(--reasoning-budget-message "${REASONING_BUDGET_MESSAGE:-[reasoning budget exhausted]}")\nexec /app/llama-server "$@" "${ROW[@]}"\n'
LCPP_ENTRY="$(python3 -c 'import json,sys; print(json.dumps(["/bin/bash","-c",sys.argv[1].encode().decode("unicode_escape"),"--"]))' "$LCPP_SCRIPT")"
LCPP_CMD='["--host","0.0.0.0","--port","8080","-m","/models/x.gguf","--jinja","--reasoning-format","deepseek"]'

mk_inspect "${tmp_work}/lcpp-8192.json"     "$LCPP_ENTRY" "$LCPP_CMD" '["REASONING_BUDGET=8192","REASONING_BUDGET_MESSAGE"]'
mk_inspect "${tmp_work}/lcpp-unset.json"    "$LCPP_ENTRY" "$LCPP_CMD" '["REASONING_BUDGET","REASONING_BUDGET_MESSAGE"]'
mk_inspect "${tmp_work}/lcpp-empty.json"    "$LCPP_ENTRY" "$LCPP_CMD" '["REASONING_BUDGET=","REASONING_BUDGET_MESSAGE"]'
mk_inspect "${tmp_work}/lcpp-4096.json"     "$LCPP_ENTRY" "$LCPP_CMD" '["REASONING_BUDGET=4096"]'
mk_inspect "${tmp_work}/lcpp-cmd-wins.json" "$LCPP_ENTRY" '["--reasoning-budget","4096"]' '["REASONING_BUDGET=8192"]'
mk_inspect "${tmp_work}/lcpp-literal.json"  '["/app/llama-server"]' '["--port","8080","--reasoning-budget","8192"]' '[]'
mk_inspect "${tmp_work}/lcpp-eq.json"       '["/app/llama-server"]' '["--reasoning-budget=8192"]' '[]'
mk_inspect "${tmp_work}/lcpp-none.json"     '["/app/llama-server"]' '["--port","8080","--reasoning-budget-message","[x]"]' '[]'
mk_inspect "${tmp_work}/lcpp-envonly.json"  '["/app/llama-server"]' '["--port","8080"]' '["LLAMA_ARG_THINK_BUDGET=8192"]'
mk_inspect "${tmp_work}/vllm-parser.json"   '["vllm","serve"]' '["/models/x","--reasoning-parser","qwen3","--port","8000"]' '[]'
mk_inspect "${tmp_work}/vllm-noparser.json" '["vllm","serve"]' '["/models/x","--port","8000"]' '[]'
mk_inspect "${tmp_work}/sgl-flag.json"      '["python3","-m","sglang.launch_server"]' '["--reasoning-parser","qwen3","--enable-custom-logit-processor"]' '[]'
mk_inspect "${tmp_work}/sgl-noflag.json"    '["python3","-m","sglang.launch_server"]' '["--reasoning-parser","qwen3"]' '[]'

# ---- runner --------------------------------------------------------------------
# run_wrapper [KEY=VAL ...] -- <wrapper args>
run_wrapper() {
  local extra_env=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do extra_env+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  : > "$ARGS_LOG"; : > "$ARGS_JOINED"
  set +e
  env PATH="${tmp_bin}:$PATH" \
      BENCHLOCAL_ARGS_LOG="$ARGS_LOG" BENCHLOCAL_JOINED_LOG="$ARGS_JOINED" \
      PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
      ${extra_env[@]+"${extra_env[@]}"} \
      bash scripts/quality-test.sh "$@" > "${tmp_work}/run.out" 2>&1
  RUN_RC=$?
  set -e
  OUT="$(cat "${tmp_work}/run.out")"
  ARGS="$(cat "$ARGS_JOINED")"
}

# The value that followed a given flag in the benchlocal argv (last occurrence).
arg_after() {
  awk -v f="$1" '$0 == f { getline; v = $0 } END { print v }' "$ARGS_LOG"
}
count_arg() {
  command grep -c -x -- "$1" "$ARGS_LOG" || true
}

LCPP="CONTAINER=llama-cpp-mock DOCKER_MOCK_INSPECT="
VLLM="CONTAINER=vllm-mock DOCKER_MOCK_INSPECT="

echo "--- A: opt-in — without the flag the argv is untouched ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick
assert_rc A 0 "$RUN_RC"
assert_not_contains A "$ARGS" "--extra-body"
assert_not_contains A "$ARGS" "--thinking-max-tokens"
assert_not_contains A "$ARGS" "thinking_token_budget"
assert_not_contains A "$OUT" "thinking budget:"

echo "--- B: llama.cpp compose shape, REASONING_BUDGET=8192 -> verified, cap derived ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget 8192
assert_rc B 0 "$RUN_RC"
assert_contains B "$OUT" "thinking budget: 8192 reasoning tokens via llama.cpp --reasoning-budget (boot flag, server-wide)"
assert_contains B "$OUT" "verified: container llama-cpp-mock boots llama-server with --reasoning-budget 8192"
assert_contains B "$OUT" "client cap: --thinking-max-tokens 12288 (= 8192 budget + 4096 answer headroom"
assert_contains B "$OUT" "sandboxed agentic packs: none in this selection"
assert_contains B "$ARGS" "--thinking-max-tokens 12288"
assert_not_contains B "$ARGS" "--extra-body"          # server-wide: nothing to send
# the comment line in the entrypoint script ("--reasoning-budget 999") was ignored
assert_not_contains B "$OUT" "999"

echo "--- C (NEGATIVE): llama.cpp flag PRESENT but REASONING_BUDGET unset -> -1 -> refused ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-unset.json" -- --quick --thinking-budget 8192
assert_rc C 2 "$RUN_RC"
assert_not_invoked C
assert_contains C "$OUT" "would NOT take effect"
assert_contains C "$OUT" "resolves --reasoning-budget to -1 (UNBOUNDED"
assert_contains C "$OUT" "REASONING_BUDGET=8192 bash scripts/switch.sh --force"
assert_contains C "$OUT" "--reasoning-budget 8192"
# an EMPTY value (`REASONING_BUDGET=`) resolves through `:-` to -1 as well
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-empty.json" -- --quick --thinking-budget 8192
assert_rc C-empty 2 "$RUN_RC"
assert_contains C-empty "$OUT" "UNBOUNDED"
# positive evidence is NEVER bypassed by the unverified acknowledgement
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-unset.json" THINKING_BUDGET_UNVERIFIED=1 -- --quick --thinking-budget 8192
assert_rc C-nobypass 2 "$RUN_RC"
assert_not_invoked C-nobypass
assert_not_contains C-nobypass "$OUT" "THINKING BUDGET UNVERIFIED"

echo "--- D (NEGATIVE): llama.cpp server carries 4096, asked 8192 -> refused ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-4096.json" -- --quick --thinking-budget 8192
assert_rc D 2 "$RUN_RC"
assert_not_invoked D
assert_contains D "$OUT" "resolves --reasoning-budget to 4096, not 8192"
assert_contains D "$OUT" "cannot change it per request"

echo "--- E (NEGATIVE): llama.cpp booted without the flag at all -> refused ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-none.json" -- --quick --thinking-budget 8192
assert_rc E 2 "$RUN_RC"
assert_not_invoked E
assert_contains E "$OUT" "booted WITHOUT --reasoning-budget (unbounded)"

echo "--- F: llama.cpp literal / = / env-only / ROW-after-cmd spellings all resolve ---"
for fx in lcpp-literal lcpp-eq lcpp-envonly lcpp-cmd-wins; do
  run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/${fx}.json" -- --quick --thinking-budget 8192
  assert_rc "F-${fx}" 0 "$RUN_RC"
  assert_contains "F-${fx}" "$OUT" "--reasoning-budget 8192"
done

echo "--- G: llama.cpp + --full -> hermes is GOVERNED by the boot flag ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --full --thinking-budget 8192
assert_rc G 0 "$RUN_RC"
assert_contains G "$OUT" "sandboxed agentic packs (hermesagent-20): governed"
assert_contains G "$ARGS" "--full"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --pack aider-polyglot-30 --thinking-budget 8192
assert_rc G-aider 0 "$RUN_RC"
assert_contains G-aider "$OUT" "sandboxed agentic packs (aider-polyglot-30): governed"

echo "--- H: vLLM with --reasoning-parser -> per-request thinking_token_budget ---"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --quick --thinking-budget 8192
assert_rc H 0 "$RUN_RC"
assert_contains H "$OUT" "via vLLM per-request thinking_token_budget"
assert_contains H "$OUT" "boots vLLM with --reasoning-parser qwen3"
[[ "$(count_arg --extra-body)" == "1" ]] || { echo "ASSERTION FAILED [H]: expected exactly one --extra-body, got $(count_arg --extra-body)" >&2; FAILED=1; }
[[ "$(arg_after --extra-body)" == '{"thinking_token_budget": 8192}' ]] || { echo "ASSERTION FAILED [H]: extra-body was '$(arg_after --extra-body)'" >&2; FAILED=1; }
assert_contains H "$ARGS" "--thinking-max-tokens 12288"

echo "--- I (NEGATIVE): vLLM without a reasoning parser -> refused ---"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-noparser.json" -- --quick --thinking-budget 8192
assert_rc I 2 "$RUN_RC"
assert_not_invoked I
assert_contains I "$OUT" "booted WITHOUT --reasoning-parser"
assert_contains I "$OUT" "VLLMValidationError"
assert_contains I "$OUT" "--reasoning-parser <name>"

echo "--- J (NEGATIVE): per-request engine + an in-sandbox agentic pack -> refused ---"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --full --thinking-budget 8192
assert_rc J-full 2 "$RUN_RC"
assert_not_invoked J-full
assert_contains J-full "$OUT" "the selection includes hermesagent-20"
assert_contains J-full "$OUT" "INSIDE the sandbox"
assert_contains J-full "$OUT" "--no-sandboxed"
assert_contains J-full "$OUT" "serve on llama.cpp"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --sandboxed-only --thinking-budget 8192
assert_rc J-sbonly 2 "$RUN_RC"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --pack aider-polyglot-30 --thinking-budget 8192
assert_rc J-aider 2 "$RUN_RC"
assert_contains J-aider "$OUT" "includes aider-polyglot-30"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --scenario hermesagent-20/HA-01 --thinking-budget 8192
assert_rc J-scn 2 "$RUN_RC"
# ...and the SAME engine is fine once the agentic packs are out of the selection
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --full --no-sandboxed --thinking-budget 8192
assert_rc J-nosb 0 "$RUN_RC"
assert_contains J-nosb "$ARGS" "--no-sandboxed-packs"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --pack cli-40 --thinking-budget 8192
assert_rc J-cli 0 "$RUN_RC"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --scenario cli-40/CLI-31 --thinking-budget 8192
assert_rc J-cliscn 0 "$RUN_RC"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --reasoning --thinking-budget 8192
assert_rc J-reasoning 0 "$RUN_RC"

echo "--- K: SGLang via /server_info (no container known) -> processor chosen from the parser ---"
SGL_INFO_OK='{"enable_custom_logit_processor": true, "reasoning_parser": "qwen3", "version": "0.5.20"}'
run_wrapper CURL_MOCK_OWNED_BY=sglang "CURL_MOCK_SERVER_INFO=${SGL_INFO_OK}" -- --quick --thinking-budget 8192
assert_rc K 0 "$RUN_RC"
assert_contains K "$OUT" "via SGLang per-request custom_logit_processor=Qwen3ThinkingBudgetLogitProcessor + custom_params.thinking_budget"
assert_contains K "$OUT" "/server_info reports enable_custom_logit_processor=true, reasoning_parser=qwen3"
K_BODY="$(arg_after --extra-body)"
assert_contains K "$K_BODY" '"custom_params": {"thinking_budget": 8192}'
assert_contains K "$K_BODY" '"custom_logit_processor": "{\"callable\": \"'
# the pickle is a by-reference GLOBAL to sglang's own class: module + qualname, nothing else
K_HEX="$(python3 -c 'import json,sys; b=json.loads(sys.argv[1]); print(json.loads(b["custom_logit_processor"])["callable"])' "$K_BODY")"
K_DECODED="$(python3 -c 'import sys; print(bytes.fromhex(sys.argv[1]).decode("latin-1"))' "$K_HEX")"
assert_contains K "$K_DECODED" $'sglang.srt.sampling.custom_logit_processor\nQwen3ThinkingBudgetLogitProcessor\n.'
# nested readback shape (v0.5.20 groups args): still found, still verified
run_wrapper CURL_MOCK_OWNED_BY=sglang 'CURL_MOCK_SERVER_INFO={"exec":{"features":{"enable_custom_logit_processor":true}},"serving":{"reasoning_parser":"glm45"}}' -- --quick --thinking-budget 8192
assert_rc K-nested 0 "$RUN_RC"
assert_contains K-nested "$OUT" "custom_logit_processor=Glm4MoeThinkingBudgetLogitProcessor"

echo "--- L (NEGATIVE): SGLang readback says the flag is OFF -> refused ---"
run_wrapper CURL_MOCK_OWNED_BY=sglang 'CURL_MOCK_SERVER_INFO={"enable_custom_logit_processor": false, "reasoning_parser": "qwen3"}' -- --quick --thinking-budget 8192
assert_rc L 2 "$RUN_RC"
assert_not_invoked L
assert_contains L "$OUT" "WITHOUT --enable-custom-logit-processor"
assert_contains L "$OUT" "ValueError"
assert_contains L "$OUT" "add --enable-custom-logit-processor to the SGLang server command"

echo "--- M: SGLang parser with no mapped processor -> refused unless named explicitly ---"
run_wrapper CURL_MOCK_OWNED_BY=sglang 'CURL_MOCK_SERVER_INFO={"enable_custom_logit_processor": true, "reasoning_parser": "mimo"}' -- --quick --thinking-budget 8192
assert_rc M 2 "$RUN_RC"
assert_not_invoked M
assert_contains M "$OUT" "no ThinkingBudgetLogitProcessor is known for reasoning_parser='mimo'"
assert_contains M "$OUT" "THINKING_BUDGET_SGLANG_PROCESSOR=<subclass>"
run_wrapper CURL_MOCK_OWNED_BY=sglang 'CURL_MOCK_SERVER_INFO={"enable_custom_logit_processor": true, "reasoning_parser": "mimo"}' THINKING_BUDGET_SGLANG_PROCESSOR=InklingThinkingBudgetLogitProcessor -- --quick --thinking-budget 8192
assert_rc M-override 0 "$RUN_RC"
assert_contains M-override "$OUT" "custom_logit_processor=InklingThinkingBudgetLogitProcessor"

echo "--- N: SGLang docker fallback when there is no readback ---"
run_wrapper CONTAINER=sglang-mock "DOCKER_MOCK_INSPECT=${tmp_work}/sgl-flag.json" -- --quick --thinking-budget 8192
assert_rc N 0 "$RUN_RC"
assert_contains N "$OUT" "container sglang-mock reports enable_custom_logit_processor=true, reasoning_parser=qwen3"
run_wrapper CONTAINER=sglang-mock "DOCKER_MOCK_INSPECT=${tmp_work}/sgl-noflag.json" -- --quick --thinking-budget 8192
assert_rc N-noflag 2 "$RUN_RC"
assert_not_invoked N-noflag
assert_contains N-noflag "$OUT" "WITHOUT --enable-custom-logit-processor"

echo "--- O (NEGATIVE): no evidence at all -> refused; explicit acknowledgement labels the run ---"
run_wrapper CURL_MOCK_OWNED_BY=llamacpp -- --quick --thinking-budget 8192
assert_rc O 2 "$RUN_RC"
assert_not_invoked O
assert_contains O "$OUT" "cannot verify the budget can take effect on this llamacpp server"
assert_contains O "$OUT" "accepted and ignored is worse than none"
assert_contains O "$OUT" "THINKING_BUDGET_UNVERIFIED=1"
run_wrapper CURL_MOCK_OWNED_BY=llamacpp THINKING_BUDGET_UNVERIFIED=1 -- --quick --thinking-budget 8192
assert_rc O-ack 0 "$RUN_RC"
assert_contains O-ack "$OUT" "THINKING BUDGET UNVERIFIED (THINKING_BUDGET_UNVERIFIED=1)"
assert_contains O-ack "$OUT" "verified: UNVERIFIED — asserted by the operator"
assert_contains O-ack "$ARGS" "--thinking-max-tokens 12288"
# vLLM with no container: same refusal, and the acknowledgement still sends the field
run_wrapper CURL_MOCK_OWNED_BY=vllm -- --quick --thinking-budget 8192
assert_rc O-vllm 2 "$RUN_RC"
assert_contains O-vllm "$OUT" "cannot verify the budget can take effect on this vllm server"

echo "--- P (NEGATIVE): engine family unknown -> refused, not guessed ---"
run_wrapper CURL_MOCK_OWNED_BY=openai -- --quick --thinking-budget 8192
assert_rc P 2 "$RUN_RC"
assert_not_invoked P
assert_contains P "$OUT" "cannot tell which engine family is serving"
# an unrecognised container name falls through to its IMAGE before giving up
run_wrapper CONTAINER=estate-thing "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" DOCKER_MOCK_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda CURL_MOCK_OWNED_BY=openai -- --quick --thinking-budget 8192
assert_rc P-image 0 "$RUN_RC"
assert_contains P-image "$OUT" "via llama.cpp --reasoning-budget"

echo "--- Q: the matched client cap ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget 8192 --thinking-max-tokens 8192
assert_rc Q-eq 2 "$RUN_RC"
assert_not_invoked Q-eq
assert_contains Q-eq "$OUT" "--thinking-max-tokens 8192 is not above --thinking-budget 8192"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget 8192 --thinking-max-tokens 4096
assert_rc Q-below 2 "$RUN_RC"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget 8192 --thinking-max-tokens 16384
assert_rc Q-above 0 "$RUN_RC"
assert_contains Q-above "$OUT" "client cap: --thinking-max-tokens 16384 (yours: 8192 budget + 8192 answer headroom)"
assert_contains Q-above "$ARGS" "--thinking-max-tokens 16384"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" THINKING_BUDGET_HEADROOM=2048 -- --quick --thinking-budget 8192
assert_rc Q-headroom 0 "$RUN_RC"
assert_contains Q-headroom "$ARGS" "--thinking-max-tokens 10240"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" THINKING_BUDGET_HEADROOM=abc -- --quick --thinking-budget 8192
assert_rc Q-badheadroom 2 "$RUN_RC"
# --max-tokens alone is NOT the cap on thinking packs (benchlocal: --thinking-max-tokens overrides it), so the derived cap still goes out
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget 8192 --max-tokens 4096
assert_rc Q-maxtok 0 "$RUN_RC"
assert_contains Q-maxtok "$ARGS" "--max-tokens 4096"
assert_contains Q-maxtok "$ARGS" "--thinking-max-tokens 12288"

echo "--- R: --resume restores the original config; the budget is a conflict ---"
printf '{}' > "${tmp_work}/prior.json"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --resume "${tmp_work}/prior.json" --thinking-budget 8192
assert_rc R 2 "$RUN_RC"
assert_contains R "$OUT" "drop: --thinking-budget"

echo "--- S: a pass-through --extra-body is MERGED, never allowed to replace the budget ---"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --quick --thinking-budget 8192 -- --extra-body '{"provider": {"only": ["x"]}}' --retry-runaways
assert_rc S 0 "$RUN_RC"
assert_contains S "$OUT" "merged your pass-through --extra-body"
[[ "$(count_arg --extra-body)" == "1" ]] || { echo "ASSERTION FAILED [S]: expected exactly one --extra-body, got $(count_arg --extra-body)" >&2; FAILED=1; }
S_BODY="$(arg_after --extra-body)"
assert_contains S "$S_BODY" '"thinking_token_budget": 8192'
assert_contains S "$S_BODY" '"provider": {"only": ["x"]}'
assert_contains S "$ARGS" "--retry-runaways"          # the rest of the pass-through survives
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --quick --thinking-budget 8192 -- --extra-body '{"thinking_token_budget": 1}'
assert_rc S-conflict 2 "$RUN_RC"
assert_not_invoked S-conflict
assert_contains S-conflict "$OUT" "two sources of truth"
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --quick --thinking-budget 8192 -- --extra-body 'not json'
assert_rc S-badjson 2 "$RUN_RC"
# without a budget the pass-through --extra-body is forwarded verbatim (unchanged behaviour)
run_wrapper CONTAINER=vllm-mock "DOCKER_MOCK_INSPECT=${tmp_work}/vllm-parser.json" -- --quick -- --extra-body '{"foo": 1}'
assert_rc S-plain 0 "$RUN_RC"
[[ "$(arg_after --extra-body)" == '{"foo": 1}' ]] || { echo "ASSERTION FAILED [S-plain]: pass-through extra-body was altered without a budget" >&2; FAILED=1; }

echo "--- T: argument validation ---"
for bad in 0 -1 abc 8192.5; do
  run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget "$bad"
  assert_rc "T-${bad}" 2 "$RUN_RC"
  assert_contains "T-${bad}" "$OUT" "--thinking-budget requires a positive integer"
done
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --thinking-budget
assert_rc T-missing 2 "$RUN_RC"

echo "--- U: the env spelling and the thinking-off leg ---"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" THINKING_BUDGET=8192 -- --quick
assert_rc U-env 0 "$RUN_RC"
assert_contains U-env "$ARGS" "--thinking-max-tokens 12288"
run_wrapper CONTAINER=llama-cpp-mock "DOCKER_MOCK_INSPECT=${tmp_work}/lcpp-8192.json" -- --quick --no-thinking --thinking-budget 8192
assert_rc U-off 0 "$RUN_RC"
assert_contains U-off "$OUT" "thinking is forced OFF on this leg — the budget is verified but inert"

if [[ "$FAILED" != "0" ]]; then
  echo "FAIL: test-quality-thinking-budget" >&2
  exit 1
fi
echo "PASS: test-quality-thinking-budget (#1383 opt-in budget, verified per engine and per pack class)"
