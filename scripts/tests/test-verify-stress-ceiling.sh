#!/usr/bin/env bash
# Test for verify-stress.sh ceiling ladder (#199) — validates:
#   1. get_n_ctx() reads llama.cpp /props → default_generation_settings.n_ctx (nested)
#   2. get_n_ctx() falls back to top-level n_ctx (some llama.cpp forks)
#   3. get_n_ctx() falls back to /v1/models → max_model_len (vLLM)
#   4. get_n_ctx() returns 0 when detection fails
#   5. get_vram_free_mb() sums across GPUs (dual + single)
#   6. Ladder rung computation is correct for various n_ctx values
#   7. SKIP_CEILING=1 / SKIP_LONGCTX=1 skip the ceiling ladder
#   8. All 8 probe headers [N/8] are present in output
#   9. Small compose skips the ladder (ceiling ≤ start)
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
assert_eq() {
  local label="$1" expected="$2" actual="$3"
  if [[ "$expected" == "$actual" ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: $label: expected '$expected', got '$actual'" >&2
    FAIL=$((FAIL + 1))
  fi
}
assert_contains() {
  local haystack="$1" needle="$2" label="${3:-}"
  if [[ "$haystack" == *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label:-assert_contains}: expected output to contain: $needle" >&2
    echo "--- output (last 30 lines) ---" >&2
    echo "$haystack" | tail -30 >&2
    FAIL=$((FAIL + 1))
  fi
}

# Extract helpers to temp file (avoids bash -c quoting issues with
# embedded python one-liners that contain single quotes).
HELPERS_FILE="$(mktemp --suffix=.sh)"
sed -n '/^get_n_ctx()/,/^}/p' scripts/verify-stress.sh > "$HELPERS_FILE"
sed -n '/^get_vram_free_mb()/,/^}/p' scripts/verify-stress.sh >> "$HELPERS_FILE"

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir" "$HELPERS_FILE"; }
trap cleanup EXIT

# Mock curl helper: match against the full arg string (not per-arg).
make_curl_mock() {
  cat > "${tmp_dir}/curl" <<EOF
#!/usr/bin/env bash
all="\$*"
case "\$all" in
$1
  *) exit 1 ;;
esac
EOF
  chmod +x "${tmp_dir}/curl"
}

# ---- Test get_n_ctx with REAL llama.cpp /props shape (nested) ----
# This is the critical test — real llama.cpp nests n_ctx inside
# default_generation_settings, NOT at top-level.
make_curl_mock '  */props*) printf '"'"'{"default_generation_settings":{"n_ctx":262144,"temperature":0.8}}'"'"'; exit 0 ;;'

result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx from /props NESTED (real llama.cpp)" "262144" "$result"

# ---- Test get_n_ctx with top-level n_ctx (some forks/old versions) ----
make_curl_mock '  */props*) printf '"'"'{"n_ctx":131072}'"'"'; exit 0 ;;'

result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx from /props TOP-LEVEL (fork/old)" "131072" "$result"

# ---- Test get_n_ctx with both nested and top-level (nested wins) ----
make_curl_mock '  */props*) printf '"'"'{"n_ctx":999,"default_generation_settings":{"n_ctx":200000}}'"'"'; exit 0 ;;'

result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx nested wins over top-level" "200000" "$result"

# ---- Test get_n_ctx fallback to /v1/models (vLLM) ----
make_curl_mock '
  */props*) exit 1 ;;
  */v1/models*) printf '"'"'{"data":[{"id":"mock","max_model_len":131072}]}'"'"'; exit 0 ;;'

result="$(ENGINE_KIND=vllm URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx fallback to /v1/models (vLLM)" "131072" "$result"

# ---- Test get_n_ctx returns 0 on total failure ----
make_curl_mock ''

result="$(ENGINE_KIND=unknown URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx returns 0 on failure" "0" "$result"

# ---- Test get_n_ctx with 512K compose (nested) ----
make_curl_mock '  */props*) printf '"'"'{"default_generation_settings":{"n_ctx":524288}}'"'"'; exit 0 ;;'

result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx 512K compose (nested)" "524288" "$result"

# ---- Test get_vram_free_mb dual GPU (no container context — MIN, not sum) ----
# TP OOMs on whichever card runs out first, so min per-card free is the honest
# margin; the old sum overstated dual-rig headroom ~2x (corrected 2026-07-04
# while dogfooding #246).
cat > "${tmp_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *memory.free*) printf "12000\n11500\n" ;;
  *index*) printf "0\n1\n" ;;
esac
EOF
chmod +x "${tmp_dir}/nvidia-smi"

result="$(CONTAINER=none PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; get_vram_free_mb")"
assert_eq "get_vram_free_mb dual-GPU min (no container)" "11500" "$result"

# ---- Test get_vram_free_mb single GPU ----
cat > "${tmp_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *memory.free*) printf "23000\n" ;;
  *index*) printf "0\n" ;;
esac
EOF
chmod +x "${tmp_dir}/nvidia-smi"

result="$(CONTAINER=none PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; get_vram_free_mb")"
assert_eq "get_vram_free_mb single-GPU" "23000" "$result"

# ---- Test get_vram_free_mb scoping: multi-GPU host, single-GPU container ----
# Simulates dual-3090 host with compose pinned to GPU 0 via DeviceRequests.
# GPU 0 has 381 MB free (model), GPU 1 has 24126 MB free (idle).
# Old behavior: sum = 24507 (inflated). Fix: query only GPU 0 → 381.
cat > "${tmp_dir}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *-i\ 0*memory.free*|*memory.free*-i\ 0*) printf "381\n" ;;
  *memory.free*)
    # When -i is passed, only return that GPU's value
    if echo "$*" | grep -q '\-i'; then
      gpu="$(echo "$*" | sed 's/.*-i *\([^ ]*\).*/\1/')"
      case "$gpu" in
        0) printf "381\n" ;;
        1) printf "24126\n" ;;
        *) printf "0\n" ;;
      esac
    else
      printf "381\n24126\n"
    fi
    ;;
  *index*) printf "0\n1\n" ;;
esac
EOF
chmod +x "${tmp_dir}/nvidia-smi"

cat > "${tmp_dir}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
case "$*" in
  *inspect*)
    printf '[{"HostConfig":{"DeviceRequests":[{"Driver":"nvidia","DeviceIDs":["0"],"Capabilities":[["compute","utility"]]}]},"Config":{"Image":"mock","Env":["NVIDIA_VISIBLE_DEVICES=all"]}}]'
    ;;
esac
MOCK_DOCKER
chmod +x "${tmp_dir}/docker"

result="$(CONTAINER=mock-container PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; get_vram_free_mb")"
assert_eq "get_vram_free_mb scoped to GPU 0 (Bug 1 fix)" "381" "$result"

# ---- Test ladder rung computation ----
compute_ladder() {
  local n_ctx="$1" start="${2:-95000}" step="${3:-30000}" frac="${4:-0.92}"
  python3 -c "
start, top, step = ${start}, int(${n_ctx} * ${frac}), ${step}
if top <= start:
    print('SKIP')
else:
    rungs = list(range(start, top, step))
    if not rungs or rungs[-1] != top:
        rungs.append(top)
    print(' '.join(str(r) for r in rungs))
"
}

# 262K compose: 95000 → 125000 → 155000 → 185000 → 215000 → 241172
ladder="$(compute_ladder 262144)"
assert_eq "ladder 262K" "95000 125000 155000 185000 215000 241172" "$ladder"

# 131K compose: 95000 → 120586
ladder="$(compute_ladder 131072)"
assert_eq "ladder 131K" "95000 120586" "$ladder"

# 200K compose: 95000 → 125000 → 155000 → 184000
ladder="$(compute_ladder 200000)"
assert_eq "ladder 200K" "95000 125000 155000 184000" "$ladder"

# 512K compose: 95000 → 125000 → ... → 482344
ladder="$(compute_ladder 524288)"
rung_count="$(echo "$ladder" | wc -w)"
assert_eq "ladder 512K rung count" "14" "$rung_count"
first="$(echo "$ladder" | awk '{print $1}')"
last="$(echo "$ladder" | awk '{print $NF}')"
assert_eq "ladder 512K first rung" "95000" "$first"
assert_eq "ladder 512K last rung" "482344" "$last"

# 32K compose: ceiling = 29440 < start 95000 → SKIP
ladder="$(compute_ladder 32768)"
assert_eq "ladder 32K (too small)" "SKIP" "$ladder"

# Custom step size: 15000
ladder="$(compute_ladder 262144 95000 15000)"
rung_count="$(echo "$ladder" | wc -w)"
assert_eq "ladder 262K step=15K rung count" "11" "$rung_count"

# Custom fraction: 0.80
ladder="$(compute_ladder 262144 95000 30000 0.80)"
last="$(echo "$ladder" | awk '{print $NF}')"
assert_eq "ladder 262K frac=0.80 last rung" "209715" "$last"

# ---- Test: full script with SKIP_CEILING / SKIP_LONGCTX ----
make_curl_mock '
  */v1/models*) printf '"'"'{"data":[{"id":"mock-model","max_model_len":262144}]}'"'"'; exit 0 ;;
  */props*) printf '"'"'{"default_generation_settings":{"n_ctx":262144}}'"'"'; exit 0 ;;
  */v1/chat/completions*) printf '"'"'{"choices":[{"message":{"content":"mock response"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}'"'"'; exit 0 ;;'

cat > "${tmp_dir}/nvidia-smi" <<'MOCK_NVIDIA'
#!/usr/bin/env bash
case "$*" in
  *memory.free*) echo "23500" ;;
  *) echo "NVIDIA-SMI mock" ;;
esac
MOCK_NVIDIA
chmod +x "${tmp_dir}/nvidia-smi"

cat > "${tmp_dir}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
printf '{"Config":{"Image":"mock","Env":[]},"Args":[]}'
MOCK_DOCKER
chmod +x "${tmp_dir}/docker"

# Test: SKIP_CEILING=1 skips probe 8
out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_CEILING=1 SKIP_LONGCTX=1 SKIP_TOOL_PREFILL=1 \
  bash scripts/verify-stress.sh 2>&1)" || true
assert_contains "$out" "[8/8]" "probe 8 header printed"
assert_contains "$out" "SKIP_CEILING=1" "SKIP_CEILING skip message"

# Test: SKIP_LONGCTX=1 also skips ceiling ladder
out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_LONGCTX=1 SKIP_TOOL_PREFILL=1 \
  bash scripts/verify-stress.sh 2>&1)" || true
assert_contains "$out" "SKIP_LONGCTX=1" "SKIP_LONGCTX also skips ceiling"

# Verify all 8 probe headers are present
for i in 1 2 3 4 5 6 7 8; do
  assert_contains "$out" "[${i}/8]" "probe ${i} header"
done

# Note: Bug 3 (no-op ladder must FAIL) is validated by code review + live test.
# The summary logic checks any_sizing_error before any_fail and emits FAIL.
# "All rungs skipped" now emits FAIL instead of skip. Unit-testing the
# summary branch requires mocking send_streaming_niah (Python urllib),
# which is beyond the curl-mock approach used here. Validated live:
# old code said "All stress checks passed" with all-400 ladder;
# new code says "filler→token sizing error" and increments FAILED.

# ---- Test: streaming NIAH helper timing extraction ----
# Extract the Python streaming helper from verify-stress.sh and test it
# with mocked HTTP responses matching the REAL llama.cpp response shape.

# Extract the Python code from send_streaming_niah
STREAMING_PY="$(mktemp --suffix=.py)"
sed -n "/^send_streaming_niah/,/^}/p" scripts/verify-stress.sh \
  | sed -n "/<<'PYEOF'/,/PYEOF/p" \
  | sed '1d;$d' > "$STREAMING_PY"

# Test 1: llama.cpp streaming response with timings object
# Real shape (confirmed against live endpoint):
#   timings: {prompt_n, prompt_ms, prompt_per_second, ...}
#   usage: {prompt_tokens, completion_tokens, ...}
MOCK_RESULT="$(mktemp --suffix=.json)"
MOCK_REQ="$(mktemp --suffix=.json)"
MOCK_TEST="$(mktemp --suffix=.py)"
echo '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' > "$MOCK_REQ"

cat > "$MOCK_TEST" <<'PYTEST'
import json, sys, urllib.request, urllib.error

# Mock streaming chunks matching REAL llama.cpp response shape
chunks = [
    {"choices": [{"delta": {"content": "Paris"}, "finish_reason": None}], "id": "1"},
    {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 14, "completion_tokens": 1, "total_tokens": 15},
        "timings": {
            "prompt_n": 14, "prompt_ms": 346.811,
            "prompt_per_token_ms": 24.772, "prompt_per_second": 40.368,
            "predicted_n": 1, "predicted_ms": 17.554,
            "predicted_per_token_ms": 17.554, "predicted_per_second": 56.968,
        },
    },
]

class MockResponse:
    def __init__(self):
        self._lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]
    def getcode(self): return 200
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self): return iter(self._lines)

urllib.request.urlopen = lambda req, timeout=None: MockResponse()

# Rewrite sys.argv so the streaming helper sees the right args
# argv[0]=script, argv[1]=req_file, argv[2]=result_file, argv[3]=url, argv[4]=timeout
streaming_py = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
exec(open(streaming_py).read())
PYTEST

python3 "$MOCK_TEST" "$STREAMING_PY" "$MOCK_REQ" "$MOCK_RESULT" "http://mock" "60" 2>&1

if [[ -f "$MOCK_RESULT" ]]; then
  result_http="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['http_code'])" 2>/dev/null)"
  result_tps="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['prefill_tps'])" 2>/dev/null)"
  result_ms="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['prefill_ms'])" 2>/dev/null)"
  result_tok="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['prompt_tokens'])" 2>/dev/null)"
  result_content="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['content'])" 2>/dev/null)"
  assert_eq "streaming: llama.cpp http_code" "200" "$result_http"
  assert_eq "streaming: llama.cpp prompt_tokens" "14" "$result_tok"
  assert_eq "streaming: llama.cpp content" "Paris" "$result_content"
  assert_eq "streaming: llama.cpp prefill_tps" "40.4" "$result_tps"
  # prefill_ms should be ~346.8
  ms_ok="$(python3 -c "print('ok' if abs(float('${result_ms}') - 346.8) < 1 else 'fail')" 2>/dev/null)"
  assert_eq "streaming: llama.cpp prefill_ms ~346.8" "ok" "$ms_ok"
else
  echo "FAIL: streaming test 1 — no result file" >&2
  FAIL=$((FAIL + 1))
fi

# Test 2: vLLM-style response (no timings, cross-engine fallback)
rm -f "$MOCK_RESULT"
cat > "$MOCK_TEST" <<'PYTEST'
import json, sys, urllib.request

chunks = [
    {"choices": [{"delta": {"content": "42"}, "finish_reason": None}]},
    {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 124000, "completion_tokens": 2, "total_tokens": 124002},
    },
]

class MockResponse:
    def __init__(self):
        self._lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]
    def getcode(self): return 200
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self): return iter(self._lines)

urllib.request.urlopen = lambda req, timeout=None: MockResponse()
streaming_py = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
exec(open(streaming_py).read())
PYTEST

python3 "$MOCK_TEST" "$STREAMING_PY" "$MOCK_REQ" "$MOCK_RESULT" "http://mock" "60" 2>&1

if [[ -f "$MOCK_RESULT" ]]; then
  result_tps="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['prefill_tps'])" 2>/dev/null)"
  result_tok="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['prompt_tokens'])" 2>/dev/null)"
  assert_eq "streaming: vLLM prompt_tokens" "124000" "$result_tok"
  # prefill_tps should be computed from prompt_tokens / TTFT (positive number)
  tps_positive="$(python3 -c "print('ok' if float('${result_tps}') > 0 else 'fail')" 2>/dev/null)"
  assert_eq "streaming: vLLM prefill_tps > 0 (cross-engine fallback)" "ok" "$tps_positive"
else
  echo "FAIL: streaming test 2 — no result file" >&2
  FAIL=$((FAIL + 1))
fi

# Test 3: HTTP 500 error handling
rm -f "$MOCK_RESULT"
cat > "$MOCK_TEST" <<'PYTEST'
import json, sys, urllib.request, urllib.error

def mock_urlopen(req, timeout=None):
    raise urllib.error.HTTPError(
        'http://mock/v1/chat/completions', 500, 'OOM', {}, None)
urllib.request.urlopen = mock_urlopen
streaming_py = sys.argv[1]
sys.argv = [sys.argv[0]] + sys.argv[2:]
exec(open(streaming_py).read())
PYTEST

python3 "$MOCK_TEST" "$STREAMING_PY" "$MOCK_REQ" "$MOCK_RESULT" "http://mock" "60" 2>&1

if [[ -f "$MOCK_RESULT" ]]; then
  result_http="$(python3 -c "import json; print(json.load(open('$MOCK_RESULT'))['http_code'])" 2>/dev/null)"
  assert_eq "streaming: HTTP 500 error handling" "500" "$result_http"
else
  echo "FAIL: streaming test 3 — no result file" >&2
  FAIL=$((FAIL + 1))
fi

rm -f "$STREAMING_PY" "$MOCK_RESULT" "$MOCK_REQ" "$MOCK_TEST"

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
echo "test-verify-stress-ceiling: ok"
