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
#  10. N_CTX: wins over detection, runs the ladder where nothing is detected, a
#      malformed one fails; a skipped check is named in the summary, which then
#      doesn't claim "paths are sound" (#1826)
#  11. A failed check is named under the summary's count; the VRAM margin is a
#      boundary check, named as an advisory with no ✗ (report.sh's exit 2); the
#      undetermined-GPU warning is said once, not on every rung (#1826)
#  12. get_n_ctx() reads a QEngine container's context from its boot log, and only a QEngine's (#1846)
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
# The extracted send_streaming_niah heredoc imports engine_http from the scripts lib;
# verify-stress.sh sets this inline when it runs the heredoc, so the standalone exec must too.
export CLUB_SCRIPTS_LIB="$ROOT_DIR/scripts/lib"

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
assert_not_contains() {
  local haystack="$1" needle="$2" label="${3:-}"
  if [[ "$haystack" != *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${label:-assert_not_contains}: output must NOT contain: $needle" >&2
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

# ---- N_CTX (#1826): an explicit context length wins, and runs where nothing is detected ----
make_curl_mock ''
result="$(ENGINE_KIND=unknown URL=http://mock CONTAINER=none N_CTX=200000 \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx: N_CTX when the server reports nothing" "200000" "$result"

make_curl_mock '  */props*) printf '"'"'{"default_generation_settings":{"n_ctx":262144}}'"'"'; exit 0 ;;'
result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none N_CTX=131072 \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx: N_CTX wins over detection" "131072" "$result"

result="$(ENGINE_KIND=llamacpp URL=http://mock CONTAINER=none N_CTX=lots \
  PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_n_ctx")"
assert_eq "get_n_ctx: a malformed N_CTX falls through to detection" "262144" "$result"

# ---- QEngine (#1846): its context is only in its boot log (no /props, no max_model_len, no flag) ----
qe_dir="$(mktemp -d)"
cat > "${qe_dir}/docker" <<'MOCK_DOCKER'
#!/usr/bin/env bash
# inspect <c> --format '{{.Config.Image}}' -> $QE_IMAGE; inspect <c> -> JSON with no context flag; logs <c> -> $QE_LOG
case "$1" in
  inspect) if [[ "${4:-}" == '{{.Config.Image}}' ]]; then echo "${QE_IMAGE:-}"
           else printf '[{"Config":{"Image":"%s","Cmd":["bash","-c","exec /usr/local/bin/qengine-entrypoint"]}}]' "${QE_IMAGE:-}"; fi ;;
  logs) printf '%b' "${QE_LOG:-}" ;;
esac
MOCK_DOCKER
chmod +x "${qe_dir}/docker"
make_curl_mock '  */v1/models*) printf '"'"'{"object":"list","data":[{"id":"qwen3.8-flash-next","object":"model","owned_by":"qengine"}]}'"'"'; exit 0 ;;'
QE_IMG="ghcr.io/bpx03/qengine@sha256:b9d1b0afe8f96547b3a05de73aff437077effcb1117c82ab2e0fe62bf000b3da"
qe_ctx() {  # <image> <log> <container>
  QE_IMAGE="$1" QE_LOG="$2" ENGINE_KIND=qengine URL=http://mock CONTAINER="$3" PATH="${qe_dir}:${tmp_dir}:$PATH" \
    bash -c "source '$HELPERS_FILE'; get_n_ctx"
}
assert_eq "get_n_ctx: QEngine 27B's boot log (max context)" "262144" \
  "$(qe_ctx "$QE_IMG" 'KV pool: 374784 tokens shared by 32 sequence slots (max context 262144)\n' qengine-qwen38-27b-dual)"
assert_eq "get_n_ctx: QEngine Flash-Next's boot log (ready, context)" "131072" \
  "$(qe_ctx "$QE_IMG" '[qflash] 16 sequence slots over a pool of 256 periods\n[qflash 46s] ready, context 131072\n' qengine-qwen38-flash-next-dual)"
assert_eq "get_n_ctx: a restarted QEngine container: the last boot's context" "65536" \
  "$(qe_ctx "$QE_IMG" '[qflash 46s] ready, context 262144\n[qflash 51s] ready, context 65536\n' qengine-qwen38-flash-next-dual)"
assert_eq "get_n_ctx: the same words in a non-QEngine container's log are not read" "0" \
  "$(qe_ctx "vllm/vllm-openai:v0.32.0" 'ready, context 131072\n' some-vllm)"
assert_eq "get_n_ctx: CONTAINER=none reads no container log" "0" \
  "$(qe_ctx "$QE_IMG" '[qflash 46s] ready, context 262144\n' none)"
rm -rf "$qe_dir"

# ---- The summary (#1826): a skipped check is named, and the run isn't called sound ----
SUMMARY_FILE="$(mktemp --suffix=.sh)"
sed -n '/^if \[\[ "\$FAILED" == "0" && \${#SKIPPED_CHECKS/,/^done$/p' scripts/verify-stress.sh > "$SUMMARY_FILE"
assert_contains "$(cat "$SUMMARY_FILE")" "skipped (not run)" "summary block extracted"
out="$(bash -c 'FAILED=0; SKIPPED_CHECKS=(); source "$1"' _ "$SUMMARY_FILE" | sed 's/\x1b\[[0-9;]*m//g')"
assert_eq "summary: nothing skipped keeps the clean line" \
  "All stress / boundary checks passed. KV-cache and prefill paths are sound for the deployed config." "$out"
out="$(bash -c 'FAILED=0; SKIPPED_CHECKS=("ceiling_ladder: could not detect n_ctx"); source "$1"' _ "$SUMMARY_FILE" \
  | sed 's/\x1b\[[0-9;]*m//g')"
assert_not_contains "$out" "paths are sound" "summary: a skip drops the 'sound' claim"
assert_not_contains "$out" "All stress / boundary checks passed" "summary: a skip isn't catalog-baseline's full pass"
assert_eq "summary: still starts 'All stress' (the c3 parser)" "All stress" "${out:0:10}"
assert_contains "$out" "1 did not run" "summary: counts the skipped check"
assert_contains "$out" "skipped (not run): ceiling_ladder: could not detect n_ctx" "summary: names the skipped check"
out="$(bash -c 'FAILED=2; SKIPPED_CHECKS=("tool_prefill: SKIP_TOOL_PREFILL=1"); source "$1"' _ "$SUMMARY_FILE" \
  | sed 's/\x1b\[[0-9;]*m//g')"
assert_contains "$out" "2 stress check(s) failed." "summary: failures still lead"
assert_contains "$out" "skipped (not run): tool_prefill: SKIP_TOOL_PREFILL=1" "summary: skips listed after failures too"
# A failed check is named under the count (#1826: "1 stress check(s) failed" under two
# all-green buckets, with nothing saying it was the VRAM margin)
out="$(bash -c 'FAILED=1; SKIPPED_CHECKS=(); FAILED_CHECKS=(); ADVISORY_CHECKS=("vram_margin: 61 MB free at the ceiling, under the 1024 MB threshold (VRAM_MARGIN_MB): headroom, not correctness"); source "$1"' _ "$SUMMARY_FILE" \
  | sed 's/\x1b\[[0-9;]*m//g')"
assert_contains "$out" "1 stress check(s) failed. See hints above." "summary: the count line is unchanged (c3's FAIL_RE)"
assert_contains "$out" "  ⚠ advisory: vram_margin: 61 MB free at the ceiling" "summary: names the margin"
assert_not_contains "$out" "✗" "summary: a margin-only run prints no ✗ (report.sh's ADVISORY test)"
assert_not_contains "$out" "All stress" "summary: a failure never reads 'All stress'"
out="$(bash -c 'FAILED=2; SKIPPED_CHECKS=(); FAILED_CHECKS=("tool_prefill: no HTTP response (timeout or container died)"); ADVISORY_CHECKS=("vram_margin: 61 MB free"); source "$1"' _ "$SUMMARY_FILE" \
  | sed 's/\x1b\[[0-9;]*m//g')"
assert_contains "$out" "  ✗ failed: tool_prefill: no HTTP response" "summary: names a failed check"
assert_contains "$out" "  ⚠ advisory: vram_margin: 61 MB free" "summary: and the margin beside it"
rm -f "$SUMMARY_FILE"

# ---- run_check (#1826): a failed check is counted, bucketed and named by its first fail() ----
RUNCHECK_FILE="$(mktemp --suffix=.sh)"
{ command grep -E '^(pass|fail)\(\)' scripts/verify-stress.sh; sed -n '/^run_check()/,/^}/p' scripts/verify-stress.sh; } > "$RUNCHECK_FILE"
out="$(bash -c 'set -euo pipefail; FAILED=0; FAILED_CHECKS=(); SKIPPED_CHECKS=()
  _RECALL_CHECKS=" ceiling_ladder "; RECALL_RUN=0; RECALL_FAIL=0; BOUND_RUN=0; BOUND_FAIL=0; source "$1"
  check_dead() { fail "no HTTP response (timeout or container died)" "hint"; }
  check_wall() { fail "ceiling ladder: wall at ~150000 tok" "hint"; fail "a second message" "hint"; }
  check_ok() { pass "fine"; }
  run_check tool_prefill check_dead; run_check ceiling_ladder check_wall; run_check ide_agent check_ok
  printf "%s|%s|%s|%s\n" "$FAILED" "$BOUND_FAIL/$BOUND_RUN" "$RECALL_FAIL/$RECALL_RUN" "$(IFS=";"; echo "${FAILED_CHECKS[*]}")"' \
  _ "$RUNCHECK_FILE" | tail -1)"
assert_eq "run_check: failures counted, bucketed, named (label once, first message)" \
  "2|1/2|1/1|tool_prefill: no HTTP response (timeout or container died);ceiling_ladder: wall at ~150000 tok" "$out"
rm -f "$RUNCHECK_FILE"

# ---- vram_margin_check (#1826): the VRAM ceiling is a BOUNDARY check that runs inside the
# ladder, so it books itself there; FAILED stays the sum of the two buckets ----
MARGIN_FILE="$(mktemp --suffix=.sh)"
sed -n '/^vram_margin_check()/,/^}/p' scripts/verify-stress.sh > "$MARGIN_FILE"
assert_contains "$(cat "$MARGIN_FILE")" "BOUND_RUN" "vram_margin_check extracted"
margin_run() {   # <free before> <free at ceiling> → "FAILED|BOUND_FAIL/BOUND_RUN|FAILED_CHECKS|ADVISORY_CHECKS" + output
  bash -c 'set -euo pipefail; FAILED=0; FAILED_CHECKS=(); ADVISORY_CHECKS=(); BOUND_RUN=5; BOUND_FAIL=0; source "$1"
    vram_margin_check "$2" "$3" 1024 91
    printf "%s|%s|%s|%s\n" "$FAILED" "$BOUND_FAIL/$BOUND_RUN" "${FAILED_CHECKS[*]-}" "${ADVISORY_CHECKS[*]-}"' _ "$MARGIN_FILE" "$1" "$2" \
    | sed 's/\x1b\[[0-9;]*m//g'
}
out="$(margin_run 640 61)"
assert_contains "$out" "VRAM margin thin at ceiling: 61 MB free < 1024 MB threshold" "margin thin: the warning line rebench-report parses"
assert_eq "margin thin: a boundary failure, named as an advisory" \
  "1|1/6||vram_margin: 61 MB free at the ceiling, under the 1024 MB threshold (VRAM_MARGIN_MB): headroom, not correctness" "$(echo "$out" | tail -1)"
assert_not_contains "$out" "✗" "margin thin: no ✗ anywhere (report.sh's ADVISORY test)"
out="$(margin_run 9000 5000)"
assert_contains "$out" "VRAM: 9000 → 5000 MB (Δ -4000 MB across ladder, margin threshold=1024 MB)" "margin ok: the drop line rebench-report parses"
assert_eq "margin ok: a boundary pass" "0|0/6||" "$(echo "$out" | tail -1)"
rm -f "$MARGIN_FILE"

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
# The undetermined-GPU warning is said once, at the ladder start, not on every rung (#1826)
err="$(CONTAINER=none PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; get_vram_free_mb" 2>&1 >/dev/null)"
assert_contains "$err" "[vram] WARN: could not determine model GPU(s) on 2-GPU host" "undetermined GPUs: warns"
err="$(CONTAINER=none PATH="${tmp_dir}:/usr/bin:/bin" \
  bash -c "source '$HELPERS_FILE'; _VRAM_GPU_WARNED=1; get_vram_free_mb" 2>&1 >/dev/null)"
assert_eq "undetermined GPUs: quiet once warned" "" "$err"
assert_eq "the ladder sets the flag after its first read" "1" \
  "$(command grep -cF 'vram_before_all="$(get_vram_free_mb)"; _VRAM_GPU_WARNED=1' scripts/verify-stress.sh)"

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
assert_contains "$out" "skipped (not run): ceiling_ladder: SKIP_LONGCTX=1" "a requested skip is listed too"

# ---- Full runs, the #1826 shapes: a server that reports its context length nowhere ----
make_curl_mock '
  */v1/models*) printf '"'"'{"data":[{"id":"mock-model"}]}'"'"'; exit 0 ;;
  */v1/chat/completions*) printf '"'"'{"choices":[{"message":{"content":"mock response"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}'"'"'; exit 0 ;;'
out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_TOOL_PREFILL=1 \
  bash scripts/verify-stress.sh 2>&1 | sed 's/\x1b\[[0-9;]*m//g')" || true
assert_contains "$out" "could not detect n_ctx" "no n_ctx anywhere: the ladder is skipped"
assert_contains "$out" "set N_CTX=" "no n_ctx anywhere: the skip names N_CTX"
assert_contains "$out" "skipped (not run): ceiling_ladder: could not detect n_ctx" "no n_ctx anywhere: listed in the summary"
assert_not_contains "$out" "paths are sound" "no n_ctx anywhere: never 'sound'"

out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_TOOL_PREFILL=1 N_CTX=131072 \
  bash scripts/verify-stress.sh 2>&1 | sed 's/\x1b\[[0-9;]*m//g')" || true
assert_contains "$out" "n_ctx from N_CTX=131072" "N_CTX: says where n_ctx came from"
assert_contains "$out" "n_ctx=131072  ladder: 95000 → 120586" "N_CTX: the ladder runs on it"
assert_not_contains "$out" "skipped (not run): ceiling_ladder" "N_CTX: the ladder is not listed as skipped"

out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_TOOL_PREFILL=1 N_CTX=128k \
  bash scripts/verify-stress.sh 2>&1 | sed 's/\x1b\[[0-9;]*m//g')" || true
assert_contains "$out" "N_CTX='128k' is not a positive whole number" "a malformed N_CTX fails the ladder"

# A host-run QEngine (CONTAINER=none): no fingerprint, no /props, no container; /v1/models' owned_by
# names it, so its VRAM margin is read as Strata's is (#1826)
make_curl_mock '
  */v1/models*) printf '"'"'{"object":"list","data":[{"id":"qwen3.8-27b","object":"model","owned_by":"qengine"}]}'"'"'; exit 0 ;;
  */v1/chat/completions*) printf '"'"'{"choices":[{"message":{"content":"mock response"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}'"'"'; exit 0 ;;'
out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=qwen3.8-27b \
  CONTAINER=none SKIP_TOOL_PREFILL=1 SKIP_LONGCTX=1 \
  bash scripts/verify-stress.sh 2>&1 | sed 's/\x1b\[[0-9;]*m//g')" || true
assert_contains "$out" "container=none  engine=qengine" "a host-run QEngine is named by /v1/models owned_by"

# A small context is covered by probe 7: skipped, but not listed as a gap
make_curl_mock '
  */v1/models*) printf '"'"'{"data":[{"id":"mock-model","max_model_len":65536}]}'"'"'; exit 0 ;;
  */v1/chat/completions*) printf '"'"'{"choices":[{"message":{"content":"mock response"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10}}'"'"'; exit 0 ;;'
out="$(PATH="${tmp_dir}:$PATH" PREFLIGHT_NO_AUTODETECT=1 URL=http://mock MODEL=mock-model \
  CONTAINER=none SKIP_TOOL_PREFILL=1 \
  bash scripts/verify-stress.sh 2>&1 | sed 's/\x1b\[[0-9;]*m//g')" || true
assert_contains "$out" "probe 7 already covers this range" "small n_ctx: the ladder is skipped as covered"
assert_not_contains "$out" "skipped (not run): ceiling_ladder" "small n_ctx: a covered skip is not a gap"

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
