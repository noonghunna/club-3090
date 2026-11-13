#!/usr/bin/env bash
# test-report-non-vllm — the rebench report and bench.sh's KV capture on an engine that isn't vLLM.
#
# Why this exists (#1878, a Strata rebench):
#   - rebench-report.py's Config table reads only vLLM's command-line flags, so every row read "?";
#   - its paste-ready Quality line was quality-test.sh's header ("Quality: line for compose schema
#     field ..."), not the line under it — on every engine;
#   - bench.sh said "KV cache type unavailable": Strata's KV is on its engine CHILD's command line
#     (`strata --serve ... --kv int8 --kv-resident 32768`), not the server process's or /props.
#
# Drives the real rebench-report.py on fixture tag dirs, and the real cap_kv_type with docker stubbed.
# ⚠️ Against the pre-fix tree this must FAIL.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
fail() { echo "FAIL: test-report-non-vllm — $1" >&2; exit 1; }
pass() { echo "  ✓ $1"; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

# --- 1. rebench-report.py -------------------------------------------------------------
mk_tag() {  # <name> <container Cmd as a JSON array> <fingerprint lines...>
  local name="$1" d="$tmp/$1" cmd="$2"; shift 2
  mkdir -p "$d"
  printf '[{"Name":"/c-%s","Config":{"Image":"img-%s","Cmd":%s,"Env":[]},"State":{"Status":"running"},"Mounts":[]}]\n' \
    "$name" "$name" "$cmd" > "$d/container-config.json"
  { echo "========== CONFIG FINGERPRINT =========="; printf '%s\n' "$@"; echo ""; echo "========== SYSTEM RAM =========="; } > "$d/bench.log"
  printf '%s\n' "Quality: line for compose schema field (paste into compose YAML header):" \
    "Quality:   toolcall-15 13/15 (87%) (--full, thinking OFF, packs tc1.0.1, 2026-10-30)" > "$d/quality-full.log"
  echo '{"packs":[{"pack_id":"toolcall-15","passed":13,"total":15,"score":0.8667}]}' > "$d/quality-full.json"
  python3 scripts/rebench-report.py "$d" >/dev/null 2>&1 || fail "rebench-report.py failed on the $name fixture"
}

mk_tag strata '["qwen"]' \
  "  endpoint       : http://localhost:8018  (mode=chat)" \
  "  model          : qwen3.8-flash-next-iq3_s" \
  "  KV cache type  : int8, 32768 cells in VRAM (--kv-resident) (source: Strata engine argv)" \
  "  CPU offload    : DETECTED via strata host-ram"
R="$tmp/strata/REPORT.md"; D="$tmp/strata/REPORT-discuss.md"
! command grep -qE '\| `--[a-z-]+` \| `?\?`? \|' "$R" || fail "Strata report still has vLLM flag rows reading '?'"
command grep -qF '| KV cache type | int8, 32768 cells in VRAM (--kv-resident) (source: Strata engine argv) |' "$R" \
  || fail "the fingerprint's KV line is not in the Config table"
command grep -qF '| model | qwen3.8-flash-next-iq3_s |' "$R" || fail "the fingerprint's model line is not in the Config table"
pass "a non-vLLM report shows the engine's own fingerprint, and no vLLM row reads '?'"
command grep -qF 'Quality:   toolcall-15 13/15' "$R" || fail "the paste-ready Quality line is missing"
! command grep -qF 'Quality: line for compose schema field' "$R" || fail "the paste-ready Quality line is still quality-test.sh's header"
pass "the paste-ready Quality line is the line, not quality-test.sh's header above it"
command grep -qE '^Config: `img-strata` · KV cache type int8' "$D" || fail "the discussion summary's Config line doesn't use the fingerprint"
command grep -qF '# served_model_name = qwen3.8-flash-next-iq3_s' "$R" || fail "the reproducer's served name is not the reported model"
pass "the discussion summary and the reproducer read the fingerprint too"

mk_tag vllm '["--model","/m","--served-model-name","qwen3.8-27b","--max-model-len","262144","--kv-cache-dtype","fp8_e4m3","--tensor-parallel-size","2"]' \
  "  endpoint       : http://localhost:8020  (mode=chat)" \
  "  model          : qwen3.8-27b" \
  "  KV cache type  : fp8_e4m3 (source: argv)"
R="$tmp/vllm/REPORT.md"
command grep -qF '| `--max-model-len` | 262144 |' "$R" || fail "a vLLM report lost its flag table"
command grep -qF '| `--kv-cache-dtype` | `fp8_e4m3` |' "$R" || fail "a vLLM report lost its KV flag row"
command grep -qE '^Config: `img-vllm` · .*`fp8_e4m3` KV · `262144` ctx' "$tmp/vllm/REPORT-discuss.md" || fail "a vLLM discussion summary changed"
pass "a vLLM report keeps its flag table and summary line"
command grep -qF -- '- **Served as:** `qwen3.8-27b` from `/m`' "$R" || fail "a vLLM report must still name the served model and path from its flags"

# #2011: QEngine (like every non-vLLM engine) has no --served-model-name / --model on its command line,
# so the Meta line read "Served as: `?` from `?`". The name comes from what the server reported, the
# path from bench.sh's `serving argv` model= token.
mk_tag qengine '["bash","-c","exec /app/server/run.sh"]' \
  "  endpoint       : http://localhost:8203  (mode=chat)" \
  "  model          : qwen3.8-flash-next" \
  "  serving argv   : model=/app/models/Qwen3.8-Flash-Next-QE-INT4" \
  "  KV cache type  : fp8_e4m3 (QEngine's only KV format)"
R="$tmp/qengine/REPORT.md"
command grep -qF -- '- **Served as:** `qwen3.8-flash-next` from `/app/models/Qwen3.8-Flash-Next-QE-INT4`' "$R" \
  || fail "a QEngine report must name what it served and from where (#2011): $(command grep -F 'Served as' "$R")"
R="$tmp/strata/REPORT.md"
command grep -qxF -- '- **Served as:** `qwen3.8-flash-next-iq3_s`' "$R" \
  || fail "with no model path anywhere, the Meta line names the served model and leaves the path out: $(command grep -F 'Served as' "$R")"
pass "the Meta line's served name and path come from the fingerprint when the command line has none (#2011)"

# --- 2. cap_kv_type on Strata: the engine child's flags -----------------------------------
bin="$tmp/bin"; mkdir -p "$bin"
# Like the real daemon, `docker top` refuses a ps format without a PID column ("Couldn't find PID field
# in ps output"): a capture asking for args alone got nothing back, live, while a looser stub passed.
cat > "$bin/docker" <<'EOF'
#!/usr/bin/env bash
[[ "$1" == top ]] || exit 1
[[ "$*" == *" -eo pid,"* ]] || { echo "Error response from daemon: Couldn't find PID field in ps output" >&2; exit 1; }
echo "PID                 COMMAND"
echo "101                 python3 -m serve.server --engine strata --port 8080 --parallel 3"
[[ "${STUB_NO_ENGINE:-0}" == 1 ]] || echo "102                 /opt/strata/components/engine/0.1.55/strata --serve --pack /data/p --max-context 131072 --kv int8 --ple-io ram --kv-resident 32768 --layer-split auto"
EOF
chmod +x "$bin/docker"
kv() { PATH="$bin:$PATH" bash -c 'source scripts/lib/capture.sh; URL=http://127.0.0.1:9; CAP_LOG=""; CONTAINER="$1"
  cap_kv_type "" "$2" 2>/dev/null || echo UNAVAILABLE' _ "$@"; }
got="$(kv strata-x strata)"
[[ "$got" == "int8, 32768 cells in VRAM (--kv-resident) (source: Strata engine argv)" ]] || fail "Strata KV capture: got '$got'"
# TabbyAPI (exllamav3) takes its KV as --cache-mode on its own command line; bench said "unavailable"
got="$(bash -c 'source scripts/lib/capture.sh; URL=http://127.0.0.1:9; CAP_LOG=""; CONTAINER=none
  cap_kv_type "python3 main.py --port 5000 --backend exllamav3 --max-seq-len 204800 --cache-size 204800 --cache-mode Q4 --gpu-split-auto true" exllamav3 2>/dev/null || echo UNAVAILABLE')"
[[ "$got" == "Q4 (source: argv)" ]] || fail "TabbyAPI KV capture from --cache-mode: got '$got'"
pass "cap_kv_type reads Strata's KV off its engine child's command line"
got="$(STUB_NO_ENGINE=1 kv strata-x strata)"
[[ "$got" == UNAVAILABLE ]] || fail "no engine child yet: expected unavailable, got '$got' (never another engine's default)"
got="$(kv none strata)"
[[ "$got" == UNAVAILABLE ]] || fail "CONTAINER=none (host-only): expected unavailable, got '$got'"
pass "no engine child, or no container: still honestly unavailable"

echo "test-report-non-vllm: ok"
