#!/usr/bin/env bash
# test-run-context — a quality report must record the sampling the engine
# APPLIED and the rig it ran on (club-3090#1396).
#
# WHY THIS TEST EXISTS
# --------------------
# vLLM and SGLang expose no sampling-defaults endpoint, so every
# `--sampling-from-server` report read "value not exposed by endpoint", and no
# report recorded TP size or GPUs — results from different rigs could not be
# compared, and in #1396 the sampling was a main suspect in a cross-rig gap.
# scripts/lib/run_context.py resolves both from the serving container; the
# wrapper hands them to benchlocal-cli (--server-defaults / --run-meta).
#
# THE ASSERTIONS HAVE TO DISCRIMINATE
# -----------------------------------
# The obvious implementation reads the flags we PASSED — vLLM's
# `override_generation_config`, SGLang's `preferred_sampling_params` — and is
# wrong on both engines: vLLM filters the override to an allowlist (a compose's
# presence_penalty never applies), and SGLang's preferred params are inert on
# /v1/chat/completions (sglang#39096). So every fixture below makes the flag
# DIFFER from what the engine applies, and asserts the applied value. A
# flag-reading implementation fails legs 1 and 4; "the report mentions the
# sampling" would pass it.
#
# Contract:
#   1-3. vLLM: the effective line wins; the filtered override without it
#        (generation_config=vllm); nothing logged → null + a note, never a guess.
#   4-5. SGLang: model generation_config + the chat adapter's fallbacks; with no
#        model defaults → the OpenAI fallbacks; preferred params are noted as
#        not in effect.
#   6.   GPUs: only those the container's CUDA sees (DeviceIDs, then
#        CUDA_VISIBLE_DEVICES indexing INTO them), power cap, PCIe, NVLink.
#   7.   --engine is required (the engine family is engine-kind.sh's call, #1282);
#        a log with no dump says so instead of emitting an empty rig.
#   8.   the wrapper passes both flags; drops --server-defaults without
#        --sampling-from-server; re-reads nothing on --resume; warns on a
#        benchlocal-cli that predates the flags.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RC_PY="${ROOT}/scripts/lib/run_context.py"
WRAPPER="${ROOT}/scripts/quality-test.sh"
fails=0
fail() { echo "✗ $*" >&2; fails=$((fails+1)); }
assert_contains() { [[ "$1" == *"$2"* ]] || { fail "expected to contain: $2"; echo "--- actual ---" >&2; echo "$1" >&2; }; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || { fail "expected NOT to contain: $2"; echo "--- actual ---" >&2; echo "$1" >&2; }; }
jq_py() { python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(eval(sys.argv[1]), sort_keys=True))" "$1"; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# ---------------------------------------------------------------- fixtures
cat > "$tmp/vllm.log" <<'LOG'
(APIServer pid=1) INFO 09-24 02:18:50 [utils.py:300] LLM engine (v0.30.0) with config: model='/models/x'
(APIServer pid=1) INFO 09-24 02:18:51 [api_utils.py:286] non-default args: {'model_tag': '/models/x', 'override_generation_config': {'temperature': 0.7, 'min_p': 0.0, 'presence_penalty': 1.5}, 'tensor_parallel_size': 4, 'quantization': 'auto_round', 'kv_cache_dtype': 'fp8_e5m2', 'max_model_len': 131072, 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 3}}
(APIServer pid=1) WARNING 09-24 02:24:16 [model.py:1772] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'min_p': 0.0}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
LOG
cat > "$tmp/vllm-genvllm.log" <<'LOG'
(APIServer pid=1) INFO 09-24 02:18:51 [api_utils.py:286] non-default args: {'generation_config': 'vllm', 'override_generation_config': {'temperature': 0.6, 'presence_penalty': 1.5}, 'tensor_parallel_size': 2}
LOG
cat > "$tmp/vllm-neutral.log" <<'LOG'
(APIServer pid=1) INFO 09-24 02:18:51 [api_utils.py:286] non-default args: {'generation_config': 'vllm', 'tensor_parallel_size': 1}
LOG
cat > "$tmp/sglang.log" <<'LOG'
[2026-09-24 01:50:02] server_args={'model_path': '/models/x', 'tp_size': 8, 'pp_size': 1, 'quantization': 'auto-round', 'kv_cache_dtype': 'fp8_e4m3', 'context_length': 262144, 'speculative_algorithm': 'DFLASH', 'speculative_num_draft_tokens': 8, 'preferred_sampling_params': '{"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}', 'sampling_defaults': 'model'}
[2026-09-24 01:52:16] Using default chat sampling params from model generation config: {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}
LOG
cat > "$tmp/sglang-nomodel.log" <<'LOG'
[2026-09-24 01:50:02] server_args={'model_path': '/models/x', 'tp_size': 2, 'sampling_defaults': 'openai'}
LOG
echo "no engine dump here" > "$tmp/empty.log"
cat > "$tmp/smi.csv" <<'CSV'
0, GPU-aaaa0000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 350.00 W, 4, 16
1, GPU-bbbb0000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 230.00 W, 4, 8
2, GPU-cccc0000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 230.00 W, 4, 8
3, GPU-dddd0000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 5090, 575.00 W, 5, 16
CSV
cat > "$tmp/nvlink-off.txt" <<'TXT'
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-bbbb0000-0000-0000-0000-000000000000)
NVML: Unable to retrieve Nvlink information as all links are inActive
GPU 2: NVIDIA GeForce RTX 3090 (UUID: GPU-cccc0000-0000-0000-0000-000000000000)
NVML: Unable to retrieve Nvlink information as all links are inActive
TXT
cat > "$tmp/nvlink-on.txt" <<'TXT'
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-bbbb0000-0000-0000-0000-000000000000)
	 Link 0: 14.062 GB/s
GPU 2: NVIDIA GeForce RTX 3090 (UUID: GPU-cccc0000-0000-0000-0000-000000000000)
	 Link 0: 14.062 GB/s
TXT
inspect() {  # $1 = DeviceIDs JSON (or null), $2 = extra Env entries (JSON list body)
  printf '[{"Config":{"Image":"vllm/vllm-openai:v0.30.0","Env":["PATH=/usr/bin"%s]},"HostConfig":{"DeviceRequests":[{"Driver":"nvidia","Count":%s,"DeviceIDs":%s}]}}]' \
    "${2:+,$2}" "$([[ "$1" == null ]] && echo -1 || echo 0)" "$1"
}
rc() { python3 "$RC_PY" "$@"; }

# ---------------------------------------------------------------- 1. vLLM effective line
echo "--- 1. vLLM: the effective (post-allowlist) defaults, not the override ---"
out="$(rc --engine vllm --boot-log "$tmp/vllm.log" 2>"$tmp/err")"
got="$(jq_py 'd["server_defaults"]' <<<"$out")"
assert_contains "$got" '"presence_penalty": 0.0'   # the override said 1.5; vLLM drops it
assert_contains "$got" '"temperature": 0.7'
assert_contains "$got" '"top_p": 0.8'           # top_p/top_k come from generation_config.json,
assert_contains "$got" '"top_k": 20'             # which only the effective line carries
assert_contains "$got" '"source": "vLLM effective defaults (generation_config.json + --override-generation-config)"'
assert_not_contains "$got" '1.5'
assert_contains "$(cat "$tmp/err")" "vLLM ignores presence_penalty=1.5 in --override-generation-config"
got="$(jq_py 'd["run_meta"]' <<<"$out")"
assert_contains "$got" '"engine": "vllm 0.30.0"'
assert_contains "$got" '"tp": "4"'
assert_contains "$got" '"spec": "mtp n=3"'
assert_contains "$got" '"kv": "fp8_e5m2"'
assert_contains "$got" '"max_ctx": "131072"'
assert_contains "$got" '"quant": "auto_round"'

# ---------------------------------------------------------------- 2-3. vLLM without the line
echo "--- 2. vLLM generation_config=vllm: the override, filtered to what vLLM honours ---"
got="$(rc --engine vllm --boot-log "$tmp/vllm-genvllm.log" 2>/dev/null | jq_py 'd["server_defaults"]')"
assert_contains "$got" '"temperature": 0.6'
assert_contains "$got" '"presence_penalty": 0.0'
assert_contains "$got" 'generation_config=vllm'
echo "--- 3. vLLM with nothing logged: null and a note, never a guess ---"
out="$(rc --engine vllm --boot-log "$tmp/vllm-neutral.log" 2>"$tmp/err")"
assert_contains "$(jq_py 'd["server_defaults"]' <<<"$out")" "null"
assert_contains "$(cat "$tmp/err")" "neutral defaults apply"

# ---------------------------------------------------------------- 4-5. SGLang
echo "--- 4. SGLang: model generation_config + chat fallbacks; preferred params NOT in effect ---"
out="$(rc --engine sglang --boot-log "$tmp/sglang.log" 2>"$tmp/err")"
got="$(jq_py 'd["server_defaults"]' <<<"$out")"
assert_contains "$got" '"temperature": 1.0'      # preferred said 0.7 — inert on chat
assert_contains "$got" '"top_p": 0.95'
assert_contains "$got" '"top_k": 20'
assert_contains "$got" '"min_p": 0.0'            # chat-adapter fallback
assert_contains "$got" '"repetition_penalty": 1.0'
assert_contains "$got" '"presence_penalty": 0.0'
assert_not_contains "$got" '0.7'
assert_contains "$(cat "$tmp/err")" "sglang#39096"
got="$(jq_py 'd["run_meta"]' <<<"$out")"
assert_contains "$got" '"tp": "8"'
assert_contains "$got" '"spec": "DFLASH n=8"'
echo "--- 5. SGLang with no model defaults: the OpenAI fallbacks ---"
got="$(rc --engine sglang --boot-log "$tmp/sglang-nomodel.log" 2>/dev/null | jq_py 'd["server_defaults"]')"
assert_contains "$got" '"top_k": -1'
assert_contains "$got" '"top_p": 1.0'
assert_contains "$got" 'OpenAI fallbacks'

# ---------------------------------------------------------------- 6. GPUs
echo "--- 6. GPUs: only the ones the container's CUDA sees ---"
inspect '["1","3"]' '"CUDA_VISIBLE_DEVICES=1"' > "$tmp/i1.json"
got="$(rc --engine vllm --boot-log "$tmp/vllm.log" --inspect-json "$tmp/i1.json" --smi-csv "$tmp/smi.csv" 2>/dev/null | jq_py 'd["run_meta"]')"
assert_contains "$got" '"gpus": "1x RTX 5090"'   # CUDA index 1 INTO DeviceIDs [1,3] = host GPU 3
assert_contains "$got" '"power_cap": "575 W"'
assert_not_contains "$got" 'nvlink'              # one GPU: nothing to link
inspect '["1","2"]' > "$tmp/i2.json"
got="$(rc --engine vllm --boot-log "$tmp/vllm.log" --inspect-json "$tmp/i2.json" --smi-csv "$tmp/smi.csv" --nvlink-txt "$tmp/nvlink-off.txt" 2>/dev/null | jq_py 'd["run_meta"]')"
assert_contains "$got" '"gpus": "2x RTX 3090"'
assert_contains "$got" '"power_cap": "230 W"'
assert_contains "$got" '"pcie": "gen4 x8"'
assert_contains "$got" '"nvlink": "no"'
got="$(rc --engine vllm --boot-log "$tmp/vllm.log" --inspect-json "$tmp/i2.json" --smi-csv "$tmp/smi.csv" --nvlink-txt "$tmp/nvlink-on.txt" 2>/dev/null | jq_py 'd["run_meta"]')"
assert_contains "$got" '"nvlink": "yes"'
inspect null '"NVIDIA_VISIBLE_DEVICES=all","CUDA_VISIBLE_DEVICES"' > "$tmp/i3.json"
got="$(rc --engine vllm --boot-log "$tmp/vllm.log" --inspect-json "$tmp/i3.json" --smi-csv "$tmp/smi.csv" 2>/dev/null | jq_py 'd["run_meta"]')"
assert_contains "$got" '"gpus": "3x RTX 3090, 1x RTX 5090"'
assert_contains "$got" '"power_cap": "350 W / 230 W / 230 W / 575 W"'

# ---------------------------------------------------------------- 7. engine is an input; no-dump honesty
echo "--- 7. --engine is required; a missing dump is said, not emitted empty ---"
python3 "$RC_PY" --boot-log "$tmp/vllm.log" >/dev/null 2>&1 && fail "run_context.py ran without --engine (it must not classify, #1282)"
out="$(rc --engine vllm --boot-log "$tmp/empty.log" 2>"$tmp/err")"
assert_contains "$(cat "$tmp/err")" "startup dump not found"
assert_not_contains "$(jq_py 'd["run_meta"]' <<<"$out")" '"tp"'
out="$(rc --engine llamacpp --boot-log "$tmp/empty.log" --inspect-json "$tmp/i2.json" 2>/dev/null)"
assert_contains "$(jq_py 'd["server_defaults"]' <<<"$out")" "null"
assert_contains "$(jq_py 'd["run_meta"]' <<<"$out")" '"engine": "llamacpp v0.30.0"'
args="$(rc --engine vllm --boot-log "$tmp/vllm.log" --emit-args --no-server-defaults 2>/dev/null)"
assert_not_contains "$args" "--server-defaults"
assert_contains "$args" $'--run-meta\ntp=4'

# ---------------------------------------------------------------- 8. the wrapper
echo "--- 8. quality-test.sh hands both to benchlocal-cli ---"
bin="$tmp/bin"; mkdir -p "$bin"
cp "$tmp/vllm.log" "$tmp/boot.log"; cp "$tmp/i2.json" "$tmp/inspect.json"
cat > "$bin/docker" <<MOCK
#!/usr/bin/env bash
case "\$1" in
  inspect)
    for a in "\$@"; do case "\$a" in --format*|'{{'*) fmt="\$a" ;; esac; done
    [[ "\$2" == "vllm-mock" || "\$3" == "vllm-mock" ]] || exit 1
    if [[ -n "\${fmt:-}" ]]; then
      case "\$*" in *Config.Image*) echo "vllm/vllm-openai:v0.30.0" ;; *RestartCount*) echo 0 ;; esac
    else cat "$tmp/inspect.json"; fi ;;
  logs) cat "$tmp/boot.log" ;;
  *) exit 1 ;;
esac
MOCK
cat > "$bin/nvidia-smi" <<MOCK
#!/usr/bin/env bash
case "\$*" in *nvlink*) cat "$tmp/nvlink-off.txt" ;; *query-gpu*) cat "$tmp/smi.csv" ;; esac
MOCK
cat > "$bin/curl" <<'MOCK'
#!/usr/bin/env bash
for a in "$@"; do case "$a" in */v1/models) printf '{"data":[{"id":"mock-model"}]}'; exit 0 ;; esac; done
exit 0
MOCK
cat > "$bin/benchlocal-cli" <<'MOCK'
#!/usr/bin/env bash
if [[ "${1:-}" == "run" && "${2:-}" == "--help" ]]; then
  [[ "${MOCK_OLD_BLC:-0}" == "1" ]] && { echo "--reasoning-effort"; exit 0; }
  echo "--reasoning-effort --run-meta --server-defaults"; exit 0
fi
printf '%s\n' "$@" > "${MOCK_ARGV}"
exit 0
MOCK
chmod +x "$bin"/*
qrun() {
  PATH="${bin}:$PATH" MOCK_ARGV="$tmp/argv" PREFLIGHT_NO_AUTODETECT=1 \
    URL=http://mock MODEL=mock-model CONTAINER=vllm-mock bash "$WRAPPER" --quick "$@" 2>&1
}
: > "$tmp/argv"; out="$(qrun --sampling-from-server)"; argv="$(cat "$tmp/argv")"
assert_contains "$argv" $'--run-meta\ntp=4'
assert_contains "$argv" $'--run-meta\ngpus=2x RTX 3090'
assert_contains "$argv" $'--server-defaults\n{"temperature":0.7,'
assert_contains "$out" "[quality-test] rig (vllm): engine=vllm 0.30.0 · tp=4"
assert_contains "$out" "server defaults resolved from the vllm boot log"
: > "$tmp/argv"; out="$(qrun)"; argv="$(cat "$tmp/argv")"
assert_contains "$argv" $'--run-meta\ntp=4'
assert_not_contains "$argv" "--server-defaults"   # benchlocal-cli refuses it without --sampling-from-server
: > "$tmp/argv"; out="$(MOCK_OLD_BLC=1 qrun --sampling-from-server)"; argv="$(cat "$tmp/argv")"
assert_contains "$out" "predates --run-meta/--server-defaults (#1396)"
assert_not_contains "$argv" "--run-meta"
echo '{}' > "$tmp/resume.partial.jsonl"
: > "$tmp/argv"; out="$(PATH="${bin}:$PATH" MOCK_ARGV="$tmp/argv" PREFLIGHT_NO_AUTODETECT=1 \
  URL=http://mock MODEL=mock-model CONTAINER=vllm-mock bash "$WRAPPER" --resume "$tmp/resume.partial.jsonl" 2>&1)"
argv="$(cat "$tmp/argv")"
[[ -n "$argv" ]] || fail "the --resume leg never reached benchlocal-cli — its assertion would be vacuous"
assert_not_contains "$argv" "--run-meta"            # the journal already holds the first session's values

if [[ "$fails" -gt 0 ]]; then
  echo "test-run-context: $fails failure(s)" >&2
  exit 1
fi
echo "test-run-context: ok"
