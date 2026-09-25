#!/usr/bin/env bash
# test-kv-offload-probe — scripts/kv-offload-probe.py must keep recognising each engine's offload counters.
#
# WHY THIS TEST EXISTS
# --------------------
# The probe's PASS needs two things: A's return is fast AND the engine's own offload counters moved. If a
# counter name drifts (an engine rename, a label reshuffle), the counter half silently reads "nothing moved"
# and every working tier reports FAIL — or, worse, a pattern that matches too broadly (the device-side
# cached_tokens source) turns a plain GPU hit into a PASS. This feeds real metric lines from both engines
# and asserts exactly which ones count as offload evidence. Offline: no server, no GPU.
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

python3 -m py_compile scripts/kv-offload-probe.py || { echo "✗ kv-offload-probe.py does not compile" >&2; exit 1; }
out="$(python3 scripts/kv-offload-probe.py --url http://127.0.0.1:1 --disk 2>&1)"
[[ "$out" == *"--disk needs --container"* ]] || { echo "✗ --disk without --container must be refused (got: $out)" >&2; exit 1; }

python3 - <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("probe", "scripts/kv-offload-probe.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
fails = []

SGL = '''# HELP sglang:cached_tokens_total Number of cached prompt tokens by source
sglang:cached_tokens_total{cache_source="device",model_name="qwen3.8-27b",tp_rank="0"} 40128.0
sglang:cached_tokens_total{cache_source="host",model_name="qwen3.8-27b",tp_rank="0"} 39488.0
sglang:cached_tokens_total{cache_source="storage",model_name="qwen3.8-27b",tp_rank="0"} 40256.0
sglang:load_back_tokens_total{cache_type="UnifiedRadixCache",pool="kv",tp_rank="0"} 80512.0
sglang:load_back_tokens_created{cache_type="UnifiedRadixCache",pool="kv",tp_rank="0"} 1.7e9
sglang:num_requests_total{model_name="qwen3.8-27b"} 12.0
'''
VLLM = '''# TYPE vllm:external_prefix_cache_hits_total counter
vllm:external_prefix_cache_hits_total{engine="0",model_name="qwen3.8-27b"} 78864.0
vllm:kv_offload_load_bytes_total{engine="0",model_name="qwen3.8-27b"} 2.9e9
vllm:gpu_prefix_cache_hits_total{engine="0",model_name="qwen3.8-27b"} 500000.0
vllm:prompt_tokens_total{engine="0",model_name="qwen3.8-27b"} 900000.0
'''
for engine, text, want in (
    ("sglang", SGL, {"sglang:cached_tokens_total [host]": 39488.0, "sglang:cached_tokens_total [storage]": 40256.0,
                     "sglang:load_back_tokens_total": 80512.0}),
    ("vllm", VLLM, {"vllm:external_prefix_cache_hits_total": 78864.0, "vllm:kv_offload_load_bytes_total": 2.9e9}),
):
    probe.http = lambda url, body=None, timeout=0, _t=text: _t
    got = probe.evidence("http://x", engine)
    if got != want:
        fails.append(f"{engine}: evidence() = {got}, want {want} (device/GPU hits and _created lines must NOT count)")
# detect(): SGLang reads /server_info; vLLM must use the kv_cache_size_tokens label, NOT num_gpu_blocks x
# block_size (on hybrid models that product read 11,840 against a real 591,422 and sized the fillers to nothing).
import json as _json
class _NotFound(Exception): pass
def fake(routes):
    def h(url, body=None, timeout=0):
        for suffix, val in routes.items():
            if url.endswith(suffix):
                if isinstance(val, Exception): raise val
                return val
        raise probe.urllib.error.URLError("404")
    return h
MODELS = {"data": [{"id": "qwen3.8-27b", "max_model_len": 262144}]}
probe.http = fake({"/v1/models": MODELS, "/server_info": {"max_total_num_tokens": 548520, "internal_states": [{}]}})
if probe.detect("http://x") != ("sglang", "qwen3.8-27b", 262144, 548520):
    fails.append(f"detect() on SGLang = {probe.detect('http://x')}, want pool 548,520 from /server_info")
VMETRICS = ('vllm:cache_config_info{block_size="848",kv_cache_size_tokens="591422",mamba_block_size="16",num_gpu_blocks="14"} 1.0\n')
probe.http = fake({"/v1/models": MODELS, "/server_info": probe.urllib.error.URLError("404"), "/metrics": VMETRICS})
if probe.detect("http://x") != ("vllm", "qwen3.8-27b", 262144, 591422):
    fails.append(f"detect() on vLLM = {probe.detect('http://x')}, want pool 591,422 from kv_cache_size_tokens (not 14 x 848)")

for f in fails:
    print("✗", f, file=sys.stderr)
sys.exit(1 if fails else 0)
PY
[[ $? -eq 0 ]] || exit 1
echo "test-kv-offload-probe: ok (compiles, arg guard, pool detection on both engines, offload counters parsed; device-side hits excluded)"
