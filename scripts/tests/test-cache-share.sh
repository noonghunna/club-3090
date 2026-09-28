#!/usr/bin/env bash
# test-cache-share — scripts/cache-share.sh reports each engine's prompt-cache
# share from its OWN counters, per engine family, and says so when it can't.
#
# WHY THIS TEST EXISTS
# --------------------
# The number is only useful if it is right for each engine: vLLM and SGLang name
# and split their counters differently (vLLM by prompt-token source, SGLang by
# cache source), a family without a cached-token counter must be reported as
# "not measured" rather than as 0 %, and --watch must report per-interval deltas,
# not cumulative totals. Offline: two stub servers stand in for the engines.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; rm -rf "$T"; }
trap cleanup EXIT
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

cat > "$T/stub.py" <<'PY'
# A fake engine: /v1/models with an owned_by, and a /metrics page whose
# counters grow by a fixed step on every scrape (so --watch sees deltas).
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, kind = int(sys.argv[1]), sys.argv[2]
hits = {"n": 0}
def metrics(n):
    if kind == "vllm":      # 1000 prompt tok/step: 900 GPU hit, 50 offload, 50 computed
        p, g, o = 10000 + 1000 * n, 8000 + 900 * n, 500 + 50 * n
        return (f'vllm:prompt_tokens_total{{engine="0",model_name="m"}} {p}\n'
                f'vllm:prompt_tokens_cached_total{{engine="0",model_name="m"}} {g + o}\n'
                f'vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_cache_hit"}} {g}\n'
                f'vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="external_kv_transfer"}} {o}\n'
                f'vllm:prompt_tokens_by_source_total{{engine="0",model_name="m",source="local_compute"}} {p - g - o}\n')
    if kind == "sglang":    # two is_streaming series; device + host cache sources
        return ('sglang:prompt_tokens_total{is_streaming="false",model_name="m"} 1000.0\n'
                f'sglang:prompt_tokens_total{{is_streaming="true",model_name="m"}} {3000 + 2000 * n}\n'
                f'sglang:cached_tokens_total{{cache_source="device",model_name="m"}} {3000 + 1900 * n}\n'
                'sglang:cached_tokens_total{cache_source="host",model_name="m"} 200.0\n'
                'sglang:cache_hit_rate{model_name="m"} 0.0\n')
    return "llamacpp:prompt_tokens_total 42\n"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            owned = {"vllm": "vllm", "sglang": "sglang", "llamacpp": "llamacpp"}[kind]
            b = json.dumps({"data": [{"id": f"{kind}-model", "owned_by": owned}]}).encode()
        elif self.path.startswith("/metrics"):
            b = metrics(hits["n"]).encode(); hits["n"] += 1
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", port), H).serve_forever()
PY
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'; }
start() { python3 "$T/stub.py" "$1" "$2" & PIDS+=($!); }
PV=$(free_port); PS=$(free_port); PL=$(free_port)
start "$PV" vllm; start "$PS" sglang; start "$PL" llamacpp
for p in "$PV" "$PS" "$PL"; do
  for _ in $(seq 1 50); do curl -sf -o /dev/null "http://127.0.0.1:$p/v1/models" && break; sleep 0.1; done
done

# 1. vLLM since boot: 10,000 prompt, 8,000 GPU + 500 offload = 85.0 %, 1,500 prefilled
out="$(bash "$ROOT/scripts/cache-share.sh" --port "$PV" 2>&1)"
[[ "$out" == *":$PV  vllm  vllm-model"* ]] || bad "vLLM header (engine via owned_by): $out"
[[ "$out" == *"10,000 prompt tok"*"85.0% from cache (GPU 80.0%, offload 5.0%)"*"1,500 prefilled"* ]] || bad "vLLM since-boot line: $out"

# 2. SGLang sums both is_streaming series and both cache sources: 4,000 prompt,
#    3,000 device + 200 host = 80.0 %
out="$(bash "$ROOT/scripts/cache-share.sh" --port "$PS" 2>&1)"
[[ "$out" == *"4,000 prompt tok"*"80.0% from cache (GPU 75.0%, offload 5.0%)"*"800 prefilled"* ]] || bad "SGLang since-boot line: $out"

# 3. llama.cpp has no cached-token counter: listed, never reported as 0 %
out="$(bash "$ROOT/scripts/cache-share.sh" --port "$PL" 2>&1)"
[[ "$out" == *"llamacpp: no cached-token counter"* && "$out" != *"% from cache"* ]] || bad "llama.cpp must be 'not measured', not 0 %: $out"

# 4. --watch prints per-INTERVAL deltas (the stub adds 2,000 prompt / 1,900 cached per scrape on SGLang)
out="$(bash "$ROOT/scripts/cache-share.sh" --port "$PS" --watch 1 --count 2 2>&1)"
n=$(command grep -cE "2,000 prompt tok · +95\.0% from cache" <<<"$out")
[[ "$n" == "2" ]] || bad "--watch must print 2 interval lines of +2,000 tok at 95.0 % (got $n): $out"

# 5. nothing serving → non-zero, a clear message
dead=$(free_port)
if bash "$ROOT/scripts/cache-share.sh" --port "$dead" >/dev/null 2>"$T/err"; then bad "nothing serving must exit non-zero"; fi
command grep -q "nothing is serving" "$T/err" || bad "nothing serving must say so: $(cat "$T/err")"

[[ $fail -eq 0 ]] && echo "test-cache-share: ok (vLLM source split, SGLang series + sources, llama.cpp not measured, --watch deltas, nothing serving)"
exit $fail
