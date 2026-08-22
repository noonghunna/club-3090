#!/usr/bin/env bash
# Tests for the #1017 fixes in verify-stress.sh:
#   1. build_niah_payload salts every haystack head with SESSION-<32 random
#      alphanumerics>. — no two rungs share a byte-identical prefix, so vLLM's
#      prefix cache can't inflate reported prefill t/s.
#   2. Prefix-cache-hit guard: cache_hit_delta flags an increase of
#      vllm:prefix_cache_hits_total across a rung (the #710 discipline);
#      missing / non-numeric metrics degrade to a no-op.
#   3. get_prefix_cache_hits parses the Prometheus counter off /metrics.
#   4. --save-json format: record_rung + finalize_save_json produce a
#      parseable curve document labeled as a ceiling/addressability check,
#      naming contaminated rungs.
#   5. End-to-end: full script against a mock vLLM (hit counter rising every
#      scrape) annotates each rung line with CACHE-HIT and writes the curve.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

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

tmp_dir="$(mktemp -d)"
cleanup() { rm -rf "$tmp_dir"; }
trap cleanup EXIT

# ---- Extract helpers under test -------------------------------------------
HELPERS_FILE="$(mktemp --suffix=.sh)"
sed -n '/^build_niah_payload()/,/^}/p' scripts/verify-stress.sh > "$HELPERS_FILE"
sed -n '/^get_prefix_cache_hits()/,/^}/p' scripts/verify-stress.sh >> "$HELPERS_FILE"
sed -n '/^cache_hit_delta()/,/^}/p' scripts/verify-stress.sh >> "$HELPERS_FILE"
sed -n '/^record_rung()/,/^}/p' scripts/verify-stress.sh > "$tmp_dir/save_helpers.sh" || true
sed -n '/^finalize_save_json()/,/^}/p' scripts/verify-stress.sh >> "$tmp_dir/save_helpers.sh"

# ===========================================================================
# 1. Salted haystack builder — distinct heads per rung (#1017)
# ===========================================================================
export MODEL="mock-model"
gen_head() { # $1 = req file path → prints content head up to first \n\n
  local req="$1"
  local sec
  sec="$(mktemp --suffix=.secret)"
  bash -c "source '$HELPERS_FILE'; build_niah_payload ${2:-50} '$sec' '$req'"
  python3 -c "import json,sys; c=json.load(open('$req'))['messages'][0]['content']; print(c.split(chr(10))[0])"
  rm -f "$sec"
}

R1="$(mktemp --suffix=.json)"; R2="$(mktemp --suffix=.json)"
H1="$(gen_head "$R1")"
H2="$(gen_head "$R2")"

assert_contains "$H1" "SESSION-" "haystack head carries SESSION- salt prefix"
if [[ "$H1" =~ ^SESSION-[a-z0-9]{32}\.$ ]]; then
  PASS=$((PASS + 1))
else
  echo "FAIL: salt head '$H1' does not match SESSION-<32 alnum>." >&2
  FAIL=$((FAIL + 1))
fi
assert_contains "$(python3 -c "import json; print(json.load(open('$R1'))['messages'][0]['content'][:200])")" \
  "history of computing" "constant filler block still present after the salt"

if [[ "$H1" != "$H2" ]]; then
  PASS=$((PASS + 1))
else
  echo "FAIL: two consecutive rungs share an identical haystack head: '$H1'" >&2
  FAIL=$((FAIL + 1))
fi

# Distinctness must hold at scale — 24 rungs, expect ≥ 20 distinct heads
# (32 alphanumerics ≈ 62^32; collision odds are negligible, threshold is
# belt-and-braces against a broken RNG).
HEADS_FILE="$(mktemp)"
for i in $(seq 1 24); do
  gen_head "$(mktemp --suffix=.json)" >> "$HEADS_FILE"
done
distinct="$(sort -u "$HEADS_FILE" | wc -l)"
rm -f "$HEADS_FILE" "$R1" "$R2"
if [[ "$distinct" -ge 20 ]]; then
  PASS=$((PASS + 1))
else
  echo "FAIL: expected >=20 distinct haystack heads out of 24 rungs, got $distinct" >&2
  FAIL=$((FAIL + 1))
fi
# ===========================================================================
# 2. Prefix-cache-hit guard — triggers on a synthetic hit increase (#710 rule)
# ===========================================================================
_VS_JSONL="${tmp_dir}/rungs.jsonl"
SAVE_OUT="${tmp_dir}/curve.json"
# 2. Prefix-cache-hit guard — triggers on a synthetic hit increase (#710 rule)
# ===========================================================================
delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta 530048 530048")"
assert_eq "equal counters → clean" "0 0" "$delta"

delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta 530048.0 530500.0")"
assert_eq "rising counters → guard fires with delta" "452 1" "$delta"
delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta 100 50")"
delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta '' ''")"
assert_eq "missing metric → no-op (llama.cpp/SGLang)" "0 0" "$delta"

delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta garbage ''")"
assert_eq "garbage metric → no-op, never crashes" "0 0" "$delta"

delta="$(bash -c "source '$HELPERS_FILE'; cache_hit_delta 100 50")"
assert_eq "falling counters (restart) → not contaminated, real delta kept" "-50 0" "$delta"

# get_prefix_cache_hits parses the Prometheus counter via curl /metrics.
cat > "${tmp_dir}/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [[ "$a" == */metrics ]] && exec printf '# TYPE vllm:prefix_cache_hits_total counter\nvllm:prefix_cache_hits_total 1624951.0\n'; done
exit 1
EOF
chmod +x "${tmp_dir}/curl"
hits="$(URL=http://mock PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_prefix_cache_hits")"
assert_eq "get_prefix_cache_hits reads counter" "1624951.0" "$hits"

cat > "${tmp_dir}/curl" <<'EOF'
#!/usr/bin/env bash
exit 1
EOF
chmod +x "${tmp_dir}/curl"
hits="$(URL=http://mock PATH="${tmp_dir}:$PATH" \
  bash -c "source '$HELPERS_FILE'; get_prefix_cache_hits")"
assert_eq "get_prefix_cache_hits empty when /metrics absent" "" "$hits"

# ===========================================================================
# 3. --save-json parser round-trip (synthetic records → final document)
# ===========================================================================
SAVE_OUT="${tmp_dir}/curve.json"
_VS_JSONL="${tmp_dir}/rungs.jsonl"
SAVE_JSON="${SAVE_OUT}" :   # record_rung only needs _VS_JSONL set
bash -c "
  source '${tmp_dir}/save_helpers.sh'
  _VS_JSONL='${_VS_JSONL}'
  record_rung ceiling_ladder 1 95000 200 recalled 95102 1022.7 113300 0 0
  record_rung ceiling_ladder 2 125000 200 recall_miss 125310 1614.3 90000 150 1
  record_rung longctx scale-150 '' 400 skipped '' '' '' 0 0
  SAVE_JSON='${SAVE_OUT}' _VS_JSONL='${_VS_JSONL}'
  finalize_save_json
" >/dev/null

parsed="$(python3 - <<PY
import json
d = json.load(open("${SAVE_OUT}"))
r = d["rungs"]
out = [
    d["schema"],
    d["kind"],
    "ceiling/addressability" in d["label"],
    len(r),
    r[0]["outcome"], r[0]["recall_ok"], r[0]["cache_clean"], r[0]["prefill_tps"], r[0]["depth_tokens"],
    r[1]["cache_clean"], r[1]["cache_hit_delta"], r[1]["target_tokens"],
    r[2]["outcome"], r[2]["depth_tokens"],
    d["prefix_cache_guard"]["metric"],
    ",".join(d["prefix_cache_guard"]["contaminated_rungs"]),
]
print("|".join(str(x) for x in out))
PY
)"
assert_eq "save-json document parses with expected fields" \
  "verify-stress-curve/1|ceiling-addressability-check|True|3|recalled|True|True|1022.7|95102|False|150|125000|skipped|None|vllm:prefix_cache_hits_total|ceiling_ladder:rung-2" \
  "$parsed"

# ===========================================================================
# 4. End-to-end: mock vLLM whose hit counter rises on every scrape
# ===========================================================================
PORT=8137
cat > "${tmp_dir}/mock_vllm.py" <<'EOF'
import json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HITS = {"n": 400000}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send(200, json.dumps({"data":[{"id":"mock-model","max_model_len":262144}]}))
        elif self.path.startswith("/metrics"):
            HITS["n"] += 1000   # every scrape rises → every rung must flag
            self._send(200,
                "# HELP vllm:prefix_cache_hits_total hits\n# TYPE vllm:prefix_cache_hits_total counter\n"
                f"vllm:prefix_cache_hits_total {HITS['n']}.0\n", "text/plain")
        else:
            self._send(404, "{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        if req.get("stream"):
            m = re.search(r"The hidden phrase is '([^']+)'", req["messages"][0]["content"])
            secret = m.group(1) if m else "crimson otter 42"
            payload = (
                f'data: {json.dumps({"choices":[{"delta":{"content":secret}}]})}\n\n'
                f'data: {json.dumps({"choices":[],"usage":{"prompt_tokens":1234,"completion_tokens":5}})}\n\n'
                "data: [DONE]\n\n")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload.encode())
        else:
            self._send(200, json.dumps({"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],
                                        "usage":{"prompt_tokens":10,"completion_tokens":1}}))

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF

E2E_OUT="${tmp_dir}/e2e-curve.json"
python3 "${tmp_dir}/mock_vllm.py" "${PORT}" & MOCK_PID=$!
sleep 0.7
PREFLIGHT_NO_AUTODETECT=1 URL="http://127.0.0.1:${PORT}" MODEL=mock-model \
  CONTAINER=none SKIP_CEILING=1 STRESS_FAST=1 VERIFY_STRESS_RECORD=0 \
  bash scripts/verify-stress.sh --save-json "${E2E_OUT}" > "${tmp_dir}/e2e.log" 2>&1 || true
kill "${MOCK_PID}" 2>/dev/null || true
wait "${MOCK_PID}" 2>/dev/null || true

out="$(sed 's/\x1b\[[0-9;]*m//g' "${tmp_dir}/e2e.log")"
assert_contains "$out" "CEILING/ADDRESSABILITY check" "header labels ladder as ceiling/addressability check"
assert_contains "$out" "CACHE-HIT +1000 during rung" "guard annotates rung whose hit counter rose"
assert_contains "$out" "prefill inflated, NOT cache-clean" "annotation says the number is inflated"
assert_contains "$out" "prefill-vs-depth curve (2 rungs) saved" "curve persisted after run"

e2e_parsed="$(python3 - <<PY
import json
d = json.load(open("${E2E_OUT}"))
r = [x for x in d["rungs"] if x["probe"] == "longctx"]
print("|".join(str(x) for x in [
    d["kind"],
    len(r),
    all(x["outcome"] == "recalled" for x in r),
    all(not x["cache_clean"] and x["cache_hit_delta"] == 1000 for x in r),
    d["prefix_cache_guard"]["contaminated_rungs"],
]))
PY
)"
assert_eq "e2e curve: recalled rungs flagged cache-dirty with delta" \
  "ceiling-addressability-check|2|True|True|['longctx:rung-scale-150', 'longctx:rung-scale-450']" \
  "$e2e_parsed"

echo ""
echo "Results: ${PASS} passed, ${FAIL} failed"
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
echo "test-verify-stress-salt: ok"
