#!/usr/bin/env bash
# concurrency-probe.sh — the #246 Phase 2 measurement: how many concurrent
# long-context streams a card's KV pool sustains cleanly.
#
# soak-test.sh is SINGLE-STREAM by design (no concurrency stress); this is its
# concurrent sibling. It fires N concurrent long requests over R rounds and
# checks completion + no-silent-empty + bounded VRAM growth — the validation
# behind an envelopes.yml `max_num_seqs` row (born-from-measurement discipline).
#
# It VALIDATES a candidate N (does the server, booted at MAX_NUM_SEQS=N, hold
# N concurrent long streams?), it does not search — kv-calc proposes N from the
# pool, this confirms it. Boot the server at the candidate MAX_NUM_SEQS first.
#
# Usage:
#   MAX_NUM_SEQS=4 bash scripts/switch.sh vllm/dual   # boot at the candidate
#   bash scripts/concurrency-probe.sh                 # probe = server's cap
#   CONCURRENCY=4 bash scripts/concurrency-probe.sh   # force N
#
# Env: URL (default http://localhost:8010) · MODEL (auto) · CONTAINER (auto for
#   VRAM) · CONCURRENCY (default: detect served max-num-seqs, else 2) · ROUNDS
#   (default 5) · PROMPT_TOKENS (default 16000) · GEN_TOKENS (default 256) ·
#   VRAM_GROWTH_MB (default 200).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

URL="${URL:-http://localhost:8010}"
ROUNDS="${ROUNDS:-5}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16000}"
GEN_TOKENS="${GEN_TOKENS:-256}"
VRAM_GROWTH_MB="${VRAM_GROWTH_MB:-200}"

MODEL="${MODEL:-$(curl -s -m 5 "${URL}/v1/models" 2>/dev/null \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo qwen3.6-27b)}"
# best-effort container for VRAM (exact-match convention; port/name heuristic)
CONTAINER="${CONTAINER:-$(docker ps --format '{{.Names}}' 2>/dev/null | grep -m1 -E 'vllm-qwen' || true)}"

# CONCURRENCY defaults to the served max-num-seqs (the thing we're validating).
if [[ -z "${CONCURRENCY:-}" ]]; then
  CONCURRENCY="$(docker inspect "$CONTAINER" --format '{{join .Config.Cmd " "}}' 2>/dev/null \
    | grep -oE 'max-num-seqs [0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
  CONCURRENCY="${CONCURRENCY:-2}"
fi

echo "[concurrency-probe] URL=$URL model=$MODEL N=$CONCURRENCY rounds=$ROUNDS prompt=${PROMPT_TOKENS}tok gen=${GEN_TOKENS}"

URL="$URL" MODEL="$MODEL" CONTAINER="$CONTAINER" CONCURRENCY="$CONCURRENCY" \
ROUNDS="$ROUNDS" PROMPT_TOKENS="$PROMPT_TOKENS" GEN_TOKENS="$GEN_TOKENS" \
VRAM_GROWTH_MB="$VRAM_GROWTH_MB" python3 - <<'PY'
import concurrent.futures as cf
import json
import os
import subprocess
import time
import urllib.request

URL = os.environ["URL"]; MODEL = os.environ["MODEL"]; N = int(os.environ["CONCURRENCY"])
ROUNDS = int(os.environ["ROUNDS"]); PTOK = int(os.environ["PROMPT_TOKENS"])
GTOK = int(os.environ["GEN_TOKENS"]); GROWTH = int(os.environ["VRAM_GROWTH_MB"])
CONTAINER = os.environ.get("CONTAINER") or ""

BLOCK = ("This section describes the history of computing in detail. Transistors "
         "were invented in 1947 at Bell Labs. The integrated circuit came a decade "
         "later. Microprocessors emerged in the 1970s and changed the world. ")
# ~0.23 tok/char; build to ~PTOK, unique salt per stream so no prefix-cache free ride
def prompt(stream, rnd):
    reps = int(PTOK / (len(BLOCK) * 0.23)) + 1
    return f"[probe s{stream} r{rnd}] " + BLOCK * reps + "\nSummarize in two sentences."

def vram_used_mb():
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return sum(int(x) for x in out.split())
    except Exception:
        return -1

def one(stream, rnd):
    body = json.dumps({"model": MODEL, "max_tokens": GTOK, "temperature": 0.0,
        "messages":[{"role":"user","content":prompt(stream,rnd)}]}).encode()
    req = urllib.request.Request(URL+"/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        toks = (r.get("usage") or {}).get("completion_tokens", 0)
        # KV-pool stress semantics: a stream "ran" if it GENERATED tokens
        # (held KV, decoded) — content-emptiness is irrelevant here (reasoning
        # models legitimately return empty content when max_tokens truncates
        # mid-<think>). silent-empty = HTTP 200 but ZERO tokens (the real
        # failure soak-test flags).
        ok = toks > 0
        return {"ok": ok, "toks": toks, "silent": toks == 0,
                "err": None, "dt": time.time()-t0}
    except Exception as e:
        return {"ok": False, "toks": 0, "silent": False, "err": str(e)[:80], "dt": time.time()-t0}

print(f"\n{'round':>5} {'done':>7} {'silent':>7} {'errors':>7} {'vram_MB':>8} {'agg_tok/s':>10}")
vram0 = vram_used_mb()   # cold idle — empty KV pool
vram_by_round = []
bad = 0
for rnd in range(1, ROUNDS+1):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        res = list(ex.map(lambda s: one(s, rnd), range(N)))
    wall = time.time()-t0
    done = sum(1 for r in res if r["ok"])
    silent = sum(1 for r in res if r["silent"])
    errs = sum(1 for r in res if r["err"])
    v = vram_used_mb(); vram_by_round.append(v)
    agg = sum(r["toks"] for r in res)/wall if wall else 0
    print(f"{rnd:>5} {done:>4}/{N:<2} {silent:>7} {errs:>7} {v:>8} {agg:>10.1f}")
    if done < N or silent or errs:
        bad += 1

# Leak signal = growth AFTER warm-up, not the cold->warm pool fill (which is
# expected: N concurrent long contexts allocate KV blocks up to the working
# set). warm baseline = after round 2 (the pool has filled); leak = final-warm.
warm_i = 1 if ROUNDS >= 3 else 0
warm = vram_by_round[warm_i]
pool_fill = warm - vram0 if vram0 >= 0 else -1
leak = vram_by_round[-1] - warm if vram0 >= 0 else -1
print(f"\n=== verdict (N={N}) ===")
print(f"  VRAM: cold {vram0} -> warm {warm} MB (pool fill {pool_fill} MB, expected) "
      f"-> final {vram_by_round[-1]} MB  (post-warm growth {leak} MB / {GROWTH} threshold)")
clean = (bad == 0) and (0 <= leak <= GROWTH)
print(f"  concurrency {N} @ ~{PTOK} tok: {'PASS — sustained clean' if clean else 'FAIL — see rounds above'}")
if clean:
    print(f"  envelope row: max_num_seqs: {N}  "
          f"validated: {{ concurrency_soak: '{N} @ ~{PTOK//1000}K, {leak} MB post-warm growth' }}")
raise SystemExit(0 if clean else 1)
PY
