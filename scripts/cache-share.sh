#!/usr/bin/env bash
# cache-share.sh — how much of each prompt the serving engine took from its
# prefix cache instead of prefilling it: the one number that says whether a
# coding-agent session is cache-friendly (docs/CODING_AGENTS.md).
#
#   bash scripts/cache-share.sh              # every serving engine, since it booted
#   bash scripts/cache-share.sh --watch 30   # …then one line per 30 s interval (Ctrl-C stops)
#   bash scripts/cache-share.sh --port 8113  # one engine (repeatable)
#
# An agent session past its first turns should sit at 90 %+. A sudden drop
# means something rewrote the FRONT of the prompt — compaction, a changed system
# prompt, reordered tool schemas — and the engine re-read the conversation.
#
# Reads the engine's own Prometheus counters; nothing is sent to the model:
#   vLLM    vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total, split by
#           vllm:prompt_tokens_by_source_total (GPU cache vs KV offload)
#   SGLang  sglang:cached_tokens_total{cache_source} / sglang:prompt_tokens_total
#           (device = GPU; host / storage = the KV offload tier)
# llama.cpp and tabbyAPI expose no cached-token counter; they are listed, not measured.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/engine-kind.sh
. "$ROOT_DIR/scripts/lib/engine-kind.sh"

WATCH=0
COUNT=0          # --count N: stop after N watch intervals (tests; 0 = until Ctrl-C)
PORTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch) WATCH="${2:?--watch needs seconds}"; shift 2 ;;
    --count) COUNT="${2:?--count needs a number}"; shift 2 ;;
    --port)  PORTS+=("${2:?--port needs a port}"); shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "cache-share: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$WATCH" =~ ^[0-9]+$ && "$COUNT" =~ ^[0-9]+$ ]] || { echo "cache-share: --watch/--count take whole numbers" >&2; exit 2; }

# 1. What is serving: /v1/models on the given ports, or on every registry port.
live="$(python3 - "$ROOT_DIR" "${PORTS[@]}" <<'PY'
import concurrent.futures as cf, json, sys, urllib.request
root, ports = sys.argv[1], [int(p) for p in sys.argv[2:]]
if not ports:
    sys.path.insert(0, f"{root}/scripts/lib")
    from litellm_sync import registry_ports, registry_variants   # the gateway sync's port list
    ports = registry_ports(registry_variants(root))
def probe(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            m = (json.load(r).get("data") or [{}])[0]
        return f"{port}\t{m.get('owned_by') or ''}\t{m.get('id') or '?'}"
    except Exception:
        return None
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    print("\n".join(x for x in ex.map(probe, ports) if x))
PY
)"
if [[ -z "$live" ]]; then
  echo "cache-share: nothing is serving${PORTS:+ on :${PORTS[*]}} (checked /v1/models)." >&2
  exit 1
fi

# 2. Engine family per port — the canonical resolver (#1282), never re-derived here.
targets=""
while IFS=$'\t' read -r port owned id; do
  targets+="${port}"$'\t'"$(engine_kind_from_owned_by "$owned")"$'\t'"${id}"$'\n'
done <<<"$live"

# 3. Read the counters; with --watch, keep reading and print per-interval deltas.
python3 - "$WATCH" "$COUNT" "$targets" <<'PY'
import re, sys, time, urllib.request
WATCH, COUNT = int(sys.argv[1]), int(sys.argv[2])
targets = [l.split("\t") for l in sys.argv[3].strip().splitlines()]
LINE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([0-9.eE+-]+)$')

def scrape(port):
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    except Exception:
        return None
    rows = []
    for ln in text.splitlines():
        m = LINE.match(ln.strip())
        if m:
            labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2) or ""))
            rows.append((m.group(1), labels, float(m.group(3))))
    return rows

def total(rows, name, **want):
    return sum(v for n, l, v in rows if n == name and all(l.get(k) == x for k, x in want.items()))

def counters(kind, rows):
    """-> (prompt tokens, from GPU cache, from offload tier) or None when not exposed."""
    if kind == "vllm":
        if not any(n == "vllm:prompt_tokens_cached_total" for n, _, _ in rows):
            return None
        prompt = total(rows, "vllm:prompt_tokens_total")
        gpu = total(rows, "vllm:prompt_tokens_by_source_total", source="local_cache_hit")
        off = total(rows, "vllm:prompt_tokens_by_source_total", source="external_kv_transfer")
        if gpu + off == 0:   # no per-source split: attribute the cached total to the GPU
            gpu = total(rows, "vllm:prompt_tokens_cached_total")
        return prompt, gpu, off
    if kind == "sglang":
        if not any(n == "sglang:cached_tokens_total" for n, _, _ in rows):
            return None
        prompt = total(rows, "sglang:prompt_tokens_total")
        gpu = total(rows, "sglang:cached_tokens_total", cache_source="device")
        off = total(rows, "sglang:cached_tokens_total") - gpu
        return prompt, gpu, off
    return None

def fmt(prompt, gpu, off):
    if prompt <= 0:
        return "no prompt tokens yet"
    cached = gpu + off
    split = f"GPU {100*gpu/prompt:.1f}%" + (f", offload {100*off/prompt:.1f}%" if off else "")
    return f"{prompt:>11,.0f} prompt tok · {100*cached/prompt:5.1f}% from cache ({split}) · {prompt-cached:,.0f} prefilled"

last = {}
for port, kind, mid in targets:
    rows = scrape(port)
    c = counters(kind, rows) if rows is not None else None
    print(f":{port}  {kind}  {mid}")
    if rows is None:
        print("  /metrics not reachable — this engine was started without metrics")
    elif c is None:
        print(f"  {kind}: no cached-token counter to read — not measured")
    else:
        print(f"  since boot  {fmt(*c)}")
        last[port] = c
if not last or not WATCH:
    sys.exit(0)
print(f"\nevery {WATCH}s — tokens since the previous line:")
n = 0
try:
    while COUNT == 0 or n < COUNT:
        time.sleep(WATCH); n += 1
        for port, kind, mid in targets:
            if port not in last:
                continue
            rows = scrape(port)
            c = counters(kind, rows) if rows else None
            if c is None:
                print(f"  {time.strftime('%H:%M:%S')} :{port}  metrics unavailable"); continue
            d = tuple(a - b for a, b in zip(c, last[port])); last[port] = c
            body = fmt(*d) if d[0] > 0 else "idle"
            print(f"  {time.strftime('%H:%M:%S')} :{port}  {body}" if len(last) > 1 else f"  {time.strftime('%H:%M:%S')}  {body}", flush=True)
except KeyboardInterrupt:
    pass
PY
