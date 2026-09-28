#!/usr/bin/env bash
# test-agent-statusline-meter — the omp and pi statusline meters stay one program,
# and report decode speed and prompt-cache share correctly.
#
# WHY THIS TEST EXISTS
# --------------------
# services/omp/extensions/tps-meter.ts and services/pi/extensions/tps-meter.ts are
# the same extension for two agents whose only difference is the package its
# type-only import names. Edited separately they drift. And the cache readout is
# easy to get subtly wrong: both agents normalise usage so that `input` is the
# UNcached part of the prompt (hit rate = cacheRead / (input + cacheRead +
# cacheWrite)), omp names the reasoning count `reasoningTokens` where pi says
# `reasoning`, and a backend that never reports cached tokens arrives as
# cacheRead 0 — which must not be shown as "0 %". The usage objects below are the
# ones omp recorded from the club gateway (vLLM/SGLang Qwen3.8).
#
# The behaviour half needs Node >= 22.6 (--experimental-strip-types) and prints
# SKIP without it; the structural half always runs.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OMP="$ROOT/services/omp/extensions/tps-meter.ts"
PI="$ROOT/services/pi/extensions/tps-meter.ts"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

# 1. one program: identical except the import line, and each import names its agent
[[ "$(head -1 "$OMP")" == 'import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";' ]] || bad "omp meter must import from @oh-my-pi/pi-coding-agent (type-only)"
[[ "$(head -1 "$PI")" == 'import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";' ]] || bad "pi meter must import from @earendil-works/pi-coding-agent (type-only)"
diff <(tail -n +2 "$OMP") <(tail -n +2 "$PI") >/dev/null || bad "the omp and pi meters differ beyond their import line — change both"

# 2. behaviour, driven through the event handlers with real usage objects
NODE_BIN="${NODE:-$(command -v node || true)}"
if [[ -z "$NODE_BIN" ]] || ! "$NODE_BIN" --experimental-strip-types -e '' 2>/dev/null; then
  echo "SKIP (behaviour): needs Node >= 22.6 for --experimental-strip-types; the structural checks ran"
else
  cp "$OMP" "$T/meter.ts"
  cat > "$T/harness.mts" <<'EOF'
import meter from "./meter.ts";
const handlers: Record<string, Function> = {};
let last = "";
meter({ on: (ev: string, fn: Function) => { handlers[ev] = fn; } } as any);
const ctx = (model: string) => ({ model: { id: model }, ui: { setStatus: (_k: string, t: string) => { last = t; } } });
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
async function turn(model: string, usage: any) {
  const c = ctx(model);
  handlers.message_start({ message: { role: "assistant" } }, c);
  await sleep(5);
  handlers.message_update({ assistantMessageEvent: { type: "text_delta" } }, c);
  await sleep(20);
  handlers.message_end({ message: { role: "assistant", usage } }, c);
  return last;
}
const out = [
  await turn("qwen3.8-27b", { input: 7667, cacheRead: 0, cacheWrite: 0, output: 131, reasoningTokens: 29 }),  // omp, cold
  await turn("qwen3.8-27b", { input: 150, cacheRead: 7680, cacheWrite: 0, output: 30, reasoningTokens: 21 }), // omp, warm
  await turn("qwen3.8-27b", { input: 400, cacheRead: 12000, cacheWrite: 0, output: 50, reasoning: 12 }),     // pi field name
  await turn("gemma", { input: 5000, cacheRead: 0, cacheWrite: 0, output: 40 }),                             // new model, no report
  await turn("gemma", { input: 6000, cacheRead: 0, cacheWrite: 0, output: 40 }),
];
console.log(JSON.stringify(out));
EOF
  res="$(cd "$T" && "$NODE_BIN" --experimental-strip-types --no-warnings harness.mts 2>&1)" || bad "the meter harness failed to run: $res"
  python3 - "$res" <<'PY' || fail=1
import json, sys
try:
    cold, warm, pi, g1, g2 = json.loads(sys.argv[1])
except Exception as e:
    print(f"✗ harness output unreadable: {e}: {sys.argv[1][:300]}", file=sys.stderr); sys.exit(1)
checks = [
    ("a cold first turn shows no cache figure (not 0 %)", "cache" not in cold),
    ("omp's reasoningTokens shows as think", "think 29" in cold and "think 21" in warm),
    ("hit rate = cacheRead / (input + cacheRead + cacheWrite): 7680 / 7830 = 98 %", "cache 98% of 7.8K" in warm),
    ("session share includes the cold turn: 7680 / 15497 = 50 %", "Σ cache 50%" in warm),
    ("pi's `reasoning` shows as think, and its split gives 97 %", "think 12" in pi and "cache 97% of 12.4K" in pi),
    ("a model switch resets the aggregate", "(n=1)" in g1),
    ("a backend that never reports cached tokens shows no cache figure", "cache" not in g1 and "cache" not in g2),
]
bad = [name for name, ok in checks if not ok]
for name in bad:
    print(f"✗ {name}", file=sys.stderr)
if bad:
    print(f"  outputs: {json.dumps([cold, warm, pi, g1, g2], ensure_ascii=False)}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
fi

[[ $fail -eq 0 ]] && echo "test-agent-statusline-meter: ok (omp == pi but the import, cache share, think field names, no false 0 %, model reset)"
exit $fail
