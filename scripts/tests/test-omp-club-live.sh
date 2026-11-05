#!/usr/bin/env bash
# test-omp-club-live.sh — the live-alias extension (services/omp/extensions/club-live.ts):
#   1. parseRenderedConfig() against render_block()'s REAL output (the empty
#      block and the user-route-after-END cases are exactly what upstream
#      review asked to see pinned);
#   2. the extension running against a stub omp host: it registers from the
#      settings file, rewrites the wire model, moves on a rendered-config
#      change through the WATCHER (the poll is 30 s, so anything caught in ~2 s
#      is the watcher — this is the watcherInstalled regression guard), keeps
#      unknown slugs thinking-free, and fails CLOSED (no gateway call, request
#      passes through) when nothing resolves;
#   3. structural pairings that must not drift.
# Leg 1 feeds the parser the real renderer's output; leg 2 is the only way to
# exercise the host-facing half (register / rewrite / move) without a rig.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"   # test-locale-utf8.sh: every python3 caller
cd "$(dirname "$0")/../.."
EXT=services/omp/extensions/club-live.ts

# --experimental-strip-types needs Node >= 22.6. Older Node: SKIP the two
# TypeScript legs loudly but still run the structural checks below (a
# hard exit 1 here once turned a Tier-0 box red for no behavioural reason).
NODE_OK=1
if ! node -e 'const [maj, min] = process.versions.node.split(".").map(Number);
              process.exit(maj > 22 || (maj === 22 && min >= 6) ? 0 : 1)' 2>/dev/null; then
  NODE_OK=0
  echo "SKIP: Node $(node --version 2>/dev/null || echo none) < 22.6 — the TypeScript legs are skipped; structural checks still run."
fi

if [[ $NODE_OK -eq 1 ]]; then
# ---- 1. parseRenderedConfig() against the REAL renderer's output ------------
# render_block() is the function that produces the generated block; driving it
# directly means a renderer format change breaks this test instead of silently
# breaking every installed alias.
python3 - <<'PY' > /tmp/club-live-render.txt
import sys
sys.path.insert(0, "scripts/lib")
import litellm_sync as ls
live = [ls.Live(8018, "qwen3.8-flash-next-iq3_s", 524288, False, True, "", 32768)]
block, n = ls.render_block(live)
assert n == 1, f"expected 1 route, got {n}"
print("model_list:\n" + block)
print("===EMPTY===")
block, n = ls.render_block([])
assert n == 0, f"expected empty block, got {n}"
print("model_list:\n" + block)
print("===SWITCHED===")
live = [ls.Live(8119, "glm-5.3-flash", 204800, False, False, "", 8192)]
block, n = ls.render_block(live)
assert n == 1, f"expected 1 route, got {n}"
print("model_list:\n" + block)
PY

node --experimental-strip-types --input-type=module -e '
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const mod = await import(pathToFileURL(process.cwd() + "/services/omp/extensions/club-live.ts").href);
const cases = readFileSync("/tmp/club-live-render.txt", "utf8").split("===EMPTY===");
const switched = cases[1].split("===SWITCHED===");
const a = mod.parseRenderedConfig(cases[0]);
if (!a || a.slug !== "qwen3.8-flash-next-iq3_s" || a.ctxWindow !== 524288 || a.maxOut !== 32768)
  throw new Error("real block parsed wrong: " + JSON.stringify(a));
const e = mod.parseRenderedConfig(switched[0]);
if (e !== null) throw new Error("empty block must resolve to nothing, got " + JSON.stringify(e));
const b = mod.parseRenderedConfig(switched[1]);
if (!b || b.slug !== "glm-5.3-flash" || b.ctxWindow !== 204800 || b.maxOut !== 8192)
  throw new Error("switched block parsed wrong: " + JSON.stringify(b));
// A user route appended after END (config.local.yaml, possibly cloud) must
// never be picked, and a route with no model_info must not inherit the next
// route window.
const c = mod.parseRenderedConfig(switched[1] + "  - model_name: anthropic/claude-opus-4-5\n");
if (!c || c.slug !== "glm-5.3-flash") throw new Error("post-END route leaked: " + JSON.stringify(c));
const d = mod.parseRenderedConfig("  # === BEGIN GENERATED LOCAL BLOCK (scripts/lib/litellm_sync.py render_block) ===\n" +
  "  - model_name: no-info-slug\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n" +
  "  - model_name: other-slug\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n    model_info:\n      max_input_tokens: 131072\n" +
  "  # === END GENERATED LOCAL BLOCK ===\n");
if (!d || d.slug !== "no-info-slug" || d.ctxWindow !== 131072)
  throw new Error("no-model_info route inherited the next route window: " + JSON.stringify(d));
console.log("parseRenderedConfig: real render_block output OK");
'

# ---- 2. the extension against a stub omp host -------------------------------
# Settings file + rendered view in a scratch dir; a fake pi records what the
# extension registers, rewrites, and moves. The rendered path does NOT exist
# at import: the extension must resolve NOTHING (fail closed), pass the wire
# request through, and never touch fetch — the removed gateway fallback must
# stay removed.
node --experimental-strip-types --input-type=module -e '
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";

const T = mkdtempSync(join(tmpdir(), "clive-"));
process.env.CLUB_LIVE_JSON = join(T, "club-live.json");
const RENDERED = join(T, "config.runtime.yaml");
const block = (slug, ctx) =>
  "  # === BEGIN GENERATED LOCAL BLOCK (scripts/lib/litellm_sync.py render_block) ===\n" +
  `  - model_name: ${slug}\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n` +
  `    model_info:\n      max_input_tokens: ${ctx}\n      max_output_tokens: 32768\n` +
  "  # === END GENERATED LOCAL BLOCK ===\n";
writeFileSync(process.env.CLUB_LIVE_JSON, JSON.stringify({
  renderedCfg: RENDERED,
  gateway: "http://127.0.0.1:4100",
  key: "sk-test-key",
  families: [{ name: "glm53", ids: ["glm-5.3-flash"],
    ladder: { efforts: ["low", "high", "max"], defaultLevel: "max" },
    maxTokens: 32768, compat: { supportsReasoningSummary: false } }],
}));
process.env.CLUB_LIVE_CFG = RENDERED;   // settingsPath/CFG honour env; the file is missing at import

let fetchTouched = false;
globalThis.fetch = () => { fetchTouched = true; throw new Error("extension must not call the gateway"); };

const registered = [];
const moved = [];
const events = {};
const pi = {
  on: (ev, h) => { events[ev] = h; },
  registerProvider: (id, cfg) => { registered.push(cfg); },
  unregisterProvider: () => {},
  registerCommand: () => {},
  setModel: async (target) => { moved.push(target); return true; },
};
const mod = await import(pathToFileURL(process.cwd() + "/services/omp/extensions/club-live.ts").href);
mod.default(pi);

// Unresolved at import (rendered file absent): names the alias honestly and
// the wire request passes through untouched.
let m = registered.at(-1).models[0];
if (!m.name.includes("not resolved")) throw new Error("unresolved state mislabelled: " + m.name);
const pass = events.before_provider_request({ payload: { model: "live", input: "x" } });
if (pass !== undefined) throw new Error("unresolved alias rewrote the request: " + JSON.stringify(pass));
writeFileSync(RENDERED, block("glm-5.3-flash", 204800));

// session_start: discovery settles, the 2 s tick resolves the GLM route.
const ctx = {
  models: { current: () => ({ provider: "club-live", id: "live" }), resolve: (s) => s },
  ui: { notify: () => {} },
  setInterval: () => {},
};
await events.session_start({}, ctx);
await new Promise((r) => setTimeout(r, 2300));
m = registered.at(-1).models[0];
if (!m.name.includes("glm-5.3-flash")) throw new Error("2 s tick did not resolve: " + m.name);
if (m.thinking?.efforts?.join() !== "low,high,max") throw new Error("family ladder not registered: " + JSON.stringify(m.thinking));
if (registered.length < 2) throw new Error("expected a re-registration on resolve");
if (registered.at(-1).compat.supportsReasoningSummary !== false) throw new Error("family compat not merged");
if (moved.at(-1) !== "club-live/live") throw new Error("resolved session was not moved via setModel: " + JSON.stringify(moved));

// The WATCHER leg: the poll is 30 s, so a change caught in ~2 s is the
// watcher. A prior regression left watcherInstalled permanently false.
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288));
await new Promise((r) => setTimeout(r, 1800));
m = registered.at(-1).models[0];
if (!m.name.includes("qwen3.8-flash-next-iq3_s")) throw new Error("watcher did not fire (watcherInstalled regression?): " + m.name);
if (moved.length < 2) throw new Error("watcher switch did not move the session: " + JSON.stringify(moved));
// A slug no family covers: thinking off, exactly like a club route discovery
// reports no override for.
if (m.reasoning !== false || m.thinking !== undefined) throw new Error("unknown family must register without thinking: " + JSON.stringify(m));
if (m.contextWindow !== 524288) throw new Error("window not taken from the rendered route: " + m.contextWindow);

// Rewrite leg: resolved alias maps the wire model to the live slug.
const out = events.before_provider_request({ payload: { model: "live", input: "x" } });
if (out?.model !== "qwen3.8-flash-next-iq3_s") throw new Error("wire rewrite wrong: " + JSON.stringify(out));
// Someone else model passes through untouched.
const other = events.before_provider_request({ payload: { model: "gpt-5", input: "x" } });
if (other !== undefined) throw new Error("foreign request modified");
if (fetchTouched) throw new Error("extension called the gateway — the fallback must stay removed");
rmSync(T, { recursive: true, force: true });
console.log("stub host: register / resolve / watcher / rewrite / fail-closed OK");
// fs.watch keeps the event loop alive by design — end the leg explicitly.
process.exit(0);
'
fi

# ---- 3. structural pairings --------------------------------------------------
# The docs must teach the alias the extension registers (the shipped
# services/omp/omp-club.yml overlay stays pinned club/<id> — valid without
# the extension).
grep -qE 'default: club-live/live' docs/CODING_AGENTS.md \
  || { echo "FAIL: docs/CODING_AGENTS.md does not teach the club-live/live alias"; exit 1; }
# The settings file is omp-setup.sh's job — the extension must read exactly
# the name the script writes (one name, two sides).
grep -q 'club-live.json' "$EXT" \
  || { echo "FAIL: extension does not read club-live.json"; exit 1; }
grep -q 'club-live.json' scripts/omp-setup.sh \
  || { echo "FAIL: omp-setup.sh does not write club-live.json"; exit 1; }
# FAIL CLOSED is the contract: the extension makes NO network call — the
# rendered view is the only source (the gateway also lists the user cloud
# routes). Comments may mention /v1/models to explain why it is NOT used.
if grep -q '\bfetch(' "$EXT"; then
  echo "FAIL: extension reaches for the network — the alias must resolve from the rendered view only"
  exit 1
fi

echo "test-omp-club-live.sh: parser-vs-real-renderer, stub-host behaviour, and pairings pass"
