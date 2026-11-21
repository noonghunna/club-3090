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
#      passes through) when nothing resolves, applies a settings-file gateway
#      change, defers a mid-turn move to agent_end, and keeps the last
#      registration when nothing is serving, reports rig-down honestly in
#      /club-live, and warns on a corrupt settings file;
#   2b. the same stub host driven by what the INSTALLER writes: the real
#      omp-setup.sh --auto-switching settings file (compute_families over the real
#      registry, unmodified) plus blocks from the real render_block, for the ids whose
#      generated ladder and base rungs differ — the menu narrowing, a stale fold name
#      landing on the route default, and the rules that need no ladder at all;
#   3. structural pairings that must not drift.
# Legs 1 and 2b feed the parser and the host the real generator's output. Leg 2 is the
# only way to reach the fallbacks the real catalog does not exercise (a family with no
# `rungs`, an off-menu default), so its families are shaped by hand.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"   # test-locale-utf8.sh: every python3 caller
cd "$(dirname "$0")/../.."
EXT=services/omp/extensions/club-live.ts

# --experimental-strip-types needs Node >= 22.6. Older Node: SKIP the three
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
RC_DIR=$(mktemp -d)
python3 - <<'PY' > "$RC_DIR/render.txt"
import sys
sys.path.insert(0, "scripts/lib")
import litellm_sync as ls
# Effort facts keyed by PORT, the way the real render supplies them (port_facts()), so
# the block carries exactly the keys litellm_sync.model_info emits and a renderer rename
# breaks this test instead of silently blinding the alias. The 8018 entry is the REAL
# Strata pair fact for qwen3.8-flash-next-iq3_s (the compose leaves REASONING_EFFORT
# unset, so the pair's default_when_unset xhigh renders; it folds high/max onto xhigh and
# never names minimal). The GLM entry is the two OMISSION cases on purpose: a `max`
# default (outside litellm's Literal) and an unprobed supports_none. The parser must read
# both as "the route did not say", not as an empty ladder or a false off-switch claim.
FACTS = {8018: {"effort": {"levels": ["low", "medium", "xhigh"],
                           "maps": {"high": "xhigh", "max": "xhigh"},
                           "default_when_unset": "xhigh", "supports_none": True}},
         8119: {"effort": {"levels": ["low", "high", "max"], "compose_default": "max"}}}
live = [ls.Live(8018, "qwen3.8-flash-next-iq3_s", 524288, False, True, "", 32768)]
block, n = ls.render_block(live, FACTS, ".")
assert n == 1, f"expected 1 route, got {n}"
print("model_list:\n" + block)
print("===EMPTY===")
block, n = ls.render_block([])
assert n == 0, f"expected empty block, got {n}"
print("model_list:\n" + block)
print("===SWITCHED===")
live = [ls.Live(8119, "glm-5.3-flash", 204800, False, False, "", 8192)]
block, n = ls.render_block(live, FACTS, ".")
assert n == 1, f"expected 1 route, got {n}"
print("model_list:\n" + block)
PY

RENDER_CASES="$RC_DIR/render.txt" node --experimental-strip-types --input-type=module -e '
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const mod = await import(pathToFileURL(process.cwd() + "/services/omp/extensions/club-live.ts").href);
const cases = readFileSync(process.env.RENDER_CASES, "utf8").split("===EMPTY===");
const switched = cases[1].split("===SWITCHED===");
const a = mod.parseRenderedConfig(cases[0]);
if (!a || a.slug !== "qwen3.8-flash-next-iq3_s" || a.ctxWindow !== 524288 || a.maxOut !== 32768)
  throw new Error("real block parsed wrong: " + JSON.stringify(a));
// The ladder the ROUTE advertises: the model rungs minus refusals, then the fold
// sources — the names the server accepts either way, so the Strata route above names five
// while its model profile has three. The parser reads this verbatim (leg 2b narrows it to
// the base rungs of the family); a renderer rename the parser misses would silently fall back
// to the family ladder and re-open #1690. The xhigh default renders because it is a
// nameable rung — only `max` and refused names are omitted.
if (a.efforts?.join() !== "low,medium,xhigh,high,max")
  throw new Error("route ladder parsed wrong: " + JSON.stringify(a.efforts));
if (a.defaultEffort !== "xhigh") throw new Error("route default parsed wrong: " + a.defaultEffort);
if (a.supportsNone !== true) throw new Error("supports_none parsed wrong: " + JSON.stringify(a.supportsNone));
const e = mod.parseRenderedConfig(switched[0]);
if (e !== null) throw new Error("empty block must resolve to nothing, got " + JSON.stringify(e));
const b = mod.parseRenderedConfig(switched[1]);
if (!b || b.slug !== "glm-5.3-flash" || b.ctxWindow !== 204800 || b.maxOut !== 8192)
  throw new Error("switched block parsed wrong: " + JSON.stringify(b));
// Both omission cases must read as ABSENT, not as an empty ladder or a false off-switch:
// `max` is outside the LiteLLM Literal type so the renderer drops the default, and an
// unprobed engine renders no supports_none at all. A parser that guessed here would clamp
// `none` to a rung the route never offered, or send `none` to a route that 400s it.
if (b.efforts?.join() !== "low,high,max") throw new Error("glm ladder parsed wrong: " + JSON.stringify(b.efforts));
if (b.defaultEffort !== undefined) throw new Error("a max default must be omitted, got: " + b.defaultEffort);
if (b.supportsNone !== undefined) throw new Error("an unprobed off-switch must be absent, got: " + b.supportsNone);
// A user route appended after END (config.local.yaml, possibly cloud) must
// never be picked, and a route with no model_info must not inherit the next
// route window.
const c = mod.parseRenderedConfig(switched[1] + "  - model_name: anthropic/claude-opus-4-5\n");
if (!c || c.slug !== "glm-5.3-flash") throw new Error("post-END route leaked: " + JSON.stringify(c));
const d = mod.parseRenderedConfig("  # === BEGIN GENERATED LOCAL BLOCK (scripts/lib/litellm_sync.py render_block) ===\n" +
  "  - model_name: no-info-slug\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n" +
  "  - model_name: other-slug\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n    model_info:\n      max_input_tokens: 262144\n" +
  "  # === END GENERATED LOCAL BLOCK ===\n");
if (!d || d.slug !== "no-info-slug" || d.ctxWindow !== 131072)
  throw new Error("no-model_info route inherited the next route window: " + JSON.stringify(d));
// The neighbouring model_info must not leak into this route — same rule as the window
// above, now for the effort fields.
if (d.efforts !== undefined || d.defaultEffort !== undefined || d.supportsNone !== undefined)
  throw new Error("no-model_info route inherited the next route effort fields: " + JSON.stringify(d));
console.log("parseRenderedConfig: real render_block output OK");
'
rm -rf "$RC_DIR"

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
const block = (slug, ctx, effort = "") =>
  "  # === BEGIN GENERATED LOCAL BLOCK (scripts/lib/litellm_sync.py render_block) ===\n" +
  `  - model_name: ${slug}\n    litellm_params:\n      api_base: http://127.0.0.1:9/v1\n` +
  `    model_info:\n      max_input_tokens: ${ctx}\n      max_output_tokens: 32768\n` + effort +
  "  # === END GENERATED LOCAL BLOCK ===\n";
// What a route advertises once the render carries the #1951 effort fields: the rungs the
// route IMPLEMENTS plus the names its fold map ALIASES (`max` -> xhigh on the Qwen
// template, `medium` -> high on Maya). These two are SYNTHETIC pairings — the flash-next
// one names `minimal` and defaults to `medium`, which no real iq3_s route does — chosen so
// the route default differs from the family default: an assertion that passes can then only
// mean the ROUTE won, with the family supplying the client facts and the base rungs. The
// real generated pairing is leg 2b below.
const FLASH_EFFORT =
  "      supports_reasoning: true\n" +
  "      reasoning_effort_levels: [low, medium, xhigh, high, max, minimal]\n" +
  "      default_reasoning_effort: medium\n" +
  "      supports_none_reasoning_effort: false\n";
// A route that DOES take `none` and declares NO default (a max default is outside the
// LiteLLM Literal, so the renderer omits it): the off-switch passes and an out-of-ladder
// value falls back to the family default.
const GLM_EFFORT =
  "      supports_reasoning: true\n" +
  "      reasoning_effort_levels: [low, high, max, medium, xhigh]\n" +
  "      supports_none_reasoning_effort: true\n";
const EMPTY_BLOCK =
  "  # === BEGIN GENERATED LOCAL BLOCK (scripts/lib/litellm_sync.py render_block) ===\n" +
  "  # (no local routes right now — nothing serving, or no key matched)\n" +
  "  # === END GENERATED LOCAL BLOCK ===\n";
const SETTINGS = {
  renderedCfg: RENDERED,
  gateway: "http://127.0.0.1:4100",
  key: "sk-test-key",
  // The first entry is deliberately malformed (no ladder/maxTokens/compat)
  // and claims the SAME slug as the good glm53 family below: the extension
  // must drop it at load, so the ladder/compat assertions after session_start
  // can only pass if the good family won.
  families: [{ name: "broken", ids: ["glm-5.3-flash"] },
    { name: "glm53", ids: ["glm-5.3-flash"],
    // No `rungs` on purpose: this is the pre-#2109 install shape, so the GLM legs below
    // also pin the fallback (intersect against `efforts` rather than drop the family).
    ladder: { efforts: ["low", "high", "max"], defaultLevel: "max" },
    maxTokens: 32768, compat: { supportsReasoningSummary: false } },
    // The shape omp-setup.sh writes for qwen3.8-flash-next-iq3_s (checked against
    // compute_families output for this id; leg 2b drives the real installer file for three
    // other ids): `efforts` carries the fold sources (high/max), `rungs` the base rungs of
    // the model only, and the family default is Strata xhigh — different from the route
    // default (`medium`) below, which is what tells the two fallbacks apart. A thinking
    // budget the family DOES declare (glm53 declares none), so the budget rule pins in both
    // directions: kept here, dropped on GLM.
    { name: "flashnext", ids: ["qwen3.8-flash-next-iq3_s"],
    ladder: { efforts: ["low", "medium", "xhigh", "high", "max"],
              rungs: ["low", "medium", "xhigh"], defaultLevel: "xhigh" },
    maxTokens: 16384, compat: { supportsReasoningSummary: false,
                                extraBody: { thinking_token_budget: 8192 } } },
    // A family whose defaultLevel names no rung anyone advertises — a hand-edited stale
    // settings file, or a fold-target default the route does not name. The registered
    // default must fall to the LOWEST rung, never the top: the top is the busiest rung the
    // route has, and the round-down rule of #1951 points the other way.
    { name: "orphan", ids: ["orphan-slug"],
    ladder: { efforts: ["low", "medium", "xhigh"], rungs: ["low", "medium", "xhigh"],
              defaultLevel: "ultra" },
    maxTokens: 8192, compat: { supportsReasoningSummary: false } },
    // A family whose default is a FOLD SOURCE: compute_families expresses a fold-target
    // default by the name that reaches it, and that name sits outside `rungs`. The route
    // does take it, so it joins the menu rather than the default falling to an arbitrary
    // rung of the ladder.
    { name: "folddefault", ids: ["fold-default-slug"],
    ladder: { efforts: ["low", "medium", "xhigh", "high"], rungs: ["low", "medium", "xhigh"],
              defaultLevel: "high" },
    maxTokens: 8192, compat: { supportsReasoningSummary: false } }],
};
writeFileSync(process.env.CLUB_LIVE_JSON, JSON.stringify(SETTINGS));
process.env.CLUB_LIVE_CFG = RENDERED;   // settingsPath/CFG honour env; the file is missing at import

let fetchTouched = false;
globalThis.fetch = () => { fetchTouched = true; throw new Error("extension must not call the gateway"); };
const waitFor = async (pred, ms) => {
  const end = Date.now() + ms;
  while (Date.now() < end && !pred()) await new Promise((r) => setTimeout(r, 100));
};

const registered = [];
const moved = [];
const events = {};
const commands = {};
const notices = [];
const pi = {
  on: (ev, h) => { events[ev] = h; },
  registerProvider: (id, cfg) => { registered.push(cfg); },
  unregisterProvider: () => {},
  registerCommand: (name, spec) => { commands[name] = spec; },
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
  ui: { notify: (m, l) => { notices.push([m, l]); } },
  setInterval: () => {},
};
await events.session_start({}, ctx);
await waitFor(() => registered.at(-1).models[0].name.includes("glm-5.3-flash"), 5000);
m = registered.at(-1).models[0];
if (!m.name.includes("glm-5.3-flash")) throw new Error("2 s tick did not resolve: " + m.name);
if (m.thinking?.efforts?.join() !== "low,high,max")
  throw new Error("a route with no effort fields must fall back to the family ladder: " + JSON.stringify(m.thinking));
if (registered.length < 2) throw new Error("expected a re-registration on resolve");
if (registered.at(-1).compat.supportsReasoningSummary !== false) throw new Error("family compat not merged");
if (moved.at(-1) !== "club-live/live") throw new Error("resolved session was not moved via setModel: " + JSON.stringify(moved));

// The WATCHER leg: the poll is 30 s, so a change caught well inside it is
// the watcher. A prior regression left the watcher never installed; poll to
// a deadline instead of sleeping a fixed debounce-sized window.
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288, FLASH_EFFORT));
await waitFor(() => registered.at(-1).models[0].name.includes("qwen3.8-flash-next-iq3_s"), 6000);
m = registered.at(-1).models[0];
if (!m.name.includes("qwen3.8-flash-next-iq3_s")) throw new Error("watcher did not fire (watcherInstalled regression?): " + m.name);
if (moved.length < 2) throw new Error("watcher switch did not move the session: " + JSON.stringify(moved));
// The rung set and the default come from the ROUTE, narrowed to the BASE rungs of the family:
// rungs [low,medium,xhigh] (its `efforts` also names high/max, its default is xhigh), while
// the route advertises those plus `high`, `max` and `minimal`, which it only ALIASES.
// Registering those would offer the user a rung the clamp then rewrites, so `minimal` must
// NOT appear here — and the default being `medium` (the route one) rather than `xhigh` (the
// family one) is what proves the route won.
if (m.thinking?.efforts?.join() !== "low,medium,xhigh")
  throw new Error("route ladder not registered: " + JSON.stringify(m.thinking));
if (m.thinking?.defaultLevel !== "medium")
  throw new Error("route default not registered: " + JSON.stringify(m.thinking));
if (m.maxTokens !== 16384) throw new Error("the reply cap must stay a family (client) fact: " + m.maxTokens);
// A slug no family covers: thinking off, exactly like a club route discovery reports no
// override for. The route advertising a ladder does NOT change that — the family entry is
// what carries the reply cap and the compat pins that make advertising thinking safe.
writeFileSync(RENDERED, block("stray-slug", 524288, FLASH_EFFORT));
await waitFor(() => registered.at(-1).models[0].name.includes("stray-slug"), 6000);
m = registered.at(-1).models[0];
if (m.reasoning !== false || m.thinking !== undefined) throw new Error("unknown family must register without thinking: " + JSON.stringify(m));
if (m.contextWindow !== 524288) throw new Error("window not taken from the rendered route: " + m.contextWindow);

// A family whose default names no rung on the menu: the LOWEST rung registers, not the top.
writeFileSync(RENDERED, block("orphan-slug", 131072, "      reasoning_effort_levels: [low, medium, xhigh]\n"));
await waitFor(() => registered.at(-1).models[0].name.includes("orphan-slug"), 6000);
m = registered.at(-1).models[0];
if (m.thinking?.efforts?.join() !== "low,medium,xhigh")
  throw new Error("orphan ladder not registered: " + JSON.stringify(m.thinking));
if (m.thinking?.defaultLevel !== "low")
  throw new Error("an off-menu family default must fall to the lowest rung, got " + JSON.stringify(m.thinking));

// A family default that is a fold source the route does advertise: it stays on the menu, so
// the registered default is a rung omp can actually be shown.
writeFileSync(RENDERED, block("fold-default-slug", 131072, "      reasoning_effort_levels: [low, medium, xhigh, high]\n"));
await waitFor(() => registered.at(-1).models[0].name.includes("fold-default-slug"), 6000);
m = registered.at(-1).models[0];
if (m.thinking?.efforts?.join() !== "low,medium,xhigh,high")
  throw new Error("a fold-source default must stay on the menu: " + JSON.stringify(m.thinking));
if (m.thinking?.defaultLevel !== "high")
  throw new Error("a fold-source family default did not register: " + JSON.stringify(m.thinking));

// Key rotation: a settings-json rewrite with a new key must land on the next
// tick — loadSettings() re-reads every round, and a moved key re-registers.
writeFileSync(process.env.CLUB_LIVE_JSON, JSON.stringify({ ...SETTINGS, key: "sk-rotated" }));
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288, FLASH_EFFORT)); // nudge the watcher
await waitFor(() => registered.at(-1).apiKey === "sk-rotated", 6000);
if (registered.at(-1).apiKey !== "sk-rotated")
  throw new Error("rotated key not picked up on the next tick: " + registered.at(-1).apiKey);
// A --gateway change must land like a rotate: the registration is compared
// by its full settings fingerprint (gateway + key + families), not the key
// alone — this leg rewrites the key to the SAME value, so a gateway-only
// change is the only thing that can explain a new baseUrl.
writeFileSync(process.env.CLUB_LIVE_JSON, JSON.stringify({ ...SETTINGS, key: "sk-rotated", gateway: "http://127.0.0.1:4200" }));
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288, FLASH_EFFORT)); // nudge the watcher
await waitFor(() => registered.at(-1).baseUrl === "http://127.0.0.1:4200/v1", 6000);
if (registered.at(-1).baseUrl !== "http://127.0.0.1:4200/v1")
  throw new Error("gateway change never re-registered (stale baseUrl): " + registered.at(-1).baseUrl);

// Rewrite leg: resolved alias maps the wire model to the live slug.
const out = events.before_provider_request({ payload: { model: "live", input: "x" } });
if (out?.model !== "qwen3.8-flash-next-iq3_s") throw new Error("wire rewrite wrong: " + JSON.stringify(out));
// Someone else model passes through untouched.
const other = events.before_provider_request({ payload: { model: "gpt-5", input: "x" } });
if (other !== undefined) throw new Error("foreign request modified");

// ---- the clamp: every row of the #1951 decision table, against the LIVE ROUTE set ----
// The live route advertises [low, medium, xhigh, high, max, minimal] with default
// `medium` and supports_none FALSE; the family is rungs [low,medium,xhigh] / default xhigh,
// with a budget. The effective ladder is the intersection, so `high`, `max` and `minimal`
// are names the route only aliases. Every expectation below is therefore reachable only
// through the route, and the `medium` vs `xhigh` split tells the two fallbacks apart.
const wire = (p) => events.before_provider_request({ payload: { model: "live", ...p } });
// In ladder: a valid request for this route, model rewritten and nothing else.
let r = wire({ reasoning_effort: "xhigh" });
if (r.reasoning_effort !== "xhigh" || r.model !== "qwen3.8-flash-next-iq3_s")
  throw new Error("in-ladder effort rewritten: " + JSON.stringify(r));
// A name the route ALIASES instead of implementing: the stale `max` a GLM registration
// leaves on a Qwen route, which folds max -> xhigh. Riding that fold is the switch-window
// bug this phase exists to close, so it lands on the ROUTE default (medium) — which is
// also what tells this row apart from the family default (xhigh).
r = wire({ reasoning_effort: "max" });
if (r.reasoning_effort !== "medium")
  throw new Error("an aliased name must land on the route default, got " + r.reasoning_effort);
// A name neither the route nor the profile has ever heard of: same destination.
r = wire({ reasoning_effort: "ultra" });
if (r.reasoning_effort !== "medium")
  throw new Error("an unknown rung must land on the route default, got " + r.reasoning_effort);
// null: "no thinking config" — the key goes, so the engine default applies.
r = wire({ reasoning_effort: null });
if ("reasoning_effort" in r) throw new Error("a null effort must be dropped, got " + JSON.stringify(r));
// absent: never inject. A raw client gets the engine default (principle 1).
r = wire({ extra: 1 });
if ("reasoning_effort" in r) throw new Error("the clamp injected an effort: " + JSON.stringify(r));
// `none` on a route that cannot turn thinking off that way: the LOWEST admitted rung,
// never the default — the default here is medium, so a wrong implementation shows up.
r = wire({ reasoning_effort: "none" });
if (r.reasoning_effort !== "low")
  throw new Error("none must fall to the lowest admitted rung, got " + r.reasoning_effort);
// Both placements: omp writes chat_template_kwargs in qwen compat mode, and the caller
// object must come back untouched.
const kwargs = { reasoning_effort: "max", other: 2 };
r = wire({ reasoning_effort: "max", chat_template_kwargs: kwargs });
if (r.chat_template_kwargs === kwargs) throw new Error("the clamp mutated the caller kwargs object");
if (r.chat_template_kwargs.reasoning_effort !== "medium" || kwargs.reasoning_effort !== "max")
  throw new Error("chat_template_kwargs effort not normalized: " + JSON.stringify(r.chat_template_kwargs));
// A budget the live family DOES declare rides through untouched.
r = wire({ thinking_token_budget: 8192 });
if (r.thinking_token_budget !== 8192) throw new Error("a declared budget must not be dropped: " + JSON.stringify(r));
// An effort-only re-render (same slug, same window) must re-register: a profile fact
// edited or a pin bump changes the rung set, and a stale registration would keep
// advertising rungs the route no longer accepts. Driven by the WATCHER, not /club-live:
// the command forces a tick, which would hide a change-detection bug.
const beforeRegs = registered.length;
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288,
  "      reasoning_effort_levels: [low, medium]\n      default_reasoning_effort: medium\n"));
await waitFor(() => registered.length > beforeRegs, 6000);
if (registered.length === beforeRegs)
  throw new Error("an effort-only change did not re-register (change detection ignores the ladder)");
// A route that now refuses xhigh must narrow the menu, not keep advertising it.
if (registered.at(-1).models[0].thinking?.efforts?.join() !== "low,medium")
  throw new Error("re-registered ladder wrong: " + JSON.stringify(registered.at(-1).models[0].thinking));
// A route that takes `none` and names no default: `none` passes, and an out-of-ladder
// value falls back to the FAMILY default because the route declared none. GLM declares no
// budget, so a budget carried over from the Qwen route is dropped here.
writeFileSync(RENDERED, block("glm-5.3-flash", 204800, GLM_EFFORT));
await waitFor(() => registered.at(-1).models[0].name.includes("glm-5.3-flash"), 6000);
const glmM = registered.at(-1).models[0];
// The route names `medium` and `xhigh` too (Maya runs them as high/max); the registered
// menu is the model ladder, and with no rendered default the family one wins.
if (glmM.thinking?.efforts?.join() !== "low,high,max")
  throw new Error("GLM route ladder not registered: " + JSON.stringify(glmM.thinking));
if (glmM.thinking?.defaultLevel !== "max")
  throw new Error("with no rendered default the family one must register: " + JSON.stringify(glmM.thinking));
r = wire({ reasoning_effort: "none" });
if (r.reasoning_effort !== "none") throw new Error("a route that supports none must keep it: " + r.reasoning_effort);
r = wire({ reasoning_effort: "minimal" });
if (r.reasoning_effort !== "max")
  throw new Error("out-of-ladder with no route default must use the family default: " + r.reasoning_effort);
// `medium` is a name Maya runs AS high, not a rung of this model. It goes to the default
// rather than being translated through the fold map: the render exposes one flat
// advertisement, so the clamp cannot tell a fold target from the route default, and the
// settled table says out-of-ladder means the default.
r = wire({ reasoning_effort: "medium" });
if (r.reasoning_effort !== "max")
  throw new Error("an aliased name on a route with no rendered default: " + r.reasoning_effort);
r = wire({ thinking_token_budget: 4096 });
if ("thinking_token_budget" in r)
  throw new Error("a budget the live family does not declare must be dropped: " + JSON.stringify(r));
writeFileSync(RENDERED, block("qwen3.8-flash-next-iq3_s", 524288, FLASH_EFFORT));
await waitFor(() => registered.at(-1).models[0].name.includes("qwen3.8-flash-next-iq3_s"), 6000);
// Mid-turn deferral: a switch landing while a turn runs registers the new
// provider but must NOT yank the session — agent_end applies the deferred
// move (the round-1 regression: a deferred move used to be dropped for good).
await events.agent_start({});
const busyMoved = moved.length;
writeFileSync(RENDERED, block("glm-5.3-flash", 204800));
await waitFor(() => registered.at(-1).models[0].name.includes("glm-5.3-flash"), 6000);
if (moved.length !== busyMoved)
  throw new Error("mid-turn switch moved the session instead of deferring: " + JSON.stringify(moved));
await events.agent_end({});
await waitFor(() => moved.length > busyMoved, 6000);
if (moved.at(-1) !== "club-live/live")
  throw new Error("deferred move not applied at agent_end: " + JSON.stringify(moved));

// Keep-last-known-good: an empty generated block (nothing serving) resolves
// to nothing; the last registration must stand — no re-register as
// "not resolved", no session move — and the wire rewrite keeps mapping to
// the last slug so the gateway can fail honestly until a scene is back.
writeFileSync(RENDERED, EMPTY_BLOCK);
await commands["club-live"].handler("", ctx);
if (!registered.at(-1).models[0].name.includes("glm-5.3-flash"))
  throw new Error("empty block did not keep the last registration: " + registered.at(-1).models[0].name);
const kept = events.before_provider_request({ payload: { model: "live", input: "x" } });
if (kept?.model !== "glm-5.3-flash")
  throw new Error("keep-last rewrite wrong: " + JSON.stringify(kept));
// /club-live must report rig-down honestly: nothing is serving (empty
// block), so the command may not claim it is.
if (!notices.some(([m]) => m.includes("nothing serving now")))
  throw new Error("/club-live claimed serving while nothing serves: " + JSON.stringify(notices));

// Corrupt settings file: an unreadable/corrupt club-live.json must warn —
// never silently downgrade to the public default key — and the alias must
// keep running on defaults with its last registration.
writeFileSync(process.env.CLUB_LIVE_JSON, "{ not json");
await commands["club-live"].handler("", ctx);
if (!notices.some(([m, l]) => l === "warning" && m.includes("settings file")))
  throw new Error("corrupt settings file never warned: " + JSON.stringify(notices));
if (fetchTouched) throw new Error("extension called the gateway — the fallback must stay removed");
rmSync(T, { recursive: true, force: true });
console.log("stub host: register / resolve / watcher / route ladder / clamp / rewrite / gateway change / deferral / keep-last / honest status / corrupt settings / fail-closed OK");
// fs.watch keeps the event loop alive by design — end the leg explicitly.
process.exit(0);
'

# ---- 2b. the stub host driven by what the INSTALLER actually writes ----------
# Leg 2's families are hand-written, and that is exactly how the round-1 review's
# blocking finding hid: for an id whose routes all fold the same way, compute_families
# puts the FOLD SOURCES in `efforts`, so a clamp intersected with that list admits a
# stale `max` and the route folds it to xhigh — the switch window phase 4 exists to
# close. This leg runs the real `omp-setup.sh --auto-switching` (real MODEL_FAMILIES,
# real registry, real compute_families) and hands the extension that settings file
# UNMODIFIED, plus rendered blocks from the real render_block + port_facts, for the
# three ids whose shapes differ:
#   thinkingcap38-27b  efforts [low,medium,xhigh,high,max,minimal] / rungs
#                      [low,medium,xhigh] / default low — a stale `max` must land on
#                      `low`, not ride the max -> xhigh fold;
#   hy4-flash          rungs [high] alone — the other four names are Maya's fold
#                      sources, so the menu narrows to the one rung the profile has;
#   deepseek-v4-flash  no family at all (the model is uncovered by MODEL_FAMILIES): the
#                      rules that need no ladder still apply — a `null` effort and a
#                      stray budget must not reach an engine that never declared either.
GEN=$(mktemp -d)
if PI_CODING_AGENT_DIR="$GEN/agent" bash scripts/omp-setup.sh --auto-switching \
     --gateway http://127.0.0.1:4100/v1 >"$GEN/omp-setup.log" 2>&1 \
   && [ -s "$GEN/agent/extensions/club-live.json" ]; then
python3 - "$GEN/agent/extensions/club-live.json" <<'PY' > "$GEN/render.txt"
import json, sys
sys.path.insert(0, "scripts/lib")
import litellm_sync as ls
from effort_families import route_names
# The generator guarantee the extension's default rule relies on: every family's default is
# a nameable rung of its own ladder. And the join's reach, stated where the reader can
# check it: no catalog id has a default outside `rungs` today (a compose that sets
# REASONING_EFFORT to a fold source would reach that shape), so the fold-source-default
# join is pinned by the synthetic leg, not by a real id.
for f in json.load(open(sys.argv[1], encoding="utf-8"))["families"]:
    l = f["ladder"]
    assert l["defaultLevel"] in l["efforts"], \
        f"{f['name']}: default not a rung of its own ladder: {l}"
    if l["defaultLevel"] not in l["rungs"]:
        print(f"  note: {f['name']} defaults outside rungs — the join is live there",
              file=sys.stderr)
variants = ls.registry_variants(".")
facts = ls.port_facts(variants)
def row(name):
    for r in variants:
        names, _ = route_names(".", r)
        if name in names:
            return r
    raise SystemExit("no registry row answers to %r: the catalog moved and this leg must "
                     "move with it" % name)
parts = []
for name in ("thinkingcap38-27b", "hy4-flash", "deepseek-v4-flash"):
    r = row(name)
    block, n = ls.render_block(
        [ls.Live(r["port"], name, r["configured_ctx"], False, False, "", 32768)], facts, ".")
    assert n == 1, f"{name}: expected 1 route, got {n}"
    parts.append(block)
print("===TC===" + parts[0] + "===HY4===" + parts[1] + "===DS4===" + parts[2])
PY
GEN_JSON="$GEN/agent/extensions/club-live.json" GEN_RENDER="$GEN/render.txt" \
node --experimental-strip-types --input-type=module -e '
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";

const T = mkdtempSync(join(tmpdir(), "clive-gen-"));
const RENDERED = join(T, "config.runtime.yaml");
process.env.CLUB_LIVE_JSON = process.env.GEN_JSON;   // the installer file, unmodified
process.env.CLUB_LIVE_CFG = RENDERED;                // scratch rendered view
const settings = JSON.parse(readFileSync(process.env.GEN_JSON, "utf8"));
const famFor = (id) => settings.families.find((f) => f.ids.includes(id));

// The field the clamp needs. If compute_families ever stops emitting it, this fails
// here instead of the clamp quietly falling back to the alias-carrying list.
const tcFam = famFor("thinkingcap38-27b");
if (!tcFam) throw new Error("the installer wrote no family for thinkingcap38-27b");
if (tcFam.ladder.rungs?.join() !== "low,medium,xhigh")
  throw new Error("installer rungs wrong for thinkingcap38-27b: " + JSON.stringify(tcFam.ladder));
if (!tcFam.ladder.efforts.includes("max"))
  throw new Error("the generated ladder should still carry the fold source max: " + JSON.stringify(tcFam.ladder.efforts));

const parts = readFileSync(process.env.GEN_RENDER, "utf8").split("===HY4===");
const TC_BLOCK = parts[0].replace("===TC===", "");
const [HY4_BLOCK, DS4_BLOCK] = parts[1].split("===DS4===");

const registered = [];
const events = {};
const pi = {
  on: (ev, h) => { events[ev] = h; },
  registerProvider: (id, cfg) => { registered.push(cfg); },
  unregisterProvider: () => {},
  registerCommand: () => {},
  setModel: async () => true,
};
globalThis.fetch = () => { throw new Error("extension must not call the gateway"); };
const waitFor = async (pred, ms) => {
  const end = Date.now() + ms;
  while (Date.now() < end && !pred()) await new Promise((r) => setTimeout(r, 100));
};
const mod = await import(pathToFileURL(process.cwd() + "/services/omp/extensions/club-live.ts").href);
mod.default(pi);
const ctx = { models: { current: () => ({ provider: "club-live", id: "live" }), resolve: (s) => s },
  ui: { notify: () => {} }, setInterval: () => {} };
await events.session_start({}, ctx);
const wire = (p) => events.before_provider_request({ payload: { model: "live", ...p } });

// thinkingcap38-27b: the route names six names, the model implements three.
writeFileSync(RENDERED, TC_BLOCK);
await waitFor(() => registered.at(-1).models[0].name.includes("thinkingcap38-27b"), 6000);
let m = registered.at(-1).models[0];
if (!m.name.includes("thinkingcap38-27b")) throw new Error("generated settings did not resolve: " + m.name);
if (m.thinking?.efforts?.join() !== "low,medium,xhigh")
  throw new Error("the menu still carries fold aliases: " + JSON.stringify(m.thinking));
if (m.thinking?.defaultLevel !== "low") throw new Error("route default not registered: " + JSON.stringify(m.thinking));
if (m.maxTokens !== 32768) throw new Error("reply cap not from the generated family: " + m.maxTokens);
// THE finding-1 rows: a stale `max` / `high` / `minimal` left on the wire by a
// registration made against another route. Against the generated `efforts` all three
// passed the membership test and the template folded them (max/high -> xhigh,
// minimal -> low by the template, not by the clamp). They must land on the route default.
for (const stale of ["max", "high", "minimal"]) {
  const kwargs = { reasoning_effort: stale };
  const r = wire({ reasoning_effort: stale, chat_template_kwargs: kwargs });
  if (r.reasoning_effort !== "low" || r.chat_template_kwargs?.reasoning_effort !== "low")
    throw new Error("stale " + stale + " rode the fold instead of the route default: " + JSON.stringify(r));
  if (kwargs.reasoning_effort !== stale) throw new Error("the clamp mutated the caller kwargs");
}
// Rungs this route really implements, and `none` it renders support for, pass untouched.
for (const ok of ["low", "medium", "xhigh", "none"]) {
  const r = wire({ reasoning_effort: ok });
  if (r.reasoning_effort !== ok) throw new Error("a rung this route takes was rewritten: " + ok + " -> " + r.reasoning_effort);
}
// The generated family declares the thinking budget, so it rides through.
if (wire({ thinking_token_budget: 8192 }).thinking_token_budget !== 8192)
  throw new Error("a budget the generated family declares was dropped");

// hy4-flash: the model profile has ONE rung; Maya folds the other four names onto it.
writeFileSync(RENDERED, HY4_BLOCK);
await waitFor(() => registered.at(-1).models[0].name.includes("hy4-flash"), 6000);
m = registered.at(-1).models[0];
if (m.thinking?.efforts?.join() !== "high")
  throw new Error("hy4 menu still carries Maya fold sources: " + JSON.stringify(m.thinking));
if (m.thinking?.defaultLevel !== "high") throw new Error("hy4 default: " + JSON.stringify(m.thinking));
const hy = wire({ reasoning_effort: "max" });
if (hy.reasoning_effort !== "high") throw new Error("hy4 stale rung not clamped: " + hy.reasoning_effort);

// deepseek-v4-flash: no family, so no ladder — and no license for a null effort or a
// budget either.
writeFileSync(RENDERED, DS4_BLOCK);
await waitFor(() => registered.at(-1).models[0].name.includes("deepseek-v4-flash"), 6000);
m = registered.at(-1).models[0];
if (m.reasoning !== false || m.thinking !== undefined)
  throw new Error("an uncovered slug must register without thinking: " + JSON.stringify(m));
const bareKwargs = { reasoning_effort: null, other: 1 };
const bare = wire({ reasoning_effort: null, thinking_token_budget: 8192, chat_template_kwargs: bareKwargs });
if ("reasoning_effort" in bare) throw new Error("a null effort reached a no-family route: " + JSON.stringify(bare));
if (bare.chat_template_kwargs?.reasoning_effort !== undefined)
  throw new Error("a null effort in chat_template_kwargs reached a no-family route: " + JSON.stringify(bare));
if (bare.chat_template_kwargs?.other !== 1) throw new Error("kwargs not preserved: " + JSON.stringify(bare.chat_template_kwargs));
if (bareKwargs.reasoning_effort !== null) throw new Error("the clamp mutated the caller kwargs");
if ("thinking_token_budget" in bare) throw new Error("a budget reached a route with no family: " + JSON.stringify(bare));
// A STRING effort on a route that registered without thinking is left alone: there is no
// ladder to clamp it to, and dropping the caller-provided rung is not settled (finding 4).
if (wire({ reasoning_effort: "high" }).reasoning_effort !== "high")
  throw new Error("a no-family route rewrote a string effort");
rmSync(T, { recursive: true, force: true });
console.log("installer-driven leg: generated families / rungs narrowing / stale fold clamp / hy4 single rung / no-family null+budget rules OK");
// fs.watch keeps the event loop alive by design — end the leg explicitly.
process.exit(0);
'
else
  echo "test-omp-club-live: SKIP the installer-driven leg (omp-setup.sh --auto-switching did not run here)" >&2
  command cat "$GEN/omp-setup.log" >&2
fi
rm -rf "$GEN"
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

echo "test-omp-club-live.sh: parser-vs-real-renderer, stub-host behaviour on hand-written and"
echo "installer-written settings, and pairings pass"
