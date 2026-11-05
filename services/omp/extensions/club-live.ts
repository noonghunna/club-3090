// club-live — keep omp roles pointed at the club alias on the model the
// club-3090 gateway is serving RIGHT NOW, including inside a running session.
//
// WHY THIS EXISTS
// The `club` provider scripts/omp-setup.sh writes discovers the gateway's
// routes, so a switch re-lists the models — but a RUNNING session stays pinned
// to the slug it started on. When a switch retires that slug, the session dies
// with "No model selected" and every subagent spawn inherits the breakage.
// This extension registers one virtual model, `club-live/live`, that never
// goes stale:
//
//   modelRoles:
//     default: club-live/live        # the alias — always the live local model
//     plan: club-live/live:xhigh     # effort suffixes work as usual
//     # a cloud orchestrator role stays literal and is NEVER touched:
//     # advisor: anthropic/claude-opus-4-5
//
// Three mechanisms, each scoped to the club-live provider alone:
// 1. before_provider_request rewrites the wire `model` from "live" to the slug
//    the gateway is serving, so the stable id always routes correctly. No
//    other request is modified — literal cloud selectors pass through.
// 2. The live slug comes from the GENERATED LOCAL BLOCK of the litellm-sync
//    rendered config — only local answering backends appear inside those
//    markers, never the user's own cloud routes from config.local.yaml. An
//    empty block means nothing local is serving: the alias stays unresolved
//    rather than pointing at a cloud route. A file watcher plus a slow poll
//    notice switch.sh within ~2 seconds.
// 3. When the slug, its window, or the gateway key changes, the provider is
//    re-registered (fresh context window, fresh key — a gateway-key rotate
//    lands without restarting omp) and a session whose CURRENT model is a
//    club-live model is moved to the new registration with pi.setModel — the
//    only API that re-points a healthy session. Sessions on other providers (a
//    cloud orchestrator) are left exactly as they are.
//
// PATHS AND KEY — the same sources the repo's own clients use, never baked:
//   config:  $CLUB_LIVE_CFG, else $CLUB3090_ROOT/services/litellm/config.runtime.yaml,
//            else ~/club-3090/services/litellm/config.runtime.yaml
//   gateway: $CLUB_LIVE_GATEWAY, else http://127.0.0.1:4000
//   key:     $CLUB_LIVE_KEY, else $LITELLM_MASTER_KEY, else the club config dir
//            (club3090.env then secrets.env), else the legacy repo .env, else
//            the public compose default — the precedence of
//            scripts/lib/club_config.py.
// The key is re-read on every refresh, so `gateway-key.sh rotate` needs no omp
// restart. A checkout outside ~/club-3090 sets CLUB3090_ROOT (the cp step
// scripts/omp-setup.sh prints says where).
//
// Subagents need no special handling: their role patterns resolve to the
// stable `club-live/live` id, which always exists.
import { readFileSync, watch } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const PROVIDER = "club-live";
const MODEL_ID = "live";
const SELECTOR = `${PROVIDER}/${MODEL_ID}`;
const POLL_MS = 30_000;
const DEBOUNCE_MS = 1_500;
// Only used while nothing has resolved yet (gateway down, rig --down): the
// alias still lists, and the next successful tick replaces it.
const DEFAULT_CTX = 131_072;
const DEFAULT_MAX_OUT = 32_768;
// The reply cap scripts/omp-setup.sh pins for every id it knows (thinking +
// answer; a small cap cuts a file write off mid-file, and omp's own catalog
// fallback would be the whole window of the smallest slug).
const REPLY_CAP = 32_768;
// The compose default every install shares — the same last resort
// scripts/omp-setup.sh uses when no key is stored.
const PUBLIC_DEFAULT_KEY = "sk-litellm-master-key";

// The served-id families, mirroring the QWEN38_IDS / FLASH_NEXT_IDS / GLM53_IDS
// lists in scripts/omp-setup.sh (test-omp-club-live.sh fails the pair if one
// side moves). They decide the effort ladder: GLM-5.3-Flash has no `medium`;
// the Qwen3.8 family has no `max`.
const GLM53_IDS: Record<string, true> = { "glm-5.3-flash": true };
const QWEN_LADDER = { efforts: ["low", "medium", "xhigh"], defaultLevel: "medium" };
const GLM_LADDER = { efforts: ["low", "high", "max"], defaultLevel: "max" };
function ladderFor(slug: string | undefined): { efforts: string[]; defaultLevel: string } {
	return slug && GLM53_IDS[slug] ? GLM_LADDER : QWEN_LADDER;
}

function env(name: string): string | undefined {
	const v = process.env[name];
	return v && v.length > 0 ? v : undefined;
}

// The checkout: CLUB3090_ROOT names it; the documented clone location is the
// default. The rendered runtime view lives inside it.
function repoRoot(): string {
	return env("CLUB3090_ROOT") ?? join(env("HOME") ?? homedir(), "club-3090");
}

// The rendered runtime view litellm-sync writes for the gateway, regenerated
// on every switch.sh launch and teardown. CLUB_LIVE_CFG overrides the file
// directly.
const RUNTIME_CFG =
	env("CLUB_LIVE_CFG") ?? join(repoRoot(), "services", "litellm", "config.runtime.yaml");
const GATEWAY = env("CLUB_LIVE_GATEWAY") ?? "http://127.0.0.1:4000";

function configDir(): string {
	return (
		env("CLUB3090_CONFIG_DIR") ??
		join(env("XDG_CONFIG_HOME") ?? join(env("HOME") ?? homedir(), ".config"), "club-3090")
	);
}

// KEY=value lines, `#` comments and the optional `export ` prefix skipped
// (both accepted by club_config.py's parse_env_file), surrounding quotes
// stripped — the shape settings.sh writes (compose_env_quote single-quotes
// what needs it).
function readEnvFile(path: string, key: string): string | undefined {
	let text: string;
	try {
		text = readFileSync(path, "utf8");
	} catch {
		return undefined;
	}
	for (const raw of text.split("\n")) {
		const line = raw.trim();
		if (!line || line.startsWith("#")) continue;
		const eq = line.indexOf("=");
		if (eq < 0) continue;
		let lhs = line.slice(0, eq).trim();
		if (lhs.startsWith("export ")) lhs = lhs.slice(7).trim();
		if (lhs !== key) continue;
		let value = line.slice(eq + 1).trim();
		if (
			(value.startsWith('"') && value.endsWith('"')) ||
			(value.startsWith("'") && value.endsWith("'"))
		)
			value = value.slice(1, -1);
		return value.length ? value : undefined;
	}
	return undefined;
}

// Same precedence as scripts/lib/club_config.py: shell env > club3090.env >
// secrets.env > the legacy repo .env, then the public default. (club_config
// counts an env var set to "" as set; here an empty value counts as unset, the
// way the compose `${LITELLM_MASTER_KEY:-…}` default treats it.) Read at every
// use, so a rotate lands on the next refresh without restarting omp.
export function resolveGatewayKey(): string {
	return (
		env("CLUB_LIVE_KEY") ??
		env("LITELLM_MASTER_KEY") ??
		readEnvFile(join(configDir(), "club3090.env"), "LITELLM_MASTER_KEY") ??
		readEnvFile(join(configDir(), "secrets.env"), "LITELLM_MASTER_KEY") ??
		readEnvFile(join(repoRoot(), ".env"), "LITELLM_MASTER_KEY") ??
		PUBLIC_DEFAULT_KEY
	);
}

interface LiveInfo {
	slug: string;
	ctxWindow: number;
	maxOut: number;
}

// The generated block between litellm_sync's BEGIN/END markers holds ONLY the
// local serving routes; the user's own routes (config.local.yaml — possibly
// cloud) are appended after END at the same indent, and an empty block carries
// just a "nothing serving" comment. Scoping to the block is what keeps the
// alias from ever resolving to a cloud route. Exported for
// test-omp-club-live.sh, which feeds it render_block()'s real output.
const BLOCK = /^ {2}# === BEGIN GENERATED LOCAL BLOCK[\s\S]*?^ {2}# === END GENERATED LOCAL BLOCK ===$/m;

export function parseRenderedConfig(text: string): LiveInfo | null {
	const block = BLOCK.exec(text);
	if (!block) return null; // not a litellm-sync rendered view
	const starts = [...block[0].matchAll(/^ {2}- model_name:/gm)].map((m) => m.index);
	if (starts.length === 0) return null; // "(nothing serving right now)"
	// The window comes from THIS route's own model_info — a route whose engine
	// reports no context must not inherit the next route's window.
	const route = block[0].slice(starts[0], starts.length > 1 ? starts[1] : block[0].length);
	const slug = /^ {2}- model_name:\s*(\S+)/.exec(route)?.[1];
	if (!slug) return null;
	const mi = /max_input_tokens:\s*(\d+)/.exec(route);
	const mo = /max_output_tokens:\s*(\d+)/.exec(route);
	return {
		slug,
		ctxWindow: mi ? Number(mi[1]) : DEFAULT_CTX,
		maxOut: mo ? Number(mo[1]) : DEFAULT_MAX_OUT,
	};
}

// ---- minimal shapes of the omp surfaces this extension touches -------------
// Deliberately structural rather than importing @oh-my-pi/pi-coding-agent:
// the extension runs under omp's loader, and these are exactly the calls it
// makes — a host-side signature change shows up as a type error here, not as
// a runtime surprise.

interface ModelLike {
	id: string;
	provider: string;
}
interface ModelFacade {
	current(): ModelLike | undefined;
	resolve(spec: string): ModelLike | undefined;
}
interface UiLike {
	notify?(message: string, level?: string): void;
}
interface CtxLike {
	models?: ModelFacade;
	ui?: UiLike;
	setInterval?(callback: () => void, ms: number): unknown;
}
interface PiLike {
	on(event: string, handler: (event: unknown, ctx: CtxLike) => unknown): void;
	setModel(model: ModelLike | string): Promise<boolean>;
	registerProvider(name: string, config: Record<string, unknown>): void;
	unregisterProvider?(name: string): void;
	registerCommand(
		name: string,
		def: { description: string; handler: (args: string, ctx: CtxLike) => unknown },
	): void;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

// Last-resort resolution for when the rendered config file is missing (never
// synced, or deleted after the gateway started): /model_group/info carries
// each route's model_info — the endpoint omp's `discovery: litellm` reads —
// and /v1/models covers older gateways. With no rendered view there are no
// config.local.yaml routes in the gateway either, so its first route is a
// local one.
async function fromGateway(): Promise<LiveInfo | null> {
	for (const path of ["/model_group/info", "/v1/models"]) {
		try {
			const res = await fetch(`${GATEWAY}${path}`, {
				headers: { Authorization: `Bearer ${resolveGatewayKey()}` },
				signal: AbortSignal.timeout(4_000),
			});
			if (!res.ok) continue;
			const data = asRecord(await res.json())?.["data"];
			if (!Array.isArray(data)) continue;
			for (const entry of data) {
				const e = asRecord(entry);
				const slug = typeof e?.model_name === "string" ? e.model_name : typeof e?.id === "string" ? e.id : "";
				if (!slug) continue;
				const info = asRecord(e?.model_info);
				const mi = Number(info?.max_input_tokens);
				const mo = Number(info?.max_output_tokens);
				return {
					slug,
					ctxWindow: Number.isFinite(mi) && mi > 0 ? mi : DEFAULT_CTX,
					maxOut: Number.isFinite(mo) && mo > 0 ? mo : DEFAULT_MAX_OUT,
				};
			}
		} catch {
			// gateway unreachable — keep the current registration
		}
	}
	return null;
}

async function resolveLive(): Promise<LiveInfo | null> {
	let text = "";
	try {
		text = readFileSync(RUNTIME_CFG, "utf8");
	} catch {
		/* rendered config missing — ask the gateway */
	}
	const parsed = parseRenderedConfig(text);
	if (parsed) return parsed;
	if (BLOCK.test(text)) return null; // rendered view present, nothing local serving
	return fromGateway();
}

export default function (pi: PiLike): void {
	// null = nothing resolved yet (fresh boot, gateway down, or rig --down).
	// Never a stale guess: a wrong slug 400s at the gateway and looks like a
	// broken rig; an unresolved alias passes the request through and the
	// gateway 400s honestly ("model 'live' not found").
	let live: LiveInfo | null = null;
	let registeredKey = "";
	let boundCtx: CtxLike | undefined;
	let agentBusy = false;
	let watcherInstalled = false;
	let refreshInFlight = false;
	let refreshPending = false;
	{
		let text = "";
		try {
			text = readFileSync(RUNTIME_CFG, "utf8");
		} catch {
			/* the 2 s tick(true) below covers the gateway path */
		}
		live = parseRenderedConfig(text);
	}

	function register(): void {
		// Re-registering the same provider id replaces the registration; drop
		// the old one first when the host exposes the teardown hook.
		registeredKey = resolveGatewayKey();
		pi.unregisterProvider?.(PROVIDER);
		pi.registerProvider(PROVIDER, {
			baseUrl: `${GATEWAY}/v1`,
			apiKey: registeredKey,
			api: "openai-completions",
			// The same wire policies scripts/omp-setup.sh pins for the `club`
			// provider (test-omp-club-live.sh tripwires the pair): vLLM /
			// SGLang / llama.cpp read thinking off the chat template, take the
			// effort through chat_template_kwargs (#1690 — without those two
			// pins every request ran at the server's default, the slowest
			// rung), and none of them take a developer role.
			compat: {
				supportsDeveloperRole: false,
				thinkingFormat: "qwen-chat-template",
				reasoningDisableMode: "none-effort",
				supportsReasoningEffort: true,
				qwenTemplateReasoningEffort: true,
				extraBody: { thinking_token_budget: 8192 },
			},
			models: [
				{
					// Stable id: roles written as `club-live/live` survive every
					// switch; before_provider_request maps it to the live slug.
					id: MODEL_ID,
					name: live ? `club-3090 live (${live.slug})` : "club-3090 live (not resolved yet)",
					reasoning: true,
					input: ["text"],
					thinking: { mode: "effort", ...ladderFor(live?.slug) },
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: live?.ctxWindow ?? DEFAULT_CTX,
					maxTokens: REPLY_CAP,
				},
			],
		});
	}
	register();

	// The only place the alias is resolved to a concrete upstream model. While
	// nothing has resolved the request passes through as-is and the gateway
	// 400s honestly ("model 'live' not found") instead of hitting a stale slug.
	pi.on("before_provider_request", (event: unknown) => {
		const payload = asRecord(asRecord(event)?.["payload"]);
		if (!payload) return undefined;
		const model = payload["model"];
		if (model !== MODEL_ID && model !== SELECTOR) return undefined;
		if (!live) return undefined;
		return { ...payload, model: live.slug };
	});

	pi.on("agent_start", async () => {
		agentBusy = true;
	});
	pi.on("agent_end", async () => {
		agentBusy = false;
	});

	async function tick(force = false): Promise<void> {
		if (refreshInFlight) {
			refreshPending = true; // a watcher hit during a poll must not be lost
			return;
		}
		refreshInFlight = true;
		try {
			do {
				refreshPending = false;
				const next = await resolveLive();
				if (!next) continue; // keep the last-known-good; the next round retries
				const changed =
					!live ||
					next.slug !== live.slug ||
					next.ctxWindow !== live.ctxWindow ||
					next.maxOut !== live.maxOut;
				const keyMoved = resolveGatewayKey() !== registeredKey;
				if (!changed && !keyMoved && !force) continue;
				live = next;
				register(); // fresh window, ladder, and key
				if (agentBusy) continue; // don't yank mid-turn; the next round retries
				// Move sessions that are ON the alias to the new registration —
				// never a session whose model is someone else's (cloud orchestrators).
				const current = boundCtx?.models?.current?.();
				if (current && current.provider === PROVIDER) {
					const target = boundCtx?.models?.resolve?.(SELECTOR) ?? SELECTOR;
					if (await pi.setModel(target)) {
						if (changed) boundCtx?.ui?.notify?.(`club-live: switched to ${next.slug}`);
					} else if (force) {
						boundCtx?.ui?.notify?.(`club-live: ${next.slug} not resolvable yet`, "warning");
					}
				} else if (changed) {
					boundCtx?.ui?.notify?.(`club-live: live model is now ${next.slug}`);
				}
			} while (refreshPending);
		} finally {
			refreshInFlight = false;
		}
	}

	pi.on("session_start", async (_event: unknown, ctx: CtxLike) => {
		boundCtx = ctx;
		ctx.setInterval?.(() => void tick(), POLL_MS);
		if (watcherInstalled) return;
		watcherInstalled = true;
		try {
			let debounce: NodeJS.Timeout | undefined;
			watch(dirname(RUNTIME_CFG), (_event2, filename) => {
				if (filename && !RUNTIME_CFG.endsWith(filename)) return;
				clearTimeout(debounce);
				debounce = setTimeout(() => void tick(), DEBOUNCE_MS);
			});
		} catch {
			// watcher unavailable (file missing at start) — the poll still covers it
		}
		// Adopt the live slug shortly after start, once discovery has settled.
		setTimeout(() => void tick(true), 2_000);
	});

	pi.registerCommand("club-live", {
		description: "Re-check the live club-3090 model and re-apply the alias",
		handler: async (_args: string, ctx: CtxLike) => {
			boundCtx = ctx;
			agentBusy = false;
			await tick(true);
			ctx.ui?.notify?.(`club-live: serving ${live?.slug ?? "nothing resolved yet"}`, "info");
		},
	});
}
