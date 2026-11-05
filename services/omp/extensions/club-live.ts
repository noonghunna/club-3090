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
// Effort suffixes resolve against the LADDER OF THE SERVING FAMILY: the
// extension registers the serving slug's own ladder (GLM has no `medium`, the
// Qwen3.8 family has no `max`), and omp CLAMPS a role's suffix to it — a
// `:medium` role on GLM runs at `low`, `:xhigh` runs at `high` (verified
// against omp's wire: the nearest lower rung is sent; no error, and an
// effort the server has never heard of never reaches it).
//
// Three mechanisms, each scoped to the club-live provider alone:
// 1. before_provider_request rewrites the wire `model` from "live" to the slug
//    the gateway is serving, so the stable id always routes correctly. No
//    other request is modified — literal cloud selectors pass through.
// 2. The live slug comes from the GENERATED LOCAL BLOCK of the litellm-sync
//    rendered config — only local answering backends appear inside those
//    markers, never the user's own cloud routes from config.local.yaml. An
//    empty block means nothing local is serving: the alias keeps its last
//    registration (requests then fail honestly at the gateway until the rig
//    is back) rather than pointing at a cloud route. The alias NEVER asks the
//    gateway which model is live — /v1/models also lists config.local.yaml
//    routes, and a fallback there could point at the cloud exactly when the
//    rig is down. If the rendered view can't be read, nothing resolves.
//    A file watcher plus a slow poll notice switch.sh within ~2 seconds; a
//    RUNNING session moves at its next turn boundary — never mid-turn.
// 3. When the slug, its window, or the settings change, the provider is
//    re-registered and a session whose CURRENT model is a club-live model is
//    moved to the new registration with pi.setModel — the only API that
//    re-points a healthy session. Sessions on other providers (a cloud
//    orchestrator) are left exactly as they are.
//    A backstop covers dispatch that skips the rewrite hook (omp's compaction
//    and skill-compression paths send the wire model as-is): the rendered
//    block carries a `live` alias route mirroring the first serving route, so
//    the gateway resolves the alias name itself. Rig down means no alias
//    route — a `live` request then 400s honestly at the gateway.
//
// SETTINGS — one JSON file next to this extension, written by
// scripts/omp-setup.sh (its --auto-switching step; re-run it after a
// gateway-key rotate or a --gateway change, exactly as for the `club`
// provider):
//   renderedCfg  host path of the file the litellm container mounts (from
//                litellm_sync.mounted_runtime_path() — a worktree setup's
//                file can live outside the checkout)
//   gateway      the gateway URL omp-setup.sh was given
//   key          the gateway key, resolved through the club_config loader —
//                the extension never re-implements that precedence
//   families     the per-family wire settings (ids, effort ladder, reply cap,
//                compat) generated from the SAME lists omp-setup.sh writes
//                into models.yml, so the alias and the `club` provider cannot
//                drift. A slug in none of the families registers WITHOUT
//                thinking — the same as a `club/<id>` route discovery reports
//                no override for.
// $CLUB_LIVE_JSON overrides the settings file's path, $CLUB_LIVE_CFG the
// rendered config's (both are how test-omp-club-live.sh points the extension
// at scratch files). Without a settings file the defaults are the documented
// clone location, http://127.0.0.1:4000, and the public compose default key.
import { readFileSync, watch, type FSWatcher } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PROVIDER = "club-live";
const MODEL_ID = "live";
const SELECTOR = `${PROVIDER}/${MODEL_ID}`;
const POLL_MS = 30_000;
const DEBOUNCE_MS = 1_500;
// Only used while nothing has resolved yet (gateway down, rig --down): the
// alias still lists, and the next successful tick replaces it.
const DEFAULT_CTX = 131_072;
const DEFAULT_MAX_OUT = 32_768;
// The compose default every install shares — the same last resort
// scripts/omp-setup.sh uses when no key is stored.
const PUBLIC_DEFAULT_KEY = "sk-litellm-master-key";

// Provider-level compat: every club engine reads thinking off the chat
// template, takes the effort through chat_template_kwargs (#1690 — without
// those pins every request ran at the server's default, the slowest rung),
// and none of them takes a developer role. Per-family additions come from
// the settings file.
const PROVIDER_COMPAT: Record<string, unknown> = {
	supportsDeveloperRole: false,
	thinkingFormat: "qwen-chat-template",
	reasoningDisableMode: "none-effort",
};

function env(name: string): string | undefined {
	const v = process.env[name];
	return v && v.length > 0 ? v : undefined;
}

interface Family {
	ids: string[];
	ladder: { efforts: string[]; defaultLevel: string };
	maxTokens: number;
	compat: Record<string, unknown>;
}
interface Settings {
	renderedCfg: string;
	gateway: string;
	key: string;
	families: Family[];
	// Set when the settings file exists but is unreadable/corrupt; an absent
	// file is not a problem — that is the documented defaults path.
	problem?: string;
}

// The settings file sits beside the installed extension; CLUB_LIVE_JSON
// relocates it (tests). Re-read on every tick, so a re-run of omp-setup.sh
// (rotate, --gateway change, new family) lands without restarting omp.
function settingsPath(): string {
	return env("CLUB_LIVE_JSON") ?? join(dirname(fileURLToPath(import.meta.url)), "club-live.json");
}
function loadSettings(): Settings {
	let raw: Record<string, unknown> = {};
	let problem: string | undefined;
	try {
		raw = JSON.parse(readFileSync(settingsPath(), "utf8")) as Record<string, unknown>;
	} catch (err) {
		// An absent file means the documented defaults. A file that IS there
		// but is unreadable or corrupt must be said out loud, not silently
		// downgraded to the public default key — the alias would 401 at the
		// gateway with nothing to point at. tick() surfaces `problem`.
		if ((err as NodeJS.ErrnoException).code !== "ENOENT")
			problem = `settings file is unreadable or corrupt (${err instanceof Error ? err.message : String(err)}); running on defaults`;
	}
	const rendered =
		env("CLUB_LIVE_CFG") ??
		(typeof raw.renderedCfg === "string"
			? raw.renderedCfg
			: join(env("HOME") ?? homedir(), "club-3090", "services", "litellm", "config.runtime.yaml"));
	return {
		renderedCfg: rendered,
		gateway: typeof raw.gateway === "string" ? raw.gateway : "http://127.0.0.1:4000",
		key: typeof raw.key === "string" && raw.key ? raw.key : PUBLIC_DEFAULT_KEY,
		// A malformed family entry degrades like an unknown slug (registers
		// without thinking) instead of handing the host a half-built model.
		families: Array.isArray(raw.families)
			? (raw.families as Family[]).filter(
					(f) =>
						!!f && Array.isArray(f.ids) && Array.isArray(f.ladder?.efforts) &&
						typeof f.ladder?.defaultLevel === "string" && typeof f.maxTokens === "number" &&
						!!f.compat && typeof f.compat === "object",
				)
			: [],
		problem,
	};
}

function familyFor(settings: Settings, slug: string | undefined): Family | undefined {
	if (!slug) return undefined;
	return settings.families.find((f) => Array.isArray(f.ids) && f.ids.includes(slug));
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

// FAIL CLOSED: the rendered view is the only source. The gateway's /v1/models
// also lists the user's config.local.yaml routes, so falling back to it could
// point the alias at a CLOUD model exactly when the rig is down. Unresolved
// instead: the request passes through and the gateway 400s honestly.
async function resolveLive(settings: Settings): Promise<LiveInfo | null> {
	let text = "";
	try {
		text = readFileSync(settings.renderedCfg, "utf8");
	} catch {
		return null;
	}
	return parseRenderedConfig(text);
}

// ---- minimal shapes of the omp surfaces this extension touches -------------
// Deliberately structural rather than importing @oh-my-pi/pi-coding-agent:
// the extension runs under omp's loader, and these are exactly the calls it
// makes — a host-side signature change shows up as a type error here, not as
// a runtime surprise.

interface ModelLike {
	id: string;
	name: string;
	reasoning: boolean;
	input: string[];
	thinking?: { mode: string; efforts: string[]; defaultLevel: string };
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	contextWindow: number;
	maxTokens: number;
}
interface ModelFacade {
	current(): { provider: string; id: string } | undefined;
	resolve?(selector: string): string | undefined;
}
interface UiLike {
	notify?(message: string, level?: string): void;
}
interface CtxLike {
	models?: ModelFacade;
	ui?: UiLike;
	setInterval?(fn: () => void, ms: number): void;
}
interface PiLike {
	on(event: string, handler: (...args: unknown[]) => unknown): void;
	registerProvider(id: string, config: Record<string, unknown>): void;
	unregisterProvider?(id: string): void;
	registerCommand(name: string, spec: { description: string; handler: (args: string, ctx: CtxLike) => Promise<void> }): void;
	setModel(model: string): Promise<boolean>;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;
}

export default function (pi: PiLike): void {
	// null = nothing resolved yet (fresh boot, gateway down, or rig --down).
	// Never a stale guess: a wrong slug 400s at the gateway and looks like a
	// broken rig; an unresolved alias passes the request through and the
	// gateway 400s honestly ("model 'live' not found").
	let live: LiveInfo | null = null;
	let settings = loadSettings();
	// Fingerprint of the settings the running registration was built from —
	// gateway, key, families. MUST stay in lockstep with register()'s
	// fingerprint line: any of them changing re-registers, so a --gateway
	// change lands on the next tick like a rotate does (what the docs promise).
	let registeredFrom = "";
	let boundCtx: CtxLike | undefined;
	let agentBusy = false;
	let pendingMove = false; // a change landed while busy: move at agent_end
	let servingNow = false; // the last resolve saw a route; false = last-known-good
	let warnedProblem = ""; // the settings problem already surfaced (warn once per change)
	let refreshInFlight = false;
	let refreshPending = false;
	let watcher: FSWatcher | undefined;
	let watchedDir = "";
	{
		let text = "";
		try {
			text = readFileSync(settings.renderedCfg, "utf8");
		} catch {
			/* the 2 s tick(true) below retries */
		}
		live = parseRenderedConfig(text);
		servingNow = live !== null;
	}

	function register(): void {
		// Re-registering the same provider id replaces the registration; drop
		// the old one first when the host exposes the teardown hook.
		registeredFrom = JSON.stringify([settings.gateway, settings.key, settings.families]);
		const fam = familyFor(settings, live?.slug);
		pi.unregisterProvider?.(PROVIDER);
		pi.registerProvider(PROVIDER, {
			baseUrl: `${settings.gateway}/v1`,
			apiKey: settings.key,
			api: "openai-completions",
			compat: { ...PROVIDER_COMPAT, ...(fam?.compat ?? {}) },
			models: [
				{
					// Stable id: roles written as `club-live/live` survive every
					// switch; before_provider_request maps it to the live slug.
					id: MODEL_ID,
					name: live ? `club-3090 live (${live.slug})` : "club-3090 live (not resolved yet)",
					// A slug in no family registers WITHOUT thinking — the same
					// as a `club/<id>` route that has no override: omp goes by
					// what discovery reports, which is nothing.
					reasoning: fam !== undefined,
					input: ["text"],
					thinking: fam ? { mode: "effort", ...fam.ladder } : undefined,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: live?.ctxWindow ?? DEFAULT_CTX,
					maxTokens: fam?.maxTokens ?? live?.maxOut ?? DEFAULT_MAX_OUT,
				} satisfies ModelLike,
			],
		});
	}
	try {
		register();
	} catch (err) {
		// A host-side rejection here must not break omp startup: the alias
		// simply does not list until the first tick retries, and tick's catch
		// surfaces the error once a session exists.
		console.error(`club-live: initial registration failed: ${err instanceof Error ? err.message : String(err)}`);
	}

	// The alias's own resolution for agent turns. omp's compaction and
	// skill-compression dispatch skip this hook and send the wire model as-is —
	// the gateway's `live` alias route (litellm_sync.render_block) resolves
	// those. While nothing has resolved AND no alias route exists (rig down),
	// the request passes through and the gateway 400s honestly.
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
		// A switch that landed mid-turn registered the new provider but
		// deferred the session move; apply it now, at the boundary.
		if (pendingMove) {
			pendingMove = false;
			void tick(true);
		}
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
				settings = loadSettings(); // rotate / --gateway / families land here
				installWatcher(); // the settings may have moved the rendered file
				const next = await resolveLive(settings);
				servingNow = next !== null;
				if (settings.problem && settings.problem !== warnedProblem) {
					warnedProblem = settings.problem;
					boundCtx?.ui?.notify?.(`club-live: ${settings.problem}`, "warning");
				}
				if (!next) continue; // keep the last-known-good; the next round retries
				const changed =
					!live ||
					next.slug !== live.slug ||
					next.ctxWindow !== live.ctxWindow ||
					next.maxOut !== live.maxOut;
				const settingsMoved =
					JSON.stringify([settings.gateway, settings.key, settings.families]) !== registeredFrom;
				if (!changed && !settingsMoved && !force) continue;
				live = next;
				register(); // fresh window, gateway, ladder, and key
				if (agentBusy) {
					pendingMove = true; // don't yank mid-turn; agent_end applies it
					continue;
				}
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
		} catch (err) {
			// A throwing host call (register/setModel) must not escape as an
			// unhandled rejection from `void tick()`; state stays last-known-good.
			boundCtx?.ui?.notify?.(
				`club-live: refresh failed (${err instanceof Error ? err.message : String(err)})`,
				"warning",
			);
		} finally {
			refreshInFlight = false;
		}
	}

	// fs.watch on the rendered config's directory. Re-armed whenever the
	// settings move it to a different directory (a re-run pointing at another
	// checkout or mount); same-directory swaps keep the existing subscription.
	function installWatcher(): void {
		const dir = dirname(settings.renderedCfg);
		if (watchedDir === dir) return;
		try {
			let debounce: NodeJS.Timeout | undefined;
			const w = watch(dir, (_event2, filename) => {
				if (filename && !settings.renderedCfg.endsWith(filename)) return;
				clearTimeout(debounce);
				debounce = setTimeout(() => void tick(), DEBOUNCE_MS);
			});
			watcher?.close();
			watcher = w;
			watchedDir = dir;
		} catch {
			// watcher unavailable (directory missing) — the poll still covers it
		}
	}

	pi.on("session_start", async (_event: unknown, ctx: CtxLike) => {
		boundCtx = ctx;
		ctx.setInterval?.(() => void tick(), POLL_MS);
		installWatcher();
		// Adopt the live slug shortly after start, once discovery has settled.
		setTimeout(() => void tick(true), 2_000);
	});

	pi.registerCommand("club-live", {
		description: "Re-check the live club-3090 model and re-apply the alias",
		handler: async (_args: string, ctx: CtxLike) => {
			boundCtx = ctx;
			// No agentBusy clobber: a manual /club-live during a turn defers
			// the move to agent_end like any other change.
			await tick(true);
			ctx.ui?.notify?.(
				live
					? servingNow
						? `club-live: serving ${live.slug}`
						: `club-live: nothing serving now — the alias still names ${live.slug} (requests fail at the gateway until a scene is back)`
					: "club-live: nothing resolved yet",
				live && servingNow ? "info" : "warning",
			);
		},
	});
}
