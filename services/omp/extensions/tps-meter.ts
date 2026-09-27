import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

// Live throughput + prompt-cache meter for the model in use.
//
// Measures per assistant message:
//   - TTFT:        message_start -> first generated delta (thinking or text),
//                  i.e. prefill + first-token latency.
//   - decode TPS:  usage.output tokens over the decode window
//                  (first delta -> message_end), so prefill is excluded and
//                  the number is real decode throughput, not request TPS.
//   - cache:       share of this request's prompt the engine served from its
//                  prefix cache instead of prefilling it:
//                  cacheRead / (input + cacheRead + cacheWrite). Both omp and pi
//                  split usage that way (`input` is the UNcached part).
//   - session aggregate for the currently selected model (resets when the
//                  model changes): sum(output) / sum(decode seconds), and the
//                  cached share sum(cacheRead) / sum(prompt) — the number to
//                  watch in a long agent session (90 %+ once past its first
//                  turns; a drop means something rewrote the front of the prompt).
//
// During a stream the status shows elapsed time + phase (thinking/writing);
// exact numbers replace it at message_end. Providers that omit usage leave
// the status untouched rather than showing a fabricated number. Cache figures
// only appear once the backend has reported a cache hit for this model: usage
// normalisation turns "not reported" into 0, which would otherwise read as 0 %.

const STATUS_KEY = "tps-meter";
// Throttle live status refreshes; deltas arrive far faster than the UI needs.
const LIVE_REFRESH_MS = 250;

type Stream = {
  t0: number; // message_start
  tFirst: number | null; // first thinking/text delta
  phase: "prefill" | "thinking" | "writing";
  lastPush: number;
};

type Usage = {
  output?: number;
  input?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number; // pi
  reasoningTokens?: number; // omp
};

const compact = (n: number): string =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`;

export default function (pi: ExtensionAPI) {
  let stream: Stream | null = null;
  let aggModel: string | null = null;
  let aggTokens = 0;
  let aggSeconds = 0;
  let aggCount = 0;
  let aggPrompt = 0;
  let aggCached = 0;
  let reportsCache = false;

  const push = (ctx: Parameters<Parameters<typeof pi.on>[1]>[1], text: string, force = false) => {
    const now = Date.now();
    if (!force && stream && now - stream.lastPush < LIVE_REFRESH_MS) return;
    if (stream) stream.lastPush = now;
    try {
      ctx.ui.setStatus(STATUS_KEY, text);
    } catch {
      // no UI (print/rpc mode) — meter is inert there
    }
  };

  pi.on("message_start", (event, ctx) => {
    if ((event.message as { role?: string }).role !== "assistant") return;
    stream = { t0: Date.now(), tFirst: null, phase: "prefill", lastPush: 0 };
    const modelId = ctx.model?.id ?? null;
    if (modelId !== aggModel) {
      aggModel = modelId;
      aggTokens = 0;
      aggSeconds = 0;
      aggCount = 0;
      aggPrompt = 0;
      aggCached = 0;
      reportsCache = false;
    }
  });

  pi.on("message_update", (event, ctx) => {
    if (!stream) return;
    const kind = event.assistantMessageEvent?.type;
    if (kind !== "thinking_delta" && kind !== "text_delta") return;
    const now = Date.now();
    if (stream.tFirst === null) {
      stream.tFirst = now;
      push(ctx, `gen ${(now - stream.t0) / 1000 >= 100 ? Math.round((now - stream.t0) / 1000) : ((now - stream.t0) / 1000).toFixed(1)}s ttft · ${kind === "thinking_delta" ? "thinking" : "writing"}…`, true);
    }
    stream.phase = kind === "thinking_delta" ? "thinking" : "writing";
    const el = (now - stream.t0) / 1000;
    const ttft = stream.tFirst !== null ? (stream.tFirst - stream.t0) / 1000 : 0;
    push(ctx, `gen ${el.toFixed(1)}s · ttft ${ttft.toFixed(1)}s · ${stream.phase}…`);
  });

  pi.on("message_end", (event, ctx) => {
    const s = stream;
    stream = null;
    if (!s) return;
    const msg = event.message as { role?: string; usage?: Usage };
    if (msg.role !== "assistant" || s.tFirst === null) return;
    const u = msg.usage;
    const out = u?.output;
    if (typeof out !== "number" || out <= 0) return;

    const tEnd = Date.now();
    const decodeS = (tEnd - s.tFirst) / 1000;
    const ttftS = (s.tFirst - s.t0) / 1000;
    if (decodeS <= 0) return;
    const tps = out / decodeS;

    aggTokens += out;
    aggSeconds += decodeS;
    aggCount += 1;
    const aggTps = aggSeconds > 0 ? aggTokens / aggSeconds : 0;

    // Prompt-cache share: this request, and the session so far.
    const cached = typeof u?.cacheRead === "number" ? u.cacheRead : 0;
    const prompt = (u?.input ?? 0) + cached + (u?.cacheWrite ?? 0);
    if (cached > 0) reportsCache = true;
    let cachePart = "";
    let aggCachePart = "";
    if (prompt > 0) {
      aggPrompt += prompt;
      aggCached += cached;
      if (reportsCache) {
        cachePart = ` · cache ${Math.round((100 * cached) / prompt)}% of ${compact(prompt)}`;
        aggCachePart = ` · Σ cache ${Math.round((100 * aggCached) / aggPrompt)}%`;
      }
    }

    const think = u?.reasoningTokens ?? u?.reasoning;
    const thinkPart = typeof think === "number" && think > 0 ? ` · think ${think.toLocaleString("en-US")}` : "";
    push(
      ctx,
      `⚡ ${tps.toFixed(1)} tok/s · ttft ${ttftS.toFixed(1)}s · out ${out.toLocaleString("en-US")}${thinkPart}${cachePart} · Σ ${aggTps.toFixed(1)} tok/s${aggCachePart} (n=${aggCount})`,
      true,
    );
  });
}
