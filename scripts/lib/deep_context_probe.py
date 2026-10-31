#!/usr/bin/env python3
"""Engine-agnostic long-context probe. Driven by scripts/deep-context-probe.sh.

Stdlib only, OpenAI-compatible chat API — works against vLLM and SGLang alike.
See the wrapper's header for why this exists (club-3090#1259) and how to read it.
"""
import json, os, sys, time, urllib.error, urllib.request
# every request to the engine goes through engine_http (scripts/lib) — it resolves the
# engine's API key (Strata 0.2+) from the URL through the repo's own helpers, lazily.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_http

URL, MODEL, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
TARGET_CTX, TURN_TOKENS = int(sys.argv[4]), int(sys.argv[5])
# Depth reply cap. Big enough that the decode window clears the floor even at
# DFlash speeds; a reply shorter than this is flagged, not silently averaged.
DEPTH_MAX_TOK = 48
# ⭐ The assistant turn written into the history is FIXED, never the model's own
# reply. See run_depth() for why; in short, feeding real replies back turns the
# conversation into a feedback loop that destroys the decode window.
CANNED_REPLY = ("one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, "
                "thirteen, fourteen, fifteen, sixteen, seventeen, eighteen, nineteen, twenty")
SESSIONS, SESSION_CTX, KV_POOL = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
BASE = URL.rstrip("/")

# ---------------------------------------------------------------------------
# WHERE PREFIX REUSE IS OBSERVABLE — measured 2026-09-13, not assumed.
#
# `usage.prompt_tokens_details.cached_tokens` is gated on a SERVER flag on both
# engines, and the gate is identical for streaming and non-streaming:
#   SGLang v0.5.19  --enable-cache-report   (usage_processor.py:40/80). Even with
#                   the flag the field is OMITTED when the count is 0. 0 of the 15
#                   shipped sglang composes pass it.
#   vLLM   v0.29.0  --enable-prompt-tokens-details (launchers/cli_args.py:132,
#                   default False). 20 shipped vllm composes pass it.
# So an ABSENT field is ambiguous — "0 cached" or "not reported" — and reading it
# as 0 is exactly the false-clean this probe exists to avoid. On this rig a
# 4,212-token prompt repeated came back TTFT 3.15s -> 0.10s, `#cached-token: 4160`
# in the engine log, and `prompt_tokens_details: null` in BOTH stream modes.
#
# Fallback: both engines publish a Prometheus counter whose DELTA across one
# request is the tokens served from prefix cache. It is engine-wide (concurrent
# traffic counts too), so the column is labelled with its source:
#   SGLang  sglang:realtime_tokens_total{mode="prefill_cache"}  needs --enable-metrics (13/15 composes). Measured.
#   vLLM    vllm:prefix_cache_hits_total   v1/metrics/loggers.py:596 — source-verified, NOT live-tested here.
# A reuse self-test right after calibration (same prompt twice) decides which
# source is live. When neither is, the column renders n/a and names the flag.
# ---------------------------------------------------------------------------
_REUSE_COUNTERS = [  # (exposed counter name, required label fragment or None)
    ("sglang:realtime_tokens_total", 'mode="prefill_cache"'),
    ("vllm:prefix_cache_hits_total", None),
]
REUSE = {"source": None, "counter": None}   # source: "usage" | "metrics" | None

# ---------------------------------------------------------------------------
# Drafter acceptance and scheduler pressure, per turn (#1259).
#
# The issue names these explicitly and says why: "a queue and a dead drafter both
# read as 'slow' from the client and are indistinguishable without them." That is
# the whole reason this matters — a user reporting "TPS fell to 25" cannot tell us
# which one they hit, and neither can a decode column on its own.
#
# A dead drafter is the sharper case: you keep paying to draft tokens that are
# then rejected, so throughput lands BELOW the no-speculation baseline. Nothing in
# a cache metric can show that.
#
# Both engines expose it as counters, so the per-turn value is a DELTA across one
# request — the same primitive the cached column already uses.
#   accepted / drafted = acceptance rate for that turn
# Names verified against a running SGLang v0.5.19 and vLLM v0.29.0 package source
# on 2026-09-13.
_SPEC_COUNTERS = {          # engine -> (accepted, drafted)
    # ⭐ spec_accept_rate, NOT spec_accept_length. The rate is a FRACTION
    # (accepted/drafted) — the same quantity vLLM reports — so the two engines
    # land on one scale and an A/B means something. accept_length is mean
    # accepted tokens per step, a different axis that silently invites comparing
    # 3.55 against 100%.
    "sglang": ("sglang:spec_accept_rate", None),
    "vllm": ("vllm:spec_decode_num_accepted_tokens_total",
             "vllm:spec_decode_num_draft_tokens_total"),
}
_PRESSURE_COUNTERS = [      # first that resolves wins
    ("vllm:num_preemptions_total", None),
    ("sglang:num_queue_reqs", None),
]
SPEC = {"accepted": None, "drafted": None, "kind": None}
PRESSURE = {"counter": None}

FLAG_HINT = ("SGLang: --enable-cache-report (+ --enable-metrics for the counter fallback); "
             "vLLM: --enable-prompt-tokens-details")

def fetch_metrics():
    try:
        with engine_http.urlopen(BASE + "/metrics", timeout=10) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None

def metric_sum(body, name, label=None):
    """Sum of every series `name{...} v` in a Prometheus text body (None if absent).
    Requires the char after the name to be `{` or a space so `foo_total` never
    matches `foo_total_created`."""
    if body is None:
        return None
    total = None
    for line in body.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name):]
        if rest[:1] not in ("{", " "):
            continue
        if label and (rest[:1] != "{" or label not in rest.split("}", 1)[0]):
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        total = (total or 0.0) + v
    return total

def detect_reuse_counter():
    body = fetch_metrics()
    for name, label in _REUSE_COUNTERS:
        if metric_sum(body, name, label) is not None:
            REUSE["counter"] = (name, label)
            return
    REUSE["counter"] = None

def detect_spec_counters():
    """Resolve the drafter-acceptance and scheduler-pressure counters, if live.

    Absence is meaningful and must not read as zero: speculation may simply be
    OFF on this slug, which is a different statement from "the drafter is dead".
    The columns render n/a in that case and the header says which."""
    body = fetch_metrics()
    acc, drf = _SPEC_COUNTERS["vllm"]
    if metric_sum(body, acc) is not None and metric_sum(body, drf) is not None:
        SPEC.update(accepted=acc, drafted=drf, kind="vllm-ratio")
    else:
        sg, _ = _SPEC_COUNTERS["sglang"]
        if metric_sum(body, sg) is not None:
            # SGLang publishes a mean accepted-length GAUGE, not a pair of
            # counters, so it is read directly rather than differenced.
            SPEC.update(accepted=sg, drafted=None, kind="sglang-gauge")
    for name, label in _PRESSURE_COUNTERS:
        if metric_sum(body, name, label) is not None:
            PRESSURE["counter"] = (name, label)
            break

def spec_view(before, after):
    """Per-turn acceptance. Returns (text, is_alarming) or (None, False).

    ⭐ A dead drafter and a queued request both present as "slow" to a client
    (#1259). Acceptance separates them, and it is the one number that explains a
    throughput BELOW the no-speculation baseline: draft cost is paid, nothing is
    accepted."""
    if SPEC["kind"] == "vllm-ratio":
        a0, a1 = metric_sum(before, SPEC["accepted"]), metric_sum(after, SPEC["accepted"])
        d0, d1 = metric_sum(before, SPEC["drafted"]), metric_sum(after, SPEC["drafted"])
        if None in (a0, a1, d0, d1) or (d1 - d0) <= 0:
            return None, False
        rate = (a1 - a0) / (d1 - d0)
        return f"{rate:.0%}", rate < 0.20
    if SPEC["kind"] == "sglang-gauge":
        v = metric_sum(after, SPEC["accepted"])
        # ⚠️ EXACTLY 0 is UNPOPULATED, not collapsed. SGLang leaves this gauge at
        # zero until a speculation step has completed, so early turns read 0.00
        # while the drafter is healthy — measured on the first live run: turns 1-2
        # read 0.00, turn 3 onwards read the true value, and the probe cried
        # "ACCEPTANCE LOW" at a working drafter. That is the absence-vs-zero trap
        # this probe exists to avoid. A genuinely dead drafter still reports a
        # small NON-zero rate, which the threshold below catches.
        if v is None or v == 0:
            return None, False
        return f"{v:.0%}", v < 0.20
    return None, False

# ⭐ The API's cached_tokens is ATTENTION-KV ONLY. jb-seo's question (disc #1178)
# is about the KV and MAMBA pools diverging — KV leaf evicted, mamba checkpoint
# stranded — which needs the engine's own pool view. SGLang publishes both pools
# on /metrics (kv_/mamba_ {used,evictable,available}_tokens), split into ACTIVE
# (running requests) vs RESIDENT (radix-cached, evictable). vLLM publishes only
# an active-KV gauge and no mamba pool at all, so on vLLM the columns say so.
#
# ⚠️ Do NOT scrape the batch log line for this. Its `full token usage` / `mamba
# usage` are ACTIVE occupancy (`num_used = capacity - available - evictable`,
# pool_stats_observer.py:220-283): they read 0.00 after every completed turn
# while 62% of the pool was radix-resident, so a divergence heuristic on them
# can never fire. The previous CONTAINER=<name> path did exactly that — and
# `--tail 40` raced a once-per-second /health generate for the newest line.
def pool_view():
    body = fetch_metrics()
    g = lambda n: metric_sum(body, n)
    kv = [g("sglang:kv_used_tokens"), g("sglang:kv_evictable_tokens"), g("sglang:kv_available_tokens")]
    if None not in kv and sum(kv) > 0:
        out = {"engine": "sglang", "kv_total": sum(kv), "kv_active": kv[0], "kv_resident": kv[1]}
        mam = [g("sglang:mamba_used_tokens"), g("sglang:mamba_evictable_tokens"), g("sglang:mamba_available_tokens")]
        if None not in mam and sum(mam) > 0:
            out.update(mamba_total=sum(mam), mamba_active=mam[0], mamba_resident=mam[1])
        return out
    kv_pct = g("vllm:kv_cache_usage_perc")
    if kv_pct is not None:
        return {"engine": "vllm", "kv_active_frac": kv_pct}
    return None

def pool_cols(p):
    """Two fixed-width columns: KV resident%, mamba resident slots/total."""
    if not p or p["engine"] != "sglang":
        return f" {'n/a':>8} {'n/a':>10}"
    kv = f"{100 * p['kv_resident'] / p['kv_total']:.0f}%"
    if "mamba_total" in p:
        mam = f"{int(p['mamba_resident'])}/{int(p['mamba_total'])}"
    else:
        mam = "n/a"
    return f" {kv:>8} {mam:>10}"

# A per-run nonce in every prompt. Without it a second run against the same
# server finds its calibration prompt, its cold reference and its sessions
# already cached: the self-test read "repeat TTFT x1.00" (which, with no counter,
# is diagnosed as prefix caching OFF) and the breadth cold reference came back
# warm, so every re-query classified as cold-ish. Seen on the second live run.
RUN = int(time.time()) % 1000000
FILLER = ("The regional logistics audit " + str(RUN) + " for corridor %d recorded stable "
          "utilisation with no deviation worth escalating during this reporting cycle. ")

def _post(msgs, max_tokens):
    payload = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
               "temperature": 0.7, "top_p": 0.8, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False},
               # An OpenAI-compatible stream omits `usage` unless asked. Without
               # include_usage, prompt_tokens returns 0 — which silently poisons
               # calibration. continuous_usage_stats (honoured by vLLM v0.29.0
               # api_utils.py:297 and SGLang v0.5.19, both measured/verified) puts
               # cumulative completion_tokens on every chunk, which is the only way
               # to count TOKENS rather than chunks under speculative decoding.
               "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    body = json.dumps(payload).encode()
    return urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                  headers=engine_http.headers(BASE))

def measure(msgs, max_tokens=24):
    """One turn. Returns a dict:
        ttft          first `choices` chunk whose delta carries anything besides `role`
                      (reasoning, content, a tool call, any other kind) — NOT the first
                      chunk with content. On a reasoning model the content field stays
                      empty through the reasoning phase, and timing on it charges
                      reasoning to prefill (the #1096 retraction); an empty `role`
                      chunk comes before the prompt read on Strata / Project Maya
                      (#1752). None if no such chunk ever arrived.
        dtps          decode tokens/s. Four bases are tried in descending fidelity and
                      the one used is reported in dtps_basis; None only when every one
                      is unmeasurable, and then dtps_why says what the stream carried.
                        usage-delta   engine-counted tokens over the usage-chunk window.
                                      Exact. Needs TWO DIFFERENT completion_tokens
                                      values, which is what the 141K depth run stopped
                                      getting from turn 17 on.
                        usage-window  engine-counted tokens over the CONTENT-chunk
                                      window. Survives an engine that sends usage once.
                        chunk-count   chunks over the content window. Reads LOW under
                                      speculative decoding — one DFlash chunk carried 8
                                      tokens (measured) — so it is a floor, not a rate.
                        wall          engine-counted tokens over (wall - ttft). Last
                                      resort: charges the inter-chunk tail to decode.
        dtps_approx   True for every basis except usage-delta (rendered as a '~' prefix).
        dtps_basis    which of the four above produced dtps, or None.
        ptok, ctok    usage.prompt_tokens / completion_tokens (0 if usage missing)
        cached_usage  prompt_tokens_details.cached_tokens, or None when ABSENT
        cached_metrics  delta of the engine's prefix-hit counter across this
                      request, or None when no counter is published
        text          assistant content
    """
    # One scrape serves the reuse, acceptance and pressure deltas alike.
    need_metrics = bool(REUSE["counter"] or SPEC["kind"] or PRESSURE["counter"])
    m_before = fetch_metrics() if need_metrics else None
    t0 = time.time(); ttft = None; text = ""; usage = None
    first = None; last = None   # (t, cumulative completion_tokens) at the first token / last usage chunk
    # Content-chunk timing, tracked INDEPENDENTLY of usage: an engine that sends
    # usage once still streams its tokens, and that window is a real decode window.
    c_first = None; c_last = None
    nchunks = 0; ncontent = 0; finish = None
    with engine_http.urlopen(_post(msgs, max_tokens), timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try: d = json.loads(line[6:])
            except Exception: continue
            now = time.time()
            u = d.get("usage")
            if u:
                usage = u
                c = u.get("completion_tokens")
                if isinstance(c, int):
                    if c >= 1 and first is None: first = (now, c)
                    last = (now, c)
            ch = (d.get("choices") or [{}])[0]
            if ch:
                nchunks += 1
                delta = ch.get("delta") or {}
                # A chunk that carries something, of any kind: Strata and Project Maya
                # send an empty `role` delta before reading the prompt (#1752). Same rule
                # as bench-agentic.sh.
                if ttft is None and any(v not in (None, "", [], {}) for k, v in delta.items() if k != "role"):
                    ttft = time.time() - t0
                piece = delta.get("content") or ""
                # Reasoning tokens are decode work too, so they open the decode
                # window — timing on `content` alone charges them to prefill (#1096).
                if piece or delta.get("reasoning_content"):
                    ncontent += 1
                    if c_first is None: c_first = now
                    c_last = now
                text += piece
                finish = ch.get("finish_reason") or finish
    wall = time.time() - t0
    u = usage or {}
    ctok = u.get("completion_tokens") or 0
    dtps = None; approx = False; why = None; basis = None
    cwin = (c_last - c_first) if (c_first is not None and c_last is not None) else 0.0
    # Two tokens is the minimum that defines a rate, and a window under 50ms is
    # scheduler noise rather than decode; below either, fall through to the next
    # basis instead of publishing a number built from one inter-chunk gap.
    def ok(win, tok): return win > 0.05 and tok >= 2
    if first and last and ok(last[0] - first[0], last[1] - first[1]):
        window = last[0] - first[0]; dtok = last[1] - first[1]; basis = "usage-delta"
    elif ok(cwin, ctok - 1):
        window = cwin; dtok = ctok - 1; basis = "usage-window"; approx = True
    elif ok(cwin, ncontent - 1):
        window = cwin; dtok = ncontent - 1; basis = "chunk-count"; approx = True
    elif ok(wall - (ttft or 0), ctok - 1):
        window = wall - (ttft or 0); dtok = ctok - 1; basis = "wall"; approx = True
    # Report n/a rather than 0.0 for an unmeasurable window (#1267): a zero here
    # is indistinguishable from a genuine silent-empty turn. ⭐ But an n/a with NO
    # REASON is the same ambiguity one level up — it reads as "slow rig" when it
    # actually means "instrument failed". So state what the stream actually carried;
    # that is what separates "the model stopped early" from "the probe went blind".
    if basis is not None:
        dtps = dtok / window
    else:
        useen = "none" if last is None else (
            f"one value ({last[1]})" if (first is None or last[1] == first[1]) else "ok")
        why = (f"{ctok} completion tok, {nchunks} chunk(s) / {ncontent} with content "
               f"over {cwin*1000:.0f}ms, finish_reason={finish or 'none'}, "
               f"usage deltas={useen}")
    ptd = u.get("prompt_tokens_details")
    cached_usage = ptd.get("cached_tokens") if isinstance(ptd, dict) else None
    cached_metrics = None
    m_after = fetch_metrics() if (need_metrics and m_before is not None) else None
    if REUSE["counter"] and m_before is not None and m_after is not None:
        a = metric_sum(m_before, *REUSE["counter"]); b = metric_sum(m_after, *REUSE["counter"])
        if a is not None and b is not None:
            cached_metrics = int(b - a)
    accept, accept_low = spec_view(m_before, m_after) if m_after is not None else (None, False)
    preempt = None
    if PRESSURE["counter"] and m_before is not None and m_after is not None:
        a = metric_sum(m_before, *PRESSURE["counter"]); b = metric_sum(m_after, *PRESSURE["counter"])
        if a is not None and b is not None:
            preempt = int(b - a)
    return {"ttft": ttft, "dtps": dtps, "dtps_approx": approx, "dtps_why": why,
            "dtps_basis": basis, "nchunks": nchunks, "ncontent": ncontent,
            "accept": accept, "accept_low": accept_low, "preempt": preempt,
            "finish_reason": finish,
            "ptok": u.get("prompt_tokens") or 0,
            "ctok": ctok, "cached_usage": cached_usage, "cached_metrics": cached_metrics,
            "text": text.strip()}

def cached_of(m):
    """(cached_tokens or None, source). None means UNOBSERVABLE, never 0."""
    if m["cached_usage"] is not None:
        return m["cached_usage"], "usage"
    if REUSE["source"] == "usage":
        # The field is proven live on this server and both engines omit/zero it
        # when nothing was reused (SGLang _details_if_cached; vLLM cached_tokens=0).
        return 0, "usage"
    if REUSE["source"] == "metrics" and m["cached_metrics"] is not None:
        return m["cached_metrics"], "metrics"
    return None, None

def fmt_cached(m):
    c, _ = cached_of(m)
    return "n/a" if c is None else f"{c:,}"

def fmt_dtps(m):
    if m["dtps"] is None: return "n/a"
    return f"{'~' if m['dtps_approx'] else ''}{m['dtps']:.1f}"

def fmt_ttft(m):
    return "n/a" if m["ttft"] is None else f"{m['ttft']:.2f}"

def http_error_text(e):
    if isinstance(e, urllib.error.HTTPError):
        try: body = e.read().decode("utf-8", "replace")[:300]
        except Exception: body = ""
        return f"HTTP {e.code}: {body}"
    return repr(e)

_TOK_PER_REP = None
_CAL = None   # the calibration prompt + its measurement, reused by the self-test

def calibrate():
    """Tokens per FILLER repetition, MEASURED not assumed.

    A words//N guess was 3.2x off on the first attempt here, which turned a
    28K-token session into 90K and thrashed the pool. The engine reports
    prompt_tokens; use it. 240 reps (~5K tokens here) so the self-test repeat is
    long enough to show reuse even on vLLM+MTP, which only reports full 1,600-
    token blocks and drops the last one (cached = (floor(N/1600)-1)*1600)."""
    global _TOK_PER_REP, _CAL
    if _TOK_PER_REP: return _TOK_PER_REP
    reps = 240
    prompt = (FILLER % 0) * reps
    m = measure([{"role": "user", "content": prompt}], max_tokens=1)
    if not m["ptok"]:
        # Falling back to a default here produced 6,000-repetition turns (~150K
        # tokens) and a 133s first turn that read as a slow rig rather than a
        # broken probe. Refuse instead of guessing.
        sys.exit("calibration failed: server reported prompt_tokens=0 — the stream is not "
                 "returning usage. Check stream_options/include_usage on this engine. "
                 "Refusing rather than guessing a token size.")
    _TOK_PER_REP = m["ptok"] / reps
    _CAL = (prompt, m)
    print(f"  calibration: {_TOK_PER_REP:.1f} tokens per repetition "
          f"({m['ptok']:,} tok over {reps} reps, ttft {fmt_ttft(m)}s)")
    return _TOK_PER_REP

def reuse_selftest():
    """Send the calibration prompt a second time and see WHERE the reuse shows up.
    Decides REUSE['source'] for the whole run. Never guesses: if the usage field is
    absent and no counter moved, the cached column is n/a for the run."""
    calibrate()
    prompt, first = _CAL
    rep = measure([{"role": "user", "content": prompt}], max_tokens=1)
    ratio = (rep["ttft"] / first["ttft"]) if (rep["ttft"] and first["ttft"]) else None
    ratio_s = f"x{ratio:.2f}" if ratio is not None else "n/a"
    if rep["cached_usage"] is not None:
        REUSE["source"] = "usage"
    elif rep["cached_metrics"] is not None and rep["cached_metrics"] > 0:
        REUSE["source"] = "metrics"
    usage_s = ("absent" if rep["cached_usage"] is None else f"{rep['cached_usage']:,}")
    ctr_s = ("no counter on /metrics" if not REUSE["counter"] else
             f"{REUSE['counter'][0]} delta {rep['cached_metrics']:,}" if rep["cached_metrics"] is not None
             else f"{REUSE['counter'][0]} unreadable")
    print(f"  reuse self-test (same {first['ptok']:,}-token prompt again): repeat TTFT {ratio_s}; "
          f"usage.prompt_tokens_details.cached_tokens: {usage_s}; {ctr_s}")
    if REUSE["source"]:
        print(f"  cached column source: {REUSE['source']}"
              + (" (engine-wide counter delta — concurrent traffic during a turn is counted too)"
                 if REUSE["source"] == "metrics" else ""))
        if REUSE["source"] == "usage" and rep["cached_usage"] == 0 and ratio is not None and ratio < 0.5:
            print("  ⚠️ the field is present but reads 0 while the repeat TTFT collapsed — the engine "
                  "under-reports reuse; treat the cached column with suspicion")
        return
    if ratio is not None and ratio < 0.5:
        print(f"  ⚠️ reuse HAPPENED (repeat TTFT {ratio_s}) but this server does NOT REPORT it. "
              f"The cached column is n/a for this run — NOT 0. To light it up: {FLAG_HINT}.")
    else:
        print(f"  ⚠️ the engine did not reuse an identical prompt (repeat TTFT {ratio_s}) and reports no "
              f"cached count — prefix caching looks OFF, or unobservable ({FLAG_HINT}). cached column: n/a.")

def chunk(tokens, seed):
    return (FILLER % seed) * max(1, int(tokens / calibrate()))

def run_depth():
    print(f"  DEPTH — one conversation to ~{TARGET_CTX:,} accumulated tokens, "
          f"~{TURN_TOKENS:,} per turn")
    reuse_selftest()
    p = pool_view()
    if p and p["engine"] == "sglang":
        print("  pool view: /metrics — kv_res = % of the KV pool that is radix-RESIDENT (reusable), "
              "mamba_res = resident state slots / pool. Active-use (running-request) occupancy is "
              "~0 between turns by construction and is not what is shown.")
    elif p and p["engine"] == "vllm":
        print("  pool view: n/a — vLLM publishes only an ACTIVE KV gauge (vllm:kv_cache_usage_perc) "
              "and no mamba pool metric, so residency is not observable on this engine.")
    else:
        print("  pool view: n/a — no pool gauges on /metrics (SGLang needs --enable-metrics; "
              "vLLM has no mamba pool metric).")
    if os.environ.get("CONTAINER"):
        print("  note: CONTAINER is no longer used — the batch log line reports ACTIVE occupancy, "
              "not residency (see pool_view in the source).")
    print("  decode_tps: engine-counted tokens over the decode window. A '~' prefix means the "
          "exact basis (two differing usage chunks) was unavailable and a fallback window was "
          "used; the basis is named per row. 'chunk-count' reads LOW under spec-dec.")
    if SPEC["kind"]:
        print(f"  accept: per-turn drafter acceptance (accepted/drafted on both engines, so it is"
              " comparable across them)."
              " A dead drafter and a queued request both read as 'slow' from the client and are"
              " indistinguishable without this (#1259); only a dead drafter puts throughput BELOW"
              " the no-speculation baseline.")
    else:
        print("  accept: n/a — no drafter-acceptance counter on /metrics. That means speculation is"
              " OFF or unreported on this slug; it does NOT mean acceptance is zero.")
    print(f"  {'turn':>4} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8} "
          f"{'decode_tps':>11} {'accept':>7} {'kv_res':>8} {'mamba_res':>10}")
    msgs = []; turn = 0; ptok = 0; prev_ttft = None; base_ctok = None
    while ptok < TARGET_CTX:
        turn += 1
        # The ask must satisfy TWO constraints that pull against each other.
        #
        # Long enough to open a decode window: "reply OK" gave 1-2 tokens and the
        # decode column was n/a on every turn, by construction.
        #
        # But it must also COMPLETE inside the cap. Asking for sixty numbers needs
        # ~90 tokens, so every early turn was truncated mid-word ("...twen") and
        # that fragment went into the history as what the assistant said. Measured
        # consequence: replies fell 48 -> 33 -> 21 -> 11 -> 7 -> 4 tokens and stuck
        # at 4, taking the decode column to n/a from 104K on. Two controls on the
        # same boot showed the engine was never involved -- a single-turn probe
        # held 48/48 to 190,670 tokens, and a multi-turn probe whose history was a
        # fixed well-formed reply showed no trend to 144,521. The probe was
        # teaching the model to be terse and then measuring the result.
        #
        # Twenty numbers is ~30 tokens: a complete reply, and still a ~150ms window
        # at 200 tok/s, well clear of the floor.
        msgs.append({"role": "user", "content": chunk(TURN_TOKENS, turn)
                     + f"\n\nTurn {turn}: count from one to twenty in words, comma separated."})
        try:
            m = measure(msgs, max_tokens=DEPTH_MAX_TOK)
        except Exception as e:
            print(f"  turn {turn}: ERROR {http_error_text(e)}"); return
        ptok = m["ptok"]
        if base_ctok is None and m["ctok"]: base_ctok = m["ctok"]
        # ⭐ Append a FIXED assistant turn, NOT the model's own reply.
        #
        # Feeding real replies back makes the conversation a feedback loop: any
        # downward drift in reply length is read by the next turn as the house
        # style and reinforced. Measured here twice. With an ask that overran the
        # cap, replies went 48 -> 4 and stuck. With an ask that completes inside
        # it, they still went 40 -> 2 by 98K. Both took the decode column to n/a
        # over exactly the depth range this probe exists to measure.
        #
        # Two same-boot controls separate cause from coincidence: a single-turn
        # probe held 48/48 at 190,670 tokens, so DEPTH does not do this; and a
        # multi-turn probe with a fixed history showed no trend to 144,521, so
        # ACCUMULATION does not either. Only the fed-back arm collapsed.
        #
        # The real reply is still measured, and its drift is still flagged below
        # — it simply does not get to set the stimulus for the next turn. That
        # keeps the decode window comparable across depths, which is the whole
        # point of the column. Drift itself is a model-behaviour finding and
        # belongs in the flags, not silently inside the thing being measured.
        msgs.append({"role": "assistant", "content": CANNED_REPLY})
        flags = ""
        if m["ttft"] is None:
            flags += "   <- NO CHOICES CHUNK (empty stream)"
        elif prev_ttft and prev_ttft > 0 and m["ttft"] > prev_ttft * 2.0:
            flags += "   <- TTFT MORE THAN DOUBLED"
        if m["ctok"] == 0:
            flags += "   <- EMPTY TURN (0 completion tokens)"
        elif not m["text"]:
            flags += f"   <- {m['ctok']} tokens but no content (reasoning-only or parser ate it)"
        # A truncated reply POISONS the history — it is the defect above, and it
        # is silent, so say so the moment it happens rather than leaving a reader
        # to infer it from a decode column that decays later.
        if m.get("finish_reason") == "length":
            flags += (f"   <- TRUNCATED at the {DEPTH_MAX_TOK}-token cap; this fragment"
                      " goes into the history and biases every later turn")
        # Reply length is measured against TURN 1 on this run, not against the cap:
        # the cap is a probe constant, turn 1 is what this model actually does with
        # this ask, so the ratio survives a change of ask or model.
        if base_ctok and m["ctok"] and m["ctok"] < base_ctok * 0.6:
            flags += (f"   <- reply shrank to {m['ctok']} tok vs {base_ctok} on turn 1"
                      f" (finish={m.get('finish_reason') or 'none'})")
        # ⭐ An n/a with no reason reads as "slow rig" when it means "instrument
        # failed" — the same ambiguity #1267 closed one level down. Name it.
        if m["dtps"] is None and m.get("dtps_why"):
            flags += f"   <- decode n/a: {m['dtps_why']}"
        elif m.get("dtps_approx") and m.get("dtps_basis"):
            flags += f"   <- decode basis: {m['dtps_basis']}"
        if m.get("accept_low"):
            flags += f"   <- DRAFTER ACCEPTANCE LOW ({m['accept']})"
        if m.get("preempt"):
            flags += f"   <- {m['preempt']} preemption(s) this turn"
        print(f"  {turn:>4} {ptok:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8} "
              f"{fmt_dtps(m):>11} {(m.get('accept') or 'n/a'):>7}"
              f"{pool_cols(pool_view())}{flags}", flush=True)
        prev_ttft = m["ttft"]
        if not ptok:
            print("  stopping: prompt_tokens came back 0 — usage vanished mid-run"); break
        if turn > 200:
            print("  stopping: 200 turns without reaching target"); break

def verdict(m, cold_ttft, pressure_known):
    """Breadth verdict. cold_ish = TTFT within 30% of the cold reference.

    ⭐ A verdict may only name a MECHANISM the probe can actually observe. What a
    row observes is "no reuse, and it cost a cold prefill". *Eviction* is one
    EXPLANATION for that, licensed only when the cache is known to be pressured.
    This took (m, cold_ttft) — no pressure input at all — and returned "CLEAN
    eviction" unconditionally, so the identical string appeared whether or not
    eviction was possible and carried no information either way (#1299).

    pressure_known is False when the probe cannot establish pressure; the wording
    then stays agnostic about cause."""
    if m["ttft"] is None:
        return "NO RESPONSE (no choices chunk)"
    cold_ish = m["ttft"] > cold_ttft * 0.7
    cached, _ = cached_of(m)
    if cached is None:
        return ("cold-like TTFT" if cold_ish else "warm-like TTFT") + " — reuse count unobservable (see self-test)"
    if cached == 0:
        if not cold_ish:
            return "fast despite cached=0 (?)"
        return ("CLEAN eviction" if pressure_known
                else "NO REUSE, cold cost — cause NOT established (pressure unknown)")
    frac = cached / m["ptok"] if m["ptok"] else 0.0
    if cold_ish:
        if frac >= 0.5:
            return f"STRANDED — {frac:.0%} reported cached but cold-cost"
        return (f"PARTIAL eviction — {frac:.0%} cached, cold-cost expected" if pressure_known
                else f"{frac:.0%} cached but cold-cost — cause NOT established (pressure unknown)")
    return f"HEALTHY reuse ({frac:.0%} cached)"

def run_breadth():
    reuse_selftest()
    p = pool_view()
    pool = KV_POOL or (int(p["kv_total"]) if p and p.get("kv_total") else 0)
    total = SESSIONS * SESSION_CTX
    print(f"  BREADTH — {SESSIONS} sessions x ~{SESSION_CTX:,} tok = ~{total:,}")
    # ⭐ The nominal pool is NOT the reusable radix budget. Measured on SGLang
    # (#1299): a 30,895-token prefix that reused perfectly (30,848 tok, 0.23s vs
    # a 24.62s cold cost) was GONE after five unrelated ~28K sessions — about 66%
    # of max_total_num_tokens. Running-request working space and the eviction
    # watermark come out of the nominal figure first.
    #
    # So `total <= pool` does NOT mean "no eviction pressure". The old wording
    # asserted exactly that and told a reader to discard verdicts that were right.
    # Nominal pool is a PLANNING ratio only; it never licenses a claim about
    # cause in either direction.
    pressure_known = False
    if pool:
        print(f"  KV pool {pool:,} tok ({'KV_POOL' if KV_POOL else 'from /metrics'}): "
              f"planned {total / pool:.2f}x nominal pool")
        pressure_known = total > pool
        if not pressure_known:
            print("  note: planned tokens are under the NOMINAL pool, but that does NOT mean there is no "
                  "eviction pressure — eviction has been measured at ~66% of nominal (#1299). Verdicts "
                  "below will not name a cause; raise SESSIONS or SESSION_CTX to exceed the pool outright.")
    else:
        print("  ⚠️ KV pool unknown (KV_POOL unset and no pool gauge on /metrics) — verdicts below will "
              "report what was observed without naming a cause.")
    if p and p.get("mamba_total"):
        print(f"  mamba pool: {int(p['mamba_total'])} state slots — one distinct prefix per slot, so "
              f"breadth can exhaust it long before KV fills (compose header arithmetic).")
    # Seed SESSIONS+1: never a session seed, and the same digit class as them —
    # "9999" tokenised ~3 tokens/rep longer than "1" and made the reference 14%
    # longer than the sessions it is compared against.
    try:
        cold = measure([{"role": "user", "content": chunk(SESSION_CTX, SESSIONS + 1) + "\n\nReply OK."}],
                       max_tokens=8)
    except Exception as e:
        print(f"  cold reference: ERROR {http_error_text(e)}"); return
    if cold["ttft"] is None:
        print("  cold reference: no choices chunk — cannot classify anything; stopping"); return
    print(f"  cold reference: {cold['ptok']:,} tok, ttft {cold['ttft']:.2f}s (measured on an "
          f"unpressured pool; later cold prefills also pay eviction, so >= this)")
    print(f"  {'open':>8} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8} {'kv_res':>8} {'mamba_res':>10}")
    sess = []
    for i in range(1, SESSIONS + 1):
        m0 = [{"role": "user", "content": chunk(SESSION_CTX, i) + f"\n\nSession {i}: reply OK."}]
        try:
            m = measure(m0, max_tokens=8)
        except Exception as e:
            print(f"  session {i}: ERROR {http_error_text(e)}"); return
        m0.append({"role": "assistant", "content": m["text"] or "OK"})
        sess.append(m0)
        print(f"  {i:>8} {m['ptok']:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8}{pool_cols(pool_view())}", flush=True)
    print(f"  re-query (order matters: each re-query is itself a prefill that can evict the next)")
    print(f"  {'session':>8} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8}   verdict")
    for idx in sorted({0, 1, len(sess) // 2, len(sess) - 1}):
        s = list(sess[idx]); s.append({"role": "user", "content": "Follow-up: reply OK."})
        try:
            m = measure(s, max_tokens=8)
        except Exception as e:
            print(f"  {idx + 1:>8}: ERROR {http_error_text(e)}"); continue
        print(f"  {idx + 1:>8} {m['ptok']:>11,} {fmt_cached(m):>9} {fmt_ttft(m):>8}   "
              f"{verdict(m, cold['ttft'], pressure_known)}",
              flush=True)

def compact_summary(sess_name, turns):
    """What a client sends after compacting: a short digest replacing the history.

    Modelled on what compaction actually does — the long conversation is thrown
    away and replaced by a summary, then work continues in the SAME session. The
    old prefix is not freed; it simply stops being referenced and waits to be
    evicted. That is the state we want to observe from the OTHER session."""
    return (f"Summary of {sess_name} so far: {turns} turns of a regional logistics "
            f"audit were reviewed. Findings were routine, utilisation stable, no "
            f"escalations. Continue from this summary.")

def run_concurrent():
    """Two deep sessions alive at once, then compact one and watch the other.

    This is the case neither existing mode covers. `depth` is one session;
    `breadth` is many shallow ones. The question both open reports turn on is
    what happens when a second long session exists, and what a COMPACTION in one
    does to the other.

    ⭐ Turns are interleaved round-robin rather than issued in parallel, and that
    is deliberate. Only one request is ever in flight, so `max_running_requests`
    stays 1 and the ACTIVE side of the pools is held constant. What accumulates
    is warm-prefix state — which is the thing under test. Issuing them in
    parallel would change both variables at once and confound exactly the
    measurement we came for.

    Prediction being tested (from a measured single-session run): one 141K
    conversation consumed 49 of 55 mamba state slots, growing ~2 per turn because
    each turn is a new distinct prefix. If slots scale with turn count rather than
    session count, two deep sessions cannot both be resident, and the second
    should force the first out well before the KV pool is exhausted."""
    reuse_selftest()
    p = pool_view()
    print(f"  CONCURRENT — {SESSIONS} sessions interleaved to ~{TARGET_CTX:,} tok each "
          f"(~{SESSIONS * TARGET_CTX:,} total), then one compacts")
    if p and p.get("mamba_total"):
        print(f"  mamba pool: {int(p['mamba_total'])} state slots. A single deep session was "
              f"measured consuming 49 of 55 — slots scale with TURN COUNT, not session count.")
    if p and p.get("kv_total"):
        print(f"  kv pool: {int(p['kv_total']):,} tok nominal. ⚠️ eviction has been measured at "
              f"~66% of nominal, so the nominal figure is a planning number only (#1299).")
    print(f"  {'phase':>16} {'sess':>5} {'turn':>5} {'prompt_tok':>11} {'cached':>9} {'ttft_s':>8} "
          f"{'decode':>8} {'accept':>7}{'  pools' if p else ''}")

    sessions = [[] for _ in range(SESSIONS)]
    turns = [0] * SESSIONS

    def step(i, phase, ask=None, reset=None):
        """One turn on session i. `reset` replaces the history (a compaction)."""
        if reset is not None:
            sessions[i] = [{"role": "user", "content": reset}]
        else:
            turns[i] += 1
            sessions[i].append({"role": "user", "content": chunk(TURN_TOKENS, (i + 1) * 1000 + turns[i])
                                + (ask or f"\n\nTurn {turns[i]}: count from one to twenty in words, comma separated.")})
        try:
            m = measure(sessions[i], max_tokens=DEPTH_MAX_TOK)
        except Exception as e:
            print(f"  {phase:>16} {chr(65+i):>5} — ERROR {http_error_text(e)}"); return None
        sessions[i].append({"role": "assistant", "content": CANNED_REPLY})
        flags = ""
        if m.get("preempt"):
            flags += f"   <- {m['preempt']} preemption(s)"
        print(f"  {phase:>16} {chr(65+i):>5} {turns[i]:>5} {m['ptok']:>11,} {fmt_cached(m):>9} "
              f"{fmt_ttft(m):>8} {fmt_dtps(m):>8} {(m.get('accept') or 'n/a'):>7}"
              f"{pool_cols(pool_view())}{flags}", flush=True)
        return m

    # --- phase 1: grow every session, round robin -------------------------
    while True:
        last = [step(i, "grow") for i in range(SESSIONS)]
        if any(m is None for m in last):
            return
        if all(m["ptok"] >= TARGET_CTX for m in last) or max(turns) > 200:
            break

    # --- phase 2: baseline reuse, every session, before anything compacts --
    base = {}
    for i in range(SESSIONS):
        m = step(i, "baseline")
        if m is None: return
        base[i] = cached_of(m)[0]

    # --- phase 3: session A compacts. Its own cost is informative, but the
    #     question is what it does to the OTHERS. -------------------------
    m = step(0, "COMPACT A", reset=compact_summary("session A", turns[0]))
    if m is None: return

    # --- phase 4: the others, immediately. Did A's compaction cost them? ---
    for i in range(1, SESSIONS):
        m = step(i, "after-compact")
        if m is None: return
        now, was = cached_of(m)[0], base.get(i)
        if now is None or was is None:
            print(f"     session {chr(65+i)}: reuse unobservable — cannot say whether the compaction "
                  f"cost it anything (see the self-test above)")
        elif was > 0 and now < was * 0.5:
            print(f"     ⚠️ session {chr(65+i)} LOST REUSE after A compacted: {was:,} -> {now:,} cached tok")
        else:
            print(f"     session {chr(65+i)} kept its prefix across A's compaction "
                  f"({was:,} -> {now:,} cached tok)")

    # --- phase 5: A regrows. The compaction itself is cheap; the expensive
    #     part is rebuilding, and THAT is what can evict a neighbour. ------
    for _ in range(3):
        if step(0, "A regrow") is None: return
    for i in range(1, SESSIONS):
        m = step(i, "after-regrow")
        if m is None: return
        now, was = cached_of(m)[0], base.get(i)
        if now is None or was is None:
            print(f"     session {chr(65+i)}: reuse unobservable")
        elif was > 0 and now < was * 0.5:
            print(f"     ⚠️ session {chr(65+i)} LOST REUSE while A rebuilt: {was:,} -> {now:,} cached tok")
        else:
            print(f"     session {chr(65+i)} survived A's rebuild ({was:,} -> {now:,} cached tok)")

detect_reuse_counter()
detect_spec_counters()
{"depth": run_depth, "breadth": run_breadth, "concurrent": run_concurrent}[MODE]()
