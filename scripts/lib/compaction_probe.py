#!/usr/bin/env python3
"""Core of scripts/compaction-probe.sh — see that file for the why.

Measures whether decode throughput degrades AFTER a context compaction at depth,
which is the shape reported in club-3090 #1052 and which nothing we shipped
measured: bench-agentic grows monotonically and never compacts.
"""
import argparse, json, os, sys, time, urllib.request
# every request to the engine goes through engine_http (scripts/lib) — it resolves the
# engine's API key (Strata 0.2+) from the URL through the repo's own helpers, lazily.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_http

_METRICS = ("vllm:request_generation_tokens_sum",
            "vllm:request_decode_time_seconds_sum",
            "vllm:num_preemptions_total")


def _post(url, payload, timeout=1800):
    req = urllib.request.Request(url + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers=engine_http.headers(url))
    with engine_http.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _metrics(url):
    """vLLM's own counters. {} when unavailable — the probe degrades to client
    timing rather than failing, but says so, because a silent fallback here would
    hand back exactly the kind of number #1096 was built on."""
    out = {}
    try:
        with engine_http.urlopen(url + "/metrics", timeout=10) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                name, _, val = line.partition(" ")
                base = name.split("{", 1)[0]
                if base in _METRICS:
                    try:
                        out[base] = out.get(base, 0.0) + float(val)
                    except ValueError:
                        pass
    except Exception:
        return {}
    return out


# ⚠️ 2000, not 200. On a THINKING model a small budget is consumed by reasoning
# and leaves a stub of content — which this probe's own thin-reply detector then
# flags as possible degeneration. Measured 2026-09-08: 3/6 post-compaction replies
# under 40 chars at max_tokens=200, entirely a budget artifact. Same trap as
# bench-agentic (#1218) and gdn-mtp-apc-repro. Give reasoning room, or the
# instrument reports on itself.
def _gen(url, model, msgs, max_tokens=2000):
    m0 = _metrics(url)
    t0 = time.time()
    d = _post(url, {"model": model, "messages": msgs, "max_tokens": max_tokens,
                    # greedy + seeded: reps must be comparable to each other and
                    # across runs, or a before/after comparison means nothing.
                    "temperature": 0.0, "seed": 1052})
    wall = time.time() - t0
    m1 = _metrics(url)
    u = d.get("usage", {}) or {}
    comp = u.get("completion_tokens", 0) or 0
    eng = None
    if m0 and m1:
        dt = (m1.get("vllm:request_decode_time_seconds_sum", 0.0)
              - m0.get("vllm:request_decode_time_seconds_sum", 0.0))
        dk = (m1.get("vllm:request_generation_tokens_sum", 0.0)
              - m0.get("vllm:request_generation_tokens_sum", 0.0))
        if dt > 0:
            eng = dk / dt
    pre = ((m1.get("vllm:num_preemptions_total", 0.0)
            - m0.get("vllm:num_preemptions_total", 0.0)) if (m0 and m1) else 0.0)
    msg = d["choices"][0]["message"]
    return {"prompt": u.get("prompt_tokens", 0) or 0, "completion": comp, "wall": wall,
            "client_tps": (comp / wall) if wall > 0 else 0.0, "engine_tps": eng,
            "preemptions": pre, "text": msg.get("content") or ""}


def _phase(url, model, msgs, reps, label, continue_session=False):
    """`continue_session=True` APPENDS each reply and a fresh user turn, so the
    conversation keeps growing the way a real agent does after a compaction.

    Without it this sent the SAME message list `reps` times and measured
    steady-state throughput at a fixed depth — which cannot see the reported
    symptom at all. The report (#1052) is "drop to around 25 and STAY [there]";
    persistence only shows up if the session actually continues.
    """
    rows = []
    msgs = list(msgs)
    for i in range(reps):
        try:
            r = _gen(url, model, msgs)
        except Exception as e:
            print(f"  {label} rep {i+1}: FAILURE — {e}", flush=True)
            return rows, e
        rows.append(r)
        if continue_session:
            msgs.append({"role": "assistant", "content": r["text"] or "ok"})
            msgs.append({"role": "user",
                         "content": f"[post-compaction turn {i+1}] Continue: describe "
                                    f"one more component of that system in detail."})
        eng = f"{r['engine_tps']:.1f}" if r["engine_tps"] else "n/a"
        warn = "  ⚠ preempted" if r["preemptions"] else ""
        print(f"  {label} rep {i+1}: ctx={r['prompt']:,} client={r['client_tps']:.1f} "
              f"engine={eng} tok/s{warn}", flush=True)
    return rows, None


def _median(xs):
    xs = sorted(x for x in xs if x)
    return xs[len(xs) // 2] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fill-frac", type=float, default=0.90)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-ctx", type=int, default=0)
    a = ap.parse_args()

    max_ctx = a.max_ctx
    if not max_ctx:
        try:
            with engine_http.urlopen(a.url + "/v1/models", timeout=10) as r:
                d = json.load(r)
            max_ctx = int(d["data"][0].get("max_model_len") or 0)
        except Exception:
            max_ctx = 0
    if not max_ctx:
        print("[compaction] could not determine max_ctx — pass MAX_CTX=", file=sys.stderr)
        return 2
    target = int(max_ctx * a.fill_frac)

    # ⚠️ PREFLIGHT. Compaction prefills a NEW sequence while the old blocks are
    # still held, so it peaks above steady state at the same depth. verify-stress
    # measured this class fillable to 91% of n_ctx with 669 MB free — under its own
    # 1024 MB margin. Above ~0.92 this stops measuring compaction and starts
    # measuring the OOM boundary; say so rather than hand back a mystery failure.
    if a.fill_frac > 0.92:
        print(f"[compaction] ⚠️  fill_frac={a.fill_frac} is above the ~0.92 that "
              f"verify-stress found addressable with any VRAM margin.")
        print( "[compaction]     A failure here is most likely the ceiling, not a "
               "compaction defect. Proceeding because you asked explicitly.")

    if not _metrics(a.url):
        print("[compaction] ⚠️  /metrics unavailable — falling back to CLIENT timing "
              "only. On a reasoning model that is the weaker instrument (#1096).")

    print(f"[compaction] max_ctx={max_ctx:,} target={target:,} ({a.fill_frac:.0%})")

    filler = ("The scheduler batches requests by token budget while the allocator pages "
              "expert weights across the link and the drafter proposes tokens ahead. ") * 60
    # ── CORRECTNESS CANARIES ──────────────────────────────────────────────
    # Throughput is only half of "behaves correctly after a compaction". Two
    # facts are planted in the pre-compaction history:
    #   KEPT    — carried INTO the summary. The model SHOULD still recall it;
    #             failing to means the compacted context is not being used.
    #   DROPPED — left behind in the discarded history and NOT in the summary.
    #             The model should NOT be able to state it. If it does, stale KV
    #             is being reused across the compaction boundary — which would be
    #             actual corruption, not slowness, and a far more serious finding
    #             than the reported TPS drop.
    CANARY_KEPT = "the deployment codename is SILVER-HARRIER-41"
    CANARY_DROPPED = "the fallback shard identifier is TEAL-PANGOLIN-77"
    msgs = [{"role": "system", "content": "You are a senior systems engineer. Be concise."}]
    msgs.append({"role": "user", "content": f"Remember two facts: {CANARY_KEPT}, "
                                            f"and {CANARY_DROPPED}."})
    msgs.append({"role": "assistant", "content": "Noted both facts."})

    print(f"=== 1. GROW to ~{target:,} ctx ===")
    ctx = turn = 0
    while ctx < target:
        msgs.append({"role": "user",
                     "content": f"[chunk {turn}] Note this material.\n{filler}"})
        try:
            r = _gen(a.url, a.model, msgs, max_tokens=32)
        except Exception as e:
            print(f"  GROW FAILED at ctx~{ctx:,}: {e}")
            print("  (at a high fill_frac this is most likely the context ceiling)")
            return 2
        msgs.append({"role": "assistant", "content": r["text"] or "noted"})
        if r["prompt"] <= ctx and turn > 3:
            print(f"  context stopped growing at {ctx:,} — stopping")
            break
        ctx = r["prompt"]
        turn += 1
        if turn % 5 == 0:
            print(f"  ctx={ctx:,} ({ctx/max_ctx:.0%})", flush=True)
        if turn > 400:
            print("  turn cap reached")
            break
    print(f"  reached ctx={ctx:,} ({ctx/max_ctx:.0%}) in {turn} turns")

    probe = [{"role": "user", "content": "Summarise the last chunk in one line."}]
    print(f"=== 2. BEFORE compaction ({a.reps} reps at depth) ===")
    before, err = _phase(a.url, a.model, msgs + probe, a.reps, "before",
                         continue_session=True)
    if err:
        print("VERDICT: crashed BEFORE compaction — not a compaction finding")
        return 2

    print("=== 3. COMPACT ===")
    compacted = [msgs[0],
                 {"role": "user", "content": "Earlier context is summarised: a GPU "
                  "inference scheduler, a paged KV allocator, expert offload, and "
                  f"speculative decoding. Also, {CANARY_KEPT}."},
                 {"role": "assistant", "content": "Understood — summary noted."}]
    compacted += msgs[-2:]
    print(f"  history {len(msgs)} msgs -> {len(compacted)} msgs")

    print(f"=== 4. AFTER compaction ({a.reps} reps) ===")
    after, err = _phase(a.url, a.model, compacted + probe, a.reps, "after",
                        continue_session=True)
    if err:
        print("VERDICT: CRASHED AFTER COMPACTION — this is the reported shape (#1052)")
        return 2

    # ── correctness, not just speed ──────────────────────────────────────
    print("=== 5. CORRECTNESS after compaction ===")
    try:
        q_kept = _gen(a.url, a.model, compacted + [
            {"role": "user", "content": "What is the deployment codename? Answer with just the codename."}])
        q_drop = _gen(a.url, a.model, compacted + [
            {"role": "user", "content": "What is the fallback shard identifier? If it is not in "
                                        "our conversation, reply exactly: NOT PRESENT."}])
        kept_ok = "SILVER-HARRIER-41" in (q_kept["text"] or "").upper()
        leaked  = "TEAL-PANGOLIN-77" in (q_drop["text"] or "").upper()
        print(f"  summary-carried fact recalled : {'YES' if kept_ok else 'NO'}"
              f"   {'' if kept_ok else '⚠️  compacted context is not being used'}")
        print(f"  dropped fact leaked           : {'YES' if leaked else 'no'}"
              f"   {'⚠️⚠️  STALE KV ACROSS THE COMPACTION BOUNDARY — corruption, not slowness' if leaked else ''}")
        print(f"    kept-probe reply : {(q_kept['text'] or '')[:90]!r}")
        print(f"    drop-probe reply : {(q_drop['text'] or '')[:90]!r}")
    except Exception as e:
        print(f"  correctness probe FAILED — {e}")

    # a reply that is merely SHORT is also a correctness signal
    thin = sum(1 for r in after if len((r["text"] or "").strip()) < 40)
    if thin:
        print(f"  ⚠️  {thin}/{len(after)} post-compaction replies were under 40 chars — "
              f"check for truncation/degeneration, not just throughput")

    bc, ac = _median([r["client_tps"] for r in before]), _median([r["client_tps"] for r in after])
    be, ae = _median([r["engine_tps"] for r in before]), _median([r["engine_tps"] for r in after])
    print("\n=== VERDICT ===")
    if bc and ac:
        print(f"  client TPS  before={bc:6.1f}  after={ac:6.1f}  ratio={ac/bc:.2f}x")
    if be and ae:
        print(f"  engine TPS  before={be:6.1f}  after={ae:6.1f}  ratio={ae/be:.2f}x")
        print("  ⚠️  POST-COMPACTION DEGRADATION (engine-side)" if ae < be * 0.6
              else "  no post-compaction degradation (engine-side)")
    else:
        print("  engine-side unavailable — client numbers only, treat as indicative")
    print(f"  preemptions  before={sum(r['preemptions'] for r in before):.0f} "
          f"after={sum(r['preemptions'] for r in after):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
