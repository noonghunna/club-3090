#!/usr/bin/env python3
"""concurrency_probe.py — one-cell measurement + matrix planner + card.

WHY THIS EXISTS
  scripts/concurrency-probe.sh used to embed this as a heredoc. The cell
  measurement (streamed TTFT vs decode, admitted vs N, NaN on errored
  aggregates) stays the same; the planner/card are the --sweep driver.

USAGE (normally invoked by concurrency-probe.sh)
  env URL=… MODEL=… CONCURRENCY=N python3 concurrency_probe.py
      one cell; prints the per-round table, verdict, and RESULT line
  python3 concurrency_probe.py --plan
      human-readable matrix plan (no server)
  python3 concurrency_probe.py --plan-tsv
      action<TAB>ctx<TAB>n<TAB>reason  (driver loop)
  python3 concurrency_probe.py --card < sweep.json
      club-3090 concurrency card on stdout
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request

# ~0.23 tok/char was calibrated on the BLOCK below (same as the heredoc).
TOK_PER_CHAR = 0.23
BLOCK = (
    "This section describes the history of computing in detail. Transistors "
    "were invented in 1947 at Bell Labs. The integrated circuit came a decade "
    "later. Microprocessors emerged in the 1970s and changed the world. "
)
UNIQUE_MIN_DEFAULT = 256
SHARE_FRAC_DEFAULT = 0.75
DEFAULT_CTX = (1024, 4096, 8192, 16384, 32768)
DEFAULT_N = (1, 2, 4, 8, 16, 32)


def _env(name, default=""):
    return os.environ.get(name, default)


def parse_size(raw):
    """'1k'/'16K'/'32768' -> token count. 1k = 1024."""
    s = str(raw).strip().lower().replace(",", "")
    if not s:
        raise ValueError("empty size")
    mult = 1
    if s.endswith("k"):
        mult = 1024
        s = s[:-1]
    elif s.endswith("m"):
        mult = 1024 * 1024
        s = s[:-1]
    return int(float(s) * mult)


def parse_size_list(raw, default=()):
    if raw is None or str(raw).strip() == "":
        return list(default)
    parts = re.split(r"[\s,]+", str(raw).strip())
    return [parse_size(p) for p in parts if p]


def parse_int_list(raw, default=()):
    if raw is None or str(raw).strip() == "":
        return list(default)
    parts = re.split(r"[\s,]+", str(raw).strip())
    return [int(p) for p in parts if p]


def fmt_ctx(n):
    # 262144 is written 262K in this repo (decimal thousands), not 256Ki.
    if n >= 100000:
        return f"{n // 1000}K"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def fmt_ttft(seconds):
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    if seconds >= 10:
        return f"{seconds:.0f}s"
    if seconds >= 0.1:
        return f"{seconds:.1f}s"
    return f"{seconds * 1000:.0f}ms"


def parse_kv_tokens_text(txt):
    """vLLM boot: 'GPU KV cache size: 210,000 tokens' (optional leading ~)."""
    if not txt:
        return None
    matches = list(
        re.finditer(r"GPU KV cache size:\s*~?([\d,]+)\s*tokens", txt, re.I)
    )
    if not matches:
        return None
    return int(matches[-1].group(1).replace(",", ""))


def detect_kv_tokens(container):
    if not container:
        return None
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "2500", container],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
        txt = (out.stdout or "") + (out.stderr or "")
    except Exception:
        return None
    return parse_kv_tokens_text(txt)


def plan_matrix(
    ctxs,
    ns,
    *,
    gen_tokens=256,
    kv_tokens=None,
    served_slots=None,
    served_max_len=None,
    n_max=None,
    ctx_max=None,
):
    """Return {ctx_list, n_list, cells, notes}. Plan-time clips only.

    A cell is skip (not run) when ctx exceeds served max-len / CTX_MAX,
    N exceeds served slots / N_MAX, or N*(ctx+gen) exceeds KV_TOKENS.
    Early-stop (fail N=2 ⇒ skip N=4) is applied at runtime, not here.
    """
    notes = []
    ctxs = [c for c in ctxs if c > 0]
    ns = sorted({n for n in ns if n > 0})
    if ctx_max:
        dropped = [c for c in ctxs if c > ctx_max]
        ctxs = [c for c in ctxs if c <= ctx_max]
        if dropped:
            notes.append(
                f"ctx > CTX_MAX={fmt_ctx(ctx_max)} dropped: "
                + " ".join(fmt_ctx(c) for c in dropped)
            )
    if served_max_len:
        dropped = [c for c in ctxs if c > served_max_len]
        ctxs = [c for c in ctxs if c <= served_max_len]
        if dropped:
            notes.append(
                f"ctx > served max-model-len {fmt_ctx(served_max_len)} dropped: "
                + " ".join(fmt_ctx(c) for c in dropped)
            )
    n_cap = None
    if n_max:
        n_cap = n_max if n_cap is None else min(n_cap, n_max)
    if served_slots:
        n_cap = served_slots if n_cap is None else min(n_cap, served_slots)
    if n_cap is not None:
        dropped_n = [n for n in ns if n > n_cap]
        ns_kept = [n for n in ns if n <= n_cap]
        if dropped_n:
            why = []
            if served_slots and n_cap == served_slots:
                why.append(f"served slots={served_slots}")
            if n_max and n_cap == n_max:
                why.append(f"N_MAX={n_max}")
            notes.append(
                "N>"
                + str(n_cap)
                + " dropped ("
                + ", ".join(why or ["cap"])
                + "): "
                + " ".join(str(n) for n in dropped_n)
            )
            ns = ns_kept
    cells = []
    for ctx in ctxs:
        for n in ns:
            reason = ""
            action = "run"
            if kv_tokens:
                need = n * (ctx + gen_tokens)
                if need > kv_tokens:
                    action = "skip"
                    reason = f"N*(ctx+gen)={need} > KV_TOKENS={kv_tokens}"
            cells.append(
                {"ctx": ctx, "n": n, "action": action, "reason": reason}
            )
    if kv_tokens:
        notes.append(f"KV_TOKENS={kv_tokens}")
    else:
        notes.append("KV_TOKENS unknown — not clipping on pool size")
    return {
        "ctx_list": ctxs,
        "n_list": ns,
        "cells": cells,
        "notes": notes,
        "gen_tokens": gen_tokens,
    }


def format_plan_text(plan, extra_header=""):
    lines = []
    if extra_header:
        lines.append(extra_header)
    by_ctx = {}
    for cell in plan["cells"]:
        by_ctx.setdefault(cell["ctx"], []).append(cell)
    lines.append("  run:")
    if not by_ctx:
        lines.append("    (empty — nothing left after clips)")
        for note in plan["notes"]:
            lines.append(f"  note: {note}")
        return "\n".join(lines)
    for ctx in plan["ctx_list"]:
        cells = by_ctx.get(ctx, [])
        run = [str(c["n"]) for c in cells if c["action"] == "run"]
        skip = [c for c in cells if c["action"] == "skip"]
        label = f"{fmt_ctx(ctx):>5}"
        row = f"    {label}:  {' '.join(run) if run else '—'}"
        if skip:
            reasons = sorted({c["reason"] for c in skip if c["reason"]})
            skip_ns = " ".join(str(c["n"]) for c in skip)
            why = reasons[0] if reasons else "clipped"
            row += f"     (skip {skip_ns} — {why})"
        lines.append(row)
    for note in plan["notes"]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def format_plan_tsv(plan):
    out = []
    for cell in plan["cells"]:
        out.append(
            f"{cell['action']}\t{cell['ctx']}\t{cell['n']}\t{cell.get('reason', '')}"
        )
    return "\n".join(out)


def prompt_text(stream, rnd, ptok, cache, share_frac, unique_min):
    """Build a ~ptok prompt. cache=cold salts FIRST (VALIDATE / fit).

    cache=shared puts an identical prefix first and the per-stream salt at
    the tail so automatic prefix caching can hit. Salt is per-stream, not
    per-round, so round 2+ can warm.
    """
    reps = int(ptok / (len(BLOCK) * TOK_PER_CHAR)) + 1
    body = BLOCK * reps
    task = "\nWrite a detailed multi-paragraph summary."
    if cache != "shared":
        # unique salt first — no prefix-cache free ride (the original probe)
        return f"[probe s{stream} r{rnd}] " + body + task

    unique_toks = max(int(unique_min), int(ptok * (1.0 - share_frac)))
    if unique_toks >= ptok:
        unique_toks = max(int(unique_min), ptok // 4)
    unique_toks = min(unique_toks, max(1, ptok - 1))
    shared_toks = max(0, ptok - unique_toks)
    shared_chars = max(0, int(shared_toks / TOK_PER_CHAR))
    unique_chars = max(1, int(unique_toks / TOK_PER_CHAR))
    shared = (BLOCK * (shared_chars // len(BLOCK) + 2))[:shared_chars]
    unique = (BLOCK * (unique_chars // len(BLOCK) + 2))[:unique_chars]
    return shared + f" [probe s{stream}] " + unique + task


def cached_from_usage(u):
    if not u:
        return 0
    ptd = u.get("prompt_tokens_details") or {}
    if isinstance(ptd, dict) and ptd.get("cached_tokens") is not None:
        try:
            return int(ptd["cached_tokens"])
        except (TypeError, ValueError):
            return 0
    if u.get("cached_tokens") is not None:
        try:
            return int(u["cached_tokens"])
        except (TypeError, ValueError):
            return 0
    return 0


def pct(xs, q):
    # p50 alone describes NO ACTUAL USER once requests queue: with Running:9 /
    # Waiting:55 the admitted streams see a short TTFT and the queued ones wait
    # several batches, so the median lands between the two populations. The tail
    # is also why `aggregate` (divided by the round WALL, set by the slowest
    # stream) comes in below N x median per-stream.
    if not xs:
        return 0.0
    ss = sorted(xs)
    return ss[min(len(ss) - 1, int(q * len(ss)))]


def engine_stats(container):
    # vLLM logs "Running: N reqs, Waiting: M reqs" and a prefix-cache hit rate.
    # Running is the concurrency the scheduler ACTUALLY admits, which can be far
    # below max_num_seqs when --max-num-batched-tokens caps the per-step budget:
    # a sweep rung labelled N=64 may only ever run ~10 wide. Reporting the label
    # without this reads as a concurrency measurement when it is a queue-depth one.
    if not container:
        return (None, None, None)
    try:
        out = subprocess.run(
            ["docker", "logs", "--tail", "400", container],
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
        txt = (out.stdout or "") + (out.stderr or "")
    except Exception:
        return (None, None, None)
    run = wait = hit = None
    for m in re.finditer(
        r"Running:\s*(\d+)\s*reqs?,\s*Waiting:\s*(\d+)\s*reqs?", txt
    ):
        run, wait = int(m.group(1)), int(m.group(2))
    for m in re.finditer(r"[Pp]refix cache hit rate:\s*([0-9.]+)", txt):
        hit = float(m.group(1))
    return (run, wait, hit)


def vram_used_mb():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return sum(int(x) for x in out.split())
    except Exception:
        return -1


def parse_result_line(line):
    """Parse 'RESULT k=v …' into a dict. Missing keys stay absent."""
    if not line:
        return {}
    s = line.strip()
    if s.startswith("RESULT "):
        s = s[7:]
    out = {}
    for part in s.split():
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _fmt_num(v, digits=2):
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def format_result(fields):
    order = [
        "N",
        "ctx",
        "clean",
        "pass",
        "mps_tps",
        "agg_tps",
        "retention",
        "leak",
        "vram_peak",
        "floor_ok",
        "ttft_ms",
        "pf_tps",
        "ttft_p95_ms",
        "tps_p05",
        "running",
        "waiting",
        "prefix_hit",
        "cache",
        "cached_toks",
        "pf_agg",
    ]
    parts = []
    seen = set()
    for k in order:
        if k in fields:
            parts.append(f"{k}={fields[k]}")
            seen.add(k)
    for k, v in fields.items():
        if k not in seen:
            parts.append(f"{k}={v}")
    return "RESULT " + " ".join(parts)


def run_probe():
    URL = os.environ["URL"]
    MODEL = os.environ["MODEL"]
    N = int(os.environ["CONCURRENCY"])
    ROUNDS = int(os.environ["ROUNDS"])
    PTOK = int(os.environ["PROMPT_TOKENS"])
    GTOK = int(os.environ["GEN_TOKENS"])
    GROWTH = int(os.environ["VRAM_GROWTH_MB"])
    REQ_TIMEOUT = float(os.environ["REQ_TIMEOUT"])
    TPS_FLOOR = float(os.environ["TPS_FLOOR"])
    RETENTION_MIN = float(os.environ["RETENTION_MIN"])
    VALIDATE = os.environ.get("VALIDATE", "0") == "1"
    CONTAINER = os.environ.get("CONTAINER") or ""
    CACHE = (os.environ.get("CACHE") or "cold").strip().lower()
    if CACHE not in ("cold", "shared"):
        CACHE = "cold"
    if VALIDATE:
        CACHE = "cold"
    SHARE_FRAC = float(os.environ.get("SHARE_FRAC") or SHARE_FRAC_DEFAULT)
    UNIQUE_MIN = int(os.environ.get("UNIQUE_MIN") or UNIQUE_MIN_DEFAULT)

    def prompt(stream, rnd):
        return prompt_text(stream, rnd, PTOK, CACHE, SHARE_FRAC, UNIQUE_MIN)

    def one(stream, rnd):
        # Streamed so we can separate TTFT (prefill) from decode and report a real
        # per-stream DECODE tok/s — the signal the bandwidth-knee sweep needs, and
        # the only honest throughput number at deep context (where prefill dominates
        # wall time). completion_tokens comes from the include_usage final chunk.
        body = json.dumps(
            {
                "model": MODEL,
                "max_tokens": GTOK,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": prompt(stream, rnd)}],
            }
        ).encode()
        req = urllib.request.Request(
            URL + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        t_first = None
        t_last = None
        toks = 0
        chunks = 0
        cached = 0
        try:
            resp = urllib.request.urlopen(req, timeout=REQ_TIMEOUT)
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)
                except Exception:
                    continue
                u = ch.get("usage")
                if u:
                    if u.get("completion_tokens"):
                        toks = u["completion_tokens"]
                    c = cached_from_usage(u)
                    if c:
                        cached = c
                ch_choices = ch.get("choices") or []
                if ch_choices:
                    d = ch_choices[0].get("delta") or {}
                    if (
                        d.get("content")
                        or d.get("reasoning_content")
                        or d.get("reasoning")
                    ):
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        chunks += 1
            dt = time.time() - t0
            if not toks:
                toks = chunks  # fall back to chunk count if usage absent
            decode_dt = (
                (t_last - t_first)
                if (t_first and t_last and t_last > t_first)
                else 0.0
            )
            tps = (toks / decode_dt) if (toks > 1 and decode_dt > 0) else 0.0
            ttft = (t_first - t0) if t_first else None
            return {
                "ok": toks > 0,
                "toks": toks,
                "silent": toks == 0,
                "err": None,
                "dt": dt,
                "ttft": ttft,
                "tps": tps,
                "cached": cached,
            }
        except Exception as e:
            return {
                "ok": False,
                "toks": 0,
                "silent": False,
                "err": str(e)[:80],
                "dt": time.time() - t0,
                "ttft": None,
                "tps": 0.0,
                "cached": 0,
            }

    print(
        f"\n{'round':>5} {'done':>7} {'silent':>7} {'errors':>7} {'vram_MB':>8} "
        f"{'agg_t/s':>8} {'per-strm':>9} {'ttft_ms':>8} {'pf_t/s':>7}"
        f" {'ttft_p95':>9} {'tps_p05':>8} {'run/wait':>10} {'pfxhit':>6}"
    )
    vram0 = vram_used_mb()
    vram_by_round = []
    mtps_by_round = []
    agg_by_round = []
    ttft_by_round = []
    pf_by_round = []
    pf_agg_by_round = []
    cached_by_round = []
    bad = 0
    err_rounds = 0
    ttft_p95_by_round = []
    tps_p05_by_round = []
    run_by_round = []
    wait_by_round = []
    hit_by_round = []
    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=N) as ex:
            res = list(ex.map(lambda s: one(s, rnd), range(N)))
        wall = time.time() - t0
        done = sum(1 for r in res if r["ok"])
        silent = sum(1 for r in res if r["silent"])
        errs = sum(1 for r in res if r["err"])
        v = vram_used_mb()
        vram_by_round.append(v)
        agg = sum(r["toks"] for r in res) / wall if wall else 0
        tps_ok = [r["tps"] for r in res if r["ok"] and r["tps"] > 0]
        mtps = statistics.median(tps_ok) if tps_ok else 0.0
        mtps_by_round.append(mtps)
        agg_by_round.append(agg)
        ttfts = [r["ttft"] for r in res if r["ok"] and r["ttft"]]
        ttft_med = statistics.median(ttfts) if ttfts else 0.0
        cached_ok = [r["cached"] for r in res if r["ok"]]
        cached_med = statistics.median(cached_ok) if cached_ok else 0.0
        uncached = max(0.0, PTOK - cached_med)
        pf = (uncached / ttft_med) if ttft_med > 0 else 0.0
        last_ttft = max(ttfts) if ttfts else 0.0
        uncached_sum = sum(max(0, PTOK - r["cached"]) for r in res if r["ok"])
        pf_agg = (uncached_sum / last_ttft) if last_ttft > 0 else 0.0
        ttft_by_round.append(ttft_med)
        pf_by_round.append(pf)
        pf_agg_by_round.append(pf_agg)
        cached_by_round.append(cached_med)
        ttft_p95 = pct(ttfts, 0.95)
        tps_p05 = pct(tps_ok, 0.05)
        ttft_p95_by_round.append(ttft_p95)
        tps_p05_by_round.append(tps_p05)
        run, wait, hit = engine_stats(CONTAINER)
        run_by_round.append(run)
        wait_by_round.append(wait)
        hit_by_round.append(hit)
        print(
            f"{rnd:>5} {done:>4}/{N:<2} {silent:>7} {errs:>7} {v:>8} "
            f"{agg:>8.1f} {mtps:>9.1f} {ttft_med * 1000:>8.0f} {pf:>7.0f}"
            f" {ttft_p95 * 1000:>9.0f} {tps_p05:>8.1f} "
            f"{('-' if run is None else run):>4}/"
            f"{('-' if wait is None else wait):<5}"
            f" {('-' if hit is None else f'{hit:.1f}'):>6}"
        )
        if done < N or silent or errs:
            bad += 1
        if errs:
            err_rounds += 1

    # VRAM: leak = post-warm growth (round 2 baseline), NOT the expected cold->warm fill.
    warm_i = 1 if ROUNDS >= 3 else 0
    warm = vram_by_round[warm_i]
    pool_fill = warm - vram0 if vram0 >= 0 else -1
    leak = vram_by_round[-1] - warm if vram0 >= 0 else -1
    vram_peak = max(vram_by_round) if vram_by_round else -1
    report_tps = mtps_by_round[-1] if mtps_by_round else 0.0
    report_agg = agg_by_round[-1] if agg_by_round else 0.0
    # An aggregate computed over a round that had errors is not a throughput number:
    # failed streams contribute toks=0 while their prefill still counts in the wall,
    # so the figure is (survivors' tokens)/(full round wall) and reads 5-10x low. It
    # has been quoted out of context as a measured regression before. Report NaN so a
    # broken arm cannot be mistaken for a slow one.
    if err_rounds:
        report_agg = float("nan")
        report_tps = float("nan")
    warm_tps = mtps_by_round[1:] if ROUNDS >= 4 else mtps_by_round
    if len(warm_tps) >= 3 and warm_tps[0] > 0:
        early = statistics.median(warm_tps[:2])
        late = statistics.median(warm_tps[-2:])
        retention = (late / early) if early > 0 else 0.0
    else:
        retention = 1.0  # too few post-warmup rounds to judge

    clean_fit = (bad == 0) and (0 <= leak <= GROWTH)
    floor_ok = (report_tps >= TPS_FLOOR) if TPS_FLOOR > 0 else True
    retention_ok = (retention >= RETENTION_MIN) if ROUNDS >= 3 else True
    PASS = clean_fit and floor_ok and (retention_ok if VALIDATE else True)

    print(f"\n=== verdict (N={N}) ===")
    print(
        f"  VRAM: cold {vram0} -> warm {warm} MB (pool fill {pool_fill} MB, expected) "
        f"-> final {vram_by_round[-1]} MB (post-warm growth {leak} MB / {GROWTH})  "
        f"peak {vram_peak} MB"
    )
    print(
        f"  per-stream decode: {report_tps:.1f} tok/s (steady) · aggregate "
        f"{report_agg:.1f} tok/s "
        f"({N} streams) · retention {retention * 100:.1f}% "
        f"(min {RETENTION_MIN * 100:.0f}%)"
        + (f" · floor {TPS_FLOOR:.0f}" if TPS_FLOOR > 0 else " · floor off")
    )
    steady_ttft = ttft_by_round[-1] if ttft_by_round else 0.0
    steady_pf = pf_by_round[-1] if pf_by_round else 0.0
    steady_pf_agg = pf_agg_by_round[-1] if pf_agg_by_round else 0.0
    steady_cached = cached_by_round[-1] if cached_by_round else 0.0
    print(
        f"  concurrent prefill: steady TTFT {steady_ttft * 1000:.0f} ms "
        f"(median of {N} streams) "
        f"· ~{steady_pf:.0f} tok/s/stream uncached @ {PTOK}-tok prompts"
        f" · agg-prefill ~{steady_pf_agg:.0f} tok/s"
        + (f" · cached ~{steady_cached:.0f} tok" if steady_cached else "")
    )
    s_p95 = ttft_p95_by_round[-1] if ttft_p95_by_round else 0.0
    s_p05 = tps_p05_by_round[-1] if tps_p05_by_round else 0.0
    s_run = run_by_round[-1] if run_by_round else None
    s_wait = wait_by_round[-1] if wait_by_round else None
    s_hit = hit_by_round[-1] if hit_by_round else None
    print(
        f"  tail: TTFT p95 {s_p95 * 1000:.0f} ms (vs p50 {steady_ttft * 1000:.0f}) "
        f"· slowest-decile stream {s_p05:.1f} tok/s (vs median {report_tps:.1f})"
    )
    if s_run is not None:
        admitted = (
            ""
            if s_run >= N
            else f"  ** engine admitted {s_run} of {N} requested — the rest QUEUED **"
        )
        print(
            f"  engine: running {s_run} · waiting {s_wait}"
            + (f" · prefix-cache hit {s_hit:.1f}%" if s_hit is not None else "")
            + admitted
        )
    flags = []
    if not clean_fit:
        flags.append("fit")
    if TPS_FLOOR > 0 and not floor_ok:
        flags.append("tps-floor")
    if VALIDATE and not retention_ok:
        flags.append("retention")
    print(
        f"  concurrency {N} @ ~{PTOK} tok: "
        f"{'PASS — sustained clean' if PASS else 'FAIL — ' + ','.join(flags)}"
    )
    if PASS:
        print(
            f"  envelope row: max_num_seqs: {N}  validated: {{ concurrency_soak: "
            f"'{N} @ ~{PTOK // 1000}K, {report_tps:.0f} tok/s/stream, {leak} MB post-warm', "
            f"vram_peak_gb: {vram_peak / 1024:.1f} }}"
        )
    fields = {
        "N": N,
        "ctx": PTOK,
        "clean": int(clean_fit),
        "pass": int(PASS),
        "mps_tps": _fmt_num(report_tps, 2),
        "agg_tps": _fmt_num(report_agg, 2),
        "retention": f"{retention:.3f}",
        "leak": leak,
        "vram_peak": vram_peak,
        "floor_ok": int(floor_ok),
        "ttft_ms": f"{steady_ttft * 1000:.0f}",
        "pf_tps": _fmt_num(steady_pf, 1),
        "ttft_p95_ms": f"{s_p95 * 1000:.0f}",
        "tps_p05": _fmt_num(s_p05, 2),
        "running": "-" if s_run is None else s_run,
        "waiting": "-" if s_wait is None else s_wait,
        "prefix_hit": "-" if s_hit is None else f"{s_hit:.1f}",
        "cache": CACHE,
        "cached_toks": f"{steady_cached:.0f}",
        "pf_agg": _fmt_num(steady_pf_agg, 1),
    }
    print(format_result(fields))
    raise SystemExit(0 if PASS else 1)


def _fnum(v):
    if v is None or v == "" or v == "-":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def render_card(record):
    """Unicode card. record is the JSON object written by --sweep."""
    model = record.get("model") or "?"
    spec = record.get("spec") or "spec off"
    gpus = record.get("gpus") or "?"
    kv = record.get("kv_tokens")
    slots = record.get("slots")
    gen = record.get("gen_tokens") or 256
    cache = record.get("cache") or "cold"
    slug = record.get("slug") or ""
    cmd = record.get("command") or "bash scripts/concurrency-probe.sh --sweep"
    rows = list(record.get("rows") or [])

    kv_s = f"KV {kv}" if kv else "KV ?"
    if isinstance(kv, int) and kv >= 1000:
        kv_s = f"KV {kv:,}".replace(",", "")
        if kv >= 1000:
            kv_s = f"KV {kv // 1000}k" if kv % 1000 == 0 else f"KV {kv}"
    slot_s = f"slots {slots}" if slots not in (None, "") else "slots ?"
    ident = model if not slug else f"{model} · {slug}"

    # vs 1-stream + star per ctx (largest clean pass N)
    by_ctx = {}
    for r in rows:
        by_ctx.setdefault(int(r["ctx"]), []).append(r)
    n1_agg = {}
    star_n = {}
    for ctx, rs in by_ctx.items():
        for r in rs:
            if int(r.get("n", 0)) == 1 and r.get("skip") in (None, "", False):
                n1_agg[ctx] = _fnum(r.get("agg"))
        # ★ / SWEET = throughput peak (highest agg among clean+pass), not the
        # largest N that merely stayed clean with the floor off.
        best_n, best_agg = None, None
        for r in rs:
            if r.get("skip") not in (None, "", False):
                continue
            if int(r.get("clean") or 0) != 1 or int(r.get("pass") or 0) != 1:
                continue
            agg = _fnum(r.get("agg"))
            if agg is None:
                continue
            if best_agg is None or agg > best_agg:
                best_agg, best_n = agg, int(r["n"])
        if best_n is not None:
            star_n[ctx] = best_n

    inner_w = 63
    bar = "─" * (inner_w - 2)

    def row_line(s=""):
        return "│  " + s.ljust(inner_w - 4) + "  │"

    out = []
    out.append("┌" + "─" * inner_w + "┐")
    out.append(row_line())
    out.append(row_line("club-3090"))
    out.append(row_line("concurrency card"))
    out.append(row_line())
    out.append(row_line(f"{ident} · {spec}"[: inner_w - 4]))
    out.append(row_line(f"{gpus} · {kv_s} · {slot_s}"))
    if by_ctx:
        ctx_span = (
            f"{fmt_ctx(min(by_ctx))}–{fmt_ctx(max(by_ctx))} in / {gen} out · cache {cache}"
        )
    else:
        ctx_span = f"{gen} out · cache {cache}"
    out.append(row_line(ctx_span))
    out.append(row_line())
    out.append(
        row_line(
            f"{'ctx':<6}{'N':>3}  {'strm':>6}  {'agg':>5}  {'vs 1-stream':<12} {'TTFT':>6}  {'VRAM':>6}"
        )
    )
    out.append(row_line(bar))

    sweet = None
    prefer_ctx = 16384 if 16384 in by_ctx else (max(star_n) if star_n else None)

    first_ctx = True
    for ctx in sorted(by_ctx):
        if not first_ctx:
            out.append(row_line())
        first_ctx = False
        for r in sorted(by_ctx[ctx], key=lambda x: int(x["n"])):
            n = int(r["n"])
            skip = r.get("skip")
            star = " ★" if star_n.get(ctx) == n and not skip else "  "
            if skip:
                reason = str(skip)
                if len(reason) > 18:
                    reason = reason[:18]
                line = (
                    f"{fmt_ctx(ctx):<6}{n:>3}  {'—':>6}  {'—':>5}  "
                    f"{'—':<12} {'—':>6}  {reason}"
                )
                out.append(row_line(line))
                continue
            strm = _fnum(r.get("strm"))
            agg = _fnum(r.get("agg"))
            ttft_s = _fnum(r.get("ttft_s"))
            vram = _fnum(r.get("vram_gb"))
            base = n1_agg.get(ctx)
            vs = "—"
            if agg is not None and base and base > 0:
                vs = f"{agg / base:.2f}×"
            strm_s = f"{strm:.1f}" if strm is not None else "—"
            agg_s = f"{agg:.0f}" if agg is not None else "—"
            vram_s = f"{vram:.1f}G" if vram is not None else "—"
            line = (
                f"{fmt_ctx(ctx):<6}{n:>3}  {strm_s:>6}  {agg_s:>5}{star} "
                f"{vs:<11} {fmt_ttft(ttft_s):>6}  {vram_s:>6}"
            )
            out.append(row_line(line))
            if prefer_ctx == ctx and star_n.get(ctx) == n:
                sweet = (ctx, n, strm, agg, ttft_s)

    out.append(row_line())
    if sweet:
        ctx, n, strm, agg, ttft_s = sweet
        strm_s = f"{strm:.0f}" if strm is not None else "?"
        agg_s = f"{agg:.0f}" if agg is not None else "?"
        out.append(
            row_line(
                f"SWEET  N={n} @ {fmt_ctx(ctx)} · {strm_s} tok/s/user · "
                f"{agg_s} tok/s rig · TTFT {fmt_ttft(ttft_s)}"
            )
        )
        out.append(row_line())
    out.append(row_line(cmd))
    out.append(row_line())
    out.append("└" + "─" * inner_w + "┘")
    return "\n".join(out)


def _clean_rows(rows):
    out = []
    for r in rows or []:
        if r.get("skip") not in (None, "", False):
            continue
        if int(r.get("clean") or 0) != 1 or int(r.get("pass") or 0) != 1:
            continue
        if _fnum(r.get("agg")) is None:
            continue
        out.append(r)
    return out


def _best_row(rows, ctx=None):
    cands = _clean_rows(rows)
    if ctx is not None:
        cands = [r for r in cands if int(r["ctx"]) == int(ctx)]
    if not cands:
        return None
    return max(cands, key=lambda r: (_fnum(r.get("agg")) or -1, int(r["n"])))


def _row_summary(r):
    if not r:
        return None
    return {
        "n": int(r["n"]),
        "ctx": int(r["ctx"]),
        "agg": _fnum(r.get("agg")),
        "strm": _fnum(r.get("strm")),
        "max_num_seqs": int(r["n"]),
        "max_model_len": int(r["ctx"]),
    }


def recommend_compose(record):
    """Pick compose knobs from the matrix.

    Two picks, because peak aggregate is almost always the short-prompt cell
    while shipped composes keep full max-model-len:
      full_ctx — peak agg at 16K (agent), else the deepest clean ctx
      max_agg  — global peak agg cell
    """
    rows = record.get("rows") or []
    clean = _clean_rows(rows)
    served = record.get("served_max_len")
    try:
        served = int(served) if served not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        served = None
    ctxs = sorted({int(r["ctx"]) for r in clean})
    agent_ctx = 16384 if 16384 in ctxs else (ctxs[-1] if ctxs else None)
    return {
        "max_agg": _row_summary(_best_row(rows)),
        "full_ctx": _row_summary(_best_row(rows, agent_ctx) if agent_ctx else None),
        "served_max_len": served,
        "engine": record.get("engine") or "",
        "slug": record.get("slug") or "",
    }


def format_recommend(rec):
    """Human block to print after the card. Empty if no clean cells."""
    if not rec or not rec.get("max_agg"):
        return (
            "=== recommend ===\n"
            "  no clean cells — nothing to recommend"
        )
    peak = rec["max_agg"]
    full = rec.get("full_ctx")
    served = rec.get("served_max_len")
    slug = rec.get("slug") or "<slug>"
    engine = (rec.get("engine") or "").lower()
    lines = ["=== recommend ==="]

    def knobs(n, ctx_keep, ctx_set=None):
        launch = [f"MAX_NUM_SEQS={n}"]
        if ctx_set is not None:
            launch.append(f"MAX_MODEL_LEN={ctx_set}")
        launch.append(f"bash scripts/switch.sh {slug}")
        out = [
            f"    MAX_NUM_SEQS={n}",
        ]
        if ctx_set is not None:
            out.append(f"    MAX_MODEL_LEN={ctx_set}")
        else:
            keep = fmt_ctx(ctx_keep) if ctx_keep else "compose default"
            out.append(f"    MAX_MODEL_LEN=<keep {keep}>")
        out.append(f"    {' '.join(launch)}")
        if engine in ("llamacpp", "llama.cpp", "ik_llama", "ik-llama"):
            c = ctx_set if ctx_set is not None else ctx_keep
            if c:
                out.append(f"    llama.cpp: -np {n} -c {c}")
        return out

    def cell_line(label, s):
        agg = f"{s['agg']:.0f}" if s.get("agg") is not None else "?"
        strm = f"{s['strm']:.0f}" if s.get("strm") is not None else "?"
        return (
            f"  {label}: {fmt_ctx(s['ctx'])} × N={s['n']}  →  "
            f"{agg} tok/s rig  ({strm} / stream)"
        )

    served_s = fmt_ctx(served) if served else "compose default"
    lines.append(f"  served max-model-len={served_s} (usually keep this)")
    lines.append("")

    if full:
        same = (
            peak
            and full
            and peak["n"] == full["n"]
            and peak["ctx"] == full["ctx"]
        )
        lines.append(cell_line("keep full ctx", full))
        lines.extend(knobs(full["n"], served or full["ctx"], ctx_set=None))
        if not same:
            lines.append("")
            lines.append(cell_line("max aggregate (short prompts only)", peak))
            lines.extend(knobs(peak["n"], served, ctx_set=peak["ctx"]))
            if served and peak["ctx"] < served:
                lines.append(
                    f"    do not drop MAX_MODEL_LEN unless traffic stays near {fmt_ctx(peak['ctx'])} — "
                    f"raising slots to {peak['n']} at {served_s} will not reproduce {peak['agg']:.0f} tok/s"
                )
        elif engine.startswith("vllm") or engine == "":
            lines.append(
                "    vLLM: MAX_NUM_SEQS is a cap, not a reservation; keep MAX_MODEL_LEN"
            )
    else:
        lines.append(cell_line("max aggregate", peak))
        lines.extend(knobs(peak["n"], served, ctx_set=peak["ctx"]))

    return "\n".join(lines)


def _plan_from_env():
    ctxs = parse_size_list(_env("CTX_SWEEP"), DEFAULT_CTX)
    ns = parse_int_list(_env("N_LIST") or _env("MATRIX_N"), DEFAULT_N)
    gen = int(_env("GEN_TOKENS") or 256)
    kv = _env("KV_TOKENS")
    kv_tokens = int(kv) if kv and kv not in ("0", "") else None
    slots = _env("SERVED_SLOTS")
    served_slots = int(slots) if slots else None
    ml = _env("SERVED_MAX_LEN")
    served_max_len = int(ml) if ml else None
    n_max = int(_env("N_MAX")) if _env("N_MAX") else None
    ctx_max = parse_size(_env("CTX_MAX")) if _env("CTX_MAX") else None
    return plan_matrix(
        ctxs,
        ns,
        gen_tokens=gen,
        kv_tokens=kv_tokens,
        served_slots=served_slots,
        served_max_len=served_max_len,
        n_max=n_max,
        ctx_max=ctx_max,
    )


def main(argv=None):
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--plan-tsv", action="store_true")
    p.add_argument("--card", action="store_true")
    p.add_argument("--detect-kv", action="store_true")
    args = p.parse_args(argv)

    if args.detect_kv:
        tok = detect_kv_tokens(_env("CONTAINER"))
        if tok:
            print(tok)
        return 0
    if args.plan or args.plan_tsv:
        plan = _plan_from_env()
        extra = _env("PLAN_HEADER")
        if args.plan:
            print(format_plan_text(plan, extra_header=extra))
        else:
            print(format_plan_tsv(plan))
        return 0
    if args.card:
        rec = json.load(sys.stdin)
        print(render_card(rec))
        rec["recommend"] = recommend_compose(rec)
        print()
        print(format_recommend(rec["recommend"]))
        return 0
    run_probe()
    return 0


if __name__ == "__main__":
    sys.exit(main())
