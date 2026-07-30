#!/usr/bin/env python3
"""Render offload-matrix-results.tsv.

The full row is 29 columns, so this offers VIEWS rather than one unreadable table:

  --perf   (default) throughput / latency / acceptance, with paired no-spec deltas
  --bw     bandwidth + utilisation: PCIe rx/tx, SM%, mem-controller%, CPU%, RAM read
  --cache  pool / hits / misses / evictions / VRAM peak
  --all    every column
  --md     GitHub-flavoured markdown (combine with any view)

Separate from the sweep driver on purpose: the TSV is the source of truth, so tables
can be re-rendered or re-grouped without touching the rig. Twice on 2026-07-29 data
sat in logs that the in-run grep never reached.

REPS: rows sharing an arm are collapsed to the MEDIAN with the rep count shown,
because single-boot error was +-5 tok/s that day and flipped a ranking.

Usage: python3 offload-matrix-render.py [results.tsv] [--perf|--bw|--cache|--all] [--md]
"""
import sys, csv, statistics as st
from collections import OrderedDict

args = sys.argv[1:]
path = next((a for a in args if not a.startswith("--")), "offload-matrix-results.tsv")
as_md = "--md" in args
view = ("bw" if "--bw" in args else "cache" if "--cache" in args
        else "all" if "--all" in args else "perf")

rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
if not rows:
    print("(no rows)"); raise SystemExit

NUM = ("agg","strm","ttft_ms","accept","pool_slots","misses","evicts",
       "rxpci","txpci","sm_pct","memctl_pct","cpu_pct","ram_rd_mbps","vram_peak","errors")

# The pairing key: a spec arm is only comparable to a no-spec arm measured at the SAME
# everything-else. `shape` is part of it because workload shape FLIPS THE SIGN of the
# speculation result (+76% agentic vs +2.7% prose on this stack) -- pairing a copy-heavy
# spec arm against a prose baseline would manufacture a gain that does not exist.
KEY = ("offload","N","ctx_slot","cache_mb","admit","throttle","shape")

INTCOL = {"ttft_ms","pool_slots","misses","evicts","vram_peak","errors"}

groups = OrderedDict()
for r in rows:
    groups.setdefault(r["arm"], []).append(r)

merged = []
for arm, rs in groups.items():
    out = dict(rs[0]); out["reps"] = str(len(rs))
    ok = [r for r in rs if r["status"] == "OK"]
    src = ok or rs
    for k in NUM:
        vals = []
        for r in src:
            try: vals.append(float(r[k]))
            except Exception: pass
        # counts stay integers; rates keep 2dp (34.10 must not render as "34.1")
        if not vals: out[k] = "-"
        elif k in INTCOL: out[k] = f"{st.median(vals):.0f}"
        else: out[k] = f"{st.median(vals):.2f}"
    if ok and len(ok) == len(rs): out["status"] = "OK"
    elif ok:                      out["status"] = "PARTIAL"
    else:                         out["status"] = rs[0]["status"]
    merged.append(out)

base = {}
for r in merged:
    if r["nmax"] == "0" and r["status"] == "OK":
        base[tuple(r.get(k) for k in KEY)] = r
def paired(r):
    return base.get(tuple(r.get(k) for k in KEY))

def ctx_h(v): return f"{int(v)//1024}K" if str(v).isdigit() else v

def delta(r, key):
    b = paired(r); v = r.get(key, "-")
    if not b or r["nmax"] == "0" or v == "-" or b.get(key, "-") == "-": return v
    try: return f"{float(v):.2f} ({(float(v)/float(b[key])-1)*100:+.1f}%)"
    except Exception: return v

VIEWS = {
 "perf":  [("arm","arm"),("reps","reps"),("N","N"),("shape","shape"),("L","_layers"),("ctx/slot","ctx_slot"),
           ("drafter","drafter"),("n-max","nmax"),("MB","max_batch"),
           ("no-spec agg","_bagg"),("agg","_dagg"),("per-strm","strm"),("accept","accept"),
           ("no-spec TTFT","_bttft"),("TTFT","ttft_ms"),("err","errors"),("status","status")],
 "bw":    [("arm","arm"),("reps","reps"),("N","N"),("n-max","nmax"),
           ("PCIe rx MB/s","rxpci"),("PCIe tx MB/s","txpci"),("SM %","sm_pct"),
           ("memctl %","memctl_pct"),("CPU %","cpu_pct"),("RAM rd MB/s*","ram_rd_mbps"),
           ("agg","agg"),("status","status")],
 "cache": [("arm","arm"),("reps","reps"),("N","N"),("n-max","nmax"),("cache MiB","cache_mb"),
           ("admit","_admit"),("pool","pool_slots"),("hits","hits_pct"),("misses","misses"),
           ("evicts","evicts"),("VRAM peak","vram_peak"),("bypass","bypass"),("status","status")],
}
cols = [(k,k) for k in rows[0].keys()] + [("reps","reps")] if view == "all" else VIEWS[view]

def cell(r, key):
    if key == "_bagg":  b = paired(r); return b["agg"] if b else "-"
    if key == "_bttft": b = paired(r); return (b["ttft_ms"] + " ms") if b else "-"
    if key == "_dagg":  return delta(r, "agg")
    if key == "_admit": return f'{r["admit"]}/{r["throttle"]}'
    if key == "_layers": return f'{r.get("offload","-")}/{r.get("layers_total","-")}'
    v = r.get(key, "-")
    if key == "ctx_slot": return ctx_h(v)
    if key == "ttft_ms" and v != "-": return f"{v} ms"
    return v

hdr = [h for h,_ in cols]
table = [[str(cell(r,k)) for _,k in cols] for r in merged]

if as_md:
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for t in table: print("| " + " | ".join(t) + " |")
else:
    w = [max([len(hdr[i])] + [len(t[i]) for t in table]) for i in range(len(hdr))]
    line = "  ".join(hdr[i].ljust(w[i]) for i in range(len(hdr)))
    print(line); print("-"*len(line))
    for t in table: print("  ".join(t[i].ljust(w[i]) for i in range(len(hdr))))

bad = [r for r in merged if r["status"] != "OK"]
if bad:
    print(f"\n⚠ {len(bad)} arm(s) NOT OK — exclude from conclusions:")
    for r in bad:
        note = {"INVALID_BYPASS": "  cache was REFUSED: this arm measured spec with the expert cache OFF",
                "NO_TOKENS":      "  zero tokens returned: every request failed",
                "REQ_ERRORS":     f"  {r.get('errors','?')} request error(s) -- see the arm log",
                "BOOT_FAIL":      "  server never started (OOM at this ctx/pool is the usual cause)",
                "TIMEOUT":        "  server did not reach /health before BOOT_TIMEOUT",
                "CACHE_DISABLED": "  expert cache requested but NEVER allocated -- try RESERVE_MB=1536",
                "CACHE_PARTIAL":  "  cache allocated on only SOME devices -- throughput blends cached and uncached",
                "PORT_BUSY":      "  port was already served by another process; arm did not boot",
                "MB_CLAMP_CONFLICT": "  n-max too high for the [1,8] MAX_BATCH clamp; cache would bypass"}.get(r["status"], "")
        print(f"   {r['arm']}: {r['status']}{note}")

print("\nviews: --perf (default) | --bw | --cache | --all ; add --md for markdown")
if view in ("bw","all"):
    print("* RAM rd MB/s is DERIVED: misses x avg expert size (1.5 MiB) / elapsed — NOT a perf counter.")
    print("  PCIe/SM/memctl are nvidia-smi dmon 1 Hz means across the whole arm (includes idle tail).")
print("agg = sum(tokens)/round wall (includes prefill); per-strm = server-side decode TPS.")
print("Workload is copy-heavy/agentic — spec gains collapse to single digits on novel generation.")
print("Spec rows pair against the n-max=0 arm at the same offload/N/ctx/cache/admit AND SHAPE.")
print("A blank 'no-spec agg' means no baseline was measured at those settings — not a zero.")
