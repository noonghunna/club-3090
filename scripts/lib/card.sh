#!/usr/bin/env bash
# card.sh — render a ready-to-paste BENCH_CARD from a bench.sh run.
#
# WHY THIS EXISTS
#   docs/BENCH_CARD.md's templates are correct and nobody fills them in by hand
#   under time pressure — which is exactly when the awkward facts (a scrape that
#   read 0.00, a cache still warming, 3 of 5 runs EOSing early) get quietly left
#   off. bench.sh already CAPTURES all of it; this renders it into the template
#   so the card is a by-product of the run instead of a chore after it.
#
# WHAT IT WILL NOT DO
#   It never invents a human judgement. The one-line verdict, the A/B decision,
#   the knob under test, "quiet box" — all render as explicit `<fill: …>`
#   placeholders. A card that guesses its own verdict is worse than no card.
#
# OUT OF SCOPE: machine-parseable JSON. That is #249's job (the measurement-record
#   producer). This emits MARKDOWN for a human to paste into an issue or a
#   BENCHMARKS discussion. When #249 lands, both should read the same record —
#   this file's RECORD SCHEMA below is the natural seam.
#
# RECORD SCHEMA (flat key=value, one per line; values may contain spaces)
#   meta.date meta.endpoint meta.mode
#   fp.model fp.ctx fp.slots fp.ftype fp.drafter fp.kv fp.argv fp.moe fp.offload
#   proto.warmups proto.runs proto.force_tokens proto.thinking proto.env
#   shape.<name>.{n,n_usable,decode_mean,decode_cv,wall_mean,wall_cv,ttft_mean_ms,ttft_std_ms}
#   prefill.<label>.{n,tps_mean,tps_cv,ttft_ms}
#   gpu.{vram_idle,vram_peak,vram_post,leak,per_device}
#   cache.<dev>.{marginal,cumulative,cum_start,slots,total_mib,type,evictions,skips}
#   pcie.{link,link_pct,link_warn,practical_mbps}
#   pcie.<phase>.<all|GPUn>.{rx_mean,rx_peak}      phase = decode | prefill
#   ram.{demand_mbps,window,ceiling_gbps}  sys.ram
#   integrity.{status,health,swap,divergence,stream_ceiling}
#
#   TWO producers, one schema:
#     * the CURRENT run  -> bench.sh fills the record in-process (exact values)
#     * a BASELINE run   -> parsed back out of a saved bench.sh log
#   They must agree. test-bench-card.sh renders a run, saves its log, re-parses
#   it, and asserts the parsed record matches the in-process one field for field
#   — otherwise an A/B would silently compare two different measurements.
#
# `command grep` only (ugrep shim), and no pgrep/pkill -f anywhere. See AGENTS.md.

[[ -n "${_CARD_SH_LOADED:-}" ]] && return 0
_CARD_SH_LOADED=1
export PYTHONUTF8="${PYTHONUTF8:-1}"

# card_kv <file> <key> <value> — append one record field. Values are written
# verbatim; empty values are SKIPPED rather than written as empty, so a renderer
# can tell "not captured" from "captured as blank".
card_kv() {
  local f="$1" k="$2" v="${3:-}"
  [[ -n "$v" ]] || return 0
  printf '%s=%s\n' "$k" "$v" >> "$f"
}

# card_kv_from_summary <record-file> <summary.json> — fold the python side's
# per-shape statistics into the record. Schema knowledge lives HERE, next to the
# parser that has to reproduce it from a log, so the two cannot drift apart.
card_kv_from_summary() {
  local rec="$1" js="$2"
  [[ -r "$js" ]] || return 1
  python3 - "$js" >> "$rec" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        d = json.load(fh)
except Exception:
    sys.exit(1)


def emit(k, v):
    if v not in (None, ""):
        print(f"{k}={v}")


for name, v in d.get("shapes", {}).items():
    # A prefill entry carries prefill_tps_mean; a decode shape carries decode.
    if "prefill_tps_mean" in v:
        p = f"prefill.{name}"
        emit(f"{p}.n", v.get("n"))
        emit(f"{p}.tps_mean", f"{v['prefill_tps_mean']:.2f}")
        emit(f"{p}.tps_cv", f"{v.get('prefill_tps_cv', 0):.1f}")
        emit(f"{p}.ttft_ms", f"{v.get('ttft_mean_ms', 0):.0f}")
    else:
        p = f"shape.{name}"
        emit(f"{p}.n", v.get("n"))
        emit(f"{p}.n_usable", v.get("n_usable"))
        emit(f"{p}.decode_mean", f"{v.get('decode_tps_mean', 0):.2f}")
        emit(f"{p}.decode_cv", f"{v.get('decode_tps_cv', 0):.1f}")
        emit(f"{p}.wall_mean", f"{v.get('wall_tps_mean', 0):.2f}")
        emit(f"{p}.wall_cv", f"{v.get('wall_tps_cv', 0):.1f}")
        emit(f"{p}.ttft_mean_ms", f"{v.get('ttft_mean_ms', 0):.0f}")
        emit(f"{p}.ttft_std_ms", f"{v.get('ttft_std_ms', 0):.0f}")
PY
}

# card_parse_log <bench-log> — extract a record from a SAVED bench.sh log.
# Strict on purpose: the baseline of an A/B is load-bearing, and half-parsing one
# produces a confident wrong delta. Exits non-zero with a reason on stderr.
card_parse_log() {
  local log="${1:-}"
  if [[ -z "$log" ]]; then
    echo "baseline unparseable: no path given (set BASELINE=<path to a saved bench.sh log>)" >&2
    return 1
  fi
  if [[ ! -r "$log" ]]; then
    echo "baseline unparseable: cannot read '$log'" >&2
    return 1
  fi
  CARD_MODE=parse _card_py "$log"
}

# card_render <mode> <current-record> [baseline-log]
#   mode = snapshot | ab
card_render() {
  local mode="$1" rec="$2" base="${3:-}"
  CARD_MODE="render_$mode" _card_py "$rec" "$base"
}

# The single embedded implementation: parser and renderer share the schema, so
# they cannot drift into disagreeing about what a field means.
_card_py() {
  CARD_MODE="${CARD_MODE:-}" \
  CARD_NOISE_BAND="${NOISE_BAND:-}" \
  CARD_EXPECTATION="${EXPECTATION:-}" \
  CARD_BOOT_POLICY="${BOOT_POLICY:-}" \
  CARD_KNOB="${KNOB:-}" \
  python3 - "$@" <<'PY'
import os, re, sys

MODE = os.environ.get("CARD_MODE", "")
FILL = "<fill: {}>"

# --- fields whose disagreement across arms CONFOUNDS an A/B ------------------
# Everything the measurement rests on. If one of these differs and it is not the
# declared knob, the two arms are not an A/B — they are two unrelated benches.
INVARIANTS = [
    ("fp.model",   "model"),
    ("fp.ftype",   "weights ftype"),
    ("fp.ctx",     "served ctx"),
    ("fp.kv",      "KV cache type"),
    ("fp.drafter", "drafter"),
    # The moe-cache config is the ARM IDENTITY: the resulting pool self-limits
    # below the requested cap, so two different configs can produce the same
    # census. A mismatch here confounds an A/B exactly as hard as a model mismatch.
    ("fp.moe",     "moe-cache config (cap + RESERVE_MB/ADMIT_AFTER/THROTTLE/MAX_BATCH)"),
    ("fp.offload", "CPU-offload method (-ot regex / --n-cpu-moe / boot log)"),
    ("fp.argv",    "serving argv (threads / ubatch / offload / ngl / split)"),
]

# Pool census SHAPE is compared with a tolerance rather than exactly: the census
# self-limits against free VRAM, so an identical config can land a few slots apart
# between boots. Beyond this it is a different cache, not noise.
POOL_TOLERANCE_PCT = 5.0


def load_record(path):
    rec = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if "=" in ln:
                k, v = ln.split("=", 1)
                rec[k.strip()] = v.strip()
    return rec


def num(s, default=None):
    try:
        return float(re.sub(r"[^0-9.eE+-]", "", str(s)))
    except (TypeError, ValueError):
        return default


# ===========================================================================
# PARSER — a saved bench.sh log back into a record
# ===========================================================================
def parse_log(path):
    # The baseline of an A/B is load-bearing. Every failure to read it exits with
    # a reason on stderr — a traceback is not an error message an operator can act
    # on, and half a parse produces a confident wrong delta.
    if not path:
        sys.stderr.write("baseline unparseable: no path given "
                         "(set BASELINE=<path to a saved bench.sh log>)\n")
        sys.exit(1)
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        sys.stderr.write(f"baseline unparseable: cannot read {path!r} ({e.strerror})\n")
        sys.exit(1)
    rec = {}
    missing = []

    def grab(pat, key, group=1, where=lines):
        for ln in where:
            m = re.search(pat, ln)
            if m:
                rec[key] = m.group(group).strip()
                return True
        return False

    # --- fingerprint block ---
    grab(r"endpoint\s+:\s*(\S+)", "meta.endpoint")
    grab(r"endpoint\s+:.*\(mode=(\w+)\)", "meta.mode")
    grab(r"^\s+model\s+:\s*(.+)$", "fp.model")
    grab(r"served ctx\s+:\s*(\d+)", "fp.ctx")
    grab(r"served ctx\s+:.*slots=(\d+)", "fp.slots")
    grab(r"weights ftype\s+:\s*(.+)$", "fp.ftype")
    grab(r"^\s+drafter\s+:\s*(.+)$", "fp.drafter")
    grab(r"KV cache type\s+:\s*(.+)$", "fp.kv")
    if rec.get("fp.kv", "").startswith("capture:"):
        del rec["fp.kv"]
    grab(r"serving argv\s+:\s*(.+)$", "fp.argv")
    grab(r"moe-cache cfg\s+:\s*(.+)$", "fp.moe")
    grab(r"CPU offload\s+:\s*(.+)$", "fp.offload")

    for ln in lines:
        m = re.match(r"=== warmups \((\d+)\) ===", ln.strip())
        if m and "proto.warmups" not in rec:
            rec["proto.warmups"] = m.group(1)
        m = re.match(r"=== measured \((\d+)\) ===", ln.strip())
        if m and "proto.runs" not in rec:
            rec["proto.runs"] = m.group(1)

    if "fp.model" not in rec:
        missing.append("CONFIG FINGERPRINT block (no 'model :' line) — was this run "
                       "produced by a bench.sh with the capture layer, and with CAPTURE=1?")

    # --- shape summaries: '=== summary [narrative] (n=5) ===' ---
    shapes = 0
    i = 0
    while i < len(lines):
        m = re.match(r"=== summary \[([^\]]+)\] \(n=(\d+)\) ===", lines[i].strip())
        if not m:
            i += 1
            continue
        name, n = m.group(1), m.group(2)
        # Stop at the next section header. A fixed-size slice bleeds into the
        # FOLLOWING summary block, and since the prefill block contains
        # "prefill tok/s", that misclassified every shape that happened to be
        # followed by a prefill summary — the shape then vanished from the A/B.
        block = []
        for ln in lines[i + 1:]:
            if ln.strip().startswith("==="):
                break
            block.append(ln)
        is_prefill = any("prefill tok/s" in b for b in block)
        key = ("prefill." if is_prefill else "shape.") + name
        rec[f"{key}.n"] = n
        for ln in block:
            mm = re.match(r"\s*(wall_TPS|decode_TPS|prefill tok/s)\s+mean=\s*([0-9.]+)"
                          r"\s+std=\s*([0-9.]+)\s+CV=\s*([0-9.]+)%", ln)
            if mm:
                label = {"wall_TPS": "wall", "decode_TPS": "decode",
                         "prefill tok/s": "tps"}[mm.group(1)]
                rec[f"{key}.{label}_mean"] = mm.group(2)
                rec[f"{key}.{label}_cv"] = mm.group(4)
            mt = re.match(r"\s*TTFT\s+mean=\s*([0-9.]+)ms\s+std=\s*([0-9.]+)ms", ln)
            if mt:
                rec[f"{key}.ttft_mean_ms"] = mt.group(1)
                rec[f"{key}.ttft_std_ms"] = mt.group(2)
                if is_prefill:
                    rec[f"{key}.ttft_ms"] = mt.group(1)
            mu = re.match(r"\s*n-usable\s+(\d+)/(\d+)", ln)
            if mu:
                rec[f"{key}.n_usable"] = mu.group(1)
        shapes += 1
        i += 1

    if not shapes:
        missing.append("no '=== summary [<shape>] (n=N) ===' block — the run produced "
                       "no measured shape")

    # --- capture sections ---
    grab(r"^\s*status:\s*([A-Z_]+)", "integrity.status")
    grab(r"health counters:\s*(.+)$", "integrity.health")
    grab(r"swap check\s+:\s*(.+)$", "integrity.swap")
    # One cross-check line per shape (#1862); bench.sh records the WORST shape, so take
    # the max here too (the first on a tie, as bench.sh's awk does). Taking the first line
    # let a saved log tick the ±5% check while a later shape was over it.
    divs = [m.group(1) for ln in lines
            for m in [re.search(r"->\s*([0-9.]+)% divergence", ln)] if m]
    if divs:
        rec["integrity.divergence"] = max(divs, key=float)
    grab(r"triad ceiling\s+:\s*([0-9.]+) GB/s", "integrity.stream_ceiling")
    m = re.search(r"idle=(\d+) MiB\s+peak=(\S+) MiB\s+post=(\S+) MiB",
                  "\n".join(lines))
    if m:
        rec["gpu.vram_idle"], rec["gpu.vram_peak"], rec["gpu.vram_post"] = m.groups()
    # "growth delta" is the same measurement under a label that does not cry wolf
    # when a lazily-allocating expert cache is active (see bench.sh).
    grab(r"(?:leak|growth) delta \(post - idle\) = (-?\d+) MiB", "gpu.leak")
    for ln in lines:
        if "growth delta (post - idle)" in ln:
            rec["gpu.delta_kind"] = "growth"
            break
        if "leak delta (post - idle)" in ln:
            rec["gpu.delta_kind"] = "leak"
            break

    # --- offload / PCIe / RAM telemetry ---
    grab(r"^\s*(mem_total_gib=.*)$", "sys.ram")
    grab(r"derived host-RAM read demand on the miss path: ~(\d+) MB/s", "ram.demand_mbps")
    for ln in lines:
        if "derived host-RAM read demand" in ln:
            rec["ram.window"] = ("whole-run (diluted)" if "WHOLE RUN" in ln else "decode window")
            break
    grab(r"of ~(\d+) GB/s STREAM-triad ceiling", "ram.ceiling_gbps")
    grab(r"triad ceiling\s+:\s*([0-9.]+) GB/s", "ram.ceiling_gbps")
    grab(r"^\s*(GPU\d+ gen=\d+/\d+ width=\d+/\d+)", "pcie.link")
    grab(r"decode rx peak = \d+ MB/s = ([0-9.]+)% of that denominator", "pcie.link_pct")
    for ln in lines:
        if "WARN: never reached" in ln:
            rec["pcie.link_warn"] = "link trained below its own maximum under load"
            break
    for ln in lines:
        m = re.match(r"\s*(decode|prefill) (all|GPU\d+) rx_mean=(\d+) rx_peak=(\d+)", ln)
        if m:
            ph, dev = m.group(1), m.group(2)
            rec[f"pcie.{ph}.{dev}.rx_mean"] = m.group(3)
            rec[f"pcie.{ph}.{dev}.rx_peak"] = m.group(4)
    # skips are reported inside the health verdict's informational tail
    for ln in lines:
        if "health counters:" in ln or "cache health" in ln:
            for dm in re.finditer(r"(CUDA\d+)[:\s]+skips=(\d+)", ln):
                rec[f"cache.{dm.group(1)}.skips"] = dm.group(2)

    # per-device cache rows stay PER DEVICE (never averaged)
    for ln in lines:
        m = re.match(r"\s*(CUDA\d+) marginal=([0-9.]+|-)% cumulative=([0-9.]+|-)%", ln)
        if m:
            rec[f"cache.{m.group(1)}.marginal"] = m.group(2)
            rec[f"cache.{m.group(1)}.cumulative"] = m.group(3)
            dv = re.search(r"devictions=(\d+)", ln)
            if dv:
                rec[f"cache.{m.group(1)}.evictions"] = dv.group(1)
        m = re.match(r"\s*(CUDA\d+) pools=\d+ slots=(\d+) total_mib=(\d+)", ln)
        if m:
            rec[f"cache.{m.group(1)}.slots"] = m.group(2)
            rec[f"cache.{m.group(1)}.total_mib"] = m.group(3)
            ty = re.search(r"type=(\S+)", ln)
            if ty and ty.group(1) != "-":
                rec[f"cache.{m.group(1)}.type"] = ty.group(1)

    if missing:
        sys.stderr.write("baseline unparseable: " + "; ".join(missing) + "\n")
        sys.exit(1)
    return rec


# ===========================================================================
# shared render helpers
# ===========================================================================
def shape_names(rec, prefix):
    out = []
    for k in rec:
        m = re.match(rf"{prefix}\.(.+)\.n$", k)
        if m:
            out.append(m.group(1))
    return sorted(set(out))


def is_offload(rec):
    """Same gate as the capture layer: the offload block renders only when the run
    actually offloaded. A generic cache line is rendered otherwise."""
    v = rec.get("fp.offload", "")
    return bool(v) and "not detected" not in v


def cache_devices(rec):
    return sorted({m.group(1) for k in rec
                   for m in [re.match(r"cache\.([^.]+)\.", k)] if m})


def cache_judgement(marg, cum):
    """cold | warming | plateaued | thrashing, from marginal vs cumulative.

    Stated as a rule on the card so a reader can disagree with it:
      marginal < 5%              -> cold (the pool is not paying for itself yet)
      |marginal - cumulative|<=2 -> plateaued (steady state reached)
      marginal > cumulative      -> warming (cumulative still catching up)
      marginal < cumulative      -> thrashing (the window did WORSE than lifetime)
    """
    m, c = num(marg), num(cum)
    if m is None or c is None:
        return "unknown"
    if m < 5:
        return "cold"
    if abs(m - c) <= 2:
        return "plateaued"
    return "warming" if m > c else "thrashing"


def pcie_slim(rec):
    """PCIe for a run that did NOT offload.

    Item 10 makes PCIe capture UNCONDITIONAL — TP all-reduce traffic is the
    interesting part on a GPU-resident multi-card run, and that is exactly the
    case with no offload block to carry it. Without this the bench output captures
    PCIe and the card silently drops it, which is the one place a card can be less
    honest than the run it describes."""
    link = rec.get("pcie.link", "")
    warn = rec.get("pcie.link_warn", "")
    dec_mean, dec_peak = rec.get("pcie.decode.all.rx_mean"), rec.get("pcie.decode.all.rx_peak")
    pre_mean, pre_peak = rec.get("pcie.prefill.all.rx_mean"), rec.get("pcie.prefill.all.rx_peak")
    if not (link or dec_mean or pre_mean):
        return []
    parts = []
    if dec_mean:
        parts.append(f"decode rx {dec_mean} / {dec_peak} MB/s mean/peak")
    if pre_mean:
        parts.append(f"prefill rx {pre_mean} / {pre_peak} MB/s mean/peak")
    pct = rec.get("pcie.link_pct")
    if pct:
        parts.append(f"{pct}% of practical link")
    head = f"**PCIe:** {link or FILL.format('link')}"
    if warn:
        head += f" ⚠️ {warn}"
    out = [head + (" · " + " · ".join(parts) if parts else "")]
    out.append("")
    out.append("> No CPU offload on this run, so this is interconnect traffic — on a multi-card "
               "tensor-parallel run that is the all-reduce path, and it is the number a topology "
               "change (link width, P2P state) moves. 1 Hz sampling under-reads bursts, so the "
               "peak sits next to the mean.")
    out.append("")
    return out


def offload_block(rec, devs):
    """The CPU-offload + expert-cache section. Rendered ONLY when offload is
    detected — on a GPU-resident run every row here would be a dash pretending to
    be data."""
    out = ["### CPU offload + expert cache", ""]
    ram = rec.get("sys.ram", "")
    rss = re.search(r"server_rss_gib=([\d.]+)", ram)
    tot = re.search(r"mem_total_gib=([\d.]+)", ram)
    swap = rec.get("integrity.swap", "")
    swap_short = "PASS" if swap.startswith("PASS") else (swap.split("—")[0].strip() or FILL.format("swap"))
    out.append(f"**Offload:** {rec.get('fp.offload', FILL.format('method'))}"
               + (f" · **RSS** {rss.group(1)} GiB / {tot.group(1)} GiB total" if rss and tot else "")
               + f" · swap {swap_short}")
    out.append(f"**Cache config:** {rec.get('fp.moe', FILL.format('cap + env knobs'))}")
    out.append("")
    out.append("> The cache config IS the arm identity — the pool self-limits below the requested "
               "cap, so pool size alone does not identify which arm this was.")
    out.append("")
    if devs:
        out.append("| device | pool (slots · MiB · type) | marginal hits | cumulative | evictions | skips† |")
        out.append("|---|---|--:|--:|--:|--:|")
        for d in devs:
            pool = " · ".join(x for x in (
                (rec.get(f"cache.{d}.slots", "") + " slots") if rec.get(f"cache.{d}.slots") else "",
                (rec.get(f"cache.{d}.total_mib", "") + " MiB") if rec.get(f"cache.{d}.total_mib") else "",
                rec.get(f"cache.{d}.type", "")) if x) or "—"
            out.append(f"| {d} | {pool} | **{rec.get(f'cache.{d}.marginal','—')}%** | "
                       f"{rec.get(f'cache.{d}.cumulative','—')}% | "
                       f"{rec.get(f'cache.{d}.evictions','—')} | {rec.get(f'cache.{d}.skips','0')} |")
        out.append("")
        out.append("†`skips` = admission throttling (insert queue exhausted under the configured "
                   "THROTTLE) — a **load signal, not a defect**. The hard-failure counters are "
                   "fill / dispatch / collect-fail; those are on the integrity panel.")
        out.append("")
        out.append("> Quote the **marginal** rate, never the cumulative: cumulative embeds the "
                   "cold-fill phase, so a *bigger* pool reports a *lower* cumulative rate on the "
                   "same traffic. Per-device rows stay separate — the hit asymmetry across cards "
                   "is routing-entropy signal, and averaging it away hides it.")
        out.append("")
    # --- PCIe ---
    link = rec.get("pcie.link", "")
    warn = rec.get("pcie.link_warn", "")
    dec_mean, dec_peak = rec.get("pcie.decode.all.rx_mean"), rec.get("pcie.decode.all.rx_peak")
    pre_mean, pre_peak = rec.get("pcie.prefill.all.rx_mean"), rec.get("pcie.prefill.all.rx_peak")
    if link or dec_mean:
        bits = [f"**PCIe** ({link or FILL.format('link')}"
                + (f" ⚠️ {warn}" if warn else "") + ")"]
        if dec_mean:
            pct = rec.get("pcie.link_pct")
            bits.append(f"decode rx {dec_mean} / {dec_peak} MB/s mean/peak"
                        + (f" ({pct}% of practical link)" if pct else ""))
        if pre_mean:
            bits.append(f"prefill rx {pre_mean} / {pre_peak} MB/s mean/peak")
        out.append(": ".join([bits[0], " · ".join(bits[1:])]) if len(bits) > 1 else bits[0])
        # Lopsided per-device traffic is worth calling out — on a layer-split MoE
        # one card can carry nearly all of the prefill transfer.
        per = [(d, num(rec.get(f"pcie.prefill.{d}.rx_mean")))
               for d in sorted({m.group(1) for k in rec
                                for m in [re.match(r"pcie\.prefill\.(GPU\d+)\.", k)] if m})]
        per = [(d, v) for d, v in per if v is not None]
        if len(per) > 1:
            hi = max(per, key=lambda x: x[1])
            lo = min(per, key=lambda x: x[1])
            if lo[1] >= 0 and hi[1] > 0 and (lo[1] == 0 or hi[1] / max(lo[1], 1e-9) >= 3):
                out.append("")
                out.append(f"> Prefill PCIe traffic is lopsided: {hi[0]} {hi[1]:.0f} MB/s vs "
                           f"{lo[0]} {lo[1]:.0f} MB/s mean. On a layer-split MoE that is expected "
                           f"(the offloaded layers are not evenly distributed), but it bounds which "
                           f"card the transfer path actually stresses.")
        out.append("")
    # --- RAM path ---
    dem = num(rec.get("ram.demand_mbps"))
    if dem is not None:
        ceil_g = num(rec.get("ram.ceiling_gbps"))
        frac = (f" of ~{ceil_g:.0f} GB/s ceiling ({dem / 10 / ceil_g:.0f}%)"
                if ceil_g else " (run with STREAM_CALIB=1 for the ceiling)")
        out.append(f"**RAM path:** derived miss demand ~{dem / 1000:.1f} GB/s{frac} — "
                   f"**derived, not measured** (misses × expert size ÷ elapsed), computed over the "
                   f"**{rec.get('ram.window', FILL.format('window'))}**.")
        out.append("")
    return out


def check(ok, label, evidence=""):
    box = "x" if ok is True else " "
    tail = f" — {evidence}" if evidence else ""
    return f"- [{box}] {label}{tail}"


def integrity_block(rec):
    """The four capture-derived additions plus the template's original four."""
    out = []
    div = num(rec.get("integrity.divergence"))
    if div is None:
        out.append(check(None, "wall-TPS scrape agrees with engine-side timings (±5%)",
                         "engine-side timings unavailable "
                         "(set SERVER_LOG=<path> on bare metal) — " + FILL.format("state which number the card quotes")))
    else:
        out.append(check(div <= 5, "wall-TPS scrape agrees with engine-side timings (±5%)",
                         f"{div:.1f}% divergence"
                         + ("" if div <= 5 else " — " + FILL.format("state which number this card quotes, and why"))))

    worst = max([num(rec[k], 0) for k in rec if k.startswith("shape.") and k.endswith(".decode_cv")] or [0])
    pworst = max([num(rec[k], 0) for k in rec if k.startswith("prefill.") and k.endswith(".tps_cv")] or [0])
    out.append(check(worst < 5 and pworst < 2,
                     "CVs within rig norm (decode <5%, prefill <2%)",
                     f"worst decode CV {worst:.1f}%, worst prefill CV {pworst:.1f}%"))
    out.append(check(None, "quiet box: no builds / downloads / other GPU work during measured runs",
                     FILL.format("confirm")))

    devs = cache_devices(rec)
    if devs:
        js = {cache_judgement(rec.get(f"cache.{d}.marginal"), rec.get(f"cache.{d}.cumulative"))
              for d in devs}
        out.append(check("plateaued" in js and len(js) == 1,
                         "cache state steady across measured runs",
                         ", ".join(sorted(js)) + " (a warming cache understates the mean)"))
    else:
        out.append(check(None, "cache state steady across measured runs", "no expert cache on this run"))

    # --- the capture-layer additions ---
    st = rec.get("integrity.status", "unknown")
    out.append(check(st == "OK", f"capture status: {st}",
                     "" if st == "OK" else "**do not quote a throughput number from this run**"))
    h = rec.get("integrity.health", "")
    if h:
        # PASS is fill/dispatch/collect-fail == 0. `skips` is admission throttling,
        # reported alongside as a tuning signal — not a defect (see capture.sh).
        out.append(check(h.startswith("PASS"), "cache hard-failure counters zero "
                                               "(fill / dispatch / collect)", h))
    s = rec.get("integrity.swap", "")
    if s:
        # WARN: pages sit in swap but were not swapped back in during the run, so the numbers stand (#1846)
        out.append(check(s.startswith(("PASS", "WARN")), "swap check", s))
    us = []
    bad = False
    for name in shape_names(rec, "shape"):
        n, u = rec.get(f"shape.{name}.n", "?"), rec.get(f"shape.{name}.n_usable")
        if u is None:
            continue
        us.append(f"{name} {u}/{n}")
        if u != n:
            bad = True
    if us:
        out.append(check(not bad, "n-usable == n for every shape", ", ".join(us)
                         + ("" if not bad else " — the CV above is not trustworthy; re-run with FORCE_TOKENS=<n>")))
    leak = num(rec.get("gpu.leak"))
    if leak is not None:
        if rec.get("gpu.delta_kind") == "growth":
            # An expert cache allocates its pool and graphs on first inference, so a
            # single run grows VRAM by design. Ticked because nothing is wrong —
            # only a monotonic rise across REPEATED runs is evidence of a leak.
            out.append(check(True, "VRAM growth accounted for",
                             f"growth delta {leak:.0f} MiB (lazy expert-cache pool/graph "
                             f"allocation; only a monotonic rise across REPEATED runs is a leak)"))
        else:
            out.append(check(abs(leak) <= 64, "VRAM returned after the run",
                             f"leak delta {leak:.0f} MiB"))
    return out


def protocol_line(rec):
    w = rec.get("proto.warmups", "?")
    r = rec.get("proto.runs", "?")
    extra = " · thinking=on" if rec.get("proto.thinking") == "1" else ""
    ft = rec.get("proto.force_tokens", "0")
    if ft not in ("0", "", None):
        extra += f" · FORCE_TOKENS={ft}"
    return (f"**Protocol:** {w} warm + {r} measured per shape · temperature=0.6 top_p=0.95{extra} · "
            + FILL.format("power cap"))


def config_line(rec):
    bits = []
    for key, pre in (("fp.ftype", ""), ("fp.kv", "KV "), ("fp.ctx", "ctx ")):
        v = rec.get(key)
        if v:
            bits.append(f"{pre}{v}")
    bits.append("spec-dec: " + (rec.get("fp.drafter") or FILL.format("drafter|none")))
    off = rec.get("fp.offload", "")
    if off and "not detected" not in off:
        bits.append("CPU offload: yes")
    return ("**Config:** " + FILL.format("engine+tag") + " · " + FILL.format("topology")
            + " · " + " · ".join(bits))


# ===========================================================================
# TEMPLATE 1 — SNAPSHOT
# ===========================================================================
def render_snapshot(rec):
    L = []
    A = L.append
    A(f"## 📊 bench.sh — {rec.get('fp.model', FILL.format('model'))} · "
      f"{FILL.format('config-slug')} · {rec.get('meta.date', FILL.format('date'))}")
    A("")
    A(config_line(rec))
    A(f"**Env:** {rec.get('proto.env') or '(defaults)'}")
    A(protocol_line(rec))
    A("")
    A("| shape | n | decode TPS | CV | wall TPS | TTFT |")
    A("|---|--:|--:|--:|--:|--:|")
    for name in shape_names(rec, "shape"):
        p = f"shape.{name}"
        n, u = rec.get(f"{p}.n", "?"), rec.get(f"{p}.n_usable")
        ncell = f"{n}" if (u is None or u == n) else f"{u}/{n} ⚠️"
        A(f"| {name} | {ncell} | {rec.get(f'{p}.decode_mean','—')} | "
          f"{rec.get(f'{p}.decode_cv','—')}% | {rec.get(f'{p}.wall_mean','—')} | "
          f"{rec.get(f'{p}.ttft_mean_ms','—')} ± {rec.get(f'{p}.ttft_std_ms','0')} ms |")
    pre = shape_names(rec, "prefill")
    if pre:
        A("")
        A("| prefill depth | n | tok/s | CV | TTFT@depth |")
        A("|---|--:|--:|--:|--:|")
        for name in pre:
            p = f"prefill.{name}"
            A(f"| {name} | {rec.get(f'{p}.n','?')} | {rec.get(f'{p}.tps_mean','—')} | "
              f"{rec.get(f'{p}.tps_cv','—')}% | {rec.get(f'{p}.ttft_ms','—')} ms |")
    A("")
    gpu = []
    if rec.get("gpu.vram_peak"):
        gpu.append(f"VRAM idle {rec.get('gpu.vram_idle','?')} → peak {rec['gpu.vram_peak']} "
                   f"→ post {rec.get('gpu.vram_post','?')} MiB")
    if rec.get("gpu.per_device"):
        gpu.append(f"per device: {rec['gpu.per_device']}")
    A(f"**GPU:** {' · '.join(gpu) if gpu else FILL.format('VRAM / power / util')} · "
      + FILL.format("power cap") + " · " + FILL.format("util"))
    devs = cache_devices(rec)
    if is_offload(rec):
        L.extend(offload_block(rec, devs))
    else:
        L.extend(pcie_slim(rec))
    if not is_offload(rec) and devs:
        # Non-offload engines with a cache of their own keep the generic line.
        parts = []
        for d in devs:
            marg = rec.get(f"cache.{d}.marginal", "—")
            cum0 = rec.get(f"cache.{d}.cum_start")
            cum = rec.get(f"cache.{d}.cumulative", "—")
            j = cache_judgement(marg, cum)
            arrow = f"{cum0}% → {cum}%" if cum0 else f"{cum}% cumulative"
            parts.append(f"{d} hits {arrow}, **marginal {marg}%** ({j})")
        A(f"**Cache (if the engine has one):** {' · '.join(parts)}")
        A("")
        A("> Quote the **marginal** rate. Cumulative embeds the cold-fill phase, so a *bigger* "
          "pool reports a *lower* cumulative rate on the same traffic. Per-device values are kept "
          "separate on purpose — a CUDA0/CUDA1 asymmetry is routing-entropy signal. "
          "Judgement rule: marginal <5% = cold; within 2pp of cumulative = plateaued; "
          "above = warming; below = thrashing.")
    A("")
    A("### Integrity")
    L.extend(integrity_block(rec))
    A("")
    A(f"**One-line verdict:** {FILL.format('what this run establishes — and what it must NOT be quoted for')}")
    return "\n".join(L)


# ===========================================================================
# TEMPLATE 2 — A/B
# ===========================================================================
def render_ab(cur, base):
    knob = os.environ.get("CARD_KNOB", "").strip()
    band_env = os.environ.get("CARD_NOISE_BAND", "").strip()

    # Noise band: given, or 2x the worst shape CV across BOTH arms. Stated either
    # way — a band with an unstated provenance is not a band.
    cvs = [num(r[k], 0) for r in (cur, base) for k in r
           if k.startswith("shape.") and k.endswith(".decode_cv")]
    if band_env:
        band = num(band_env, 5.0)
        how = "NOISE_BAND env, supplied by the operator"
    elif cvs:
        band = 2 * max(cvs)
        how = f"derived: 2x the worst decode CV across both arms ({max(cvs):.1f}%)"
    else:
        band = 5.0
        how = "default 5% — no CVs available to derive from"

    # --- invariants ---
    diffs = []
    for key, label in INVARIANTS:
        a, b = base.get(key), cur.get(key)
        if a is None and b is None:
            continue
        if a != b:
            diffs.append((key, label, a, b))
    # Pool census SHAPE, per device, with a tolerance.
    #
    # ONLY checked when the requested cache CONFIG matched. If the config differs,
    # a different pool is the INTENDED CONSEQUENCE of the knob, not a confound —
    # flagging it there would make every cache-sizing A/B read as confounded. When
    # the config is identical and the realized pool still differs materially, that
    # IS a confound: the two boots had different VRAM available to them, so the
    # arms are not comparable for a reason outside the config.
    same_cfg = base.get("fp.moe") is not None and base.get("fp.moe") == cur.get("fp.moe")
    for d in (sorted(set(cache_devices(cur)) | set(cache_devices(base))) if same_cfg else []):
        a, b = base.get(f"cache.{d}.slots"), cur.get(f"cache.{d}.slots")
        if a is None or b is None:
            continue
        av, bv = num(a), num(b)
        if av and bv and abs(bv - av) / av * 100 > POOL_TOLERANCE_PCT:
            diffs.append((f"cache.{d}.slots", f"pool census {d} (slots, ±{POOL_TOLERANCE_PCT:.0f}%)",
                          a, b))
    if knob:
        attributed = [d for d in diffs if knob.lower() in (d[0] + " " + d[1]).lower()
                      or knob.lower() in str(d[2]).lower() or knob.lower() in str(d[3]).lower()]
        confounds = [d for d in diffs if d not in attributed]
    else:
        attributed, confounds = [], diffs

    L = []
    A = L.append
    if confounds:
        A("> ## ⛔ CONFOUNDED")
        A(">")
        A("> These arms differ in " + str(len(confounds)) + " value(s) that must be held constant. "
          "This is not an A/B — it is two unrelated benches, and the Δ column below cannot be "
          "attributed to any knob. Re-run with them matched, or name the intended knob with "
          "`KNOB=<text>` if one of these IS the thing under test.")
        A("")
    A(f"## ⚖️ bench.sh A/B — {knob or FILL.format('knob')}: "
      f"A: {FILL.format('arm A name')} vs B: {FILL.format('arm B name')} · "
      f"{cur.get('fp.model', FILL.format('model'))} · {cur.get('meta.date', FILL.format('date'))}")
    A("")
    held = [lab for key, lab in INVARIANTS
            if base.get(key) is not None and base.get(key) == cur.get(key)]
    A(f"**Held constant:** {', '.join(held) if held else FILL.format('list what was held')}")
    A(f"**Boot policy:** {os.environ.get('CARD_BOOT_POLICY') or FILL.format('same-boot | boot-per-arm ×N')}"
      f"   **Noise band:** ±{band:.1f}% ({how})")
    A("")
    A(f"**Pre-registered expectation:** {os.environ.get('CARD_EXPECTATION') or 'none'}")
    A("")
    A("| metric | A: baseline | B: this run | Δ | verdict |")
    A("|---|--:|--:|--:|---|")

    def row(label, a, b, lower_is_better=False, kind="perf"):
        av, bv = num(a), num(b)
        if av is None or bv is None or av == 0:
            A(f"| {label} | {a or '—'} | {b or '—'} | — | not comparable (missing in one arm) |")
            return
        d = (bv - av) / av * 100
        if abs(d) <= band:
            verdict = "within band — no signal"
        elif kind == "cost":
            # A resource row is NOT good or bad on its own — spending more pool to
            # buy throughput may be exactly the intended trade. Report the spend
            # and leave the judgement to the human, who has the VRAM budget.
            verdict = ("**+{:.1f}% resource spend**" if d > 0 else "**{:.1f}% resource freed**").format(d)
        else:
            better = (d < 0) if lower_is_better else (d > 0)
            verdict = "outside band — real " + ("improvement" if better else "regression")
        A(f"| {label} | {av:.2f} | {bv:.2f} | {d:+.1f}% | {verdict} |")

    for name in sorted(set(shape_names(cur, "shape")) | set(shape_names(base, "shape"))):
        row(f"decode {name}", base.get(f"shape.{name}.decode_mean"), cur.get(f"shape.{name}.decode_mean"))
    for name in sorted(set(shape_names(cur, "shape")) | set(shape_names(base, "shape"))):
        row(f"TTFT {name} (ms)", base.get(f"shape.{name}.ttft_mean_ms"),
            cur.get(f"shape.{name}.ttft_mean_ms"), lower_is_better=True)
    for name in sorted(set(shape_names(cur, "prefill")) | set(shape_names(base, "prefill"))):
        label = name[8:] if name.startswith("prefill-") else name
        row(f"prefill {label}", base.get(f"prefill.{name}.tps_mean"), cur.get(f"prefill.{name}.tps_mean"))
    # CACHE DELTAS — only when both arms actually offloaded.
    #
    # The MARGINAL hit rate is THE boot-comparable cache metric, and this is the row
    # that catches an under-warmed pool: a cumulative rate would show the bigger
    # pool as WORSE (it spent longer filling), so an A/B read off cumulative numbers
    # rejects the better config. Marginal is measured over the same window in both
    # arms, so it compares like with like.
    if is_offload(cur) and is_offload(base):
        for d in sorted(set(cache_devices(cur)) | set(cache_devices(base))):
            row(f"marginal hits {d} (%)", base.get(f"cache.{d}.marginal"), cur.get(f"cache.{d}.marginal"))
        if cur.get("pcie.prefill.all.rx_peak") and base.get("pcie.prefill.all.rx_peak"):
            row("prefill PCIe rx peak (MB/s)", base.get("pcie.prefill.all.rx_peak"),
                cur.get("pcie.prefill.all.rx_peak"), kind="cost")
    # The hidden cost. A knob that buys throughput by spending a persistent
    # resource (pool slots, KV headroom, VRAM) looks free until this row exists —
    # several configs have been adopted-then-reverted for exactly that.
    for d in sorted(set(cache_devices(cur)) | set(cache_devices(base))):
        row(f"pool slots {d}", base.get(f"cache.{d}.slots"), cur.get(f"cache.{d}.slots"), kind="cost")
    row("VRAM peak (MiB)", base.get("gpu.vram_peak"), cur.get("gpu.vram_peak"), kind="cost")

    A("")
    A("**Invariants (must match across arms or the A/B is confounded):**")
    A("")
    if not diffs:
        A("| field | value | |")
        A("|---|---|---|")
        for key, label in INVARIANTS:
            if cur.get(key) is not None:
                A(f"| {label} | `{cur[key]}` | ✅ match |")
    else:
        A("| field | A: baseline | B: this run | |")
        A("|---|---|---|---|")
        rendered = set()
        for key, label in INVARIANTS:
            a, b = base.get(key), cur.get(key)
            if a is None and b is None:
                continue
            rendered.add(key)
            if a == b:
                A(f"| {label} | `{a}` | `{b}` | ✅ match |")
            else:
                tag = "🔧 the declared knob" if (key, label, a, b) in attributed else "⛔ **CONFOUND**"
                A(f"| {label} | `{a or '(absent)'}` | `{b or '(absent)'}` | {tag} |")
        for key, label, a, b in diffs:
            if key in rendered:
                continue
            tag = "🔧 the declared knob" if (key, label, a, b) in attributed else "⛔ **CONFOUND**"
            A(f"| {label} | `{a or '(absent)'}` | `{b or '(absent)'}` | {tag} |")
    A("")
    A("### Integrity — arm B (this run)")
    L.extend(integrity_block(cur))
    A("")
    A("### Integrity — arm A (baseline)")
    L.extend(integrity_block(base))
    A("")
    A(f"**Decision:** {FILL.format('ADOPT / REJECT / PARK')} — "
      f"{FILL.format('one sentence, naming the decisive metric')}")
    return "\n".join(L)


# ===========================================================================
if MODE == "parse":
    r = parse_log(sys.argv[1])
    for k in sorted(r):
        print(f"{k}={r[k]}")
    sys.exit(0)

if MODE == "render_snapshot":
    print(render_snapshot(load_record(sys.argv[1])))
    sys.exit(0)

if MODE == "render_ab":
    cur = load_record(sys.argv[1])
    baseline = sys.argv[2] if len(sys.argv) > 2 else ""
    if not baseline:
        sys.stderr.write("baseline unparseable: CARD=ab needs BASELINE=<path to a saved bench.sh log>\n")
        sys.exit(1)
    print(render_ab(cur, parse_log(baseline)))
    sys.exit(0)

sys.stderr.write(f"card.sh: unknown mode {MODE!r}\n")
sys.exit(2)
PY
}
