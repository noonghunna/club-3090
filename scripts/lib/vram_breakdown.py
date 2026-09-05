#!/usr/bin/env python3
"""Break a GPU's VRAM down into the buffers that actually hold it.

Called by capture.sh::cap_vram_breakdown from bench.sh. Reads the engine's own
boot log for named buffers, then reconciles against nvidia-smi so the gap
between "what the engine told us" and "what the card reports" is visible rather
than silent -- that gap is where a draft context hides.

Two traps this encodes, both paid for on 2026-08-31:
  * moe-cache pool: take the LAST report per (device, pool index). A drafter load
    triggers a SECOND allocation and both are logged; summing all matches
    double-counts (measured 13,947 against a live 6,820 -- 2.05x).
  * A pool figure read at boot is a LOWER BOUND. The pool grows into free VRAM
    as traffic admits experts, until it hits the free-minus-reserve floor.

Scope: the BYTE split only. Pool slot counts, hit rates and cache health belong to
CAPTURE: EXPERT CACHE and are deliberately not duplicated here -- one owner per
number, so the two sections can never disagree.
"""
import json
import re
import subprocess
import sys


def _per_dev(text, pat, agg=False):
    out = {}
    for m in re.finditer(pat, text):
        dev, val = m.group(1), float(m.group(2))
        out[dev] = out.get(dev, 0.0) + val if agg else val
    return out


def breakdown(text):
    """Parse the engine log -> the named-buffer dicts.

    Returns (model, compute, kv, state, pool, slots, allocs).  Pure -- no
    subprocess, no printing -- so the text and --json emitters cannot drift,
    and tests can drive it without a GPU."""
    # A GPU-pinned drafter emits the same `model buffer size` line as the
    # target model -- the two weights are additive, so last-wins would report
    # the drafter's and drop the target's into unaccounted.
    model = _per_dev(text, r"load_tensors:\s+(CUDA\d) model buffer size =\s+([\d.]+) MiB", agg=True)
    compute = _per_dev(text, r"sched_reserve:\s+(CUDA\d) compute buffer size =\s+([\d.]+) MiB")
    kv = _per_dev(text, r"llama_kv_cache:\s+(CUDA\d) KV buffer size =\s+([\d.]+) MiB", agg=True)
    state = _per_dev(text, r"_comp_state:\s+(CUDA\d) \S+ \S+ state buffer size =\s+([\d.]+) MiB", agg=True)
    # GLM names this buffer `llama_memory_recurrent ... RS` -- same kind of
    # state as DeepSeek's `_comp_state`, so it folds into the `state` column.
    for d, v in _per_dev(text, r"llama_memory_recurrent:\s+(CUDA\d) RS buffer size =\s+([\d.]+) MiB", agg=True).items():
        state[d] = state.get(d, 0.0) + v

    # LAST report per (device, pool) -- never sum every match, see module docstring
    last = {}
    for m in re.finditer(
        r"\[moe-cache\] (CUDA\d) pool\[(\d)\]:.*?slots=(\d+).*?total=(\d+) MiB", text
    ):
        last[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    pool, slots = {}, {}
    for (dev, idx), (sl, mib) in last.items():
        pool[dev] = pool.get(dev, 0) + mib
        slots[dev] = slots.get(dev, 0) + sl
    allocs = {}
    for m in re.finditer(r"\[moe-cache\] (CUDA\d) pool\[(\d)\]:", text):
        k = (m.group(1), m.group(2))
        allocs[k] = allocs.get(k, 0) + 1
    return model, compute, kv, state, pool, slots, allocs


def _bus_key(bus: str) -> str:
    """nvidia-smi may print the domain short ("0000:01:00.0") or long
    ("00000000:0000:01:00.0") -- compare the last two segments."""
    return ":".join(bus.strip().lower().split(":")[-2:])


def _prepare_devices(text) -> dict:
    """{CUDA ordinal: PCI bus id} from llama.cpp's device announcement
    (`llama_prepare_model_devices: using device CUDA0 (name) (0000:01:00.0)`)
    -- the log's own ordinal -> physical-GPU mapping (#1118 §4, option A)."""
    out = {}
    for m in re.finditer(
        r"llama_prepare_model_devices:\s+using device (CUDA\d) \([^)]*\) "
        r"\(([0-9A-Fa-f:.]+)\)", text):
        out[m.group(1)] = m.group(2)
    return out


def _join_smi(prep: dict, by_bus: dict, by_index: dict, warnings: list) -> dict:
    """Join log ordinals to physical GPUs by PCI bus id.  All-or-nothing: if
    every logged ordinal resolves, the joined map is exact under any
    ESTATE_GPUS ordering; if anything is missing (older log, odd nvidia-smi),
    fall back to index order and say so rather than half-map."""
    if prep and all(_bus_key(bus) in by_bus for bus in prep.values()):
        return {d: by_bus[_bus_key(bus)] for d, bus in prep.items()}
    if prep:
        warnings.append(
            "CUDA ordinals could not be joined to physical GPUs by PCI bus id "
            "-- falling back to index order (#1118 §4)")
    return dict(by_index)


def _fetch_smi() -> tuple[dict, dict]:
    """(by_bus, by_index): the same readings keyed by normalized PCI bus id
    and by CUDA-style index, so callers join whichever way the log allows."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        ).stdout
        by_bus, by_index = {}, {}
        for line in out.strip().splitlines():
            i, bus, used, tot = [x.strip() for x in line.split(",")]
            by_bus[_bus_key(bus)] = (float(used), float(tot))
            by_index["CUDA" + i] = (float(used), float(tot))
        return by_bus, by_index
    except Exception:
        return {}, {}


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--json"]
    as_json = len(argv) != len(sys.argv) - 1
    if len(argv) < 1:
        return 1
    try:
        text = open(argv[0], encoding="utf-8", errors="replace").read()
    except OSError:
        return 1

    model, compute, kv, state, pool, slots, allocs = breakdown(text)
    warnings: list[str] = []
    prep = _prepare_devices(text)
    by_bus, by_index = _fetch_smi()
    smi = _join_smi(prep, by_bus, by_index, warnings)

    devs = sorted(set(model) | set(smi) | set(pool))
    if not devs:
        return 1

    n = max(allocs.values()) if allocs else 0
    if n > 1:
        warnings.append(
            "%d moe-cache allocations logged per pool -- figures take the "
            "LAST (summing them double-counts)" % n)

    devices = []
    text_lines = []
    for dev in devs:
        parts, acc = [], 0.0
        entry = {"device": dev, "model": None, "kv": None, "state": None,
                 "compute": None, "pool": None, "used": None, "total": None,
                 "unaccounted": None}
        for label, src in (("model", model), ("kv", kv), ("state", state),
                           ("compute", compute), ("pool", pool)):
            if dev in src:
                parts.append("%s=%.0f" % (label, src[dev]))
                acc += src[dev]
                entry[label] = round(src[dev])
        if dev in smi:
            used, tot = smi[dev]
            entry["used"], entry["total"] = round(used), round(tot)
            parts.append("total=%.0f/%.0fMiB(%.0f%%)" % (used, tot, 100 * used / tot))
            if acc:
                unacc = used - acc
                if unacc < 0:
                    # The components come from the boot log; the total is live
                    # nvidia-smi.  A negative gap means the log is STALER than
                    # the card (container restarted / shrunk) -- clamp and say
                    # so instead of printing a nonsense negative (#1118).
                    warnings.append(
                        "%s: components (%.0f MiB) exceed the live total "
                        "(%.0f MiB) -- the log is staler than the card; "
                        "unaccounted clamped to 0" % (dev, acc, used))
                    unacc = 0.0
                entry["unaccounted"] = round(unacc)
                # everything the card reports that the engine did not name:
                # draft context, CUDA runtime, fragmentation
                parts.append("unaccounted=%.0f" % unacc)
        devices.append(entry)
        text_lines.append("  %s: %s" % (dev, "  ".join(parts)))

    if as_json:
        print(json.dumps({"devices": devices, "warnings": warnings}))
        return 0
    for line in text_lines:
        print(line)
    if warnings:
        for w in warnings:
            print("  WARNING: %s" % w)
    if pool:
        print("  pool slots + hit rate: see CAPTURE: EXPERT CACHE below (this line is the "
              "BYTE split only)")
    return 0

    # LAST report per (device, pool) -- never sum every match, see module docstring
    last = {}
    for m in re.finditer(
        r"\[moe-cache\] (CUDA\d) pool\[(\d)\]:.*?slots=(\d+).*?total=(\d+) MiB", text
    ):
        last[(m.group(1), m.group(2))] = (int(m.group(3)), int(m.group(4)))
    pool, slots, allocs = {}, {}, {}
    for (dev, idx), (sl, mib) in last.items():
        pool[dev] = pool.get(dev, 0) + mib
        slots[dev] = slots.get(dev, 0) + sl
    for m in re.finditer(r"\[moe-cache\] (CUDA\d) pool\[(\d)\]:", text):
        k = (m.group(1), m.group(2))
        allocs[k] = allocs.get(k, 0) + 1

    smi = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", timeout=10,
        ).stdout
        for line in out.strip().splitlines():
            i, used, tot = [x.strip() for x in line.split(",")]
            smi["CUDA" + i] = (float(used), float(tot))
    except Exception:
        pass

    devs = sorted(set(model) | set(smi) | set(pool))
    if not devs:
        return 1

    for dev in devs:
        parts, acc = [], 0.0
        for label, src in (("model", model), ("kv", kv), ("state", state),
                           ("compute", compute), ("pool", pool)):
            if dev in src:
                parts.append("%s=%.0f" % (label, src[dev]))
                acc += src[dev]
        if dev in smi:
            used, tot = smi[dev]
            parts.append("total=%.0f/%.0fMiB(%.0f%%)" % (used, tot, 100 * used / tot))
            if acc:
                # everything the card reports that the engine did not name:
                # draft context, CUDA runtime, fragmentation
                parts.append("unaccounted=%.0f" % (used - acc))
        print("  %s: %s" % (dev, "  ".join(parts)))

    n = max(allocs.values()) if allocs else 0
    if n > 1:
        print("  WARNING: %d moe-cache allocations logged per pool — figures above take the "
              "LAST (summing them double-counts)" % n)
    if pool:
        print("  pool slots + hit rate: see CAPTURE: EXPERT CACHE below (this line is the "
              "BYTE split only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
