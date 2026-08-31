#!/usr/bin/env bash
# Drift-guard for the registry `offload` facet.
#
# WHY THIS EXISTS: `offload` records whether a compose is RESIDENCY-CAPABLE,
# but for a long time it held the value "n-cpu-moe" — a name implying a
# `--n-cpu-moe` flag that NO compose on this stack has ever passed. Every
# CPU-offload compose here uses `-ot`, so the value could not be checked
# against the flag, and 4 entries drifted unnoticed across two model families
# (2026-08-29): deepseek-flash-{dual,multi4}-q8-moecache and
# inkling-small-{dual,multi4}-iq4xs-moecache claimed `residency` while
# declaring no residency contract at all.
#
# The contract, both directions:
#   (a) offload == "residency"        <=> the compose declares the residency
#       header contract (CPU-Offload-Bundle-MiB + CPU-Offload-MoE-Layers) AND
#       carries OT_G pin slots in its `-ot` regex, which the launcher
#       (resolve_offload_residency) fills in at boot;
#   (b) offload == "tensor-override"  => passes `-ot`, declares NO residency
#       contract;
#   (c) offload == "n-cpu-moe"        => the compose really does pass
#       `--n-cpu-moe`. Legal in principle; nothing uses it today, and the
#       retired spelling must not come back by copy-paste.
#
# ⚠️ Deliberately checks the CONTRACT, not the flag: `-ot` is present on every
# CPU-offload compose, so asserting on it distinguishes nothing.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import re, sys, pathlib, yaml

BUNDLE = re.compile(r'^#\s*CPU-Offload-Bundle-MiB:\s*\d+', re.M)
LAYERS = re.compile(r'^#\s*CPU-Offload-MoE-Layers:\s*\d+', re.M)

def facts(text):
    """(residency_contract, has_ot, ot_slots, has_ncm) read from a compose."""
    try:
        d = yaml.safe_load(text)
        svc = d["services"][list(d["services"])[0]]
        cmd = [str(x) for x in (svc.get("command") or [])]
    except Exception:
        cmd = []
    has_ot = "-ot" in cmd
    slots = cmd[cmd.index("-ot") + 1].count("OT_G") if has_ot else 0
    has_ncm = any("n-cpu-moe" in c for c in cmd) or bool(
        re.search(r'--n-cpu-moe', text))
    contract = bool(BUNDLE.search(text)) and bool(LAYERS.search(text))
    return contract, has_ot, slots, has_ncm

def expected(contract, has_ot, slots, has_ncm):
    if has_ncm and not has_ot:
        return "n-cpu-moe"
    if contract and slots > 0:
        return "residency"
    if has_ot:
        return "tensor-override"
    return None          # not a CPU-offload compose; offload should be unset

fail = []

# ---- SELF-TEST FIRST: a gate that cannot fail is not a gate ----------------
OT = "      - '-ot'\n      - '{rx}'\n"
def mk(rx, contract):
    hdr = ("# CPU-Offload-Bundle-MiB: 2848\n# CPU-Offload-MoE-Layers: 38\n"
           if contract else "# CPU-Offload-Host-RAM-GB: 121\n")
    return (hdr + "services:\n  s:\n    image: x\n    command:\n"
            + OT.format(rx=rx))

RES_RX = r'${OT_G0:-blk\.99999\.ffn_up_exps\.weight=CUDA0},blk\.[0-9]+\.ffn_up_exps\.weight=CPU'
PLAIN_RX = r'blk\.[0-9]+\.ffn_up_exps\.weight=CPU'
cases = [
    ("residency compose",            mk(RES_RX, True),   "residency"),
    ("plain -ot compose",            mk(PLAIN_RX, False), "tensor-override"),
    ("OT_G slots but NO contract",   mk(RES_RX, False),  "tensor-override"),
    ("contract but NO OT_G slots",   mk(PLAIN_RX, True), "tensor-override"),
]
for label, text, want in cases:
    got = expected(*facts(text))
    if got != want:
        fail.append(f"SELF-TEST '{label}': expected {want!r}, derived {got!r}")
    else:
        print(f"  self-test ok: {label} -> {want}")

# ---- the real catalog ------------------------------------------------------
reg = yaml.safe_load(open("scripts/lib/profiles/registry.yaml", encoding="utf-8"))["entries"]
checked = 0
for slug, e in sorted(reg.items()):
    cp = e.get("compose_path")
    if not cp:
        continue
    p = pathlib.Path(cp)
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    want = expected(*facts(text))
    have = e.get("offload")
    if want is None:
        # non-offload compose: `offload` may legitimately be unset or a
        # non-CPU backend value (uva / prefetch) this guard does not own.
        if have in ("residency", "tensor-override", "n-cpu-moe"):
            fail.append(f"{slug}: offload={have!r} but the compose has no CPU-offload flags")
        continue
    checked += 1
    if have != want:
        fail.append(f"{slug}: offload={have!r} but the compose says {want!r} ({cp})")

print(f"  checked {checked} CPU-offload slug(s) against their composes")
if fail:
    print("\nFAIL: registry `offload` drifted from the composes:", file=sys.stderr)
    for f in fail:
        print(f"  ⛔ {f}", file=sys.stderr)
    sys.exit(1)
print("test-compose-offload-drift: ok")
PY
