#!/usr/bin/env python3
"""
Hardened, fail-closed patch for sgl-project/sglang#33713 — MAMBA component
nodes pruned instead of downgraded on device eviction, breaking host-tier
loadback.

HARDENED vs the dormant /patch/apply_patch.py:
  * Idempotent: re-running `apply` on an already-patched tree is a no-op.
  * `--check` is a READ-ONLY per-edit status gate (applied / pristine /
    absent) — safe to run before booting, never mutates.
  * FAIL-CLOSED on anchor drift: if a pristine anchor is missing OR a file is
    in a half-patched state, `apply` exits non-zero (so the entrypoint.d
    wrapper refuses to boot unpatched).
  * Edits apply to a single file, unified_tree_core.py (the v0.5.19
    unified-cache live path). The two edits are a match-side pair:

      Edit 1 — traversal dead-node check (~L783):
        Before: child.evicted and not child.backuped            (FULL-only)
        After : child.evicted and not any(cd.host_value is not None
                                                for cd in child.component_data)
        A node is only "dead" (stops the match walk) if NO component — FULL
        or MAMBA — has a host copy. Before, a mamba-only host node was seen
        as dead and the walk stopped, so a mamba host backup could never be
        discovered for loadback.

      Edit 2 — _is_host_leaf membership (~L1840):
        Before: not node.backuped                                 (FULL-only)
        After : not any(cd.host_value is not None
                    for cd in node.component_data)
        A node is an evictable host leaf if ANY component has a host copy,
        not only if the FULL (KV) component does.

  * NOTE ON VALIDATION (see worklog/bench/mamba-33713/): on our GDN model
    (Qwen3.5) this match-side change does NOT alter the demote-vs-prune fork
    (the evict decision at ~L1340 still keys off the FULL component), and the
    KV-level "no loadback" symptom does not reproduce here — it reproduces on
    the issue's KDA model (Ling-3.0-flash). This patch is included per request
    and is fail-closed, but treat it as HYPOTHESIS-ONLY until validated against
    a KDA model. It is NOT the evict-fork fix the reporter's data points to.

Usage (inside the container, via the entrypoint.d wrapper):
    python /patch/apply_33713.py           # apply (idempotent), fail-closed
    python /patch/apply_33713.py --check  # read-only per-edit status gate
"""

import argparse
import re
import sys
from pathlib import Path

TARGET = Path("/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py")

# component_data is a list[ComponentData] indexed by ComponentType (int);
# iterate it directly, not .values().
def _any_host(cd_list):
    return "any(cd.host_value is not None for cd in %s)" % cd_list

EDITS = [
    {
        "tag": "E1",
        "kind": "traversal dead-node check",
        "pristine": (
            "            # HiCache: dead node (evicted + not backuped) \u2014 stop traversal\n"
            "            if child.evicted and not child.backuped:\n"
            "                break"
        ),
        "patched": (
            "            # HiCache: dead node (evicted + no component has host data) \u2014 stop traversal\n"
            "            if child.evicted and not " + _any_host("child.component_data") + ":\n"
            "                break"
        ),
    },
    {
        "tag": "E2",
        "kind": "_is_host_leaf membership",
        "pristine": (
            "        if not node.backuped:\n"
            "            return False"
        ),
        "patched": (
            "        if not " + _any_host("node.component_data") + ":\n"
            "            return False"
        ),
    },
]

# the 2-line anchors are unique in v0.5.19 (verified); E2's bare single line
# appears 3x, so we always match on the 2-line block.
def state_of(text, ed):
    if ed["patched"] in text:
        return "PATCHED"
    if ed["pristine"] in text:
        return "pristine"
    return "ABSENT"

def read_or_die(p):
    if not p.exists():
        sys.exit("ERROR: target file not found: %s" % p)
    return p.read_text()

def apply_edits(src):
    changed = False
    for ed in EDITS:
        st = state_of(src, ed)
        if st == "PATCHED":
            continue  # idempotent no-op
        if st == "pristine":
            if src.count(ed["pristine"]) != 1:
                sys.exit("ERROR: E%s anchor not unique (count=%d) — refusing." %
                         (ed["tag"], src.count(ed["pristine"])))
            src = src.replace(ed["pristine"], ed["patched"], 1)
            changed = True
        else:
            sys.exit("ERROR: E%s anchor ABSENT and not patched (drift) — refusing to boot." % ed["tag"])
    return src, changed

def cmd_apply(p):
    src = read_or_die(p)
    try:
        new, changed = apply_edits(src)
    except SystemExit:
        raise
    if changed:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(new)
        tmp.replace(p)
    for ed in EDITS:
        print("%s  %-26s -> %s" % (ed["tag"], ed["kind"], state_of(p.read_text(), ed)))
    print("APPLY " + ("OK" if changed else "(no-op: tree already patched)"))

def cmd_check(p):
    if not p.exists():
        print("CHECK ERROR: target missing: %s" % p)
        return 1
    src = p.read_text()
    bad = False
    for ed in EDITS:
        st = state_of(src, ed)
        print("%s  %-26s : %s" % (ed["tag"], ed["kind"], st))
        if st != "PATCHED":
            bad = True
    print("CHECK " + ("PASS (all edits applied)" if not bad else "FAIL (some edit not applied)"))
    return 0 if not bad else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "sgl#33713 patch")
    ap.add_argument("--check", action="store_true", help="read-only: per-edit status gate (safe pre-flight)")
    ap.add_argument("--target", help="override target path (off-image test)")
    a = ap.parse_args(argv)
    p = Path(a.target) if a.target else TARGET
    if a.check:
        sys.exit(cmd_check(p))
    cmd_apply(p)

if __name__ == "__main__":
    main()
