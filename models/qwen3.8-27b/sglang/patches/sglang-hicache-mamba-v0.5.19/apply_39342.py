#!/usr/bin/env python3
"""
Production patch for sgl-project/sglang#39342 — mixed-chunk corrupts the
mamba radix cache.

WHAT IT FIXES
  prepare_for_extend stamps req.kv.mamba_last_track_seqlen for each incoming
  prefill. merge_batch (called from mix_with_running) then unconditionally
  nulls the batch mamba_track_* tensors, so the GDN checkpoint write is
  skipped (guarded by track_mask is not None). The per-req claim is never
  rolled back, and mamba_component.prepare_for_caching_req (the v0.5.19
  unified-cache live path) reads mamba_last_track_seqlen to size the mamba
  donation. Result: a stale ping-pong slot (never written for this seqlen) is
  donated into the unified radix tree under a mismatched key.

WHY THE FIVE EDITS ARE A LOAD-BEARING UNIT (not optional / not reorderable)
  * B (mix_with_running) is what *sets* the rollback flag C keys off, and
    clears the stale seqlen so it cannot size anything. It is scoped to the
    INCOMING PREFILL reqs only (the head of mix_with_running, before
    merge_batch) — NOT the running decode reqs, whose mamba stamps point at
    checkpoints written by their own (possibly non-mixed) prefill and remain
    valid. A1/A2 plumb + reset that flag on Req.
  * C returns 0 when the claim was rolled back. The caller does
        if cl is not None: effective_cache_len = min(len(token_ids), cl)
    so 0 truncates the effective cache length to 0. In cache_unfinished_req
    that routes into the existing `if effective_cache_len <= 0: return`
    guard (no insert, no match, no depth assertion). In cache_finished_req
    there is NO such guard: effective_cache_len==0 reaches self.insert() with
    an empty RadixKey, which unified_tree_core.begin_insert short-circuits
    (len(key)==0 -> InsertResult(prefix_len=0, mamba_exist=True,
    last_device_node=root)) before the walk/commit — so the mamba commit (and
    edit D's guard) is never reached on this path and the request's KV rows
    are simply freed. Both paths end with no insert, no assertion, no stale
    donation. The KV cost of not caching this request is the safe analogue of
    pristine, which cached a stale mamba slot.
  * D is a defensive guard in commit_insert_component_data for a None
    mamba_value; it is not strictly reached in the return-0 path but keeps the
    tree walk safe if any future path leaves mamba_value unset.
  * Cost: the rolled-back request's KV prefix is not cached (cold re-prefill on
    next reuse). This is the safe analogue of pristine, which cached garbage.
  Run all five together; they are validated as a set below.

MIGRATION (v1 -> v2)
  v1 placed the B rollback at the TAIL of mix_with_running (after merge_batch),
  which iterated the combined self.reqs = incoming prefills + running decode
  reqs. That over-reached: it cleared the decode reqs' valid stamps and set
  their rollback flag, so their KV prefix was silently dropped from the cache
  (a pervasive cache-hit regression under sustained mixed-chunk load) and it
  would not surface in an answer-correctness or crash-only test. v2 relocates
  the rollback to the HEAD of mix_with_running, before merge_batch, where
  self.reqs is ONLY the incoming prefills. This script migrates a v1 tree to
  v2 (reverts the old tail-B, then applies the head-B); a pristine tree is
  applied directly to v2. See --check for the five-edit state.

TARGETS (lmsysorg/sglang:v0.5.19 layout, verified against the image)
  schedule_batch.py   (Req-level, version-stable)   edits A1 A2 B (migr) B2 (head)
  mamba_component.py  (v0.5.19 unified-cache path)   edits C D

SAFETY
  * Idempotent: re-running `apply` on an already-patched (v2) tree is a
    no-op.
  * All-or-nothing ACROSS THE UNIT (two-pass): apply first validates EVERY
    edit in BOTH files read-only, and only if the whole load-bearing unit is
    pristine-or-patched does it write anything. This is what prevents a
    half-written state where schedule_batch.py is patched but
    mamba_component.py is not (the v1-crash shape, where B clears the stamp
    but C never reads the flag).
  * Each edit requires a byte-unique anchor; any missing/duplicate anchor
    aborts the run (no partial writes) with a clear message.
  * Writes are atomic (tmp file in same dir + ast.parse + os.replace) so a
    crash never leaves a half-patched file.
  * `check` is a read-only gate (safe to wire into compose pre-flight).
  * `revert` returns the files to pristine by inverting each edit (NEW->OLD),
    gated on the per-edit marker so it only touches what was actually applied.
  * Version-gated: warns if the installed sglang version differs from the
    version the anchors were verified against, without hard-failing (anchors
    are content-based and will simply be reported as missing).

USAGE (inside the container, at /sgl-workspace/... ; or any root via --root)
  python apply_39342.py            # apply (default; two-pass, all-or-nothing)
  python apply_39342.py --check    # read-only status of every edit
  python apply_39342.py --revert   # undo (only what was applied)
  python apply_39342.py --report   # full per-edit table + version + summary
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Targets & anchors
# ---------------------------------------------------------------------------

DEFAULT_SB = "/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py"
DEFAULT_MCM = "/sgl-workspace/sglang/python/sglang/srt/mem_cache/unified_cache/components/mamba_component.py"

# Version the anchors were verified against (content match is authoritative;
# this only warns on drift).
KNOWN_GOOD_VERSION = "0.5.19"


def _edit(tag, kind, old, new):
    return {"tag": tag, "kind": kind, "old": old, "new": new}


EDITS = {
    "schedule_batch.py": [
        _edit(
            "A1",
            "Req field: mamba_mixed_rollback",
            "    # Deferred clear: newly allocated mamba slot needs zeroing on forward stream\n"
            "    mamba_needs_clear: bool = False\n",
            "    # Deferred clear: newly allocated mamba slot needs zeroing on forward stream\n"
            "    mamba_needs_clear: bool = False\n"
            "    # SGL-39342 fix: set when this prefill's mamba claim is rolled back (mixed\n"
            "    # batch orphaned it — the claimed ping-pong slot is never written).\n"
            "    mamba_mixed_rollback: bool = False\n",
        ),
        _edit(
            "A2",
            "Req reset: clear rollback flag",
            "        self.kv.mamba_ping_pong_track_buffer = None\n"
            "        self.kv.mamba_next_track_idx = None\n"
            "        self.kv.mamba_last_track_idx = None\n",
            "        self.kv.mamba_ping_pong_track_buffer = None\n"
            "        self.kv.mamba_next_track_idx = None\n"
            "        self.kv.mamba_last_track_idx = None\n"
            "        self.mamba_mixed_rollback = False\n",
        ),
        # --- B (MIGRATION, tail): revert the v1 tail-B rollback to pristine. ---
        # old = the v1 "new" text; new = the pristine tail. On a pristine tree
        # this is already-patched (pristine tail present) and is skipped; on a
        # v1 tree it reverts the over-reaching tail-B so the head-B can take its
        # place. Runs BEFORE the head-B so the rollback lives in exactly one
        # spot.
        _edit(
            "B",
            "mix_with_running tail: revert v1 rollback (migration)",
            "        self.is_prefill_only = False\n"
            "        # SGL-39342 fix: the incoming prefill reqs' mamba claim (stamped in\n"
            "        # prepare_for_extend) is orphaned by the track-tensor nulling in\n"
            "        # merge_batch; the mixed forward never writes that slot. Roll it\n"
            "        # back so mamba_component does not donate a stale slot.\n"
            "        for req in self.reqs:\n"
            "            if req.kv.mamba_last_track_seqlen is not None:\n"
            "                if req.kv.mamba_last_track_idx is not None:\n"
            "                    req.kv.mamba_next_track_idx = req.kv.mamba_last_track_idx\n"
            "                req.kv.mamba_last_track_seqlen = None\n"
            "                req.mamba_mixed_rollback = True\n"
            "\n"
            "    def convert_decode_to_extend(self):\n",
            "        self.is_prefill_only = False\n"
            "\n"
            "    def convert_decode_to_extend(self):\n",
        ),
        # --- B2 (HEAD, the actual v2 fix): scoped to incoming prefill reqs. ---
        # At the head of mix_with_running, before merge_batch, self.reqs is ONLY
        # the incoming prefill/extend reqs. The running decode reqs are not yet
        # merged in, so their (valid) stamps are left untouched.
        _edit(
            "B2",
            "mix_with_running head: roll back orphaned prefill mamba claim",
            "    def mix_with_running(self, running_batch: ScheduleBatch):\n"
            "        self.forward_mode = ForwardMode.MIXED\n"
            "        running_bs = running_batch.batch_size()\n",
            "    def mix_with_running(self, running_batch: ScheduleBatch):\n"
            "        self.forward_mode = ForwardMode.MIXED\n"
            "        running_bs = running_batch.batch_size()\n"
            "        # SGL-39342 fix (v2): at this point self.reqs is ONLY the incoming\n"
            "        # prefill/extend reqs (running_batch is not merged in yet). Their\n"
            "        # mamba claim was stamped in prepare_for_extend, but the mixed forward\n"
            "        # (merge_batch nulls mamba_track_*) never writes that ping-pong slot.\n"
            "        # Roll back the claim so mamba_component does not donate a stale slot.\n"
            "        # We deliberately do NOT touch the running decode reqs: their stamps\n"
            "        # point at checkpoints written by their own (possibly non-mixed)\n"
            "        # prefill and remain valid — rolling those back (v1) silently dropped\n"
            "        # their KV from the cache.\n"
            "        for req in self.reqs:\n"
            "            if req.kv.mamba_last_track_seqlen is not None:\n"
            "                if req.kv.mamba_last_track_idx is not None:\n"
            "                    req.kv.mamba_next_track_idx = req.kv.mamba_last_track_idx\n"
            "                req.kv.mamba_last_track_seqlen = None\n"
            "                req.mamba_mixed_rollback = True\n",
        ),
    ],
    "mamba_component.py": [
        _edit(
            "C",
            "prepare_for_caching_req: skip mamba value on rollback",
            "        if self.cache.enable_mamba_extra_buffer:\n"
            "            cache_len = req.kv.mamba_last_track_seqlen\n"
            "        else:\n"
            "            cache_len = token_ids_len\n",
            "        # SGL-39342 fix: if the mamba claim was rolled back (mixed batch),\n"
            "        # the claimed ping-pong slot was never written — don't donate it,\n"
            "        # and cap the KV insert to 0 (no insert). This prevents the\n"
            "        # KV-insert-vs-mamba-match depth mismatch that fires the\n"
            "        # `assert new_prefix_len <= len(new_indices)` in cache_unfinished_req.\n"
            "        # Cost: this request's KV prefix is not cached (cold re-prefill next\n"
            "        # time). This is the safe analogue of pristine's behaviour, which\n"
            "        # would cache a stale mamba slot (the #39342 corruption).\n"
            "        if (\n"
            "            self.cache.enable_mamba_extra_buffer\n"
            "            and getattr(req, \"mamba_mixed_rollback\", False)\n"
            "        ):\n"
            "            return 0  # no insert; caller sees effective_cache_len=0\n"
            "        if self.cache.enable_mamba_extra_buffer:\n"
            "            cache_len = req.kv.mamba_last_track_seqlen\n"
            "        else:\n"
            "            cache_len = token_ids_len\n",
        ),
        _edit(
            "D",
            "commit_insert_component_data: tolerate None mamba_value",
            "        assert params.mamba_value is not None\n",
            "        if params.mamba_value is None:\n"
            "            return\n",
        ),
    ],
}

FILES = {
    "schedule_batch.py": DEFAULT_SB,
    "mamba_component.py": DEFAULT_MCM,
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def is_patched(text, ed):
    # NEW present => patched. (For additive edits A1/A2/C the OLD text remains a
    # substring of NEW, so we key off NEW, not "OLD absent".)
    return ed["new"] in text


def is_pristine(text, ed):
    return ed["old"] in text and not is_patched(text, ed)


def read_or_die(p):
    if not p.exists():
        sys.exit(f"ERROR: target file not found: {p}")
    return p.read_text()


def atomic_write(p, content):
    ast.parse(content)  # fail before touching the on-disk file
    tmp = p.with_suffix(p.suffix + ".39342.tmp")
    tmp.write_text(content)
    os.replace(tmp, p)


def detect_version():
    try:
        return importlib.metadata.version("sglang")
    except Exception:
        pass
    try:
        import sglang  # type: ignore

        return getattr(sglang, "__version__", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def run_edits(path, name, edits, direction):
    """Apply (old->new) or revert (new->old) a file's edits.

    direction='apply': every edit must be pristine (old present, new absent)
        or already-patched (new present); already-patched edits are skipped.
    direction='revert': every edit must be patched (new present).

    Per-file all-or-nothing: if ANY edit in a file cannot be applied/reverted
    (missing or ambiguous anchor), the whole file is left untouched. Returns
    (changed, per_edit, all_ok).
    """
    text = read_or_die(path)
    changed = False
    all_ok = True
    per_edit = []
    for ed in edits:
        if direction == "apply":
            if is_patched(text, ed):
                per_edit.append((ed["tag"], "already-patched"))
                continue  # skip (would be a duplicate write)
            if is_pristine(text, ed):
                if text.count(ed["old"]) != 1:
                    per_edit.append((ed["tag"], "AMBIGUOUS"))
                    all_ok = False
                    continue
                text = text.replace(ed["old"], ed["new"], 1)
                changed = True
                per_edit.append((ed["tag"], "applied"))
            else:
                per_edit.append((ed["tag"], "ABSENT"))
                all_ok = False  # hold the write for this whole file
        else:  # revert
            if is_patched(text, ed):
                if text.count(ed["new"]) != 1:
                    per_edit.append((ed["tag"], "AMBIGUOUS"))
                    all_ok = False
                    continue
                text = text.replace(ed["new"], ed["old"], 1)
                changed = True
                per_edit.append((ed["tag"], "reverted"))
            else:
                per_edit.append((ed["tag"], "not-patched (left)" if is_pristine(text, ed) else "ABSENT"))
    if changed and all_ok:
        atomic_write(path, text)
        return True, per_edit, all_ok
    # Partial/ambiguous within this file -> do not write; report the held state.
    return changed, per_edit, all_ok


def validate_file(path, name, edits):
    """Read-only pre-flight for cmd_apply's pass 1: every edit must be
    pristine (old present, new absent, unique) or already-patched (new
    present). Never writes. Returns (all_ok, per_edit)."""
    text = read_or_die(path)
    all_ok = True
    per_edit = []
    for ed in edits:
        if is_patched(text, ed):
            per_edit.append((ed["tag"], "already-patched"))
        elif is_pristine(text, ed) and text.count(ed["old"]) == 1:
            per_edit.append((ed["tag"], "pristine"))
        elif is_pristine(text, ed):
            per_edit.append((ed["tag"], "AMBIGUOUS"))
            all_ok = False
        else:
            per_edit.append((ed["tag"], "ABSENT"))
            all_ok = False
    return all_ok, per_edit


def cmd_apply(root_overrides):
    ver = detect_version()
    if ver and ver != KNOWN_GOOD_VERSION:
        print(f"WARNING: installed sglang is {ver}; anchors were verified against "
              f"{KNOWN_GOOD_VERSION}. Continuing (content-gated).")

    # PASS 1 — read-only validation of EVERY edit in EVERY file, no writes.
    # The edits are a load-bearing unit across 2 files; writing file A and only
    # then finding file B drifted would leave a half-written crash state (B
    # clears the stamp, C never reads the flag). Validate the whole unit first.
    pre_ok = True
    for name, edits in EDITS.items():
        path = Path(root_overrides.get(name) or FILES[name])
        all_ok, per_edit = validate_file(path, name, edits)
        print(f"[{name}] " + (", ".join(f"{t}={s}" for t, s in per_edit)))
        if not all_ok:
            pre_ok = False
    if not pre_ok:
        sys.exit("\nERROR: one or more anchors are missing/ambiguous — NOTHING was "
                 "written (all-or-nothing across the load-bearing unit). "
                 "Inspect with --report.")

    # PASS 2 — every edit is pristine-or-patched, so these writes cannot fail
    # mid-way; perform them.
    any_applied = False
    for name, edits in EDITS.items():
        path = Path(root_overrides.get(name) or FILES[name])
        changed, _per_edit, _all_ok = run_edits(path, name, edits, "apply")
        if changed:
            any_applied = True
    print("APPLY " + ("OK" if any_applied else "(no-op: tree already patched)"))


def cmd_check(root_overrides):
    rc = 0
    for name, edits in EDITS.items():
        path = Path(root_overrides.get(name) or FILES[name])
        if not path.exists():
            print(f"[{name}] MISSING {path}")
            rc = 1
            continue
        text = path.read_text()
        for ed in edits:
            if is_patched(text, ed):
                print(f"[{name}] {ed['tag']}: PATCHED")
            elif is_pristine(text, ed):
                print(f"[{name}] {ed['tag']}: PRISTINE (not applied)")
                rc = 1
            else:
                print(f"[{name}] {ed['tag']}: ABSENT (anchor not found)")
                rc = 1
    print("CHECK " + ("OK: all edits applied" if rc == 0 else "NOT fully applied"))
    return rc


def cmd_revert(root_overrides):
    for name, edits in EDITS.items():
        path = Path(root_overrides.get(name) or FILES[name])
        changed, per_edit, all_ok = run_edits(path, name, edits, "revert")
        print(f"[{name}] " + (", ".join(f"{t}={s}" for t, s in per_edit) if changed
                              else "(nothing to revert)"))


def cmd_report(root_overrides):
    ver = detect_version()
    print(f"sglang version : {ver or 'unknown'}  (verified against {KNOWN_GOOD_VERSION})")
    print(f"stable marker  : SGL-39342\n")
    hdr = f"{'file':<22}{'tag':<5}{'state':<14}{'anchor'}"
    print(hdr)
    print("-" * len(hdr))
    for name, edits in EDITS.items():
        path = Path(root_overrides.get(name) or FILES[name])
        if not path.exists():
            print(f"{name:<22}{'-':<5}{'MISSING':<14}{'-'}")
            continue
        text = path.read_text()
        for ed in edits:
            if is_patched(text, ed):
                st = "PATCHED"
            elif is_pristine(text, ed):
                st = "pristine"
            else:
                st = "ABSENT"
            print(f"{name:<22}{ed['tag']:<5}{st:<14}{ed['kind']}")
    print("\nNote: the five edits are a load-bearing unit (B2 sets the flag C reads,")
    print("C's `return 0` needs D to skip the commit). Apply/revert them together.")
    print("B is the v1->v2 migration (reverts the old tail rollback); B2 is the")
    print("head-scoped rollback (the actual v2 fix).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    _doc = __doc__ if __doc__ else ""
    ap = argparse.ArgumentParser(description=_doc.splitlines()[1] if _doc else "sgl#39342 patch")
    ap.add_argument("--check", action="store_true", help="read-only: report each edit's state (safe pre-flight gate)")
    ap.add_argument("--revert", action="store_true", help="invert applied edits back to pristine")
    ap.add_argument("--report", action="store_true", help="full per-edit table + version")
    ap.add_argument("--sb", help="override schedule_batch.py path (off-image test)")
    ap.add_argument("--mcm", help="override mamba_component.py path (off-image test)")
    args = ap.parse_args(argv)

    root_overrides = {}
    if args.sb:
        root_overrides["schedule_batch.py"] = args.sb
    if args.mcm:
        root_overrides["mamba_component.py"] = args.mcm

    if args.check:
        sys.exit(cmd_check(root_overrides))
    if args.revert:
        cmd_revert(root_overrides)
        return
    if args.report:
        cmd_report(root_overrides)
        return
    cmd_apply(root_overrides)


if __name__ == "__main__":
    main()
