#!/usr/bin/env bash
set -euo pipefail

# test-kvcalc-version.sh — v0.8.0 [F] STEP F6 (club-3090 #147).
#
# CONTRACT-5 (iii) — the MANDATORY G2 regression. Asserts the
# `kv_calc_version` that [E] stamps into every §6.2 capture manifest is a
# real FUNCTION of the kv-calc content surface, not a hand-bumped literal:
#
#   kv_calc_version = "kvcalc-v0.8.0+" + sha256(<content>)[:12]
#   <content> = bytes(tools/kv-calc.py)
#               + sorted-concat(scripts/lib/profiles/calibration/*.yml)
#
# (a) shape: `kvcalc-v0.8.0+<12 lowercase hex>`
# (b) STABLE across repeated imports/derivations with unchanged content
# (c) CHANGES when `tools/kv-calc.py` content changes (the kv-calc MATH)
# (d) CHANGES when a `calibration/*.yml` is added / edited (the CORPUS)
# (e) the bare-fallback path yields exactly `"kvcalc-v0.8.0"` when the
#     content surface is unreadable (honest degrade, never crash [E]/pull)
#
# (c)/(d) are the load-bearing assertions: if they cannot DISTINGUISH a
# content change the test is worthless. They are made real by copying the
# REAL repo content surface into an isolated tmp tree, perturbing the copy
# only, recomputing over the copy, and asserting the digest moves. The
# REAL repo files are NEVER mutated.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))

from scripts.lib.profiles import pull  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


BASE = "kvcalc-v0.8.0"
SHAPE = re.compile(r"^kvcalc-v0\.8\.0\+[0-9a-f]{12}$")

# ---------------------------------------------------------------------------
# (a) shape — the module-level cached constant + a fresh derivation.
# ---------------------------------------------------------------------------
v_const = pull._KV_CALC_VERSION
check(isinstance(v_const, str) and SHAPE.match(v_const) is not None,
      f"(a) _KV_CALC_VERSION is `kvcalc-v0.8.0+<12 lowercase hex>` "
      f"(got {v_const!r})")
check(pull._KV_CALC_VERSION_BASE == BASE,
      f"(a) base label is exactly {BASE!r} (got "
      f"{pull._KV_CALC_VERSION_BASE!r})")

v_derived = pull._derive_kv_calc_version(ROOT)
check(SHAPE.match(v_derived) is not None,
      f"(a) _derive_kv_calc_version(REPO) has the same shape "
      f"(got {v_derived!r})")
digest = pull._kv_calc_content_hash(ROOT)
check(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{12}", digest)
      is not None,
      f"(a) _kv_calc_content_hash(REPO) is 12 lowercase hex "
      f"(got {digest!r})")

# ---------------------------------------------------------------------------
# (b) STABLE across repeated derivations with unchanged content.
# ---------------------------------------------------------------------------
runs = [pull._derive_kv_calc_version(ROOT) for _ in range(5)]
check(len(set(runs)) == 1 and runs[0] == v_const == v_derived,
      f"(b) derivation is STABLE across 5 calls + equals the cached "
      f"constant (got {sorted(set(runs))})")
hruns = [pull._kv_calc_content_hash(ROOT) for _ in range(5)]
check(len(set(hruns)) == 1,
      f"(b) content hash is STABLE across repeated reads "
      f"(got {sorted(set(hruns))})")


# ---------------------------------------------------------------------------
# Helper: build an ISOLATED tmp mirror of just the content surface so the
# hash function can be exercised over a perturbed copy WITHOUT touching the
# real repo. Layout matches what _kv_calc_content_hash() reads:
#   <tmp>/tools/kv-calc.py
#   <tmp>/scripts/lib/profiles/calibration/*.yml
# ---------------------------------------------------------------------------
def _mirror(tmp: Path) -> Path:
    (tmp / "tools").mkdir(parents=True, exist_ok=True)
    cal = tmp / "scripts" / "lib" / "profiles" / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools" / "kv-calc.py", tmp / "tools" / "kv-calc.py")
    src_cal = ROOT / "scripts" / "lib" / "profiles" / "calibration"
    for f in sorted(src_cal.glob("*.yml")):
        shutil.copy2(f, cal / f.name)
    return tmp


with tempfile.TemporaryDirectory() as _td:
    base_root = _mirror(Path(_td) / "base")
    h_base = pull._kv_calc_content_hash(base_root)

    # Sanity: an untouched faithful mirror reproduces the REAL repo hash
    # (proves the function reads exactly the documented content surface).
    check(h_base == pull._kv_calc_content_hash(ROOT),
          f"mirror sanity: faithful copy reproduces the real repo content "
          f"hash (mirror {h_base!r} == repo "
          f"{pull._kv_calc_content_hash(ROOT)!r})")
    check(pull._derive_kv_calc_version(base_root) == v_const,
          "mirror sanity: faithful copy reproduces the cached "
          "_KV_CALC_VERSION")

    # -----------------------------------------------------------------------
    # (c) CHANGES when tools/kv-calc.py content changes (kv-calc MATH).
    #     Append one byte to the COPY only; the real file is untouched.
    # -----------------------------------------------------------------------
    c_root = _mirror(Path(_td) / "c")
    kvp = c_root / "tools" / "kv-calc.py"
    kvp.write_bytes(kvp.read_bytes() + b"\n# F6 regression perturbation\n")
    h_c = pull._kv_calc_content_hash(c_root)
    v_c = pull._derive_kv_calc_version(c_root)
    check(h_c is not None and h_c != h_base,
          f"(c) a tools/kv-calc.py content change MOVES the content hash "
          f"({h_base!r} -> {h_c!r}) — the version is a real function of "
          f"the kv-calc math, NOT a constant")
    check(SHAPE.match(v_c) is not None and v_c != v_const,
          f"(c) the derived version string CHANGES with kv-calc.py "
          f"(got {v_c!r}, was {v_const!r})")

    # -----------------------------------------------------------------------
    # (d) CHANGES when a calibration/*.yml is added OR edited (CORPUS).
    # -----------------------------------------------------------------------
    # (d1) EDIT an existing corpus file in the copy.
    d1_root = _mirror(Path(_td) / "d1")
    d1_cal = d1_root / "scripts" / "lib" / "profiles" / "calibration"
    some_yml = sorted(d1_cal.glob("*.yml"))[0]
    some_yml.write_bytes(some_yml.read_bytes()
                         + b"\n# F6 corpus edit perturbation\n")
    h_d1 = pull._kv_calc_content_hash(d1_root)
    v_d1 = pull._derive_kv_calc_version(d1_root)
    check(h_d1 is not None and h_d1 != h_base and h_d1 != h_c,
          f"(d) EDITING a calibration/*.yml MOVES the content hash "
          f"({h_base!r} -> {h_d1!r}) — corpus is in the hash")
    check(SHAPE.match(v_d1) is not None and v_d1 != v_const,
          f"(d) the derived version CHANGES on a corpus edit "
          f"(got {v_d1!r})")

    # (d2) ADD a new corpus file to the copy.
    d2_root = _mirror(Path(_td) / "d2")
    d2_cal = d2_root / "scripts" / "lib" / "profiles" / "calibration"
    (d2_cal / "zzz-f6-new-model.yml").write_bytes(b"# new calibration row\n")
    h_d2 = pull._kv_calc_content_hash(d2_root)
    v_d2 = pull._derive_kv_calc_version(d2_root)
    check(h_d2 is not None and h_d2 != h_base,
          f"(d) ADDING a calibration/*.yml MOVES the content hash "
          f"({h_base!r} -> {h_d2!r}) — add/remove is detected")
    check(SHAPE.match(v_d2) is not None and v_d2 != v_const,
          f"(d) the derived version CHANGES when a corpus file is added "
          f"(got {v_d2!r})")

    # Distinctness: math-change vs corpus-edit vs corpus-add are 3 distinct
    # hashes — no dimension silently collides (the G2 defect, removed).
    check(len({h_base, h_c, h_d1, h_d2}) == 4,
          f"(c)/(d) math-change / corpus-edit / corpus-add yield DISTINCT "
          f"hashes (no collision): "
          f"{sorted({h_base, h_c, h_d1, h_d2})}")

# ---------------------------------------------------------------------------
# (e) bare-fallback: unreadable content surface -> exactly "kvcalc-v0.8.0",
#     and the hash helper returns None (never raises — [E]/pull must not
#     crash over a provenance-label refinement).
# ---------------------------------------------------------------------------
missing = Path("/nonexistent-f6-kvcalc-root-zzz")
check(pull._kv_calc_content_hash(missing) is None,
      "(e) _kv_calc_content_hash returns None when content unreadable "
      "(no exception escapes)")
check(pull._derive_kv_calc_version(missing) == BASE,
      f"(e) _derive_kv_calc_version degrades to the bare {BASE!r} when "
      f"content unreadable (got "
      f"{pull._derive_kv_calc_version(missing)!r})")

# An empty dir (paths absent, no exception) also degrades cleanly.
with tempfile.TemporaryDirectory() as _et:
    empty = Path(_et)
    check(pull._kv_calc_content_hash(empty) is None
          and pull._derive_kv_calc_version(empty) == BASE,
          f"(e) empty repo root -> None hash + bare {BASE!r} "
          f"(robust degrade, never crash)")

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll F6 CONTRACT-5 (iii) kv_calc_version regression assertions "
      "passed (a-e).")
PY

echo "test-kvcalc-version.sh OK"
