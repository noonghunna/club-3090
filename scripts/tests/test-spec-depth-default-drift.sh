#!/usr/bin/env bash
# Drift-guard for the speculative-decoding depth default inside a compose.
#
# A compose that ships a drafter states its depth default in up to three places,
# and they are edited independently:
#
#   (a) the ARG the engine actually receives   ->  - '${SPEC_N:-<n>}'
#   (b) the entrypoint's gate + banner default ->  _spec_n="$${SPEC_N:-<n>}"
#   (c) the header Toggle: line                ->  depth (default <n>)
#
# Only (a) changes behaviour. (b) prints "[spec] drafter ON (SPEC_N=<n>)" and is
# what an operator reads back to confirm what booted; (c) is what they read
# before booting. When they disagree the compose lies about itself in the two
# places anyone actually looks.
#
# This is not hypothetical: the GLM dual composes carried a MEASURED n=2 in (b)
# while (a) still handed the engine 3, so a ~+8-10% decode win sat in the tree
# for days without ever reaching a default boot, and the banner claimed the win
# was live. Caught by hand 2026-09-08; this guard is why it cannot repeat.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) — repo sources are full of unicode and a
# genuine non-UTF-8 locale otherwise crashes the read (#779). See
# test-compose-status-drift.sh for the full rationale; guarded by
# test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import re, sys
from pathlib import Path

ARG  = re.compile(r"-\s*'\$\{SPEC_N:-(\d+)\}'")
EP   = re.compile(r'_spec_n="\$\$\{SPEC_N:-(\d+)\}"')
HDR  = re.compile(r"depth \(default (\d+)\)")

bad, checked = [], 0
for f in sorted(Path("models").rglob("*.yml")):
    text = f.read_text(encoding="utf-8")
    if "SPEC_N" not in text:
        continue
    arg = set(ARG.findall(text))
    ep  = set(EP.findall(text))
    hdr = set(HDR.findall(text))
    if not arg and not ep:
        continue          # documents the knob without shipping a default
    checked += 1

    # Within one site, two different defaults is always wrong.
    for name, vals in (("arg", arg), ("entrypoint", ep), ("header", hdr)):
        if len(vals) > 1:
            bad.append(f"{f}: {name} declares MORE THAN ONE default: {sorted(vals)}")

    # Across sites they must agree. The arg is authoritative — it is the only
    # one the engine reads — so report the others as drifted FROM it.
    if len(arg) == 1:
        truth = next(iter(arg))
        if len(ep) == 1 and next(iter(ep)) != truth:
            bad.append(
                f"{f}: engine gets SPEC_N={truth} but the banner announces "
                f"{next(iter(ep))} (_spec_n default)"
            )
        if len(hdr) == 1 and next(iter(hdr)) != truth:
            bad.append(
                f"{f}: engine gets SPEC_N={truth} but the header documents "
                f"default {next(iter(hdr))}"
            )

# A guard that silently checks nothing is worse than no guard: the composes it
# targets could be renamed away and this would still pass.
if checked < 5:
    print(f"FAIL: only {checked} compose(s) carried a SPEC_N default — "
          f"expected the drafter composes to be found. Bad path or regex drift?")
    sys.exit(1)

if bad:
    print(f"FAIL: spec-depth default drift in {len(bad)} place(s):")
    for b in bad:
        print(f"  - {b}")
    print("\nThe ARG is what the engine receives; make the banner and header match it")
    print("(or change the arg, if the OTHER two are the intended value).")
    sys.exit(1)

print(f"OK: spec-depth defaults agree across arg/banner/header in {checked} compose(s)")
PY
