#!/usr/bin/env bash
# test-baselines — guards for scripts/lib/profiles/baselines.yml + its
# registry-emit join (catalog-baselines slice 1).
#
# REDs on: schema violations · unknown slugs · ctx parity breaks (functional
# slugs: compose MAX_MODEL_LEN/CTX_SIZE default must equal registry max_ctx) ·
# a seeded slug missing its joined baseline in the --json contract.
# WARNs (never reds) on: pin-staleness (engine_pin != current pin) — pin bumps
# must not block on immediate re-bench; the debt just stays visible.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --- 1-3. schema + slug membership + ctx parity (pure python asserts) --------
python3 - <<'PY'
import re
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

doc = yaml.safe_load(Path("scripts/lib/profiles/baselines.yml").read_text())
assert doc.get("schema_version") == 1, "schema_version must be 1"
rows = doc.get("baselines") or {}
assert rows, "baselines.yml has no rows"

QUALITY_RE = re.compile(r"^\d{1,3}/150$")
# rig_class key format (slice 3): <count>x<card-token>-<interconnect-ish tail>,
# e.g. 2x3090-pcie / 2x5090-pcie / 1xdgx-spark. Format check, NOT a fixed set —
# a new GPU model must not require a guard edit.
RIG_CLASS_RE = re.compile(r"^\d+x[a-z0-9][a-z0-9-]*$")
errors = []


def check_row(where, row, submission=False):
    """Shared field validation — primary rows and submission rows carry the
    same measurement fields; submissions ADD required source/tier semantics."""
    for k in ("narr_tps", "code_tps"):
        if not isinstance(row.get(k), (int, float)):
            errors.append(f"{where}.{k}: required numeric")
    if not isinstance(row.get("date"), date):
        errors.append(f"{where}.date: required YYYY-MM-DD")
    for k in ("engine_pin", "rig", "submitted_by"):
        if not (isinstance(row.get(k), str) and row[k].strip()):
            errors.append(f"{where}.{k}: required non-empty string")
    pw = row.get("power_cap_w")
    if not (isinstance(pw, list) and pw and all(isinstance(x, int) for x in pw)):
        errors.append(f"{where}.power_cap_w: required list of ints")
    # tier (slice 3): primary rows are ALWAYS the local bar; submissions carry
    # the trust verdict (submitted = unverified, reproduced = independently re-run)
    tier = row.get("tier")
    if submission:
        if tier not in ("submitted", "reproduced"):
            errors.append(f"{where}.tier: must be submitted|reproduced")
        if not (isinstance(row.get("source"), str) and row["source"].strip()):
            errors.append(f"{where}.source: required non-empty string (disc/PR link)")
    elif tier != "local":
        errors.append(f"{where}.tier: primary rows must be tier: local")
    # optional, typed when present
    if "ttft_ms" in row and not isinstance(row["ttft_ms"], (int, float)):
        errors.append(f"{where}.ttft_ms: numeric")
    for k in ("quality_8pk", "quality_8pk_think_on"):
        if k in row and not QUALITY_RE.match(str(row[k])):
            errors.append(f"{where}.{k}: must look like 'P/150'")
    if "ctx_validated" in row:
        cv = row["ctx_validated"]
        ok = (isinstance(cv, dict) and isinstance(cv.get("tokens"), int)
              and isinstance(cv.get("niah"), str))
        if not ok:
            errors.append(f"{where}.ctx_validated: {{tokens: int, niah: str}}")
    if "prefill_tps" in row:
        pf = row["prefill_tps"]
        ok = (isinstance(pf, dict) and pf
              and all(isinstance(v, (int, float)) for v in pf.values()))
        if not ok:
            errors.append(f"{where}.prefill_tps: dict of numeric depth points (e.g. {{10k: N, 90k: M}})")
    if "source_tag" in row and not isinstance(row["source_tag"], str):
        errors.append(f"{where}.source_tag: string")


for slug, row in rows.items():
    where = f"baselines[{slug}]"
    if slug not in COMPOSE_REGISTRY:
        errors.append(f"{where}: unknown registry slug"); continue
    subs = row.get("submissions")
    has_primary = any(k in row for k in ("narr_tps", "code_tps"))
    if not has_primary and subs is None:
        errors.append(f"{where}: empty entry (needs a primary row and/or submissions)")
        continue
    if has_primary:
        check_row(where, row)
    if subs is not None:
        if not (isinstance(subs, dict) and subs):
            errors.append(f"{where}.submissions: must be a non-empty map keyed by rig_class")
        else:
            for rc, srow in subs.items():
                swhere = f"{where}.submissions[{rc}]"
                if not RIG_CLASS_RE.match(str(rc)):
                    errors.append(f"{swhere}: rig_class key must look like '2x5090-pcie'")
                if not isinstance(srow, dict):
                    errors.append(f"{swhere}: must be a row map"); continue
                check_row(swhere, srow, submission=True)
                if srow.get("rig") != rc:
                    errors.append(f"{swhere}.rig: must equal the rig_class key ({srow.get('rig')!r} != {rc!r})")

# ctx parity (functional slugs): compose ctx-env default == registry max_ctx.
CTX_RE = re.compile(r"\$\{(?:MAX_MODEL_LEN|CTX_SIZE|MAX_CTX)[^:}]*:-(\d+)\}")
for slug, e in COMPOSE_REGISTRY.items():
    if e["status"] not in ("production", "caveats"):
        continue
    try:
        txt = Path(e["compose_path"]).read_text()
    except OSError:
        errors.append(f"ctx-parity[{slug}]: compose unreadable: {e['compose_path']}")
        continue
    m = CTX_RE.search(txt)
    if m and int(m.group(1)) != e["max_ctx"]:
        errors.append(
            f"ctx-parity[{slug}]: compose default {m.group(1)} != registry max_ctx {e['max_ctx']}"
        )

if errors:
    print("test-baselines: FAIL", file=sys.stderr)
    for err in errors:
        print(f"  ✗ {err}", file=sys.stderr)
    sys.exit(1)
print(f"  ✓ schema + slug membership + ctx parity ({len(rows)} rows)")
PY

# --- 4-5. the join contract + staleness WARNs --------------------------------
# shellcheck source=/dev/null
source scripts/lib/registry-emit.sh
json="$(registry_variant_rows_json "$ROOT_DIR")"
# Env (not a pipe): the heredoc below owns the python interpreter's stdin —
# the same trick registry-emit itself uses for REGISTRY_TAB.
EMIT_JSON="$json" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

import yaml

d = json.loads(os.environ["EMIT_JSON"])
by_slug = {v["slug"]: v for v in d["variants"]}
rows = yaml.safe_load(Path("scripts/lib/profiles/baselines.yml").read_text())["baselines"]

missing = [s for s in rows if not (by_slug.get(s) or {}).get("baseline")]
if missing:
    print(f"test-baselines: FAIL — seeded slugs missing joined baseline: {missing}",
          file=sys.stderr)
    sys.exit(1)

# every joined row must carry the computed staleness verdict key
bad = [s for s in rows if "stale" not in by_slug[s]["baseline"]]
if bad:
    print(f"test-baselines: FAIL — joined rows missing 'stale': {bad}", file=sys.stderr)
    sys.exit(1)

# slice 3: submissions must ride the join with a per-submission staleness
# verdict (same mechanism as the primary row — the pin comparison is
# rig-independent)
sub_bad = []
for s in rows:
    for rc, srow in (by_slug[s]["baseline"].get("submissions") or {}).items():
        if "stale" not in srow:
            sub_bad.append(f"{s}[{rc}]")
if sub_bad:
    print(f"test-baselines: FAIL — joined submissions missing 'stale': {sub_bad}",
          file=sys.stderr)
    sys.exit(1)

stale = [s for s in rows if by_slug[s]["baseline"]["stale"] is True]
for s in stale:
    b = by_slug[s]["baseline"]
    print(f"  WARN: {s} baseline is STALE — measured on {b['engine_pin']!r}, "
          f"current pin {b['current_pin']!r} (re-bench owed; row stays, badge shows)")
print(f"  ✓ join contract ({len(rows)} joined, {len(stale)} stale-warned)")
PY

echo "test-baselines: ok"
