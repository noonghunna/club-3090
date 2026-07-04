#!/usr/bin/env bash
# catalog-baseline.sh — induct a validated rebench run as a slug's SHIPPED
# baseline row in scripts/lib/profiles/baselines.yml (catalog-baselines §2.3).
#
#   bash scripts/catalog-baseline.sh <slug> --from-tag <rebench-tag> [options]
#
# One mechanism, three entry points: ⑤ Promote (c3 producer lane) · maintainer
# CLI · the rebench-full completion prompt (re-validation after pin bumps).
#
# It VALIDATES the tag dir covers the gates (verify-full pass · bench n>=5 ·
# quality run present), extracts the display projection (decode TPS · TTFT ·
# 8-pack arms · ctx_validated with the NIAH verdict · provenance), UPSERTS the
# YAML row textually (comments preserved), and prints a unified diff for
# review. It NEVER commits — the row ships via PR (design §2: PR-reviewed).
#
# Options:
#   --from-tag <tag>     rebench tag under results/rebench/ (required)
#   --tag-dir <dir>      explicit tag dir (overrides --from-tag lookup; for
#                        tags living in another checkout/worktree)
#   --engine-pin <spec>  override the measured pin (default: the slug's CURRENT
#                        resolved pin — correct when inducting right after the
#                        gate ran on it)
#   --rig <class>        hardware fingerprint class (default: derived from
#                        nvidia-smi, e.g. 2x3090-pcie)
#   --power-cap-w <csv>  per-card caps, e.g. "370,420" (default: nvidia-smi)
#   --submitted-by <who> provenance (default: $USER)
#   --tps-only           induct without a quality run (WARNS; the 8-pack gate
#                        is normally required)
#   --dry-run            print the diff, do not write
#   --baselines-file <f> target YAML (default scripts/lib/profiles/baselines.yml;
#                        tests point this at a copy)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLUG="${1:-}"
[[ -n "$SLUG" && "$SLUG" != --* ]] || {
  echo "usage: catalog-baseline.sh <slug> --from-tag <rebench-tag> [--dry-run] (see header)" >&2
  exit 2
}
shift

TAG="" TAG_DIR="" ENGINE_PIN="" RIG="" POWER="" SUBMITTED_BY="${USER:-unknown}"
TPS_ONLY=0 DRY_RUN=0 BASELINES_FILE="scripts/lib/profiles/baselines.yml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-tag)     TAG="$2"; shift 2 ;;
    --tag-dir)      TAG_DIR="$2"; shift 2 ;;
    --engine-pin)   ENGINE_PIN="$2"; shift 2 ;;
    --rig)          RIG="$2"; shift 2 ;;
    --power-cap-w)  POWER="$2"; shift 2 ;;
    --submitted-by) SUBMITTED_BY="$2"; shift 2 ;;
    --tps-only)     TPS_ONLY=1; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --baselines-file) BASELINES_FILE="$2"; shift 2 ;;
    *) echo "[catalog-baseline] unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TAG" || -n "$TAG_DIR" ]] || { echo "[catalog-baseline] --from-tag (or --tag-dir) is required" >&2; exit 2; }
TAG_DIR="${TAG_DIR:-$ROOT_DIR/results/rebench/$TAG}"
TAG="${TAG:-$(basename "$TAG_DIR")}"

SLUG="$SLUG" TAG="$TAG" TAG_DIR="$TAG_DIR" ENGINE_PIN="$ENGINE_PIN" RIG="$RIG" \
POWER="$POWER" SUBMITTED_BY="$SUBMITTED_BY" TPS_ONLY="$TPS_ONLY" DRY_RUN="$DRY_RUN" \
BASELINES_FILE="$BASELINES_FILE" \
python3 - <<'PY'
import difflib
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, ".")
from scripts.lib.profiles.compat import load_profiles  # noqa: E402
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402
from scripts.lib.profiles.launch_compat import ProfileError, resolve_variant_pin  # noqa: E402

slug = os.environ["SLUG"]
tag = os.environ["TAG"]
tag_dir = Path(os.environ["TAG_DIR"])
tps_only = os.environ["TPS_ONLY"] == "1"
dry_run = os.environ["DRY_RUN"] == "1"


def die(msg: str) -> None:
    print(f"[catalog-baseline] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def warn(msg: str) -> None:
    print(f"[catalog-baseline] WARN: {msg}", file=sys.stderr)


if slug not in COMPOSE_REGISTRY:
    die(f"unknown registry slug {slug!r}")
if not tag_dir.is_dir():
    die(f"tag dir not found: {tag_dir}")

# ── gate validation (refuse-loud; the row must stand on a full gate) ─────────
verify_log = tag_dir / "verify-full.log"
if not (verify_log.is_file() and "All checks passed" in verify_log.read_text(errors="replace")):
    die(f"verify-full gate not covered: {verify_log} missing or not all-pass")

internal = tag_dir / "_internal.json"
if not internal.is_file():
    die(f"{internal} missing — run rebench-report.py over the tag dir first")
rec = json.loads(internal.read_text())
bench = rec.get("bench") or {}
narr, code = bench.get("narrative") or {}, bench.get("code") or {}
if narr.get("decode_tps_mean") is None or code.get("decode_tps_mean") is None:
    die("bench gate not covered: no decode TPS means in _internal.json")
bench_log = (tag_dir / "bench.log").read_text(errors="replace") if (tag_dir / "bench.log").is_file() else ""
n_runs = len(re.findall(r"^\s*run-\d+", bench_log, re.M)) // 2 if bench_log else 0
if bench_log and n_runs < 3:
    die(f"bench gate not covered: {n_runs} measured runs per prompt (< 3, the protocol minimum)")
if bench_log and n_runs < 5:
    warn(f"bench ran n={n_runs} (below the n=5 canonical target — RUNS=5 on the next gate); "
         "the row comment records the actual n")


def _quality_from(path: Path):
    if not path.is_file():
        return None
    try:
        q = json.loads(path.read_text())
        packs = q.get("packs") or []
        p = sum(int(x.get("passed") or 0) for x in packs)
        t = sum(int(x.get("total") or 0) for x in packs)
        return f"{p}/{t}" if t else None
    except (ValueError, TypeError):
        return None


q_off = _quality_from(tag_dir / "quality-full.json")
q_on = _quality_from(tag_dir / "quality-full-thinking.json")
if not q_off and not q_on:
    if not tps_only:
        die("quality gate not covered: no quality-full*.json in the tag dir "
            "(re-run with --tps-only to induct a TPS-only row, WARNED)")
    warn("inducting WITHOUT a quality run (--tps-only) — 8pk stays empty")

# ── ctx_validated: the verify-stress ceiling ladder verdict ──────────────────
ctx_tokens = None
niah = None
stress_text = ""
stress = tag_dir / "verify-stress.log"
if stress.is_file():
    stress_text = stress.read_text(errors="replace")
    m = re.search(r"all (\d+) rungs passed — fillable to (\d+) tok", stress_text)
    if m:
        ctx_tokens = int(m.group(2))
        niah = f"clean@{round(ctx_tokens / 1000)}K"
    elif "All stress / boundary checks passed" in stress_text:
        niah = "passed (no ceiling-ladder line parsed)"
else:
    warn("no verify-stress.log — ctx_validated omitted")

# ── prefill probe (catalog-baselines 2c) + anchor calibration ─────────────────
# Reuse THE parser (measurement_record.parse_bench_output) — one grammar for
# bench output, never a second regex set here.
prefill_by_ctx = {}
ttft_by_ctx = {}
if bench_log:
    from scripts.lib.profiles.measurement_record import parse_bench_output

    bm = parse_bench_output(bench_log)
    prefill_by_ctx = dict(bm.prefill_tps_by_ctx)
    ttft_by_ctx = dict(bm.ttft_ms_by_ctx)
    # Anchor calibration: the probe's DEEP warm anchor vs the NIAH ladder's
    # nearest rung measure the same point — agreement certifies the ladder's
    # whole depth curve; divergence is itself a finding (design §2.1.1).
    if prefill_by_ctx and stress_text:
        rungs = [
            (int(a), float(p))
            for a, p in re.findall(r"rung \d+/\d+: .*?actual=(\d+)K tok.*?prefill=([0-9.]+) t/s", stress_text)
        ]
        deep = max(((int(d), v) for d, v in prefill_by_ctx.items()), default=None)
        if deep and rungs:
            depth_k = deep[0] / 1000
            nearest = min(rungs, key=lambda r: abs(r[0] - depth_k))
            if abs(nearest[0] - depth_k) <= 20:  # comparable depths only
                ratio = deep[1] / nearest[1] if nearest[1] else 0
                if 0.7 <= ratio <= 1.3:
                    print(f"[catalog-baseline] anchor calibration OK: probe {deep[1]:.0f} t/s @{depth_k:.0f}K "
                          f"vs ladder {nearest[1]:.0f} t/s @{nearest[0]}K (ratio {ratio:.2f}) — depth curve certified",
                          file=sys.stderr)
                else:
                    warn(f"anchor DIVERGENCE: probe {deep[1]:.0f} t/s @{depth_k:.0f}K vs ladder "
                         f"{nearest[1]:.0f} t/s @{nearest[0]}K (ratio {ratio:.2f}, tolerance 0.7-1.3) — "
                         "warm-vs-cold or config drift; investigate before trusting the depth curve")

# ── provenance ────────────────────────────────────────────────────────────────
engine_pin = os.environ["ENGINE_PIN"]
if not engine_pin:
    try:
        exports = resolve_variant_pin(load_profiles(), slug)
        if "VLLM_NIGHTLY_SHA" not in exports:
            engine_pin = next(iter(exports.values()))
    except ProfileError:
        pass
    if not engine_pin:
        # compose-image default (ik/llama.cpp class) — same rule as the emit join
        txt = (Path(COMPOSE_REGISTRY[slug]["compose_path"])).read_text(errors="replace")
        m = re.search(r"^\s*image:\s*[\"']?(?:\$\{[A-Z_0-9]+:-)?([^\s}\"']+)\}?", txt, re.M)
        engine_pin = m.group(1) if m else ""
if not engine_pin:
    die("could not resolve the engine pin — pass --engine-pin explicitly")

rig = os.environ["RIG"]
power = os.environ["POWER"]
if not rig or not power:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        names = [l.split(",")[0].strip() for l in out]
        caps = [str(round(float(l.split(",")[1]))) for l in out]
        if not rig and names:
            short = re.sub(r"NVIDIA GeForce RTX\s*", "", names[0]).strip().replace(" ", "").lower()
            rig = f"{len(names)}x{short}-pcie"
        if not power and caps:
            power = ",".join(caps)
    except Exception:
        pass
if not rig:
    die("could not derive --rig (nvidia-smi unavailable) — pass it explicitly")
if not power:
    die("could not derive --power-cap-w — pass it explicitly")
power_list = "[" + ", ".join(x.strip() for x in power.split(",")) + "]"

ttft = narr.get("ttft_ms_mean")
row_date = date.fromtimestamp(
    max(f.stat().st_mtime for f in tag_dir.iterdir())
).isoformat()

# ── build the row text (matches the file's hand-written shape) ───────────────
lines = [f"  {slug}:"]
lines.append(f"    # inducted by catalog-baseline.sh from rebench tag {tag} ({datetime.now().date().isoformat()});")
lines.append(f"    # evidence: verify-full pass · bench n={n_runs or '?'} · "
             + ("quality both arms" if (q_off and q_on) else ("quality one arm" if (q_off or q_on) else "TPS-ONLY (no quality)"))
             + (" · NIAH ladder" if ctx_tokens else ""))
lines.append(f"    narr_tps: {round(float(narr['decode_tps_mean']), 2)}")
lines.append(f"    code_tps: {round(float(code['decode_tps_mean']), 2)}")
if ttft is not None:
    lines.append(f"    ttft_ms: {round(float(ttft))}")
if prefill_by_ctx:
    inner = ", ".join(
        f"{int(k) // 1000}k: {v:.0f}"
        for k, v in sorted(prefill_by_ctx.items(), key=lambda x: int(x[0]))
    )
    lines.append(f"    prefill_tps: {{ {inner} }}")
if q_off:
    lines.append(f'    quality_8pk: "{q_off}"')
if q_on:
    lines.append(f'    quality_8pk_think_on: "{q_on}"')
if ctx_tokens:
    lines.append(f'    ctx_validated: {{ tokens: {ctx_tokens}, niah: "{niah}" }}')
lines.append(f"    date: {row_date}")
lines.append(f'    engine_pin: "{engine_pin}"')
lines.append(f'    rig: "{rig}"')
lines.append(f"    power_cap_w: {power_list}")
lines.append(f'    source_tag: "{tag}"')
lines.append(f'    submitted_by: "{os.environ["SUBMITTED_BY"]}"')
row_text = "\n".join(lines) + "\n"

# ── upsert (textual splice — comments elsewhere in the file preserved) ────────
bl_path = Path(os.environ["BASELINES_FILE"])
old = bl_path.read_text()
block_re = re.compile(
    rf"^  {re.escape(slug)}:\n(?:^(?:    .*|\s*)\n)*?(?=^  \S|^#|^\S|\Z)", re.M
)
if block_re.search(old):
    new = block_re.sub(row_text + "\n", old, count=1)
    action = "replaced"
else:
    # append before the top-level wave-2 footer comment block (or at EOF)
    m = re.search(r"^# ─+\n# SEED WAVE 2", old, re.M)
    ins = m.start() if m else len(old)
    new = old[:ins] + row_text + "\n" + old[ins:]
    action = "added"

# self-check: the produced file must still parse + the row must round-trip
import yaml  # noqa: E402

parsed = yaml.safe_load(new)
got = (parsed.get("baselines") or {}).get(slug)
assert got and isinstance(got.get("narr_tps"), (int, float)), "upsert self-check failed"

diff = "".join(
    difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile="baselines.yml", tofile=f"baselines.yml ({action} {slug})",
    )
)
print(diff or "[catalog-baseline] no change (row identical)")
if dry_run:
    print(f"\n[catalog-baseline] DRY RUN — {action} row for {slug} NOT written")
else:
    bl_path.write_text(new)
    print(f"\n[catalog-baseline] {action} baseline row for {slug} (from {tag}).")
    print("[catalog-baseline] next: review the diff, run bash scripts/tests/test-baselines.sh, PR it.")
PY
