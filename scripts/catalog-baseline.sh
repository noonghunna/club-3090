#!/usr/bin/env bash
# catalog-baseline.sh — induct a validated rebench run as a slug's SHIPPED
# baseline row in scripts/lib/profiles/baselines.yml (catalog-baselines §2.3).
#
#   bash scripts/catalog-baseline.sh <slug> --from-tag <rebench-tag> [options]
#   bash scripts/catalog-baseline.sh <slug> --from-bundle <tgz|dir> \
#        --source <disc/PR link> --submitted-by <handle> [options]
#
# One mechanism, three entry points: ⑤ Promote (c3 producer lane) · maintainer
# CLI · the rebench-full completion prompt (re-validation after pin bumps).
# PLUS the slice-3 cross-rig path: --from-bundle ingests a VOLUNTEER's rebench
# bundle into the slug's `submissions:` map (keyed by rig_class, tier:
# submitted). In bundle mode provenance comes FROM THE BUNDLE — rig/power from
# rig.txt, engine pin from container-config.json — NEVER from this rig
# (nvidia-smi / resolve_variant_pin would stamp OUR fingerprint onto foreign
# numbers). --source and --submitted-by are REQUIRED (no $USER default).
#
# It VALIDATES the tag dir covers the gates (verify-full pass · bench n>=3,
# n<5 warned · quality run present), extracts the display projection (decode
# TPS · TTFT · 8-pack arms · prefill depth points · ctx_validated with the
# NIAH verdict · provenance), UPSERTS the YAML row textually (comments
# preserved), and prints a unified diff for review. It NEVER commits — the
# row ships via PR (design §2: PR-reviewed).
#
# READS THE TAG ARTIFACTS AS TRUTH: if a harness fix rescored saved results,
# MATERIALIZE the rescore first (`benchlocal-cli rescore <json> --in-place`)
# or this tool inducts the pre-fix numbers — docs/QUALITY_TEST.md "Rescoring
# saved results".
#
# Options:
#   --from-tag <tag>     rebench tag under results/rebench/ (required unless
#                        --from-bundle; WITH --from-bundle it selects the tag
#                        when the bundle carries more than one)
#   --tag-dir <dir>      explicit tag dir (overrides --from-tag lookup; for
#                        tags living in another checkout/worktree)
#   --from-bundle <p>    a volunteer's rebench bundle (.tgz or an extracted
#                        dir) → ingest into the slug's submissions: map
#   --source <url>      (bundle mode, REQUIRED) provenance link — the
#                        discussion comment / PR the bundle was submitted in
#   --engine-pin <spec>  override the measured pin (default: the slug's CURRENT
#                        resolved pin; bundle mode: container-config.json)
#   --rig <class>        hardware fingerprint class (default: derived from
#                        nvidia-smi, e.g. 2x3090-pcie; bundle mode: rig.txt.
#                        In bundle mode this is also the submissions map key)
#   --power-cap-w <csv>  per-card caps, e.g. "370,420" (default: nvidia-smi;
#                        bundle mode: rig.txt)
#   --submitted-by <who> provenance (default: $USER; bundle mode: REQUIRED —
#                        the volunteer's handle, never defaulted)
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

TAG="" TAG_DIR="" BUNDLE="" SOURCE="" ENGINE_PIN="" RIG="" POWER=""
SUBMITTED_BY="" TPS_ONLY=0 DRY_RUN=0
BASELINES_FILE="scripts/lib/profiles/baselines.yml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-tag)     TAG="$2"; shift 2 ;;
    --tag-dir)      TAG_DIR="$2"; shift 2 ;;
    --from-bundle)  BUNDLE="$2"; shift 2 ;;
    --source)       SOURCE="$2"; shift 2 ;;
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

FROM_BUNDLE=0
if [[ -n "$BUNDLE" ]]; then
  # ── bundle mode (slice 3): extract + locate the tag dir INSIDE the bundle ──
  FROM_BUNDLE=1
  [[ -n "$SOURCE" ]] || { echo "[catalog-baseline] bundle mode: --source <disc/PR link> is required" >&2; exit 2; }
  [[ -n "$SUBMITTED_BY" ]] || { echo "[catalog-baseline] bundle mode: --submitted-by <handle> is required (never defaulted to \$USER)" >&2; exit 2; }
  [[ -z "$TAG_DIR" ]] || { echo "[catalog-baseline] --tag-dir and --from-bundle are mutually exclusive" >&2; exit 2; }
  if [[ -d "$BUNDLE" ]]; then
    BUNDLE_DIR="$BUNDLE"
  else
    [[ -f "$BUNDLE" ]] || { echo "[catalog-baseline] bundle not found: $BUNDLE" >&2; exit 2; }
    BUNDLE_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUNDLE_DIR"' EXIT
    tar xzf "$BUNDLE" -C "$BUNDLE_DIR"
  fi
  # a tag dir is any dir carrying _internal.json (the rebench-report record)
  mapfile -t _tag_dirs < <(find "$BUNDLE_DIR" -name _internal.json -printf '%h\n' | sort)
  if [[ ${#_tag_dirs[@]} -eq 0 ]]; then
    echo "[catalog-baseline] no tag dir (_internal.json) inside the bundle" >&2; exit 1
  elif [[ ${#_tag_dirs[@]} -eq 1 ]]; then
    TAG_DIR="${_tag_dirs[0]}"
  else
    [[ -n "$TAG" ]] || { echo "[catalog-baseline] bundle carries ${#_tag_dirs[@]} tags — select one with --from-tag:" >&2
                         printf '  %s\n' "${_tag_dirs[@]##*/}" >&2; exit 2; }
    TAG_DIR=""
    for d in "${_tag_dirs[@]}"; do [[ "$(basename "$d")" == "$TAG" ]] && TAG_DIR="$d"; done
    [[ -n "$TAG_DIR" ]] || { echo "[catalog-baseline] tag $TAG not in the bundle" >&2; exit 2; }
  fi
  TAG="${TAG:-$(basename "$TAG_DIR")}"
else
  [[ -n "$TAG" || -n "$TAG_DIR" ]] || { echo "[catalog-baseline] --from-tag (or --tag-dir / --from-bundle) is required" >&2; exit 2; }
  SUBMITTED_BY="${SUBMITTED_BY:-${USER:-unknown}}"
  TAG_DIR="${TAG_DIR:-$ROOT_DIR/results/rebench/$TAG}"
  TAG="${TAG:-$(basename "$TAG_DIR")}"
fi

SLUG="$SLUG" TAG="$TAG" TAG_DIR="$TAG_DIR" ENGINE_PIN="$ENGINE_PIN" RIG="$RIG" \
POWER="$POWER" SUBMITTED_BY="$SUBMITTED_BY" TPS_ONLY="$TPS_ONLY" DRY_RUN="$DRY_RUN" \
FROM_BUNDLE="$FROM_BUNDLE" SOURCE="$SOURCE" \
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
# n from the canonical summary headers ONLY — a raw run-line count also picks
# up the prefill-probe blocks' run lines and over-states n (found on the first
# probe-enabled induction: n=5 read as n=8)
_ns = re.findall(r"^=== summary \[(?:narrative|code)\] \(n=(\d+)\) ===$", bench_log, re.M)
n_runs = min(int(x) for x in _ns) if _ns else 0
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
# Two regimes: local induction reads THIS rig (nvidia-smi + resolved pin);
# bundle mode reads THE BUNDLE (rig.txt + container-config.json) and NEVER
# falls back to local state — that would stamp our fingerprint onto foreign
# numbers (the slice-3 trust boundary).
from_bundle = os.environ["FROM_BUNDLE"] == "1"
source = os.environ["SOURCE"]

engine_pin = os.environ["ENGINE_PIN"]
rig = os.environ["RIG"]
power = os.environ["POWER"]

if from_bundle:
    if not engine_pin:
        cc = tag_dir / "container-config.json"
        if cc.is_file():
            try:
                data = json.loads(cc.read_text(errors="replace"))
                engine_pin = ((data[0].get("Config") or {}).get("Image") or "").strip()
            except (ValueError, IndexError, AttributeError, TypeError):
                pass
        if not engine_pin:
            die("bundle mode: no readable Config.Image in container-config.json — pass --engine-pin")
    rig_txt = (tag_dir / "rig.txt").read_text(errors="replace") if (tag_dir / "rig.txt").is_file() else ""
    names = re.findall(r"^GPU \d+: (.+?) \(UUID", rig_txt, re.M)
    if not rig:
        if not names:
            die("bundle mode: no GPU lines in rig.txt — pass --rig")
        short = re.sub(r"NVIDIA\s+(GeForce\s+)?(RTX\s+)?", "", names[0]).strip().replace(" ", "").lower()
        rig = f"{len(names)}x{short}-pcie"
    if not power:
        caps = re.findall(r"^power_cap_w:\s*([\d.]+)", rig_txt, re.M)
        if not caps:
            die("bundle mode: no power_cap_w in rig.txt — pass --power-cap-w")
        vals = [str(round(float(c))) for c in caps]
        if len(vals) == 1 and len(names) > 1:
            vals = vals * len(names)  # one global cap line → replicate per card
        power = ",".join(vals)
else:
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
# The SAME fields serve both shapes; only indent + head + tier/source differ:
#   local:  "  <slug>:"        + 4-space fields + tier: local
#   bundle: "      <rig>:"     + 8-space fields + source: + tier: submitted
#           (spliced under the slug's "    submissions:" map)
tool = "catalog-baseline.sh --from-bundle" if from_bundle else "catalog-baseline.sh"
fields = [f"# inducted by {tool} from rebench tag {tag} ({datetime.now().date().isoformat()});"]
fields.append(f"# evidence: verify-full pass · bench n={n_runs or '?'} · "
              + ("quality both arms" if (q_off and q_on) else ("quality one arm" if (q_off or q_on) else "TPS-ONLY (no quality)"))
              + (" · NIAH ladder" if ctx_tokens else ""))
fields.append(f"narr_tps: {round(float(narr['decode_tps_mean']), 2)}")
fields.append(f"code_tps: {round(float(code['decode_tps_mean']), 2)}")
if ttft is not None:
    fields.append(f"ttft_ms: {round(float(ttft))}")
if prefill_by_ctx:
    inner = ", ".join(
        f"{int(k) // 1000}k: {v:.0f}"
        for k, v in sorted(prefill_by_ctx.items(), key=lambda x: int(x[0]))
    )
    fields.append(f"prefill_tps: {{ {inner} }}")
if q_off:
    fields.append(f'quality_8pk: "{q_off}"')
if q_on:
    fields.append(f'quality_8pk_think_on: "{q_on}"')
if ctx_tokens:
    fields.append(f'ctx_validated: {{ tokens: {ctx_tokens}, niah: "{niah}" }}')
fields.append(f"date: {row_date}")
fields.append(f'engine_pin: "{engine_pin}"')
fields.append(f'rig: "{rig}"')
fields.append(f"power_cap_w: {power_list}")
if from_bundle:
    fields.append(f'source: "{source}"')
fields.append(f'source_tag: "{tag}"')
fields.append(f'submitted_by: "{os.environ["SUBMITTED_BY"]}"')
fields.append("tier: submitted" if from_bundle else "tier: local")

if from_bundle:
    row_text = f"      {rig}:\n" + "".join(f"        {f}\n" for f in fields)
else:
    row_text = f"  {slug}:\n" + "".join(f"    {f}\n" for f in fields)

# ── upsert (textual splice — comments elsewhere in the file preserved) ────────
bl_path = Path(os.environ["BASELINES_FILE"])
old = bl_path.read_text()
block_re = re.compile(
    rf"^  {re.escape(slug)}:\n(?:^(?:    .*|\s*)\n)*?(?=^  \S|^#|^\S|\Z)", re.M
)
# an existing submissions: sub-map inside a slug block (6+-space content lines)
SUBMAP_RE = re.compile(
    r"^    submissions:\n(?:^(?:      .*|\s*)\n)*?(?=^    \S|^  \S|^#|^\S|\Z)", re.M
)
m_block = block_re.search(old)

if from_bundle:
    # bundle mode touches ONLY the submissions map — the primary row (if any)
    # is the local bar and stays byte-identical.
    if m_block:
        block = m_block.group(0)
        body = block.rstrip("\n") + "\n"          # block content
        trail = block[len(body):]                 # blank-line row separator(s)
        sub_re = re.compile(
            rf"^      {re.escape(rig)}:\n(?:^(?:        .*|\s*)\n)*?(?=^      \S|^    \S|^  \S|^#|^\S|\Z)",
            re.M,
        )
        m_hdr = re.search(r"^    submissions:\n", body, re.M)
        if m_hdr:
            m_sub = sub_re.search(body)
            if m_sub:
                # ONE row per rig_class — newest replaces; older stays in git
                new_body = body[:m_sub.start()] + row_text + body[m_sub.end():]
                action = f"replaced submission [{rig}] in"
            else:
                new_body = body[:m_hdr.end()] + row_text + body[m_hdr.end():]
                action = f"added submission [{rig}] to"
        else:
            new_body = body + "    submissions:\n" + row_text
            action = f"added submissions map [{rig}] to"
        new = old[:m_block.start()] + new_body + trail + old[m_block.end():]
    else:
        # submission-only entry: a slug measured on hardware we don't have
        entry_text = (
            f"  {slug}:\n"
            "    # cross-rig submissions only — no local baseline row yet\n"
            "    submissions:\n" + row_text
        )
        m = re.search(r"^# ─+\n# (?:KNOWN GAPS|SEED WAVE 2)", old, re.M)
        ins = m.start() if m else len(old)
        new = old[:ins] + entry_text + "\n" + old[ins:]
        action = f"added submission-only entry [{rig}] as"
else:
    if m_block:
        # preserve an existing submissions: sub-map across the primary-row
        # replace (the block regex would otherwise swallow it)
        m_sub = SUBMAP_RE.search(m_block.group(0))
        keep = (m_sub.group(0).rstrip("\n") + "\n") if m_sub else ""
        new = old[:m_block.start()] + row_text + keep + "\n" + old[m_block.end():]
        action = "replaced"
    else:
        # append before the top-level gap-list footer comment block (or at EOF)
        m = re.search(r"^# ─+\n# (?:KNOWN GAPS|SEED WAVE 2)", old, re.M)
        ins = m.start() if m else len(old)
        new = old[:ins] + row_text + "\n" + old[ins:]
        action = "added"

# self-check: the produced file must still parse + the row must round-trip
import yaml  # noqa: E402

parsed = yaml.safe_load(new)
got = (parsed.get("baselines") or {}).get(slug)
if from_bundle:
    sub_got = (got or {}).get("submissions", {}).get(rig)
    assert sub_got and isinstance(sub_got.get("narr_tps"), (int, float)), "upsert self-check failed (submission)"
    assert sub_got.get("tier") == "submitted" and sub_got.get("source"), "upsert self-check failed (provenance)"
else:
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
