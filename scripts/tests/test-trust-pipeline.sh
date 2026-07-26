#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

# test-trust-pipeline.sh — v0.8.0 [F] STEP F4 (club-3090 #147).
#
# Contract test for CONTRACT-3 + CONTRACT-3a: the §6.2 inbound-trust
# pipeline raw->candidate->validated->tier1. The test IS the spec; the
# code is fixed to it. NO live Docker / GPU / network — fixture capture
# dirs are built in a tmp tree (byte-exact [E] schema, parsed via F1's
# read_capture_bundle for realism, classified via F2/F3's classify), and
# the F4 trust pipeline is run over them.
#
# Coverage (every CONTRACT-3 / CONTRACT-3a / brief F4 assertion as a
# failing-then-passing check):
#   * tampered submission_fingerprint -> stops at `raw`, reason
#     `fingerprint-mismatch` (never reaches candidate).
#   * valid fingerprint + all-green required smoke -> reaches
#     validated/tier1; graduation_set = exactly the green caps.
#   * `partial` anchor with streaming:unsmoked -> graduates ONLY green
#     caps; the unsmoked cap is NOT in graduation_set; cannot reach Tier-1
#     for it.
#   * a `red` cap -> never graduates that cap (the #145-class guard).
#   * single submission, no maintainer promotion, consensus_n=2 -> stops
#     at candidate (reason insufficient-consensus); + a 2nd matching
#     submission -> consensus reached -> validated; OR maintainer_promoted
#     alone -> validated.
#   * consensus-key DISCRIMINATION: differing on ANY of the 9 fields does
#     NOT count as consensus.
#   * CONTRACT-3a: a DERIVED anchor (no curated COMPOSE_REGISTRY backing),
#     otherwise valid -> stops at validated, reason
#     `derived-tier1-deferred-v0.8.1`, NO calibration row; a CURATED-
#     compose-backed anchor -> reaches tier1 and emits a calibration-row-
#     shaped record matching the compat.py schema.
#   * success-anchor validation requires NO predicted-vs-actual delta
#     (a clean success anchor validates with no delta input at all).
#   * the real on-disk capture(s) under .pull-captures/: promote() does
#     not crash; the re-derived fingerprint MATCHES the real [E]-minted
#     one (a strong real-data check — a MISMATCH is a loud finding).
#   * kv-calc untouched: importing trust_pipeline does not break
#     `tools/kv-calc.py --calibration` (still Overall: 22/22 (100%)).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles.loop_input import read_capture_bundle  # noqa: E402
from scripts.lib.profiles.classifier import classify  # noqa: E402
from scripts.lib.profiles.trust_pipeline import (  # noqa: E402
    TrustStage,
    promote,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Re-derive submission_fingerprint EXACTLY as [E] mints it
# (capture.py:671-680 8-tuple + capture.py:457-460 \x1f-join sha256
# hexdigest). The fixture builder uses this so fixtures carry a VALID
# fingerprint by construction (and we tamper it explicitly to test the
# mismatch path).
# ---------------------------------------------------------------------------
def mint_fingerprint(m: dict) -> str:
    parts = [
        m["model"],
        m["club3090_commit"],
        m["topology_summary_canonical"],
        str(m["quant_label"]),
        m["kv_calc_version"],
        str(m["engine_version"]),
        m["utc_ts"],
        m["outcome"],
    ]
    h = hashlib.sha256()
    h.update("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Byte-exact [E] schema fixtures (mirror scripts/lib/profiles/capture.py;
# parsed back through F1 for realism, same discipline as test-classifier.sh).
# ---------------------------------------------------------------------------
def mk_manifest(**over) -> dict:
    m = {
        "schema": 1,
        "slug": "Org/My-Model",
        "utc_ts": "20260517T000000Z",
        "submission_fingerprint": "PLACEHOLDER",
        "model": "Org/My-Model",
        "quant_label": "BFloat16",
        "arch_family": "LlamaForCausalLM",
        "topology_class": "1x24576MiB",
        "engine_pin": "vllm/vllm-openai:nightly-abc123",
        "engine_version": "vllm/vllm-openai:nightly-abc123",
        "kv_calc_version": "kvcalc-v0.8.0",
        "selected_ctx": 32768,
        "kv_format": "fp8_e5m2",
        "smoke_capability_set": ["plain-chat", "streaming"],
        "topology_summary_canonical": "[(NVIDIA GeForce RTX 3090, 24576)]",
        "model_id": "Org/My-Model",
        "failure_class": None,
        "club3090_commit": "cafef00d",
        "outcome": "ok",
        "capture_points": ["gate", "download", "boot", "smoke"],
    }
    m.update(over)
    # Mint a VALID fingerprint over the FINAL field values unless the
    # caller explicitly pinned one (the tamper test pins a bad value).
    if m["submission_fingerprint"] == "PLACEHOLDER":
        m["submission_fingerprint"] = mint_fingerprint(m)
    return m


def write_bundle(d: Path, *, manifest=None, pt1=None, pt3=None,
                 pt4=None, pt5=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    man = manifest if manifest is not None else mk_manifest()
    arts = {
        "manifest.json": man,
        "pt1-gate.json": pt1 if pt1 is not None else {
            "schema": 1, "point": "gate", "slug": man["slug"],
            "confidence": "estimated-lower-bound",
            "raw_verdict": "fits-clean", "terminal": "confirm→proceed",
            "profile_like": "vllm/minimal", "hardware_sm": 8.6,
            "predicted_b_breakdown": None,
        },
        "pt2-download.json": {
            "point": "download", "ok": True, "files": ["model.safetensors"],
            "bytes": 123, "sha_verified": True, "failure": None,
        },
        "pt3-boot.json": pt3 if pt3 is not None else {
            "point": "boot", "ok": True, "seconds": 81.9, "failure": None,
        },
        "pt4-smoke.json": pt4 if pt4 is not None else {
            "point": "smoke",
            "smoke_capability_set": ["plain-chat", "streaming"],
            "results": {"plain-chat": "green", "streaming": "green"},
            "partial": False, "results_detail": {},
        },
    }
    for name, obj in arts.items():
        (d / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    if pt5 is not None:
        (d / "pt5-override-capture.json").write_text(
            json.dumps(pt5, indent=2), encoding="utf-8")
    return d


tmp = Path(tempfile.mkdtemp())


def load(d: Path):
    fi = read_capture_bundle(d)
    return fi, classify(fi)


# ---------------------------------------------------------------------------
# 1. Tampered submission_fingerprint -> stop at `raw`,
#    reason `fingerprint-mismatch` (never reaches candidate).
# ---------------------------------------------------------------------------
bad = write_bundle(
    tmp / "tamper",
    manifest=mk_manifest(submission_fingerprint="deadbeef" * 8),
)
fi, cl = load(bad)
tr = promote(fi, cl, consensus_n=2)
check(tr.stage is TrustStage.RAW and tr.reason == "fingerprint-mismatch",
      f"tampered submission_fingerprint -> stage=raw reason="
      f"fingerprint-mismatch (got {tr.stage.value}/{tr.reason})")
check(tr.fingerprint_ok is False and not tr.at_least(TrustStage.CANDIDATE),
      "tampered fingerprint never reaches candidate (fingerprint_ok=False)")

# ---------------------------------------------------------------------------
# 2. Valid fingerprint + all-green required smoke + maintainer promotion
#    -> reaches validated; graduation_set = exactly the green caps.
#    (This anchor is DERIVED -> stops at validated, see test 7 for the
#     CONTRACT-3a reason; here we assert the graduation set + validated.)
# ---------------------------------------------------------------------------
ok = write_bundle(tmp / "allgreen")
fi, cl = load(ok)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check(tr.fingerprint_ok is True,
      "valid (re-derived) submission_fingerprint -> fingerprint_ok=True")
check(tr.graduation_set == frozenset({"plain-chat", "streaming"}),
      f"graduation_set = exactly the green caps "
      f"(got {sorted(tr.graduation_set)})")
check(tr.at_least(TrustStage.VALIDATED),
      "valid fingerprint + all-green + maintainer promotion -> "
      ">= validated")

# ---------------------------------------------------------------------------
# 3. `partial` anchor with streaming:unsmoked -> graduates ONLY the green
#    cap; the unsmoked cap is NOT in graduation_set; cannot Tier-1 for it.
# ---------------------------------------------------------------------------
partial = write_bundle(
    tmp / "partial",
    manifest=mk_manifest(outcome="partial"),
    pt4={
        "point": "smoke",
        "smoke_capability_set": ["plain-chat"],
        "results": {"plain-chat": "green", "streaming": "unsmoked"},
        "partial": True, "results_detail": {},
    },
)
fi, cl = load(partial)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check("plain-chat" in tr.graduation_set
      and "streaming" not in tr.graduation_set,
      "partial anchor: ONLY the green cap graduates; the unsmoked cap is "
      "NOT in graduation_set (the #145-class capability-scoped rule)")
check(tr.graduation_set == frozenset({"plain-chat"}),
      f"partial graduation_set is exactly {{plain-chat}} "
      f"(got {sorted(tr.graduation_set)})")

# ---------------------------------------------------------------------------
# 4. A `red` cap -> never graduates that cap (the #145-class guard).
# ---------------------------------------------------------------------------
red = write_bundle(
    tmp / "redcap",
    manifest=mk_manifest(outcome="failed"),
    pt4={
        "point": "smoke",
        "smoke_capability_set": ["plain-chat", "streaming"],
        "results": {"plain-chat": "green", "streaming": "red"},
        "partial": False,
        "results_detail": {"streaming": {"status": 500, "error": "boom"}},
    },
)
fi, cl = load(red)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check("streaming" not in tr.graduation_set
      and tr.graduation_set == frozenset({"plain-chat"}),
      "a `red` cap NEVER graduates (only the green cap is in "
      "graduation_set) — the #145-class guard")

# An anchor with NO green capability at all cannot be a success anchor for
# anything -> stops at candidate.
allred = write_bundle(
    tmp / "allred",
    manifest=mk_manifest(outcome="failed"),
    pt4={
        "point": "smoke", "smoke_capability_set": ["plain-chat"],
        "results": {"plain-chat": "red"}, "partial": False,
        "results_detail": {"plain-chat": {"status": 500, "error": "x"}},
    },
)
fi, cl = load(allred)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check(tr.stage is TrustStage.CANDIDATE
      and tr.reason == "no-green-capability",
      f"no green capability -> stops at candidate "
      f"(got {tr.stage.value}/{tr.reason})")

# ---------------------------------------------------------------------------
# 5. Consensus: single submission, no maintainer -> stops at candidate
#    (insufficient-consensus). +2nd matching -> validated. maintainer
#    alone -> validated.
# ---------------------------------------------------------------------------
solo = write_bundle(tmp / "solo")
fi, cl = load(solo)
tr = promote(fi, cl, consensus_n=2)  # 1/2, no maintainer
check(tr.stage is TrustStage.CANDIDATE
      and tr.reason == "insufficient-consensus"
      and tr.consensus_count == 1 and tr.consensus_reached is False,
      f"single submission, no maintainer, n=2 -> candidate / "
      f"insufficient-consensus (got {tr.stage.value}/{tr.reason}, "
      f"count={tr.consensus_count})")

# A 2nd, byte-identical submission -> its consensus_key() matches -> 2/2.
solo2 = write_bundle(tmp / "solo2")
fi2, _ = load(solo2)
tr = promote(fi, cl, prior_submissions=[fi2], consensus_n=2)
check(tr.consensus_count == 2 and tr.consensus_reached is True
      and tr.at_least(TrustStage.VALIDATED),
      f"a 2nd matching submission -> consensus 2/2 -> >= validated "
      f"(got count={tr.consensus_count}, stage={tr.stage.value})")

# maintainer_promoted=True ALONE (no consensus) -> validated.
tr = promote(fi, cl, consensus_n=2, maintainer_promoted=True)
check(tr.at_least(TrustStage.VALIDATED) and tr.consensus_reached is False,
      "maintainer_promoted=True alone (no consensus) -> >= validated")

# A raw consensus-key tuple is also a valid prior (the matching primitive
# is decoupled from storage — F4 ships the primitive, not a daemon).
tr = promote(fi, cl, prior_submissions=[fi2.consensus_key()],
             consensus_n=2)
check(tr.consensus_count == 2,
      "a raw consensus-key tuple counts as a matching prior submission")

# ---------------------------------------------------------------------------
# 6. Consensus-key DISCRIMINATION: differ on ANY of the 9 fields -> NOT
#    consensus (materially-different runs can't accidentally agree).
# ---------------------------------------------------------------------------
diffs = {
    "model": "Org/Different-Model",
    "quant_label": "awq",
    "arch_family": "Qwen3MoeForCausalLM",
    "topology_class": "2x24576MiB",
    "engine_pin": "vllm/vllm-openai:nightly-zzz999",
    "kv_calc_version": "kvcalc-v0.8.0+deadbeefcafe",
    "selected_ctx": 65536,
    "kv_format": "turboquant_3bit_nc",
    "smoke_capability_set": ["plain-chat"],
}
base = write_bundle(tmp / "disc_base")
fi_base, cl_base = load(base)
for i, (field, val) in enumerate(diffs.items()):
    over = {field: val}
    # engine_pin/engine_version travel together (consensus uses engine_pin).
    if field == "engine_pin":
        over["engine_version"] = val
    other = write_bundle(tmp / f"disc_{i}", manifest=mk_manifest(**over))
    fi_o, _ = load(other)
    tr = promote(fi_base, cl_base, prior_submissions=[fi_o], consensus_n=2)
    check(tr.consensus_count == 1 and tr.consensus_reached is False,
          f"consensus discrimination: differing on `{field}` does NOT "
          f"count as consensus (count stayed {tr.consensus_count})")

# ---------------------------------------------------------------------------
# 7. CONTRACT-3a: DERIVED anchor (no curated COMPOSE_REGISTRY backing),
#    otherwise fully valid -> stops at validated, reason
#    `derived-tier1-deferred-v0.8.1`, NO calibration row.
# ---------------------------------------------------------------------------
derived = write_bundle(
    tmp / "derived",
    manifest=mk_manifest(model="Qwen/Qwen2.5-0.5B-Instruct",
                         model_id="Qwen/Qwen2.5-0.5B-Instruct",
                         slug="Qwen/Qwen2.5-0.5B-Instruct"),
)
fi, cl = load(derived)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check(tr.stage is TrustStage.VALIDATED
      and tr.reason == "derived-tier1-deferred-v0.8.1"
      and tr.curated is False,
      f"DERIVED anchor -> stops at validated / "
      f"derived-tier1-deferred-v0.8.1, curated=False "
      f"(got {tr.stage.value}/{tr.reason}/curated={tr.curated})")
check(tr.calibration_row is None,
      "DERIVED anchor emits NO calibration row (CONTRACT-3a (b): "
      "classifier+dedup only this phase)")
check(not tr.at_least(TrustStage.TIER1),
      "DERIVED anchor never reaches Tier-1")

# ---------------------------------------------------------------------------
# 8. CONTRACT-3a: CURATED-compose-backed anchor -> reaches tier1 and emits
#    a calibration-row-shaped record matching the compat.py schema.
#    `Lorbus/Qwen3.6-27B-int4-AutoRound` is a real hf_repos slug of the
#    curated catalog model `qwen3.6-27b`, which has COMPOSE_REGISTRY
#    entries (e.g. `vllm/default`). profile_like names a real curated key.
# ---------------------------------------------------------------------------
curated = write_bundle(
    tmp / "curated",
    manifest=mk_manifest(model="Lorbus/Qwen3.6-27B-int4-AutoRound",
                         model_id="Lorbus/Qwen3.6-27B-int4-AutoRound",
                         slug="Lorbus/Qwen3.6-27B-int4-AutoRound",
                         quant_label="autoround-int4",
                         kv_format="turboquant_3bit_nc",
                         selected_ctx=48000),
    pt1={
        "schema": 1, "point": "gate",
        "slug": "Lorbus/Qwen3.6-27B-int4-AutoRound",
        "confidence": "exact", "raw_verdict": "fits-clean",
        "terminal": "proceed", "profile_like": "vllm/default",
        "hardware_sm": 8.6, "predicted_b_breakdown": None,
    },
)
fi, cl = load(curated)
tr = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
check(tr.curated is True,
      f"CURATED anchor: model slug resolves to a catalog model with a "
      f"COMPOSE_REGISTRY entry (curated={tr.curated})")
check(tr.stage is TrustStage.TIER1 and tr.reason == "tier1-curated",
      f"CURATED + validated -> reaches TIER1 (got "
      f"{tr.stage.value}/{tr.reason})")
row = tr.calibration_row
EXPECTED_ROW_KEYS = {
    "compose", "vram_gb", "measured_peak_gb", "ctx_override",
    "status", "engine_pin", "genesis_pin", "source",
}
check(isinstance(row, dict) and set(row.keys()) == EXPECTED_ROW_KEYS,
      f"calibration_row matches the compat.py calibration/<model>.yml "
      f"row schema (keys={sorted(row.keys()) if row else None})")
check(row["compose"] in {"vllm/default"} or (
        row["compose"] and row["compose"].startswith("vllm/")),
      f"calibration_row.compose is a real COMPOSE_REGISTRY key "
      f"(got {row['compose']!r})")
check(row["status"] == "candidate-tier1" and row["measured_peak_gb"] is None,
      "calibration_row is status=candidate-tier1 (NEVER 'active') and "
      "measured_peak_gb is None (F4 never benchmarks / writes YAML / "
      "runs kv-calc)")
check(row["ctx_override"] == 48000 and abs(row["vram_gb"] - 24.0) < 0.01,
      f"calibration_row carries the anchor's selected_ctx + per-card "
      f"vram_gb (ctx_override={row['ctx_override']}, "
      f"vram_gb={row['vram_gb']})")

# Cross-check the shape is consumable by compat: every key compat's
# calibration_status reads (status/compose/vram_gb/ctx_override) present.
from scripts.lib.profiles.compat import (  # noqa: E402
    CalibrationProfile, Profiles, calibration_status, load_profiles,
)
prof = load_profiles()
prof_with = Profiles(
    hardware=prof.hardware, models=prof.models,
    workloads=prof.workloads, engines=prof.engines,
    drafters=prof.drafters,
    calibration=dict(prof.calibration),
)
check(all(k in row for k in ("status", "compose", "vram_gb",
                             "ctx_override")),
      "calibration_row carries every field compat.calibration_status "
      "reads (status/compose/vram_gb/ctx_override)")

# ---------------------------------------------------------------------------
# 9. Success-anchor validation requires NO predicted-vs-actual delta.
#    A clean success anchor validates with NO delta input at all
#    (CONTRACT-3 critical clarification: delta is the §6.1 kv-calc-bug
#    branch, NOT success validation). `tolerance` is accepted-and-ignored.
# ---------------------------------------------------------------------------
clean = write_bundle(tmp / "nodelta")  # pt3 ok, no pt5, no delta anywhere
fi, cl = load(clean)
tr_nodelta = promote(fi, cl, maintainer_promoted=True, consensus_n=2)
tr_tol = promote(fi, cl, maintainer_promoted=True, consensus_n=2,
                 tolerance=0.05)
check(tr_nodelta.at_least(TrustStage.VALIDATED),
      "clean success anchor validates with NO delta input at all "
      "(success validation = topology + (consensus OR maintainer))")
check(tr_tol.stage is tr_nodelta.stage
      and tr_tol.reason == tr_nodelta.reason,
      "passing a `tolerance` has NO effect on a success-anchor verdict "
      "(accepted-and-ignored — delta is the §6.1 branch, not §6.2 success)")

# ---------------------------------------------------------------------------
# 10. REAL on-disk capture(s) under .pull-captures/: promote() does not
#     crash; the re-derived fingerprint MATCHES the real [E]-minted one.
#     (A MISMATCH here is a LOUD real-data finding — not papered over.)
# ---------------------------------------------------------------------------
real_root = root / ".pull-captures"
real_dirs: list[Path] = []
if real_root.is_dir():
    for slug_dir in sorted(real_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(slug_dir.iterdir()):
            if ts_dir.is_dir() and (ts_dir / "manifest.json").is_file():
                real_dirs.append(ts_dir)

if not real_dirs:
    print("SKIP: no real on-disk .pull-captures/ bundle "
          "(graceful — not a failure)")
else:
    for rd in real_dirs:
        try:
            rfi = read_capture_bundle(rd)
            rcl = classify(rfi)
            rtr = promote(rfi, rcl, consensus_n=2)
        except Exception as exc:
            check(False, f"REAL capture {rd} promote() crashed: {exc!r}")
            continue
        claimed = rfi.manifest.get("submission_fingerprint")
        match = (rtr.rederived_fingerprint == claimed)
        check(match,
              f"REAL [E] capture {rd.name}: F4 re-derived "
              f"submission_fingerprint MATCHES the [E]-minted one "
              f"(model={rfi.model_id!r}; "
              f"{'OK' if match else 'MISMATCH — LOUD FINDING'})")
        check(rtr.stage in set(TrustStage),
              f"REAL [E] capture {rd.name}: promote() returns a valid "
              f"stage (got {rtr.stage.value}/{rtr.reason})")

# ---------------------------------------------------------------------------
if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll F4 §6.2 inbound-trust pipeline (CONTRACT-3 + 3a) "
      "assertions passed.")
PY

echo "test-trust-pipeline.sh OK"
