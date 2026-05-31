#!/usr/bin/env bash
set -euo pipefail

# test-loop-input.sh — v0.8.0 [F] STEP F1 (club-3090 #147).
#
# Contract test for CONTRACT-1: the `FInput` capture-bundle reader. The
# test IS the spec; the code is fixed to it. NO live Docker / GPU /
# network — fixture capture dirs are built in a tmp tree (mirroring the
# byte-exact `[E]` schema from scripts/lib/profiles/capture.py) and parsed.
#
# Coverage (every CONTRACT-1 / brief F1 assertion as a failing-then-passing
# check):
#   * a well-formed bundle (pt1-4 + manifest, NO pt5) loads; fields
#     accessible; pt5_override is None; raw_bundle_path defaults None and
#     is NOT synthesized by [E].
#   * a bundle WITH pt5 loads; pt5_override is the dict.
#   * forward-compat: future-additive pt1.predicted_b_breakdown /
#     pt3.failure_log_excerpt / pt3.actual present -> STILL loads.
#   * schema violations raise CaptureBundleError: manifest.schema != 1,
#     a missing required ptN, a wrong `point`.
#   * key normalization: quant_label LOWERCASED; arch_family VERBATIM;
#     model_id / engine_version alias accessors return the right manifest
#     values.
#   * consensus_key() = the §6.2 9-tuple (exact order/fields);
#     dedup_tuple() = the §6.3 7-tuple; dedup_hash() = sha256[:12].
#   * failure_class surfaced as None; outcome surfaced raw but NO accessor
#     claims it is the §6.1 class enum; a non-null [E] failure_class is
#     rejected.
#   * runs against a REAL on-disk .pull-captures/ dir if any exist
#     (skips gracefully if none).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles.loop_input import (  # noqa: E402
    BaseCaptureBundle,
    CaptureBundleError,
    FInput,
    FInputGate,
    read_capture_bundle,
    read_gate_bundle,
)

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


def raises(fn, msg: str) -> None:
    try:
        fn()
    except CaptureBundleError:
        print(f"PASS: {msg}")
    except Exception as exc:  # wrong exception type is a failure
        print(f"FAIL: {msg} (raised {type(exc).__name__}, not "
              f"CaptureBundleError)", file=sys.stderr)
        failures.append(msg)
    else:
        print(f"FAIL: {msg} (did not raise)", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Byte-exact [E] schema fixtures (mirror scripts/lib/profiles/capture.py).
# ---------------------------------------------------------------------------
def mk_manifest(**over) -> dict:
    m = {
        "schema": 1,
        "slug": "Org/My-Model",
        "utc_ts": "20260517T000000Z",
        "submission_fingerprint": "deadbeef" * 8,
        "model": "Org/My-Model",
        "quant_label": "BFloat16",          # mixed case ON PURPOSE
        "arch_family": "LlamaForCausalLM",  # used VERBATIM
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
        "outcome": "partial",
        "capture_points": ["gate", "download", "boot", "smoke"],
    }
    m.update(over)
    return m


def write_bundle(d: Path, *, manifest=None, pt1=None, pt3=None,
                 pt5=None, drop=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    arts = {
        "manifest.json": manifest if manifest is not None else mk_manifest(),
        "pt1-gate.json": pt1 if pt1 is not None else {
            "schema": 1, "point": "gate", "slug": "Org/My-Model",
            "confidence": "estimated-lower-bound",
            "raw_verdict": "fits-clean", "terminal": "confirm→proceed",
            "profile_like": "vllm/minimal", "hardware_sm": 8.6,
        },
        "pt2-download.json": {
            "point": "download", "ok": True, "files": ["model.safetensors"],
            "bytes": 123, "sha_verified": True, "failure": None,
        },
        "pt3-boot.json": pt3 if pt3 is not None else {
            "point": "boot", "ok": True, "seconds": 81.9, "failure": None,
        },
        "pt4-smoke.json": {
            "point": "smoke",
            "smoke_capability_set": ["plain-chat", "streaming"],
            "results": {"plain-chat": "green", "streaming": "green"},
            "partial": True, "results_detail": {},
        },
    }
    if drop:
        arts.pop(drop, None)
    for name, obj in arts.items():
        (d / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    if pt5 is not None:
        (d / "pt5-override-capture.json").write_text(
            json.dumps(pt5, indent=2), encoding="utf-8")
    return d


tmp = Path(tempfile.mkdtemp())

# ---------------------------------------------------------------------------
# 1. Well-formed bundle (no pt5) loads; fields accessible; pt5 None.
# ---------------------------------------------------------------------------
b1 = write_bundle(tmp / "b1")
fi = read_capture_bundle(b1)
check(isinstance(fi, FInput), "well-formed bundle -> FInput")
check(fi.pt5_override is None,
      "no pt5 file -> pt5_override is None")
check(fi.raw_bundle_path is None,
      "raw_bundle_path defaults None (NOT produced/synthesized by [E])")
check(fi.manifest["submission_fingerprint"] == "deadbeef" * 8
      and fi.pt2_download["ok"] is True
      and fi.pt3_boot["seconds"] == 81.9
      and fi.pt4_smoke["partial"] is True
      and fi.pt1_gate["raw_verdict"] == "fits-clean",
      "all required pt1-4 + manifest fields accessible")

# raw_bundle_path is ONLY ever the explicit caller value.
fi_attach = read_capture_bundle(b1, raw_bundle_path="/some/report.md")
check(fi_attach.raw_bundle_path == Path("/some/report.md"),
      "raw_bundle_path is exactly the explicit caller value when supplied")

# ---------------------------------------------------------------------------
# 2. Bundle WITH pt5 loads; pt5_override is the dict.
# ---------------------------------------------------------------------------
pt5_obj = {
    "point": "override_capture",
    "predicted_b_breakdown": {"weights": 1.0, "kv": 0.5},
    "actual": {"boot_peak_mib": 23000, "gpu_worker_reported_mib": 22800},
    "predicted_vs_actual_delta_mib": 500,
    "exit_error_summary": None,
    "calibration_signal_not_validated": True,
}
b2 = write_bundle(tmp / "b2", pt5=pt5_obj)
fi2 = read_capture_bundle(b2)
check(fi2.pt5_override == pt5_obj,
      "pt5 present -> pt5_override is the parsed dict")

# ---------------------------------------------------------------------------
# 3. Forward-compat: future-additive pt1/pt3 keys still load.
# ---------------------------------------------------------------------------
b3 = write_bundle(
    tmp / "b3",
    pt1={
        "schema": 1, "point": "gate", "slug": "Org/My-Model",
        "confidence": "estimated-lower-bound", "raw_verdict": "wont-fit",
        "terminal": "override-accepted", "profile_like": "vllm/minimal",
        "hardware_sm": 8.6,
        "predicted_b_breakdown": {"weights": 2.0},  # F3/G6-A-i additive
    },
    pt3={
        "point": "boot", "ok": False, "seconds": 0.0,
        "failure": "server did not become ready before timeout",
        "failure_log_excerpt": "torch.cuda.OutOfMemoryError ...",  # F3 A-ii
        "actual": {"attempted_alloc_mib": 1234,                    # F3 A-ii'
                   "gpu_worker_reported_mib": 23456},
    },
)
fi3 = read_capture_bundle(b3)
check(fi3.pt1_gate["predicted_b_breakdown"] == {"weights": 2.0}
      and fi3.pt3_boot["actual"]["attempted_alloc_mib"] == 1234
      and fi3.pt3_boot["failure_log_excerpt"].startswith("torch.cuda"),
      "forward-additive pt1.predicted_b_breakdown / pt3.failure_log_excerpt "
      "/ pt3.actual present -> STILL loads (no rejection of future keys)")

# ---------------------------------------------------------------------------
# 4. Schema / shape violations raise CaptureBundleError.
# ---------------------------------------------------------------------------
b4a = write_bundle(tmp / "b4a", manifest=mk_manifest(schema=2))
raises(lambda: read_capture_bundle(b4a),
       "manifest.schema != 1 raises CaptureBundleError")

b4b = write_bundle(tmp / "b4b", drop="pt3-boot.json")
raises(lambda: read_capture_bundle(b4b),
       "missing required pt3-boot.json raises CaptureBundleError")

b4c = write_bundle(tmp / "b4c", drop="manifest.json")
raises(lambda: read_capture_bundle(b4c),
       "missing required manifest.json raises CaptureBundleError")

b4d = write_bundle(tmp / "b4d", pt3={
    "point": "BOGUS", "ok": True, "seconds": 1.0, "failure": None})
raises(lambda: read_capture_bundle(b4d),
       "wrong pt3 `point` value raises CaptureBundleError")

b4e = write_bundle(tmp / "b4e",
                   manifest=mk_manifest(failure_class="genuine-oom"))
raises(lambda: read_capture_bundle(b4e),
       "non-null [E] manifest.failure_class raises CaptureBundleError "
       "([F] classifies, not [E])")

raises(lambda: read_capture_bundle(tmp / "does-not-exist"),
       "non-existent capture dir raises CaptureBundleError")

# ---------------------------------------------------------------------------
# 5. Key normalization + alias accessors.
# ---------------------------------------------------------------------------
fi = read_capture_bundle(b1)
check(fi.quant_label == "bfloat16",
      f"quant_label LOWERCASED for keying (raw 'BFloat16' -> "
      f"{fi.quant_label!r})")
check(fi.arch_family == "LlamaForCausalLM",
      f"arch_family used VERBATIM, NOT re-normalized (got "
      f"{fi.arch_family!r})")
check(fi.model_id == "Org/My-Model" == fi.manifest["model"],
      "model_id alias accessor == manifest['model']")
check(fi.engine_version == "vllm/vllm-openai:nightly-abc123"
      == fi.manifest["engine_pin"],
      "engine_version alias accessor == manifest['engine_pin']")

# ---------------------------------------------------------------------------
# 6. consensus_key() = §6.2 9-tuple; dedup_tuple() = §6.3 7-tuple.
# ---------------------------------------------------------------------------
ck = fi.consensus_key()
check(isinstance(ck, tuple) and len(ck) == 9,
      f"consensus_key() is the §6.2 9-tuple (len={len(ck)})")
check(ck == (
    "Org/My-Model", "bfloat16", "LlamaForCausalLM", "1x24576MiB",
    "vllm/vllm-openai:nightly-abc123", "kvcalc-v0.8.0", 32768,
    "fp8_e5m2", ("plain-chat", "streaming")),
    "consensus_key() exact order/fields/normalization (§6.2)")
check(ck[1] == "bfloat16" and ck[2] == "LlamaForCausalLM",
      "consensus_key() pos1 quant lowercased, pos2 arch verbatim")

dt = fi.dedup_tuple()
check(isinstance(dt, tuple) and len(dt) == 7,
      f"dedup_tuple() is the §6.3 7-tuple (len={len(dt)})")
check(dt == (
    "Org/My-Model", "bfloat16", "LlamaForCausalLM", "kvcalc-v0.8.0",
    "vllm/vllm-openai:nightly-abc123", None, "1x24576MiB"),
    "dedup_tuple() exact order/fields/normalization (§6.3)")
check(dt[5] is None,
      "dedup_tuple() pos5 failure_class is None on an [E] bundle")

dh = fi.dedup_hash()
check(isinstance(dh, str) and len(dh) == 12
      and all(c in "0123456789abcdef" for c in dh),
      f"dedup_hash() is sha256[:12] hex (got {dh!r})")

# ---------------------------------------------------------------------------
# 7. failure_class surfaced None; outcome raw, NOT a §6.1 class enum.
# ---------------------------------------------------------------------------
check(fi.failure_class is None,
      "failure_class surfaced as None ([F] computes it later, not [E])")
check(fi.outcome == "partial",
      "outcome raw value surfaced (interim [E] 3-state)")
# Binding rule: there must be NO accessor claiming `outcome` is the §6.1
# class. Assert no method/property name asserts a class taxonomy on it.
api = set(dir(fi))
check("failure_class" in api and "outcome" in api,
      "FInput exposes raw `failure_class` + `outcome` accessors")
check(not any("class_enum" in n or "six_one_class" in n
              or "classify" in n for n in api),
      "no FInput accessor claims `outcome` is the §6.1 class enum "
      "(binding rule: [F] derives failure_class itself downstream)")

# ---------------------------------------------------------------------------
# 7b. v0.8.2 CONTRACT-1.1 — `BaseCaptureBundle` protocol + schema==1
#     byte-identity (the V1 RED-LINE) + `read_gate_bundle()` (schema==2).
# ---------------------------------------------------------------------------
import hashlib as _hl  # noqa: E402

# (a) The protocol-lift is byte-identity-preserving: the SAME schema==1
#     bundle yields a byte-identical FInput + identical dedup_tuple +
#     dedup_hash + consensus_key PRE/POST (the F3-class risk). We assert
#     determinism (two reads of the same bytes are identical) AND that
#     FInput satisfies BaseCaptureBundle BY CONSTRUCTION.
fi_a = read_capture_bundle(b1)
fi_b = read_capture_bundle(b1)
check(fi_a.manifest == fi_b.manifest
      and fi_a.dedup_tuple() == fi_b.dedup_tuple()
      and fi_a.dedup_hash() == fi_b.dedup_hash()
      and fi_a.consensus_key() == fi_b.consensus_key(),
      "V1 RED-LINE: a real schema==1 bundle -> byte-identical FInput + "
      "identical dedup_tuple/dedup_hash/consensus_key (protocol lift is "
      "a pure static retype)")
# The literal expected dedup_hash for b1's manifest — pinned so a future
# refactor that perturbs the schema==1 serialization fails LOUDLY here.
_b1_dt = ("Org/My-Model", "bfloat16", "LlamaForCausalLM", "kvcalc-v0.8.0",
          "vllm/vllm-openai:nightly-abc123", None, "1x24576MiB")
_b1_h = _hl.sha256("\x1f".join(str(p) for p in _b1_dt)
                   .encode("utf-8")).hexdigest()[:12]
check(fi_a.dedup_hash() == _b1_h,
      f"V1 RED-LINE: schema==1 dedup_hash is the pinned byte-exact value "
      f"{_b1_h} (no drift from the protocol lift)")
check(isinstance(fi_a, BaseCaptureBundle),
      "FInput satisfies BaseCaptureBundle (runtime_checkable Protocol — "
      "by construction; no isinstance(finput, FInput) anywhere in F2/F5)")
check(fi_a.is_gate_only is False,
      "FInput.is_gate_only defaults False (a schema==1 full bundle is "
      "NEVER gate-only; additive default => schema==1 byte-identical)")

# (b) read_gate_bundle(): a schema==2 gate-only bundle -> FInputGate. It
#     validates ONLY the always-present row + outcome=='hard-block' +
#     failure_class is None; it MUST NOT reuse the 22-key validator.
def write_gate(d: Path, *, manifest=None, pt1=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    gm = manifest if manifest is not None else {
        "schema": 2, "slug": "Org/Gate-Model",
        "utc_ts": "20260518T000000Z", "club3090_commit": "cafef00d",
        "outcome": "hard-block",
        "abort_reason": "engine-support-unknown/no-arch-row",
        "failure_class": None,
        "model": "Org/Gate-Model", "model_id": "Org/Gate-Model",
        "arch_family": None, "quant_label": None,
        "topology_class": None, "topology_summary_canonical": None,
        "selected_ctx": None, "kv_format": None,
        "smoke_capability_set": None, "engine_pin": None,
        "engine_version": None, "kv_calc_version": None,
        "submission_fingerprint": None, "is_gate_only": True,
        "capture_points": ["gate"],
    }
    gp = pt1 if pt1 is not None else {
        "schema": 2, "point": "gate", "slug": "Org/Gate-Model",
        "confidence": "ESTIMATED_LOWER_BOUND", "raw_verdict": None,
        "profile_like": "vllm/minimal", "hardware_sm": 8.6,
        "predicted_b_breakdown": None,
        "abort_reason": "engine-support-unknown/no-arch-row",
        "detail": "[C0] no-arch-row", "is_gate_only": True,
    }
    (d / "manifest.json").write_text(json.dumps(gm, indent=2),
                                     encoding="utf-8")
    (d / "pt1-gate.json").write_text(json.dumps(gp, indent=2),
                                     encoding="utf-8")
    return d


g1 = write_gate(tmp / "g1")
fg = read_gate_bundle(g1)
check(isinstance(fg, FInputGate),
      "read_gate_bundle: schema==2 -> FInputGate")
check(isinstance(fg, BaseCaptureBundle),
      "FInputGate satisfies BaseCaptureBundle (protocol — F2/F5 consume "
      "it through the same surface)")
check(fg.pt2_download is None and fg.pt3_boot is None
      and fg.pt4_smoke is None and fg.pt5_override is None,
      "FInputGate: pt2-5 are None (gate-only terminated pre-download — "
      "satisfies the Optional[dict] protocol slots)")
check(fg.is_gate_only is True and fg.outcome == "hard-block"
      and fg.failure_class is None
      and fg.abort_reason == "engine-support-unknown/no-arch-row",
      "FInputGate: is_gate_only / outcome / failure_class / abort_reason")
check(fg.arch_family is None and fg.model_id == "Org/Gate-Model"
      and fg.quant_label == "none",
      "FInputGate: arch null pre-deriver, model_id the slug, quant_label "
      "_norm_quant(None)=='none' (same defensive discipline as FInput)")
# dedup_tuple uses .get(k,None) — crash-safe on the degraded manifest;
# the null-topology hash is DETERMINISTIC + stable (CONTRACT-1.1).
dtg = fg.dedup_tuple()
check(isinstance(dtg, tuple) and len(dtg) == 7 and dtg[5] is None
      and dtg[6] is None,
      f"FInputGate.dedup_tuple(): 7-tuple, .get-tolerant, fc+topo None "
      f"(got {dtg})")
dhg1 = fg.dedup_hash()
dhg2 = read_gate_bundle(g1).dedup_hash()
_exp_g = _hl.sha256("\x1f".join(str(p) for p in (
    "Org/Gate-Model", "none", None, None, None, None, None))
    .encode("utf-8")).hexdigest()[:12]
check(dhg1 == dhg2 == _exp_g,
      f"FInputGate.dedup_hash(): DETERMINISTIC with null topology "
      f"(str(None)=='None'); stable across reads (got {dhg1})")

# Schema mismatch: read_gate_bundle rejects schema==1; read_capture_bundle
# rejects schema==2 (the two readers are strictly separate).
raises(lambda: read_gate_bundle(b1),
       "read_gate_bundle rejects a schema==1 bundle (schema!=2)")
raises(lambda: read_capture_bundle(g1),
       "read_capture_bundle rejects a schema==2 gate bundle (untouched "
       "schema==1 path stays strict)")

# read_gate_bundle MUST NOT reuse the 22-key validator: a gate manifest
# WITHOUT the post-C0 keys still parses (it only requires the always-
# present row). And the [F]-classifies invariant is enforced.
g_bad_oc = write_gate(tmp / "gbo", manifest={
    "schema": 2, "slug": "x", "utc_ts": "t", "club3090_commit": "c",
    "outcome": "ok", "abort_reason": "hard-block", "failure_class": None})
raises(lambda: read_gate_bundle(g_bad_oc),
       "read_gate_bundle rejects outcome != 'hard-block'")
g_bad_fc = write_gate(tmp / "gbf", manifest={
    "schema": 2, "slug": "x", "utc_ts": "t", "club3090_commit": "c",
    "outcome": "hard-block", "abort_reason": "hard-block",
    "failure_class": "genuine-oom"})
raises(lambda: read_gate_bundle(g_bad_fc),
       "read_gate_bundle rejects a non-null failure_class (the gate "
       "emitter NEVER classifies — [F]'s job)")
g_min = write_gate(tmp / "gmin", manifest={
    "schema": 2, "slug": "x", "utc_ts": "t", "club3090_commit": "c",
    "outcome": "hard-block", "abort_reason": "disk-short",
    "failure_class": None})
check(isinstance(read_gate_bundle(g_min), FInputGate),
      "read_gate_bundle: a MINIMAL always-present-row-only manifest "
      "parses (does NOT reuse the 22-key schema-1 _require_keys)")

# ---------------------------------------------------------------------------
# 8. REAL on-disk capture dir(s) under .pull-captures/ (skip if none).
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
    print("SKIP: no real on-disk .pull-captures/ bundle to parse "
          "(graceful — not a failure)")
else:
    for rd in real_dirs:
        try:
            manifest = json.loads((rd / "manifest.json").read_text())
            schema = manifest.get("schema")
            if schema == 1:
                rfi = read_capture_bundle(rd)
                ok = (
                    isinstance(rfi, FInput)
                    and rfi.manifest["schema"] == 1
                    and rfi.failure_class is None
                    and len(rfi.consensus_key()) == 9
                    and len(rfi.dedup_tuple()) == 7
                    and len(rfi.dedup_hash()) == 12
                    # CONTRACT-5 G1: live topology serialized (full GPU name).
                    and isinstance(
                        rfi.manifest["topology_summary_canonical"], str)
                )
                check(ok, f"REAL [E] capture parses + keys build: {rd.name} "
                          f"(model={rfi.model_id!r})")
            elif schema == 2:
                gfi = read_gate_bundle(rd)
                ok = (
                    isinstance(gfi, FInputGate)
                    and gfi.manifest["schema"] == 2
                    and gfi.failure_class is None
                    and gfi.is_gate_only is True
                )
                check(ok, f"REAL gate capture parses: {rd.name} "
                          f"(model={gfi.model_id!r})")
            else:
                check(False, f"REAL capture {rd} has unsupported schema "
                             f"{schema!r}")
        except Exception as exc:
            check(False, f"REAL capture {rd} failed to parse: {exc!r}")

# ---------------------------------------------------------------------------
if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll F1 FInput capture-bundle reader (CONTRACT-1) assertions "
      "passed.")
PY

echo "test-loop-input.sh OK"
