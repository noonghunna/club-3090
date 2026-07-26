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

# test-classifier.sh — v0.8.0 [F] STEP F2 (club-3090 #147).
#
# Contract test for CONTRACT-2 Tier-2 + Appendix A: the §6.1 semantic-
# fingerprint classifier. The test IS the spec; the code is fixed to it.
# NO live Docker / GPU / network — fixture capture dirs are built in a tmp
# tree (byte-exact [E] schema, reused via F1's read_capture_bundle for
# realism) and classified.
#
# Coverage (every CONTRACT-2 / Appendix-A / §6.1 assertion as a
# failing-then-passing check):
#   * EACH Appendix A row maps to its STATED §6.1 class:
#       Cliff-2 OOM            -> genuine-oom
#       prefill-cliff GDN OOM  -> genuine-oom
#       #145 streaming dead    -> quant-unsupported (via pt4, boot green)
#       AWQ/quant mis-load     -> quant-unsupported
#       Genesis overlay drift  -> overlay-arch-drift
#       Ampere/FA3 SM90 kernel -> kernel-unsupported
#       cold-start then green  -> benign-cold-start (should_file=False)
#       served-name 404 ctrl   -> benign-cold-start (should_file=False)
#       unmatched              -> unknown (should_file=False + review_queue)
#   * route_as_kv_calc_bug is False for EVERY F2 path incl. genuine-oom
#     (Tier-1 is F3 — F2 must hard-wire False).
#   * output ALWAYS in the 6-enum ∪ unknown — an out-of-enum 7th DB value
#     degrades to `unknown` (no 7th value can leak).
#   * works with pt3.failure_log_excerpt ABSENT (today's shipped [E]
#     schema, bare pt3.failure) AND present (F3 forward-compat).
#   * tier seam: F2 only ever emits TIER2 / NONE_UNKNOWN, never TIER1.
#   * REAL on-disk .pull-captures/ bundle (a success/`partial`): classify
#     returns a valid enum and does NOT misclassify as a fileable failure.

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

from scripts.lib.profiles.loop_input import read_capture_bundle  # noqa: E402
from scripts.lib.profiles.classifier import (  # noqa: E402
    FailureClass,
    Tier,
    classify,
)

ENUM_VALUES = {c.value for c in FailureClass}
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Byte-exact [E] schema fixtures (mirror scripts/lib/profiles/capture.py),
# parsed back through F1's reader for realism (CONTRACT-1 boundary).
# ---------------------------------------------------------------------------
def mk_manifest(**over) -> dict:
    m = {
        "schema": 1,
        "slug": "Org/My-Model",
        "utc_ts": "20260517T000000Z",
        "submission_fingerprint": "deadbeef" * 8,
        "model": "Org/My-Model",
        "quant_label": "BFloat16",
        "arch_family": "Qwen3NextForCausalLM",
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
        "outcome": "failed",
        "capture_points": ["gate", "download", "boot", "smoke"],
    }
    m.update(over)
    return m


def write_bundle(d: Path, *, manifest=None, pt3=None, pt4=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    arts = {
        "manifest.json": manifest if manifest is not None else mk_manifest(),
        "pt1-gate.json": {
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
            "point": "boot", "ok": False, "seconds": 0.0,
            "failure": "server did not become ready before timeout",
        },
        "pt4-smoke.json": pt4 if pt4 is not None else {
            "point": "smoke",
            "smoke_capability_set": ["plain-chat", "streaming"],
            "results": {"plain-chat": "unsmoked", "streaming": "unsmoked"},
            "partial": True, "results_detail": {},
        },
    }
    for name, obj in arts.items():
        (d / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return d


tmp = Path(tempfile.mkdtemp())


def fi(name, **kw):
    return read_capture_bundle(write_bundle(tmp / name, **kw))


def boot_fail(failure=None, excerpt=None):
    pt3 = {"point": "boot", "ok": False, "seconds": 0.0,
           "failure": failure}
    if excerpt is not None:
        pt3["failure_log_excerpt"] = excerpt           # F3 forward-compat
        pt3["actual"] = {"attempted_alloc_mib": 1234,  # F2 must NOT read
                         "gpu_worker_reported_mib": 23456}
    return pt3


# ---------------------------------------------------------------------------
# Appendix A — every row maps to its STATED §6.1 class.
# ---------------------------------------------------------------------------
# Row 1: Cliff-2 accumulated-ctx OOM (~21-26K) — torch.cuda.OOM.
r = classify(fi("a1", pt3=boot_fail(
    failure="torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
            "allocate 512.00 MiB")))
check(r.failure_class is FailureClass.GENUINE_OOM,
      f"Appendix-A Cliff-2 OOM -> genuine-oom (got {r.failure_class.value})")
check(r.route_as_kv_calc_bug is False,
      "genuine-oom route_as_kv_calc_bug=False with NO Tier-1 inputs "
      "present (F3 honest-degrade — never a confidently-wrong kv-calc bug)")
check(r.should_file is True,
      "genuine-oom should_file=True (classified+filed, not a kv-calc bug)")
check(r.tier is Tier.TIER1,
      f"F3: OOM signature -> Tier-1 fast-path decides genuine-oom "
      f"(got {r.tier.value})")
check(r.matched_rule == "tier1-oom-fastpath",
      "F3: Tier-1 fast-path stamps matched_rule=tier1-oom-fastpath")
check(r.tier1_inputs is not None
      and r.tier1_inputs.get("predicted_b_breakdown") is None,
      "F3 honest-degrade: tier1_inputs surfaced, predicted side absent "
      "(bare pt3.failure, no pt1.predicted_b_breakdown / pt3.actual)")

# Row 2: prefill-cliff OOM (~50-60K, DeltaNet GDN) — works with the F3
# forward-compat failure_log_excerpt PRESENT.
r = classify(fi("a2", pt3=boot_fail(
    failure="server did not become ready before timeout",
    excerpt="...gated_delta_net forward... torch.cuda.OutOfMemoryError: "
            "CUDA out of memory ...")))
check(r.failure_class is FailureClass.GENUINE_OOM,
      f"Appendix-A prefill-cliff GDN OOM -> genuine-oom "
      f"(got {r.failure_class.value})")
check(r.route_as_kv_calc_bug is False,
      "prefill-cliff genuine-oom route_as_kv_calc_bug=False (F2)")
check(r.error_substring and "torch.cuda" in r.error_substring,
      "F3 forward-compat: failure_log_excerpt used when PRESENT")

# Row 3: #145 qwen3_coder streaming dead — boot+plain-chat green,
# streaming red. Maps via pt4, NOT a boot failure.
r = classify(fi("a3",
    pt3={"point": "boot", "ok": True, "seconds": 80.0, "failure": None},
    pt4={"point": "smoke",
         "smoke_capability_set": ["plain-chat", "streaming"],
         "results": {"plain-chat": "green", "streaming": "red"},
         "partial": False,
         "results_detail": {"streaming": {"status": 200, "error": ""}}}))
check(r.failure_class is FailureClass.QUANT_UNSUPPORTED,
      f"Appendix-A #145 streaming-dead -> quant-unsupported (got "
      f"{r.failure_class.value})")
check(r.matched_rule == "streaming-dead-boot-green-145",
      "#145 matched via pt4_results rule (not a pt3 boot failure)")

# Row 4: AWQ/quant mis-load on derived (no --quantization).
r = classify(fi("a4", pt3=boot_fail(
    failure="ValueError: Model QuantConfig: the AWQ weight is not "
            "supported for this dtype")))
check(r.failure_class is FailureClass.QUANT_UNSUPPORTED,
      f"Appendix-A AWQ mis-load -> quant-unsupported (got "
      f"{r.failure_class.value})")

# Row 5: Genesis-required engine on clean image (overlay drift).
r = classify(fi("a5", pt3=boot_fail(
    failure="AttributeError: module 'vllm' has no attribute "
            "'genesis_patch' (patch not applied)")))
check(r.failure_class is FailureClass.OVERLAY_ARCH_DRIFT,
      f"Appendix-A Genesis overlay drift -> overlay-arch-drift (got "
      f"{r.failure_class.value})")

# Row 6: Ampere-unsupported kernel (FA3 / SM90-only path).
r = classify(fi("a6", pt3=boot_fail(
    failure="RuntimeError: FlashAttention-3 requires Hopper SM90; "
            "compute capability 8.6 not supported on this gpu")))
check(r.failure_class is FailureClass.KERNEL_UNSUPPORTED,
      f"Appendix-A Ampere/FA3 SM90 -> kernel-unsupported (got "
      f"{r.failure_class.value})")

# Row 7: first-request-after-boot cold start (slow, then green). pt3
# timeout but pt4 green -> benign-cold-start, SUPPRESSED (not filed).
r = classify(fi("a7",
    pt3={"point": "boot", "ok": False, "seconds": 0.0,
         "failure": "server did not become ready before timeout"},
    pt4={"point": "smoke",
         "smoke_capability_set": ["plain-chat"],
         "results": {"plain-chat": "green"},
         "partial": False, "results_detail": {}}))
check(r.failure_class is FailureClass.BENIGN_COLD_START,
      f"Appendix-A cold-start-then-green -> benign-cold-start (got "
      f"{r.failure_class.value})")
check(r.should_file is False,
      "benign-cold-start SUPPRESSED — should_file=False (§6.1 acceptance)")
check(r.route_as_kv_calc_bug is False and r.review_queue is False,
      "benign-cold-start never files / never review-queue / never kv-calc")

# Row 8: [E] bug #1 served-model-name 404 — historical negative control.
r = classify(fi("a8",
    pt3={"point": "boot", "ok": True, "seconds": 80.0, "failure": None},
    pt4={"point": "smoke",
         "smoke_capability_set": ["plain-chat"],
         "results": {"plain-chat": "red"},
         "partial": False,
         "results_detail": {"plain-chat": {"status": 404,
                                            "error": "model not found"}}}))
check(r.failure_class is FailureClass.BENIGN_COLD_START,
      f"Appendix-A served-name-404 negative control -> benign-cold-start "
      f"(got {r.failure_class.value})")
check(r.should_file is False,
      "negative-control 404 should_file=False (in-enum benign-cold-start)")

# Row 9: anything unmatched -> unknown -> review queue, never files.
r = classify(fi("a9", pt3=boot_fail(
    failure="some entirely novel boot error nobody has classified yet")))
check(r.failure_class is FailureClass.UNKNOWN,
      f"Appendix-A unmatched -> unknown (got {r.failure_class.value})")
check(r.should_file is False and r.review_queue is True,
      "unknown -> should_file=False + review_queue=True (maintainer queue)")
check(r.route_as_kv_calc_bug is False,
      "unknown NEVER auto-files a kv-calc bug (§6.1)")
check(r.tier is Tier.NONE_UNKNOWN,
      f"unmatched decided as NONE_UNKNOWN (got {r.tier.value})")

# ---------------------------------------------------------------------------
# Cross-cutting: pt3.failure_log_excerpt ABSENT (today's shipped [E]
# schema) classifies off the bare pt3.failure string.
# ---------------------------------------------------------------------------
r = classify(fi("noexcerpt", pt3={
    "point": "boot", "ok": False, "seconds": 0.0,
    "failure": "torch.cuda.OutOfMemoryError: CUDA out of memory"}))
check("failure_log_excerpt" not in (read_capture_bundle(
          write_bundle(tmp / "noexcerpt2", pt3={
              "point": "boot", "ok": False, "seconds": 0.0,
              "failure": "torch.cuda.OutOfMemoryError"})).pt3_boot),
      "fixture confirms pt3.failure_log_excerpt ABSENT (shipped [E] schema)")
check(r.failure_class is FailureClass.GENUINE_OOM,
      "classifies off bare pt3.failure when failure_log_excerpt ABSENT")

# ---------------------------------------------------------------------------
# Out-of-enum guard: no 7th value can ever leak. Point the classifier at a
# poisoned DB whose matcher class is an invalid 7th value -> must degrade
# to `unknown`, and EVERY result must be in the 6-enum.
# ---------------------------------------------------------------------------
poison = tmp / "poison.yml"
poison.write_text(
    "schema: 1\n"
    "exact_fingerprints: {}\n"
    "condition_matchers:\n"
    "  - id: poisoned\n"
    "    kind: log_substring\n"
    "    any: ['poison-signal']\n"
    "    class: not-a-real-class-7th-value\n",
    encoding="utf-8")
r = classify(fi("poisoned", pt3=boot_fail(
    failure="this contains a poison-signal token")),
    fingerprint_db_path=poison)
check(r.failure_class is FailureClass.UNKNOWN,
      f"out-of-enum 7th DB class degrades to `unknown` (got "
      f"{r.failure_class.value}) — no 7th value can leak")
check(r.failure_class.value in ENUM_VALUES,
      "poisoned-DB result still strictly in the 6-member §6.1 enum")

# ---------------------------------------------------------------------------
# F3 — §6.1 Tier-1 fast-path (CONTRACT-2 A-i/A-ii/A-ii′/A-iii). Tier-1 plugs
# IN FRONT of Tier-2: the OOM signature -> always genuine-oom, decided by
# Tier.TIER1. route_as_kv_calc_bug=True ONLY when ALL THREE inputs present
# (pt1.predicted_b_breakdown + pt3.actual.attempted_alloc_mib +
# pt3.actual.gpu_worker_reported_mib); else honest-degrade (False). pt5 >
# pt3-triple > bare precedence. classifier reads structured fields only.
# ---------------------------------------------------------------------------
PRED = {"weights_gb": 9.0, "kv_gb": 12.0, "overhead_gb": 1.5}


def write_f3_bundle(name, *, pt1_extra=None, pt3=None, pt5=None):
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    pt1 = {
        "schema": 1, "point": "gate", "slug": "Org/My-Model",
        "confidence": "estimated-lower-bound", "raw_verdict": "wont-fit",
        "terminal": "override-accepted", "profile_like": "vllm/minimal",
        "hardware_sm": 8.6, "predicted_b_breakdown": None,
    }
    if pt1_extra:
        pt1.update(pt1_extra)
    arts = {
        "manifest.json": mk_manifest(),
        "pt1-gate.json": pt1,
        "pt2-download.json": {
            "point": "download", "ok": True, "files": ["model.safetensors"],
            "bytes": 1, "sha_verified": True, "failure": None},
        "pt3-boot.json": pt3 if pt3 is not None else {
            "point": "boot", "ok": False, "seconds": 0.0,
            "failure": "server did not become ready before timeout"},
        "pt4-smoke.json": {
            "point": "smoke",
            "smoke_capability_set": ["plain-chat", "streaming"],
            "results": {"plain-chat": "unsmoked", "streaming": "unsmoked"},
            "partial": True, "results_detail": {}},
    }
    if pt5 is not None:
        arts["pt5-override-capture.json"] = pt5
    for nm, obj in arts.items():
        (d / nm).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return read_capture_bundle(d)


# (1) genuine-oom with ALL THREE inputs present -> route_as_kv_calc_bug TRUE
#     + a predicted-vs-actual delta. (pt3-triple source.)
fa = write_f3_bundle("f3_all3",
    pt1_extra={"predicted_b_breakdown": PRED},
    pt3={"point": "boot", "ok": False, "seconds": 3.0,
         "failure": "server did not become ready before timeout",
         "failure_log_excerpt":
             "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
             "allocate 2.50 GiB",
         "actual": {"attempted_alloc_mib": 2560,
                    "gpu_worker_reported_mib": 22880}})
r = classify(fa)
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      f"F3 (1): OOM + all-3 -> genuine-oom via Tier-1 (got "
      f"{r.failure_class.value}/{r.tier.value})")
check(r.route_as_kv_calc_bug is True,
      "F3 (1): all-3-present -> route_as_kv_calc_bug=True (CONTRACT-2 "
      "A-ii′ + §11)")
check(r.predicted_vs_actual_delta_mib == 22880 - int(round(22.5 * 1024)),
      f"F3 (1): predicted-vs-actual delta = gpu_worker_peak - sum([B]) "
      f"(got {r.predicted_vs_actual_delta_mib})")
check((r.tier1_inputs or {}).get("source") == "pt3+pt1",
      f"F3 (1): inputs resolved from the pt3+pt1 triple "
      f"(got {(r.tier1_inputs or {}).get('source')})")

# (2) genuine-oom MISSING one input (no gpu_worker peak) -> classified
#     genuine-oom but route_as_kv_calc_bug FALSE (honest degrade).
fm = write_f3_bundle("f3_miss",
    pt1_extra={"predicted_b_breakdown": PRED},
    pt3={"point": "boot", "ok": False, "seconds": 3.0,
         "failure": "timeout",
         "failure_log_excerpt":
             "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
             "allocate 2.50 GiB",
         "actual": {"attempted_alloc_mib": 2560,
                    "gpu_worker_reported_mib": None}})
r = classify(fm)
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      "F3 (2): still classified genuine-oom via Tier-1 when an input "
      "is missing")
check(r.route_as_kv_calc_bug is False,
      "F3 (2): missing gpu_worker_reported_mib -> route_as_kv_calc_bug "
      "FALSE (honest degrade, never confidently-wrong)")
check(r.should_file is True,
      "F3 (2): genuine-oom still should_file=True (filed as a normal "
      "issue, just NOT a kv-calc bug)")

# (3) precedence: pt5 structured fields WIN over the pt3 triple (A-iii).
#     pt5 carries its own predicted_b_breakdown + actual; pt3's triple
#     would be incomplete, but pt5 supplies all three -> route TRUE, and
#     the resolved source is pt5.
fp5 = write_f3_bundle("f3_pt5",
    pt1_extra={"predicted_b_breakdown": None},
    pt3={"point": "boot", "ok": False, "seconds": 1.0,
         "failure": "timeout",
         "failure_log_excerpt":
             "torch.cuda.OutOfMemoryError: CUDA out of memory",
         "actual": {"attempted_alloc_mib": None,
                    "gpu_worker_reported_mib": None}},
    pt5={"point": "override_capture",
         "predicted_b_breakdown": PRED,
         "actual": {"boot_peak_mib": 23900,
                    "gpu_worker_reported_mib": 23900},
         "predicted_vs_actual_delta_mib": 1376,
         "exit_error_summary": "torch.cuda.OutOfMemoryError: CUDA OOM",
         "calibration_signal_not_validated": True})
r = classify(fp5)
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      "F3 (3): OOM -> genuine-oom via Tier-1 with pt5 present")
check((r.tier1_inputs or {}).get("source") == "pt5",
      f"F3 (3): A-iii precedence — pt5 structured fields WIN over the "
      f"pt3 triple (got {(r.tier1_inputs or {}).get('source')})")
check(r.route_as_kv_calc_bug is True,
      "F3 (3): pt5 supplies all three -> route_as_kv_calc_bug=True")

# (4) Tier-1 MISS (no OOM signature anywhere) -> falls straight through to
#     the UNCHANGED Tier-2 (AWQ quant mis-load -> quant-unsupported, the
#     exact F2 behaviour, byte-for-byte unaffected).
fmiss = write_f3_bundle("f3_t1miss",
    pt3={"point": "boot", "ok": False, "seconds": 0.0,
         "failure": "ValueError: Model QuantConfig: the AWQ weight is not "
                    "supported for this dtype"})
r = classify(fmiss)
check(r.tier is Tier.TIER2
      and r.failure_class is FailureClass.QUANT_UNSUPPORTED,
      f"F3 (4): no OOM signature -> Tier-1 misses, falls through to "
      f"UNCHANGED Tier-2 (got {r.tier.value}/{r.failure_class.value})")
check(r.route_as_kv_calc_bug is False
      and r.predicted_vs_actual_delta_mib is None
      and r.tier1_inputs is None,
      "F3 (4): a Tier-2 result carries NO Tier-1 telemetry (Tier-1 never "
      "ran) — F2 behaviour byte-unchanged")

# ---------------------------------------------------------------------------
# F8-fix — real modern vLLM v0.21.0+ KV-cache-too-large phrasing. The on-rig
# F8 validator induced a GENUINE vLLM KV-cache-too-large failure: vLLM
# nightly bf610c2f raises a CLEAN `ValueError` from
# `_check_enough_kv_cache_memory` — NOT torch.cuda.OutOfMemoryError — so the
# SHIPPED classic-torch-only Tier-1 OOM signature MISSED this real, common,
# kv-calc-relevant failure (it fell through to Tier-2 -> `unknown`, Tier-1
# never fired). These lines are copied VERBATIM from a captured real
# vLLM KV-OOM log (the gpu_worker-available line + the ValueError).
# With the F8-fix the widened signature MUST fire
# Tier-1 -> genuine-oom, and with the new pt3.actual parse + a
# pt1.predicted_b_breakdown all three inputs are present -> route TRUE.
F8_REAL_EXCERPT = (
    "f8d-oom-vllm  | (EngineCore pid=81) INFO 05-17 05:13:55 "
    "[gpu_worker.py:462] Available KV cache memory: 20.89 GiB\n"
    "f8d-oom-vllm  | (EngineCore pid=81) ERROR 05-17 05:13:55 "
    "[core.py:1159] ValueError: To serve at least one request with the "
    "models's max seq len (2000000), (22.89 GiB KV cache is needed, which "
    "is larger than the available KV cache memory (20.89 GiB). Based on "
    "the available memory, the estimated maximum model length is 1825488."
)
# 22.89 GiB -> 23439 MiB ; 20.89 GiB -> 21391 MiB (what [E]'s parse emits;
# the F8-fixed capture.py is what produces pt3.actual on a real bundle).
f8 = write_f3_bundle("f3_f8_real",
    pt1_extra={"predicted_b_breakdown": PRED},
    pt3={"point": "boot", "ok": False, "seconds": 4.0,
         "failure": "server did not become ready before timeout",
         "failure_log_excerpt": F8_REAL_EXCERPT,
         "actual": {"attempted_alloc_mib": 23439,
                    "gpu_worker_reported_mib": 21391}})
r = classify(f8)
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      f"F8-fix: real vLLM v0.21.0+ 'KV cache is needed, which is larger "
      f"than the available KV cache memory' ValueError -> genuine-oom via "
      f"Tier-1 (got {r.failure_class.value}/{r.tier.value})")
check(r.matched_rule == "tier1-oom-fastpath",
      "F8-fix: real vLLM KV-OOM stamps the Tier-1 fast-path rule")
check(r.route_as_kv_calc_bug is True,
      "F8-fix: real vLLM KV-OOM + all-3 inputs present -> "
      "route_as_kv_calc_bug=True (the kv-calc-bug signal this fix unblocks)")
check(r.predicted_vs_actual_delta_mib == 21391 - int(round(22.5 * 1024)),
      f"F8-fix: sane non-None predicted-vs-actual delta = gpu_worker "
      f"available (21391) - sum([B]) (got "
      f"{r.predicted_vs_actual_delta_mib})")
check((r.tier1_inputs or {}).get("source") == "pt3+pt1",
      "F8-fix: inputs resolved from the pt3+pt1 triple")

# F8-fix: the older vLLM "No available memory for the cache blocks"
# phrasing must ALSO fire Tier-1 genuine-oom (defensive coverage).
r = classify(fi("f8_cacheblocks", pt3=boot_fail(
    failure="ValueError: No available memory for the cache blocks. Try "
            "increasing `gpu_memory_utilization` when initializing the "
            "engine.")))
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      f"F8-fix: 'No available memory for the cache blocks' -> genuine-oom "
      f"via Tier-1 (got {r.failure_class.value}/{r.tier.value})")

# F8-fix CLASSIC NO-REGRESSION: the shipped classic
# torch.cuda.OutOfMemoryError / 'Tried to allocate' / 'peak memory' path
# STILL classifies genuine-oom via Tier-1 (the widening is purely
# additive — the classic alternative is FIRST in the signature regex).
r = classify(fi("f8_classic_noregress", pt3=boot_fail(
    failure="torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to "
            "allocate 2.50 GiB")))
check(r.failure_class is FailureClass.GENUINE_OOM
      and r.tier is Tier.TIER1,
      "F8-fix NO-REGRESSION: classic torch.cuda.OutOfMemoryError STILL "
      "classifies genuine-oom via Tier-1 (widening is additive)")

# F8-fix honest-degrade preserved: a NON-OOM novel error still falls
# through to unknown (the widened signature does NOT over-match).
r = classify(fi("f8_nonoom", pt3=boot_fail(
    failure="RuntimeError: some entirely unrelated novel startup error")))
check(r.failure_class is FailureClass.UNKNOWN
      and r.route_as_kv_calc_bug is False,
      "F8-fix: a non-OOM novel error STILL -> unknown / kvbug False "
      "(widened signature does not over-match; honest-degrade preserved)")

# Sweep every Appendix-A fixture: output ALWAYS ∈ the 6-enum; the kv-calc
# routing gate stays False for EVERY Appendix-A seed (the OOM rows a1/a2
# have NO Tier-1 inputs -> honest degrade; the rest are Tier-2). Only the
# OOM-signature rows (a1 Cliff-2, a2 prefill-cliff GDN) reach Tier-1; every
# non-OOM row still falls through to the unchanged Tier-2 (no F2 regress).
_OOM_ROWS = {"a1", "a2"}
for nm in ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"):
    rr = classify(read_capture_bundle(tmp / nm))
    check(rr.failure_class.value in ENUM_VALUES,
          f"{nm}: failure_class ∈ §6.1 6-enum ({rr.failure_class.value})")
    if nm in _OOM_ROWS:
        check(rr.tier is Tier.TIER1,
              f"{nm}: OOM signature -> F3 Tier-1 fast-path (got "
              f"{rr.tier.value})")
    else:
        check(rr.tier is not Tier.TIER1,
              f"{nm}: non-OOM -> never Tier-1 (falls through to "
              f"unchanged Tier-2; got {rr.tier.value})")
    check(rr.route_as_kv_calc_bug is False,
          f"{nm}: route_as_kv_calc_bug False (a1/a2 honest-degrade — no "
          f"Tier-1 inputs; rest Tier-2)")

# ---------------------------------------------------------------------------
# v0.8.2 CONTRACT-1.1 M1 — the additive `gate_abort_reason` matcher kind +
# the conservative seed. A gate-only (schema==2) FInputGate has empty
# error text + no pt3/pt4, so it falls through every shipped [E]-bundle
# rule to the new gate rules. Only `engine-support-unknown/no-arch-row` ->
# `kernel-unsupported` (should_file=True, public-filed — the §10-R9 lever);
# `runtime-incompatible`/`disk-short`/`hard-block` -> `unknown`
# (should_file=False + review_queue=True — review-queued, NOT public-filed).
# ---------------------------------------------------------------------------
from scripts.lib.profiles.loop_input import read_gate_bundle  # noqa: E402


def _gate_fi(nm, abort_reason):
    d = tmp / nm
    d.mkdir(parents=True, exist_ok=True)
    gm = {
        "schema": 2, "slug": "Org/Gate", "utc_ts": "20260518T000000Z",
        "club3090_commit": "cafef00d", "outcome": "hard-block",
        "abort_reason": abort_reason, "failure_class": None,
        "model": "Org/Gate", "model_id": "Org/Gate",
        "arch_family": None, "quant_label": None,
        "topology_class": None, "topology_summary_canonical": None,
        "selected_ctx": None, "kv_format": None,
        "smoke_capability_set": None, "engine_pin": None,
        "engine_version": None, "kv_calc_version": None,
        "submission_fingerprint": None, "is_gate_only": True,
        "capture_points": ["gate"],
    }
    gp = {
        "schema": 2, "point": "gate", "slug": "Org/Gate",
        "confidence": "ESTIMATED_LOWER_BOUND", "raw_verdict": None,
        "profile_like": "vllm/minimal", "hardware_sm": 8.6,
        "predicted_b_breakdown": None, "abort_reason": abort_reason,
        "detail": "x", "is_gate_only": True,
    }
    (d / "manifest.json").write_text(json.dumps(gm), encoding="utf-8")
    (d / "pt1-gate.json").write_text(json.dumps(gp), encoding="utf-8")
    return read_gate_bundle(d)


# /no-arch-row -> kernel-unsupported, PUBLIC-FILED (should_file=True).
rg = classify(_gate_fi("g-narr", "engine-support-unknown/no-arch-row"))
check(rg.failure_class is FailureClass.KERNEL_UNSUPPORTED,
      f"M1: engine-support-unknown/no-arch-row -> kernel-unsupported "
      f"(the §10-R9 lever) (got {rg.failure_class.value})")
check(rg.should_file is True and rg.review_queue is False,
      "M1: /no-arch-row is PUBLIC-FILED (should_file=True, NOT "
      "review-queued) — it must aggregate/dedup/volume-rank")
check(rg.matched_rule == "gate-engine-support-no-arch-row",
      f"M1: matched the additive gate_abort_reason rule (got "
      f"{rg.matched_rule!r})")
check(rg.route_as_kv_calc_bug is False,
      "M1: a gate-only bundle is NEVER a kv-calc bug (Tier-1 is F3; no "
      "OOM signature on a gate bundle)")

# runtime-incompatible / disk-short / hard-block -> unknown, REVIEW-QUEUED
# (NOT public-filed) — deliberately suppressed (correct-refusal class).
for _ar in ("engine-support-unknown/runtime-incompatible",
            "disk-short", "hard-block"):
    rq = classify(_gate_fi(f"g-{_ar.replace('/', '-')}", _ar))
    check(rq.failure_class is FailureClass.UNKNOWN,
          f"M1: {_ar} -> unknown (conservative catch-all) "
          f"(got {rq.failure_class.value})")
    check(rq.should_file is False and rq.review_queue is True,
          f"M1: {_ar} is REVIEW-QUEUED, NOT public-filed "
          f"(unknown ∈ _SUPPRESSED_NEVER_FILED; should_file=False, "
          f"review_queue=True — never silently dropped)")

# The raw abort_reason SURVIVES redaction into pt1-gate (H2 maintainer-
# distinguishability: a /no-arch-row vs a kernel bug must be tellable).
_h2 = _gate_fi("g-h2", "engine-support-unknown/no-arch-row")
check(_h2.pt1_gate.get("abort_reason")
      == "engine-support-unknown/no-arch-row"
      and _h2.abort_reason == "engine-support-unknown/no-arch-row",
      "M1/H2: the raw abort_reason survives into pt1_gate (a maintainer "
      "can distinguish 'add a registry row' from 'file a kernel bug')")

# §6.1-NEUTRALITY: the additive kind does NOT perturb the shipped
# schema==1 path. The SAME [E] bundle classifies byte-identically + a
# schema==1 FInput has NO pt1_gate.abort_reason so the gate rules are
# inert for it (they only fire on a gate-only bundle).
_s1 = fi("neutral-s1", pt3=boot_fail(
    failure="torch.cuda.OutOfMemoryError: CUDA out of memory"))
_r1a = classify(_s1)
_r1b = classify(fi("neutral-s1b", pt3=boot_fail(
    failure="torch.cuda.OutOfMemoryError: CUDA out of memory")))
check(_r1a.failure_class is FailureClass.GENUINE_OOM
      and _r1a.failure_class == _r1b.failure_class
      and _r1a.matched_rule == _r1b.matched_rule
      and _r1a.tier == _r1b.tier,
      "V1: schema==1 classification is byte-identical post-protocol-lift "
      "+ the additive gate_abort_reason kind is inert for [E] bundles "
      "(no pt1_gate.abort_reason -> the gate rules never fire)")
from scripts.lib.profiles.loop_input import BaseCaptureBundle  # noqa: E402
check(isinstance(_s1, BaseCaptureBundle),
      "V1: the schema==1 FInput classify() consumed satisfies "
      "BaseCaptureBundle (the retype is a pure static change)")

# ---------------------------------------------------------------------------
# REAL on-disk .pull-captures/ bundle (a success/`partial`): classify
# returns a valid enum and does NOT misclassify as a fileable failure.
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
    print("SKIP: no real on-disk .pull-captures/ bundle (graceful)")
else:
    for rd in real_dirs:
        rfi = read_capture_bundle(rd)
        rr = classify(rfi)
        check(rr.failure_class.value in ENUM_VALUES,
              f"REAL {rd.name}: classify -> valid §6.1 enum "
              f"({rr.failure_class.value})")
        # The one real bundle is boot-ok + all caps green/unsmoked (no
        # red, no failure) — a success/`partial`. It must NOT be filed as
        # a failure and must NEVER route as a kv-calc bug.
        boot_ok = bool((rfi.pt3_boot or {}).get("ok"))
        no_red = not any(
            v == "red"
            for v in ((rfi.pt4_smoke or {}).get("results") or {}).values())
        if boot_ok and no_red:
            check(rr.route_as_kv_calc_bug is False,
                  f"REAL success/partial {rd.name}: never a kv-calc bug")
            check(rr.failure_class is not FailureClass.GENUINE_OOM,
                  f"REAL success/partial {rd.name}: NOT misclassified as "
                  f"genuine-oom")

# ---------------------------------------------------------------------------
if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll F2 §6.1 Tier-2 classifier (CONTRACT-2 / Appendix A) "
      "assertions passed.")
PY

echo "test-classifier.sh OK"
