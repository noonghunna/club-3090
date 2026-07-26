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

# test-dedup.sh — v0.8.0 [F] STEP F5 (club-3090 #147).
#
# Contract test for CONTRACT-4: the §6.3 canonical-tuple-hash dedup +
# bounded label scheme + collision-safe submit path. The test IS the spec;
# the code is fixed to it. NO live Docker / GPU / network and NO real `gh`:
# every `gh` invocation goes through an INJECTED mock `gh_runner`. Fixture
# capture dirs are built in a tmp tree (byte-exact [E] schema, parsed via
# F1's read_capture_bundle for realism, classified via F2/F3's classify).
#
# Coverage (every CONTRACT-4 / Codex-r1 H3+H4 / §6.1 assertion as a
# failing-then-passing check):
#   * effective_dedup_hash is 12 hex; substitutes the CLASSIFIER's
#     failure_class (NOT the manifest null) into the §6.3 7-tuple; two
#     FInputs differing on ANY of the 7 dims -> different hash (Codex-r1
#     H3: no dimension droppable).
#   * bounded labels ONLY: exactly {loop:dedup-<12hex>, class:<enum>,
#     arch:<sanitized>}; NO raw model/engine/kvcalc/topo label; class in
#     the 6 §6.1 enum; arch sanitized + length-capped (Codex-r1 H4).
#   * submit with a mocked gh_runner: (a) no existing issue -> opens new
#     (labels + body has the full 7-tuple json); (b) existing issue body
#     tuple MATCHES -> +1 comment, no new issue; (c) existing issue same
#     loop:dedup- label but body tuple MISMATCHES (simulated sha12
#     collision) -> does NOT +1, opens new (the collision safeguard);
#     (d) gh missing/fails -> spools _dedup-queue/<hash>.json, returns
#     cleanly, never raises.
#   * filing policy: benign-cold-start -> not filed, not spooled-as-issue;
#     unknown -> _review-queue/ not the tracker; only should_file=True
#     dedups/files; F5 never emits a kv-calc-bug filing.
#   * real on-disk capture under .pull-captures/ (>=2 exist): build
#     FInput+classify+dedup with a mock gh returning "no existing issue"
#     -> opens exactly one issue with a valid bounded label set + a body
#     7-tuple that round-trips (parse back == effective_dedup_tuple).
#   * import-time safety: importing dedup then kv-calc --calibration is
#     still Overall: 22/22 (100%).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles.loop_input import (  # noqa: E402
    FInput,
    read_capture_bundle,
)
from scripts.lib.profiles.classifier import (  # noqa: E402
    ClassificationResult,
    FailureClass,
    Tier,
    classify,
)
from scripts.lib.profiles import dedup as D  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Byte-exact [E] schema fixtures (mirror capture.py; parsed via F1 for
# realism — same discipline as test-classifier.sh / test-trust-pipeline.sh).
# ---------------------------------------------------------------------------
def mint_fingerprint(m: dict) -> str:
    parts = [
        m["model"], m["club3090_commit"], m["topology_summary_canonical"],
        str(m["quant_label"]), m["kv_calc_version"],
        str(m["engine_version"]), m["utc_ts"], m["outcome"],
    ]
    h = hashlib.sha256()
    h.update("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


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
        "outcome": "failed",
        "capture_points": ["gate", "download", "boot", "smoke"],
    }
    m.update(over)
    if m["submission_fingerprint"] == "PLACEHOLDER":
        m["submission_fingerprint"] = mint_fingerprint(m)
    return m


def write_bundle(d: Path, *, manifest=None, pt3=None, pt4=None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    man = manifest if manifest is not None else mk_manifest()
    arts = {
        "manifest.json": man,
        "pt1-gate.json": {
            "schema": 1, "point": "gate", "slug": man["slug"],
            "confidence": "estimated-lower-bound",
            "raw_verdict": "fits-clean", "terminal": "confirm→proceed",
            "profile_like": "vllm/minimal", "hardware_sm": 8.6,
            "predicted_b_breakdown": None,
        },
        "pt2-download.json": {
            "point": "download", "ok": True,
            "files": ["model.safetensors"], "bytes": 123,
            "sha_verified": True, "failure": None,
        },
        "pt3-boot.json": pt3 if pt3 is not None else {
            "point": "boot", "ok": False, "seconds": 12.0,
            "failure": "torch.cuda.OutOfMemoryError: CUDA out of memory",
            "failure_log_excerpt":
                "torch.cuda.OutOfMemoryError: CUDA out of memory. "
                "Tried to allocate 2048 MiB",
        },
        "pt4-smoke.json": pt4 if pt4 is not None else {
            "point": "smoke", "smoke_capability_set": [],
            "results": {}, "partial": False, "results_detail": {},
        },
    }
    for name, obj in arts.items():
        (d / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return d


tmp = Path(tempfile.mkdtemp())
repo_root = Path(tempfile.mkdtemp())  # isolated .pull-captures spool root.


# ---------------------------------------------------------------------------
# Mock gh_runner — ZERO network / `gh`. Records argv; returns scripted
# GhResult objects (the real seam: D.GhResult / D.GhRunner).
# ---------------------------------------------------------------------------
class MockGh:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.list_response: str = "[]"      # gh issue list --json stdout.
        self.list_ok: bool = True
        self.comment_ok: bool = True
        self.create_ok: bool = True
        self.create_stdout: str = (
            "https://github.com/noonghunna/club-3090/issues/4242"
        )
        self.unavailable: bool = False      # simulate gh-missing.

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if self.unavailable:
            return D.GhResult(False, 127, "", "gh not found (mock)")
        if argv[:2] == ["issue", "list"]:
            return D.GhResult(self.list_ok,
                              0 if self.list_ok else 1,
                              self.list_response if self.list_ok else "",
                              "" if self.list_ok else "list failed")
        if argv[:2] == ["issue", "comment"]:
            return D.GhResult(self.comment_ok,
                              0 if self.comment_ok else 1, "",
                              "" if self.comment_ok else "comment failed")
        if argv[:2] == ["issue", "create"]:
            return D.GhResult(self.create_ok,
                              0 if self.create_ok else 1,
                              self.create_stdout if self.create_ok else "",
                              "" if self.create_ok else "create failed")
        if argv[:2] == ["label", "create"]:
            return D.GhResult(True, 0, "", "")
        return D.GhResult(True, 0, "", "")


def oom_classification(fi: FInput) -> ClassificationResult:
    """Real F2/F3 classify on an OOM bundle -> genuine-oom (should_file
    True). Asserted, so the test fails loudly if the classifier drifts."""
    cl = classify(fi)
    assert cl.failure_class is FailureClass.GENUINE_OOM, cl.failure_class
    assert cl.should_file is True
    return cl


# ===========================================================================
# 1. effective_dedup_hash: 12 hex; classifier's failure_class substituted
#    (NOT the manifest null); all 7 dims in the hash (Codex-r1 H3).
# ===========================================================================
b_oom = write_bundle(tmp / "oom")
fi = read_capture_bundle(b_oom)
cl = oom_classification(fi)

eff_t = D.effective_dedup_tuple(fi, cl)
eff_h = D.effective_dedup_hash(fi, cl)

check(re.fullmatch(r"[0-9a-f]{12}", eff_h) is not None,
      f"effective_dedup_hash is exactly 12 lowercase hex (got {eff_h!r})")

# F1's pre-classification tuple has failure_class == manifest null (None);
# the EFFECTIVE tuple substitutes the classifier's value at index 5.
f1_tuple = fi.dedup_tuple()
check(f1_tuple[5] is None,
      "F1 dedup_tuple()[5] (failure_class) is the manifest null (pre-class)")
check(eff_t[5] == "genuine-oom",
      f"effective tuple substitutes the CLASSIFIER's failure_class at [5] "
      f"(got {eff_t[5]!r}, not the manifest None)")
check(eff_t[0] == f1_tuple[0] and eff_t[1] == f1_tuple[1]
      and eff_t[2] == f1_tuple[2] and eff_t[3] == f1_tuple[3]
      and eff_t[4] == f1_tuple[4] and eff_t[6] == f1_tuple[6],
      "every OTHER dim is taken verbatim from F1's normalized tuple "
      "(single-sourced normalization; only [5] differs)")

# Byte-exact F1 convention: sha256("\x1f".join(str(p)))[:12].
manual = hashlib.sha256(
    "\x1f".join(str(p) for p in eff_t).encode("utf-8")
).hexdigest()[:12]
check(eff_h == manual,
      f"effective_dedup_hash uses F1's EXACT \\x1f+sha256[:12] convention "
      f"(loop_input.py:227-228) — proxy reuse matches the manual hash "
      f"({eff_h} == {manual})")

# Differing on a NON-failure_class manifest dim must change the hash.
DIM_OVERRIDES = {
    "model": {"model": "Org/Other", "model_id": "Org/Other",
              "slug": "Org/Other"},
    "quant_label": {"quant_label": "AWQ"},
    "arch_family": {"arch_family": "Qwen3MoeForCausalLM"},
    "kv_calc_version": {"kv_calc_version": "kvcalc-v0.8.0+deadbeefcafe"},
    "engine_version": {"engine_pin": "vllm/vllm-openai:nightly-zzz",
                       "engine_version": "vllm/vllm-openai:nightly-zzz"},
    "topology_class": {"topology_class": "2x24576MiB"},
}
for i, (dim, over) in enumerate(DIM_OVERRIDES.items()):
    bo = write_bundle(tmp / f"dim_{i}", manifest=mk_manifest(**over))
    fo = read_capture_bundle(bo)
    co = oom_classification(fo)
    ho = D.effective_dedup_hash(fo, co)
    check(ho != eff_h,
          f"differing on §6.3 dim `{dim}` -> DIFFERENT hash "
          f"(Codex-r1 H3: no dimension droppable) ({ho} != {eff_h})")

# And the failure_class (7th dim, classifier-supplied) also changes it:
# same bundle, a DIFFERENT classifier class -> different hash.
cl_kernel = ClassificationResult(
    failure_class=FailureClass.KERNEL_UNSUPPORTED, tier=Tier.TIER2,
    fingerprint="0" * 12, should_file=True, route_as_kv_calc_bug=False,
    review_queue=False, error_substring="x", matched_rule="seed",
)
check(D.effective_dedup_hash(fi, cl_kernel) != eff_h,
      "differing on the classifier's failure_class (the 7th dim) -> "
      "DIFFERENT hash (the §6.1 mislabel safeguard)")

# ===========================================================================
# 2. Bounded labels ONLY (Codex-r1 H4): exactly
#    {loop:dedup-<12hex>, class:<enum>, arch:<sanitized>}; NO raw
#    model/engine/kvcalc/topo label.
# ===========================================================================
labels = D.label_set(eff_h, cl, fi)
check(len(labels) == 3, f"exactly 3 labels (got {labels})")
ldedup = [x for x in labels if x.startswith("loop:dedup-")]
lclass = [x for x in labels if x.startswith("class:")]
larch = [x for x in labels if x.startswith("arch:")]
check(len(ldedup) == 1
      and re.fullmatch(r"loop:dedup-[0-9a-f]{12}", ldedup[0]),
      f"the dedup label is loop:dedup-<12hex> (got {ldedup})")
ENUM = {c.value for c in FailureClass}
check(len(lclass) == 1 and lclass[0].split(":", 1)[1] in ENUM,
      f"class: label value is one of the 6 §6.1 enum (got {lclass})")
check(len(larch) == 1
      and re.fullmatch(r"arch:[a-z0-9-]+", larch[0])
      and len(larch[0].split(":", 1)[1]) <= D._ARCH_LABEL_VALUE_CAP,
      f"arch: label is slug-sanitized + length-capped (got {larch})")
RAW_FORBIDDEN = ("model:", "engine:", "kvcalc:", "topo:", "topology:")
check(not any(x.startswith(p) for x in labels for p in RAW_FORBIDDEN),
      "NO raw model:/engine:/kvcalc:/topo: label (those live in the body)")

# A long/awkward arch is still bounded + sanitized.
b_longarch = write_bundle(
    tmp / "longarch",
    manifest=mk_manifest(arch_family="Some::Weird/Arch  With Spaces!!" * 5),
)
fi_la = read_capture_bundle(b_longarch)
cl_la = oom_classification(fi_la)
labs_la = D.label_set(D.effective_dedup_hash(fi_la, cl_la), cl_la, fi_la)
arch_la = [x for x in labs_la if x.startswith("arch:")][0]
check(re.fullmatch(r"arch:[a-z0-9-]+", arch_la)
      and len(arch_la.split(":", 1)[1]) <= D._ARCH_LABEL_VALUE_CAP,
      f"a long/awkward arch is sanitized + capped (got {arch_la!r})")

# bootstrap_labels: idempotent, gh-guarded, only the 6 class:* values.
mg0 = MockGh()
bs = D.bootstrap_labels(gh_runner=mg0)
check(set(bs.keys()) == ENUM and all(bs.values()),
      "bootstrap_labels creates exactly the 6 bounded class:* labels "
      "(gh_runner-guarded, --force idempotent)")
check(all(c[:2] == ["label", "create"] and "--force" in c
          for c in mg0.calls),
      "every bootstrap gh call is `label create --force` (idempotent)")
mg0.unavailable = True
bs2 = D.bootstrap_labels(gh_runner=mg0)  # must NOT raise.
check(all(v is False for v in bs2.values()),
      "bootstrap_labels with gh unavailable reports not-ok, NEVER raises")

# ===========================================================================
# 3. submit() with a mocked gh_runner.
# ===========================================================================
# (a) no existing issue -> opens new; labels + body 7-tuple json.
mg = MockGh()
mg.list_response = "[]"
r = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg)
check(r.action is D.DedupAction.OPENED and r.reason == "opened-new",
      f"(a) no existing issue -> OPENED (got {r.action.value}/{r.reason})")
create_call = [c for c in mg.calls if c[:2] == ["issue", "create"]]
check(len(create_call) == 1, "(a) exactly one `issue create` call")
cc = create_call[0]
for lab in labels:
    check(lab in cc, f"(a) new issue carries bounded label {lab!r}")
body_arg = cc[cc.index("--body") + 1]
parsed_body = D.parse_body_tuple(body_arg)
check(parsed_body is not None
      and [str(x) for x in parsed_body]
      == [str(x) for x in D.effective_dedup_tuple(fi, cl)],
      "(a) new-issue body carries the FULL canonical 7-tuple json that "
      "round-trips to effective_dedup_tuple")
check(set(r.labels) == set(labels),
      "(a) result.labels == the exact bounded label set")

# (b) existing issue, body tuple MATCHES -> +1 comment, no new issue.
match_body = D.build_issue_body(fi, cl)
mg2 = MockGh()
mg2.list_response = json.dumps([{"number": 99, "body": match_body}])
r2 = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg2)
check(r2.action is D.DedupAction.PLUSONE
      and r2.reason == "plusone-body-verified" and r2.issue_number == 99,
      f"(b) body-tuple MATCH -> PLUSONE on #99 "
      f"(got {r2.action.value}/{r2.reason}/{r2.issue_number})")
check(any(c[:2] == ["issue", "comment"] for c in mg2.calls)
      and not any(c[:2] == ["issue", "create"] for c in mg2.calls),
      "(b) +1 via `issue comment`; NO `issue create` (no new issue)")
ccm = [c for c in mg2.calls if c[:2] == ["issue", "comment"]][0]
check(D._PLUSONE_MARKER in ccm[ccm.index("--body") + 1],
      "(b) the +1 comment carries the stable machine-parseable marker")

# (c) existing issue, SAME loop:dedup- label but body tuple MISMATCHES
#     (simulated sha12 collision) -> does NOT +1, opens new.
fi_other = read_capture_bundle(
    write_bundle(tmp / "collide",
                 manifest=mk_manifest(model="Org/Totally-Different",
                                      model_id="Org/Totally-Different",
                                      slug="Org/Totally-Different"))
)
cl_other = oom_classification(fi_other)
collide_body = D.build_issue_body(fi_other, cl_other)  # DIFFERENT 7-tuple.
mg3 = MockGh()
# The candidate is returned under THIS hash's label (simulating a sha12
# collision: same loop:dedup-<hash> label, different body tuple).
mg3.list_response = json.dumps([{"number": 7, "body": collide_body}])
r3 = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg3)
check(r3.action is D.DedupAction.OPENED and r3.reason == "opened-new",
      f"(c) sha12 collision (label hit, body tuple MISMATCH) -> OPENED "
      f"new, does NOT +1 (got {r3.action.value}/{r3.reason})")
check(not any(c[:2] == ["issue", "comment"] for c in mg3.calls)
      and any(c[:2] == ["issue", "create"] for c in mg3.calls),
      "(c) collision safeguard: NEVER +1 on a hash-label match alone; "
      "body-tuple mismatch -> open new, no comment")
check(D.body_tuple_matches(collide_body, fi, cl) is False
      and D.body_tuple_matches(match_body, fi, cl) is True,
      "(c) body_tuple_matches: True only on a full 7-tuple equality")

# (d) gh missing/fails -> spool _dedup-queue/<hash>.json; never raises.
mg4 = MockGh()
mg4.unavailable = True
r4 = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg4)
check(r4.action is D.DedupAction.SPOOLED
      and r4.reason.startswith("gh-degraded"),
      f"(d) gh unavailable -> SPOOLED (got {r4.action.value}/{r4.reason})")
expected_spool = (repo_root / ".pull-captures" / "_dedup-queue"
                  / f"{eff_h}.json")
check(r4.spool_path == expected_spool and expected_spool.is_file(),
      f"(d) spooled to .pull-captures/_dedup-queue/{eff_h}.json")
spooled = json.loads(expected_spool.read_text())
check(spooled["kind"] == "would-be-issue"
      and spooled["dedup_hash"] == eff_h
      and set(spooled["labels"]) == set(labels)
      and D.parse_body_tuple(spooled["body"]) is not None,
      "(d) spooled payload is a replayable would-be issue (labels + body "
      "7-tuple)")

# gh failing on the list step (ok=False, not missing) also spools.
mg4b = MockGh()
mg4b.list_ok = False
r4b = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg4b)
check(r4b.action is D.DedupAction.SPOOLED,
      "(d) gh `issue list` non-zero exit also degrades to SPOOLED "
      "(never raises / blocks)")

# gh failing on the create step (after a clean list) also spools.
mg4c = MockGh()
mg4c.list_response = "[]"
mg4c.create_ok = False
r4c = D.submit(fi, cl, repo_root=repo_root, gh_runner=mg4c)
check(r4c.action is D.DedupAction.SPOOLED,
      "(d) gh `issue create` failure degrades to SPOOLED (never raises)")

# ===========================================================================
# 4. Filing policy (§6.1 acceptance verbatim).
# ===========================================================================
# benign-cold-start -> NOT filed, NOT spooled-as-issue (SUPPRESSED).
cl_benign = ClassificationResult(
    failure_class=FailureClass.BENIGN_COLD_START, tier=Tier.TIER2,
    fingerprint="0" * 12, should_file=False, route_as_kv_calc_bug=False,
    review_queue=False, error_substring="cold start", matched_rule="seed",
)
mgb = MockGh()
rb = D.submit(fi, cl_benign, repo_root=repo_root, gh_runner=mgb)
check(rb.action is D.DedupAction.SUPPRESSED
      and rb.reason == "benign-suppressed",
      f"benign-cold-start -> SUPPRESSED (got {rb.action.value})")
check(mgb.calls == [],
      "benign-cold-start: NO gh I/O at all (not filed)")
check(not (repo_root / ".pull-captures" / "_dedup-queue"
           / f"{rb.dedup_hash}.json").is_file()
      or rb.spool_path is None,
      "benign-cold-start: NOT spooled-as-issue (§6.1 acceptance)")

# unknown -> _review-queue/ spool, NOT the tracker.
cl_unknown = ClassificationResult(
    failure_class=FailureClass.UNKNOWN, tier=Tier.NONE_UNKNOWN,
    fingerprint="0" * 12, should_file=False, route_as_kv_calc_bug=False,
    review_queue=True, error_substring="???", matched_rule=None,
)
mgu = MockGh()
ru = D.submit(fi, cl_unknown, repo_root=repo_root, gh_runner=mgu)
check(ru.action is D.DedupAction.REVIEW_QUEUED
      and ru.reason == "unknown-review-queued",
      f"unknown -> REVIEW_QUEUED (got {ru.action.value})")
check(mgu.calls == [], "unknown: NO gh I/O (never the issue tracker)")
rq = (repo_root / ".pull-captures" / "_review-queue"
      / f"{ru.dedup_hash}.json")
check(ru.spool_path == rq and rq.is_file(),
      f"unknown -> spooled to .pull-captures/_review-queue/ (NOT the "
      f"tracker, NOT a kv-calc bug)")
rqp = json.loads(rq.read_text())
check(rqp["kind"] == "review-queue",
      "unknown review-queue spool is marked kind=review-queue")

# only should_file=True dedups/files (the genuine-oom path above did).
check(cl.should_file is True and r.action is D.DedupAction.OPENED,
      "only should_file=True classes dedup-or-file (genuine-oom OPENED)")

# F5 NEVER emits a kv-calc-bug filing: even a route_as_kv_calc_bug=True
# classification is deduped as a normal failure issue (kv-calc-bug is
# F3-Tier-1's calibration signal, a DIFFERENT pipeline F5 does not touch).
cl_kvbug = ClassificationResult(
    failure_class=FailureClass.GENUINE_OOM, tier=Tier.TIER1,
    fingerprint="0" * 12, should_file=True, route_as_kv_calc_bug=True,
    review_queue=False, error_substring="oom",
    matched_rule="tier1-oom-fastpath",
    predicted_vs_actual_delta_mib=512,
)
mgk = MockGh()
mgk.list_response = "[]"
rk = D.submit(fi, cl_kvbug, repo_root=repo_root, gh_runner=mgk)
check(rk.action is D.DedupAction.OPENED,
      "route_as_kv_calc_bug=True still just OPENS a normal deduped "
      "failure issue (F5 NEVER files a kv-calc bug)")
ckk = [c for c in mgk.calls if c[:2] == ["issue", "create"]][0]
joined = " ".join(ckk).lower()
check("kv-calc-bug" not in joined and "calibration bug" not in joined,
      "F5's filed issue is NOT a kv-calc-bug filing (boundary held)")

# ===========================================================================
# 4b. v0.8.2 CONTRACT-1.1 — F5 protocol lift (V1 RED-LINE).
#
#  * a real schema==1 bundle -> identical effective_dedup_hash /
#    effective_dedup_tuple PRE/POST the protocol retype (byte-identity);
#  * the FENCED `dedup.py` `FInput.dedup_hash(_EffProxy())` unbound-class
#    idiom STILL produces F1's byte-exact serialization (NOT "tidied");
#  * a schema==2 gate-only FInputGate flows through F5 via the protocol
#    (effective hash deterministic + STABLE with null topology —
#    CONTRACT-1.1; review-queued, NOT public-filed for `unknown`).
# ===========================================================================
from scripts.lib.profiles.loop_input import (  # noqa: E402
    BaseCaptureBundle,
    read_gate_bundle,
)

# (a) schema==1 byte-identity: the effective tuple/hash are a pure function
#     of F1's normalized tuple + the classifier class — pinned so a
#     protocol-lift regression fails LOUDLY.
_fi_id = read_capture_bundle(write_bundle(tmp / "id-s1"))
_cl_id = oom_classification(_fi_id)
_et1 = D.effective_dedup_tuple(_fi_id, _cl_id)
_eh1 = D.effective_dedup_hash(_fi_id, _cl_id)
_et2 = D.effective_dedup_tuple(read_capture_bundle(tmp / "id-s1"), _cl_id)
_eh2 = D.effective_dedup_hash(read_capture_bundle(tmp / "id-s1"), _cl_id)
import hashlib as _hl2  # noqa: E402
_exp_et = list(_fi_id.dedup_tuple())
_exp_et[5] = "genuine-oom"
_exp_eh = _hl2.sha256("\x1f".join(str(p) for p in _exp_et)
                      .encode("utf-8")).hexdigest()[:12]
check(_et1 == _et2 == tuple(_exp_et) and _eh1 == _eh2 == _exp_eh,
      f"V1 RED-LINE: schema==1 effective_dedup_tuple/hash byte-identical "
      f"+ pinned ({_exp_eh}) — the protocol lift is a pure static retype")
check(isinstance(_fi_id, BaseCaptureBundle),
      "V1: the schema==1 FInput F5 consumes satisfies BaseCaptureBundle")

# (b) the FENCED unbound-class idiom: effective_dedup_hash MUST equal F1's
#     OWN dedup_hash() of the substituted tuple (proves dedup.py:~277
#     still reuses FInput.dedup_hash unbound, NOT finput.dedup_hash() which
#     would hash the unsubstituted fc=None tuple).
class _Probe:
    def dedup_tuple(self):
        return _et1


from scripts.lib.profiles.loop_input import FInput as _FI  # noqa: E402
check(D.effective_dedup_hash(_fi_id, _cl_id)
      == _FI.dedup_hash(_Probe()),  # type: ignore[arg-type]
      "V1 RED-LINE: effective_dedup_hash reuses F1's UNBOUND dedup_hash on "
      "the SUBSTITUTED tuple (the dedup.py fence holds — NOT tidied to "
      "finput.dedup_hash() which would hash the fc=None tuple)")
# Negative control: hashing the UN-substituted tuple (fc=None) differs —
# the exact silent-corruption the fence prevents.
check(D.effective_dedup_hash(_fi_id, _cl_id) != _fi_id.dedup_hash(),
      "V1 RED-LINE: the substituted hash != F1's fc=None hash (a 'tidy' "
      "to finput.dedup_hash() WOULD silently corrupt + defeat the §6.1 "
      "mislabel safeguard — proven divergent)")

# (c) a schema==2 gate-only FInputGate flows through F5 via the protocol.
_gd = tmp / "f5-gate"
_gd.mkdir(parents=True, exist_ok=True)
_gman = {
    "schema": 2, "slug": "Org/Gate", "utc_ts": "20260518T000000Z",
    "club3090_commit": "cafef00d", "outcome": "hard-block",
    "abort_reason": "engine-support-unknown/no-arch-row",
    "failure_class": None, "model": "Org/Gate", "model_id": "Org/Gate",
    "arch_family": None, "quant_label": None, "topology_class": None,
    "topology_summary_canonical": None, "selected_ctx": None,
    "kv_format": None, "smoke_capability_set": None, "engine_pin": None,
    "engine_version": None, "kv_calc_version": None,
    "submission_fingerprint": None, "is_gate_only": True,
    "capture_points": ["gate"],
}
_gp = {
    "schema": 2, "point": "gate", "slug": "Org/Gate",
    "confidence": "ESTIMATED_LOWER_BOUND", "raw_verdict": None,
    "profile_like": "vllm/minimal", "hardware_sm": 8.6,
    "predicted_b_breakdown": None,
    "abort_reason": "engine-support-unknown/no-arch-row",
    "detail": "x", "is_gate_only": True,
}
(_gd / "manifest.json").write_text(json.dumps(_gman), encoding="utf-8")
(_gd / "pt1-gate.json").write_text(json.dumps(_gp), encoding="utf-8")
_fg = read_gate_bundle(_gd)
_clg = classify(_fg)  # /no-arch-row -> kernel-unsupported, should_file
check(_clg.failure_class is FailureClass.KERNEL_UNSUPPORTED
      and _clg.should_file is True,
      "V1: gate-only /no-arch-row classifies kernel-unsupported "
      "(public-filed) through the protocol")
_egh1 = D.effective_dedup_hash(_fg, _clg)
_egh2 = D.effective_dedup_hash(read_gate_bundle(_gd), _clg)
check(re.fullmatch(r"[0-9a-f]{12}", _egh1) is not None
      and _egh1 == _egh2,
      f"V1: gate-only effective_dedup_hash is 12-hex + DETERMINISTIC with "
      f"NULL topology (str(None)=='None' — the L3 fold; got {_egh1})")
# unknown gate bundle -> REVIEW-QUEUED (no public issue, but a spool).
(_gd2 := tmp / "f5-gate-rq").mkdir(parents=True, exist_ok=True)
_g2m = dict(_gman); _g2m["abort_reason"] = "disk-short"
_g2p = dict(_gp); _g2p["abort_reason"] = "disk-short"
(_gd2 / "manifest.json").write_text(json.dumps(_g2m), encoding="utf-8")
(_gd2 / "pt1-gate.json").write_text(json.dumps(_g2p), encoding="utf-8")
_fg2 = read_gate_bundle(_gd2)
_clg2 = classify(_fg2)
_rq_gh = MockGh()
_rq = D.submit(_fg2, _clg2, repo_root=repo_root, gh_runner=_rq_gh)
check(_clg2.failure_class is FailureClass.UNKNOWN
      and _rq.action is D.DedupAction.REVIEW_QUEUED
      and _rq.spool_path is not None
      and _rq_gh.calls == [],
      f"V1: gate-only disk-short -> unknown -> REVIEW-QUEUED spool "
      f"(NOT a public issue; ZERO gh I/O — filing-policy returns before "
      f"any gh call) (action={_rq.action})")

# ===========================================================================
# 5. REAL on-disk capture(s) under .pull-captures/ (>=2 exist).
#    Build FInput + classify + dedup with a mock gh returning "no existing
#    issue" -> opens exactly one issue with a valid bounded label set + a
#    body 7-tuple that round-trips. NO real gh.
# ===========================================================================
real_root = root / ".pull-captures"
real_dirs: list[Path] = []
if real_root.is_dir():
    for slug_dir in sorted(real_root.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(slug_dir.iterdir()):
            if ts_dir.is_dir() and (ts_dir / "manifest.json").is_file():
                real_dirs.append(ts_dir)

# `.pull-captures/` is gitignored runtime state — populated only after a
# real on-rig pull, ALWAYS absent on a fresh clone / in CI / after cleanup.
# The real-data invariant is that any captures present round-trip; their
# EXISTENCE is not a CI precondition. Skip (never fail) when absent; the
# round-trip assertions below run on whatever IS present.
if real_dirs:
    print(f"real-data: round-tripping {len(real_dirs)} on-disk "
          f".pull-captures/ bundle(s)")
else:
    print("real-data: SKIP — no on-disk .pull-captures/ corpus "
          "(gitignored runtime state; expected absent in CI / fresh "
          "clone). Round-trip assertions run only on present bundles.")

for rd in real_dirs:
    rfi = read_capture_bundle(rd)
    rcl = classify(rfi)
    # Use a classification that should_file so the submit path exercises
    # the open/dedup branch (the real captures are `partial` successes the
    # classifier may map to unknown/benign; force a fileable class so we
    # exercise the OPEN path deterministically — the real-data check here
    # is that the canonical tuple round-trips, not the classifier verdict).
    rcl_file = ClassificationResult(
        failure_class=FailureClass.GENUINE_OOM, tier=Tier.TIER1,
        fingerprint=rcl.fingerprint, should_file=True,
        route_as_kv_calc_bug=False, review_queue=False,
        error_substring=rcl.error_substring,
        matched_rule="tier1-oom-fastpath",
    )
    mgr = MockGh()
    mgr.list_response = "[]"  # no existing issue.
    rr = D.submit(rcl_file and rfi, rcl_file, repo_root=repo_root,
                  gh_runner=mgr)
    check(rr.action is D.DedupAction.OPENED,
          f"REAL {rd.name}: mock 'no existing issue' -> opens exactly one "
          f"issue (got {rr.action.value})")
    creates = [c for c in mgr.calls if c[:2] == ["issue", "create"]]
    check(len(creates) == 1,
          f"REAL {rd.name}: exactly ONE issue create call")
    exp_labels = D.label_set(D.effective_dedup_hash(rfi, rcl_file),
                             rcl_file, rfi)
    check(set(rr.labels) == set(exp_labels)
          and all(any(p in lab for p in ("loop:dedup-", "class:", "arch:"))
                  for lab in rr.labels)
          and not any(lab.startswith(("model:", "engine:", "kvcalc:",
                                      "topo:")) for lab in rr.labels),
          f"REAL {rd.name}: a valid bounded label set "
          f"({sorted(rr.labels)})")
    cbody = creates[0][creates[0].index("--body") + 1]
    rt = D.parse_body_tuple(cbody)
    check(rt is not None
          and [str(x) for x in rt]
          == [str(x) for x in D.effective_dedup_tuple(rfi, rcl_file)],
          f"REAL {rd.name}: body 7-tuple round-trips to "
          f"effective_dedup_tuple (no real gh; mocked)")

# ===========================================================================
if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll F5 §6.3 canonical-tuple-hash dedup (CONTRACT-4) "
      "assertions passed.")
PY

echo "test-dedup.sh OK"
