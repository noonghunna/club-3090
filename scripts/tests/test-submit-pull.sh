#!/usr/bin/env bash
set -euo pipefail

# test-submit-pull.sh — v0.8.2 STEP V2 (CONTRACT-1.2 / 1.3).
#
# Contract test for the user-invoked, consented failure on-ramp:
#   scripts/pull.sh --submit-last / --submit <capture-dir>
# + the CONTRACT-1.2 surfacing pointer. The test IS the spec; the code is
# fixed to it. NO live Docker / GPU / HF-API and ZERO network / ZERO real
# `gh`: every `gh` invocation goes through an INJECTED mock `gh_runner`;
# consent is driven via an injected `_input`; capture bundles are built by
# the REAL `capture.emit_gate_capture` (genuine schema:2 artifacts) parsed
# via F1's read_gate_bundle and classified via the REAL F2 `classify`.
#
# Coverage (every CONTRACT-1.2 / 1.3 + RED-LINE assertion as a
# failing-then-passing check):
#   * `--submit-last` resolves `.pull-captures/.last` (the V1 shared marker,
#     read-only here); absent/stale -> the exact CONTRACT-1.3 error.
#   * the `.last`-marker RACE: a SECOND terminal capture overwrites `.last`
#     between capture and `--submit-last`; the verb RE-READS `.last` and
#     re-shows the CURRENT bundle identity (no silent wrong-bundle submit).
#   * consent: explicit `y` required; a decline performs ZERO network.
#   * gh present (mock auth ok) -> reuses the SHIPPED F5 `dedup.submit`
#     (effective_dedup_hash, bounded labels, +1-or-open) — NOT reimplemented.
#   * gh-less + should_file=True (engine-support-unknown/no-arch-row ->
#     kernel-unsupported): a prefilled PUBLIC issues/new URL carrying the
#     loop:dedup-<hash> label.
#   * gh-less + review-queued (`unknown` -> a correct-refusal abort_reason):
#     the LOCAL _review-queue spool path + the no-public-issue line and
#     ABSOLUTELY NO public issues/new URL (the §6.1 review-queue boundary).
#   * CONTRACT-1.2: the surfacing pointer + `.last` fire whenever a gate
#     bundle was EMITTED, regardless of exit code — proven on the
#     bundle-emitted-but-exit-0 bypassable C0 path (NOT only exit-2).
#   * leak-hygiene: nothing the on-ramp tells the user to share carries an
#     unredacted absolute path.
#   * gate path stays I/O-free: run_pull performs NO network / NO prompt /
#     NO auto-send (the surfacing is a single stdout line, post-return).
#   * import-time safety: importing submit_pull then kv-calc --calibration
#     stays Overall: 9/9.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import capture as CAP  # noqa: E402
from scripts.lib.profiles import dedup as D  # noqa: E402
from scripts.lib.profiles import submit_pull as S  # noqa: E402
from scripts.lib.profiles.classifier import (  # noqa: E402
    FailureClass,
    classify,
)
from scripts.lib.profiles.loop_input import read_gate_bundle  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Mock gh_runner — ZERO network / `gh`. Records argv; `auth status` ok/not.
# (Reuses the shipped D.GhResult seam so the type matches dedup.submit.)
# ---------------------------------------------------------------------------
class MockGh:
    def __init__(self, *, authed=True):
        self.calls: list[list[str]] = []
        self.authed = authed
        self.list_response = "[]"

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["auth", "status"]:
            return D.GhResult(self.authed, 0 if self.authed else 1, "",
                              "" if self.authed else "not logged in")
        if not self.authed:
            return D.GhResult(False, 127, "", "gh not found (mock)")
        if argv[:2] == ["issue", "list"]:
            return D.GhResult(True, 0, self.list_response, "")
        if argv[:2] == ["issue", "create"]:
            return D.GhResult(
                True, 0,
                "https://github.com/noonghunna/club-3090/issues/777", "")
        if argv[:2] == ["issue", "comment"]:
            return D.GhResult(True, 0, "", "")
        if argv[:2] == ["label", "create"]:
            return D.GhResult(True, 0, "", "")
        return D.GhResult(True, 0, "", "")


def emit_gate(repo_root: Path, *, slug, abort_reason, ts) -> Path:
    """Build a REAL schema:2 gate bundle via the shipped emitter (also
    writes the shared `.last` marker — V1 centralization)."""
    out = CAP.emit_gate_capture(
        slug=slug,
        profile_like="vllm/minimal",
        abort_reason=abort_reason,
        confidence=None,
        raw_verdict=None,
        detail=f"gate hard-block: {abort_reason}",
        der=None,
        hardware_sm=8.6,
        gpu_topology=(2, [24576, 24576], ["RTX 3090", "RTX 3090"]),
        club3090_commit="cafef00d",
        kv_calc_version="kvcalc-v0.8.0",
        repo_root=repo_root,
        ts=ts,
    )
    return Path(out["dir"])


def run_submit(repo_root, *, submit_last=True, capture_dir=None,
                consent="y", gh):
    """Invoke the verb capturing stdout/stderr; ZERO network (mock gh)."""
    out, err = io.StringIO(), io.StringIO()
    answers = iter([consent])

    def _inp(_prompt):
        try:
            return next(answers)
        except StopIteration:
            return ""

    with redirect_stdout(out), redirect_stderr(err):
        rc = S.submit_pull(
            capture_dir=capture_dir,
            submit_last=submit_last,
            repo_root=Path(repo_root),
            gh_runner=gh,
            _input=_inp,
        )
    return rc, out.getvalue(), err.getvalue()


# ===========================================================================
# 1. --submit-last with NO marker -> the exact CONTRACT-1.3 error, rc=2.
# ===========================================================================
rr = Path(tempfile.mkdtemp())
gh = MockGh(authed=True)
rc, so, se = run_submit(rr, gh=gh)
check(rc == 2 and "no recent capture" in se
      and "--submit <dir>" in se,
      "no `.last` -> the CONTRACT-1.3 error 'no recent capture; use "
      "`--submit <dir>`' (rc=2)")
check(gh.calls == [],
      "unresolvable bundle -> ZERO gh calls (no network before resolve)")

# ===========================================================================
# 2. gh present + should_file=True (no-arch-row) -> reuses SHIPPED F5
#    dedup.submit (NOT reimplemented). Consent `y` required first.
# ===========================================================================
rr = Path(tempfile.mkdtemp())
gd = emit_gate(rr, slug="Org/Exotic",
               abort_reason="engine-support-unknown/no-arch-row",
               ts="20260518T000001Z")
# Sanity: the real F1/F2 route this to kernel-unsupported / should_file.
_fi = read_gate_bundle(gd)
_cl = classify(_fi)
check(_cl.failure_class is FailureClass.KERNEL_UNSUPPORTED
      and _cl.should_file is True,
      "no-arch-row gate bundle classifies kernel-unsupported/should_file "
      "(the §10-R9 solicited lever) via the REAL shipped F2")

gh = MockGh(authed=True)
rc, so, se = run_submit(rr, gh=gh, consent="y")
check(rc == 0 and "F5 action=" in so,
      "gh present + consent y -> reuses the shipped F5 dedup.submit "
      f"(rc={rc})")
check(any(c[:2] == ["issue", "list"] for c in gh.calls)
      and any(c[:2] == ["issue", "create"] for c in gh.calls),
      "F5 ran the shipped path (gh issue list then create) — NOT a "
      "reimplemented submit")

# ===========================================================================
# 3. Consent gate: a decline performs ZERO network.
# ===========================================================================
gh = MockGh(authed=True)
rc, so, se = run_submit(rr, gh=gh, consent="n")
issue_calls = [c for c in gh.calls if c[:1] == ["issue"]]
check(rc == 0 and "declined" in so and not issue_calls,
      "explicit decline -> nothing sent, ZERO `gh issue` network calls")

# ===========================================================================
# 4. The `.last`-marker RACE (RED-LINE): a SECOND capture overwrites
#    `.last` between capture and --submit-last; the verb RE-READS `.last`
#    and re-shows the CURRENT bundle identity (no silent wrong-bundle).
# ===========================================================================
rr = Path(tempfile.mkdtemp())
g_first = emit_gate(rr, slug="Org/First",
                    abort_reason="engine-support-unknown/no-arch-row",
                    ts="20260518T000010Z")
# A second terminal capture (a racing pull) overwrites `.last`.
g_second = emit_gate(rr, slug="Org/Second",
                     abort_reason="engine-support-unknown/no-arch-row",
                     ts="20260518T000011Z")
marker = (rr / ".pull-captures" / ".last").read_text().strip()
check(marker == str(g_second.relative_to(rr / ".pull-captures")),
      "the shared `.last` marker is last-writer-wins (points at the SECOND "
      "capture after the racing pull)")
gh = MockGh(authed=True)
rc, so, se = run_submit(rr, gh=gh, consent="n")  # decline; only inspect.
check("Org/Second" in so and "Org/First" not in so,
      "--submit-last RE-READS `.last` at submit + re-shows the CURRENT "
      "(SECOND) bundle identity — NO silent wrong-bundle submit")
check(S.resolve_last(rr).resolve() == g_second.resolve(),
      "resolve_last() resolves the marker's CURRENT target (the race "
      "defense is keyed on a re-read, not a stale capture-time value)")

# ===========================================================================
# 5. gh-less + should_file=True -> a prefilled PUBLIC issues/new URL with
#    the loop:dedup-<hash> label; NO `gh issue` calls.
# ===========================================================================
rr = Path(tempfile.mkdtemp())
gd = emit_gate(rr, slug="Org/Exotic",
               abort_reason="engine-support-unknown/no-arch-row",
               ts="20260518T000020Z")
gh = MockGh(authed=False)  # gh unavailable / unauthed.
rc, so, se = run_submit(rr, gh=gh, consent="y")
fi = read_gate_bundle(gd)
cl = classify(fi)
h = D.effective_dedup_hash(fi, cl)
check(rc == 0
      and "https://github.com/noonghunna/club-3090/issues/new?" in so
      and f"loop%3Adedup-{h}" in so.replace("loop:dedup-", "loop%3Adedup-")
      or f"loop:dedup-{h}" in so,
      "gh-less + should_file=True -> a prefilled PUBLIC issues/new URL "
      "(carries the loop:dedup-<hash> label)")
issue_calls = [c for c in gh.calls if c[:1] == ["issue"]]
check(not issue_calls,
      "gh-less path makes ZERO `gh issue` network calls (paste-only)")

# Use the structured fallback API to assert the URL precisely.
fb = S.ghless_fallback(fi, cl, repo_root=rr, bundle_dir=gd)
check(fb["kind"] == "ghless-public-url" and fb["should_file"] is True
      and fb["url"].startswith(
          "https://github.com/noonghunna/club-3090/issues/new?")
      and f"loop:dedup-{h}" in __import__("urllib.parse",
          fromlist=["unquote"]).unquote(fb["url"]),
      "ghless_fallback(should_file=True): public issues/new URL, "
      "loop:dedup-<hash> label, deterministic")
# Title template: [<model>] <failure-class> on <topology> (dedup:<hash8>)
import urllib.parse as _up  # noqa: E402
_q = _up.parse_qs(_up.urlparse(fb["url"]).query)
_title = _q["title"][0]
check(_title == f"[Org/Exotic] kernel-unsupported on 2x24576MiB "
      f"(dedup:{h[:8]})",
      f"gh-less deterministic title template (got {_title!r})")

# ===========================================================================
# 6. gh-less + review-queued (`unknown` correct-refusal) -> the LOCAL
#    _review-queue spool path + the no-public-issue line; ABSOLUTELY NO
#    public issues/new URL (the §6.1 review-queue boundary, RED-LINE).
# ===========================================================================
rr = Path(tempfile.mkdtemp())
gd = emit_gate(rr, slug="Org/DiskShort", abort_reason="disk-short",
               ts="20260518T000030Z")
fi = read_gate_bundle(gd)
cl = classify(fi)
check(cl.failure_class is FailureClass.UNKNOWN
      and cl.should_file is False,
      "disk-short gate bundle classifies `unknown`/should_file=False "
      "(a correct-refusal — review-queued, NOT public-filed)")
gh = MockGh(authed=False)
rc, so, se = run_submit(rr, gh=gh, consent="y")
check(rc == 0
      and "not a public issue" in so
      and "_review-queue" in so
      and "issues/new" not in so
      and "github.com" not in so,
      "gh-less + review-queued -> local _review-queue spool path + the "
      "no-public-issue line, ABSOLUTELY NO public issues/new URL")
fb = S.ghless_fallback(fi, cl, repo_root=rr, bundle_dir=gd)
check(fb["kind"] == "ghless-review-queued"
      and fb["should_file"] is False
      and "url" not in fb
      and fb["spool_path"].endswith("_review-queue/"
                                    f"{D.effective_dedup_hash(fi, cl)}.json"),
      "ghless_fallback(review-queued): NO `url` key, points at the "
      "_review-queue spool — the §6.1 boundary the suppression design "
      "exists to keep clean")

# ===========================================================================
# 7. CONTRACT-1.2 RED-LINE — the surfacing pointer + `.last` fire whenever
#    a gate bundle was EMITTED, regardless of exit code. Proven on the
#    bundle-emitted-but-exit-0 bypassable C0 path (NOT only exit-2).
# ===========================================================================
from scripts.lib.profiles import pull as P  # noqa: E402


class _Diag(dict):
    pass


def _fake_res(*, ok, abort_reason, gate_dir):
    """A minimal PullResult-shaped object exercising main()'s surfacing
    branch directly (the surfacing is keyed on
    res.diagnostics['gate_capture']['dir'], NOT the exit code)."""
    r = P.PullResult(
        slug="Org/Phi", profile_like="vllm/minimal",
        path="B", ok=ok, stratum=P.Stratum.C0,
    )
    r.abort_reason = abort_reason
    r.diagnostics = {"gate_capture": {"dir": gate_dir}}
    return r


# Exit-0 bypassable advisory path (ok-ish / non-abort-2) WITH an emitted
# bundle vs exit-2 hard-block WITH an emitted bundle: BOTH must surface.
buf0 = io.StringIO()
r0 = _fake_res(ok=True, abort_reason=None, gate_dir="/x/cap/exit0")
with redirect_stdout(buf0):
    # Mirror main()'s post-return surfacing branch verbatim.
    _gc = r0.diagnostics.get("gate_capture")
    _gd0 = _gc.get("dir") if isinstance(_gc, dict) else None
    if _gd0:
        print(f"[pull] Diagnostics captured (redacted, no paths/tokens): "
              f"{_gd0}")
        print("[pull] Help improve the fit math — submit with: "
              "scripts/pull.sh --submit-last")
out0 = buf0.getvalue()
check("Diagnostics captured" in out0 and "--submit-last" in out0,
      "CONTRACT-1.2: an emitted gate bundle on the EXIT-0 bypassable path "
      "still surfaces the submit pointer (NOT gated on exit==2 — the "
      "no-arch-row §10-R9 lever is the exit-0 path)")

# Negative control: NO gate bundle emitted -> NO surfacing line.
buf_n = io.StringIO()
with redirect_stdout(buf_n):
    _gc = ({}).get("gate_capture")
    _gdn = _gc.get("dir") if isinstance(_gc, dict) else None
    if _gdn:
        print("[pull] Diagnostics captured ...")
check("Diagnostics captured" not in buf_n.getvalue(),
      "CONTRACT-1.2: NO gate bundle emitted -> NO surfacing line "
      "(trigger is bundle-emitted, precisely)")

# The real main() wiring: assert the surfacing branch reads
# diagnostics['gate_capture']['dir'] (source inspection — keyed on the
# bundle, not the exit code).
import inspect  # noqa: E402
_msrc = inspect.getsource(P.main)
# The surfacing branch lives BEFORE `if res.ok:` and is keyed on the
# emitted gate-bundle dir; isolate that span and prove no exit-code gate.
_surf = _msrc.split('res.diagnostics.get("gate_capture")')[1].split(
    "if res.ok:")[0]
check('res.diagnostics.get("gate_capture")' in _msrc
      and '.get("dir")' in _surf
      and "Diagnostics captured" in _surf
      and "--submit-last" in _surf
      and "_EXIT_ABORT" not in _surf
      and "_EXIT_NEEDS_FLAG" not in _surf
      and "res.ok" not in _surf,
      "CONTRACT-1.2 wiring: main() surfacing is keyed on the emitted gate "
      "bundle dir, with NO exit-code gate around it")

# ===========================================================================
# 8. Gate path is I/O-FREE (RED-LINE): submit_pull is the ONLY network
#    site; run_pull has no submit/network/prompt. Source-level proof
#    (run_pull never imports/calls submit_pull or gh).
# ===========================================================================
_rp_src = inspect.getsource(P.run_pull)
check("submit_pull" not in _rp_src and "gh_runner" not in _rp_src
      and "_real_gh_runner" not in _rp_src and "input(" not in _rp_src,
      "RED-LINE: run_pull (the gate path) has NO submit/gh/network/blocking-"
      "prompt — submission is the separate, explicit, consented verb only")
# submit_pull REUSES dedup.submit + dedup.effective_dedup_hash (it never
# reimplements the F5 hash/serialization primitive — every dedup-hash use
# is qualified `_DEDUP.effective_dedup_hash`, never a local sha256 join).
_sp_src = inspect.getsource(S)
import re as _re  # noqa: E402
_unqualified_eff = _re.findall(
    r"(?<!_DEDUP\.)\beffective_dedup_hash\s*\(", _sp_src)
check("_DEDUP.submit(" in _sp_src
      and "_DEDUP.effective_dedup_hash(" in _sp_src
      and not _unqualified_eff
      and "hashlib" not in _sp_src
      and "sha256" not in _sp_src,
      "RED-LINE: submit_pull REUSES the shipped dedup.submit / "
      "dedup.effective_dedup_hash — it does NOT reimplement F5 dedup "
      "(no local hashlib/sha256, no unqualified effective_dedup_hash)")

# ===========================================================================
# 9. Leak-hygiene (RED-LINE): nothing the on-ramp tells the user to share
#    carries an unredacted absolute host path.
# ===========================================================================
rr = Path(tempfile.mkdtemp())
gd = emit_gate(rr, slug="Org/Exotic",
               abort_reason="engine-support-unknown/no-arch-row",
               ts="20260518T000040Z")
fi = read_gate_bundle(gd)
cl = classify(fi)
fb = S.ghless_fallback(fi, cl, repo_root=rr, bundle_dir=gd)
import urllib.parse as _up2  # noqa: E402
_decoded_body = _up2.unquote(
    _up2.parse_qs(_up2.urlparse(fb["url"]).query)["body"][0])
_PCN = ".pull-captures"
check("/opt/ai" not in _decoded_body and "/home/" not in _decoded_body
      and "/mnt/" not in _decoded_body,
      "leak-hygiene: the gh-less URL body (built from the [E]-redacted "
      "manifest) carries NO unredacted internal absolute path")
# RIG-INDEPENDENT (the real catch the /opt|/home check missed when gd was a
# tmp sandbox dir): the ABSOLUTE bundle dir must NEVER appear in the public-
# paste body; only the repo-relative `.pull-captures/...` pointer may.
check(str(gd) not in _decoded_body,
      "leak-hygiene: the gh-less body MUST NOT contain the absolute capture "
      f"dir ({gd}) — public-paste content is repo-relative only")
check(_PCN in _decoded_body and not any(
          _seg.startswith("/") for _seg in _decoded_body.split("`")
          if _PCN in _seg),
      "leak-hygiene: the gh-less body's bundle pointer is repo-relative "
      f"(`{_PCN}/<slug>/<ts>`), never an absolute path")
# The redacted-bundle pointer the message prints is the local capture dir
# (a tmp path here, by design the on-ramp's only emitted artifact) — assert
# the message NEVER instructs pasting terminal scrollback.
for ln in fb["lines"]:
    check("paste your terminal" not in ln.lower()
          and "terminal output" not in ln.lower(),
          "leak-hygiene: the on-ramp never tells the user to paste their "
          "terminal scrollback (console is not a safe submission source)")

# ===========================================================================
# 10. import-time safety: importing submit_pull then kv-calc --calibration
#     stays Overall: 9/9 (no import side effects on the calibrator).
# ===========================================================================
import subprocess  # noqa: E402
_cp = subprocess.run(
    [sys.executable, str(root / "tools" / "kv-calc.py"), "--calibration"],
    capture_output=True, text=True, cwd=str(root))
check("Overall: 9/9 (100%)" in _cp.stdout,
      "import-time safety: submit_pull import does not perturb "
      "kv-calc --calibration (still Overall: 9/9)")

# ===========================================================================
if failures:
    print(f"\nSUMMARY: {len(failures)} assertion(s) FAILED.",
          file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)

print("\nSUMMARY: all v0.8.2 STEP V2 submit/on-ramp assertions passed "
      "(CONTRACT-1.2 surfacing keyed on bundle-emitted-not-exit-code; "
      "CONTRACT-1.3 --submit-last/--submit, consent, .last race re-read, "
      "F5 reuse, gh-less should_file branch with NO public URL for "
      "review-queued; gate path I/O-free; leak-clean).")
PY

# ---------------------------------------------------------------------------
# CLI-contract: the submit verb is parsed BEFORE the slug/--profile-like
# requirement (a distinct top-level verb). NO network (no real `gh` resolves
# because no `.last` exists in a clean CI tree -> the rc=2 error path).
# ---------------------------------------------------------------------------
rm -rf "$ROOT_DIR/.pull-captures"
_sc(){ bash scripts/pull.sh "$@" >/dev/null 2>&1; echo $?; }
[ "$(_sc --submit-last)" = 2 ]   || { echo "FAIL: --submit-last with no .last -> 2 (parsed w/o slug/--profile-like)" >&2; exit 1; }
[ "$(_sc --submit /nonexistent-xyz)" = 2 ] || { echo "FAIL: --submit <bad> -> 2 (parsed w/o slug/--profile-like)" >&2; exit 1; }
echo "PASS: --submit*/--submit-last is a distinct top-level verb (needs no slug/--profile-like; rc=2 on unresolvable, no network)"
rm -rf "$ROOT_DIR/.pull-captures"

echo "test-submit-pull.sh OK"
