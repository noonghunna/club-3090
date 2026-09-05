#!/usr/bin/env bash
# test-pull-download-ladder.sh — the #804 download resilience ladder.
#
# The ladder exists because huggingface_hub CLIENT-SIDE refuses the classic
# (non-Xet) HTTP path for files over 50 GB — verbatim, from hf 1.25.1:
#
#   The file is too large to be downloaded using the regular download method.
#    Install `hf_xet` with `pip install hf_xet` for xet-powered downloads.
#
# ...while the server serves the bytes fine (resolve 302 -> CAS, HTTP 200,
# resumable, sha256 in x-linked-etag). Hit in the wild on a 96 GB single-file
# GGUF. Retrying rung 1 cannot succeed, so the ladder escalates instead.
#
# Asserted here (all hermetic — NO network, NO multi-GB transfer, every rung's
# transport injected):
#   1. the refusal signature is recognised, and ordinary flakiness is NOT
#      (a false positive sends a recoverable error down the slow path; a false
#      negative burns 30 opaque retries on a hard wall);
#   2. rung 1 success does NOT escalate — the classic path stays first and
#      unchanged, and Xet is never touched;
#   3. NO opt-in -> rung 2 is SKIPPED with the reason announced, rung 3 runs;
#   4. opt-in but hf_xet not importable -> skipped, and NOTHING auto-installs;
#   5. opt-in + hf_xet importable -> rung 2 runs, followed by the MANDATORY
#      independent sha256 gate;
#   6. a rung-2/rung-3 artifact whose sha256 mismatches HARD-FAILS and is
#      DELETED (a corrupt weight left on disk is worse than none — someone
#      will serve it);
#   7. a file the hub publishes no sha256 for is an unverifiable-artifact hard
#      fail on the fallback rungs, never waved through;
#   8. rung 3's curl argv is single-stream + resumable and points at the
#      RESOLVE endpoint, never a signed CAS URL (those expire in ~3600 s, so a
#      cached one turns a long resume into a 403);
#   9. NO parallel-chunk downloader appears anywhere in the executable path —
#      aria2c has silently corrupted multi-GB files on this rig at the correct
#      size, so it is banned by test, not by convention;
#  10. end-to-end through downloader.download_model: the refusal escalates,
#      the result reports which rung delivered, and the ladder's verification
#      is not silently re-done on a 96 GB file.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import io
import os
import sys
import tempfile
import tokenize
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import downloader as DL     # noqa: E402
from scripts.lib.profiles import hf_fetch as HF       # noqa: E402
from scripts.lib.profiles.einput import EInput        # noqa: E402

failures: list[str] = []


def check(cond, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# The verbatim refusal, as `hf download` surfaces it (issue #804 body).
REFUSAL = (
    "Error: Invalid value. The file is too large to be downloaded using the "
    "regular download method. Install hf_xet with pip install hf_xet for "
    "xet-powered downloads."
)
BIG = "laguna-s-2.1-Q4_K_M.gguf"
BODY = b"NINETY-SIX-GIGABYTES-PRETEND"
SHA = hashlib.sha256(BODY).hexdigest()
TREE = [{"type": "file", "path": BIG, "size": len(BODY),
         "lfs": {"sha256": SHA, "size": len(BODY)}}]


class Log:
    """Capture the announcements; the ladder's auditability IS the feature."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        self.lines.append(str(msg))

    def text(self) -> str:
        return "\n".join(self.lines)


def curl_runner_ok(body=BODY):
    """Stand in for curl: honour `-o <dest>` and write `body`. Records argv."""
    seen = {"argv": None, "calls": 0}

    def _run(argv, dest):
        seen["argv"] = list(argv)
        seen["calls"] += 1
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(body)
        return 0

    _run.seen = seen
    return _run


# ---------------------------------------------------------------------------
# 1. refusal signature — precise, both directions.
# ---------------------------------------------------------------------------
check(HF.is_xet_refusal(REFUSAL),
      "the verbatim too-large refusal is recognised as a hard wall")
check(HF.is_xet_refusal(REFUSAL.upper()),
      "signature match is case-insensitive (CLI casing varies)")
for transient in (
    "ConnectionResetError: [Errno 104] Connection reset by peer",
    "requests.exceptions.ReadTimeout: HTTPSConnectionPool(...)",
    "OSError: [Errno 28] No space left on device",
    "the file is too large for this filesystem",     # 'too large', no hf_xet
    "Xet Storage is enabled for this repo, but the 'hf_xet' package is not "
    "installed.",                                     # a WARNING, not the wall
):
    check(not HF.is_xet_refusal(transient),
          f"ordinary failure is NOT mistaken for the wall: {transient[:44]!r}")

# ---------------------------------------------------------------------------
# 2. rung 1 success does not escalate (classic stays first and untouched).
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    f = DL.HubFetcher()
    f.xet_available = lambda: (_ for _ in ()).throw(
        AssertionError("Xet must not even be probed on a rung-1 success"))
    _orig_which, _orig_run = DL.shutil.which, DL.subprocess.run
    DL.shutil.which = lambda n: "/usr/bin/hf" if n == "hf" else None

    def _run_ok(argv, **kw):
        d = Path(argv[argv.index("--local-dir") + 1])
        (d / BIG).parent.mkdir(parents=True, exist_ok=True)
        (d / BIG).write_bytes(BODY)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    DL.subprocess.run = _run_ok
    try:
        out = f.snapshot(repo_id="org/Repo", local_dir=td, allow_patterns=[BIG])
    finally:
        DL.shutil.which, DL.subprocess.run = _orig_which, _orig_run
    check(out == [BIG] and f.rung == "classic",
          f"rung 1 (classic LFS) success returns without escalating "
          f"(rung={f.rung!r} files={out})")

# ---------------------------------------------------------------------------
# 3. NO opt-in -> rung 2 skipped WITH a reason; rung 3 delivers + verifies.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    log, curl = Log(), curl_runner_ok()
    res = HF.escalate(
        "poolside/Laguna-S-2.1-GGUF", Path(td), [BIG],
        classic_error=REFUSAL, allow_xet=False, token=None, log=log,
        curl_runner=curl,
        xet_runner=lambda *a: (_ for _ in ()).throw(
            AssertionError("rung 2 must NOT run without an opt-in")),
        xet_available=lambda: True,      # available, but NOT permitted
        tree_fn=lambda: TREE,
    )
    check(res.ok and res.rung == "raw-resolve",
          f"no opt-in -> rung 3 (raw resolve) delivers (rung={res.rung!r})")
    check(res.verified == [BIG],
          f"rung 3 output is sha256-verified, not merely fetched "
          f"({res.verified})")
    check("rung 2 (Xet) SKIPPED" in log.text()
          and "opt-in required" in log.text(),
          "rung 2 skip is ANNOUNCED with its reason (never silent)")
    check("rung 1" in log.text() and "policy wall" in log.text(),
          "the announcement explains WHY rung 1 could not be retried")
    check("DONE via rung 3" in log.text()
          and "sha256-verified" in log.text(),
          "the completion line names the rung AND what was verified")

# ---------------------------------------------------------------------------
# 4. opt-in but hf_xet missing -> skipped, and the ladder NEVER installs it.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    log, curl = Log(), curl_runner_ok()
    res = HF.escalate(
        "org/Big", Path(td), [BIG], classic_error=REFUSAL, allow_xet=True,
        token=None, log=log, curl_runner=curl,
        xet_runner=lambda *a: (_ for _ in ()).throw(
            AssertionError("rung 2 must NOT run when hf_xet is unavailable")),
        xet_available=lambda: False,
        tree_fn=lambda: TREE,
    )
    check(res.ok and res.rung == "raw-resolve",
          "opted in but hf_xet unimportable -> falls through to rung 3")
    check("not importable" in log.text()
          and "never auto-installs" in log.text(),
          "the skip says hf_xet is missing AND that we will not install it")

# ---------------------------------------------------------------------------
# 5. opt-in + hf_xet importable -> rung 2 runs, THEN the mandatory sha gate.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    log = Log()
    xet_calls = {"n": 0, "xet": None}

    def _xet(repo, local_dir, includes, xet):
        xet_calls["n"] += 1
        xet_calls["xet"] = xet
        p = Path(local_dir) / BIG
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(BODY)
        return 0, ""

    res = HF.escalate(
        "org/Big", Path(td), [BIG], classic_error=REFUSAL, allow_xet=True,
        token=None, log=log, xet_runner=_xet, xet_available=lambda: True,
        curl_runner=lambda *a: (_ for _ in ()).throw(
            AssertionError("rung 3 must not run after a rung-2 success")),
        tree_fn=lambda: TREE,
    )
    check(res.ok and res.rung == "xet" and xet_calls["n"] == 1,
          f"rung 2 runs when opted in AND available (rung={res.rung!r})")
    check(xet_calls["xet"] is True,
          "rung 2 re-runs the SAME hf download with Xet ENABLED")
    check(res.verified == [BIG],
          "rung 2's artifact still goes through the INDEPENDENT sha256 gate "
          "(a wrapper-reported 'verified' has lied on this rig before)")

# ---------------------------------------------------------------------------
# 6. sha mismatch on a fallback rung -> hard fail + the artifact is DELETED.
# ---------------------------------------------------------------------------
def _xet_corrupt(repo, local_dir, includes, xet):
    p = Path(local_dir) / BIG
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"CORRUPT")
    return 0, ""


for rung_name, kwargs in (
    ("rung 3", dict(allow_xet=False, curl_runner=curl_runner_ok(b"CORRUPT"),
                    xet_available=lambda: False)),
    ("rung 2", dict(allow_xet=True, xet_available=lambda: True,
                    xet_runner=_xet_corrupt)),
):
    with tempfile.TemporaryDirectory() as td:
        log = Log()
        err = None
        try:
            HF.escalate("org/Big", Path(td), [BIG], classic_error=REFUSAL,
                        token=None, log=log, tree_fn=lambda: TREE, **kwargs)
        except HF.LadderError as exc:
            err = exc
        check(err is not None and err.token == "sha-mismatch",
              f"{rung_name}: sha256 mismatch is a HARD fail "
              f"(token={getattr(err, 'token', None)!r})")
        check(not (Path(td) / BIG).exists(),
              f"{rung_name}: the mismatching artifact is DELETED, not left "
              f"on disk for someone to serve")

# ---------------------------------------------------------------------------
# 7. no published sha256 -> unverifiable artifact, hard fail + delete.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    log = Log()
    no_hash_tree = [{"type": "file", "path": BIG, "size": len(BODY)}]
    err = None
    try:
        HF.escalate("org/Big", Path(td), [BIG], classic_error=REFUSAL,
                    allow_xet=False, token=None, log=log,
                    curl_runner=curl_runner_ok(), xet_available=lambda: False,
                    tree_fn=lambda: no_hash_tree)
    except HF.LadderError as exc:
        err = exc
    check(err is not None and err.token == "no-etag",
          f"a fallback artifact with no published sha256 hard-fails "
          f"(token={getattr(err, 'token', None)!r}) — never waved through")
    check(not (Path(td) / BIG).exists(),
          "the unverifiable artifact is deleted too")

# ---------------------------------------------------------------------------
# 8. rung 3's curl argv: single stream, resumable, RESOLVE endpoint.
# ---------------------------------------------------------------------------
argv = HF.curl_argv("poolside/Laguna-S-2.1-GGUF", BIG, Path("/tmp/x.gguf"))
joined = " ".join(argv)
check(argv[0].endswith("curl"), f"rung 3 transport is curl (got {argv[0]!r})")
check("-L" in argv, "curl follows the 302 (-L) — afresh on every attempt")
check("-C" in argv and argv[argv.index("-C") + 1] == "-",
      "curl resumes from whatever is on disk (-C -)")
check("--retry-all-errors" in argv, "curl retries transient errors")
url = argv[-1]
check(url.startswith("https://huggingface.co/poolside/Laguna-S-2.1-GGUF/"
                     "resolve/main/"),
      f"the URL is the RESOLVE endpoint, re-signed per attempt — NOT a "
      f"cached CAS link that expires in ~3600s (got {url})")
check("cas-bridge" not in url and "X-Xet-Cas-Uid" not in url
      and "Expires=" not in url,
      "no signed/expiring CAS URL is ever baked into the retry command")
for parallel in ("-Z", "--parallel", "--parallel-max"):
    check(parallel not in argv,
          f"rung 3 is SINGLE-STREAM: no {parallel} (parallel-chunk fetchers "
          f"have corrupted multi-GB files on this rig at the correct size)")

# ---------------------------------------------------------------------------
# 9. no parallel-chunk downloader anywhere in the EXECUTABLE path.
#    Comments/docstrings may name aria2c (they explain the ban); code may not.
# ---------------------------------------------------------------------------
for mod in ("scripts/lib/profiles/hf_fetch.py",
            "scripts/lib/profiles/downloader.py"):
    src = (root / mod).read_text(encoding="utf-8")
    code = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        code.append(tok.string)
    blob = " ".join(code).lower()
    check("aria2" not in blob,
          f"{mod}: aria2c appears only in prose, never in executable code")

# ---------------------------------------------------------------------------
# 10. END TO END through download_model: refusal -> escalation -> reported.
# ---------------------------------------------------------------------------
API = {
    "sha": "cafebabe",
    "siblings": [
        {"rfilename": "config.json", "size": 12},
        {"rfilename": "model.safetensors", "size": len(BODY),
         "lfs": {"sha256": SHA, "size": len(BODY)}},
    ],
}


def mk_einput(slug, hf_home, api):
    der = SimpleNamespace(slug=slug, profile={"_hf_api": api})
    return EInput(
        slug=slug, terminal="proceed", is_override_accepted=False, der=der,
        runtime={}, selected_files=[], hf_home=Path(hf_home), c2a=None,
        hardware_sm=8.6, visible_gpu_count=1, per_gpu_vram_mib=[24576],
        selected_gpu_indices=[0], selected_gpu_vram_mib=[24576],
        topology_summary="(RTX 3090, 24576)", club3090_commit="deadbeef",
        diagnostics={},
    )


with tempfile.TemporaryDirectory() as td:
    log = Log()
    ei = mk_einput("org/Huge", td, API)
    fetcher = DL.HubFetcher(API, log=log)
    fetcher.xet_available = lambda: False
    hashed = {"n": 0}
    _orig_sha = DL._sha256_file

    def _counting_sha(p):
        hashed["n"] += 1
        return _orig_sha(p)

    def _curl(argv, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(BODY)
        return 0

    fetcher.curl_runner = _curl
    _orig_which, _orig_run = DL.shutil.which, DL.subprocess.run
    DL.shutil.which = lambda n: "/usr/bin/hf" if n == "hf" else None
    DL._sha256_file = _counting_sha

    def _run_refuse(argv, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr=REFUSAL)

    DL.subprocess.run = _run_refuse
    try:
        # config.json is not LFS -> rung 3 fetches it too; give it bytes.
        def _curl_multi(argv, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(
                BODY if Path(dest).name.endswith(".safetensors") else b"{}")
            return 0

        fetcher.curl_runner = _curl_multi
        # only the safetensors publishes a sha256; a non-LFS metadata file has
        # none, which on a fallback rung is (correctly) a hard fail -> restrict
        # the pull to the weight so the happy path is what we assert.
        API_W = {"sha": "cafebabe", "siblings": [API["siblings"][1]]}
        ei = mk_einput("org/Huge", td, API_W)
        fetcher.api = API_W
        r = DL.download_model(ei, fetcher=fetcher)
    finally:
        DL.shutil.which, DL.subprocess.run = _orig_which, _orig_run
        DL._sha256_file = _orig_sha

    check(r.ok and r.failure is None,
          f"end-to-end: the too-large refusal ESCALATES instead of failing "
          f"(ok={r.ok} failure={r.failure!r})")
    check(r.rung == "raw-resolve",
          f"the result reports WHICH rung delivered the bytes "
          f"(rung={r.rung!r}) — logs stay auditable")
    check(hashed["n"] == 0,
          f"the ladder's own sha256 is not re-computed by the stage that "
          f"follows it (a 96 GB re-read for an identical comparison); "
          f"extra hashes={hashed['n']}")
    check("delivered via the #804 fallback ladder" in log.text(),
          "the completion announcement names the ladder + the rung")
    final = DL.pull_dir(Path(td), "org/Huge")
    check(final.is_dir() and not (final / ".incomplete").exists(),
          "atomic move + .incomplete cleanup are unchanged on the ladder path")

# ---------------------------------------------------------------------------
# 11. a NON-refusal rung-1 failure keeps its pre-existing mapping (no
#     escalation, no behaviour change for the 99% case).
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    f = DL.HubFetcher()
    f.curl_runner = lambda *a: (_ for _ in ()).throw(
        AssertionError("a transient error must NOT trigger the fallback"))
    _orig_which, _orig_run = DL.shutil.which, DL.subprocess.run
    DL.shutil.which = lambda n: "/usr/bin/hf" if n == "hf" else None
    DL.subprocess.run = lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="401 Client Error: Unauthorized")
    err = None
    try:
        f.snapshot(repo_id="org/Gated", local_dir=td, allow_patterns=[BIG])
    except BaseException as exc:
        err = exc
    finally:
        DL.shutil.which, DL.subprocess.run = _orig_which, _orig_run
    check(isinstance(err, DL.GatedError),
          f"a 401 still maps to GatedError — the ladder ONLY intercepts the "
          f"too-large refusal (got {type(err).__name__})")

# ---------------------------------------------------------------------------
# 12. PREFLIGHT: say "this repo needs Xet or the fallback" BEFORE the doomed
#     rung-1 attempt, not after 30 opaque retries (#804 acceptance bullet 3).
# ---------------------------------------------------------------------------
small = {"a.safetensors": HF.FileMeta("a.safetensors", size=8 * 1024**3,
                                      sha256="a" * 64)}
verdict, note = HF.preflight_transport(["a.safetensors"], small)
check(verdict == "classic" and note == "",
      f"a normal repo preflights as 'classic' with no noise ({verdict!r})")

huge = {BIG: HF.FileMeta(BIG, size=96 * 1024**3, sha256=SHA)}
verdict, note = HF.preflight_transport([BIG], huge)
check(verdict == "needs-fallback",
      f"a >50 GB file preflights as needs-fallback ({verdict!r})")
check("50 GB" in note,
      f"the ceiling is quoted the way upstream defines it (decimal 50 GB, "
      f"not a binary 46.57 GB nobody will recognise): {note}")
check("policy wall" in note and "retries cannot help" in note,
      "the preflight says retrying is futile, so nobody sits through 30 of them")
check("--allow-xet" in note and "rung 3" in note,
      f"the preflight names BOTH escape routes: {note}")
check(BIG in note and "89" in note or "96" in note,
      f"the offending file and its size are named: {note}")

# The same warning reaches the operator through download_model's own log.
with tempfile.TemporaryDirectory() as td:
    log = Log()
    BIG_API = {"siblings": [
        {"rfilename": "model.safetensors", "size": 96 * 1024**3,
         "lfs": {"sha256": SHA, "size": 96 * 1024**3}}]}
    ei = mk_einput("org/Big", td, BIG_API)
    f = DL.HubFetcher(BIG_API, log=log)

    def _snap(repo_id, local_dir, allow_patterns):
        raise DL.DiskError("stop here — the preflight line is what we assert")

    f.snapshot = _snap
    DL.download_model(ei, fetcher=f)
    check("[download] preflight:" in log.text() and "50 GB" in log.text(),
          f"download_model announces the transport verdict BEFORE fetching "
          f"(pull.sh streams this straight into c3's lane):\n{log.text()}")

# ---------------------------------------------------------------------------
# 13. the module CLI exists and is self-describing (so a surface that shells
#     out to `hf download` today can adopt the ladder without importing).
# ---------------------------------------------------------------------------
check(callable(getattr(HF, "_main", None)) and callable(getattr(HF, "fetch", None)),
      "hf_fetch exposes a standalone fetch() + a CLI entry point")

# ---------------------------------------------------------------------------
# #1132 -- a missing `hf` binary must ESCALATE to rung 3 instead of dying with
# a bare exit 1, and _main must announce the failure token + detail on stderr.
# ---------------------------------------------------------------------------
import subprocess as _sp

_orig_tree, _orig_which, _orig_run, _orig_raw = (
    HF.repo_tree, HF.shutil.which, HF.subprocess.run, HF.raw_resolve_fetch)
try:
    HF.repo_tree = lambda *a, **k: TREE

    # (a) tool unavailable -> rung 3 serves the pull; entry names the cause.
    # null BOTH probe names: a rig with `huggingface-cli` installed would
    # otherwise run the real downloader here.
    HF.shutil.which = lambda n: None if n in ("hf", "huggingface-cli") else _orig_which(n)
    def _fake_raw(repo, name, local_dir, **kw):
        dest = Path(local_dir) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(BODY)
        return dest
    HF.raw_resolve_fetch = _fake_raw
    log127 = Log()
    with tempfile.TemporaryDirectory() as td:
        def _runner_127(repo, local_dir, includes, xet):
            return 127, "neither `hf` nor `huggingface-cli` is on PATH"
        res = HF.fetch("org/Repo", Path(td), [BIG], allow_xet=False,
                       log=log127)
    check(res.ok and res.rung == "raw-resolve",
          f"#1132 missing hf binary: rung 3 serves the pull "
          f"(rung={res.rung!r} ok={res.ok})")
    check(res.verified == [BIG], "#1132 missing hf: sha256 gate still runs")
    check("never stood a chance" in log127.text()
          and "not on PATH" in log127.text(),
          "#1132 missing hf: entry announcement names the real cause")
    check("rung 3" in log127.text(), "#1132 missing hf: escalation reached rung 3")

    # (b) a NON-tool failure (network/auth) stays a hard failure -- no escalation.
    HF.shutil.which = lambda n: "/usr/bin/hf" if n == "hf" else None
    HF.subprocess.run = lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="", stderr="connection reset by peer")
    lognet = Log()
    with tempfile.TemporaryDirectory() as td:
        def _runner_netfail(repo, local_dir, includes, xet):
            return 1, "connection reset by peer"
        res = HF.fetch("org/Repo", Path(td), [BIG], allow_xet=False,
                       log=lognet)
    check(not res.ok and res.failure == "fallback-failed",
          "#1132 non-tool rung-1 failure stays a hard failure")
    check("rung 3" not in lognet.text(),
          "#1132 non-tool failure does NOT escalate")

    # (c) _main announces the failure token + detail on stderr, then exits 1.
    HF.subprocess.run = _orig_run
    probe = (
        "import sys; sys.path.insert(0, %r); "
        "from scripts.lib.profiles import hf_fetch as HF;"
        "HF.fetch = lambda *a, **k: HF.LadderResult("
        "ok=False, failure='fallback-failed', "
        "detail='rung 1 failed (rc=1): boom');"
        "sys.exit(HF._main(['some/repo', '--local-dir', 'x']))" % str(root)
    )
    proc = _sp.run([sys.executable, "-c", probe], capture_output=True, text=True)
    check(proc.returncode == 1, "#1132 _main still exits 1 on failure")
    check("FAILED: fallback-failed" in proc.stderr and "boom" in proc.stderr,
          "#1132 _main announces the failure token + detail on stderr")
finally:
    HF.repo_tree, HF.shutil.which, HF.subprocess.run, HF.raw_resolve_fetch = (
        _orig_tree, _orig_which, _orig_run, _orig_raw)

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll #804 download-ladder assertions passed.")
PY

echo "test-pull-download-ladder.sh OK"
