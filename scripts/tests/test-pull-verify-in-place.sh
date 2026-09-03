#!/usr/bin/env bash
# test-pull-verify-in-place.sh — #812: verify-in-place for metadata-less local
# files, and announce WHY a re-download starts.
#
# The reported failure (Discord, lifeform): setup/pull re-downloads a model on
# first run even though the file is already at the target path. Mechanism: it
# was placed by hand, so no `.cache/huggingface/download/<f>.metadata` sits
# beside it, the skip check cannot prove completeness, and a full re-pull starts
# — silently, for hours. The instinct is right (a half-copied 40 GB file must
# never be silently trusted); the silence is the bug.
#
# Asserted here (hermetic — no network, no real transfer):
#   1. a correct, metadata-less file is ADOPTED after a hash match, with ZERO
#      bytes re-downloaded (the fetcher is not called at all);
#   2. adoption writes the HF metadata stub in huggingface_hub's exact 3-line
#      format at its exact path, so `hf download` skips the file from now on;
#   3. a TRUNCATED file is rejected with a message naming BOTH sizes, then
#      re-downloaded;
#   4. a corrupt-but-right-sized file is caught by the hash and re-downloaded
#      (this is the exact shape of the silent-corruption incident this rig has
#      already been burned by: correct size, wrong bytes);
#   5. without VERIFY_IN_PLACE the file is still re-downloaded — but the reason
#      is stated, along with how to opt into verifying instead;
#   6. NO code path starts a download without printing why;
#   7. the word "verified" NEVER appears for a size-only observation. A wrapper
#      on this rig once printed "DONE (hash-verified)" over silent corruption;
#      this suite makes that phrasing a test failure, not a style note;
#   8. a partially-present pull dir adopts what it can and fetches only the
#      remainder (one missing sibling must not re-pull three good shards).
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
import os
import re
import sys
import tempfile
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


WEIGHT = "model.safetensors"
CONF = "config.json"
BODY = b"FORTY-GIGABYTES-OF-WEIGHTS-PRETEND" * 4
CONF_BODY = b'{"architectures":["Qwen3NextForCausalLM"]}'
SHA = hashlib.sha256(BODY).hexdigest()
CONF_SHA = hashlib.sha256(CONF_BODY).hexdigest()

API = {
    "sha": "1234567890abcdef1234567890abcdef12345678",
    "siblings": [
        {"rfilename": WEIGHT, "size": len(BODY),
         "lfs": {"sha256": SHA, "size": len(BODY)}},
        {"rfilename": CONF, "size": len(CONF_BODY),
         "lfs": {"sha256": CONF_SHA, "size": len(CONF_BODY)}},
    ],
}


class Log:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, msg):
        self.lines.append(str(msg))

    def text(self):
        return "\n".join(self.lines)


class CountingFetcher:
    """Records whether — and for what — the network was engaged."""

    def __init__(self, log=None):
        self.calls = 0
        self.allow_seen = None
        self.log = log or HF.default_log
        self.rung = "classic"
        self.ladder_verified = set()

    def snapshot(self, repo_id, local_dir, allow_patterns):
        self.calls += 1
        self.allow_seen = list(allow_patterns)
        out = []
        for name in allow_patterns:
            p = Path(local_dir) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(BODY if name == WEIGHT else CONF_BODY)
            out.append(name)
        return out

    def head_etag(self, repo_id, filename):
        return None


def mk_einput(slug, hf_home):
    der = SimpleNamespace(slug=slug, profile={"_hf_api": API})
    return EInput(
        slug=slug, terminal="proceed", is_override_accepted=False, der=der,
        runtime={}, selected_files=[], hf_home=Path(hf_home), c2a=None,
        hardware_sm=8.6, visible_gpu_count=1, per_gpu_vram_mib=[24576],
        selected_gpu_indices=[0], selected_gpu_vram_mib=[24576],
        topology_summary="(RTX 3090, 24576)", club3090_commit="deadbeef",
        diagnostics={},
    )


def place(pull: Path, name: str, body: bytes) -> None:
    """Hand-place a file exactly as a user would: bytes only, no metadata."""
    pull.mkdir(parents=True, exist_ok=True)
    (pull / name).write_bytes(body)


def env(**kw):
    """Scoped env for the VERIFY_IN_PLACE opt-in."""
    class _Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kw}
            for k, v in kw.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        def __exit__(self, *a):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return _Ctx()


# ---------------------------------------------------------------------------
# 1 + 2. correct metadata-less files + VERIFY_IN_PLACE=1 -> adopted, 0 bytes,
#        HF metadata stub written in the exact format at the exact path.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE="1",
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/Placed", td)
    pull = DL.pull_dir(Path(td), "org/Placed")
    place(pull, WEIGHT, BODY)
    place(pull, CONF, CONF_BODY)
    log = Log()
    f = CountingFetcher(log)
    r = DL.download_model(ei, fetcher=f)

    check(r.ok and f.calls == 0,
          f"a hand-placed, CORRECT model is adopted — the fetcher is never "
          f"called (ok={r.ok} snapshot_calls={f.calls})")
    check(sorted(r.adopted) == sorted([WEIGHT, CONF]),
          f"every file is reported as adopted ({r.adopted})")
    check(r.sha_verified is True,
          "sha_verified is True only because an actual sha256 was computed")
    check("verified in place" in log.text(),
          f"the adoption is ANNOUNCED as verified in place:\n{log.text()}")
    check("0 bytes re-downloaded" in log.text(),
          "the announcement states that nothing went over the wire")

    mp = HF.metadata_path(pull, WEIGHT)
    check(mp == pull / ".cache" / "huggingface" / "download"
          / f"{WEIGHT}.metadata",
          f"the stub lands where huggingface_hub looks for it ({mp})")
    lines = mp.read_text(encoding="utf-8").splitlines()
    check(len(lines) >= 3 and lines[0] == API["sha"] and lines[1] == SHA,
          f"the stub is huggingface_hub's 3-line format "
          f"(commit / etag=sha256 / timestamp); got {lines[:3]}")
    check(float(lines[2]) >= (pull / WEIGHT).stat().st_mtime - 1,
          "the stub's timestamp is not older than the file, or hf would "
          "discard it as outdated and re-download anyway")
    md = HF.read_metadata(pull, WEIGHT)
    check(md is not None and md["etag"] == SHA,
          f"the stub reads back through the same rules hf applies ({md})")

    # Re-running now takes the cheap path: the record exists, no re-hash.
    log2 = Log()
    f2 = CountingFetcher(log2)
    hashed = {"n": 0}
    _orig = HF.sha256_file
    HF.sha256_file = lambda p, **kw: (hashed.__setitem__("n", hashed["n"] + 1)
                                      or _orig(p, **kw))
    try:
        r2 = DL.download_model(ei, fetcher=f2)
    finally:
        HF.sha256_file = _orig
    check(r2.ok and f2.calls == 0 and hashed["n"] == 0,
          f"the SECOND run costs neither a download nor a re-hash "
          f"(calls={f2.calls} hashes={hashed['n']})")

# ---------------------------------------------------------------------------
# 3. truncated file -> rejected naming BOTH sizes, then re-downloaded.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE="1",
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/Truncated", td)
    pull = DL.pull_dir(Path(td), "org/Truncated")
    half = BODY[: len(BODY) // 2]
    place(pull, WEIGHT, half)
    place(pull, CONF, CONF_BODY)
    log = Log()
    f = CountingFetcher(log)
    r = DL.download_model(ei, fetcher=f)

    txt = log.text()
    check(str(len(half)) in txt and str(len(BODY)) in txt,
          f"the size-mismatch message names BOTH sizes "
          f"(local={len(half)} remote={len(BODY)}):\n{txt}")
    check("truncated or stale" in txt,
          "the message says what a size mismatch means, not just that it is one")
    check(f.calls == 1 and r.ok,
          f"the truncated file IS re-downloaded (calls={f.calls} ok={r.ok})")
    check(WEIGHT in (f.allow_seen or []),
          f"the re-fetch includes the truncated file ({f.allow_seen})")
    check(CONF not in (f.allow_seen or []),
          f"...and NOT the sibling that verified clean ({f.allow_seen}) — "
          f"one bad file must not re-pull the good ones")

# ---------------------------------------------------------------------------
# 4. right size, WRONG bytes -> the hash catches it (the silent-corruption
#    shape this rig has already been burned by).
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE="1",
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/Corrupt", td)
    pull = DL.pull_dir(Path(td), "org/Corrupt")
    corrupt = b"X" * len(BODY)          # correct size, wrong content
    place(pull, WEIGHT, corrupt)
    place(pull, CONF, CONF_BODY)
    log = Log()
    f = CountingFetcher(log)
    r = DL.download_model(ei, fetcher=f)
    txt = log.text()
    check("does NOT match" in txt and "corrupt" in txt.lower(),
          f"a right-sized, wrong-bytes file is caught BY HASH and named as "
          f"corrupt:\n{txt}")
    check(f.calls == 1 and WEIGHT in (f.allow_seen or []),
          "the corrupt file is re-downloaded")

# ---------------------------------------------------------------------------
# 5 + 6 + 7. no opt-in -> still re-downloads, but says why, says how to opt in,
#            and NEVER calls a size check "verified".
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE=None,
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/NoOptIn", td)
    pull = DL.pull_dir(Path(td), "org/NoOptIn")
    place(pull, WEIGHT, BODY)          # byte-perfect, but unhashed
    place(pull, CONF, CONF_BODY)
    log = Log()
    f = CountingFetcher(log)
    r = DL.download_model(ei, fetcher=f)
    txt = log.text()

    check(f.calls == 1,
          "without the opt-in the file is re-downloaded (a half-copy must "
          "never be silently trusted)")
    check("cannot verify it without hashing" in txt,
          f"the reason is stated in plain words:\n{txt}")
    check("VERIFY_IN_PLACE=1" in txt and "--verify-in-place" in txt,
          "the announcement names BOTH the env var and the pull.sh flag")
    check(re.search(r"re-downloading \(.*over the wire\)", txt) is not None,
          "the announcement states the cost of proceeding")
    check("verified" not in txt.lower(),
          f"the word 'verified' NEVER appears for a size-only observation — "
          f"a wrapper on this rig once printed 'DONE (hash-verified)' over "
          f"silent corruption, and that must be untellable here:\n{txt}")
    check(r.sha_verified is True,
          "the completed download is still hash-verified by the normal stage")

# ---------------------------------------------------------------------------
# 6b. the announce-WHY rule holds even with an EMPTY destination: no code path
#     starts a download without printing why.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE=None,
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/Fresh", td)
    log = Log()

    class OrderedFetcher(CountingFetcher):
        def snapshot(self, repo_id, local_dir, allow_patterns):
            # capture what had been announced BEFORE the first byte moved
            self.pre = list(self.log.lines) if hasattr(self.log, "lines") else []
            return super().snapshot(repo_id, local_dir, allow_patterns)

    f = OrderedFetcher(log)
    r = DL.download_model(ei, fetcher=f)
    pre = "\n".join(getattr(f, "pre", []))
    check(r.ok and f.calls == 1, "a fresh pull still downloads normally")
    check("[download] starting:" in pre,
          f"the 'starting' announcement is emitted BEFORE the fetch, not "
          f"after:\n{pre}")
    check("reason:" in pre and str(ei.slug) in pre,
          f"the announcement names the repo and the reason:\n{pre}")

# ---------------------------------------------------------------------------
# 8. partial presence -> adopt what verifies, fetch only the remainder.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td, env(VERIFY_IN_PLACE="1",
                                              CLUB3090_VERIFY_IN_PLACE=None):
    ei = mk_einput("org/Partial", td)
    pull = DL.pull_dir(Path(td), "org/Partial")
    place(pull, WEIGHT, BODY)          # the expensive one is already correct
    log = Log()
    f = CountingFetcher(log)
    r = DL.download_model(ei, fetcher=f)
    check(r.ok and f.allow_seen == [CONF],
          f"only the MISSING file is fetched; the verified multi-GB weight is "
          f"kept (allow_seen={f.allow_seen})")
    check(r.adopted == [WEIGHT],
          f"the kept file is reported as adopted ({r.adopted})")
    check((DL.pull_dir(Path(td), "org/Partial") / WEIGHT).read_bytes() == BODY,
          "the adopted file survives the atomic move into the final dir")
    check("keeping 1 verified file" in log.text(),
          f"the split is announced:\n{log.text()}")

# ---------------------------------------------------------------------------
# 9. plan_local_file phrasing contract, in isolation.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    (d / WEIGHT).write_bytes(BODY)
    meta_ok = HF.FileMeta(WEIGHT, size=len(BODY), sha256=SHA, source="api")
    meta_nohash = HF.FileMeta(WEIGHT, size=len(BODY), sha256=None)

    e = HF.plan_local_file(d, WEIGHT, meta_ok, do_hash=True)
    check(e.action == "adopt" and e.hashed and "verified in place" in e.message,
          f"hash match -> adopt, and only then is it called verified ({e})")

    e = HF.plan_local_file(d, WEIGHT, meta_nohash, do_hash=True)
    check(e.action == "download" and e.reason == "no-remote-hash"
          and "verified" not in e.message.replace("cannot be verified", ""),
          f"no published hash -> honest 'cannot be verified', re-download "
          f"({e.message})")

    e = HF.plan_local_file(d, "absent.bin", meta_ok, do_hash=True)
    check(e.action == "download" and e.reason == "absent",
          f"a missing file is simply 'absent' ({e.reason})")

    e = HF.plan_local_file(d, WEIGHT, meta_ok, do_hash=False)
    check(e.action == "download" and e.reason == "verify-not-enabled"
          and not e.hashed,
          f"size agreement alone NEVER adopts ({e.reason}) — silent "
          f"corruption looks exactly like this")

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll #812 verify-in-place / announce-why assertions passed.")
PY

echo "test-pull-verify-in-place.sh OK"
