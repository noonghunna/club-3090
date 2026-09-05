"""Download resilience ladder + verify-in-place (club-3090 #804 / #812).

Two jobs, one module, because they share the same plumbing (the HF repo tree
API, the non-redirect resolve HEAD, and the sha256 gate):

#804 — the three-rung ladder
----------------------------
``huggingface_hub`` **client-side refuses** the classic (non-Xet) HTTP path for
any file whose published size exceeds ``constants.MAX_HTTP_DOWNLOAD_SIZE``
(50 GB, hard-coded, no env override) — verbatim, from
``file_download.py::http_get`` on hf 1.25.1::

    The file is too large to be downloaded using the regular download method.
     Install `hf_xet` with `pip install hf_xet` for xet-powered downloads.

Retrying is futile; it is a policy wall, not flakiness. Meanwhile the server
serves the bytes fine: the resolve endpoint 302s to the CAS bridge, plain HTTP
200, resumable, and hands back the canonical sha256 in ``x-linked-etag``.

    rung 1  classic LFS   the current default — UNCHANGED, always first
                          (resume via .incomplete + the best reliability record)
    rung 2  Xet           OPT-IN ONLY (``ALLOW_XET=1`` / ``pull.sh --allow-xet``)
                          and only when ``hf_xet`` is ALREADY importable — this
                          module NEVER installs it. Xet has a stall/corruption
                          history on this stack, so a mandatory INDEPENDENT
                          sha256 gate runs afterwards.
    rung 3  raw resolve   single-stream ``curl -L -C - --retry-all-errors``
                          against the *resolve* endpoint, stall watchdog, then
                          the same mandatory sha256 gate.

Guardrails that are not negotiable here:

* **No parallel-chunk downloaders.** ``aria2c`` silently corrupts multi-GB
  files (right size, wrong bytes) — it must never appear in any rung. Guarded
  by ``test-pull-download-ladder.sh``.
* **Rung 2 never engages silently.** No opt-in -> announce why it was skipped
  and fall to rung 3.
* **Re-resolve per retry.** The 302 target is a *signed* CAS URL that expires
  (``Expires=`` ~3600 s). Resume must therefore go through
  ``huggingface.co/<repo>/resolve/<rev>/<file>`` on every attempt and let curl
  follow the redirect afresh — caching the signed URL turns a 6-hour resume
  into a 403. Guarded by a test on the curl argv.
* **A rung-2/rung-3 artifact that fails sha256 is DELETED**, not left on disk
  for someone to serve.
* Every rung announces which rung ran and what was verified, so logs are
  auditable.

#812 — verify-in-place + announce WHY
-------------------------------------
A file placed by hand has no ``.cache/huggingface/download/<f>.metadata`` beside
it, so the downloader cannot tell "complete" from "half-copied 40 GB" and
re-pulls. The instinct is right; the silence is the bug. So:

* **The announcement is unconditional.** Nothing here starts a transfer without
  first printing one honest line per present file saying what was found and
  what will happen to it.
* **Adoption requires a real hash.** ``VERIFY_IN_PLACE=1`` opts into hashing
  (it costs a full read of a multi-GB file); on a sha256 match the file is
  adopted and the HF metadata stub written, so ``hf download`` skips it
  forever after. Size agreement alone is NEVER called "verified" — a wrapper
  on this rig once printed "DONE (hash-verified)" over silent corruption at the
  correct size, and that is exactly the failure this module refuses to repeat.

Wording contract (asserted by the tests): the token ``verified`` appears in an
announcement **only** when a sha256 was actually computed and compared. A
size-only observation says "size matches" and nothing stronger.

Stdlib only — this runs under the bare ``python3`` that ``scripts/pull.sh``
uses, where ``huggingface_hub`` is NOT importable (it exists on this rig only as
a ``uv tool`` exposing the ``hf`` CLI).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_HF = "https://huggingface.co"
_NET_TIMEOUT = 60
_SHA_CHUNK = 8 * 1024 * 1024

# The verbatim client-side refusal (hf 1.25.1 file_download.py). Matched on a
# lowercased haystack; both markers must be present so an unrelated "too large"
# from the filesystem or a proxy cannot false-trigger an escalation.
_XET_REFUSAL_MARKERS = (
    "too large to be downloaded using the regular download method",
    "hf_xet",
)

# The wall's real trigger, for the honest announcement (NOT a Xet property per
# se: any file over this size is refused on the classic HTTP path, Xet-backed
# repo or not). huggingface_hub constants.MAX_HTTP_DOWNLOAD_SIZE.
MAX_HTTP_DOWNLOAD_SIZE = 50 * 1000 * 1000 * 1000


# ---------------------------------------------------------------------------
# announcement plumbing
# ---------------------------------------------------------------------------
def default_log(msg: str) -> None:
    """Announcements go to stderr so they interleave with `hf`'s own progress
    bars and never pollute a `--json` stdout contract."""
    print(msg, file=sys.stderr, flush=True)


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "unknown size"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.2f} {unit}"
        f /= 1024.0
    return f"{f:.2f} TB"  # pragma: no cover - unreachable


def short_sha(s: Optional[str]) -> str:
    return f"{s[:12]}…" if s else "?"


# ---------------------------------------------------------------------------
# Xet refusal signature
# ---------------------------------------------------------------------------
def is_xet_refusal(blob: str) -> bool:
    """True iff `blob` (a merged stdout+stderr of a failed `hf download`) is the
    client-side too-large refusal, NOT ordinary flakiness. Both markers are
    required — retrying this is futile, so a false positive would send a
    recoverable network error down the slow fallback and a false negative would
    burn 30 opaque retries on a hard policy wall."""
    low = (blob or "").lower()
    return all(m in low for m in _XET_REFUSAL_MARKERS)


def _is_tool_unavailable(rc: int, blob: str) -> bool:
    """True iff rung 1 failed because the `hf` CLI is not installed at all
    (rc=127, or the which() probe's message) — a fallback-ELIGIBLE cause:
    rung 3 needs only curl, so a missing binary must not kill the ladder
    (#1132).  Distinct from the 50 GB refusal, which is a policy wall."""
    return rc == 127 or "is on PATH" in blob


def hf_xet_importable(python: Optional[str] = None) -> bool:
    """Is `hf_xet` ALREADY importable in the environment the `hf` CLI runs under?

    Deliberately a probe, never an install: rung 2 is opt-in *and* BYO-package.
    The `hf` CLI on this rig is a `uv tool` / pipx venv, so we probe the
    interpreter that sits beside the resolved `hf` binary; failing that, our
    own. A probe failure reads as "not importable" (fall to rung 3), never as
    an exception."""
    cands: list[str] = []
    if python:
        cands.append(python)
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf_bin:
        sibling = Path(hf_bin).resolve().parent / "python3"
        if sibling.exists():
            cands.append(str(sibling))
    cands.append(sys.executable or "python3")
    for exe in cands:
        try:
            proc = subprocess.run(
                [exe, "-c", "import hf_xet"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0:
            return True
    return False


def allow_xet_enabled(env: Optional[dict] = None) -> bool:
    """Operator opt-in for rung 2. `pull.sh --allow-xet` exports
    CLUB3090_ALLOW_XET; the bare ALLOW_XET=1 from the issue is honoured too."""
    e = os.environ if env is None else env
    return (e.get("CLUB3090_ALLOW_XET") or e.get("ALLOW_XET") or "") == "1"


def verify_in_place_enabled(env: Optional[dict] = None) -> bool:
    """Opt-in for hashing an already-present file instead of re-downloading it
    (#812 item 2). The *announcement* (item 1) is unconditional and does not
    consult this."""
    e = os.environ if env is None else env
    return (e.get("CLUB3090_VERIFY_IN_PLACE") or e.get("VERIFY_IN_PLACE")
            or "") == "1"


# ---------------------------------------------------------------------------
# HF metadata: repo tree + the non-redirect resolve HEAD
# ---------------------------------------------------------------------------
@dataclass
class FileMeta:
    """What the hub publishes about one file. `sha256` is the canonical LFS
    hash (`x-linked-etag` on the non-redirected resolve hop == the model API's
    `siblings[].lfs.sha256`); None means genuinely unverifiable."""
    name: str
    size: Optional[int] = None
    sha256: Optional[str] = None
    commit: Optional[str] = None
    source: str = ""          # "api" | "resolve-head" | ""


def _token(env: Optional[dict] = None) -> Optional[str]:
    """HF token from the env, else the CLI's token file. nohup/cron/non-login
    shells don't source a profile, so an unexported token silently degrades to
    anonymous -> rate-limited -> looks exactly like a stall (the rig wrapper
    learned this the slow way)."""
    e = os.environ if env is None else env
    tok = e.get("HF_TOKEN") or e.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (
        Path(e.get("HF_HOME", "")) / "token" if e.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "token",
    ):
        if p is None:
            continue
        try:
            t = p.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if t:
            return t
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Do NOT follow the 302 into the CAS bridge.

    Following it is the trap: the post-redirect CAS response carries a
    CAS-*blob* ETag (the xet hash) and NO `x-linked-etag`, so a redirect-
    following HEAD reads as "no hash published" on every Xet-backed repo. The
    sha256 lives on the FIRST hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_head(repo_id: str, filename: str, *, revision: str = "main",
                 token: Optional[str] = None,
                 timeout: int = _NET_TIMEOUT,
                 opener: Optional[Any] = None) -> FileMeta:
    """Non-redirect-following HEAD on the resolve endpoint -> size + sha256.

    Reads `x-linked-etag` (canonical LFS sha256) / `x-linked-size` /
    `x-repo-commit` off the 302 itself. `opener` is injectable so tests never
    touch the network."""
    url = f"{_HF}/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    op = opener or urllib.request.build_opener(_NoRedirect())
    hdrs = None
    try:
        with op.open(req, timeout=timeout) as resp:
            hdrs = resp.headers          # 200 (a small non-LFS file)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308):
            hdrs = exc.headers           # the hop we actually want
        else:
            return FileMeta(name=filename)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return FileMeta(name=filename)
    if hdrs is None:                     # pragma: no cover - defensive
        return FileMeta(name=filename)

    def _h(*names: str) -> Optional[str]:
        for n in names:
            v = hdrs.get(n)
            if v:
                return v.strip().strip('"')
        return None

    sha = _h("x-linked-etag", "X-Linked-Etag")
    size = _h("x-linked-size", "X-Linked-Size") or _h("content-length")
    try:
        size_i = int(size) if size is not None else None
    except ValueError:
        size_i = None
    return FileMeta(
        name=filename, size=size_i,
        sha256=sha.lower() if sha else None,
        commit=_h("x-repo-commit", "X-Repo-Commit"),
        source="resolve-head" if sha else "",
    )


def repo_tree(repo_id: str, *, revision: str = "main",
              token: Optional[str] = None, timeout: int = _NET_TIMEOUT,
              urlopen: Optional[Callable] = None) -> list[dict]:
    """`/api/models/<repo>/tree/<rev>?recursive=1` — path + size + lfs.oid for
    every file. Needed to expand `--include` globs for rung 3 (curl fetches one
    named file at a time) and to price/verify an on-disk file. Returns [] on any
    failure; the caller degrades to "no metadata to verify against" rather than
    inventing one."""
    url = (f"{_HF}/api/models/{repo_id}/tree/{revision}"
           f"?recursive=1&expand=1")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    fn = urlopen or urllib.request.urlopen
    try:
        with fn(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def tree_meta(tree: Iterable[dict]) -> dict[str, FileMeta]:
    """`{path: FileMeta}` from a repo-tree payload. `lfs.oid`/`lfs.sha256` is
    the canonical sha256 for LFS files; a plain (non-LFS) file publishes only a
    size, and stays hash-unverifiable — which the announcement says out loud
    instead of implying otherwise."""
    out: dict[str, FileMeta] = {}
    for e in tree or []:
        if e.get("type") not in (None, "file"):
            continue
        name = e.get("path")
        if not name:
            continue
        lfs = e.get("lfs") if isinstance(e.get("lfs"), dict) else {}
        sha = lfs.get("sha256") or lfs.get("oid")
        size = lfs.get("size") if lfs.get("size") is not None else e.get("size")
        try:
            size_i = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_i = None
        out[name] = FileMeta(
            name=name, size=size_i,
            sha256=str(sha).lower() if sha else None,
            source="api" if sha else "",
        )
    return out


def api_meta(api: dict) -> dict[str, FileMeta]:
    """`{path: FileMeta}` from the `/api/models/<repo>?blobs=true` payload the
    deriver already fetched and stashed at `der.profile['_hf_api']` — free, and
    redirect-immune. Preferred over a network round-trip when present."""
    out: dict[str, FileMeta] = {}
    for s in (api or {}).get("siblings", []) or []:
        if not isinstance(s, dict):
            continue
        name = s.get("rfilename")
        if not name:
            continue
        lfs = s.get("lfs") if isinstance(s.get("lfs"), dict) else {}
        sha = lfs.get("sha256")
        size = lfs.get("size") if lfs.get("size") is not None else s.get("size")
        try:
            size_i = int(size) if size is not None else None
        except (TypeError, ValueError):
            size_i = None
        out[name] = FileMeta(
            name=name, size=size_i,
            sha256=str(sha).strip().strip('"').lower() if sha else None,
            source="api" if sha else "",
        )
    return out


def expand_includes(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """Resolve `--include` globs against a repo file list, preserving pattern
    order and de-duplicating. Exact filenames are valid globs, so the
    single-file case falls out for free."""
    all_names = list(names)
    out: list[str] = []
    seen: set[str] = set()
    for pat in patterns:
        for n in all_names:
            if n == pat or fnmatch.fnmatch(n, pat):
                if n not in seen:
                    seen.add(n)
                    out.append(n)
    return out


# ---------------------------------------------------------------------------
# hashing + the HF local-dir metadata stub
# ---------------------------------------------------------------------------
def sha256_file(path: Path, *, chunk: int = _SHA_CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def metadata_path(local_dir: Path, filename: str) -> Path:
    """The path `huggingface_hub._local_folder.get_local_download_paths`
    computes, so a stub we write is the same file `hf download` reads."""
    return (Path(local_dir) / ".cache" / "huggingface" / "download"
            / f"{os.path.join(*filename.split('/'))}.metadata")


def read_metadata(local_dir: Path, filename: str) -> Optional[dict]:
    """`{'commit': …, 'etag': …, 'timestamp': float}` or None.

    Mirrors `_local_folder.read_download_metadata`, INCLUDING its staleness
    rule: metadata older than the file's mtime is ignored (the file changed
    after the record was written, so the record no longer describes it)."""
    mp = metadata_path(local_dir, filename)
    try:
        lines = mp.read_text(encoding="utf-8").splitlines()
        commit, etag, ts = lines[0].strip(), lines[1].strip(), float(lines[2])
    except (OSError, IndexError, ValueError):
        return None
    try:
        st = (Path(local_dir) / filename).stat()
    except OSError:
        return None
    if st.st_mtime - 1 > ts:
        return None                      # outdated record — do not trust it
    return {"commit": commit, "etag": etag.strip('"'), "timestamp": ts}


def write_metadata(local_dir: Path, filename: str, *, commit: str,
                   etag: str) -> Path:
    """Write the 3-line stub (`commit_hash` / `etag` / `timestamp`) that makes
    `hf download` treat the file as already-fetched.

    ONLY ever called after a full sha256 match — this is the adoption receipt
    for a byte-verified file, never a shortcut around verification."""
    mp = metadata_path(local_dir, filename)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(f"{commit or ''}\n{etag}\n{time.time()}\n", encoding="utf-8")
    return mp


# ---------------------------------------------------------------------------
# #812 — verify-in-place decisions + the unconditional announcement
# ---------------------------------------------------------------------------
@dataclass
class PlanEntry:
    name: str
    action: str        # "adopt" | "download"
    reason: str        # machine token
    message: str       # the honest human line (announced verbatim)
    local_size: Optional[int] = None
    remote_size: Optional[int] = None
    hashed: bool = False


def plan_local_file(local_dir: Path, name: str, meta: Optional[FileMeta], *,
                    do_hash: bool,
                    hasher: Callable[[Path], str] = sha256_file) -> PlanEntry:
    """Decide — and phrase — what happens to one already-present file.

    Order is deliberate: size first (a cheap, certain reject that needs no
    read), then the downloader's own metadata record, then the opt-in hash.
    ``adopt`` is reachable ONLY through a computed sha256 match or a metadata
    record the downloader itself wrote; a bare size agreement is reported as
    "size matches" and still re-downloads, because size agreement is exactly
    what silent corruption looks like."""
    fp = Path(local_dir) / name
    try:
        lsize = fp.stat().st_size
    except OSError:
        return PlanEntry(name, "download", "absent",
                         f"{name}: not present — downloading")

    rsize = meta.size if meta else None
    rsha = meta.sha256 if meta else None
    h = human_bytes(lsize)

    # 1. size mismatch -> certain reject, names BOTH sizes, no read needed.
    if rsize is not None and lsize != rsize:
        return PlanEntry(
            name, "download", "size-mismatch",
            f"found {name} ({h}, {lsize} bytes) but the repo publishes "
            f"{human_bytes(rsize)} ({rsize} bytes) — truncated or stale, "
            f"re-downloading",
            local_size=lsize, remote_size=rsize)

    # 2. the downloader's own record (what `hf download` itself consults).
    md = read_metadata(local_dir, name)
    if md and md.get("etag") and rsha and md["etag"].lower() == rsha:
        return PlanEntry(
            name, "adopt", "metadata-match",
            f"found {name} ({h}) with an HF download record for sha256 "
            f"{short_sha(rsha)} — already fetched by the downloader, skipping",
            local_size=lsize, remote_size=rsize)

    # 3. nothing published to check against -> say so, and say the cost.
    if rsha is None:
        extra = (f" and the size matches ({h})"
                 if rsize is not None and lsize == rsize else "")
        return PlanEntry(
            name, "download", "no-remote-hash",
            f"found {name} ({h}) but the repo publishes no sha256 for it, so "
            f"it cannot be verified{extra} — re-downloading ({h} over the "
            f"wire)",
            local_size=lsize, remote_size=rsize)

    # 4. verifiable, but hashing is opt-in (a full read of a multi-GB file).
    if not do_hash:
        return PlanEntry(
            name, "download", "verify-not-enabled",
            f"found {name} ({h}) but cannot verify it without hashing "
            f"(no HF metadata beside it) — re-downloading ({h} over the wire). "
            f"Set VERIFY_IN_PLACE=1 (or pull.sh --verify-in-place) to sha256 "
            f"the local copy instead and adopt it on a match",
            local_size=lsize, remote_size=rsize)

    # 5. the real check.
    try:
        actual = hasher(fp)
    except OSError as exc:
        return PlanEntry(
            name, "download", "hash-unreadable",
            f"found {name} ({h}) but it could not be read for hashing "
            f"({exc}) — re-downloading",
            local_size=lsize, remote_size=rsize)
    if actual.lower() == rsha:
        return PlanEntry(
            name, "adopt", "hash-match",
            f"verified in place: {name} ({h}) sha256 {short_sha(rsha)} "
            f"matches the repo — adopting, 0 bytes re-downloaded",
            local_size=lsize, remote_size=rsize, hashed=True)
    return PlanEntry(
        name, "download", "hash-mismatch",
        f"found {name} ({h}) but its sha256 {short_sha(actual)} does NOT "
        f"match the repo's {short_sha(rsha)} — corrupt, re-downloading",
        local_size=lsize, remote_size=rsize, hashed=True)


def plan_and_announce(local_dir: Path, names: Iterable[str],
                      meta: dict[str, FileMeta], *, do_hash: bool,
                      log: Callable[[str], None] = default_log,
                      prefix: str = "[verify]",
                      hasher: Callable[[Path], str] = sha256_file,
                      commit: str = "") -> list[PlanEntry]:
    """Announce, unconditionally, what happens to every target file — then
    write the adoption receipt for each verified one.

    This is #812's rule "no code path starts a download without printing why"
    made mechanical: the caller cannot obtain the plan without the lines being
    emitted."""
    entries: list[PlanEntry] = []
    for name in names:
        e = plan_local_file(local_dir, name, meta.get(name), do_hash=do_hash,
                            hasher=hasher)
        entries.append(e)
        if e.reason == "absent":
            continue                     # summarised by the caller, not spammed
        log(f"{prefix} {e.message}")
        if e.action == "adopt" and e.reason == "hash-match":
            m = meta.get(name)
            try:
                write_metadata(local_dir, name, commit=commit,
                               etag=(m.sha256 if m and m.sha256 else ""))
            except OSError as exc:       # pragma: no cover - defensive
                log(f"{prefix} (could not write the adoption record for "
                    f"{name}: {exc} — it will be re-checked next time)")
    return entries


# ---------------------------------------------------------------------------
# #804 — the ladder
# ---------------------------------------------------------------------------
def preflight_transport(names: Iterable[str],
                        meta: dict[str, FileMeta]) -> tuple[str, str]:
    """Will the classic path be REFUSED for this pull? `(verdict, message)`.

    Answered from published sizes BEFORE a byte moves, so the operator sees
    "this repo needs Xet or the fallback" up front instead of watching a
    doomed rung-1 attempt (#804 acceptance bullet 3). `verdict` is "classic"
    (nothing over the wall) or "needs-fallback"."""
    over = [(n, meta[n].size) for n in names
            if meta.get(n) and (meta[n].size or 0) > MAX_HTTP_DOWNLOAD_SIZE]
    if not over:
        return "classic", ""
    listed = ", ".join(f"{n} ({human_bytes(s)})" for n, s in over[:3])
    more = f" (+{len(over) - 3} more)" if len(over) > 3 else ""
    # Say "50 GB" — upstream defines the constant decimally (50 * 1000**3), so
    # rendering it in binary units ("46.57 GB") would not match the number in
    # anyone's error message or in huggingface_hub's own source.
    return "needs-fallback", (
        f"{len(over)} file(s) exceed the 50 GB ceiling huggingface_hub "
        f"enforces on the classic HTTP path — "
        f"{listed}{more}. Rung 1 will be REFUSED client-side (a policy wall, "
        f"not flakiness, so retries cannot help). This will escalate: rung 3 "
        f"(single-stream resumable curl, ~5 MB/s, always works), or rung 2 "
        f"(Xet, faster) if you pass --allow-xet and hf_xet is installed.")


class LadderError(RuntimeError):
    """A rung failed in a way the ladder cannot escalate past. `.token` maps to
    the caller's structured failure vocabulary."""

    def __init__(self, token: str, detail: str = ""):
        super().__init__(detail or token)
        self.token = token
        self.detail = detail


@dataclass
class LadderResult:
    ok: bool
    rung: str = ""                 # "classic" | "xet" | "raw-resolve"
    files: list[str] = field(default_factory=list)
    verified: list[str] = field(default_factory=list)
    hash_source: str = ""          # "api" | "resolve-head" | ""
    failure: Optional[str] = None
    detail: str = ""
    notes: list[str] = field(default_factory=list)


def _curl_bin() -> Optional[str]:
    return shutil.which("curl")


def curl_argv(repo_id: str, filename: str, dest: Path, *,
              revision: str = "main", token: Optional[str] = None,
              connect_timeout: int = 30, curl: str = "curl") -> list[str]:
    """Rung 3's exact command line, as its own function so a test can assert
    the shape without running it.

    Two properties are load-bearing:
      * the URL is the **resolve endpoint**, never a signed CAS URL — curl
        re-follows the 302 on every invocation, so a resume hours later gets a
        fresh signature instead of a 403 on an `Expires`d one;
      * `-C -` + a single stream. No `-Z`, no range splitting, no aria2c —
        parallel-chunk fetchers have silently corrupted multi-GB files on this
        rig (right size, wrong bytes)."""
    url = f"{_HF}/{repo_id}/resolve/{revision}/{urllib.parse.quote(filename)}"
    argv = [
        curl, "-L",                       # follow the 302 afresh each attempt
        "-C", "-",                        # resume from whatever is on disk
        "--retry-all-errors", "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", str(connect_timeout),
        "--fail",                         # an HTTP error is a non-zero exit
        "-sS",                            # no \r progress spam; the watchdog
                                          # reports growth on its own line
        "-o", str(dest),
    ]
    if token:
        argv += ["-H", f"Authorization: Bearer {token}"]
    argv.append(url)
    return argv


def _run_with_stall_watchdog(argv: list[str], watch: Path, *,
                             stall_timeout: int,
                             log: Callable[[str], None],
                             runner: Optional[Callable] = None,
                             progress_every: int = 60) -> int:
    """Run `argv`, killing it if `watch` stops growing for `stall_timeout`.

    A silent hang (rate-limit, dead peer, a lock wait) produces no error on its
    own — only a watchdog turns it into a retryable failure. The same thread
    emits a periodic line-oriented progress note, because curl runs with `-sS`
    (a `\\r` progress meter would shred the c3 lane's line reader, and a
    multi-hour download that prints nothing is the opaque-download complaint
    this whole change exists to fix). `runner` is injectable so tests exercise
    the ladder without spawning curl."""
    if runner is not None:
        return int(runner(argv, watch))
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8")
    stop = threading.Event()

    def _watch() -> None:
        last, flat, since_report = -1, 0, 0
        while not stop.wait(15):
            try:
                cur = watch.stat().st_size
            except OSError:
                cur = 0
            if cur > last:
                last, flat = cur, 0
            else:
                flat += 15
            since_report += 15
            if since_report >= progress_every:
                since_report = 0
                log(f"[hf-fetch]   {watch.name}: {human_bytes(cur)} on disk")
            if stall_timeout > 0 and flat >= stall_timeout:
                log(f"[hf-fetch] no byte growth for {stall_timeout}s "
                    f"({human_bytes(cur)} on disk) — killing the stalled "
                    f"attempt; the retry resumes from here")
                try:
                    proc.kill()
                except OSError:          # pragma: no cover - defensive
                    pass
                return

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    try:
        out, _ = proc.communicate()
    finally:
        stop.set()
    if proc.returncode != 0 and out:
        log(f"[hf-fetch] curl: {out.strip()[-400:]}")
    return int(proc.returncode)


def raw_resolve_fetch(repo_id: str, filename: str, local_dir: Path, *,
                      revision: str = "main", token: Optional[str] = None,
                      stall_timeout: int = 300, max_retries: int = 6,
                      expected_size: Optional[int] = None,
                      log: Callable[[str], None] = default_log,
                      runner: Optional[Callable] = None) -> Path:
    """Rung 3 for ONE file: single-stream resumable curl against the resolve
    endpoint, re-resolved on every attempt, with a stall watchdog.

    Raises `LadderError` on exhaustion. Does NOT verify — the caller runs the
    mandatory sha256 gate over everything the rung produced, so verification
    can never be skipped per-file by accident."""
    curl = _curl_bin()
    if curl is None and runner is None:
        raise LadderError(
            "fallback-unavailable",
            "the raw-resolve fallback needs `curl` on PATH (install it, or "
            "opt into rung 2 with --allow-xet if hf_xet is installed)")
    dest = Path(local_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = curl_argv(repo_id, filename, dest, revision=revision, token=token,
                     curl=curl or "curl")
    for attempt in range(1, max_retries + 1):
        # `-C -` against an already-complete file gets a 416 and `--fail` turns
        # that into a non-zero exit — a resume that finished on the previous
        # attempt would otherwise look like a failure. Size agreement is NOT
        # trust: the mandatory sha256 gate still runs over this file.
        if expected_size is not None:
            try:
                if dest.stat().st_size == expected_size:
                    return dest
            except OSError:
                pass
        rc = _run_with_stall_watchdog(argv, dest, stall_timeout=stall_timeout,
                                      log=log, runner=runner)
        if rc == 0:
            return dest
        if attempt >= max_retries:
            raise LadderError(
                "fallback-failed",
                f"curl gave up on {filename} after {max_retries} attempts "
                f"(last rc={rc})")
        log(f"[hf-fetch] rung 3: {filename} attempt {attempt}/{max_retries} "
            f"failed (rc={rc}) — retrying in 8s (re-resolving the URL; the "
            f"signed CAS link expires, the resolve endpoint does not)")
        time.sleep(8 if runner is None else 0)
    raise LadderError("fallback-failed", filename)  # pragma: no cover


def verify_fetched(local_dir: Path, names: Iterable[str],
                   meta: dict[str, FileMeta], *,
                   log: Callable[[str], None] = default_log,
                   rung: str,
                   hasher: Callable[[Path], str] = sha256_file) -> list[str]:
    """The MANDATORY sha256 gate for rungs 2 and 3.

    Rung 1 delegates integrity to `hf download` (which hashes as it writes);
    rungs 2 and 3 do not get that for free, and both have a corruption story on
    this rig, so every file they produce is hashed here against the hub's
    published sha256. A mismatch DELETES the artifact and raises — a corrupt
    multi-GB weight left on disk is worse than no weight, because someone will
    serve it.

    A file the hub publishes no sha256 for is an unverifiable-artifact hard
    fail on these rungs; it is NOT quietly waved through."""
    verified: list[str] = []
    for name in names:
        fp = Path(local_dir) / name
        m = meta.get(name)
        if m is None or not m.sha256:
            try:
                fp.unlink()
            except OSError:
                pass
            raise LadderError(
                "no-etag",
                f"rung {rung} fetched {name} but the hub publishes no sha256 "
                f"for it — refusing to trust an unverifiable artifact "
                f"(deleted)")
        actual = hasher(fp)
        if actual.lower() != m.sha256:
            try:
                fp.unlink()
            except OSError:
                pass
            raise LadderError(
                "sha-mismatch",
                f"rung {rung} fetched {name} but sha256 {short_sha(actual)} "
                f"!= the hub's {short_sha(m.sha256)} — artifact DELETED")
        verified.append(name)
        log(f"[hf-fetch] sha256 OK: {name} ({short_sha(m.sha256)}, "
            f"source={m.source or 'hub'})")
    return verified


def escalate(repo_id: str, local_dir: Path, includes: list[str], *,
             classic_error: str,
             meta: Optional[dict[str, FileMeta]] = None,
             revision: str = "main",
             allow_xet: Optional[bool] = None,
             token: Optional[str] = None,
             log: Callable[[str], None] = default_log,
             xet_runner: Optional[Callable] = None,
             curl_runner: Optional[Callable] = None,
             xet_available: Optional[Callable[[], bool]] = None,
             tree_fn: Optional[Callable] = None,
             stall_timeout: int = 300,
             entry_note: str = "") -> LadderResult:
    """Rungs 2 and 3, entered after rung 1 failed in a fallback-ELIGIBLE way:
    the too-large refusal, or the `hf` CLI missing entirely (#1132 — rung 3
    needs only curl).

    Each rung announces why the previous one could not deliver, so a log shows
    the whole decision chain rather than a bare "downloading (again)"."""
    token = token if token is not None else _token()
    allow = allow_xet_enabled() if allow_xet is None else allow_xet
    notes: list[str] = []

    if entry_note:
        log(f"[hf-fetch] rung 1 (classic LFS) never stood a chance: {entry_note}")
    else:
        log("[hf-fetch] rung 1 (classic LFS) REFUSED this repo: huggingface_hub "
            "client-side blocks the plain HTTP path for files over 50 GB. This is "
            "a hard policy wall, not flakiness — retrying it cannot succeed.")
    if classic_error:
        log(f"[hf-fetch]   hub said: {classic_error.strip()[:300]}")

    # --- enumerate the concrete files (curl fetches one named file at a time)
    metas: dict[str, FileMeta] = dict(meta or {})
    if not metas:
        tf = tree_fn or (lambda: repo_tree(repo_id, revision=revision,
                                           token=token))
        metas = tree_meta(tf())
    names = expand_includes(metas.keys(), includes) if metas else []
    if not names:
        # No tree -> fall back to treating literal (non-glob) includes as names
        # and pricing them from a per-file resolve HEAD.
        names = [p for p in includes if not any(c in p for c in "*?[")]
        for n in names:
            metas.setdefault(n, resolve_head(repo_id, n, revision=revision,
                                             token=token))
    if not names:
        raise LadderError(
            "fallback-failed",
            "could not enumerate the repo tree, so the fallback has no "
            "concrete filenames to fetch (network down, or the repo is gated)")
    log(f"[hf-fetch] escalating for {len(names)} file(s): "
        f"{', '.join(names[:4])}{' …' if len(names) > 4 else ''}")

    # --- rung 2: Xet, opt-in only, never auto-installed --------------------
    avail_fn = xet_available or hf_xet_importable
    if not allow:
        notes.append("rung 2 (Xet) skipped: not opted in")
        log("[hf-fetch] rung 2 (Xet) SKIPPED — operator opt-in required "
            "(ALLOW_XET=1 or pull.sh --allow-xet). Xet has a stall/restart-"
            "from-zero history on this stack, so it never engages silently.")
    elif not avail_fn():
        notes.append("rung 2 (Xet) skipped: hf_xet not importable")
        log("[hf-fetch] rung 2 (Xet) SKIPPED — opted in, but `hf_xet` is not "
            "importable in the hf CLI's environment. This never auto-installs "
            "it; run `uv tool install --with hf_xet huggingface-hub` (or "
            "`pipx inject huggingface-hub hf_xet`) yourself to enable it.")
    else:
        log("[hf-fetch] rung 2 (Xet) — opted in and hf_xet is importable; "
            "retrying the SAME `hf download` with Xet enabled. An independent "
            "sha256 gate runs afterwards regardless of what it reports.")
        rc, blob = _hf_download(repo_id, local_dir, includes, xet=True,
                               runner=xet_runner)
        if rc == 0:
            verified = verify_fetched(local_dir, names, metas, log=log,
                                      rung="2 (Xet)")
            src = next((metas[n].source for n in verified if metas.get(n)), "")
            log(f"[hf-fetch] DONE via rung 2 (Xet) — {len(verified)} file(s), "
                f"sha256-verified against the hub's published hashes "
                f"({src or 'hub'}).")
            return LadderResult(ok=True, rung="xet", files=list(names),
                                verified=verified, hash_source=src,
                                notes=notes)
        notes.append(f"rung 2 (Xet) failed rc={rc}")
        log(f"[hf-fetch] rung 2 (Xet) FAILED (rc={rc}) — falling through to "
            f"the raw-resolve fallback. {blob.strip()[:200]}")

    # --- rung 3: raw resolve + curl ---------------------------------------
    log("[hf-fetch] rung 3 (raw resolve + curl) — single-stream, resumable, "
        "re-resolved per retry. Slower than either rung above (~5 MB/s "
        "observed) but universal. No parallel-chunk downloader is used: they "
        "have corrupted multi-GB files on this stack at the correct size.")
    for n in names:
        m = metas.get(n)
        log(f"[hf-fetch] rung 3: fetching {n} ({human_bytes(m.size if m else None)})")
        raw_resolve_fetch(repo_id, n, local_dir, revision=revision,
                          token=token, log=log, runner=curl_runner,
                          stall_timeout=stall_timeout,
                          expected_size=(m.size if m else None))
    verified = verify_fetched(local_dir, names, metas, log=log,
                              rung="3 (raw resolve)")
    src = next((metas[n].source for n in verified if metas.get(n)), "")
    log(f"[hf-fetch] DONE via rung 3 (raw resolve) — {len(verified)} file(s), "
        f"sha256-verified against the hub's published hashes "
        f"({src or 'hub'}).")
    return LadderResult(ok=True, rung="raw-resolve", files=list(names),
                        verified=verified, hash_source=src, notes=notes)


def _hf_download(repo_id: str, local_dir: Path, includes: list[str], *,
                 xet: bool, runner: Optional[Callable] = None) -> tuple[int, str]:
    """One `hf download` invocation. `xet=False` is the classic path
    (HF_HUB_DISABLE_XET=1); `xet=True` is rung 2. Returns (rc, stdout+stderr)."""
    if runner is not None:
        return runner(repo_id, str(local_dir), list(includes), xet)
    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        return 127, "neither `hf` nor `huggingface-cli` is on PATH"
    argv = [hf, "download", repo_id, "--local-dir", str(local_dir)]
    for pat in includes:
        argv += ["--include", pat]
    env = dict(os.environ)
    env["HF_HUB_DISABLE_XET"] = "0" if xet else "1"
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    tok = _token()
    if tok:
        env.setdefault("HF_TOKEN", tok)
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, encoding="utf-8")
    return proc.returncode, f"{proc.stdout or ''}\n{proc.stderr or ''}"


def fetch(repo_id: str, local_dir: Path, includes: list[str], *,
          revision: str = "main", allow_xet: Optional[bool] = None,
          do_verify_in_place: Optional[bool] = None,
          token: Optional[str] = None,
          log: Callable[[str], None] = default_log) -> LadderResult:
    """The whole thing, for callers that are NOT `downloader.download_model`:
    preflight -> announce/verify-in-place -> rung 1 -> escalate on the refusal.

    `download_model` drives the rungs itself (it owns atomic staging and the
    CONTRACT-3 failure vocabulary); this is the standalone entry the module CLI
    exposes so a GGUF pull or a shell script gets the same ladder + the same
    honest announcements without reimplementing either."""
    token = token if token is not None else _token()
    allow = allow_xet_enabled() if allow_xet is None else allow_xet
    do_hash = (verify_in_place_enabled() if do_verify_in_place is None
               else do_verify_in_place)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    metas = tree_meta(repo_tree(repo_id, revision=revision, token=token))
    names = expand_includes(metas.keys(), includes) if metas else list(includes)
    if metas and not names:
        return LadderResult(ok=False, failure="fallback-failed",
                            detail=f"no file in {repo_id} matches {includes}")

    verdict, note = preflight_transport(names, metas)
    if verdict == "needs-fallback":
        log(f"[hf-fetch] preflight: {note}")

    plan = plan_and_announce(local_dir, names, metas, do_hash=do_hash, log=log,
                             prefix="[hf-fetch]")
    todo = [e.name for e in plan if e.action == "download"]
    adopted = [e.name for e in plan if e.action == "adopt"]
    if not todo:
        log(f"[hf-fetch] SKIPPED — all {len(names)} file(s) already present "
            f"and accounted for. 0 bytes re-downloaded.")
        return LadderResult(ok=True, rung="", files=list(names),
                            verified=[e.name for e in plan if e.hashed])
    log(f"[hf-fetch] starting: {repo_id} -> {local_dir} "
        f"({len(todo)} file(s) to fetch"
        + (f", {len(adopted)} kept" if adopted else "") + ")")

    rc, blob = _hf_download(repo_id, local_dir, todo, xet=False)
    if rc == 0:
        log(f"[hf-fetch] DONE via rung 1 (classic LFS) — {len(todo)} file(s); "
            f"`hf download` hashes as it writes, so no separate gate is run "
            f"on this rung.")
        return LadderResult(ok=True, rung="classic", files=list(names))
    # #1132: the too-large refusal escalates, and so does a missing `hf`
    # binary -- rung 3 needs only curl, so a bare exit would waste the one
    # rung that could have served the pull.  Anything else (network, auth)
    # stays a hard failure.
    tool_missing = _is_tool_unavailable(rc, blob)
    if not (is_xet_refusal(blob) or tool_missing):
        return LadderResult(ok=False, failure="fallback-failed",
                            detail=f"rung 1 failed (rc={rc}): "
                                   f"{blob.strip()[-400:]}")
    try:
        return escalate(repo_id, local_dir, todo, classic_error=blob,
                        meta=metas, revision=revision, allow_xet=allow,
                        token=token, log=log,
                        entry_note=("`hf`/`huggingface-cli` is not on PATH"
                                    if tool_missing else ""))
    except LadderError as exc:
        log(f"[hf-fetch] FAILED: {exc.detail or exc.token}")
        return LadderResult(ok=False, failure=exc.token, detail=exc.detail)


def _main(argv: list[str]) -> int:
    """`python3 scripts/lib/profiles/hf_fetch.py <repo> --local-dir DIR ...`

    A thin CLI over `fetch()` / `preflight_transport()`, so surfaces that
    shell out to `hf download` directly can adopt the ladder + the honest
    announcements without importing anything."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="hf_fetch.py",
        description="HF download with the #804 resilience ladder and the "
                    "#812 verify-in-place / announce-why preflight.")
    ap.add_argument("repo")
    ap.add_argument("--local-dir", required=True)
    ap.add_argument("--include", action="append", default=[],
                    help="glob or exact filename; repeatable (default: all)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--allow-xet", action="store_true",
                    help="permit rung 2 (needs hf_xet already installed; "
                         "this never installs it)")
    ap.add_argument("--verify-in-place", action="store_true",
                    help="sha256 an already-present file and adopt it on a "
                         "match instead of re-downloading")
    ap.add_argument("--preflight", action="store_true",
                    help="report whether the classic path can serve this "
                         "pull, then exit")
    args = ap.parse_args(argv)

    includes = args.include or ["*"]
    if args.preflight:
        metas = tree_meta(repo_tree(args.repo, revision=args.revision,
                                    token=_token()))
        names = expand_includes(metas.keys(), includes)
        verdict, note = preflight_transport(names, metas)
        print(f"transport: {verdict}")
        if note:
            print(note)
        return 0

    res = fetch(args.repo, Path(args.local_dir), includes,
                revision=args.revision, allow_xet=args.allow_xet or None,
                do_verify_in_place=args.verify_in_place or None)
    if not res.ok:
        # #1132 -- a bare `exit 1` discarded the failure token + detail,
        # contradicting the #804/#812 auditability contract exactly when the
        # pull failed.  Announce, then exit.
        default_log(f"[hf-fetch] FAILED: {res.failure}"
                    + (f" — {res.detail.strip()[-400:]}" if res.detail else ""))
    return 0 if res.ok else 1


if __name__ == "__main__":       # pragma: no cover - CLI wrapper
    sys.exit(_main(sys.argv[1:]))
