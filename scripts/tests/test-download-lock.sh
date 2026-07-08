#!/usr/bin/env bash
# Guard: the per-repo download lock in downloader.download_model (club-3090
# #617 — repeated ① Bring [D] presses spawned 5 concurrent hf downloads racing
# into the same .incomplete). Asserts:
#   1. a 2nd download while a LIVE holder exists is REFUSED (failure=in-progress,
#      detail carries the holder pid) — never races/rmtrees the first's staging;
#   2. a STALE lock (dead holder — a crashed/SIGKILL'd download) is RECLAIMED
#      and the fetch proceeds, and the lock is RELEASED afterwards;
#   3. read_active_download reflects live-vs-stale.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import sys, os, tempfile, subprocess, time
sys.path.insert(0, "scripts")
from pathlib import Path
from lib.profiles import downloader as DL

tmp = Path(tempfile.mkdtemp())
slug = "org/Test-FP8"

class EI:
    slug = "org/Test-FP8"
    hf_home = str(tmp)

# 1) LIVE holder → refuse ---------------------------------------------------
holder = subprocess.Popen(["sleep", "60"])
try:
    lock = DL.download_lock_dir(tmp, slug)
    lock.mkdir(parents=True)
    (lock / "pid").write_text(
        f"{holder.pid}\n2026-01-01T00:00:00+00:00\n", encoding="utf-8"
    )
    r = DL.download_model(EI())
    assert r.ok is False and r.failure == "in-progress", \
        f"expected in-progress refusal, got ok={r.ok} failure={r.failure}"
    assert f"pid={holder.pid}" in (r.detail or ""), \
        f"detail must carry holder pid: {r.detail!r}"
    act = DL.read_active_download(tmp, slug)
    assert act and act["pid"] == holder.pid, f"read_active live: {act}"
    print("PASS 1: live holder refused (failure=in-progress, pid in detail)")
finally:
    holder.terminate()
    holder.wait()

# holder dead now → the lock is stale
assert DL.read_active_download(tmp, slug) is None, \
    "dead-holder lock must read stale (None)"
print("PASS 2: dead-holder lock reads stale (None) → reclaimable")

# 2) STALE lock → reclaim + proceed + release -------------------------------
# stub the fetch so no network/deriver is needed; prove we got PAST the lock.
sentinel = DL.DownloadResult(ok=True, local_dir=str(tmp / "x"), files=["f"])
DL._download_model_impl = lambda einput, *, fetcher=None: sentinel  # type: ignore
lock = DL.download_lock_dir(tmp, slug)
if not lock.exists():
    lock.mkdir(parents=True)
(lock / "pid").write_text("2147483646\nSTALE\n", encoding="utf-8")  # dead pid
r = DL.download_model(EI())
assert r is sentinel, "stale lock must be reclaimed and the fetch proceed"
assert not lock.exists(), "lock must be RELEASED (rmdir) after the fetch"
print("PASS 3: stale lock reclaimed → fetch proceeded → lock released")

# 4) MID-ACQUIRE window → refuse (pidless + FRESH lock = a holder just mkdir'd
#    and hasn't written its pid yet; reclaiming it would let BOTH callers
#    proceed — the very dup race the lock closes) -----------------------------
def _must_not_fetch(einput, *, fetcher=None):
    raise AssertionError("must NOT reach the fetch for a fresh pidless lock")
DL._download_model_impl = _must_not_fetch  # type: ignore
lock = DL.download_lock_dir(tmp, slug)
lock.mkdir(parents=True, exist_ok=True)
(lock / "pid").unlink(missing_ok=True)         # no pid yet, mtime = now (fresh)
r = DL.download_model(EI())
assert r.ok is False and r.failure == "in-progress", \
    f"fresh pidless lock (mid-acquire) must be refused, got ok={r.ok} failure={r.failure}"
assert lock.exists(), "must NOT reclaim a fresh pidless lock (holder mid-acquire)"
print("PASS 4: fresh pidless lock (mid-acquire) refused — not stolen")

# 5) OLD pidless lock → reclaim (a genuinely broken/abandoned lock, past grace)
sentinel2 = DL.DownloadResult(ok=True, local_dir=str(tmp / "y"), files=["g"])
DL._download_model_impl = lambda einput, *, fetcher=None: sentinel2  # type: ignore
lock = DL.download_lock_dir(tmp, slug)
lock.mkdir(parents=True, exist_ok=True)
(lock / "pid").unlink(missing_ok=True)
old = time.time() - (DL._LOCK_ACQUIRE_GRACE_S + 5)
os.utime(lock, (old, old))                     # age past the grace window
r = DL.download_model(EI())
assert r is sentinel2, "old pidless lock must be reclaimed and the fetch proceed"
assert not lock.exists(), "lock released after reclaim + fetch"
print("PASS 5: old pidless lock reclaimed → fetch proceeded → lock released")

print("OK test-download-lock")
PY
