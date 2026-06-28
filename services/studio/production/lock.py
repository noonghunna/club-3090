"""Process-local single-flight lock for v0a (CLI/admin).

v0a is CLI/admin-only, so a host file-lock is enough to guarantee one production
job at a time (two `run`s can't render into the same ComfyUI queue concurrently).
The durable queue + OWUI-exposed job lock are v0b/v1.
"""
from __future__ import annotations

import errno
import fcntl
import os

from . import config


class ProductionLockHeld(RuntimeError):
    pass


class ProductionLock:
    """Non-blocking flock on `<productions>/.run.lock`.

    Usage:
        with ProductionLock():
            ...   # raises ProductionLockHeld immediately if another run holds it
    """

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(config.PRODUCTIONS_DIR, ".run.lock")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fd = None

    def acquire(self) -> "ProductionLock":
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(self._fd)
            self._fd = None
            if e.errno in (errno.EAGAIN, errno.EACCES):
                raise ProductionLockHeld(
                    f"another production run holds {self.path} — v0a is single-flight"
                ) from e
            raise
        os.ftruncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode())
        return self

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> "ProductionLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
