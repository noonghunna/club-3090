"""Small shared helpers."""
from __future__ import annotations

import hashlib
import subprocess


class FFError(RuntimeError):
    pass


def sh(cmd: list[str], timeout: int = 600) -> str:
    """Run a subprocess, raise FFError with tail of stderr on failure. Returns stdout."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise FFError(f"{cmd[0]} failed ({r.returncode}): {(r.stderr or '')[-600:]}")
    return r.stdout


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
