"""c3 application-session logging.

The app log is opt-in and deliberately separate from the always-on download
logs.  It never serializes settings or process environments: callers log event
names and exception objects only.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional


def config_dir() -> Path:
    """Return ``C3_CONFIG_DIR`` or the normal per-user c3 config directory."""
    base = os.environ.get("C3_CONFIG_DIR")
    return Path(base) if base else Path.home() / ".config" / "club-3090"


def env_log_override(environ: Mapping[str, str]) -> Optional[bool]:
    """Parse the strict per-launch ``C3_LOG=1|0`` override."""
    value = str(environ.get("C3_LOG") or "").strip()
    if value == "1":
        return True
    if value == "0":
        return False
    return None


class SessionLog:
    """One opt-in Python log file for the lifetime of a c3 app instance."""

    KEEP = 10

    def __init__(self) -> None:
        self.path: Optional[Path] = None
        self._handler: Optional[logging.FileHandler] = None
        try:
            log_dir = config_dir() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            self.path = log_dir / f"c3-session-{ts}.log"
            handler = logging.FileHandler(self.path, encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)sZ %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
            formatter.converter = time.gmtime
            handler.setFormatter(formatter)
            handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(handler)
            logging.getLogger("club3090_cockpit").setLevel(logging.INFO)
            self._handler = handler
            self._prune(log_dir)
            logging.getLogger("club3090_cockpit").info("c3 session log started")
        except OSError:
            self.path = None
            self._handler = None

    def _prune(self, log_dir: Path) -> None:
        try:
            logs = sorted(
                log_dir.glob("c3-session-*.log"),
                key=lambda path: path.stat().st_mtime,
            )
            for old in logs[: max(0, len(logs) - self.KEEP)]:
                old.unlink(missing_ok=True)
        except OSError:
            pass

    def close(self) -> None:
        handler = self._handler
        if handler is None:
            return
        try:
            logging.getLogger("club3090_cockpit").info("c3 session log stopped")
            handler.flush()
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
            self._handler = None
