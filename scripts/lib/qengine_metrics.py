#!/usr/bin/env python3
"""qengine_metrics — publish QEngine's metrics port on 127.0.0.1, only when asked.

WHY
---
QEngine serves a live dashboard, ``/metrics`` (Prometheus) and ``/api/stats`` (its
``qetop``) on a second port, 9393 by default, independent of the API port. The
image's entrypoint passes ``--metrics-port "${QE_METRICS_PORT:-9393}"`` and binds it
to 0.0.0.0 inside the container; 0 turns it off. The composes don't publish it,
because the dashboard has a write endpoint (POST /api/reset) and no auth.

Setting ``QE_METRICS_PORT=<port>`` opts in: the compose forwards it to the engine
(bare passthrough, so the engine listens on that port), and switch.sh adds this
override at ``up``, which publishes it on the host's loopback only:
``127.0.0.1:<port>:<port>``. Compose can't publish a port conditionally, hence the
override. Unset or 0 writes none: the port stays unpublished (0 also turns the
engine's metrics server off).

    python3 scripts/lib/qengine_metrics.py override --slug SLUG --compose PATH --port VALUE

prints the override file's path, or nothing (no port asked for, or a compose with no
services). Exit 2, with a ``Fix:`` line, when VALUE is not a port number. The file
lives in ``<data dir>/compose-overrides/`` beside the slug labels (the same path for
every checkout) and holds the resolved port, not a ``${…}`` reference, because
compose records its path and switch.sh's teardown reads it again, when the
setting may be gone. Standard library only: this is on the launcher path.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUBDIR = "compose-overrides"
_PORT_RE = re.compile(r"^(0|[1-9][0-9]{0,4})$")


def parse_port(value: str) -> int | None:
    """The port QE_METRICS_PORT asks for: None when unset, 0 for QEngine's "off".
    Raises ValueError for anything that isn't 0-65535 written plainly."""
    value = (value or "").strip()
    if not value:
        return None
    if not _PORT_RE.match(value) or int(value) > 65535:
        raise ValueError(value)
    return int(value)


def override_path(slug: str, environ=None) -> Path:
    from scripts.lib import club_config

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-") or "slug"
    return club_config.data_dir(environ) / SUBDIR / f"qengine-metrics-{safe}.yml"


def write_override(slug: str, compose: Path, port: int | None, environ=None) -> Path | None:
    """The override publishing ``port`` on 127.0.0.1 for every service of ``compose``,
    or None when there is nothing to publish."""
    from scripts.lib.slug_label import service_names

    if not port:
        return None
    names = service_names(Path(compose).read_text(encoding="utf-8", errors="replace"))
    if not names:
        return None
    body = ["# Written by scripts/lib/qengine_metrics.py: QEngine's metrics and dashboard port,",
            "# published on 127.0.0.1 only, because QE_METRICS_PORT is set.",
            "services:"]
    for name in names:
        body += [f"  {name}:", "    ports:", f'      - "127.0.0.1:{port}:{port}"']
    text = "\n".join(body) + "\n"
    path = override_path(slug, environ)
    try:
        if path.read_text(encoding="utf-8") == text:
            return path
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


# Reading the port from the host: it isn't published unless asked for, so tools read it inside the
# container. The image runs Python, so no curl is needed; QE_METRICS_PORT moves it, as the
# entrypoint does. An unauthenticated port of the engine's own, not the API, so this is not an
# engine request and carries no key (engine_http is for those).
_READ_IN_CONTAINER = ("import os, sys, urllib.request; p = os.environ.get('QE_METRICS_PORT') or '9393'; "
                      "sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:' + p + '{path}', "
                      "timeout=2).read().decode('utf-8', 'replace'))")


def read_in_container(container: str, path: str = "/metrics", timeout: float = 5) -> str:
    """QEngine's metrics-port page ``path`` read inside ``container``, or "" when it can't be read
    (another engine, the metrics server off, no python3 in the image)."""
    if not container:
        return ""
    try:
        r = subprocess.run(["docker", "exec", container, "python3", "-c", _READ_IN_CONTAINER.format(path=path)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except Exception:
        return ""
    return r.stdout if r.returncode == 0 else ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qengine_metrics.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("override", help="write the port override; print its path (or nothing)")
    o.add_argument("--slug", required=True)
    o.add_argument("--compose", required=True, help="the compose file being launched")
    o.add_argument("--port", default="", help="QE_METRICS_PORT's value (empty = not set)")
    c = sub.add_parser("check", help="exit 2 unless --port is empty or a port number")
    c.add_argument("--port", default="")
    a = ap.parse_args(argv)
    try:
        port = parse_port(a.port)
    except ValueError:
        print(f"ERROR: QE_METRICS_PORT='{a.port}' is not a port number.", file=sys.stderr)
        print("Fix: QE_METRICS_PORT=<1-65535> publishes QEngine's metrics and dashboard on "
              "127.0.0.1:<port>; leave it unset to keep them unpublished, or 0 to turn "
              "QEngine's metrics server off.", file=sys.stderr)
        return 2
    if a.cmd == "check":
        return 0
    path = write_override(a.slug, Path(a.compose), port)
    if path:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
