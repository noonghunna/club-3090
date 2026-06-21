#!/usr/bin/env bash
# CONTRACT — LMCache entrypoint knobs must be forwarded through compose env.
#
# The LMCache compose starts `lmcache server` from an in-container `bash -c`
# entrypoint and intentionally uses escaped `$${LMCACHE_*:-...}` tokens so bash,
# not Docker Compose, chooses defaults at runtime. Docker Compose will not pass
# host-shell values into that bash process unless each knob is declared in the
# service `environment:` block. Missing declarations silently force defaults
# (e.g. L1=30, L2 off) even when the user launches with LMCACHE_L1_GB/LMCACHE_L2.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import re
import sys
from pathlib import Path

import yaml

compose_files = sorted(Path("models").glob("*/*/compose/*/*/*.yml"))
runtime_lmcache_var = re.compile(r"\$\$\{?(LMCACHE_[A-Za-z0-9_]+)")


def env_keys(service):
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    keys = set()
    if isinstance(env, list):
        for item in env:
            if not isinstance(item, str):
                continue
            keys.add(item.split("=", 1)[0])
    return keys


def entrypoint_script(service):
    ep = service.get("entrypoint")
    if not isinstance(ep, list) or len(ep) < 3:
        return ""
    if ep[0] in {"bash", "sh", "/bin/bash", "/bin/sh"} and ep[1] == "-c":
        return ep[2] or ""
    return ""


failures = []
checked = 0
for compose in compose_files:
    data = yaml.safe_load(compose.read_text()) or {}
    for service_name, service in sorted((data.get("services") or {}).items()):
        service = service or {}
        script = entrypoint_script(service)
        if "lmcache server" not in script:
            continue
        checked += 1
        used = set(runtime_lmcache_var.findall(script))
        # LMCACHE_DISABLE_BANNER is consumed by the image, not by the entrypoint
        # command construction, but it is still declared in environment.
        used.discard("LMCACHE_DISABLE_BANNER")
        declared = env_keys(service)
        missing = sorted(used - declared)
        if missing:
            failures.append(
                f"{compose} [{service_name}]: entrypoint uses {', '.join(missing)} "
                "but compose does not forward them in environment:"
            )

print(f"[lmcache-env] scanned {len(compose_files)} composes; {checked} LMCache entrypoint(s)")
if failures:
    print("FAIL:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    sys.exit(1)
print("[lmcache-env] PASS: LMCache runtime knobs are declared in compose environment")
PY
