#!/usr/bin/env bash
# CONTRACT — conditional compose entrypoints must $$-escape runtime bash tokens.
#
# Docker Compose v5.1+ interpolates entrypoint/command STRING VALUES at parse
# time, against the host env. A `bash -c` entrypoint that relies on a variable
# set at RUNTIME (e.g. _NVLINK_ENABLED, exported by scripts/detect_nvlink.sh
# inside the container) must therefore escape every `$` that bash — not Compose
# — should expand, by writing `$$`. An un-escaped `${_NVLINK_ENABLED:-0}` is
# resolved to its `:-` default BEFORE the container runs, silently baking the
# NVLink branch dead: `if [ "0" = "1" ]` is always false -> always the PCIe
# branch, regardless of hardware. This was live on 9 dual/multi composes until
# club-3090 #436 fixed it repo-wide (lmcache via #429/#433). See
# LEARNINGS.md "2026-06-18 — Docker Compose v5.1+ interpolates entrypoint strings".
#
# This guard fails if any shipped compose carries an un-escaped runtime token in
# a `bash -c` entrypoint:
#   ${_NVLINK_ENABLED...}      -> must be $${_NVLINK_ENABLED...}
#   ${VAR[@]} (bash array)     -> must be $${VAR[@]}   (Compose errors otherwise)
#   $@ / $0  (in a detect_nvlink entrypoint)  -> must be $$@ / $$0
#   ${VLLM_ENFORCE_EAGER:+...} -> must be $${VLLM_ENFORCE_EAGER:+...}
#
# Verify the live resolution of any single compose with:
#   docker compose -f <file> config   # a baked `[ "0" = "1" ]` == interpolated too early
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import re
import sys
from pathlib import Path

import yaml

compose_files = sorted(Path("models").glob("*/*/compose/*/*/*.yml"))

# An "un-escaped" token is a single `$` NOT preceded by another `$`.
UNESC_NVLINK = re.compile(r"(?<!\$)\$\{_NVLINK_ENABLED")
UNESC_ENFORCE = re.compile(r"(?<!\$)\$\{VLLM_ENFORCE_EAGER:\+")
UNESC_ARRAY = re.compile(r"(?<!\$)\$\{[A-Za-z_][A-Za-z0-9_]*\[@\]\}")
UNESC_ATPARAM = re.compile(r"(?<!\$)\$[@0]")

_SHELLS = {"bash", "sh", "/bin/bash", "/bin/sh"}


def entrypoint_script(service):
    """Return the inline script of a `bash -c <script>` entrypoint, else None."""
    ep = service.get("entrypoint")
    if not isinstance(ep, list) or len(ep) < 3:
        return None
    if ep[0] in _SHELLS and "-c" in ep[1:2]:
        return ep[2]
    return None


failures = []
checked = 0
with_detect = 0

for compose in compose_files:
    try:
        data = yaml.safe_load(compose.read_text()) or {}
    except Exception as e:  # noqa: BLE001
        failures.append(f"{compose}: YAML parse error: {e}")
        continue
    rel = compose  # already repo-relative (glob is rooted at cwd == repo root)
    for sname, service in sorted((data.get("services") or {}).items()):
        script = entrypoint_script(service or {})
        if script is None:
            continue
        checked += 1
        # _NVLINK_ENABLED is ALWAYS a runtime var — never legitimately un-escaped.
        if UNESC_NVLINK.search(script):
            failures.append(
                f"{rel} [{sname}]: un-escaped ${{_NVLINK_ENABLED}} in entrypoint — "
                f"Compose v5.1+ bakes it to its default at parse time (NVLink branch dies). "
                f"Use $${{_NVLINK_ENABLED:-0}}."
            )
        # A bash array expansion un-escaped errors out on Compose v5.1+ entirely.
        if UNESC_ARRAY.search(script):
            failures.append(
                f"{rel} [{sname}]: un-escaped bash array ${{VAR[@]}} in entrypoint — "
                f"Compose v5.1+ rejects it ('invalid interpolation format'). Escape as $${{VAR[@]}}."
            )
        if "detect_nvlink.sh" in script:
            with_detect += 1
            if UNESC_ATPARAM.search(script):
                failures.append(
                    f"{rel} [{sname}]: un-escaped $@/$0 in a detect_nvlink entrypoint — "
                    f"use $$@/$$0 so bash (not Compose) expands the serve args."
                )
            if UNESC_ENFORCE.search(script):
                failures.append(
                    f"{rel} [{sname}]: un-escaped ${{VLLM_ENFORCE_EAGER:+...}} in entrypoint — "
                    f"escape as $${{VLLM_ENFORCE_EAGER:+...}}."
                )

print(
    f"[nvlink-escape] scanned {len(compose_files)} composes; "
    f"{checked} have a `bash -c` entrypoint; {with_detect} source detect_nvlink.sh"
)
if failures:
    print("FAIL:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("[nvlink-escape] PASS: every conditional entrypoint $$-escapes its runtime bash tokens")
PY
