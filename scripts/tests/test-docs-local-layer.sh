#!/usr/bin/env bash
# Drift-guard for the LOCAL model layer's DISCOVERABILITY (#1142).
#
# The layer itself has worked since 2026-08-22 — gitignored profiles-local/,
# promote.py defaulting to --layer local, export_pr.py completing the community
# loop, get_registry() merging both layers. What failed was ROUTING: AGENTS.md
# and the user-facing BYOM docs never mentioned any of it, so agents and users
# followed the curated-catalog steps and wrote scripts/lib/profiles/registry.yaml
# — a TRACKED file — which then conflicts on the next `git pull`. That is the
# exact failure the local layer exists to prevent.
#
# This guard asserts the docs keep pointing at the layer and at the local -> PR
# path, so the routing cannot silently rot back.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail=0
need() {  # need <file> <needle> <why>
  if ! command grep -qF -- "$2" "$1"; then
    printf '  FAIL %-34s missing %-24s (%s)\n' "$1" "$2" "$3"
    fail=1
  fi
}

# The agent-facing contract must pose the choice AND name both tools.
need AGENTS.md                            "profiles-local"          "agents must know the local layer exists"
need AGENTS.md                            "export_pr.py"            "agents must know local -> PR is one command"
need AGENTS.md                            "C3_ALLOW_CORE_PROMOTE"   "the core gate must be stated, not discovered"

# The user-facing BYOM guide is where a community user actually lands.
need docs/BRING_YOUR_OWN.md               "profiles-local"          "BYOM guide must cover the local layer"
need docs/BRING_YOUR_OWN.md               "export_pr.py"            "its share/contribute section must name the tool"
need docs/BRING_YOUR_OWN.md               "promote.py"              "the local write path must be shown"

# The catalog doc must offer the local layer as a path, not only the core one.
need docs/ADDING_MODELS.md                "profiles-local"          "must list the local layer among the paths"
need docs/ADDING_MODELS.md                "BRING_YOUR_OWN.md"       "must hand off to the BYOM guide"

# The layer's own README is the last stop for someone already inside it.
need scripts/lib/profiles-local/README.md "export_pr.py"            "local layer must document its exit to a PR"

# README must give a BYOM entry point at all.
need README.md                            "BRING_YOUR_OWN.md"       "no BYOM pointer from the front door"

# The layer must stay gitignored — the whole premise. Structure files stay tracked.
for f in registry.local.json models.d/x.yml composes/x/vllm/dual/base.yml; do
  if ! git check-ignore -q "scripts/lib/profiles-local/$f"; then
    printf '  FAIL %-34s NOT gitignored — a git pull could clobber a user model\n' "$f"
    fail=1
  fi
done
for f in README.md .gitignore; do
  if ! git ls-files --error-unmatch "scripts/lib/profiles-local/$f" >/dev/null 2>&1; then
    printf '  FAIL %-34s should stay TRACKED (the layer structure is committed)\n' "$f"
    fail=1
  fi
done

# promote.py must keep defaulting to the safe layer, and keep the core gate.
python3 - <<'PY' || fail=1
import re, subprocess, sys
src = open("scripts/lib/profiles/promote.py", encoding="utf-8").read()
ok = True

# BEHAVIOUR, not the literal. This check used to assert `default="local"` in the
# argparse call — and went red when #1155 changed it to `default=None` + a
# `args.layer or "local"` resolution, which preserves the behaviour exactly. A
# guard that tests a spelling instead of an outcome fails on a correct change and
# teaches people to edit the guard. Ask the tool what it does instead.
help_out = subprocess.run(
    [sys.executable, "scripts/lib/profiles/promote.py", "--help"],
    capture_output=True, text=True,
).stdout
if "local (default" not in help_out:
    print("  FAIL promote.py --help no longer states that LOCAL is the default")
    ok = False
if not re.search(r'choices=\("local",\s*"core"\)', src):
    print("  FAIL promote.py --layer no longer offers exactly local|core"); ok = False
# a defaulted (None) layer must still resolve to local, never to core
if not re.search(r'args\.layer\s*=\s*args\.layer\s*or\s*"local"', src):
    print("  FAIL a defaulted --layer no longer resolves to 'local'"); ok = False
if '_CORE_GATE_ENV = "C3_ALLOW_CORE_PROMOTE"' not in src:
    print("  FAIL promote.py core gate env renamed/removed"); ok = False
sys.exit(0 if ok else 1)
PY

if [[ $fail -ne 0 ]]; then
  echo "test-docs-local-layer: FAIL — see #1142 (BYOM routing)"
  exit 1
fi
echo "test-docs-local-layer: PASS — local layer is documented, routed and gitignored"
