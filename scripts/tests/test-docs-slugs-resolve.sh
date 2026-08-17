#!/usr/bin/env bash
# Every slug named in a DOCUMENTED COMMAND must exist in the registry, and
# non-functional slugs must be shown with --force.
#
# WHY THIS EXISTS
# ---------------
# 2026-08-17: six separate places told users to run commands that could not
# work. None was caught by any gate, because nothing asserted that a slug
# appearing in prose still exists:
#
#   README.md            first explicit command was `beellama/dflash` — deprecated,
#                        labelled "single-card BLESSED default"
#   SINGLE_CARD.md       Quick start ran ik-llama/iq4ks-mtp + llamacpp/default
#                        (both deprecated, no --force); vllm/minimal — the only
#                        production single-card slug — was absent entirely
#   DUAL_CARD.md         3 of 4 --variant lines named absent slugs
#   MULTI_CARD.md        topology table named vllm/default + vllm/long-text
#   FAQ.md               `--set-default vllm/dual-turbo` told users to PERSIST a
#                        broken default; the 5-step troubleshooting ladder had
#                        3 of 5 rungs on absent slugs
#   discussion #17       onboarding path used a compose path the restructure removed
#
# `launch.sh` exits 2 with "unknown compose variant", so the failure is clean —
# but a new user cannot tell a stale doc from their own mistake.
#
# SCOPE — deliberately narrow, so the gate stays trustworthy
# ----------------------------------------------------------
# Only lines invoking `switch.sh` or `launch.sh` are inspected. That is what makes
# this checkable at all: the repo has 106 `vllm/pull`, 51 `vllm/issues`, 30
# `vllm/patches` and 27 `vllm/vllm-openai` strings in prose, all GitHub URLs or
# paths that happen to match a slug shape. None of them appear inside a launch
# command. A gate that scanned all prose would be noise and would get disabled.
#
# Resolver tokens are legal and NOT slugs: `<engine>/default` and `<model>/default`
# are resolved by model_default_target(), documented in FAQ.md.

set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs — repo
# docs are full of unicode (— × → ⚠) and a non-UTF-8, non-C locale otherwise
# decodes reads/stdout/argv with the locale codec (#779). Guarded by
# test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from scripts.lib.profiles.compose_registry import (
    COMPOSE_REGISTRY,
    FUNCTIONAL_STATUSES,
)

# Engine prefixes and model ids are DERIVED from the registry, never hardcoded —
# a new engine or model must not silently fall outside the gate.
ENGINES = sorted({k.split("/", 1)[0] for k in COMPOSE_REGISTRY})
MODELS = sorted({e["model"] for e in COMPOSE_REGISTRY.values() if e.get("model")})

# Which docs a user actually follows. Contributor-track and diagnostics prose are
# out of scope on purpose; those aren't copy-paste onboarding paths.
TARGETS = [
    "README.md",
    "CONTRIBUTING.md",
    *sorted(str(p.relative_to(root)) for p in (root / "docs").glob("*.md")),
]

# Match a launcher invocation and capture only what sits in ARGUMENT position.
# Matching the whole line over-fires: tables and retirement notes mention slugs in
# prose while also containing a launcher call elsewhere on the same line, and
# `VLLM_IMAGE=vllm/vllm-openai:latest` is an image reference, not a slug.
ARG = re.compile(
    r"\b(?:switch|launch)\.sh\b"                 # the launcher
    r"((?:\s+(?:--[\w-]+(?:=\S+)?|[A-Z_]+=\S+))*)"  # intervening flags / env
    r"\s+([A-Za-z0-9][\w./-]*)"                    # the argument we care about
)
# --variant / --set-default take the slug as the NEXT token.
FLAG_ARG = re.compile(r"--(?:variant|set-default)\s+([A-Za-z0-9][\w./-]*)")
# A trailing :tag means it's a container image, never a slug.
IMAGEISH = re.compile(r"^\S+:[\w.-]+$")

def check_line(rel, lineno, line, sink, counters):
    """Classify one line. Appends problems to `sink`. Pure — used by the self-test."""
    if "switch.sh" not in line and "launch.sh" not in line:
        return
    has_force = "--force" in line or "FORCE=1" in line

    cands = set()
    for m in ARG.finditer(line):
        cands.add(m.group(2))
    for m in FLAG_ARG.finditer(line):
        cands.add(m.group(1))

    for tok in cands:
        if IMAGEISH.match(tok):          # vllm/vllm-openai:latest
            continue
        if "/" not in tok:               # `switch.sh --list`, a bare model id
            continue
        prefix, _, name = tok.partition("/")

        # ⚠️ REGISTRY LOOKUP MUST COME FIRST. `llamacpp/default` is BOTH a real
        # (deprecated) slug AND matches the `<engine>/default` resolver shape.
        # Checking the resolver pattern first made this gate SKIP it — and it is the
        # most-referenced deprecated slug in the docs (29 mentions), so that bug
        # would have defeated the main thing this test exists for. It was caught by
        # the self-test below, not by reading the code. Do not reorder these.
        entry = COMPOSE_REGISTRY.get(tok)

        if entry is None and name == "default":
            if prefix in ENGINES or prefix in MODELS:
                continue
            sink.append(
                f"{rel}:{lineno}: `{tok}` names neither an engine nor a model "
                f"in the registry — the resolver will reject it"
            )
            continue

        if prefix not in ENGINES:
            continue                     # not a slug shape we own

        counters["checked"] += 1
        if entry is None:
            sink.append(
                f"{rel}:{lineno}: `{tok}` is NOT in compose_registry.py — "
                f"this command cannot run (launch.sh exits 2)"
            )
            continue

        status = entry.get("status", "production")
        if status not in FUNCTIONAL_STATUSES and not has_force:
            sink.append(
                f"{rel}:{lineno}: `{tok}` is status={status} (non-functional) "
                f"but the command has no --force — it will be refused"
            )
        elif status not in FUNCTIONAL_STATUSES:
            counters["force_ok"] += 1


# ---------------------------------------------------------------------------
# SELF-TEST — a gate that silently stops catching things is worse than no gate.
# Each case below is one the real docs got wrong at some point.
# ---------------------------------------------------------------------------
_dep = next((k for k, v in COMPOSE_REGISTRY.items()
             if v.get("status") == "deprecated" and k.endswith("/default")), None)
_prod = next((k for k, v in COMPOSE_REGISTRY.items()
              if v.get("status") == "production"), None)

CASES = [
    ("absent slug",                 "bash scripts/switch.sh vllm/does-not-exist",              True),
    ("--variant absent",            "bash scripts/launch.sh --variant vllm/nope",              True),
    ("--set-default absent",        "bash scripts/switch.sh --set-default vllm/nope",          True),
    ("bogus <model>/default",       "bash scripts/switch.sh not-a-real-model/default",         True),
    ("production slug",             f"bash scripts/switch.sh {_prod}",                         False),
    ("<engine>/default resolver",   "bash scripts/switch.sh vllm/default",                     False),
    ("docker image, not a slug",
     "VLLM_IMAGE=vllm/vllm-openai:latest bash scripts/launch.sh --variant " + str(_prod),      False),
    ("slug in prose beside a call",
     "The retired `llamacpp/mtp` is listed by `bash scripts/switch.sh --list`.",               False),
]
if _dep:
    CASES += [
        ("deprecated, no --force",  f"bash scripts/switch.sh {_dep}",                          True),
        ("deprecated with --force", f"bash scripts/switch.sh --force {_dep}",                  False),
    ]

_self_fail = []
for label, line, should_flag in CASES:
    sink, ctr = [], {"checked": 0, "force_ok": 0}
    check_line("<self-test>", 0, line, sink, ctr)
    if bool(sink) != should_flag:
        _self_fail.append(
            f"self-test '{label}': expected {'a problem' if should_flag else 'no problem'}, "
            f"got {sink or 'none'}"
        )
if _self_fail:
    print("test-docs-slugs-resolve: FAIL (the gate itself is broken)")
    for e in _self_fail:
        print(f"  ⛔ {e}")
    sys.exit(1)

errors = []
counters = {"checked": 0, "force_ok": 0}

for rel in TARGETS:
    path = root / rel
    if not path.exists():
        continue
    for lineno, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        check_line(rel, lineno, line, errors, counters)

checked = counters["checked"]
force_ok = counters["force_ok"]

if errors:
    print("test-docs-slugs-resolve: FAIL")
    for e in errors:
        print(f"  ⛔ {e}")
    print()
    print(f"  {len(errors)} problem(s) across {len(TARGETS)} doc(s).")
    print("  Fix: name a slug that exists (`bash scripts/switch.sh --list`), add")
    print("  --force for a non-functional one, or use the `<model>/default` resolver.")
    sys.exit(1)

print(
    f"test-docs-slugs-resolve: ok "
    f"self-test {len(CASES)}/{len(CASES)} · {checked} slug refs in launcher commands across {len(TARGETS)} docs, "
    f"{force_ok} non-functional correctly gated behind --force)"
)
PY
