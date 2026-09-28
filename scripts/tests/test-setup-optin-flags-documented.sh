#!/usr/bin/env bash
# Gate: every optional `WITH_*` download flag that setup.sh READS must be named
# in the surface a user can reach — `bash scripts/setup.sh --help`
# (club-3090#1304).
#
# THE DEFECT
# ----------
# vllm/gemma-26ba4b-single needs an external MTP drafter that setup.sh fetches
# only under WITH_ASSISTANT_DRAFT=1. That flag existed in the model profile, in
# setup.sh's implementation, and in a test — and in no user-facing surface at
# all, while its two siblings WITH_DFLASH_DRAFT and WITH_VISION were documented
# in the file's header comment. A user who did not set it got a bare
# missing-config.json traceback out of vLLM naming neither the cause nor either
# fix (re-run setup with the flag, or boot the slug with SPEC_N=0).
#
# The header comment is not the surface this asserts on: reading it means
# opening the script. `--help` is what a stuck user actually types, so that is
# where the flags have to be. (Keeping the header comment in sync is still good
# practice — it is just not the thing that rescues anybody.)
#
# SCOPE: flags setup.sh reads in CODE. A `WITH_*` name appearing only inside a
# comment is documentation about some other script and is not a knob this file
# offers, so the extractor ignores comment lines.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
setup = root / "scripts/setup.sh"

READ_RE = re.compile(r"\$\{(WITH_[A-Z0-9_]+)")


def flags_read(text):
    """WITH_* flags the script reads in code (comment lines excluded)."""
    found = set()
    for line in text.split("\n"):
        if line.lstrip().startswith("#"):
            continue
        found.update(READ_RE.findall(line))
    return found


def undocumented(text, help_out):
    return sorted(f for f in flags_read(text) if f not in help_out)


# --- positive control FIRST -------------------------------------------------
# Prove the extractor sees a read flag, ignores a comment-only mention, and that
# a help text missing a flag is actually flagged. Without this, a scanner that
# quietly matched nothing would pass on any tree.
PROBE = (
    "#   WITH_MENTIONED_ONLY  a flag some other script owns\n"
    'if [[ "${WITH_PLANTED:-0}" == "1" ]]; then :; fi\n'
)
if flags_read(PROBE) != {"WITH_PLANTED"}:
    sys.exit(f"FAIL: positive control — extractor read {flags_read(PROBE)}, expected exactly WITH_PLANTED")
if undocumented(PROBE, "help text with no flags") != ["WITH_PLANTED"]:
    sys.exit("FAIL: positive control — an undocumented flag was not flagged")
if undocumented(PROBE, "... WITH_PLANTED Set to 1 to ...") != []:
    sys.exit("FAIL: positive control — a documented flag was wrongly flagged")
print("  ✓ extractor reads code-only flags and flags the undocumented ones")

# --- the real scan ----------------------------------------------------------
text = setup.read_text(encoding="utf-8")
help_out = subprocess.run(
    ["bash", str(setup), "--help"], capture_output=True, text=True,
).stdout

read = sorted(flags_read(text))
if not read:
    sys.exit("FAIL: no WITH_* flags found in setup.sh — the extractor is looking at the wrong thing")

missing = undocumented(text, help_out)
if missing:
    print("test-setup-optin-flags-documented: FAIL")
    for f in missing:
        print(f"  ⛔ {f} is read by scripts/setup.sh but never named in `setup.sh --help`")
    print()
    print("  Fix: add the flag to usage() (and to the header comment beside its siblings).")
    print("  An opt-in download nobody can discover is an opt-out download in practice.")
    sys.exit(1)

print(f"test-setup-optin-flags-documented: ok ({len(read)} opt-in flags — "
      f"{', '.join(read)} — all named in --help)")
PY
