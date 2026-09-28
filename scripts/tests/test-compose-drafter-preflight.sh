#!/usr/bin/env bash
# Gate: vllm/gemma-26ba4b-single must refuse to boot with a message that names
# BOTH ways out when its external MTP drafter is missing (club-3090#1304).
#
# THE DEFECT
# ----------
# setup.sh fetches this drafter only under WITH_ASSISTANT_DRAFT=1, a flag that
# appeared in no user-facing surface, while the slug defaults to SPEC_N=4. So
# the ordinary path — `setup.sh gemma-4-26b-a4b`, then `switch.sh
# vllm/gemma-26ba4b-single` — reached vLLM with no drafter on disk and produced
# a bare missing-config.json. That error names neither the cause (an un-fetched
# optional download) nor either fix, and a user on Discord reasonably concluded
# the download should have been automatic, hand-downloaded a file, renamed it to
# match, and hit a second error.
#
# Both fixes have to be IN the message, because a user who reads it is by
# definition someone who did not know about the first one:
#   - re-run setup with WITH_ASSISTANT_DRAFT=1, or
#   - boot without the drafter using SPEC_N=0 (standardised across all 27 vLLM
#     composes by #1048 — it needs no downloads and no file edits).
#
# This is the vLLM twin of the llama.cpp fail-loud check the deepseek moecache
# composes carry (#1054). That family resolves its drafter path out of `command:`
# argv and is exercised by test-spec-toggle-contract.sh; this one embeds the path
# in the entrypoint, so it is asserted here.
#
# Hermetic: the entrypoint is extracted and run with `vllm` stubbed. No docker,
# no image, no weights, no GPU.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - "$ROOT_DIR" <<'PY'
import os
import pathlib
import re
import subprocess
import sys
import tempfile

root = pathlib.Path(sys.argv[1])
compose = root / "models/gemma-4-26b-a4b/vllm/compose/single/awq/int8.yml"
DRAFTER = "gemma-4-26b-a4b-it-assistant"

text = compose.read_text(encoding="utf-8")


def entrypoint_body(t):
    """The `entrypoint:` block-scalar body, dedented."""
    lines = t.split("\n")
    for i, l in enumerate(lines):
        if not re.match(r"^\s*entrypoint:\s*$", l):
            continue
        j = i + 1
        while j < len(lines) and not re.match(r"^\s*-\s*\|\s*$", lines[j]):
            j += 1
        if j >= len(lines):
            return None
        ind = len(lines[j]) - len(lines[j].lstrip()) + 2
        k = j + 1
        out = []
        while k < len(lines):
            if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) < ind:
                break
            out.append(lines[k][ind:] if len(lines[k]) > ind else "")
            k += 1
        return "\n".join(out)
    return None


body = entrypoint_body(text)
if not body:
    sys.exit(f"FAIL: could not extract the entrypoint block from {compose.name}")


def run(drafter_present, env=None):
    """Run the entrypoint with `vllm` stubbed; return (rc, combined output)."""
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d)
        stub = dp / "vllm"
        stub.write_text('#!/bin/bash\nfor a in "$@"; do printf "ARG:%s\\n" "$a"; done\nexit 0\n')
        stub.chmod(0o755)
        etc = dp / "etc" / "club3090"
        for sub in set(re.findall(r"/etc/club3090/([\w.-]+)/install\.sh", body)):
            (etc / sub).mkdir(parents=True, exist_ok=True)
            (etc / sub / "install.sh").write_text("#!/bin/bash\nexit 0\n")
        etc.mkdir(parents=True, exist_ok=True)
        hfc = dp / "hf"
        hfc.mkdir(parents=True, exist_ok=True)
        if drafter_present:
            (hfc / DRAFTER).mkdir(parents=True, exist_ok=True)
            (hfc / DRAFTER / "config.json").write_text("{}", encoding="utf-8")
        script = dp / "ep.sh"
        script.write_text(
            body.replace("$$", "$")
                .replace("/etc/club3090", str(etc))
                .replace("/root/.cache/huggingface", str(hfc))
        )
        e = {k: v for k, v in os.environ.items() if not k.startswith("SPEC")}
        e.update(env or {})
        e["PATH"] = f"{d}:{os.environ['PATH']}"
        r = subprocess.run(["bash", str(script), "--", "--model", "x"],
                           capture_output=True, text=True, encoding="utf-8",
                           env=e, timeout=60)
        return r.returncode, (r.stdout or "") + (r.stderr or "")


# --- 1. drafter MISSING at the default depth => refuse, and say how -----------
rc, out = run(drafter_present=False)
if rc == 0:
    sys.exit("FAIL: the drafter is absent and the compose still handed the request to vLLM "
             "— that is the bare missing-config.json path #1304 is about:\n" + out)
if "ARG:--speculative-config" in out:
    sys.exit("FAIL: the compose still passed --speculative-config with no drafter on disk")
missing = [needle for needle in ("WITH_ASSISTANT_DRAFT", "SPEC_N=0") if needle not in out]
if missing:
    sys.exit(f"FAIL: the refusal message never names {', '.join(missing)} — a user who reads "
             f"it is by definition someone who did not already know:\n{out}")
print("  ✓ a missing drafter refuses the boot and names BOTH routes "
      "(WITH_ASSISTANT_DRAFT=1 / SPEC_N=0)")

# --- 2. the check must not fire when the drafter IS present ------------------
rc, out = run(drafter_present=True)
if rc != 0:
    sys.exit(f"FAIL: the drafter is present and the compose refused anyway (rc={rc}):\n{out}")
if "ARG:--speculative-config" not in out:
    sys.exit(f"FAIL: drafter present but --speculative-config never reached the engine:\n{out}")
print("  ✓ with the drafter present the check is inert and the drafter is passed through")

# --- 3. SPEC_N=0 must not need the drafter at all ----------------------------
# The whole point of the escape hatch: it works on a machine that never
# downloaded the file. A check that fired before the SPEC_N test would break it.
rc, out = run(drafter_present=False, env={"SPEC_N": "0"})
if rc != 0:
    sys.exit(f"FAIL: SPEC_N=0 with no drafter on disk was refused (rc={rc}) — "
             f"the documented escape hatch must not require the download:\n{out}")
if "ARG:--speculative-config" in out:
    sys.exit("FAIL: SPEC_N=0 still passed --speculative-config")
print("  ✓ SPEC_N=0 boots with no drafter on disk (the escape hatch stays usable)")

print("test-compose-drafter-preflight: ok (club-3090#1304)")
PY
