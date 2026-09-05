#!/usr/bin/env bash
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs --
# repo sources carry unicode and a non-UTF-8 locale breaks the parser paths
# (#779).  Enforced by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"
# test-vram-breakdown.sh — regression for club-3090 #1171.
#
# The VRAM breakdown parser (`scripts/lib/vram_breakdown.py`) reported the
# DRAFTER's weights as `model=` on any slug with a GPU-pinned drafter
# (`_per_dev` last-wins vs. two additive `load_tensors` lines), and never
# matched GLM's `llama_memory_recurrent ... RS` state line (only DeepSeek's
# `_comp_state`).  Fix: `model` aggregates, and the RS line folds into the
# `state` column.
#
# Fixture-driven (deterministic log text, no Docker/GPU/network): asserts the
# per-device `model=`/`state=` tokens, which do not depend on nvidia-smi being
# present.  The `total=`/`unaccounted=` figures DO depend on a live nvidia-smi
# and are deliberately not asserted here; the real GLM+DFlash2 boot acceptance
# (model=3969, CUDA1 unaccounted 6080 -> 2561) is recorded in the PR.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import subprocess
import sys
import tempfile
from pathlib import Path

PARSER = ["python3", "scripts/lib/vram_breakdown.py"]
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


def _log_path(log_text: str) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(log_text)
    return f.name


def run(log_text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(log_text)
        path = f.name
    r = subprocess.run(PARSER + [path], capture_output=True, text=True)
    Path(path).unlink()
    if r.returncode != 0:
        failures.append(f"parser exited {r.returncode}: {r.stderr[:200]}")
    return r.stdout


# --- #1171: two-model boot (GLM-5.3-Flash target + DFlash2 drafter on CUDA1)
glm = """\
0.12.593.897 I load_tensors:        CUDA0 model buffer size =  3524.46 MiB
0.12.593.898 I load_tensors:        CUDA1 model buffer size =  3313.42 MiB
1.00.332.286 I llama_kv_cache:      CUDA0 KV buffer size =   337.50 MiB
1.00.358.673 I llama_memory_recurrent:      CUDA0 RS buffer size =   231.19 MiB
1.00.361.303 I llama_memory_recurrent:      CUDA1 RS buffer size =   205.50 MiB
1.02.627.213 I sched_reserve:      CUDA0 compute buffer size =  5329.04 MiB
1.02.627.225 I sched_reserve:      CUDA1 compute buffer size =  5208.35 MiB
1.03.247.930 I load_tensors:        CUDA1 model buffer size =   655.73 MiB
1.03.561.019 I llama_kv_cache:      CUDA1 KV buffer size =    80.00 MiB
"""
out = run(glm)
check("model=3969" in out,
      "drafter no longer overwrites model=: CUDA1 model= 3313.42+655.73=3969")
check("state=231" in out and "state=206" in out,
      "GLM recurrent RS lines are counted in state= per device")
check("model=3524" in out, "CUDA0 target model is unchanged")

# --- DeepSeek `_comp_state` must still parse into state= (no regression)
ds = """\
0.10.000.000 I load_tensors:        CUDA0 model buffer size =  1000.00 MiB
0.11.000.000 I llama_per_fixed_state:      _comp_state: CUDA0 q4_0 1 state buffer size =  128.00 MiB
"""
out = run(ds)
check("model=1000" in out, "single-model boot: model= is the one line, untouched")
check("state=128" in out, "DeepSeek _comp_state still parses into state=")

# --- no-drafter boot: single load_tensors line per device stays correct
single = """\
0.12.000.000 I load_tensors:        CUDA0 model buffer size =  2048.00 MiB
0.12.000.001 I load_tensors:        CUDA1 model buffer size =  1024.00 MiB
"""
out = run(single)
check("model=2048" in out and "model=1024" in out,
      "no-drafter boot: one line per device, values unchanged")

# --- #1118: --json emits one object per device with the full field set
import json as _json
import os as _os
import subprocess as _sp

r = _sp.run(PARSER + ["--json", _log_path(glm)], capture_output=True, text=True)
data = _json.loads(r.stdout)
devs = {d["device"]: d for d in data["devices"]}
check(set(devs) == {"CUDA0", "CUDA1"}, "--json: one object per device")
c1 = devs["CUDA1"]
check(c1["model"] == 3969 and c1["state"] == 206 and c1["kv"] == 80
      and c1["pool"] is None and c1["compute"] == 5208,
      "--json: component fields carry the parsed values (absent -> null)")
check(isinstance(data.get("warnings"), list), "--json: warnings list is present")

# --- #1118: negative unaccounted (stale log vs a live, nearly-empty card)
# must be CLAMPED to 0 and flagged in warnings -- never a negative other.
# A fake nvidia-smi on PATH simulates the stopped-container total (100 MiB
# used against ~10 GiB of log-derived components).
fake_dir = tempfile.mkdtemp(prefix="vram-smi-")
fake_smi = Path(fake_dir) / "nvidia-smi"
fake_smi.write_text('#!/bin/sh\necho "0, 100, 24576"\necho "1, 100, 24576"\n')
fake_smi.chmod(0o755)
env = dict(_os.environ)
env["PATH"] = fake_dir + ":" + env.get("PATH", "")
with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
    f.write(glm)
    stale_path = f.name
r = _sp.run(PARSER + ["--json", stale_path], capture_output=True, text=True,
            env=env)
Path(stale_path).unlink()
if r.returncode != 0:
    failures.append(f"--json (stale) parser exited {r.returncode}")
else:
    data = _json.loads(r.stdout)
    stale = [d for d in data["devices"] if d["device"] == "CUDA1"][0]
    check(stale["unaccounted"] == 0,
          "negative unaccounted is CLAMPED to 0, never negative")
    check(stale["model"] == 3969,
          "components still reported alongside the clamp")
    check(any("staler than the card" in w for w in data["warnings"]),
          "the clamp is flagged in warnings (staleness cue)")
Path(fake_smi).unlink()

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll vram_breakdown #1171 assertions passed.")
PY

echo "test-vram-breakdown.sh OK"
