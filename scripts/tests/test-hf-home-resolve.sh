#!/usr/bin/env bash
# Guard: deriver.resolve_hf_home precedence (club-3090 #617-followup). A bare
# `pull.sh <repo>` — only `.env`'s MODEL_DIR set, no explicit HF_HOME — must land
# on the MODEL DISK (MODEL_DIR/.cache/huggingface), NOT silently on ~/.cache (the
# footgun that misplaced a brought model's weights). Precedence asserted:
#   --hf-home > $HF_HOME > $MODEL_DIR/.cache/hf (env OR .env) > $XDG > ~/.cache
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import os, sys, tempfile
sys.path.insert(0, "scripts")
from pathlib import Path
from lib.profiles import deriver as D

def clear():
    for k in ("HF_HOME", "MODEL_DIR", "XDG_CACHE_HOME"):
        os.environ.pop(k, None)

# 1) --hf-home wins over everything
clear()
os.environ["HF_HOME"] = "/tmp/should-lose"
assert D.resolve_hf_home("/tmp/explicit") == Path("/tmp/explicit"), "--hf-home must win"
print("PASS 1: --hf-home wins")

# 2) $HF_HOME wins over MODEL_DIR
clear()
os.environ["MODEL_DIR"] = "/mnt/models/huggingface"
os.environ["HF_HOME"] = "/tmp/hfhome"
assert D.resolve_hf_home() == Path("/tmp/hfhome"), "$HF_HOME must win over MODEL_DIR"
print("PASS 2: $HF_HOME > MODEL_DIR")

# 3) MODEL_DIR (env) → MODEL_DIR/.cache/huggingface  (THE fix)
clear()
os.environ["MODEL_DIR"] = "/mnt/models/huggingface"
assert D.resolve_hf_home() == Path("/mnt/models/huggingface/.cache/huggingface"), \
    "MODEL_DIR must resolve to the model disk, not ~/.cache"
print("PASS 3: MODEL_DIR (env) → model disk")

# 4) MODEL_DIR read from .env when env unset (the bare-pull.sh case), via a
#    temp REPO_ROOT so the assertion is deterministic regardless of the live .env
tmp = Path(tempfile.mkdtemp())
(tmp / ".env").write_text('export MODEL_DIR="/data/models"\n', encoding="utf-8")
orig = D.REPO_ROOT
D.REPO_ROOT = tmp
try:
    clear()
    assert D.resolve_hf_home() == Path("/data/models/.cache/huggingface"), \
        "bare call must read MODEL_DIR from .env → model disk"
    print("PASS 4: MODEL_DIR from .env → model disk")

    # 5) no MODEL_DIR anywhere → the ~/.cache fallback is preserved
    (tmp / ".env").write_text("PORT=8010\n", encoding="utf-8")
    clear()
    assert D.resolve_hf_home() == Path.home() / ".cache" / "huggingface", \
        "no MODEL_DIR anywhere → ~/.cache fallback"
    print("PASS 5: no MODEL_DIR → ~/.cache fallback preserved")
finally:
    D.REPO_ROOT = orig

print("OK test-hf-home-resolve")
PY
