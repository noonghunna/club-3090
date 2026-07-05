#!/usr/bin/env bash
# test-artifact-inventory — the deriver's stage-1 INSPECT for the Bring funnel
# (bring-funnel design §2/§2b: artifact_inventory + inspect_repo).
#
# Offline: fixture API dicts only — NO network, NO weights. Guards the funnel's
# trust surface: a GGUF-only repo must be a first-class bring (not
# unsupported-format), variants must group multi-part files, mmproj must never
# masquerade as a servable variant, and lineage must ride along.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import sys

sys.path.insert(0, ".")
from scripts.lib.profiles.deriver import artifact_inventory  # noqa: E402

GB = 1024 ** 3


def api(siblings, card=None):
    d = {"siblings": [{"rfilename": n, "size": s} for n, s in siblings]}
    if card:
        d["cardData"] = card
    return d


# ── 1. GGUF-only repo: first-class, grouped, sorted by size ─────────────────
inv = artifact_inventory(api([
    ("Qwen-27B-Q4_K_M.gguf", 17 * GB),
    ("Qwen-27B-Q8_0-00001-of-00002.gguf", 15 * GB),
    ("Qwen-27B-Q8_0-00002-of-00002.gguf", 14 * GB),
    ("UD-Q5_K_XL/Qwen-27B-UD-Q5_K_XL.gguf", 20 * GB),   # subdir layout
    ("mmproj-F16.gguf", 1 * GB),
    ("README.md", 100),
], card={"base_model": "Qwen/Qwen3.6-27B"}))
assert inv["formats"] == ["gguf"], inv["formats"]
assert inv["safetensors"] is None
quants = [v["quant"] for v in inv["gguf_variants"]]
assert quants == ["Q4_K_M", "UD-Q5_K_XL", "Q8_0"], quants   # size-sorted
q8 = inv["gguf_variants"][2]
assert q8["parts"] == 2 and abs(q8["size_gb"] - 29.0) < 0.01, q8
assert inv["gguf_mmproj"] == ["mmproj-F16.gguf"]            # NOT a variant
assert inv["lineage_base_model"] == "Qwen/Qwen3.6-27B"
print("  ✓ gguf-only: grouped, multi-part summed, mmproj split, lineage rides")

# ── 2. safetensors-only repo (adapters excluded) ─────────────────────────────
inv = artifact_inventory(api([
    ("model-00001-of-00002.safetensors", 15 * GB),
    ("model-00002-of-00002.safetensors", 14 * GB),
    ("adapter_model.safetensors", 1 * GB),               # adapter — excluded
    ("model.safetensors.index.json", 90_000),
    ("config.json", 900),
]))
assert inv["formats"] == ["safetensors"], inv["formats"]
assert len(inv["safetensors"]["weight_files"]) == 2
assert abs(inv["safetensors"]["size_gb"] - 29.0) < 0.01
assert inv["gguf_variants"] == []
print("  ✓ safetensors-only: adapter excluded, sharded set sized")

# ── 3. both formats in one repo ──────────────────────────────────────────────
inv = artifact_inventory(api([
    ("model.safetensors", 30 * GB),
    ("model-Q4_K_M.gguf", 17 * GB),
]))
assert inv["formats"] == ["safetensors", "gguf"], inv["formats"]
print("  ✓ mixed repo carries both formats")

# ── 4. empty / unservable repo ───────────────────────────────────────────────
inv = artifact_inventory(api([("README.md", 100)]))
assert inv["formats"] == [] and inv["safetensors"] is None and inv["gguf_variants"] == []
print("  ✓ unservable repo → empty formats (inspect_repo maps this to error)")

# ── 5. tokenless gguf never dropped (keys by stem) ───────────────────────────
inv = artifact_inventory(api([("weird-model.gguf", 5 * GB)]))
assert len(inv["gguf_variants"]) == 1 and inv["gguf_variants"][0]["quant"] == "weird-model"
print("  ✓ unparseable quant token keys by stem (never silently dropped)")
PY

# CLI shim: --inventory is required; bad usage exits 2 without network
if python3 scripts/lib/profiles/deriver.py some/repo >/dev/null 2>&1; then
  echo "FAIL: CLI without --inventory must refuse" >&2; exit 1
fi
echo "  ✓ CLI refuses without --inventory (no accidental network mode)"

echo "test-artifact-inventory: ok"
