#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# test-repack-prefix.sh — smoke test for tools/repack_prefix.py
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

TOOL="$REPO/tools/repack_prefix.py"
SRC="$TMP/src"
DST="$TMP/dst"

echo "=== test-repack-prefix: generate dummy checkpoint ==="
mkdir -p "$SRC" "$DST"

export SRCD="$SRC"
python3 <<'PYEOF'
import json, struct, os

srcd = os.environ['SRCD']

data = b'\x00\x00\x00\x00' * 8
hdr = {
    'model.language_model.layers.0.self_attn.q_proj.qweight': {
        'dtype': 'F32', 'shape': [8], 'data_offsets': [0, 32],
    },
    'model.language_model.layers.0.self_attn.k_proj.qweight': {
        'dtype': 'F32', 'shape': [8], 'data_offsets': [32, 64],
    },
}
blob = json.dumps(hdr, separators=(',', ':')).encode()
pad = (8 - (len(blob) % 8)) % 8
blob = blob + b' ' * pad

with open(os.path.join(srcd, 'model-00001-of-00001.safetensors'), 'wb') as f:
    f.write(struct.pack('<Q', len(blob)))
    f.write(blob)
    f.write(data * 2)

idx = {
    'metadata': {'total_size': 64},
    'weight_map': {
        'model.language_model.layers.0.self_attn.q_proj.qweight': 'model-00001-of-00001.safetensors',
        'model.language_model.layers.0.self_attn.k_proj.qweight': 'model-00001-of-00001.safetensors',
    },
}
json.dump(idx, open(os.path.join(srcd, 'model.safetensors.index.json'), 'w'))

cfg = {
    'architectures': ['Qwen3_5ForCausalLM'],
    'model_type': 'qwen3_5',
    'quantization_config': {
        'quant_method': 'auto-round',
        'bits': 4,
        'block_name_to_quantize': ['model.language_model.layers'],
        'extra_config': {
            'model.language_model.layers.0.self_attn.q_proj': {'bits': 16},
        },
    },
}
json.dump(cfg, open(os.path.join(srcd, 'config.json'), 'w'))
print('generated dummy checkpoint in ' + srcd)
PYEOF

# ── Run repack_prefix.py strip ──────────────────────────────────────────────
echo "=== test-repack-prefix: strip direction ==="
python3 "$TOOL" --src "$SRC" --dst "$DST/stripped" --direction strip || {
    echo "FAIL: strip run"; exit 1
}

# ── Verify strip ────────────────────────────────────────────────────────────
echo "=== test-repack-prefix: verify strip ==="
export DSTD="$DST"
python3 <<'PYEOF'
import json, os, struct

dstd = os.environ['DSTD']
out = os.path.join(dstd, 'stripped')

idx = json.load(open(os.path.join(out, 'model.safetensors.index.json')))
for k in idx['weight_map']:
    assert k.startswith('model.'), 'key ' + k + ' should start with model.'
    assert not k.startswith('model.language_model.'), 'key ' + k + ' should be stripped'
print('  index: ' + str(len(idx['weight_map'])) + ' keys OK')

cfg = json.load(open(os.path.join(out, 'config.json')))
bnq = cfg['quantization_config']['block_name_to_quantize']
assert isinstance(bnq, list) and bnq == ['model.layers'], 'bnq=' + str(bnq)
ec = cfg['quantization_config']['extra_config']
for k in ec:
    assert k.startswith('model.layers.'), 'extra_config key ' + k + ' should start with model.layers.'
print('  config.json quant keys OK')

with open(os.path.join(out, 'model-00001-of-00001.safetensors'), 'rb') as f:
    hlen = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(hlen))
    for k in hdr:
        if k == '__metadata__':
            continue
        assert k.startswith('model.'), 'safetensors key ' + k + ' should start with model.'
        assert not k.startswith('model.language_model.'), 'safetensors key ' + k + ' should be stripped'
    remaining = f.read()
    assert len(remaining) == 64, 'expected 64 data bytes, got ' + str(len(remaining))
    assert all(b == 0 for b in remaining), 'data bytes should be all zeros'
print('  safetensors header + data integrity OK')

print('strip: PASS')
PYEOF

# ── Roundtrip: strip -> inject ──────────────────────────────────────────────
echo "=== test-repack-prefix: roundtrip (strip then inject) ==="
# Use the stripped output as source for inject
python3 "$TOOL" --src "$DST/stripped" --dst "$DST/roundtripped" --direction inject || {
    echo "FAIL: roundtrip inject run"; exit 1
}

python3 <<'PYEOF'
import json, os, struct

dstd = os.environ['DSTD']
out = os.path.join(dstd, 'roundtripped')

idx = json.load(open(os.path.join(out, 'model.safetensors.index.json')))
for k in idx['weight_map']:
    assert k.startswith('model.language_model.'), 'key ' + k + ' should have language_model prefix'
print('  index: ' + str(len(idx['weight_map'])) + ' keys OK')

cfg = json.load(open(os.path.join(out, 'config.json')))
bnq = cfg['quantization_config']['block_name_to_quantize']
assert bnq == ['model.language_model.layers'], 'bnq=' + str(bnq)
ec = cfg['quantization_config']['extra_config']
for k in ec:
    assert k.startswith('model.language_model.layers.'), 'extra_config key ' + k + ' should be prefixed'
print('  config.json quant keys OK')

print('roundtrip: PASS')
PYEOF

# ── Dry-run ─────────────────────────────────────────────────────────────────
echo "=== test-repack-prefix: dry-run ==="
python3 "$TOOL" --src "$SRC" --dst "$DST/dry" --direction strip --dry-run || {
    echo "FAIL: dry-run"; exit 1
}
echo "dry-run: PASS"

echo ""
echo "=== ALL PASS ==="
