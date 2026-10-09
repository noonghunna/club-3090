#!/usr/bin/env bash
# scripts/glm53-convert-mainline-gguf.py: mainline-format GLM-5.3 GGUF (arch `glm5-next`) -> llamacpp-club3090 (`glm5next`).
#
# A synthetic 2-shard GGUF is built IN-TEST from stdlib struct (no real weights, no network). Asserts:
#   1. converted shard 1: arch `glm5next`, `glm5-next.` prefix renamed, `expert_shared_feed_forward_length` added (2048),
#      `tokenizer.ggml.pre` glm5 -> glm4, `index_share_mtp` dropped, filler key present.
#   2. header length unchanged and every byte after the KV section (tensor infos + tensor data) identical.
#   3. shard 2 is a relative symlink to the original; the originals are untouched.
#   4. a second run is a no-op ("already converted"); --check-only writes nothing.
#   5. honest failures: a non-glm5-next GGUF and a dir without shard 1 exit non-zero.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONV="$ROOT_DIR/scripts/glm53-convert-mainline-gguf.py"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

python3 - "$TMP" <<'EOF'
import struct, sys, os
d = sys.argv[1]
def s(x): b = x.encode(); return struct.pack('<Q', len(b)) + b
def kv_str(k, v): return s(k) + struct.pack('<I', 8) + s(v)
def kv_u32(k, v): return s(k) + struct.pack('<I', 4) + struct.pack('<I', v)
def kv_bool(k, v): return s(k) + struct.pack('<I', 7) + struct.pack('<B', v)
def gguf(path, arch, kvs, tensor=True):
    ti = b''
    data = b''
    if tensor:  # one F32 tensor [4], offset 0
        ti = s('blk.0.attn_norm.weight') + struct.pack('<I', 1) + struct.pack('<Q', 4) + struct.pack('<I', 0) + struct.pack('<Q', 0)
        data = struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)
    body = b''.join(kvs)
    hdr = b'GGUF' + struct.pack('<IQQ', 3, 1 if tensor else 0, len(kvs)) + body + ti
    pad = (-len(hdr)) % 32
    open(path, 'wb').write(hdr + b'\0' * pad + data)
os.makedirs(f'{d}/src'); os.makedirs(f'{d}/other')
kvs = [kv_str('general.architecture', 'glm5-next'), kv_u32('glm5-next.block_count', 46),
       kv_u32('glm5-next.expert_feed_forward_length', 2048), kv_bool('glm5-next.attention.indexer.index_share_mtp', 1),
       kv_bool('glm5-next.attention.indexer.kpool_select_tail', 1), kv_str('tokenizer.ggml.pre', 'glm5'),
       kv_u32('split.count', 2)]
gguf(f'{d}/src/M-00001-of-00002.gguf', 'glm5-next', kvs)
gguf(f'{d}/src/M-00002-of-00002.gguf', 'glm5-next', [kv_u32('split.count', 2)])
gguf(f'{d}/other/X-00001-of-00001.gguf', 'llama', [kv_str('general.architecture', 'llama')])
EOF

fail() { echo "FAIL: $*"; exit 1; }
cp -a "$TMP/src" "$TMP/src.orig"
python3 "$CONV" "$TMP/src" "$TMP/dst" --check-only >/dev/null
[[ ! -e "$TMP/dst" ]] || fail "--check-only wrote files"
out="$(python3 "$CONV" "$TMP/src" "$TMP/dst")" || fail "conversion exited non-zero"
grep -q 'wrote' <<<"$out" || fail "no 'wrote' line: $out"

python3 - "$TMP" "$CONV" <<'EOF'
import sys, os, filecmp
from pathlib import Path
d, conv = sys.argv[1], sys.argv[2]
ns = {}; exec(open(conv).read().split('if __name__')[0], ns)
src, dst = Path(f'{d}/src/M-00001-of-00002.gguf'), Path(f'{d}/dst/M-00001-of-00002.gguf')
_, _, a, a_end, a_ti = ns['read_header'](src); _, _, b, b_end, b_ti = ns['read_header'](dst)
kv = {k: v for k, t, v in b}; keys = set(kv)
def sv(k): return ns['str_value'](b, k)
assert sv('general.architecture') == 'glm5next', sv('general.architecture')
assert sv('tokenizer.ggml.pre') == 'glm4', sv('tokenizer.ggml.pre')
assert 'glm5next.block_count' in keys and not any(k.startswith('glm5-next.') for k in keys), keys
assert int.from_bytes(kv['glm5next.expert_shared_feed_forward_length'], 'little') == 2048
assert 'glm5next.attention.indexer.index_share_mtp' not in keys
assert 'general.club3090_conversion' in keys
assert (a_end, a_ti) == (b_end, b_ti), ((a_end, a_ti), (b_end, b_ti))
x, y = src.read_bytes(), dst.read_bytes()
assert len(x) == len(y) and x[a_end:] == y[b_end:], 'bytes after the KV section differ'
link = Path(f'{d}/dst/M-00002-of-00002.gguf')
assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.resolve() == Path(f'{d}/src/M-00002-of-00002.gguf').resolve()
for f in os.listdir(f'{d}/src'):
    assert filecmp.cmp(f'{d}/src/{f}', f'{d}/src.orig/{f}', shallow=False), f'original {f} modified'
print('ok')
EOF

grep -q 'already converted' <<<"$(python3 "$CONV" "$TMP/src" "$TMP/dst")" || fail "second run was not a no-op"
python3 "$CONV" "$TMP/other" "$TMP/dst2" >/dev/null 2>&1 && fail "non-glm5-next GGUF was accepted"
mkdir -p "$TMP/empty"; python3 "$CONV" "$TMP/empty" "$TMP/dst3" >/dev/null 2>&1 && fail "dir without shard 1 was accepted"
echo "PASS test-glm53-convert-mainline-gguf"
