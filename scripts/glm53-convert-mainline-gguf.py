#!/usr/bin/env python3
"""Make a mainline-format GLM-5.3-Flash GGUF loadable by the llamacpp-club3090 engine.

GGUFs converted by mainline llama.cpp (ggml-org/llama.cpp#27773: avar6, AesSedai, ...) declare the architecture
`glm5-next`. The llamacpp-club3090 engine carries GLM support from #27754, which reads `glm5next`. The tensors are
named and shaped the same in both; only shard 1's key/value metadata differs. This script writes a converted copy of
shard 1 into DST_DIR and symlinks every other shard next to it, so the published files stay byte-for-byte as
downloaded (and keep passing setup's sha256 check).

What changes in shard 1's metadata, nothing else:
  - general.architecture `glm5-next` -> `glm5next`, and every `glm5-next.` key prefix -> `glm5next.`
  - adds `glm5next.expert_shared_feed_forward_length` (u32 2048): #27773 does not write it, the engine reads it
  - `tokenizer.ggml.pre` `glm5` -> `glm4`: mainline maps both names to the same pre-tokenizer; the engine knows `glm4`
  - drops `attention.indexer.index_share_mtp` / `kpool_select_tail` (mainline-only, never read by the engine)
A filler string key keeps the header the same length, so the data section and every tensor offset are unchanged.

usage: glm53-convert-mainline-gguf.py SRC_DIR DST_DIR [--check-only]
Idempotent: if DST_DIR already holds a converted shard 1 with the right header, it does nothing.
Pure Python (no numpy, no gguf-py), so it runs on the host python3 that setup.sh uses.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

OLD, NEW = "glm5-next", "glm5next"
DROP = {"glm5-next.attention.indexer.index_share_mtp"}
DROP_IF_NEEDED = ["glm5-next.attention.indexer.kpool_select_tail", "glm5-next.attention.indexer.types"]
ADD_U32 = {"glm5next.expert_shared_feed_forward_length": 2048}
FILLER_KEY = "general.club3090_conversion"
NOTE = "glm5-next->glm5next for llamacpp-club3090 (scripts/glm53-convert-mainline-gguf.py); tensors unchanged"
T_U32, T_STR, T_ARR = 4, 8, 9
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def pstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def read_header(path: Path):
    with open(path, "rb") as f:
        buf = f.read(64 * 1024 * 1024)  # KV + tensor infos are far below 64 MiB
    if buf[:4] != b"GGUF":
        sys.exit(f"{path}: not a GGUF file")
    ver, n_t, n_kv = struct.unpack_from("<IQQ", buf, 4)
    if ver != 3:
        sys.exit(f"{path}: GGUF v{ver}, expected v3")
    off = 24

    def rstr(o):
        n, = struct.unpack_from("<Q", buf, o)
        return buf[o + 8:o + 8 + n].decode(), o + 8 + n

    def skip(t, o):
        if t in FIXED:
            return o + FIXED[t]
        if t == T_STR:
            return rstr(o)[1]
        if t == T_ARR:
            at, n = struct.unpack_from("<IQ", buf, o)
            o += 12
            if at in FIXED:
                return o + FIXED[at] * n
            for _ in range(n):
                o = skip(at, o)
            return o
        raise ValueError(f"unknown GGUF value type {t}")

    kvs = []
    for _ in range(n_kv):
        key, off = rstr(off)
        t, = struct.unpack_from("<I", buf, off)
        off += 4
        v0 = off
        off = skip(t, off)
        kvs.append((key, t, buf[v0:off]))
    kv_end = off
    for _ in range(n_t):
        _, off = rstr(off)
        nd, = struct.unpack_from("<I", buf, off)
        off += 4 + 8 * nd + 4 + 8
    return buf, n_t, kvs, kv_end, off


def str_value(kvs, key):
    for k, t, v in kvs:
        if k == key and t == T_STR:
            return v[8:].decode()
    return None


def build(kvs, extra_drop):
    have = {k for k, _, _ in kvs}
    out = []
    for k, t, v in kvs:
        if k in DROP or k in extra_drop:
            continue
        if k == "general.architecture":
            v = pstr(NEW)
        if k == "tokenizer.ggml.pre" and v == pstr("glm5"):
            v = pstr("glm4")
        if k.startswith(OLD + "."):
            k = NEW + k[len(OLD):]
        out.append(pstr(k) + struct.pack("<I", t) + v)
    for k, val in ADD_U32.items():
        if k not in have and k.replace(NEW, OLD, 1) not in have:
            out.append(pstr(k) + struct.pack("<I", T_U32) + struct.pack("<I", val))
    return out


def new_header(buf, n_t, kvs, kv_end):
    fixed = len(pstr(FILLER_KEY)) + 4 + 8
    extra_drop: set[str] = set()
    for attempt in range(len(DROP_IF_NEEDED) + 1):
        body = b"".join(build(kvs, extra_drop))
        saved = (kv_end - 24) - len(body)
        if saved >= fixed:
            break
        if attempt < len(DROP_IF_NEEDED):
            extra_drop.add(DROP_IF_NEEDED[attempt])
    else:
        sys.exit("cannot keep the header length; this GGUF needs a full rewrite")
    pad = saved - fixed
    note = (NOTE + " " * pad)[:pad]
    n_kv = len(build(kvs, extra_drop)) + 1
    header = buf[:4] + struct.pack("<IQQ", 3, n_t, n_kv) + body + pstr(FILLER_KEY) + struct.pack("<I", T_STR) + pstr(note)
    assert len(header) == kv_end, (len(header), kv_end)
    return header, sorted((DROP | extra_drop) & {k for k, _, _ in kvs})


def is_converted(path: Path) -> bool:
    try:
        _, _, kvs, _, _ = read_header(path)
    except (OSError, SystemExit, ValueError, struct.error):
        return False
    keys = {k for k, _, _ in kvs}
    return (str_value(kvs, "general.architecture") == NEW and FILLER_KEY in keys
            and all(k in keys for k in ADD_U32) and str_value(kvs, "tokenizer.ggml.pre") != "glm5")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src_dir", type=Path)
    ap.add_argument("dst_dir", type=Path)
    ap.add_argument("--check-only", action="store_true", help="print the plan, write nothing")
    a = ap.parse_args()

    shard1 = sorted(a.src_dir.glob("*-00001-of-*.gguf"))
    if len(shard1) != 1:
        sys.exit(f"expected exactly one *-00001-of-*.gguf in {a.src_dir}, found {len(shard1)}")
    src = shard1[0]
    prefix = src.name.split("-00001-of-")[0]
    shards = sorted(a.src_dir.glob(f"{prefix}-*-of-*.gguf"))
    dst = a.dst_dir / src.name

    if dst.exists() and is_converted(dst) and dst.stat().st_size == src.stat().st_size:
        print(f"[convert] already converted: {dst}")
        return 0

    buf, n_t, kvs, kv_end, ti_end = read_header(src)
    arch = str_value(kvs, "general.architecture")
    if arch == NEW:
        sys.exit(f"{src} is already glm5next; nothing to convert (point the slug at it directly)")
    if arch != OLD:
        sys.exit(f"{src}: architecture {arch!r}, expected {OLD!r}")
    header, dropped = new_header(buf, n_t, kvs, kv_end)
    print(f"[convert] {src.name}: {len(kvs)} keys, header {kv_end} bytes unchanged, {n_t} tensors in shard 1, "
          f"{len(shards)} shards; dropping {dropped}; adding {list(ADD_U32)}")
    if a.check_only:
        return 0

    a.dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as f:
        f.seek(0)
        f.write(header)
    _, n_t2, _, kv_end2, ti_end2 = read_header(tmp)
    if (n_t2, kv_end2, ti_end2) != (n_t, kv_end, ti_end):
        tmp.unlink()
        sys.exit("layout changed after rewrite; aborting")
    with open(src, "rb") as x, open(tmp, "rb") as y:  # the tensor-info section must be byte-identical
        x.seek(kv_end)
        y.seek(kv_end)
        if x.read(ti_end - kv_end) != y.read(ti_end - kv_end):
            tmp.unlink()
            sys.exit("tensor-info section differs after rewrite; aborting")
    os.replace(tmp, dst)
    for s in shards:
        if s == src:
            continue
        link = a.dst_dir / s.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(s, a.dst_dir))
    print(f"[convert] wrote {dst} (+{len(shards) - 1} shard links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
