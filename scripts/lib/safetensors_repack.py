"""Low-level safetensors header-rename + verbatim-data-copy primitives.

Used by:
  - tools/repack_prefix.py  (standalone CLI)
  - scripts/lib/repackage_autoround_int4.py  (AutoRound INT4 repackager)

STDLIB ONLY.  Reads default to utf-8 per repo convention.
"""
from __future__ import annotations

import json
import struct

COPY_CHUNK = 32 * 1024 * 1024  # 32 MiB


# ---------------------------------------------------------------------------
# Renamer factories
# ---------------------------------------------------------------------------

def make_renamer(direction: str):
    """Return a key-renaming callable.

    direction "strip":  model.language_model.* -> model.*
    direction "inject": model.* -> model.language_model.*
    """
    if direction == "strip":
        old, new = "model.language_model.", "model."
    elif direction == "inject":
        old, new = "model.", "model.language_model."
    else:
        raise ValueError(f"unknown direction: {direction!r}")

    def rn(k: str) -> str:
        if k == "__metadata__":
            return k
        if direction == "inject" and k.startswith(new):
            return k  # already prefixed — don't double-inject
        if k.startswith(old):
            return new + k[len(old):]
        return k  # lm_head.*, mtp.* untouched
    return rn


# ---------------------------------------------------------------------------
# Header I/O
# ---------------------------------------------------------------------------

def read_header(path: str):
    """Read a safetensors file's JSON header.  Returns (header_length, dict)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    return hlen, hdr


# ---------------------------------------------------------------------------
# Core rewrite
# ---------------------------------------------------------------------------

def rewrite_file(src: str, dst: str, rename_key) -> dict:
    """Copy *src* -> *dst* renaming header keys; stream-copy data verbatim.

    Returns {new_key: nbytes} for building the index.
    """
    hlen, hdr = read_header(src)
    meta = hdr.get("__metadata__")
    new_hdr: dict = {}
    if meta is not None:
        new_hdr["__metadata__"] = meta
    sizes: dict[str, int] = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        nk = rename_key(k)
        if nk in new_hdr:
            raise ValueError(f"key collision after rename: {nk}")
        new_hdr[nk] = v
        off = v["data_offsets"]
        sizes[nk] = off[1] - off[0]

    blob = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(blob) % 8)) % 8
    blob = blob + b" " * pad

    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fin.seek(8 + hlen)  # skip original header -> data section
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        while True:
            chunk = fin.read(COPY_CHUNK)
            if not chunk:
                break
            fout.write(chunk)
    return sizes


# ---------------------------------------------------------------------------
# Config.json re-prefixing  (Qwen3.5/3.6 block_name_to_quantize + extra_config)
# ---------------------------------------------------------------------------

def reprefix_config(cfg: dict, direction: str) -> dict:
    """Re-prefix ``quantization_config`` keys in a config.json dict.

    ``block_name_to_quantize`` entries are exact string matches
    (e.g. ``"model.layers"`` <-> ``"model.language_model.layers"``).
    ``extra_config`` keys are prefix-matched with a trailing dot so that
    ``model.layers.*`` never matches inside ``model.language_model.layers.*``.

    Mutates and returns *cfg*.
    """
    qc = cfg.get("quantization_config")
    if qc is None:
        return cfg

    if direction == "strip":
        old_bn, new_bn = "model.language_model.layers", "model.layers"
        old_ec, new_ec = "model.language_model.layers.", "model.layers."
    else:
        old_bn, new_bn = "model.layers", "model.language_model.layers"
        old_ec, new_ec = "model.layers.", "model.language_model.layers."

    bnq = qc.get("block_name_to_quantize")
    if isinstance(bnq, list):
        qc["block_name_to_quantize"] = [
            new_bn if b == old_bn else b for b in bnq
        ]
    elif isinstance(bnq, str) and bnq == old_bn:
        qc["block_name_to_quantize"] = new_bn

    ec = qc.get("extra_config")
    if isinstance(ec, dict):
        qc["extra_config"] = {}
        for k, v in ec.items():
            if k.startswith(old_ec):
                qc["extra_config"][new_ec + k[len(old_ec):]] = v
            else:
                qc["extra_config"][k] = v

    return cfg
