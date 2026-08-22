#!/usr/bin/env bash
# P3 — GGUF header KV probe in the deriver (ModelSpec proposal §3 option 1).
#
# A minimal synthetic GGUF (header + KV pairs + one tokenizer array) is built
# IN-TEST from stdlib struct — no real GGUF file, NO live network. The remote
# range path is a recorded FixtureFetcher that records every requested Range
# and asserts the probe stays bounded (one range-GET of the header budget,
# never a full file fetch).
#
# Asserts:
#   1. local .gguf parse → header magic/version/tensor-count + KV pairs.
#   2. mapping → the deriver spec-facts shape (block_count→num_hidden_layers,
#      embedding_length→hidden_size, …) with provenance gguf-header +
#      confidence estimated-lower-bound (docs/PULL.md tier language).
#   3. GGUF MHA convention: absent attention.head_count_kv ⇒ kv_heads == heads
#      (lossless, not a guess) and the assumption is RECORDED.
#   4. head_dim: attention.key_length wins; else hidden/heads when divisible.
#   5. remote bounded probe: exactly ONE range request bytes=0-(budget-1);
#      truncated input still maps the arch KVs that arrived; `truncated` set.
#   6. honest failures: non-GGUF magic → None; GGUF v1 → None; missing
#      general.architecture → None; 404/NetworkError → None.
#   7. general.file_type → quant label (15 → Q4_K_M); unknown → None.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

# Deterministic: probes must see NO token regardless of the host environment.
os.environ.pop("HF_TOKEN", None)

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import deriver as D  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# Synthetic GGUF builder (v2/v3 little-endian, stdlib struct only)
# ---------------------------------------------------------------------------
def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def build_gguf(kv: dict, version: int = 3, tensor_count: int = 291) -> bytes:
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kv))
    for key, (vt, val) in kv.items():
        out += _s(key) + struct.pack("<I", vt)
        if vt == 8:
            out += _s(val)
        elif vt == 9:  # array of strings (the tokenizer shape)
            out += struct.pack("<I", 8) + struct.pack("<Q", len(val))
            for item in val:
                out += _s(item)
        else:
            out += struct.pack(D._GGUF_SCALAR_FMTS[vt], val)
    return out


ARCH_KV = {
    "general.architecture": (8, "llama"),
    "general.name": (8, "Synth-7B"),
    "general.file_type": (4, 15),              # LLAMA_FTYPE_MOSTLY_Q4_K_M
    "general.quantization_version": (10, 2),
    "llama.block_count": (4, 32),
    "llama.embedding_length": (10, 4096),
    "llama.attention.head_count": (10, 32),
    "llama.attention.head_count_kv": (10, 8),
    "llama.attention.key_length": (10, 128),
    "llama.context_length": (10, 131072),
}
# A converter-shaped repo: the big vocab array comes AFTER the arch keys.
TOKENIZER_KV = {"tokenizer.ggml.tokens": (9, ["token-" + "x" * 16] * 4096)}

GGUF_BLOB = build_gguf({**ARCH_KV, **TOKENIZER_KV})


# ---------------------------------------------------------------------------
# 1. local .gguf parse — header fields + every KV pair
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "Synth-7B-Q4_K_M.gguf"
    p.write_bytes(GGUF_BLOB)

    h = D.read_gguf_header(str(p))
    check(h is not None and not h["truncated"], "local: full header parses untruncated")
    check(h["version"] == 3 and h["tensor_count"] == 291,
          "local: magic/version/tensor-count read")
    check(h["kv_count"] == len(ARCH_KV) + len(TOKENIZER_KV), "local: kv_count matches")
    check(h["kv"].get("general.architecture") == "llama"
          and h["kv"].get("llama.block_count") == 32
          and h["kv"].get("llama.attention.head_count_kv") == 8,
          "local: KV pairs extracted past the tokenizer array")
    check(isinstance(h["kv"].get("tokenizer.ggml.tokens"), list)
          and len(h["kv"]["tokenizer.ggml.tokens"]) == 4096,
          "local: string arrays parse")

    facts = D.gguf_facts_from_file(str(p), model_id="org/Synth-7B-GGUF", weight_gb=4.2)

check(facts["model_id"] == "org/Synth-7B-GGUF", "map: model_id passthrough")
check(facts["arch"] == "llama", "map: general.architecture → arch")
check(facts["hidden_size"] == 4096, "map: embedding_length → hidden_size")
check(facts["num_hidden_layers"] == 32, "map: block_count → num_hidden_layers")
check(facts["num_attn_heads"] == 32, "map: head_count → num_attn_heads")
check(facts["num_kv_heads"] == 8, "map: head_count_kv → num_kv_heads")
check(facts["head_dim_attn"] == 128, "map: attention.key_length → head_dim_attn")
check(facts["max_ctx_supported"] == 131072, "map: context_length → max_ctx_supported")
check(facts["weights_total_gb"] == 4.2, "map: weight_gb passthrough")
check(facts["valid_tp"] == [1, 2], "map: policy valid_tp default")
check(facts["model_family"] is None, "map: family stays None (no config.json — never fabricated)")
check(facts["confidence"] == "estimated-lower-bound",
      "map: confidence matches docs/PULL.md tier (derived ≠ curated exact)")
check(facts["facts_provenance"] == "gguf-header", "map: provenance marked gguf-header")
check(facts["gguf"]["general_name"] == "Synth-7B", "map: general.name extracted")
check(facts["gguf"]["quant_label"] == "Q4_K_M" and facts["gguf"]["file_type"] == 15,
      "map: general.file_type 15 → Q4_K_M quant label")
check(facts["gguf"]["quantization_version"] == 2, "map: quantization_version extracted")
check(facts["gguf"]["kv_heads_assumed_equal"] is False,
      "map: explicit head_count_kv NOT flagged as assumed")

# ---------------------------------------------------------------------------
# 2. MHA convention — head_count_kv omitted ⇒ kv_heads == heads (recorded)
# ---------------------------------------------------------------------------
mha = D.gguf_spec_facts(
    {"version": 3, "truncated": False,
     "kv": {"general.architecture": "llama",
            "llama.embedding_length": 4096, "llama.block_count": 32,
            "llama.attention.head_count": 32}},
    model_id="m",
)
check(mha["num_kv_heads"] == 32 and mha["gguf"]["kv_heads_assumed_equal"] is True,
      "map: absent head_count_kv ⇒ kv_heads == heads, assumption recorded")

# head_dim derived when key_length absent and hidden divisible by heads
mha_nokd = D.gguf_spec_facts(
    {"version": 3, "truncated": False,
     "kv": {"general.architecture": "llama",
            "llama.embedding_length": 4096, "llama.block_count": 32,
            "llama.attention.head_count": 32}},
)
check(mha_nokd["head_dim_attn"] == 128, "map: head_dim_attn derived hidden//heads")

# ---------------------------------------------------------------------------
# 3. remote bounded probe — ONE range request, truncated-but-sufficient
# ---------------------------------------------------------------------------
class RangeFetcher:
    """Records every requested Range; serves slices of the recorded blob."""

    def __init__(self, blob: bytes, *, fail_after: int = 0):
        self.blob = blob
        self.calls: list[tuple] = []
        self._fail_after = fail_after

    def get(self, url, headers=None, range_=None):
        self.calls.append((url, range_))
        if self._fail_after and len(self.calls) > self._fail_after:
            raise D.NetworkError("boom")
        if range_ is None:
            return D.FetchResponse(status=200, body=self.blob)  # full fetch = BUG
        lo, hi = range_
        return D.FetchResponse(status=206, body=self.blob[lo:hi + 1])


URL = f"{D._HF_RESOLVE}/org/Synth-7B-GGUF/resolve/main/Synth-7B-Q4_K_M.gguf"

# 3a. full blob within budget → complete parse
f_full = RangeFetcher(GGUF_BLOB)
h = D.probe_gguf_header("org/Synth-7B-GGUF", "Synth-7B-Q4_K_M.gguf", f_full, None)
check(h is not None and not h["truncated"], "remote: full header within budget parses")
check(f_full.calls == [(URL, (0, D._GGUF_REMOTE_PROBE_BYTES - 1))],
      "remote: EXACTLY ONE bounded range-GET bytes=0-(budget-1), never a full fetch")
facts_r = D.gguf_spec_facts(h, model_id="org/Synth-7B-GGUF", weight_gb=4.2)
check(facts_r["num_hidden_layers"] == 32 and facts_r["num_kv_heads"] == 8,
      "remote: arch KVs map identically to the local path")

# ---------------------------------------------------------------------------
# 2b. per-layer head-count ARRAYS (real-world hybrid attention, e.g. laguna):
#     uniform array collapses losslessly; variable array → None + flag.
# ---------------------------------------------------------------------------
uniform = D.gguf_spec_facts(
    {"version": 3, "kv": {"general.architecture": "llama",
                          "llama.block_count": 4,
                          "llama.attention.head_count": [32, 32, 32, 32],
                          "llama.attention.head_count_kv": 8}},
)
check(uniform["num_attn_heads"] == 32
      and uniform["gguf"]["head_count_variable"] is False,
      "array: UNIFORM per-layer head_count collapses losslessly")
variable = D.gguf_spec_facts(
    {"version": 3, "kv": {"general.architecture": "laguna",
                          "laguna.block_count": 48,
                          "laguna.embedding_length": 3072,
                          "laguna.attention.head_count": [48, 72] * 24,
                          "laguna.attention.key_length": 128,
                          "laguna.context_length": 1048576}},
)
check(variable["num_attn_heads"] is None
      and variable["gguf"]["head_count_variable"] is True,
      "array: VARIABLE per-layer head_count → None + flag (never averaged)")
check(variable["num_hidden_layers"] == 48
      and variable["head_dim_attn"] == 128,
      "array: scalar dims around a variable array still map")

# ---------------------------------------------------------------------------
# 4. honest failures
# ---------------------------------------------------------------------------
check(D.parse_gguf_header(lambda n: b"NOPE" + b"\x00" * 64) is None,
      "fail: non-GGUF magic → None")
check(D.parse_gguf_header(lambda n: b"GGUF" + struct.pack("<I", 1)) is None,
      "fail: GGUF v1 → None (v2/v3 only)")
check(D.gguf_spec_facts({"version": 3, "kv": {"llama.block_count": 32}}) is None,
      "fail: missing general.architecture → None (dims are arch-keyed)")
check(D.gguf_spec_facts({"version": 3, "kv": {}}) is None,
      "fail: empty KV → None")
f404 = RangeFetcher(b"")
f404.blob = b""  # 404-ish empty body
check(D.probe_gguf_header("org/X", "x.gguf", f404, None) is None,
      "fail: empty remote body → None")


class BoomFetcher:
    def get(self, url, headers=None, range_=None):
        raise D.NetworkError("timeout")


check(D.probe_gguf_header("org/X", "x.gguf", BoomFetcher(), None) is None,
      "fail: NetworkError → None (structured, never a traceback)")
check(D.read_gguf_header("/nonexistent/path.gguf") is None,
      "fail: missing local file → None")

# 3b. a tokenizer array big enough to push parsing PAST the budget —
#     the arch dims STILL map; truncation is reported, never hidden.
big_tok = build_gguf(
    {**ARCH_KV, "tokenizer.ggml.tokens": (9, ["token-" + "x" * 64] * 20000)}
)
f_cut = RangeFetcher(big_tok)
h2 = D.probe_gguf_header("org/Synth-7B-GGUF", "Synth-7B-Q4_K_M.gguf", f_cut, None)
check(h2 is not None and h2["truncated"], "remote: over-budget header reports truncated")
check(len(f_cut.calls) == 1, "remote: truncation still costs exactly ONE range-GET")
facts_c = D.gguf_spec_facts(h2)
check(facts_c is not None and facts_c["hidden_size"] == 4096
      and facts_c["num_hidden_layers"] == 32
      and facts_c["gguf"]["truncated"] is True,
      "remote: arch KVs that arrived before the cut still map + truncation surfaced")

# unknown file_type → quant_label None (basename regex is the caller fallback)
check(D.gguf_spec_facts(
    {"version": 3, "kv": {"general.architecture": "llama",
                          "general.file_type": 999}})["gguf"]["quant_label"] is None,
      "map: unknown general.file_type → quant_label None (honest, no guess)")

# ---------------------------------------------------------------------------
if failures:
    print(f"\n{len(failures)} FAILURE(S)", file=sys.stderr)
    sys.exit(1)
print("\nAll GGUF-header assertions passed.")
PY
