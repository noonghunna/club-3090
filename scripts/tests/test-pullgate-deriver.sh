#!/usr/bin/env bash
# v0.8.0 Pull-Gate P2 — [A] transformers deriver + §4 ModelProfile/confidence
# + variant-scoped hf_repos schema.
#
# Recorded fixtures only — NO live network. A FixtureFetcher replays saved
# config.json / HF-siblings-API / safetensors-header byte responses and
# asserts the header probe stays range-bounded (never a full file fetch).
#
# Asserts:
#   1. deriver happy path (Llama-class dense fixture) -> correct spec fields
#      + confidence estimated-lower-bound.
#   2. Tier-1: a slug in a curated variant's hf_repos -> exact (model,variant)
#      + confidence exact.
#   3. each stratum-1 structured error path returns the structured error
#      (no traceback, no full-file download):
#      repo-not-found / gated-no-token / unsupported-format /
#      ambiguous-weight-set / quant-dtype-unknown (N>16MiB AND malformed).
#   4. header-probe byte-range correctness: requests bytes=8-(8+N-1),
#      reads exactly N.
#   5. schema: all hf_repos slugs globally unique; no gguf variant carries
#      hf_repos; load_profiles() actually surfaces hf_repos (regression vs
#      the old drop-unknown-keys behaviour).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

# Deterministic: the "gated, no token" path must see NO token regardless of
# the host environment ($HF_TOKEN may be set on dev rigs).
os.environ.pop("HF_TOKEN", None)

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib.profiles import deriver as D  # noqa: E402
from scripts.lib.profiles.compat import load_profiles  # noqa: E402

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures.append(msg)


# ---------------------------------------------------------------------------
# FixtureFetcher: replays recorded responses; records every requested Range
# so tests can assert the header probe never fetches a whole weight file.
# ---------------------------------------------------------------------------
def st_header_bytes(header_obj: dict) -> bytes:
    blob = json.dumps(header_obj).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob


class FixtureFetcher:
    def __init__(self, routes: dict):
        # routes: url -> dict | bytes | callable(range_)->FetchResponse
        self.routes = routes
        self.calls: list[tuple[str, object]] = []

    def get(self, url, headers=None, range_=None):
        self.calls.append((url, range_))
        if url not in self.routes:
            return D.FetchResponse(status=404, body=b"")
        spec = self.routes[url]
        if callable(spec):
            return spec(range_)
        if isinstance(spec, D.FetchResponse):
            if range_ is not None and spec.status == 200:
                lo, hi = range_
                return D.FetchResponse(status=206, body=spec.body[lo : hi + 1])
            return spec
        if isinstance(spec, dict):
            body = json.dumps(spec).encode("utf-8")
            return D.FetchResponse(status=200, body=body)
        if isinstance(spec, bytes):
            if range_ is not None:
                lo, hi = range_
                return D.FetchResponse(status=206, body=spec[lo : hi + 1])
            return D.FetchResponse(status=200, body=spec)
        raise AssertionError(f"bad fixture spec for {url}")


CFG = f"{D._HF_RESOLVE}/{{slug}}/resolve/main/config.json"
API = f"{D._HF_API}/{{slug}}?blobs=true"
WF = f"{D._HF_RESOLVE}/{{slug}}/resolve/main/{{f}}"

profiles = load_profiles()


# ---------------------------------------------------------------------------
# 1. Happy path — Llama-class dense fixture, head_dim derived
# ---------------------------------------------------------------------------
DENSE_CFG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "torch_dtype": "bfloat16",
}
DENSE_API = {
    "siblings": [
        {"rfilename": "config.json", "size": 700},
        {"rfilename": "tokenizer.json", "size": 1_800_000},
        {"rfilename": "model-00001-of-00002.safetensors", "size": 7_000_000_000},
        {"rfilename": "model-00002-of-00002.safetensors", "size": 6_500_000_000},
        {"rfilename": "model.safetensors.index.json", "size": 40_000},
    ]
}
slug = "fixtures/llama-dense-7b"
f = FixtureFetcher(
    {
        CFG.format(slug=slug): DENSE_CFG,
        API.format(slug=slug): DENSE_API,
    }
)
res = D.derive(slug, fetcher=f, profiles=profiles)
check(res.error is None, f"happy path: no stratum-1 error (got {res.error})")
check(
    res.confidence == D.Confidence.ESTIMATED_LOWER_BOUND,
    f"happy path: confidence estimated-lower-bound (got {res.confidence})",
)
check(res.generic_dense_eligible is True, "happy path: generic-dense eligible")
sp = res.spec or {}
check(sp.get("hidden_size") == 4096, "happy path spec.hidden_size == 4096")
check(sp.get("num_hidden_layers") == 32, "happy path spec.num_hidden_layers == 32")
check(sp.get("num_attn_heads") == 32, "happy path spec.num_attn_heads == 32")
check(sp.get("num_kv_heads") == 8, "happy path spec.num_kv_heads == 8")
check(
    sp.get("head_dim_attn") == 128,
    f"happy path spec.head_dim_attn derived == 128 (got {sp.get('head_dim_attn')})",
)
check(
    abs(sp.get("weights_total_gb", 0) - round(13_500_000_000 / (1024 ** 3), 4)) < 1e-3,
    f"happy path spec.weights_total_gb summed selected blobs "
    f"(got {sp.get('weights_total_gb')})",
)
check(
    sp.get("model_family") == "generic-dense",
    "happy path spec.model_family == generic-dense",
)
check(
    (res.profile or {}).get("arch") == "LlamaForCausalLM",
    "happy path profile.arch == LlamaForCausalLM",
)
# torch_dtype path used -> NO weight-file fetch at all.
check(
    not any(".safetensors" in c[0] and "/resolve/main/" in c[0] for c in f.calls),
    "happy path: no weight-file fetch (torch_dtype short-circuits probe)",
)

# ---------------------------------------------------------------------------
# 2. Tier-1 curated lookup -> exact
# ---------------------------------------------------------------------------
tier1_slug = "Lorbus/Qwen3.6-27B-int4-AutoRound"
f2 = FixtureFetcher({})  # must NOT need network for a curated hit
res2 = D.derive(tier1_slug, fetcher=f2, profiles=profiles)
check(res2.error is None, f"tier1: no error (got {res2.error})")
check(res2.confidence == D.Confidence.EXACT, f"tier1: confidence exact (got {res2.confidence})")
check(
    res2.tier1 is not None
    and res2.tier1.model_id == "qwen3.6-27b"
    and res2.tier1.weights_variant == "autoround-int4",
    f"tier1: resolves to (qwen3.6-27b, autoround-int4) (got {res2.tier1})",
)
check(len(f2.calls) == 0, "tier1: zero network calls for a curated hit")
# Case-insensitive match.
res2b = D.derive(tier1_slug.lower(), fetcher=FixtureFetcher({}), profiles=profiles)
check(
    res2b.tier1 is not None and res2b.tier1.model_id == "qwen3.6-27b",
    "tier1: slug matched case-insensitively",
)

# ---------------------------------------------------------------------------
# 3. stratum-1 structured errors
# ---------------------------------------------------------------------------
# repo-not-found (HF 404 on the model API — repo existence authority)
s = "fixtures/does-not-exist"
fnf = FixtureFetcher(
    {API.format(slug=s): D.FetchResponse(status=404, body=b"")}
)
rnf = D.derive(s, fetcher=fnf, profiles=profiles)
check(
    rnf.error is not None
    and rnf.error.kind == D.DeriverErrorKind.REPO_NOT_FOUND,
    f"stratum-1 repo-not-found on HF 404 (got {rnf.error})",
)

# gated-no-token (HF 401, no token)
s = "fixtures/gated-model"
fg = FixtureFetcher(
    {API.format(slug=s): D.FetchResponse(status=401, body=b"")}
)
rg = D.derive(s, fetcher=fg, profiles=profiles)  # no hf_token
check(
    rg.error is not None
    and rg.error.kind == D.DeriverErrorKind.GATED_NO_TOKEN,
    f"stratum-1 gated-no-token on HF 401 w/o token (got {rg.error})",
)

# unsupported-format (no *.safetensors — GGUF-only repo)
s = "fixtures/gguf-only"
GGUF_API = {
    "siblings": [
        {"rfilename": "config.json", "size": 700},
        {"rfilename": "model-Q4_K_M.gguf", "size": 9_000_000_000},
    ]
}
fu = FixtureFetcher(
    {
        CFG.format(slug=s): {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        API.format(slug=s): GGUF_API,
    }
)
ru = D.derive(s, fetcher=fu, profiles=profiles)
check(
    ru.error is not None
    and ru.error.kind == D.DeriverErrorKind.UNSUPPORTED_FORMAT,
    f"stratum-1 unsupported-format (no safetensors) (got {ru.error})",
)
check(
    not any(c[0].endswith(".gguf") for c in fu.calls),
    "unsupported-format: never fetched the .gguf weight file",
)

# ambiguous-weight-set (two complete top-level safetensors, no index)
s = "fixtures/ambiguous"
AMB_API = {
    "siblings": [
        {"rfilename": "config.json", "size": 700},
        {"rfilename": "pytorch_a.safetensors", "size": 5_000_000_000},
        {"rfilename": "alt_b.safetensors", "size": 5_000_000_000},
    ]
}
fa = FixtureFetcher(
    {
        CFG.format(slug=s): {"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
        API.format(slug=s): AMB_API,
    }
)
ra = D.derive(s, fetcher=fa, profiles=profiles)
check(
    ra.error is not None
    and ra.error.kind == D.DeriverErrorKind.AMBIGUOUS_WEIGHT_SET,
    f"stratum-1 ambiguous-weight-set (got {ra.error})",
)

# quant-dtype-unknown (header N > 16 MiB)
s = "fixtures/huge-header"
HUGE_N = D._MAX_HEADER_BYTES + 1
huge_first8 = struct.pack("<Q", HUGE_N)
NOQ_CFG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}  # NO quantization_config, NO torch_dtype
ONE_SF_API = {
    "siblings": [
        {"rfilename": "config.json", "size": 700},
        {"rfilename": "model.safetensors", "size": 13_000_000_000},
    ]
}
fh = FixtureFetcher(
    {
        CFG.format(slug=s): NOQ_CFG,
        API.format(slug=s): ONE_SF_API,
        WF.format(slug=s, f="model.safetensors"): huge_first8,
    }
)
rh = D.derive(s, fetcher=fh, profiles=profiles)
check(
    rh.error is not None
    and rh.error.kind == D.DeriverErrorKind.QUANT_DTYPE_UNKNOWN,
    f"stratum-1 quant-dtype-unknown (header N>16MiB) (got {rh.error})",
)
# Bounded: only the first 8 bytes were ever requested for the weight file.
wf_calls = [
    c for c in fh.calls if c[0] == WF.format(slug=s, f="model.safetensors")
]
check(
    all(c[1] == (0, 7) for c in wf_calls) and len(wf_calls) >= 1,
    f"quant-dtype-unknown (N>16MiB): only bytes=0-7 fetched, no full file "
    f"(weight-file ranges={[c[1] for c in wf_calls]})",
)

# quant-dtype-unknown (malformed header JSON)
s = "fixtures/malformed-header"
bad_json = b"{ this is : not json "
mal_blob = struct.pack("<Q", len(bad_json)) + bad_json
fm = FixtureFetcher(
    {
        CFG.format(slug=s): NOQ_CFG,
        API.format(slug=s): ONE_SF_API,
        WF.format(slug=s, f="model.safetensors"): mal_blob,
    }
)
rm = D.derive(s, fetcher=fm, profiles=profiles)
check(
    rm.error is not None
    and rm.error.kind == D.DeriverErrorKind.QUANT_DTYPE_UNKNOWN,
    f"stratum-1 quant-dtype-unknown (malformed header) (got {rm.error})",
)

# ---------------------------------------------------------------------------
# 4. header-probe byte-range correctness: bytes=8-(8+N-1), exactly N bytes
# ---------------------------------------------------------------------------
s = "fixtures/probe-range"
hdr_obj = {"__metadata__": {"dtype": "F8_E5M2"}, "weight": {"dtype": "F8_E5M2"}}
hdr_blob = json.dumps(hdr_obj).encode("utf-8")
N = len(hdr_blob)
full = struct.pack("<Q", N) + hdr_blob + b"\x00" * 4096  # tail = "weight data"
fp = FixtureFetcher(
    {
        CFG.format(slug=s): NOQ_CFG,
        API.format(slug=s): ONE_SF_API,
        WF.format(slug=s, f="model.safetensors"): full,
    }
)
rp = D.derive(s, fetcher=fp, profiles=profiles)
wf_url = WF.format(slug=s, f="model.safetensors")
ranges = [c[1] for c in fp.calls if c[0] == wf_url]
check(
    rp.error is None and rp.generic_dense_eligible is True,
    f"probe-range: header probe succeeded via fp8 dtype (err={rp.error})",
)
check(
    ranges == [(0, 7), (8, 8 + N - 1)],
    f"probe-range: requested exactly [(0,7),(8,{8 + N - 1})] "
    f"(== bytes=8-(8+N-1)); got {ranges}",
)
check(
    (rp.profile or {}).get("effective_bpw") == 8.0,
    f"probe-range: effective_bpw resolved to 8.0 from fp8 header "
    f"(got {(rp.profile or {}).get('effective_bpw')})",
)
# Never requested the weight tail (no unbounded / full-file range).
check(
    all(r is not None and r[1] - r[0] + 1 <= max(8, N) for r in ranges),
    f"probe-range: no full-file fetch — every range bounded (got {ranges})",
)

# ---------------------------------------------------------------------------
# 5. schema invariants
# ---------------------------------------------------------------------------
seen: dict[str, str] = {}
dup = []
gguf_with_repos = []
surfaced = 0
for m in profiles.models.values():
    for variant, meta in m.weights.items():
        repos = meta.get("hf_repos", []) or []
        # regression vs old drop-unknown-keys: the key must exist (normalized)
        assert "hf_repos" in meta, f"{m.id}.{variant} missing normalized hf_repos"
        surfaced += len(repos)
        if repos and str(meta.get("format", "")).lower() == "gguf":
            gguf_with_repos.append(f"{m.id}.{variant}")
        for slug in repos:
            k = slug.strip().lower()
            if k in seen:
                dup.append(f"{slug} on {seen[k]} and {m.id}.{variant}")
            seen[k] = f"{m.id}.{variant}"

check(not dup, f"schema: all hf_repos slugs globally unique (dups={dup})")
check(
    not gguf_with_repos,
    f"schema: no gguf/non-safetensors variant carries hf_repos "
    f"(offenders={gguf_with_repos})",
)
check(
    surfaced > 0,
    f"schema: load_profiles() surfaces hf_repos (regression vs "
    f"drop-unknown-keys) — {surfaced} slugs visible",
)
# hf_repos_for / all_hf_repos accessors work and a known curated slug is present.
qm = profiles.models["qwen3.6-27b"]
check(
    "Lorbus/Qwen3.6-27B-int4-AutoRound" in qm.hf_repos_for("autoround-int4"),
    "schema: ModelProfile.hf_repos_for('autoround-int4') exposes Lorbus slug",
)
check(
    qm.hf_repos_for("gguf") == (),
    "schema: gguf variant hf_repos_for() is empty tuple",
)
check(
    isinstance(qm.all_hf_repos(), dict)
    and qm.all_hf_repos()["autoround-int4"],
    "schema: ModelProfile.all_hf_repos() returns per-variant map",
)

# Curated slug that (hypothetically) resolves to a gguf variant -> the
# deriver surfaces stratum-1 unsupported-format honestly. We exercise the
# code path directly via a synthetic profiles object would be heavy; instead
# assert the invariant holds in the real catalog (no gguf hf_repos) AND that
# derive() returns unsupported-format if a tier1 hit's variant were gguf.
# (Real catalog has none, so this is covered by the gguf_with_repos check
# above + the deriver's explicit guard.)

# compressed-tensors config_groups bit-width resolution (the Agents-A1
# FP8-dynamic producer-zero dogfood finding, 2026-07-02): bits live nested at
# config_groups.<g>.weights.num_bits, NOT top-level — resolve_quant_dtype must
# read them (previously -> quant-dtype-unknown for every llm-compressor
# checkpoint: FP8-dynamic, INT8 W8A8, INT4 pack-quantized).
CT_FP8_CFG = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "config_groups": {
        "group_0": {
            "weights": {"num_bits": 8, "type": "float", "strategy": "channel", "dynamic": False},
            "input_activations": {"num_bits": 8, "type": "float", "dynamic": True},
            "targets": ["Linear"],
        }
    },
}
wf, bpw, err = D.resolve_quant_dtype("fixtures/ct-fp8", {"quantization_config": CT_FP8_CFG}, [], None, None)
check(
    err is None and wf == "compressed-tensors" and bpw == 8.0,
    f"compressed-tensors FP8-dynamic config_groups -> bpw 8.0 (got wf={wf} bpw={bpw} err={err})",
)
CT_INT4_CFG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {"weights": {"num_bits": 4, "type": "int", "group_size": 128}}
    },
}
wf4, bpw4, err4 = D.resolve_quant_dtype("fixtures/ct-int4", {"quantization_config": CT_INT4_CFG}, [], None, None)
check(
    err4 is None and bpw4 == 4.0,
    f"compressed-tensors INT4 pack-quantized config_groups -> bpw 4.0 (got bpw={bpw4} err={err4})",
)
# mixed groups: the WIDEST weights num_bits wins (footprint-dominant).
CT_MIXED_CFG = {
    "quant_method": "compressed-tensors",
    "config_groups": {
        "g0": {"weights": {"num_bits": 4, "type": "int"}},
        "g1": {"weights": {"num_bits": 8, "type": "float"}},
    },
}
_, bpwm, errm = D.resolve_quant_dtype("fixtures/ct-mixed", {"quantization_config": CT_MIXED_CFG}, [], None, None)
check(
    errm is None and bpwm == 8.0,
    f"compressed-tensors mixed groups -> widest bpw 8.0 (got bpw={bpwm} err={errm})",
)

if failures:
    print(f"\n{len(failures)} assertion(s) failed.", file=sys.stderr)
    sys.exit(1)
print("\nAll Pull-Gate deriver assertions passed.")
PY

echo "test-pullgate-deriver.sh OK"
