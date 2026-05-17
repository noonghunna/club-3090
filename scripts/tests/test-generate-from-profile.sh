#!/usr/bin/env bash
set -euo pipefail

# test-generate-from-profile.sh — v0.8.0 [E] STEP E1 (club-3090 #141 / #147).
#
# Contract test for the ADDITIVE derived-emission path: EInput (CONTRACT-1),
# derived_emittable() (CONTRACT-5 gate), generate_from_profile()
# (CONTRACT-2 derived-vllm template + quant/dtype dispatch). The test is the
# spec; the code is fixed to it. NO live network / GPU / Docker — every
# EInput is a constructed fixture.
#
# Coverage:
#   e1      positive clean derived case: patchless default-vLLM compose;
#           container --model path + :ro mount; image from install.spec;
#           NVIDIA_VISIBLE_DEVICES set; kv_arg-mapped --kv-cache-dtype;
#           NO qwen parser / chat-template / reasoning-parser.
#   e1-neg  CONTRACT-5 negative matrix, each: NO compose emitted (Refuse) +
#           the EXACT structured reason token:
#             vllm/gemma-int8       -> overlay-feature
#             vllm/gemma-int8-tq3   -> kv
#             vllm/tools-text       -> drafter
#             clean tp2 on 1 GPU    -> gpu-count
#             a pip-install engine  -> engine-install-method
#             autoround weight_fmt  -> unsupported-quant-for-derived:autoround
#             quant + no torch_dtype-> unsupported-quant-for-derived:missing-torch-dtype
#   + generate("vllm/minimal") still emits (additive: registry path intact).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - "$ROOT_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
sys.path.insert(0, str(root))

from scripts.lib import generate_compose as gc  # noqa: E402
from scripts.lib.profiles.einput import EInput  # noqa: E402
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402

errors: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def mk_der(*, weight_format=None, torch_dtype=None, selected=None, slug="org/Model"):
    return SimpleNamespace(
        slug=slug,
        profile={
            "weight_format": weight_format,
            "torch_dtype": torch_dtype,
            "selected_weight_files": selected or [],
        },
    )


def mk_einput(profile_like, *, der, gpu_count=2, sel_gpus=(0, 1),
              slug="org/My-Model", trc=False):
    rt = dict(COMPOSE_REGISTRY[profile_like])
    return EInput(
        slug=slug,
        terminal="proceed",
        is_override_accepted=False,
        der=der,
        runtime=rt,
        selected_files=[],
        hf_home=Path("/data/hf"),
        c2a=None,
        hardware_sm=8.6,
        visible_gpu_count=gpu_count,
        per_gpu_vram_mib=[24576] * gpu_count,
        selected_gpu_indices=list(sel_gpus),
        selected_gpu_vram_mib=[24576] * len(sel_gpus),
        topology_summary="(RTX 3090, 24576)x2",
        club3090_commit="deadbeef",
        diagnostics={"_root": str(root), "trc_permitted": trc},
    )


# ==========================================================================
# e1 — positive clean derived case.
#   vllm/gemma-a4b-single: vllm-nightly-clean (required_genesis:false,
#   vendored_overlays:[], install.method:docker_image), kv bf16,
#   drafter None, required_engine_features []. tp=1 -> 1 GPU.
#   weight_format bfloat16 (pure dtype row -> --quantization omitted).
# ==========================================================================
der_clean = mk_der(weight_format="bfloat16", torch_dtype="bfloat16",
                    slug="org/My-Model")
ei = mk_einput("vllm/gemma-a4b-single", der=der_clean,
               gpu_count=1, sel_gpus=(0,), slug="org/My-Model")

ok, reason = gc.derived_emittable(ei)
check(ok and reason is None,
      f"e1: derived_emittable must pass clean case, got ({ok}, {reason})")

text, meta = gc.generate_from_profile(root, ei)

# resolved image == engine install.spec (NOT a bare ${VLLM_NIGHTLY_SHA}).
clean_engine = gc.load_engine(root, "vllm-nightly-clean")
spec = clean_engine["install"]["spec"]
check(f"image: {spec}" in text, "e1: image must be the resolved engine install.spec")
check("${VLLM_NIGHTLY_SHA}" not in text,
      "e1: derived compose must NOT contain a bare ${VLLM_NIGHTLY_SHA}")
check(meta["resolved_image"] == spec, "e1: meta.resolved_image must record the spec")

# --model is the CONTAINER path; the :ro mount carries host->container.
san = "org-my-model"
check(f"      - /models/club3090/pulls/{san}" in text,
      "e1: --model must be the container path")
check(f"      - /data/hf/club3090/pulls/{san}:/models/club3090/pulls/{san}:ro"
      in text, "e1: :ro volume mount host->container missing/incorrect")
check(meta["served_model_name"] == san,
      f"e1: served-model-name must be sanitized slug, got {meta['served_model_name']!r}")
check(f"      - {san}" in text, "e1: --served-model-name not the sanitized slug")

# NVIDIA_VISIBLE_DEVICES = exactly selected_gpu_indices.
check("      - NVIDIA_VISIBLE_DEVICES=0" in text,
      "e1: NVIDIA_VISIBLE_DEVICES must equal selected_gpu_indices")

# pure bfloat16 row: --quantization OMITTED, --dtype bfloat16.
check("--quantization" not in text,
      "e1: pure-dtype row must NOT emit --quantization")
check("      - --dtype" in text and "      - bfloat16" in text,
      "e1: --dtype bfloat16 expected for a pure bfloat16 derived model")
check(meta["quantization"] is None and meta["dtype"] == "bfloat16",
      "e1: meta quant/dtype mismatch for pure-bfloat16")

# kv bf16 -> kv_arg() returns None -> --kv-cache-dtype flag OMITTED.
check("--kv-cache-dtype" not in text,
      "e1: bf16 kv_format must omit --kv-cache-dtype (kv_arg emission-only)")
check(meta["kv_cache_dtype_arg"] is None, "e1: meta.kv_cache_dtype_arg must be None")

# PATCHLESS + NO curated qwen constants. The header documents what is
# deliberately NOT emitted (so those tokens legitimately appear ABOVE
# `services:`); the contract is they must not appear in the emitted
# SERVICE BODY. Split on the services: key.
svc_body = text.split("\nservices:", 1)[1]
for forbidden in ("--chat-template", "--reasoning-parser",
                  "--default-chat-template-kwargs", "--enable-auto-tool-choice",
                  "--tool-call-parser", "auto_round", "patches/"):
    check(forbidden not in svc_body,
          f"e1: derived service body must NOT contain {forbidden!r} (patchless default-vLLM)")
check(meta["patchless"] is True, "e1: meta.patchless must be True")
check("--trust-remote-code" not in text,
      "e1: --trust-remote-code must NOT be emitted when trc not resolved-permitted")

# A clean derived case that NEEDS a dtype (fp8 row + torch_dtype) maps a
# kv_arg-mapped --kv-cache-dtype (fp8_e5m2 -> fp8_e5m2).
der_fp8 = mk_der(weight_format="fp8", torch_dtype="bfloat16")
ei_fp8 = mk_einput("vllm/qwen-a3b-preview-single", der=der_fp8,
                   gpu_count=1, sel_gpus=(0,))
ok2, _ = gc.derived_emittable(ei_fp8)
check(ok2, "e1: fp8+torch_dtype clean single-GPU case must be emittable")
t2, m2 = gc.generate_from_profile(root, ei_fp8)
check("      - --quantization" in t2 and "      - fp8" in t2,
      "e1: fp8 weight_format must emit --quantization fp8")
check("      - --kv-cache-dtype" in t2 and "      - fp8_e5m2" in t2,
      "e1: fp8_e5m2 kv_format must emit --kv-cache-dtype fp8_e5m2 (kv_arg map)")
check(m2["dtype"] == "bfloat16", "e1: fp8 row --dtype from torch_dtype")

# trc resolved-permitted -> --trust-remote-code emitted.
ei_trc = mk_einput("vllm/gemma-a4b-single", der=der_clean,
                   gpu_count=1, sel_gpus=(0,), trc=True)
t3, m3 = gc.generate_from_profile(root, ei_trc)
check("      - --trust-remote-code" in t3 and m3["trc_emitted"] is True,
      "e1: --trust-remote-code MUST emit when [C0] trc gate resolved permitted")


# ==========================================================================
# e1-neg — CONTRACT-5 negative matrix. Each: Refuse, NO emit, exact token.
# ==========================================================================
def expect_refuse(ei, token: str, label: str) -> None:
    try:
        gc.generate_from_profile(root, ei)
        errors.append(f"{label}: expected Refuse[{token}], emitted a compose")
        return
    except gc.Refuse as r:
        check(str(r) == token or str(r).startswith(token),
              f"{label}: refuse {str(r)!r} != expected {token!r}")
    # derived_emittable() must independently refuse with the same token
    # (gate is pure; never a half-emit).
    ok, reason = gc.derived_emittable(ei)
    check(not ok and reason is not None and reason.startswith(token.split(':')[0])
          and reason == token,
          f"{label}: derived_emittable token {reason!r} != {token!r}")


# overlay-feature: vllm/gemma-int8 carries required_engine_features
# (compose_registry.py:258). It ALSO rides a Genesis engine with vendored
# overlays, so the property-driven gate short-circuits on the FIRST failing
# clause (engine-install-method). To isolate the overlay-feature token we
# point at a clean engine but keep gemma-int8's required_engine_features —
# proving the COMPOSE_REGISTRY runtime-entry clause fires independently.
ei_of = mk_einput("vllm/gemma-int8", der=mk_der(weight_format="bfloat16",
                                                torch_dtype="bfloat16"))
ei_of.runtime["engine"] = "vllm-nightly-clean"
ei_of.runtime["kv_format"] = "bf16"
expect_refuse(ei_of,
              "derived-runtime-unsupported:overlay-feature",
              "e1-neg/overlay-feature")

# kv: vllm/gemma-int8-tq3 uses turboquant_3bit_nc KV. It ALSO has
# required_engine_features + a Genesis engine; the gate short-circuits on
# the FIRST failing clause (engine-install-method), so to isolate the kv
# token we synthesize a runtime that is engine-clean but TQ3-KV.
ei_kv = mk_einput("vllm/gemma-a4b", der=mk_der(weight_format="bfloat16",
                                               torch_dtype="bfloat16"))
ei_kv.runtime["kv_format"] = "turboquant_3bit_nc"
expect_refuse(ei_kv, "derived-runtime-unsupported:kv", "e1-neg/kv")

# drafter: vllm/tools-text has drafter=qwen-mtp-builtin. Engine
# vllm-nightly-clean is clean, no required features, kv fp8_e5m2 -> the
# first failing clause is drafter.
expect_refuse(
    mk_einput("vllm/tools-text", der=mk_der(weight_format="bfloat16",
                                            torch_dtype="bfloat16"),
              gpu_count=1, sel_gpus=(0,)),
    "derived-runtime-unsupported:drafter", "e1-neg/drafter")

# gpu-count: a clean no-drafter TP=2 shape on a simulated 1-GPU einput.
expect_refuse(
    mk_einput("vllm/gemma-a4b", der=mk_der(weight_format="bfloat16",
                                           torch_dtype="bfloat16"),
              gpu_count=1, sel_gpus=(0,)),
    "derived-runtime-unsupported:gpu-count", "e1-neg/gpu-count")

# engine-install-method: a pip-install engine (vllm-stable, install.method
# pip). No registry entry uses it; synthesize the runtime to point at it.
ei_pip = mk_einput("vllm/gemma-a4b-single",
                   der=mk_der(weight_format="bfloat16", torch_dtype="bfloat16"),
                   gpu_count=1, sel_gpus=(0,))
ei_pip.runtime["engine"] = "vllm-stable"
expect_refuse(ei_pip, "derived-runtime-unsupported:engine-install-method",
              "e1-neg/engine-install-method")

# unsupported-quant-for-derived:autoround — clean shape, weight_format
# autoround (explicit CONTRACT-5 reject).
expect_refuse(
    mk_einput("vllm/gemma-a4b-single",
              der=mk_der(weight_format="autoround", torch_dtype="bfloat16"),
              gpu_count=1, sel_gpus=(0,)),
    "derived-runtime-unsupported:unsupported-quant-for-derived:autoround",
    "e1-neg/autoround")

# unsupported-quant-for-derived:missing-torch-dtype — a quantized row
# (awq) with NO torch_dtype and no callable header probe -> fail-closed.
expect_refuse(
    mk_einput("vllm/gemma-a4b-single",
              der=mk_der(weight_format="awq", torch_dtype=None, selected=[]),
              gpu_count=1, sel_gpus=(0,)),
    "derived-runtime-unsupported:unsupported-quant-for-derived:missing-torch-dtype",
    "e1-neg/missing-torch-dtype")

# unsupported-quant-for-derived (bare) — weight_format not in the dispatch
# table at all (never guess a --quantization).
expect_refuse(
    mk_einput("vllm/gemma-a4b-single",
              der=mk_der(weight_format="some-exotic-quant",
                         torch_dtype="bfloat16"),
              gpu_count=1, sel_gpus=(0,)),
    "derived-runtime-unsupported:unsupported-quant-for-derived",
    "e1-neg/unsupported-quant")


# ==========================================================================
# Additive invariant — the registry-key generate() path still emits.
# ==========================================================================
mtext, mmeta = gc.generate(root, "vllm/minimal")
check("GENERATED by scripts/generate-compose.sh" in mtext,
      "additive: generate('vllm/minimal') must still emit the [D] compose")
check(mmeta["profile"] == "vllm/minimal",
      "additive: generate() meta must be unchanged shape")


# ==========================================================================
if errors:
    print("[generate-from-profile] FAIL")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("[generate-from-profile] PASS: e1 positive (patchless default-vLLM, "
      "container --model + :ro mount, install.spec image, NVIDIA_VISIBLE_DEVICES, "
      "kv_arg-mapped --kv-cache-dtype, no qwen constants), e1-neg CONTRACT-5 "
      "matrix (overlay-feature/kv/drafter/gpu-count/engine-install-method/"
      "autoround/missing-torch-dtype) all structured-refuse with no emit, "
      "generate() registry path intact")
PY
