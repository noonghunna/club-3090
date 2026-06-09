"""Static compose-to-profile bridge for v0.7.0.

The registry intentionally mirrors the shipped compose files. It is not a
generator and it does not attempt to normalize away historical variants.
"""

# Slug lifecycle / availability statuses — the canonical health flag.
#
# These are the registry-side equivalent of the compose `Status:` header enum
# (see the repo CLAUDE.md "Status enum" table). The compose-header emoji maps
# to one of these words; the drift-guard test asserts the two never diverge.
#
#   functional → launches normally (production) or with a one-line notice
#                (caveats).
#   (NA)       → surfaced in --list but not reliable: launch warns and requires
#                --force so a user can't *unknowingly* boot a broken slug.
STATUS_VALUES = (
    "production",      # ✅ Production — recommended, fully validated.
    "caveats",         # ⚠️ Production w/ caveats — works under documented limits.
    "experimental",    # 🧪 Experimental — under active validation; may not boot.
    "preview",         # 👁️ Preview — known quality issues; tracked, not for prod.
    "upstream-gated",  # ⏸️ Upstream-gated — blocked by external action (pin/PR/HW).
    "deprecated",      # 🗑️ Deprecated — kept for reference; flagged for removal.
)

# Statuses that launch without --force. Everything else is "(NA)".
FUNCTIONAL_STATUSES = frozenset({"production", "caveats"})

# Compose `Status:` header emoji → registry status word. The header may carry
# trailing prose after the canonical token (e.g. "✅ Production (NEW — ...)");
# matching is by the leading emoji, so prose is tolerated.
COMPOSE_STATUS_EMOJI = {
    "✅": "production",
    "⚠️": "caveats",
    "🧪": "experimental",
    "👁️": "preview",
    "⏸️": "upstream-gated",
    "🗑️": "deprecated",
}


def _entry(
    *,
    model,
    weights_variant,
    workload,
    engine,
    drafter,
    kv_format,
    tp,
    max_ctx,
    max_num_seqs,
    mem_util,
    compose_path,
    default_port,
    kvcalc_key=None,
    requires_nvlink=False,
    required_engine_features=None,
    recommended_engine_features=None,
    required_sm=None,
    status="production",
    status_note=None,
    category=None,
):
    if status not in STATUS_VALUES:
        raise ValueError(
            f"{compose_path}: status={status!r} not in {STATUS_VALUES}"
        )
    entry = {
        "model": model,
        "weights_variant": weights_variant,
        "workload": workload,
        "engine": engine,
        "drafter": drafter,
        "kv_format": kv_format,
        "tp": tp,
        "pp": 1,
        "max_ctx": max_ctx,
        "max_num_seqs": max_num_seqs,
        "mem_util": mem_util,
        "compose_path": compose_path,
        "requires_nvlink": requires_nvlink,
        "required_engine_features": list(required_engine_features or []),
        "default_port": default_port,
        "gpu_assignment_mode": "contiguous",
        "kvcalc_key": kvcalc_key,
        "status": status,
        "status_note": status_note,
    }
    if recommended_engine_features:
        entry["recommended_engine_features"] = list(recommended_engine_features)
    if required_sm is not None:
        entry["required_sm"] = required_sm
    if category is not None:
        entry["category"] = category
    return entry


def compose_header_status(text):
    """Map a compose file's profile-schema `Status:` header to a status word.

    Reads ONLY the `Status:` line inside the leading `# Profile (at-a-glance):`
    comment block (the structured schema), stopping at the `# ---` separator so
    a free-form `# Status: ...` prose line further down can't be mistaken for it.
    Returns the status word (one of STATUS_VALUES) or None if no canonical
    emoji is found. Matching is by the leading enum emoji, so trailing prose
    after the canonical token (e.g. "✅ Production (NEW — ...)") is tolerated.
    """
    in_schema = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# Profile (at-a-glance):"):
            in_schema = True
            continue
        if not in_schema:
            continue
        # The schema block ends at the dashed separator line.
        if stripped.startswith("# --") or stripped.startswith("#--"):
            break
        # Match "#   Status:    <emoji> ..." within the schema block.
        body = stripped.lstrip("#").strip()
        if body.startswith("Status:"):
            value = body[len("Status:"):].strip()
            for emoji, word in COMPOSE_STATUS_EMOJI.items():
                if value.startswith(emoji):
                    return word
            return None
    return None


COMPOSE_REGISTRY = {
    # Qwen 3.6 27B, vLLM single-card.
    "vllm/minimal": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=32768, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml",
        default_port=8020,
        kvcalc_key="qwen3.6-27b:minimal",
    ),

    # Qwen 3.6 27B, vLLM dual/multi-card.
    "vllm/dual": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
        default_port=8010,
        kvcalc_key="qwen3.6-27b:dual",
    ),

    # --- Qwen "fast" / "max accuracy" tiers (2026-06-07) -----------------------
    # A symmetric 4-slug family across dual (2-card) and multi4 (4-card):
    #   *-fast = AutoRound INT4 weights + fp8_e5m2 KV  (peak TPS, the proven path)
    #   *-max  = official FP8 weights   + int8-PTH KV  (higher fidelity @ 262K)
    # The 2-card duals are the on-rig validation proxies for the 4-card multi4s
    # (this dev rig has 2× 3090); the multi4 configs are byte-identical to their
    # dual sibling apart from TP and the gpu-count, so they ship 🧪 Experimental
    # until a real ≥4-card host validates them.
    #
    # `vllm/qwen-27b-dual-fast` is an explicit alias of `vllm/dual` (same compose,
    # same port) — it just names the fast tier in the symmetric family. The
    # (qwen,vllm,dual) DEFAULT stays "vllm/dual" (the long-established slug).
    "vllm/qwen-27b-dual-fast": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
        default_port=8010,
        kvcalc_key="qwen3.6-27b:dual",
        status_note="Alias of vllm/dual — names the 'fast' tier in the fast/max family (AutoRound INT4 + fp8_e5m2 KV + MTP n=3, TP=2 @262K). Same compose + port as vllm/dual; production-validated there (129/150). Pair with vllm/qwen-27b-dual-max for higher fidelity.",
    ),
    "vllm/qwen-27b-dual-max": _entry(
        model="qwen3.6-27b", weights_variant="fp8", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml",
        default_port=8013,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwen3.6-27B 'max accuracy' tier, 2-card: official FP8 weights (e4m3, embedded MTP head) + int8-PTH KV + MTP n=3, TP=2 @262K. 🧪 Experimental — live-validated 2026-06-07 (boots + serves @262K, KV pool 295K tok / 1.13x concurrency via int8-PTH, MarlinFP8 W8A16 on Ampere, coherent + MTP active; ~56 TPS decode). 8-pack A/B (--full, same harness 2026-06-07): 110/150 vs fast 109 vs balanced 105 — a TIE (det 65/64/64; spread within noise). The 8-pack (short-ctx) does NOT separate the quants; FP8 + int8-PTH differentiate on KV fidelity, not behavioral quality — a long-ctx NIAH A/B (where int8-PTH should matter) is the open follow-up. Slowest of the three (~56 vs fast ~89 code) with the smallest KV pool (1.13x). Also the validation proxy for vllm/qwen-27b-multi-max (same config @ TP=4).",
    ),
    "vllm/qwen-27b-dual-balanced": _entry(
        model="qwen3.6-27b", weights_variant="awq-bf16-int4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/awq-bf16-int4/int8.yml",
        default_port=8016,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwen3.6-27B 'balanced' tier, 2-card: cyankiwi AWQ-BF16-INT4 (int4 group-32 body + BF16 mtp head, compressed-tensors) + int8-PTH KV + MTP n=3, TP=2 @262K. 🧪 Experimental — live-validated 2026-06-07 (Marlin WNA16, KV pool 370K tok / 1.41x, ~67 TPS decode). 8-pack A/B (--full, same harness): 105/150 vs fast 109 vs max 110 — a TIE (balanced nominally lowest). ⚠️ HONEST STANDING: balanced is DOMINATED by the fast tier (vllm/dual) — slower (~67 vs ~89 code TPS), SMALLER KV pool (370K/1.41x vs fast's measured 622K/2.37x — the 27GB AWQ weights leave less KV room than fast's 17.5GB autoround), and tied/below on the 8-pack. Its ONLY possible edge is int8-PTH KV fidelity > fast's fp8_e5m2 (int8-PTH and fp8 are the SAME size — a fidelity bet, not a memory one), UNPROVEN since the short-ctx 8-pack is blind to it. Keep ONLY if the long-ctx NIAH A/B (#470) shows int8-PTH materially better high-ctx recall; else deprecate (fast wins on speed AND pool).",
    ),
    "vllm/qwen-27b-multi-fast": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=4, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/mtp.yml",
        default_port=8014,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwen3.6-27B 'fast' tier, 4-card (TP=4): AutoRound INT4 + fp8_e5m2 KV + MTP n=3 @262K. 🧪 Experimental (cross-rig) — byte-identical to vllm/dual (≡ vllm/qwen-27b-dual-fast) apart from TP=4 + gpu-count; vllm/dual @TP=2 is the on-rig validation proxy (this dev rig has 2× 3090). The extra cards buy ~2x aggregate KV headroom at 262K. Validate on a real ≥4× 3090 host before promotion.",
    ),
    "vllm/qwen-27b-multi-max": _entry(
        model="qwen3.6-27b", weights_variant="fp8", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=4, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/fp8/mtp.yml",
        default_port=8015,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwen3.6-27B 'max accuracy' tier, 4-card (TP=4): official FP8 weights + int8-PTH KV + MTP n=3 @262K. 🧪 Experimental (cross-rig) — byte-identical to vllm/qwen-27b-dual-max apart from TP=4 + gpu-count; the dual-max @TP=2 is the on-rig validation proxy. FP8 + int8-PTH = the highest-fidelity Qwen path at full 262K, with TP=4 relieving the dual-max's tight (1.13x) KV pool. Validate on a real ≥4× 3090 host before promotion.",
    ),

    # Qwen 3.6 27B, llama.cpp single-card.
    # `llamacpp/default` is an alias for `llamacpp/mtp` (collapsed 2026-05-22):
    # the old Q3_K_XL vanilla compose was retired and `default` now points at
    # the MTP compose. max_ctx = the 200K max-safe default (262K boots but walls
    # ~125K at fill — see docs/CLIFFS.md; runtime CTX_SIZE default is 200000).
    "llamacpp/default": _entry(
        model="qwen3.6-27b", weights_variant="unsloth-q4km", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),
    "llamacpp/mtp": _entry(
        model="qwen3.6-27b", weights_variant="unsloth-q4km", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),
    "llamacpp/bounded-thinking": _entry(
        model="qwen3.6-27b", weights_variant="unsloth-q4km", workload="tool-heavy",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/bounded-thinking.yml",
        default_port=8020,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="New structured-CoT port; live grammar + MTP validation pending.",
    ),
    "llamacpp/mtp-vision": _entry(
        model="qwen3.6-27b", weights_variant="unsloth-q4km", workload="vision-coding",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        # 150K @ 1M-px (IMAGE_MAX_TOKENS=1024) — re-tuned 2026-05-25 (PR #227); was a
        # stale 49152. Full-res 4M-px OOMs at fill, so 1M-px is the safe default.
        tp=1, max_ctx=150000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/unsloth-q4km/mtp-vision.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),

    # ik_llama.cpp — IQ4_KS (ubergarm). Same engine family as llamacpp, but the
    # IQK quant is ~0.5-0.8 GB leaner on weights → best fit for VRAM-tight
    # single-card (sub-24 GB, shared GPU, WSL display overhead). Its own image
    # (ikawrakow/ik-llama-cpp), so unaffected by mainline llama.cpp drift.
    "ik-llama/iq4ks-mtp": _entry(
        model="qwen3.6-27b", weights_variant="ubergarm-iq4ks", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),
    "ik-llama/iq4ks-mtp-vision": _entry(
        model="qwen3.6-27b", weights_variant="ubergarm-iq4ks", workload="vision-coding",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=163840, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/mtp-vision.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),
    "ik-llama/iq4ks-two-stage": _entry(
        model="qwen3.6-27b", weights_variant="ubergarm-iq4ks", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ubergarm-iq4ks/two-stage.yml",
        default_port=8020,
        kvcalc_key="SKIP",
    ),

    # Qwen3.6-27B beellama.cpp DFlash — single-card DEFAULT (DFlash spec-dec,
    # Q5_K_S target + Anbeeld DFlash-IQ4_XS draft, q5_0(K)/q4_1(V) KV). beellama
    # is a llama.cpp-family engine (kvcalc SKIP, like ik-llama). Promoted to the
    # single-GPU default 2026-05-30: code-throughput leader (~100 TPS) + slight
    # 8-pack quality edge (107 vs ik 99, think-off) + output-lossless spec-dec.
    # Served via our UNOFFICIAL multi-arch image (sm_86/89/120 = 3090/4090/5090);
    # sm_89/sm_120 are compiled but unvalidated on our 3090-only rig — see Caveats
    # in the compose. kv_format reflects K-side precision (V is q4_1).
    "beellama/dflash": _entry(
        model="qwen3.6-27b", weights_variant="beellama-q5ks-dflash", workload="fast-chat",
        engine="beellama-local", drafter="anbeeld-qwen-dflash", kv_format="q5_0",
        tp=1, max_ctx=102400, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/single/beellama-q5ks-dflash/dflash.yml",
        default_port=8060,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="Single-GPU default. Launchers inject Anbeeld's official beellama.cpp server-cuda-v0.3.0 image (sm_86/89 = 3090/4090); sm_89 compiled-not-validated on club-3090's 3090-only rig. 5090/sm_120: prefix BEELLAMA_IMAGE=ghcr.io/noonghunna/beellama-cpp:multiarch-v0.3.0-efe856397 (sm_120 compiled-not-validated). Usable ctx ceiling 160K (200K OOMs on prefill); ships 102K. DFlash prose is net-positive on tok/s (+27% vs no-spec, re-tested 2026-06-03); the earlier 'prose-DFlash regression' is RETRACTED — it was an AR over-read + wrong baseline (docs/UPSTREAM.md).",
    ),

    # Qwen3.6-27B PRISM-PRO-DQ (Ex0bit dynamic-quant GGUF) — community-experimental, ik-llama.
    "ik-llama/prism-pro-dq-mtp": _entry(
        model="qwen3.6-27b", weights_variant="ex0bit-prism-pro-dq", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=122880, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ex0bit-prism-pro-dq/mtp.yml",
        default_port=8020,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="PRISM-PRO-DQ community dynamic-quant GGUF — eval-only, not yet validated.",
    ),
    "ik-llama/prism-pro-dq-long": _entry(
        model="qwen3.6-27b", weights_variant="ex0bit-prism-pro-dq", workload="long-ctx-single",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=180000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ex0bit-prism-pro-dq/long.yml",
        default_port=8052,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="PRISM-PRO-DQ community dynamic-quant GGUF — eval-only, not yet validated.",
    ),
    "ik-llama/prism-pro-dq-two-stage": _entry(
        model="qwen3.6-27b", weights_variant="ex0bit-prism-pro-dq", workload="tool-heavy",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/single/ex0bit-prism-pro-dq/two-stage.yml",
        default_port=8020,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="PRISM-PRO-DQ community dynamic-quant GGUF — eval-only, not yet validated.",
    ),
    "ik-llama/prism-pro-dq-dual": _entry(
        model="qwen3.6-27b", weights_variant="ex0bit-prism-pro-dq", workload="tool-heavy",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=2, max_ctx=196608, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/dual/ex0bit-prism-pro-dq/mtp.yml",
        default_port=8053,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="PRISM-PRO-DQ community dynamic-quant GGUF — eval-only, not yet validated.",
    ),
    "ik-llama/prism-pro-dq-dual-vision": _entry(
        model="qwen3.6-27b", weights_variant="ex0bit-prism-pro-dq", workload="vision-coding",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/ik-llama/compose/dual/ex0bit-prism-pro-dq/mtp-vision.yml",
        default_port=8010,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="PRISM-PRO-DQ community dynamic-quant GGUF — eval-only, not yet validated.",
    ),
    # Qwen3.6-35B-A3B APEX-MTP (mudler MoE GGUF — Compact + Quality) — community-experimental, ik-llama.
    "ik-llama/apex-mtp-compact": _entry(
        model="qwen3.6-35b-a3b", weights_variant="mudler-apex-compact", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=163840, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/mtp.yml",
        default_port=8054,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="APEX-MTP community MoE GGUF — eval-only bring-up lane, not yet validated.",
    ),
    "ik-llama/byteshape-iq4xs-mtp": _entry(
        model="qwen3.6-35b-a3b", weights_variant="byteshape-iq4xs", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/ik-llama/compose/single/byteshape-iq4xs/mtp.yml",
        default_port=8058,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="byteshape IQ4_XS 4.19bpw MoE GGUF (embedded MTP head) — community intake from PR #293 (@Rhonstin). Single-card 35B-A3B, q4_0 KV + --fit → 262K. First-party validated 2026-06-02 on 1× 3090: verify-full all-pass, verify-stress 8/8 (NIAH→240K, no Cliff), bench n=5 (narrative 113/116 · code 129/137 wall/decode TPS, CV<2.3%), 8-pack 110/150 (≈ author's 111/150), soak-continuous PASS (0 err, 0 VRAM growth, 0/25 silent-empty). Caveat: single-rig; agent packs modest (hermes 55%, cli 42%) as typical for the class. Intake fixes vs #293: image cu13, port 8058.",
    ),
    "ik-llama/apex-mtp-compact-long": _entry(
        model="qwen3.6-35b-a3b", weights_variant="mudler-apex-compact", workload="long-ctx-single",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=1, max_ctx=196608, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/long.yml",
        default_port=8056,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="APEX-MTP community MoE GGUF — eval-only bring-up lane, not yet validated.",
    ),
    # @laurimyllari's --fit + asymmetric q8_0(K)/q5_0(V) KV config from
    # discussion #241, retuned + measured on 1× 3090. +7% narr / +4% code
    # over the q4/q4 mtp.yml sibling on the same APEX I-Compact GGUF.
    # kv_format "q8_0" reflects K-side precision; V is q5_0 (see compose).
    "ik-llama/apex-fit-q8q5": _entry(
        model="qwen3.6-35b-a3b", weights_variant="mudler-apex-compact", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=1, max_ctx=196608, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/fit-mtp.yml",
        default_port=8057,
        kvcalc_key="SKIP",
    ),
    "ik-llama/apex-mtp-quality-dual": _entry(
        model="qwen3.6-35b-a3b", weights_variant="mudler-apex-quality", workload="long-ctx-single",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=2, max_ctx=196608, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/ik-llama/compose/dual/mudler-apex-quality/mtp.yml",
        default_port=8055,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="APEX-MTP community MoE GGUF — eval-only bring-up lane, not yet validated.",
    ),

    # Gemma 4 31B, vLLM. Lean v0.21.0 set: bf16 default, int8 long-context, single-card fp8 risk path.
    "vllm/gemma-mtp-tp1": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="fp8_e4m3",
        tp=1, max_ctx=8192, max_num_seqs=256, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/single/autoround-int4/fp8-mtp.yml",
        default_port=8031, required_sm=9.0,
        kvcalc_key="gemma-4-31b:gemma-single",
        status="deprecated",
        status_note="Dead on Ampere: no fp8 KV path for Gemma 4 on sm_86 (attention asserts kv ∈ {fp8,fp8_e4m3,nvfp4} — rejects fp8_e5m2; fp8/fp8_e4m3 need the fp8e4nv kernel sm_86 lacks; nvfp4 Blackwell-only). Live-confirmed stock v0.22.0 2026-05-31. Single-card → beellama/gemma-dflash; dual → vllm/gemma-bf16-mtp. See compose Caveats.",
    ),
    "vllm/gemma-bf16-mtp": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="bf16",
        tp=2, max_ctx=131072, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml",
        default_port=8030,
        kvcalc_key="gemma-4-31b:gemma-dual",
    ),
    "vllm/gemma-int8-mtp": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
        kvcalc_key="gemma-4-31b:gemma-dual-int8",
    ),

    # Gemma-4-31B unsloth QAT W4A16 (compressed-tensors int4) — QAT-int4 fidelity alt to
    # autoround-int4. Same dual / int8-PTH-KV(#40391) / assistant-MTP path as gemma-int8-mtp.
    "vllm/gemma-31b-qat-w4a16-dual": _entry(
        model="gemma-4-31b", weights_variant="qat-w4a16", workload="multi-stream-tenant",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/qat-w4a16/int8.yml",
        default_port=8033, required_engine_features=["int8_per_token_head"],
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Gemma-4-31B unsloth QAT W4A16 (compressed-tensors int4) + int8-PTH KV (#40391) + assistant MTP n=4, dual TP=2 @262K. Boots clean on stock vllm-gemma-stable (NO #44494 workaround — tower-based Gemma4ForConditionalGeneration, unlike the 12B unified arch). 8-pack A/B 2026-06-07: 109/150 vs the autoround-int4 int8.yml's 105 (+4, within ±5-7 8-pack noise ≈ tie; real instructfollow edge IF 15-vs-8 offset by hermes/cli/TC/RM). WEAKER spec-decode though: MTP accept-len ~2.3 (n=3, the n-swept default) vs autoround's ~3.9 → less speedup. n-swept @370W: n2 72.9/86.1 · n3 74.0/87.7 · n4 71.6/87.8 (all within ~3%; n=3 = top narr + tied code + ~20% less drafting → set as the default vs autoround's n=4, since the QAT-int4's fast acceptance decay favors a lower n). Still slower than autoround (106/139 @230W — power not matched; the AL gap is the clean signal). Comparable QUALITY but slower — autoround-int4 (gemma-int8-mtp) stays the clear default. NIAH/soak not yet run.",
    ),

    # Gemma-4-12B (gemma4_unified arch — vLLM PR #44429, merged 2026-06-03),
    # dual-3090 bf16 on the ephemeral vllm/vllm-openai:gemma4-unified preview
    # image (== today's vLLM main; Gemma-4 fixes baked in except the local
    # p-RoPE cache sizing overlay below).
    # bf16 weights (~24 GB) don't fit one 24 GB card with KV → TP=2 mandatory.
    # Gemma-4-12B vLLM (gemma4_unified arch-preview image, ephemeral tag — pin a
    # digest before promotion). 256K works on the STOCK image: google/gemma-4-12B-it
    # ships max_position_embeddings=262144 (upstream config fix, vllm#39914), so the
    # former local vllm-gemma4-prope-longctx overlay was dropped 2026-06-04 (NIAH
    # 140K–241K validated overlay-free). kvcalc routes through the shared Gemma dense
    # path (gemma4_unified TEXT backbone == gemma4-swa-dense KV family). Only the MTP
    # variants ship (the no-drafter base composes were pruned — MTP is lossless and
    # fits the full 262144, so the bases bought nothing).
    "vllm/gemma-12b-dual-bf16-mtp": _entry(
        model="gemma-4-12b", weights_variant="bf16", workload="fast-chat",
        engine="vllm-gemma4-unified", drafter="gemma-12b-it-assistant", kv_format="bf16",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.90,
        compose_path="models/gemma-4-12b/vllm/compose/dual/bf16/mtp.yml",
        default_port=8036,
        kvcalc_key="gemma-4-12b:gemma-dual",
        status="caveats",
        status_note="Gemma-4-12B (gemma4_unified, vLLM PR #44429) dual-3090 bf16 + assistant spec-dec (n=4). ⚠️ Production w/ caveats: rebench-full 2026-06-04 (verify-full + bench + verify-stress + 8-pack 94/150 + soak PASS); 256K NIAH overlay-free (stock config fix vllm#39914). CAVEAT: ephemeral gemma4-unified arch-preview image (0.1.dev) — pin a digest; promotes to Production on a STABLE vLLM gemma4_unified release.",
    ),
    "vllm/gemma-12b-single-int8-mtp": _entry(
        model="gemma-4-12b", weights_variant="autoround-int8", workload="fast-chat",
        engine="vllm-gemma4-unified", drafter="gemma-12b-it-assistant", kv_format="bf16",
        tp=1, max_ctx=262144, max_num_seqs=4, mem_util=0.92,
        compose_path="models/gemma-4-12b/vllm/compose/single/autoround-int8/mtp.yml",
        default_port=8038,
        kvcalc_key="gemma-4-12b:gemma-single-int8-mtp",
        status="caveats",
        status_note="Gemma-4-12B Intel AutoRound INT8 (W8A16) + assistant external drafter (n=4) single 3090 on the gemma4_unified arch-preview image. ⚠️ Production w/ caveats: validated 2026-06-04 (bench + 256K NIAH + 8-pack 105/150 + soak PASS). MTP fits the full 262144 (drafter resident, KV pool ~310K tok, 1.18x at 262K, ~20.7 GB). n-sweep code-gen: n=4 117 TPS / accept_len 3.67 (n=5 122.5 code-max via SPEC_N=5) vs ~50 no-MTP. 8-pack on par with the bf16 dual's 94/150 (INT8≈bf16). bf16 KV only. CAVEAT: ephemeral arch-preview image (0.1.dev) — pin a digest; promotes to Production on a STABLE vLLM gemma4_unified release.",
    ),
    # QAT W4A16 (compressed-tensors int4) single-card — the sub-24 GB path: int4
    # weights (~7 GB) fit 16 GB cards where the INT8 (~13 GB) won't. Loses ~10pp to
    # the INT8 single on the 8-pack (int4-vs-int8 fidelity), so on a 24 GB card prefer
    # the INT8. Needs the vendored gemma4-unified-vision-unquant workaround (vLLM #44494).
    "vllm/gemma-12b-qat-w4a16-single": _entry(
        model="gemma-4-12b", weights_variant="qat-w4a16", workload="fast-chat",
        engine="vllm-gemma4-unified", drafter="gemma-12b-it-assistant", kv_format="bf16",
        tp=1, max_ctx=262144, max_num_seqs=4, mem_util=0.94,
        compose_path="models/gemma-4-12b/vllm/compose/single/qat-w4a16/mtp.yml",
        default_port=8039,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Gemma-4-12B unsloth QAT W4A16 (compressed-tensors int4) + assistant external drafter (n=4) single 3090 on the gemma4_unified arch-preview image. 🧪 Experimental — for sub-24 GB VRAM: the int4 weights (~7 GB) fit 16 GB cards where the INT8 (~13 GB) won't (vLLM #44494 commenter ran the same checkpoint on a 16 GB 4060 Ti). Full gate 2026-06-06: bench 99/131 TPS (MTP accept_len 2.37), 262K NIAH clean to 240K, 8-pack 95/150 — ~10pp below the INT8 single's 105 (the int4-vs-int8 cost), so on a 24 GB card the INT8 single is preferred. CAVEAT: requires the vendored gemma4-unified-vision-unquant workaround (sitecustomize forces the vision embedder unquantized — vLLM #44494 + missing num_soft_tokens) + the ephemeral gemma4-unified arch-preview image. Re-test on the upstream #44494 fix + a stable gemma4_unified release.",
    ),

    # Gemma-4-12B single-card GGUF (Q8_K_XL) — the two engine-native single-3090
    # paths that fit the bf16-too-big model in 24 GB. Both llama.cpp-family
    # (kvcalc SKIP, no vLLM kv-calc); 256K via `--override-kv` p-RoPE; no
    # spec-dec (gemma4 MTP draft arch unmerged, llama.cpp#23398). NIAH-clean to
    # 246K on a single 3090.
    "beellama/gemma-12b-single-q8kxl": _entry(
        model="gemma-4-12b", weights_variant="beellama-q8kxl", workload="fast-chat",
        engine="beellama-local", drafter=None, kv_format="q5_0",
        tp=1, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/gemma-4-12b/beellama/compose/single/beellama-q8kxl/base.yml",
        default_port=8067,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Gemma-4-12B Q8_K_XL single-3090 on beellama.cpp (q5_0(K)/q4_1(V) KV, 256K via --override-kv, no spec-dec). NIAH-clean to 246K. Launchers inject Anbeeld's official server-cuda-v0.3.0 image (sm_86). v0.3.0 is a DEV branch → experimental; promote on a stable tag.",
    ),
    "llamacpp/gemma-12b-single-q8kxl": _entry(
        model="gemma-4-12b", weights_variant="unsloth-q8kxl", workload="fast-chat",
        engine="llama-cpp-local", drafter=None, kv_format="q8_0",
        tp=1, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/gemma-4-12b/llama-cpp/compose/single/unsloth-q8kxl/base.yml",
        default_port=8069,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Gemma-4-12B Q8_K_XL single-3090 on mainline llama.cpp (q8_0 KV, 256K via --override-kv, no spec-dec). NIAH-clean to 246K. The no-fork mainline sibling of beellama/gemma-12b-single-q8kxl; no Gemma-4 spec-dec until llama.cpp#23398 merges.",
    ),

    # Gemma-4-31B beellama.cpp DFlash — single-card DEFAULT (Q4_K_S target +
    # Anbeeld DFlash-IQ4_XS draft, q5_0(K)/q4_1(V) KV). The ONLY viable fast
    # single-card Gemma-4 path: does SWA windowed KV (big ctx) AND Gemma-4
    # spec-dec, where vLLM is FA-walled (head_dim=512), ik-llama walls ~24K, and
    # stock llama.cpp is ~12 TPS (no FA_ALL_QUANTS). Promoted to single-GPU
    # default 2026-05-30 (no functional default existed before). llama.cpp-family
    # → kvcalc SKIP. Re-point to the no-fork mainline path when llama.cpp#23398
    # (Gemma-4 MTP) merges — see docs/UPSTREAM.md.
    "beellama/gemma-dflash": _entry(
        model="gemma-4-31b", weights_variant="beellama-q4ks-dflash", workload="fast-chat",
        engine="beellama-local", drafter="anbeeld-gemma-dflash", kv_format="q5_0",
        tp=1, max_ctx=128000, max_num_seqs=1, mem_util=None,
        compose_path="models/gemma-4-31b/beellama/compose/single/beellama-q4ks-dflash/dflash.yml",
        default_port=8061,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="Single-GPU default — the only viable fast single-card Gemma-4 path on Ampere. Launchers inject Anbeeld's official beellama.cpp server-cuda-v0.3.0 image (sm_86/89 = 3090/4090); sm_89 compiled-not-validated on club-3090's 3090-only rig. 5090/sm_120: prefix BEELLAMA_IMAGE=ghcr.io/noonghunna/beellama-cpp:multiarch-v0.3.0-efe856397 (sm_120 compiled-not-validated). DFlash prose is net-positive on tok/s (+28–31% vs no-spec, re-tested 2026-06-03; earlier 'prose regression' RETRACTED — AR over-read + wrong baseline); re-point to mainline llama.cpp#23398 Gemma-4 MTP when it merges — docs/UPSTREAM.md.",
    ),
    # Dual-card beellama Gemma-4 (layer-split, 262K) — PARKED upstream-gated 2026-05-31.
    # Boots + recalls 262K fine, but DFlash spec-dec is broken on multi-GPU in our pinned
    # build (07ac3ce): drafter decode fails, accept 0.357, ~24/38 TPS; --device-draft crashes.
    # Fixes live on Anbeeld's v0.3.0 dev branch (414 commits ahead) but no tagged release yet.
    # Re-test (DFlash-fix AND --spec-type mtp) when a beellama release lands. docs/UPSTREAM.md.
    "beellama/gemma-dflash-dual": _entry(
        model="gemma-4-31b", weights_variant="beellama-q4ks-dflash", workload="fast-chat",
        engine="beellama-local", drafter="anbeeld-gemma-dflash", kv_format="q5_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/gemma-4-31b/beellama/compose/dual/beellama-q4ks-dflash/dflash.yml",
        default_port=8062,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Dual-card beellama Gemma-4 (layer-split, 262K) on v0.3.0 — RELEASED experimental for community v0.3.0 testing (Anbeeld's request, club-3090#288). Multi-GPU DFlash FIXED on v0.3.0 (GPU cross-ring; validated sm_86 2026-06-01, FA_ALL_QUANTS=1; image injected from beellama-local install.spec = Anbeeld's official server-cuda-v0.3.0 commit tag). Earlier 'v0.3.0-wide DFlash-on-PROSE regression' RETRACTED (2026-06-03) — did NOT reproduce on qwen single+dual or gemma single (DFlash prose net-positive +27–58% vs no-spec); was an AR over-read + wrong baseline. This gemma-dual not separately re-benched. Promote experimental→caveats when Anbeeld tags a STABLE release. docs/UPSTREAM.md.",
    ),

    # ------------------------------------------------------------------
    # beellama v0.3.0 Q8_K_XL dual-card composes — RELEASED experimental for
    # community v0.3.0 testing (Anbeeld's request, club-3090#288). Image
    # injected centrally from engines/beellama-local.yml install.spec
    # (Anbeeld's official server-cuda-v0.3.0 commit tag). Validated 2× 3090
    # sm_86 2026-06-01. kvcalc_key=SKIP (llama.cpp family — no vLLM kv-calc).
    # ------------------------------------------------------------------
    "beellama/qwen-mtp-dual": _entry(
        model="qwen3.6-27b", weights_variant="beellama-q8kxl-mtp", workload="fast-chat",
        engine="beellama-local", drafter="unsloth-mtp-gguf", kv_format="q5_0",
        tp=2, max_ctx=65536, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/dual/beellama-q8kxl-mtp/mtp.yml",
        default_port=8064,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Dual-card beellama Qwen3.6-27B Q8_K_XL + embedded MTP head (--spec-type draft-mtp, unsloth-mtp-gguf drafter). v0.3.0 sm_86 2026-06-01: boots + coherent, MTP active (code accept ~0.90, ~58 TPS decode). Ships 65536 safe first-boot ctx; validated robust to ~160K (262K impossible — DeltaNet recurrent draft state hard-pins to one card). High-fidelity Q8 sibling of vllm/dual fp8-mtp. Promote experimental→caveats on a STABLE Anbeeld tag.",
    ),
    "beellama/qwen-dflash-dual": _entry(
        model="qwen3.6-27b", weights_variant="beellama-q8kxl-dflash", workload="fast-chat",
        engine="beellama-local", drafter="anbeeld-qwen-dflash", kv_format="q5_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/dual/beellama-q8kxl-dflash/dflash.yml",
        default_port=8065,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Dual-card beellama Qwen3.6-27B Q8_K_XL + DFlash (Anbeeld DFlash-IQ4_XS draft, --spec-type dflash). v0.3.0 sm_86 2026-06-01: boots + coherent at full 262K (fixed draft footprint; tensor-split 0.575,0.425 → ~21.2 GB/card). DFlash prose net-positive on tok/s (+52% vs no-spec @262K, re-tested 2026-06-03; earlier 'prose regression' RETRACTED — AR over-read + wrong baseline). Tool-grammar-neutral spec-dec for Qwen agents (club-3090#237). Promote on a STABLE tag.",
    ),
    "beellama/gemma-q8-dflash-dual": _entry(
        model="gemma-4-31b", weights_variant="beellama-q8kxl-dflash", workload="fast-chat",
        engine="beellama-local", drafter="anbeeld-gemma-dflash", kv_format="q5_0",
        tp=2, max_ctx=196608, max_num_seqs=1, mem_util=None,
        compose_path="models/gemma-4-31b/beellama/compose/dual/beellama-q8kxl-dflash/dflash.yml",
        default_port=8066,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Dual-card beellama Gemma-4-31B Q8_K_XL + DFlash (Anbeeld DFlash-IQ4_XS draft). v0.3.0 sm_86 2026-06-01: 192K balanced ceiling (tensor-split 0.55,0.45 → ~21.4/21.9 GB; 262K OOMs — Gemma full-attn layers grow KV). High-fidelity Q8 sibling of beellama/gemma-dflash-dual (q4ks). Earlier 'v0.3.0-wide DFlash-on-PROSE regression' RETRACTED (2026-06-03) — didn't reproduce on qwen single+dual or gemma single; was an AR over-read + wrong baseline. This gemma-Q8-dual not separately re-benched. Promote on a STABLE tag.",
    ),

    # Gemma 4 26B-A4B MoE — AWQ on vLLM v0.22.0. AWQ-4bit (compressed-tensors) MoE
    # experts resolve to Marlin WNA16 MoE on Ampere sm_86; the AutoRound INT4-mixed
    # variant is Ampere-dead (uint8b128, no W4A16 kernel) and was archived.
    # Single (#465, 2026-06-06): INT8-PTH KV via vendored PR #40391 (vllm-gemma-stable)
    # lifts the single-card ceiling to long context (240K NIAH-clean) vs the prior
    # bf16/16K path. Dual stays bf16/262K (no overlay; PR #40886 is in v0.22.0).
    "vllm/gemma-26ba4b-single": _entry(
        model="gemma-4-26b-a4b", weights_variant="awq", workload="fast-chat",
        engine="vllm-gemma-stable", drafter="gemma-26b-it-assistant", kv_format="int8_per_token_head",
        tp=1, max_ctx=176000, max_num_seqs=256, mem_util=0.94,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/single/awq/int8.yml",
        default_port=8040,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="AWQ MoE + external MTP (n=4) + INT8-PTH KV via vendored PR #40391 on vLLM v0.22.0 (vllm-gemma-stable). Gate PASS 2026-06-06 (rebench gemma-26ba4b-int8r): verify-full ✓, bench 168 narr / 217 code TPS @370W (MTP AL 3.0-3.8), verify-stress NIAH→161K ✓, soak 20x5 PASS 0-growth, quality 109/150 think-ON (~ gemma-4-31B 107/150) / 98/150 think-OFF. Caveats: needs the #40391 overlay (not in stock v0.22.0); 176K @ mem_util 0.94 (262K only WITHOUT the MTP drafter — 0.96 OOMs the cudagraph-capture tail); think-OFF agentic/extraction softer (cli-40 30%, DataExtract 60% — recover to 52%/73% with thinking). INT8-PTH lifted single-card ctx from the prior bf16/16K.",
    ),
    "vllm/gemma-26ba4b-dual": _entry(
        model="gemma-4-26b-a4b", weights_variant="awq", workload="fast-chat",
        engine="vllm-stable", drafter="gemma-26b-it-assistant", kv_format="bf16",
        tp=2, max_ctx=262144, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/dual/awq/mtp.yml",
        default_port=8041,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="AWQ + external MTP (gemma-26b-it-assistant n=4) on stock v0.22.0 — MTP +55% TPS (134->208, AL 3.55) validated 2026-06-05. Max ctx 262K (model max; KV pool 806,821 tok at 262144/0.92, 2x 3090) boot+coherence validated 2026-06-06. Promote after rebench-full + soak.",
    ),
    "vllm/qwen-a3b-preview-single": _entry(
        model="qwen3.6-35b-a3b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=8192, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/single/autoround-int4/preview.yml",
        default_port=8050,
        kvcalc_key="qwen3.6-35b-a3b:qwen-a3b-preview-single",
        status="preview",
        status_note="MoE onboarding smoke — Cliff 2 mitigations unavailable without Genesis. Do NOT use for long-ctx.",
    ),
    "vllm/qwen-35b-a3b-dual": _entry(
        model="qwen3.6-35b-a3b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml",
        default_port=8051,
        kvcalc_key="qwen3.6-35b-a3b:qwen-35b-a3b-dual",
    ),

    # Qwen3.6-40B-Deckard — dense 40B uncensored community merge, llama.cpp dual.
    # First dual llama.cpp compose in the catalog. Q6_K GGUF (31 GB) requires both
    # cards; layer-split via -ts 1,1. MTP n=2 sweet spot (41.6 tok/s, 0.81 accept).
    # kv_format q8_0 (K+V). kvcalc SKIP (llama.cpp family — no vLLM kv-calc).
    "llamacpp/deckard40B-dual-mtp": _entry(
        model="qwen3.6-40b-deckard", weights_variant="mtp-q6k", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=2, max_ctx=131072, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/mtp-q6k/mtp.yml",
        default_port=8199,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Dense 40B uncensored Qwen3.6 merge (Q6_K MTP GGUF, 31 GB) on dual 3090 llama.cpp. MTP n=2 sweet spot (~41.6 tok/s, 0.81 accept). 128K ctx ceiling @q8_0 KV (192K OOMs). Dual-only. verify-full 8/8, 8-pack 105/150 (MTP off==on, spec-dec lossless), soak-continuous PASS (0 MiB growth, 0/25 silent-empty). 🧪 pending profile config.json arch confirm before ✅.",
        category="uncensored",
    ),
}



DEFAULTS = {
    ("qwen3.6-27b", "vllm", "single"): "vllm/minimal",
    ("qwen3.6-27b", "vllm", "dual"): "vllm/dual",
    ("qwen3.6-27b", "llamacpp", "single"): "llamacpp/default",
    ("qwen3.6-27b", "ik-llama", "single"): "ik-llama/iq4ks-mtp",
    ("qwen3.6-27b", "beellama", "single"): "beellama/dflash",
    # No vLLM single-card Gemma default: fp8 KV is hardware-impossible on Ampere
    # sm_86 (vllm/gemma-mtp-tp1 deprecated 2026-05-31) and no bf16 single compose
    # ships. Single-card Gemma → beellama/gemma-dflash (the curated walk picks it).
    ("gemma-4-31b", "beellama", "single"): "beellama/gemma-dflash",
    # Dual default is gemma-int8-mtp: full 262K + vision + 4 streams (the full-context
    # priority). It rides v0.21.0 + the vendored #40391 per-head-KV overlay (the one
    # gemma config that can't follow stable). gemma-bf16-mtp stays as the stable v0.22.0
    # no-overlay 32K fallback — kept, not deprecated, just no longer the default.
    ("gemma-4-31b", "vllm", "dual"): "vllm/gemma-int8-mtp",
    ("gemma-4-26b-a4b", "vllm", "single"): "vllm/gemma-26ba4b-single",
    ("gemma-4-26b-a4b", "vllm", "dual"): "vllm/gemma-26ba4b-dual",
    ("qwen3.6-35b-a3b", "vllm", "single"): "vllm/qwen-a3b-preview-single",
    ("qwen3.6-35b-a3b", "vllm", "dual"): "vllm/qwen-35b-a3b-dual",
    # Deckard: only one compose (dual llama.cpp MTP), so the dual default is trivial.
    ("qwen3.6-40b-deckard", "llamacpp", "dual"): "llamacpp/deckard40B-dual-mtp",
}


# --- PR-B: model-default resolver knobs (maintainer-owned, design §13.3) ----
#
# Two curated tables drive the `<model>/default` resolver. They are maintainer
# knobs — edited by PR, never auto-grown. See docs/model-default-resolver
# design + the repo CLAUDE.md "Default rule" note.

# A SHORT opt-in shortlist of models eligible to be the *bare-launch* default
# (`launch.sh` with no model + no pin → first INSTALLED model on this list →
# its `<model>/default`). This is NOT an exhaustive ranking of the catalog:
#   - Models absent from it are fully runnable by name (`--model X` /
#     `X/default`); they are simply never the auto-default.
#   - New models are NOT auto-added. Adding a model touches nothing here;
#     promote one explicitly only when desired.
#   - Order within the (short) list = the tiebreak for "first installed".
RECOMMENDED_DEFAULT_MODELS = ["qwen3.6-27b", "gemma-4-31b"]

# Which engine wins, per detected topology, when resolving `<model>/default`
# with no user pin. The resolver walks this list in order and picks the FIRST
# engine that has a functional DEFAULTS[(model, engine, topology)] entry (i.e.
# whose status is NOT in the (NA) set). This is the whole recommendation
# policy expressed as data — reorder a row to change a recommendation, no code
# change, any topology.
#
# Engine identifiers are the slug-prefix form used in DEFAULTS keys:
#   vllm · ik-llama · llamacpp   (+ beellama, aspirational — see below).
# `beellama` is ranked but has ZERO registry entries today (blocked on an
# upstream Docker image — see docs/UPSTREAM.md). The resolver skips it
# naturally (no DEFAULTS hit) → no behavior change until it is onboarded, at
# which point it AUTO-PROMOTES to the single-GPU default with no resolver edit.
ENGINE_PREFERENCE = {
    "single": ["beellama", "ik-llama", "llamacpp", "vllm"],
    "dual": ["vllm", "ik-llama", "llamacpp", "beellama"],
    "multi": ["vllm", "ik-llama", "llamacpp", "beellama"],
}


def _topology_family(topology):
    """Map a concrete topology to its ENGINE_PREFERENCE family.

    Concrete topologies are `single` · `dual` · `multi4` · `multiN`; the
    preference table keys on the family `single` · `dual` · `multi`.
    """
    if topology == "single":
        return "single"
    if topology == "dual":
        return "dual"
    if topology.startswith("multi"):
        return "multi"
    return topology


def _nearest_lower_topology(topology):
    """Degradation order (design §6): notice + nearest-lower topology.

    multiN → dual → single → None. Returns the next topology to try, or None
    when there is nowhere lower to fall.
    """
    if topology.startswith("multi"):
        return "dual"
    if topology == "dual":
        return "single"
    return None


def engine_set():
    """The closed set of engine namespace-prefixes (DEFAULTS keys + ranked).

    `X/default` dispatch (design §13.1): `X ∈ engine_set` → engine
    recommendation; else `X ∈ model_set` → model default; else error.
    Engines and model-ids are disjoint by construction.
    """
    engines = set()
    for _model, engine, _topology in DEFAULTS:
        engines.add(engine)
    for ranked in ENGINE_PREFERENCE.values():
        engines.update(ranked)
    return engines


def model_set():
    """The set of model-ids that appear in DEFAULTS (the runnable catalog)."""
    return {model for (model, _engine, _topology) in DEFAULTS}


def _functional_default(model, engine, topology):
    """A DEFAULTS slug for (model, engine, topology) whose status is functional.

    Returns the slug only when an entry exists AND its registry status is NOT
    in the (NA) set (experimental/preview/upstream-gated/deprecated) — a
    broken/preview config must never become someone's auto-default (§12.5).
    Returns None otherwise.
    """
    slug = DEFAULTS.get((model, engine, topology))
    if not slug:
        return None
    entry = COMPOSE_REGISTRY.get(slug)
    if entry is None:
        return None
    if entry.get("status", "production") not in FUNCTIONAL_STATUSES:
        return None
    return slug


def curated_default_target(model, topology):
    """Curated fallback (§4): walk ENGINE_PREFERENCE[family], first functional
    DEFAULTS slug wins. Returns the slug, or None if no functional curated
    default exists for (model, topology).
    """
    family = _topology_family(topology)
    for engine in ENGINE_PREFERENCE.get(family, []):
        slug = _functional_default(model, engine, topology)
        if slug:
            return slug
    return None


def community_default_target(model, topology, hw_class=None):  # noqa: ARG001
    """Community-ranked best config — the FUTURE middle precedence rung (§13.4).

    Contract: returns a ranked slug when the submissions/ranking app exists;
    returns None today (always skipped). The resolver inserts a non-None result
    BETWEEN the user pin and the curated fallback. v1 ships this stub returning
    None so the ladder rung is real, not aspirational; a test asserts it is
    skipped.
    """
    return None


def model_default_pin_key(model):
    """The .env key for a per-model user pin (design §13.2).

    `CLUB3090_DEFAULT_<MODELID uppercased, non-alnum→_>`, e.g.
    qwen3.6-27b → CLUB3090_DEFAULT_QWEN3_6_27B.
    """
    suffix = "".join(c if c.isalnum() else "_" for c in model).upper()
    return f"CLUB3090_DEFAULT_{suffix}"


def model_of_slug(slug):
    """The model-id a slug belongs to, or None if the slug is unknown."""
    entry = COMPOSE_REGISTRY.get(slug)
    return entry.get("model") if entry else None


def slug_topology(slug):
    """The topology family a slug serves, derived from its compose_path.

    compose_path is `models/<model>/<engine>/compose/<topology>/<quant>/...`.
    Returns `single`/`dual`/`multi` (the ENGINE_PREFERENCE family) or None.
    """
    entry = COMPOSE_REGISTRY.get(slug)
    if not entry:
        return None
    cp = entry.get("compose_path", "")
    if "/compose/" not in cp:
        return None
    after = cp.split("/compose/", 1)[1]
    topo = after.split("/", 1)[0]
    return _topology_family(topo)
