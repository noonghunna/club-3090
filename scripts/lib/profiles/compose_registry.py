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
    "vllm/default": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="tool-heavy",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=48000, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/tq3-mtp.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:long-vision",
    ),
    "vllm/long-text": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=180000, max_num_seqs=1, mem_util=0.93,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-text.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:long-text",
        status="caveats",
        status_note="Cliff 2b at >50K single-prompt context (not IDE-agent safe).",
    ),
    "vllm/long-text-no-mtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter=None, kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-text-no-mtp.yml",
        default_port=8021, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:long-text-no-mtp",
        status="caveats",
        status_note="Cliff 2b at >50K single-prompt context (not IDE-agent safe).",
    ),
    "vllm/long-vision": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="vision-coding",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=145000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/long-vision.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:long-vision",
        status="caveats",
        status_note="Cliff 2b at >50K single-prompt context (not IDE-agent safe).",
    ),
    "vllm/bounded-thinking": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="tool-heavy",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=180000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/bounded-thinking.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:bounded-thinking",
    ),
    "vllm/tools-text": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="tool-heavy",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=1, max_ctx=75000, max_num_seqs=1, mem_util=0.97,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/tools-text.yml",
        default_port=8020,
        kvcalc_key="qwen3.6-27b:tools-text",
    ),
    "vllm/minimal": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=32768, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml",
        default_port=8020,
        kvcalc_key="qwen3.6-27b:minimal",
    ),

    # Qwen 3.6 27B, vLLM dual/multi-card.
    "vllm/dual": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml",
        default_port=8010, recommended_engine_features=["marlin_pad_sub_tile_n"],
        kvcalc_key="qwen3.6-27b:dual",
    ),
    "vllm/dual-turbo": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/turbo.yml",
        default_port=8011, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:dual-turbo",
        recommended_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-dflash": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="vision-coding",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=185000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/dflash.yml",
        default_port=8012, required_engine_features=["marlin_pad_sub_tile_n"],
        kvcalc_key="qwen3.6-27b:dual-dflash",
    ),
    "vllm/dual-dflash-noviz": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/dflash-noviz.yml",
        default_port=8013, required_engine_features=["marlin_pad_sub_tile_n"],
        kvcalc_key="qwen3.6-27b:dual-dflash-noviz",
    ),
    "vllm/dual-bf16": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="bf16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/bf16.yml",
        default_port=8012,
        kvcalc_key="qwen3.6-27b:dual-bf16",
        status="experimental",
        status_note="BF16 KV on Qwen3-Next DeltaNet not validated on this stack; matched-config A/B vs Gemma bf16.",
    ),
    "vllm/dual-int8": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-full", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/int8.yml",
        default_port=8011, required_engine_features=["int8_per_token_head"],
        kvcalc_key="qwen3.6-27b:dual-int8",
        status="experimental",
        status_note="INT8 PTH KV on Qwen3-Next DeltaNet not validated on this stack; matched-config A/B vs Gemma int8.",
    ),
    "vllm/dual-tq3-mtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-mtp.yml",
        default_port=8013, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:dual-tq3-mtp",
        status="deprecated",
        status_note="Tombstoned 2026-05-11 — Genesis-free TQ3+MTP needs 4 of 5 upstream fixes not yet landed. Use dual-tq3-mtp-genesis (gated) or dual-turbo.",
    ),
    "vllm/dual-tq3-mtp-genesis": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-mtp-genesis.yml",
        default_port=8015, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:dual-tq3-mtp-genesis",
        status="upstream-gated",
        status_note="Genesis pin parked/drifted — won't boot clean until the pin re-validates. See docs/UPSTREAM.md (Genesis row).",
    ),
    "vllm/dual-tq3-nomtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter=None, kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/tq3-nomtp.yml",
        default_port=8014, required_engine_features=["turboquant_3bit_nc"],
        kvcalc_key="qwen3.6-27b:dual-tq3-nomtp",
    ),
    "vllm/dual-carnice-bf16mtp": _entry(
        model="qwen3.6-27b", weights_variant="carnice-bf16mtp", workload="long-ctx-single",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/carnice-bf16mtp/bf16-mtp.yml",
        default_port=8070,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="Carnice fine-tune MTP AL=2.0 vs Lorbus 3.4-3.8 (working but suboptimal TPS).",
    ),
    "vllm/dual-qwopus-bf16mtp": _entry(
        model="qwen3.6-27b", weights_variant="qwopus-bf16mtp", workload="long-ctx-single",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/qwopus-bf16mtp/bf16-mtp.yml",
        default_port=8071,
        kvcalc_key="SKIP",
        status="preview",
        status_note="Qwopus fine-tune preview: line repetition + NIAH drop + silent-empty turn-5 in soak.",
    ),
    "vllm/dual4": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-nightly-clean", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=4, max_ctx=262144, max_num_seqs=4, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/fp8-mtp.yml",
        default_port=8015,
        kvcalc_key="qwen3.6-27b:dual4",
    ),
    "vllm/dual4-dflash": _entry(
        model="qwen3.6-27b", weights_variant="autoround-int4", workload="long-ctx-single",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=4, max_ctx=262144, max_num_seqs=2, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/dflash.yml",
        default_port=8016, required_engine_features=["marlin_pad_sub_tile_n"],
        kvcalc_key="qwen3.6-27b:dual4-dflash",
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
        status="upstream-gated",
        status_note="Boot-OOMs on Ampere 24 GB; needs 32 GB+ GPU or Ampere-aware fp8 dispatch. See compose Caveats.",
    ),
    "vllm/gemma-mtp": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=4, mem_util=0.92,
        compose_path="models/gemma-4-31b/vllm/compose/dual/autoround-int4/bf16-mtp.yml",
        default_port=8030,
        kvcalc_key="gemma-4-31b:gemma-dual",
    ),
    "vllm/gemma-int8": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=98304, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
        kvcalc_key="gemma-4-31b:gemma-dual-int8",
    ),

    # v0.7.3 MoE onboarding — Gemma 4 26B-A4B + Qwen 3.6 35B-A3B.
    # Both target the unconstrained-nightly engine (vllm-nightly-clean) which
    # rides nightly-bf610c2f (2026-05-15, post-PR-#42521). Gemma is the
    # shippable path; Qwen 35B-A3B is preview-only until Genesis v7.73.x
    # re-anchors on a post-#42521 nightly.
    "vllm/gemma-a4b-single": _entry(
        model="gemma-4-26b-a4b", weights_variant="autoround-int4-mixed", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="bf16",
        tp=1, max_ctx=8192, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/single/autoround-int4-mixed/bf16.yml",
        default_port=8040,
        kvcalc_key="gemma-4-26b-a4b:gemma-a4b-single",
        status="experimental",
        status_note="v0.7.3 MoE onboarding — first-boot smoke, validation pending.",
    ),
    "vllm/gemma-a4b": _entry(
        model="gemma-4-26b-a4b", weights_variant="autoround-int4-mixed", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/dual/autoround-int4-mixed/bf16.yml",
        default_port=8041,
        kvcalc_key="gemma-4-26b-a4b:gemma-a4b",
        status="experimental",
        status_note="v0.7.3 MoE onboarding — primary bench target, validation pending.",
    ),
    "vllm/gemma-a4b-awq": _entry(
        model="gemma-4-26b-a4b", weights_variant="awq", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/dual/awq/bf16.yml",
        default_port=8042,
        kvcalc_key="gemma-4-26b-a4b:gemma-a4b-awq",
        status="experimental",
        status_note="v0.7.3 MoE onboarding — AWQ path with PR #40886 overlay, validation pending.",
    ),
    "vllm/gemma-a4b-awq-mtp": _entry(
        model="gemma-4-26b-a4b", weights_variant="awq", workload="fast-chat",
        engine="vllm-nightly-clean", drafter="gemma-26b-it-assistant", kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/dual/awq/mtp.yml",
        default_port=8043,
        kvcalc_key="gemma-4-26b-a4b:gemma-a4b-awq-mtp",
        status="experimental",
        status_note="v0.7.3 MoE onboarding — AWQ + MTP, validation pending.",
    ),
    "vllm/qwen-a3b-preview-single": _entry(
        model="qwen3.6-35b-a3b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=8192, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/single/autoround-int4/preview.yml",
        default_port=8050,
        kvcalc_key="qwen3.6-35b-a3b:qwen-a3b-preview-single",
        status="preview",
        status_note="MoE onboarding smoke — Cliff 2 mitigations unavailable without Genesis. Do NOT use for long-ctx.",
    ),
    "vllm/qwen-35b-a3b-dual": _entry(
        model="qwen3.6-35b-a3b", weights_variant="autoround-int4", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml",
        default_port=8051,
        kvcalc_key="qwen3.6-35b-a3b:qwen-35b-a3b-dual",
    ),
}



DEFAULTS = {
    ("qwen3.6-27b", "vllm", "single"): "vllm/default",
    ("qwen3.6-27b", "vllm", "dual"): "vllm/dual",
    ("qwen3.6-27b", "vllm", "multi4"): "vllm/dual4",
    ("qwen3.6-27b", "llamacpp", "single"): "llamacpp/default",
    ("qwen3.6-27b", "ik-llama", "single"): "ik-llama/iq4ks-mtp",
    ("gemma-4-31b", "vllm", "single"): "vllm/gemma-mtp-tp1",
    ("gemma-4-31b", "vllm", "dual"): "vllm/gemma-mtp",
    ("gemma-4-26b-a4b", "vllm", "single"): "vllm/gemma-a4b-single",
    ("gemma-4-26b-a4b", "vllm", "dual"): "vllm/gemma-a4b",
    ("qwen3.6-35b-a3b", "vllm", "single"): "vllm/qwen-a3b-preview-single",
    ("qwen3.6-35b-a3b", "vllm", "dual"): "vllm/qwen-35b-a3b-dual",
}
