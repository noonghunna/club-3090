"""Static compose-to-profile bridge for v0.7.0.

The registry intentionally mirrors the shipped compose files. It is not a
generator and it does not attempt to normalize away historical variants.
"""


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
    requires_nvlink=False,
    required_engine_features=None,
    recommended_engine_features=None,
    required_sm=None,
):
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
    }
    if recommended_engine_features:
        entry["recommended_engine_features"] = list(recommended_engine_features)
    if required_sm is not None:
        entry["required_sm"] = required_sm
    return entry


COMPOSE_REGISTRY = {
    # Qwen 3.6 27B, vLLM single-card.
    "vllm/default": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="tool-heavy",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=48000, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/single/docker-compose.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/long-text": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=180000, max_num_seqs=1, mem_util=0.93,
        compose_path="models/qwen3.6-27b/vllm/compose/single/long-text.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/long-text-no-mtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter=None, kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/long-text-no-mtp.yml",
        default_port=8021, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/long-vision": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="vision-coding",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=145000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/long-vision.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/bounded-thinking": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="tool-heavy",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=1, max_ctx=180000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/single/bounded-thinking.yml",
        default_port=8020, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/tools-text": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="tool-heavy",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=1, max_ctx=75000, max_num_seqs=1, mem_util=0.97,
        compose_path="models/qwen3.6-27b/vllm/compose/single/tools-text.yml",
        default_port=8020,
    ),
    "vllm/minimal": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="fast-chat",
        engine="vllm-nightly-mtp", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=32768, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/single/minimal.yml",
        default_port=8020,
    ),

    # Qwen 3.6 27B, vLLM dual/multi-card.
    "vllm/dual": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/docker-compose.yml",
        default_port=8010, recommended_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-turbo": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/turbo.yml",
        default_port=8011, required_engine_features=["turboquant_3bit_nc"],
        recommended_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-dflash": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="vision-coding",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=185000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/dflash.yml",
        default_port=8012, required_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-dflash-noviz": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/dflash-noviz.yml",
        default_port=8013, required_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-bf16": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="bf16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/bf16.yml",
        default_port=8012,
    ),
    "vllm/dual-int8": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-full", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/int8.yml",
        default_port=8011, required_engine_features=["int8_per_token_head"],
    ),
    "vllm/dual-tq3-mtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/tq3-mtp.yml",
        default_port=8013, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/dual-tq3-mtp-genesis": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/tq3-mtp-genesis.yml",
        default_port=8015, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/dual-tq3-nomtp": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter=None, kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/tq3-nomtp.yml",
        default_port=8014, required_engine_features=["turboquant_3bit_nc"],
    ),
    "vllm/dual-carnice-bf16mtp": _entry(
        model="qwen3.6-27b", weights_variant="carnice_bf16mtp", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/carnice-bf16mtp.yml",
        default_port=8070,
    ),
    "vllm/dual-qwopus-bf16mtp": _entry(
        model="qwen3.6-27b", weights_variant="qwopus_bf16mtp", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/qwopus-bf16mtp.yml",
        default_port=8071,
    ),
    "vllm/dual-nvlink": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/nvlink.yml",
        default_port=8014, requires_nvlink=True, recommended_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-nvlink-turbo": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/nvlink-turbo.yml",
        default_port=8017, requires_nvlink=True, required_engine_features=["turboquant_3bit_nc"],
        recommended_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-nvlink-dflash": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="vision-coding",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=185000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/nvlink-dflash.yml",
        default_port=8018, requires_nvlink=True, required_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual-nvlink-dflash-noviz": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/nvlink-dflash-noviz.yml",
        default_port=8019, requires_nvlink=True, required_engine_features=["marlin_pad_sub_tile_n"],
    ),
    "vllm/dual4": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-mtp", drafter="qwen-mtp-builtin", kv_format="fp8_e5m2",
        tp=4, max_ctx=262144, max_num_seqs=4, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/docker-compose.yml",
        default_port=8015,
    ),
    "vllm/dual4-dflash": _entry(
        model="qwen3.6-27b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-dflash", drafter="zlab-qwen-dflash", kv_format="fp16",
        tp=4, max_ctx=262144, max_num_seqs=2, mem_util=0.95,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/dflash.yml",
        default_port=8016, required_engine_features=["marlin_pad_sub_tile_n"],
    ),

    # Qwen 3.6 27B, llama.cpp single-card.
    "llamacpp/default": _entry(
        model="qwen3.6-27b", weights_variant="gguf", workload="long-ctx-single",
        engine="llama-cpp-mainline", drafter=None, kv_format="q4_0",
        tp=1, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/docker-compose.yml",
        default_port=8020,
    ),
    "llamacpp/concurrent": _entry(
        model="qwen3.6-27b", weights_variant="gguf", workload="multi-stream-tenant",
        engine="llama-cpp-mainline", drafter=None, kv_format="q4_0",
        tp=1, max_ctx=192000, max_num_seqs=4, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/concurrent.yml",
        default_port=8020,
    ),

    # Gemma 4 31B, vLLM.
    "vllm/gemma-mtp-tp1": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="fast-chat",
        engine="vllm-nightly-mtp", drafter="gemma-it-assistant", kv_format="fp8_e4m3",
        tp=1, max_ctx=8192, max_num_seqs=256, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/single/docker-compose.yml",
        default_port=8031, required_sm=9.0,
    ),
    "vllm/gemma-mtp": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="fast-chat",
        engine="vllm-nightly-mtp", drafter="gemma-it-assistant", kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=4, mem_util=0.92,
        compose_path="models/gemma-4-31b/vllm/compose/dual/docker-compose.yml",
        default_port=8030,
    ),
    "vllm/gemma-int8": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-full", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=98304, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
    ),
    "vllm/gemma-int8-262k": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-full", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
    ),
    "vllm/gemma-dflash": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="fast-chat",
        engine="vllm-nightly-dflash", drafter="gemma-dflash", kv_format="bf16",
        tp=2, max_ctx=32768, max_num_seqs=4, mem_util=0.92,
        compose_path="models/gemma-4-31b/vllm/compose/dual/dflash.yml",
        default_port=8032,
    ),
    "vllm/gemma-dflash-int8": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-full", drafter="gemma-dflash", kv_format="int8_per_token_head",
        tp=2, max_ctx=65536, max_num_seqs=2, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/dflash-int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
    ),
    "vllm/gemma-int8-tq3": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="multi-stream-tenant",
        engine="vllm-nightly-full", drafter="gemma-it-assistant", kv_format="turboquant_3bit_nc",
        tp=2, max_ctx=98304, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/int8-tq3.yml",
        default_port=8034, required_engine_features=["turboquant_3bit_nc"], required_sm=9.0,
    ),
    "vllm/gemma-bf16": _entry(
        model="gemma-4-31b", weights_variant="autoround_int4", workload="long-ctx-single",
        engine="vllm-nightly-mtp", drafter="gemma-it-assistant", kv_format="bf16",
        tp=2, max_ctx=200000, max_num_seqs=1, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/bf16.yml",
        default_port=8033,
    ),
    "vllm/gemma-awq": _entry(
        model="gemma-4-31b", weights_variant="awq", workload="multi-stream-tenant",
        engine="vllm-nightly-full", drafter="gemma-it-assistant", kv_format="bf16",
        tp=2, max_ctx=65536, max_num_seqs=4, mem_util=0.85,
        compose_path="models/gemma-4-31b/vllm/compose/dual/awq.yml",
        default_port=8033,
    ),

    # v0.7.3 MoE onboarding — Gemma 4 26B-A4B + Qwen 3.6 35B-A3B.
    # Both target the unconstrained-nightly engine (vllm-nightly-clean) which
    # rides nightly-bf610c2f (2026-05-15, post-PR-#42521). Gemma is the
    # shippable path; Qwen 35B-A3B is preview-only until Genesis v7.73.x
    # re-anchors on a post-#42521 nightly.
    "vllm/gemma-a4b": _entry(
        model="gemma-4-26b-a4b", weights_variant="autoround_int4_mixed", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="bf16",
        tp=1, max_ctx=8192, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/single/docker-compose.yml",
        default_port=8040,
    ),
    "vllm/qwen-a3b-preview": _entry(
        model="qwen3.6-35b-a3b", weights_variant="autoround_int4", workload="fast-chat",
        engine="vllm-nightly-clean", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=8192, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/single/preview.yml",
        default_port=8050,
    ),
}

