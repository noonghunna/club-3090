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
    "incubating",      # 🐣 Incubating — pre-experimental: works but not ready for the
                       #    actionable list (niche / fails the standard gate by design).
                       #    HIDDEN from `switch.sh --list` by default; revealed by `--all`.
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
    "🐣": "incubating",
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
    fallback_sm=None,
    status="production",
    status_note=None,
    category=None,
    weights_companions=None,
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
        # Extra weight-variant keys (a DFlash draft / mmproj projector) this slug's
        # compose mounts from a separate subdir, BEYOND the core weights_variant.
        # The serve-cockpit Download action fetches these alongside the core so the
        # slug actually serves.  Bare keys, scoped to this entry's model.
        "weights_companions": list(weights_companions or []),
    }
    if recommended_engine_features:
        entry["recommended_engine_features"] = list(recommended_engine_features)
    if required_sm is not None:
        entry["required_sm"] = required_sm
    if fallback_sm is not None:
        # Weight-only fallback floor. required_sm = the NATIVE-kernel SM;
        # fallback_sm = the lowest SM where the format still RUNS via a
        # weight-only fallback kernel (e.g. NVFP4 -> Marlin W4A16, floor
        # sm 7.5 per vLLM marlin_utils_fp4). In the band
        # [fallback_sm, required_sm) the gates ALLOW with a fallback
        # annotation instead of refusing (kv-calc emits `hw_fallback`;
        # c3 shows the slug with a ⚑ badge instead of hiding it).
        # Live-confirmed on 2x3090 sm_86 2026-07-11: NVFP4-27B boots,
        # 69.7/85.5 TPS, 8-pack 110/150 (ties the fp8 tier's 109).
        entry["fallback_sm"] = fallback_sm
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
    #   *-max  = official FP8 weights   + fp8/e4m3 KV  (higher-fidelity weights @ 262K)
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
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e4m3",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/fp8/mtp.yml",
        default_port=8013,
        kvcalc_key="SKIP",
        status="production",
        status_note="Qwen3.6-27B 'max accuracy' tier, 2-card: official FP8 weights (embedded MTP head) + fp8/e4m3 KV (flipped from int8-PTH in #594) + MTP n=3, TP=2 @262K. fp8/e4m3 routes KV attention to FlashInfer (int8-PTH is TRITON_ATTN-only): decode stays FLAT at depth — 2.3x int8-PTH @35K — where int8-PTH craters. Full v0.24.0 gate: verify-full 9/9, verify-stress fillable to 240,636 tok, soak-continuous PASS (0 err / 0 growth / 100% retention, p50 decode 125.5), 8-pack --full 109/150 (ties int8-PTH's 107, quality-neutral despite fp8 scale=1.0 — vLLM disables calculate_kv_scales on Qwen3-Next hybrid). KV pool 295K tok / 1.13x concurrency (smallest pool of the tiers; FP8 weights use MarlinFP8 W8A16 on Ampere — memory win, not decode). The highest-fidelity weight tier; consumer Blackwell (5090+) gets native FP8 GEMM via the launcher's DeepGEMM-disable. Also the validation proxy for vllm/qwen-27b-multi-max (same config @ TP=4).",
    ),
    "vllm/qwen-27b-dual-lmcache": _entry(
        model="qwen3.6-27b", weights_variant="fp8", workload="long-ctx-single",
        engine="vllm-lmcache", drafter="qwen-mtp-builtin", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm-lmcache/compose/dual/fp8/lmcache.yml",
        default_port=8017,
        kvcalc_key="SKIP",
        status="incubating",
        status_note="Qwen3.6-27B 'max accuracy + LMCache KV-offload' tier, 2-card: byte-identical serving fidelity to vllm/qwen-27b-dual-max (FP8 weights + int8-PTH KV + MTP n=3 @262K, TP=2) PLUS an LMCache tiered persistent prefix-KV cache (MP/HMA connector, lmcache 0.4.7). 🐣 Incubating — OPT-IN, hidden from switch.sh --list (--force to launch). Live-validated 2026-06-17 (club-3090 #133): boots + serves @262K (util 0.92, KV pool 279K tok / 1.07x), MTP active (~83% accept), and a CONTROLLED A/B (toggle ONLY the connector, same image+config) shows decode 74 narr / 94 code TPS == WITHOUT LMCache → ZERO decode penalty (offload is async/overlapped; an earlier 'halves decode' claim was an uncontrolled-measurement error, retracted — see #133). LMCache caches each session's prefix KV in CPU RAM (L1, --l1-size-gb 30 default ≈ ~4 realistic 50K sessions / <1 full 262K @ ~131 KB/token measured cache footprint — 7x the GPU's 18.9 int8-PTH rate, which only sets max ctx) + optional disk (L2 fs adapter, env-gated LMCACHE_L2=1 or LMCACHE_L2_ADAPTER, off by default, ~131 KB/token; rehydrate 4.8s vs 43s cold ~9x, survives restarts; unbounded — size disk per the INTERNALS table). THREE reasons incubating not production: (1) runs LMCache's third-party image (lmcache/vllm-openai, DIGEST-pinned — the tag is MUTABLE, and it bundles a newer vLLM 0.23.1-dev than our v0.22.0 pin; ✅-promotion wants LMCache on our own image), (2) L2 disk tier is unbounded (no size cap — disk-fill is on the operator; preflight soft-warns), (3) 38 GB image pulled on-demand. RAM-gated: preflight rejects --l1-size-gb over-allocation (the l1=100→reboot incident); needs ~58 GB free at l1=30, cap ~50 GB on a 94 GB rig; shm_size must track l1. Best for many concurrent long sessions kept warm (cold→warm TTFT ~7-8x) — efschu's high-context multi-session use case. Standard duals (vllm/dual, vllm/qwen-27b-dual-max) stay LMCache-free.",
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
        status="production",
        status_note="Qwen3.6-27B 'fast' tier, 4-card (TP=4): AutoRound INT4 + fp8_e5m2 KV + MTP n=3 @262K. Byte-identical to vllm/dual (≡ vllm/qwen-27b-dual-fast) apart from TP=4 + gpu-count; vllm/dual @TP=2 is the on-rig proxy (this dev rig has 2× 3090). Promoted 2026-07-05 on the cross-rig validation the header required — #584 (@ryanmpelletier, 4× 3090): verify-full 9/9, verify-stress clean to 240K, soak PASS, bench n=5. The 4 cards buy concurrency — KV pool 1.77M/6.77× vs the 2-card 622K/2.37× (single-stream decode ~flat). Quality is TP-invariant, carried from the vllm/dual proxy (109/150); a 4-card 8-pack confirmation is the open follow-up.",
    ),
    "vllm/qwen-27b-multi-max": _entry(
        model="qwen3.6-27b", weights_variant="fp8", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e4m3",
        tp=4, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/multi4/fp8/mtp.yml",
        default_port=8015,
        kvcalc_key="SKIP",
        status="production",
        status_note="Qwen3.6-27B 'max accuracy' tier, 4-card (TP=4): official FP8 weights + fp8/e4m3 KV (flipped from int8-PTH alongside dual-max #594/#595) + MTP n=3 @262K. Byte-identical to vllm/qwen-27b-dual-max apart from TP=4 + gpu-count (dual-max @TP=2 is the on-rig proxy; this dev rig has 2 cards). Promoted 2026-07-07 on the clean v0.24.0 4-card validation the caveats required — #584 (@ryanmpelletier, 4× 3090 x16): verify-full 9/9, verify-stress clean to 240K, soak PASS, bench n=5, 8-pack 111/150 (ties the dual-max proxy 109 → quality TP-invariant). Corroborated by a 2nd 4-card rig — #625 (@MoppelMat, 4× 3090 mixed x4/x8, 300 W: decode 79/102, prefill-90K clean, soak PASS). fp8/e4m3 routes KV attention to FlashInfer (int8-PTH is Triton-only) → flat decode at depth. TP=4 buys the 6.77x KV pool; single-stream decode ~flat vs 2-card.",
    ),

    # Qwen3.6-27B NVFP4 (nvidia modelopt) — the community-validation Hopper/
    # Blackwell tier. AUTHORED BLIND on this sm_86 dev rig (NVFP4 cannot boot
    # here): required_sm=9.0 gates launch to NVIDIA's supported set (Hopper
    # sm_90 / Blackwell sm_100+ incl. 5090 sm_120, GB10 sm_121); the first
    # community booter is the validation, not a confirmation. NVFP4 *KV* stays
    # off everywhere (consumer Blackwell has no FP4 FMHA — see hardware
    # rtx-5090.yml note); fp8_e4m3 KV is the FP4-era KV. NOTE (corrected
    # 2026-07-06): hf_quant_config DECLARES kv_cache_quant_algo=FP8 but the
    # checkpoint ships NO k_scale/v_scale tensors (index-verified) → runs at
    # scale=1.0, the #594-quality-tied regime. Same for the 35B-A3B sibling.
    "vllm/qwen-27b-single-nvfp4": _entry(
        model="qwen3.6-27b", weights_variant="nvfp4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e4m3",
        tp=1, max_ctx=65536, max_num_seqs=1, mem_util=0.85,
        compose_path="models/qwen3.6-27b/vllm/compose/single/nvfp4/mtp.yml",
        default_port=8076, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="qwen3.6-27b:nvfp4-single",
        status="experimental",
        status_note="Qwen3.6-27B NVFP4 (nvidia modelopt MIXED_PRECISION: NVFP4 FFN + FP8 attention + FP8 KV scales + unquantized MTP head), single Hopper/Blackwell card (native sm_90+ — H100 / 5090 / RTX 6000 Pro / GB10; fallback_sm=7.5: sub-9.0 cards RUN it via the Marlin W4A16 weight-only fallback since vLLM v0.24 — no native-FP4 speed edge, and this single-card config needs >24 GB VRAM regardless). 🧪 AUTHORED BLIND on the sm_86 dev rig, community-validated on two 5090s (#613 @guybrush01 + #617 @paulp83). Root cause of the original OOM was MTP, not ctx: MTP-on at 98K left no room for the draft head + cudagraphs + GDN prefill scratch. Default is now MTP-on + MAX_MODEL_LEN=65536 + mem_util=0.85 — the config that keeps MTP's ~2x AND fits (verify-stress all-pass, 131/155 TPS decode, MTP accept ~3.2, ~1.4 GB VRAM free @ 65K). SPEC=off trades MTP for more ctx (81K/98K @ 71 TPS) and is the tight-system-RAM path (MTP's draft load OOM-kills a 28 GB host, #617). 80 GB+ cards raise MAX_MODEL_LEN toward 262K with MTP on. fp8/e4m3 KV runs scale=1.0 (FP8 KV declared in hf_quant_config, NO k_scale/v_scale tensors shipped — index-verified, the #594-tied regime). ~2.5x smaller than bf16, NVIDIA MMLU-Pro/GSM8K deltas <1% vs bf16 per the model card. 8-pack think-off measured 2026-07-11 on the dual sibling (sm_86 Marlin fallback, weight-identical): 110/150 — ties the fp8 tier's 109; native-FP4 activation path still owed (stays 🧪). No DEFAULTS row (opt-in only).",
    ),
    "vllm/qwen-27b-dual-nvfp4": _entry(
        model="qwen3.6-27b", weights_variant="nvfp4", workload="long-ctx-single",
        engine="vllm-stable", drafter="qwen-mtp-builtin", kv_format="fp8_e4m3",
        tp=2, max_ctx=262144, max_num_seqs=2, mem_util=0.92,
        compose_path="models/qwen3.6-27b/vllm/compose/dual/nvfp4/mtp.yml",
        default_port=8077, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="qwen3.6-27b:nvfp4-dual",
        status="experimental",
        status_note="Qwen3.6-27B NVFP4 (nvidia modelopt MIXED_PRECISION — see single-nvfp4) at TP=2 @262K full ctx, 2x Hopper/Blackwell native (the 2x 5090 configuration is the primary community target; fallback_sm=7.5). LIVE-VALIDATED ON 2x3090 sm_86 2026-07-11 via the Marlin W4A16 fallback: boots @262K + fp8 KV + MTP n=3 (accept 97%+), 69.7 narr / 85.5 code decode TPS, 23.77 GB/card, 8-pack think-off 110/150 — statistically TIES the fp8 production tier's 109 (weight-identical path; native-FP4 activations unmeasured). On Ampere it works but has NO edge (~20% slower than the AutoRound tier for the same model) — prefer AutoRound/fp8 there; this slug's value on sub-sm_90 cards is models/cases where NVFP4 is the only quant. First native-FP4 community boot + rebench-full still wanted (funnel). Mirrors vllm/qwen-27b-dual-max's shape (TP=2 + MTP n=3 + fp8/e4m3 KV + vision @262K) with NVFP4 weights instead of FP8: ~11 GB/card weights vs dual-max's 14.5 — bigger KV pool headroom on 32 GB cards. On native-FP4 GEMM parts (Blackwell) NVIDIA claims near-fp8 throughput at 2.5x less weight memory. No DEFAULTS row (opt-in only).",
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
        weights_companions=("gguf_mmproj_f16",),  # mmproj vision projector the compose mounts
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
        weights_companions=("gguf_mmproj_f16",),  # mmproj vision projector the compose mounts
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
        weights_companions=("anbeeld-dflash-iq4xs",),  # DFlash draft GGUF the compose mounts
        default_port=8060,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="Single-GPU default. Launchers inject the beellama engine pin (Anbeeld's official v0.3.2-preview digest — engines/beellama-local.yml install.spec); sm_86 verified on this rig 2026-07-04 (verify-full all-pass), sm_89 runs the sm_80 cubins. 5090/sm_120: official images build on CUDA 12.4 and carry NO sm_120 (upstream ask Anbeeld#85) — self-build with CUDA_DOCKER_ARCH=120 + CUDA_VERSION=12.8.1 (engine notes have the verified recipe) or the unmaintained v0.3.0-feature-level snapshot ghcr.io/noonghunna/beellama-cpp:multiarch-v0.3.0-efe856397. Usable ctx ceiling 160K (200K OOMs on prefill); ships 102K. DFlash prose is net-positive on tok/s (+27% vs no-spec, re-tested 2026-06-03); the earlier 'prose-DFlash regression' is RETRACTED — it was an AR over-read + wrong baseline (docs/UPSTREAM.md).",
    ),

    # Qwopus3.6-27B-Coder (Jackrong coder fine-tune of Qwen3.6-27B) — single 3090, Q5_K_M
    # GGUF + embedded MTP head + KVarN-4 KV. The first KVarN compose; needs the v0.3.2
    # preview KVarN engine build (digest-pinned in engines/beellama-local.yml).
    "beellama/qwopus-coder": _entry(
        model="qwen3.6-27b", weights_variant="qwopus-coder-mtp-q5km", workload="fast-chat",
        engine="beellama-local", drafter="qwopus-mtp-gguf", kv_format="kvarn4",
        tp=1, max_ctx=163840, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/single/qwopus-coder-mtp-q5km/mtp.yml",
        default_port=8067,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwopus3.6-27B-Coder (Jackrong) Q5_K_M GGUF + EMBEDDED MTP head (--spec-type draft-mtp) + KVarN-4 KV, single 3090. Launch with --force (experimental). REQUIRES the KVarN engine build (beellama v0.3.2 PREVIEW, digest-pinned) — the v0.3.0/earlier images reject --cache-type-k kvarn4. 2026-06-12 on sm_86: embedded MTP head loads, verify-full all-pass, NIAH needle @72K (= q5_0/q4_1 control), bench ~46/58 TPS narr/code (≈ q5_0/q4_1 — KVarN decode-neutral), 8-pack quality 104/103 think-off/on ≈ q5_0/q4_1 102/107 (quality-neutral; disc #329). Ships 160K (MTP-on ceiling; 230K via the no-MTP env opt-in in the compose). Launcher path (switch.sh --force) + soak-continuous PASS (0-growth, 0/25 silent-empty, 100% retention). beellama v0.3.2 is a rolling PRE-RELEASE → stays experimental (un-park on a stable Anbeeld tag; full verify-stress NIAH ladder pending for ⚠️ promotion).",
    ),

    # Carnice-V2-27B (stuchapin — Hermes-style agentic SFT of Qwen3.6-27B) — single 3090,
    # Q5_K_M GGUF + embedded MTP head + KVarN-4 KV. Sibling of beellama/qwopus-coder; the
    # embedded head is mtp_num_hidden_layers=1 so DRAFT_N_MAX=1 (author warns n=3 is wrong).
    "beellama/carnice-v2-single-q5km-mtp": _entry(
        model="qwen3.6-27b", weights_variant="carnice-v2-q5km", workload="fast-chat",
        engine="beellama-local", drafter="carnice-mtp-gguf", kv_format="kvarn4",
        tp=1, max_ctx=163840, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/single/carnice-v2-q5km/mtp-kvarn4.yml",
        default_port=8068,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Carnice-V2-27B (stuchapin — kai-os/Carnice-V2-27b, Hermes-style agentic SFT of Qwen3.6-27B) Q5_K_M GGUF + EMBEDDED MTP head (--spec-type draft-mtp, n=1) + KVarN-4 KV, single 3090. Launch with --force (experimental). REQUIRES the KVarN engine build (beellama v0.3.2 PREVIEW, digest-pinned) — v0.3.0/earlier reject --cache-type-k kvarn4. FULLY VALIDATED 2026-06-14 (rebench-full, reasoning-on): engine-compat PASS (beellama LOADS the PR#22673-fused GGUF — the card's 'mainline fails to load' does NOT apply), verify-full all-pass, verify-stress 8/8 (NIAH clean to 150K), soak PASS (0-growth, 0/100 silent-empty, 100.5% retention, p50 49.5 TPS), bench 46.8/50.5 TPS narr/code, MTP accept ~94%. 8-pack reasoning-on 110/150 — BEATS sibling beellama/qwopus-coder 103/150 (edge is agentic/instruct: hermes 13 vs 10, instructfollow 15, reasonmath 13; dataextract 10 = Qwen-family number-format floor). Ships 160K (MTP-on default; 176K measured ceiling, 192K OOMs; 230K via the no-MTP env opt-in). n=1 default; n=2 is a +12%-TPS opt-in (DRAFT_N_MAX=2, doesn't crash on our single-card kvarn4 unlike the author's 2×3090) pending a dedicated soak. beellama v0.3.2 is a rolling PRE-RELEASE → stays experimental (un-park on a stable Anbeeld tag, #455).",
    ),
    "beellama/carnice-v2-dual-q8-mtp": _entry(
        model="qwen3.6-27b", weights_variant="carnice-v2-q8", workload="fast-chat",
        engine="beellama-local", drafter="carnice-mtp-gguf", kv_format="q8_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/beellama/compose/dual/carnice-v2-q8/mtp-q8kv.yml",
        default_port=8070,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Carnice-V2-27B Q8_0 + EMBEDDED MTP head (--spec-type draft-mtp, n=1) + q8_0/q8_0 KV, DUAL 3090 (layer-split -ts 0.55,0.45). The dual / higher-quant follow-through requested in #403 (Q5_K_M single = beellama/carnice-v2-single-q5km-mtp). FULLY VALIDATED 2026-06-16 (rebench-full): verify-full ALL-PASS, bench n=5 narr 40.7/code 44.0 decode TPS, verify-stress 8/8 (NIAH->240K), soak fresh 20x5 PASS (0 growth, 0/100 silent-empty, p50 42.2, 100% retention), 8-pack think-OFF 103/150 / think-ON 105/150 (in-band, q8_0 KV held quality). MTP accept ~81%, full 262K fits @ -ts 0.55,0.45 (21.9/21.2 GB/card, ~2.5GB free). KV A/B: chose q8_0 over the originally-spec'd kvarn6 — q8_0 prefills +17% (1003 vs 860 t/s; kvarn6's software-compression compute throttled prefill), higher-fidelity (q8 > q6-class), reference path (kvarn6 = Anbeeld 'experimental'), fits 262K on dual. -b/-ub/--no-mmap A/B'd FLAT; n=2 = +13% validated-stable opt-in (DRAFT_N_MAX=2). DFlash A/B RULED OUT (base-27B drafter ~10% accept on the fine-tune; no Carnice-matched drafter exists) → MTP-only. Launch --force. beellama v0.3.2 rolling pre-release → experimental (#455).",
    ),
    # Qwen3.6-27B-MTP-pi-reasoning (bytkim) — reasoning fine-tune, Q4_K_M GGUF + embedded
    # MTP head + q4_0/q4_0 KV, single 3090, reasoning-ON, mainline llama.cpp. Chosen over a
    # beellama/q4_0-q4_1 path: mainline decodes ~25% faster on identical weights (~41 vs ~33
    # t/s), trading beellama's 262K unified-KV ceiling for speed at a ~200K mainline ceiling.
    "llamacpp/qwen27b-pi-reasoning-single": _entry(
        model="qwen3.6-27b", weights_variant="pi-reasoning-q4km", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q4_0",
        tp=1, max_ctx=200000, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-27b/llama-cpp/compose/single/pi-reasoning-q4km/mtp.yml",
        default_port=8063,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Qwen3.6-27B-MTP-pi-reasoning (bytkim — 'Pi-style' reasoning-supervised CODING/terminal-agent fine-tune of Qwen3.6-27B) Q4_K_M GGUF + EMBEDDED MTP head (--spec-type draft-mtp, n=2) + q4_0/q4_0 KV, single 3090, reasoning-ON, on MAINLINE llama.cpp (server-cuda-b9246, PR #22673). Launch with --force (experimental). CONFIG FOLLOWS THE MODEL CARD: temp 1.0 / top-p 0.95 / top-k 0 / min-p 0 (NOT the stack's 0.6/20), reasoning ON, q4_0/q4_0 KV. Card recommends MTP n=3; on-rig A/B found n=2 marginally faster (within noise) — kept n=2, MTP_DRAFT_N_MAX=3 matches the card. Card notes presence-penalty 1.5 for DIRECT/instruct (REASONING=off) mode. CONTEXT (measured 2026-06-17): 200K-alloc fills ~188K usable with correct needle recall (22.7 GB / ~1.8 GB free); decode ~23 t/s at ~188K depth. Do NOT alloc 262K — FA scratch grows with alloc, so 262K OOMs at ~176K (LESS usable than 200K); full 262K usable is beellama-only. Author TESTED only 128K, so 128-188K is engine-proven but past the card's validated window (CTX_SIZE=131072 for strict compliance). FULL REBENCH-FULL VALIDATED 2026-06-18 (--with-8pack-thinking=both): bench @370W NARRATIVE 47.4/47.9, CODE 54.2/55.3 wall/decode, PP 1030 tok/s (n=5, CV<2%); @230W cap 28.5/32.9 (mainline -42% 370->230W — power-sensitive). verify-stress 8/8 (NIAH recall to 183K @ 91% of the 200K pool). 8-pack 104/150 think-off / 106/150 think-on (cohort: carnice 110, ik 107, qwopus 103). soak PASS (0 MiB growth, 0/100 silent-empty, p50 54.5, 102% retention). The MTP head is NOT weaker than base: a matched-power A/B (230W) put it DEAD-EVEN with base Qwen3.6-27B MTP (73% vs 72% accept, identical decode), and 47.9/55.3 @370W is ~on par with base 50.3/58.9 (within ~5% on canonical prompts). Verbose even thinking-off (a few deterministic-pack misses were finish_reason=length truncations — give it generous max_tokens). MTP gives ~+43% over no-MTP. Mainline llama.cpp = no patches, follows upstream.",
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
        weights_companions=("gguf_mmproj_f16",),  # mmproj vision projector the compose mounts
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
    "llamacpp/hauhaucs-35ba3b-dual": _entry(
        model="qwen3.6-35b-a3b", weights_variant="morikomorizz-q6kp", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/mtp.yml",
        default_port=8073,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Uncensored HauhauCS-Aggressive 35B-A3B (morikomorizz Q6_K_P, embedded MTP head) on mainline llama.cpp b9570, dual-card -ts 0.55,0.45 q8_0 KV @ 262K. MTP loads clean (the prior HauhauCS-MTP ret=-3 was ik-llama/older-build, not mainline). Validated 2026-06-14: verify-stress 8/8 (NIAH→240K), bench.sh n=3 @262K (narr 113.4 / code ~150 decode TPS, CV<1%; n=3 vs n=1 = -9% prose/+10% code), soak 20x5 PASS (0 growth, 0/100 silent-empty, p50 162.4, 99.6% retention). 8-pack think-OFF 103/150, think-ON 105/150 (wash). Maintainer-set non-default knobs: MTP n=3 (code-max, -4% prose vs n=1) + REASONING=on (vs stack thinking-off default). Community GGUF, digest-UNPINNED + uncensored → experimental. No DEFAULTS row (opt-in only).",
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
        status="deprecated",
        status_note="DEPRECATED 2026-07-02 (v0.24.0 consolidation): superseded by vllm/gemma-31b-dual (cyankiwi bf16 @224K on STOCK v0.24.0, overlay-free). This rode vllm-gemma-stable v0.22.0 + the #42006 overlay at 131K bf16; the v0.24.0 bf16 path reaches ~224K on stock with no overlay, so this no longer earns its keep. Kept (not deleted) for history.",
    ),
    "vllm/gemma-int8-mtp": _entry(
        model="gemma-4-31b", weights_variant="autoround-int4", workload="multi-stream-tenant",
        engine="vllm-gemma-stable", drafter="gemma-it-assistant", kv_format="int8_per_token_head",
        tp=2, max_ctx=262144, max_num_seqs=4, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/autoround-int4/int8.yml",
        default_port=8032, required_engine_features=["int8_per_token_head"],
        kvcalc_key="gemma-4-31b:gemma-dual-int8",
        status="deprecated",
        status_note="DEPRECATED 2026-07-02 (v0.24.0 consolidation): the 262K int8-PTH+#40391 path on vllm-gemma-stable v0.22.0. Superseded by vllm/gemma-31b-dual (bf16 @224K, stock v0.24.0, overlay-free) as the single dual slug. NOT migrated to v0.24.0 — int8-PTH silently craters recall past ~32K there (#40391 open/unmerged upstream, verified 2026-07-01). Kept for history + as the 262K reference; when PR #40391 merges, a v0.24.0 int8-PTH 262K path can return overlay-free. required_engine_features/kv_format kept so pull-gate + launch-compat assertions stay meaningful.",
    ),

    # Gemma-4-31B cyankiwi QAT-AWQ-INT4 (compressed-tensors, lm_head bf16) — the v0.24.0
    # OVERLAY-FREE dual on vllm-stable. On v0.24.0 the autoround/qat-w4a16 duals live on
    # vllm-gemma-stable (autoround crashes tie_weights, qat-w4a16 carries #40391/#42006);
    # cyankiwi's lm_head-excluded checkpoint dodges tie_weights, #40391 is native, and the
    # native ParserEngine handles tools — so this folds onto vllm-stable, no overlays.
    # MTP DISABLED (Gemma-4 MTP×tools broken on v0.24.0 — vLLM #39043/#42006; see compose caveat).
    "vllm/gemma-31b-dual": _entry(
        model="gemma-4-31b", weights_variant="qat-awq-int4", workload="multi-stream-tenant",
        engine="vllm-stable", drafter=None, kv_format="bf16",
        tp=2, max_ctx=229376, max_num_seqs=2, mem_util=0.95,
        compose_path="models/gemma-4-31b/vllm/compose/dual/qat-awq-int4/base.yml",
        default_port=8032,
        kvcalc_key="gemma-4-31b:gemma-dual",
        status="caveats",
        status_note="Gemma-4-31B cyankiwi QAT-AWQ-INT4 (compressed-tensors, lm_head bf16), dual TP=2 BF16 KV @224K, stock vLLM v0.24.0 OVERLAY-FREE — the consolidation 31b (folds onto vllm-stable, retires the 31b's vllm-gemma-stable dependence). BF16 (not int8-PTH): on v0.24.0 int8-PTH allocates 262K but SILENTLY craters recall past ~32K (needs #40391, open/conflicting upstream — verified 2026-07-01, both cyankiwi + w4a16 crater identically; the SAME cyankiwi weights on v0.22.0+#40391 recall clean to 112K+). bf16 KV is overlay-free + no cliff. rebench-full VALIDATED 2026-07-02 @0.95/229376: verify-full 9/9 (tools+streaming-tool-calls+reasoning clean, MTP-off), verify-stress ALL 5 ceiling rungs to 210K (91%) with healthy VRAM margin 1162MB>1024 (the 0.97/245K thin-margin flag is resolved at 0.95/224K), bench decode ~59 TPS (CV 0.1%, TTFT 69ms), soak PASS (0 err · 0 MiB growth · 0/100 silent · p50 58.7 · 99.6% retention). tie_weights dodged (lm_head excluded), gemma4 tool+reasoning parsers native (#45588). CAVEATS: (1) MTP DISABLED — Gemma-4 MTP×tool-calling broken on v0.24.0 (vLLM #39043; #42006 closed-unmerged); (2) ~224K ceiling (bf16 ~2×/tok vs int8-PTH's 262K) — int8-PTH+#40391 (262K) returns free when #40391 merges. Supersedes gemma-int8-mtp/gemma-bf16-mtp/qat-w4a16 for v0.24.0. 8-pack deferred (maintainer call).",
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
        status="deprecated",
        status_note="DEPRECATED 2026-07-02 (v0.24.0 consolidation): QAT-int4 data point on vllm-gemma-stable v0.22.0, superseded by vllm/gemma-31b-dual (bf16, stock v0.24.0) as the single dual slug. Kept for history. ORIGINAL NOTE: Gemma-4-31B unsloth QAT W4A16 (compressed-tensors int4) + int8-PTH KV (#40391) + assistant MTP n=4, dual TP=2 @262K. Boots clean on stock vllm-gemma-stable (NO #44494 workaround — tower-based Gemma4ForConditionalGeneration, unlike the 12B unified arch). 8-pack A/B 2026-06-07: 109/150 vs the autoround-int4 int8.yml's 105 (+4, within ±5-7 8-pack noise ≈ tie; real instructfollow edge IF 15-vs-8 offset by hermes/cli/TC/RM). WEAKER spec-decode though: MTP accept-len ~2.3 (n=3, the n-swept default) vs autoround's ~3.9 → less speedup. n-swept @370W: n2 72.9/86.1 · n3 74.0/87.7 · n4 71.6/87.8 (all within ~3%; n=3 = top narr + tied code + ~20% less drafting → set as the default vs autoround's n=4, since the QAT-int4's fast acceptance decay favors a lower n). Still slower than autoround (106/139 @230W — power not matched; the AL gap is the clean signal). Comparable QUALITY but slower — autoround-int4 (gemma-int8-mtp) stays the clear default. NIAH/soak not yet run.",
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
        engine="vllm-stable", drafter=None, kv_format="bf16",  # v0.24.0: gemma4_unified native (#44429) → vllm-stable; MTP off (Gemma-4 MTP×tools broken #39043/#42006)
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
        status_note="Gemma-4-12B Q8_K_XL single-3090 on beellama.cpp (q5_0(K)/q4_1(V) KV, 256K via --override-kv, no spec-dec). NIAH-clean to 246K. Launchers inject the beellama engine pin (Anbeeld's official v0.3.2-preview digest — engines/beellama-local.yml install.spec; no sm_120 until Anbeeld#85). Preview = rolling pre-release → experimental; promote on a stable tag.",
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
        weights_companions=("anbeeld-dflash-iq4xs",),  # DFlash draft GGUF the compose mounts
        default_port=8061,
        kvcalc_key="SKIP",
        status="caveats",
        status_note="Single-GPU default — the only viable fast single-card Gemma-4 path on Ampere. Launchers inject the beellama engine pin (Anbeeld's official v0.3.2-preview digest — engines/beellama-local.yml install.spec); sm_89 runs the sm_80 cubins. 5090/sm_120: official images build on CUDA 12.4 and carry NO sm_120 (upstream ask Anbeeld#85) — self-build with CUDA_DOCKER_ARCH=120 + CUDA_VERSION=12.8.1 (engine notes) or the unmaintained v0.3.0-feature-level snapshot ghcr.io/noonghunna/beellama-cpp:multiarch-v0.3.0-efe856397. DFlash prose is net-positive on tok/s (+28–31% vs no-spec, re-tested 2026-06-03; earlier 'prose regression' RETRACTED — AR over-read + wrong baseline); re-point to mainline llama.cpp#23398 Gemma-4 MTP when it merges — docs/UPSTREAM.md.",
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
        weights_companions=("anbeeld-dflash-iq4xs",),  # DFlash draft GGUF the compose mounts
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
        weights_companions=("anbeeld-dflash-iq4xs",),  # DFlash draft GGUF the compose mounts
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
        weights_companions=("anbeeld-dflash-iq4xs",),  # DFlash draft GGUF the compose mounts
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
        engine="vllm-stable", drafter=None, kv_format="bf16",  # MTP off on v0.24.0 (Gemma-4 MTP×tools broken, vLLM #39043/#42006)
        tp=2, max_ctx=262144, max_num_seqs=256, mem_util=0.92,
        compose_path="models/gemma-4-26b-a4b/vllm/compose/dual/awq/mtp.yml",
        default_port=8041,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="AWQ + external MTP (gemma-26b-it-assistant n=4) on stock v0.22.0 — MTP +55% TPS (134->208, AL 3.55) validated 2026-06-05. Max ctx 262K (model max; KV pool 806,821 tok at 262144/0.92, 2x 3090) boot+coherence validated 2026-06-06. Promote after rebench-full + soak.",
    ),
    "vllm/diffusiongemma-dual": _entry(
        model="diffusiongemma-26b-a4b", weights_variant="fp8", workload="fast-chat",
        engine="vllm-diffusion-gemma", drafter=None, kv_format="bf16",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.82,
        compose_path="models/diffusiongemma-26b-a4b/vllm/compose/dual/fp8/base.yml",
        default_port=8042,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="DiffusionGemma dLLM (vLLM's first) on Ampere via the OFFICIAL vllm/vllm-openai:gemma image (digest-pinned; dgemma arch baked in) + 3 bind-mounted Ampere/TP fixes (marlin-K-pad x2 + diffusion_gemma TP-vocab/dtype) — NOT in :gemma since vLLM tests H100/TP=1. Eager-only, gemma4 tool+reasoning parsers. 262K (NIAH->250K), 8-pack 100/150 (5-pack 84%), ~177/180 TPS typical (peak ~1100 low-entropy). max_new_tokens lifted 256->16384 (the model self-terminates ~1.2-1.8K words; no one-shot 10K). Experimental: visible in --list (NA), launch needs --force. Supersedes the 123-file sideload (PR #358); re-pin+rebase the 3 fixes if :gemma is re-pushed. 2026-06-11.",
    ),
    # DEFAULTS: intentionally NOT added — 'experimental' is non-functional, so it
    # degrades out of the curated <model>/default walk; reachable only by explicit
    # slug `vllm/diffusiongemma-dual` (launch requires --force).
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

    # Qwen3.6-35B-A3B NVFP4 (nvidia modelopt MoE) — community-validated
    # Hopper/Blackwell tier, sibling of the 27B nvfp4 pair. AUTHORED BLIND on
    # this sm_86 rig (required_sm=9.0 gates launch). MoE + unified-memory is
    # the marquee pairing: 3B active params suit big-capacity/lower-bandwidth
    # parts (GB10 Spark), so the SINGLE slug is the primary ask there. NO MTP:
    # our measured finding on this MoE is that the built-in head shares the
    # MoE forward and is NET-NEGATIVE (-51%; learnings/qwen3.6-35b-a3b.md) —
    # base serving only. fp8/e4m3 KV at scale=1.0 — SAME as the 27B nvfp4
    # (FP8 KV declared in hf_quant_config, no scale tensors shipped,
    # index-verified 2026-07-06; the #594-quality-tied regime).
    "vllm/qwen-35b-a3b-single-nvfp4": _entry(
        model="qwen3.6-35b-a3b", weights_variant="nvfp4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e4m3",
        tp=1, max_ctx=131072, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/single/nvfp4/fp8.yml",
        default_port=8078, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="qwen3.6-35b-a3b:nvfp4-single",
        status="caveats",
        status_note="Qwen3.6-35B-A3B NVFP4 (nvidia modelopt MIXED_PRECISION MoE: NVFP4 expert FFNs + FP8 attention; ~23.4 GB), single Hopper/Blackwell card (native sm_90+; fallback_sm=7.5 — sub-9.0 runs via the Marlin W4A16 fallback, validated on the 27B sibling 2026-07-11, but this 23.4 GB single-card config needs a 32 GB card regardless). Authored on the sm_86 dev rig, FIRST community validation on a single RTX 5090 in #619 (@paulp83): boots clean on vLLM v0.24.0 (quant=modelopt_mixed), verify-full 9/9 (tool-calls + streaming + reasoning), verify-stress needle-clean (9.8K + 29K), soak-continuous PASS (15 MiB growth / 100% retention / 0 err / p50 311 TPS). Decode 255.8 narr / 257.9 code TPS @ 60 ms TTFT — ~2.5-3x our 2x3090 AutoRound tier (native FP4 GEMM). VRAM 30.6/32 GB @131K — TIGHT but flat through soak on a 32 GB card. ⚠️ PROMOTED to Production w/ caveats 2026-07-09 on TWO independent 5090 validations (#619 @paulp83 + #612/#652 @guybrush01, within noise: verify-full 9/9, verify-stress to ~120K, soak PASS ~311 TPS). CAVEAT: 8-pack quality NOT yet cross-rig-measured (sandboxes weren't built on these runs; quality run pending, gated on #492) — reverts to 🧪 if a quality run regresses. THE GB10/DGX-Spark single-card ask: 3B-active MoE suits unified-memory parts — GB10 128 GB runs the full 262K via MAX_MODEL_LEN env (131K default sized for 5090 32 GB). NO MTP by design: the built-in head is net-negative on this MoE (-51% measured on the AutoRound tier; @paulp83's speculative_config=None confirms it loaded MTP-off). fp8/e4m3 KV @ scale=1.0 (FP8 KV declared in hf_quant_config, no scale tensors shipped — same as the 27B nvfp4). No DEFAULTS row (opt-in only).",
    ),
    "vllm/qwen-35b-a3b-dual-nvfp4": _entry(
        model="qwen3.6-35b-a3b", weights_variant="nvfp4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e4m3",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/dual/nvfp4/fp8.yml",
        default_port=8079, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="qwen3.6-35b-a3b:nvfp4-dual",
        status="experimental",
        status_note="Qwen3.6-35B-A3B NVFP4 (see single-nvfp4) at TP=2 @262K full ctx, 2x Hopper/Blackwell native (2x 5090 primary community target; ~11.7 GB/card weights; fallback_sm=7.5 — sub-9.0 cards run it via the Marlin W4A16 fallback, validated on the 27B sibling 2026-07-11, though on Ampere the AutoRound tier serves this model faster). 🧪 first community boot + rebench-full validates. Mirrors vllm/qwen-35b-a3b-dual's shape (no drafter — MTP net-negative on this MoE, vision on, thinking off) with NVFP4 weights + fp8/e4m3 KV instead of AutoRound + e5m2. No DEFAULTS row (opt-in only).",
    ),

    "vllm/qwen-35b-a3b-dual-nvfp4-fast": _entry(
        model="qwen3.6-35b-a3b", weights_variant="nvfp4-fast", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e4m3",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.92,
        compose_path="models/qwen3.6-35b-a3b/vllm/compose/dual/nvfp4-fast/fp8.yml",
        default_port=8081, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="qwen3.6-35b-a3b:nvfp4-dual",
        status="caveats",
        status_note="Qwen3.6-35B-A3B NVFP4-Fast (unsloth compressed-tensors MIXED: true W4A4 NVFP4 expert FFNs + FP8-dynamic attention; quant auto-detects — NOT modelopt), TP=2 @262K. THE AMPERE-VALIDATED NVFP4 PATH — inverse of dual-nvfp4: FIRST-PARTY VALIDATED on the reference 2x3090 2026-07-11 (first MoE-FP4 fallback boot anywhere, MARLIN NvFp4 MoE backend): decode 179.5/179.4 (n=5, CV<=0.8%) + 8-pack think-off 103/150 = DOUBLE STATISTICAL TIE with the AutoRound tier (182.3/182.3, 104-equiv) at full 262K, 22.46 GB/card; cli-40 20/40 = best measured on this MoE. See BENCHMARKS 2026-07-11. PROMOTED to Production w/ caveats 2026-07-11 on the full gate: verify-stress 8/8 (NIAH to 240,635 = 91%, ceiling margin 1,801 MB) + soak-continuous PASS (0 err, 0 growth, 100% retention) + bench + 8-pack. CAVEATS: streaming-toolcall+thinking-on finish=length (known family class, verify-full check 6; non-streaming unaffected); native-FP4 quality unvalidated (numbers = Ampere W4A16 bound). NATIVE FP4 (sm_90+) UNVALIDATED — there the silicon quantizes activations too (true W4A4; family is activation-quant-sensitive), so the native quality number is the arc's missing datapoint. Ships real calibrated k/v scale tensors (nvidia's export ships none) and they LOAD (in-worker verified 2026-07-11 on the 27B sibling); measured effect vs scale=1.0 on this family: none (27B A/B quality/NIAH tie). mtp.* head shipped unquantized but OFF (net-negative on this MoE at TP=2). On Ampere, pick the AutoRound tier unless you specifically want the NVFP4 artifact. No DEFAULTS row (opt-in only).",
    ),

    # Agents-A1 — InternScience's 35B agentic MoE (Qwen3-Next MoE arch, OWN model
    # per its card's base_model; NOT a qwen fine-tune slug). Official FP8-dynamic
    # compressed-tensors checkpoint; on Ampere sm_86 vLLM serves it Marlin FP8-MoE
    # WEIGHT-ONLY (no native FP8 compute — activation quant inert; sm_89+ runs the
    # real checkpoint). No MTP head shipped (safetensors-verified) → drafter-free.
    # T2 producer-zero validation 2026-07-03: brought via pull.sh route-C sibling
    # swap + generate-compose (first model onboarded through the lane end-to-end).
    "vllm/agents-a1-dual": _entry(
        model="agents-a1", weights_variant="fp8-dynamic", workload="long-ctx-single",
        engine="vllm-stable", drafter=None, kv_format="fp8_e5m2",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=0.92,
        compose_path="models/agents-a1/vllm/compose/dual/fp8-dynamic/fp8.yml",
        default_port=8072,
        kvcalc_key="agents-a1:agents-a1-dual",
        status="caveats",
        status_note="Agents-A1 FP8-dynamic dual (TP=2) @262K — the agentic thinking-ON specialist. Full gate PASS 2026-07-03 (rebench agents-a1-fp8-dual): verify-full ✓ · bench 154.0/153.8 decode TPS (TTFT ~129ms, CV 0.1%) · verify-stress 8/8 incl. staggered NIAH exact-recall to 240K (91% of 262K, VRAM Δ0 across the ladder) · soak-continuous PASS (0 growth, 0/100 silent-empty, 99.8% retention) · 8-pack OFF 105/150 / ON 110/150 (post benchlocal #79+#81 harness). vs the qwen3.6-35b-a3b incumbent: general capability TIES (ON 110=110), ~13% slower decode (154 vs 178) — but cli-40 thinking-ON 23/40 vs 17/40 (+6, the highest cli-40 on this stack) + toolcall 15/15 (OFF). CAVEATS: (1) hermes-20 REGRESSES with thinking ON (12→9 — genuine model behavior: wrong-path rm -rf claimed as success, duplicate cron; verified not-harness), (2) Ampere = weight-only FP8 (activation quant inert), (3) not a general upgrade — reach for it on tool/CLI-agent work with thinking ENABLED (that's where the 23/40 lives), (4) Blackwell sm_120: upstream v0.24.0 kernel bug crashes boot — uncomment VLLM_TEST_FORCE_FP8_MARLIN=1 in the compose (#548). Vision retained. Dual-only (36 GB weights).",
    ),

    # Qwen3.6-40B-Deckard — dense 40B uncensored community merge, llama.cpp dual.
    # First dual llama.cpp compose in the catalog. Q6_K GGUF (31 GB) requires both
    # cards; layer-split via -ts 1,1. MTP n=2 sweet spot (41.6 tok/s, 0.81 accept).
    # kv_format q8_0 (K+V). kvcalc SKIP (llama.cpp family — no vLLM kv-calc).
    "llamacpp/deckard40B-dual-mtp": _entry(
        model="qwen3.6-40b-deckard", weights_variant="piehsoft-q6k", workload="fast-chat",
        engine="llama-cpp-local", drafter="qwen-mtp-builtin", kv_format="q8_0",
        tp=2, max_ctx=131072, max_num_seqs=1, mem_util=None,
        compose_path="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/mtp.yml",
        default_port=8199,
        kvcalc_key="SKIP",
        status="production",
        status_note="Dense 40B uncensored Qwen3.6 merge (Q6_K MTP GGUF, 31 GB) on dual 3090 llama.cpp. Arch CONFIRMED qwen35-dense (standard GQA, 97 layers) from the GGUF header. MTP n=2 sweet spot (~41.6 tok/s, 0.81 accept). 128K ctx ceiling @q8_0 KV (192K OOMs). Dual-only. verify-full 8/8, verify-stress 8/8, 8-pack 105/150 (MTP off==on, spec-dec lossless), soak-continuous PASS (0 MiB growth, 0/25 silent-empty). First uncensored + first dual-llama.cpp compose in the catalog.",
        category="uncensored",
    ),

    # Tess-4-27B — Qwen3.5-based dense 27B (migtissera Q4_K_M GGUF), llama.cpp dual.
    # First EXTERNAL-MTP compose in the catalog: the nextn head ships as a SEPARATE
    # GGUF (mtp-Tess-*.gguf), engaged via --spec-draft-model + --spec-type draft-mtp
    # (contrast Deckard's embedded head). kv_format q4_0 (K+V). kvcalc SKIP.
    "llamacpp/tess-dual-mtp": _entry(
        model="tess-4-27b", weights_variant="migtissera-q4km", workload="fast-chat",
        engine="llama-cpp-local", drafter="tess-mtp-gguf", kv_format="q4_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/tess-4-27b/llama-cpp/compose/dual/migtissera-q4km/mtp.yml",
        default_port=8020,
        kvcalc_key="SKIP",
        status="production",
        status_note="Tess-4-27B (migtissera Q4_K_M GGUF, 16 GB) — Qwen3.5-based dense 27B instruct/agentic fine-tune on dual 3090 llama.cpp. Arch qwen35-dense (dense = non-MoE; HYBRID attention, 48 linear + 16 full — corrected 2026-07-11) — same family as Deckard-40B. EXTERNAL MTP n=2 (separate mtp-*.gguf draft via --spec-draft-model, spec_method mtp_gguf) — first external-draft compose in the catalog. q4_0 KV, 262K ctx. Live-validated 2026-07-09 on server-cuda-b9246: decode ~52 narrative / 68 code tok/s (TTFT 233 ms), prefill ~1.3K tok/s; verify-stress 8/8 (NIAH ladder clean to 240,634 tok = 91% of 262K, ~5.9 GB free at deepest fill); soak-continuous PASS (0 err, 0/100 silent-empty, p50 66.4 tok/s, 96.3% retention). Quality (benchlocal --full): core 8-pack 115/150 (77%) think-off, 118/150 (79%) think-on — ties-to-edges the qwen3.6-27b dual-max reference (109) and LEADS the agentic packs (hermesagent 15/20 vs 9, cli-40 25/40 vs 20). PROMOTED caveats->production 2026-07-12: the streaming+thinking finish=length caveat does NOT reproduce on the shipped b9967 + 16K-reasoning-budget config (3/3 clean incl. parallel 2-tool) and the shipped-config quality refresh passed both modes (OFF 116 / ON 117 single-draw, no pack regression). Trades ~1/2 the qwen-dual throughput for a quality tie/edge + vision-capable base + smaller footprint (~12.7+17.2 GB layer-split vs ~22 GB/card TP=2).",
    ),

    "vllm/tess-dual-nvfp4": _entry(
        model="tess-4-27b", weights_variant="nvfp4", workload="fast-chat",
        engine="vllm-stable", drafter=None, kv_format="fp8_e4m3",
        tp=2, max_ctx=131072, max_num_seqs=2, mem_util=0.92,
        compose_path="models/tess-4-27b/vllm/compose/dual/nvfp4/fp8.yml",
        default_port=8082, required_sm=9.0, fallback_sm=7.5,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Tess-4-27B NVFP4 (migtissera compressed-tensors, W4A4 recipe; Marlin W4A16 weight-only on Ampere) at TP=2 @131K + fp8 KV, spec-off — THE FASTEST TESS on 2x24GB: 62.4 tok/s decode (CV 0.2%) vs the llama.cpp catalog entry's 57.9 with MTP, and the first vLLM-servable Tess on consumer cards (validated 2026-07-11, BENCHMARKS). Froggeric template pinned (repo template stock-broken). Drafter-less BY FORENSIC RESULT: grafted MTP head = 0% accept in vLLM (works only token-fed in llama.cpp), EAGLE3 ~40% = net-negative on this trunk (club #662). kvcalc SKIP: Tess is a qwen35 HYBRID (KV on 16/64 layers) and no hybrid kv-calc model exists for it yet — follow-up at promotion. A0 8-pack LANDED 2026-07-12: 106/150 off / 113/150 on — UNDERSHOOTS the GGUF bar (116/117), gap cli-40-concentrated (17 vs 25 off); deterministic packs tie+ (RM-off 14/15 best-ever). Pin-fallback TRIGGERED per the pre-registered rule: huginnfork NVFP4A16 weights-only A/B + FP8 W8A16 precision arm are the next gates; stress/soak also still unrun. Slug stays experimental; llamacpp/tess-dual-mtp (now Production) remains the tess recommendation.",
    ),

    # Ornith-1.0-9B — DeepReinforce agentic-coding RL fine-tune. Qwen3-Next DENSE-FFN
    # HYBRID (arch=qwen35: 8 full-attn + 24 GDN/DeltaNet layers, NON-MoE) — only 8/32
    # layers carry GQA KV so 262K fits at 4.25 GiB KV. No MTP head → drafter-free
    # ngram self-spec on ik_llama. Lean single-card 9B. 🧪 niche (loses to gemma-12b).
    "ik-llama/ornith9b-single": _entry(
        model="ornith-1.0-9b", weights_variant="deepreinforce-q4km", workload="fast-chat",
        engine="llama-cpp-local", drafter=None, kv_format="q8_0",
        tp=1, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/ornith-1.0-9b/ik-llama/compose/single/deepreinforce-q4km/ngram.yml",
        default_port=8070,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Ornith-1.0-9B (DeepReinforce agentic-coding RL) on ik_llama single 3090, Q4_K_M + q8_0 KV, full 262K. Arch CONFIRMED qwen35 Qwen3-Next DENSE-FFN HYBRID (8 full-attn + 24 GDN/DeltaNet layers, NON-MoE) — only 8/32 layers carry GQA KV, so 262K fits at 4.25 GiB KV (13.4 GiB total, ~10 GiB headroom). No MTP head → drafter-free ngram self-spec (--spec-type ngram-map-k, ~0.59 accept, +30% warm on copy-heavy gen; works DESPITE the DeltaNet hybrid, where DFlash/EAGLE are KV-rollback-blocked). Full gate PASS 2026-06-25: bench ~102/103 TPS (TTFT 144ms, CV<1%), verify-stress 8/8 incl NIAH→0.92×262K, soak-continuous PASS (0 MiB growth, 0/100 silent-empty, p50 104 TPS, 98.1% retention). 8-pack think-OFF 91/150 / think-ON 95/150 (temp 0.6). cli-40 7/40 is partly a termination-protocol artifact (model solves but mis-routes the <solution> sign-off via the bash tool + loops → agent_loop_exhausted; a hardened agent prompt lifts it 7→13/40). NICHE ONLY — gemma-4-12b beats it on quality (105/150) AND speed (117/122 vs 102 TPS) at +7 GiB; pick Ornith only for the lean 13.4 GiB footprint / 16 GB-card fit. Self-grabbed official GGUF, ik digest-pinned → 🧪.",
    ),

    # Ornith-1.0-35B — DeepReinforce agentic-coding RL fine-tune of Qwen3.6-35B-A3B
    # (qwen35moe MoE hybrid, 40L, 256 experts / 8 active ~3B, NO MTP head). Dual 3090,
    # Q8_0 GGUF, full 262K, drafter-free ngram (opt-in; Q8 dual is tight). 🧪 — ties the
    # base on the 8-pack, EDGES it on aider (15/30 vs 12-13) → coding-leaning lane.
    "ik-llama/ornith35b-dual": _entry(
        model="ornith-1.0-35b", weights_variant="deepreinforce-q8", workload="fast-chat",
        engine="llama-cpp-local", drafter=None, kv_format="q8_0",
        tp=2, max_ctx=262144, max_num_seqs=1, mem_util=None,
        compose_path="models/ornith-1.0-35b/ik-llama/compose/dual/deepreinforce-q8/ngram.yml",
        default_port=8071,
        kvcalc_key="SKIP",
        status="experimental",
        status_note="Ornith-1.0-35B (DeepReinforce agentic-coding RL fine-tune of Qwen3.6-35B-A3B) on ik_llama dual 3090, Q8_0 GGUF, full 262K. Arch CONFIRMED qwen35moe (MoE hybrid, 40L: 10 full-attn + 30 GDN, 256 experts / 8 active, ~3B active of 34.66B) — only ~10 layers carry GQA KV so 262K KV is ~2.7 GB. NO MTP head → drafter-free ngram (--spec-type ngram-map-k, OPT-IN: Q8 weights nearly fill dual so ngram needs n_max=32 + a balanced -ts to fit 262K). Full gate PASS 2026-06-26: bench ~108.8/108.6 TPS, verify-stress 8/8 (NIAH→240K), soak-continuous PASS (0 growth, 0/100 silent-empty, p50 112). 8-pack think-OFF 105/150 / think-ON 105/150 — TIES the base qwen3.6-35b-a3b (byteshape 110) within noise; thinking adds nothing. EDGES the base on real coding: aider-polyglot-30 15/30 (OFF==ON) vs the base's 12-13/30, corroborated by bugfind 15/15. Coding-leaning 35B-A3B; the base stays the pick for general use. Self-grabbed official GGUF, ik digest-pinned → 🧪.",
    ),

    # VibeThinker-3B — WeiboAI verifiable-reasoning fine-tune of Qwen2.5-Coder-3B
    # (Qwen2 dense). First dense-family + first sub-4B model in the catalog.
    "vllm/vibethinker-3b-single": _entry(
        model="vibethinker-3b", weights_variant="bf16", workload="long-ctx-single",
        engine="vllm-stable", drafter=None, kv_format="fp8_e5m2",
        tp=1, max_ctx=131072, max_num_seqs=1, mem_util=0.40,
        compose_path="models/vibethinker-3b/vllm/compose/single/bf16/fp8.yml",
        default_port=8074,
        kvcalc_key="SKIP",
        status="incubating",
        status_note="VibeThinker-3B (WeiboAI) — bf16 Qwen2 dense verifiable-reasoning model (SFT+RL fine-tune of Qwen2.5-Coder-3B) on a single 3090, vLLM v0.22.0 (vllm-stable). bf16 weights (~5.8 GB) + fp8_e5m2 KV (storage-only A/B'd 2026-06-16 — math/code answers identical to bf16 KV; halves cache, quality-neutral). full 131072 ctx, ~110 TPS. Single-concurrency sized: max_num_seqs=1 + mem_util 0.40 → ~9.8 GB total (174K-token / 1.33x KV pool, full 131K kept), freeing ~14 GB to co-reside with a 27B; ~9.8 GB is near the floor (5.8 GB bf16 weights immovable). Output quality is temp-governed not mem-governed: temp 0.6 coherent, the card's temp 1.0 is unstable on short prompts (degenerate loops) — overridable via TEMP. fp8 WEIGHTS rejected (break this quant-sensitive 3B: non-terminating empty output). Sampling per the tech report: temp 1.0 / top_p 0.95 / top_k -1. Live-validated 2026-06-16: serves clean, correct reasoning + code (verify-full output-quality + thinking-mode PASS); --reasoning-parser qwen3 splits <think> blocks correctly. ⚠️ ALWAYS-REASONING: it emits a <think> trace before every answer with NO way to disable it (system prompt / /no_think ignored; authors document no controls). Omit max_tokens (vLLM's large default → reasons briefly then answers) or set generously (8K-40K; authors use up to 40960); a small explicit max_tokens truncates mid-reason with no answer. NO tool-calling (emits bare JSON not <tool_call>; authors don't support it — intentionally unwired). Consequence: FAILS verify-full's fixed-small-budget checks (basic 30 / streaming 120 / tool 200-256 tok) → 5/9 — a harness-vs-always-reasoning mismatch, NOT a serving defect. Stays 🐣 incubating: does not pass the standard functional gate; math/code/STEM reasoning only, not general/agentic.",
    ),
    "llamacpp/vibethinker-3b-single": _entry(
        model="vibethinker-3b", weights_variant="prithivmlmods-q8", workload="long-ctx-single",
        engine="llama-cpp-local", drafter=None, kv_format="q8_0",
        tp=1, max_ctx=131072, max_num_seqs=1, mem_util=None,
        compose_path="models/vibethinker-3b/llama-cpp/compose/single/prithivmlmods-q8/q8kv.yml",
        default_port=8075,
        kvcalc_key="SKIP",
        status="incubating",
        status_note="VibeThinker-3B (WeiboAI) — prithivMLmods Q8_0 GGUF on a single 3090, mainline llama.cpp (server-cuda). Q8 weights + q8_0/q8_0 KV, full 131072 ctx, -b 4096 -ub 2048. The PERFORMANCE-MAX VibeThinker path: live-validated 2026-06-16 ~166 TPS decode (vs ~110 vLLM bf16), prefill ~6,630 tok/s (-b 4096/-ub 2048 = +20% over 2048/512; -ub is the lever, -b 8192 adds nothing), ~6.2 GB at full 131K (3.3 GB Q8 + ~2.6 GB q8_0 KV), CV 0.1%, stable at temp 0.6 AND 1.0. KEY: llama.cpp Q8_0 is near-lossless → quality INTACT, where vLLM's fp8 weight-quant broke this quant-sensitive 3B (non-terminating empty output) — so on llama.cpp you get the quantization win with no quality cost. NO MTP head in this GGUF (plain qwen2 conversion) → no self-spec-dec (not needed at this speed). Reasoning: emits <think>...</think> but llama.cpp's deepseek parser does NOT split it (Qwen2.5 template doesn't declare reasoning) → trace stays inline in content (answer after </think> clean). ⚠️ ALWAYS-REASONING + NO tool-calling → FAILS verify-full's fixed-small-budget + tool checks (5/9, same as the vLLM sibling) — by design, not a serving defect. 🐣 incubating: math/code/STEM reasoning specialist, not general/agentic. Better-performing sibling to vllm/vibethinker-3b-single. Tool-free quality (benchlocal, temp 0.6, 2026-06-16, thinking-ON): gsm-symbolic-30 30/30 (100%), reasonmath-15 12/15 (80%); one-shot coding (sandbox executes, no tool-calls) humaneval-plus-30 29/30 (97%) + lcb-v6-30 25/30 (83%, 2 losses=token_limit on hardest); instructfollow-15 15/15 (100%), structoutput-15 12/15 (80%), dataextract-15 6/15 (40%). The packs' thinking-OFF defaults had zeroed dataextract / capped structoutput at 60% via token_limit truncation (always-reasoning vs bounded budget); --enable-thinking restores them. dataextract's residual 40% is genuine — full config sweep {temp 0/0.6/1.0 x budget 16K/32K} all land 27-40% (0.6 is the sweet spot; greedy and 1.0 both = 27%), same value-mismatch + type-coercion failures → config can NOT recover it; valid JSON, field-level extraction errors (reasoning specialist, not an extractor). Sweep also validates the compose default temp 0.6 (beats 0 and 1.0). toolcall/hermes/cli N/A (no tool-calling).",
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
    # Dual default is gemma-31b-dual: cyankiwi bf16 @224K on STOCK vLLM v0.24.0, OVERLAY-FREE
    # (⚠️ Production w/ caveats, validated 2026-07-02 — verify-full 9/9 + verify-stress to 210K with
    # healthy VRAM margin + soak PASS). Supersedes the v0.22.0 gemma-int8-mtp (int8-PTH+#40391, 262K),
    # now DEPRECATED: int8-PTH silently craters recall on v0.24.0 (#40391 open/unmerged upstream) — the
    # 262K int8-PTH path returns when #40391 merges. bf16 trades ~224K vs 262K for zero overlays.
    ("gemma-4-31b", "vllm", "dual"): "vllm/gemma-31b-dual",
    ("gemma-4-26b-a4b", "vllm", "single"): "vllm/gemma-26ba4b-single",
    ("gemma-4-26b-a4b", "vllm", "dual"): "vllm/gemma-26ba4b-dual",
    ("qwen3.6-35b-a3b", "vllm", "single"): "vllm/qwen-a3b-preview-single",
    ("qwen3.6-35b-a3b", "vllm", "dual"): "vllm/qwen-35b-a3b-dual",
    # Deckard: only one compose (dual llama.cpp MTP), so the dual default is trivial.
    ("qwen3.6-40b-deckard", "llamacpp", "dual"): "llamacpp/deckard40B-dual-mtp",
    # Tess-4-27B: single dual llama.cpp compose (external MTP); ⚠️ caveats = functional.
    ("tess-4-27b", "llamacpp", "dual"): "llamacpp/tess-dual-mtp",
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
