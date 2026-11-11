# w4a8-inc-moe

> ⚠️ **Not ported to vLLM 0.33.0** (the `vllm-stable` pin from 2026-11-10). 0.33 rewrote
> `inc_wna16_scheme.py` (318 → 265 lines) and the `if use_marlin:` block this patch anchors on is gone, so
> with `W4A8=1` the installer refuses and the boot stops. The default, W4A16, never runs it. Porting means
> re-deriving the four fixes below against 0.33's MoE path and re-checking them end to end on 64 GB cards.

Makes `VLLM_MARLIN_INPUT_DTYPE=int8` (W4A8) work for Marlin MoE experts. Companion to
`qwen3.6-27b/vllm/patches/w4a8-int8-act`, which fixes the dense Marlin linears; run it
after that one. No-op without int8 activations.

**Symptom.** On stock v0.31.0 with `w4a8-int8-act`, Qwen3.8-Flash-Next (AutoRound/INC,
MoE) answers every prompt with `!!!!!!!!`, and a compressed-tensors MoE drafter drops to
about 1.8 accepted tokens per step.

**Bugs fixed** (details in `patch_w4a8_inc_moe.py`):

1. INC MoE never gets the input dtype: `_resolve_gptq_moe` builds `AutoGPTQMoEMethod`
   directly, so its experts are packed for 16-bit activations while `MarlinExperts`
   reads the env and feeds int8.
2. `AutoGPTQMoEMethod` asserts 8-bit weights under int8 activations. The check is
   inverted (the dense path requires 4-bit under the same message).
3. The per-layer int8 global scale from `marlin_act_int8_process_scales` is registered
   but never reaches the kernel, in both `AutoGPTQMoEMethod` and
   `CompressedTensorsWNA16MoEMethod`.
4. Negative AutoRound group scales: Marlin's int8 path stores scales as unsigned int16.
   Folded at load as in the dense patch (q -> min(16 - q, 15), s -> |s|), gated on the
   scale sign because AutoRound ships qzeros even for symmetric weights.

**Checked.** Kernel-level test against a torch reference at the model's shapes (hidden
2560, expert width 640, top-10): relative error 0.6% at 16-bit activations, 2.1% int8.
End to end on 2x CMP 170HX: GSM8K (400, greedy, thinking off) 386 vs 388 for W4A16;
draft acceptance 3.52.

**Is it worth it here?** Mixed. Prefill +6-9%; decode +15% on code, -3..5% on prose, and
a c=8 GSM8K run took 280 s vs 257 s. Off by default in the CMP compose.

**Upstream.** Not yet filed; add `docs/UPSTREAM.md` rows (four bugs) before filing.

**Re-anchoring.** Seven anchors across `fused_moe/oracle/int_wna16.py`,
`quantization/inc/schemes/inc_wna16_scheme.py`, `quantization/auto_gptq.py` and
`compressed_tensors_moe/compressed_tensors_moe_wna16.py`. All are checked before any file
is written; missing or not unique -> the installer exits 1.
