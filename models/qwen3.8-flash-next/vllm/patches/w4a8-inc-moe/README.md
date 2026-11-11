# w4a8-inc-moe

> **Ported to vLLM 0.33.0** (2026-11-11). 0.33 moved the INC GPTQ MoE builder out of `inc_wna16_scheme.py`
> into `INCWNA16MoEScheme._build_gptq_method` in `inc_wna16_moe.py`, with the same logic one indent deeper.
> The script patches that file when it exists and `inc_wna16_scheme.py` otherwise, so v0.32.0 still applies.
> The other three target files are byte-identical between the two releases.
>
> **Checked on 0.33:** offline on the image, and again on v0.32.0:
> - it applies after `w4a8-int8-act`;
> - a second run is a no-op, and `verify` passes;
> - the patched modules import;
> - with the env unset, nothing is touched.
>
> A CPU check showed the 0.33 builder now hands the method `torch.int8`; unpatched, it gets `None`.
>
> **Not checked on 0.33:** a boot, the kernel-level test and GSM8K. The reference rig has no 64 GB cards
> and no AutoRound MoE checkpoint. The figures below are v0.31.0's.

Makes `VLLM_MARLIN_INPUT_DTYPE=int8` (W4A8) work for Marlin MoE experts. Companion to
`qwen3.6-27b/vllm/patches/w4a8-int8-act`, which fixes the dense Marlin linears; run it
after that one. No-op without int8 activations.

**Symptom.** On stock v0.31.0 with `w4a8-int8-act`, Qwen3.8-Flash-Next (AutoRound/INC,
MoE) answers every prompt with `!!!!!!!!`, and a compressed-tensors MoE drafter drops to
about 1.8 accepted tokens per step.

**Bugs fixed** (details in `patch_w4a8_inc_moe.py`):

1. INC MoE never gets the input dtype: `_resolve_gptq_moe` (from 0.33,
   `INCWNA16MoEScheme._build_gptq_method`) builds `AutoGPTQMoEMethod` directly, so its experts are packed for 16-bit activations while `MarlinExperts`
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

**Upstream.** `docs/UPSTREAM.md` has the row. The third-party PRs for bugs 1, 3 and 4 were closed
unmerged on 2026-11-04, and bug 2 has none.

**Re-anchoring.** Seven anchors across `fused_moe/oracle/int_wna16.py`, the INC file
(`quantization/inc/schemes/inc_wna16_moe.py` from 0.33, `inc_wna16_scheme.py` before it),
`quantization/auto_gptq.py` and `compressed_tensors_moe/compressed_tensors_moe_wna16.py`. All are checked before any file
is written; missing or not unique -> the installer exits 1.
