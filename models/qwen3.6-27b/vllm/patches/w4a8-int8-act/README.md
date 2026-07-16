# W4A8 int8-activation enablement (Qwen3.6-27B AutoRound) — #609

`VLLM_MARLIN_INPUT_DTYPE=int8` is a **stock vLLM** serve-time toggle (since ≤v0.22):
it dynamic-quantizes activations to int8 per token and runs the W4A16 Marlin GEMMs
on the INT8 tensor cores — the "W4A8" everyone wants on Ampere. Two upstream bugs
block it on this model's checkpoints; these patches fix both. Measured on the
reference 2×3090 (dual `fp8-mtp`, MTP n=3, 262K; single-variable env-only A/B through this exact
compose): **prefill +50% (1,674→2,513 tok/s) · decode neutral (+~3%, within noise) ·
MTP acceptance unchanged · 8-pack 110/150 off / 111/150 on — ties the W4A16
baseline on both legs.** Full forensics + community discussion: club-3090 #609.

| file | fixes |
|---|---|
| `patch-inc-w4a8.py` | vLLM routes `quant_method: auto-round` onto its `inc` path, which builds the Marlin method without threading `input_dtype` → the env is **silently ignored** for every AutoRound checkpoint. |
| `patch-negscale-fold.py` | Marlin's int8 path quantizes group scales to int16 but the kernel reads them **unsigned**; AutoRound legitimately emits ~50% **negative scales** (fine in W4A16) → garbage. Load-time sign-fold (`q → clamp(16−q,0,15)`, `s → |s|`; exact except q==0 in negative groups ≈ 0.61% of weights — 8-pack-verified harmless). Also: an actionable error replaces the misleading `"W8A8 is not supported"` assert for asymmetric checkpoints, and a warn-and-proceed guard covers any asym route that survives it. |
| `install.sh` | Runs both, idempotent. **No-op without the env** — safe to run on every boot. Hard-fails the boot only if the anchors drifted *and* the user asked for int8 (never a silent half-wire). |

Enable per launch:

```bash
VLLM_MARLIN_INPUT_DTYPE=int8 bash scripts/switch.sh vllm/dual
```

Requirements: **positive-symmetric int4** weights (the shipped autoround checkpoint
qualifies via the fold; asymmetric AWQ refuses loudly). Qwen validated on the composes' fp16 AND
bf16; prefer bf16 for other model families (Gemma overflows fp16 under int8 dequant). Compressed-
tensors positive-sym checkpoints need no patches at all (the hook is stock there).

Upstream status: both bugs to be filed (see docs/UPSTREAM.md rows); anchors verified
byte-identical on v0.24.0 and v0.25.1 (Marlin csrc unchanged v0.23→v0.25.1).
