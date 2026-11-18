# vllm-humming-w4a8 — W4A8 on vLLM v0.34.0's humming backend

v0.34.0 removed Marlin's 8-bit activations (vllm#65542), and with them the
`VLLM_MARLIN_INPUT_DTYPE=int8` toggle our AutoRound INT4 tiers ran on.
`VLLM_MARLIN_INPUT_DTYPE` is now a startup error if it is set at all. W4A8
moves to `--quantization humming` with
`VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"int8"}'`. The composes select it
from `W4A8` (default `1`); `W4A8=0` serves Marlin W4A16 under
`--quantization auto_round` and does not run this patch.

On our checkpoints humming needs two fixes, and one log line proves int8 is in
use. `install.sh` runs `apply.py`, which patches
`vllm/model_executor/layers/quantization/humming.py` in place:

| # | What | Upstream | Without it |
|---|---|---|---|
| 1 | Run `UnquantizedLinearMethod.__init__` after the weight loader's fallback swaps `self.__class__` | issue vllm#68194, fix in open PR vllm#68202 (the same lines) | Load fails with `'UnquantizedLinearMethod' object has no attribute '_gemm_impl'`. `HummingConfig` ignores AutoRound's `extra_config`, so the BF16 layers (vision tower, `linear_attn.in_proj_ba`, `mtp.fc`: 318 per TP pair on Qwen3.8-27B) take the fallback. |
| 2 | Drop the scalar placeholder `weight` that `create_weights` registers, before `replace_layer_parameters` | issue vllm#68195, filed by us; no upstream PR | `replace_layer_parameters` keeps each layer's checkpoint `qweight` storage beside the converted `weight`: 15.17 GiB of weights a card instead of 9.19, and 262,144 context no longer fits on 24 GB cards. |
| 3 | Print `[humming-w4a8] effective activations a_dtype=…` once per worker, after `check_and_fallback_input_schema` | diagnostic, ours | Nothing in vLLM's log names the activation dtype humming settled on, and `/proc/<pid>/environ` is overwritten by `setproctitle`, so int8 could not be confirmed. |

**Measured** (2026-11-17, Qwen3.8-27B AutoRound INT4, 2× RTX 3090, 230 W, v0.34.0
with `--no-enable-extensible-kv-cache`): with fixes 1 and 2, `dual-fast` loads
9.17 GiB a card, pools 602,678 tokens at 262,144 and passes verify-full (MTP
3.05, vision 4/4). Its prefill is 2,555 / 1,626 tok/s at 10K / 90K, against
2,030 / 1,409 for v0.33.0's Marlin W4A8 and 1,476 / 1,114 for v0.34.0's W4A16.
Decode is level. Rows: `BENCHMARKS.md` and `learnings/qwen3.8-27b.md` (stack docs).

**Behaviour.** Idempotent: each change carries the marker `# club3090 humming-w4a8`.
A fix already present upstream is reported and skipped: fix 1 when the
`__init__` call follows the swap, fix 2 when `create_weights` no longer
registers the scalar placeholder. A missing anchor for fix 1 or 2 stops the
boot. A missing anchor for the log line only warns.

**Don't** add `"allow_fallback": true` to the input config: on sm_86 with g128
weights it silently reverts to 16-bit activations.

**Drop triggers.** Fix 1: vllm#68202 merged and in the pinned release. Fix 2:
an upstream fix for vllm#68195 in the pinned release. Then delete that part of
`apply.py`. The log line goes with the last fix.
