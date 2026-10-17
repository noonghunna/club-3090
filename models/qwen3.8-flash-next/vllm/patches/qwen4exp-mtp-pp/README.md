# qwen4exp-mtp-pp

Lets the Qwen3.8-Flash-Next MTP drafter run with `--pipeline-parallel-size > 1` on stock vLLM.

**Bug.** The drafter lives on the last PP rank only and the speculator hands it the target's
hidden states directly, but `Qwen4ExpMTP` model `forward` branches on the *global* PP rank. On
any rank but the first it takes the intermediate-tensors path, which the speculator never
fills: `assert intermediate_tensors is not None` (`models/qwen4_exp/nvidia/mtp.py`) in the
memory-profiling dummy run. Hit on vLLM v0.31.0, PP=2, 2x CMP 170HX, 2026-10-17.

**Fix** (three hunks): branch on whether hidden states were passed; give the drafter its own
input embedding under PP (the target's copy is on rank 0 and is not shared across ranks);
build the draft lm_head with the checkpoint's quant config so a quantized head can load.

**Provenance.** The `mtp.py` hunks of Minachist's `flash-next-mtp-01-enable.patch`
(Apache-2.0, `vllm-patch/3x24GB+MTP/`). Its `qsa.py` / `platforms/interface.py` hunks belong
to Minachist's host-KV mode, which we do not use, and are left out.
Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>
