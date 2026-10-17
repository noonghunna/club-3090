# qwen4exp-pp-ple-rank0

Lets Qwen3.8-Flash-Next run with pipeline parallelism (`--pipeline-parallel-size 2`) on the
stock `vllm/vllm-openai` image.

**Why.** Stock vLLM (checked on v0.31.0 and v0.32.0rc3) refuses PP > 1 for any Qwen4Exp
model with an n-gram PLE, because later pipeline ranks never receive the raw `input_ids`
the PLE reads. Flash-Next has one PLE layer (`ple_layer_ids: [2]`, decoder layer 1), so it
is always on rank 0 and the refusal is stricter than needed.

**What it changes** (marker `# [club3090 qwen4exp-pp-ple-rank0]`, three edits):

| File (under the vllm package) | Change |
|---|---|
| `model_executor/models/config.py` | refuse PP only when a PLE layer is outside rank 0 |
| `models/qwen4_exp/nvidia/model_state.py` | same check, worker side |
| `v1/core/kv_cache_utils.py` | a KV group listing every rank's layers allocates only this rank's |

**Provenance.** A port of the PP parts of Minachist's `flash-next-vllm.patch` (Apache-2.0),
from `huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, `vllm-patch/`.
Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>

**Re-anchoring at a pin bump.** Each anchor must occur exactly once; otherwise the installer
exits 1 and the container does not start. Find the new text of each check in the installed
vLLM and update `EDITS` in `patch_pp_ple_rank0.py`.

Status: 🧪 experiment (dual CMP 170HX). Not yet in `scripts/lib/profiles/patches.yml`.
