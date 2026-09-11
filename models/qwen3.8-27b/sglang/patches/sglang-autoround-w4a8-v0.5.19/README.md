# SGLang AutoRound W4A8 (v0.5.19 re-cut)

**What it does:** adds INT8-activation (W4A8) GEMM to SGLang's AutoRound INT4
path. Weights stay the AutoRound INT4 (group 128, symmetric) — activations run
as INT8 on the tensor cores via the Marlin kernel family, on top of SGLang's
existing `gptq_kernels.py` AutoRound dispatch. On Ampere (sm_86) this is the
dense-GEMM route to W4A8; no FP8 compute needed (which sm_86 lacks).

**Activation is unconditional-with-patch.** Unlike the vLLM sibling (which is
gated behind `VLLM_MARLIN_INPUT_DTYPE=int8`), the SGLang patch flips
`quant_args_marlin.experimental_w4a8 = True` on every Marlin AutoRound layer,
with no env var. So *applying the patch* is what turns W4A8 on — a container
with the patch applied but the shape-guard/`in_proj_ba` data fix absent will
serve a *mix* of W4A8 and W4A16 layers (the non-conformant GDN shards fall back),
never a silent all-BF16 path. The installer hard-fails on a partial patch, so
the "half-wired" state is caught at boot, not in the field.

**Provenance:** re-cut of jb-seo's `0004-autoround-w4a8.patch` from the
club-3090 contribution (cut against SGLang **v0.5.18**) onto **v0.5.19** —
the first release with DFlash2 upstream, so one engine carries W4A8 + DFlash2.
Adaptations vs the v0.5.18 cut:

| File | Change |
| --- | --- |
| `gptq_kernels.py` | byte-identical v0.5.18→v0.5.19 — applies verbatim |
| `auto_round.py` | 1-line context fix: `if isinstance(layer, (LinearBase, ParallelLMHead)):` → `if is_linear:` (v0.5.19 refactored the gate) |
| 8 new files (`gptq_marlin_w4a8.py` + 7 CUDA sources) | new files, no context dependency |

## The shape guard — this PR's real fix (NOT in jb-seo's cut)

AutoRound's W4A8 eligibility gate omitted the per-shard `K%128==0` /
`N%64==0` constraint that `prepare_w4a8` hard-asserts. On the Qwen3.8-27B
GDN (hybrid linear-attention) layers the merged `in_proj_ba` tensor is
**N=96 → 48/rank at TP4**, which is *not* 64-divisible. Without the guard a
naive W4A8 pass crashes in `gptq_marlin_repack` with
`size_n=48 is not divisible by 64` at weight load.

The guard pre-checks each shard's K/N **before** routing it to the W4A8
path; non-conformant shards log

```
[Marlin] W4A8 skipped (shard ... not 128/64 divisible) — using standard Marlin W4A16
```

and fall back to standard Marlin W4A16 for that layer. Consequence on
Qwen3.8-27B at TP4: the GDN merged-projection layers run W4A16 (their N is
the problem), everything else (the bulk of the dense FFN/attention weight)
runs W4A8. That is the measured configuration the benches in
`results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/` were taken with.

**Necessity was proven by isolation:** stock v0.5.19 *without* the guard,
run on the same checkpoint, crashes identically at `gptq_marlin_repack.cuh:
311: size_n=48`. The rest of the patch is exonerated — the guard is the fix.

## Checkpoint data fix (separate from the patch — DO NOT SKIP)

Stock SGLang crashes on the Avuja Qwen3.8-27B AutoRound checkpoint at
weight-load **even without W4A8**: the GDN `in_proj_a`/`in_proj_b` layers are
BF16 in the checkpoint (`extra_config` bits:16) but SGLang builds a merged
`in_proj_ba` module and routes it through GPTQ-Marlin, which asserts
`size_n % 64 == 0`. The fix is **data-side**, in the target's
`config.json`:

```json
"extra_config": {
  ".*in_proj_ba.*": {"bits": 16, "data_type": "fp"}
}
```

Avuja's published checkpoint already carries this (backup of the pristine
file: `config.json.bak-sglang-test`). If you point the compose at another
AutoRound Qwen3.8-27B checkpoint (biMEMO, Vishva007, …), apply the same
one-liner first or the stack will crash-loop at weight load. This is a
checkpoint property, not a patch property — the patch cannot and does not
fix it.

## Install

`install.sh` (mounted at `/etc/club3090/w4a8/` by the compose) is idempotent
and content-marker-based (works whether or not the SGLang tree is a git
repo):

- 3/3 markers present → "already applied — skipping"
- 0/3 → applies the patch (`git apply` if the tree is a repo, `patch(1) -p1`
  otherwise), then re-checks
- 1–2/3 → hard fail: partial state, recreate the container

It is invoked from the compose entrypoint **before** `sglang serve`.
A failed apply is fatal by design: a half-wired W4A8 would serve a mix of
INT8- and BF16-activation layers silently.

## Pinned image

`lmsysorg/sglang:v0.5.19` (2026-09-05 release, the DFlash2 cut). Per the
repo's engine-pinning policy (AGENTS.md "Engine image pinning"), this compose
pins the exact release the patch was cut against — a rolling tag would break
the `auto_round.py` anchor the moment upstream touches that gate again.

## Validation status

- **v0.5.19 isolation run (no W4A8, no DFlash2):** reproduced the
  `size_n=48` Marlin crash on stock — the guard's target.
- **Full boot (TP4, DFlash2 n=8, fp8 KV, bf16 SSM, 262K ctx):** all quantized
  linears prepared W4A8 with 0 skip-errors, DFlash2 accept length ~4.4–4.5,
  KV pool 731,047 tokens (fp8), mamba pool 84 slots (bf16), `max_running
  requests = 16` not capped. Coherent output verified (math exact, prose
  coherent) — W4A8 numerics are clean on this checkpoint.
- **Bench:** `results/sglang-q38-ar-w4a8-dflash2-tp4-20260908/` — full
  bench-ultimate run (prefill ladder 512→16K, decode saturation c=1..16,
  narrative + code) head-to-head vs the vLLM 0.27.1 baseline on the same
  model/quant/drafter. SGLang wins c=1 decode by ~45%; vLLM closes the gap
  by c=8; prefill within 2–17%. See the BENCHMARKS.md row for the headline
  numbers.
