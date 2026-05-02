# PFlash Integration Feasibility

Date: 2026-05-02

Scope: evaluate Luce-Org PFlash as a possible long-context prefill accelerator
for the club-3090 vLLM stack: vLLM nightly
`0.20.1rc1.dev16+g7a1eb8ac2`, Sandermage Genesis runtime patches, RTX 3090
SM86, AutoRound INT4, TQ3/fp8 KV, Qwen3.6-27B.

Verdict: PFlash is not fundamentally incompatible with vLLM if treated as a
lossy prompt-compression stage before target admission. It is fundamentally
incompatible with a simple Genesis/kernel-only port. The hard part is not the
block-sparse kernel; it is adding a prefill-time second-model coordinator,
tokenizer handoff, request rewrite, memory scheduling, and quality policy to
vLLM.

## 1. PFlash Architecture Decoded

The implementation lives mostly under `lucebox-hub/dflash/`; `pflash/` is the
bench/server harness. The public README says PFlash is an in-process
speculative-prefill path: a Qwen3-0.6B BF16 drafter scores a long prompt, then
the Qwen3.6-27B Q4_K_M target only prefills selected spans. Runtime is the
`dflash` C++/CUDA daemon, not Python/PyTorch/Triton.

Important source paths:

- `dflash/src/qwen3_drafter.{h,cpp}`: `DrafterContext`,
  `load_drafter()`, `free_drafter()`, `drafter_score_and_compress()`.
- `dflash/src/qwen3_0p6b_drafter.h`: Qwen3-0.6B drafter weight layout and
  `forward_qwen3_0p6b_drafter()`.
- `dflash/src/qwen3_0p6b_graph.cpp`: custom drafter forward, K/Q capture, tail
  scoring.
- `dflash/src/flashprefill.{h,cpp}` and `flashprefill_kernels.cu`:
  `mean_K -> score -> select -> sparse_fwd`.
- `dflash/src/bsa_launcher.cu`: converts PFlash block indices to BSA
  blockmask and dispatches FA2-derived Block-Sparse-Attention.
- `dflash/scripts/_prefill_hook.py`: server-side tokenizer handoff and
  `park/unpark/free drafter` dance.
- `dflash/test/test_dflash.cpp`: daemon commands `compress`, `generate`,
  `park`, `unpark`, `free drafter`.

Drafter lifetime:

- In the normal `test_dflash` daemon path, the drafter is lazy-loaded on the
  first `compress <ids.bin> <keep_x1000> <drafter.gguf>` command and can be
  released by `free drafter`.
- The server helper intentionally parks target/draft weights, runs
  compression, frees the drafter, and unparks target/draft so this fits on
  24 GB.
- The phase-split benchmark daemon `dflash/test/pflash_daemon.cpp` is a
  separate mode that loads the Qwen3-0.6B drafter once and only serves
  compression requests.

Drafter scoring:

- This is a full pass of the small drafter over the full source prompt, not a
  lexical heuristic.
- `qwen3_0p6b_graph.cpp` captures per-layer K for all source tokens and Q for
  the last `n_lookahead` tokens. Then it computes tail attention
  `Q_tail @ K^T / sqrt(D)`, applies softmax with the tail mask, takes max over
  heads/layers, and produces `running_max[n_lookahead, S]`.
- `qwen3_drafter.cpp` reduces that to per-token scores by mean over lookahead,
  applies 1D average-pool smoothing, then scores chunks.

Selection:

- There are two different selection mechanisms.
- Inside the drafter's FlashPrefill attention, `FlashPrefillConfig` uses
  `block_size=128`, attention sinks, a recent-window rule, and alpha-threshold
  block selection (`DFLASH_FP_ALPHA`). This determines sparse attention blocks
  for the drafter forward.
- For final prompt compression, `drafter_score_and_compress()` uses chunk-top-K:
  default `chunk_size=32`, `n_keep = floor(n_chunks * keep_ratio)`, partial sort
  by chunk mean score, chronological sort, then adjacent selected chunks are
  span-merged. This yields approximately `keep_ratio * S` source tokens before
  tokenizer effects.

Target prefill:

- PFlash does not preserve gaps or original token positions. The compressed
  drafter-token stream is decoded back to text, then re-tokenized with the
  target tokenizer and chat template. See `pflash/tests/bench_niah_cpp.py` and
  `dflash/scripts/_prefill_hook.py`.
- The target sees a continuous compressed prompt. Its KV cache is populated
  normally for that shorter prompt.

Kernels and layout:

- PFlash brings custom CUDA/C++ kernels: `compute_mean_vector_bf16`,
  `compute_block_score_bf16`, GPU `block_select`, and sparse forward.
- The fast sparse forward can call mit-han-lab BSA via `bsa_launcher.cu`;
  hardcoded support is BF16, head_dim 128, block_size 128, SM80+.
- Tensor layout is contiguous `[B, S, H, D]` / `[B, S, Hk, D]`. BSA receives a
  dense contiguous Q/K/V pointer plus a blockmask. It does not interoperate
  directly with vLLM PagedAttention block tables.

I could not use the X announcement as an architecture source, and the requested
Lucebox blog URL was blocked by the browsing tool. The repo README and
`pflash/README.md` mirror the relevant architecture details and link to the
blog, so the conclusions above are source-code-backed rather than based on
secondary summaries.

## 2. Compatibility With vLLM Prefill

Current club-3090 vLLM long-text path:

- Compose: `models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml`.
- Image: `vllm/vllm-openai:nightly-7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8`.
- Boot patches: `python3 -m vllm._genesis.patches.apply_all`, then local
  sidecars such as `patch_tolist_cudagraph.py` and
  `patch_inputs_embeds_optional.py`.
- Runtime args include `--enable-chunked-prefill`,
  `--max-num-batched-tokens 4128`, `--max-num-seqs 1`,
  `--kv-cache-dtype turboquant_3bit_nc`, and
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

Relevant vLLM surfaces at the pinned commit:

- `vllm/v1/core/sched/scheduler.py`: `Scheduler.schedule()` splits request
  progress into scheduled token counts; `_mamba_block_aligned_split()` enforces
  hybrid/Mamba scheduling constraints; `update_draft_token_ids()` ignores draft
  tokens while `request.is_prefill_chunk` is true.
- `vllm/v1/worker/gpu_model_runner.py`: `GPUModelRunner.__init__()` creates
  decode-time proposers/drafters when `speculative_config` is set;
  `_update_states()` and `_prepare_inputs()` turn `SchedulerOutput` into input
  tensors; `_dummy_sampler_run()` handles spec-decode warmup/profiling.
- `vllm/model_executor/models/qwen3_next.py`: `Qwen3NextDecoderLayer` chooses
  `GatedDeltaNetAttention` or `Qwen3NextAttention`; `Qwen3NextModel.forward()`
  drives the target layers.
- `vllm/v1/spec_decode/eagle.py` and `llm_base_proposer.py`: decode-time
  proposer model surfaces, not prefill-compression surfaces.

Mapping PFlash elements:

1. Drafter as a separate scheduling pass:
   This does not map cleanly to vLLM's existing chunked-prefill scheduler.
   vLLM schedules target-model tokens in chunks. PFlash needs a pre-admission
   full-source pass with a separate model, then it rewrites the prompt before
   target scheduling. That is a new request state, e.g. `WAITING_COMPRESSION`,
   not another target prefill chunk.

2. Target compressed prompt:
   This maps cleanly if PFlash is treated as prompt rewriting. vLLM can prefill
   the compressed target IDs normally, with existing PagedAttention/KV layout,
   prefix caching, and MTP decode after prefill. It does not map cleanly if the
   goal is to preserve original absolute positions or sparse gaps.

3. KV layout:
   No target KV change is needed for the prompt-rewrite interpretation. A
   direct PFlash kernel port into target attention would require adapting
   contiguous Q/K/V + BSA blockmask logic to vLLM's varlen/PagedAttention
   metadata. PFlash's source kernels assume dense contiguous source tensors,
   not paged KV block tables.

4. Attention backend:
   To reproduce PFlash speed, vLLM would need a drafter attention backend or
   custom model runner path equivalent to Luce's `flashprefill_forward_bf16()`.
   The natural vLLM location would be under `vllm/v1/attention/backends/` plus
   model code that exposes Q/K/V capture for Qwen3-0.6B. A pure Python wrapper
   around the Luce C++ entry point would not fit vLLM's current model execution
   abstraction cleanly.

5. Second model:
   Required. vLLM already has decode-time draft/proposer support, but PFlash's
   Qwen3-0.6B drafter is a prefill-time scorer. Existing
   `SpeculativeConfig` is tied to generation, verifier scheduling, draft token
   IDs, and rejection/acceptance. It is not a general prefill-drafter API.

6. Tokenizer handoff:
   Required. PFlash uses the drafter tokenizer to select spans, decodes the
   selected drafter IDs back to text, and re-tokenizes/render-chat-templates for
   the target. vLLM would need this before building its final `Request` token
   IDs. This is above attention/kernel level.

Conclusion: possible as a native vLLM feature, but only as request-level
prefill compression. Not possible as a drop-in attention backend for the
existing target prefill loop.

## 3. Could This Live As Genesis Monkey-Patches?

Genesis can patch existing vLLM Python and reroute existing execution surfaces.
club-3090 already uses this for:

- Boot-time source edits via `vllm._genesis.patches.apply_all`.
- Attention/kernel routing and SM86-specific sparse-V behavior
  (`GENESIS_ENABLE_PN26_SPARSE_V=1`).
- GDN/prefill chunking (`GENESIS_ENABLE_P103=1`,
  `GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1`).
- Activation/op wrappers (`GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1`,
  `GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1`).
- Local sidecars such as
  `models/qwen3.6-27b/vllm/patches/patch_inputs_embeds_optional.py`.

PFlash work that could live in Genesis or club-3090 sidecars:

- Add an opt-in HTTP/request preprocessor that calls an external compressor
  service before vLLM admission.
- Patch OpenAI serving code to rewrite the last user message above a token
  threshold, similar to Luce's `dflash/scripts/server.py`.
- Add env-gated routing and diagnostics.
- Possibly compile and load a small custom CUDA extension for a block-sparse
  helper if it is isolated from scheduler/model lifecycle.

PFlash work that does not fit Genesis well:

- Loading and memory-managing a second LLM inside vLLM workers.
- Parking/unparking target model weights around compression on 24 GB.
- Adding a new request lifecycle state before target prefill.
- Integrating a prefill drafter with vLLM's scheduler, cache profiler, memory
  accounting, CUDA graph warmup, and error handling.
- Capturing per-layer Q/K from an arbitrary drafter model and running tail
  scoring without first-class model-runner support.

The engine-level coordination cost dominates. Genesis could prototype a
"proxy compressor before vLLM" path, but a true in-process PFlash port should
not be a runtime text-patch stack.

## 4. Integration Paths

### Path A: Native vLLM Port

Scope:

- Add a `PrefillCompressionConfig`, separate from `SpeculativeConfig`.
- Add request state before normal admission: tokenize/render source, run
  prefill-drafter, replace prompt text/IDs, then submit target request.
- Add a Qwen3-0.6B drafter model runner with Q/K capture and tail scoring.
- Add/port FlashPrefill/BSA kernels as a vLLM extension/backend.
- Add memory profiling for target + prefill drafter + temporary sparse-attn
  scratch.
- Define compatibility with chat templates, tool-calling, structured outputs,
  prefix caching, LoRA, multimodal inputs, and multi-tenant batching.

Effort: huge, likely 2-4 months for a narrow Qwen-only prototype; 6+ months
for upstream-quality design and tests.

Risk: medium-high. The prompt-rewrite design is architecturally possible, but
quality policy and scheduler/memory integration are substantial.

Next step:

- Prototype outside vLLM first: a local PFlash compression microservice that
  returns compressed text; then feed that text into current vLLM. If quality
  holds on our RAG/codebase tasks, write an upstream design issue/PR plan.

### Path B: Genesis Monkey-Patch

Scope:

- Patch vLLM's OpenAI serving layer to call an external PFlash compressor for
  prompts over a threshold.
- Keep the PFlash daemon as a separate process/container, not inside vLLM.
- Rewrite the last user message or full prompt before vLLM request creation.
- Add safe skips for tool schemas, structured outputs, vision, and short
  prompts.

Effort: medium for an external-compressor proof of concept, about 1-3 weeks.
Large and brittle if attempting in-process second-model loading, likely
4-8 weeks with high failure risk.

Risk: high for in-process monkey-patch; medium for external proxy. Genesis can
patch Python serving flow, but it cannot robustly own vLLM worker memory,
second-model lifecycle, or CUDA graph/profiling integration.

Hard limit:

- This path is not a real PFlash port. It is a vLLM-front prompt compressor.
  It gives up shared allocator benefits and adds interprocess/service
  lifecycle complexity.

Next step:

- Do not add this to Sandermage's Genesis tree now. If testing is needed, make
  it a club-3090 local experiment: `pflash-proxy` or `pflash-preprocessor`
  container in front of vLLM.

### Path C: Adopt lucebox-hub As Third Engine

Scope:

- Build `lucebox-hub/dflash` with `DFLASH27B_ENABLE_BSA=ON` and SM86.
- Fetch/convert target, DFlash draft, and Qwen3-0.6B drafter weights.
- Run `dflash/scripts/server.py` or `server_tools.py` with
  `--prefill-compression auto`.
- Bench side-by-side against vLLM single-card, vLLM TP=2, and llama.cpp.

Effort: small-medium, about 2-5 days for bench harness and routing; 1-2 weeks
if packaging as a stable club-3090 engine variant.

Risk: low architectural mismatch, medium product risk. It is their native
engine, so PFlash fits. The cost is feature parity.

What we give up versus vLLM:

- Our Genesis patch stack and vLLM nightly features.
- Qwen3 Coder tool-parser behavior, structured output integration, and
  vLLM request scheduling.
- Existing MTP path; lucebox uses DFlash/DDTree decode, not vLLM MTP.
- AutoRound INT4 and TQ3 path as configured in club-3090; lucebox validates
  GGUF Q4_K_M target and its own KV quantization.
- Mature OpenAI-compatible behavior for all client quirks. The current prompt
  notes greedy-only sampling, empty-prompt regression, and chat-template quirks.

Next step:

- Treat lucebox-hub as a third-engine candidate, not a vLLM port. Run NIAH and
  one real club-3090 long-RAG/codebase corpus through it once daemon-mode
  stability is acceptable.

## 5. Recommendation

Do not try to port PFlash into Genesis now.

Native vLLM integration is possible, but it is a first-class engine feature,
not a patch. The minimum viable vLLM design is a prefill-compression stage that
loads/runs a separate drafter, rewrites the prompt, then submits the compressed
target prompt to the existing scheduler. That requires scheduler/request
lifecycle, memory accounting, tokenizer policy, and model-runner work.

For club-3090's current workload, the cost-benefit is unfavorable today:

- Single-card vLLM now has a confirmed 60K-safe MTP-on long-text envelope after
  Genesis v7.69 + local vLLM #35975 backport + lower mem-util.
- Dual-card vLLM already covers the 200K+ class.
- llama.cpp covers the long-context single-card fallback at lower feature
  richness.
- PFlash's current validation is strongest on NIAH single-needle, not our
  tool-heavy IDE agent or mixed long-RAG workload.

Best near-term path:

1. Wait for lucebox-hub daemon/server maturity.
2. Add it as a third engine when it can run stable OpenAI-compatible requests
   with acceptable sampling/chat-template behavior.
3. Bench it on our actual long single-shot prompts, not only NIAH.
4. Revisit a native vLLM design only if the community starts an upstream
   prefill-compression feature or PFlash proves decisive on real workloads.

Possible interim experiment:

- Build a club-3090 `pflash-proxy` that compresses text through lucebox-hub and
  forwards compressed prompts to vLLM. This is useful for quality measurement.
  It should not be mistaken for production integration.

Bottom line: PFlash is promising for the exact pain point "long-context
single-card TTFT", but the practical integration path today is adoption as a
separate engine, not Genesis monkey-patching and not a quick vLLM kernel port.

## Addendum (2026-05-02 evening) - submodule + PR #78 audit

### What was previously missing (now seen)

- `/tmp/lucebox-hub` was unshallowed and submodules were initialized.
- Populated submodules:
  - `dflash/deps/llama.cpp`: 149M, pinned at
    `b6ffab4a9d3ee7dc2bd39354c86f6bb11ab15420`.
  - `dflash/deps/Block-Sparse-Attention`: 113M, pinned at
    `49d6c39e4dc0303442cda3bb758b3925d4399c49`.
  - BSA's CUTLASS submodule: `a75b4ac483166189a45290783cb0a18af5ff0ea5`.
- PR #78 was inspected with `gh pr view 78 --repo Luce-Org/lucebox-hub
  --comments --json title,body,files,commits,comments,state,mergedAt,author,url`
  and `gh pr diff 78`.
- No earlier sections are contradicted. The update mainly sharpens Path C:
  phase-split is now a reproducible PFlash-compression harness, not a complete
  dual-GPU target serving architecture.

### PR #78 dual-GPU phase-split - implications for Path C

PR #78: `bench(pflash): add dual-GPU PFlash phase-split harness`, merged
2026-05-02 as Lucebox commit `c5a6996`; PR commit
`de318816ed969e2f536e1714738dcf6279e34bbc`.

Changed files:

- `dflash/CMakeLists.txt`: adds `pflash_daemon` target.
- `dflash/test/pflash_daemon.cpp`: persistent compressor daemon.
- `dflash/scripts/phase_split_dual_gpu.py`: compression/report harness.
- `dflash/docs/SPEC_PREFILL.md`: documents the harness.

What "phase-split" means:

- It keeps the Qwen3-0.6B PFlash drafter resident in `pflash_daemon`.
- `scripts/phase_split_dual_gpu.py` sends counted drafter-token IDs to that
  daemon.
- `--pflash-gpu` pins only the PFlash compression phase to a selected CUDA GPU.
- It writes compressed token/text artifacts plus JSON/Markdown timing and GPU
  resource reports.
- It does not run target prefill, target decode, DFlash/DDTree decode, or a
  two-GPU serving loop. The PR body explicitly says it "does not measure or
  modify decode."

VRAM implication:

- For experiments, yes: it removes the single-process 24 GB park/unpark dance
  from the compression phase because the drafter can live as its own resident
  process on the PFlash GPU.
- For production Path C, only partially: a complete third engine still needs a
  target process/daemon for generation and routing between compressed artifact
  and target request. PR #78 measures PFlash as a phase, not end-to-end serving.

Performance/capacity evidence in the PR:

- Hardware: two 22 GB RTX 2080 Ti GPUs, not RTX 3090, A100, or H100.
- Workload: synthetic NIAH capacity sweep only.
- Reported local passing points:
  - single-GPU/co-resident PFlash + target: 24,573 source tokens -> 1,309
    compressed tokens.
  - dual-GPU phase split: 196,605 -> 9,949 tokens.
  - dual-GPU phase split: 229,365 -> 11,573 tokens.
  - dual-GPU phase split: 262,125 -> 13,229 tokens.
- The PR says the 262K point was the largest passing point in that local run,
  not a universal limit.
- No real RAG/codebase/tool workload benchmark is included.

Stability claims:

- PR #78 does not claim to fix greedy-only sampling, empty-prompt regressions,
  chat-template quirks, or OpenAI server stability. Comments contain one
  maintainer/contributor acknowledgement and no technical discussion resolving
  those gaps.

Path C effect:

- More attractive for benchmarking PFlash compression as a separate component.
- Not enough to call lucebox-hub a production-ready third engine for club-3090.
- Updated effort estimate:
  - PFlash-only compression bench on our prompts: lower than before, about
    1-3 days because PR #78 provides the daemon/report harness.
  - Packaged third-engine variant: still 1-2 weeks, because target generation,
    OpenAI behavior, sampling/chat-template quirks, and routing remain outside
    the phase-split PR.

### BSA kernel + Luce's llama.cpp fork - implications for Path A

BSA submodule:

- BSA README says it is modified from FlashAttention 2.4.2, supports fp16 and
  bf16, and supports head dimensions 32, 64, and 128.
- It supports dense, streaming, and block-sparse masks with block size 128 for
  block streaming/block-sparse modes.
- It has both causal and non-causal kernel instantiations, e.g.
  `flash_fwd_block_hdim128_bf16_sm80.cu` and
  `flash_fwd_block_hdim128_bf16_causal_sm80.cu`.
- `flash_fwd_launch_template.h` has an SM86/SM89-specific hdim128 path:
  for non-causal it uses a `128 x 32` tile to get two CTAs per SM; for causal
  it uses `64 x 64`.
- The implementation is FA2-derived and SM80+ friendly. There is no FA3
  requirement for Ampere.
- Luce's `dflash/src/bsa_launcher.cu` still hardcodes the PFlash use case more
  narrowly than BSA itself: BF16, `head_dim=128`, `block_size=128`, and
  non-causal `run_mha_fwd_block_<cutlass::bfloat16_t, 128, false>()`.

Luce llama.cpp fork:

- I diffed the pinned submodule commit `b6ffab4a9` against
  `ggml-org/llama.cpp` upstream master from merge-base
  `fae3a28070fe4026f87bd6a544aba1b2d1896566`.
- Luce-specific commits at the pinned submodule:
  - `b16de6590`: tree-mode `ssm_conv` + `gated_delta_net` CUDA kernels.
  - `137228317`: add `TQ3_0` TurboQuant 3.5 bpv KV cache type.
  - `3e80ebc8a`: default chunked FA threshold to 0; only TQ3_0 forces chunked.
- Diff size: 41 files, 1,827 insertions, 36 deletions.
- Deltas are concentrated in ggml/CUDA internals:
  - `ggml/src/ggml-cuda/fattn-chunked.{cu,cuh}` and `fattn.cu`.
  - `ggml/src/ggml-cuda/gated_delta_net.cu`.
  - `ggml/src/ggml-cuda/ssm-conv.cu`.
  - TQ3 quant/copy/dequant helpers and template instantiations.
  - `ggml/include/ggml.h` adds `GGML_TYPE_TQ3_0`,
    `GGML_OP_TURBO_WHT`, `ggml_ssm_conv_tree()`,
    `ggml_gated_delta_net_tree()`, and
    `ggml_gated_delta_net_tree_persist()`.
  - `src/llama-kv-cache.cpp` disables normal attn rotation for TQ3_0 KV.
- I did not find sampler or high-level server scheduler rewrites in the pinned
  diff. `common/arg.cpp` only adds `GGML_TYPE_TQ3_0` to accepted KV cache
  types. This suggests lucebox-hub keeps much of standard llama.cpp behavior,
  but its CUDA/ggml model execution path is materially forked.

Path A effect:

- Seeing BSA does not make a vLLM port small. The kernel is portable in
  principle on SM86, but PFlash's vLLM problem is still prefill-drafter
  lifecycle, tokenizer rewrite, memory scheduling, and request integration.
- BSA's actual constraints match the original memo: Ampere is fine; PFlash's
  Luce launcher is the narrow hdim128/BF16/non-causal path; vLLM would still
  need new glue around contiguous Q/K/V and blockmask construction.

### Path C verdict refresh

Adoption became more attractive as an experimental third-engine bench target,
not as an immediate production route.

- Better: PR #78 gives us a clean way to benchmark compression on one GPU and
  inspect compressed artifacts and resource peaks.
- Unchanged: daemon/OpenAI stability gaps were not addressed by PR #78.
- Unchanged: no evidence yet for real club-3090 RAG/codebase workloads.
- Unchanged: production third-engine work still needs server hardening,
  routing, sampling/chat-template validation, and feature-parity decisions.

Recommended next Path C experiment:

1. Build `pflash_daemon`.
2. Run `phase_split_dual_gpu.py run-prompt` on one real long club-3090 prompt.
3. Feed the compressed artifact to current vLLM and llama.cpp manually.
4. Compare answer quality before spending time packaging lucebox-hub serving.

### Path A verdict refresh

No change. A native vLLM PFlash port remains a first-class engine feature, not
a Genesis patch and not a kernel-only job. BSA being SM86-capable removes one
hardware concern, but the scheduler/request/memory/tokenizer work remains the
dominant cost.

## Sources

- Lucebox hub README:
  https://github.com/Luce-Org/lucebox-hub
- PFlash README:
  https://github.com/Luce-Org/lucebox-hub/tree/main/pflash
- PFlash daemon notes:
  https://github.com/Luce-Org/lucebox-hub/blob/main/dflash/docs/SPEC_PREFILL.md
- PFlash source inspected locally from `Luce-Org/lucebox-hub`:
  `dflash/src/qwen3_drafter.cpp`, `dflash/src/flashprefill.cpp`,
  `dflash/src/bsa_launcher.cu`, `dflash/scripts/_prefill_hook.py`,
  `dflash/test/test_dflash.cpp`.
- vLLM pinned source inspected at commit
  `7a1eb8ac2ec4ea69338c51dc7afd4b15010abfa8`:
  `vllm/v1/core/sched/scheduler.py`,
  `vllm/v1/worker/gpu_model_runner.py`,
  `vllm/model_executor/models/qwen3_next.py`.
- Local club-3090 surfaces:
  `models/qwen3.6-27b/vllm/compose/docker-compose.long-text.yml`,
  `models/qwen3.6-27b/vllm/patches/README.md`,
  `scripts/setup.sh`.
