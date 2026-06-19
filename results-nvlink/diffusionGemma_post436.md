# club-3090 rig report

Generated: 2026-06-19 05:53:07 UTC

_Redacted output (paths, host, user, tokens). Re-run with `--no-redact` for full data._

## System

- **OS:** Debian GNU/Linux 13 (trixie)
- **Kernel:** 7.0.2-7-pve
- **Environment:** lxc (virtualized)
- **Locale:** C
- **Timezone:** UTC
- **Uptime:** up 4 days, 1 hour, 43 minutes

## CPU + RAM

- **CPU:** Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz (24 threads)
- **RAM:** 92Gi total, 81Gi available

## Disk

- **<MODEL_DIR>:** 107G available, ext4 filesystem
- **/var/lib/docker:** 126G available, ext4 filesystem

## GPU hardware

- **GPU 0:** NVIDIA GeForce RTX 3090 | 24576 MiB | driver 610.43.02 | VBIOS 94.02.42.80.37 | persistence=Disabled
  - **Power:** limit=350.00 W (default=350.00 W, max=366.00 W) | current_draw=188.45 W
  - **PCIe:** x16 lanes negotiated (GPU max x16, Gen up to 3) | bus 00000000:17:00.0
- **GPU 1:** NVIDIA GeForce RTX 3090 | 24576 MiB | driver 610.43.02 | VBIOS 94.02.42.C0.15 | persistence=Disabled
  - **Power:** limit=350.00 W (default=350.00 W, max=366.00 W) | current_draw=25.69 W
  - **PCIe:** x16 lanes negotiated (GPU max x16, Gen up to 3) | bus 00000000:65:00.0
- **ECC mode:** [N/A] (3090s don't have ECC; expect N/A)

### NVLink

<details><summary>NVLink link status</summary>

```
GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-1aff65f8-8897-eaee-29e4-6871ef7491a0)
         Link 0: 14.062 GB/s
         Link 1: 14.062 GB/s
         Link 2: 14.062 GB/s
         Link 3: 14.062 GB/s
GPU 1: NVIDIA GeForce RTX 3090 (UUID: GPU-fef3f583-99bd-ac7f-c761-eddb679067e0)
         Link 0: 14.062 GB/s
         Link 1: 14.062 GB/s
         Link 2: 14.062 GB/s
         Link 3: 14.062 GB/s
```

</details>

### Topology

<details><summary>PCIe / GPU topology matrix</summary>

```
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     4,6,9,15,17     0               N/A
GPU1    NV4      X      4,6,9,15,17     0               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```

</details>

### PCIe / P2P detail (lspci)

_lspci not available (pciutils not installed) — showing P2P capability matrix instead._

        GPU0    GPU1
 GPU0   X       OK
 GPU1   OK      X

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  DR   = Disabled by regkey
  U    = Unknown

### Full nvidia-smi

<details><summary>Full nvidia-smi output</summary>

```
Fri Jun 19 05:53:08 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:17:00.0 Off |                  N/A |
| 32%   28C    P8            188W /  350W |   23074MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:65:00.0  On |                  N/A |
| 30%   33C    P8             25W /  350W |   23106MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

</details>

## Display / desktop state

- **$DISPLAY:** unset (headless)
- **Display processes running:** none detected
- **GPU 0 idle VRAM:** 23074 MiB (held by running `vllm-diffusiongemma-26b-a4b-fp8-tp2`)
- **GPU 1 idle VRAM:** 23106 MiB (held by running `vllm-diffusiongemma-26b-a4b-fp8-tp2`)

## Container runtime

- **Docker:** 29.6.0
- **docker compose (v2):** 5.1.4
- **NVIDIA Container Toolkit:** 1.19.1

## Stack version

- **club-3090:** `v0.8.7-202-gd7bffb7` (branch: `master`, SHA `d7bffb7`)
- **GENESIS_PIN default:** `7b9fd319` (per scripts/setup.sh)
- **Cached vLLM images:**
  - tag `<none>` digest `8` (days ago)
  - tag `v0.22.0` digest `sha256:0fec7ec5f3e6bc168e54899935fb0557da908a4832a1dbc88e2debcf2f889416` (2 weeks ago)
## Profile state

- **Profile schema version:** 1
- **Profile counts:** 9 hardware, 8 models, 5 workloads, 13 engines, 11 drafters
- **Compose registry:** 53 entries
- **Canonical scenarios:** 9
- **Calibration:**
  - gemma-4-12b: 1 rows
  - gemma-4-26b-a4b: 0 rows
  - gemma-4-31b: 3 rows
  - qwen3.6-27b: 2 rows
  - qwen3.6-35b-a3b: 1 rows
- **Active estate:** none (`~/.club3090/estate.yml` not found)

## KV math calibration

- Overall: 7/7 (100%)
- No FAIL rows. kv-calc projections should agree with measured VRAM within the ±1.5 GB error band.
<details><summary>Full kv-calc --calibration output</summary>

```
========================================================================================
Calibration — predicted per-card VRAM vs measured BENCHMARKS rows
========================================================================================

  Predicted = weights + activation + overhead + drafter + (KV capped at available).
  Budget = mem_util × VRAM. Measured = nvidia-smi peak during bench (target ≈ budget).
  Verdict ✓ iff PASS/TIGHT and measured < VRAM (boot OK).

== qwen3.6-27b ==
  compose                     predicted    budget   measured  verdict
  ─────────────────────────   ─────────  ────────  ─────────  ───────
  dual                          19.91 GB   22.80 GB    23.60 GB    PASS ✓
  minimal@64K                   21.60 GB   21.60 GB    22.40 GB   TIGHT ✓

  Verdict accuracy: 2/2 (100%)

== qwen3.6-35b-a3b ==
  compose                     predicted    budget   measured  verdict
  ─────────────────────────   ─────────  ────────  ─────────  ───────
  qwen-35b-a3b-dual             13.68 GB   22.08 GB    22.10 GB    PASS ✓

  Verdict accuracy: 1/1 (100%)

== gemma-4-31b ==
  compose                     predicted    budget   measured  verdict
  ─────────────────────────   ─────────  ────────  ─────────  ───────
  gemma-dual-int8               22.80 GB   22.80 GB    22.20 GB   TIGHT ✓
  gemma-dual                    22.80 GB   22.80 GB    22.50 GB   TIGHT ✓
  gemma-dual-int8@256K          22.80 GB   22.80 GB    22.50 GB   TIGHT ✓

  Verdict accuracy: 3/3 (100%)

== gemma-4-12b ==
  compose                     predicted    budget   measured  verdict
  ─────────────────────────   ─────────  ────────  ─────────  ───────
  gemma-dual                    21.60 GB   21.60 GB    21.60 GB   TIGHT ✓

  Verdict accuracy: 1/1 (100%)

Overall: 7/7 (100%)

Notes:
  - This is a directional estimator (±1.5 GB error band on the breakdown).
  - vLLM's `gpu_worker.py` boot log is the authoritative source.
  - If predicted PASS but measured > budget, file an issue with `scripts/report.sh --bench`.
```

</details>

## Active container

- **Name:** `vllm-diffusiongemma-26b-a4b-fp8-tp2`
- **Engine:** `vllm`
- **Status:** Up 2 minutes
- **Ports:** 0.0.0.0:8020->8000/tcp, [::]:8020->8000/tcp
- **Image:** `vllm/vllm-openai:gemma`
- **Image digest:** `sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b`
- **Build tag (OCI version):** `vllm/vllm-openai:v74b5964f02c7e023fadd3004cfac8a61c52eef1f`
- **Upstream commit (OCI revision):** `74b5964f02c7e023fadd3004cfac8a61c52eef1f`
- **Upstream source:** https://github.com/vllm-project/vllm

### Container Python / CUDA versions

- **PyTorch:** `torch=2.11.0+cu130 torch_cuda_build=13.0 cudnn=91900`
- **vLLM:** `0.22.1rc1.dev357+g74b5964f0`
- **nvidia-smi inside container:**
  ```
  0, NVIDIA GeForce RTX 3090, 610.43.02
  1, NVIDIA GeForce RTX 3090, 610.43.02
  ```

### Boot log highlights

**KV pool sizing:**
```
(Worker_TP0 pid=50) INFO 06-19 05:51:39 [gpu_worker.py:480] Available KV cache memory: 4.36 GiB
(EngineCore pid=40) INFO 06-19 05:51:39 [kv_cache_utils.py:1744] GPU KV cache size: 406,513 tokens
(EngineCore pid=40) INFO 06-19 05:51:39 [kv_cache_utils.py:1745] Maximum concurrency for 262,144 tokens per request: 1.55x
```

**Engine config (CLI flags + engine init):**
```
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:273] non-default args: {'model_tag': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'chat_template': '/vllm-workspace/examples/tool_chat_template_gemma4.jinja', 'enable_auto_tool_choice': True, 'tool_call_parser': 'gemma4', 'model': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'trust_remote_code': True, 'max_model_len': 262144, 'enforce_eager': True, 'served_model_name': ['diffusiongemma-26b-a4b'], 'hf_overrides': {'diffusion_sampler': 'entropy_bound', 'diffusion_entropy_bound': 0.1}, 'override_generation_config': {'max_new_tokens': 16384}, 'attention_backend': 'TRITON_ATTN', 'reasoning_parser': 'gemma4', 'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.82, 'max_num_batched_tokens': 2048, 'max_num_seqs': 1}
(EngineCore pid=40) INFO 06-19 05:51:07 [core.py:113] Initializing a V1 LLM engine (v0.22.1rc1.dev357+g74b5964f0) with config: model='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', speculative_config=None, tokenizer='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=compressed-tensors, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='gemma4', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=diffusiongemma-26b-a4b, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
```


### Full boot log (first 200 lines)

<details><summary>First 200 lines of docker logs</summary>

```
[nvlink] detected NVLink (NV4) between GPU0-GPU1 — enabling NVLink mode
[nvlink] P2P ENABLED — NCCL_P2P_LEVEL=NVL, custom all-reduce ON, expandable_segments OFF
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]        █     █     █▄   ▄█
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.22.1rc1.dev357+g74b5964f0
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]   █▄█▀ █     █     █     █  model   ~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:339]
(APIServer pid=1) INFO 06-19 05:50:41 [api_utils.py:273] non-default args: {'model_tag': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'chat_template': '/vllm-workspace/examples/tool_chat_template_gemma4.jinja', 'enable_auto_tool_choice': True, 'tool_call_parser': 'gemma4', 'model': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'trust_remote_code': True, 'max_model_len': 262144, 'enforce_eager': True, 'served_model_name': ['diffusiongemma-26b-a4b'], 'hf_overrides': {'diffusion_sampler': 'entropy_bound', 'diffusion_entropy_bound': 0.1}, 'override_generation_config': {'max_new_tokens': 16384}, 'attention_backend': 'TRITON_ATTN', 'reasoning_parser': 'gemma4', 'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.82, 'max_num_batched_tokens': 2048, 'max_num_seqs': 1}
(APIServer pid=1) WARNING 06-19 05:50:41 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_URL
(APIServer pid=1) WARNING 06-19 05:50:41 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_IMAGE_TAG
(APIServer pid=1) WARNING 06-19 05:50:41 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND
(APIServer pid=1) WARNING 06-19 05:50:41 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_PIPELINE
(APIServer pid=1) WARNING 06-19 05:50:41 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT
(APIServer pid=1) INFO 06-19 05:50:52 [model.py:611] Resolved architecture: DiffusionGemmaForBlockDiffusion
(APIServer pid=1) INFO 06-19 05:50:52 [model.py:1750] Using max model len 262144
(APIServer pid=1) INFO 06-19 05:50:53 [vllm.py:1011] Asynchronous scheduling is enabled.
(APIServer pid=1) WARNING 06-19 05:50:53 [vllm.py:1067] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(APIServer pid=1) WARNING 06-19 05:50:53 [vllm.py:1075] TORCH_COMPILE_DISABLE is set, disabling torch.compile. This is equivalent to setting -cc.mode=none
(APIServer pid=1) WARNING 06-19 05:50:53 [vllm.py:1109] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=1) INFO 06-19 05:50:53 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(APIServer pid=1) INFO 06-19 05:50:53 [vllm.py:1285] Cudagraph is disabled under eager mode
(APIServer pid=1) WARNING 06-19 05:50:53 [vllm.py:2081] Model Runner V2 does not yet support the thinking_token_budget request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required.
(APIServer pid=1) INFO 06-19 05:50:55 [compilation.py:321] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=40) INFO 06-19 05:51:07 [core.py:113] Initializing a V1 LLM engine (v0.22.1rc1.dev357+g74b5964f0) with config: model='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', speculative_config=None, tokenizer='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=compressed-tensors, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='gemma4', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=diffusiongemma-26b-a4b, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=40) INFO 06-19 05:51:07 [multiproc_executor.py:140] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=172.20.0.2 (local), world_size=2, local_world_size=2
(Worker pid=50) INFO 06-19 05:51:17 [parallel_state.py:1568] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:49309 backend=nccl
(Worker pid=51) INFO 06-19 05:51:17 [parallel_state.py:1568] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:49309 backend=nccl
(Worker pid=50) INFO 06-19 05:51:17 [pynccl.py:113] vLLM is using nccl==2.28.9
(Worker pid=50) WARNING 06-19 05:51:18 [symm_mem.py:66] SymmMemCommunicator: Device capability 8.6 not supported, communicator is not available.
(Worker pid=51) WARNING 06-19 05:51:18 [symm_mem.py:66] SymmMemCommunicator: Device capability 8.6 not supported, communicator is not available.
(Worker pid=50) INFO 06-19 05:51:18 [cuda_communicator.py:245] Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=50) INFO 06-19 05:51:18 [cuda_communicator.py:245] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=50) INFO 06-19 05:51:18 [parallel_state.py:1903] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=50) INFO 06-19 05:51:18 [gpu_worker.py:303] Using V2 Model Runner
(Worker_TP1 pid=51) INFO 06-19 05:51:18 [model_runner.py:269] Loading model from scratch...
(Worker_TP0 pid=50) INFO 06-19 05:51:18 [model_runner.py:269] Loading model from scratch...
(Worker_TP0 pid=50) INFO 06-19 05:51:19 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP1 pid=51) INFO 06-19 05:51:19 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=50) INFO 06-19 05:51:19 [fp8.py:419] Using MARLIN Fp8 MoE backend out of potential backends: ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', 'DEEPGEMM', 'VLLM_CUTLASS', 'TRITON', 'MARLIN', 'BATCHED_DEEPGEMM', 'BATCHED_VLLM_CUTLASS', 'BATCHED_TRITON', 'XPU', 'CPU'].
(Worker_TP0 pid=50) INFO 06-19 05:51:19 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP1 pid=51) INFO 06-19 05:51:19 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=50) INFO 06-19 05:51:19 [weight_utils.py:922] Filesystem type for checkpoints: EXT4. Checkpoint size: 25.33 GiB. Available RAM: 92.61 GiB.
(Worker_TP0 pid=50) INFO 06-19 05:51:19 [weight_utils.py:945] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:14<00:00, 14.86s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:14<00:00, 14.86s/it]
(Worker_TP0 pid=50)
(Worker_TP0 pid=50) INFO 06-19 05:51:34 [default_loader.py:397] Loading weights took 15.02 seconds
(Worker_TP0 pid=50) WARNING 06-19 05:51:34 [marlin_utils_fp8.py:103] Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.
(Worker_TP0 pid=50) WARNING 06-19 05:51:34 [marlin.py:126] Padding FP8 Marlin K dimension from 1056 to 1088 for RowParallelLinear. Extra activation columns are zero-padded at apply time.
(Worker_TP0 pid=50) WARNING 06-19 05:51:34 [marlin_utils_fp8.py:261] Padding FP8 MoE Marlin intermediate dimension from 352 to 384. Gate/up shards and down-proj reduction are zero-padded before repack.
(Worker_TP0 pid=50) INFO 06-19 05:51:34 [fp8.py:625] Using MoEPrepareAndFinalizeNoDPEPModular
(Worker_TP0 pid=50) INFO 06-19 05:51:36 [model_runner.py:290] Model loading took 14.59 GiB and 17.632303 seconds
(Worker_TP0 pid=50) INFO 06-19 05:51:36 [topk_topp_sampler.py:55] Using FlashInfer for top-p & top-k sampling.
(Worker_TP1 pid=51) INFO 06-19 05:51:36 [model_runner.py:290] Model loading took 14.59 GiB and 17.677372 seconds
(Worker_TP0 pid=50) INFO 06-19 05:51:39 [gpu_worker.py:480] Available KV cache memory: 4.36 GiB
(EngineCore pid=40) INFO 06-19 05:51:39 [kv_cache_utils.py:1744] GPU KV cache size: 406,513 tokens
(EngineCore pid=40) INFO 06-19 05:51:39 [kv_cache_utils.py:1745] Maximum concurrency for 262,144 tokens per request: 1.55x
(Worker_TP0 pid=50) WARNING 06-19 05:51:39 [diffusion_gemma.py:1132] DiffusionGemma does not support repetition/frequency/presence penalties; ignoring them for this request.
(Worker_TP1 pid=51) INFO 06-19 05:51:44 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(Worker_TP0 pid=50) INFO 06-19 05:51:44 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(EngineCore pid=40) INFO 06-19 05:51:45 [core.py:316] init engine (profile, create kv cache, warmup model) took 8.95 s
(EngineCore pid=40) INFO 06-19 05:51:48 [vllm.py:1011] Asynchronous scheduling is enabled.
(EngineCore pid=40) WARNING 06-19 05:51:48 [vllm.py:1067] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=40) WARNING 06-19 05:51:48 [vllm.py:1075] TORCH_COMPILE_DISABLE is set, disabling torch.compile. This is equivalent to setting -cc.mode=none
(EngineCore pid=40) WARNING 06-19 05:51:48 [vllm.py:1109] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=40) INFO 06-19 05:51:48 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(EngineCore pid=40) INFO 06-19 05:51:48 [vllm.py:1285] Cudagraph is disabled under eager mode
(EngineCore pid=40) WARNING 06-19 05:51:48 [vllm.py:2081] Model Runner V2 does not yet support the thinking_token_budget request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required.
(EngineCore pid=40) INFO 06-19 05:51:48 [compilation.py:321] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=1) INFO 06-19 05:51:48 [api_server.py:579] Supported tasks: ['generate']
(APIServer pid=1) INFO 06-19 05:51:48 [parser_manager.py:37] "auto" tool choice has been enabled.
(APIServer pid=1) WARNING 06-19 05:51:48 [model.py:1502] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'max_tokens': 16384}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1) INFO 06-19 05:51:48 [hf.py:548] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1) INFO 06-19 05:52:44 [base.py:227] Multi-modal warmup completed in 55.636s
(APIServer pid=1) INFO 06-19 05:52:44 [base.py:227] Readonly multi-modal warmup completed in 0.145s
(APIServer pid=1) INFO 06-19 05:52:44 [api_server.py:583] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:37] Available routes are:
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /docs, Methods: GET, HEAD
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /generative_scoring, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=1) INFO 06-19 05:52:44 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
(APIServer pid=1) INFO:     172.20.0.1:39568 - "GET /v1/models HTTP/1.1" 200 OK
```

</details>

## Recent failed boot attempts

_No recently-exited vLLM or llama.cpp containers found._

## verify-full.sh output

<details><summary>verify-full output</summary>

```
[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)
[autodetect] served model='diffusiongemma-26b-a4b' (from http://localhost:8020/v1/models; set MODEL= to override)
Running FULL functional test against http://localhost:8020
  model=diffusiongemma-26b-a4b  container=vllm-diffusiongemma-26b-a4b-fp8-tp2  engine=vllm

[1/9] Server reachable on /v1/models ...
  ✓ server is serving
[2/9] Genesis patches applied ...
  ⊘ no Genesis marker in logs (container restarted, or Genesis not loaded) (skipped)
[warmup] priming engine (cold cudagraph/JIT, up to 180s, not scored) ...
[warmup] engine warm
[3/9] Basic completion — capital of France ...
  ✓ reply contains 'Paris'
[4/9] Tool calling ...
  ✓ tool_calls[] populated with get_weather
[5/9] Streaming (SSE) ...
  ✗ suspiciously few chunks (1) for 120 max_tokens
    → SSE may be buffering. Final text: Code hides hidden flaws,
One small typo breaks the path,
Logic found at last.
[6/9] Streaming tool-calls (thinking-on) ...
  ✓ streamed delta.tool_calls (get_weather) + finish_reason=tool_calls, no <tool_call> leak
[7/9] Thinking / reasoning mode ...
  ✓ reasoning 188 chars, content 9 chars (finish=stop)
    reasoning: *   Question: "What is 2+2?"     *   Constraint: "One-line a...
    content:   2+2 is 4....
[8/9] Output quality / cascade detection (2K-token completion) ...
  ✓ output OK — 8585 chars, variety=0.615, max_line_repeat=0, finish=stop
[9/9] MTP acceptance length threshold ...
  ⊘ no SpecDecoding metrics in logs (compose may not have spec-decode enabled) (skipped)

1 check(s) failed. See hints above.
```

</details>

## verify-stress.sh output

<details><summary>verify-stress output (7 boundary checks incl. Cliff 2 needle recall)</summary>

```
[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)
[autodetect] served model='diffusiongemma-26b-a4b' (from http://localhost:8020/v1/models; set MODEL= to override)
Running STRESS / boundary test against http://localhost:8020
  model=diffusiongemma-26b-a4b  container=vllm-diffusiongemma-26b-a4b-fp8-tp2  engine=vllm
  This script does the heavy stuff (longctx needle ladder + ~25K-token tool prefill).
  For the fast functional smoke (~2 min), use verify-full.sh instead.

[1/8] Long-context needle small rungs (10K / 30K) ...
    ✓   9672 tokens: recalled 'crimson otter 19' (got: crimson otter 19 )  prefill=6355.4 t/s (2s)
    ✓  28874 tokens: recalled 'sapphire platypus 35' (got: sapphire platypus 35 )  prefill=6138.1 t/s (5s)
  ✓ all long-ctx depths recalled secret correctly
[2/8] Tool response prefill OOM (~25K-token mock tool response) ...
  ✓ tool prefill OK — text response (781 chars, finish=stop)
[3/8] IDE-agent one-shot prompt (sys + tool schemas + user request) ...
  ✓ IDE-agent one-shot OK — 46 completion tokens (79 chars), finish=stop
[4/8] Multi-turn agent prompt (sys + tools + 4-turn history) ...
  ✓ multi-turn agent OK
[5/8] LCB-coding shape (LeetCode-style problem + structured plan) ...
  ✓ LCB-coding shape OK
[6/8] Reasoning-heavy (math problem + max_tokens=8192) ...
  ✓ reasoning-heavy OK — 1629 completion tokens
[7/8] Long-context needle large rungs (60K / 90K — Cliff 2 territory) ...
    ✓  57673 tokens: recalled 'turquoise platypus 75' (got: turquoise platypus 75 )  prefill=4229.9 t/s (14s)
    ✓  89673 tokens: recalled 'golden capybara 63' (got: golden capybara 63 )  prefill=4190.2 t/s (21s)
  ✓ all long-ctx depths recalled secret correctly
[8/8] Context ceiling ladder (staggered NIAH from ~95000 → ~0.92 × n_ctx) ...
    n_ctx=262144  ladder: 95000 → 125000 → 155000 → 185000 → 215000 → 241172 (6 rungs)
    calibrated: scale=100 → 6417 tokens (tok/scale_unit=64.17)
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    VRAM free (ladder start): 2031 MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 1/6: target=95K  actual=94K tok (36%)  recalled 'violet falcon 41'  prefill=3365.8 t/s (28s)  VRAM_free=2031MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 2/6: target=125K  actual=124K tok (47%)  recalled 'violet iguana 51'  prefill=2755.7 t/s (45s)  VRAM_free=2031MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 3/6: target=155K  actual=154K tok (58%)  recalled 'silver iguana 55'  prefill=2403.2 t/s (64s)  VRAM_free=2031MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 4/6: target=185K  actual=184K tok (70%)  recalled 'emerald iguana 40'  prefill=2022.2 t/s (91s)  VRAM_free=2031MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    △ rung 5/6: target=215K  actual=214K tok (81%)  recall MISS (got: 'crimson chilla ') — quality ceiling reached  prefill=1879.5 t/s (114s)  VRAM_free=2031MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)

  ✓ ceiling ladder: quality ceiling at 214474 tok (81% of n_ctx=262144) — recall miss, passed up to 184520 tok
    VRAM: 2031 → 2031 MB (Δ -0 MB across ladder, margin threshold=1024 MB)

All stress / boundary checks passed. KV-cache and prefill paths are sound for the deployed config.
```

</details>

## soak-test.sh (SOAK_MODE=continuous) output

<details><summary>soak-test stdout (5-session × 5-turn ramping conversation, ~25 min)</summary>

```
[soak] running soak test against http://localhost:8020 (model=diffusiongemma-26b-a4b, container=vllm-diffusiongemma-26b-a4b-fp8-tp2)
[soak] mode=continuous sessions=5 turns=5 max_growth=200MiB timeout=1800s
[soak] output=results/report-soak-20260619-060010
[soak] session 1/5
[soak]   turn 1/5: status=200 wall=374ms ttft=374ms decode_tps=0.0 vram=46184MiB
[soak]   turn 2/5: status=200 wall=966ms ttft=966ms decode_tps=0.0 vram=46184MiB
[soak]   turn 3/5: status=200 wall=1393ms ttft=1393ms decode_tps=0.0 vram=46184MiB
[soak]   turn 4/5: status=200 wall=1989ms ttft=1989ms decode_tps=0.0 vram=46184MiB
[soak]   turn 5/5: status=200 wall=5018ms ttft=3527ms decode_tps=317.091 vram=46184MiB
[soak] warm baseline after session 1: 46184 MiB
[soak] session 2/5
[soak]   turn 1/5: status=200 wall=364ms ttft=364ms decode_tps=0.0 vram=46184MiB
[soak]   turn 2/5: status=200 wall=346ms ttft=346ms decode_tps=0.0 vram=46184MiB
[soak]   turn 3/5: status=200 wall=386ms ttft=386ms decode_tps=0.0 vram=46184MiB
[soak]   turn 4/5: status=200 wall=2085ms ttft=2085ms decode_tps=0.0 vram=46184MiB
[soak]   turn 5/5: status=200 wall=3682ms ttft=1917ms decode_tps=248.681 vram=46184MiB
[soak] session 3/5
[soak]   turn 1/5: status=200 wall=363ms ttft=363ms decode_tps=0.0 vram=46184MiB
[soak]   turn 2/5: status=200 wall=283ms ttft=283ms decode_tps=0.0 vram=46184MiB
[soak]   turn 3/5: status=200 wall=455ms ttft=455ms decode_tps=0.0 vram=46184MiB
[soak]   turn 4/5: status=200 wall=435ms ttft=435ms decode_tps=0.0 vram=46184MiB
[soak]   turn 5/5: status=200 wall=3763ms ttft=1825ms decode_tps=220.917 vram=46184MiB
[soak] session 4/5
[soak]   turn 1/5: status=200 wall=262ms ttft=262ms decode_tps=0.0 vram=46184MiB
[soak]   turn 2/5: status=200 wall=345ms ttft=345ms decode_tps=0.0 vram=46184MiB
[soak]   turn 3/5: status=200 wall=384ms ttft=384ms decode_tps=0.0 vram=46184MiB
[soak]   turn 4/5: status=200 wall=436ms ttft=436ms decode_tps=0.0 vram=46184MiB
[soak]   turn 5/5: status=200 wall=3236ms ttft=2013ms decode_tps=335.18 vram=46184MiB
[soak] session 5/5
[soak]   turn 1/5: status=200 wall=255ms ttft=255ms decode_tps=0.0 vram=46184MiB
[soak]   turn 2/5: status=200 wall=347ms ttft=347ms decode_tps=0.0 vram=46184MiB
[soak]   turn 3/5: status=200 wall=453ms ttft=452ms decode_tps=0.0 vram=46184MiB
[soak]   turn 4/5: status=200 wall=435ms ttft=435ms decode_tps=0.0 vram=46184MiB
[soak]   turn 5/5: status=200 wall=3764ms ttft=1381ms decode_tps=181.675 vram=46184MiB

[soak] summary
[soak]   verdict              PASS
[soak]   boot_vram_mib        46184
[soak]   max_vram_mib         46184
[soak]   max_growth_mib       0 / 200
[soak]   errors               0
[soak]   silent_empty         0 / 25 (0.0%)
[soak]   p50_decode_tps       248.68
[soak]   p95_ttft_ms          2071
[soak]   tps_retention        100.0%
[soak]   note                 PASS = no failure signal on this sample;
[soak]                        not patch validation (topology alone can
[soak]                        sidestep what overlays target). See
[soak]                        scripts/soak-test.sh --help and docs/CLIFFS.md.
[soak] artifacts: results/report-soak-20260619-060010
```

</details>

**Soak summary** (`results/report-soak-20260619-060010/summary.md`):

# Soak test summary

- Verdict: **PASS**
- Boot VRAM baseline: 46184 MiB
- Max VRAM observed: 46184 MiB
- Max growth observed: 0 MiB
- Sessions completed: 5
- Request errors: 0
- Silent-empty turns (HTTP 200 + 0 completion tokens): 0 / 25 (0.0%)

| Metric | Value |
|---|---:|
| p50 decode TPS | 248.68 |
| p95 decode TPS | 331.56 |
| first-5 median TPS | 248.68 |
| last-5 median TPS | 248.68 |
| TPS retention | 100.0% |
| p50 TTFT | 435 ms |
| p95 TTFT | 2071 ms |
| TTFT first/last ratio | 1.00x |
| VRAM oscillation | 0 MiB |

## Recommendation

- Runtime VRAM growth and throughput retention stayed within v1 soak thresholds.

## bench.sh output

<details><summary>bench output (3 warmups + 5 measured per prompt)</summary>

```
[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)
[autodetect] served model='diffusiongemma-26b-a4b' (from http://localhost:8020/v1/models; set MODEL= to override)

========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========
=== warmups (3) ===
  warm-1     wall=  3.49s  ttft=  1177ms  toks=1000  wall_TPS=286.33  decode_TPS=431.79
  warm-2     wall=  3.55s  ttft=   996ms  toks=1000  wall_TPS=281.78  decode_TPS=391.77
  warm-3     wall=  4.16s  ttft=  1324ms  toks=1000  wall_TPS=240.64  decode_TPS=353.14

=== measured (5) ===
  run-1      wall=  4.24s  ttft=  1166ms  toks=1000  wall_TPS=236.00  decode_TPS=325.58
  run-2      wall=  3.66s  ttft=  1000ms  toks=1000  wall_TPS=272.90  decode_TPS=375.39
  run-3      wall=  3.83s  ttft=   941ms  toks=1000  wall_TPS=261.00  decode_TPS=346.02
  run-4      wall=  4.74s  ttft=  1104ms  toks=1000  wall_TPS=210.95  decode_TPS=275.01
  run-5      wall=  4.07s  ttft=  1002ms  toks=1000  wall_TPS=245.97  decode_TPS=326.42

=== summary [narrative] (n=5) ===
  wall_TPS       mean= 245.37   std= 23.85   CV= 9.7%   min=210.95   max=272.90
  decode_TPS     mean= 329.68   std= 36.65   CV=11.1%   min=275.01   max=375.39
  TTFT          mean=  1043ms  std=   90ms  min=941ms  max=1166ms
  PP tok/s       mean=   7.72   std=  1.55   CV=20.1%   min=5.40   max=9.10

========== CODE (prompt=78 chars, max_tokens=800) ==========
=== warmups (3) ===
  warm-1     wall=  2.44s  ttft=   983ms  toks= 777  wall_TPS=318.14  decode_TPS=532.42
  warm-2     wall=  2.51s  ttft=   940ms  toks= 720  wall_TPS=286.83  decode_TPS=458.57
  warm-3     wall=  2.41s  ttft=  1062ms  toks= 746  wall_TPS=310.08  decode_TPS=555.06

=== measured (5) ===
  run-1      wall=  2.80s  ttft=   834ms  toks= 784  wall_TPS=280.13  decode_TPS=399.12
  run-2      wall=  2.02s  ttft=   727ms  toks= 744  wall_TPS=368.35  decode_TPS=575.28
  run-3      wall=  2.20s  ttft=   845ms  toks= 726  wall_TPS=330.42  decode_TPS=536.99
  run-4      wall=  2.12s  ttft=   943ms  toks= 763  wall_TPS=359.88  decode_TPS=647.96
  run-5      wall=  2.18s  ttft=   836ms  toks= 732  wall_TPS=335.86  decode_TPS=544.67

=== summary [code] (n=5) ===
  wall_TPS       mean= 334.93   std= 34.51   CV=10.3%   min=280.13   max=368.35
  decode_TPS     mean= 540.80   std= 90.52   CV=16.7%   min=399.12   max=647.96
  TTFT          mean=   837ms  std=   77ms  min=727ms  max=943ms
  PP tok/s       mean=   7.68   std=  2.24   CV=29.1%   min=5.40   max=10.40

=== GPU state ===
0, 100 %, 23076 MiB, 24576 MiB, 345.61 W, 48
1, 100 %, 23108 MiB, 24576 MiB, 329.72 W, 53

=== Last 3 SpecDecoding metrics ===
```

</details>

## bench-agentic.sh output

<details><summary>bench-agentic output (1 session x 12 default turns, curve-shape estimate; ~8 min estimate)</summary>

```
[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)

========================================================================
SESSION 1/1 — 12 turns, context grows to ~29,033 tokens
========================================================================
  Turn  Prompt tok   TTFT ms  Decode TPS  Result chars
  ----- ---------- --------- ----------- -------------
  1            826       404    193233.5           307
  2            999       352    211582.1           249
  3          1,173       356    543706.1           278
  4          1,383       300    281407.0         8,373
  5          4,734       928   1048576.0         8,912
  6          7,743      1073    239272.4         3,106
  7          9,090       606    539985.2         6,495
  8         11,216      1730   2029501.9         2,576
  9         12,529       497    230175.2        25,250
  10        22,733      2760   1642109.6        17,407
  11        29,611      2291    271210.6        21,299
  12        38,210      4364   2783830.1        21,883  ⚠ tool-call miss (synthetic result injected)


========================================================================
SUMMARY — multi-turn prefill stress (1 session(s) × 12 turns)
========================================================================
  tool-call misses: 1/12 turns — ramp continued via synthetic results (#255); depth/curve unaffected, but tool-call reliability is degraded at depth on this config.
  Turn  Prompt tok   TTFT ms   σ ms  Decode TPS  Notes
  ----- ---------- --------- ------ -----------  ───────────────────────────────────
  1            826       404      0    193233.5  cold-start (compile/warmup — excluded from growth)
  2            999       352      0    211582.1  warm baseline
  3          1,173       356      0    543706.1
  4          1,383       300      0    281407.0
  5          4,734       928      0   1048576.0  ↑  TTFT 2.6× warm-baseline
  6          7,743      1073      0    239272.4  ↑  TTFT 3.0× warm-baseline
  7          9,090       606      0    539985.2  ~  TTFT 1.7× warm-baseline
  8         11,216      1730      0   2029501.9  ⚠  TTFT 4.9× warm-baseline (O(n)-like growth for this arch_class)
  9         12,529       497      0    230175.2  ~  TTFT 1.4× warm-baseline
  10        22,733      2760      0   1642109.6  ⚠  TTFT 7.8× warm-baseline (O(n)-like growth for this arch_class)
  11        29,611      2291      0    271210.6  ⚠  TTFT 6.5× warm-baseline (O(n)-like growth for this arch_class)
  12        38,210      4364      0   2783830.1  ⚠  TTFT 12.4× warm-baseline (O(n)-like growth for this arch_class)

────────────────────────────────────────────────────────────────────────
  TTFT growth by accumulated context (12 turns, 1 sessions):
    Turn 1 (cold):            404 ms TTFT  — compile/warmup, excluded from growth
    Turn 2 (warm base):      352 ms TTFT @ 999 prompt tokens
    Turn 12:                 4364 ms TTFT @ 38,210 prompt tokens
    Context grew 38.2×,  TTFT grew 12.4× (warm baseline → last turn)
    ~  TTFT sub-linear for this cell (12.4× vs 38.2× context).
    (Full-context O(n) growth would approach 38.2× with context)

  Note — DeltaNet/SSM state is NOT prefix-cacheable on vLLM Qwen3-Next cells.
  Attention KV caching can still work, but recurrent-state recomputation scales
  O(n) with sequence length. Prior single-card 24 GB vLLM Qwen3-Next observations
  saw degradation above ~35K tokens and timeouts around ~74K; treat those as
  informational per-arch_class guideposts. llama.cpp is not affected.

=== GPU state ===
0, 100 %, 23080 MiB, 24576 MiB, 343.21 W, 49
1, 100 %, 23112 MiB, 24576 MiB, 341.03 W, 54

=== Last 3 SpecDecoding metrics ===
```

</details>

---

_Generated by `bash scripts/report.sh`. Flags: `--verify` (verify-full), `--stress` (verify-stress 7/7 incl. Cliff 2 needles), `--soak` (SOAK_MODE=continuous, catches Cliff 2b), `--bench` (canonical TPS), `--agentic` (multi-turn TTFT/decode curve-shape, ~8 min estimate), `--full` (all five, ~43 min estimate). Use `--no-redact` to disable redaction (internal sharing only)._