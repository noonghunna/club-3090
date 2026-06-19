# club-3090 rig report

Generated: 2026-06-14 04:58:44 UTC

_Redacted output (paths, host, user, tokens). Re-run with `--no-redact` for full data._

## System

- **OS:** Debian GNU/Linux 13 (trixie)
- **Kernel:** 7.0.2-7-pve
- **Environment:** lxc (virtualized)
- **Locale:** C
- **Timezone:** UTC
- **Uptime:** up 48 minutes

## CPU + RAM

- **CPU:** Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz (24 threads)
- **RAM:** 48Gi total, 40Gi available

## Disk

- **<MODEL_DIR>:** 133G available, ext4 filesystem
- **/var/lib/docker:** 57G available, ext4 filesystem

## GPU hardware

- **GPU 0:** NVIDIA GeForce RTX 3090 | 24576 MiB | driver 610.43.02 | VBIOS 94.02.42.80.37 | persistence=Disabled
  - **Power:** limit=350.00 W (default=350.00 W, max=366.00 W) | current_draw=188.97 W
  - **PCIe:** x16 lanes negotiated (GPU max x16, Gen up to 3) | bus 00000000:17:00.0
- **GPU 1:** NVIDIA GeForce RTX 3090 | 24576 MiB | driver 610.43.02 | VBIOS 94.02.42.C0.15 | persistence=Disabled
  - **Power:** limit=350.00 W (default=350.00 W, max=366.00 W) | current_draw=25.87 W
  - **PCIe:** x16 lanes negotiated (GPU max x16, Gen up to 3) | bus 00000000:65:00.0
- **ECC mode:** [N/A] (3090s don't have ECC; expect N/A)

### NVLink

<details><summary>NVLink link status</summary>


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


</details>

### Topology

<details><summary>PCIe / GPU topology matrix</summary>


        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     0-5     0               N/A
GPU1    NV4      X      0-5     0               N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks


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


Sun Jun 14 04:58:45 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:17:00.0 Off |                  N/A |
| 33%   28C    P8            188W /  350W |   23166MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:65:00.0  On |                  N/A |
| 30%   33C    P8             25W /  350W |   22938MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+


</details>

## Display / desktop state

- **$DISPLAY:** unset (headless)
- **Display processes running:** none detected
- **GPU 0 idle VRAM:** 23166 MiB (held by running `vllm-diffusiongemma-26b-a4b-fp8-tp2`)
- **GPU 1 idle VRAM:** 22938 MiB (held by running `vllm-diffusiongemma-26b-a4b-fp8-tp2`)

## Container runtime

- **Docker:** 29.5.2
- **docker compose (v2):** 5.1.4
- **NVIDIA Container Toolkit:** 1.19.1

## Stack version

- **club-3090:** `v0.8.7-150-g99ced8e` (branch: `master`, SHA `99ced8e`)
- **GENESIS_PIN default:** `7b9fd319` (per scripts/setup.sh)
- **Cached vLLM images:**
  - tag `<none>` digest `3` (days ago)
  - tag `gemma4-unified` digest `sha256:e828735fba48bca2cf9701864d41693c91953394c5b1455b4668edd7563ed450` (10 days ago)
  - tag `v0.22.0` digest `sha256:0fec7ec5f3e6bc168e54899935fb0557da908a4832a1dbc88e2debcf2f889416` (2 weeks ago)
## Profile state

- **Profile schema version:** 1
- **Profile counts:** 9 hardware, 7 models, 5 workloads, 12 engines, 10 drafters
- **Compose registry:** 46 entries
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
  
  0, NVIDIA GeForce RTX 3090, 610.43.02
  1, NVIDIA GeForce RTX 3090, 610.43.02
  

### Boot log highlights

**KV pool sizing:**

(Worker_TP0 pid=42) INFO 06-14 04:57:27 [gpu_worker.py:480] Available KV cache memory: 4.43 GiB
(EngineCore pid=32) INFO 06-14 04:57:27 [kv_cache_utils.py:1744] GPU KV cache size: 413,470 tokens
(EngineCore pid=32) INFO 06-14 04:57:27 [kv_cache_utils.py:1745] Maximum concurrency for 262,144 tokens per request: 1.58x


**Engine config (CLI flags + engine init):**

(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:273] non-default args: {'model_tag': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'chat_template': '/vllm-workspace/examples/tool_chat_template_gemma4.jinja', 'enable_auto_tool_choice': True, 'tool_call_parser': 'gemma4', 'model': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'trust_remote_code': True, 'max_model_len': 262144, 'enforce_eager': True, 'served_model_name': ['diffusiongemma-26b-a4b'], 'hf_overrides': {'diffusion_sampler': 'entropy_bound', 'diffusion_entropy_bound': 0.1}, 'override_generation_config': {'max_new_tokens': 16384}, 'attention_backend': 'TRITON_ATTN', 'reasoning_parser': 'gemma4', 'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.82, 'max_num_batched_tokens': 2048, 'max_num_seqs': 1}
(EngineCore pid=32) INFO 06-14 04:57:02 [core.py:113] Initializing a V1 LLM engine (v0.22.1rc1.dev357+g74b5964f0) with config: model='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', speculative_config=None, tokenizer='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=compressed-tensors, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='gemma4', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=diffusiongemma-26b-a4b, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')



### Full boot log (first 200 lines)

<details><summary>First 200 lines of docker logs</summary>


(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]        █     █     █▄   ▄█
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.22.1rc1.dev357+g74b5964f0
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]   █▄█▀ █     █     █     █  model   ~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:339]
(APIServer pid=1) INFO 06-14 04:56:37 [api_utils.py:273] non-default args: {'model_tag': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'chat_template': '/vllm-workspace/examples/tool_chat_template_gemma4.jinja', 'enable_auto_tool_choice': True, 'tool_call_parser': 'gemma4', 'model': '~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', 'trust_remote_code': True, 'max_model_len': 262144, 'enforce_eager': True, 'served_model_name': ['diffusiongemma-26b-a4b'], 'hf_overrides': {'diffusion_sampler': 'entropy_bound', 'diffusion_entropy_bound': 0.1}, 'override_generation_config': {'max_new_tokens': 16384}, 'attention_backend': 'TRITON_ATTN', 'reasoning_parser': 'gemma4', 'tensor_parallel_size': 2, 'gpu_memory_utilization': 0.82, 'max_num_batched_tokens': 2048, 'max_num_seqs': 1}
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_DISABLE_CUSTOM_ALL_REDUCE
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_COMMIT
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_PIPELINE
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_BUILD_URL
(APIServer pid=1) WARNING 06-14 04:56:37 [envs.py:2111] Unknown vLLM environment variable detected: VLLM_IMAGE_TAG
(APIServer pid=1) INFO 06-14 04:56:48 [model.py:611] Resolved architecture: DiffusionGemmaForBlockDiffusion
(APIServer pid=1) INFO 06-14 04:56:48 [model.py:1750] Using max model len 262144
(APIServer pid=1) INFO 06-14 04:56:48 [vllm.py:1011] Asynchronous scheduling is enabled.
(APIServer pid=1) WARNING 06-14 04:56:48 [vllm.py:1067] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(APIServer pid=1) WARNING 06-14 04:56:48 [vllm.py:1075] TORCH_COMPILE_DISABLE is set, disabling torch.compile. This is equivalent to setting -cc.mode=none
(APIServer pid=1) WARNING 06-14 04:56:48 [vllm.py:1109] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(APIServer pid=1) INFO 06-14 04:56:48 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(APIServer pid=1) INFO 06-14 04:56:48 [vllm.py:1285] Cudagraph is disabled under eager mode
(APIServer pid=1) WARNING 06-14 04:56:48 [vllm.py:2081] Model Runner V2 does not yet support the thinking_token_budget request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required.
(APIServer pid=1) INFO 06-14 04:56:51 [compilation.py:321] Enabled custom fusions: norm_quant, act_quant
(EngineCore pid=32) INFO 06-14 04:57:02 [core.py:113] Initializing a V1 LLM engine (v0.22.1rc1.dev357+g74b5964f0) with config: model='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', speculative_config=None, tokenizer='~/.cache/huggingface/diffusiongemma-26b-a4b-it-fp8-dynamic', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=262144, download_dir=None, load_format=auto, tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=False, quantization=compressed-tensors, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='gemma4', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=diffusiongemma-26b-a4b, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['all'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_rope_kvcache_cat_mla': False, 'fuse_act_padding': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto', linear_backend='auto')
(EngineCore pid=32) INFO 06-14 04:57:02 [multiproc_executor.py:140] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=172.22.0.2 (local), world_size=2, local_world_size=2
(Worker pid=42) INFO 06-14 04:57:12 [parallel_state.py:1568] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:51485 backend=nccl
(Worker pid=43) INFO 06-14 04:57:12 [parallel_state.py:1568] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:51485 backend=nccl
(Worker pid=42) INFO 06-14 04:57:12 [pynccl.py:113] vLLM is using nccl==2.28.9
(Worker pid=42) WARNING 06-14 04:57:12 [symm_mem.py:66] SymmMemCommunicator: Device capability 8.6 not supported, communicator is not available.
(Worker pid=43) WARNING 06-14 04:57:12 [symm_mem.py:66] SymmMemCommunicator: Device capability 8.6 not supported, communicator is not available.
(Worker pid=42) INFO 06-14 04:57:12 [cuda_communicator.py:245] Using ['CUSTOM', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=42) INFO 06-14 04:57:12 [cuda_communicator.py:245] Using ['PYNCCL'] all-reduce backends (in dispatch order) for group 'ep:0' out of potential backends: ['NCCL_SYMM_MEM', 'QUICK_REDUCE', 'FLASHINFER', 'CUSTOM', 'SYMM_MEM', 'PYNCCL'].
(Worker pid=42) INFO 06-14 04:57:12 [parallel_state.py:1903] rank 0 in world size 2 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank 0, EPLB rank N/A
(Worker pid=42) INFO 06-14 04:57:12 [gpu_worker.py:303] Using V2 Model Runner
(Worker_TP1 pid=43) INFO 06-14 04:57:13 [model_runner.py:269] Loading model from scratch...
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [model_runner.py:269] Loading model from scratch...
(Worker_TP1 pid=43) INFO 06-14 04:57:13 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [fp8.py:419] Using MARLIN Fp8 MoE backend out of potential backends: ['AITER', 'FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS', 'DEEPGEMM', 'VLLM_CUTLASS', 'TRITON', 'MARLIN', 'BATCHED_DEEPGEMM', 'BATCHED_VLLM_CUTLASS', 'BATCHED_TRITON', 'XPU', 'CPU'].
(Worker_TP1 pid=43) INFO 06-14 04:57:13 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [cuda.py:318] Using AttentionBackendEnum.TRITON_ATTN backend.
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [weight_utils.py:922] Filesystem type for checkpoints: EXT4. Checkpoint size: 25.33 GiB. Available RAM: 109.98 GiB.
(Worker_TP0 pid=42) INFO 06-14 04:57:13 [weight_utils.py:945] Auto-prefetch is disabled because the filesystem (EXT4) is not a recognized network FS (NFS/Lustre). If you want to force prefetching, start vLLM with --safetensors-load-strategy=prefetch.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:08<00:00,  8.40s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:08<00:00,  8.40s/it]
(Worker_TP0 pid=42)
(Worker_TP0 pid=42) INFO 06-14 04:57:22 [default_loader.py:397] Loading weights took 8.43 seconds
(Worker_TP0 pid=42) WARNING 06-14 04:57:22 [marlin_utils_fp8.py:103] Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads.
(Worker_TP0 pid=42) WARNING 06-14 04:57:22 [marlin.py:126] Padding FP8 Marlin K dimension from 1056 to 1088 for RowParallelLinear. Extra activation columns are zero-padded at apply time.
(Worker_TP0 pid=42) WARNING 06-14 04:57:22 [marlin_utils_fp8.py:261] Padding FP8 MoE Marlin intermediate dimension from 352 to 384. Gate/up shards and down-proj reduction are zero-padded before repack.
(Worker_TP0 pid=42) INFO 06-14 04:57:22 [fp8.py:625] Using MoEPrepareAndFinalizeNoDPEPModular
(Worker_TP0 pid=42) INFO 06-14 04:57:23 [model_runner.py:290] Model loading took 14.59 GiB and 10.956314 seconds
(Worker_TP1 pid=43) INFO 06-14 04:57:23 [model_runner.py:290] Model loading took 14.59 GiB and 10.976225 seconds
(Worker_TP0 pid=42) INFO 06-14 04:57:23 [topk_topp_sampler.py:55] Using FlashInfer for top-p & top-k sampling.
(Worker_TP0 pid=42) INFO 06-14 04:57:27 [gpu_worker.py:480] Available KV cache memory: 4.43 GiB
(EngineCore pid=32) INFO 06-14 04:57:27 [kv_cache_utils.py:1744] GPU KV cache size: 413,470 tokens
(EngineCore pid=32) INFO 06-14 04:57:27 [kv_cache_utils.py:1745] Maximum concurrency for 262,144 tokens per request: 1.58x
(Worker_TP0 pid=42) WARNING 06-14 04:57:27 [diffusion_gemma.py:1132] DiffusionGemma does not support repetition/frequency/presence penalties; ignoring them for this request.
(Worker_TP0 pid=42) INFO 06-14 04:57:32 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(Worker_TP1 pid=43) INFO 06-14 04:57:32 [jit_monitor.py:54] Kernel JIT monitor activated — Triton JIT compilations during inference will be logged as warnings.
(EngineCore pid=32) INFO 06-14 04:57:32 [core.py:316] init engine (profile, create kv cache, warmup model) took 9.10 s
(EngineCore pid=32) INFO 06-14 04:57:35 [vllm.py:1011] Asynchronous scheduling is enabled.
(EngineCore pid=32) WARNING 06-14 04:57:35 [vllm.py:1067] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
(EngineCore pid=32) WARNING 06-14 04:57:35 [vllm.py:1075] TORCH_COMPILE_DISABLE is set, disabling torch.compile. This is equivalent to setting -cc.mode=none
(EngineCore pid=32) WARNING 06-14 04:57:35 [vllm.py:1109] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
(EngineCore pid=32) INFO 06-14 04:57:35 [kernel.py:270] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'], fused_add_rms_norm=['vllm_c', 'native'])
(EngineCore pid=32) INFO 06-14 04:57:35 [vllm.py:1285] Cudagraph is disabled under eager mode
(EngineCore pid=32) WARNING 06-14 04:57:35 [vllm.py:2081] Model Runner V2 does not yet support the thinking_token_budget request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required.
(EngineCore pid=32) INFO 06-14 04:57:35 [compilation.py:321] Enabled custom fusions: norm_quant, act_quant
(APIServer pid=1) INFO 06-14 04:57:35 [api_server.py:579] Supported tasks: ['generate']
(APIServer pid=1) INFO 06-14 04:57:35 [parser_manager.py:37] "auto" tool choice has been enabled.
(APIServer pid=1) WARNING 06-14 04:57:35 [model.py:1502] Default vLLM sampling parameters have been overridden by the model's `generation_config.json`: `{'max_tokens': 16384}`. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
(APIServer pid=1) INFO 06-14 04:57:36 [hf.py:548] Detected the chat template content format to be 'openai'. You can set `--chat-template-content-format` to override this.
(APIServer pid=1) INFO 06-14 04:58:29 [base.py:227] Multi-modal warmup completed in 53.055s
(APIServer pid=1) INFO 06-14 04:58:29 [base.py:227] Readonly multi-modal warmup completed in 0.145s
(APIServer pid=1) INFO 06-14 04:58:29 [api_server.py:583] Starting vLLM server on http://0.0.0.0:8000
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:37] Available routes are:
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /openapi.json, Methods: GET, HEAD
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /docs, Methods: GET, HEAD
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: GET, HEAD
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /redoc, Methods: GET, HEAD
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /load, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /version, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /health, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /metrics, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /tokenize, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /detokenize, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/models, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /ping, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /ping, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /invocations, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/chat/completions/batch, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/responses, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/completions, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/messages, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/messages/count_tokens, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /generative_scoring, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/chat/completions/render, Methods: POST
(APIServer pid=1) INFO 06-14 04:58:29 [launcher.py:46] Route: /v1/completions/render, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
(APIServer pid=1) INFO:     172.22.0.1:34576 - "GET /v1/models HTTP/1.1" 200 OK


</details>

## Recent failed boot attempts

_No recently-exited vLLM or llama.cpp containers found._

## verify-full.sh output

<details><summary>verify-full output</summary>


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
    → SSE may be buffering. Final text: Code hides hidden flaw,
One small typo breaks the flow,
Logic found at last.
[6/9] Streaming tool-calls (thinking-on) ...
  ✓ streamed delta.tool_calls (get_weather) + finish_reason=tool_calls, no <tool_call> leak
[7/9] Thinking / reasoning mode ...
  ✓ reasoning 151 chars, content 9 chars (finish=stop)
    reasoning: *   Question: "What is 2+2?"     *   Constraint: "One-line a...
    content:   2+2 is 4....
[8/9] Output quality / cascade detection (2K-token completion) ...
  ✓ output OK — 8591 chars, variety=0.665, max_line_repeat=0, finish=stop
[9/9] MTP acceptance length threshold ...
  ⊘ no SpecDecoding metrics in logs (compose may not have spec-decode enabled) (skipped)

1 check(s) failed. See hints above.


</details>

## verify-stress.sh output

<details><summary>verify-stress output (7 boundary checks incl. Cliff 2 needle recall)</summary>


[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)
[autodetect] served model='diffusiongemma-26b-a4b' (from http://localhost:8020/v1/models; set MODEL= to override)
Running STRESS / boundary test against http://localhost:8020
  model=diffusiongemma-26b-a4b  container=vllm-diffusiongemma-26b-a4b-fp8-tp2  engine=vllm
  This script does the heavy stuff (longctx needle ladder + ~25K-token tool prefill).
  For the fast functional smoke (~2 min), use verify-full.sh instead.

[1/8] Long-context needle small rungs (10K / 30K) ...
    ✓   9673 tokens: recalled 'violet platypus 40' (got: violet platypus 40 )  prefill=4259.6 t/s (2s)
    ✓  28871 tokens: recalled 'amber otter 70' (got: amber otter 70 )  prefill=4397.8 t/s (7s)
  ✓ all long-ctx depths recalled secret correctly
[2/8] Tool response prefill OOM (~25K-token mock tool response) ...
  ✓ tool prefill OK — text response (814 chars, finish=stop)
[3/8] IDE-agent one-shot prompt (sys + tool schemas + user request) ...
  ✓ IDE-agent one-shot OK — 52 completion tokens (79 chars), finish=stop
[4/8] Multi-turn agent prompt (sys + tools + 4-turn history) ...
  ✓ multi-turn agent OK
[5/8] LCB-coding shape (LeetCode-style problem + structured plan) ...
  ✓ LCB-coding shape OK
[6/8] Reasoning-heavy (math problem + max_tokens=8192) ...
  ✓ reasoning-heavy OK — 1620 completion tokens
[7/8] Long-context needle large rungs (60K / 90K — Cliff 2 territory) ...
    ✓  57673 tokens: recalled 'amber axolotl 31' (got: amber axolotl 31 )  prefill=3142.1 t/s (18s)
    ✓  89673 tokens: recalled 'violet chinchilla 54' (got: violet chinchilla 54 )  prefill=3447.0 t/s (26s)
  ✓ all long-ctx depths recalled secret correctly
[8/8] Context ceiling ladder (staggered NIAH from ~95000 → ~0.92 × n_ctx) ...
    n_ctx=262144  ladder: 95000 → 125000 → 155000 → 185000 → 215000 → 241172 (6 rungs)
    calibrated: scale=100 → 6417 tokens (tok/scale_unit=64.17)
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    VRAM free (ladder start): 2107 MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 1/6: target=95K  actual=94K tok (36%)  recalled 'turquoise axolotl 59'  prefill=2660.7 t/s (36s)  VRAM_free=2107MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 2/6: target=125K  actual=124K tok (47%)  recalled 'crimson chinchilla 93'  prefill=2287.2 t/s (55s)  VRAM_free=2107MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 3/6: target=155K  actual=154K tok (58%)  recalled 'amber chinchilla 67'  prefill=2023.6 t/s (76s)  VRAM_free=2107MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    ✓ rung 4/6: target=185K  actual=184K tok (70%)  recalled 'silver iguana 51'  prefill=1735.8 t/s (106s)  VRAM_free=2107MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)
    △ rung 5/6: target=215K  actual=214K tok (81%)  recall MISS (got: 'sapphire axolot ') — quality ceiling reached  prefill=1637.8 t/s (131s)  VRAM_free=2107MB
    [vram] WARN: could not determine model GPU(s) on 2-GPU host — summing all (margin may be inflated)

  ✓ ceiling ladder: quality ceiling at 214474 tok (81% of n_ctx=262144) — recall miss, passed up to 184520 tok
    VRAM: 2107 → 2107 MB (Δ -0 MB across ladder, margin threshold=1024 MB)

All stress / boundary checks passed. KV-cache and prefill paths are sound for the deployed config.


</details>

## soak-test.sh (SOAK_MODE=continuous) output

<details><summary>soak-test stdout (5-session × 5-turn ramping conversation, ~25 min)</summary>


[soak] ERROR: no running club-3090 container found (vllm-/llama-cpp-/ik-llama-/beellama-/sglang- × qwen36-27b/qwen36-35b-a3b/gemma-4-31b); set CONTAINER=... or CONTAINER=none for host engines
scripts/soak-test.sh: line 262: results/report-soak-20260614-050702/nvidia-smi-final.csv: No such file or directory
scripts/soak-test.sh: line 265: results/report-soak-20260614-050702/docker-stats-final.jsonl: No such file or directory
[soak] artifacts: results/report-soak-20260614-050702


</details>
_soak summary.md not produced — check stdout above_

## bench.sh output

<details><summary>bench output (3 warmups + 5 measured per prompt)</summary>


[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)
[autodetect] served model='diffusiongemma-26b-a4b' (from http://localhost:8020/v1/models; set MODEL= to override)

========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========
=== warmups (3) ===
  warm-1     wall=  4.32s  ttft=   991ms  toks=1000  wall_TPS=231.28  decode_TPS=300.03
  warm-2     wall=  4.62s  ttft=  1224ms  toks=1000  wall_TPS=216.26  decode_TPS=294.08
  warm-3     wall=  4.57s  ttft=  1095ms  toks=1000  wall_TPS=218.58  decode_TPS=287.39

=== measured (5) ===
  run-1      wall=  5.61s  ttft=  1867ms  toks=1000  wall_TPS=178.32  decode_TPS=267.33
  run-2      wall=  4.69s  ttft=  1285ms  toks=1000  wall_TPS=213.20  decode_TPS=293.61
  run-3      wall=  4.29s  ttft=  1284ms  toks=1000  wall_TPS=232.99  decode_TPS=332.42
  run-4      wall=  4.56s  ttft=  1223ms  toks=1000  wall_TPS=219.13  decode_TPS=299.40
  run-5      wall=  4.96s  ttft=  1349ms  toks=1000  wall_TPS=201.62  decode_TPS=276.98

=== summary [narrative] (n=5) ===
  wall_TPS       mean= 209.05   std= 20.56   CV= 9.8%   min=178.32   max=232.99
  decode_TPS     mean= 293.95   std= 25.03   CV= 8.5%   min=267.33   max=332.42
  TTFT          mean=  1402ms  std=  264ms  min=1223ms  max=1867ms
  PP tok/s       mean=4293.30   std=9591.05   CV=223.4%   min=0.00   max=21450.30

========== CODE (prompt=78 chars, max_tokens=800) ==========
=== warmups (3) ===
  warm-1     wall=  2.61s  ttft=   892ms  toks= 765  wall_TPS=292.58  decode_TPS=444.15
  warm-2     wall=  2.70s  ttft=   837ms  toks= 743  wall_TPS=275.60  decode_TPS=399.61
  warm-3     wall=  3.53s  ttft=  1417ms  toks= 704  wall_TPS=199.21  decode_TPS=332.60

=== measured (5) ===
  run-1      wall=  3.09s  ttft=   901ms  toks= 800  wall_TPS=258.62  decode_TPS=364.92
  run-2      wall=  3.72s  ttft=  1607ms  toks= 800  wall_TPS=214.83  decode_TPS=377.85
  run-3      wall=  2.89s  ttft=  1164ms  toks= 752  wall_TPS=260.44  decode_TPS=436.24
  run-4      wall=  2.50s  ttft=   774ms  toks= 739  wall_TPS=295.59  decode_TPS=428.12
  run-5      wall=  3.96s  ttft=   966ms  toks= 721  wall_TPS=182.28  decode_TPS=241.18

=== summary [code] (n=5) ===
  wall_TPS       mean= 242.35   std= 44.13   CV=18.2%   min=182.28   max=295.59
  decode_TPS     mean= 369.66   std= 78.18   CV=21.1%   min=241.18   max=436.24
  TTFT          mean=  1082ms  std=  325ms  min=774ms  max=1607ms
  PP tok/s       mean=   6.88   std=  2.23   CV=32.3%   min=5.40   max=10.40

=== GPU state ===
0, 100 %, 23168 MiB, 24576 MiB, 339.25 W, 48
1, 100 %, 22940 MiB, 24576 MiB, 325.64 W, 58

=== Last 3 SpecDecoding metrics ===


</details>

## bench-agentic.sh output

<details><summary>bench-agentic output (1 session x 12 default turns, curve-shape estimate; ~8 min estimate)</summary>


[autodetect] using running container=vllm-diffusiongemma-26b-a4b-fp8-tp2 url=http://localhost:8020  (skip: PREFLIGHT_NO_AUTODETECT=1)

========================================================================
SESSION 1/1 — 12 turns, context grows to ~29,033 tokens
========================================================================
  Turn  Prompt tok   TTFT ms  Decode TPS  Result chars
  ----- ---------- --------- ----------- -------------
  1            826       852    339162.0           307
  2          1,026       401    369384.7           249
  3          1,228       404    666984.1           278
  4          1,441       405    254567.6         8,373
  5          4,792      1101    643298.2         8,912
  6          7,801      1256    273541.6         3,106
  7          9,149       666    506615.4         6,495
  8         11,275      2248   2729307.9         2,576  ⚠ tool-call miss (synthetic result injected)
  9         12,721       558    221400.2        25,250
  10        22,925      3471    590747.0        17,407
  11        29,803      4851    896218.8        21,299  ⚠ tool-call miss (synthetic result injected)
  12        38,517      5591   2665871.2        21,883  ⚠ tool-call miss (synthetic result injected)


========================================================================
SUMMARY — multi-turn prefill stress (1 session(s) × 12 turns)
========================================================================
  tool-call misses: 3/12 turns — ramp continued via synthetic results (#255); depth/curve unaffected, but tool-call reliability is degraded at depth on this config.
  Turn  Prompt tok   TTFT ms   σ ms  Decode TPS  Notes
  ----- ---------- --------- ------ -----------  ───────────────────────────────────
  1            826       852      0    339162.0  cold-start (compile/warmup — excluded from growth)
  2          1,026       401      0    369384.7  warm baseline
  3          1,228       404      0    666984.1
  4          1,441       405      0    254567.6
  5          4,792      1101      0    643298.2  ↑  TTFT 2.7× warm-baseline
  6          7,801      1256      0    273541.6  ↑  TTFT 3.1× warm-baseline
  7          9,149       666      0    506615.4  ~  TTFT 1.7× warm-baseline
  8         11,275      2248      0   2729307.9  ⚠  TTFT 5.6× warm-baseline (O(n)-like growth for this arch_class)
  9         12,721       558      0    221400.2
  10        22,925      3471      0    590747.0  ⚠  TTFT 8.7× warm-baseline (O(n)-like growth for this arch_class)
  11        29,803      4851      0    896218.8  ⚠  TTFT 12.1× warm-baseline (O(n)-like growth for this arch_class)
  12        38,517      5591      0   2665871.2  ⚠  TTFT 13.9× warm-baseline (O(n)-like growth for this arch_class)

────────────────────────────────────────────────────────────────────────
  TTFT growth by accumulated context (12 turns, 1 sessions):
    Turn 1 (cold):            852 ms TTFT  — compile/warmup, excluded from growth
    Turn 2 (warm base):      401 ms TTFT @ 1,026 prompt tokens
    Turn 12:                 5591 ms TTFT @ 38,517 prompt tokens
    Context grew 37.5×,  TTFT grew 13.9× (warm baseline → last turn)
    ~  TTFT sub-linear for this cell (13.9× vs 37.5× context).
    (Full-context O(n) growth would approach 37.5× with context)

  Note — DeltaNet/SSM state is NOT prefix-cacheable on vLLM Qwen3-Next cells.
  Attention KV caching can still work, but recurrent-state recomputation scales
  O(n) with sequence length. Prior single-card 24 GB vLLM Qwen3-Next observations
  saw degradation above ~35K tokens and timeouts around ~74K; treat those as
  informational per-arch_class guideposts. llama.cpp is not affected.

=== GPU state ===
0, 100 %, 23168 MiB, 24576 MiB, 343.17 W, 49
1, 100 %, 22940 MiB, 24576 MiB, 333.26 W, 59

=== Last 3 SpecDecoding metrics ===


</details>

---

_Generated by `bash scripts/report.sh`. Flags: `--verify` (verify-full), `--stress` (verify-stress 7/7 incl. Cliff 2 needles), `--soak` (SOAK_MODE=continuous, catches Cliff 2b), `--bench` (canonical TPS), `--agentic` (multi-turn TTFT/decode curve-shape, ~8 min estimate), `--full` (all five, ~43 min estimate). Use `--no-redact` to disable redaction (internal sharing only)._