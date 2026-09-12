# HYPERMAX validation — September 12, 2026

Status: experimental. Functional and throughput validation passed. The full
context ladder and continuous soak are still running; this report does not
claim production readiness.

## Configuration

- Two RTX 3090 cards, SM86, PCIe P2P enabled, no NVLink, 250 W per card.
- AMD EPYC 7K62, 373.3 GiB RAM, NVIDIA driver 595.71.05.
- Stock `vllm/vllm-openai:v0.29.0`, image digest
  `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
- PyTorch 2.13.0+cu130, FlashInfer 0.6.18. FA2 source and CUTLASS revisions
  are pinned in `install.py`; both extensions were built inside this image.
- Official Qwen FP8 target revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`;
  syvai DFlash2 W4A16 revision `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.
- TP=2, DFlash2 n=7, FP8 E4M3 KV for target and draft, LBNHC layout, BF16
  compute, 262144 context, one sequence, 2048 batched tokens, one image.
- NCCL P2P enabled; custom all-reduce disabled. Thinking OFF for throughput.

The actual compose was started through `switch.sh --force
vllm/qwen38-27b-dual-hypermax`, with `MODEL_DIR` pointing at installed weights,
`ENABLE_THINKING=false`, `NCCL_P2P_DISABLE=0`, and a loopback test port.
Target and draft full CUDA Graph capture succeeded.

## Results

`verify-full.sh`: all 10 checks passed, including streaming tool calls,
thinking, the 2K output check and vision ground truth 4/4.

Canonical `bench.sh`: 3 warmups and 5 measured runs per decode workload.
Sampler explicitly sent: temperature=0.6, top_p=0.95, top_k=20, min_p=0.
Narrative max_tokens=1000; quicksort max_tokens=800.

| Metric | Narrative | Code |
|---|---:|---:|
| Decode tok/s, mean ± sample SD | 99.88 ± 2.79 | 191.93 ± 9.12 |
| Decode CV | 2.8% | 4.8% |
| Wall tok/s | 98.54 | 181.06 |
| TTFT | 136 ms | 143 ms |

Prefill uses fresh haystacks, one warmup per depth and 3/1 measured runs:

| Prompt depth | Input tokens | Prefill tok/s |
|---|---|---:|
| About 10K | 10003, 10003, 10360 | 1750.64 ± 8.40 |
| About 90K | 93331 | 1387.22 (one measured run) |

Per-card VRAM peak during verify + bench: **23808 / 23808 MiB**, sampled every
500 ms. Minimum physical free memory: **319 / 319 MiB**. This is below the
standard stress gate's 1024 MiB margin; do not interpret a successful request
as production memory headroom. Engine logs were captured with the benchmark;
windowed throughput is indicative, not interchangeable with request timing.

## Checks and limitations

- Compose render, registry canonicalization, profile diagnostics on dual
  RTX 3090, patch attribution and KV-calculator calibration checks passed.
  KV-calculator projection is unavailable for this model/external-draft
  combination; the compose uses the measured fixed KV allocation.
- Both CUDA libraries built successfully. A second install reused the build.
  Negative checks refused a different vLLM version and a modified backend hash.
- The full 152-script catalog sweep ran. Socket-based fixture tests needed
  reruns outside the sandbox. Final counts are pending those reruns.
- Six unrelated capture/benchmark fixture failures also reproduce on release
  v0.11.0: test-bench-capture, test-pull, test-pullemit-capture, test-loop-input,
  test-submit-pull and test-trust-pipeline. They concern the triad worker-count
  assertion and capture manifest schema/outcome fields.
- Full-context stress and continuous soak: pending. The original 0.27.1
  measurements are historical; none of them are substituted for 0.29.0 checks.

Reproduce serving checks with `URL`, `MODEL=qwen3.8-27b` and
`CONTAINER=vllm-qwen38-27b-dual-hypermax` set for the running profile:

```bash
bash scripts/verify-full.sh
bash scripts/bench.sh
bash scripts/verify-stress.sh
SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 bash scripts/soak-test.sh
```
