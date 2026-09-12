# HYPERMAX validation — September 12, 2026

Status: experimental. Functional and throughput validation passed. The full
context ladder returned correct answers but failed the 1 GiB free-VRAM margin.
Continuous soak, quality, exact-context and combined vision/context checks
finished. The profile does not meet the production memory-margin gate.

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

Quality (`--medium --no-thinking --sampling-from-server --strict-thinking`):
**63/75** — ToolCall 14/15, InstructFollow 13/15, Structured Output 15/15,
Data Extraction 12/15, ReasonMath 9/15. Thinking validity passed for all packs.
Runner 0.9.9; packs tc1.0.1, if1.0.0, so1.1.0, de1.2.0, rm1.0.0.
The live server supplied temperature=0.7, top_p=0.8, top_k=20, min_p=0.
The historical 0.27.1 run scored 62/75 with the checkpoint's 1.0/0.95 sampling
defaults, so the score difference is not a controlled engine A/B.

Thinking ON (`--quick --enable-thinking --sampling-from-server
--strict-thinking`): **27/30**, ToolCall 13/15 and InstructFollow 14/15.
All 30 responses contained reasoning; strict thinking validity passed.
This leg inherited the same live server sampling row as the OFF leg; it tests
the request-level thinking switch, not a restart into the thinking sampler row.

- Compose render, registry canonicalization, profile diagnostics on dual
  RTX 3090, patch attribution and KV-calculator calibration checks passed.
  KV-calculator projection is unavailable for this model/external-draft
  combination; the compose uses the measured fixed KV allocation.
- Both CUDA libraries built successfully. A second install reused the build.
  Negative checks refused a different vLLM version and a modified backend hash.
- The full 152-script catalog sweep ran: **146 passed, 6 failed** after reruns
  of socket-based fixture tests outside the sandbox. All five guards affected
  by the new compose/installer wiring passed on the final tree.
- Six unrelated capture/benchmark fixture failures also reproduce on release
  v0.11.0: test-bench-capture, test-pull, test-pullemit-capture, test-loop-input,
  test-submit-pull and test-trust-pipeline. They concern the triad worker-count
  assertion and capture manifest schema/outcome fields.
- `verify-stress.sh`: all response checks passed, including the six-step
  ceiling ladder from about 94K through 240K actual input tokens. The last
  step recalled its secret at 1011.6 input tok/s. The command exited 1 solely
  because physical free VRAM was 299 MiB, below the 1024 MiB margin. The
  threshold was not lowered. This uniform-haystack test checks addressability,
  not retrieval quality on natural documents.
- Continuous soak: **PASS**, 5 sessions × 5 turns, zero errors, zero measured
  VRAM growth (47656 MiB total throughout), 100% throughput retention by the
  harness metric. This is a bounded sample, not a long-term stability claim.
- The original 0.27.1 measurements are historical; none substitute for these
  0.29.0 checks.

The exact context-boundary probe passed: **261000 input tokens**, zero cached
tokens, 11 output tokens, exact code recall, **266.603 seconds**. A request with
263000 input tokens was rejected with HTTP 400. Peak VRAM during this probe
was 23828 MiB on each card. The API probes are included as `check_context.py`
and `check_vision_context.py`; their `--url` and `--model` arguments allow
repeating the checks on another host.

The combined vision/context probe passed all five facts (document key, three
colored shapes and the number). It processed a 640×480 source fixture at about
4 MP, producing **4070 image tokens**. The image-only request took 3.815 s;
the combined **259872-token** request took **266.634 s**, with zero cached
tokens. Two follow-ups at 259932 and 259992 tokens also passed, in 8.210 s and
8.162 s, each reusing 255440 cached tokens. This tests processing/memory and
simple visual facts, not general high-resolution vision quality.

Across every serving test, per-card peak VRAM was **23888 / 23888 MiB** and
minimum physical free memory was **239 / 239 MiB**, sampled every 500 ms.
The standard stress margin remains failed; no threshold or context reduction
was used to turn that result green.

Reproduce serving checks with `URL`, `MODEL=qwen3.8-27b` and
`CONTAINER=vllm-qwen38-27b-dual-hypermax` set for the running profile:

```bash
bash scripts/verify-full.sh
bash scripts/bench.sh
bash scripts/verify-stress.sh
SOAK_MODE=continuous SOAK_SESSIONS=5 SOAK_TURNS=5 bash scripts/soak-test.sh
```
