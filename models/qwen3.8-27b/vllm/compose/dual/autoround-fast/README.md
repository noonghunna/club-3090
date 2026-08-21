# Qwen3.8-27B DFlash15 fast (canonical)

Canonical TP2 WSL profile: the complete `Qwen3.8-27B-W4A16-AutoRound-fast`
target plus the external `Qwen3.8-27B-DFlash2-W4A16` drafter, `SPEC_N=15`,
lookup drafting, BF16 KV, FlashAttention and synchronous scheduling.

## Cache layout

`MODEL_DIR` must contain these directories:

```text
qwen3.8-27b-autoround-fast/
qwen3.8-27b-dflash2-w4a16/
```

The target is the single repo
`born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast`; download it without assembling
or hard-linking model parts. Put the DFlash2 repo in the second directory using
the drafter's published checkpoint.

## WSL prerequisites

The compose mounts native WSL Ninja and CUDA headers because the DFlash2 top-k
path JIT-compiles on first use:

```bash
# NINJA_BIN defaults to /usr/bin/ninja
# CUDA_HEADERS_DIR must contain curand.h and the CUDA 13 headers
NINJA_BIN=/usr/bin/ninja CUDA_HEADERS_DIR=/home/cristian/cuda13-include
```

Override either path when the host uses a different location. The entrypoint
also applies the DFlash2 backport, lookup/split-KV, hybrid-KV/CUDAGraph,
Qwen embedding, prefix-cache safety and WSL-UVA compatibility patches. P2P
and custom all-reduce stay disabled.

## Launch

```bash
cd /home/cristian/club-3090-upstream
# Replace MODEL_DIR with the cache root containing both model directories.
MODEL_DIR=/home/cristian/validation-cache \
NINJA_BIN=/usr/bin/ninja \
CUDA_HEADERS_DIR=/home/cristian/cuda13-include \
SPEC_N=15 GPU_MEMORY_UTILIZATION=0.85 MAX_MODEL_LEN=244320 \
MAX_NUM_SEQS=1 KV_CACHE_MEMORY=9000000000 CG_MAX=16 \
VLLM_V2_CUDAGRAPH_MEM_MIB=1900 \
CLUB3090_RESTART=no docker compose -p dflash15-fast \
  -f models/qwen3.8-27b/vllm/compose/dual/autoround-fast/dflash15.yml up -d
```

These values are the measured ceiling profile. `MAX_NUM_SEQS=2` is an
experimental variant; launch it explicitly with `MAX_NUM_SEQS=2` and keep the
context/pool unchanged while comparing it.

The compose also enables `--long-prefill-token-threshold=4096`,
`--enable-chunked-prefill`, default `enable_thinking=false`/`reasoning_effort=low`,
and the club sampler through `--override-generation-config`. Per-request
parameters still override these defaults.

## Checks

```bash
curl -fsS http://localhost:8113/health
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  bash scripts/bench.sh                 # 3 warmups + 5 measured
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  bash scripts/verify-full.sh
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  bash scripts/verify-stress.sh
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  bash scripts/soak-test.sh --continuous

# Experimental MAX_NUM_SEQS=2 checks:
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  CONCURRENCY=2 ROUNDS=5 PROMPT_TOKENS=16000 GEN_TOKENS=256 \
  bash scripts/concurrency-probe.sh
URL=http://localhost:8113 MODEL=qwen3.8-27b \
  CONTAINER=vllm-qwen38-27b-dual-dflash15-fast \
  SESSIONS=1 TURNS=12 bash scripts/bench-agentic.sh
```

The canonical upstream-image evidence is kept outside the repository while
being reproduced: `canonical-dflash15-244k-bench.log` and `-bench-2.log`
(93.44/160.87 and 93.94/166.77 decode TPS), `-verify-full.log`,
`-verify-stress.log` (224319-token fill), and `-soak.log` (73 MiB growth,
0 errors). The full report also records the WSL2 PCIe/custom-all-reduce state.

### MAX_NUM_SEQS=2 evidence (experimental)

- `dflash15-maxseq2-bench.log`: **92.30 narrative / 161.29 code decode TPS**;
  CV 6.9% / 6.8%, TTFT 78 / 72 ms, peak 37,486 MiB, acceptance 5.41–5.64.
- `dflash15-maxseq2-concurrency.log`: 5 rounds at 2×16K prompts, 0 errors,
  0 silent responses, **81.0 tok/s per stream**, 99.1% retention, peak 41,958
  MiB, 0 MiB post-warm growth.
- `dflash15-maxseq2-bench-agentic.log`: 1×12-turn ramp, 0 tool-call misses;
  TTFT 1.31 s at 1.5K tokens → 9.26 s at 35.4K tokens (7.1× growth).
  This is the expected GDN/SSM recurrent-prefill cost, not a concurrency failure.

`bench.sh` sends `enable_thinking=false` by default for a decode comparison;
this is request-level only. The server does not force thinking off, so model
card/client requests may send `enable_thinking=true`.

The 244K ceiling has a thin VRAM margin under long-context stress. Keep the
single stream and pinned KV pool for the canonical profile; lower context or
KV memory before increasing concurrency.
