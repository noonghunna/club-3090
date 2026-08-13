# Migrating vLLM from Linux to Windows

> A guide for club-3090 users migrating from Linux to native Windows — reuse your existing models, cache, and configs with minimal friction.

**Produced by BlackBox_Labs**  
**Last updated:** 2026-08-13  \
**Tested on:** club-3090 dual-3090 → Windows 10/11, Python 3.12, CUDA 13.3

---

> **Path conventions:** In this guide `$VLLM_HOME` is the directory where you installed vLLM — the Python venv, models, NCCL DLL, configs, and helper scripts all live under it. On the author's machine this was `D:\vllm-build`, but **use any drive with enough free space** (models are 15–50 GB). CUDA Toolkit and Visual Studio Build Tools install to their standard Windows locations (typically `C:\Program Files\...`) and are referenced as such below.


## Table of Contents

1. [Why Migrate?](#why-migrate)
2. [What Changes and What Stays](#what-changes-and-what-stays)
3. [Prerequisites](#prerequisites)
4. [Step 1 — Install the Windows Toolchain](#step-1--install-the-windows-toolchain)
5. [Step 2 — Set Up the vLLM Virtual Environment](#step-2--set-up-the-vllm-virtual-environment)
6. [Step 3 — Reuse Your Linux Models](#step-3--reuse-your-linux-models)
7. [Step 4 — Build vLLM from Source (Custom CUDA)](#step-4--build-vllm-from-source-custom-cuda)
8. [Step 5 — Set Up NCCL for Multi-GPU](#step-5--set-up-nccl-for-multi-gpu)
9. [Step 6 — Port Your club-3090 Compose Configs](#step-6--port-your-club-3090-compose-configs)
10. [Step 7 — Launch and Verify](#step-7--launch-and-verify)
11. [Known Differences from Linux](#known-differences-from-linux)
12. [Troubleshooting](#troubleshooting)
13. [Credits & Acknowledgements](#credits--acknowledgements)

---

## Why Migrate?

If you're running [club-3090](https://github.com/BlackBox-Labs/club-3090) on Linux and considering Windows, here's why this migration matters:

- **No Docker/WSL2 overhead** — native Windows, native performance
- **Reuse your existing model weights** — no redownloading 50 GB models
- **Keep your configs** — the vLLM CLI flags are identical, just port the invocation
- **Multi-GPU works** — with NCCL for Windows, tensor/pipeline parallelism across 2×3090 works on native Windows

The key difference: on Linux you `pip install vllm` and it just works. On Windows, you need to build from source (or use a pre-built wheel) because there are no official vLLM wheels for Windows.

---

## What Changes and What Stays

| Component | Linux | Windows |
|-----------|-------|---------|
| Model weights | `~/models/` or wherever you put them | Same files, just different path |
| vLLM CLI flags | Identical | Identical |
| Docker compose | `.yml` files | Not needed — direct CLI |
| Python venv | `~/.venv` or `venv/` | `$VLLM_HOME\venv\` (or wherever) |
| CUDA toolkit | `/usr/local/cuda` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x` |
| NCCL | System library | `nccl-windows` DLL (see Step 5) |
| Build system | GCC + CUDA nvcc | MSVC + CUDA nvcc |
| Line continuation | `\` | `^` (CMD) or `` ` `` (PowerShell) |

**Nothing changes** about your model weights, quantization, chat templates, or API usage. The only friction is the build and environment setup.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Windows | 10 21H2+ or 11 | 64-bit |
| Python | 3.12.x | Not 3.13+ (vLLM doesn't support yet) |
| CUDA Toolkit | 12.6–13.3 | Match your PyTorch CUDA version |
| MSVC Build Tools | VS 2022 (v143) | Required for compiling C++ extensions |
| GPU | NVIDIA RTX 3090 (12 GB+) | 2× for dual-card configs |
| Disk | 100 GB+ free | For models + build artifacts |
| Git | Any version | For cloning vllm-windows repo |

> **Important:** This guide assumes you already have model weights from your Linux setup. If not, you'll need to download them (see Step 3).

---

## Step 1 — Install the Windows Toolchain

### Python 3.12

1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run the installer — **check "Add python.exe to PATH"** before installing
3. Verify:

```bash
python --version
# Expected: Python 3.12.x
```

### CUDA Toolkit

1. Download from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Select Windows → x86_64 → your preferred version (12.8 or 13.3)
3. Run the installer — default path is fine
4. Verify:

```bash
nvcc --version
# Expected: release 12.8 or 13.3
```

> **If `nvcc` is not recognized**, add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\bin` to your system PATH.

### MSVC Build Tools

1. Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Run the installer
3. Select **C++ build tools** — ensure **MSVC v143 - VS 2022 C++ x64/x86 build tools** is checked
4. Install (uncheck everything else to save space)

Your MSVC path will be something like:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
```

You'll need this path for the build step.

---

## Step 2 — Set Up the vLLM Virtual Environment

Create an isolated venv on your data drive:

```bash
cd $VLLM_HOME
python -m venv vllm-build\venv
```

Activate it:

**Command Prompt:**
```cmd
$VLLM_HOME\venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
$VLLM_HOME\venv\Scripts\Activate.ps1
```

Verify:

```bash
python --version
# Should show Python 3.12.x from the venv path
```

---

## Step 3 — Reuse Your Linux Models

### If your models are already on the same machine

If you migrated your entire `$VLLM_HOME\` (or your system drive) drive, your model files are already there. Just point vLLM to the same directory:

```bash
# Example: if your Linux setup had models at ~/models/
# and that's now at $VLLM_HOME\models\
vllm serve $VLLM_HOME\models\qwen3.6-27b --port 8000
```

### If your models are on a different drive or machine

Copy them over. The safetensors files are platform-agnostic — no conversion needed:

```bash
# From Linux, copy to your Windows machine
# rsync / scp / Windows File Share / external drive — whatever works

# Example: copy from a network share
robocopy $VLLM_HOME\linux-models $VLLM_HOME\models\qwen3.6-27b /E /R:3 /W:5
```

### Verify model integrity

Before launching, verify your model files are intact:

```bash
python -c "
import os, json
model_dir = 'D:/models/qwen3.6-27b'
# Check for safetensors index
index_file = os.path.join(model_dir, 'model.safetensors.index.json')
if os.path.exists(index_file):
    with open(index_file) as f:
        idx = json.load(f)
    files = idx.get('weight_map', {})
    print(f'Model has {len(set(files.values()))} safetensors files')
    print(f'Total weights: {len(files)} tensors')
else:
    print('No safetensors index found — checking for single-file model...')
    single = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(single):
        print('Single safetensors file found')
    else:
        print('ERROR: No safetensors model found')
"
```

### If you need to download fresh models

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $VLLM_HOME\models\qwen2.5-7b-instruct
```

For quantized models (recommended for 24 GB VRAM):

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir $VLLM_HOME\models\qwen2.5-7b-instruct-awq
```

---

## Step 4 — Build vLLM from Source (Custom CUDA)

This is the key step for migration. Building from source lets you:
- Use a **custom CUDA version** (e.g., CUDA 13.3 when PyTorch only ships cu12x)
- **Reuse cached build artifacts** from any previous Linux build
- **Minimize re-downloads** — the source repo is small (~100 MB); the heavy lifting is in the compiled wheels

### Clone the Windows-specific branch

```bash
cd $VLLM_HOME
git clone --single-branch --branch vllm-for-windows https://github.com/SystemPanic/vllm-windows.git
cd $VLLM_HOMEvllm-windows
```

> **Do NOT clone the `main` branch.** The `vllm-for-windows` branch contains Windows-specific patches, build scripts, and CUDA configurations.

### Set up the MSVC compiler environment

Open **Command Prompt** (not PowerShell) and run:

```cmd
:: Update this path to match your VS installation
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64
```

If the command succeeds, your prompt won't change — that's normal. The compiler environment is set in the background.

### Set build environment variables

```cmd
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10

:: Optional: enable cuDNN (if installed)
set USE_CUDNN=1
set CUDNN_LIBRARY_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\lib\x64
set CUDNN_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\include

:: Optional: enable cuSPARSELt
set USE_CUSPARSELT=1
set CUSPARSELT_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\include

:: Optional: enable cuDSS (if installed)
set USE_CUDSS=1
set CUDSS_LIBRARY_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\lib\x64
set CUDSS_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.x\include
```

> **CUDA 13.0–13.2 alignment patch:** If you're on CUDA 13.0 through 13.2, run this as Administrator BEFORE building:
> ```cmd
> python $VLLM_HOME\vllm-windows\fix_cuda_13_align.py
> ```
> CUDA 13.3+ has the fix built-in and does not need this patch.

### Install PyTorch matching your CUDA version

```cmd
:: For CUDA 12.1
pip install torch==2.11+cu121 torchaudio==2.11+cu121 torchvision==0.26.0+cu121 --index-url https://download.pytorch.org/whl/cu121

:: For CUDA 12.4
pip install torch==2.11+cu124 torchaudio==2.11+cu124 torchvision==0.26.0+cu124 --index-url https://download.pytorch.org/whl/cu124

:: For CUDA 12.6
pip install torch==2.11+cu126 torchaudio==2.11+cu126 torchvision==0.26.0+cu126 --index-url https://download.pytorch.org/whl/cu126

:: For CUDA 13.0
pip install torch==2.11+cu130 torchaudio==2.11+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130

:: For CUDA 13.3
pip install torch==2.11+cu133 torchaudio==2.11+cu133 torchvision==0.26.0+cu133 --index-url https://download.pytorch.org/whl/cu133
```

> **Pro tip for migration:** If you have a Linux machine with PyTorch wheels cached, you can copy those `.whl` files directly into the Windows venv's `pip` cache folder (`~/.cache/pip/`) to avoid re-downloading.

### Install vLLM build dependencies

```cmd
pip install -r requirements\build\cuda.txt
pip install -r requirements\cuda.txt
pip install -r requirements\windows.txt
```

### Build and install

```cmd
pip install . --no-build-isolation -vvv
```

This will:
1. Compile CUDA kernels (triton, flashinfer, etc.)
2. Build C++ extensions
3. Install vLLM into your venv

**Expected time:** 20–45 minutes depending on CPU cores and SSD speed.

> **Tip:** The `-vvv` flag gives verbose output so you can see progress. The build will show kernel compilation stages — don't interrupt it.

### Verify the installation

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:

```
vLLM: 0.26.0
PyTorch: 2.11.0+cu133
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## Step 5 — Set Up NCCL for Multi-GPU

For dual-card tensor/pipeline parallelism (like club-3090's `vllm/dual` config), you need NCCL for Windows. The official NVIDIA NCCL doesn't ship a Windows DLL.

### Option A — Pre-built nccl-windows release (recommended)

1. Go to [https://github.com/SystemPanic/nccl-windows/releases](https://github.com/SystemPanic/nccl-windows/releases)
2. Download the latest release
3. Extract `nccl.dll` to a known location:

```bash
mkdir $VLLM_HOME\nccl-windows\install\bin
:: Copy nccl.dll here
```

4. Set the environment variable before running vLLM:

```cmd
set VLLM_NCCL_SO_PATH=$VLLM_HOME\nccl-windows\install\bin\nccl.dll
```

### Option B — Build NCCL from source

If you need a custom build:

```bash
cd $VLLM_HOME
git clone --branch nccl-windows https://github.com/SystemPanic/nccl-windows.git
cd $VLLM_HOMEnccl-windows
make -j src.build
make pkg.txz.build
# Extract the resulting tarball to your desired install location
```

### Why NCCL matters for club-3090 migrants

Your club-3090 dual-card configs use `--tensor-parallel-size 2` or `--pipeline-parallel-size 2`. Without NCCL for Windows, multi-GPU parallelism falls back to slower PCIe ring communication or fails entirely. The `VLLM_NCCL_SO_PATH` environment variable tells vLLM where to find the Windows-compiled NCCL DLL.

---

## Step 6 — Port Your club-3090 Compose Configs

This is where the migration pays off. Your club-3090 compose configs are just vLLM CLI invocations — they translate directly to Windows commands.

### Example: Porting a club-3090 dual config

Your club-3090 `vllm/dual` compose might look like this on Linux:

```yaml
# From club-3090 compose
container_name: vllm-dual
command: >
  vllm serve Qwen/Qwen2.5-7B-Instruct
    --served-model-name qwen-8010
    --quantization awq
    --dtype float16
    --tensor-parallel-size 2
    --pipeline-parallel-size 1
    --max-model-len 262144
    --gpu-memory-utilization 0.90
    --max-num-seqs 2
    --max-num-batched-tokens 8192
    --kv-cache-dtype fp8_e4m3
    --trust-remote-code
    --enable-prefix-caching
    --enable-chunked-prefill
    --port 8000
```

On Windows, this becomes a single CMD invocation:

```cmd
vllm serve $VLLM_HOME\models\qwen2.5-7b-instruct-awq ^
  --served-model-name qwen-8010 ^
  --quantization awq ^
  --dtype float16 ^
  --tensor-parallel-size 2 ^
  --pipeline-parallel-size 1 ^
  --max-model-len 262144 ^
  --gpu-memory-utilization 0.90 ^
  --max-num-seqs 2 ^
  --max-num-batched-tokens 8192 ^
  --kv-cache-dtype fp8_e4m3 ^
  --trust-remote-code ^
  --enable-prefix-caching ^
  --enable-chunked-prefill ^
  --host 0.0.0.0 ^
  --port 8000
```

### Key differences to note

| Linux | Windows |
|-------|---------|
| `\` (backslash) line continuation | `^` (caret) in CMD, `` ` `` in PowerShell |
| `--model` | `--model` (same, but use local path if model is downloaded) |
| Docker `--network host` | Not needed — `--host 0.0.0.0` binds to all interfaces |
| `.env` file for NCCL | `set VLLM_NCCL_SO_PATH=...` before the command |
| `docker compose up` | Direct `vllm serve` command |

### Creating a launcher script

For convenience, create a PowerShell launcher script similar to club-3090's `launch.sh`:

```powershell
# $VLLM_HOME\launch-qwen3.6-27b.ps1
# Ported from club-3090 vllm/dual config

$env:VLLM_NCCL_SO_PATH = "$VLLM_HOME\nccl-windows\install\bin\nccl.dll"
$env:NCCL_P2P_DISABLE = "1"
$env:NCCL_CUMEM_ENABLE = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False,max_split_size_mb:512"
$env:XDG_CACHE_HOME = "$VLLM_HOME\cache"
$env:TRITON_CACHE_DIR = "$VLLM_HOME\cache\triton"

vllm serve $VLLM_HOME\models\qwen3.6-27b-autoround-int4 `
  --served-model-name qwen-8010 `
  --quantization auto_round `
  --dtype float16 `
  --tensor-parallel-size 2 `
  --pipeline-parallel-size 1 `
  --max-model-len 262144 `
  --gpu-memory-utilization 0.90 `
  --max-num-seqs 2 `
  --max-num-batched-tokens 8192 `
  --kv-cache-dtype fp8_e4m3 `
  --trust-remote-code `
  --chat-template "$VLLM_HOME\configs\chat_template.jinja" `
  --reasoning-parser qwen3 `
  --default-chat-template-kwargs '{"enable_thinking": false}' `
  --enable-auto-tool-choice `
  --tool-call-parser qwen3_coder `
  --enable-prefix-caching `
  --enable-chunked-prefill `
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' `
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}' `
  --host 0.0.0.0 `
  --port 8010 `
  --disable-custom-all-reduce
```

Make it executable:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\$VLLM_HOME\launch-qwen3.6-27b.ps1
```

---

## Step 7 — Launch and Verify

### Start the server

```bash
vllm serve $VLLM_HOME\models\your-model --port 8000
```

On first run, vLLM will:
1. Load model weights into GPU memory (1–5 minutes)
2. Compile triton kernels (5–15 minutes on first run)
3. Start the HTTP server

> **Be patient on the first run.** The kernel compilation is a one-time cost. Subsequent launches will be much faster.

### Test the endpoint

```bash
curl -sf http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"qwen-8010\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, who are you?\"}], \"max_tokens\": 50}"
```

### Verify multi-GPU (if applicable)

```bash
python -c "
import torch
print(f'GPUs detected: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"
```

Expected output for dual 3090:

```
GPUs detected: 2
  GPU 0: NVIDIA GeForce RTX 3090
    Memory: 24.0 GB
  GPU 1: NVIDIA GeForce RTX 3090
    Memory: 24.0 GB
```

### Run a benchmark (optional)

The club-3090 repository ships native PowerShell benchmark and reporting scripts — **do not write your own**. The scripts handle logging, engine-side metrics capture, and consolidated reporting automatically.

```powershell
# Canonical benchmark (narrative + code, 3 warmup + 5 measured runs each)
.\scripts\bench.ps1

# With custom parameters
.\scripts\bench.ps1 -Runs 10 -MaxTokensNarr 2000 -Only narr

# Quick directional A/B sweep (1 warmup, 1 measured run, narrative only)
.\scripts\bench.ps1 --quick

# Engine-side metrics capture (requires SERVER_LOG for bare-metal)
$env:SERVER_LOG = "$VLLM_HOME\logs\server.log"
.\scripts\bench.ps1

# Full triage report with all stages
.\scripts\report.ps1 -Full > my-rig-report.md

# Individual stages
.\scripts\report.ps1 -Verify    # verify-full (~1-2 min)
.\scripts\report.ps1 -Stress    # verify-stress (~10-20 min)
.\scripts\report.ps1 -Soak      # soak-test continuous (~25 min)
.\scripts\report.ps1 -Bench     # canonical TPS (~5 min)
```

**Available scripts:**

| Script | Purpose |
|--------|---------|
| `bench.ps1` | Canonical benchmark — narrative + code, wall_TPS, decode_TPS, TTFT, engine-side metrics |
| `report.ps1` | Triage report — hardware, GPU state, engine logs, stage gating, verdict accounting |
| `verify-full.ps1` | Functional tests — server reachability, completion, tool calling, streaming, quality |
| `verify-stress.ps1` | Stress tests — long-context needles, tool prefill OOM, context ceiling ladder |
| `soak-test.ps1` | Runtime soak test — VRAM accretion, TTFT growth, decode TPS retention |
| `capture.ps1` | Shared library — engine-side log scraping (expert cache, spec-decoding, bypass rates) |

**Key features:**

- **Engine-side metrics**: Expert cache census, spec-decoding acceptance rates, bypass counters, decode/prefill timings (via `capture.ps1`)
- **Config fingerprinting**: Model flags, KV type, offload detection, moe-cache environment variables
- **Stage gating**: Engine liveness probe between stages — skips remaining if engine dies mid-run
- **Verdict accounting**: PASS/FAIL/ADVISORY per stage with exit code reflecting worst verdict
- **CARD rendering**: Snapshot cards and A/B comparison against baseline logs
- **Short-EOS detection**: Excludes degenerate early-terminating runs (<25% of max_tokens)
- **Decode granularity**: Auto-detects token vs canvas (dLLM) — marks decode_TPS as n/a for canvas models
- **STREAM calibration**: Optional RAM-bandwidth ceiling measurement via numpy
- **BENCH_MOCK**: CI-friendly mock output mode
- **Comprehensive redaction**: Paths, hostnames, ports automatically redacted from reports

---

## Known Differences from Linux

### 1. Line continuation syntax

Linux uses `\` for line continuation. Windows CMD uses `^`, PowerShell uses `` ` ``.

```bash
# Linux
vllm serve /models/qwen2.5-7b \
  --port 8000 \
  --tensor-parallel-size 2

# Windows CMD
vllm serve $VLLM_HOME\models\qwen2.5-7b ^
  --port 8000 ^
  --tensor-parallel-size 2

# Windows PowerShell
vllm serve $VLLM_HOME\models\qwen2.5-7b `
  --port 8000 `
  --tensor-parallel-size 2
```

### 2. Path separators

Linux uses `/`. Windows uses `\` (or `/` in most tools, including vLLM).

```bash
# Linux — both work
vllm serve ~/models/qwen2.5-7b
vllm serve /home/user/models/qwen2.5-7b

# Windows — prefer forward slashes for vLLM
vllm serve D:/models/qwen2.5-7b
# or backslashes (also works in most cases)
vllm serve $VLLM_HOME\models\qwen2.5-7b
```

### 3. NCCL is not pre-installed

Linux ships with NCCL as part of the NVIDIA driver stack. Windows requires the separate `nccl-windows` DLL. Always set `VLLM_NCCL_SO_PATH` for multi-GPU workloads.

### 4. No Docker compose

Linux users of club-3090 use `docker compose up`. On Windows native, you run `vllm serve` directly. The CLI flags are identical — just port the invocation from the `.yml` to a shell command or PowerShell script.

### 5. CUDA memory allocator behavior

Windows may handle CUDA memory allocation differently. If you see OOM errors that didn't occur on Linux, try:

```cmd
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:512
```

### 6. Triton kernel cache location

On Linux, Triton caches compiled kernels in `~/.triton/cache/`. On Windows, set:

```cmd
set TRITON_CACHE_DIR=$VLLM_HOME\cache\triton
```

This puts the cache on your fast data drive and avoids permission issues with your system drive.

---

## Troubleshooting

### "NCCL error: unhandled error" on multi-GPU

```cmd
:: Add these before your vllm serve command
set NCCL_P2P_DISABLE=1
set NCCL_CUMEM_ENABLE=0
set VLLM_NCCL_SO_PATH=$VLLM_HOME\nccl-windows\install\bin\nccl.dll
```

#### ⚠ Gotcha — do NOT pin `NCCL_SOCKET_IFNAME=lo` / `GLOO_SOCKET_IFNAME=lo`

`lo` is the **Linux** loopback interface name. The nccl-windows build does not recognise it,
and pinning it produces a hard **`NCCL error: invalid usage`** at startup (seen on a 2×3090,
TP=2 boot: weights load, then NCCL init aborts). Leave both **UNSET** — with
`NCCL_P2P_DISABLE=1`, NCCL auto-selects shared memory for intra-node tensor parallelism,
which is correct on Windows. This is the #1 cause of a multi-GPU boot that "loads the
weights and then dies on NCCL init".

#### Loopback rendezvous (WiFi-independent)

To keep the rendezvous off the NIC entirely, pin it to loopback before launching:

```cmd
set VLLM_HOST_IP=127.0.0.1
```

This binds c10d/gloo to `127.0.0.1` (always up). A benign one-time hostname probe
(`client socket has failed to connect to [HOSTNAME]:<port> (10049)`) may still print, then
resolve to `127.0.0.1` — harmless. The `invalid usage` error above is not.

### "ModuleNotFoundError: No module named 'numpy._core...'"

Multiple Python environments conflicting. Clean rebuild:

```bash
deactivate
rmdir /s /q $VLLM_HOME\venv
python -m venv $VLLM_HOME\venv
$VLLM_HOME\venv\Scripts\activate
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r $VLLM_HOME\vllm-windows\requirements\build\cuda.txt
pip install -r $VLLM_HOME\vllm-windows\requirements\cuda.txt
pip install -r $VLLM_HOME\vllm-windows\requirements\windows.txt
pip install $VLLM_HOME\vllm-windows --no-build-isolation -vvv
```

### "CUDA out of memory" on dual-GPU

- Verify both GPUs are detected: `python -c "import torch; print(torch.cuda.device_count())"`
- Ensure `--tensor-parallel-size` matches your GPU count
- Reduce `--gpu-memory-utilization` from 0.90 to 0.80
- Close GPU-intensive apps (Discord overlay, Chrome, etc.)

### "MSVC compiler not found"

Install MSVC Build Tools (Step 1). Then rebuild:

```bash
pip uninstall vllm -y
pip cache purge
pip install $VLLM_HOME\vllm-windows --no-build-isolation -vvv
```

### vLLM starts but never responds on first run

Normal — kernel compilation takes 5–15 minutes on first run. Wait patiently. You can monitor progress in the terminal. Subsequent launches will be much faster.

### Model loads but produces garbled output

- Verify the model files are intact (Step 3)
- Check that the quantization type matches the model:
  - AWQ models → `--quantization awq`
  - AutoRound models → `--quantization auto_round`
  - Compressed-tensors (AWQ) → `--quantization compressed-tensors`
  - Standard FP16/BF16 → no `--quantization` flag needed
- Try `--dtype float16` or `--dtype bfloat16` explicitly

### "pip install vllm" fails with compilation errors

Ensure you've cloned the `vllm-for-windows` branch, not `main`:

```bash
cd $VLLM_HOMEvllm-windows
git branch --show-current
# Should show: vllm-for-windows
```

If it shows `main`, switch:

```bash
git fetch origin vllm-for-windows
git checkout vllm-for-windows
```

---

## What's Next

Now that you're running on native Windows:

- **Model management:** Use `huggingface-cli` to download and manage models
- **Quantization:** Use AWQ, GPTQ, or AutoRound for smaller model sizes (4-bit = ~60% less VRAM)
- **Speculative decoding:** `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` for faster inference
- **Tool use:** `--enable-auto-tool-choice --tool-call-parser qwen3_coder` for function calling
- **Prefix caching:** `--enable-prefix-caching` for repeated prompts (saves compute)
- **Chunked prefill:** `--enable-chunked-prefill` for better throughput with long prompts

### Creating a model launcher menu

For a club-3090-like experience, create a PowerShell menu script:

```powershell
# $VLLM_HOME\vllm-menu.ps1

$models = @(
    @{ Name = "Qwen3.6-27B AutoRound INT4"; Path = "$VLLM_HOME\models\qwen3.6-27b-autoround-int4"; Quant = "auto_round"; Dtype = "float16"; TP = 2; Util = 0.90; Port = 8010; ModelName = "qwen-8010"; MTP = 3 },
    @{ Name = "Qwen3.6-35B-A3B AutoRound INT4"; Path = "$VLLM_HOME\models\qwen3.6-35b-a3b-autoround-int4"; Quant = "auto_round"; Dtype = "float16"; TP = 2; Util = 0.70; Port = 8010; ModelName = "qwen-8010"; MTP = $null },
    @{ Name = "Tess-4-27B AutoRound W4A16"; Path = "$VLLM_HOME\models\Tess-4-27B-AutoRound-W4A16-Tuning"; Quant = "auto_round"; Dtype = "float16"; TP = 2; Util = 0.90; Port = 8010; ModelName = "qwen-8010"; MTP = 3 },
    @{ Name = "AgentWorld-35B-A3B AWQ INT4"; Path = "$VLLM_HOME\models\Qwen-AgentWorld-35B-A3B-AWQ-INT4"; Quant = "compressed-tensors"; Dtype = "bfloat16"; TP = 2; Util = 0.70; Port = 8010; ModelName = "qwen-8010"; MTP = $null; LangModelOnly = $true }
)

Write-Host "=== vLLM Model Launcher (Windows) ===" -ForegroundColor Cyan
Write-Host "Ported from club-3090 Linux configs" -ForegroundColor Gray
Write-Host ""

for ($i = 0; $i -lt $models.Count; $i++) {
    Write-Host "[$($i+1)] $($models[$i].Name)" -ForegroundColor White
}

Write-Host ""
$choice = Read-Host "Select model (or 'q' to quit)"

if ($choice -eq 'q') { exit }

$idx = [int]$choice - 1
if ($idx -lt 0 -or $idx -ge $models.Count) {
    Write-Host "Invalid selection." -ForegroundColor Red
    exit
}

$m = $models[$idx]

Write-Host ""
Write-Host "Loading $($m.Name)..." -ForegroundColor Yellow
Write-Host "  Path: $($m.Path)"
Write-Host "  GPUs: 2x RTX 3090, TP=$($m.TP)"
Write-Host "  Port: $($m.Port)"
Write-Host ""

$env:VLLM_NCCL_SO_PATH = "$VLLM_HOME\nccl-windows\install\bin\nccl.dll"
$env:NCCL_P2P_DISABLE = "1"
$env:NCCL_CUMEM_ENABLE = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False,max_split_size_mb:512"
$env:XDG_CACHE_HOME = "$VLLM_HOME\cache"
$env:TRITON_CACHE_DIR = "$VLLM_HOME\cache\triton"

$cmd = "vllm serve $($m.Path) --served-model-name $($m.ModelName) --quantization $($m.Quant) --dtype $($m.Dtype) --tensor-parallel-size $($m.TP) --pipeline-parallel-size 1 --max-model-len 262144 --gpu-memory-utilization $($m.Util) --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8_e4m3 --trust-remote-code --host 0.0.0.0 --port $($m.Port) --disable-custom-all-reduce"

if ($m.MTP) {
    $cmd += " --speculative-config `"{`"method`":`"mtp`",`"num_speculative_tokens`":$($m.MTP)}`"`""
}

if ($m.LangModelOnly) {
    $cmd += " --language-model-only"
}

Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""
Invoke-Expression $cmd
```

> **Note — the shipped launcher is YAML-driven, not this hardcoded array.** The example
> above is the *old* club-3090 port style kept for illustration. The production
> `$VLLM_HOME\scripts\vllm-menu.ps1` reads all four profiles from YAML under
> `$VLLM_HOME\configs\<model>\...` (edit-and-launch — no code changes). Its **Stop**
> function is **scoped**: it kills only the `:8010` listen-tree plus any `python.exe` under
> your vLLM build venv (e.g. `$VLLM_HOME\venv`). It never blanket-kills python, so any
> other python tools on the box are left running. If you adapt this script, do the same — match
> on the listen socket / venv path, never `Get-Process python | Stop-Process -Force`.

---

## Credits & Acknowledgements

This guide was produced by **BlackBox_Labs** based on real-world migration experience from the [club-3090](https://github.com/BlackBox-Labs/club-3090) Linux setup to native Windows on dual RTX 3090 hardware.

### Projects referenced and credited

| Project | What it provides | Link |
|---------|-----------------|------|
| **vLLM** | The core inference engine — PagedAttention, continuous batching, high-throughput serving | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **vllm-windows** (SystemPanic) | Pre-built Windows wheels and the `vllm-for-windows` branch with Windows-specific patches, CUDA 13+ support, and build instructions | [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) |
| **nccl-windows** (SystemPanic) | Windows-compiled NCCL library enabling tensor and pipeline parallelism across multiple GPUs on Windows | [SystemPanic/nccl-windows](https://github.com/SystemPanic/nccl-windows) |
| **club-3090** (BlackBox_Labs) | Curated configs, benchmarks, and multi-engine recipes for RTX 3090 dual-card setups — the source of truth for dual-card vLLM configs | [BlackBox-Labs/club-3090](https://github.com/BlackBox-Labs/club-3090) |
| **PyTorch** (PyTorch team / NVIDIA) | GPU-accelerated tensor library with CUDA support | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **FlashInfer** | Optimized attention kernels for vLLM | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| **Triton** | NVIDIA's domain-specific language for writing GPU kernels | [triton-lang/triton](https://github.com/triton-lang/triton) |

### Special thanks

- **SystemPanic** for maintaining the vllm-windows and nccl-windows projects that make native Windows vLLM possible
- **club-3090 community** for the dual-card vLLM configs, benchmarks, and quantization recipes that served as the migration baseline
- **NVIDIA** for GPU hardware support and CUDA toolchain
- **vLLM contributors** — originally developed at UC Berkeley's Sky Computing Lab, now maintained by 2000+ contributors

### Citation

If you use vLLM for your research, please cite the original paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

---

## Quick Reference

| Component | Version |
|-----------|---------|
| Python | 3.12.x |
| PyTorch | 2.11.0+cu12x / cu13x |
| vLLM | 0.26.0+ (built from vllm-for-windows branch) |
| CUDA Toolkit | 12.6–13.3 |
| cuDNN | 9.x (optional) |
| MSVC | VS 2022 Build Tools (v143) |
| NCCL | SystemPanic/nccl-windows (Windows builds) |
| Source repo | `git clone --branch vllm-for-windows https://github.com/SystemPanic/vllm-windows.git` |
