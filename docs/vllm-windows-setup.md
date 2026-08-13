# Setting Up vLLM Natively on Windows

> A beginner-to-intermediate guide for installing and running vLLM on Windows 10/11 — no Docker required.

**Produced by BlackBox_Labs**  
**Last updated:** 2026-08-13  \
**Tested on:** Windows 10/11, Python 3.12, CUDA 12.8, RTX 3090 × 2

---

> **Path conventions:** In this guide `$VLLM_HOME` is the directory where you installed vLLM — the Python venv, models, NCCL DLL, configs, and helper scripts all live under it. On the author's machine this was `D:\vllm-build`, but **use any drive with enough free space** (models are 15–50 GB). CUDA Toolkit and Visual Studio Build Tools install to their standard Windows locations (typically `C:\Program Files\...`) and are referenced as such below.


## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Drive Layout](#drive-layout)
4. [Step 1 — Install Python 3.12](#step-1--install-python-312)
5. [Step 2 — Install NVIDIA CUDA Toolkit](#step-2--install-nvidia-cuda-toolkit)
6. [Step 3 — Install MSVC Build Tools](#step-3--install-msvc-build-tools)
7. [Step 4 — Create the vLLM Virtual Environment](#step-4--create-the-vllm-virtual-environment)
8. [Step 5 — Install vLLM (Two Methods)](#step-5--install-vllm-two-methods)
9. [Step 6 — Download a Model and Run a Server](#step-6--download-a-model-and-run-a-server)
10. [Step 7 — Using the OpenAI-Compatible API](#step-7--using-the-openai-compatible-api)
11. [Windows-Specific Environment Variables](#windows-specific-environment-variables)
12. [Multi-GPU Setup](#multi-gpu-setup)
13. [Troubleshooting](#troubleshooting)
14. [Credits & Acknowledgements](#credits--acknowledgements)

---

## Overview

vLLM is one of the fastest open-source LLM inference engines. On Linux it installs with a single `pip install vllm`. On Windows, it needs a few extra steps:

- **CUDA Toolkit** for GPU compute
- **MSVC Build Tools** for compiling C++ extensions (triton, flashinfer, etc.)
- **Custom NCCL** for multi-GPU tensor/pipeline parallelism (Windows doesn't ship with NCCL)

This guide walks you through every step. By the end you'll have a local vLLM server answering OpenAI-compatible API requests.

> **Why native Windows?** No Docker, no WSL2, no VMs. Just Python + CUDA + vLLM running directly on Windows.

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) | RTX 3090/4090 (24 GB VRAM) |
| VRAM | 12 GB | 24 GB+ (multi-GPU) |
| RAM | 16 GB | 32 GB+ |
| Disk | 50 GB free | 100 GB+ (models are large) |
| Python | 3.10–3.12 | 3.12 (latest stable) |
| OS | Windows 10 21H2+ | Windows 11 |

> **Note:** vLLM requires an NVIDIA GPU. AMD/Intel GPUs are not supported.

---

## Drive Layout

vLLM + models consume significant space. We recommend:

| Drive | Purpose |
|-------|---------|
| `C:\` | Windows, Python, CUDA Toolkit, Build Tools (system apps) |
| `$VLLM_HOME\` | vLLM installation, virtual environment, models, cache |

Keep Python and CUDA on `C:\` (they're small). Put the venv and models on a separate drive — a model can be 15–50 GB.

---

## Step 1 — Install Python 3.12

### Download

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the **Windows installer** for Python 3.12 (e.g., `python-3.12.x-amd64.exe`)
3. Run the installer

### Critical settings

When the installer opens, **check this box first**:

```
☑ Add python.exe to PATH
```

Then click **Install Now**.

### Verify

Open a new terminal (Command Prompt or PowerShell) and run:

```bash
python --version
```

You should see:

```
Python 3.12.x
```

If you get "python is not recognized", close and reopen your terminal, or add Python to PATH manually.

---

## Step 2 — Install NVIDIA CUDA Toolkit

vLLM needs the NVIDIA CUDA toolkit to compile and run GPU code.

### Download

1. Go to the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Select:
   - **Operating System:** Windows
   - **Architecture:** x86_64
   - **Version:** 12.6 or 12.8 (compatible with PyTorch 2.x)
   - **Installer Type:** exe (local)
3. Download and run the installer

### Installation notes

- Use the **Custom (Advanced)** install
- Keep the default install path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- The installer will add CUDA to your system PATH automatically

### Verify CUDA

Open a new terminal and run:

```bash
nvcc --version
```

You should see something like:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Cuda compilation tools, release 12.8, V12.8.x
```

> **Important:** If `nvcc` is not recognized, the CUDA installer didn't add it to PATH. Manually add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your system PATH.

---

## Step 3 — Install MSVC Build Tools

vLLM and its dependencies (like `flashinfer`, `triton`) need to compile C++ extensions. This requires the Microsoft Visual C++ Build Tools.

### Download

1. Go to [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Download the **Build Tools for Visual Studio** (not the full IDE — it's much smaller)
3. Run the installer

### What to select

In the Workloads screen, check:

```
☑ C++ build tools
```

In the "Installation details" panel on the right, ensure:

```
☑ MSVC v143 - VS 2022 C++ x64/x86 build tools
```

You can uncheck everything else. Click **Install**.

### Verify

The build tools will be at something like:

```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64
```

You'll need this path later when setting up the environment.

---

## Step 4 — Create the vLLM Virtual Environment

We use a virtual environment to keep vLLM isolated from your other Python projects.

### Create the venv

Open a terminal and navigate to your data drive (e.g., `$VLLM_HOME\`):

```bash
cd $VLLM_HOME
python -m venv vllm-build\venv
```

### Activate the venv

**Command Prompt:**
```cmd
$VLLM_HOME\venv\Scripts\activate.bat
```

**PowerShell:**
```powershell
$VLLM_HOME\venv\Scripts\Activate.ps1
```

Your prompt should change to show `(venv)`. Verify:

```bash
python --version
```

Should show **Python 3.12.x** from the venv path.

---

## Step 5 — Install vLLM (Two Methods)

### Method A — Pre-built Wheel (Recommended for most users)

The easiest path: install a pre-built wheel from the [vllm-windows](https://github.com/SystemPanic/vllm-windows) project.

1. Go to [https://github.com/SystemPanic/vllm-windows/releases](https://github.com/SystemPanic/vllm-windows/releases)
2. Find the latest release that matches your Python, PyTorch, and CUDA versions
3. Download the `.whl` file
4. Install it:

```bash
pip install path\to\downloaded\vllm‑cp312‑cp312‑win_amd64.whl
```

> **Matching your environment:** The wheel filename encodes the Python version, PyTorch version, and CUDA version. For example `vllm-0.26.0-cp312-cp312-win_amd64.whl` targets Python 3.12. Check the release page for the exact version.

### Method B — Build from Source (For custom CUDA versions or bleeding edge)

If the pre-built wheel doesn't match your setup, build from source using the `vllm-for-windows` branch.

#### Prerequisites

- MSVC Build Tools installed (Step 3)
- CUDA Toolkit installed (Step 2)
- Python 3.12 in your venv

#### Build steps

Open **Command Prompt** (not PowerShell) and run:

```cmd
:: Set your VS installation path (update if different)
set "VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"

:: Launch the x64 compiler environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Clone the Windows-specific branch
cd $VLLM_HOME
git clone --single-branch --branch vllm-for-windows https://github.com/SystemPanic/vllm-windows.git
cd $VLLM_HOMEvllm-windows

:: Set build environment variables
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set MAX_JOBS=10

:: (Optional) Enable cuDNN if installed
set USE_CUDNN=1
set CUDNN_LIBRARY_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64
set CUDNN_INCLUDE_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include
```

> **CUDA 13.0–13.2 note:** These CUDA versions have 128-byte alignment in `cuda.h`. MSVC doesn't support passing over-aligned types by value. CUDA 13.3+ fixes this. If you're on CUDA 13.0–13.2, run this as Administrator first:
> ```cmd
> python $VLLM_HOME\vllm-windows\fix_cuda_13_align.py
> ```

#### Install dependencies and build

```cmd
:: Install PyTorch matching your CUDA version
pip install torch==2.11+cu121 torchaudio==2.11+cu121 torchvision==0.26.0+cu121 --index-url https://download.pytorch.org/whl/cu121

:: Install vLLM build requirements
pip install -r requirements\build\cuda.txt
pip install -r requirements\cuda.txt
pip install -r requirements\windows.txt

:: Build and install
pip install . --no-build-isolation -vvv
```

This may take 20–40 minutes as it compiles CUDA kernels.

### Verify the installation (both methods)

```bash
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:

```
vLLM: 0.26.0
PyTorch: 2.11.0+cu121
CUDA available: True
```

---

## Step 6 — Download a Model and Run a Server

### Create a models directory

```bash
mkdir $VLLM_HOME\models
```

### Download a model

Install the Hugging Face CLI and download a model:

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir $VLLM_HOME\models\qwen2.5-7b-instruct
```

> **Note:** Models can be 15–50 GB. For quantized models (INT4/INT8), use the `-awq` or `-gptq` variants which are smaller.

### Start the server

```bash
vllm serve $VLLM_HOME\models\qwen2.5-7b-instruct ^
  --host 0.0.0.0 ^
  --port 8000 ^
  --dtype auto ^
  --max-model-len 4096
```

On the first run, vLLM will:
1. Load the model weights into GPU memory (1–5 minutes)
2. Compile the inference engine (triton kernels, etc.)
3. Start the HTTP server

> **PowerShell note:** In PowerShell, use backticks `` ` `` for line continuation instead of `^`.

### Test it

Open a new terminal and run:

```bash
curl http://localhost:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, who are you?\"}], \"max_tokens\": 50}"
```

If you see a JSON response with the model's reply, congratulations — vLLM is running!

---

## Step 7 — Using the OpenAI-Compatible API

vLLM exposes an OpenAI-compatible API. You can use it with any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## Windows-Specific Environment Variables

For multi-GPU setups or performance tuning, set these before running `vllm serve`:

### Command Prompt (.bat / .cmd)

```bat
@echo off
set NCCL_P2P_DISABLE=1
set NCCL_CUMEM_ENABLE=0
set VLLM_NCCL_SO_PATH=$VLLM_HOME\nccl-windows\install\bin\nccl.dll
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set VLLM_NO_USAGE_STATS=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:512
set XDG_CACHE_HOME=$VLLM_HOME\cache
set TRITON_CACHE_DIR=$VLLM_HOME\cache\triton
vllm serve $VLLM_HOME\models\your-model --port 8000
```

### PowerShell (.ps1)

```powershell
$env:NCCL_P2P_DISABLE = "1"
$env:NCCL_CUMEM_ENABLE = "0"
$env:VLLM_NCCL_SO_PATH = "$VLLM_HOME\nccl-windows\install\bin\nccl.dll"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:VLLM_NO_USAGE_STATS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False,max_split_size_mb:512"
$env:XDG_CACHE_HOME = "$VLLM_HOME\cache"
$env:TRITON_CACHE_DIR = "$VLLM_HOME\cache\triton"
vllm serve $VLLM_HOME\models\your-model --port 8000
```

### Why these matter on Windows

| Variable | Purpose |
|----------|---------|
| `NCCL_P2P_DISABLE=1` | Disables GPU-to-GPU NVLink P2P (unreliable on Windows, forces PCIe ring) |
| `NCCL_CUMEM_ENABLE=0` | Disables cumulative memory allocator (can cause OOM on Windows) |
| `VLLM_NCCL_SO_PATH` | Points vLLM to the Windows-compiled NCCL DLL (see Step 5.5) |
| `VLLM_HOST_IP=127.0.0.1` | Pins NCCL/gloo rendezvous to loopback — multi-GPU works without touching WiFi/Ethernet (see Troubleshooting) |
| `PYTORCH_CUDA_ALLOC_CONF` | Reduces CUDA memory fragmentation |
| `XDG_CACHE_HOME` / `TRITON_CACHE_DIR` | Puts Triton kernel cache on your fast data drive |

---

## Step 5.5 — NCCL for Multi-GPU (Windows)

For tensor parallelism or pipeline parallelism across multiple GPUs, you need a Windows-compiled NCCL library. The official NCCL from NVIDIA doesn't ship a Windows DLL.

### Option A — Use the pre-built nccl-windows release

1. Go to [https://github.com/SystemPanic/nccl-windows/releases](https://github.com/SystemPanic/nccl-windows/releases)
2. Download the latest release
3. Extract the `nccl.dll` to a known location, e.g., `$VLLM_HOME\nccl-windows\install\bin\nccl.dll`
4. Set the environment variable:

```bash
set VLLM_NCCL_SO_PATH=$VLLM_HOME\nccl-windows\install\bin\nccl.dll
```

### Option B — Build NCCL from source

If you need a custom build:

1. Clone the repository:

```bash
cd $VLLM_HOME
git clone --branch nccl-windows https://github.com/SystemPanic/nccl-windows.git
cd $VLLM_HOMEnccl-windows
```

2. Build:

```bash
make -j src.build
```

3. Install:

```bash
make pkg.txz.build
# Extract the resulting tarball to your desired install location
```

4. Set `VLLM_NCCL_SO_PATH` to the `nccl.dll` in your install directory's `bin/` folder.

> **Note:** The `nccl-windows` project is a community effort providing Windows-compatible NCCL builds. The official NVIDIA NCCL does not support Windows natively.

---

## Multi-GPU Setup

If you have multiple GPUs (e.g., 2× RTX 3090), use tensor parallelism:

```bash
vllm serve $VLLM_HOME\models\qwen2.5-7b-instruct ^
  --host 0.0.0.0 ^
  --port 8000 ^
  --dtype auto ^
  --tensor-parallel-size 2 ^
  --max-model-len 4096
```

The `--tensor-parallel-size 2` flag tells vLLM to split the model across both GPUs.

Verify your GPUs are detected:

```bash
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

For pipeline parallelism:

```bash
vllm serve $VLLM_HOME\models\your-model ^
  --tensor-parallel-size 1 ^
  --pipeline-parallel-size 2 ^
  --port 8000
```

---

## Troubleshooting

### "No module named 'numpy._core._multiarray_umath'"

This happens when multiple Python environments conflict (e.g., the venv inherits numpy from a system install). Fix:

```bash
deactivate
rmdir /s /q $VLLM_HOME\venv
python -m venv $VLLM_HOME\venv
$VLLM_HOME\venv\Scripts\activate
pip install numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm
```

### "ModuleNotFoundError: No module named 'torch'"

PyTorch wasn't installed. Reinstall:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"

- Reduce `--max-model-len`
- Use a smaller or quantized model
- Close GPU-intensive applications (Discord overlay, Chrome, etc.)
- For multi-GPU, ensure `--tensor-parallel-size` matches your GPU count

### "nccl error: unhandled error" / "NCCL error: invalid usage"

On Windows, NCCL (NVIDIA Collective Communications Library) can be tricky. Start with the safe baseline:

1. Set `NCCL_P2P_DISABLE=1` (disables GPU-to-GPU direct communication — on Windows this forces the PCIe ring / shared-memory path)
2. Set `VLLM_NCCL_SO_PATH` to a known-good `nccl.dll` from the nccl-windows project
3. Use `--disable-custom-all-reduce` flag

#### ⚠ Gotcha: do NOT set `NCCL_SOCKET_IFNAME=lo` or `GLOO_SOCKET_IFNAME=lo` on Windows

`lo` is the **Linux** loopback interface name. The nccl-windows build does not recognise
`lo`, and pinning it produces a hard **`NCCL error: invalid usage`** at startup (observed on
a 2×3090, TP=2 boot: weights load, then NCCL init aborts). Leave `NCCL_SOCKET_IFNAME` /
`GLOO_SOCKET_IFNAME` **UNSET** — with `NCCL_P2P_DISABLE=1`, NCCL auto-selects shared memory
for intra-node tensor parallelism, which is correct. This is the single most common cause
of a multi-GPU boot that "loads the weights and then dies on NCCL init".

#### Loopback rendezvous (WiFi-independent multi-GPU)

To keep the multiprocess rendezvous off the network card entirely (handy if you don't want
vLLM touching WiFi/Ethernet), pin it to loopback:

```powershell
$env:VLLM_HOST_IP = "127.0.0.1"
```

This binds the c10d/gloo rendezvous to `127.0.0.1`, which is always up. A benign one-time
hostname probe may still print in the log — `client socket has failed to connect to
[HOSTNAME]:<port> (system error: 10049)` — and then immediately connect via `127.0.0.1`.
That line is harmless; the `invalid usage` error above is not.

### Stopping vLLM without killing other python

The stop script (`$VLLM_HOME\scripts\stop-vllm.ps1`) and the launcher menu's "Stop all" are
**scoped**: they kill only (1) the process tree that holds the `:8010` listen socket, and
(2) any `python.exe` whose path is under your vLLM build venv (e.g. `$VLLM_HOME\venv`). They do **NOT** run
`Get-Process python | Stop-Process -Force`. A blanket python kill would also terminate any
other python tools on the box (IDEs, tooling, terminals) and drop your session.
If you write your own stop logic, match on the listen socket / venv path — never blanket-kill
python.

### "MSVC compiler not found"

You need the MSVC Build Tools (Step 3). Install them, then:

```bash
pip uninstall vllm -y
pip install vllm
```

### "vllm serve starts but never responds"

This is normal on first run — vLLM compiles triton kernels which can take 5–15 minutes. Wait patiently. You can monitor progress in the terminal output.

### "ImportError: DLL load failed"

Usually a CUDA/driver mismatch. Check:

```bash
nvidia-smi
```

Your driver should support your CUDA version. If the driver is too old, update it from [NVIDIA's driver page](https://www.nvidia.com/Download/index.aspx).

### "pip install vllm fails to compile"

Common causes:
- MSVC Build Tools not installed (Step 3)
- Python version mismatch (vLLM requires 3.10–3.12, not 3.13+)
- Missing CUDA toolkit (nvcc not on PATH)
- Insufficient disk space for build artifacts

Try installing a pre-built wheel first (faster, no compilation):

```bash
pip install --no-build-isolation vllm
```

If that fails, ensure your environment is clean:

```bash
pip uninstall vllm -y
pip cache purge
pip install vllm
```

---

## What's Next

- **Model management:** Use `huggingface-cli` to download and manage models
- **Quantization:** Use AWQ, GPTQ, or AutoRound for smaller model sizes (4-bit models use ~60% less VRAM)
- **Speculative decoding:** Enable with `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` for faster inference
- **Tool use:** Enable with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` for function calling
- **Prefix caching:** Enable with `--enable-prefix-caching` for repeated prompts (saves compute on shared context)
- **Chunked prefill:** Enable with `--enable-chunked-prefill` for better throughput with long prompts

---

## Credits & Acknowledgements

This guide was produced by **BlackBox_Labs** based on real-world deployment experience running vLLM natively on Windows with dual RTX 3090 GPUs.

### Projects referenced and credited

| Project | What it provides | Link |
|---------|-----------------|------|
| **vLLM** | The core inference engine — PagedAttention, continuous batching, high-throughput serving | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **vllm-windows** (SystemPanic) | Pre-built Windows wheels and the `vllm-for-windows` branch with Windows-specific patches, CUDA 13+ support, and build instructions | [SystemPanic/vllm-windows](https://github.com/SystemPanic/vllm-windows) |
| **nccl-windows** (SystemPanic) | Windows-compiled NCCL library enabling tensor and pipeline parallelism across multiple GPUs on Windows | [SystemPanic/nccl-windows](https://github.com/SystemPanic/nccl-windows) |
| **PyTorch** (PyTorch team / NVIDIA) | GPU-accelerated tensor library with CUDA support | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **FlashInfer** | Optimized attention kernels for vLLM | [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |
| **Triton** | NVIDIA's domain-specific language for writing GPU kernels | [triton-lang/triton](https://github.com/triton-lang/triton) |

### Special thanks

- **SystemPanic** for maintaining the vllm-windows and nccl-windows projects that make native Windows vLLM possible
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
| PyTorch | 2.11.0+cu12x |
| vLLM | 0.26.0+ |
| CUDA Toolkit | 12.6–13.3 |
| cuDNN | 9.x |
| MSVC | VS 2022 Build Tools (v143) |
| NCCL | SystemPanic/nccl-windows (Windows builds) |
