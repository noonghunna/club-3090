# qwen4exp-ple-mmap

Keeps Qwen3.8-Flash-Next's n-gram PLE table in a file on NVMe instead of pinned host
RAM, on the stock `vllm/vllm-openai` image. Stock vLLM only offers the table on the GPU or
pinned in RAM (`--engram-config cpu_offload`), which costs ~48 GiB of RAM for the FP8 table.

**How it works** (Minachist's design): the table is a raw file, memory-mapped. The rows a
step needs are gathered on the host (decode-01, hoisted out of the CUDA graph by the
model-state hook in decode-02) into a small pinned buffer and copied to the GPU. A
look-ahead thread prefetches the pages a long prompt will touch (ttft-01) and advises the
kernel about upcoming gathers (decode-06). Hot rows live in the page cache, which the
kernel can reclaim; cold rows are read from the drive.

**Use**

| Env | Meaning |
|---|---|
| `VLLM_PLE_MMAP_PATH` | the table file. Missing -> written from the checkpoint on first boot (slow), then reused |
| `VLLM_PLE_MMAP_REBUILD=1` | rewrite the file on the next boot |
| `VLLM_PLE_PREFETCH=0` | turn off ttft-01 and decode-06 |
| `VLLM_PLE_HOST_GATHER=0` | gather inside the model (needs PIECEWISE + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`) |

Constraints: ETP=1, so **PP only, not TP** (pair with `qwen4exp-pp-ple-rank0`). Run with
`--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",...}'` as Minachist
ships it. The file is the table in the checkpoint's storage dtype: 51.2 GB for an FP8 PLE
(albucino), ~102 GB for BF16.

**Provenance.** Port to stock vLLM v0.31.0 of Minachist's `flash-next-vllm.patch` (its
ngram_embedding hunks only), `decode-01-ple-host-gather`, `decode-02-model-state-hook`,
`ttft-01-ple-page-prefetch` and `decode-06-ple-gather-advise` (Apache-2.0),
`huggingface.co/Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, `vllm-patch/`.
Club changes: moved past upstream's common/ refactor (the compressed-tensors PLE hunks are
dropped; the mmap class reuses the common base classes), added the missing imports.
Co-authored-by: Minachist <84688797+aiueonicked@users.noreply.github.com>

**Versions.** One diff per vLLM version (`ple-mmap-v<version>.diff`); the installer refuses
any other version. v0.32.0rc3: 2 hunks fail, needs a re-port.

Status: 🧪 experiment (dual CMP 170HX). Not yet in `scripts/lib/profiles/patches.yml`.
