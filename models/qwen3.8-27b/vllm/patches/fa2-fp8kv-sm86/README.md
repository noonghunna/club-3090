# FlashAttention with FP8 KV

Experimental plugin for `vllm/qwen38-27b-dual-ultramax` on vLLM 0.29.0.
The target and DFlash2 drafter retain FP8 E4M3 KV storage. Attention computes
in BF16; Ampere has no native FP8 arithmetic. GDN and vision are unchanged.

The plugin registers `FLASH_ATTN` through vLLM's public backend API for both
target and draft. It does not edit or subclass the FlashInfer backend.
Native FA2 prefill unpacks
one bounded KV block at a time and merges partial results in FP32. If its
workspace allocation fails, paged FA2 handles the same request.

The init service extracts two compiled libraries and a plugin wheel from a
pinned artifact image into a named volume. The serving container mounts it
read-only. `install_artifact.py` checks the artifact identity, payload SHA256,
PyTorch/CUDA/Python/C++ ABI, compiled SM and vLLM metadata/layout interface
before installing the wheel offline. Installing FA2 needs no source download
or CUDA C++ compiler. vLLM still performs its normal torch.compile/Triton
warmup. The artifact manifest records the source revision and build pins.

The image includes `LICENSE`, `NOTICE` and the upstream licenses. The source
repository contains an upstream FlashAttention submodule, a file map and a
patch that reproduces the exact diff against upstream FA2. See the FA2 row in
[`docs/UPSTREAM.md`](../../../../../docs/UPSTREAM.md) for the dependency pin.

The mixed-prefill update passes all nine CUDA regressions and the full
upstream GPU suite on each RTX 3090 with vLLM 0.29. The adapter checks also
pass on vLLM 0.30, and the actual FP8 compose passes verify-full with
SPEC_N=0, 98304 context and 4 GB KV/card. The update processes long full-causal
mixed batches as per-request paged chunks of at most 2048 query tokens,
using exact GPU context lengths. `envelope.sh` remains enabled for conservative
rollout and users overriding the artifact with an older version. The model
smoke does not revalidate the shipped 262144-context DFlash2 configuration.
See REVIEW_VALIDATION.md and the FA2 row in the upstream tracker above.

SM89 and SM120 binaries are included, but those
targets have compilation coverage only. SM90 and SM100 retain stock native
FlashAttention without installing the plugin; no new GPU measurements cover
that route. Other SMs and mixed architectures are refused before weights load.
Registry `required_sm` remains a lower bound; `supported_sm` provides the
explicit set for launcher gates.

The extension uses LBNHC (NHD) layout, BF16 queries and E4M3 KV. The
backend advertises one layout and uses it for both target and draft. This
avoids reading the drafter's copied CacheConfig before the target's layout RPC
has reached it in vLLM 0.29.0. The tested head
geometries are `(head_size, local_kv_heads)` = `(256, 1)`, `(256, 2)` and
`(128, 4)`. DCP, attention sinks and other geometries are rejected. The compose
targets TP=2 and one sequence. Fixed split counts preserve CUDA Graph replay.

Launch with `bash scripts/switch.sh --force vllm/qwen38-27b-dual-ultramax`.
The compose filename is `dflash2.yml`. Set `MODEL_DIR` to a directory
containing `qwen3.8-27b-fp8` and `qwen3.8-27b-dflash2-w4a16`.
Use `SPEC_N=0` or `SPEC=off` to disable speculative decoding. Set
`NCCL_P2P_DISABLE=0` only on a host with proven peer access.

The 262144-token limit comes from fixed KV reservation
(`KV_CACHE_MEMORY_BYTES=6335076762` per card), batch size 2048 and scheduler
settings. `gpu_memory_utilization` does not constrain this fixed reservation.
`long_prefill_token_threshold=0` disables the separate long-prefill threshold;
chunked prefill still obeys the batch budget. These settings affect memory and
scheduling, and do not establish a kernel-specific context-capacity gain.
The profile allows one sequence and one image up to about 4 MP.

The adapter for vLLM 0.27.1 and the plugin for vLLM 0.29 use different
integration methods. Measurements must identify which one ran.

The API boundary probes are included beside the adapter. Run from the repo
root after the model is ready; each full-context request takes several minutes:

```bash
python3 models/qwen3.8-27b/vllm/patches/fa2-fp8kv-sm86/check_context.py --url http://127.0.0.1:8110
python3 models/qwen3.8-27b/vllm/patches/fa2-fp8kv-sm86/check_vision_context.py --url http://127.0.0.1:8110
```

The first sends exactly 261000 input tokens and rejects a 263000-token request.
The second uses the repository's vision fixture, exercises about 4 MP of image
processing, fills about 260K combined tokens, and checks two follow-up turns.
`VALIDATION.md` records the September 12 adapter run. `REVIEW_VALIDATION.md`
records the native plugin comparison and artifact checks.
