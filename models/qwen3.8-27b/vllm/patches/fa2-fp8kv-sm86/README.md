# FA2 FP8 KV on SM86

Experimental sidecar for `vllm/qwen38-27b-dual-hypermax` on vLLM 0.29.0.
The target and DFlash2 drafter retain FP8 E4M3 KV storage. Attention computes
in BF16; Ampere has no native FP8 arithmetic. GDN and vision are unchanged.

The stock `FLASHINFER` registry entry supplies the KV-cache update/layout
contract. A small subclass replaces attention execution and metadata planning.
No FlashInfer attention kernel runs on this path. Native FA2 prefill unpacks
one bounded KV block at a time and merges partial results in FP32. If its
workspace allocation fails, paged FA2 handles the same request.

The installer checks the vLLM version and stock backend SHA256 before adding
one import at the end of that module. It downloads SHA256-checked source
archives at the revisions in `install.py`, builds both extensions in the stock
image, and caches them under the model's engine cache. The build key includes
the source revisions, PyTorch, CUDA, FlashInfer and vLLM versions. A manifest
checks cached libraries and Python helpers before reuse. First boot needs
network access and a CUDA compiler; later boots reuse the verified cache.

Sources and upstream notices remain in the cache. Kernel licensing and source
provenance are in the downloaded project's `LICENSE` and `NOTICE`; CUTLASS
retains its own license. The local vLLM adapter is Apache-2.0, derived from the
stock vLLM metadata/attention contract. See the FA2 row in
[`docs/UPSTREAM.md`](../../../../../docs/UPSTREAM.md) for the dependency pin.

Only SM86, LBNHC (NHD) layout, BF16 queries and E4M3 KV are supported. The
backend advertises one layout and uses it for both target and draft. This
avoids reading the drafter's copied CacheConfig before the target's layout RPC
has reached it in vLLM 0.29.0. The tested head
geometries are `(head_size, local_kv_heads)` = `(256, 1)`, `(256, 2)` and
`(128, 4)`. DCP, attention sinks and other geometries are rejected. The compose
targets TP=2 and one sequence. Fixed split counts preserve CUDA Graph replay.

Launch with `bash scripts/switch.sh --force vllm/qwen38-27b-dual-hypermax`.
The compose filename remains `dflash2-fp8-fa2.yml`. Set `MODEL_DIR` to a directory
containing `qwen3.8-27b-fp8` and `qwen3.8-27b-dflash2-w4a16`.
Use `SPEC_N=0` or `SPEC=off` to disable speculative decoding. Set
`NCCL_P2P_DISABLE=0` only on a host with proven peer access.

The 262K context, one 4 MP image and fixed KV allocation leave little memory
headroom on a 24 GB card. This is an experimental profile, not a production
recommendation. Results from vLLM 0.27.1 are historical and must not be labeled
as measurements of this 0.29.0 sidecar.
