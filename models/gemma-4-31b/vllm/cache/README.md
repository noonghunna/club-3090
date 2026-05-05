# vLLM compile/Triton caches (host-mounted)

Per-variant artifacts for `torch.compile` (Inductor) + Triton kernel JIT.
Mounted into containers at `/root/.cache/vllm/torch_compile_cache` and
`/root/.triton/cache`. First boot of a fresh variant warms the cache
(~3-7 min depending on TP); subsequent boots reuse cached graphs and skip
recompile (~2 min).

Mirrors the pattern in `models/qwen3.6-27b/vllm/cache/`.

Safe to delete (`rm -rf cache/triton/* cache/torch_compile/*`) — only
costs you one slow cold start to regenerate. The two compose variants
(`gemma-mtp.yml` TP=2 and `gemma-mtp-tp1.yml` TP=1) share this directory
but key off their own config hash, so no cross-contamination.
