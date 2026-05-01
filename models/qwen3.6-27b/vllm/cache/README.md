# vLLM compile/Triton caches (host-mounted)

Per-variant artifacts for `torch.compile` (Inductor) + Triton kernel JIT.
Mounted into containers at `/root/.cache/vllm/torch_compile_cache` and
`/root/.triton/cache`. First boot of a fresh variant warms the cache
(~60-90 sec); subsequent boots reuse cached graphs and skip recompile.

Pattern lifted from Sandermage's `start_27b_int4_TQ_k8v4.sh:31-32`.

Safe to delete (`rm -rf cache/triton/* cache/torch_compile/*`) — only
costs you one slow cold start to regenerate. Variants share the cache
directory but key off their own config hash, so no cross-contamination.
