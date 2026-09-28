# vLLM compile/Triton caches (host-mounted)

> **The launchers no longer use this folder (#1466).** `switch.sh`, `launch.sh`, `gpu-mode`,
> the estate planner and c3 mount these caches from `~/.cache/club-3090/<engine image>/`
> (`CLUB3090_CACHE_DIR` moves it), shared by every checkout and every model that runs the
> same image. This folder is only what a plain `docker compose up` falls back to when
> nothing is set. A cache an older version left here is dead weight:
> `bash scripts/settings.sh caches` shows its size and removes it after asking, and prints
> the `sudo rm -rf` line for folders docker created as root.

Per-variant artifacts for `torch.compile` (Inductor) + Triton kernel JIT.
Mounted into containers at `/root/.cache/vllm/torch_compile_cache` and
`/root/.triton/cache`. First boot of a fresh variant warms the cache
(~3-7 min depending on TP); subsequent boots reuse cached graphs and skip
recompile (~2 min).

Mirrors the pattern in `models/qwen3.6-27b/vllm/cache/`.

Safe to delete (`rm -rf cache/triton/* cache/torch_compile/*`) — only
costs you one slow cold start to regenerate. The two compose variants
(`dual.yml` TP=2 and `single.yml` TP=1) share this directory
but key off their own config hash, so no cross-contamination.
