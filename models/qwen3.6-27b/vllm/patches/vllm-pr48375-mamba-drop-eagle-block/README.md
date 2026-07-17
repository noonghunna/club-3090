# Vendored vllm#48375 — honor `drop_eagle_block` in MambaManager

**Upstream:** bug [vllm#43559](https://github.com/vllm-project/vllm/issues/43559)
(MTP × prefix-caching corrupts recurrent-state KV on Qwen3-Next hybrids; ~20% acc
drop reproduced upstream on 35B-A3B) · fix [vllm#48375](https://github.com/vllm-project/vllm/pull/48375)
(open) — `MambaManager.find_longest_cache_hit` receives `drop_eagle_block` but
silently ignores it; the fix lowers the hit-search ceiling by one page so the
final page's recurrent-state snapshot (possibly taken over later-rejected draft
tokens) is recomputed instead of reused.

**What ships here:** the PR's 9-line source delta as an idempotent, anchor-checked
install script (`install.sh` → `patch_mamba_drop_eagle_block.py`), mounted at
`/etc/club3090/pr48375/` and invoked from the compose entrypoint before serve.
Anchor drift → the container **refuses to boot** rather than serving unpatched.

**Runtime behavior:** inert under the shipped `--no-enable-prefix-caching`
default (#720) — the patched code path only executes when prefix caching is
enabled. For users who flip `PREFIX_CACHE_ARG=--enable-prefix-caching`, the fix
is active; measured cost is ~5% fewer prefix-cache hit tokens (the by-design
one-block search reduction), correctness-neutral otherwise.

**Validation (2026-07-17, reference 2×3090, v0.25.1):** boots + serves both
hybrid families (27B dense-hybrid, 35B-A3B MoE) with MTP n=3 + prefix-ON;
tool-calls clean 12/12 post-agent probes per arm (probe-validity asserted via
`vllm:prefix_cache_hits_total`); MTP accept-len 3.2–3.9 intact. Full arm matrix
+ the non-repro bounds discussion: club-3090 #710.

**Drop trigger:** vllm#48375 merges upstream AND the `vllm-stable` pin moves past
it → remove the mounts + entrypoint call + this dir; `patches.yml` row retires.
