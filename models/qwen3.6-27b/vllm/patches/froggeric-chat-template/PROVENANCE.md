# froggeric chat template provenance

Source: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

## Current vendored snapshot

- Date vendored for re-eval: 2026-05-17
- Upstream revision: `c31fd393e531dbacd92b6deb99a2037cc949f950`
- Upstream timestamp: 2026-05-16T13:44:07Z
- Upstream release label: v19
- Local file SHA256: `4649b3fa3db3fda4d51173ed4ff0175fde7ece8bbceb9d595d04d862020c9746`
- Status: **ADOPTED 2026-05-18 (#150)** — re-eval PASSED on `vllm/dual` (template-only A/B: hermesagent-20 +10pp 50→60%, 7 packs flat, TPS-neutral). See docs/UPSTREAM.md froggeric row (authoritative).

## Previous vendored snapshot

- Local introducing commit: `84498d47aaf7a2fdb7c0203d53bb64414a64b6c1`
- Local file SHA256: `94e944287ffaf8c3ed8b5840a0c92fd4ca3caefa721f4f5e31e92605e63f1ad4`
- Upstream revision: unrecoverable. The introducing commit cited the HF repo
  but no commit/release pin, and the previous local file did not exactly match
  any current upstream qwen3.5/qwen3.6 archived template v8-v19 or main commit
  in the HF repo history available on 2026-05-17.

## Rejected candidate — v21.3 (2026-07-11)

- Upstream revision: `23a40b0bd4d197c31d39e3c442fd2cd6100b3971` (2026-07-03, release label v21.3)
- Candidate file SHA256: `d203f3342d8a7f8474dd55563eece3a26e71b21c6f667c9db9c93b762b3bf997`
- Status: **REJECTED** — failed the adoption gate on `vllm/dual` (fp8-mtp, v0.24.0).
- Gate results: TPS-neutral ✓ (74.68 decode vs banked v19 73.98, same-session); toolcall-15 15/15 ✓;
  **hermesagent-20 6/20 (30%) ✗ with p95 pinned at the 300s pack timeout** — scenarios
  STALLING in multi-turn tool loops, not failing on content. Total 106/150 vs v19's 109.
- The stall signature is mechanical (suspects: v21's `preserve_thinking` default flip,
  payload-truncation defaults, or the reworked tool-error scoping) and hits exactly the
  pack the v19 adoption was won on (#150: hermes +10pp). v21's prefix-cache/streaming
  fixes do not outweigh a 30% hermes pack.
- Next re-vendor attempt should start by isolating the hermes stall (hermes-only run
  with `--sandbox-log-dir`) before re-running the full gate. Candidate + A/B artifacts
  were session-local (not vendored); re-fetch by revision above.
