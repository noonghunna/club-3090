# Qwen3.8-Flash-Next — Changelog

Dated history for Qwen3.8-Flash-Next configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-10-18 — first slug on stock vLLM: `vllm/qwen38-flash-next-dual` (🧪)

Flash-Next on our own `vllm-stable` pin (`vllm/vllm-openai:v0.31.0`), no custom image, for
**2× CMP 170HX 64 GB** (sm_80, the A100's chip). Port 8196. Findings: discussion #1735;
PR #1736.

Layout, chosen by measurement on one rig (Gen2 x8 per card, no P2P, 175 W cap): pipeline
parallel over the two cards with a 26/22 layer split (TP saturates the link: 45/63 vs
117/163 tok/s narr/code), the 51.2 GB FP8 PLE table memory-mapped from NVMe instead of
pinned in RAM (host RAM ~19 GB instead of ~83 GB, decode within 3%), FP8 e4m3 KV (pool
1.99M tokens), MTP n=3 through albucino's compact draft with an INT4 copy of `lm_head`
as the draft head, `--prefix-match-unit 16` and the shared-prefix checkpoint.

Seven new overlays under `models/qwen3.8-flash-next/vllm/patches/` (rows in
`scripts/lib/profiles/patches.yml`, upstream bugs in `docs/UPSTREAM.md`):

- `qwen4exp-pp-ple-rank0` and `qwen4exp-mtp-pp`: stock refuses PP for PLE models and the
  MTP drafter asserts under PP. Ports of Minachist's patches (Apache-2.0).
- `qwen4exp-ple-mmap`: Minachist's mmap PLE, ported to v0.31.0; refuses other versions.
- `qwen4exp-ple-fp8-bytecopy` and `qwen4exp-qsa-fp8kv-ampere`: two `fp8e4nv` compile
  failures on Ampere (pinned-RAM PLE copy, QSA attention reading FP8 KV). The e4m3
  decode is halt95's (Merlin).
- `qwen4exp-mtp-draft-head`: the drafter's own W4A16 `lm_head`, built on first boot
  (port of AlexBrandes/vllm 14457f89e; quantizer MIT). c=8 code 668 -> 754, json
  745 -> 809 tok/s.
- `w4a8-inc-moe`: four vLLM bugs that make W4A8 answer `!!!!` on AutoRound MoE models.
  Opt-in `W4A8=1`; GSM8K 386/400 vs 388/400, decode mixed, so it stays off.

Vision is on, `GPU_MEMORY_UTILIZATION` is 0.95 (pool 2.23M tokens, ~2.7 GB left per
card), and the slug shares the Qwen3.8-27B serving contract: reasoning budget by effort
(`vllm-effort-budget` + `vllm-default-thinking-budget`: low 4,096 / medium 16,384 / xhigh
32,768) and the model card's sampler row chosen by `ENABLE_THINKING`.

Measured 2026-10-18 at 175 W. `report.sh --full` PASS: verify-full incl. vision 4/4,
verify-stress ladder to 240,657 tokens, soak-continuous 0 MiB growth, bench-agentic.
`bench.sh`: 126.1 / 174.0 narr/code, prefill 5,159 @10K / 7,557 @90K, peak 62.8 / 62.2 GB.
sparkDash protocol (greedy, thinking off, 400 forced tokens), aggregate decode
c=1/c=8: code 186/695, prose 163/620, json 192/840 tok/s.

Also: the `vllm-stable` engine lists the `qwen4exp` family; compat allows `fp8_e4m3` KV
for `qwen4exp` on sm_80+ (the overlay above); a `2x-cmp-170hx-64gb` canonical scenario;
`diagnose-profile` picks a card of a slug's `supported_sm`.

Known gaps: `KV_OFFLOAD_GB` does not start under PP=2 (vLLM OffloadingConnector bug);
measured on one rig only.

## 2026-10-18 — `vllm/qwen38-flash-next-dual` moves to vLLM v0.32.0

The slug follows `vllm-stable` to the stock `vllm/vllm-openai:v0.32.0` (#1740).

Overlays:

- `qwen4exp-ple-mmap`: re-ported to v0.32.0 (`ple-mmap-v0.32.0.diff`). Upstream's PLE
  prefetch interface changed (`start_prefetch(ngram_ids)`, `finalize_prefetch(total_tokens)`,
  a `reduce_results` argument).
- `qwen4exp-qsa-fp8kv-ampere`: also turns off v0.32.0's fused QSA prepare (vllm#57097) for
  FP8 KV below sm_89. That launch writes the cache through an `fp8e4nv` cast, which Triton
  can't compile on Ampere; the write goes back to `reshape_and_cache_flash`.
- `qwen4exp-ple-fp8-bytecopy`: retired. vllm#60142, in v0.32.0, is the same fix.
- The rest apply unchanged.

Measured 2026-10-18 at 175 W, with vLLM's v0.32.0 release wheel installed into the v0.31.0
image (the v0.32.0 Docker image was not published yet; same torch, CUDA and Python).
`report.sh --full` PASS: verify-full incl. vision 4/4, verify-stress ladder to 240,659
tokens, soak-continuous 0 MiB growth, bench-agentic. `bench.sh`: 126.3 / 171.6 narr/code,
prefill 5,171 @10K / 7,546 @90K, the same as v0.31.0 within noise; decode at 90K depth 148
vs 129 tok/s (one sample each). KV pool 2.23M tokens, as on v0.31.0 (the first boot on a
cold kernel cache got 2.06M: 1.4 GiB more non-torch memory on stage 0).

Quality (`quality-test.sh --full`, benchlocal-cli v0.11.0, on v0.31.0, thinking budget
65,536 at every effort), pass@1 / pass@3 of 150: instruct 123 / 131, low 122 / 135, medium
128 / 139, xhigh 130 / 141. Analysis and raw cards: #1737.

## 2026-10-18 — `vllm/qwen38-flash-next-dual`: one layout, gated to 64 GB cards (PR #1736 review)

- Renamed from `vllm/qwen38-flash-next-dual-cmp170hx`; the compose is now
  `vllm/compose/dual/albucino-autoround-int4-fp8ple/mtp.yml`, container
  `vllm-qwen38-flash-next-dual`. Engine `vllm-stable`, which lists the `qwen4exp` family.
- Ships PP=2 only, with the measured defaults. The `PARALLEL`, `EP`, `COMPILE` and
  `CUDAGRAPH_SIZES` knobs are gone; the compile mode follows `PLE_PLACEMENT` (nvme lean,
  ram stock), as it did by default.
- The entrypoint reads both cards with `nvidia-smi` and refuses to start below 61,440 MiB
  per card, or with fewer than two cards. The message names
  `strata/qwen3.8-flash-next-iq3s-dual` and `bucko-vllm/qwen3.8-flash-next-ple-dual` for
  3090s. Header: `Requires-min-vram-gb: 64`, `Requires-min-gpu-count: 2`,
  `Tensor-parallel: 2` (the card count preflight looks for; the layout stays PP=2).
- Knob names as on the other Qwen3.8 vLLM composes, defaults unchanged: `MAMBA_SSM_DTYPE`
  is now `MAMBA_SSM_CACHE_DTYPE`; new `SERVED_NAME`, `DRAFT_SAMPLE_METHOD` (default greedy,
  vLLM's default and what everything here was measured with), `ASYNC_SCHED`, `OFFLINE`
  (default 1, as the fixed `HF_HUB_OFFLINE=1` was), `PROMPT_TOKENS_DETAILS_ARG`,
  `MAMBA_CACHE_MODE` (default align) and `MAMBA_BLOCK_SIZE`.
- `--disable-custom-all-reduce` dropped: PP=2 runs TP=1 per stage, so there is no all-reduce.
- Booted on the published `vllm/vllm-openai:v0.32.0` image (`sha256:befccf0e…`) at stock
  clocks, 175 W: verify-full PASS incl. vision 4/4; `bench.sh` 122.9 / 170.9 narr/code,
  prefill 5,147 @10K / 7,610 @90K, peak 63.2 / 62.4 GB, KV pool 2,232,041 tokens. Within
  noise of the earlier wheel-in-v0.31.0-image run.
- `docs/UPSTREAM.md`: the overlay rows now name the upstream PRs that cover them (vllm#60234,
  #58815, #49039 / #61405 / #48926 / #59895, #50653), and a row for vllm#61281, the prefix-cache
  miss under concurrent prefills that `--prefix-match-unit 16` makes likely (read from the
  source, not reproduced).

## 2026-10-31 — RoPE-extension launch settings on every Strata slug (`ROPE_SCALING` / `ROPE_SCALE`)

- New launcher knobs `ROPE_SCALING` (`none`/`linear`/`yarn`) and `ROPE_SCALE` (factor ≥ 1),
  written into the run config as the engine args `--rope-scaling` / `--rope-scale` at every
  boot, forwarded by all five Strata composes and catalogued in `launch-knobs.json` — so
  `switch.sh <slug> --set ROPE_SCALING=yarn` (or a one-off `ROPE_SCALING=yarn CONTEXT=524288
  bash scripts/switch.sh <slug>`) extends the existing slug instead of a dedicated one.
  The pinned engine (0.1.54, image 0.2.8) and setup already had YaRN; this is plumbing.
- Past the trained 262144 setup resolves yarn itself (factor ctx/262144); the knobs override
  that at every boot. `none` past 262144 is refused; `linear`/`yarn` with no factor derive
  the same value as setup. The launcher strips the flags it wrote — including ones setup
  folded into its own record — and leaves flags setup wrote itself alone, so unscale-back
  past 262144 is refused the same way setup refuses it.
- Raising `CONTEXT` re-runs setup at the next boot (it sizes the KV and expert cache); the
  run-config folder is shared, so one slug walks 131072 → 524288 → back without a second
  slug or a re-download — the packs and MTP layer stay in `Strata-data`.
- YaRN's mscale applies at every position: inside 262144 the run is a slightly different
  model from the stock one (upstream's greedy comparison diverges after ~19 tokens). An
  ordinary-RoPE KV session can't be restored into a yarn engine; replay the conversation.
  Upstream with this scheme: 8/8 needles at 512K
  (Strata's `bench/results/2026-09-28-rope-scaling`), 1M 3-fact 3/3 YaRN vs 0/3 ordinary
  (Strata's `bench/results/2026-10-09-rope-yarn-long-context`).
- RAM at a raised CONTEXT: the pinned KV pool IS allocated at start (engine default: the
  whole `--max-context`; only `--kv-grow` defers it, single-GPU only) and grows ~13.7
  KB/token (int8): +5.4 GB at 524288 over 131K, +12.7 GB at 1048576 — against 22-24 GB
  measured headroom while the 131K slug runs (2026-10-30). `KV=q4_0` (half the pool,
  upstream-benched) is the fallback at 1M.
- `test-strata-config-knobs.sh` covers the apply/strip/refusal paths; `test-launch-knobs.sh`
  proves the catalogue domains through `strata_config.py validate` (program checks) and the
  delivery through the compose scan. Bench gates — 8-pack, a >262144 needle probe on this
  rig, a within-262144 quality A/B against the unscaled boot, MTP acceptance under yarn —
  pending.
