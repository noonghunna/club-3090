# GLM-5.3-Flash — Changelog

Dated history for GLM-5.3-Flash configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-10-07 — dual UD-IQ3_XXS: `UBATCH` 2048 → 1024, `MOE_ADMIT_AFTER` 64 → 32 (measured on 2× 5090; 2× 3090 untested)

`llamacpp-club3090/glm53-flash-dual-iq3xxs-moecache` changes two defaults. Both were measured
on 2× RTX 5090 (TR PRO 3975WX, 8× DDR4-3200, PCIe 4), CTX 204,800, DFlash2 n=2, with
alternated boots:

- **`UBATCH` 1024** (was 2048): short-prompt decode code 33.55 ± 0.84 vs 29.59 ± 0.28 (+13%,
  4 vs 2 boots), narrative +13%; prefill −25 to −31% (~32K 526 → 364, ~190K 385 → 290 tok/s).
  The same mechanism as the 4096 → 2048 step: GLM's kpool indexer masks scale with ubatch,
  and the freed compute buffer goes to the expert pool (marginal hit ~63% → ~68%).
- **`MOE_ADMIT_AFTER` 32** (was 64, inherited from DeepSeek and never measured on GLM): ms per
  verify cycle at ~32K occupied 64.3 vs 72.2 (−11%, 3 vs 4 boots); ~190K −9% (inside noise);
  short prompts unchanged. 16 was a wash overall in a one-boot screen, and 128 was worse at depth.

With both, through `switch.sh` on this rig: narrative / code **34.49 / 36.95** tok/s (3 boots),
`verify-stress` PASS (ceiling 188,019 tok, 91%; boundary 5/5). See BENCHMARKS.

⛔ **Not yet run on 2× 3090.** The 3090 pool is ~4× smaller at 205K, so ub 1024 should help at
least as much there, and admit 32 could either help or churn. Restore the previous behaviour
with `UBATCH=2048 MOE_ADMIT_AFTER=64`. Whether ub 1024 avoids ggml-org/llama.cpp#28282 (the
long-prefill crash at 2048) is untested.
