# GLM-5.3-Flash — Changelog

Dated history for GLM-5.3-Flash configs in this repo. Append-only — add a new entry, don't rewrite past ones.

## 2026-10-07 — dual UD-IQ3_XXS: recommended `UBATCH=1024 MOE_ADMIT_AFTER=32` override for 32 GB+ cards (measured on 2× 5090; defaults unchanged; 2× 3090 untested)

`llamacpp-club3090/glm53-flash-dual-iq3xxs-moecache` keeps its shipped defaults
(`UBATCH=2048`, `MOE_ADMIT_AFTER=64`) and adds a documented override recommendation for
32 GB+ cards (e.g. RTX 5090), measured on 2× RTX 5090 (TR PRO 3975WX, 8× DDR4-3200, PCIe 4),
CTX 204,800, DFlash2 n=2, with alternated boots:

- **`UBATCH=1024`** (default stays 2048): short-prompt decode code 33.55 ± 0.84 vs 29.59 ± 0.28
  (+13%, 4 vs 2 boots), narrative +13%; prefill −25 to −31% (~32K 526 → 364, ~190K 385 → 290
  tok/s). The same mechanism as the 4096 → 2048 step: GLM's kpool indexer masks scale with
  ubatch, and the freed compute buffer goes to the expert pool (marginal hit ~63% → ~68%).
- **`MOE_ADMIT_AFTER=32`** (default stays 64, inherited from DeepSeek and never measured on
  GLM): ms per verify cycle at ~32K occupied was lower at 32 than at 64 — two drafts of this
  entry recorded 64.3 vs 72.2 and 64.3 vs 73.9 and which is correct is unknown (raw per-boot
  logs were not kept); re-measure before quoting a percentage. ~190K −9% (inside noise); short
  prompts unchanged. 16 was a wash overall in a one-boot screen, and 128 was worse at depth.

With both overrides, through `switch.sh` on this rig: narrative / code **34.49 / 36.95** tok/s
(3 boots), `verify-stress` PASS (ceiling 188,019 tok, 91%; boundary 5/5). See BENCHMARKS.

⛔ **Not yet run on 2× 3090.** The 3090 pool is ~4× smaller at 205K, so `UBATCH=1024` should
help at least as much there, and `MOE_ADMIT_AFTER=32` could either help or churn. The defaults
stay `UBATCH=2048 MOE_ADMIT_AFTER=64` until a 3090 run is on record; if it shows the same gain,
flip the defaults then. Whether `UBATCH=1024` also avoids ggml-org/llama.cpp#28282 (the
long-prefill crash at 2048) is untested.
