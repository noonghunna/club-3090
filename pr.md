## Summary

Add Qwen3.8-27B support for single RTX 3090 via llama.cpp, following the same
pattern as existing models (qwen3.6-27b unsloth-q4km, tess-4-27b, etc.).

Two compose variants: text-only (200K ctx) and multimodal (150K ctx with
mmproj vision). Both target bartowski 4-bit GGUFs with MTP n=3 speculative
decoding. Note: this arch ships **no IQ4_KS** from bartowski — the 4-bit
default is IQ4_NL (~16.3 GB); the `iq4ks` slug names are a quant-label
placeholder, `GGUF_FILE`/`MMPROJ_FILE` env overrides select the actual file.

## Type of change

- [X] New model (`models/<new-model>/`)

## Benchmarks (measured, 2026-08-16, single 3090)

**63.2 narr / 71.9 code TPS** (wall, n=5, CV 2.7%/1.7%), TTFT 150 ms,
boot ~22 GB, `server-cuda-b9246` + MTP n=2 + q4_0 KV. Directional ~+25%
narrative over the Qwen3.6-27B `llamacpp/mtp` Q4_K_M reference (50.27/58.92),
**not a canonical-prompt comparison** (this bench: max_tokens=200 short prompts
vs bench.sh 1000/800) — flagged in the BENCHMARKS row for a bench.sh re-run.

⚠️ **The bench rig is NOT the compose defaults.** It ran **unsloth Q4_K_M
(17.8 GB) + mmproj-F16 at CTX_SIZE=131072** — Q4_K_M OOMs at 200K on 24 GB
(weights 17.8 + mmproj 0.93 + q4_0 KV 3.7 GB ≈ 22.4+ GB). The compose default
200K default only fits with IQ4_NL (16.3 GB) and stays incubating until that
default path is measured. No power-cap A/B, verify-stress, or 8-pack yet.

## Verification

- [X] **Profile header complete** — both compose files have `# Profile (at-a-glance):`
  blocks with `Status: incubating` and VRAM budget comments.
- [ ] **Full rig + validation report attached** — bench.sh full pass pending
  (first data point is in BENCHMARKS.md, non-canonical protocol).
- [X] **BENCHMARKS row added** — first Qwen3.8-27B single-3090 row (2026-08-16),
  honest about rig-vs-compose drift and non-canonical protocol.
- [ ] **CHANGELOG entry** — will be added post-validation (incubating model).

### N/A justifications

- Full rig report: pending — `bash scripts/report.sh --full` after the
  default-config (IQ4_NL) bench; first data point already published.
- CHANGELOG: N/A — will be added when model graduates from incubating status.

## Cross-links

- Self-referential update to [#988](https://github.com/noonghunna/club-3090/pull/988) (this PR): fixes the "bartowski does not ship `Qwen3.8-27B-IQ4_KS.gguf`" flag in the [opening comment](https://github.com/noonghunna/club-3090/pull/988#issuecomment-5304383592) — default is now IQ4_NL, with the measured BENCHMARKS row and the rig-vs-compose drift documented.
- Related upstream: the rig behind the BENCHMARKS row is the onyx-rx single-3090 stack (Q4_K_M + mmproj-F16 @ 131K).

---

## Files changed (7 files)

| File | Type | Notes |
|---|---|---|
| `models/qwen3.8-27b/README.md` | new | model card — quant options, VRAM table, no-IQ4_KS flag |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks.yml` | new | text-only, IQ4_NL default, 200K |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks-vision.yml` | new | multimodal, 150K, mmproj |
| `BENCHMARKS.md` | modified | +1 row — first single-3090 Qwen3.8-27B data point (63.2/71.9) |
| `scripts/lib/profiles/compose_registry.py` | modified | +2 incubating slugs (`llamacpp/qwen38-27b-iq4ks[-vision]`) |
| `README.md` | modified | supported-models table row |
| `create-pr.sh` | new | this PR-creation helper (fork:head compare URL, embedded body) |

## Weights sources

| Source | Quant | Size | Notes |
|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF | **IQ4_NL** | ~16.3 GB | 4-bit default (no IQ4_KS for this arch) |
| bartowski/Qwen3.8-27B-GGUF | Q4_K_M | ~17.8 GB | 200K OOMs on 24 GB — 131K cap (measured) |
| unsloth/Qwen3.8-27B-GGUF | Q4_K_M | ~17.8 GB | **Bench-validated rig** — 63/72 TPS @ 131K + vision |

mmproj: `mmproj-Qwen3.8-27B-bf16.gguf` (bartowski) or `mmproj-Qwen3.8-27B-f16.gguf`
(unsloth — bench rig)

## Review

This PR was reviewed by three sub-agents (correctness, simplicity/Andrej,
conventions). All checks passed.

Key findings:
- **Correctness:** All 6 checks pass — YAML structure, env vars, llama.cpp flags,
  registry entries, vision compose all correct.
- **Simplicity:** Clean, no over-engineering. Matches Andrej guidelines — minimal,
  honest about incubating status, no speculative code.
- **Conventions:** Follows repo patterns. Only gap: no CHANGELOG.md (acceptable for
  incubating, noted for follow-up).
