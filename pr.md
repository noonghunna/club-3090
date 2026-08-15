## Summary

Add Qwen3.8-27B support for single RTX 3090 via llama.cpp, following the same
pattern as existing models (qwen3.6-27b unsloth-q4km, tess-4-27b, etc.).

Two compose variants: text-only (200K ctx) and multimodal (150K ctx with
mmproj-BF16 vision). Both use bartowski IQ4_KS weights (~14.8 GB) with MTP n=3
speculative decoding. VRAM budget: ~21.3 GB text / ~22.7 GB vision on 24 GB.

## Type of change

- [X] New model (`models/<new-model>/`)

## Verification

- [X] **Profile header complete** — both compose files have `# Profile (at-a-glance):`
  blocks with `Status: incubating` and VRAM budget comments.
- [ ] **Full rig + validation report attached** — new model, pending benchmarks.
- [ ] **BENCHMARKS row added** — pending first bench run.
- [ ] **CHANGELOG entry** — will be added post-validation (incubating model).

### N/A justifications

- Full rig report: N/A — new model, unbenchmarked (status = incubating). Will
  add `bash scripts/report.sh --full` output after first validation run.
- BENCHMARKS row: N/A — pending first bench run. Will add once TPS numbers are
  measured.
- CHANGELOG: N/A — will be added when model graduates from incubating status.

## Cross-links

- Closes #
- Related upstream:

---

## Files changed (5 files, +313 lines)

| File | Type | Lines |
|---|---|---|
| `models/qwen3.8-27b/README.md` | new | +93 |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks.yml` | new | +91 |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks-vision.yml` | new | +102 |
| `scripts/lib/profiles/compose_registry.py` | modified | +24 |
| `README.md` | modified | +1 |

## Weights sources

| Source | Quant | Size | Notes |
|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF | **IQ4_KS** | ~14.8 GB | Default, broad quant selection |
| bartowski/Qwen3.8-27B-GGUF | IQ4_NL | ~15.6 GB | Nearly identical quality |
| unsloth/Qwen3.8-27B-GGUF | IQ4_KS | ~15.0 GB | Dynamic V3.0 MTP |

mmproj: `mmproj-Qwen3.8-27B-bf16.gguf` (bartowski) or `mmproj-BF16.gguf` (unsloth)

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
