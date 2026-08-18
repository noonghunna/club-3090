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

## Benchmarks (canonical `scripts/bench.sh`, 2026-08-17, single 3090 @ verified 370 W)

**58.05 narr / 68.38 code wall TPS** (n=5, CV 2.9% / 1.5%; decode 58.57 / 69.78),
TTFT 150 / 148 ms, PP 1265 tok/s @10K and 911 @90K, VRAM 22440 MiB flat
(idle → peak → post, 0 MiB leak). Mainline llama.cpp + MTP n=2
(`--spec-type draft-mtp`) + q4_0 KV, `-c 131072 -b 1024 -ub 1024`.

**+17% narrative / +19% code over the Qwen3.6-27B `llamacpp/mtp` reference,
compared at matched metric** — wall 49.69/57.50 → 58.05/68.38 (+16.8% / +18.9%),
decode 50.27/58.92 → 58.57/69.78 (+16.5% / +18.4%). Both metrics agree, so the
delta is robust.

**Replicated twice on the same warm engine.** An earlier run in which ~318 MB of
the `llama-server` process was paged to swap produced 57.83 / 68.50 wall — within
≤1.8% on every metric, including PP and TTFT. Swap was then fully disabled
(`swapoff -a`, `VmSwap` 325,956 kB → 36 kB, engine never restarted) and the run
repeated. The paged memory was cold and did not affect throughput.

🐛 **Upstream nit — `bench.sh`'s swap check false-positives.** On the swap-free
run it still printed `swap check FAIL — 36 kB of the SERVING PROCESS is paged out
to swap; every number in this run is suspect` while, three lines earlier, reporting
`swap_total_gib=0.0 swap_used_gib=0.0 server_swap_mib=0`. With no swap device
mounted there is nowhere to page, and 36 kB is 0.0002% of an 18.3 GiB RSS.
Suggest short-circuiting the check when `swap_total == 0`, or thresholding on a
fraction of RSS rather than any nonzero `VmSwap`. Happy to open a separate PR.

⚠️ **The bench rig is NOT the compose defaults.** It ran **unsloth Q4_K_M
(17.8 GB) + mmproj-F16 at CTX_SIZE=131072** — Q4_K_M OOMs at 200K on 24 GB
(weights 17.8 + mmproj 0.93 + q4_0 KV 3.7 GB ≈ 22.4+ GB). The compose default
200K only fits with IQ4_NL (16.3 GB). Both default files are now downloaded and
byte-verified, but that path has **never been booted**, so `iq4ks.yml` stays
incubating until it is measured.

## Verification

- [X] **Profile header complete** — both compose files have `# Profile (at-a-glance):`
  blocks with `Status: incubating` and VRAM budget comments.
- [X] **Full rig + validation report attached** — `scripts/report.sh --full`,
  all five stages exit 0. Report pasted as a PR comment.
- [X] **`verify-full.sh` PASS** — 7 checks passed, 2 skipped (Genesis and the
  MTP-acceptance log scrape are vLLM-only; see N/A below).
- [X] **`verify-stress.sh` 8/8 PASS** — Cliff-2 needles recalled at 58,569 and
  91,070 tokens; ceiling ladder fillable to **120,320 tok (91% of n_ctx=131072)**
  with 1684 MB margin and 0 MB VRAM drift across the ladder.
- [X] **`SOAK_MODE=continuous` 5×5 PASS** — 0 MiB growth against the 200 MiB
  threshold, 0 errors, 0/25 silent-empty, 100% TPS retention, p50 decode 94.2.
  No Cliff 2b signature.
- [X] **`bench.sh` run** — 3 warmups + 5 measured, narrative and code, plus the
  prefill-10K/90K probes for prompt throughput. See swap caveat above.
- [X] **BENCHMARKS row added** — canonical row dated 2026-08-17, superseding the
  non-canonical 2026-08-16 entry and disclosing the swap caveat.
- [ ] **CHANGELOG entry** — not yet added; see N/A below.

### N/A justifications

- **Genesis patch check + MTP acceptance length**: N/A — llama.cpp engine.
  Genesis is vLLM-only, and the MTP-acceptance check parses vLLM log format.
  Both were skipped cleanly by `verify-full.sh` rather than failing.
- **Draft/MTP acceptance rate**: not reported — requires `SERVER_LOG=<path>` or a
  container name; this run used `CONTAINER=none` against a host endpoint.
- **Quality 8-pack**: N/A for this run — `benchlocal-cli` is not installed on the
  rig, so `quality-test.sh` could not run. Not a blocker for an incubating model.
- **Power-cap A/B**: not run. The single 370 W point is measured and verified;
  a sweep needs `sudo power-cap-sweep.sh` and is left for a follow-up.
- **CHANGELOG**: deferred until the model graduates from incubating, at which
  point the IQ4_NL default path will also have been measured.

## Cross-links

- Self-referential update to [#988](https://github.com/noonghunna/club-3090/pull/988) (this PR): fixes the "bartowski does not ship `Qwen3.8-27B-IQ4_KS.gguf`" flag in the [opening comment](https://github.com/noonghunna/club-3090/pull/988#issuecomment-5304383592) — default is now IQ4_NL, with the canonical BENCHMARKS row and the rig-vs-compose drift documented.
- Related upstream: the rig behind the BENCHMARKS row is the onyx-rx single-3090 stack (Q4_K_M + mmproj-F16 @ 131K).

---

## Files changed (8 files)

| File | Type | Notes |
|---|---|---|
| `models/qwen3.8-27b/README.md` | new | model card — quant options, VRAM table, no-IQ4_KS flag, what's working / not working |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks.yml` | new | text-only, IQ4_NL default, 200K |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks-vision.yml` | new | multimodal, 150K, mmproj |
| `scripts/lib/profiles/models/qwen3.8-27b.yml` | new | model profile — required for the registry entries to resolve (see Review) |
| `BENCHMARKS.md` | modified | +1 canonical row (58.05/68.38 @ 370 W), supersedes the 2026-08-16 entry |
| `scripts/lib/profiles/compose_registry.py` | modified | +2 incubating slugs (`llamacpp/qwen38-27b-iq4ks[-vision]`) |
| `README.md` | modified | supported-models table row |
| `create-pr.sh` | new | this PR-creation helper (fork:head compare URL, embedded body) |

## Weights sources

| Source | Quant | Size | Notes |
|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF | **IQ4_NL** | 16,325,830,240 B (~16.3 GB) | 4-bit default (no IQ4_KS for this arch); downloaded, **unbooted** |
| bartowski/Qwen3.8-27B-GGUF | Q4_K_M | 17,772,537,440 B (~17.8 GB) | 200K OOMs on 24 GB — 131K cap (measured) |
| unsloth/Qwen3.8-27B-GGUF | Q4_K_M | 17,772,537,440 B (~17.8 GB) | **Bench-validated rig** — 58.05/68.38 wall TPS @ 131K + vision, 370 W |

mmproj (both from **bartowski**): `mmproj-Qwen3.8-27B-bf16.gguf` (931,145,952 B —
compose default, unbenched) or `mmproj-Qwen3.8-27B-f16.gguf` (927,607,008 B — the
one the bench rig actually mounted). Note unsloth ships *differently named and
sized* projectors in its own repo (`mmproj-F16.gguf` = 927,607,488 B,
`mmproj-BF16.gguf` = 931,146,432 B), so the two are not interchangeable.

## Review

An earlier three-sub-agent review of this PR reported "all checks passed". A
follow-up pass found that conclusion was wrong, and the following were fixed:

- **Registry entries could not resolve (blocker).** Both new slugs declare
  `model="qwen3.8-27b"`, but no `scripts/lib/profiles/models/qwen3.8-27b.yml`
  existed — the only model of 18 in the registry missing one. This crashed
  `estate_cli.py validate` and the kv-calc calibration with
  `CrossReferenceError` during the report run. Fixed by adding the profile;
  architecture constants come from `Qwen/Qwen3.8-27B` `config.json` →
  `text_config` (64 layers = 48 linear-attention + 16 full-attention, hidden
  5120, 24/4 heads, head_dim 256, max_position_embeddings 262144).
  `kv_calc_supported: false` mirrors the registry's `kvcalc_key="SKIP"` — the
  constants are accurate, but KV projections have not been validated against
  measured VRAM for this model.
- **Top-level `README.md` claimed `IQ4_KS/NL`** as if both quants ship, and
  `~15 GB weights` — contradicting every other file in the PR. Corrected to
  `IQ4_NL/Q4_K_M` and ~16.3 GB.
- **Compose quick-start pointed weights at the wrong directory.** The documented
  `hf download --local-dir $MODEL_DIR/qwen3.8-27b-gguf/iq4ks` does not match the
  `GGUF_FILE` default `qwen3.8-27b-gguf/Qwen3.8-27B-IQ4_NL.gguf`, so following
  the card as written left the compose unable to find the file. Fixed in both.
- **Model card was missing `What's working` / `What's not working today`**,
  which sibling cards carry — the sections that matter most for a model whose
  own VRAM table marks 3 of 4 rows untested. Added, with the IQ4_NL/200K path
  explicitly listed as unvalidated.
- **`weights_companions=("gguf_mmproj_bf16",)`** referenced an alias registered
  nowhere. Changed to the conventional `gguf_mmproj_f16`, which is also the
  projector the bench rig actually mounted.
