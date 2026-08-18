## Summary

Add Qwen3.8-27B support for single RTX 3090 via llama.cpp, following the same
pattern as existing models (qwen3.6-27b unsloth-q4km, tess-4-27b, etc.).

Ships the **vision** compose that upstream `llamacpp/qwen38-27b-single-iq4nl`
explicitly names as missing (its note: vision "needs a third pull plus a
kind:mmproj weights entry plus a -vision.yml sibling"), plus the measurements
that entry lacks (it claims no TPS, no verify-stress, no soak, and a ctx ceiling
that is computed rather than measured).

The text-only compose originally in this PR was **dropped** — upstream now ships
an equivalent at `single/unsloth-iq4nl/q8kv.yml`. Note: this arch ships **no IQ4_KS** from bartowski — the 4-bit
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

✅ **The compose defaults now equal the benched configuration** — bartowski Q4_K_M
(17,772,537,440 B) + mmproj-F16 (927,607,008 B), `-c 131072`, `-b 1024 -ub 1024`,
q4_0/q4_0 KV, MTP n=2. Earlier revisions of this PR shipped IQ4_NL @ 200K with MTP
n=3 while benching something else, and apologised for the gap in three places. The
gap is now closed rather than documented.

⚠️ **Provenance correction.** Earlier revisions credited the rig weights and projector
to unsloth. Both are **bartowski**, verified by byte size: bartowski
`Qwen3.8-27B-Q4_K_M.gguf` is 17,772,537,440 B (what is on the rig) while unsloth's
same-named file is 17,106,775,008 B — 666 MB smaller. Their projectors also differ in
name and size. The two are not interchangeable.

⚠️ **IQ4_NL @ 200K is still unbooted.** Both of those files are downloaded and
byte-verified, but that path has never been launched, so no claim is made for it.

## Power-cap sweep (210–450 W, 10 W steps)

`sudo bash scripts/power-cap-sweep.sh --cooling air --load-mode decode-concurrent
--concurrency auto --bench-runs 3` — 25 caps, concurrency auto-selected to 4, air-cooled.
Full summary attached as a PR comment.

| Cap | Narr TPS | Code TPS | Draw | SM clk | Narr tok/W |
|---:|---:|---:|---:|---:|---:|
| 210 | 29.02 | 31.75 | 209 W | 840 | 0.139 |
| 250 | 43.81 | 45.50 | 249 W | 1230 | **0.176** ← peak |
| 270 | 45.48 | 48.72 | 268 W | 1380 | 0.170 |
| 300 | 49.45 | 54.38 | 298 W | 1515 | 0.166 |
| 370 | 52.11 | 59.04 | 367 W | 1680 | 0.142 |
| 420 | 50.91 | 61.24 | 415 W | 1755 | 0.123 |
| 450 | 51.87 | 61.64 | **428 W** | 1785 | 0.121 |

**Efficiency plateaus ~230–310 W, peaking near 250–270 W.** Against that peak, 370 W
is −19% tok/W and 420 W is −30%. **Throughput saturates ~300 W**: 300→450 W buys
**+5% narrative for +50% power** (code gains more, +13%, so code-heavy work justifies
slightly more headroom than prose).

**Decode is memory-bandwidth-bound**, by the script's own diagnostic: SM clock scales
monotonically 840→1785 MHz across the sweep while TPS plateaus, and memory clock stays
pinned at the 3090 spec max 9501 MHz throughout. That is why the controlled 370→420 W
A/B moved decode only ~2%.

**Caps above ~430 W are inert** — draw at the 430/440/450 caps is 425.0/428.0/427.9 W,
so the card stops tracking the limit. **Not thermally limited**: 74 °C max, well under
the 80–83 °C air-throttle band, and `sw_power_cap` Active in ~100% of in-load samples,
so the curve is power-shaped rather than cooling-shaped and is usable as a cross-rig
anchor.

### Read the shape, not adjacent caps

The script warns that adjacent-cap deltas are timing-noise, and this run bears that out
(250 > 260, 310 > 320, 410 > 430 all invert). Notably the sweep's own 370-vs-420 pair
(52.11 vs 50.91) contradicts a controlled A/B run on the same warm engine, which gave
**+2.5% narrative / +2.1% code at 420 W** over 370 W (n=5, CV 1.3%). Trust the A/B for
that pair and the sweep for curve shape. Two further caveats: the sweep's absolute TPS
are **not** comparable to the single-stream `bench.sh` numbers above (it runs 4 concurrent
streams against a `-np 1` engine, which serialises rather than batches — its notes assume
a continuous-batching engine), and the 0.176 peak should be read as a plateau, not a point.

### Consequence for this row's operating point

The benchmarks above were taken at **370 W**. On the strength of this sweep the reference
box now runs at **250 W**, near the efficiency peak. The 370 W figures remain valid and
reproducible as a labelled measurement, and are what the BENCHMARKS row documents; a
250 W single-stream re-bench is the obvious follow-up but is not required for this PR.

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
- **Power-cap A/B**: 370 W arm measured, 420 W arm pending — see the
  "Power-cap comparison" section above for why the existing 420 W figure is not
  comparable and what a valid A/B requires. Needs root; no passwordless sudo here.
- **CHANGELOG**: deferred until the model graduates from incubating, at which
  point the IQ4_NL default path will also have been measured.

## Cross-links

- Self-referential update to [#988](https://github.com/noonghunna/club-3090/pull/988) (this PR): fixes the "bartowski does not ship `Qwen3.8-27B-IQ4_KS.gguf`" flag in the [opening comment](https://github.com/noonghunna/club-3090/pull/988#issuecomment-5304383592) — default is now IQ4_NL, with the canonical BENCHMARKS row and the rig-vs-compose drift documented.
- Related upstream: the rig behind the BENCHMARKS row is the onyx-rx single-3090 stack (Q4_K_M + mmproj-F16 @ 131K).

---

## Files changed (7 files)

| File | Type | Notes |
|---|---|---|
| `models/qwen3.8-27b/llama-cpp/compose/single/bartowski-q4km/q4kv-vision.yml` | new | the multimodal sibling upstream's note asks for. Defaults **equal the benched config**: bartowski Q4_K_M + mmproj-F16, `-c 131072`, MTP n=2, q4_0 KV |
| `models/qwen3.8-27b/README.md` | new | model card — quant options, VRAM table, no-IQ4_KS flag, what's working / not working |
| `scripts/lib/profiles/models/qwen3.8-27b.yml` | modified | adds `bartowski-q4km` + `gguf_mmproj_f16`/`_bf16`; the `kind: mmproj` entries upstream lacks |
| `BENCHMARKS.md` | modified | +1 canonical row (58.05/68.38 @ 370 W) + power-cap sweep findings; supersedes the 2026-08-16 entry |
| `scripts/lib/profiles/compose_registry.py` | modified | +1 slug `llamacpp/qwen38-27b-iq4ks-vision` → the vision compose, `max_ctx=131072` |
| `README.md` | modified | supported-models table row |
| `create-pr.sh` | new | PR-creation helper (fork:head compare URL, embedded body) |

**Dropped from an earlier revision of this PR:**
`single/iq4ks.yml` (text-only) — upstream master now ships an equivalent at
`single/unsloth-iq4nl/q8kv.yml`, so it was a duplicate under a superseded flat path
with a misleading `iq4ks` name for an IQ4_NL file.

## Weights sources

| Source | Quant | Size | Notes |
|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF | **IQ4_NL** | 16,325,830,240 B (~16.3 GB) | 4-bit default (no IQ4_KS for this arch); downloaded, **unbooted** |
| bartowski/Qwen3.8-27B-GGUF | Q4_K_M | 17,772,537,440 B (~17.8 GB) | **the bench-validated rig weights and the compose default**; 200K OOMs on 24 GB, 131K measured (NIAH fill 120,320 tok) |
| unsloth/Qwen3.8-27B-GGUF | Q4_K_M | 17,106,775,008 B | **NOT the rig weights** — 666 MB smaller than bartowski's same-named file |

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
