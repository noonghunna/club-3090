# ULTRAMAX review validation — September 13, 2026

## Artifact update — October 4, 2026

This section records the mixed-prefill fix. The September results below
remain the record for the previous artifact.

- Source: `7646a2d8a8d903847d81f506c6a0d26bc212c97f`, included in upstream
  v0.1.0; the PR and release are tracked in `docs/UPSTREAM.md`.
- Image: `ghcr.io/antonprokopyev/fa2-fp8kv-sm86@sha256:ec26059d1c7a7eb4b4d916ddac495a0b937eae8e4594297f7ed625368c30cd7e`.
- Artifact ID: `b0ce0370c7ca5c0d0e688e744f3665f3a099407fc2a38a4102d81eaaf5707a2d`.
- Build ABI: Linux x86-64, CPython 3.12, PyTorch 2.13.0+cu130, CUDA 13.0,
  C++11 ABI enabled. Anonymous digest pull and checksum-verified export pass.

The installed wheel and CUDA libraries passed the entire upstream GPU suite
on each RTX 3090 with vLLM 0.29.0: six numerical shapes, eighteen decode graph
replays, twelve window cases, fifteen full-kernel cases, six prefill shapes
plus all finite E4M3 encodings, nine mixed-prefill regressions, and three
backend cases. The prefill check reported 3040.05 MiB peak PyTorch allocation
on each card. The old artifact failed six of the same nine mixed-prefill GPU
cases with the sequence-envelope error; the new artifact passed all nine.
All nine mixed-prefill regressions and backend tests also passed with stock
vLLM 0.30.0 (`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`).

The actual `dual/fp8/dflash2.yml` compose booted Qwen3.8-27B FP8 on both
cards with the new artifact and vLLM 0.30.0. The test used `SPEC_N=0`,
`MAX_MODEL_LEN=98304`, and `KV_CACHE_MEMORY_BYTES=4000000000` per card.
`verify-full.sh` passed, including vision 4/4. `verify-stress.sh` also passed:
5/5 boundary checks and 2/2 recall ladder groups at approximately 10K, 30K,
60K and 90K tokens. The extra ceiling ladder was skipped because 92% of
98304 is below its 95000-token starting rung. The canonical `bench.sh`
invocation completed with exit 0, but its final stdout table was not retained
in the capture; no TPS numbers are claimed from that run.

The host's loaded NVIDIA
module was 595.71.05 while its userspace libraries were 595.91.07; isolated
test containers used SHA256-verified official NVIDIA 595.71.05 libraries
and explicit GPU device mounts. No system driver files were changed.

The compose status remains experimental. These results do not validate
the shipped 262144-context DFlash2 configuration, TP=4/8, SM89/SM120,
complete quality or soak. The scheduler envelope guard remains enabled
for conservative rollout and older artifact overrides. No throughput
improvement is claimed.

Offline guards pass: `test-fa2-artifact`, `test-fa2-envelope`,
`test-patch-attribution`, `test-compose-status-drift`,
`test-compose-registry-disk`, `test-profiles-compat`, and `test-launch-compat`.

## Previous artifact validation

The revised configuration uses a native `FLASH_ATTN` plugin for target and
DFlash2. It does not replace the FlashInfer registry entry. The profile stays
experimental. SM86 has runtime measurements; SM89 and SM120 have compiled
binaries but no physical-GPU validation. SM90/SM100 retain stock FlashAttention
and have no new measurements in this report.

## Controlled comparison

All four arms used the same two RTX 3090 cards at 250 W, PCIe read-P2P enabled,
no NVLink, TP=2 and stock vLLM 0.29.0. The rig has an EPYC 7K62, 373.3 GiB RAM
and NVIDIA driver 595.71.05. Custom all-reduce was disabled.

- Target FP8 revision: `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- DFlash2 W4A16 revision: `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.
- DFlash n=7, explicit FP8 E4M3 target/draft KV, BF16 attention compute.
- Context 262144, one sequence, batch 2048, long-prefill threshold 0,
  fixed KV allocation 6335076762 bytes per card, vision enabled.
- Same GDN and DFlash dense-KV fixes, prefix caching and chat template.
- Canonical narrative/code: 3 warm + 5 measured, max_tokens 1000/800,
  temperature 0.6, top_p 0.95, top_k 20, min_p 0, thinking OFF.
- Quality: `--medium --no-thinking --strict-thinking`, pack sampling defaults
  (temperature 0), not server sampling. All packs reported valid thinking OFF.

The stock control changes only the layout contract: its backend advertises
LBNHC, and the metadata builder and implementation return LBNHC from
`kv_cache_layout`. Attention execution and graph-support declarations remain
stock FlashInfer. This avoids the copied draft CacheConfig reading a layout
before the target's layout RPC reaches it. Both arms therefore use explicit
FP8 draft KV; the control does not silently switch the drafter to BF16.

The two eager arms isolate attention implementation while removing graph
policy as a variable. The compiled stock arm preserves its normal graph
policy; it does not force `UNIFORM_BATCH`. The plugin arm supports full graph
capture. Thus the compiled comparison measures the serving stacks, including
their graph capabilities. It is not a pure CUDA-kernel speedup measurement.

| Arm | Narrative decode / wall tok/s | Code decode / wall tok/s | Decode CV, narrative / code | Peak MiB, GPU 0 / 1 | Minimum free MiB, GPU 0 / 1 |
|---|---:|---:|---:|---:|---:|
| Stock layout control, eager | 31.03 / 30.80 | 58.46 / 57.05 | 2.5% / 5.0% | 23664 / 23664 | 463 / 463 |
| FLASH_ATTN plugin, eager | 32.73 / 32.54 | 59.68 / 58.32 | 1.9% / 10.2% | 23002 / 23002 | 1125 / 1125 |
| Stock layout control, compiled | 75.06 / 74.20 | 140.82 / 133.97 | 4.1% / 2.4% | 23756 / 23756 | 371 / 371 |
| FLASH_ATTN plugin, graphs | 98.61 / 97.23 | 191.17 / 182.70 | 6.2% / 1.5% | 22948 / 22948 | 1179 / 1179 |

VRAM was sampled every 500 ms across each arm's verify, benchmark, greedy and
quality phases. These are short-context measurements, not a full-context
memory guarantee. CPU artifact compilation overlapped part of the compiled
stock benchmark; this is a timing limitation of that comparison. The eager
control and acceptance counts remain useful, but the compiled ratios should
not be treated as an isolated, noise-free kernel result.

All four arms passed verify-full 10/10, including vision 4/4.

| Arm | ToolCall | InstructFollow | Structured Output | Data Extraction | ReasonMath | Total |
|---|---:|---:|---:|---:|---:|---:|
| Stock eager | 14 | 13 | 15 | 11 | 11 | 64/75 |
| Plugin eager | 14 | 13 | 15 | 11 | 11 | 64/75 |
| Stock compiled | 14 | 13 | 15 | 10 | 11 | 63/75 |
| Plugin graphs | 13 | 13 | 15 | 11 | 11 | 63/75 |

Each pack has 15 cases. Versions: tc1.0.1, if1.0.0, so1.1.0, de1.2.0, rm1.0.0.
The one-case ToolCall difference is 6.7 percentage points. Equal aggregate
scores do not establish equal outputs or statistical quality equivalence.

## Acceptance counters

Counters below are before/after differences from `/metrics`, after counters
remain unchanged for 12 seconds. Each narrative/code phase includes all
3 warmups and 5 measured requests. The engine counter creation timestamp was
checked for restarts; accepted-position sums were checked against accepted
tokens. No idle log window was used to estimate whole-phase acceptance.

Mean acceptance length = `1 + accepted_tokens / draft_steps`.
Position acceptance = `accepted_at_position / draft_steps`; the seven
positions below are raw counts, not rounded ratios inferred from logs.

| Arm | Phase | Draft steps | Accepted tokens | Accepted at positions 0 through 6 | Mean length |
|---|---|---:|---:|---|---:|
| Stock eager | Narrative | 2594 | 5291 | 1973, 1347, 839, 521, 314, 187, 110 | 3.0397 |
| Plugin eager | Narrative | 2556 | 5355 | 1943, 1316, 854, 555, 339, 218, 130 | 3.0951 |
| Stock compiled | Narrative | 2625 | 5374 | 1992, 1355, 869, 543, 310, 199, 106 | 3.0472 |
| Plugin graphs | Narrative | 2622 | 5297 | 1996, 1344, 822, 502, 323, 196, 114 | 3.0202 |
| Stock eager | Code | 818 | 3886 | 755, 673, 612, 535, 481, 433, 397 | 5.7506 |
| Plugin eager | Code | 863 | 4107 | 790, 705, 639, 566, 508, 466, 433 | 5.7590 |
| Stock compiled | Code | 772 | 3576 | 700, 630, 560, 503, 447, 388, 348 | 5.6321 |
| Plugin graphs | Code | 829 | 3936 | 756, 686, 612, 552, 487, 437, 406 | 5.7479 |
| Stock eager | Greedy | 758 | 2908 | 666, 537, 445, 390, 332, 292, 246 | 4.8364 |
| Plugin eager | Greedy | 774 | 2897 | 672, 548, 447, 379, 328, 282, 241 | 4.7429 |
| Stock compiled | Greedy | 840 | 3155 | 728, 588, 473, 410, 362, 318, 276 | 4.7560 |
| Plugin graphs | Greedy | 843 | 3218 | 724, 604, 504, 430, 368, 319, 269 | 4.8173 |

The extra greedy phase sends the same eight requests at temperature 0,
top_p 0.95, top_k 20, min_p 0, seed 20260913 and zero presence penalty:
attention essay, quicksort, JSON primes, binary search counterexample, LRU
cache, average train speed, TCP/UDP comparison and sorted-list merge.
Stock eager and plugin eager produced identical text on 3/8 requests; the
others diverged after 263–1143 common characters. Bit-identical output is not
claimed. Kernel reference checks, task scores and acceptance distributions
provide separate evidence.

The large acceptance jump and flat tail from the review's isolated log window
did not reproduce under matched phases. Stock already reaches mean length
5.75 on code versus 3.04 on narrative. Workload and measurement window are
therefore plausible explanations for the earlier difference, but those older
windows lack enough matched raw evidence to establish their exact cause.

## Prebuilt artifact

- Source: `AntonProkopyev/fa2-fp8kv-sm86@0fa02cbb760fbcc4a94ba1ee376e89825f8f43f4`.
- Image: `ghcr.io/antonprokopyev/fa2-fp8kv-sm86@sha256:da040941fa048fd5fdfce520503341c0beda4ce41436a1c1fecaf3f8a99777c7`.
- Artifact ID: `731d1942112a7cf35be4f0add0919072c06cac47e3bad4f674ff0157edad0e3f`.
- Build/runtime ABI: PyTorch 2.13.0+cu130, CUDA 13.0, CPython 3.12,
  Linux x86-64, C++11 ABI enabled.
- `cuobjdump` confirms SM86, SM89 and SM120 cubins in both libraries.
- Offline export and repeated export passed; corrupting the cached library
  caused an explicit checksum failure on the next export.
- The package is public. Pulling the pinned digest with an empty Docker
  configuration succeeded, followed by offline extraction into an empty
  directory. No GitHub credentials are required by consumers.
- Offline install from the actual image passed on two RTX 3090 cards.
- `check_backend.py` passed native KV-write byte checks, FP32 attention
  references and CUDA Graph replay with changing lengths. Cases cover
  causal D256, noncausal D128 with a 2048-token window, and bounded prefill.
  Maximum absolute errors were 0.0046932, 0.0004857 and 0.0053701 respectively.
- Twelve CPU consumer-contract checks pass, including bad identity, payload,
  ABI, missing SM binary, unsupported/mixed GPU, insufficient GPU count,
  path escape and missing wheel. Native SM90/SM100 branch checks use mocks;
  they are not GPU validation.
- Three additional checks execute the compose entrypoint with simulated PCIe,
  PCIe P2P and NVLink detection. They confirm that custom all-reduce is enabled
  only for NVLink and the requested allocator is preserved when it is disabled.

The four-arm comparison used the preceding SM86 plugin build. The published
artifact adds SM89/SM120 compilation and matching host guards. Its actual
compose receives a separate validation pass below.

## Actual compose validation

The final compose retains the tested 6335076762-byte reservation and a
267493-token KV pool. Combined vision leaves 1021 MiB free per card, below
the unchanged 1024 MiB gate. This is a disclosed experimental-profile limit.
Reducing the reservation by 64 or 32 MiB preserved request capacity but
evicted the near-limit prefix cache: a continuation reprocessed all 259932
tokens in about 266 seconds instead of reusing 255440 cached tokens in about
8 seconds. An 8 MiB reduction preserved the cache but did not increase physical
free memory. None of those reservation changes ships.

The final `dual/fp8/dflash2.yml` was booted through `switch.sh --force
vllm/qwen38-27b-dual-ultramax` with the verified local image, loopback port
8011, installed weights, thinking OFF and proven PCIe P2P enabled. The init
service and read-only artifact volume were exercised. Full target/draft graph
capture succeeded. Logs confirm `disable_custom_all_reduce=True` on this rig.

- Verify-full: 10/10, including vision 4/4.
- Canonical benchmark: 103.34 / 191.91 decode tok/s (CV 4.3% / 3.8%);
  101.95 / 182.79 wall tok/s, narrative/code respectively.
- Prefill: about 1754.6 tok/s at 10K and 1384.0 at 93331 input tokens.
  One warmup per depth; three measured 10K requests and one measured 93K.
- Quality quick: 26/30, ToolCall 13/15 and InstructFollow 13/15, valid thinking
  OFF, same pack versions as above.
- Registry canonicalization, profile diagnostics and KV-calculator calibration
  pass. KV projection for this model/external drafter remains unavailable.
- Exact context: 261000 input tokens, 11 output tokens, 265.136 seconds,
  zero cached tokens, exact verification-code recall. Peak memory was
  23026 / 23026 MiB. A 263000-token request returned HTTP 400.
- Combined vision: 259872 input tokens, including 4070 image tokens, zero
  cached tokens, 265.744 seconds. All five fixture facts were retained.
  Follow-ups at 259932 and 259992 tokens took 8.140 and 8.184 seconds, with
  255440 cached tokens and all five facts retained on each turn.
  Peak memory across these checks was 23106 / 23106 MiB (1021 MiB free).

- The concluding stress run completed all response probes and all ten fresh
  needle depths. Its last rung recalled the secret at 240662 input tokens,
  1014.1 prefill tok/s, with zero cached tokens. **The command exited 1 solely
  for the memory margin:** 1021 MiB free per card versus the unchanged 1024 MiB
  threshold. This is not a full stress PASS. The 4 MP vision path was warmed
  before the stress run; peak memory stayed 23106 / 23106 MiB.

- Continuous soak: **PASS**, five sessions × five turns. Zero errors, zero
  silent-empty responses, zero VRAM growth (46212 MiB total throughout),
  100.0% throughput retention. All 25 decode measurements came from the
  engine's `/metrics` time-per-output-token counters; p50 was 168.48 tok/s.
  This is a bounded multi-turn sample, not a production guarantee.

The fixed KV reservation bypasses `gpu_memory_utilization`; the 262K capacity
comes from reservation and scheduling settings, not a proven kernel-only gain.
No margin threshold is reduced to obtain a passing result.

A preliminary context attempt was interrupted when an existing catalog test
invoked the real launcher and removed the serving container. Docker recorded
external kill/stop/destroy events. Its dependent failures are excluded from
the model results. The remaining catalog tests run with an unavailable Docker
socket, and the GPU validation above was restarted separately. A preliminary
artifact run also used the detector's PCIe custom-all-reduce setting; its
timings are excluded from the final compose numbers above.

## Catalog tests

The full 160-script sweep completed. After fixing and rerunning the affected
guards, 154 pass and six fail. Two tests added by master `c4ed9ca` also pass,
bringing coverage to 156/162. Its soak engine-rate fix is included in the
concluding soak run. All 15 final guards for the changed behavior
pass, including artifact installation, SM compatibility, mounts, attribution,
sampler/spec toggles, restart policy, executable modes and launcher parity.

Five failures match the existing v0.11.0 release baseline: `test-bench-capture`
(triad worker-count assertion), `test-loop-input`, `test-pull`,
`test-pullemit-capture` and `test-submit-pull` (capture manifest `outcome` fields).
The sixth, `test-report-resolved-config`, reproduces on clean master `2dcbb99`
with the same two missing-flag assertions. These failures are not counted as
passes and are outside the FA2 integration changes.
