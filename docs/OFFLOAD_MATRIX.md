# Offload Matrix — sweeping CPU-offloaded MoE serving configs

`scripts/offload-matrix.sh` boots a `llama-server` per configuration, drives concurrent requests at it, scrapes throughput + cache + bandwidth telemetry, and appends one row per arm to a TSV. `scripts/offload-matrix-render.py` turns that TSV into readable tables.

It exists because CPU-expert-offloaded MoE serving has **too many interacting knobs to tune one at a time**, and because several of those knobs fail *silently* — producing plausible numbers that are measuring something other than what you think.

> ### ⚠ Engine scope: llama.cpp and its forks only
>
> Not vLLM, SGLang, TensorRT-LLM, MLC or ExLlama. This is not a flag-syntax limitation — the tool measures a mechanism only llama.cpp implements this way.
>
> **llama.cpp offload computes the offloaded experts *on the CPU*.** Activations cross PCIe; throughput is bound by RAM bandwidth and core count. **vLLM's CPU offload streams expert *weights* to the GPU and computes there** — bound by PCIe bandwidth. The dimensions here (offloaded layer count, a VRAM cache of hot CPU-resident experts, its admission policy) have no counterpart in that design, so the numbers would not be comparable even if the flags were translated. Benchmark those engines with their own tools.

---

## 1. Quick start

```bash
# smallest useful run: one no-spec baseline + one speculative arm
MODEL=/path/to/model.gguf bash scripts/offload-matrix.sh

# see what a sweep would run, without running it
PLAN=1 MODEL=/path/to/model.gguf SWEEP_OFFLOAD="19 28" SWEEP_N="1 4" bash scripts/offload-matrix.sh

# read the results (TSV is the source of truth; re-render freely)
python3 scripts/offload-matrix-render.py offload-matrix-out/offload-matrix-results.tsv
python3 scripts/offload-matrix-render.py <tsv> --bw      # bandwidth / utilisation
python3 scripts/offload-matrix-render.py <tsv> --cache   # pool / hits / evictions
python3 scripts/offload-matrix-render.py <tsv> --md      # markdown, for pasting
```

`MODEL` is the only required input — everything else is auto-detected (thread count, GPU count and tensor split, total layer count, expert count, `--moe-cache` support) or has a default. Nothing rig-specific is baked in.

---

## 2. The dimensions

Each is a **space-separated list**; the tool runs the cartesian product, one server boot per arm.

| Env | Meaning | Default |
|---|---|---|
| `SWEEP_OFFLOAD` | Number of layers whose experts go to CPU (`-ot` regex generated, not hand-written) | empty = no offload |
| `SWEEP_N` | Concurrency (`-np`) | `1` |
| `SWEEP_NMAX` | Draft depth. **`0` = the paired no-spec baseline** | `0` |
| `SWEEP_DRAFTER` | Drafter type (`ngram-mod`, `suffix`, … ; model-based drafters need `DRAFTER_GGUF`) | `ngram-mod` |
| `SWEEP_CTX` | Total context (`-c`); per-slot ctx = this ÷ N | `32768` |
| `SWEEP_CACHE` | Per-device VRAM expert cache, MiB (`--moe-cache`) | `0` = flag not passed |
| `SWEEP_ADMIT` | Cache admission policy, `ADMIT/THROTTLE` pairs | `default` |
| `SWEEP_SHAPE` | Workload: `copy`, `novel`, `code` | `copy` |
| `REPS` | Repeats per arm — **each rep is a fresh boot** | `1` |

Other knobs worth knowing: `ROUNDS` (default 4; **round 0 is a discarded warm-up**), `GEN` (max tokens, 900), `RESERVE_MB`, `KV_TYPE`, `UBATCH`, `THREADS`, `PORT`, `OUT_DIR`, `PROBE_TIMEOUT` (seconds the `n_layer` probe waits, 300), `PROMPT_FILE` (replace the built-in shapes with your own static prompt), `HETERO`.

> **⚠ `SWEEP_DRAFTER` × `SWEEP_NMAX` does not make drafters comparable on its own.** `SWEEP_NMAX` is *draft length*, and it maps correctly to whichever flag the selected drafter uses (`--spec-ngram-mod-n-max`, `--spec-ngram-simple-size-m`, `--draft-max`, …). What the sweep does **not** control is each drafter's **lookup length**, and the defaults differ: `ngram-simple` needs a 12-gram match (`--spec-ngram-simple-size-n`, default 12) where `ngram-mod` matches on 24 (`--spec-ngram-mod-n-match`). Equal `n-max` therefore means equal draft length and *different matching behaviour* — a drafter that shows no gain may simply be **failing to find a match**, not drafting badly. Pin the match lengths through `EXTRA_ARGS` (and re-check them against your fork's `--help`, the defaults are fork-specific) before ranking drafters against each other.

### Presets

`PRESET=<name>` fills in a sensible dimension set; anything you set explicitly still wins.

| Preset | Sweeps | Use when |
|---|---|---|
| `smoke` | one baseline + one spec arm | verifying the tool works on your rig |
| `spec` | `n-max 0,3`, REPS 2 | "is speculation worth it here?" |
| `drafters` | several drafters at fixed depth | picking a drafter |
| `depth` | `n-max 0,2,3,4,5` | tuning draft depth |
| `concurrency` | `N 1,2,4,6,8` | finding the aggregate-throughput knee |
| `offload` | several offload counts | finding the VRAM/speed trade point |
| `cache` | cache sizes × admission | tuning the expert cache |
| `full` | N × depth × offload | overnight |

---

## 3. Using it effectively

### 3a. Respect the dependency order

The dimensions are **not independent**. Measuring a downstream one on an unsettled base wastes the run:

```
1. offload layers   BASE — changes hit rate, which changes miss-streaming,
                    which changes the acceptance a drafter must clear to pay for itself
2. cache + admission tunes that base (admission alone moved throughput 5.4% on the
                    reference rig — rank drafters at the wrong admission and you rank
                    them handicapped)
3. ctx              bounded by what layout+cache leave free
4. concurrency (N)  interacts with ctx (per-slot ctx = total ÷ N)
5. drafter / depth  the most sensitive to everything above
```

Sweep top-down, in stages, rather than one giant product. `PLAN=1` prints the staged sequence. The tool warns when a sweep inverts this order.

### 3b. Repeat before you rank

On the reference rig, **within a boot** the per-request decode rate is essentially noise-free (0.9% spread across four requests). **Between boots at identical config it is low single digits** — three consecutive boots of the same arm landed within 0.5% of each other (37.24 / 37.09 / 37.26 tok/s).

So a single boot can resolve a large effect, but it cannot resolve a small one, and it gives you no way to tell which case you are in. `REPS` re-boots per rep and the renderer reports **medians** — use `REPS≥3` for any ranking decision.

> **What actually bites is not boot noise — it is a config difference you did not notice.** The 12–22% swings that motivated this section originally turned out to be an arm running with the expert cache silently disabled (§5a), not variance. Before attributing a gap to noise, check `status`, the pool lines, and §5's trap list. Noise on this rig is small; silent misconfiguration is not.

### 3c. Workload shape flips the *sign* of speculative results

| Shape | What it is | Speculation on the reference rig |
|---|---|---|
| `copy` | reproduce a JSON doc with edits — agentic/tool-loop-like | **+73%** |
| `novel` | 800-word essay (byte-identical to `bench.sh`) | **+2.7%** |
| `code` | quicksort implementation (byte-identical to `bench.sh`) | mid |

Benchmark the shape that matches your workload. A spec-dec gain measured on `copy` does not transfer to prose. The renderer refuses to pair a spec arm against a baseline of a different shape for exactly this reason.

`novel` and `code` are byte-identical to `bench.sh`'s canonical prompts so rows can be laid against existing bench results; the consequence is they are **identical across slots** at `N>1`, which flatters multi-slot numbers through expert-union overlap. Use `copy` (seeded per slot) for concurrency work.

---

## 4. Reading the output

The TSV is append-only and is the source of truth — the renderer never mutates it, so you can re-group and re-render without touching the rig.

- `agg` — aggregate tokens/sec across all slots, including prefill (what a fleet sees)
- `per-strm` — server-side decode rate for one stream (what a single user feels)
- spec rows show `agg` with a **delta against their paired no-spec baseline**, matched on offload/N/ctx/cache/admit **and shape**
- `RAM rd MB/s` is **derived** (misses × avg expert size ÷ elapsed), not a hardware counter
- PCIe/SM/mem-controller figures are `nvidia-smi dmon` means across the arm, including the idle tail

### Arm status — an arm that is not `OK` must be excluded

| Status | Meaning |
|---|---|
| `OK` | usable |
| `CACHE_DISABLED` | expert cache was requested but **never allocated** — see §5a |
| `INVALID_BYPASS` | cache exists but **at least one** decode was refused it. The engine warns **once per session**, so the `bypass` count bounds nothing about how much of the run ran uncached — the arm is not usable for cache-dimension conclusions, but its throughput is not necessarily wrong. See §5b |
| `MB_CLAMP_CONFLICT` | `n-max` too high for the batch clamp; refused pre-boot |
| `PORT_BUSY` | something already served that port; arm did not boot |
| `NO_TOKENS` | zero tokens returned — every request failed |
| `REQ_ERRORS` | some requests errored (count in the `errors` column; first message in the arm log) |
| `BOOT_FAIL` / `TIMEOUT` | server never started / never became ready (OOM at high ctx is the usual cause) |

The renderer lists non-`OK` arms separately with the reason. **Do not read a number off a non-`OK` row.**

Every arm writes a log to `OUT_DIR` whose header records the fully-resolved server command line, so any single arm can be reproduced by hand without re-deriving auto-detected values.

---

## 5. Traps that produce *plausible wrong numbers*

These are the reason the tool has as many refusal paths as it does. All were found the hard way.

### 5a. The expert-cache reserve can silently disable the cache

The cache reserves VRAM before sizing its pool. If the reserve consumes the free budget — easy at high context on 24 GB cards — **the pool is never created and the run measures the no-cache path**, logging `no cache budget` rather than any bypass warning.

Detected as `CACHE_DISABLED` (absence of a `[moe-cache] enabled:` line). Fix by lowering the reserve (`RESERVE_MB=1536`) or reducing ctx/pool.

Measured cost of missing this on the reference rig — identical config, only the reserve differing:

| arm | cache off | cache on | cache worth |
|---|---|---|---|
| no-spec | 30.14 | 33.89 | **+12.4%** |
| spec (`ngram-mod` depth 3) | 47.25 | 58.64 | **+24.1%** |

Speculation roughly doubles the cache's value — a draft-verify step touches more experts per unit wall-clock, so misses cost more. **Measuring either feature with the other silently off understates it.**

### 5b. The cache batch clamp must satisfy `MAX_BATCH ≥ n-max + 1`

Below that, the cache refuses **every** decode and the arm measures the no-cache path while looking healthy. The tool computes the clamp for you and refuses arms the engine's `[1,8]` clamp cannot satisfy (`MB_CLAMP_CONFLICT` at `n-max ≥ 8`).

Note "cache requested" and "cache active" are different propositions. Only the second is worth asserting.

### 5c. Hit rate does not predict throughput

On the reference rig a 28-layer offload reached 82%/88% hit rates and was still **11% slower** than a 19-layer offload at 71%/56%. CPU layer count dominates. Optimise for measured throughput, not for the hit-rate column.

### 5d. Measure a server you actually booted

The tool refuses to start an arm if the port already answers (`PORT_BUSY`), because a foreign server would satisfy the readiness probe while the arm's own boot bind-fails — every row then measures the same untouched config and reports `OK`.

---

## 6. Inheriting config from a running server

With `MODEL` unset and a `llama-server` already running, the tool reads that process's `/proc/<pid>/cmdline` and inherits `-m`, `-t`, `-ub`, `-ct`, `-ngl`, rope flags, and the current `-ot` offload count as a sweep anchor. Explicit env always wins; `INHERIT=0` disables.

It will then **refuse to run** while that server is up, since every arm boots its own and they would compete for VRAM. Stop it first (the inherited config is already captured, so re-running after the kill keeps your settings).

---

## 7. Requirements

- `llama-server` (`LLAMA_SERVER=`, or on `PATH`)
- a GGUF **MoE** model. On a dense model the offload regex matches nothing, so `-ot` is a silent no-op and every offload column is meaningless. The tool refuses that case **when it can detect it** — the expert count comes from the layer probe, which is skipped if you set `LAYERS_TOTAL`. In that case it warns instead; set `N_EXPERT=<n>` to arm the guard.
- `nvidia-smi` for GPU telemetry — optional; those columns become `-` without it
- The **cache dimensions** (`SWEEP_CACHE`, `SWEEP_ADMIT`) and the pool/hit columns need a build with the VRAM MoE expert cache (`--moe-cache`), which is **not in mainline llama.cpp**. The tool auto-detects it and degrades gracefully: on mainline, the offload / concurrency / ctx / drafter / shape dimensions all still work, and the cache dimensions are disabled with a note.

## 8. What it does not do

Compare engines, tune sampling, measure quality, or test correctness of output. It measures **serving throughput and latency under a configuration**. Pair it with `scripts/quality-test.sh` before shipping any config it recommends — a fast config that degrades output is not a win.
