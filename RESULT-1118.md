# RESULT-1118 — estate-bar VRAM breakdown with per-GPU drill-down

## Branch

`feat/1118-estate-vram` (off `origin/master`)

```
 scripts/lib/vram_breakdown.py        |  ~95 ++++++-------
 scripts/tests/test-vram-breakdown.sh |  ~75 +++++++
 tools/serve-cockpit/club3090_cockpit/app.py      | 164 ++++++++++++++++++-
 tools/serve-cockpit/club3090_cockpit/services.py |  43 ++++++
 tools/serve-cockpit/tests/test_app_headless.py   | 151 +++++++++++++++++
 6 files changed, 522 insertions(+), 11 deletions(-)
```

Two commits: `29fef847` (parser `--json` + clamp), `ca7f3b8f` (c3 panel).

## What changed and why

1. **Parser (`scripts/lib/vram_breakdown.py`)**: the parse moved into a pure
   `breakdown(text)`; `main()` gained a `--json` flag emitting one object per
   device (`model/kv/state/compute/pool/used/total/unaccounted`) plus a
   `warnings` list. Human output stays the default and is unchanged for
   existing fixtures. `unaccounted` is **clamped at 0 with a warning** when
   the log-derived components exceed the live nvidia-smi total (stale boot
   log / container restart) — a negative `other` is a staleness artifact,
   not memory.
2. **Service (`club3090_cockpit/services.py`)**: `CockpitData.vram_breakdown(container)`
   — `docker logs` through the injected read runner (same seam as
   `bootlog_solve`), temp file, parser `--json` subprocess. Honest
   `ok=False` failures.
3. **Panel (`app.py`, RailStatus)**: the estate card renders the split —
   aggregate across all GPUs by default; `[G]` cycles estate → per-GPU →
   back. The log parse is cached per serving container and re-runs on a
   600 s stride (or when the serving container changes); the **totals stay
   on the fast nvidia-smi poll**. Degrades to today's used/total card when
   there is no split (non-llama engine, nothing serving, parse failure).
   `other` is clamped and marked `≥`/`⚠`; the moe-cache multi-allocation
   warning is surfaced, not swallowed.

## Acceptance evidence

Parser `--json` on the live GLM+DFlash2 boot log:

```json
{"devices": [
  {"device": "CUDA0", "model": 3524, "kv": 1238, "state": 231,
   "compute": 5329, "pool": 11714, "used": 22938, "total": 24576,
   "unaccounted": 902},
  {"device": "CUDA1", "model": 3969, "kv": 1111, "state": 206,
   "compute": 5240, "pool": 9903, "used": 22992, "total": 24576,
   "unaccounted": 2563}],
 "warnings": ["2 moe-cache allocations logged per pool"]}
```

Panel (headless render, 40 cols):

```
=== AGGREGATE ===                      === CUDA1 (after [G]) ===
Estate                                 Estate
██████████ GPU0 0/0G                   ██████████ GPU0 0/0G
██████████ GPU1 0/0G                   ██████████ GPU1 0/0G
kv pool 61%                            kv pool 61%
VRAM split · estate                    VRAM split · CUDA1
  model    7G                           model    4G
  pool     21G                          pool     10G
  compute  10G                          compute  5G
  kv       2G                           kv       1G
  state    0G                           state    0G
  other    3G ⚠                         other    3G ⚠
  split from boot log · 0m old · [G] cycle
  ⚠ 2 moe-cache allocations logged per pool
```

(GPU bars read 0/0G only because the headless test has no live nvidia-smi;
in a real session they carry the live totals.)

## Suites

- Guard suite: **1 known master failure** — `test-litellm-generate.sh`
  ("incubating deepseek route lacks the '# status:' annotation") reproduces
  verbatim on pristine `origin/master` (EXIT=1, same message); this branch
  touches no litellm files. Everything else green, including the extended
  `test-vram-breakdown.sh` (12 assertions).
- Cockpit suite (worktree venv, `pytest`): **1124 passed, 1 skipped** —
  includes the 5 new `TestEstateVramSplit` tests (aggregate render, CUDA1
  drill-down, negative-`other` clamp, no-split degradation, worker+`[G]`
  wiring) and the negative-unaccounted parser cases in the guard test.

## §4 findings — CUDA ordinal ↔ nvidia-smi index — RESOLVED: option A implemented
**RESOLVED — option A implemented** on this branch (maintainer endorsed in
review): `_prepare_devices()` extracts `{ordinal: PCI bus id}` from
`llama_prepare_model_devices`, `_fetch_smi()` queries `pci.bus_id` as well,
and `_join_smi()` keys the readings by normalized bus id (domain-short vs
domain-long forms).  All-or-nothing, with option C as the automatic
fallback: if any logged ordinal cannot be resolved, the parser falls back
to index order and **says so in `warnings`** (the panel surfaces it).  Logs
without the prepare line keep today's index behaviour unchanged.

Covered by `test-vram-breakdown.sh`: a crossed-bus fixture (log buses
swapped vs nvidia-smi index order) proves an index-join would attribute
backwards while the bus join is correct; an unresolvable-bus fixture
proves the fallback is announced.

What the stack gives us today (verified on the live boot log):

```
llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090) (0000:01:00.0) - 23858 MiB free
llama_prepare_model_devices: using device CUDA1 (NVIDIA GeForce RTX 3090) (0000:02:00.0) - 13850 MiB free
```

The engine logs a **PCI bus ID per CUDA ordinal**. Options, with costs:

| option | how | cost / risk |
|---|---|---|
| **A. PCI-bus join in the parser** | parse `llama_prepare_model_devices` → `{ordinal: pci_bus_id}`; add `pci.bus_id` to the nvidia-smi query and key the smi lookup by bus id | ~15 lines + fixture lines in the parser. Exact under ANY `ESTATE_GPUS` reorder/subset. Needs the `llama_prepare_model_devices` line (present on the moe-cache engine; older/other engines fall back) |
| B. `ESTATE_GPUS` order mapping (panel-side) | read the compose's `device_ids` CSV; ordinal *i* ↔ i-th entry | needs compose discovery + parse in c3; **the runtime does not guarantee CSV order** (devices enumerate by bus order), so this can silently mis-attribute — the exact failure §4 warns about |
| C. Disable drill-down when unverifiable | if the log has no `prepare_model_devices` lines → aggregate-only, dim cue | zero wrongness; per-GPU mode unavailable on rigs whose logs lack the line |

**Question: may I implement A (with C as the automatic fallback when the log
has no PCI lines)?** It is a small parser change but touches the smi query —
beyond the `--json` scope you approved. Until then, the shipped panel keeps
the per-GPU drill-down **enabled but keyed by ordinal** (correct on this
rig's default `ESTATE_GPUS=0,1`), and the aggregate view is always correct.

## Deliberately left alone

- **Drafter sub-rows** (`└ drafter weights/KV/compute` from the issue's
  reference table): the parser's contract has no per-context attribution —
  drafter compute is #1171 defect 3 (open by instruction), and splitting
  drafter weights needs positional parsing the log does not label. The panel
  renders the six owned components; `other`/`⚠` carries what cannot be
  attributed (on DFlash2 boots that includes the drafter's 1,606 MiB compute).
- **mmproj row**: `clip_model_loader` emits no buffer-size line — not
  derivable; would need an engine-side log line (per the maintainer comment).
- **Pool slots / hit rates**: owned by `CAPTURE: EXPERT CACHE` (one owner per
  number). The panel surfaces the parser's moe-cache *warning* only.
- **`test-bench-capture.sh` failure on master** (#1137, reopened): pre-existing,
  reproduced on `origin/master`; not touched.
