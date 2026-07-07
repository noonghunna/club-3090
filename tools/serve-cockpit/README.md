# club-3090 serve cockpit (`c3`)

A **lazydocker-style terminal UI** for the club-3090 AI inference stack — the front door for
*discover → serve → operate → validate* on a consumer NVIDIA rig. Wraps the same registry, launchers,
and scripts you'd run by hand (`switch.sh`, `launch.sh`, `gpu-mode`, `health.sh`, the verify/bench
suite) behind one keyboard-driven screen.

> **Status:** working — read paths (catalog, estate, containers, health) are live; write paths
> (serve / scene-switch / downloads) are guarded by a confirm + a single-writer reconcile lease.

## What's inside

Two modes, shown as a tab bar; the producer mode is hidden in the lean view.

- **`1` Run & Operate** *(always shown)*
  - **Catalog** — the full registry of model variants; filter, inspect, and **serve** one (`⏎`).
  - **Orchestration** — live GPU cards, the `gpu-mode` scenes (incl. `ai-studio`), and supporting services; switch scene / stop.
  - **Containers** — running/stopped services with engine + port; drill into **Logs / Top / Config**, start a stopped one.
  - **Doctor** — "is it serving correctly?" — `health.sh` live + `verify` / `verify-full` reads, basic/full reports, and the power-cap sweep.
- **`2` Bring & Validate** *(producer lane — hidden in lean view)*
  - The add-a-model pipeline: **① Bring** (fit-check an HF repo) → **② Serve** (generate a compose + serve untested) → … → **⑤ Promote**.

Launch shows both modes by default; **`c3 --lean`** (or **`[C]`** in-app) gives the consumer view
(Run & Operate only).

## How to run

The cockpit depends on an in-repo sibling package, **`club3090-tui-core`** (`tools/tui-core/`), which
is **not on PyPI** — install both from the checkout.

**With [uv](https://docs.astral.sh/uv/) (recommended — one command; the local path is wired in `pyproject.toml`):**

```bash
uv pip install -e tools/serve-cockpit
c3
```

**With plain pip (install the core first, then the cockpit):**

```bash
pip install -e tools/tui-core
pip install -e tools/serve-cockpit
c3
```

Either way the launch is `c3` (or `python -m club3090_cockpit`). The app finds the repo root from its
own location; override with **`C3_REPO_ROOT=/path/to/club-3090`** if you installed it elsewhere.

**First run — set your Model Dir + HF token:** press **`S`** to open Settings, set your **Model Dir**
(where weights download to) and your **HuggingFace token** (needed for gated / private repos), then
**`Ctrl+S`** to save (`HF_HOME` is auto-derived under the model dir). Then hit **`r`** to browse the catalog.

**Keeping current:** the cockpit moves fast — after a `git pull`, **re-run the install**
(`uv pip install -e tools/serve-cockpit`) to pick up new deps (e.g. PyYAML) and UI changes.

## Keybindings

| Key | Action |
|-----|--------|
| `1` / `2` | Run & Operate · Bring & Validate |
| `↑ ↓ ← →` | move within / between the tab bar and content |
| `⏎` | primary action for the focused row (serve / start / download / confirm) |
| `k` | stop a service / cancel a download |
| `f` | force-start (experimental — skips the fit gate) |
| `r` | refresh the catalog (re-reads the registry) |
| `S` | settings — set Model Dir + HF token (`Ctrl+S` saves) |
| `n` | new cluster — Operate · Orchestration: compose a model + GPU set (fit-checked, gated) |
| `Y` | copy the focused context to the clipboard |
| `.` | toggle the left rail (full-width content) |
| `C` | toggle lean view (hide / restore the Bring & Validate mode) |
| `?` | help · `q` quit |

## Clusters (multiple models on one host)

A **cluster** = one model pinned to a chosen GPU set + port (an estate instance). The **Operate · Orchestration** tab shows a **cluster view** — each cluster with its GPUs stacked and a placement health badge (`✓ placed` / `⚠ PLACEMENT MISMATCH`, so you see where models *actually* landed) — and `[n]` opens a **New-cluster** modal (name · slug · GPU set, prefilled with the free GPUs). The create is fit-checked against the selected GPUs and routed through the confirm gate.

Same capability headless via the CLI: `bash scripts/cluster.sh create/list/status/up/down/rm` (`--json` on reads). Full guide → [`docs/CLUSTERS.md`](../../docs/CLUSTERS.md).

## Running tests

Fully headless — no TTY, GPU, Docker, or script calls (a conftest blocks any real spawn).

```bash
uv pip install -e "tools/serve-cockpit[dev]"   # or: pip install pytest pytest-asyncio
cd tools/serve-cockpit && pytest
```
