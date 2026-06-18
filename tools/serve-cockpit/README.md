# club3090 serve cockpit (`c3`)

**Status: 🧪 Experimental**

A lazydocker-style TUI for the club-3090 AI inference stack — the front door for
discover → serve → manage on a consumer NVIDIA rig.

## Phase 1 scope (this package)

This is the **walking skeleton**: every navigation node is present and renders
something so layout, density, and keybindings can be signed off before the
plumbing work begins.  The only real data source wired in Phase 1 is the
Discover → Catalog tab, which is populated from the live registry via
`registry_variant_rows` (read-only, no side effects).  Everything else
(Serve, Estate/Containers, Validate, and all row/scene/container actions)
is a static placeholder stub — it will be wired in Phase 3.

## How to run

Install into the repo environment (editable):

```bash
cd tools/serve-cockpit
pip install -e .
```

Then launch:

```bash
c3
# or equivalently:
python -m club3090_cockpit
```

The app resolves the repo root automatically from `__file__` location.
Override with `C3_REPO_ROOT=/path/to/club-3090` if installed elsewhere.

## Keybindings

| Key | Action |
|-----|--------|
| `1` | Discover mode |
| `2` | Serve mode |
| `3` | Estate mode |
| `4` | Validate mode |
| `r` | Refresh catalog (re-reads registry only) |
| `⏎` | Primary action (no-op in Phase 1 — pops notice) |
| `?` | Help |
| `q` | Quit |

## Running tests

```bash
cd tools/serve-cockpit
pip install -e ".[dev]"
# or: pip install pytest pytest-asyncio
pytest
```

Tests are fully headless: no TTY, no GPU, no Docker, no script calls.
