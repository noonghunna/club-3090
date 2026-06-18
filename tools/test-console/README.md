# club3090-test-console (c3t)

A lazydocker-style terminal UI for the club-3090 AI inference stack test suite.

## What it does

The TUI wraps the stack's existing test scripts (`bench.sh`, `verify-full.sh`, `quality-test.sh`, etc.) and:

1. **Auto-detects** the serving model + endpoint (no more wrong-port/wrong-MODEL headaches)
2. **Runs tests** with one keystroke — full pipeline or individual tests
3. **Streams live progress** with structured parsing (bench TPS bars, NIAH ladder, quality counters, soak gauges)
4. **Shows GPU stats** (VRAM, utilization, power draw/cap, temperature) in real-time

## Quick start

```bash
# From the repo root:
bash scripts/c3t

# Or install with uv:
cd tools/test-console
uv sync
uv run c3t
```

## Setup

```bash
cd tools/test-console

# Option A: uv (recommended)
uv sync

# Option B: pip + venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Keybindings

| Key | Action |
|-----|--------|
| `↑/↓` or `j/k` | Move in test menu |
| `Enter` | Run selected test (default config) |
| `c` | Configure selected test |
| `x` | Stop current run |
| `r` | Re-detect serving target |
| `m` | Manual target override |
| `f` | Toggle log follow/scroll-lock |
| `Tab` | Cycle pane focus |
| `?` | Help |
| `q` | Quit |

## Test catalog

| Test | Script | Duration |
|------|--------|----------|
| Smoke | `verify.sh` | ~15s |
| Functional | `verify-full.sh` | ~2min |
| Speed bench | `bench.sh` | ~5min |
| Stress / NIAH | `verify-stress.sh` | ~15min |
| Quality packs | `quality-test.sh` | 5-90min |
| Soak / stability | `soak-test.sh` | ~20min |
| ★ FULL rebench | `rebench-full.sh` | ~45min-4h |

## Architecture

```
tools/test-console/
├── pyproject.toml
├── README.md
├── club3090_test_console/
│   ├── __init__.py
│   ├── __main__.py      # Entry point
│   ├── app.py           # Main Textual App
│   ├── app.tcss         # Textual CSS
│   ├── detect.py        # Endpoint/model auto-detection
│   ├── parsers.py       # Output parsers per test
│   ├── runner.py        # Subprocess management
│   └── widgets/
│       ├── target_pane.py   # Target status display
│       ├── test_menu.py     # Test selection menu
│       └── live_pane.py     # Structured progress + log
└── tests/
    ├── test_parsers.py  # Parser unit tests (offline)
    └── test_detect.py   # Detection unit tests (offline)
```

## Development

```bash
# Run tests (no GPU needed)
cd tools/test-console
uv run pytest -v

# Run the TUI
uv run c3t
```

## Design decisions

- **Python + Textual** over Go+BubbleTea: reuses the stack's Python ecosystem and can import `compose_registry` directly
- **No script modifications**: the TUI is a pure wrapper — it spawns existing scripts and parses their output
- **No global installs**: uses `uv` / project-local `.venv` for isolation
- **XDG state**: run history persists under `~/.local/state/club3090-test-console/`, never in the repo tree
- **Headless-testable**: all parsers and detection logic are unit-testable without a GPU
