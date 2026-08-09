# club3090-env — point Claude Code at the running endpoint

> Switch club-3090 composes without touching `settings.json` again.

When you switch composes with `switch.sh`, every config changes the port **and** the model name. This tool auto-detects the running endpoint and prints `export` lines so your shell always points Claude Code at the right model — zero manual editing.

## How it works

```
┌──────────────────────────────────────────────────────────────┐
│  User runs: eval "$(bash scripts/club3090-env.sh)" │
│       ↓                                                      │
│  1. docker ps → finds running club-3090 container            │
│  2. Extract host port from port binding                      │
│  3. curl /v1/models → retries with backoff (up to 5x)       │
│     → picks most specific model ID                           │
│  4. Prints export lines to stdout                             │
│       ↓                                                      │
│  eval applies them → ANTHROPIC_BASE_URL + ANTHROPIC_MODEL    │
│       ↓                                                      │
│  Claude Code starts → connects to the right endpoint         │
└──────────────────────────────────────────────────────────────┘
```

**Boot race:** The engine can take 10–20s to finish loading after `switch.sh`. The script retries the `/v1/models` query up to 5 times with increasing backoff (2s, 3s, 4s, 5s). If the endpoint still doesn't respond, it falls back to the container command line or container name.

Detection chain (first match wins):
1. **Live endpoint** — queries `/v1/models` on the detected port, picks the longest (most specific) model ID
2. **Container CLI** — extracts `--served-model-name` from the container command
3. **Container name** — derives from the Docker container name (e.g., `vllm-qwen36-27b-dual-nvfp4` → `qwen3.6-27b-nvfp4`)

## Prerequisites

- Docker running, with a club-3090 container active
- `curl` in PATH

## Install

### One-time: add an alias to your shell profile

Add this line to `~/.bashrc` or `~/.zshrc`:

```bash
alias club3090-env='eval "$(bash /path/to/club-3090/scripts/club3090-env.sh)"'
```

> Replace `/path/to/club-3090` with the **absolute** path to your club-3090 checkout.

That's it. No `settings.json` editing. No hooks. No Python dependency for the write path.

### Alternative: inline in shell profile

If you prefer the detection to run automatically on every new terminal, put this directly in your shell profile instead of an alias:

```bash
eval "$(bash /path/to/club-3090/scripts/club3090-env.sh)" 2>/dev/null || true
```

The `2>/dev/null || true` suppresses the "no container running" message on terminals where nothing is booted yet.

## Usage

**Switch compose, source env, start Claude Code:**

```bash
bash scripts/switch.sh vllm/dual
club3090-env      # prints export lines → eval applies them
claude            # connects to the right port + model
```

Output you'll see:
```
[claude-settings] URL: http://localhost:8077 → http://localhost:8010
[claude-settings] Model: qwen3.6-27b-nvfp4 → qwen3.6-27b-autoround
```

If nothing changed:
```
[claude-settings] Already correct: http://localhost:8010 / qwen3.6-27b-autoround
```

**Switch mid-session:**
```bash
bash scripts/switch.sh vllm/single
club3090-env      # picks up the new port + model
# No need to restart Claude Code if it was launched from this shell
# (env vars propagate to child processes)
```

**Dry-run (see what would be set without applying):**
```bash
bash scripts/club3090-env.sh
# Shows [claude-settings] lines + export lines
# Pipe to grep to see just the diagnostics:
bash scripts/club3090-env.sh | grep '^\[claude-settings\]'
```

## Configuration

| Setting | Default | Notes |
|---|---|---|
| Hook timeout | `30` seconds | Covers retries while the engine boots + endpoint query |
| Container name match | `vllm-`, `llamacpp-`, `sglang-`, `beellama-`, `ik-llama-`, `llama-cpp-` | Add prefixes to the `grep -E` line in the shell script if you use custom container names |

### Remote serving

If your model runs on another machine (not `localhost`), the script detects the binding IP from Docker. If you bind to `0.0.0.0`, it defaults to `localhost`. To serve on a specific interface, set `BIND_HOST` in your compose or `.env` — the script picks it up from the port binding.

## Troubleshooting

**"No club-3090 container running"** — No matching container was found. Check `docker ps` and ensure the container name starts with one of the recognized prefixes (`vllm-`, `llamacpp-`, etc.).

**"Could not parse port from binding"** — The port binding format was unexpected. Check with `docker ps --filter name=<container>` and verify it shows `IP:PORT->INT_PORT/protocol`.

**"Could not detect model name"** — The `/v1/models` endpoint didn't respond after 5 retries and fallbacks failed. The engine may still be booting. Wait a minute and run the script manually: `bash scripts/club3090-env.sh`.

**API error on first start, works on second** — This is normal if your engine takes >20s to boot (large models, cold load). The retry loop covers most cases, but if the engine is exceptionally slow, run the script manually once the engine is ready.

## Files

| File | Purpose |
|---|---|
| `scripts/club3090-env.sh` | Shell entry point — detects container, port, model; emits export lines |
| `club3090-env.md` | This file |