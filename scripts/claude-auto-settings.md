# Claude Code Auto-Settings

> Switch club-3090 composes without touching `settings.json` again.

When you switch composes with `switch.sh`, every config changes the port **and** the model name. This tool auto-detects the running endpoint and updates your Claude Code `~/.claude/settings.json` so the CLI always points at the right model — zero manual editing.

## How it works

```
┌──────────────────────────────────────────────────────────────┐
│  Claude Code session starts                                  │
│       ↓                                                      │
│  SessionStart hook fires update-claude-settings.sh           │
│       ↓                                                      │
│  1. docker ps → finds running club-3090 container            │
│  2. Extract host port from port binding                      │
│  3. curl /v1/models → retries with backoff (up to 5x)       │
│     → picks most specific model ID                           │
│  4. Atomically writes ~/.claude/settings.json                │
│       ↓                                                      │
│  Claude Code continues, now pointing at the right endpoint   │
└──────────────────────────────────────────────────────────────┘
```

**Boot race:** The engine can take 10–20s to finish loading after `switch.sh`. The script retries the `/v1/models` query up to 5 times with increasing backoff (2s, 3s, 4s, 5s). If the endpoint still doesn't respond, it falls back to the container command line or container name — but won't overwrite last-good settings unless it has a valid model name.

Detection chain (first match wins):
1. **Live endpoint** — queries `/v1/models` on the detected port, picks the longest (most specific) model ID
2. **Container CLI** — extracts `--served-model-name` from the container command
3. **Container name** — derives from the Docker container name (e.g., `vllm-qwen36-27b-dual-nvfp4` → `qwen3.6-27b-nvfp4`)

## Prerequisites

- Docker running, with a club-3090 container active
- `curl`, `python3` in PATH
- Claude Code installed with `~/.claude/settings.json`

## Install

### Quick install (one-shot)

```bash
# 1. Ensure the scripts are in your club-3090 repo
cd /path/to/club-3090

# 2. Add the SessionStart hook to your Claude Code settings
#    Replace the path below if your repo lives elsewhere.
python3 - << 'EOF'
import json, os

settings_path = os.path.expanduser("~/.claude/settings.json")
with open(settings_path) as f:
    settings = json.load(f)

# Ensure env vars exist for local serving
if "env" not in settings:
    settings["env"] = {}
settings["env"].setdefault("ANTHROPIC_BASE_URL", "http://localhost:8000")
settings["env"].setdefault("ANTHROPIC_API_KEY", "dummy")
settings["env"].setdefault("ANTHROPIC_MODEL", "qwen3.6-27b")

# Add the hook — adjust the script path to match your repo location
if "hooks" not in settings:
    settings["hooks"] = {}

settings["hooks"]["SessionStart"] = [
    {
        "hooks": [
            {
                "type": "command",
                "command": "bash /path/to/club-3090/scripts/update-claude-settings.sh",
                "timeout": 30
            }
        ]
    }
]

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print(f"Installed hook in {settings_path}")
EOF
```

> **Important:** Replace `/path/to/club-3090` in the `command` string with the **absolute** path to your club-3090 checkout.

### Manual install

Edit `~/.claude/settings.json` and add a `hooks` block:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_MODEL": "qwen3.6-27b"
  },
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash /path/to/club-3090/scripts/update-claude-settings.sh",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## Usage

**Nothing.** That's the point.

1. Switch composes as usual:
   ```bash
   bash scripts/switch.sh vllm/dual
   ```
2. Start (or restart) Claude Code.
3. The hook fires, detects the new port and model, and updates settings automatically.

You'll see a one-liner in the hook output:
```
[claude-settings] URL: http://localhost:8077 → http://localhost:8010
[claude-settings] Model: qwen3.6-27b-nvfp4 → qwen3.6-27b-autoround
[claude-settings] Updated /home/paul/.claude/settings.json
```

Or if nothing changed:
```
[claude-settings] Already correct: http://localhost:8010 / qwen3.6-27b-autoround
```

## Manual run

You can invoke the script anytime to force an update:

```bash
bash /path/to/club-3090/scripts/update-claude-settings.sh
```

Useful if you switch composes mid-session and want Claude Code to pick up the change without restarting.

## Configuration

| Setting | Default | Notes |
|---|---|---|
| `SETTINGS_FILE` | `~/.claude/settings.json` | Hardcoded in the shell script; edit if you use a custom path |
| Hook timeout | `30` seconds | Covers retries while the engine boots + endpoint query + file write |
| Container name match | `vllm-`, `llamacpp-`, `sglang-`, `beellama-`, `ik-llama-`, `llama-cpp-` | Add prefixes to the `grep -E` line in the shell script if you use custom container names |

### Remote serving

If your model runs on another machine (not `localhost`), the script detects the binding IP from Docker. If you bind to `0.0.0.0`, it defaults to `localhost`. To serve on a specific interface, set `BIND_HOST` in your compose or `.env` — the script picks it up from the port binding.

## Troubleshooting

**"No club-3090 container running"** — No matching container was found. Check `docker ps` and ensure the container name starts with one of the recognized prefixes (`vllm-`, `llamacpp-`, etc.).

**"Could not parse port from binding"** — The port binding format was unexpected. Check with `docker ps --filter name=<container>` and verify it shows `IP:PORT->INT_PORT/protocol`.

**"Could not detect model name"** — The `/v1/models` endpoint didn't respond after 5 retries and fallbacks failed. The engine may still be booting. Wait a minute and run the script manually: `bash scripts/update-claude-settings.sh`.

**API error on first start, works on second** — This is normal if your engine takes >20s to boot (large models, cold load). The retry loop covers most cases, but if the engine is exceptionally slow, increase the hook `timeout` in `settings.json` or run the script manually once the engine is ready.

**Hook doesn't fire** — Verify the `hooks` block is in `~/.claude/settings.json` with valid JSON. The `command` path must be absolute.

## Files

| File | Purpose |
|---|---|
| `scripts/update-claude-settings.sh` | Shell entry point — detects container, port, model |
| `scripts/update-claude-settings.py` | Python helper — atomically writes `settings.json` |
| `README.md` | This file |