# model-switch — HTTP control for swapping the served model

**Status: 🧪 experimental** (opt-in; validate on your rig before relying on it).

A tiny stdlib HTTP service that wraps [`scripts/switch.sh`](../../scripts/switch.sh) so an
experiment harness can swap the running model programmatically instead of shelling out —
and, because much of the catalog is experimental, **keep the desired model alive**. Only one
model fits in VRAM at a time on a 1–2 GPU rig, so this is the "now serve model X, keep it
healthy, tell me when it's ready" primitive.

It adds **no orchestration logic** — `switch.sh` stays the single source of truth (registry
lookup, down→up, readiness probe). This is the HTTP analogue of what `tools/serve-cockpit`
does as a TUI. **stdlib only, no pip installs.**

## What it does beyond a raw switch
- **Hardware-aware listing** — `/models` hides slugs that can't fit this GPU count (e.g.
  `multi4` on a 2-GPU host) and tombstoned statuses; `?all=1` shows everything.
- **Explicit-consent force** — a non-production or oversized slug requires `{"force": true}`
  (mirrors `switch.sh --force`). See the caution below.
- **Rollback** — if a switch fails, the previously-healthy model is restored.
- **Self-healing watchdog** — re-launches a crashed/wedged desired model; on a crash-loop it
  tears the model down and marks the service `degraded` instead of thrashing forever.

## Endpoints

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/healthz` | no | `{"ok": true}` |
| GET | `/` | no | **self-discovery manifest** — every route, method, auth policy, and body shape (point an agent here) |
| GET | `/status` | yes | current/desired slug, `docker_state`, `degraded`, `watchdog{…}` |
| GET | `/models` | yes | `{"host_topology","gpu_count","available":[{slug,…,gpu_eligible,requires_force,recommended}]}` |
| POST | `/switch` | yes | blocks until ready → `{"ok","slug","model","took_s"}`; rolls back on failure |
| POST | `/heal` | yes | recover to a slug (`{"slug"\|"model"}`) or re-launch the current desired model |
| POST | `/pull` | yes | **download** a model's weights + companions (async; poll `GET /status .download`) |
| POST | `/down` | yes | `switch.sh --down` + stand the watchdog down |

> This table is human-facing convenience. The **authoritative** endpoint list is the `ROUTES`
> table in `server.py`, served unauthenticated at `GET /` — so a client/agent can learn the whole
> API in one call, and it can't drift from the code. On `/models`, `recommended` = the slug is
> `production` or production-with-`caveats` (vs experimental/preview/incubating/upstream-gated/
> deprecated); `status` is the exact lifecycle, `requires_force` the switching-consent signal.

`POST /switch` / `/heal` body accepts an exact registry slug or a model id, plus optional force:
```json
{"slug": "vllm/gemma-31b-dual"}          // any slug from GET /models
{"model": "gemma-4-31b"}                 // resolved to the model's curated default slug
{"slug": "vllm/qwen-27b-multi-fast", "force": true}   // required for non-prod / oversized slugs
```
Errors: `400` unknown/ambiguous, or **force required without `"force": true`** ·
`401` bad token · `409` a transition is already in progress · `500`
`{ok:false, detail:<switch.sh log tail>}` (with `rolled_back_to` / `rollback_failed`).

### ⚠️ `force` disables safety checks
`{"force": true}` sets `FORCE=1` for `switch.sh`, which bypasses its free-VRAM, SM, and
hardware-fit preflights — not just the status gate. Only force a slug you know fits. The
authorization is **persisted**, so the watchdog re-launches a force-authorized model the same
way; an *externally* CLI-switched experimental model is adopted but **not** auto-forced (heal
it via `POST /heal {"slug":…,"force":true}` if you want that).

## Downloading weights (`/pull`)
`/switch` to a model whose weights aren't on disk returns **`409 {"error":"weights_missing",
"pull":"POST /pull {...}"}`** (the same check fires for watchdog heals — a missing-weights model
degrades instead of thrashing). Fetch them with `/pull`, which wraps `scripts/setup.sh` (core
weights **+** every `weights_companions` — mmproj / DFlash the compose mounts):

```bash
curl -s -H "$AUTH" -XPOST $BASE/pull -d '{"slug":"ik-llama/ornith9b-single"}'   # → 202, async
curl -s -H "$AUTH" $BASE/status | jq .download    # {state: downloading|ready|error, last_line, ...}
```
- **Async & coarse:** returns `202` immediately; poll `GET /status .download`. `200 {state:ready,
  already:true}` if already present; `507` if the disk hasn't room.
- **`{model}` resolves only a *functional curated default*** — experimental/non-default catalog
  entries (e.g. Ornith) must be pulled by **`{"slug": …}`** (same rule as `/switch`).
- **Runs concurrently** with a serving model (disk/network, not GPU); a service restart interrupts
  it — re-`POST /pull` resumes (`setup.sh` idempotent + HF resume). One download at a time (`409`).

### Where weights land — `MODEL_DIR`
Everything (`/pull`, the weights pre-check, `/status.model_dir`) uses **`MODEL_DIR`**, the same
knob `setup.sh`/`switch.sh`/the composes read. Precedence: shell/systemd env `MODEL_DIR` >
repo-root `.env` `MODEL_DIR=` > `<repo>/models-cache` (default). Set it once in `.env`
(e.g. `MODEL_DIR=/mnt/ssd/models`) so the service, downloads, and serving all agree. `HF_HOME` is
pinned to `$MODEL_DIR/.cache/huggingface` so HF staging lands on the same disk.

## Self-healing model
The watchdog polls the desired model's `/health`. If it stays down past `HEAL_GRACE_S` (or
`BOOT_GRACE_S` while first booting) it re-launches it. Repeated failures spend a rolling
budget (`MAX_HEAL_FAILURES` within `HEAL_BUDGET_WINDOW_S`, persisted across restarts); once
spent, the service `--down`s the model and goes `degraded` — clear it with `POST /heal`. A
docker-daemon outage reads as `docker_state:"unknown"` and **never** triggers healing.

## Run the rig

The control API is the entry point — it drives the model **and** keeps it alive. Two steps:
start the API, then tell it which model to serve. All config comes from the repo-root `.env`
(`VLLM_API_KEY`, `PORT`, `MODEL_DIR`); you never hand-type env vars.

**Dev / no systemd** — the `control-api.sh` wrapper loads `.env` and manages a pidfile + log:
```bash
bash scripts/control-api.sh start          # launch the API (detached, watchdog on)
TOKEN=$(grep -E '^VLLM_API_KEY=' .env | cut -d= -f2)
curl -s --max-time 420 -H "Authorization: Bearer $TOKEN" \
     -XPOST localhost:8099/switch -d '{"slug":"vllm/dual"}'   # bring up the model (~3 min)
bash scripts/control-api.sh status         # UP + /status
# control-api.sh {start|stop|restart|status|logs}
```

**Production — systemd** (survives reboots; the desired model is persisted, so the watchdog
restores it on boot):
```bash
sed -e "s|/opt/club-3090|$PWD|g" -e "s|User=youruser|User=$USER|" \
    scripts/systemd/club3090-model-switch.service | sudo tee /etc/systemd/system/club3090-model-switch.service
sudo systemctl daemon-reload
sudo systemctl enable --now club3090-model-switch.service
# then bring the model up once (persisted; watchdog keeps/restores it thereafter):
curl -s --max-time 420 -H "Authorization: Bearer $VLLM_API_KEY" \
     -XPOST localhost:8099/switch -d '{"slug":"vllm/dual"}'
```

**From another device** (over a VPN — do not expose the port directly): point the base at the
tailnet HTTPS front, e.g. `BASE=https://<host>.ts.net:8443` (`tailscale serve --https=8443 8099`).

> `switch.sh`'s readiness probe defaults to the unauthenticated `/health`, so the CLI
> (`bash scripts/switch.sh vllm/dual`) also works with `VLLM_API_KEY` set — but the API path
> above is preferred: one entry point, and the watchdog then keeps the model up.

## Config (env — systemd loads them from the repo-root `.env`)

| Var | Default | Purpose |
|---|---|---|
| `CLUB3090_API_TOKEN` | — | Control-endpoint bearer token. Falls back to `VLLM_API_KEY`. If neither is set, the endpoint is **unauthenticated** (loopback only). |
| `MODEL_SWITCH_PORT` | `8099` | Listen port. |
| `MODEL_SWITCH_BIND` | `127.0.0.1` | Bind address. Keep on loopback; expose via a VPN, not `0.0.0.0`. |
| `MODEL_SWITCH_WATCHDOG` | `1` | Background self-healing on/off. |
| `MODEL_SWITCH_WATCH_INTERVAL_S` | `15` | Watchdog poll period. |
| `MODEL_SWITCH_HEAL_GRACE_S` | `60` | Down time before healing a previously-healthy model. |
| `MODEL_SWITCH_BOOT_GRACE_S` | `300` | Unready time allowed while a model is booting. |
| `MODEL_SWITCH_STABILITY_WINDOW_S` | `300` | Continuous health before the heal budget resets. |
| `MODEL_SWITCH_HEAL_BUDGET_WINDOW_S` | `600` | Rolling window for the crash-loop budget. |
| `MODEL_SWITCH_MAX_HEAL_FAILURES` | `3` | Heals allowed in-window before degrading. |
| `MODEL_SWITCH_STATE_DIR` | `$STATE_DIRECTORY`/XDG | Where the desired-model state persists. |
| `MODEL_SWITCH_GPU_COUNT` | `nvidia-smi -L` | Override the detected GPU count. |
| `PORT` | slug default | Model http port used for the `/health` readiness probe. |
| `SWITCH_SCRIPT` / `DOCKER_BIN` | `scripts/switch.sh` / `docker` | Overridable (used by the test to stub them). |

## Example

```bash
TOKEN=...  # your CLUB3090_API_TOKEN (or VLLM_API_KEY)
curl -s -H "Authorization: Bearer $TOKEN" localhost:8099/status
curl -s -XPOST -H "Authorization: Bearer $TOKEN" localhost:8099/switch -d '{"model":"gemma-4-31b"}'
# → blocks ~1-2 min, then {"ok":true,"slug":"vllm/gemma-31b-dual","model":"gemma-4-31b","took_s":97.3}
```

## Security

Bind to loopback (default) and set a token. To reach it from other devices, front it with a
VPN — e.g. Tailscale on its own HTTPS port: `tailscale serve --https=8443 8099`. Do **not**
expose the port directly to the internet; a model-switch endpoint is a control plane.
