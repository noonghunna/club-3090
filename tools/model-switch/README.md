# model-switch — HTTP control for swapping the served model

**Status: 🧪 experimental** (opt-in; validate on your rig before relying on it).

A tiny stdlib HTTP service that wraps [`scripts/switch.sh`](../../scripts/switch.sh) so an
experiment harness can swap the running model programmatically instead of shelling out.
Only one model fits in VRAM at a time on a 1–2 GPU rig, so this is the "now serve model X,
tell me when it's ready" primitive.

It adds **no orchestration logic** — `switch.sh` stays the single source of truth (registry
lookup, down→up, readiness probe). This is the HTTP analogue of what `tools/serve-cockpit`
does as a TUI. **stdlib only, no pip installs.**

## Endpoints

| Method | Path | Auth | Returns |
|---|---|---|---|
| GET | `/healthz` | no | `{"ok": true}` |
| GET | `/status` | yes | `{"current_model","ready","port","container"}` |
| GET | `/models` | yes | `{"available":[{"slug","model","status","port"}, …]}` |
| POST | `/switch` | yes | blocks until ready → `{"ok","slug","model","took_s"}` |

`POST /switch` body accepts either an exact registry slug or a model id:
```json
{"slug": "vllm/gemma-31b-dual"}      // any slug from GET /models
{"model": "gemma-4-31b"}             // resolved to the model's curated default slug
```
Errors: `400` unknown/ambiguous slug-or-model · `401` bad token · `409` a switch is already
in progress · `500` `{ok:false, detail:<switch.sh log tail>}`.

## Run

```bash
# Foreground (dev):
python3 tools/model-switch/server.py

# Or as a systemd service (survives reboots, auto-restarts):
#   edit scripts/systemd/club3090-model-switch.service (User=, repo path), then:
sudo cp scripts/systemd/club3090-model-switch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now club3090-model-switch.service
```

## Config (env — systemd loads them from the repo-root `.env`)

| Var | Default | Purpose |
|---|---|---|
| `CLUB3090_API_TOKEN` | — | Control-endpoint bearer token. Falls back to `VLLM_API_KEY`. If neither is set, the endpoint is **unauthenticated** (loopback only). |
| `MODEL_SWITCH_PORT` | `8099` | Listen port. |
| `MODEL_SWITCH_BIND` | `127.0.0.1` | Bind address. Keep on loopback; expose via a VPN, not `0.0.0.0`. |
| `PORT` | slug default | Model http port used for the `/health` readiness probe. |
| `SWITCH_SCRIPT` | `scripts/switch.sh` | Overridable (used by the test to stub the switch). |

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

## Readiness note

The service probes the **unauthenticated `/health`** endpoint (not the auth-gated
`/v1/models`) when waiting for a switch, so it works whether or not the model is protected
with `VLLM_API_KEY`. It requires no change to `switch.sh`.
