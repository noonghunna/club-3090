#!/usr/bin/env bash
# hermes-setup.sh — point Hermes Agent at this rig's local models, through the
# LiteLLM gateway.
#
#   bash scripts/hermes-setup.sh            # write/refresh the `club` provider, print the next steps
#   bash scripts/hermes-setup.sh --print    # show the provider, change nothing
#   bash scripts/hermes-setup.sh --gateway http://gateway-host:4000/v1   # a gateway on another box
#
# WHAT IT WRITES — `providers.club` in Hermes's config.yaml ($HERMES_HOME/config.yaml,
# ~/.hermes/config.yaml by default), through `hermes config set`, whose writer keeps
# the file's comments and layout. Hermes can't read a context window from the
# gateway, so the model list is a SNAPSHOT: every route the gateway serves, with the
# window it reports, plus `qwen3.8-27b` (the id every Qwen3.8 slug serves; 262,144
# when nothing is up). Re-run it after switching to a slug with a different window
# or model. A `club` provider this script didn't write is refused; the old config is
# backed up.
#
# WHAT IT DOES NOT WRITE — your default model or reasoning effort. Pick the club
# model with `hermes model`, per run with `hermes chat --provider custom:club -m
# qwen3.8-27b`, or in a session with `/model custom:club:qwen3.8-27b`.
# docs/CODING_AGENTS.md ("Hermes Agent — setup") has what was measured.
#
# THE KEY — the gateway's LITELLM_MASTER_KEY from your club-3090 settings
# (~/.config/club-3090/secrets.env; `bash scripts/gateway-key.sh rotate` makes one),
# else the public default every install shares. LITELLM_MASTER_KEY set in the shell
# wins (a gateway on another box). After a rotate, re-run this script.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PRINT=0
GATEWAY="http://127.0.0.1:4000/v1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --print) PRINT=1; shift ;;
    --gateway) GATEWAY="${2:?--gateway needs a URL}"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "hermes-setup: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

HERMES_BIN="${HERMES_BIN:-hermes}"      # the tests point this at a stub
MARK="club-3090 local models (scripts/hermes-setup.sh)"
# The key through the one settings loader (#1466, #1467) — never read out of the
# compose file, whose line is `${LITELLM_MASTER_KEY:-…}` now, not the key.
# shellcheck source=lib/club-config.sh
. "$ROOT_DIR/scripts/lib/club-config.sh"
KEY="$(club_config_get LITELLM_MASTER_KEY "$ROOT_DIR" || true)"
KEY="${KEY:-sk-litellm-master-key}"   # the compose's default, when nothing is stored

PROVIDER_JSON="$(python3 - "$GATEWAY" "$KEY" "$MARK" <<'PY'
import json, sys, urllib.request

gateway, key, mark = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
ALWAYS = "qwen3.8-27b"      # every Qwen3.8 slug serves it, so it is listed even when nothing is up
FALLBACK_WINDOW = 262144    # the dual-fast slugs' window, for an id the gateway isn't serving now

root = gateway[:-3] if gateway.endswith("/v1") else gateway
req = urllib.request.Request(root + "/model_group/info", headers={"Authorization": f"Bearer {key}"})
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.load(r).get("data", [])
except Exception as e:  # noqa: BLE001 — any failure means "no snapshot"
    print(f"[hermes-setup] ⚠️  couldn't read {root}/model_group/info ({e.__class__.__name__}); "
          f"writing {ALWAYS} with a {FALLBACK_WINDOW:,}-token window — re-run with the slug up.", file=sys.stderr)
    data = []
# routes that carry model_info were rendered from a serving engine; the rest (a
# config.local.yaml cloud route, say) aren't this rig's models
routes = {g["model_group"]: int(g["max_input_tokens"]) for g in data if g.get("model_group") and g.get("max_input_tokens")}
models = {ALWAYS: {"context_length": routes.get(ALWAYS, FALLBACK_WINDOW)}}
models.update({mid: {"context_length": w} for mid, w in sorted(routes.items()) if mid != ALWAYS})
print(json.dumps({
    "name": mark,
    "api": gateway,
    "api_key": key,
    # Hermes auto-detects otherwise; the gateway speaks chat completions for every route.
    "transport": "chat_completions",
    "default_model": ALWAYS,
    "models": models,
}))
PY
)"

if [[ $PRINT -eq 1 ]]; then
  python3 -c 'import json, sys; print(json.dumps({"providers": {"club": json.loads(sys.argv[1])}}, indent=2))' "$PROVIDER_JSON"
  exit 0
fi

if ! command -v "$HERMES_BIN" >/dev/null 2>&1; then
  echo "[hermes-setup] Hermes Agent is not on PATH. Install it (https://hermes-agent.nousresearch.com), or add this" >&2
  echo "               under \`providers:\` in ~/.hermes/config.yaml yourself (bash scripts/hermes-setup.sh --print)." >&2
  exit 1
fi

CONFIG="${HERMES_HOME:-$HOME/.hermes}/config.yaml"
existing="$("$HERMES_BIN" config get providers.club 2>&1 || true)"
if [[ "$existing" != "Config key not set"* ]]; then
  name="$("$HERMES_BIN" config get providers.club.name 2>&1 || true)"
  if [[ "$name" != "$MARK" ]]; then
    echo "[hermes-setup] $CONFIG already has a \`club\` provider this script didn't write —" >&2
    echo "               rename or remove it (hermes config unset providers.club), then re-run." >&2
    exit 1
  fi
  action="refreshed"
else
  action="added"
fi

if [[ -f "$CONFIG" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  cp -p "$CONFIG" "$CONFIG.bak-$stamp"          # -p keeps the mode: the file holds API keys
  echo "[hermes-setup] backed up $CONFIG -> $CONFIG.bak-$stamp"
fi
# `config set` replaces the whole mapping (checked against Hermes 0.21.5), so a model
# that dropped out of the snapshot goes with it.
"$HERMES_BIN" config set providers.club "$PROVIDER_JSON" >/dev/null
[[ "$("$HERMES_BIN" config get providers.club.name 2>&1 || true)" == "$MARK" ]] \
  || { echo "[hermes-setup] hermes config set did not take — check: $HERMES_BIN config get providers.club" >&2; exit 1; }
python3 - "$PROVIDER_JSON" "$action" "$CONFIG" <<'PY'
import json, sys
models = json.loads(sys.argv[1])["models"]
listed = ", ".join(f"{mid} ({cfg['context_length']:,} ctx)" for mid, cfg in models.items())
print(f"[hermes-setup] {sys.argv[2]} the `club` provider in {sys.argv[3]}: {listed}")
PY

cat <<EOF

Next:
  1. Serve a model:   bash scripts/switch.sh --force vllm/qwen38-27b-dual-fast   (experimental; the slug docs/CODING_AGENTS.md recommends for agents)
  2. Use it:          hermes chat --provider custom:club -m qwen3.8-27b
     or make it the default with \`hermes model\`; in a session: /model custom:club:qwen3.8-27b
     Effort:          /reasoning none|low|medium|high|xhigh (high and max run as xhigh, minimal as low)
  3. Re-run this script after switching to a slug with a different context window
     or model — Hermes keeps the snapshot it was given.

  Scripted:  hermes chat --provider custom:club -m qwen3.8-27b -q "..." --oneshot </dev/null
EOF
