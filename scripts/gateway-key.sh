#!/usr/bin/env bash
# gateway-key.sh — give the LiteLLM gateway a key of its own, instead of the public
# default every club-3090 install shares (club-3090#1467).
#
#   bash scripts/gateway-key.sh status    # public default or your own key? (never prints it)
#   bash scripts/gateway-key.sh rotate    # store a new random key, print the next steps
#
# WHY — unless a key is stored, services/litellm/docker-compose.yml gives the gateway
# `sk-litellm-master-key`, the same on every install. The gateway listens on every
# interface, so anyone on your network who knows club-3090 can use your GPUs and the
# cloud routes in your config.local.yaml, which spend your own API keys.
#
# WHERE IT LIVES — LITELLM_MASTER_KEY in ~/.config/club-3090/secrets.env (0600;
# CLUB3090_CONFIG_DIR moves it), written by the settings writer
# (scripts/lib/club_config.py). gpu-mode hands it to docker compose. It is never
# printed; to paste it somewhere, read it from that file.
#
# WHAT A ROTATE CHANGES — nothing until the gateway is recreated: the running one
# keeps its key, so clients keep working. From then on every client still sending
# the old key gets `400 No connected db.`. This script restarts nothing: it prints
# the commands that recreate only the gateway (the way gpu-mode starts it) and
# re-run the omp / pi / Hermes setups whose `club` provider points at this rig. A
# key you pasted by hand (Claude Code's ANTHROPIC_API_KEY, an Open WebUI connection)
# you update by hand.
#
# CLUB3090_DIR picks the clone whose settings and gateway to use (default: this
# script's, as for gpu-mode); CLUB3090_GATEWAY_URL the gateway to probe
# (default http://127.0.0.1:4000).
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${CLUB3090_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
# shellcheck source=lib/club-config.sh
. "$SCRIPT_DIR/lib/club-config.sh"

KEY_NAME="LITELLM_MASTER_KEY"
DEFAULT_KEY="sk-litellm-master-key"          # the compose's fallback; public by definition
DEFAULT_GATEWAY="http://127.0.0.1:4000/v1"   # what omp/pi/hermes-setup.sh write by default
GATEWAY_URL="${CLUB3090_GATEWAY_URL:-http://127.0.0.1:4000}"
GATEWAY_URL="${GATEWAY_URL%/}"; GATEWAY_URL="${GATEWAY_URL%/v1}"
SECRETS="$(club_config_dir)/secrets.env"

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }
say()  { printf '%s\n' "$*"; }
warn() { printf '[gateway-key] %s\n' "$*" >&2; }

# The key as docker compose will get it — the value gpu-mode writes into its
# --env-file — as "<source><TAB><value>": the config file that sets it, or "shell"
# when this shell overrides a stored one. Empty when nothing stores a key. The
# value only ever lands in a variable; nothing here prints it.
_key_line() {
  club_config_resolve "$ROOT_DIR" | awk -F'\t' -v k="$KEY_NAME" '$1 == k { print $2 "\t" $3; exit }'
}

# HTTP status of GET /v1/models with <key>, 000 when nothing answers. The header
# goes in on stdin (-H @-), so the key is never on a command line.
_probe() {
  curl -s -o /dev/null -w '%{http_code}' -m 5 -H @- "$GATEWAY_URL/v1/models" <<<"Authorization: Bearer $1" 2>/dev/null || true
}

# One line per agent whose config holds the `club` provider OUR setup script wrote:
#   <agent><TAB><setup script><TAB><config file><TAB><gateway url, empty if unreadable>
_agent_setups() {
  local omp_yml="${PI_CODING_AGENT_DIR:-$HOME/.omp/agent}/models.yml"
  local pi_json="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/models.json"
  local hermes_yml="${HERMES_HOME:-$HOME/.hermes}/config.yaml" url
  python3 - "$omp_yml" "$pi_json" <<'PY'
import io, json, re, sys
def read(p):
    try:
        return io.open(p, encoding="utf-8").read()
    except OSError:
        return ""
omp, pi = sys.argv[1], sys.argv[2]
t = read(omp)
if ">>> club-3090 local models" in t and "<<< club-3090 local models" in t:
    block = t.split(">>> club-3090 local models", 1)[1].split("<<< club-3090 local models", 1)[0]
    m = re.search(r"^\s*baseUrl:\s*(\S+)", block, re.M)
    print(f"omp\tomp-setup.sh\t{omp}\t{m.group(1) if m else ''}")
t = read(pi)
if "scripts/pi-setup.sh" in t:
    try:
        url = json.loads(t)["providers"]["club"].get("baseUrl", "")
    except Exception:          # noqa: BLE001 — unreadable: its setup's default gateway
        url = ""
    print(f"pi\tpi-setup.sh\t{pi}\t{url}")
PY
  if [[ -f "$hermes_yml" ]] && command grep -qF 'club-3090 local models (scripts/hermes-setup.sh)' "$hermes_yml"; then
    url="$("${HERMES_BIN:-hermes}" config get providers.club.api 2>/dev/null || true)"
    [[ "$url" == http* ]] || url=""
    printf 'hermes\thermes-setup.sh\t%s\t%s\n' "$hermes_yml" "$url"
  fi
}

# Is <url> this rig's gateway, so this rig's key belongs in it? Unreadable counts
# as yes: the setup scripts default to it.
_is_local_gateway() {
  local url="${1:-}" host mine
  [[ -z "$url" ]] && return 0
  host="${url#*://}"; host="${host%%/*}"; host="${host%:*}"
  case "$host" in 127.0.0.1|localhost|"[::1]") return 0 ;; esac
  mine="${GATEWAY_URL#*://}"; mine="${mine%%/*}"; mine="${mine%:*}"
  [[ "$host" == "$mine" ]]
}

# The command that refreshes one agent's provider, keeping the gateway it points at.
_setup_cmd() {
  local script="$1" url="$2"
  if [[ -n "$url" && "$url" != "$DEFAULT_GATEWAY" ]]; then
    printf 'bash %s/scripts/%s --gateway %s' "$ROOT_DIR" "$script" "$url"
  else
    printf 'bash %s/scripts/%s' "$ROOT_DIR" "$script"
  fi
}

cmd_status() {
  local line src="" val="" code
  line="$(_key_line)"
  if [[ -n "$line" ]]; then src="${line%%$'\t'*}"; val="${line#*$'\t'}"; fi
  if [[ -z "$val" || "$val" == "$DEFAULT_KEY" ]]; then
    say "Gateway key: the PUBLIC DEFAULT, the same on every club-3090 install."
    say "  Anyone on your network can use the gateway, and the cloud routes in your config.local.yaml."
    say "  Give it a key of its own: bash $ROOT_DIR/scripts/gateway-key.sh rotate"
    val="$DEFAULT_KEY"
  else
    case "$src" in
      secrets.env) say "Gateway key: your own (per install), stored in $SECRETS." ;;
      shell)       say "Gateway key: your own, from LITELLM_MASTER_KEY in this shell (it overrides the stored one)." ;;
      club3090.env) say "Gateway key: your own, from club3090.env (the settings file, not the 0600 secrets file)."
                    say "  Move it: python3 $SCRIPT_DIR/lib/club_config.py unset LITELLM_MASTER_KEY, then bash $ROOT_DIR/scripts/gateway-key.sh rotate" ;;
      *)           say "Gateway key: your own, from $src. Its place is secrets.env — a rotate stores a new one there: bash $ROOT_DIR/scripts/gateway-key.sh rotate" ;;
    esac
  fi
  if [[ -n "${LITELLM_MASTER_KEY+x}" && "$src" != shell ]]; then
    say "  LITELLM_MASTER_KEY is also set in this shell: the agent setup scripts use it, gpu-mode (sudo) does not."
  fi
  code="$(_probe "$val")"
  case "$code" in
    200) say "Running gateway ($GATEWAY_URL): accepts it." ;;
    000) say "Running gateway ($GATEWAY_URL): not answering." ;;
    *)   say "Running gateway ($GATEWAY_URL): REFUSES it (HTTP $code) — it was started with another key; recreate it (bash $ROOT_DIR/scripts/gateway-key.sh rotate prints how)." ;;
  esac
  local agent script file url n=0
  while IFS=$'\t' read -r agent script file url; do
    [[ -n "$agent" ]] || continue
    n=$((n + 1))
    if _is_local_gateway "$url"; then
      say "Agent setup: $agent ($file) carries this gateway's key — refresh after a rotate: $(_setup_cmd "$script" "$url")"
    else
      say "Agent setup: $agent ($file) points at $url, not this rig's gateway."
    fi
  done < <(_agent_setups)
  [[ $n -gt 0 ]] || say "Agent setups: none found (omp, pi and Hermes setups write ~/.omp/agent/models.yml, ~/.pi/agent/models.json, ~/.hermes/config.yaml)."
  return 0
}

cmd_rotate() {
  local line src=""
  if [[ -n "${LITELLM_MASTER_KEY+x}" ]]; then
    warn "LITELLM_MASTER_KEY is set in this shell, and the shell wins over the stored key."
    warn "Run 'unset LITELLM_MASTER_KEY', then rotate again. Nothing was changed."
    return 1
  fi
  line="$(_key_line)"
  [[ -n "$line" ]] && src="${line%%$'\t'*}"
  if [[ "$src" == club3090.env ]]; then
    warn "LITELLM_MASTER_KEY is set in club3090.env, which wins over secrets.env, so a new key there would not take."
    warn "Remove it first (python3 $SCRIPT_DIR/lib/club_config.py unset LITELLM_MASTER_KEY), then rotate again. Nothing was changed."
    return 1
  fi
  # Generated and stored inside one python3: the key is never on a command line,
  # in another process's environment, or on stdout.
  python3 - "$SCRIPT_DIR/lib" "$KEY_NAME" <<'PY'
import os, secrets, stat, sys
sys.path.insert(0, sys.argv[1])
import club_config
path = club_config.set_values({sys.argv[2]: "sk-club-" + secrets.token_hex(16)}, "secrets")
# The writer creates secrets.env 0600 and keeps an existing file's mode; one made
# looser by hand is tightened, since it now holds this key.
if stat.S_IMODE(os.stat(path).st_mode) & 0o077:
    os.chmod(path, 0o600)
    print(f"[gateway-key] tightened {path} to 0600", file=sys.stderr)
PY
  line="$(_key_line)"
  if [[ "${line%%$'\t'*}" != secrets.env || "${line#*$'\t'}" == "$DEFAULT_KEY" ]]; then
    warn "wrote a new key to $SECRETS, but it is not the one in effect — check: bash $0 status"
    return 1
  fi
  say "[gateway-key] New gateway key stored in $SECRETS (0600). It is not shown; that file holds it."
  say ""
  say "The running gateway keeps its old key until it is recreated, so clients keep working until then."
  say "  1. Recreate ONLY the gateway on the new key, the way gpu-mode starts it (needs sudo):"
  say "       cd $ROOT_DIR"
  cat <<'EOF'
       bash scripts/lib/litellm-sync.sh --no-restart
       CLUB3090_COMPOSE_ENV_FILE="$(python3 scripts/lib/club_config.py compose-env-file --root .)"
       (cd services/litellm && sudo docker compose --env-file "$CLUB3090_COMPOSE_ENV_FILE" up -d); rm -f "$CLUB3090_COMPOSE_ENV_FILE"
     (Any gpu-mode mode that starts the gateway does the same, but also switches models.)
     Check it took: bash scripts/gateway-key.sh status   → "Running gateway …: accepts it."
EOF
  local agent script file url any=0
  while IFS=$'\t' read -r agent script file url; do
    [[ -n "$agent" ]] || continue
    if ! _is_local_gateway "$url"; then
      say "  -  $agent points at $url, not this rig's gateway: left out."
      continue
    fi
    [[ $any -eq 1 ]] || say "  2. Then re-run the agent setups that hold the old key (pi and Hermes read the gateway, so after step 1):"
    any=1
    say "       $(_setup_cmd "$script" "$url")"
  done < <(_agent_setups)
  [[ $any -eq 1 ]] || say "  2. (no omp / pi / Hermes setup found that needs re-running)"
  say "  3. By hand, wherever you pasted the old key: Claude Code (ANTHROPIC_API_KEY in its"
  say "     settings.json), an Open WebUI connection to :4000. The key is in $SECRETS."
  say "  Until they have it, those clients get '400 No connected db.' from the recreated gateway."
}

case "${1:-status}" in
  status|show-status) cmd_status ;;
  rotate)
    [[ $# -le 1 ]] || { warn "rotate takes no options"; exit 2; }
    cmd_rotate ;;
  -h|--help|help)     usage ;;
  *) warn "unknown command '$1' — status | rotate (see --help)"; exit 2 ;;
esac
