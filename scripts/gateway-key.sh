#!/usr/bin/env bash
# gateway-key.sh — give the LiteLLM gateway a key of its own, instead of the public
# default every club-3090 install shares (club-3090#1467).
#
#   bash scripts/gateway-key.sh status          # public default or your own key? (never prints it)
#   bash scripts/gateway-key.sh rotate          # store a new random key, print the next steps
#   bash scripts/gateway-key.sh rotate --apply  # store it AND do the steps: recreate the gateway,
#                                               #   re-run the omp / pi / Hermes setups
#   bash scripts/gateway-key.sh init            # first run only: a key of its own on a FRESH
#                                               #   install, else nothing (setup.sh runs it)
#
# WHY — unless a key is stored, services/litellm/docker-compose.yml gives the gateway
# `sk-litellm-master-key`, the same on every install. The gateway listens on every
# interface, so anyone on your network who knows club-3090 can use your GPUs and the
# cloud routes in your config.local.yaml, which spend your own API keys.
#
# WHERE IT LIVES — LITELLM_MASTER_KEY in ~/.config/club-3090/secrets.env (0600;
# CLUB3090_CONFIG_DIR moves it), written by the settings writer
# (scripts/lib/club_config.py). gpu-mode hands it to docker compose: the gateway gets
# it, and so does a NEW Open WebUI volume's gateway connection. It is never printed;
# to paste it somewhere, read it from that file.
#
# WHAT A ROTATE CHANGES — nothing until the gateway is recreated: the running one
# keeps its key, so clients keep working. From then on every client still sending
# the old key gets `400 No connected db.`. Plain `rotate` restarts nothing and prints
# the steps; `rotate --apply` does them: `gpu-mode gateway` (recreates ONLY the
# gateway, starting it if it was stopped; needs sudo), then the omp / pi / Hermes
# setups whose `club` provider points at this rig. A key you pasted by hand (Claude
# Code's ANTHROPIC_API_KEY, an existing Open WebUI connection) you update by hand.
#
# FIRST RUN (`init`) — stores a new key ONLY when nothing shows this machine has used
# the gateway, so no client can be holding the current one. It keeps the current
# key, and says why, when ANY of these holds:
#   - LITELLM_MASTER_KEY is set in the shell, club3090.env, secrets.env or the repo
#     .env (even empty);
#   - services/litellm/config.runtime.yaml exists in this checkout (gpu-mode and
#     switch.sh render it whenever they start or sync the gateway; `gpu-mode off`
#     removes the container but not this file);
#   - Docker has a `litellm` or `open-webui` container (running or stopped), or an
#     Open WebUI data volume (its gateway connection keeps the key it was made with);
#   - Docker is installed but can't be asked (not running, or no permission without
#     sudo, which init never uses): no container or volume can be ruled out. Docker
#     not installed at all counts as neither;
#   - something answers at the gateway's address (CLUB3090_GATEWAY_URL);
#   - an omp, pi or Hermes config holds the provider our setup script wrote.
# A key pasted by hand (Claude Code) is not looked for: every gpu-mode scene that starts
# the gateway also starts Open WebUI, and its volume outlives `gpu-mode off`.
# It never prints the key and never changes an existing one; `rotate` is the way to
# a new key on an existing install.
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
# Container names from services/litellm and services/openwebui docker-compose.yml.
GATEWAY_CONTAINERS=(litellm open-webui)

# The header comment above (line 2 up to the first non-comment line).
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
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

# The setup script and arguments that refresh one agent's provider, keeping the
# gateway it points at — one word per line, for `mapfile`.
_setup_argv() {
  local script="$1" url="$2"
  printf '%s\n' "$ROOT_DIR/scripts/$script"
  [[ -n "$url" && "$url" != "$DEFAULT_GATEWAY" ]] && printf '%s\n' --gateway "$url"
  return 0
}
# …and the same as a command line to print.
_setup_cmd() {
  local argv
  mapfile -t argv < <(_setup_argv "$1" "$2")
  printf 'bash %s' "${argv[*]}"
}

# Generate a key and store it in secrets.env. Generated and stored inside one
# python3: the key is never on a command line, in another process's environment,
# or on stdout. Fails with one line (no traceback) when the file can't be written.
_store_new_key() {
  python3 - "$SCRIPT_DIR/lib" "$KEY_NAME" <<'PY'
import os, secrets, stat, sys
sys.path.insert(0, sys.argv[1])
import club_config
try:
    path = club_config.set_values({sys.argv[2]: "sk-club-" + secrets.token_hex(16)}, "secrets")
    # The writer creates secrets.env 0600 and keeps an existing file's mode; one made
    # looser by hand is tightened, since it now holds this key.
    if stat.S_IMODE(os.stat(path).st_mode) & 0o077:
        os.chmod(path, 0o600)
        print(f"[gateway-key] tightened {path} to 0600", file=sys.stderr)
except (OSError, club_config.ConfigError) as e:
    print(f"[gateway-key] could not store a key in {club_config.target_path('secrets')}: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

# 0 when the key in effect is one this script just stored in secrets.env.
_new_key_in_effect() {
  local line
  line="$(_key_line)"
  [[ "${line%%$'\t'*}" == secrets.env && -n "${line#*$'\t'}" && "${line#*$'\t'}" != "$DEFAULT_KEY" ]]
}

cmd_status() {
  local line src="" val="" code
  line="$(_key_line)"
  if [[ -n "$line" ]]; then src="${line%%$'\t'*}"; val="${line#*$'\t'}"; fi
  if [[ -z "$val" || "$val" == "$DEFAULT_KEY" ]]; then
    say "Gateway key: the PUBLIC DEFAULT, the same on every club-3090 install."
    say "  Anyone on your network can use the gateway, and the cloud routes in your config.local.yaml."
    say "  Give it a key of its own: bash $ROOT_DIR/scripts/gateway-key.sh rotate --apply"
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
    *)   say "Running gateway ($GATEWAY_URL): REFUSES it (HTTP $code) — it was started with another key; recreate it: bash $ROOT_DIR/scripts/gpu-mode.sh gateway" ;;
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

# What only you can update: clients that hold the key because you pasted it.
_manual_steps() {
  say "  By hand, wherever the old key was pasted (the new one is the LITELLM_MASTER_KEY= line of $SECRETS):"
  say "    - Claude Code: ANTHROPIC_API_KEY in its settings.json (docs/CODING_AGENTS.md, Claude Code)."
  say "    - Open WebUI: its connection to the gateway keeps the key it was created with."
  say "      Admin → Settings → Connections → http://host.docker.internal:4000/v1 → set the key → Save."
  say "      (Only a NEW Open WebUI volume takes the stored key by itself.)"
  say "  Until they have it, those clients get '400 No connected db.' from the recreated gateway."
}

cmd_rotate() {
  local apply="$1" line src=""
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
  _store_new_key || return 1
  if ! _new_key_in_effect; then
    warn "wrote a new key to $SECRETS, but it is not the one in effect — check: bash $0 status"
    return 1
  fi
  say "[gateway-key] New gateway key stored in $SECRETS (0600). It is not shown; that file holds it."
  say ""
  if [[ "$apply" == 1 ]]; then
    _apply
    return
  fi
  say "The running gateway keeps its old key until it is recreated, so clients keep working until then."
  say "  (bash $ROOT_DIR/scripts/gateway-key.sh rotate --apply does steps 1 and 2 for you.)"
  say "  1. Recreate ONLY the gateway on the new key (needs sudo; no model is started or stopped):"
  say "       bash $ROOT_DIR/scripts/gpu-mode.sh gateway"
  say "     It waits for the gateway and checks it took (bash $ROOT_DIR/scripts/gateway-key.sh status"
  say "     → \"Running gateway …: accepts it.\")."
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
  say "  3."
  _manual_steps
}

# rotate --apply, after the key is stored: recreate the gateway through gpu-mode (the
# one start path that hands compose the settings), then refresh the agent setups on
# this rig's gateway — after, because pi and Hermes read its routes with the key.
_apply() {
  local agent script file url argv rc out failed=() done_=() left=()
  say "── 1. Recreating the gateway on the new key: bash $ROOT_DIR/scripts/gpu-mode.sh gateway (needs sudo)"
  if ! CLUB3090_DIR="$ROOT_DIR" bash "$ROOT_DIR/scripts/gpu-mode.sh" gateway; then
    say ""
    warn "the gateway was NOT recreated on the new key (see above). The key is stored; nothing else was changed."
    warn "Fix that, then: bash $ROOT_DIR/scripts/gpu-mode.sh gateway — and re-run the agent setups: bash $0 status lists them."
    return 1
  fi
  say ""
  say "── 2. Refreshing the agent setups on this rig's gateway"
  while IFS=$'\t' read -r agent script file url; do
    [[ -n "$agent" ]] || continue
    if ! _is_local_gateway "$url"; then
      say "  -  $agent points at $url, not this rig's gateway: left out."
      left+=("$agent")
      continue
    fi
    mapfile -t argv < <(_setup_argv "$script" "$url")
    rc=0; out="$(bash "${argv[@]}" 2>&1)" || rc=$?
    if [[ $rc -eq 0 ]]; then
      say "  ✓ $agent: $(_setup_cmd "$script" "$url")"
      done_+=("$agent")
    else
      say "  ✗ $agent: $(_setup_cmd "$script" "$url") failed (exit $rc):"
      printf '%s\n' "$out" | sed 's/^/      /'
      failed+=("$agent")
    fi
  done < <(_agent_setups)
  [[ ${#done_[@]} -gt 0 || ${#failed[@]} -gt 0 || ${#left[@]} -gt 0 ]] \
    || say "  (no omp / pi / Hermes setup found that needs re-running)"
  say ""
  say "── 3. Left for you"
  _manual_steps
  if [[ ${#failed[@]} -gt 0 ]]; then
    warn "setup(s) that failed still hold the old key: ${failed[*]} — fix and re-run them (commands above)."
    return 1
  fi
  return 0
}

# The first sign that this machine has already used the gateway (see FIRST RUN in the
# header), as one line to print; exit 1 when there is none. Ordered cheapest first;
# Docker is asked before any agent or Claude Code file is read.
_existing_install() {
  local line src="" val name names vols agent script file url
  if [[ -n "${LITELLM_MASTER_KEY+x}" ]]; then
    say "LITELLM_MASTER_KEY is set in this shell — that is the key; nothing generated."
    return 0
  fi
  line="$(_key_line)"
  if [[ -n "$line" ]]; then
    src="${line%%$'\t'*}"; val="${line#*$'\t'}"
    if [[ -z "$val" || "$val" == "$DEFAULT_KEY" ]]; then
      say "LITELLM_MASTER_KEY is set in $src, to the public default — kept. A key of its own: bash $ROOT_DIR/scripts/gateway-key.sh rotate --apply"
    else
      say "a gateway key is already stored ($src) — nothing to do."
    fi
    return 0
  fi
  if [[ -e "$ROOT_DIR/services/litellm/config.runtime.yaml" ]]; then
    say "the gateway has run from this checkout before (services/litellm/config.runtime.yaml exists), so clients may hold its public default key — kept."
    return 0
  fi
  if command -v docker >/dev/null 2>&1; then
    if ! docker info >/dev/null 2>&1; then
      say "Docker can't be asked whether a gateway already exists (not running, or no permission) — no key generated."
      return 0
    fi
    names="$(docker ps -a --format '{{.Names}}' 2>/dev/null)" || {
      say "Docker can't list its containers — no key generated."; return 0; }
    for name in "${GATEWAY_CONTAINERS[@]}"; do
      if command grep -qxF "$name" <<<"$names"; then
        say "a '$name' container exists, so clients may hold the gateway's public default key — kept."
        return 0
      fi
    done
    vols="$(docker volume ls -q 2>/dev/null)" || {
      say "Docker can't list its volumes — no key generated."; return 0; }
    name="$(command grep -m1 -E '(^|_)open-webui-data$' <<<"$vols" || true)"
    if [[ -n "$name" ]]; then
      say "Open WebUI's data volume ($name) exists; its gateway connection keeps the key it was made with — kept."
      return 0
    fi
  fi
  # Something already answering where the gateway lives (one not started by Docker).
  if [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$GATEWAY_URL/health/liveliness" 2>/dev/null || true)" != 000 ]]; then
    say "something already answers at $GATEWAY_URL — kept."
    return 0
  fi
  while IFS=$'\t' read -r agent script file url; do
    [[ -n "$agent" ]] || continue
    say "$agent's config ($file) already holds a gateway key (from $script) — kept."
    return 0
  done < <(_agent_setups)
  return 1
}

cmd_init() {
  local why
  if why="$(_existing_install)"; then
    say "[gateway-key] $why"
    case "$why" in
      *"already stored"*|*"set in this shell"*|*"to the public default"*) ;;
      *) say "[gateway-key]   A key of its own for the gateway: bash $ROOT_DIR/scripts/gateway-key.sh rotate --apply" ;;
    esac
    return 0
  fi
  _store_new_key || return 1
  if ! _new_key_in_effect; then
    warn "wrote a key to $SECRETS, but it is not the one in effect — check: bash $0 status"
    return 1
  fi
  say "[gateway-key] Fresh install: the gateway gets a key of its own, stored in $SECRETS (0600; not shown)."
  say "[gateway-key]   gpu-mode hands it to the gateway and to a new Open WebUI; the omp / pi / Hermes setups read it."
  say "[gateway-key]   Claude Code needs it pasted in (docs/CODING_AGENTS.md). Check: bash $ROOT_DIR/scripts/gateway-key.sh status"
}

case "${1:-status}" in
  status|show-status) cmd_status ;;
  rotate)
    case "${2:-}" in
      "")      [[ $# -le 1 ]] || { warn "rotate takes only --apply"; exit 2; }; cmd_rotate 0 ;;
      --apply) [[ $# -le 2 ]] || { warn "rotate takes only --apply"; exit 2; }; cmd_rotate 1 ;;
      *)       warn "rotate takes only --apply (got '$2')"; exit 2 ;;
    esac ;;
  init)
    [[ $# -le 1 ]] || { warn "init takes no options"; exit 2; }
    cmd_init ;;
  -h|--help|help)     usage ;;
  *) warn "unknown command '$1' — status | rotate [--apply] | init (see --help)"; exit 2 ;;
esac
