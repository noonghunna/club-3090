#!/usr/bin/env bash
# test-openwebui-gateway-key — Open WebUI's connection to the gateway gets the gateway's
# key (#1467).
#
# WHY THIS TEST EXISTS
# --------------------
# services/openwebui/docker-compose.yml gave the gateway connection the placeholder
# `sk-noauth`. The gateway refuses any key but its own (`400 No connected db.`), so a
# fresh Open WebUI listed no gateway model at all. The connection's key is now
# `${LITELLM_MASTER_KEY:-sk-litellm-master-key}` — the gateway's key from the settings,
# else its public default — and must reach compose the way the gateway's own does:
# through gpu-mode's per-call settings file (sudo strips the environment).
#   1. `docker compose config` renders the stored key into OPENAI_API_KEYS, paired with
#      the gateway URL; nothing stored → the public default; OWUI_OPENAI_API_KEYS
#      still overrides the whole list.
#   2. gpu-mode's start path (`gpu-mode upgrade` of a drifted open-webui) hands compose
#      a settings file that renders the same.
# Open WebUI v0.11 only SEEDS its database from this variable on a new volume; that is
# documented in the compose, not testable offline.
#
# Offline: `docker compose config` only renders; nothing is created. For leg 2,
# `docker` / `sudo` are PATH shims prepended INLINE: `compose … up` is turned into
# `compose … config` with the same --env-file, under a throwaway project name.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; fail=1; }
DEFAULT="sk-litellm-master-key"
KEY="test-key-$(printf '%s' "$$-$RANDOM" | md5sum | cut -c1-24)"
unset LITELLM_MASTER_KEY HF_TOKEN OWUI_OPENAI_API_KEYS OWUI_OPENAI_API_BASE_URLS   # a real token in this shell stays out of anything resolved here
if ! docker compose version >/dev/null 2>&1; then
  echo "test-openwebui-gateway-key: SKIP (docker compose not available)"; exit 0
fi
REAL_DOCKER="$(command -v docker)"
PROJECT="c3test-owui-$$-$RANDOM"

CFG="$T/cfg"; mkdir -p "$CFG"; printf 'LITELLM_MASTER_KEY=%s\n' "$KEY" > "$CFG/secrets.env"; chmod 600 "$CFG/secrets.env"
# render <dir> <env file> → "<OPENAI_API_BASE_URLS>\n<OPENAI_API_KEYS>"
render() {
  (cd "$1" && env -u LITELLM_MASTER_KEY docker compose -p "$PROJECT" --env-file "$2" -f docker-compose.yml config --format json 2>/dev/null) \
    | python3 -c 'import json,sys; e=json.load(sys.stdin)["services"]["open-webui"]["environment"]; print(e["OPENAI_API_BASE_URLS"]); print(e["OPENAI_API_KEYS"])' 2>/dev/null
}

# ── 1. the compose ───────────────────────────────────────────────────────────
echo "1. services/openwebui/docker-compose.yml"
envf="$(CLUB3090_CONFIG_DIR="$CFG" python3 "$ROOT/scripts/lib/club_config.py" compose-env-file --root "$T/no-repo")"
got="$(render "$ROOT/services/openwebui" "$envf")"; rm -f "$envf"
[[ "$got" == "http://host.docker.internal:4000/v1;http://host.docker.internal:8090/v1"$'\n'"$KEY;sk-noauth" ]] \
  && ok "a key stored only in secrets.env becomes the gateway connection's key (paired with :4000)" \
  || bad "rendered: ${got//$KEY/<key>}"
got="$(render "$ROOT/services/openwebui" /dev/null)"
[[ "$(sed -n 2p <<<"$got")" == "$DEFAULT;sk-noauth" ]] && ok "nothing stored → the gateway's public default (what the gateway itself falls back to)" \
  || bad "with nothing stored: $(sed -n 2p <<<"$got")"
printf 'OWUI_OPENAI_API_KEYS=a;b;c\nOWUI_OPENAI_API_BASE_URLS=http://x:1/v1;http://x:2/v1;http://x:3/v1\n' > "$CFG/club3090.env"
envf="$(CLUB3090_CONFIG_DIR="$CFG" python3 "$ROOT/scripts/lib/club_config.py" compose-env-file --root "$T/no-repo")"
got="$(render "$ROOT/services/openwebui" "$envf")"; rm -f "$envf" "$CFG/club3090.env"
[[ "$(sed -n 2p <<<"$got")" == "a;b;c" ]] && ok "OWUI_OPENAI_API_KEYS still overrides the whole list" || bad "override: $(sed -n 2p <<<"$got")"

# ── 2. through gpu-mode's start path ─────────────────────────────────────────
echo "2. gpu-mode (sudo docker compose + the per-call settings file)"
C="$T/club"; mkdir -p "$C/scripts/lib" "$C/services"
cp "$ROOT/scripts/gpu-mode.sh" "$C/scripts/"; cp "$ROOT/scripts/lib/club-config.sh" "$ROOT/scripts/lib/club_config.py" "$C/scripts/lib/"
cp -r "$ROOT/services/openwebui" "$C/services/"
mkdir -p "$T/bin"
cat > "$T/bin/sudo" <<'EOF'
#!/usr/bin/env bash
while [[ "${1:-}" == *=* ]]; do shift; done
exec "$@"
EOF
# docker: `compose … config` renders for real (throwaway project); `compose … up` is
# rendered instead of run, and its OPENAI_API_KEYS saved; open-webui is "running" on
# an older image, so `upgrade` restarts it.
cat > "$T/bin/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  info) exit 0 ;;
  compose)
    shift; args=()
    for a in "\$@"; do [[ "\$a" == up || "\$a" == -d ]] && continue; args+=("\$a"); done
    if [[ " \$* " == *" up "* ]]; then
      "$REAL_DOCKER" compose -p "$PROJECT" "\${args[@]}" config --format json \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["open-webui"]["environment"]["OPENAI_API_KEYS"])' > "\$STUB_OUT"
      exit 0
    fi
    exec "$REAL_DOCKER" compose -p "$PROJECT" "\$@" ;;
  inspect)
    case "\$4" in
      *Config.Image*)  echo ghcr.io/open-webui/open-webui:v0.10.0 ;;
      *State.Running*) echo true ;;
    esac ;;
  *) exit 0 ;;
esac
EOF
chmod +x "$T/bin/"*
for b in docker sudo; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done
out="$(CLUB3090_CONFIG_DIR="$CFG" CLUB3090_DIR="$C" STUB_OUT="$T/owui-keys" PATH="$T/bin:$PATH" bash "$C/scripts/gpu-mode.sh" upgrade 2>&1)"
[[ "$(cat "$T/owui-keys" 2>/dev/null)" == "$KEY;sk-noauth" ]] \
  && ok "gpu-mode starting Open WebUI hands compose the stored key for the gateway connection" \
  || bad "gpu-mode's open-webui start rendered OPENAI_API_KEYS='$(sed "s/$KEY/<key>/" "$T/owui-keys" 2>/dev/null)' — $(tail -3 <<<"$out")"
[[ "$out" != *"$KEY"* ]] || bad "gpu-mode printed the key"

[ "$fail" -eq 0 ] && echo "test-openwebui-gateway-key: ok (OPENAI_API_KEYS = the stored gateway key, else its default; override wins; delivered through gpu-mode's settings file)"
exit "$fail"
