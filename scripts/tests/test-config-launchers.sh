#!/usr/bin/env bash
# test-config-launchers — the launchers and tools read settings through the ONE loader
# (club-3090#1466, phases 1b-1 and 1b-2), driven through their real entry points.
#
# For each reader: a value only in club3090.env reaches it, and the shell beats it.
# club3090.env also outranks the repo .env, so on a checkout whose .env sets the same
# key (a maintainer's rig) the config value must still win. This test never writes
# the checkout's own .env — it only adds a temporary CLUB3090_CONFIG_DIR.
#
# ⚠️ launch.sh used to `source` .env, which beat the shell for every key unless
# MODEL_DIR was exported. The shell now wins everywhere; arm 2 pins that change on a
# key other than MODEL_DIR.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
C="$T/cfg"; mkdir -p "$C"
cat > "$C/club3090.env" <<'EOF'
MODEL_DIR=/from/club3090-env
IK_LLAMA_IMAGE=cfg/ik-llama:config
MODEL_SWITCH_PORT=9123
C3_ALLOW_CORE_PROMOTE=1
EOF
# A docker that answers nothing, so no launcher touches the real engine.
mkdir -p "$T/bin"; printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/docker"; chmod +x "$T/bin/docker"
# launch.sh must stop at model selection. If it ever got as far as delegating, the mock
# switch records it (and does nothing), and the last arm fails.
printf '#!/usr/bin/env bash\necho "$*" >> "%s"\nexit 1\n' "$T/switch-called" > "$T/bin/switch-mock"; chmod +x "$T/bin/switch-mock"
run() { env -u MODEL_DIR -u IK_LLAMA_IMAGE PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$C" "$@"; }

# 1. switch.sh
got="$(run timeout 60 bash scripts/switch.sh --list 2>&1 | command grep -m1 -oE '\[switch\] MODEL_DIR=.*')"
[[ "$got" == "[switch] MODEL_DIR=/from/club3090-env" ]] && ok "switch.sh reads MODEL_DIR from club3090.env" || bad "switch.sh: '$got'"
got="$(run MODEL_DIR=/from/shell timeout 60 bash scripts/switch.sh --list 2>&1 | command grep -m1 -oE '\[switch\] MODEL_DIR=.*')"
[[ "$got" == "[switch] MODEL_DIR=/from/shell" ]] && ok "switch.sh: the shell beats club3090.env" || bad "switch.sh shell-wins: '$got'"

# 2. launch.sh — stops at model selection (unknown model), before any docker work.
out="$(run SWITCH="$T/bin/switch-mock" COMPOSE_BIN=: timeout 60 bash scripts/launch.sh --no-preflight --model zzz-no-such-model 2>&1)"
command grep -qF 'not installed under /from/club3090-env.' <<<"$out" && ok "launch.sh reads MODEL_DIR from club3090.env" \
  || bad "launch.sh MODEL_DIR: $(command grep -m1 'not installed' <<<"$out")"
command grep -qF 'ik-llama image pinned: cfg/ik-llama:config' <<<"$out" && ok "launch.sh reads a non-MODEL_DIR key from club3090.env" \
  || bad "launch.sh image pin: $(command grep -m1 'image pinned' <<<"$out")"
out="$(run IK_LLAMA_IMAGE=shell/ik-llama:mine SWITCH="$T/bin/switch-mock" COMPOSE_BIN=: timeout 60 bash scripts/launch.sh --no-preflight --model zzz-no-such-model 2>&1)"
command grep -qF 'ik-llama image pinned: shell/ik-llama:mine' <<<"$out" \
  && ok "launch.sh: the shell beats saved settings for every key now, not only MODEL_DIR" \
  || bad "launch.sh shell-wins: $(command grep -m1 'image pinned' <<<"$out")"

# 3. A value that expected `source`-style expansion is flagged, not silently used.
X="$T/x"; mkdir -p "$X"; printf 'MODEL_DIR=$HOME/models\n' > "$X/club3090.env"
out="$(env -u MODEL_DIR PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$X" SWITCH="$T/bin/switch-mock" COMPOSE_BIN=: timeout 60 bash scripts/launch.sh --no-preflight --model zzz-no-such-model 2>&1)"
command grep -qE "WARN: MODEL_DIR \(from club3090.env\) contains '\\\$VAR'" <<<"$out" \
  && ok "launch.sh warns when a saved value expected shell expansion" || bad "no expansion warning: $(command grep -m1 -i warn <<<"$out")"

# 4. The Python tools.
got="$(run python3 -c 'import sys; sys.path.insert(0, "."); from scripts.lib.profiles import estate_cli as e; print(e.load_dotenv().get("MODEL_DIR"))')"
[[ "$got" == "/from/club3090-env" ]] && ok "estate_cli.load_dotenv reads club3090.env" || bad "estate_cli: '$got'"
got="$(run MODEL_DIR=/from/shell python3 -c 'import sys; sys.path.insert(0, "."); from scripts.lib.profiles import estate_cli as e; print(e.load_dotenv().get("MODEL_DIR"))')"
[[ "$got" == "/from/shell" ]] && ok "estate_cli: the shell wins" || bad "estate_cli shell-wins: '$got'"
got="$(run python3 -c 'import sys; sys.path.insert(0, "."); from scripts.lib.profiles import deriver as d; print(d._model_dir_from_env_or_dotenv())')"
[[ "$got" == "/from/club3090-env" ]] && ok "deriver reads MODEL_DIR from club3090.env" || bad "deriver: '$got'"
# promote.py / export_pr.py as DIRECT scripts: the import path that needs sys.path help.
out="$(run python3 scripts/lib/profiles/promote.py --spec-file "$T/no-such-spec.json" --root "$T" 2>&1)"
command grep -qF 'C3_ALLOW_CORE_PROMOTE read from club3090.env' <<<"$out" \
  && ok "promote.py (direct script) reads the gate from club3090.env and says so" \
  || bad "promote.py: $(command grep -m2 -E 'C3_ALLOW|Error|Traceback' <<<"$out")"
out="$(run python3 scripts/lib/profiles/export_pr.py --check --spec-file "$T/no-such-spec.json" --out "$T/out" 2>&1)"
command grep -qE 'ImportError|ModuleNotFoundError' <<<"$out" && bad "export_pr.py (direct script) failed to import the loader: $(command grep -m1 -E 'Error' <<<"$out")" \
  || ok "export_pr.py (direct script) imports the loader"

# 5. The model-switch server loads settings at import, before reading its constants.
got="$(run python3 -c 'import importlib.util as u; s = u.spec_from_file_location("srv", "tools/model-switch/server.py"); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.PORT)' 2>&1 | tail -1)"
[[ "$got" == "9123" ]] && ok "model-switch server takes MODEL_SWITCH_PORT from club3090.env (no EnvironmentFile needed)" \
  || bad "model-switch server PORT: '$got'"
command grep -q '^EnvironmentFile=' scripts/systemd/club3090-model-switch.service \
  && bad "the systemd unit still loads the repo .env itself" || ok "the systemd unit no longer reads the repo .env itself"

# 6. gpu-mode: `sudo docker compose` gets settings through a 0600 temp env file,
#    removed on exit. `service-images` is read-only (docker compose config, no sudo).
if docker info >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  G6="$T/g6"; mkdir -p "$G6" "$T/tmp"
  printf 'SPARK_DASHBOARD_IMAGE=cfg/spark-dashboard:from-config\n' > "$G6/club3090.env"
  out="$(env -u SPARK_DASHBOARD_IMAGE TMPDIR="$T/tmp" CLUB3090_CONFIG_DIR="$G6" timeout 180 bash scripts/gpu-mode.sh service-images 2>&1)"
  command grep -qF 'cfg/spark-dashboard:from-config' <<<"$out" \
    && ok "gpu-mode: a setting only in club3090.env reaches docker compose (service image pin)" \
    || bad "gpu-mode service-images didn't see the config value: $(command grep -m1 spark <<<"$out")"
  left="$(ls -A "$T/tmp" | command grep -E "^club3090-compose-" || true)"
  [[ -z "$left" ]] && ok "gpu-mode removes its temp env file on exit" || bad "gpu-mode left its env file: $left"
else
  ok "gpu-mode arm skipped (no docker access here)"
fi

# 7. comfyui-paths.sh (sourced by the studio scripts): MODEL_DIR, LANIP, HF_TOKEN only.
C7="$T/c7"; mkdir -p "$C7"
printf 'MODEL_DIR=/from/club3090-env\nLANIP=203.0.113.7\n' > "$C7/club3090.env"
printf 'HF_TOKEN=hf_from_secrets\n' > "$C7/secrets.env"
got="$(env -u MODEL_DIR -u LANIP -u HF_TOKEN -u C3_PATHS_NO_ENV CLUB3090_CONFIG_DIR="$C7" \
        bash -c '. services/comfyui/comfyui-paths.sh; printf "%s|%s|%s" "$MODEL_DIR" "$LANIP" "$HF_TOKEN"')"
[[ "$got" == "/from/club3090-env|203.0.113.7|hf_from_secrets" ]] \
  && ok "comfyui-paths.sh reads MODEL_DIR / LANIP from club3090.env and HF_TOKEN from secrets.env" \
  || bad "comfyui-paths.sh: '$got'"
got="$(env -u LANIP -u HF_TOKEN -u C3_PATHS_NO_ENV MODEL_DIR= CLUB3090_CONFIG_DIR="$C7" \
        bash -c '. services/comfyui/comfyui-paths.sh; printf "%s" "$MODEL_DIR"')"
[[ "$got" == "/from/club3090-env" ]] && ok "comfyui-paths.sh: an EMPTY exported MODEL_DIR still falls through to the settings (as before)" \
  || bad "comfyui-paths.sh empty MODEL_DIR: '$got'"
got="$(env -u MODEL_DIR -u LANIP -u HF_TOKEN C3_PATHS_NO_ENV=1 CLUB3090_CONFIG_DIR="$C7" \
        bash -c '. services/comfyui/comfyui-paths.sh; printf "%s|%s" "${LANIP:-}" "${HF_TOKEN:-}"')"
[[ "$got" == "|" ]] && ok "comfyui-paths.sh: C3_PATHS_NO_ENV=1 still skips the settings" || bad "C3_PATHS_NO_ENV: '$got'"

# 8. download_director.sh — an `hf` that does nothing, so it only reports its destination.
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/hf"; chmod +x "$T/bin/hf"
out="$(env -u MODEL_DIR PATH="$T/bin:$PATH" CLUB3090_CONFIG_DIR="$C7" timeout 60 bash services/comfyui/download_director.sh 2>&1)"
command grep -qF '→ /from/club3090-env/qwen3.5-4b-gguf/' <<<"$out" && ok "download_director.sh takes MODEL_DIR from club3090.env" \
  || bad "download_director.sh: $(command grep -m1 '→' <<<"$out")"

[[ -e "$T/switch-called" ]] && bad "launch.sh went past model selection and called switch: $(cat "$T/switch-called")" \
  || ok "launch.sh never got as far as calling switch (the mock was never invoked)"

[[ $fail -eq 0 ]] && echo "test-config-launchers: ok" || echo "test-config-launchers: FAIL"
exit $fail
