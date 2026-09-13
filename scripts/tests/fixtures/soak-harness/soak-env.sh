#!/usr/bin/env bash
# Shared scaffolding for the soak-harness tests (#809, #829).
#
# Boots the scriptable SSE stub endpoint, puts a fake nvidia-smi on PATH whose
# reading is driven by a file the stub writes, and runs scripts/soak-test.sh
# against it in host mode (CONTAINER=none). No GPU, no docker, no engine.
#
# Source this from a test, then call:
#   soak_env_init                 — create the sandbox (sets SOAK_ENV_DIR)
#   soak_stub_start <plan> [E=V]  — boot the stub (sets SOAK_STUB_URL / _PID);
#                                   extra KEY=VAL go to the stub process, e.g.
#                                   STUB_TPOT_TPS=123.456 to serve /metrics
#   soak_stub_stop                — stop it (by PID; never pkill -f, see below)
#   soak_stub_docker <log-file>   — put a fake `docker` on PATH whose `logs`
#                                   prints <log-file> on STDERR (#1268)
#   soak_run <out-dir> [env...]   — run soak-test.sh, capture stdout+rc
#
# `pkill -f` / `pgrep -f` are deliberately NOT used anywhere here: the pattern
# self-matches the test's own command line and has killed the invoking shell on
# this rig. The stub is always stopped by the PID we started.

# shellcheck disable=SC2034
SOAK_ENV_FIXTURES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

soak_env_init() {
  SOAK_ENV_DIR="$(mktemp -d)"
  SOAK_ENV_BIN="${SOAK_ENV_DIR}/bin"
  SOAK_VRAM_FILE="${SOAK_ENV_DIR}/vram.txt"
  mkdir -p "$SOAK_ENV_BIN"
  echo "20000" > "$SOAK_VRAM_FILE"

  # Fake nvidia-smi. soak-test.sh calls it three ways:
  #   --query-gpu=memory.used                       (vram_mib)
  #   --query-gpu=index,memory.used,utilization.gpu (append_gpu_snapshot)
  #   --query-gpu=index,name,...                    (capture_state)
  # All read the same scripted VRAM value, so a test can make the "GPU" report
  # a healthy figure during a live session and a corpse figure after a death.
  cat > "${SOAK_ENV_BIN}/nvidia-smi" <<SMI
#!/usr/bin/env bash
v="\$(cat "${SOAK_VRAM_FILE}" 2>/dev/null || echo 0)"
for a in "\$@"; do
  case "\$a" in
    --query-gpu=memory.used) echo "\$v"; exit 0 ;;
    --query-gpu=index,memory.used,utilization.gpu) echo "0, \$v, 42"; exit 0 ;;
    --query-gpu=index,name,memory.used*) echo "0, StubGPU, \$v, 24576, 42, 210.0, 60"; exit 0 ;;
  esac
done
echo "\$v"
SMI
  chmod +x "${SOAK_ENV_BIN}/nvidia-smi"
}

soak_env_cleanup() {
  soak_stub_stop
  [[ -n "${SOAK_ENV_DIR:-}" ]] && rm -rf "$SOAK_ENV_DIR"
}

soak_stub_start() {
  local plan="$1"; shift
  local port_file="${SOAK_ENV_DIR}/port.txt"
  rm -f "$port_file"
  env "$@" python3 "${SOAK_ENV_FIXTURES}/stub-endpoint.py" "$port_file" "$plan" "$SOAK_VRAM_FILE" \
    >"${SOAK_ENV_DIR}/stub.log" 2>&1 &
  SOAK_STUB_PID=$!
  local waited=0
  while [[ ! -s "$port_file" ]]; do
    sleep 0.05
    waited=$((waited + 1))
    if (( waited > 200 )); then
      echo "FAIL: stub endpoint did not come up" >&2
      cat "${SOAK_ENV_DIR}/stub.log" >&2 || true
      return 1
    fi
  done
  SOAK_STUB_URL="http://127.0.0.1:$(cat "$port_file")"
}

soak_stub_stop() {
  if [[ -n "${SOAK_STUB_PID:-}" ]] && kill -0 "$SOAK_STUB_PID" 2>/dev/null; then
    kill "$SOAK_STUB_PID" 2>/dev/null || true
    wait "$SOAK_STUB_PID" 2>/dev/null || true
  fi
  SOAK_STUB_PID=""
}

# soak_stub_docker <log-file> — fake `docker` for the engine-log counter path.
#
# Two things it exists to prove, both of which have burned this repo before:
#   1. `docker logs` output must be read from STDERR as well as stdout — every
#      engine here logs to stderr, so a helper reading only stdout scores a live
#      counter as "never fired". This stub prints ONLY on stderr.
#   2. the harness must bound the scrape to the turn it is measuring. Every
#      invocation's argv is appended to ${SOAK_ENV_DIR}/docker-argv.log so a
#      test can assert the --since window was actually passed.
soak_stub_docker() {
  local log_file="$1"
  : > "${SOAK_ENV_DIR}/docker-argv.log"
  cat > "${SOAK_ENV_BIN}/docker" <<DOCKER
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "${SOAK_ENV_DIR}/docker-argv.log"
case "\$1" in
  logs) cat "${log_file}" >&2 ;;
  inspect)
    for a in "\$@"; do
      [[ "\$a" == "{{.State.Running}}" ]] && { echo true; exit 0; }
    done
    exit 0 ;;
  stats) echo '{}' ;;
  *) : ;;
esac
exit 0
DOCKER
  chmod +x "${SOAK_ENV_BIN}/docker"
}

# soak_run <script-root> <out-dir> [KEY=VAL ...]
# Sets SOAK_OUT (combined stdout+stderr) and SOAK_RC.
soak_run() {
  local root="$1"; shift
  local out_dir="$1"; shift
  set +e
  SOAK_OUT="$(
    env PATH="${SOAK_ENV_BIN}:$PATH" \
        CONTAINER=none \
        ENDPOINT="$SOAK_STUB_URL" \
        MODEL=stub-model \
        SOAK_OUTPUT="$out_dir" \
        SOAK_MODE=fresh \
        SOAK_SESSIONS="${SOAK_SESSIONS:-3}" \
        SOAK_TURNS="${SOAK_TURNS:-5}" \
        SOAK_REQ_TIMEOUT_S=20 \
        SOAK_TIMEOUT_S=300 \
        SOAK_NO_CHAT_TEMPLATE_KWARGS=1 \
        "$@" \
        bash "${root}/scripts/soak-test.sh" 2>&1
  )"
  SOAK_RC=$?
  set -e
}
