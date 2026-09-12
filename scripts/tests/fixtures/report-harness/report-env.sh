#!/usr/bin/env bash
# Shared scaffolding for the report.sh stage-gating tests (#813, #830).
#
# Builds a throwaway repo root containing the REAL scripts/report.sh plus
# scripted stand-ins for the five stage scripts, boots a stub endpoint whose
# liveness the stubs can revoke mid-run, and runs report.sh against it. No GPU,
# no engine, no container.
#
# `pkill -f` / `pgrep -f` are deliberately NOT used: the pattern self-matches
# the test's own command line and has killed the invoking shell on this rig.
# The stub is always stopped by the PID we started.

REPORT_ENV_FIXTURES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOAK_FIXTURES="$(cd "${REPORT_ENV_FIXTURES}/../soak-harness" && pwd)"

# report_env_init <real-repo-root>
report_env_init() {
  REPORT_SRC_ROOT="$1"
  REPORT_ENV_DIR="$(mktemp -d)"
  REPORT_FAKE_ROOT="${REPORT_ENV_DIR}/repo"
  REPORT_ALIVE_FILE="${REPORT_ENV_DIR}/engine-alive"
  REPORT_TRACE="${REPORT_ENV_DIR}/stage-trace"
  REPORT_FAKE_BIN="${REPORT_ENV_DIR}/bin"
  mkdir -p "${REPORT_FAKE_ROOT}/scripts" "$REPORT_FAKE_BIN"

  # Neutralise the two host probes that would otherwise reach the rig's REAL
  # docker daemon and GPUs. `docker info` failing is report.sh's documented
  # "docker daemon unreachable — skipping" path, so the hardware sections
  # degrade cleanly and the test never touches a live container or endpoint.
  printf '#!/usr/bin/env bash\nexit 1\n' > "${REPORT_FAKE_BIN}/docker"
  printf '#!/usr/bin/env bash\nexit 1\n' > "${REPORT_FAKE_BIN}/nvidia-smi"
  chmod +x "${REPORT_FAKE_BIN}/docker" "${REPORT_FAKE_BIN}/nvidia-smi"

  cp "${REPORT_SRC_ROOT}/scripts/report.sh" "${REPORT_FAKE_ROOT}/scripts/report.sh"
  chmod +x "${REPORT_FAKE_ROOT}/scripts/report.sh"
  # report.sh sources scripts/lib/*.sh relative to its resolved repo root.
  ln -s "${REPORT_SRC_ROOT}/scripts/lib" "${REPORT_FAKE_ROOT}/scripts/lib"
  : > "$REPORT_ALIVE_FILE"
  : > "$REPORT_TRACE"
}

# report_nvsmi_stub — replace the hard-fail nvidia-smi stub with a scripted one.
# The script body is read from stdin. Additive: the default stub stays `exit 1`
# for every existing caller, so this only affects tests that opt in.
report_nvsmi_stub() {
  cat > "${REPORT_FAKE_BIN}/nvidia-smi"
  chmod +x "${REPORT_FAKE_BIN}/nvidia-smi"
}

report_env_cleanup() {
  report_stub_stop
  [[ -n "${REPORT_ENV_DIR:-}" ]] && rm -rf "$REPORT_ENV_DIR"
}

# report_stage_stub <script-name> <exit-code> [body-line ...]
# Writes a stand-in stage script that records that it ran, prints the given
# lines, and exits with the given code.
report_stage_stub() {
  local name="$1" rc="$2"; shift 2
  local path="${REPORT_FAKE_ROOT}/scripts/${name}"
  {
    echo '#!/usr/bin/env bash'
    echo "echo \"${name}\" >> \"${REPORT_TRACE}\""
    local line
    for line in "$@"; do
      printf 'printf %%s\\\\n %q\n' "$line"
    done
    echo "exit ${rc}"
  } > "$path"
  chmod +x "$path"
}

# Make a stage stub also revoke the endpoint's liveness — an engine that dies
# during that stage.
report_stage_kills_engine() {
  local name="$1"
  local path="${REPORT_FAKE_ROOT}/scripts/${name}"
  # Insert the revocation just before the trailing exit.
  local rc
  rc="$(tail -1 "$path")"
  sed -i '$d' "$path"
  {
    echo "rm -f \"${REPORT_ALIVE_FILE}\""
    echo "$rc"
  } >> "$path"
}

report_stub_start() {
  local port_file="${REPORT_ENV_DIR}/port.txt"
  local plan="${REPORT_ENV_DIR}/plan"
  printf 'ok\n' > "$plan"
  rm -f "$port_file"
  python3 "${SOAK_FIXTURES}/stub-endpoint.py" "$port_file" "$plan" \
    "${REPORT_ENV_DIR}/vram.txt" "$REPORT_ALIVE_FILE" \
    >"${REPORT_ENV_DIR}/stub.log" 2>&1 &
  REPORT_STUB_PID=$!
  local waited=0
  while [[ ! -s "$port_file" ]]; do
    sleep 0.05
    waited=$((waited + 1))
    if (( waited > 200 )); then
      echo "FAIL: report stub endpoint did not come up" >&2
      cat "${REPORT_ENV_DIR}/stub.log" >&2 || true
      return 1
    fi
  done
  REPORT_STUB_URL="http://127.0.0.1:$(cat "$port_file")"
}

report_stub_stop() {
  if [[ -n "${REPORT_STUB_PID:-}" ]] && kill -0 "$REPORT_STUB_PID" 2>/dev/null; then
    kill "$REPORT_STUB_PID" 2>/dev/null || true
    wait "$REPORT_STUB_PID" 2>/dev/null || true
  fi
  REPORT_STUB_PID=""
}

# report_run [args...] — sets REPORT_OUT and REPORT_RC.
# URL= is honoured by report.sh's endpoint resolver ahead of container probing,
# so no fake docker is needed for the liveness path.
report_run() {
  : > "$REPORT_TRACE"
  set +e
  REPORT_OUT="$(
    env PATH="${REPORT_FAKE_BIN}:$PATH" \
        URL="${REPORT_STUB_URL:-}" \
        HOME="${REPORT_ENV_DIR}/home" \
        REPORT_FULL_CALIBRATION=0 \
        bash "${REPORT_FAKE_ROOT}/scripts/report.sh" "$@" 2>&1
  )"
  REPORT_RC=$?
  set -e
}

# Like report_run but with NO endpoint hint at all — the "cannot probe" case.
report_run_unprobeable() {
  : > "$REPORT_TRACE"
  set +e
  REPORT_OUT="$(
    env -u URL -u ENDPOINT \
        PATH="${REPORT_FAKE_BIN}:$PATH" \
        HOME="${REPORT_ENV_DIR}/home" \
        REPORT_FULL_CALIBRATION=0 \
        bash "${REPORT_FAKE_ROOT}/scripts/report.sh" "$@" 2>&1
  )"
  REPORT_RC=$?
  set -e
}

# Did a stage actually execute?
report_stage_ran() {
  command grep -qx "$1" "$REPORT_TRACE"
}
