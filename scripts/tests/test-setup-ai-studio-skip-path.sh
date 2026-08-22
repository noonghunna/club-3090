#!/usr/bin/env bash
# setup-ai-studio.sh skip-path regression (#715 gap 5).
#
# The invariant: SKIP_BUILD / SKIP_DOWNLOAD skip ONLY the build / download —
# the bring-up (gpu-mode ai-studio) and the rest of the flow MUST still run.
# @MoppelMat's #686 install "printed the two skip lines and produced nothing";
# whatever the mechanism on his checkout (gaps 1+4 compounding is the leading
# read), this pins the invariant so a future refactor can't reintroduce it.
#
# Also exercises #715 gap 1 for free: the run pre-creates the studio bind-mount
# dirs USER-OWNED under a tmp MODEL_DIR before any (stubbed) docker call.
#
# Everything external is stubbed: docker / nvidia-smi via PATH shims, the
# bring-up via the GPU_MODE_BIN hook. No container, no GPU, no .env writes
# (LANIP + MODEL_DIR pinned via env; C3 paths derive under the tmp dir).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }

# --- PATH shims --------------------------------------------------------------
mkdir -p "$TMP/bin"
cat > "$TMP/bin/docker" <<'SH'
#!/usr/bin/env bash
# minimal docker stub for the setup-ai-studio skip-path test
case "$1" in
  compose) exit 0 ;;                      # `docker compose version`
  info)    exit 0 ;;
  ps)
    # both invocations: the port-gate names+ports format and the -a names+status check
    if [[ "$*" == *'{{.Names}} {{.Ports}}'* ]]; then
      echo "open-webui 0.0.0.0:8080->8080/tcp"
    else
      printf 'open-webui\tUp 2 minutes\n'
    fi
    exit 0 ;;
  *) exit 0 ;;
esac
SH
cat > "$TMP/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
if [[ "${1:-}" == "-L" ]]; then
  echo "GPU 0: STUB RTX 3090 (UUID: GPU-stub-0)"
  echo "GPU 1: STUB RTX 3090 (UUID: GPU-stub-1)"
  exit 0
fi
exit 0
SH
chmod +x "$TMP/bin/docker" "$TMP/bin/nvidia-smi"

# --- gpu-mode stub (the GPU_MODE_BIN testability hook) -----------------------
cat > "$TMP/fake-gpu-mode.sh" <<SH
#!/usr/bin/env bash
echo "FAKE-GPU-MODE \$*" >> "$TMP/gpu-mode-calls"
exit 0
SH
chmod +x "$TMP/fake-gpu-mode.sh"

# --- run: both skips set → the bring-up must still happen --------------------
out="$(PATH="$TMP/bin:$PATH" \
  SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_PIPE=1 SKIP_OWUI_WIRING=1 SKIP_DISK_CHECK=1 \
  ASSUME_YES=1 LANIP=127.0.0.1 MODEL_DIR="$TMP/models" C3_PATHS_NO_ENV=1 \
  GPU_MODE_BIN="$TMP/fake-gpu-mode.sh" \
  bash "$ROOT_DIR/scripts/setup-ai-studio.sh" 2>&1)" || fail "setup exited non-zero under skip flags:
$out"

# the two skip lines printed (the flags took effect) …
grep -q "SKIP_BUILD set"    <<<"$out" || fail "missing the SKIP_BUILD skip line"
grep -q "SKIP_DOWNLOAD set" <<<"$out" || fail "missing the SKIP_DOWNLOAD skip line"
# … AND the bring-up still ran (#715 gap 5 — the invariant)
grep -q "\[3/4\] Starting the studio" <<<"$out" || fail "bring-up step banner missing — skip flags removed functionality:
$out"
[ -f "$TMP/gpu-mode-calls" ] || fail "gpu-mode was never invoked under skip flags"
grep -q "FAKE-GPU-MODE ai-studio" "$TMP/gpu-mode-calls" || fail "gpu-mode not called with ai-studio: $(cat "$TMP/gpu-mode-calls")"

# gap 1 side-assert: the studio bind-mount dirs were pre-created USER-OWNED
for d in ComfyUI models input output user pip-cache; do
  [ -d "$TMP/comfyui/$d" ] || fail "bind-mount dir not pre-created: \$COMFYUI_ROOT/$d"
  [ -w "$TMP/comfyui/$d" ] || fail "pre-created dir not writable: \$COMFYUI_ROOT/$d"
done
[ -d "$TMP/models" ] || fail "MODEL_DIR not pre-created"

# --- 4b gateway-first wiring (openwebui → LiteLLM collapse follow-up) --------
# The OWUI_REGISTER_BIN / OWUI_UNREGISTER_BIN hooks let us assert exactly which
# connections a default run wires: :4000 gateway + :8090 director registered,
# the pre-collapse per-port direct connections unregistered, :4000 NEVER
# unregistered, and an OWUI_OPENAI_API_BASE_URLS override left untouched.
cat > "$TMP/fake-owui-register.sh" <<SH
#!/usr/bin/env bash
echo "\$*" >> "$TMP/owui-register-calls"
exit 0
SH
cat > "$TMP/fake-owui-unregister.sh" <<SH
#!/usr/bin/env bash
echo "\$*" >> "$TMP/owui-unregister-calls"
exit 0
SH
chmod +x "$TMP/fake-owui-register.sh" "$TMP/fake-owui-unregister.sh"

# Scenario A: default run → gateway-first registration output.
out="$(PATH="$TMP/bin:$PATH" \
  SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_PIPE=1 SKIP_DISK_CHECK=1 \
  ASSUME_YES=1 LANIP=127.0.0.1 MODEL_DIR="$TMP/models" C3_PATHS_NO_ENV=1 \
  GPU_MODE_BIN="$TMP/fake-gpu-mode.sh" \
  OWUI_REGISTER_BIN="$TMP/fake-owui-register.sh" \
  OWUI_UNREGISTER_BIN="$TMP/fake-owui-unregister.sh" \
  bash "$ROOT_DIR/scripts/setup-ai-studio.sh" 2>&1)" || fail "setup exited non-zero without SKIP_OWUI_WIRING:
$out"

grep -q "gateway-first" <<<"$out" || fail "wiring banner missing:
$out"
[ -f "$TMP/owui-register-calls" ] || fail "owui-register was never invoked"
grep -qx "4000" "$TMP/owui-register-calls" || fail ":4000 gateway not registered: $(cat "$TMP/owui-register-calls")"
grep -qx "8090" "$TMP/owui-register-calls" || fail ":8090 director not registered: $(cat "$TMP/owui-register-calls")"
[ ! -e "$TMP/owui-unregister-calls" ] && fail "no unregisters happened" || true
for port in 8010 8051 8032 8038 8199; do
  grep -qx "$port" "$TMP/owui-unregister-calls" || fail "per-port connection :$port not dropped: $(cat "$TMP/owui-unregister-calls")"
done
grep -qx "4000" "$TMP/owui-unregister-calls" && fail ":4000 was unregistered — setup must not fight the gateway topology"
echo "  ✓ default wiring registers :4000 + :8090 and drops legacy per-port connections"

# Scenario B: OWUI_OPENAI_API_BASE_URLS override → setup stays out of the way.
rm -f "$TMP/owui-register-calls" "$TMP/owui-unregister-calls"
out="$(PATH="$TMP/bin:$PATH" \
  SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_PIPE=1 SKIP_DISK_CHECK=1 \
  ASSUME_YES=1 LANIP=127.0.0.1 MODEL_DIR="$TMP/models" C3_PATHS_NO_ENV=1 \
  GPU_MODE_BIN="$TMP/fake-gpu-mode.sh" \
  OWUI_REGISTER_BIN="$TMP/fake-owui-register.sh" \
  OWUI_UNREGISTER_BIN="$TMP/fake-owui-unregister.sh" \
  OWUI_OPENAI_API_BASE_URLS="http://host.docker.internal:8010/v1" \
  bash "$ROOT_DIR/scripts/setup-ai-studio.sh" 2>&1)" || fail "setup exited non-zero under the override:
$out"

grep -q "OWUI_OPENAI_API_BASE_URLS set" <<<"$out" || fail "override notice missing:
$out"
[ ! -e "$TMP/owui-register-calls" ] || fail "override run still registered connections: $(cat "$TMP/owui-register-calls")"
[ ! -e "$TMP/owui-unregister-calls" ] || fail "override run still unregistered connections: $(cat "$TMP/owui-unregister-calls")"
echo "  ✓ OWUI_OPENAI_API_BASE_URLS override leaves stored connections untouched"

echo "test-setup-ai-studio-skip-path: ok"
