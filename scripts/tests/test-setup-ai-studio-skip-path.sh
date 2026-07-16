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

echo "test-setup-ai-studio-skip-path: ok"
