#!/usr/bin/env bash
# test-setup-saved-settings — setup.sh saves settings through the ONE writer (club-3090#1466, 1c).
#
# setup.sh writes two settings, and both used to land in files of their own:
#
#   MODEL_DIR  the interactive "where should weights go?" answer. It was `sed -i`'d or
#              appended into <repo>/.env — one checkout only, and a path holding `|` or
#              `&` broke the sed. It now goes to club3090.env via club_config_set, which
#              every checkout reads, and a value the readers would disagree on is refused
#              with the writer's message instead of being mangled.
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False  the WSL2 boot-crash workaround.
#              It was written to models/<model>/vllm/compose/.env, a file docker compose
#              never reads: every compose lives in <topology>/<quant>/, and compose loads
#              .env from the compose FILE's directory, not from the working directory. The
#              last leg below shows both halves on a real compose (when docker is here).
#
# Driven through setup.sh itself. The prompt needs a terminal, so `script` provides
# one; every run stops at SKIP_MODEL=1 with a mocked nvidia-smi and docker, so nothing
# downloads and no container is touched. The checkout is a fixture (symlinks to this
# tree) so its legacy .env is ours, and the settings dir is a temp dir.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# A checkout whose .env and compose dir are ours: setup.sh derives ROOT_DIR from its own
# path with a logical `cd`, so a symlinked scripts/ makes the fixture the root.
FX="$T/checkout"; mkdir -p "$FX/models/qwen3.6-27b/vllm/compose"
ln -s "$ROOT/scripts" "$FX/scripts"
# nvidia-smi: one 3090, so preflight passes; docker: absent (setup only warns).
mkdir -p "$T/bin"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  -L) echo "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-test)" ;;
  *compute_cap*) echo "8.6" ;;
  *) echo "0, NVIDIA GeForce RTX 3090, 24576, 8.6" ;;
esac
EOF
printf '#!/usr/bin/env bash\nexit 1\n' > "$T/bin/docker"
chmod +x "$T/bin/nvidia-smi" "$T/bin/docker"

# setup <config-dir> [VAR=val ...] — setup.sh qwen3.6-27b, stopped at SKIP_MODEL.
setup() {
  local cfg="$1"; shift
  env -u MODEL_DIR -u HF_TOKEN -u PYTORCH_CUDA_ALLOC_CONF PATH="$T/bin:$PATH" \
    CLUB3090_CONFIG_DIR="$cfg" PREFLIGHT_DISK_GB=0 SKIP_MODEL=1 "$@" \
    timeout 120 bash "$FX/scripts/setup.sh" qwen3.6-27b 2>&1
}
# setup_tty <config-dir> <answers> [VAR=val ...] — the same run on a terminal, answers on stdin.
setup_tty() {
  local cfg="$1" answers="$2"; shift 2
  local cmd="bash '$FX/scripts/setup.sh' qwen3.6-27b"
  printf '%b' "$answers" | env -u MODEL_DIR -u HF_TOKEN -u PYTORCH_CUDA_ALLOC_CONF PATH="$T/bin:$PATH" \
    CLUB3090_CONFIG_DIR="$cfg" PREFLIGHT_DISK_GB=0 SKIP_MODEL=1 "$@" \
    timeout 120 script -qec "$cmd" /dev/null 2>&1 | tr -d '\r'
}
saved() { command grep -E "^$2=" "$1/club3090.env" 2>/dev/null; }

# ── MODEL_DIR ────────────────────────────────────────────────────────────────
if command -v script >/dev/null 2>&1; then
  C1="$T/c1"
  out="$(setup_tty "$C1" "3\n$T/weights\n\n")"
  if command grep -qF 'Where should I put model weights?' <<<"$out"; then
    [[ "$(saved "$C1" MODEL_DIR)" == "MODEL_DIR=$T/weights" ]] \
      && ok "the MODEL_DIR prompt saves the answer to club3090.env" \
      || bad "MODEL_DIR not in club3090.env: '$(saved "$C1" MODEL_DIR)' — $(command grep -m1 -E 'saved|Save' <<<"$out")"
    command grep -qF "saved to $C1/club3090.env" <<<"$out" && ok "…and says where it went" \
      || bad "no 'saved to <club3090.env>' line: $(command grep -m2 -E 'saved|→' <<<"$out")"
    [[ ! -e "$FX/.env" ]] && ok "…and no longer writes the checkout's .env" || bad "setup.sh wrote $FX/.env: $(cat "$FX/.env")"
    command grep -qF '[model]   SKIP_MODEL=1' <<<"$out" && ok "…and the run carries on with it (reached SKIP_MODEL)" \
      || bad "run did not reach SKIP_MODEL: $(tail -3 <<<"$out")"
  else
    bad "the MODEL_DIR prompt never appeared: $(head -5 <<<"$out")"
  fi

  # The saved value is what the next run uses — no prompt.
  out="$(setup_tty "$C1" "")"
  command grep -qF 'Where should I put model weights?' <<<"$out" && bad "prompted again although MODEL_DIR is saved" \
    || { command grep -qF "Model dir:    $T/weights" <<<"$out" && ok "the next run takes MODEL_DIR from club3090.env without asking" \
         || bad "next run's model dir: $(command grep -m1 'Model dir' <<<"$out")"; }

  # Declining saves nothing.
  C2="$T/c2"
  out="$(setup_tty "$C2" "3\n$T/weights\nn\n")"
  [[ ! -e "$C2/club3090.env" && ! -e "$FX/.env" ]] && ok "answering 'n' saves nothing" \
    || bad "'n' still saved: $(cat "$C2/club3090.env" "$FX/.env" 2>/dev/null)"

  # A value the readers would read differently is refused, said, and not fatal.
  C3="$T/c3"
  out="$(setup_tty "$C3" "3\n$T/we\$ights\n\n")"
  command grep -qF "value contains '\$'" <<<"$out" && ok "a path with '\$' is refused with the writer's reason" \
    || bad "no refusal reason shown: $(command grep -m2 -E 'saved|ERROR|→' <<<"$out")"
  [[ -z "$(saved "$C3" MODEL_DIR)" ]] && ok "…and is not stored" || bad "refused value stored: $(saved "$C3" MODEL_DIR)"
  command grep -qF '[model]   SKIP_MODEL=1' <<<"$out" && ok "…and setup still carries on with it for this run" \
    || bad "a refused save aborted setup: $(tail -3 <<<"$out")"
else
  echo "  - MODEL_DIR prompt legs skipped: no 'script' (util-linux) to provide a terminal"
fi

# The legacy checkout .env is still READ (a rig that hasn't migrated keeps working).
printf 'MODEL_DIR=%s/legacy-weights\n' "$T" > "$FX/.env"
out="$(setup "$T/c-empty")"
command grep -qF "Model dir:    $T/legacy-weights" <<<"$out" && ok "a MODEL_DIR only in the legacy repo .env is still used" \
  || bad "legacy .env MODEL_DIR ignored: $(command grep -m1 'Model dir' <<<"$out")"
rm -f "$FX/.env"

# ── WSL2: PYTORCH_CUDA_ALLOC_CONF ────────────────────────────────────────────
printf 'Linux version 6.6.87.2-microsoft-standard-WSL2 (gcc) #1 SMP\n' > "$T/proc-wsl"
printf 'Linux version 6.8.0-142-generic (buildd@lcy02) #1 SMP\n'      > "$T/proc-linux"
W1="$T/w1"
out="$(setup "$W1" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-wsl")"
[[ "$(saved "$W1" PYTORCH_CUDA_ALLOC_CONF)" == "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False" ]] \
  && ok "WSL2: the allocator workaround is saved to club3090.env" \
  || bad "WSL2: not saved: '$(saved "$W1" PYTORCH_CUDA_ALLOC_CONF)' — $(command grep -m2 wsl2 <<<"$out")"
[[ -z "$(ls -A "$FX/models/qwen3.6-27b/vllm/compose")" ]] && ok "WSL2: nothing is written into the compose dir any more" \
  || bad "WSL2: compose dir got: $(ls -A "$FX/models/qwen3.6-27b/vllm/compose")"
out="$(setup "$W1" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-wsl")"
command grep -qF 'already has the expandable_segments:False override' <<<"$out" && ok "WSL2: a second run sees it and leaves it" \
  || bad "WSL2 re-run: $(command grep -m2 wsl2 <<<"$out")"

# A value already set — in the settings or the legacy .env — is never replaced.
W2="$T/w2"; mkdir -p "$W2"; printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' > "$W2/club3090.env"
out="$(setup "$W2" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-wsl")"
[[ "$(cat "$W2/club3090.env")" == "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" ]] \
  && ok "WSL2: a saved PYTORCH_CUDA_ALLOC_CONF is not clobbered" || bad "WSL2 clobbered: $(cat "$W2/club3090.env")"
command grep -qF 'WARN' <<<"$out" && ok "WSL2: …and setup warns that it lacks the override" || bad "no WARN: $(command grep -m2 wsl2 <<<"$out")"
W3="$T/w3"; printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' > "$FX/.env"
out="$(setup "$W3" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-wsl")"
[[ ! -e "$W3/club3090.env" ]] && ok "WSL2: a value in the legacy repo .env counts as set too" || bad "WSL2 wrote over a legacy value: $(cat "$W3/club3090.env")"
rm -f "$FX/.env"

# An old compose-dir .env is pointed out, never touched.
W4="$T/w4"; printf '# mine\nGPU_MEMORY_UTILIZATION=0.94\n' > "$FX/models/qwen3.6-27b/vllm/compose/.env"
before="$(cksum < "$FX/models/qwen3.6-27b/vllm/compose/.env")"
out="$(setup "$W4" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-wsl")"
[[ "$(cksum < "$FX/models/qwen3.6-27b/vllm/compose/.env")" == "$before" ]] && ok "WSL2: an existing compose-dir .env is left exactly as it was" \
  || bad "WSL2 changed the compose-dir .env"
command grep -qF 'is not read by docker compose' <<<"$out" && ok "WSL2: …and setup says it is not read" || bad "no note about the old file: $(command grep -m3 wsl2 <<<"$out")"
rm -f "$FX/models/qwen3.6-27b/vllm/compose/.env"

# Not WSL2 → nothing.
W5="$T/w5"
out="$(setup "$W5" MODEL_DIR="$T/weights" SETUP_PROC_VERSION="$T/proc-linux")"
[[ ! -e "$W5/club3090.env" ]] && ! command grep -q '\[wsl2\]' <<<"$out" && ok "not WSL2: nothing saved, nothing said" \
  || bad "non-WSL2 run: $(cat "$W5/club3090.env" 2>/dev/null) $(command grep -m1 wsl2 <<<"$out")"

# ── does the saved value reach the container? ───────────────────────────────
# switch.sh loads the settings (club_config_load) and runs `cd <model>/<engine>/compose
# && docker compose -f <topology>/<quant>/<file> up`. Render a real vLLM compose that way.
# `config` only renders; nothing starts.
if docker compose version >/dev/null 2>&1; then
  P="$T/proof"; mkdir -p "$P/models/qwen3.6-27b/vllm"
  ln -s "$ROOT/scripts" "$P/scripts"
  cp -r "$ROOT/models/qwen3.6-27b/vllm/compose" "$P/models/qwen3.6-27b/vllm/"
  render() {  # render <config-dir> → the container's PYTORCH_CUDA_ALLOC_CONF
    ( cd "$P/models/qwen3.6-27b/vllm/compose" && env -u PYTORCH_CUDA_ALLOC_CONF CLUB3090_CONFIG_DIR="$1" bash -c '
        source "$2/scripts/lib/club-config.sh"; club_config_load "$2"
        docker compose -f single/autoround-int4/minimal.yml config' _ "$1" "$P" 2>/dev/null ) \
      | command grep -m1 -oE 'PYTORCH_CUDA_ALLOC_CONF: .*'
  }
  printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False\n' > "$P/models/qwen3.6-27b/vllm/compose/.env"
  got="$(render "$T/c-empty")"
  [[ "$got" == "PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True" ]] \
    && ok "the OLD location (models/<model>/vllm/compose/.env) never reached the container: '$got'" \
    || bad "the compose-dir .env is read after all ('$got') — re-check the move"
  rm -f "$P/models/qwen3.6-27b/vllm/compose/.env"
  got="$(render "$W1")"
  [[ "$got" == "PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False" ]] \
    && ok "the value setup.sh saved reaches the container through the loader: '$got'" \
    || bad "saved value did not reach the compose: '$got'"
else
  echo "  - compose render legs skipped: no 'docker compose' here"
fi

[[ $fail -eq 0 ]] && echo "test-setup-saved-settings: ok" || echo "test-setup-saved-settings: FAIL"
exit $fail
