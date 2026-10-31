#!/usr/bin/env bash
# benchlocal-path.sh — find benchlocal-cli when this shell's PATH misses the directory it was installed to.
#
# Source it; it defines one function and runs nothing at source time.
#
# Why this exists: pip --user, uv tool and pipx all put benchlocal-cli in ~/.local/bin (uv: $UV_TOOL_BIN_DIR or
# $XDG_BIN_HOME first). A login shell has that on PATH, but a non-login one often doesn't (cron, a systemd unit,
# some IDE or tmux setups). The 512K rebench on #1888 ran every other step, then failed its 8-pack at
# "benchlocal-cli not found on $PATH" with benchlocal-cli installed and working in the user's own shell.

# club_benchlocal_on_path — 0 when benchlocal-cli is runnable after this call, else 1. When it is only in one of
# the user bin directories, that directory goes to the front of PATH (exported, so child scripts find it too)
# and one line on stderr says so.
club_benchlocal_on_path() {
  command -v benchlocal-cli >/dev/null 2>&1 && return 0
  local d
  for d in "${UV_TOOL_BIN_DIR:-}" "${XDG_BIN_HOME:-}" "${HOME:-}/.local/bin"; do
    [[ -n "$d" && -x "$d/benchlocal-cli" ]] || continue
    export PATH="$d:$PATH"
    echo "[benchlocal] using ${d}/benchlocal-cli (that directory isn't on this shell's PATH)" >&2
    return 0
  done
  return 1
}
