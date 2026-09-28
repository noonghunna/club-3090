#!/usr/bin/env bash
#
# Guard: a compose that bind-mounts a cache directory INTO the repo tree must run
# the container with the host's GID, or it leaves root-owned files nobody can
# delete without sudo.
#
# vLLM runs as root. When a compose maps a host path under models/ to a container
# cache path (/root/.triton/cache, /root/.cache/vllm/torch_compile_cache), every
# file it writes there is root-owned on the host. `user: "0:${DOCKER_GID:-1000}"`
# keeps UID 0 — which the image needs — while giving the files the invoking
# user's GROUP, so they come out group-writable and clean up normally.
#
# Measured cost of getting this wrong: three worktrees on the reference rig
# accumulated 140-406 MB each of root-owned triton/torch_compile cache that the
# owning user could not remove. Setting TRITON_CACHE_DIR does NOT help — these
# composes bind-mount that exact container path back into the tree, so the env
# var has no bearing on where the files land.
#
# _archive/ composes are exempt: they are historical records, not launched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

mapfile -t mounts < <(command grep -rlE '^\s+- \.\..*cache.*:/root/' "${ROOT}/models" --include='*.yml' 2>/dev/null \
                      | command grep -v '/_archive/' | sort)
[[ ${#mounts[@]} -gt 0 ]] || bad "found composes that mount a cache path" "at least one" "none — the scan matched nothing"

missing=()
for f in "${mounts[@]}"; do
  command grep -q 'user: "0:' "$f" || missing+=("${f#${ROOT}/}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  bad "every cache-mounting compose sets the host GID" "user: \"0:\${DOCKER_GID:-1000}\" in all ${#mounts[@]}" \
      $'\n'"$(printf '    %s\n' "${missing[@]}")"
else
  ok "all ${#mounts[@]} live cache-mounting composes run with the host GID"
fi

# The UID must stay 0 — the vLLM images expect root. A well-meaning "fix" that
# sets user: "1000:1000" breaks the container instead of the cleanup.
wrong_uid=()
for f in "${mounts[@]}"; do
  if command grep -qE '^\s*user:\s*"[^0]' "$f"; then wrong_uid+=("${f#${ROOT}/}"); fi
done
if [[ ${#wrong_uid[@]} -gt 0 ]]; then
  bad "UID stays 0" "user: \"0:...\"" $'\n'"$(printf '    %s\n' "${wrong_uid[@]}")"
else
  ok "every one keeps UID 0 (the images need root); only the GID is the host's"
fi

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-compose-cache-ownership" >&2; exit 1; fi
echo "PASS: test-compose-cache-ownership"
