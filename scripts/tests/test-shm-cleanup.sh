#!/usr/bin/env bash
# test-shm-cleanup — scripts/lib/shm-cleanup.sh removes ONLY orphaned engine shared-memory segments,
# and switch.sh runs it before every boot.
#
# WHY THIS TEST EXISTS
# --------------------
# With `ipc: host`, engine shared-memory segments live in the HOST /dev/shm and survive an unclean
# container stop; 65 GB had piled up on the reference rig by 2026-09-25 (vllm#57303 alone leaks the
# whole KV-offload RAM tier). The cleanup has to be aggressive enough to reclaim that and careful
# enough never to touch a segment something is still using. This runs the REAL helper against a
# scratch directory seeded with one file per rule and asserts exactly which ones it removes:
#   removed : root-owned, engine-named, old, unmapped (psm_* and vllm_offload_*.mmap)
#   kept    : too young; not root-owned; not an engine name; held open by a live process.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fails=0
fail() { echo "✗ $*" >&2; fails=$((fails+1)); }

# Static: switch.sh must call the helper before `compose up`, and never let it fail the boot.
sw="$(command grep -nE 'lib/shm-cleanup\.sh" \|\| true' scripts/switch.sh | head -1 | cut -d: -f1)"
up="$(command grep -nE 'COMPOSE_BIN\} -f "\$\{file\}" up -d' scripts/switch.sh | head -1 | cut -d: -f1)"
[[ -n "$sw" && -n "$up" && "$sw" -lt "$up" ]] || fail "switch.sh must run lib/shm-cleanup.sh (|| true) before compose up (cleanup line='${sw}', up line='${up}')"

IMG=""
if docker info >/dev/null 2>&1; then
  for i in alpine:3.20 alpine:3 alpine:latest busybox:latest; do
    docker image inspect "$i" >/dev/null 2>&1 && { IMG="$i"; break; }
  done
fi
if [[ -z "$IMG" ]]; then
  [[ "$fails" -eq 0 ]] && { echo "test-shm-cleanup: ok (static only — no docker or no local alpine/busybox image, functional leg skipped)"; exit 0; }
  exit 1
fi

scratch="$(mktemp -d)"; holder=""
cleanup() {
  [[ -n "$holder" ]] && docker rm -f "$holder" >/dev/null 2>&1
  docker run --rm -v "$scratch:/s" "$IMG" sh -c 'rm -rf /s/* /s/.[!.]*' >/dev/null 2>&1
  rm -rf "$scratch"
}
trap cleanup EXIT

# Seed as root (the way a container leaves them), then age everything but "psm_young" by 30 minutes.
docker run --rm -v "$scratch:/s" "$IMG" sh -c '
  for f in psm_orphan vllm_offload_club3090-test.mmap psm_young psm_held sgl_shm_mq_held other_file; do
    head -c 4096 /dev/zero > /s/$f; done
  for f in psm_orphan vllm_offload_club3090-test.mmap psm_held sgl_shm_mq_held other_file; do
    touch -d "@$(( $(date +%s) - 1800 ))" /s/$f; done' || fail "could not seed the scratch dir"
head -c 4096 /dev/zero > "$scratch/psm_user_owned"; touch -d '30 minutes ago' "$scratch/psm_user_owned"
# A live process holding two old segments: one mapped path-style (fd), both must survive.
holder="$(docker run -d --pid=host -v "$scratch:/dev/shm" "$IMG" sh -c 'exec 3</dev/shm/psm_held 4</dev/shm/sgl_shm_mq_held; sleep 120')"
sleep 1

CLUB3090_SHM_DIR="$scratch" CLUB3090_SHM_MIN_AGE_MIN=5 bash scripts/lib/shm-cleanup.sh 2>"$scratch.log"
rc=$?
[[ "$rc" -eq 0 ]] || fail "shm-cleanup.sh exited $rc (it must never fail a boot)"
for f in psm_orphan vllm_offload_club3090-test.mmap; do
  [[ -e "$scratch/$f" ]] && fail "orphan $f was NOT removed"
done
for f in psm_young psm_held sgl_shm_mq_held psm_user_owned other_file; do
  [[ -e "$scratch/$f" ]] || fail "$f was removed but must be kept (young / held / user-owned / non-engine)"
done
command grep -q 'removed 2 orphaned' "$scratch.log" || fail "expected '[shm] removed 2 orphaned …' (got: $(cat "$scratch.log"))"
# Opt-out: with CLUB3090_SHM_CLEANUP=0 nothing is touched, even a fresh orphan.
docker run --rm -v "$scratch:/s" "$IMG" sh -c 'head -c 4096 /dev/zero > /s/psm_optout; touch -d "@$(( $(date +%s) - 1800 ))" /s/psm_optout'
CLUB3090_SHM_CLEANUP=0 CLUB3090_SHM_DIR="$scratch" bash scripts/lib/shm-cleanup.sh 2>/dev/null
[[ -e "$scratch/psm_optout" ]] || fail "CLUB3090_SHM_CLEANUP=0 still removed a file"
rm -f "$scratch.log"

if [[ "$fails" -gt 0 ]]; then
  echo "test-shm-cleanup: $fails failure(s)" >&2
  exit 1
fi
echo "test-shm-cleanup: ok (switch.sh runs it before compose up; 2 orphans removed; young / held / user-owned / non-engine kept; opt-out honoured)"
