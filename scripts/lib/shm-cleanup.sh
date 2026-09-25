#!/usr/bin/env bash
# shm-cleanup.sh — remove ORPHANED engine shared-memory segments from the host /dev/shm before a boot.
#
# WHY: the club composes run with `ipc: host`, so the shared-memory segments vLLM / SGLang / PyTorch
# create (psm_*, sgl_*, sglang_*, torch_*, sem.mp-*, vllm_offload_*.mmap) live in the HOST /dev/shm
# and outlive any container that does not exit cleanly. /dev/shm is tmpfs: every orphan is RAM that
# never comes back until someone deletes it. On 2026-09-25 the reference rig had 65 GB resident with
# nothing running — 339 orphans since 09-18, including two 32 GiB KV-offload regions (vllm#57303).
#
# SAFETY — a file is removed only when ALL of these hold:
#   1. its name matches one of the engine patterns above;
#   2. it is owned by root (created inside a container — a user's own process owns its segments);
#   3. it is older than CLUB3090_SHM_MIN_AGE_MIN minutes (default 5), so a peer container in the middle
#      of creating-then-mapping a segment is never raced;
#   4. no live process on the host maps it or holds an fd on it (read from /proc in the host PID
#      namespace).
# The files are root-owned in the sticky /dev/shm, so deleting them needs root: this runs one short
# alpine container with --pid=host + SYS_PTRACE (to read other processes' /proc/<pid>/maps and fds).
#
# Best-effort by contract: it NEVER fails a boot. No docker, no image and no network to pull it all
# mean "skip with a one-line note". CLUB3090_SHM_CLEANUP=0 disables it.
#
# Usage: source it and call `club_shm_cleanup`, or run it directly.
#   CLUB3090_SHM_DIR           directory to clean (default /dev/shm; tests point it elsewhere)
#   CLUB3090_SHM_MIN_AGE_MIN   minimum age in minutes (default 5)
#   CLUB3090_SHM_IMAGE         helper image (default: the first LOCAL one of alpine:3.20, alpine:3,
#                              alpine:latest, busybox:latest; pulls alpine:3 only when none is present)

club_shm_cleanup() {
  [[ "${CLUB3090_SHM_CLEANUP:-1}" == "0" ]] && return 0
  local dir="${CLUB3090_SHM_DIR:-/dev/shm}" age="${CLUB3090_SHM_MIN_AGE_MIN:-5}"
  local image="${CLUB3090_SHM_IMAGE:-}"
  [[ -d "$dir" ]] || return 0
  [[ "$age" =~ ^[0-9]+$ ]] || age=5
  # Fast path: nothing old, root-owned and engine-named → no container at all.
  local cand
  cand="$(find "$dir" -maxdepth 1 -type f -user 0 -mmin "+${age}" \
            \( -name 'psm_*' -o -name 'sgl_*' -o -name 'sglang_*' -o -name 'torch_*' \
               -o -name 'sem.mp-*' -o -name 'vllm_offload_*.mmap' \) 2>/dev/null | head -1)"
  [[ -n "$cand" ]] || return 0
  command -v docker >/dev/null 2>&1 || { echo "[shm] orphaned engine segments in ${dir}, but no docker to remove them — skipped" >&2; return 0; }
  if [[ -z "$image" ]]; then
    local i
    for i in alpine:3.20 alpine:3 alpine:latest busybox:latest; do
      docker image inspect "$i" >/dev/null 2>&1 && { image="$i"; break; }
    done
  fi
  if [[ -z "$image" ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    image="${image:-alpine:3}"
    timeout 60 docker pull -q "$image" >/dev/null 2>&1 \
      || { echo "[shm] could not pull ${image} to clean orphaned segments in ${dir} — skipped (CLUB3090_SHM_CLEANUP=0 silences this)" >&2; return 0; }
  fi
  local out
  # shellcheck disable=SC2016  # the script runs inside the helper container
  out="$(timeout 120 docker run --rm --pid=host --cap-add SYS_PTRACE -v "${dir}:/shm" -e AGE="$age" \
    "$image" sh -c '
      inuse="$(for p in /proc/[0-9]*; do
                 cat "$p/maps" 2>/dev/null
                 for fd in "$p"/fd/*; do readlink "$fd" 2>/dev/null; done
               done | grep -oE "/dev/shm/[^ ]+" | sed "s/ (deleted)\$//" | sort -u)"
      n=0; kb=0
      for f in $(find /shm -maxdepth 1 -type f -user 0 -mmin "+$AGE" \
                   \( -name "psm_*" -o -name "sgl_*" -o -name "sglang_*" -o -name "torch_*" \
                      -o -name "sem.mp-*" -o -name "vllm_offload_*.mmap" \)); do
        b="${f#/shm/}"
        printf "%s\n" "$inuse" | grep -qxF "/dev/shm/$b" && continue
        k=$(du -k "$f" | cut -f1)
        rm -f "$f" && n=$((n+1)) && kb=$((kb+k))
      done
      echo "$n $kb"' 2>/dev/null)" || { echo "[shm] cleanup helper failed — skipped" >&2; return 0; }
  local n kb
  read -r n kb <<< "$out"
  if [[ "${n:-0}" =~ ^[0-9]+$ && "${n:-0}" -gt 0 ]]; then
    awk -v n="$n" -v kb="${kb:-0}" -v d="$dir" 'BEGIN{printf "[shm] removed %d orphaned engine shared-memory segment(s) from %s, %.1f GB returned to RAM\n", n, d, kb/1048576}' >&2
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  club_shm_cleanup
fi
