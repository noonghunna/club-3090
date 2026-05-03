#!/usr/bin/env bash
#
# Pull the latest club-3090 stack and re-run setup. The one-shot upgrade path.
#
# What this does (in order):
#   1. Refuses to run on a dirty tree — commit or stash your local edits first.
#   2. Bails on a non-master branch (forks / contributors should pull manually).
#   3. git pull --ff-only origin master  (no merge commits, no rebase ambiguity)
#   4. bash scripts/setup.sh <model>     (re-pins Genesis, re-vendors Marlin —
#                                          idempotent, fast on second run)
#   5. Tells you to restart whatever variant you had running.
#
# What this does NOT do:
#   - Pull the vLLM image (the SHA is pinned in compose; setup.sh's
#     instructions cover the docker pull when needed).
#   - Restart your container — that's a deliberate user action, in case you
#     want to A/B old vs new before bringing the new variant up.
#
# Usage:
#   bash scripts/update.sh                    # uses default model (qwen3.6-27b)
#   bash scripts/update.sh qwen3.6-27b        # explicit
#   bash scripts/update.sh --dry-run          # show what would happen, change nothing
#   bash scripts/update.sh --force            # skip the "behind origin" check (re-runs setup anyway)
#
# Exit codes:
#   0 — already up-to-date or successfully updated
#   1 — dirty tree / wrong branch / missing dep / git pull failed / setup.sh failed

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-qwen3.6-27b}"
DRY_RUN=0
FORCE=0

# --- arg parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    -*) echo "Unknown flag: $1"; exit 1 ;;
    *)  MODEL="$1"; shift ;;
  esac
done

cd "$ROOT_DIR"

run() {
  # Echo + execute (or just echo, in dry-run).
  echo "[update] \$ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

# --- 1. dep checks ---
command -v git >/dev/null 2>&1 || { echo "[update] ERROR: 'git' not found in PATH." >&2; exit 1; }
[[ -d "${ROOT_DIR}/.git" ]] || { echo "[update] ERROR: ${ROOT_DIR} is not a git repo." >&2; exit 1; }

# --- 2. branch check ---
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ "$current_branch" != "master" ]]; then
  echo "[update] ERROR: not on master (current: ${current_branch})." >&2
  echo "         This script only updates the master branch from origin." >&2
  echo "         If you're on a feature branch, switch first:  git checkout master" >&2
  echo "         If you've forked, pull manually from your fork's upstream." >&2
  exit 1
fi

# --- 3. dirty-tree check ---
# git status --porcelain prints one line per modified / untracked file.
# Empty output = clean tree. We refuse to operate on dirty trees so we
# don't clobber the rare user who's edited a compose locally.
dirty=$(git status --porcelain 2>/dev/null)
if [[ -n "$dirty" ]]; then
  echo "[update] ERROR: working tree has local changes — refusing to update." >&2
  echo "" >&2
  echo "$dirty" | sed 's/^/         /' >&2
  echo "" >&2
  echo "         Commit or stash your changes first, then re-run:" >&2
  echo "           git stash && bash scripts/update.sh && git stash pop" >&2
  echo "         Or if these were experiments you don't need:" >&2
  echo "           git restore .  &&  git clean -fd  &&  bash scripts/update.sh" >&2
  exit 1
fi

# --- 4. fetch + behind check ---
echo "[update] fetching origin/master..."
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[update] (dry-run) git fetch --quiet origin master"
else
  git fetch --quiet origin master || { echo "[update] ERROR: git fetch failed." >&2; exit 1; }
fi

behind=$(git rev-list --count HEAD..origin/master 2>/dev/null || echo 0)
ahead=$(git rev-list --count origin/master..HEAD 2>/dev/null || echo 0)

if [[ "$behind" == "0" && "$ahead" == "0" ]]; then
  echo "[update] up-to-date with origin/master."
  if [[ $FORCE -eq 0 ]]; then
    echo "[update] (use --force to re-run setup.sh anyway, e.g. after a manual edit)"
    exit 0
  fi
  echo "[update] --force passed; re-running setup.sh anyway."
elif [[ "$ahead" != "0" && "$behind" == "0" ]]; then
  echo "[update] you are ${ahead} commit(s) ahead of origin/master (no remote changes)."
  echo "         Nothing to pull. If you want to push:  git push origin master"
  exit 0
elif [[ "$ahead" != "0" && "$behind" != "0" ]]; then
  echo "[update] ERROR: branch has diverged (${ahead} ahead, ${behind} behind)." >&2
  echo "         Resolve manually:  git pull --rebase  (or)  git pull origin master" >&2
  exit 1
else
  echo "[update] ${behind} commit(s) behind origin/master — pulling..."
fi

# --- 5. pull (only if behind) ---
if [[ "$behind" != "0" ]]; then
  run git pull --ff-only origin master
fi

# --- 6. re-run setup.sh ---
echo ""
echo "[update] re-running setup.sh ${MODEL} (re-pins Genesis, re-vendors Marlin)..."
echo ""
run bash "${ROOT_DIR}/scripts/setup.sh" "$MODEL"

# --- 7. next-step hint ---
echo ""
echo "[update] ✓ done."
echo ""
running=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(vllm-qwen36-27b|llama-cpp-qwen36-27b)' | head -1 || true)
if [[ -n "$running" ]]; then
  echo "[update] A club-3090 container is currently running: ${running}"
  echo "[update] To pick up the latest config + Genesis tree, restart it:"
  echo "[update]   bash scripts/switch.sh <variant>"
  echo "[update] (Use 'bash scripts/switch.sh --list' to see available variants.)"
else
  echo "[update] Next:  bash scripts/launch.sh   (or bash scripts/switch.sh <variant>)"
fi
