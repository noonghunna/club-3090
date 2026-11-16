#!/usr/bin/env bash
#
# Pull the latest club-3090 stack and re-run setup. The one-shot upgrade path.
#
# What this does (in order):
#   1. Refuses to run on a dirty tree — commit or stash your local edits first.
#   2. Bails on a non-master branch (forks / contributors should pull manually).
#   3. git pull --ff-only origin master  (no merge commits, no rebase ambiguity)
#   4. bash scripts/setup.sh <slug>      (re-fetches missing weights — idempotent,
#                                          fast on second run) for the weights in
#                                          use: the slug each running club-3090
#                                          container was launched as. Nothing
#                                          running → this step is skipped.
#   5. Tells you the switch.sh command that restarts what you had running.
#
# What this does NOT do:
#   - Pull the vLLM image (the SHA is pinned in compose; setup.sh's
#     instructions cover the docker pull when needed).
#   - Restart your container — that's a deliberate user action, in case you
#     want to A/B old vs new before bringing the new variant up.
#
# Usage:
#   bash scripts/update.sh                    # checks the running slug's weights (none running: skipped)
#   bash scripts/update.sh <slug>             # that slug's weights + companions, e.g. qengine/qwen38-27b-dual
#   bash scripts/update.sh qwen3.6-27b        # a model's setup.sh default (MODEL=<model> works too)
#   bash scripts/update.sh --dry-run          # show what would happen, change nothing
#   bash scripts/update.sh --force            # skip the "behind origin" check (re-runs setup anyway)
#
# Exit codes:
#   0 — already up-to-date or successfully updated
#   1 — dirty tree / wrong branch / missing dep / git pull failed / setup.sh failed

# shellcheck source=lib/club-containers.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/lib/club-containers.sh"
set -euo pipefail
# Dot decimals whatever the user's locale: under es_ES / de_DE mawk read "1.65" as 1 and bash printf rejected "9.0" (#1926, #159).
# LC_ALL would override LC_NUMERIC, so its value moves to the other categories first. Guarded by test-numeric-locale.sh.
[[ -z "${LC_ALL:-}" ]] || { for _lc in LANG LC_CTYPE LC_COLLATE LC_MESSAGES LC_TIME LC_MONETARY; do export "$_lc=$LC_ALL"; done; unset LC_ALL _lc; }; export LC_NUMERIC=C

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# What setup.sh checks: an argument, else MODEL, else the running slugs (step 6).
# There is no model default any more (#2050): `qwen3.6-27b` checked — and on a rig
# without it would have downloaded — the weights of a model the rig may never run.
MODEL="${MODEL:-}"
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
# -e, not -d: in a git worktree .git is a file, so -d called a worktree "not a git repo".
[[ -e "${ROOT_DIR}/.git" ]] || { echo "[update] ERROR: ${ROOT_DIR} is not a git repo." >&2; exit 1; }

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

# --- 6. re-run setup.sh for the weights in use ---
# "Which club containers are running?" — registry-derived. This asked the question
# with a TWO-NAME list (vllm-qwen36-27b|llama-cpp-qwen36-27b), so it stayed
# silent for every other model and engine on the stack, not just the new ones.
# ⚠️ Not caught by test-engine-kind-resolver arm 5: that arm polices the ENGINE
# prefix alternation, and a model-specific name is legitimate elsewhere (report.sh
# deliberately prefers `name=vllm-qwen36` before falling back to the full set).
# Each row is `name<TAB>slug`: the launchers stamp the slug a container runs as its
# club3090.slug label (scripts/lib/slug_label.py), empty for one started without it.
_ps_fmt=$'{{.Names}}\t{{.Label "club3090.slug"}}'
_club_re="$(club_container_re)"
running_rows=$(docker ps --format "$_ps_fmt" 2>/dev/null | while IFS=$'\t' read -r _n _s; do
  printf '%s\n' "$_n" | command grep -qE "$_club_re" && printf '%s\t%s\n' "$_n" "$_s"
done || true)
running_slugs=$(printf '%s\n' "$running_rows" | awk -F'\t' 'NF>1 && $2!="" {print $2}' | sort -u)
unlabelled=$(printf '%s\n' "$running_rows" | awk -F'\t' '$1!="" && $2=="" {print $1}')

echo ""
if [[ -n "$MODEL" ]]; then
  echo "[update] re-running setup.sh ${MODEL} (re-fetches missing weights)..."
  echo ""
  run bash "${ROOT_DIR}/scripts/setup.sh" "$MODEL"
elif [[ -n "$running_slugs" ]]; then
  # setup.sh takes the slug form: that slug's weights variant and its
  # weights_companions, and nothing else (#2050). An array, not a here-string, so
  # setup.sh keeps the terminal's stdin for its prompts.
  mapfile -t _slugs <<<"$running_slugs"
  for slug in "${_slugs[@]}"; do
    echo "[update] re-running setup.sh ${slug} (the running slug's weights; re-fetches missing ones)..."
    echo ""
    run bash "${ROOT_DIR}/scripts/setup.sh" "$slug"
  done
fi
if [[ -z "$MODEL" && -n "$unlabelled" ]]; then
  echo "[update] not checked: $(tr '\n' ' ' <<<"$unlabelled")— running without a club3090.slug label"
  echo "[update]   (started by hand, or before the launchers stamped one), so its weights are unknown."
fi
if [[ -z "$MODEL" && -z "$running_slugs" ]]; then
  [[ -n "$unlabelled" ]] || echo "[update] nothing is serving, so no weights are in use to check."
  echo "[update] skipping setup.sh. To check a slug's weights:  bash scripts/update.sh <slug>"
  echo "[update]   (bash scripts/switch.sh --list shows the slugs)"
fi

# --- 7. next-step hint ---
echo ""
echo "[update] ✓ done."
echo ""
if [[ -n "$running_rows" ]]; then
  echo "[update] To pick up the latest config, restart what is running:"
  while IFS=$'\t' read -r name slug; do
    [[ -n "$name" ]] || continue
    if [[ -n "$slug" ]]; then
      echo "[update]   bash scripts/switch.sh ${slug}      # ${name}"
    else
      echo "[update]   bash scripts/switch.sh <variant>    # ${name} (no slug label)"
    fi
  done <<<"$running_rows"
  echo "[update] (Use 'bash scripts/switch.sh --list' to see available variants.)"
else
  echo "[update] Next:  bash scripts/launch.sh   (or bash scripts/switch.sh <variant>)"
fi

# Support services (Open WebUI, LiteLLM, Qdrant, SearXNG, spark-dashboard): a pull
# can bump a service's pinned image, but nothing already running moves to it —
# switch.sh, reboots and `docker restart` all keep the old image. Report drift
# (only when there is some) and point at the one command that applies it.
# Read-only; SVC_SUDO_FLAGS=-n so this never stops to prompt for a password.
if [[ $DRY_RUN -eq 0 ]] && command -v docker >/dev/null 2>&1; then
  svc_report=$(SVC_SUDO_FLAGS=-n bash "${ROOT_DIR}/scripts/gpu-mode.sh" service-images 2>/dev/null || true)
  if [[ "$svc_report" == *"gpu-mode upgrade"* ]]; then
    echo ""
    echo "[update] Some running support services are behind their pinned image:"
    echo "$svc_report"
  fi
fi
