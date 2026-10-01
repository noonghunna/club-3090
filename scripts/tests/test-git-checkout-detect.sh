#!/usr/bin/env bash
# test-git-checkout-detect — no script decides "is this a git checkout?" with a DIRECTORY test.
#
# In a git worktree (and a submodule) .git is a FILE that points at the real git dir. Three
# scripts tested it with -d, so from a worktree report.sh printed "club-3090: not a git repo",
# update.sh refused with "is not a git repo", and preflight's drift check bailed early. They
# now use -e (the repo root must itself be the checkout; `git rev-parse --is-inside-work-tree`
# would also say yes for a copy sitting inside some unrelated repo).
#
# Tree-wide: any shell `-d …/.git` test or Python is_dir()/isdir() on a .git path under
# scripts/ or tools/ fails this guard.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

hits="$(command grep -rnE --include='*.sh' --include='*.py' \
  -e '-d[[:space:]]+"?[^[:space:]"]*\.git"?([[:space:]]|\]|$)' \
  -e '\.git["'\'']?\)?\.is_dir\(\)' -e 'isdir\([^)]*\.git' \
  scripts tools 2>/dev/null | command grep -v '^scripts/tests/' || true)"
if [[ -n "$hits" ]]; then
  echo "  ✗ a directory-only .git check (a worktree's .git is a file — use -e / exists()):" >&2
  printf '%s\n' "$hits" | sed 's/^/      /' >&2
  exit 1
fi
echo "  ✓ no directory-only .git checks under scripts/ or tools/"
echo "test-git-checkout-detect: ok"
