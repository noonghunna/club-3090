"""Copy the parts of the checkout a pytest fixture needs: what git would ship, not what the rig left behind.

A helper, not a test module (test-pytest-modules.sh runs only test_*.py). Used by test_promote.py and
test_local_registry.py, whose fixtures build a throwaway repo root per test case.

Why not shutil.copytree: under models/ a rig that has served something keeps gitignored engine data and
state (Strata's packs and drafter, 2.3 GB on one contributor's checkout, #1983), sometimes in root-owned
0700 directories. copytree walked all of it, once per test case: ~143 GB written to TMPDIR in one run, a
46 GB /tmp filled part-way, and 63 cases red with `shutil.Error: Permission denied` on directories the
tests never read. git's own list (tracked files plus untracked ones that aren't ignored, so a contributor's
new compose still comes along) is the tree the tests are about.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Never copied, whatever git lists (each fixture says why): build caches written by containers, often
# root-owned (#1142), and the real local layer of whoever runs the suite (#1316).
SKIP_PARTS = frozenset({"__pycache__", "cache", "profiles-local"})


def _git_listed(repo: Path, rel: str) -> list[str] | None:
    """Tracked + untracked-not-ignored files under ``rel``, or None when ``repo`` isn't a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", rel],
            capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def copy_trees(repo: Path, dest: Path, rels) -> None:
    for rel in rels:
        files = _git_listed(repo, rel)
        if files is None:
            # Not a git checkout (an unpacked release archive): the old walk, with the same skips.
            shutil.copytree(repo / rel, dest / rel, ignore=shutil.ignore_patterns(*SKIP_PARTS))
            continue
        for f in files:
            if SKIP_PARTS.intersection(Path(f).parts):
                continue
            src = repo / f
            if not src.is_file():
                continue  # still in the index but deleted from the work tree
            dst = dest / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
