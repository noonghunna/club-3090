"""Entry point for the c3 command."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Launch the serve cockpit TUI."""
    # Resolve repo root from this file's location:
    #   <repo>/tools/serve-cockpit/club3090_cockpit/__main__.py
    #   parents[0] = club3090_cockpit/
    #   parents[1] = serve-cockpit/
    #   parents[2] = tools/
    #   parents[3] = <repo root>
    # Override with C3_REPO_ROOT if installed outside the tree.
    env_root = os.environ.get("C3_REPO_ROOT")
    repo_root = Path(env_root) if env_root else Path(__file__).resolve().parents[3]

    if not (repo_root / "scripts").is_dir():
        print(
            f"Error: club-3090 repo root not found at {repo_root} "
            f"(no scripts/ dir).  Run via the repo tree, or set C3_REPO_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import CockpitApp

    app = CockpitApp(repo_root=repo_root)
    app.run()


if __name__ == "__main__":
    main()
