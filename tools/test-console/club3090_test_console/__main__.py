"""Entry point for c3t command."""

import os
import sys
from pathlib import Path


def _suppress_event_loop_cleanup_error():
    """Suppress harmless 'Event loop is closed' errors from asyncio subprocess cleanup."""
    # Python 3.8+ has sys.unraisablehook for exceptions in __del__ methods
    if hasattr(sys, 'unraisablehook'):
        original_hook = sys.unraisablehook
        
        def filtered_hook(unraisable):
            # Suppress 'Event loop is closed' from asyncio subprocess cleanup
            if (unraisable.exc_type is RuntimeError and 
                unraisable.exc_value and 
                'Event loop is closed' in str(unraisable.exc_value)):
                return
            # Also suppress 'Cannot run the event loop while another loop is running'
            if (unraisable.exc_type is RuntimeError and
                unraisable.exc_value and
                'loop is' in str(unraisable.exc_value).lower()):
                return
            original_hook(unraisable)
        
        sys.unraisablehook = filtered_hook


def main():
    """Launch the test console TUI."""
    _suppress_event_loop_cleanup_error()
    
    # Resolve the repo root from this file's location (…/<repo>/tools/test-console/
    # club3090_test_console/__main__.py → parents[3] == repo root), so the tool works
    # from any clone. Override with C3T_REPO_ROOT if the package is installed elsewhere.
    env_root = os.environ.get("C3T_REPO_ROOT")
    repo_root = Path(env_root) if env_root else Path(__file__).resolve().parents[3]
    if not (repo_root / "scripts").is_dir():
        print(
            f"Error: club-3090 repo root not found at {repo_root} "
            f"(no scripts/ dir). Run via scripts/c3t, or set C3T_REPO_ROOT.",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import TestConsoleApp
    
    app = TestConsoleApp(repo_root=repo_root)
    app.run()


if __name__ == "__main__":
    main()
