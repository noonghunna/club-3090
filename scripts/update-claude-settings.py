#!/usr/bin/env python3
"""Update ~/.claude/settings.json with the detected club-3090 endpoint.

Called by update-claude-settings.sh with three arguments:
  1. Path to settings.json
  2. Base URL (e.g., http://localhost:8077)
  3. Model name (e.g., qwen3.6-27b-nvfp4)
"""
import json
import os
import sys
import tempfile


def main():
    settings_file = sys.argv[1]
    base_url = sys.argv[2]
    model_name = sys.argv[3]

    with open(settings_file, "r") as f:
        settings = json.load(f)

    if "env" not in settings:
        settings["env"] = {}

    old_url = settings["env"].get("ANTHROPIC_BASE_URL", "")
    old_model = settings["env"].get("ANTHROPIC_MODEL", "")

    settings["env"]["ANTHROPIC_BASE_URL"] = base_url
    settings["env"]["ANTHROPIC_MODEL"] = model_name

    changed = False
    if old_url != base_url:
        print(
            f'[claude-settings] URL: {old_url or "(none)"} → {base_url}'
        )
        changed = True
    if old_model != model_name:
        print(
            f'[claude-settings] Model: {old_model or "(none)"} → {model_name}'
        )
        changed = True

    if not changed:
        print(f"[claude-settings] Already correct: {base_url} / {model_name}")
        sys.exit(0)

    # Atomic write: temp file then rename
    dir_name = os.path.dirname(settings_file)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, settings_file)
        print(f"[claude-settings] Updated {settings_file}")
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()