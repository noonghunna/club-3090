"""Lane capability registry loader — 'what the machine can do.'

Loads `capabilities.yaml` and compresses it into the compact text the planner is
given (the planner may choose lanes ONLY from here). The executor + schema remain
authoritative; this is the planner's menu, not the enforcement.
"""
from __future__ import annotations

import os

import yaml

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capabilities.yaml")


def load(path: str = REGISTRY_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def video_lanes(reg: dict) -> list[str]:
    return list((reg.get("video_lanes") or {}).keys())


def audio_lanes(reg: dict) -> list[str]:
    return list((reg.get("audio_lanes") or {}).keys())


def prompt_slice(reg: dict) -> str:
    """The compact lane menu + rules handed to the planner."""
    lines = ["LANES YOU MAY USE (choose ONLY from these):"]
    for name, c in (reg.get("video_lanes") or {}).items():
        win = c.get("native_window_seconds", 5.0)
        fps = c.get("native_fps", 16)
        lines.append(
            f"- VIDEO lane '{name}': mode t2v, vivid PROSE prompt, each shot <= {win:.1f}s "
            f"@ {fps}fps, SILENT (no audio in the clip). {c.get('when_to_use', '')}"
        )
    for name, c in (reg.get("audio_lanes") or {}).items():
        lines.append(
            f"- AUDIO lane '{name}': {c.get('kind')}, format {c.get('prompt_format')}. "
            f"{c.get('when_to_use', '')}"
        )
    lines.append("RULES:")
    for r in reg.get("rules", []):
        lines.append(f"- {r}")
    return "\n".join(lines)
