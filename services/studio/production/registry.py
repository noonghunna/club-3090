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


def prompt_slice(reg: dict, video_lane: str | None = None) -> str:
    """The compact lane menu + rules handed to the planner.

    When `video_lane` is given (the operator's pin), only THAT lane's contract is shown —
    the planner can't choose the video lane anyway (it's forced), and showing the pinned
    lane's real fps/window/audio prevents the 4B from planning against the wrong lane's
    physics (e.g. Wan's 16 fps / silent when LTX at 24 fps is pinned).
    """
    lines = ["LANES YOU MAY USE (choose ONLY from these):"]
    vlanes = reg.get("video_lanes") or {}
    items = ([(video_lane, vlanes[video_lane])] if video_lane and video_lane in vlanes
             else list(vlanes.items()))
    for name, c in items:
        win = c.get("native_window_seconds", 5.0)
        fps = c.get("native_fps", 16)
        audio = ("SILENT (no audio in the clip)" if c.get("audio_behavior") == "none"
                 else "the clip has native audio, but the film's soundtrack is the narration + music layer")
        lines.append(
            f"- VIDEO lane '{name}': mode t2v, vivid PROSE prompt, each shot <= {win:.1f}s "
            f"@ {fps}fps — {audio}. {c.get('when_to_use', '')}"
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
