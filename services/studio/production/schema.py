"""ProductionPlanV1 — the validatable plan contract (v0a subset).

stdlib dataclasses + an explicit `validate()` (host has no pydantic). The shape is
forward-compatible with the full schema in the design doc; v0a only *enforces* the
slice it executes (single pinned video lane = wan, t2v shots, one timeline entry
per shot). Fields the design reserves but v0a ignores (creative_intent, style_bible,
asset_dependencies, image_policy, takes, ...) are simply not modelled here yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class PlanError(ValueError):
    """Raised when a plan is structurally invalid (clear, actionable message)."""


@dataclass
class Delivery:
    aspect: str = "16:9"
    width: int = 832          # Wan native lo-res; v0a renders at delivery dims (no rescale)
    height: int = 480
    fps: int = 16             # Wan native rate
    codec: str = "h264"
    loudness_lufs: float = -14.0
    subtitles: bool = False


@dataclass
class Shot:
    id: str
    lane: str                 # v0a: must equal project.video_lane ("wan")
    mode: str                 # v0a: "t2v"
    target_seconds: float
    prompt_intent: str
    narration: str = ""
    seed: int = 0
    validators: list = field(
        default_factory=lambda: ["non_empty", "duration", "no_audio_expected"]
    )


@dataclass
class Music:
    lane: str = "ace-step"
    tags: str = "ambient, reflective, cinematic, instrumental"
    lyrics: str = "[instrumental]"
    seconds: float = 0.0      # 0 => derived from total timeline duration at run time
    seed: int = 0


@dataclass
class Project:
    title: str
    tone: str = ""
    target_seconds: float = 0.0
    video_lane: str = "wan"


@dataclass
class TimelineEntry:
    clip: str                 # references Shot.id
    narration_offset: float = 0.0
    music_level_db: float = -18.0
    transition_in: str = "cut"


@dataclass
class ProductionPlanV1:
    project: Project
    shots: list                       # list[Shot]
    delivery: Delivery
    timeline: list                    # list[TimelineEntry]
    music: Optional[Music] = None
    schema_version: str = "ProductionPlanV1"

    # -- construction -------------------------------------------------------
    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ProductionPlanV1":
        try:
            project = Project(**d["project"])
            shots = [Shot(**s) for s in d["shots"]]
            delivery = Delivery(**d.get("delivery", {}))
            timeline = [TimelineEntry(**t) for t in d.get("timeline", [])]
            music = Music(**d["music"]) if d.get("music") else None
        except (KeyError, TypeError) as e:
            raise PlanError(f"malformed plan: {e}") from e
        plan = ProductionPlanV1(
            project=project, shots=shots, delivery=delivery,
            timeline=timeline, music=music,
            schema_version=d.get("schema_version", "ProductionPlanV1"),
        )
        plan.validate()
        return plan

    # -- v0a validation -----------------------------------------------------
    def validate(self) -> None:
        if self.schema_version != "ProductionPlanV1":
            raise PlanError(f"unsupported schema_version {self.schema_version!r}")
        if self.project.video_lane != "wan":
            raise PlanError(
                f"v0a supports only video_lane='wan' (got {self.project.video_lane!r})"
            )
        if not self.shots:
            raise PlanError("plan has no shots")

        ids = [s.id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise PlanError(f"duplicate shot ids: {ids}")

        for s in self.shots:
            if s.lane != self.project.video_lane:
                raise PlanError(
                    f"shot {s.id}: lane {s.lane!r} != pinned video_lane "
                    f"{self.project.video_lane!r} (v0a pins one video lane)"
                )
            if s.mode != "t2v":
                raise PlanError(f"shot {s.id}: v0a supports only mode='t2v' (got {s.mode!r})")
            if s.target_seconds <= 0:
                raise PlanError(f"shot {s.id}: target_seconds must be > 0")

        # Timeline: one entry per shot, all references valid, order = play order.
        known = set(ids)
        seen: set[str] = set()
        for t in self.timeline:
            if t.clip not in known:
                raise PlanError(f"timeline references unknown clip {t.clip!r}")
            if t.clip in seen:
                raise PlanError(f"timeline references clip {t.clip!r} twice")
            seen.add(t.clip)
        if seen != known:
            missing = known - seen
            raise PlanError(f"timeline missing shots: {sorted(missing)}")

        if self.delivery.fps <= 0 or self.delivery.width <= 0 or self.delivery.height <= 0:
            raise PlanError("delivery fps/width/height must be > 0")

    def shot_by_id(self, sid: str) -> Shot:
        for s in self.shots:
            if s.id == sid:
                return s
        raise PlanError(f"no shot {sid!r}")

    def ordered_shots(self) -> list:
        """Shots in timeline (play) order."""
        return [self.shot_by_id(t.clip) for t in self.timeline]
