"""ProductionPlanV1 — the validatable plan contract (v0a subset).

stdlib dataclasses + an explicit `validate()` (host has no pydantic). The shape is
forward-compatible with the full schema in the design doc. One pinned video lane per
job (operator-chosen via stack.py — validated against the production-renderable set,
never silently picked), continuity-aware shots (t2v / i2v), one timeline entry per
shot. Fields the design reserves but v0a ignores (creative_intent, style_bible,
takes, ...) are simply not modelled here yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .stack import KEYFRAME_LANES, VIDEO_LANES, wired_keyframe_lanes, wired_video_lanes


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
class Character:
    """A recurring on-screen character — the Character Bible (Tier A: SEMANTIC identity).

    The canonical block (description + wardrobe + props) is injected by the executor into
    every keyframe + Wan shot prompt this character appears in, so the same person reads
    consistently across shots. This is *less drift*, NOT a guaranteed face lock — that is
    Tier B (reference-image conditioning), for which `reference_asset_id` is RESERVED here
    but unused in Tier A.
    """
    id: str
    name: str
    description: str                 # canonical visual appearance (build, age, face, hair)
    role: str = ""                   # protagonist | antagonist | supporting | …
    wardrobe: str = ""               # signature clothing
    props: str = ""                  # signature props (hat, cane, badge …)
    negative: str = ""               # negative drift notes (what NOT to render)
    seed: int = 0                    # stable per-character seed
    reference_asset_id: Optional[str] = None   # RESERVED for Tier B — unused in Tier A


@dataclass
class Shot:
    id: str
    lane: str                 # must equal project.video_lane (one pinned lane per job)
    mode: str                 # "t2v" | "i2v"
    target_seconds: float
    prompt_intent: str
    narration: str = ""
    seed: int = 0
    start_from: Optional[str] = None   # None=t2v · "prev_last_frame"=chain · "<asset id>"=hero/keyframe
    characters: list = field(default_factory=list)   # Character ids appearing in this shot
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
class Assembly:
    """Deterministic post-production knobs (not 'brain' work)."""
    transition_seconds: float = 0.6   # dissolve length between clips
    duck_db: int = 6                  # bed duck depth under narration (gentler than the old ~12)
    normalize: bool = True


@dataclass
class AssetTask:
    """A generated image asset produced in pre-production (v0b-images)."""
    id: str
    role: str            # hero_keyframe | reference | storyboard | title_card
    lane: str            # an image lane (e.g. "chroma")
    prompt: str
    seed: int = 0
    width: int = 832
    height: int = 480
    characters: list = field(default_factory=list)   # Character ids this keyframe depicts


@dataclass
class Project:
    title: str
    tone: str = ""
    target_seconds: float = 0.0
    video_lane: str = "wan"           # operator-pinned (stack.py); default wan
    continuity: str = "none"          # none | chain | hero | storyboard (v0b-images)
    image_policy: dict = field(default_factory=dict)   # role -> image lane


@dataclass
class TimelineEntry:
    clip: str                 # references Shot.id
    narration_offset: float = 0.0
    music_level_db: float = -18.0
    transition_in: str = "dissolve"   # "dissolve" (default) | "cut" — transition INTO this clip


@dataclass
class ProductionPlanV1:
    project: Project
    shots: list                       # list[Shot]
    delivery: Delivery
    timeline: list                    # list[TimelineEntry]
    music: Optional[Music] = None
    assembly: Assembly = field(default_factory=Assembly)
    asset_tasks: list = field(default_factory=list)   # list[AssetTask] (v0b-images)
    characters: list = field(default_factory=list)    # list[Character] (Character Bible)
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
            _a = d.get("assembly", {}) or {}
            assembly = Assembly(
                transition_seconds=float(_a.get("transition_seconds", 0.6)),
                duck_db=int(_a.get("duck_db", 6)),
                normalize=bool(_a.get("normalize", True)),
            )
            asset_tasks = [AssetTask(**a) for a in (d.get("asset_tasks") or [])]
            characters = [Character(**c) for c in (d.get("characters") or [])]
        except (KeyError, TypeError, ValueError) as e:
            raise PlanError(f"malformed plan: {e}") from e
        plan = ProductionPlanV1(
            project=project, shots=shots, delivery=delivery,
            timeline=timeline, music=music, assembly=assembly,
            asset_tasks=asset_tasks, characters=characters,
            schema_version=d.get("schema_version", "ProductionPlanV1"),
        )
        plan.validate()
        return plan

    # -- v0a validation -----------------------------------------------------
    def validate(self) -> None:
        if self.schema_version != "ProductionPlanV1":
            raise PlanError(f"unsupported schema_version {self.schema_version!r}")
        # The video lane is an operator/stack decision (see stack.py). Validate it is a
        # known lane the production executor can actually render — never silently picked.
        vl = self.project.video_lane
        if vl not in VIDEO_LANES:
            raise PlanError(
                f"unknown video_lane {vl!r}; choose from {sorted(VIDEO_LANES)}"
            )
        if not VIDEO_LANES[vl]["wired"]:
            raise PlanError(
                f"video_lane {vl!r} is a known studio lane but not yet wired into the "
                f"production executor (renders today: {wired_video_lanes()})"
            )
        if not self.shots:
            raise PlanError("plan has no shots")

        ids = [s.id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise PlanError(f"duplicate shot ids: {ids}")

        if self.project.continuity not in ("none", "chain", "hero", "storyboard"):
            raise PlanError(
                f"project.continuity must be none|chain|hero|storyboard "
                f"(got {self.project.continuity!r})"
            )
        asset_ids = {a.id for a in self.asset_tasks}
        if len(asset_ids) != len(self.asset_tasks):
            raise PlanError("duplicate asset_task ids")

        # keyframe assets must use a wired image lane (the continuity-quality lever).
        for a in self.asset_tasks:
            if a.lane not in KEYFRAME_LANES:
                raise PlanError(
                    f"asset {a.id}: unknown keyframe lane {a.lane!r}; "
                    f"choose from {sorted(KEYFRAME_LANES)}"
                )
            if not KEYFRAME_LANES[a.lane]["wired"]:
                raise PlanError(
                    f"asset {a.id}: keyframe lane {a.lane!r} not wired "
                    f"(renders today: {wired_keyframe_lanes()})"
                )

        # continuity needs image->video: any i2v shot requires the pinned lane to have i2v.
        if any(s.mode == "i2v" for s in self.shots) and not VIDEO_LANES[vl]["i2v"]:
            raise PlanError(
                f"plan has i2v shots but video_lane {vl!r} has no i2v workflow "
                f"(use continuity='none' or a video lane with i2v)"
            )

        # Character Bible: unique ids; every shot/asset character ref resolves; a
        # reserved reference_asset_id (Tier B) must point at a declared asset_task.
        char_ids = [c.id for c in self.characters]
        if len(char_ids) != len(set(char_ids)):
            raise PlanError(f"duplicate character ids: {char_ids}")
        known_chars = set(char_ids)
        for c in self.characters:
            if not c.description.strip():
                raise PlanError(f"character {c.id!r}: empty canonical description")
            if c.reference_asset_id and c.reference_asset_id not in asset_ids:
                raise PlanError(
                    f"character {c.id!r}: reference_asset_id {c.reference_asset_id!r} "
                    f"is not a declared asset_task id {sorted(asset_ids)}"
                )
        for s in self.shots:
            unknown = [cid for cid in s.characters if cid not in known_chars]
            if unknown:
                raise PlanError(f"shot {s.id}: unknown character ids {unknown} "
                                f"(declared: {sorted(known_chars)})")
        for a in self.asset_tasks:
            unknown = [cid for cid in a.characters if cid not in known_chars]
            if unknown:
                raise PlanError(f"asset {a.id}: unknown character ids {unknown} "
                                f"(declared: {sorted(known_chars)})")

        for s in self.shots:
            if s.lane != self.project.video_lane:
                raise PlanError(
                    f"shot {s.id}: lane {s.lane!r} != pinned video_lane "
                    f"{self.project.video_lane!r} (one pinned video lane per job)"
                )
            if s.mode not in ("t2v", "i2v"):
                raise PlanError(f"shot {s.id}: mode must be 't2v' or 'i2v' (got {s.mode!r})")
            if s.target_seconds <= 0:
                raise PlanError(f"shot {s.id}: target_seconds must be > 0")
            if s.mode == "i2v" and not s.start_from:
                raise PlanError(f"shot {s.id}: mode 'i2v' requires start_from")
            # missing-dependency rejection: start_from must resolve.
            if s.start_from and s.start_from != "prev_last_frame" and s.start_from not in asset_ids:
                raise PlanError(
                    f"shot {s.id}: start_from {s.start_from!r} is neither 'prev_last_frame' "
                    f"nor a declared asset_task id {sorted(asset_ids)}"
                )

        # Timeline: one entry per shot, all references valid, order = play order.
        known = set(ids)
        seen: set[str] = set()
        for t in self.timeline:
            if t.clip not in known:
                raise PlanError(f"timeline references unknown clip {t.clip!r}")
            if t.clip in seen:
                raise PlanError(f"timeline references clip {t.clip!r} twice")
            if t.transition_in not in ("cut", "dissolve"):
                raise PlanError(
                    f"timeline {t.clip}: transition_in must be 'cut' or 'dissolve' "
                    f"(got {t.transition_in!r})"
                )
            seen.add(t.clip)
        if seen != known:
            missing = known - seen
            raise PlanError(f"timeline missing shots: {sorted(missing)}")

        # asset-DAG ordering: the FIRST shot (play order) can't chain from a previous clip.
        ordered = self.ordered_shots()
        if ordered and ordered[0].start_from == "prev_last_frame":
            raise PlanError(
                f"first shot {ordered[0].id!r} cannot start_from 'prev_last_frame' (no previous clip)"
            )

        if self.delivery.fps <= 0 or self.delivery.width <= 0 or self.delivery.height <= 0:
            raise PlanError("delivery fps/width/height must be > 0")

    def shot_by_id(self, sid: str) -> Shot:
        for s in self.shots:
            if s.id == sid:
                return s
        raise PlanError(f"no shot {sid!r}")

    def asset_task_by_id(self, aid: str):
        for a in self.asset_tasks:
            if a.id == aid:
                return a
        raise PlanError(f"no asset_task {aid!r}")

    def ordered_shots(self) -> list:
        """Shots in timeline (play) order."""
        return [self.shot_by_id(t.clip) for t in self.timeline]

    def character_by_id(self, cid: str):
        for c in self.characters:
            if c.id == cid:
                return c
        raise PlanError(f"no character {cid!r}")

    def character_block(self, ids) -> tuple[str, str]:
        """Compose the (positive, negative) canonical Character-Bible text for `ids`.

        The executor prepends `positive` to the keyframe / Wan shot prompt and adds
        `negative` to the negative prompt where the lane supports it — so the same
        character reads consistently. Empty strings when there are no character ids.
        This is SEMANTIC identity (less drift), not a face lock (that is Tier B).
        """
        pos, neg = [], []
        by_id = {c.id: c for c in self.characters}
        for cid in (ids or []):
            c = by_id.get(cid)
            if not c:
                continue
            head = (c.name + (f" ({c.role})" if c.role else "")).strip()
            bits = [f"{head}: {c.description}".strip(": ").strip()]
            if c.wardrobe.strip():
                bits.append("wearing " + c.wardrobe.strip())
            if c.props.strip():
                bits.append("with " + c.props.strip())
            pos.append(", ".join(b for b in bits if b))
            if c.negative.strip():
                neg.append(c.negative.strip())
        return "; ".join(pos), ", ".join(neg)
