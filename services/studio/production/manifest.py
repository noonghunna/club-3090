"""Typed, extensible production manifest.

Records (a) run-level provenance — registry/workflow versions, seeds, the exact
ffmpeg command, delivery profile — so a whole run reproduces, and (b) per-artifact
records discriminated by `type`. v0a only writes `type="media"` records, but the
shape already admits `type="llm_prompt"` so v0b prompt-provenance slots in with no
redesign.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Artifact:
    id: str
    type: str            # "media" | "llm_prompt"  (v0a writes only "media")
    role: str            # "shot" | "narration" | "music_bed" | "final"
    path: str            # relative to the production dir
    lane: str = ""
    seed: int = 0
    prompt_hash: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    validation: dict = field(default_factory=dict)


@dataclass
class Manifest:
    job_id: str
    title: str
    created_utc: str
    backend: str                          # "live" | "synthetic"
    schema_version: str = "ProductionPlanV1"
    stack: dict = field(default_factory=dict)               # operator-chosen lanes (stack.ProductionStack)
    delivery: dict = field(default_factory=dict)
    workflow_versions: dict = field(default_factory=dict)   # {"wan": sha, "ace-step": sha}
    seeds: list = field(default_factory=list)
    ffmpeg_cmds: list = field(default_factory=list)         # exact assemble commands
    artifacts: list = field(default_factory=list)           # list[Artifact]
    final: str = ""
    exit_criteria: dict = field(default_factory=dict)

    def add(self, art: Artifact) -> Artifact:
        self.artifacts.append(art)
        return art

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifacts"] = [asdict(a) if isinstance(a, Artifact) else a for a in self.artifacts]
        return d

    def write(self, prod_dir: str) -> str:
        path = os.path.join(prod_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @staticmethod
    def read(prod_dir: str) -> dict:
        with open(os.path.join(prod_dir, "manifest.json")) as f:
            return json.load(f)
