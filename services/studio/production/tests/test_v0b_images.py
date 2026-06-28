"""Offline v0b-images tests — the asset-DAG + both continuity modes (synthetic).

Covers the user's asks: asset-DAG ordering (hero generated before the shots that
consume it) and missing-dependency rejection. The synthetic backend renders i2v as
a clip that literally begins from the start frame, so chain/hero continuity is
exercised end-to-end offline (no GPU).

Run:  python3 -m unittest services.studio.production.tests.test_v0b_images -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from ..schema import PlanError, ProductionPlanV1

_PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _load(name, mut=None):
    with open(os.path.join(_PROD, "plans", name)) as f:
        d = json.load(f)
    if mut:
        mut(d)
    return d


class TestAssetDAGSchema(unittest.TestCase):
    def test_chain_plan_valid(self):
        p = ProductionPlanV1.from_dict(_load("lighthouse_chain.json"))
        self.assertEqual(p.project.continuity, "chain")
        self.assertEqual([s.mode for s in p.ordered_shots()], ["t2v", "i2v", "i2v"])
        self.assertEqual(p.ordered_shots()[1].start_from, "prev_last_frame")

    def test_hero_plan_valid(self):
        p = ProductionPlanV1.from_dict(_load("lighthouse_hero.json"))
        self.assertEqual(p.project.continuity, "hero")
        self.assertEqual([a.id for a in p.asset_tasks], ["hero"])
        self.assertTrue(all(s.start_from == "hero" for s in p.shots))

    def test_missing_asset_dependency_rejected(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(
                "lighthouse_hero.json", lambda d: d["shots"][0].update(start_from="nope")))

    def test_i2v_without_start_from_rejected(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(
                "lighthouse_chain.json", lambda d: d["shots"][1].pop("start_from")))

    def test_first_shot_cannot_chain(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(
                "lighthouse_chain.json",
                lambda d: d["shots"][0].update(mode="i2v", start_from="prev_last_frame")))

    def test_bad_continuity_rejected(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(
                "lighthouse_chain.json", lambda d: d["project"].update(continuity="morph")))


class TestContinuityE2E(unittest.TestCase):
    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_chain_mode_renders_with_handoff_frames(self):
        from ..executor import run_production
        from ..validators import ffprobe
        plan = ProductionPlanV1.from_dict(_load("lighthouse_chain.json"))
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="chain",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            self.assertTrue(s["manifest"]["exit_criteria"]["all_validators_pass"])
            self.assertTrue(ffprobe(s["final"]).has_audio)
            # the i2v hand-off frames were extracted for the chained shots
            self.assertTrue(os.path.isfile(os.path.join(td, "chain", "assets", "s2_start.png")))
            self.assertTrue(os.path.isfile(os.path.join(td, "chain", "assets", "s3_start.png")))

    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_hero_mode_asset_dag_orders(self):
        from ..executor import run_production
        plan = ProductionPlanV1.from_dict(_load("lighthouse_hero.json"))
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="hero",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            self.assertTrue(s["manifest"]["exit_criteria"]["all_validators_pass"])
            # pre-production generated the hero BEFORE the shots that consume it
            self.assertTrue(os.path.isfile(os.path.join(td, "hero", "assets", "hero.png")))
            roles = {a["role"] for a in s["manifest"]["artifacts"]}
            self.assertIn("hero_keyframe", roles)


class TestStoryboardMode(unittest.TestCase):
    def test_storyboard_plan_valid(self):
        p = ProductionPlanV1.from_dict(_load("lighthouse_storyboard.json"))
        self.assertEqual(p.project.continuity, "storyboard")
        # one keyframe per shot, each shot i2v from ITS OWN keyframe
        self.assertEqual([a.id for a in p.asset_tasks], ["kf_s1", "kf_s2", "kf_s3"])
        self.assertEqual([s.start_from for s in p.shots], ["kf_s1", "kf_s2", "kf_s3"])
        self.assertTrue(all(s.mode == "i2v" for s in p.shots))

    def test_planner_authors_storyboard(self):
        from ..planner import plan_from_brief
        from ..registry import load
        from .test_v0b import GOOD_PLAN, StubLLM
        plan, _ = plan_from_brief("b", load(), llm=StubLLM(["t", GOOD_PLAN]), continuity="storyboard")
        self.assertEqual(plan.project.continuity, "storyboard")
        # GOOD_PLAN has 2 shots -> 2 per-shot keyframes; each shot starts from its own
        self.assertEqual(len(plan.asset_tasks), len(plan.shots))
        for s in plan.shots:
            self.assertEqual(s.start_from, f"kf_{s.id}")
            self.assertEqual(s.mode, "i2v")
        # shared style bible prefixes every keyframe prompt
        self.assertTrue(all("cohesive cinematic style" in a.prompt for a in plan.asset_tasks))

    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_storyboard_renders_per_shot_keyframes(self):
        from ..executor import run_production
        plan = ProductionPlanV1.from_dict(_load("lighthouse_storyboard.json"))
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="sb",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            self.assertTrue(s["manifest"]["exit_criteria"]["all_validators_pass"])
            for kf in ("kf_s1", "kf_s2", "kf_s3"):
                self.assertTrue(os.path.isfile(os.path.join(td, "sb", "assets", kf + ".png")))


class TestPlannerContinuity(unittest.TestCase):
    """The planner deterministically wires continuity (4B stays creative)."""
    def setUp(self):
        from ..registry import load
        self.reg = load()

    def test_planner_applies_chain(self):
        from ..planner import plan_from_brief
        from .test_v0b import GOOD_PLAN, StubLLM
        plan, _ = plan_from_brief("b", self.reg, llm=StubLLM(["t", GOOD_PLAN]), continuity="chain")
        self.assertEqual(plan.project.continuity, "chain")
        self.assertEqual(plan.shots[0].mode, "t2v")
        self.assertTrue(all(s.mode == "i2v" and s.start_from == "prev_last_frame"
                            for s in plan.shots[1:]))

    def test_planner_applies_hero(self):
        from ..planner import plan_from_brief
        from .test_v0b import GOOD_PLAN, StubLLM
        plan, _ = plan_from_brief("b", self.reg, llm=StubLLM(["t", GOOD_PLAN]), continuity="hero")
        self.assertEqual(plan.project.continuity, "hero")
        self.assertEqual([a.id for a in plan.asset_tasks], ["hero"])
        self.assertTrue(all(s.mode == "i2v" and s.start_from == "hero" for s in plan.shots))
        self.assertEqual(plan.project.image_policy.get("hero_keyframe_lane"), "chroma")


if __name__ == "__main__":
    unittest.main()
