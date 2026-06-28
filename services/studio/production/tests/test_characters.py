"""Offline tests for the Character Bible (Tier A — SEMANTIC identity, no GPU).

Tier A is *less drift*, not a face lock: a canonical character block (description +
wardrobe + props, plus negative-drift notes) is injected by the executor into every
keyframe + Wan shot prompt the character appears in, so the same person reads
consistently. These tests cover the schema, the (positive, negative) composition, the
planner wiring (cast → keyframes), the character_bible.json + manifest provenance, and
self-gating (no characters → no-op). Reference-image conditioning (face lock) is Tier B.

Run:  python3 -m unittest services.studio.production.tests.test_characters -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from ..planner import apply_continuity, extract_json, normalize, plan_from_brief
from ..registry import load
from ..schema import PlanError, ProductionPlanV1
from .test_v0b import StubLLM

_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")

_CAST = {
    "id": "det", "name": "Det. Marlowe", "role": "protagonist",
    "description": "tall weathered man, 40s, square jaw, grey eyes, dark hair",
    "wardrobe": "charcoal trench coat, grey fedora", "props": "a lit cigarette",
    "negative": "clean-shaven, young, bright colors", "seed": 111,
}


def _plan(**over):
    d = {
        "project": {"title": "Neon Noir", "video_lane": "wan", "continuity": "none"},
        "characters": [dict(_CAST)],
        "shots": [{"id": "s1", "lane": "wan", "mode": "t2v", "target_seconds": 5,
                   "prompt_intent": "Marlowe under a streetlight", "characters": ["det"]}],
        "timeline": [{"clip": "s1", "transition_in": "dissolve"}],
    }
    d.update(over)
    return d


# director-style plan (characters + per-shot cast) for the planner/StubLLM path
GOOD_PLAN_CHARS = json.dumps({
    "project": {"title": "Noir", "tone": "moody", "video_lane": "wan"},
    "characters": [dict(_CAST)],
    "shots": [
        {"id": "s1", "target_seconds": 5, "prompt_intent": "streetlight", "characters": ["det"], "narration": "one"},
        {"id": "s2", "target_seconds": 5, "prompt_intent": "smoky bar", "characters": ["det"], "narration": "two"},
    ],
    "music": {"lane": "ace-step", "tags": "noir", "lyrics": "[instrumental]", "seed": 7},
    "timeline": [{"clip": "s1", "transition_in": "dissolve"}, {"clip": "s2", "transition_in": "dissolve"}],
})


class TestCharacterSchema(unittest.TestCase):
    def test_valid_plan_with_cast(self):
        p = ProductionPlanV1.from_dict(_plan())
        self.assertEqual([c.name for c in p.characters], ["Det. Marlowe"])
        self.assertEqual(p.shots[0].characters, ["det"])
        self.assertIsNone(p.characters[0].reference_asset_id)   # Tier B field reserved, unused

    def test_duplicate_character_ids_rejected(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_plan(characters=[dict(_CAST), dict(_CAST)]))

    def test_unknown_shot_character_rejected(self):
        d = _plan()
        d["shots"][0]["characters"] = ["ghost"]
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(d)

    def test_empty_description_rejected(self):
        c = dict(_CAST, description="")
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_plan(characters=[c]))

    def test_dangling_reference_asset_id_rejected(self):
        # reserved Tier B field — if set, must point at a declared asset_task
        c = dict(_CAST, reference_asset_id="nope")
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_plan(characters=[c]))


class TestCharacterBlock(unittest.TestCase):
    def test_positive_and_negative_compose(self):
        p = ProductionPlanV1.from_dict(_plan())
        pos, neg = p.character_block(["det"])
        self.assertIn("Det. Marlowe (protagonist)", pos)
        self.assertIn("charcoal trench coat", pos)        # wardrobe
        self.assertIn("a lit cigarette", pos)             # props
        self.assertEqual(neg, "clean-shaven, young, bright colors")

    def test_empty_for_no_ids(self):
        p = ProductionPlanV1.from_dict(_plan())
        self.assertEqual(p.character_block([]), ("", ""))
        self.assertEqual(p.character_block(None), ("", ""))


class TestPlannerWiresCast(unittest.TestCase):
    def test_storyboard_copies_cast_to_keyframes(self):
        data = apply_continuity(normalize(extract_json(GOOD_PLAN_CHARS), video_lane="wan"),
                                "storyboard", image_lane="chroma")
        plan = ProductionPlanV1.from_dict(data)
        self.assertEqual([a.characters for a in plan.asset_tasks], [["det"], ["det"]])

    def test_hero_keyframe_gets_union_cast(self):
        data = apply_continuity(normalize(extract_json(GOOD_PLAN_CHARS), video_lane="wan"),
                                "hero", image_lane="chroma")
        plan = ProductionPlanV1.from_dict(data)
        self.assertEqual(plan.asset_tasks[0].characters, ["det"])   # union across shots

    def test_planner_threads_characters(self):
        plan, _ = plan_from_brief("b", load(), llm=StubLLM(["t", GOOD_PLAN_CHARS]), continuity="storyboard")
        self.assertEqual([c.id for c in plan.characters], ["det"])
        self.assertTrue(all(a.characters == ["det"] for a in plan.asset_tasks))


class TestNegativeInjection(unittest.TestCase):
    def test_chroma_and_hidream_take_text_negative(self):
        import json as _json
        import os as _os
        from .. import config, lanes
        for lane, probe in (("chroma", lambda wf: wf["neg"]["inputs"]["text"]),
                            ("hidream", lambda wf: wf["cond"]["inputs"]["negative_prompt"])):
            with open(_os.path.join(config.WORKFLOW_DIR, lanes._IMAGE_WORKFLOWS[lane])) as fh:
                wf = _json.load(fh)
            lanes._apply_image_negative(wf, lane, "no hats")
            self.assertIn("no hats", probe(wf))

    def test_turbo_lanes_are_noop(self):
        # zimage/krea feed neg from a conditioning node (cfg=1 ignores it) → no text neg
        import json as _json
        import os as _os
        from .. import config, lanes
        for lane in ("zimage", "krea"):
            with open(_os.path.join(config.WORKFLOW_DIR, lanes._IMAGE_WORKFLOWS[lane])) as fh:
                wf = _json.load(fh)
            before = _json.dumps(wf)
            lanes._apply_image_negative(wf, lane, "no hats")   # must not raise / mutate
            self.assertEqual(_json.dumps(wf), before)


class TestCharacterBibleEndToEnd(unittest.TestCase):
    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_renders_with_bible_and_provenance(self):
        from ..executor import run_production
        data = apply_continuity(normalize(extract_json(GOOD_PLAN_CHARS), video_lane="wan"),
                                "storyboard", image_lane="chroma")
        plan = ProductionPlanV1.from_dict(data)
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="noir",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            self.assertTrue(s["manifest"]["exit_criteria"]["all_validators_pass"])
            self.assertEqual(s["manifest"]["characters"], [{"id": "det", "name": "Det. Marlowe",
                                                            "role": "protagonist"}])
            bible = os.path.join(td, "noir", "character_bible.json")
            self.assertTrue(os.path.isfile(bible))
            self.assertEqual(json.load(open(bible))[0]["name"], "Det. Marlowe")
            self.assertIn("character_bible", {a["role"] for a in s["manifest"]["artifacts"]})

    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_self_gating_no_characters(self):
        # a character-less brief writes no bible and injects nothing (no-op)
        from ..executor import run_production
        d = _plan(characters=[])
        d["shots"][0].pop("characters", None)
        d["shots"][0]["narration"] = "a quiet city at night"   # give the mix an audio track
        plan = ProductionPlanV1.from_dict(d)
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="plain",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            self.assertEqual(s["manifest"]["characters"], [])
            self.assertFalse(os.path.exists(os.path.join(td, "plain", "character_bible.json")))


if __name__ == "__main__":
    unittest.main()
