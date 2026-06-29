"""Offline v0b planner tests (stdlib unittest, no model/GPU).

A stub LLM returns canned director replies so the full planner — prompt
construction, JSON extraction, lenient normalization, and the validator-repair
loop — is exercised without the 4B (mirrors v0a's synthetic backend).

Run:  python3 -m unittest services.studio.production.tests.test_v0b -v
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from ..planner import PlannerError, extract_json, normalize, plan_from_brief
from ..registry import audio_lanes, load, prompt_slice, video_lanes

_PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLAN = os.path.join(_PROD, "plans", "lighthouse_3shot.json")
_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")

GOOD_PLAN = """{
  "project": {"title": "Test", "tone": "calm", "target_seconds": 10, "video_lane": "wan"},
  "shots": [
    {"id": "s1", "lane": "wan", "mode": "t2v", "target_seconds": 5, "prompt_intent": "a", "narration": "one", "seed": 1},
    {"id": "s2", "lane": "wan", "mode": "t2v", "target_seconds": 5, "prompt_intent": "b", "narration": "two", "seed": 2}
  ],
  "music": {"lane": "ace-step", "tags": "calm, ambient", "lyrics": "[instrumental]", "seed": 3},
  "timeline": [
    {"clip": "s1", "transition_in": "dissolve"},
    {"clip": "s2", "transition_in": "dissolve"}
  ]
}"""

# A documentary brief where the 4B disobeyed and snuck in a fictional protagonist
# (the "Arif" failure, 2026-06-29) — the planner must strip it for a documentary.
DOC_PLAN_WITH_SNEAKY_CHARACTER = """{
  "project": {"title": "The Architect's Line", "tone": "reflective", "target_seconds": 10, "video_lane": "wan"},
  "characters": [{"id": "p", "name": "Arif", "role": "protagonist", "description": "young man", "seed": 9}],
  "shots": [
    {"id": "s1", "lane": "wan", "mode": "t2v", "target_seconds": 5, "prompt_intent": "ancient ruins of Mohenjo-daro", "characters": ["p"], "narration": "The Indus Valley civilization thrived here.", "seed": 1},
    {"id": "s2", "lane": "wan", "mode": "t2v", "target_seconds": 5, "prompt_intent": "1947 partition crowds", "characters": ["p"], "narration": "In 1947 the subcontinent was partitioned.", "seed": 2}
  ],
  "music": {"lane": "ace-step", "tags": "documentary", "lyrics": "[instrumental]", "seed": 3},
  "timeline": [
    {"clip": "s1", "transition_in": "dissolve"},
    {"clip": "s2", "transition_in": "dissolve"}
  ]
}"""


class StubLLM:
    """Returns a queue of canned director responses; records the calls."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, *, max_tokens, temperature, **kwargs):
        self.calls.append(messages)   # **kwargs absorbs enable_thinking (critic stage)
        return self.responses.pop(0)


class TestRegistry(unittest.TestCase):
    def test_load_and_slice(self):
        reg = load()
        self.assertIn("wan", video_lanes(reg))
        self.assertIn("kokoro", audio_lanes(reg))
        self.assertIn("ace-step", audio_lanes(reg))
        s = prompt_slice(reg)
        self.assertIn("wan", s)
        self.assertIn("RULES", s)
        self.assertIn("Pin ONE video_lane", s)   # the pinning rule is handed to the planner


class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.reg = load()

    def test_happy_path(self):
        llm = StubLLM(["a short treatment", GOOD_PLAN])
        plan, arts = plan_from_brief("a 10s calm clip", self.reg, llm=llm)
        self.assertEqual([s.id for s in plan.shots], ["s1", "s2"])
        self.assertEqual(plan.project.video_lane, "wan")
        self.assertEqual(len(llm.calls), 2)                 # treatment + plan, no repair
        self.assertEqual([a.type for a in arts], ["llm_prompt", "llm_prompt"])
        self.assertEqual([a.role for a in arts], ["treatment", "plan"])

    def test_fenced_and_trailing_chatter(self):
        llm = StubLLM(["t", "```json\n" + GOOD_PLAN + "\n```\nhope that helps!"])
        plan, _ = plan_from_brief("b", self.reg, llm=llm)
        self.assertEqual(len(plan.shots), 2)

    def test_repair_loop_recovers(self):
        bad = '{"project": {"video_lane": "ltx"}, "shots": []}'   # invalid: not wan, no shots
        llm = StubLLM(["t", bad, GOOD_PLAN])
        plan, arts = plan_from_brief("b", self.reg, llm=llm)
        self.assertEqual(len(plan.shots), 2)
        self.assertEqual(len(llm.calls), 3)                 # treatment + plan + 1 repair
        self.assertEqual(arts[-1].role, "repair_1")

    def test_repair_exhausted_raises(self):
        llm = StubLLM(["t", "nonsense", "still no json", "nope", "still nope"])
        with self.assertRaises(PlannerError):
            plan_from_brief("b", self.reg, llm=llm, max_repairs=3)

    def test_normalize_fills_defaults(self):
        data = normalize({"project": {"video_lane": "wan"},
                          "shots": [{"prompt_intent": "x", "narration": "n"}]})
        self.assertEqual(data["shots"][0]["id"], "s1")
        self.assertEqual(data["shots"][0]["lane"], "wan")
        self.assertEqual(data["shots"][0]["mode"], "t2v")
        self.assertTrue(data["timeline"])                   # auto-built from shots
        self.assertIn("music", data)
        self.assertEqual(data["delivery"]["width"], 832)

    def test_extract_json_balanced(self):
        self.assertEqual(extract_json('prefix {"a": {"b": 1}} suffix')["a"]["b"], 1)
        with self.assertRaises(ValueError):
            extract_json("no json here")


class TestProvenanceNotCountedAsValidator(unittest.TestCase):
    """Regression: llm_prompt provenance must NOT fail all_validators_pass.

    (Live v0b run #1 reported rc=1 because the planner's prompt records — which carry
    no media 'ok' — were counted as failed validators.)
    """
    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_llm_prompt_excluded_from_exit_criteria(self):
        from ..executor import run_production
        from ..manifest import Artifact
        from ..schema import ProductionPlanV1
        with open(_PLAN) as f:
            plan = ProductionPlanV1.from_dict(json.load(f))
        prov = [Artifact(id="prompt.plan", type="llm_prompt", role="plan",
                         path="prompts/plan.json", validation={"response_hash": "x"})]
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="prov",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td,
                               extra_artifacts=prov)
            ec = s["manifest"]["exit_criteria"]
            self.assertTrue(ec["all_validators_pass"])   # media all pass; provenance excluded
            types = {a["type"] for a in s["manifest"]["artifacts"]}
            self.assertIn("llm_prompt", types)           # provenance still carried
            self.assertIn("media", types)


class TestDeriveShots(unittest.TestCase):
    """Size the film from the brief's stated duration (~5s/shot), not a fixed test count."""

    def test_minutes_and_seconds(self):
        from ..planner import derive_shots
        self.assertEqual(derive_shots("a 1 minute video on the history of pakistan")[0], 12)
        self.assertEqual(derive_shots("a 45 second noir short")[0], 9)
        self.assertEqual(derive_shots("make a 30s clip")[0], 6)

    def test_caps_runaway_requests(self):
        from ..planner import derive_shots
        self.assertEqual(derive_shots("a 10 minute epic")[0], 24)   # capped, not 120 shots

    def test_default_when_no_duration(self):
        from ..planner import derive_shots
        shots, secs = derive_shots("a noir detective short")
        self.assertEqual(shots, 4)
        self.assertIsNone(secs)


class TestDocumentaryFormat(unittest.TestCase):
    """A documentary/factual brief must NOT become a character-driven fiction — the 4B
    used to invent a protagonist ("Arif") for a documentary on Pakistan's history
    (diagnosed 2026-06-29). Format is detected from the brief, steered in the prompts,
    AND enforced deterministically in normalize so the failure can't recur."""

    def setUp(self):
        self.reg = load()

    def test_detect_format(self):
        from ..prompts import detect_format
        for b in ["a documentary on the history of pakistan",
                  "give me a 2 minute documentory on history of pakistan",   # user's typo
                  "explain how vaccines work", "a guide to the solar system",
                  "the biography of Tesla", "a tutorial on knife sharpening",
                  "an overview of the Roman empire"]:
            self.assertEqual(detect_format(b), "documentary", b)
        for b in ["a 15s noir detective short", "make a 30s noir short, use sulphur",
                  "a lone wanderer crosses a desert at dusk", "two robots fall in love"]:
            self.assertEqual(detect_format(b), "narrative", b)

    def test_documentary_prompt_suppresses_character_bible(self):
        from ..prompts import build_plan_system
        doc = build_plan_system(self.reg, n_shots=6, video_lane="ltx", fmt="documentary")
        self.assertIn("DOCUMENTARY", doc)
        self.assertIn('"characters": []', doc)            # empty cast in the shape
        self.assertNotIn("Detective Marlowe", doc)        # no protagonist example
        self.assertNotIn("CHARACTER BIBLE", doc)
        nar = build_plan_system(self.reg, n_shots=6, video_lane="ltx", fmt="narrative")
        self.assertIn("CHARACTER BIBLE", nar)             # narrative keeps the bible
        self.assertIn("Detective Marlowe", nar)

    def test_normalize_strips_characters_in_documentary(self):
        data = normalize(
            {"project": {"video_lane": "wan"},
             "characters": [{"id": "p", "name": "Arif"}],
             "shots": [{"prompt_intent": "ruins", "characters": ["p"], "narration": "n"}]},
            fmt="documentary")
        self.assertEqual(data["characters"], [])
        self.assertEqual(data["shots"][0]["characters"], [])
        self.assertEqual(data["project"]["format"], "documentary")

    def test_normalize_keeps_characters_in_narrative(self):
        data = normalize(
            {"project": {"video_lane": "wan"},
             "characters": [{"id": "p", "name": "Marlowe"}],
             "shots": [{"prompt_intent": "alley", "characters": ["p"], "narration": "n"}]},
            fmt="narrative")
        self.assertEqual(data["characters"], [{"id": "p", "name": "Marlowe"}])
        self.assertEqual(data["shots"][0]["characters"], ["p"])
        self.assertEqual(data["project"]["format"], "narrative")

    def test_planner_routes_documentary_brief_and_strips_protagonist(self):
        # End-to-end: a doc brief + a disobedient 4B that emitted "Arif" → stripped.
        llm = StubLLM(["a factual treatment", DOC_PLAN_WITH_SNEAKY_CHARACTER])
        plan, _ = plan_from_brief("a documentary on the history of pakistan", self.reg, llm=llm)
        self.assertEqual(plan.project.format, "documentary")
        self.assertEqual(plan.characters, [])
        self.assertTrue(all(s.characters == [] for s in plan.shots))
        # narration stays factual content the model wrote (we don't rewrite it)
        self.assertIn("1947", " ".join(s.narration for s in plan.shots))


if __name__ == "__main__":
    unittest.main()
