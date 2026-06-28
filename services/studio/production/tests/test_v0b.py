"""Offline v0b planner tests (stdlib unittest, no model/GPU).

A stub LLM returns canned director replies so the full planner — prompt
construction, JSON extraction, lenient normalization, and the validator-repair
loop — is exercised without the 4B (mirrors v0a's synthetic backend).

Run:  python3 -m unittest services.studio.production.tests.test_v0b -v
"""
from __future__ import annotations

import unittest

from ..planner import PlannerError, extract_json, normalize, plan_from_brief
from ..registry import audio_lanes, load, prompt_slice, video_lanes

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


class StubLLM:
    """Returns a queue of canned director responses; records the calls."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, *, max_tokens, temperature):
        self.calls.append(messages)
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


if __name__ == "__main__":
    unittest.main()
