"""Offline tests for the semantic plan critic (services/studio/production/critic.py) and its
integration into the planner's repair loop (Codex F5/F6/F7). No model/GPU — a stub LLM drives
the LLM critique; the deterministic checks need nothing.

Run:  python3 -m unittest services.studio.production.tests.test_critic -v
"""
from __future__ import annotations

import copy
import json
import unittest

from .. import critic
from ..planner import extract_json, plan_from_brief
from ..registry import load
from ..schema import ProductionPlanV1
from ..util import strip_reasoning
from .test_v0b import GOOD_PLAN, StubLLM

_BASE = json.loads(GOOD_PLAN)


def _plan(mutate=None):
    d = copy.deepcopy(_BASE)
    if mutate:
        mutate(d)
    return ProductionPlanV1.from_dict(d)


class TestDeterministicIssues(unittest.TestCase):
    def test_clean_plan_has_no_issues(self):
        self.assertEqual(critic.deterministic_issues(_plan()), [])

    def test_overlong_narration_flagged(self):
        # 5 s shot → budget ~15 words; 25 words trips it.
        long = " ".join(["word"] * 25)
        p = _plan(lambda d: d["shots"][0].update(narration=long))
        issues = critic.deterministic_issues(p)
        self.assertTrue(any("narration" in i and "s1" in i for i in issues), issues)

    def test_near_duplicate_shots_flagged(self):
        same = "a lone detective walks through the rain-slick alley at night"
        def dup(d):
            d["shots"][0]["prompt_intent"] = same
            d["shots"][1]["prompt_intent"] = same
        issues = critic.deterministic_issues(_plan(dup))
        self.assertTrue(any("near-duplicate" in i for i in issues), issues)

    def test_documentary_character_leak_flagged(self):
        def add_char(d):
            d["characters"] = [{"id": "p", "name": "Arif", "role": "protagonist",
                                "description": "young man", "seed": 9}]
            d["shots"][0]["characters"] = ["p"]
        p = _plan(add_char)
        issues = critic.deterministic_issues(p, fmt="documentary")
        self.assertTrue(any("DOCUMENTARY" in i and "Arif" in i for i in issues), issues)
        # the same plan is fine as a narrative
        self.assertEqual([i for i in critic.deterministic_issues(p, fmt="narrative")
                          if "DOCUMENTARY" in i], [])


class TestParseAndNormalizeCritique(unittest.TestCase):
    def test_parse_plain(self):
        self.assertEqual(critic.parse_critique('{"off_topic": [2], "issues": []}')["off_topic"], [2])

    def test_parse_strips_thinking(self):
        raw = '<think>let me consider {fake: 1} carefully</think>\n{"off_topic": [], "issues": []}'
        self.assertEqual(critic.parse_critique(raw), {"off_topic": [], "issues": []})

    def test_parse_garbage_is_none(self):
        for bad in ["", "no json", "{unbalanced"]:
            self.assertIsNone(critic.parse_critique(bad), bad)

    def test_normalize(self):
        # off_topic shot numbers → one issue each
        out = critic.normalize_critique({"off_topic": [1, 3], "issues": []})
        self.assertEqual(len(out), 2)
        self.assertTrue(all("off-topic" in i for i in out), out)
        # explicit issues pass through (blank dropped)
        self.assertEqual(critic.normalize_critique({"off_topic": [], "issues": ["a", " ", "b"]}), ["a", "b"])
        # clean / empty / non-dict → no blocking issues
        self.assertEqual(critic.normalize_critique({"off_topic": [], "issues": []}), [])
        self.assertEqual(critic.normalize_critique({}), [])
        self.assertEqual(critic.normalize_critique(None), [])


class TestCritiqueFailsOpen(unittest.TestCase):
    def test_llm_error_degrades_to_deterministic_only(self):
        def boom(*a, **k):
            raise RuntimeError("director down")
        # a plan with an overlong narration still surfaces the deterministic issue; the LLM error
        # is swallowed (fail-open) rather than raising.
        p = _plan(lambda d: d["shots"][0].update(narration=" ".join(["w"] * 30)))
        issues = critic.critique(p, "brief", "narrative", call=boom)
        self.assertTrue(issues and all("narration" in i for i in issues), issues)

    def test_no_call_means_deterministic_only(self):
        self.assertEqual(critic.critique(_plan(), "brief", "narrative", call=None), [])


class TestExtractJsonStripsThinking(unittest.TestCase):
    def test_thinking_block_with_decoy_json_is_ignored(self):
        raw = '<think>maybe {"project": {"title": "WRONG"}}</think>\n' + GOOD_PLAN
        self.assertEqual(extract_json(raw)["project"]["title"], "Test")

    def test_strip_reasoning_unclosed(self):
        # an unclosed <think> (truncated output) → drop the tag, keep what follows so the JSON
        # remains findable by extract_json.
        out = strip_reasoning('<think>reasoning... {"a":1}')
        self.assertNotIn("<think>", out)
        self.assertIn('{"a":1}', out)


class TestPlannerCriticIntegration(unittest.TestCase):
    def setUp(self):
        self.reg = load()

    def test_critic_off_by_default_no_extra_calls(self):
        llm = StubLLM(["treatment", GOOD_PLAN])
        plan, _ = plan_from_brief("a 10s calm clip", self.reg, llm=llm)   # use_critic defaults False
        self.assertEqual(len(llm.calls), 2)                              # treatment + plan, no critique

    def test_critic_passes_a_clean_plan(self):
        llm = StubLLM(["treatment", GOOD_PLAN, '{"ok": true, "issues": []}'])
        plan, arts = plan_from_brief("a 10s calm clip", self.reg, llm=llm, use_critic=True)
        self.assertEqual(len(llm.calls), 3)                              # treatment + plan + critique
        self.assertEqual(len(plan.shots), 2)
        self.assertNotIn("critic_repair", [a.role for a in arts])

    def test_critic_triggers_one_repair_then_ships(self):
        # plan #1 is flagged by the critic → one repair → plan #2 ships (no second critique).
        llm = StubLLM(["treatment", GOOD_PLAN,
                       '{"ok": false, "issues": ["shot s1 is off-topic"]}', GOOD_PLAN])
        plan, arts = plan_from_brief("a 10s calm clip", self.reg, llm=llm, use_critic=True)
        roles = [a.role for a in arts]
        self.assertIn("critic", roles)
        self.assertIn("critic_repair_1", roles)
        self.assertEqual(len(llm.calls), 4)        # treatment + plan + critique + critic-repair
        self.assertEqual(len(plan.shots), 2)


if __name__ == "__main__":
    unittest.main()
