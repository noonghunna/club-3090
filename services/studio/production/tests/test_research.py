"""Offline tests for the SearXNG research client (services/studio/production/research.py).
The network is mocked — no live SearXNG, no GPU.

Run:  python3 -m unittest services.studio.production.tests.test_research -v
"""
from __future__ import annotations

import io
import json
import unittest
from unittest import mock

from .. import research


def _fake_urlopen(payload):
    class _Ctx:
        def __enter__(self):
            return io.BytesIO(json.dumps(payload).encode())

        def __exit__(self, *a):
            return False
    return _Ctx()


class TestWebSearch(unittest.TestCase):
    def test_parses_dedups_skips_empty_and_caps(self):
        payload = {"results": [
            {"title": "A", "content": "fact one about 1947 partition", "url": "u1"},
            {"title": "A2", "content": "fact one about 1947 partition", "url": "u2"},  # dup prefix → skip
            {"title": "B", "content": "fact two about the 1971 war", "url": "u3"},
            {"title": "C", "content": "", "url": "u4"},                                # empty → skip
            {"title": "D", "content": "fact three about 1958", "url": "u5"},
        ]}
        with mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(payload)):
            out = research.web_search("pakistan", n=2)
        self.assertEqual([r["content"] for r in out],
                         ["fact one about 1947 partition", "fact two about the 1971 war"])

    def test_fails_open_on_network_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("searxng down")):
            self.assertEqual(research.web_search("q"), [])

    def test_fails_open_on_bad_json(self):
        class _Bad:
            def __enter__(self):
                return io.BytesIO(b"not json")

            def __exit__(self, *a):
                return False
        with mock.patch("urllib.request.urlopen", return_value=_Bad()):
            self.assertEqual(research.web_search("q"), [])


class TestNotes(unittest.TestCase):
    def test_format_notes(self):
        self.assertEqual(research.format_notes([]), "")
        s = research.format_notes([{"title": "BBC", "content": "In 1947 the subcontinent was partitioned", "url": "u"}])
        self.assertIn("RESEARCH NOTES", s)
        self.assertIn("1947 the subcontinent was partitioned", s)
        self.assertIn("[BBC]", s)

    def test_research_notes_empty_when_no_results(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError):
            self.assertEqual(research.research_notes("a documentary on pakistan"), "")

    def test_research_notes_formats_live_shape(self):
        payload = {"results": [{"title": "Wiki", "content": "On October 7, 1958 a military coup occurred", "url": "u"}]}
        with mock.patch("urllib.request.urlopen", return_value=_fake_urlopen(payload)):
            notes = research.research_notes("pakistan")
        self.assertIn("October 7, 1958", notes)


class TestPlannerResearchIntegration(unittest.TestCase):
    """use_research grounds a DOCUMENTARY plan in real facts; narrative / off → no research."""

    def setUp(self):
        from ..registry import load
        self.reg = load()

    def test_documentary_research_grounds_both_stages(self):
        from ..planner import plan_from_brief
        from .test_v0b import GOOD_PLAN, StubLLM
        llm = StubLLM(["a factual treatment", GOOD_PLAN])
        with mock.patch.object(research, "research_notes",
                               return_value="RESEARCH NOTES: In 1947 the subcontinent was partitioned") as m:
            plan, arts = plan_from_brief("a documentary on the history of pakistan", self.reg,
                                         llm=llm, use_research=True)
        m.assert_called_once()
        self.assertIn("research", [a.role for a in arts])              # provenance recorded
        self.assertIn("RESEARCH NOTES", llm.calls[0][1]["content"])    # treatment user grounded
        self.assertIn("RESEARCH NOTES", llm.calls[1][1]["content"])    # plan user grounded

    def test_no_research_for_narrative_or_when_off(self):
        from ..planner import plan_from_brief
        from .test_v0b import GOOD_PLAN, StubLLM
        with mock.patch.object(research, "research_notes") as m:
            plan_from_brief("a 15s noir short", self.reg, llm=StubLLM(["t", GOOD_PLAN]), use_research=True)   # narrative
            plan_from_brief("a documentary on pakistan", self.reg, llm=StubLLM(["t", GOOD_PLAN]), use_research=False)  # off
        m.assert_not_called()

    def test_research_fail_open_still_plans(self):
        # research returns '' (search down) → plan still produced, no research provenance
        from ..planner import plan_from_brief
        from .test_v0b import GOOD_PLAN, StubLLM
        with mock.patch.object(research, "research_notes", return_value=""):
            plan, arts = plan_from_brief("a documentary on pakistan", self.reg,
                                         llm=StubLLM(["t", GOOD_PLAN]), use_research=True)
        self.assertNotIn("research", [a.role for a in arts])
        self.assertEqual(len(plan.shots), 2)


if __name__ == "__main__":
    unittest.main()
