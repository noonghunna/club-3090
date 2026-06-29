"""Offline tests for the conversation-intent classifiers (services/studio/director_intent.py).

This is the logic baked into the OWUI Production lane that decides greeting vs confirm vs
question vs film-brief. It used to live as untestable locals inside the pipe's pipe() method;
extracting it (Codex F13) makes the highest-risk UX logic table-testable. The F1 regression —
"can you make a 30s noir short?" being dropped as a question — is locked down here.

Run:  python3 -m unittest services.studio.production.tests.test_director_intent -v
"""
from __future__ import annotations

import os
import sys
import unittest

# director_intent.py is a sibling of build_studio_pipe.py at services/studio/ (it is injected
# into the pipe at build time). Put that dir on the path so we test the real source.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import director_intent as di  # noqa: E402


class TestGenerationRequest(unittest.TestCase):
    def test_creation_asks_are_generation_requests(self):
        for t in ["make a noir short", "create a one minute video on history of pakistan?",
                  "can you make a 30-second noir short?", "do a 1 minute documentary on Pakistan",
                  "give me a 2 minute documentory on history of pakistan.",
                  "I want a short film about robots", "let's build a trailer", "render a clip of the sea",
                  "put together a 30s montage"]:
            self.assertTrue(di.is_generation_request(t), t)

    def test_non_creation_turns_are_not(self):
        for t in ["hi", "what other options do we have?", "can we do some research first?",
                  "do you have other models?", "thanks", "use ltx", "no music", "go"]:
            self.assertFalse(di.is_generation_request(t), t)

    def test_looks_documentary_matches_detect_format(self):
        # The pipe's looks_documentary must agree with the server's detect_format (no drift).
        from ..prompts import detect_format
        for b in ["a documentary on the history of pakistan", "explain how vaccines work",
                  "a guide to the solar system", "the biography of Tesla",
                  "a 15s noir detective short", "two robots fall in love", "a lighthouse at dawn"]:
            self.assertEqual(di.looks_documentary(b), detect_format(b) == "documentary", b)


class TestClassifiers(unittest.TestCase):
    def test_confirm_and_greeting_are_exact(self):
        for t in ["go", "GO", "yes", "do it", "build it", "ok go", "  go. "]:
            self.assertTrue(di.is_confirm(t), t)
        for t in ["hi", "Hello", "thanks", "test"]:
            self.assertTrue(di.is_greeting(t), t)
        # a brief that merely contains a confirm word is NOT a bare confirm
        self.assertFalse(di.is_confirm("go make a noir film"))

    def test_question_shape(self):
        for t in ["what other options?", "how long will it take?", "can we research first?"]:
            self.assertTrue(di.is_question(t), t)
        for t in ["a lone detective in the rain", "make a noir short"]:
            self.assertFalse(di.is_question(t), t)

    def test_compound_confirms(self):
        # "ok do it" etc. must read as a confirm so they don't leak into brief detection
        for t in ["ok do it", "yes go ahead", "do it now", "ok go", "yes please", "go on", "sure lets go"]:
            self.assertTrue(di.is_confirm(t), t)
        # NOT bare confirms: a compound confirm+stack, or any real content
        for t in ["go with ltx", "make a noir film", "do some research", "a noir short"]:
            self.assertFalse(di.is_confirm(t), t)


class TestBriefCandidate(unittest.TestCase):
    def test_generation_request_beats_question_shape(self):
        # The F1 fix: a creation ask phrased as a question is STILL a brief.
        for t in ["can you make a 30-second noir short?", "do a 1 minute documentary on Pakistan",
                  "create a one minute video on history of pakistan?"]:
            self.assertTrue(di.is_brief_candidate(t), t)

    def test_chat_and_smalltalk_are_not_briefs(self):
        for t in ["hi", "what other options do we have?", "can we do some research first?", "go"]:
            self.assertFalse(di.is_brief_candidate(t), t)

    def test_plain_brief_is_a_candidate(self):
        self.assertTrue(di.is_brief_candidate("a lone detective walks a rain-slick alley at night"))

    def test_pure_override_is_not_a_brief(self):
        # a short stack tweak ("use ltx") is supplied as pure_override=True by the caller
        self.assertFalse(di.is_brief_candidate("use ltx", pure_override=True))


class TestPickBrief(unittest.TestCase):
    def test_picks_first_real_brief_across_a_conversation(self):
        turns = ["hi", "can we do some research first?", "use ltx",
                 "make a 30s noir short", "go"]
        po = lambda t: t.strip().lower() in ("use ltx",)
        self.assertEqual(di.pick_brief(turns, po), "make a 30s noir short")

    def test_the_real_failing_transcript_now_captures_a_brief(self):
        # Exact shape of the chat that dove into chit-chat with no brief (2026-06-29).
        turns = ["hi", "Do I need to upload any files before starting?",
                 "can we do some research first?",
                 "create a one minute video on history of pakistan?"]
        self.assertEqual(di.pick_brief(turns),
                         "create a one minute video on history of pakistan?")

    def test_all_smalltalk_yields_no_brief(self):
        self.assertEqual(di.pick_brief(["hi", "hello", "what can you do?"]), "")

    def test_documentary_subject_is_a_brief_even_as_a_question(self):
        self.assertTrue(di.is_brief_candidate("dig history of pakistan?"))
        self.assertTrue(di.is_brief_candidate("the history of jazz"))

    def test_the_ok_do_it_transcript_captures_the_real_brief(self):
        # Regression (2026-06-29): "dig history of pakistan?" was dropped as a question and the
        # confirm phrase "ok do it" became the film brief. Now the subject is the brief, and
        # "ok do it" is a confirm.
        turns = ["hi", "can you search the web?", "can you research?",
                 "dig history of pakistan?", "ok do it"]
        self.assertEqual(di.pick_brief(turns), "dig history of pakistan?")
        self.assertTrue(di.is_confirm("ok do it"))


class TestConfirmWord(unittest.TestCase):
    """has_confirm_word corroborates the LLM confirm so a render never fires on a hallucination."""

    def test_start_signals(self):
        for t in ["go", "go with ltx", "yes do it", "render it", "let's go", "ok go", "build it now"]:
            self.assertTrue(di.has_confirm_word(t), t)

    def test_non_starts(self):
        for t in ["use ltx", "what about hidream?", "a noir short", "30 seconds"]:
            self.assertFalse(di.has_confirm_word(t), t)


class TestParseControllerJson(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(di.parse_controller_json('{"intent": "confirm", "confirm": true}')["confirm"], True)
        self.assertEqual(di.parse_controller_json('```json\n{"a": 1}\n```')["a"], 1)
        # leading chatter + trailing text → still pulls the object
        self.assertEqual(di.parse_controller_json('sure! {"a": {"b": 2}} hope that helps')["a"]["b"], 2)

    def test_garbage_is_none(self):
        for bad in ["", "no json here", "{unbalanced", "[1,2,3]"]:
            self.assertIsNone(di.parse_controller_json(bad), bad)


class TestNormalizeDecision(unittest.TestCase):
    def test_valid_decision_keeps_only_lane_picks(self):
        d = di.normalize_decision({
            "intent": "stack", "brief": "a noir short",
            "stack_patch": {"video_lane": "ltx", "keyframe_lane": "hidream", "music": False, "seconds": 30},
            "confirm": False, "reply": "Switching to LTX."})
        self.assertEqual(d["brief"], "a noir short")
        # music/seconds are NOT taken from the LLM (keyword floor owns them) — lanes only
        self.assertEqual(d["stack_patch"], {"video_lane": "ltx", "keyframe_lane": "hidream"})
        self.assertNotIn("music", d["stack_patch"])
        self.assertNotIn("seconds", d["stack_patch"])
        self.assertEqual(d["intent"], "stack")
        self.assertEqual(d["reply"], "Switching to LTX.")

    def test_drops_unknown_lanes_and_bad_types(self):
        d = di.normalize_decision({
            "intent": "stack",
            "stack_patch": {"video_lane": "veo", "keyframe_lane": "dalle", "continuity": "morph",
                            "music": "yes", "seconds": 99999}})
        self.assertEqual(d["stack_patch"], {})   # every invalid value dropped, none coerced

    def test_intent_defaulted_and_confirm_coerced(self):
        self.assertEqual(di.normalize_decision({"brief": "x"})["intent"], "brief")
        self.assertEqual(di.normalize_decision({"brief": ""})["intent"], "smalltalk")
        self.assertEqual(di.normalize_decision({"confirm": 1, "brief": "x"})["confirm"], True)

    def test_none_on_non_dict(self):
        self.assertIsNone(di.normalize_decision(None))
        self.assertIsNone(di.normalize_decision("nope"))

    def test_controller_system_lists_the_valid_slots(self):
        s = di.build_controller_system()
        for tok in ("wan", "ltx", "sulphur", "10eros", "chroma", "hidream", "storyboard", "confirm"):
            self.assertIn(tok, s)


class TestDecideAction(unittest.TestCase):
    """The pipe routes a resolved turn through decide_action — the core branch logic."""

    def test_build_only_when_brief_and_confirmed(self):
        self.assertEqual(di.decide_action("a noir short", True, "confirm"), "build")

    def test_confirm_without_brief_needs_a_brief(self):
        self.assertEqual(di.decide_action("", True, "confirm"), "need_brief")

    def test_no_brief_is_chat(self):
        self.assertEqual(di.decide_action("", False, "smalltalk"), "chat")
        self.assertEqual(di.decide_action("", False, "question"), "chat")

    def test_brief_with_question_or_smalltalk_is_chat(self):
        self.assertEqual(di.decide_action("a noir short", False, "question"), "chat")
        self.assertEqual(di.decide_action("a noir short", False, "smalltalk"), "chat")
        self.assertEqual(di.decide_action("a noir short", False, "cancel"), "chat")

    def test_brief_with_plan_change_shows_proposal(self):
        for intent in ("brief", "revise", "stack"):
            self.assertEqual(di.decide_action("a noir short", False, intent), "proposal", intent)

    def test_confirmed_wins_over_intent(self):
        # a corroborated confirm builds even if the LLM labeled the intent "stack" ("go with ltx")
        self.assertEqual(di.decide_action("a noir short", True, "stack"), "build")


if __name__ == "__main__":
    unittest.main()
