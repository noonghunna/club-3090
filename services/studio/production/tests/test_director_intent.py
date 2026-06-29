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


if __name__ == "__main__":
    unittest.main()
