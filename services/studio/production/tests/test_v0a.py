"""Offline v0a tests (stdlib unittest, no pydantic/pytest, no GPU/services).

Run:  python3 -m unittest services.studio.production.tests.test_v0a -v
      (from the repo root)

The end-to-end test uses the `synthetic` backend (ffmpeg lavfi) so it exercises the
WHOLE pipeline — render -> validate -> tts -> bed -> assemble -> manifest -> a real
MP4 with an audio track — without ComfyUI, TTS, or a GPU.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from .. import assemble, validators
from ..lock import ProductionLock, ProductionLockHeld
from ..schema import PlanError, ProductionPlanV1

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = os.path.join(HERE, "plans", "lighthouse_3shot.json")
HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _load(extra=None):
    with open(PLAN) as f:
        d = json.load(f)
    if extra:
        extra(d)
    return d


class TestSchema(unittest.TestCase):
    def test_valid_plan_loads(self):
        plan = ProductionPlanV1.from_dict(_load())
        self.assertEqual(plan.project.video_lane, "wan")
        self.assertEqual(len(plan.shots), 3)
        self.assertEqual([s.id for s in plan.ordered_shots()], ["s1", "s2", "s3"])

    def test_rejects_non_wan_video_lane(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(lambda d: d["project"].update(video_lane="ltx")))

    def test_rejects_shot_lane_mismatch(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(lambda d: d["shots"][0].update(lane="sulphur")))

    def test_rejects_duplicate_shot_ids(self):
        def dup(d):
            d["shots"][1]["id"] = "s1"
            d["timeline"][1]["clip"] = "s1"
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(dup))

    def test_rejects_timeline_missing_shot(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(lambda d: d["timeline"].pop()))

    def test_rejects_non_t2v_mode(self):
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(_load(lambda d: d["shots"][0].update(mode="i2v")))


class TestAssembleCommand(unittest.TestCase):
    def test_full_mix_graph(self):
        cmd = assemble.build_mix_command(
            "/tmp/final.mp4",
            clips=["a.mp4", "b.mp4", "c.mp4"],
            narrations=[("v1.wav", 200), ("v2.wav", 5000)],
            bed="bed.wav", fps=16, lufs=-14.0, bed_level_db=-18,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertIn("concat=n=3:v=1:a=0[vid]", fc)
        self.assertIn("adelay=200|200", fc)
        self.assertIn("sidechaincompress", fc)        # bed ducked under narration
        self.assertIn("asplit=2[nkey][nmix]", fc)      # narration split, used once each
        self.assertIn("loudnorm=I=-14.0", fc)
        self.assertIn("-shortest", cmd)
        self.assertEqual(cmd[-1], "/tmp/final.mp4")

    def test_bed_only_no_narration(self):
        cmd = assemble.build_mix_command(
            "/tmp/f.mp4", clips=["a.mp4"], narrations=[], bed="bed.wav",
            fps=16, lufs=-14.0, bed_level_db=-18)
        fc = cmd[cmd.index("-filter_complex") + 1]
        self.assertNotIn("sidechaincompress", fc)
        self.assertIn("loudnorm", fc)

    def test_db_to_linear(self):
        self.assertAlmostEqual(assemble.db_to_linear(0), 1.0)
        self.assertAlmostEqual(assemble.db_to_linear(-6), 0.501, places=2)


class TestValidators(unittest.TestCase):
    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_silent_clip_and_audio(self):
        with tempfile.TemporaryDirectory() as td:
            vid = os.path.join(td, "v.mp4")
            wav = os.path.join(td, "a.wav")
            from ..util import sh
            sh(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                "color=c=0x102030:s=320x240:r=16", "-t", "2", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-an", vid])
            sh(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                "sine=frequency=440:duration=2", "-c:a", "pcm_s16le", wav])
            self.assertTrue(validators.run_all(["non_empty", "no_audio_expected", "duration"],
                                               vid, target_seconds=2)["ok"])
            self.assertTrue(validators.run_all(["non_empty", "audio_present"], wav)["ok"])
            # a silent clip must FAIL audio_present
            self.assertFalse(validators.check("audio_present", vid)[0])


class TestLock(unittest.TestCase):
    def test_single_flight(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, ".run.lock")
            a = ProductionLock(p).acquire()
            try:
                with self.assertRaises(ProductionLockHeld):
                    ProductionLock(p).acquire()
            finally:
                a.release()
            ProductionLock(p).acquire().release()   # free again


class TestEndToEndSynthetic(unittest.TestCase):
    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_full_pipeline_offline(self):
        from ..executor import run_production
        with tempfile.TemporaryDirectory() as td:
            plan = ProductionPlanV1.from_dict(_load())
            summary = run_production(plan, backend_name="synthetic", job_id="test-job",
                                     now_iso="2026-01-01T00:00:00Z", productions_dir=td)
            man = summary["manifest"]
            # a real final MP4 with an audio track exists
            self.assertTrue(os.path.isfile(summary["final"]))
            fpr = validators.ffprobe(summary["final"])
            self.assertGreater(fpr.duration, 10)        # ~15s of clips
            self.assertTrue(fpr.has_audio)
            # manifest: typed records, run-level provenance, exit criteria
            self.assertEqual({a["type"] for a in man["artifacts"]}, {"media"})
            roles = sorted({a["role"] for a in man["artifacts"]})
            self.assertEqual(roles, ["final", "music_bed", "narration", "shot"])
            self.assertEqual(len(man["ffmpeg_cmds"]), 1)
            self.assertIn("wan", man["workflow_versions"])
            self.assertTrue(man["exit_criteria"]["all_validators_pass"])
            self.assertTrue(man["exit_criteria"]["final_has_audio"])
            # manifest.json was written + reloads
            with open(os.path.join(summary["prod_dir"], "manifest.json")) as f:
                reloaded = json.load(f)
            self.assertEqual(reloaded["job_id"], "test-job")


if __name__ == "__main__":
    unittest.main()
