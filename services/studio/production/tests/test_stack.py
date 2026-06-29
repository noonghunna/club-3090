"""Offline tests for the operator-chosen production stack (stdlib unittest, no GPU).

Covers the user's ask: the stack (video lane · keyframe lane · continuity · audio) is
explicit, defaulted-but-visible, OVERRIDABLE, validated against what the executor can
render, recorded in the manifest — and the director NEVER silently picks a video model
(the lane pin is forced through normalize/the planner).

Run:  python3 -m unittest services.studio.production.tests.test_stack -v
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from .. import stack as S
from ..planner import normalize, plan_from_brief
from ..registry import load
from ..schema import PlanError, ProductionPlanV1
from .test_v0b import GOOD_PLAN, StubLLM

_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


class TestResolveStack(unittest.TestCase):
    def test_defaults_are_auto_and_visible(self):
        s = S.resolve_stack()
        self.assertEqual((s.video_lane, s.keyframe_lane, s.continuity), ("wan", "chroma", "storyboard"))
        self.assertTrue(s.auto)                      # nothing pinned → auto (still shown)
        self.assertTrue(s.narration and s.music)
        self.assertIn("auto defaults", S.describe_stack(s))

    def test_explicit_pin_clears_auto(self):
        s = S.resolve_stack(video_lane="wan", keyframe_lane="hidream", continuity="hero", music=False)
        self.assertFalse(s.auto)
        self.assertEqual(s.keyframe_lane, "hidream")
        self.assertFalse(s.music)
        self.assertIn("HiDream", S.describe_stack(s))
        self.assertIn("no music", S.describe_stack(s))

    def test_auto_token_and_blank_normalize_to_default(self):
        for v in ("auto", "", "  AUTO  ", None):
            self.assertEqual(S.resolve_stack(video_lane=v).video_lane, "wan")

    def test_unknown_video_lane_rejected(self):
        with self.assertRaises(S.StackError):
            S.resolve_stack(video_lane="nope")

    def test_all_video_lanes_wired_and_resolve(self):
        # wan + the LTX family (ltx/sulphur/10eros) all render via the executor now.
        for lane in ("wan", "ltx", "sulphur", "10eros"):
            self.assertEqual(S.resolve_stack(video_lane=lane).video_lane, lane)
        self.assertEqual(sorted(S.wired_video_lanes()), ["10eros", "ltx", "sulphur", "wan"])

    def test_capabilities_match_wired_video_lanes(self):
        # F8 drift guard: every wired video lane (stack.py) must have a capability contract
        # the planner is shown, else the 4B plans against the wrong lane's physics.
        from ..registry import load, video_lanes
        reg_lanes = set(video_lanes(load()))
        for lane in S.wired_video_lanes():
            self.assertIn(lane, reg_lanes, f"{lane} is wired but missing from capabilities.yaml")

    def test_prompt_slice_shows_only_the_pinned_lane_with_its_physics(self):
        # The planner sees ONLY the pinned lane's real fps/audio — not a Wan-shaped default.
        from ..registry import load, prompt_slice
        reg = load()
        ltx = prompt_slice(reg, video_lane="ltx")
        self.assertIn("'ltx'", ltx)
        self.assertNotIn("'wan'", ltx)               # other lanes are filtered out
        self.assertIn("24fps", ltx)                  # LTX is 24 fps, not Wan's 16
        self.assertNotIn("SILENT", ltx)              # LTX has native audio
        wan = prompt_slice(reg, video_lane="wan")
        self.assertIn("16fps", wan)
        self.assertIn("SILENT", wan)                 # Wan clips are silent

    def test_unknown_keyframe_and_continuity_rejected(self):
        with self.assertRaises(S.StackError):
            S.resolve_stack(keyframe_lane="zzz")
        with self.assertRaises(S.StackError):
            S.resolve_stack(continuity="morph")

    def test_continuity_needs_i2v(self):
        # a hypothetical wired video lane WITHOUT i2v can't serve a continuity mode.
        fake = dict(S.VIDEO_LANES)
        fake["flat"] = {"label": "no-i2v", "wired": True, "i2v": False}
        with mock.patch.dict(S.VIDEO_LANES, fake, clear=True):
            self.assertEqual(S.resolve_stack(video_lane="flat", continuity="none").continuity, "none")
            for c in ("storyboard", "hero", "chain"):
                with self.assertRaises(S.StackError):
                    S.resolve_stack(video_lane="flat", continuity=c)

    def test_wired_keyframe_lanes_resolve(self):
        for k in S.wired_keyframe_lanes():
            self.assertEqual(S.resolve_stack(keyframe_lane=k).keyframe_lane, k)

    def test_keyframe_lanes_are_tiered_and_hidream_is_not_default(self):
        # the everyday default is a fast lane (chroma); hidream is the quality tier,
        # NOT the default — Wan downscales away most of its 2560×1440 advantage.
        self.assertEqual(S.DEFAULT_KEYFRAME, "chroma")
        self.assertEqual(S.KEYFRAME_LANES["chroma"]["tier"], "default")
        self.assertEqual(S.KEYFRAME_LANES["zimage"]["tier"], "fast")
        self.assertEqual(S.KEYFRAME_LANES["hidream"]["tier"], "quality")
        self.assertEqual(S.KEYFRAME_LANES["krea"]["tier"], "aesthetic")
        for k in S.wired_keyframe_lanes():
            self.assertTrue(S.KEYFRAME_LANES[k].get("tier"), f"{k} has no tier")
        self.assertIn("default=chroma", S.lane_help())

    def test_ideogram_keyframe_rejected_needs_json_not_prose(self):
        # ideogram is a design/title-card lane (structured JSON), not a prose keyframe lane.
        self.assertNotIn("ideogram", S.wired_keyframe_lanes())
        with self.assertRaises(S.StackError):
            S.resolve_stack(keyframe_lane="ideogram")


class TestSchemaLaneValidation(unittest.TestCase):
    def _good(self):
        return {
            "project": {"title": "T", "video_lane": "wan", "continuity": "none"},
            "shots": [{"id": "s1", "lane": "wan", "mode": "t2v", "target_seconds": 5,
                       "prompt_intent": "x"}],
            "timeline": [{"clip": "s1", "transition_in": "dissolve"}],
        }

    def test_ltx_family_video_lanes_accepted(self):
        for lane in ("ltx", "sulphur", "10eros"):
            d = self._good()
            d["project"]["video_lane"] = lane
            d["shots"][0]["lane"] = lane
            ProductionPlanV1.from_dict(d)   # wired now → must not raise

    def test_unknown_video_lane_rejected(self):
        d = self._good()
        d["project"]["video_lane"] = "made-up"
        d["shots"][0]["lane"] = "made-up"
        with self.assertRaises(PlanError):
            ProductionPlanV1.from_dict(d)

    def test_unknown_keyframe_lane_rejected(self):
        d = self._good()
        d["shots"][0].update(mode="i2v", start_from="kf")
        d["asset_tasks"] = [{"id": "kf", "role": "storyboard_keyframe", "lane": "zzz",
                             "prompt": "p", "seed": 1, "width": 832, "height": 480}]
        with self.assertRaises(PlanError) as cm:
            ProductionPlanV1.from_dict(d)
        self.assertIn("keyframe lane", str(cm.exception))

    def test_wired_keyframe_lanes_accepted(self):
        for k in S.wired_keyframe_lanes():
            d = self._good()
            d["shots"][0].update(mode="i2v", start_from="kf")
            d["asset_tasks"] = [{"id": "kf", "role": "storyboard_keyframe", "lane": k,
                                 "prompt": "p", "seed": 1, "width": 832, "height": 480}]
            ProductionPlanV1.from_dict(d)   # must not raise


class TestPlannerHonoursStack(unittest.TestCase):
    def setUp(self):
        self.reg = load()

    def test_keyframe_lane_override_flows_to_assets(self):
        st = S.resolve_stack(keyframe_lane="hidream", continuity="storyboard")
        plan, _ = plan_from_brief("b", self.reg, llm=StubLLM(["t", GOOD_PLAN]), stack=st)
        self.assertTrue(plan.asset_tasks)
        self.assertTrue(all(a.lane == "hidream" for a in plan.asset_tasks))
        self.assertEqual(plan.project.image_policy.get("storyboard_keyframe_lane"), "hidream")

    def test_music_off_drops_bed(self):
        st = S.resolve_stack(music=False, continuity="none")
        plan, _ = plan_from_brief("b", self.reg, llm=StubLLM(["t", GOOD_PLAN]), stack=st)
        self.assertIsNone(plan.music)

    def test_narration_off_strips_voiceover(self):
        st = S.resolve_stack(narration=False, continuity="none")
        plan, _ = plan_from_brief("b", self.reg, llm=StubLLM(["t", GOOD_PLAN]), stack=st)
        self.assertTrue(all(not s.narration.strip() for s in plan.shots))

    def test_normalize_forces_pinned_lane_over_model_output(self):
        # the 4B "chose" ltx; the pin (wan) must win — planner never lets the model pick.
        data = normalize({"project": {"video_lane": "ltx"},
                          "shots": [{"prompt_intent": "x", "lane": "ltx"}]}, video_lane="wan")
        self.assertEqual(data["project"]["video_lane"], "wan")
        self.assertEqual(data["shots"][0]["lane"], "wan")


class TestKeyframeWorkflowWiring(unittest.TestCase):
    """The data-plane: every WIRED keyframe lane has a real workflow + a patch that
    injects prompt/size/seed into nodes that actually exist (no live render needed)."""

    def test_wired_lanes_have_workflow_and_patch(self):
        from .. import lanes
        for lane in S.wired_keyframe_lanes():
            self.assertIn(lane, lanes._IMAGE_WORKFLOWS, f"{lane} missing workflow")
            self.assertIn(lane, lanes._IMAGE_PATCH, f"{lane} missing patch")

    def test_no_unwired_lane_is_wired_in_lanes(self):
        from .. import lanes
        for lane, meta in S.KEYFRAME_LANES.items():
            if not meta["wired"]:
                self.assertNotIn(lane, lanes._IMAGE_WORKFLOWS,
                                 f"{lane} is unwired in stack but present in lanes")

    def test_hidream_dims_clamped_above_node_minimum(self):
        # HiDream-O1's sampler rejects a side < 512 (step 32). Delivery dims (832x480)
        # must be scaled up, not passed through, or ComfyUI 400s the graph.
        from .. import lanes
        w, h = lanes._hidream_dims(832, 480)
        self.assertGreaterEqual(min(w, h), 512)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertGreater(w, h)            # landscape aspect preserved

    def test_patch_injects_into_real_workflow_nodes(self):
        import copy
        import json
        from .. import config, lanes
        for lane in S.wired_keyframe_lanes():
            path = os.path.join(config.WORKFLOW_DIR, lanes._IMAGE_WORKFLOWS[lane])
            self.assertTrue(os.path.isfile(path), f"missing workflow file for {lane}: {path}")
            with open(path) as fh:
                wf = json.load(fh)
            base = copy.deepcopy(wf)
            lanes._IMAGE_PATCH[lane](wf, "a vivid keyframe", 832, 480, 4242)
            # the patch must have CHANGED the graph (i.e. it hit real nodes, not no-ops)
            self.assertNotEqual(wf, base, f"{lane} patch was a no-op (node keys drifted?)")
            blob = json.dumps(wf)
            self.assertIn("a vivid keyframe", blob, f"{lane}: prompt not injected")
            self.assertIn("4242", blob, f"{lane}: seed not injected")


class TestComfyOutputSelection(unittest.TestCase):
    """pick_output prefers a saved `output` artifact over a `temp` preview — the real
    HiDream history (a temp .jpg from the sampler + a SaveImage .png to /output)."""

    def test_prefers_output_over_temp_preview(self):
        from .. import comfy
        outputs = {
            "sampler": {"images": [{"filename": "hidream_o1_abc.jpg", "subfolder": "", "type": "temp"}]},
            "save": {"images": [{"filename": "studio_hidream_00011_.png", "subfolder": "", "type": "output"}]},
        }
        self.assertEqual(comfy.pick_output(outputs, "image"), ("studio_hidream_00011_.png", ""))

    def test_falls_back_to_only_candidate(self):
        from .. import comfy
        outputs = {"save": {"images": [{"filename": "chroma_1.png", "subfolder": "", "type": "output"}]}}
        self.assertEqual(comfy.pick_output(outputs, "image"), ("chroma_1.png", ""))

    def test_none_when_no_match(self):
        from .. import comfy
        self.assertIsNone(comfy.pick_output({}, "image"))


class TestStackInManifest(unittest.TestCase):
    @unittest.skipUnless(_HAVE_FFMPEG, "ffmpeg/ffprobe required")
    def test_explicit_stack_recorded(self):
        import json
        from ..executor import run_production
        _PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(_PROD, "plans", "lighthouse_storyboard.json")) as f:
            plan = ProductionPlanV1.from_dict(json.load(f))
        st = S.resolve_stack(keyframe_lane="hidream", continuity="storyboard", music=False)
        with tempfile.TemporaryDirectory() as td:
            s = run_production(plan, backend_name="synthetic", job_id="m",
                               now_iso="2026-01-01T00:00:00Z", productions_dir=td, stack=st)
            rec = s["manifest"]["stack"]
            self.assertEqual(rec["keyframe_lane"], "hidream")
            self.assertFalse(rec["music"])

    def test_stack_from_plan_reconstructs(self):
        import json
        _PROD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(_PROD, "plans", "lighthouse_hero.json")) as f:
            plan = ProductionPlanV1.from_dict(json.load(f))
        st = S.stack_from_plan(plan)
        self.assertEqual(st.video_lane, "wan")
        self.assertEqual(st.continuity, "hero")
        self.assertEqual(st.keyframe_lane, "chroma")   # from image_policy / asset lane


class TestLtxWorkflows(unittest.TestCase):
    """The shared LTX-family builder used by both the interactive pipe and the executor."""

    def test_family_membership(self):
        from .. import ltx_workflows as L
        self.assertTrue(all(L.is_ltx_family(x) for x in ("ltx", "sulphur", "10eros")))
        self.assertFalse(L.is_ltx_family("wan"))

    def test_render_graph_sizes_at_24fps_native_res(self):
        from .. import ltx_workflows as L
        wf, w, h = L.render_graph("sulphur", prompt="noir alley", seconds=5, seed=7)
        self.assertEqual((w, h), (1280, 720))                 # native dev res (not Wan 832×480)
        self.assertEqual(wf["5"]["inputs"]["text"], "noir alley")
        self.assertEqual(wf["16"]["inputs"]["noise_seed"], 7)
        self.assertEqual(wf["10"]["inputs"]["value"], 120)    # 5 s × 24 fps

    def test_i2v_conditions_on_image(self):
        from .. import ltx_workflows as L
        wf, w, h = L.render_graph("ltx", prompt="x", seconds=4, seed=1, mode="i2v", image_name="kf.png")
        self.assertEqual((w, h), (768, 512))
        self.assertEqual(wf["100"]["inputs"]["image"], "kf.png")
        self.assertEqual(wf["10"]["inputs"]["value"], 96)     # 4 s × 24 fps

    def test_build_workflows_has_all_six(self):
        from .. import ltx_workflows as L
        self.assertEqual(sorted(L.build_workflows()),
                         sorted(["ltx-t2v", "ltx-i2v", "sulphur-t2v", "sulphur-i2v",
                                 "10eros-t2v", "10eros-i2v"]))


if __name__ == "__main__":
    unittest.main()
