"""v0a CLI: brief plan -> finished MP4.

    python -m services.studio.production.run PLAN.json [--backend live|synthetic]

CLI/admin only, single-flight (a host file-lock). `live` drives ComfyUI + Kokoro on
the rig; `synthetic` uses ffmpeg lavfi stand-ins so the whole pipeline runs offline.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

from . import config
from .executor import run_production
from .lock import ProductionLock, ProductionLockHeld
from .schema import PlanError, ProductionPlanV1
from .stack import StackError, describe_stack, lane_help, resolve_stack, stack_from_plan


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "production"


def _job_id(title: str) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{_slug(title)}-{stamp}-{os.urandom(2).hex()}"


def _print_summary(summary: dict) -> bool:
    man = summary["manifest"]
    ec = man["exit_criteria"]
    print("\n" + "=" * 64)
    print(f"  PRODUCTION COMPLETE — {man['title']}  [{man['backend']}]")
    print("=" * 64)
    print(f"  final:      {summary['final']}")
    print(f"  artifacts:  {len(man['artifacts'])}   dir: {summary['prod_dir']}")
    print("  --- exit criteria " + "-" * 45)
    print(f"  all validators pass : {ec['all_validators_pass']}")
    print(f"  VO within tolerance : {ec['vo_within_tolerance']}   {ec['vo_detail']}")
    print(f"  final has audio     : {ec['final_has_audio']}")
    print(f"  final duration      : {ec['final_duration_s']}s  (clips {ec['total_clip_s']}s)")
    print("  --- the decisive (human) question " + "-" * 29)
    print("  > is this assembled MP4 better than picking lanes by hand?  [you decide]")
    print("=" * 64)
    failed = [a["id"] for a in man["artifacts"]
              if a["type"] == "media" and not a["validation"].get("ok")]
    if failed:
        print(f"  ⚠ validators FAILED on: {failed}")
    return bool(ec["all_validators_pass"] and ec["vo_within_tolerance"] and ec["final_has_audio"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="studio-production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="static plan (v0a) or a brief the 4B plans (v0b) -> MP4",
        epilog="The production stack is operator-chosen, defaulted-but-visible, and "
               "recorded in the manifest — the director never silently picks a model.\n\n"
               + lane_help())
    ap.add_argument("plan", nargs="?", help="path to a ProductionPlanV1 JSON (v0a path)")
    ap.add_argument("--brief", default=None,
                    help='a one-line brief; the 4B director plans it (v0b), e.g. --brief "60s doc on lighthouses"')
    ap.add_argument("--shots", type=int, default=3, help="target shot count for --brief planning")
    # -- the production stack (operator-chosen; 'auto' = the visible default) --
    ap.add_argument("--video-lane", default="auto",
                    help="video model for every shot: wan (default, renders today) · "
                         "ltx/sulphur/10eros (roadmap — not yet wired in the executor). 'auto' = wan.")
    ap.add_argument("--keyframe-lane", default="auto",
                    help="image model for continuity keyframes: chroma (default) · zimage · "
                         "krea · hidream · ideogram. 'auto' = chroma.")
    ap.add_argument("--continuity", default="auto",
                    help="storyboard (per-shot keyframes + shared style bible, DEFAULT) · hero "
                         "(one shared keyframe) · chain (i2v from prev frame) · none (independent "
                         "t2v). 'auto' = storyboard.")
    ap.add_argument("--no-music", dest="music", action="store_false", default=True,
                    help="omit the ACE-Step music bed")
    ap.add_argument("--no-narration", dest="narration", action="store_false", default=True,
                    help="omit Kokoro voiceover (visuals + music only)")
    ap.add_argument("--voice", default="", help="Kokoro voice id for narration (default af_heart)")
    ap.add_argument("--backend", choices=["live", "synthetic"], default="live",
                    help="live = ComfyUI+Kokoro on the rig; synthetic = offline ffmpeg stand-ins")
    ap.add_argument("--job-id", default=None, help="override the generated job id")
    ap.add_argument("--productions-dir", default=config.PRODUCTIONS_DIR,
                    help=f"base dir (default {config.PRODUCTIONS_DIR})")
    args = ap.parse_args(argv)

    if bool(args.plan) == bool(args.brief):
        print('[args] give exactly one of: a plan file, or --brief "..."', file=sys.stderr)
        return 2

    # Resolve + ECHO the stack up front (validates against what the executor renders,
    # so an unwired/unknown lane fails fast with a clear message — before the lock).
    try:
        stack = resolve_stack(
            video_lane=args.video_lane, keyframe_lane=args.keyframe_lane,
            continuity=args.continuity, narration=args.narration,
            music=args.music, voice=args.voice,
        )
    except StackError as e:
        print(f"[stack] {e}", file=sys.stderr)
        return 2

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    extra_artifacts: list = []

    lock = ProductionLock(os.path.join(args.productions_dir, ".run.lock"))
    try:
        lock.acquire()
    except ProductionLockHeld as e:
        print(f"[locked] {e}", file=sys.stderr)
        return 3
    try:
        if args.brief:
            # brief path: the CLI stack pins the lanes; the director plans within it.
            print(describe_stack(stack), file=sys.stderr)
            from . import planner, registry
            job_id = args.job_id or _job_id(args.brief)
            prod_dir = os.path.join(args.productions_dir, job_id)
            os.makedirs(prod_dir, exist_ok=True)
            try:
                reg = registry.load()
                print(f"[plan] director planning the brief into ~{args.shots} shots…", file=sys.stderr)
                plan, extra_artifacts = planner.plan_from_brief(
                    args.brief, reg, n_shots=args.shots, stack=stack,
                    prompts_dir=os.path.join(prod_dir, "prompts"),
                )
            except planner.PlannerError as e:
                print(f"[plan error] {e}", file=sys.stderr)
                return 2
            print(f"[plan] director produced {len(plan.shots)} shots: "
                  f"{plan.project.title!r}", file=sys.stderr)
        else:
            # static-plan path: the plan file is self-describing — its own encoded stack
            # wins over CLI lane flags (which target --brief planning). We echo what the
            # plan actually encodes so the operator still SEES the stack.
            try:
                with open(args.plan) as f:
                    plan = ProductionPlanV1.from_dict(json.load(f))
            except (OSError, json.JSONDecodeError, PlanError) as e:
                print(f"[plan error] {e}", file=sys.stderr)
                return 2
            job_id = args.job_id or _job_id(plan.project.title)
            stack = stack_from_plan(plan)
            print(describe_stack(stack), file=sys.stderr)

        summary = run_production(plan, backend_name=args.backend, job_id=job_id,
                                 now_iso=now_iso, productions_dir=args.productions_dir,
                                 extra_artifacts=extra_artifacts, stack=stack)
    finally:
        lock.release()

    ok = _print_summary(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
