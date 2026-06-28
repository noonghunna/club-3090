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
    failed = [a["id"] for a in man["artifacts"] if not a["validation"].get("ok")]
    if failed:
        print(f"  ⚠ validators FAILED on: {failed}")
    return bool(ec["all_validators_pass"] and ec["vo_within_tolerance"] and ec["final_has_audio"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="studio-production", description="v0a: static plan -> MP4")
    ap.add_argument("plan", help="path to a ProductionPlanV1 JSON")
    ap.add_argument("--backend", choices=["live", "synthetic"], default="live",
                    help="live = ComfyUI+Kokoro on the rig; synthetic = offline ffmpeg stand-ins")
    ap.add_argument("--job-id", default=None, help="override the generated job id")
    ap.add_argument("--productions-dir", default=config.PRODUCTIONS_DIR,
                    help=f"base dir (default {config.PRODUCTIONS_DIR})")
    args = ap.parse_args(argv)

    try:
        with open(args.plan) as f:
            plan = ProductionPlanV1.from_dict(json.load(f))
    except (OSError, json.JSONDecodeError, PlanError) as e:
        print(f"[plan error] {e}", file=sys.stderr)
        return 2

    job_id = args.job_id or _job_id(plan.project.title)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    lock = ProductionLock(os.path.join(args.productions_dir, ".run.lock"))
    try:
        lock.acquire()
    except ProductionLockHeld as e:
        print(f"[locked] {e}", file=sys.stderr)
        return 3
    try:
        summary = run_production(plan, backend_name=args.backend, job_id=job_id,
                                 now_iso=now_iso, productions_dir=args.productions_dir)
    finally:
        lock.release()

    ok = _print_summary(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
