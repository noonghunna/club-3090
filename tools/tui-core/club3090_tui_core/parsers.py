"""Output parsers for each test script.

Each parser is a stateful class that processes lines and emits structured events.
Regex patterns are derived from the actual script output formats (Section 10 of spec).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TestType(str, Enum):
    VERIFY = "verify"
    VERIFY_FULL = "verify-full"
    BENCH = "bench"
    VERIFY_STRESS = "verify-stress"
    QUALITY = "quality"
    SOAK = "soak"
    REBENCH = "rebench-full"


class Status(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


def strip_ansi(text: str) -> str:
    """Strip ANSI/SGR escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@dataclass
class ParseEvent:
    """A structured event parsed from a test output line."""
    event_type: str           # e.g., "bench_run", "summary_metric", "niah_rung", "verdict"
    data: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


# ============================================================================
# bench.sh parser
# ============================================================================

class BenchParser:
    """Parser for bench.sh output."""

    # Section: ========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========
    SECTION_RE = re.compile(r"^========== (\w+) \(prompt=")

    # Warmup/measured: run-1  wall= 11.53s  ttft=   118ms  toks=1000  wall_TPS= 86.68  decode_TPS= 87.58
    RUN_RE = re.compile(
        r"^\s+(warm|run)-(\d+)\s+wall=\s*([\d.]+)s\s+ttft=\s*(\d+)ms\s+"
        r"toks=\s*(\d+)\s+wall_TPS=\s*([\d.]+)\s+decode_TPS=\s*([\d.]+)"
    )

    # Summary: wall_TPS  mean=  84.14  std=  2.41  CV= 2.9%
    SUMMARY_RE = re.compile(
        r"^\s+(wall_TPS|decode_TPS)\s+mean=\s*([\d.]+).*CV=\s*([\d.]+)%"
    )

    # TTFT summary
    TTFT_RE = re.compile(r"^\s+TTFT\s+mean=\s*(\d+)ms")

    # Spec decoding metrics
    SPEC_DEC_RE = re.compile(r"^=== Last \d+ SpecDecoding metrics ===")

    def __init__(self):
        self.current_section: Optional[str] = None
        self.runs: dict[str, list[dict]] = {"narrative": [], "code": []}
        self.summary: dict[str, dict] = {}
        self.total_runs: int = 0
        self.warmups: int = 0
        self.measured_runs: int = 0

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        """Parse a single line and return an event if matched."""
        clean = strip_ansi(line)

        # Section header
        m = self.SECTION_RE.match(clean)
        if m:
            self.current_section = m.group(1).lower()
            return ParseEvent("bench_section", {"section": self.current_section}, line)

        # Run line (warmup or measured)
        m = self.RUN_RE.match(clean)
        if m:
            run_type, run_num = m.group(1), int(m.group(2))
            data = {
                "type": run_type,
                "run": run_num,
                "wall_s": float(m.group(3)),
                "ttft_ms": int(m.group(4)),
                "tokens": int(m.group(5)),
                "wall_tps": float(m.group(6)),
                "decode_tps": float(m.group(7)),
                "section": self.current_section,
            }
            if self.current_section and run_type == "run":
                self.runs[self.current_section].append(data)
            return ParseEvent("bench_run", data, line)

        # Summary metric
        m = self.SUMMARY_RE.match(clean)
        if m:
            metric, mean, cv = m.group(1), float(m.group(2)), float(m.group(3))
            data = {"metric": metric, "mean": mean, "cv": cv, "section": self.current_section}
            if self.current_section:
                self.summary.setdefault(self.current_section, {})[metric] = data
            return ParseEvent("summary_metric", data, line)

        # TTFT summary
        m = self.TTFT_RE.match(clean)
        if m:
            data = {"ttft_mean_ms": int(m.group(1)), "section": self.current_section}
            if self.current_section:
                self.summary.setdefault(self.current_section, {})["ttft"] = data
            return ParseEvent("summary_ttft", data, line)

        # Spec decoding header
        if self.SPEC_DEC_RE.match(clean):
            return ParseEvent("spec_dec_header", {}, line)

        return None


# ============================================================================
# verify.sh / verify-full.sh parser
# ============================================================================

class VerifyParser:
    """Parser for verify.sh and verify-full.sh output."""

    # [3/9] Basic completion — capital of France ...
    STEP_RE = re.compile(r"^\[(\d+)/(\d+)\] (.+?) \.\.\.")

    #   ✓ reply contains 'Paris'
    #   ✗ tool-call request failed
    #   ⊘ Genesis patches applied (skipped)
    CHECK_RE = re.compile(r"^\s+([✓✗⊘]) (.+)")

    #     → Check docker logs vllm-qwen36-27b
    HINT_RE = re.compile(r"^\s+→ (.+)")

    # All checks passed.
    PASS_RE = re.compile(r"^All checks passed\.")

    # 3 check(s) failed.
    FAIL_RE = re.compile(r"^(\d+) check\(s\) failed\.")

    def __init__(self):
        self.steps: list[dict] = []
        self.current_step: Optional[dict] = None
        self.total_steps: int = 0
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        clean = strip_ansi(line)

        # Step header
        m = self.STEP_RE.match(clean)
        if m:
            step_num, total, name = int(m.group(1)), int(m.group(2)), m.group(3)
            self.total_steps = total
            self.current_step = {"num": step_num, "total": total, "name": name, "checks": []}
            self.steps.append(self.current_step)
            return ParseEvent("verify_step", {"step": step_num, "total": total, "name": name}, line)

        # Check result
        m = self.CHECK_RE.match(clean)
        if m:
            glyph, msg = m.group(1), m.group(2)
            status = {"✓": "passed", "✗": "failed", "⊘": "skipped"}[glyph]
            if status == "passed":
                self.passed += 1
            elif status == "failed":
                self.failed += 1
            else:
                self.skipped += 1

            check = {"glyph": glyph, "message": msg, "status": status, "hint": ""}
            if self.current_step:
                self.current_step["checks"].append(check)
            return ParseEvent("verify_check", check, line)

        # Hint
        m = self.HINT_RE.match(clean)
        if m and self.current_step and self.current_step["checks"]:
            hint = m.group(1)
            self.current_step["checks"][-1]["hint"] = hint
            return ParseEvent("verify_hint", {"hint": hint}, line)

        # Final verdict
        if self.PASS_RE.match(clean):
            return ParseEvent("verdict", {"status": Status.PASSED, "message": "All checks passed"}, line)

        m = self.FAIL_RE.match(clean)
        if m:
            count = int(m.group(1))
            return ParseEvent("verdict", {"status": Status.FAILED, "failed": count, "message": f"{count} check(s) failed"}, line)

        return None


# ============================================================================
# verify-stress.sh parser (incl. NIAH)
# ============================================================================

class StressParser:
    """Parser for verify-stress.sh output."""

    # [1/8] Long-context needle small rungs (10K / 30K) ...
    PROBE_RE = re.compile(r"^\[(\d+)/8\] (.+?) \.\.\.")

    # ✓  10000 tokens: recalled '…' (got: …)
    # △  30000 tokens: recall MISS (…) — system OK, quality ceiling reached
    TOKEN_RE = re.compile(r"^\s+([✓△✗⊘])\s+(\d+) tokens:")

    # ✓ rung 1/6: target=95K  actual=95K tok (36%)  recalled '…'  prefill=… t/s (…s)  VRAM_free=…MB
    RUNG_RE = re.compile(
        r"^\s+([✓△✗⊘]) rung (\d+)/(\d+): target=(\d+)K"
    )

    # n_ctx=262000  ladder: 95000 → 125000 → ...
    LADDER_RE = re.compile(r"^\s+n_ctx=(\d+)\s+ladder:")

    # VRAM free (ladder start): 12345 MB
    VRAM_FREE_RE = re.compile(r"^\s+VRAM free \(ladder start\): (\d+) MB")

    # All stress / boundary checks passed.
    PASS_RE = re.compile(r"^All stress")

    # 3 stress check(s) failed.
    FAIL_RE = re.compile(r"^(\d+) stress check")

    def __init__(self):
        self.probes: list[dict] = []
        self.current_probe: Optional[dict] = None
        self.niah_results: list[dict] = []  # rung/token results
        self.ladder_info: dict = {}

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        clean = strip_ansi(line)

        # Probe header
        m = self.PROBE_RE.match(clean)
        if m:
            probe_num, name = int(m.group(1)), m.group(2)
            self.current_probe = {"num": probe_num, "name": name, "results": []}
            self.probes.append(self.current_probe)
            return ParseEvent("stress_probe", {"probe": probe_num, "name": name}, line)

        # Token-level result (probes 1, 7)
        m = self.TOKEN_RE.match(clean)
        if m:
            glyph, tokens = m.group(1), int(m.group(2))
            status = {"✓": "passed", "△": "partial", "✗": "failed", "⊘": "skipped"}[glyph]
            result = {"tokens": tokens, "status": status, "glyph": glyph}
            self.niah_results.append(result)
            if self.current_probe:
                self.current_probe["results"].append(result)
            return ParseEvent("niah_token", result, line)

        # Rung-level result (probe 8 ceiling ladder)
        m = self.RUNG_RE.match(clean)
        if m:
            glyph, rung, total, target_k = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            status = {"✓": "passed", "△": "partial", "✗": "failed", "⊘": "skipped"}[glyph]
            result = {"rung": rung, "total": total, "target_k": target_k, "status": status, "glyph": glyph}
            self.niah_results.append(result)
            if self.current_probe:
                self.current_probe["results"].append(result)
            return ParseEvent("niah_rung", result, line)

        # Ladder info
        m = self.LADDER_RE.match(clean)
        if m:
            self.ladder_info["n_ctx"] = int(m.group(1))
            return ParseEvent("niah_ladder", {"n_ctx": int(m.group(1))}, line)

        # VRAM free
        m = self.VRAM_FREE_RE.match(clean)
        if m:
            self.ladder_info["vram_free_mb"] = int(m.group(1))
            return ParseEvent("niah_vram", {"vram_free_mb": int(m.group(1))}, line)

        # Verdicts
        if self.PASS_RE.match(clean):
            return ParseEvent("verdict", {"status": Status.PASSED, "message": "All stress checks passed"}, line)

        m = self.FAIL_RE.match(clean)
        if m:
            count = int(m.group(1))
            return ParseEvent("verdict", {"status": Status.FAILED, "failed": count, "message": f"{count} stress check(s) failed"}, line)

        return None


# ============================================================================
# quality-test.sh parser
# ============================================================================

class QualityParser:
    """Parser for quality-test.sh output."""

    # [1/15] TC-01 ✓ passed (2.3s)
    # [7/15] TC-07 ✗ verifier_fail (3.1s)
    SCENARIO_RE = re.compile(
        r"^\s+\[(\d+)/(\d+)\] (\S+) ([✓✗]) (\w+) \(([\d.]+)s\)"
    )

    # Pack ID prefix mapping
    PACK_PREFIXES = {
        "TC": "toolcall-15",
        "IF": "instructfollow-15",
        "SO": "structoutput-15",
        "DE": "dataextract-15",
        "RM": "reasonmath-15",
        "BF": "bugfind-15",
        "CL": "cli-40",
        "HA": "hermesagent-20",
        "HE": "humaneval-plus-30",
        "LC": "lcb-v6-30",
        "GS": "gsm-symbolic-30",
        "GP": "gpqa-diamond",
    }

    # TOTAL line
    TOTAL_RE = re.compile(r"TOTAL.*?(\d+)/(\d+)")

    def __init__(self):
        self.scenarios: list[dict] = []
        self.packs: dict[str, dict] = {}  # pack_id -> {passed, total, scenarios}
        self.total_passed: int = 0
        self.total_count: int = 0

    def _pack_from_prefix(self, scenario_id: str) -> str:
        prefix = scenario_id[:2].upper()
        return self.PACK_PREFIXES.get(prefix, "unknown")

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        clean = strip_ansi(line)

        # Scenario result
        m = self.SCENARIO_RE.match(clean)
        if m:
            num, total, scenario_id, glyph, failure_mode, elapsed = (
                int(m.group(1)), int(m.group(2)), m.group(3),
                m.group(4), m.group(5), float(m.group(6))
            )
            passed = glyph == "✓"
            pack_id = self._pack_from_prefix(scenario_id)

            result = {
                "num": num, "total": total,
                "scenario_id": scenario_id,
                "passed": passed,
                "failure_mode": failure_mode if not passed else "",
                "elapsed_s": elapsed,
                "pack_id": pack_id,
            }
            self.scenarios.append(result)

            # Update pack totals
            pack = self.packs.setdefault(pack_id, {"passed": 0, "total": 0, "scenarios": []})
            pack["total"] += 1
            pack["scenarios"].append(result)
            if passed:
                pack["passed"] += 1
                self.total_passed += 1
            self.total_count += 1

            return ParseEvent("quality_scenario", result, line)

        # TOTAL line
        m = self.TOTAL_RE.match(clean)
        if m:
            self.total_passed = int(m.group(1))
            self.total_count = int(m.group(2))
            return ParseEvent("quality_total", {
                "passed": self.total_passed,
                "total": self.total_count,
            }, line)

        return None


# ============================================================================
# soak-test.sh parser
# ============================================================================

class SoakParser:
    """Parser for soak-test.sh output."""

    # [soak] mode=fresh sessions=20 turns=5 max_growth=200MiB timeout=1800s
    MODE_RE = re.compile(r"^\[soak\] mode=(\w+) sessions=(\d+) turns=(\d+)")

    # [soak] session 1/20
    SESSION_RE = re.compile(r"^\[soak\] session (\d+)/(\d+)")

    # [soak]   turn 1/5: status=200 wall=5159ms ttft=481ms decode_tps=42.113 vram=43104MiB
    TURN_RE = re.compile(
        r"^\[soak\]\s+turn (\d+)/(\d+): status=(\d+) wall=(\d+)ms "
        r"ttft=(\d+)ms decode_tps=([\d.]+) vram=(\d+)MiB"
    )

    # [soak]   verdict              PASS
    VERDICT_RE = re.compile(r"^\[soak\]\s+verdict\s+(PASS|FAIL)")

    # [soak]   silent_empty         0 / 100 (0.0%)
    METRIC_RE = re.compile(r"^\[soak\]\s+(silent_empty|tps_retention|p50_decode_tps|max_growth_mib)\s+(.+)")

    # [soak] warm baseline after session 1: 43104 MiB
    BASELINE_RE = re.compile(r"^\[soak\] warm baseline after session (\d+): (\d+) MiB")

    def __init__(self):
        self.mode: str = ""
        self.total_sessions: int = 0
        self.total_turns: int = 0
        self.current_session: int = 0
        self.turns: list[dict] = []
        self.baseline_vram: int = 0
        self.verdict: Optional[str] = None
        self.metrics: dict = {}

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        clean = strip_ansi(line)

        # Mode/config line
        m = self.MODE_RE.match(clean)
        if m:
            self.mode = m.group(1)
            self.total_sessions = int(m.group(2))
            self.total_turns = int(m.group(3))
            return ParseEvent("soak_config", {
                "mode": self.mode,
                "sessions": self.total_sessions,
                "turns": self.total_turns,
            }, line)

        # Session start
        m = self.SESSION_RE.match(clean)
        if m:
            self.current_session = int(m.group(1))
            return ParseEvent("soak_session", {
                "session": self.current_session,
                "total": int(m.group(2)),
            }, line)

        # Turn result
        m = self.TURN_RE.match(clean)
        if m:
            turn_data = {
                "turn": int(m.group(1)),
                "total": int(m.group(2)),
                "status": int(m.group(3)),
                "wall_ms": int(m.group(4)),
                "ttft_ms": int(m.group(5)),
                "decode_tps": float(m.group(6)),
                "vram_mib": int(m.group(7)),
                "session": self.current_session,
            }
            self.turns.append(turn_data)
            return ParseEvent("soak_turn", turn_data, line)

        # Baseline
        m = self.BASELINE_RE.match(clean)
        if m:
            self.baseline_vram = int(m.group(2))
            return ParseEvent("soak_baseline", {"vram_mib": self.baseline_vram}, line)

        # Verdict
        m = self.VERDICT_RE.match(clean)
        if m:
            self.verdict = m.group(1)
            status = Status.PASSED if self.verdict == "PASS" else Status.FAILED
            return ParseEvent("verdict", {
                "status": status,
                "verdict": self.verdict,
            }, line)

        # Metrics
        m = self.METRIC_RE.match(clean)
        if m:
            key, value = m.group(1), m.group(2)
            self.metrics[key] = value
            return ParseEvent("soak_metric", {"key": key, "value": value}, line)

        return None


# ============================================================================
# rebench-full.sh parser (orchestrator)
# ============================================================================

class RebenchParser:
    """Parser for rebench-full.sh orchestrator output."""

    # [verify-full] running…
    STEP_RUNNING_RE = re.compile(r"^\[([\w-]+)\] running…")

    # [verify-full] ✓ 96s — log: results/rebench/<tag>/verify-full.log
    STEP_PASS_RE = re.compile(r"^\[([\w-]+)\] ✓ (\d+)s")

    # [bench] ✗ 14s — failed (rc=1) — log: …
    STEP_FAIL_RE = re.compile(r"^\[([\w-]+)\] ✗ (\d+)s — failed \(rc=(\d+)\)")

    # [quality-full] skipped — 8-pack is opt-in
    STEP_SKIP_RE = re.compile(r"^\[([\w-]+)\] skipped")

    #  report:      results/rebench/<tag>/REPORT.md
    REPORT_RE = re.compile(r"^\s+report:\s+(.+REPORT\.md)")

    # artifacts:   results/rebench/<tag>
    ARTIFACTS_RE = re.compile(r"^\s+artifacts:\s+(.+)")

    # rebench complete
    COMPLETE_RE = re.compile(r"^\s*rebench complete")

    STEP_ORDER = ["verify-full", "bench", "verify-stress", "quality-full", "quality-thinking", "soak"]

    def __init__(self):
        self.steps: dict[str, dict] = {}
        self.current_step: Optional[str] = None
        self.report_path: str = ""
        self.artifacts_dir: str = ""
        self.complete: bool = False

    def parse_line(self, line: str) -> Optional[ParseEvent]:
        clean = strip_ansi(line)

        # Step running
        m = self.STEP_RUNNING_RE.match(clean)
        if m:
            step = m.group(1)
            self.current_step = step
            self.steps[step] = {"status": Status.RUNNING, "elapsed_s": 0}
            return ParseEvent("rebench_step_start", {"step": step}, line)

        # Step passed
        m = self.STEP_PASS_RE.match(clean)
        if m:
            step, elapsed = m.group(1), int(m.group(2))
            self.steps[step] = {"status": Status.PASSED, "elapsed_s": elapsed}
            self.current_step = None
            return ParseEvent("rebench_step_done", {
                "step": step, "status": Status.PASSED, "elapsed_s": elapsed,
            }, line)

        # Step failed
        m = self.STEP_FAIL_RE.match(clean)
        if m:
            step, elapsed, rc = m.group(1), int(m.group(2)), int(m.group(3))
            self.steps[step] = {"status": Status.FAILED, "elapsed_s": elapsed, "rc": rc}
            self.current_step = None
            return ParseEvent("rebench_step_done", {
                "step": step, "status": Status.FAILED, "elapsed_s": elapsed, "rc": rc,
            }, line)

        # Step skipped
        m = self.STEP_SKIP_RE.match(clean)
        if m:
            step = m.group(1)
            self.steps[step] = {"status": Status.SKIPPED, "elapsed_s": 0}
            return ParseEvent("rebench_step_done", {
                "step": step, "status": Status.SKIPPED, "elapsed_s": 0,
            }, line)

        # Report path
        m = self.REPORT_RE.match(clean)
        if m:
            self.report_path = m.group(1)
            return ParseEvent("rebench_report", {"path": self.report_path}, line)

        # Artifacts dir
        m = self.ARTIFACTS_RE.match(clean)
        if m:
            self.artifacts_dir = m.group(1)
            return ParseEvent("rebench_artifacts", {"dir": self.artifacts_dir}, line)

        # Complete
        if self.COMPLETE_RE.match(clean):
            self.complete = True
            return ParseEvent("rebench_complete", {}, line)

        return None


def get_parser(test_type: TestType):
    """Factory to get the right parser for a test type."""
    parsers = {
        TestType.VERIFY: VerifyParser,
        TestType.VERIFY_FULL: VerifyParser,
        TestType.BENCH: BenchParser,
        TestType.VERIFY_STRESS: StressParser,
        TestType.QUALITY: QualityParser,
        TestType.SOAK: SoakParser,
        TestType.REBENCH: RebenchParser,
    }
    return parsers[test_type]()
