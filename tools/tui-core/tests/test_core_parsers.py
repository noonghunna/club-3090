"""Tests for the core parsers module."""

from __future__ import annotations

import pytest

from club3090_tui_core.parsers import (
    BenchParser,
    VerifyParser,
    StressParser,
    QualityParser,
    SoakParser,
    RebenchParser,
    Status,
    strip_ansi,
)


def test_strip_ansi():
    assert strip_ansi("\033[32m✓\033[0m test") == "✓ test"
    assert strip_ansi("no codes here") == "no codes here"
    assert strip_ansi("\033[1;31m✗\033[0m fail") == "✗ fail"


class TestBenchParser:
    def test_section_header(self):
        p = BenchParser()
        event = p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        assert event is not None
        assert event.event_type == "bench_section"
        assert event.data["section"] == "narrative"

    def test_run_line(self):
        p = BenchParser()
        p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        event = p.parse_line("  run-1     wall= 11.53s  ttft=   118ms  toks=1000  wall_TPS= 86.68  decode_TPS= 87.58")
        assert event is not None
        assert event.event_type == "bench_run"
        assert event.data["type"] == "run"
        assert event.data["decode_tps"] == 87.58

    def test_summary_metric(self):
        p = BenchParser()
        p.parse_line("========== NARRATIVE (prompt=65 chars, max_tokens=1000) ==========")
        event = p.parse_line("  wall_TPS  mean=  84.14  std=  2.41  CV= 2.9%")
        assert event is not None
        assert event.event_type == "summary_metric"
        assert event.data["mean"] == 84.14


class TestVerifyParser:
    def test_step_header(self):
        p = VerifyParser()
        event = p.parse_line("[1/9] Basic completion ...")
        assert event is not None
        assert event.event_type == "verify_step"
        assert event.data["step"] == 1
        assert event.data["total"] == 9

    def test_check_passed(self):
        p = VerifyParser()
        event = p.parse_line("  ✓ reply contains 'Paris'")
        assert event is not None
        assert event.event_type == "verify_check"
        assert event.data["status"] == "passed"

    def test_check_failed(self):
        p = VerifyParser()
        event = p.parse_line("  ✗ tool-call request failed")
        assert event is not None
        assert event.data["status"] == "failed"

    def test_all_passed_verdict(self):
        p = VerifyParser()
        event = p.parse_line("All checks passed.")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.PASSED

    def test_failure_verdict(self):
        p = VerifyParser()
        event = p.parse_line("3 check(s) failed.")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.FAILED
        assert event.data["failed"] == 3


class TestStressParser:
    def test_rung_line(self):
        p = StressParser()
        line = "  ✓ rung 1/6: target=95K  actual=95K tok (36%)  recalled 'needle'  prefill=10 t/s (1s)  VRAM_free=8000MB"
        event = p.parse_line(line)
        assert event is not None
        assert event.event_type == "niah_rung"
        assert event.data["rung"] == 1
        assert event.data["target_k"] == 95
        assert event.data["status"] == "passed"

    def test_token_line(self):
        p = StressParser()
        event = p.parse_line("  ✓  10000 tokens: recalled 'needle' (got: needle)")
        assert event is not None
        assert event.event_type == "niah_token"
        assert event.data["tokens"] == 10000
        assert event.data["status"] == "passed"

    def test_stress_pass_verdict(self):
        p = StressParser()
        event = p.parse_line("All stress checks passed.")
        assert event is not None
        assert event.data["status"] == Status.PASSED


class TestQualityParser:
    def test_scenario_passed(self):
        p = QualityParser()
        event = p.parse_line("  [1/15] TC-01 ✓ passed (2.3s)")
        assert event is not None
        assert event.event_type == "quality_scenario"
        assert event.data["passed"] is True
        assert event.data["scenario_id"] == "TC-01"

    def test_scenario_failed(self):
        p = QualityParser()
        event = p.parse_line("  [7/15] TC-07 ✗ verifier_fail (3.1s)")
        assert event is not None
        assert event.data["passed"] is False

    def test_total_line(self):
        p = QualityParser()
        event = p.parse_line("TOTAL 10/15")
        assert event is not None
        assert event.event_type == "quality_total"
        assert event.data["passed"] == 10
        assert event.data["total"] == 15


class TestSoakParser:
    def test_session_line(self):
        p = SoakParser()
        event = p.parse_line("[soak] session 1/20")
        assert event is not None
        assert event.event_type == "soak_session"
        assert event.data["session"] == 1

    def test_turn_line(self):
        p = SoakParser()
        event = p.parse_line("[soak]   turn 1/5: status=200 wall=5159ms ttft=481ms decode_tps=42.113 vram=43104MiB")
        assert event is not None
        assert event.event_type == "soak_turn"
        assert event.data["decode_tps"] == 42.113

    def test_verdict_pass(self):
        p = SoakParser()
        event = p.parse_line("[soak]   verdict              PASS")
        assert event is not None
        assert event.event_type == "verdict"
        assert event.data["status"] == Status.PASSED


class TestRebenchParser:
    def test_step_running(self):
        p = RebenchParser()
        event = p.parse_line("[verify-full] running…")
        assert event is not None
        assert event.event_type == "rebench_step_start"
        assert event.data["step"] == "verify-full"

    def test_step_passed(self):
        p = RebenchParser()
        event = p.parse_line("[verify-full] ✓ 96s — log: results/rebench/tag/verify-full.log")
        assert event is not None
        assert event.event_type == "rebench_step_done"
        assert event.data["status"] == Status.PASSED

    def test_step_failed(self):
        p = RebenchParser()
        event = p.parse_line("[bench] ✗ 14s — failed (rc=1) — log: results/bench.log")
        assert event is not None
        assert event.event_type == "rebench_step_done"
        assert event.data["status"] == Status.FAILED
        assert event.data["rc"] == 1
