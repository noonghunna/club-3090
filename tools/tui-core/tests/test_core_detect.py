"""Tests for the core detection module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from club3090_tui_core.detect import (
    ServingTarget,
    GpuInfo,
    PORT_MAP_BROAD_RE,
    _classify_engine,
    _classify_engine_from_container,
    match_target_to_registry,
)
from club3090_tui_core.registry import VariantRow


# ============================================================================
# Test port regex
# ============================================================================

class TestPortRegex:
    def test_vllm_port(self):
        m = PORT_MAP_BROAD_RE.search("0.0.0.0:8010->8000/tcp")
        assert m is not None
        assert m.group(1) == "8010"
        assert m.group(2) == "8000"

    def test_llamacpp_port(self):
        m = PORT_MAP_BROAD_RE.search("0.0.0.0:8020->8080/tcp")
        assert m is not None
        assert m.group(1) == "8020"
        assert m.group(2) == "8080"

    def test_sglang_port(self):
        m = PORT_MAP_BROAD_RE.search("0.0.0.0:30000->30000/tcp")
        assert m is not None
        assert m.group(1) == "30000"
        assert m.group(2) == "30000"

    def test_ipv6_loopback(self):
        m = PORT_MAP_BROAD_RE.search("[::]:8011->8000/tcp")
        assert m is not None
        assert m.group(1) == "8011"

    def test_non_engine_port_ignored(self):
        m = PORT_MAP_BROAD_RE.search("0.0.0.0:8188->8188/tcp")
        assert m is None


# ============================================================================
# Test engine classification
# ============================================================================

class TestEngineClassification:
    def test_from_port(self):
        assert _classify_engine("8000") == "vllm"
        assert _classify_engine("8080") == "llamacpp"
        assert _classify_engine("30000") == "sglang"
        assert _classify_engine("9999") == "unknown"

    def test_from_container_name(self):
        assert _classify_engine_from_container("vllm-qwen36-27b") == "vllm"
        assert _classify_engine_from_container("llama-cpp-pi-reasoning") == "llamacpp"
        assert _classify_engine_from_container("ik-llama-cpp-dual") == "llamacpp"
        assert _classify_engine_from_container("sglang-main") == "sglang"
        assert _classify_engine_from_container("beellama-dflash") == "beellama"
        assert _classify_engine_from_container("random-container") == "unknown"


# ============================================================================
# Test ServingTarget
# ============================================================================

class TestServingTarget:
    def test_is_localhost(self):
        t = ServingTarget(url="http://localhost:8010")
        assert t.is_localhost is True

        t = ServingTarget(url="http://127.0.0.1:8010")
        assert t.is_localhost is True

        t = ServingTarget(url="http://192.168.1.50:8010")
        assert t.is_localhost is False

    def test_is_active(self):
        t = ServingTarget(url="http://localhost:8010", model="test-model", health="serving")
        assert t.is_active is True

        t = ServingTarget(url="http://localhost:8010", model="test-model", health="unreachable")
        assert t.is_active is False


# ============================================================================
# Test registry matching — with VariantRow objects (core path)
# ============================================================================

def _make_variant_row(**kwargs) -> VariantRow:
    defaults = dict(
        slug="vllm/dual",
        switch_engine="vllm",
        launch_engine="vllm",
        compose_dir="",
        file="",
        port=8010,
        model="qwen",
        engine="vllm",
        kvcalc_key="fp8",
        container="vllm_qwen",
        compose_path="",
        status="production",
        ctx_label="",
        status_note="",
    )
    defaults.update(kwargs)
    return VariantRow(**defaults)


class TestRegistryMatchingVariantRow:
    """match_target_to_registry with VariantRow objects."""

    def test_match_by_port(self):
        target = ServingTarget(host_port=8010, container="vllm-test")
        variants = [_make_variant_row(port=8010, slug="vllm/dual", status="production")]
        result = match_target_to_registry(target, variants)
        assert result.slug == "vllm/dual"
        assert result.status == "production"

    def test_match_by_container_name(self):
        target = ServingTarget(host_port=9999, container="vllm-qwen36-27b")
        variants = [_make_variant_row(container="vllm_qwen36_27b", port=8010, slug="vllm/dual")]
        result = match_target_to_registry(target, variants)
        assert result.slug == "vllm/dual"

    def test_no_match(self):
        target = ServingTarget(host_port=9999, container="unknown-thing")
        variants = [_make_variant_row(container="vllm_qwen", port=8010)]
        result = match_target_to_registry(target, variants)
        assert result.slug == ""

    def test_kvcalc_key_propagated(self):
        target = ServingTarget(host_port=8010)
        variants = [_make_variant_row(port=8010, kvcalc_key="qwen3.6-27b:fp8-mtp")]
        result = match_target_to_registry(target, variants)
        assert result.kv_format == "qwen3.6-27b:fp8-mtp"

    def test_status_note_propagated(self):
        target = ServingTarget(host_port=8010)
        variants = [_make_variant_row(port=8010, status_note="DFlash prose regression")]
        result = match_target_to_registry(target, variants)
        assert result.status_note == "DFlash prose regression"


class TestRegistryMatchingDictCompat:
    """match_target_to_registry with dict — backward-compat path."""

    def test_match_by_port_dict(self):
        target = ServingTarget(host_port=8010, container="vllm-test")
        variants = [
            {"slug": "vllm/dual", "port": 8010, "model": "qwen", "engine": "vllm",
             "kvcalc_key": "fp8", "status": "production", "container": "vllm_qwen",
             "compose_dir": "", "file": "", "switch_engine": "", "launch_engine": "",
             "compose_path": "", "ctx_label": "", "status_note": ""},
        ]
        result = match_target_to_registry(target, variants)
        assert result.slug == "vllm/dual"
        assert result.status == "production"

    def test_match_by_container_dict(self):
        target = ServingTarget(host_port=9999, container="vllm-qwen36-27b")
        variants = [
            {"slug": "vllm/dual", "port": 8010, "model": "qwen", "engine": "vllm",
             "kvcalc_key": "fp8", "status": "production", "container": "vllm_qwen36_27b",
             "compose_dir": "", "file": "", "switch_engine": "", "launch_engine": "",
             "compose_path": "", "ctx_label": "", "status_note": ""},
        ]
        result = match_target_to_registry(target, variants)
        assert result.slug == "vllm/dual"


class TestRegistryMatchExactNotPrefix:
    """Regression: a slug whose container name is a PREFIX of the actually-running
    container must NOT shadow the real one — the old substring match mislabelled
    ``vllm-...-dual-max`` as ``vllm/dual`` and ``...-carnice-v2-dual`` as the single
    (and then misdirected the targeted stop to a container that wasn't running)."""

    # registry order puts the shorter (prefix) container FIRST — the worst case.
    VARIANTS = [
        _make_variant_row(container="vllm-qwen36-27b-dual", port=8010, slug="vllm/dual"),
        _make_variant_row(container="vllm-qwen36-27b-dual-max", port=8013,
                          slug="vllm/qwen-27b-dual-max", status="experimental"),
        _make_variant_row(container="beellama-carnice-v2", port=8068,
                          slug="beellama/carnice-v2-single-q5km-mtp", status="experimental"),
        _make_variant_row(container="beellama-carnice-v2-dual", port=8070,
                          slug="beellama/carnice-v2-dual-q8-mtp", status="experimental"),
    ]

    def test_dual_max_not_shadowed_by_dual(self):
        target = ServingTarget(container="vllm-qwen36-27b-dual-max", host_port=8013)
        assert match_target_to_registry(target, self.VARIANTS).slug == "vllm/qwen-27b-dual-max"

    def test_carnice_dual_not_shadowed_by_single(self):
        target = ServingTarget(container="beellama-carnice-v2-dual", host_port=8070)
        assert (
            match_target_to_registry(target, self.VARIANTS).slug
            == "beellama/carnice-v2-dual-q8-mtp"
        )

    def test_prefix_slug_still_matches_its_own_container(self):
        target = ServingTarget(container="vllm-qwen36-27b-dual", host_port=8010)
        assert match_target_to_registry(target, self.VARIANTS).slug == "vllm/dual"

    def test_port_only_when_no_container(self):
        target = ServingTarget(container="", host_port=8013)
        assert match_target_to_registry(target, self.VARIANTS).slug == "vllm/qwen-27b-dual-max"

    def test_docker_scale_suffix_uses_longest_substring(self):
        # A docker project/scale suffix (<name>-1) has no exact match → pass 3
        # picks the LONGEST containing registry container, never a prefix sibling.
        target = ServingTarget(container="beellama-carnice-v2-dual-1", host_port=0)
        assert (
            match_target_to_registry(target, self.VARIANTS).slug
            == "beellama/carnice-v2-dual-q8-mtp"
        )
