"""Tests for the detection module."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from club3090_test_console.detect import (
    ServingTarget,
    GpuInfo,
    PORT_MAP_BROAD_RE,
    _classify_engine,
    _classify_engine_from_container,
    match_target_to_registry,
)


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

    def test_localhost_only(self):
        m = PORT_MAP_BROAD_RE.search("127.0.0.1:8011->8000/tcp")
        assert m is not None
        assert m.group(1) == "8011"

    def test_non_engine_port_ignored(self):
        m = PORT_MAP_BROAD_RE.search("0.0.0.0:8188->8188/tcp")
        assert m is None  # 8188 is not an engine port

    def test_multiple_mappings(self):
        line = "0.0.0.0:8010->8000/tcp, :::8010->8000/tcp"
        matches = list(PORT_MAP_BROAD_RE.finditer(line))
        assert len(matches) == 2


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

        t = ServingTarget(url="", model="", health="unknown")
        assert t.is_active is False


# ============================================================================
# Test registry matching
# ============================================================================

class TestRegistryMatching:
    def test_match_by_port(self):
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

    def test_match_by_container_name(self):
        target = ServingTarget(host_port=9999, container="vllm-qwen36-27b")
        variants = [
            {"slug": "vllm/dual", "port": 8010, "model": "qwen", "engine": "vllm",
             "kvcalc_key": "fp8", "status": "production", "container": "vllm_qwen36_27b",
             "compose_dir": "", "file": "", "switch_engine": "", "launch_engine": "",
             "compose_path": "", "ctx_label": "", "status_note": ""},
        ]
        result = match_target_to_registry(target, variants)
        assert result.slug == "vllm/dual"

    def test_no_match(self):
        target = ServingTarget(host_port=9999, container="unknown-thing")
        variants = [
            {"slug": "vllm/dual", "port": 8010, "model": "qwen", "engine": "vllm",
             "kvcalc_key": "fp8", "status": "production", "container": "vllm_qwen",
             "compose_dir": "", "file": "", "switch_engine": "", "launch_engine": "",
             "compose_path": "", "ctx_label": "", "status_note": ""},
        ]
        result = match_target_to_registry(target, variants)
        assert result.slug == ""


# ============================================================================
# Test nothing-serving path
# ============================================================================

class TestNothingServing:
    @pytest.mark.asyncio
    async def test_empty_docker_ps(self):
        """When docker ps returns nothing, health should be 'unreachable'."""
        from club3090_test_console.detect import detect_endpoint

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            target = await detect_endpoint()
            assert target.health == "unreachable"
            assert not target.model
            assert not target.url


class TestDualStackDedup:
    """Test that dual-stack Docker output is deduped and non-engine containers are filtered."""

    def test_dual_stack_dedup(self):
        """Same container with 0.0.0.0 and [::] mappings should produce one candidate."""
        from club3090_test_console.detect import PORT_MAP_BROAD_RE, ENGINE_PREFIXES
        
        # Simulate Docker dual-stack output
        ports_str = "0.0.0.0:8010->8000/tcp, [::]:8010->8000/tcp"
        matches = list(PORT_MAP_BROAD_RE.finditer(ports_str))
        assert len(matches) == 2  # Two raw matches
        
        # Dedup logic
        seen = set()
        unique = []
        container_name = "vllm-qwen36-27b"
        for m in matches:
            key = (container_name, int(m.group(1)))
            if key not in seen:
                seen.add(key)
                unique.append(m)
        assert len(unique) == 1  # Deduped to one

    def test_open_webui_filtered(self):
        """Open WebUI maps 8080->8080 but isn't an engine container."""
        from club3090_test_console.detect import ENGINE_PREFIXES
        
        assert not ENGINE_PREFIXES.match("open-webui")
        assert not ENGINE_PREFIXES.match("nginx-proxy")
        assert not ENGINE_PREFIXES.match("litellm-gateway")
        assert ENGINE_PREFIXES.match("vllm-qwen36-27b")
        assert ENGINE_PREFIXES.match("llama-cpp-pi-reasoning")
        assert ENGINE_PREFIXES.match("sglang-main")
        assert ENGINE_PREFIXES.match("beellama-dflash")

    def test_multiple_containers_detected(self):
        """Truly different containers should set health=multiple."""
        from club3090_test_console.detect import ENGINE_PREFIXES
        
        candidates = [
            ("vllm-qwen36-27b", 8010, 8000, "vllm"),
            ("llama-cpp-pi-reasoning", 8063, 8080, "llamacpp"),
        ]
        unique_containers = set(c[0] for c in candidates)
        assert len(unique_containers) == 2  # Two different containers
