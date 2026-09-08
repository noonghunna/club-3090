"""Tests for the core detection module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from club3090_tui_core.detect import (
    ServingTarget,
    GpuInfo,
    PORT_MAP_BROAD_RE,
    PORT_MAP_ANY_RE,
    _classify_engine,
    _classify_engine_from_container,
    _registry_claims,
    detect_endpoint,
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


class TestMatchConfidence:
    """F9 (slug masquerade): the match GRADE is recorded on ``match_confidence``
    so renderers can distinguish a verified identity (exact container) from a
    shape guess (port/substring fallback).  Regression source: the Agents-A1
    bring (2026-07-03) served on a sibling's port and was PRESENTED as
    ``vllm/qwen-35b-a3b-dual`` with no hint it was a different model."""

    VARIANTS = [
        _make_variant_row(container="vllm-qwen36-35b-a3b-dual", port=8051,
                          slug="vllm/qwen-35b-a3b-dual"),
        _make_variant_row(container="vllm-qwen36-27b-dual", port=8010, slug="vllm/dual"),
    ]

    def test_exact_container_is_identity(self):
        target = ServingTarget(container="vllm-qwen36-27b-dual", host_port=8010)
        result = match_target_to_registry(target, self.VARIANTS)
        assert result.slug == "vllm/dual"
        assert result.match_confidence == "identity"

    def test_port_fallback_is_shape(self):
        # The A1 masquerade: an unknown (brought) container on a sibling's port
        # still slug-matches — but MUST be graded a shape guess, not an identity.
        target = ServingTarget(
            container="vllm-agents-a1-dual", host_port=8051, model="agents-a1"
        )
        result = match_target_to_registry(target, self.VARIANTS)
        assert result.slug == "vllm/qwen-35b-a3b-dual"
        assert result.match_confidence == "shape"

    def test_substring_fallback_is_shape(self):
        target = ServingTarget(container="vllm-qwen36-27b-dual-1", host_port=0)
        result = match_target_to_registry(target, self.VARIANTS)
        assert result.slug == "vllm/dual"
        assert result.match_confidence == "shape"

    def test_unmatched_is_empty(self):
        target = ServingTarget(container="comfyui", host_port=7861)
        result = match_target_to_registry(target, self.VARIANTS)
        assert result.slug == ""
        assert result.match_confidence == ""


# ============================================================================
# #1219 — detection is REGISTRY-FIRST, not container-name-prefix-first
# ============================================================================


def _row(slug, container, port, engine="vllm"):
    return VariantRow(
        slug=slug, switch_engine=engine, launch_engine=engine, compose_dir="d",
        file="f.yml", port=port, model="m", engine=engine, kvcalc_key="k",
        container=container, compose_path="p", status="production",
        ctx_label="32K", status_note="",
    )


def _proc(ps_output: str):
    """Stub for `docker ps --format {{.Names}}|{{.Ports}}`."""
    mock = AsyncMock()
    mock.communicate = AsyncMock(return_value=(ps_output.encode(), b""))
    return mock


def _run(coro):
    """Drive a coroutine without pytest-asyncio.

    ⚠️ pytest-asyncio is NOT installed for this package, so an ``async def`` test
    is SKIPPED, not run — a green line that proves nothing.  Baseline before these
    tests was "25 passed", with zero skips; anything that reports skips here is a
    test that silently did not execute.
    """
    return asyncio.run(coro)


# A local slug can satisfy NEITHER heuristic by construction: the container is
# named by whatever the user's compose calls it, and a third-party engine listens
# on whatever internal port it likes.  Both are true of the bucko recipe that
# surfaced this (2026-09-08).
LOCAL_PS = "qwen38-flash-next-ple|0.0.0.0:20272->9000/tcp"


class TestRegistryClaims:
    def test_builds_map_from_variant_rows(self):
        claims = _registry_claims([_row("x/y", "my-engine", 20272)])
        assert "my-engine" in claims
        assert claims["my-engine"][1] == 20272

    def test_normalizes_underscores(self):
        claims = _registry_claims([_row("x/y", "my_engine", 1)])
        assert "my-engine" in claims

    def test_ignores_rows_without_a_container(self):
        assert _registry_claims([_row("x/y", "", 1)]) == {}

    def test_accepts_dict_rows(self):
        claims = _registry_claims([{"container": "c", "port": 42}])
        assert claims["c"][1] == 42

    def test_empty_and_none(self):
        assert _registry_claims(None) == {}
        assert _registry_claims([]) == {}


class TestRegistryFirstDetection:
    """The bug: a registered container that matches neither heuristic is invisible."""

    def test_local_slug_invisible_without_registry(self):
        """Baseline — this is the reported failure, and it must stay reproducible."""
        with patch("asyncio.create_subprocess_exec", return_value=_proc(LOCAL_PS)):
            target = _run(detect_endpoint())
        assert target.health == "unreachable"
        assert not target.url

    def test_local_slug_detected_when_registry_claims_it(self):
        rows = [_row("bucko-vllm/flash-next", "qwen38-flash-next-ple", 20272)]
        with patch("asyncio.create_subprocess_exec", return_value=_proc(LOCAL_PS)):
            target = _run(detect_endpoint(variants=rows))
        assert target.container == "qwen38-flash-next-ple"
        assert target.host_port == 20272
        assert target.internal_port == 9000   # NOT one of the three curated ports
        # The reported symptom is an EMPTY target.url: with no url, health.sh
        # falls back to the curated default port and reports "not reachable" over
        # a plainly loaded model.  Resolving the url is the fix.
        assert target.url == "http://localhost:20272"
        # `health` stays "unreachable" here because nothing is actually listening
        # on 20272 in a unit test — that probe is Step 4 and is not what #1219 is
        # about.  Asserting on it would be asserting on the stub, not the fix.

    def test_engine_family_comes_from_the_registry_row(self):
        """Engine identity is lineage (the profile's type), not the container name."""
        rows = [_row("x/y", "qwen38-flash-next-ple", 20272, engine="vllm")]
        with patch("asyncio.create_subprocess_exec", return_value=_proc(LOCAL_PS)):
            target = _run(detect_endpoint(variants=rows))
        assert target.engine == "vllm"

    def test_unclaimed_non_engine_container_still_ignored(self):
        """Registry-first must not become 'accept everything'."""
        ps = "open-webui|0.0.0.0:8080->8080/tcp\nqdrant|0.0.0.0:6333->6333/tcp"
        with patch("asyncio.create_subprocess_exec", return_value=_proc(ps)):
            target = _run(detect_endpoint(variants=[_row("x/y", "not-running", 1)]))
        # open-webui publishes 8080->8080 and so still trips the port heuristic;
        # what matters is that qdrant (neither prefix nor engine port) never does.
        assert target.container != "qdrant"

    def test_heuristics_still_work_for_unregistered_containers(self):
        ps = "vllm-hand-run|0.0.0.0:8000->8000/tcp"
        with patch("asyncio.create_subprocess_exec", return_value=_proc(ps)):
            target = _run(detect_endpoint(variants=[]))
        assert target.container == "vllm-hand-run"
        assert target.engine == "vllm"

    def test_claimed_container_preferred_over_heuristic_match(self):
        ps = ("vllm-some-other|0.0.0.0:8000->8000/tcp\n"
              "qwen38-flash-next-ple|0.0.0.0:20272->9000/tcp")
        rows = [_row("bucko-vllm/flash-next", "qwen38-flash-next-ple", 20272)]
        with patch("asyncio.create_subprocess_exec", return_value=_proc(ps)):
            target = _run(detect_endpoint(variants=rows))
        assert target.container == "qwen38-flash-next-ple"

    def test_registry_host_port_wins_over_a_second_mapping(self):
        """An engine that also publishes metrics must resolve to its API port."""
        ps = "qwen38-flash-next-ple|0.0.0.0:9100->9100/tcp, 0.0.0.0:20272->9000/tcp"
        rows = [_row("x/y", "qwen38-flash-next-ple", 20272)]
        with patch("asyncio.create_subprocess_exec", return_value=_proc(ps)):
            target = _run(detect_endpoint(variants=rows))
        assert target.host_port == 20272

    def test_omitting_variants_is_unchanged_behaviour(self):
        ps = "vllm-qwen36-27b-minimal|0.0.0.0:8020->8000/tcp"
        with patch("asyncio.create_subprocess_exec", return_value=_proc(ps)):
            a = _run(detect_endpoint())
            b = _run(detect_endpoint(variants=None))
        assert a.container == b.container == "vllm-qwen36-27b-minimal"


class TestAnyPortRegex:
    def test_matches_arbitrary_internal_port(self):
        m = PORT_MAP_ANY_RE.search("0.0.0.0:20272->9000/tcp")
        assert m and m.group(1) == "20272" and m.group(2) == "9000"

    def test_broad_regex_still_rejects_arbitrary_internal_port(self):
        """The narrow regex must stay narrow — it gates UNCLAIMED containers."""
        assert not PORT_MAP_BROAD_RE.search("0.0.0.0:20272->9000/tcp")
