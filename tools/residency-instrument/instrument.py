#!/usr/bin/env python3
"""Cliff 2 residency instrumentation for vLLM/Genesis containers.

The module is designed to be volume-mounted into a vLLM container and loaded by
``sitecustomize.py``. It is intentionally observational: it monkey-patches
request, engine, and worker boundaries to write CSV snapshots, but it does not
change scheduling, memory policy, or kernels.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import inspect
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import fcntl
except Exception:  # pragma: no cover - Windows is not a target here.
    fcntl = None  # type: ignore[assignment]


MIB = 1024 * 1024
_INSTALLED = False
_LOG_LOCK = threading.Lock()
_REQUEST_SEQ = 0
_ENGINE_STEP = 0
_WORKER_STEP = 0
_MTP_BYTES_CACHE: int | None = None
_EMPTY_CACHE_COUNT = 0


FIELDNAMES = [
    "session",
    "turn",
    "request_seq",
    "engine_step",
    "worker_step",
    "event",
    "phase",
    "timestamp",
    "pid",
    "process",
    "rank",
    "request_id",
    "t_pre_request",
    "t_post_request",
    "total_allocated_mib",
    "total_reserved_mib",
    "free_mib",
    "device_total_mib",
    "kv_cache_used_blocks",
    "kv_cache_free_blocks",
    "kv_cache_total_blocks",
    "kv_cache_usage_pct",
    "kv_cache_used_mib",
    "kv_cache_page_bytes_per_block",
    "genesis_pool_count",
    "genesis_pn12_pool_mib",
    "genesis_other_pools_mib",
    "genesis_total_pools_mib",
    "genesis_prealloc_mib",
    "genesis_tq_pool_mib",
    "genesis_gdn_pool_mib",
    "genesis_moe_pool_mib",
    "mtp_draft_resident_mib",
    "flashinfer_workspace_mib",
    "cuda_graph_private_mib",
    "fragmentation_reserved_unallocated_mib",
    "request_input_tokens",
    "request_output_tokens",
    "ttft_ms",
    "decode_tps",
    "status",
    "error",
    "note",
]

TURN_FIELDNAMES = [
    "session",
    "turn",
    "t_pre_request",
    "t_post_request",
    "total_allocated_mib",
    "total_reserved_mib",
    "free_mib",
    "kv_cache_used_blocks",
    "kv_cache_used_mib",
    "genesis_pool_count",
    "genesis_pn12_pool_mib",
    "genesis_other_pools_mib",
    "mtp_draft_resident_mib",
    "flashinfer_workspace_mib",
    "cuda_graph_private_mib",
    "fragmentation_reserved_unallocated_mib",
    "request_input_tokens",
    "request_output_tokens",
    "ttft_ms",
    "decode_tps",
    "status",
    "error",
    "request_id",
    "engine_step",
    "worker_step",
    "raw_event_note",
]


def _log_path() -> str | None:
    return os.environ.get("RESIDENCY_LOG_PATH") or os.environ.get(
        "GENESIS_RESIDENCY_LOG"
    )


def _mib(nbytes: int | float | None) -> str:
    if nbytes is None:
        return ""
    try:
        return f"{float(nbytes) / MIB:.3f}"
    except Exception:
        return ""


def _now() -> float:
    return time.time()


def _rank() -> str:
    for key in ("RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(key)
        if value:
            return f"{key}={value}"
    return ""


def _process_name() -> str:
    argv0 = Path(sys.argv[0]).name if sys.argv else ""
    try:
        import multiprocessing

        mp_name = multiprocessing.current_process().name
    except Exception:
        mp_name = ""
    if mp_name and mp_name != "MainProcess":
        return f"{argv0}:{mp_name}"
    return argv0


def _write_row(row: dict[str, Any]) -> None:
    path = _log_path()
    if not path:
        return
    normalized = {name: row.get(name, "") for name in FIELDNAMES}
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            with open(path, "a+", newline="") as fh:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                fh.seek(0, os.SEEK_END)
                needs_header = fh.tell() == 0
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                if needs_header:
                    writer.writeheader()
                writer.writerow(normalized)
                fh.flush()
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception as exc:  # pragma: no cover - instrumentation must not fail vLLM.
        sys.stderr.write(f"[residency] failed to write row: {exc!r}\n")


def _safe_import(module_name: str) -> Any | None:
    try:
        __import__(module_name)
        return sys.modules[module_name]
    except Exception:
        return None


def _tensor_storage_key(tensor: Any) -> tuple[str, int] | None:
    try:
        if not getattr(tensor, "is_cuda", False):
            return None
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()))
    except Exception:
        return None


def _tensor_nbytes(tensor: Any) -> int:
    try:
        return int(tensor.nbytes)
    except Exception:
        try:
            return int(tensor.element_size() * tensor.numel())
        except Exception:
            return 0


def _sum_tensors(value: Any, seen: set[tuple[str, int]], depth: int = 0) -> tuple[int, int]:
    if depth > 6:
        return 0, 0
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None and isinstance(value, torch.Tensor):
        key = _tensor_storage_key(value)
        if key is None or key in seen:
            return 0, 0
        seen.add(key)
        return _tensor_nbytes(value), 1
    if isinstance(value, dict):
        total = count = 0
        for item in value.values():
            b, c = _sum_tensors(item, seen, depth + 1)
            total += b
            count += c
        return total, count
    if isinstance(value, (list, tuple, set)):
        total = count = 0
        for item in value:
            b, c = _sum_tensors(item, seen, depth + 1)
            total += b
            count += c
        return total, count
    return 0, 0


def _registry_bytes(module_name: str, owner_name: str, method_name: str) -> int:
    mod = _safe_import(module_name)
    if mod is None:
        return 0
    owner = getattr(mod, owner_name, None)
    method = getattr(owner, method_name, None)
    if method is None:
        return 0
    try:
        info = method()
    except Exception:
        return 0
    if isinstance(info, int):
        return info
    if isinstance(info, dict):
        return int(info.get("total_bytes") or info.get("total") or 0)
    return 0


def _pn12_bytes() -> int:
    return _registry_bytes(
        "vllm._genesis.kernels.ffn_intermediate_cache",
        "FFNIntermediateCache",
        "total_pooled_bytes",
    )


def _genesis_snapshot() -> dict[str, Any]:
    pool_tokens = ("POOL", "POOLS", "BUFFER", "BUFFERS", "REGISTRY", "CACHE")
    seen: set[tuple[str, int]] = set()
    total_bytes = 0
    pool_count = 0

    for name, mod in list(sys.modules.items()):
        if not name.startswith("vllm._genesis"):
            continue
        try:
            members = vars(mod)
        except Exception:
            continue
        for attr, value in members.items():
            upper = attr.upper()
            if any(token in upper for token in pool_tokens):
                b, c = _sum_tensors(value, seen)
                total_bytes += b
                pool_count += c
            if inspect.isclass(value) and getattr(value, "__module__", "") == name:
                for cls_attr, cls_value in vars(value).items():
                    cls_upper = cls_attr.upper()
                    if any(token in cls_upper for token in pool_tokens):
                        b, c = _sum_tensors(cls_value, seen)
                        total_bytes += b
                        pool_count += c

    pn12_bytes = _pn12_bytes()
    prealloc_bytes = _registry_bytes(
        "vllm._genesis.prealloc", "GenesisPreallocBuffer", "get_registry_info"
    )
    tq_bytes = _registry_bytes(
        "vllm._genesis.kernels.dequant_buffer",
        "TurboQuantBufferManager",
        "get_registry_info",
    )
    gdn_bytes = (
        _registry_bytes(
            "vllm._genesis.kernels.gdn_gating_buffer",
            "GdnGatingBufferManager",
            "get_registry_info",
        )
        + _registry_bytes(
            "vllm._genesis.kernels.gdn_core_attn_manager",
            "GdnCoreAttnManager",
            "get_registry_info",
        )
        + _registry_bytes(
            "vllm._genesis.kernels.fla_kkt_buffer",
            "FlaKktBufferManager",
            "get_registry_info",
        )
    )
    moe_mod = _safe_import("vllm._genesis.kernels.moe_intermediate_cache")
    moe_bytes = 0
    if moe_mod is not None and hasattr(moe_mod, "get_registry_info"):
        try:
            moe_bytes = int(moe_mod.get_registry_info().get("total_bytes") or 0)
        except Exception:
            moe_bytes = 0

    total_bytes = max(total_bytes, pn12_bytes, prealloc_bytes, tq_bytes, gdn_bytes)
    other_bytes = max(total_bytes - pn12_bytes, 0)
    return {
        "genesis_pool_count": pool_count,
        "genesis_pn12_pool_mib": _mib(pn12_bytes),
        "genesis_other_pools_mib": _mib(other_bytes),
        "genesis_total_pools_mib": _mib(total_bytes),
        "genesis_prealloc_mib": _mib(prealloc_bytes),
        "genesis_tq_pool_mib": _mib(tq_bytes),
        "genesis_gdn_pool_mib": _mib(gdn_bytes),
        "genesis_moe_pool_mib": _mib(moe_bytes),
    }


def _torch_snapshot(include_slow: bool = False) -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        free, total = torch.cuda.mem_get_info(device)
        stats = torch.cuda.memory_stats(device)
        graph_private = 0
        for key, value in stats.items():
            low = key.lower()
            if not key.endswith(".current"):
                continue
            if ("graph" in low or "private" in low) and (
                "reserved_bytes" in low or "allocated_bytes" in low
            ):
                try:
                    graph_private += int(value)
                except Exception:
                    pass
        out = {
            "total_allocated_mib": _mib(allocated),
            "total_reserved_mib": _mib(reserved),
            "free_mib": _mib(int(free)),
            "device_total_mib": _mib(int(total)),
            "cuda_graph_private_mib": _mib(graph_private),
            "fragmentation_reserved_unallocated_mib": _mib(max(reserved - allocated, 0)),
        }
        if include_slow:
            out.update(_genesis_snapshot())
            out["flashinfer_workspace_mib"] = _mib(_flashinfer_workspace_bytes())
            out["mtp_draft_resident_mib"] = _mib(_mtp_draft_resident_bytes())
        return out
    except Exception:
        return {}


def _flashinfer_workspace_bytes() -> int:
    seen: set[tuple[str, int]] = set()
    total = 0

    for mod_name, mod in list(sys.modules.items()):
        if "flashinfer" not in mod_name.lower():
            continue
        try:
            members = vars(mod)
        except Exception:
            continue
        for attr, value in members.items():
            if "workspace" not in attr.lower():
                continue
            b, _ = _sum_tensors(value, seen)
            total += b

    for obj in gc.get_objects():
        try:
            cls = obj.__class__
            class_label = f"{cls.__module__}.{cls.__name__}".lower()
            if "flashinfer" not in class_label:
                continue
            for attr in dir(obj):
                if "workspace" not in attr.lower():
                    continue
                try:
                    b, _ = _sum_tensors(getattr(obj, attr), seen)
                except Exception:
                    b = 0
                total += b
        except Exception:
            continue
    return total


def _module_parameter_bytes(module: Any, seen: set[tuple[str, int]]) -> int:
    total = 0
    try:
        params = list(module.parameters(recurse=True))
    except Exception:
        return 0
    for param in params:
        key = _tensor_storage_key(param)
        if key is None or key in seen:
            continue
        seen.add(key)
        total += _tensor_nbytes(param)
    return total


def _drafter_object_bytes(obj: Any, seen: set[tuple[str, int]], depth: int = 0) -> int:
    if obj is None or depth > 3:
        return 0
    total = _module_parameter_bytes(obj, seen)
    for attr in (
        "drafter",
        "model",
        "draft_model",
        "draft_model_runner",
        "model_runner",
        "proposer",
    ):
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        if child is obj:
            continue
        total += _drafter_object_bytes(child, seen, depth + 1)
    return total


def _mtp_draft_resident_bytes() -> int:
    global _MTP_BYTES_CACHE
    if _MTP_BYTES_CACHE:
        return _MTP_BYTES_CACHE
    try:
        import torch
    except Exception:
        return 0
    seen: set[tuple[str, int]] = set()
    total = 0
    needles = ("draft", "proposer", "mtp")
    for obj in gc.get_objects():
        try:
            cls = obj.__class__
            label = f"{cls.__module__}.{cls.__name__}".lower()
            if hasattr(obj, "drafter"):
                total += _drafter_object_bytes(getattr(obj, "drafter"), seen)
                continue
            if not any(needle in label for needle in needles):
                continue
            if isinstance(obj, torch.nn.Module):
                total += _module_parameter_bytes(obj, seen)
            else:
                total += _drafter_object_bytes(obj, seen)
        except Exception:
            continue
    if total:
        _MTP_BYTES_CACHE = total
    return total


def _kv_snapshot(engine_core: Any) -> dict[str, Any]:
    try:
        manager = engine_core.scheduler.kv_cache_manager
        block_pool = manager.block_pool
        total_blocks = int(getattr(block_pool, "num_gpu_blocks", 0) or 0)
        free_blocks = int(block_pool.get_num_free_blocks())
        used_blocks = max(total_blocks - free_blocks, 0)
        page_bytes = _kv_page_bytes_per_block(manager)
        used_mib = _mib(used_blocks * page_bytes) if page_bytes else ""
        usage_pct = (
            f"{(used_blocks / total_blocks) * 100:.3f}" if total_blocks else ""
        )
        return {
            "kv_cache_used_blocks": used_blocks,
            "kv_cache_free_blocks": free_blocks,
            "kv_cache_total_blocks": total_blocks,
            "kv_cache_usage_pct": usage_pct,
            "kv_cache_used_mib": used_mib,
            "kv_cache_page_bytes_per_block": page_bytes or "",
        }
    except Exception:
        return {}


def _kv_page_bytes_per_block(manager: Any) -> int:
    try:
        cfg = manager.kv_cache_config
        groups = getattr(cfg, "kv_cache_groups", None) or []
        page_bytes = 0
        for group in groups:
            spec = getattr(group, "kv_cache_spec", None)
            if spec is None and isinstance(group, dict):
                spec = group.get("kv_cache_spec")
            value = getattr(spec, "page_size_bytes", 0)
            if callable(value):
                value = value()
            page_bytes += int(value or 0)
        return page_bytes
    except Exception:
        return 0


def _prompt_token_count(prompt: Any) -> str:
    try:
        if hasattr(prompt, "prompt_token_ids"):
            ids = getattr(prompt, "prompt_token_ids")
            return str(len(ids)) if ids is not None else ""
        if isinstance(prompt, dict):
            for key in ("prompt_token_ids", "token_ids", "input_ids"):
                if key in prompt and prompt[key] is not None:
                    return str(len(prompt[key]))
        if isinstance(prompt, (list, tuple)) and prompt and all(
            isinstance(x, int) for x in prompt[: min(len(prompt), 8)]
        ):
            return str(len(prompt))
    except Exception:
        return ""
    return ""


def _output_token_count(last_output: Any) -> str:
    try:
        outputs = getattr(last_output, "outputs", None)
        if outputs:
            token_ids = getattr(outputs[0], "token_ids", None)
            if token_ids is not None:
                return str(len(token_ids))
        completion = getattr(last_output, "completion_token_ids", None)
        if completion is not None:
            return str(len(completion))
    except Exception:
        return ""
    return ""


def _sample_slow(step: int) -> bool:
    every_raw = os.environ.get("RESIDENCY_SLOW_EVERY_N", "10")
    try:
        every = max(int(every_raw), 1)
    except Exception:
        every = 10
    return step % every == 0


def _base_row(event: str, phase: str = "", **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "event": event,
        "phase": phase,
        "timestamp": f"{_now():.6f}",
        "pid": os.getpid(),
        "process": _process_name(),
        "rank": _rank(),
    }
    row.update(extra)
    return row


def _log_snapshot(
    event: str,
    *,
    phase: str = "",
    include_slow: bool = False,
    engine_core: Any | None = None,
    **extra: Any,
) -> None:
    drafter = extra.pop("_drafter", None)
    row = _base_row(event, phase, **extra)
    row.update(_torch_snapshot(include_slow=include_slow))
    if engine_core is not None:
        row.update(_kv_snapshot(engine_core))
    if include_slow and "genesis_pool_count" not in row:
        row.update(_genesis_snapshot())
    if include_slow and drafter is not None:
        direct_mtp_bytes = _drafter_object_bytes(drafter, set())
        if direct_mtp_bytes:
            row["mtp_draft_resident_mib"] = _mib(direct_mtp_bytes)
    _write_row(row)


def _empty_cache_on_idle(engine_core: Any) -> None:
    global _EMPTY_CACHE_COUNT
    if os.environ.get("RESIDENCY_EMPTY_CACHE_ON_IDLE", "0") != "1":
        return
    try:
        if engine_core.scheduler.has_requests():
            return
    except Exception:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        _EMPTY_CACHE_COUNT += 1
        _log_snapshot(
            "empty_cache_before",
            include_slow=True,
            engine_core=engine_core,
            note=f"idle_empty_cache_count={_EMPTY_CACHE_COUNT}",
        )
        torch.cuda.empty_cache()
        _log_snapshot(
            "empty_cache_after",
            include_slow=True,
            engine_core=engine_core,
            note=f"idle_empty_cache_count={_EMPTY_CACHE_COUNT}",
        )
    except Exception as exc:
        _write_row(_base_row("empty_cache_error", error=repr(exc)))


def _patch_async_llm_generate() -> None:
    mod = _safe_import("vllm.v1.engine.async_llm")
    if mod is None:
        return
    cls = getattr(mod, "AsyncLLM", None)
    if cls is None or getattr(cls.generate, "_residency_wrapped", False):
        return
    original = cls.generate

    async def wrapped(self: Any, prompt: Any, sampling_params: Any, request_id: str, *args: Any, **kwargs: Any):
        global _REQUEST_SEQ
        _REQUEST_SEQ += 1
        seq = _REQUEST_SEQ
        t0 = _now()
        input_tokens = _prompt_token_count(prompt)
        _log_snapshot(
            "pre_request",
            phase="pre",
            include_slow=True,
            request_seq=seq,
            request_id=request_id,
            request_input_tokens=input_tokens,
            t_pre_request=f"{t0:.6f}",
        )
        status = "ok"
        error = ""
        last_output = None
        try:
            async for output in original(
                self,
                prompt,
                sampling_params,
                request_id,
                *args,
                **kwargs,
            ):
                last_output = output
                yield output
        except BaseException as exc:
            status = "exception"
            error = f"{exc.__class__.__name__}: {exc}"
            raise
        finally:
            _log_snapshot(
                "post_request",
                phase="post",
                include_slow=True,
                request_seq=seq,
                request_id=request_id,
                request_input_tokens=input_tokens,
                request_output_tokens=_output_token_count(last_output),
                t_pre_request=f"{t0:.6f}",
                t_post_request=f"{_now():.6f}",
                status=status,
                error=error,
            )

    wrapped._residency_wrapped = True  # type: ignore[attr-defined]
    cls.generate = wrapped


def _patch_engine_core() -> None:
    mod = _safe_import("vllm.v1.engine.core")
    if mod is None:
        return
    cls = getattr(mod, "EngineCore", None)
    if cls is None:
        return
    for method_name in ("step", "step_with_batch_queue"):
        original = getattr(cls, method_name, None)
        if original is None or getattr(original, "_residency_wrapped", False):
            continue

        def make_wrapper(orig: Any, label: str):
            def wrapped(self: Any, *args: Any, **kwargs: Any):
                global _ENGINE_STEP
                result = orig(self, *args, **kwargs)
                _empty_cache_on_idle(self)
                _ENGINE_STEP += 1
                interval = max(
                    int(os.environ.get("RESIDENCY_ENGINE_STEP_INTERVAL", "1")), 1
                )
                if _ENGINE_STEP % interval == 0:
                    _log_snapshot(
                        "engine_step",
                        include_slow=_sample_slow(_ENGINE_STEP),
                        engine_core=self,
                        engine_step=_ENGINE_STEP,
                        note=label,
                    )
                return result

            wrapped._residency_wrapped = True  # type: ignore[attr-defined]
            return wrapped

        setattr(cls, method_name, make_wrapper(original, method_name))


def _patch_worker_execute() -> None:
    targets = (
        ("vllm.v1.worker.gpu_worker", "GPUWorker"),
        ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    )
    for module_name, class_name in targets:
        mod = _safe_import(module_name)
        if mod is None:
            continue
        cls = getattr(mod, class_name, None)
        if cls is None:
            continue
        original = getattr(cls, "execute_model", None)
        if original is None or getattr(original, "_residency_wrapped", False):
            continue

        def make_wrapper(orig: Any, label: str):
            def wrapped(self: Any, *args: Any, **kwargs: Any):
                global _WORKER_STEP
                result = orig(self, *args, **kwargs)
                _WORKER_STEP += 1
                interval = max(
                    int(os.environ.get("RESIDENCY_WORKER_STEP_INTERVAL", "1")), 1
                )
                if _WORKER_STEP % interval == 0:
                    _log_snapshot(
                        "worker_execute_model",
                        include_slow=_sample_slow(_WORKER_STEP),
                        worker_step=_WORKER_STEP,
                        _drafter=getattr(self, "drafter", None),
                        note=label,
                    )
                return result

            wrapped._residency_wrapped = True  # type: ignore[attr-defined]
            return wrapped

        setattr(cls, "execute_model", make_wrapper(original, f"{module_name}.{class_name}"))


def install() -> None:
    """Install monkey-patches. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED or not _log_path():
        return
    _INSTALLED = True
    _write_row(_base_row("install", note="residency instrumentation installed"))
    for patcher in (_patch_async_llm_generate, _patch_engine_core, _patch_worker_execute):
        try:
            patcher()
        except Exception as exc:  # pragma: no cover - observational only.
            _write_row(_base_row("install_error", error=f"{patcher.__name__}: {exc!r}"))


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _ts(row: dict[str, str]) -> float:
    try:
        return float(row.get("timestamp") or 0)
    except Exception:
        return 0.0


def _last_before(rows: list[dict[str, str]], timestamp: float, event: str) -> dict[str, str]:
    best: dict[str, str] = {}
    for row in rows:
        if row.get("event") != event:
            continue
        if _ts(row) <= timestamp + 2.0:
            best = row
        else:
            break
    return best


def _last_nonempty_before(
    rows: list[dict[str, str]], timestamp: float, event: str, field: str
) -> str:
    best = ""
    for row in rows:
        if row.get("event") != event:
            continue
        if _ts(row) > timestamp + 2.0:
            break
        value = row.get(field, "")
        if value not in ("", None):
            best = value
    return best


def _choose(*values: str) -> str:
    for value in values:
        if value not in ("", None):
            return value
    return ""


def join_residency(residency_csv: str, turn_log_csv: str, output_csv: str) -> None:
    raw_rows = sorted(_read_csv(residency_csv), key=_ts)
    turn_rows = _read_csv(turn_log_csv)
    post_rows = [r for r in raw_rows if r.get("event") == "post_request"]
    pre_by_seq = {
        r.get("request_seq", ""): r for r in raw_rows if r.get("event") == "pre_request"
    }

    out_rows: list[dict[str, str]] = []
    for idx, turn in enumerate(turn_rows):
        post = post_rows[idx] if idx < len(post_rows) else {}
        end_ts = _ts(post) if post else 0.0
        engine = _last_before(raw_rows, end_ts, "engine_step") if end_ts else {}
        worker = _last_before(raw_rows, end_ts, "worker_execute_model") if end_ts else {}
        pre = pre_by_seq.get(post.get("request_seq", ""), {}) if post else {}
        carry = {}
        if end_ts:
            for field in (
                "genesis_pool_count",
                "genesis_pn12_pool_mib",
                "genesis_other_pools_mib",
                "mtp_draft_resident_mib",
                "flashinfer_workspace_mib",
                "cuda_graph_private_mib",
                "fragmentation_reserved_unallocated_mib",
            ):
                carry[field] = _last_nonempty_before(
                    raw_rows, end_ts, "worker_execute_model", field
                )
        source = {**post}
        for key in FIELDNAMES:
            if not source.get(key):
                source[key] = _choose(worker.get(key, ""), engine.get(key, ""), pre.get(key, ""))
        out_rows.append(
            {
                "session": turn.get("session_id", ""),
                "turn": turn.get("turn_id", ""),
                "t_pre_request": _choose(post.get("t_pre_request", ""), pre.get("t_pre_request", "")),
                "t_post_request": post.get("t_post_request", ""),
                "total_allocated_mib": _choose(worker.get("total_allocated_mib", ""), source.get("total_allocated_mib", "")),
                "total_reserved_mib": _choose(worker.get("total_reserved_mib", ""), source.get("total_reserved_mib", "")),
                "free_mib": _choose(worker.get("free_mib", ""), source.get("free_mib", "")),
                "kv_cache_used_blocks": engine.get("kv_cache_used_blocks", ""),
                "kv_cache_used_mib": engine.get("kv_cache_used_mib", ""),
                "genesis_pool_count": _choose(worker.get("genesis_pool_count", ""), carry.get("genesis_pool_count", ""), source.get("genesis_pool_count", "")),
                "genesis_pn12_pool_mib": _choose(worker.get("genesis_pn12_pool_mib", ""), carry.get("genesis_pn12_pool_mib", ""), source.get("genesis_pn12_pool_mib", "")),
                "genesis_other_pools_mib": _choose(worker.get("genesis_other_pools_mib", ""), carry.get("genesis_other_pools_mib", ""), source.get("genesis_other_pools_mib", "")),
                "mtp_draft_resident_mib": _choose(worker.get("mtp_draft_resident_mib", ""), carry.get("mtp_draft_resident_mib", ""), source.get("mtp_draft_resident_mib", "")),
                "flashinfer_workspace_mib": _choose(worker.get("flashinfer_workspace_mib", ""), carry.get("flashinfer_workspace_mib", ""), source.get("flashinfer_workspace_mib", "")),
                "cuda_graph_private_mib": _choose(worker.get("cuda_graph_private_mib", ""), carry.get("cuda_graph_private_mib", ""), source.get("cuda_graph_private_mib", "")),
                "fragmentation_reserved_unallocated_mib": _choose(worker.get("fragmentation_reserved_unallocated_mib", ""), carry.get("fragmentation_reserved_unallocated_mib", ""), source.get("fragmentation_reserved_unallocated_mib", "")),
                "request_input_tokens": post.get("request_input_tokens", ""),
                "request_output_tokens": post.get("request_output_tokens", ""),
                "ttft_ms": turn.get("ttft_ms", ""),
                "decode_tps": turn.get("decode_tps", ""),
                "status": turn.get("status", post.get("status", "")),
                "error": turn.get("error", post.get("error", "")),
                "request_id": post.get("request_id", ""),
                "engine_step": engine.get("engine_step", ""),
                "worker_step": worker.get("worker_step", ""),
                "raw_event_note": _choose(worker.get("note", ""), engine.get("note", ""), post.get("note", "")),
            }
        )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TURN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    join = sub.add_parser("join", help="join raw residency rows with soak turn-log.csv")
    join.add_argument("--residency-csv", required=True)
    join.add_argument("--turn-log", required=True)
    join.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "join":
        join_residency(args.residency_csv, args.turn_log, args.output)
        return 0
    install()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
