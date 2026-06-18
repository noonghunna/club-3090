"""Auto-detection of serving model + endpoint.

Replicates the logic from scripts/preflight.sh::preflight_autodetect_endpoint
and preflight_autodetect_model.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

# Engine-internal ports: 8000=vLLM, 8080=llama.cpp, 30000=SGLang
ENGINE_INTERNAL_PORTS = {"8000", "8080", "30000"}

# Recognized engine-family container prefixes
ENGINE_PREFIXES = re.compile(r"^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)")

# Port mapping regex: matches 0.0.0.0:8011->8000/tcp, [::]:8011->8000/tcp, 127.0.0.1:8011->8000/tcp
PORT_MAP_RE = re.compile(
    r"(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}|\[::\]):(\d+)->(8000|8080|30000)/tcp"
)

# All known engine-port patterns (also match without IP prefix)
PORT_MAP_BROAD_RE = re.compile(
    r":(\d+)->(8000|8080|30000)/tcp"
)


@dataclass
class GpuInfo:
    """Information about a single GPU."""
    index: int
    utilization: int = 0       # %
    mem_used_mib: int = 0
    mem_total_mib: int = 0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    temp_c: int = 0


@dataclass
class ServingTarget:
    """Resolved serving target — model + endpoint + metadata."""
    url: str = ""
    model: str = ""
    container: str = ""
    engine: str = ""           # vllm | llamacpp | ik-llama | sglang | beellama | unknown
    host_port: int = 0
    internal_port: int = 0
    slug: str = ""             # registry slug if matched
    kv_format: str = ""
    max_ctx: int = 0
    tp: int = 0
    status: str = ""           # registry status
    status_note: str = ""
    health: str = "unknown"    # serving | unreachable | multiple
    gpus: list[GpuInfo] = field(default_factory=list)

    @property
    def is_localhost(self) -> bool:
        return "localhost" in self.url or "127." in self.url or "[::1]" in self.url

    @property
    def is_active(self) -> bool:
        return bool(self.url and self.model and self.health == "serving")


def _classify_engine(internal_port: str) -> str:
    """Map internal port to engine family."""
    return {"8000": "vllm", "8080": "llamacpp", "30000": "sglang"}.get(internal_port, "unknown")


def _classify_engine_from_container(name: str) -> str:
    """Refine engine from container name prefix."""
    if name.startswith("vllm-"):
        return "vllm"
    if name.startswith("llama-cpp-") or name.startswith("ik-llama-"):
        return "llamacpp"
    if name.startswith("sglang-"):
        return "sglang"
    if name.startswith("beellama-"):
        return "beellama"
    return "unknown"


async def detect_endpoint(container_name: Optional[str] = None) -> ServingTarget:
    """Detect the currently-serving model and endpoint.

    Args:
        container_name: If set, detect only from this container.

    Returns:
        A ServingTarget with whatever was resolved.
    """
    target = ServingTarget()

    # Step 1: docker ps
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "--format", "{{.Names}}|{{.Ports}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        lines = stdout.decode().strip().split("\n")
    except (asyncio.TimeoutError, FileNotFoundError, Exception):
        target.health = "unreachable"
        return target

    # Step 2: find inference containers
    # Filter to recognized engine prefixes FIRST to exclude Open WebUI and other non-inference containers
    candidates: list[tuple[str, int, int, str]] = []  # (name, host_port, internal_port, engine)
    seen: set[tuple[str, int]] = set()  # dedupe by (container_name, host_port) for dual-stack

    for line in lines:
        if "|" not in line:
            continue
        name, ports_str = line.split("|", 1)

        # Only consider recognized engine containers
        if not ENGINE_PREFIXES.match(name):
            continue

        for match in PORT_MAP_BROAD_RE.finditer(ports_str):
            host_port = int(match.group(1))
            internal_port = int(match.group(2))

            # Dedupe dual-stack mappings (same container, same host port)
            key = (name, host_port)
            if key in seen:
                continue
            seen.add(key)

            engine = _classify_engine_from_container(name)
            if engine == "unknown":
                engine = _classify_engine(str(internal_port))
            candidates.append((name, host_port, internal_port, engine))

    if not candidates:
        target.health = "unreachable"
        return target

    # If caller pinned a container, filter to it
    if container_name:
        candidates = [c for c in candidates if c[0] == container_name]
        if not candidates:
            target.health = "unreachable"
            return target

    # Step 3: prefer recognized engine prefix; else first match
    preferred = [c for c in candidates if ENGINE_PREFIXES.match(c[0])]
    chosen = preferred[0] if preferred else candidates[0]

    name, host_port, internal_port, engine = chosen
    target.container = name
    target.host_port = host_port
    target.internal_port = internal_port
    target.engine = engine
    target.url = f"http://localhost:{host_port}"

    # Check for truly different containers (not just multiple ports on same container)
    unique_containers = set(c[0] for c in candidates)
    if len(unique_containers) > 1:
        target.health = "multiple"
    else:
        target.health = "serving"

    # Step 4: probe /v1/models
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{target.url}/v1/models")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    target.model = data[0].get("id", "")
                    if target.health == "multiple":
                        pass  # keep multiple
                    else:
                        target.health = "serving"
            else:
                target.health = "unreachable"
    except Exception:
        target.health = "unreachable"

    # Step 5: GPU info
    target.gpus = await get_gpu_info()

    return target


async def get_gpu_info() -> list[GpuInfo]:
    """Query nvidia-smi for GPU stats."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        gpus = []
        for line in stdout.decode().strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                gpus.append(GpuInfo(
                    index=int(parts[0]),
                    utilization=int(float(parts[1])),
                    mem_used_mib=int(float(parts[2])),
                    mem_total_mib=int(float(parts[3])),
                    power_draw_w=float(parts[4]),
                    power_limit_w=float(parts[5]),
                    temp_c=int(float(parts[6])),
                ))
        return gpus
    except Exception:
        return []


async def detect_from_registry(repo_root: str) -> list["VariantRow"]:
    """Enumerate registry variants via registry-emit.sh.

    Returns list[VariantRow] parsed from tab-delimited output.  This is a thin
    alias onto the converged parser in ``registry.py`` so there is exactly one
    tab parser + one ``VariantRow`` contract on the stack (the dict-returning
    duplicate that used to live here is gone).
    """
    from .registry import detect_from_registry_async
    return await detect_from_registry_async(repo_root)


def match_target_to_registry(target: ServingTarget, variants: list) -> ServingTarget:
    """Enrich a ServingTarget with registry metadata by matching port/container.

    Accepts list[VariantRow] (core) or list[dict] (compat).
    """
    for v in variants:
        # Support both VariantRow objects and dicts
        if hasattr(v, "container"):
            container = v.container
            port = v.port
            slug = v.slug
            kvcalc_key = v.kvcalc_key
            status = v.status
            status_note = v.status_note
        else:
            container = v.get("container", "")
            port = v.get("port", 0)
            slug = v.get("slug", "")
            kvcalc_key = v.get("kvcalc_key", "")
            status = v.get("status", "")
            status_note = v.get("status_note", "")

        if target.container and container.replace("_", "-") in target.container:
            target.slug = slug
            target.kv_format = kvcalc_key
            target.status = status
            target.status_note = status_note
            return target
        if port == target.host_port:
            target.slug = slug
            target.kv_format = kvcalc_key
            target.status = status
            target.status_note = status_note
            return target
    return target
