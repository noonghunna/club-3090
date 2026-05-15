#!/usr/bin/env python3
"""Estate planner operations for club-3090.

The estate layer is intentionally a thin orchestration wrapper around the
existing compose registry and validate_estate() profile checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CLUB3090_LOG_LEVEL", "ERROR")

from scripts.lib.profiles.canonical_scenarios import CANONICAL_SCENARIOS  # noqa: E402
from scripts.lib.profiles.compat import (  # noqa: E402
    InstanceSpec,
    ProfileError,
    calibration_status,
    load_profiles,
    validate_estate,
)
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY  # noqa: E402
from scripts.lib.profiles.launch_compat import _hardware_id_from_gpu, resolve_engine_pin  # noqa: E402


SUPPORTED_ESTATE_SCHEMA_VERSIONS = {1}
DEFAULT_ESTATE_PATH = Path("~/.club3090/estate.yml").expanduser()
DEFAULT_BOOT_LOG_DIR = Path("/tmp/club3090-estate-boot")
BOOT_LOG_KEEP = 5


class EstateCliError(Exception):
    """User-facing estate CLI failure."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    mem_mib: int
    sm: float
    hardware_id: str


@dataclass(frozen=True)
class BootOutcome:
    index: int
    total: int
    inst: InstanceSpec
    ok: bool
    elapsed_s: int
    log_path: Path
    error: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_dotenv() -> dict[str, str]:
    env = dict(os.environ)
    path = REPO_ROOT / ".env"
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env.setdefault(key, value)
    return env


def safe_name(name: str) -> str:
    out = re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-_")
    return out or "instance"


def project_name(name: str) -> str:
    return f"estate-{safe_name(name)}"


def container_name(name: str) -> str:
    return f"club3090-{safe_name(name)}"


def parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def estate_path(value: str | None) -> Path:
    return Path(value).expanduser() if value else DEFAULT_ESTATE_PATH


def same_path(a: Path, b: Path) -> bool:
    return a.expanduser().resolve(strict=False) == b.expanduser().resolve(strict=False)


def parse_fake_gpus(value: str) -> list[GpuInfo]:
    out = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            idx, name, mem_mib, sm = raw.split(":", 3)
        except ValueError as exc:
            raise EstateCliError(f"invalid CLUB3090_FAKE_GPUS entry `{raw}`") from exc
        hardware_id = _hardware_id_from_gpu(name, int(mem_mib), float(sm))
        out.append(GpuInfo(int(idx), name, int(mem_mib), float(sm), hardware_id))
    return sorted(out, key=lambda gpu: gpu.index)


def detect_gpus_from_host() -> list[GpuInfo]:
    fake = os.environ.get("CLUB3090_FAKE_GPUS")
    if fake:
        return parse_fake_gpus(fake)

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise EstateCliError(f"nvidia-smi GPU query failed: {proc.stderr.strip() or proc.stdout.strip()}")
    out = []
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 4:
            raise EstateCliError(f"could not parse nvidia-smi GPU row `{raw}`")
        idx, name, mem_mib, sm = parts
        hardware_id = _hardware_id_from_gpu(name, int(mem_mib), float(sm))
        out.append(GpuInfo(int(idx), name, int(mem_mib), float(sm), hardware_id))
    return sorted(out, key=lambda gpu: gpu.index)


def parse_estate_yaml(path: Path) -> tuple[dict[str, Any], list[InstanceSpec]]:
    if not path.exists():
        raise EstateCliError(f"estate file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise EstateCliError(f"failed to parse estate YAML: {exc}") from exc

    version = data.get("schema_version")
    if version not in SUPPORTED_ESTATE_SCHEMA_VERSIONS:
        raise EstateCliError(f"unsupported estate schema_version={version}; supported={sorted(SUPPORTED_ESTATE_SCHEMA_VERSIONS)}")
    estate = data.get("estate")
    if not isinstance(estate, list):
        raise EstateCliError("estate file must contain an `estate:` list")

    instances = []
    seen = set()
    for i, item in enumerate(estate, start=1):
        if not isinstance(item, dict):
            raise EstateCliError(f"estate entry #{i} must be a mapping")
        name = str(item.get("name") or "").strip()
        compose = str(item.get("compose") or item.get("compose_name") or "").strip()
        gpus = item.get("gpus")
        port = item.get("port")
        if not name:
            raise EstateCliError(f"estate entry #{i} missing name")
        if name in seen:
            raise EstateCliError(f"duplicate estate instance name `{name}`")
        seen.add(name)
        if not compose:
            raise EstateCliError(f"estate entry `{name}` missing compose")
        if not isinstance(gpus, list) or not gpus:
            raise EstateCliError(f"estate entry `{name}` must set gpus: [..]")
        if not isinstance(port, int):
            raise EstateCliError(f"estate entry `{name}` must set integer port")
        try:
            gpu_indices = tuple(int(gpu) for gpu in gpus)
        except (TypeError, ValueError) as exc:
            raise EstateCliError(f"estate entry `{name}` has non-integer GPU index") from exc
        instances.append(InstanceSpec(name=name, compose_name=compose, gpu_indices=gpu_indices, port=port))
    return data, instances


def synthesize_hardware_from_doc(data: dict[str, Any], profiles) -> list:
    rig = data.get("rig") if isinstance(data.get("rig"), dict) else {}
    hardware_id = rig.get("hardware_id")
    gpu_count = rig.get("gpu_count")
    if not hardware_id or not isinstance(gpu_count, int):
        raise EstateCliError("no GPUs detected and estate file rig.hardware_id/gpu_count fallback is missing")
    if hardware_id == "mixed":
        raise EstateCliError("cannot synthesize mixed hardware from estate file; run on the target rig")
    if hardware_id not in profiles.hardware:
        raise EstateCliError(f"estate rig.hardware_id `{hardware_id}` is not a known hardware profile")
    return [profiles.hardware[hardware_id] for _ in range(gpu_count)]


def hardware_for_estate(data: dict[str, Any], profiles) -> tuple[list, list[GpuInfo]]:
    try:
        gpus = detect_gpus_from_host()
        return [profiles.hardware[gpu.hardware_id] for gpu in gpus], gpus
    except EstateCliError:
        hardware = synthesize_hardware_from_doc(data, profiles)
        gpus = [GpuInfo(i, hardware[i].display_name, int(hardware[i].vram_gb * 1024), hardware[i].sm, hardware[i].id) for i in range(len(hardware))]
        return hardware, gpus


def detect_nvlink_pairs() -> tuple[bool, list[tuple[int, int]]]:
    mode = os.environ.get("NVLINK_MODE", "auto")
    fake_pairs = os.environ.get("CLUB3090_FAKE_NVLINK_PAIRS", "")
    if fake_pairs:
        pairs = []
        for item in fake_pairs.split(","):
            if not item.strip():
                continue
            a, b = item.replace(":", "-").split("-", 1)
            pairs.append(tuple(sorted((int(a), int(b)))))
        return True, sorted(set(pairs))
    if mode == "force_on":
        gpus = detect_gpus_from_host()
        if len(gpus) >= 2:
            return True, [(gpus[0].index, gpus[1].index)]
        return True, []
    if mode == "force_off":
        return False, []

    proc = subprocess.run(["nvidia-smi", "topo", "-m"], text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        return False, []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return False, []
    header = re.findall(r"GPU(\d+)", lines[0])
    pairs = set()
    for line in lines[1:]:
        cols = line.split()
        if not cols or not cols[0].startswith("GPU"):
            continue
        row_idx = int(cols[0][3:])
        for col_name, value in zip(header, cols[1:]):
            col_idx = int(col_name)
            if row_idx < col_idx and re.match(r"NV\d+", value):
                pairs.add((row_idx, col_idx))
    return bool(pairs), sorted(pairs)


def validate_doc(path: Path):
    profiles = load_profiles()
    data, instances = parse_estate_yaml(path)
    hardware, gpus = hardware_for_estate(data, profiles)
    nvlink_active, nvlink_pairs = detect_nvlink_pairs()
    result = validate_estate(instances, hardware, profiles, nvlink_active, nvlink_pairs)
    return profiles, data, instances, hardware, gpus, nvlink_active, nvlink_pairs, result


def print_validation_summary(instances: list[InstanceSpec], result) -> None:
    print(f"Estate validation: {'PASS' if result.valid else 'FAIL'}")
    for inst in instances:
        inst_result = result.per_instance.get(inst.name)
        marker = "✓" if inst_result and inst_result.valid else "✗"
        print(f"  {marker} {inst.name}: {inst.compose_name} GPUs={list(inst.gpu_indices)} port={inst.port}")
        if inst_result and not inst_result.valid:
            for reason in inst_result.reasons:
                print(f"      - {reason}")
    if result.cross_instance_failures:
        print("  Cross-instance failures:")
        for failure in result.cross_instance_failures:
            print(f"      - {failure}")
    for note in result.notes:
        print(f"  note: {note}")


def estate_doc(instances: list[InstanceSpec], gpus: list[GpuInfo], nvlink_active: bool, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    hardware_ids = {gpu.hardware_id for gpu in gpus}
    now = utc_now()
    return {
        "schema_version": 1,
        "created": (existing or {}).get("created", now),
        "updated": now,
        "rig": {
            "hardware_id": next(iter(hardware_ids)) if len(hardware_ids) == 1 else "mixed",
            "gpu_count": len(gpus),
            "nvlink_active": bool(nvlink_active),
        },
        "estate": [
            {
                "name": inst.name,
                "compose": inst.compose_name,
                "gpus": list(inst.gpu_indices),
                "port": inst.port,
            }
            for inst in instances
        ],
    }


def write_estate(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".estate.", suffix=".tmp", delete=False) as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def persist_default_estate_source(path: Path, data: dict[str, Any], instances: list[InstanceSpec], gpus: list[GpuInfo], nvlink_active: bool) -> None:
    if same_path(path, DEFAULT_ESTATE_PATH):
        return
    write_estate(DEFAULT_ESTATE_PATH, estate_doc(instances, gpus, nvlink_active, data))
    print(f"[estate] wrote {DEFAULT_ESTATE_PATH} from {path}")


def compose_abs_path(compose_name: str) -> Path:
    entry = COMPOSE_REGISTRY.get(compose_name)
    if not entry:
        raise EstateCliError(f"unknown compose `{compose_name}`")
    path = REPO_ROOT / entry["compose_path"]
    if not path.exists():
        raise EstateCliError(f"compose file not found for `{compose_name}`: {path}")
    return path


def compose_env(inst: InstanceSpec) -> dict[str, str]:
    env = load_dotenv()
    joined = ",".join(str(gpu) for gpu in inst.gpu_indices)
    env.update(
        {
            "ESTATE_GPUS": joined,
            "ESTATE_PORT": str(inst.port),
            "ESTATE_CONTAINER": container_name(inst.name),
            "CUDA_VISIBLE_DEVICES": joined,
            "NVIDIA_VISIBLE_DEVICES": joined,
            "PORT": str(inst.port),
        }
    )
    entry = COMPOSE_REGISTRY.get(inst.compose_name)
    if entry:
        profiles = load_profiles()
        engine = profiles.engines[entry["engine"]]
        if engine.type == "vllm":
            env.update(resolve_engine_pin(profiles, entry["engine"]))
    return env


def compose_cmd() -> list[str]:
    return shlex.split(os.environ.get("COMPOSE_BIN", "docker compose"))


def boot_log_dir() -> Path:
    return Path(os.environ.get("CLUB3090_ESTATE_BOOT_LOG_DIR", str(DEFAULT_BOOT_LOG_DIR))).expanduser()


def instance_log_path(inst: InstanceSpec) -> Path:
    return boot_log_dir() / f"{safe_name(inst.name)}.log"


def rotate_log(path: Path, keep: int = BOOT_LOG_KEEP) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(keep - 1, 0, -1):
        src = path.with_name(f"{path.name}.{i}")
        dst = path.with_name(f"{path.name}.{i + 1}")
        if src.exists():
            src.replace(dst)
    if path.exists():
        path.replace(path.with_name(f"{path.name}.1"))


def prepare_instance_log(inst: InstanceSpec) -> Path:
    path = instance_log_path(inst)
    rotate_log(path)
    path.touch(mode=0o600, exist_ok=True)
    return path


def summarize_error(error: str, limit: int = 220) -> str:
    line = " ".join(part.strip() for part in str(error).splitlines() if part.strip())
    if not line:
        line = "unknown error"
    return line if len(line) <= limit else line[: limit - 1] + "…"


def append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def run_compose(inst: InstanceSpec, action: str, log_path: Path | None = None) -> None:
    cmd = compose_cmd() + ["-p", project_name(inst.name), "-f", str(compose_abs_path(inst.compose_name)), action]
    if action == "up":
        cmd.append("-d")
    if log_path is not None:
        append_log(log_path, f"$ {' '.join(cmd)}")
        with log_path.open("a", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=compose_env(inst), text=True, stdout=fh, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=compose_env(inst), text=True)
    if proc.returncode != 0:
        raise EstateCliError(f"`{' '.join(cmd)}` failed with exit {proc.returncode}")


def container_running(name: str) -> bool:
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name(name)],
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def docker_logs_tail(name: str, lines: int = 30) -> str:
    proc = subprocess.run(["docker", "logs", "--tail", str(lines), container_name(name)], text=True, capture_output=True, check=False)
    return (proc.stdout + proc.stderr).strip()


def endpoint_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=3) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def wait_ready(inst: InstanceSpec, timeout: int) -> None:
    start = time.monotonic()
    last_line = 0
    print(f"[estate] waiting for {inst.name} http://localhost:{inst.port}/v1/models (timeout {timeout}s)...")
    while True:
        if endpoint_ready(inst.port):
            elapsed = int(time.monotonic() - start)
            print(f"[estate] ✓ {inst.name} ready after {elapsed}s")
            return
        if not container_running(inst.name):
            logs = docker_logs_tail(inst.name)
            raise EstateCliError(f"container {container_name(inst.name)} stopped during boot\n{logs}")
        elapsed = int(time.monotonic() - start)
        if elapsed >= timeout:
            raise EstateCliError(f"timeout waiting for {inst.name} after {timeout}s; logs: docker logs {container_name(inst.name)}")
        if elapsed // 30 > last_line:
            last_line = elapsed // 30
            print(f"[estate]   {elapsed}s elapsed for {inst.name}, still waiting...")
        time.sleep(4)


def wait_ready_quiet(inst: InstanceSpec, timeout: int) -> int:
    start = time.monotonic()
    poll_interval = max(float(os.environ.get("CLUB3090_ESTATE_POLL_INTERVAL", "4")), 0.1)
    while True:
        if endpoint_ready(inst.port):
            return int(time.monotonic() - start)
        if not container_running(inst.name):
            logs = docker_logs_tail(inst.name)
            raise EstateCliError(f"container {container_name(inst.name)} stopped during boot\n{logs}")
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise EstateCliError(f"timeout waiting for {inst.name} after {timeout}s; logs: docker logs {container_name(inst.name)}")
        time.sleep(min(poll_interval, max(timeout - elapsed, 0.1)))


def select_instances(instances: list[InstanceSpec], only: set[str] | None) -> list[InstanceSpec]:
    if not only:
        return instances
    aliases: dict[str, InstanceSpec] = {}
    for inst in instances:
        aliases[inst.name] = inst
        aliases[container_name(inst.name)] = inst
    missing = sorted(name for name in only if name not in aliases)
    if missing:
        raise EstateCliError(f"--only references unknown instance(s): {', '.join(missing)}")
    selected_names = {aliases[name].name for name in only}
    return [inst for inst in instances if inst.name in selected_names]


def command_validate(args: argparse.Namespace) -> int:
    try:
        _, _, instances, _, _, _, _, result = validate_doc(estate_path(args.file))
    except EstateCliError as exc:
        print(f"[estate] ERROR: {exc}", file=sys.stderr)
        return 2
    print_validation_summary(instances, result)
    return 0 if result.valid else 1


def effective_parallel_jobs(requested: int | None, total: int) -> int:
    if requested is None:
        return min(total, 4)
    if requested < 1:
        raise EstateCliError("--parallel-jobs must be >= 1")
    return min(requested, total, 4)


def boot_instance_parallel(index: int, total: int, inst: InstanceSpec, timeout: int, log_path: Path) -> BootOutcome:
    start = time.monotonic()
    try:
        append_log(log_path, f"[estate] booting {inst.name}: {inst.compose_name} GPUs={list(inst.gpu_indices)} port={inst.port}")
        run_compose(inst, "up", log_path=log_path)
        elapsed_s = wait_ready_quiet(inst, timeout)
        append_log(log_path, f"[estate] healthy after {elapsed_s}s")
        return BootOutcome(index=index, total=total, inst=inst, ok=True, elapsed_s=elapsed_s, log_path=log_path)
    except Exception as exc:
        elapsed_s = int(time.monotonic() - start)
        summary = summarize_error(str(exc))
        append_log(log_path, f"[estate] ERROR after {elapsed_s}s: {summary}")
        return BootOutcome(index=index, total=total, inst=inst, ok=False, elapsed_s=elapsed_s, log_path=log_path, error=summary)


def boot_instances_parallel(selected: list[InstanceSpec], timeout: int, requested_jobs: int | None, stagger_s: float) -> int:
    if stagger_s < 0:
        raise EstateCliError("--parallel-stagger must be >= 0")
    total = len(selected)
    jobs = effective_parallel_jobs(requested_jobs, total)
    print(f"[estate] parallel boot: {total} instance(s), jobs={jobs}, stagger={stagger_s:g}s")

    futures: dict[concurrent.futures.Future[BootOutcome], tuple[int, InstanceSpec, Path]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, inst in enumerate(selected, start=1):
            log_path = prepare_instance_log(inst)
            print(f"[estate] [{i}/{total}] booting {inst.name} on GPUs {','.join(str(g) for g in inst.gpu_indices)} port {inst.port}... (started)")
            future = pool.submit(boot_instance_parallel, i, total, inst, timeout, log_path)
            futures[future] = (i, inst, log_path)
            if i < total and stagger_s > 0:
                time.sleep(stagger_s)

        outcomes = [future.result() for future in concurrent.futures.as_completed(futures)]

    outcomes.sort(key=lambda outcome: outcome.index)
    healthy = 0
    failed: list[BootOutcome] = []
    for outcome in outcomes:
        if outcome.ok:
            healthy += 1
            print(f"[estate] [{outcome.index}/{outcome.total}] {outcome.inst.name} ✓ healthy after {outcome.elapsed_s}s")
        else:
            failed.append(outcome)
            print(
                f"[estate] [{outcome.index}/{outcome.total}] {outcome.inst.name} ✗ failed after {outcome.elapsed_s}s: {outcome.error}"
            )

    print(f"[estate] Summary: {healthy}/{total} healthy, {len(failed)} failed.")
    for outcome in failed:
        print(f"[estate] Failed instance: {outcome.inst.name}. See {outcome.log_path}")
    return 0 if not failed else 1


def command_boot(args: argparse.Namespace) -> int:
    path = estate_path(args.file)
    try:
        _, data, instances, _, gpus, nvlink_active, _, result = validate_doc(path)
        print_validation_summary(instances, result)
        if not result.valid:
            return 1
        selected = select_instances(instances, parse_only(args.only))
        persist_default_estate_source(path, data, instances, gpus, nvlink_active)
        if getattr(args, "parallel", False) and len(selected) > 1:
            return boot_instances_parallel(
                selected,
                args.timeout,
                getattr(args, "parallel_jobs", None),
                getattr(args, "parallel_stagger", 15.0),
            )
        total = len(selected)
        for i, inst in enumerate(selected, start=1):
            print(f"[estate] [{i}/{total}] booting {inst.name}: {inst.compose_name} GPUs={list(inst.gpu_indices)} port={inst.port}")
            run_compose(inst, "up")
            wait_ready(inst, args.timeout)
        print("[estate] all selected instances are healthy")
        return 0
    except EstateCliError as exc:
        print(f"[estate] ERROR: {exc}", file=sys.stderr)
        return 1


def command_down(args: argparse.Namespace) -> int:
    path = estate_path(args.file)
    try:
        _, instances = parse_estate_yaml(path)
        selected = select_instances(instances, parse_only(args.only))
        for inst in selected:
            print(f"[estate] stopping {inst.name} ({container_name(inst.name)})")
            run_compose(inst, "down")
        return 0
    except EstateCliError as exc:
        print(f"[estate] ERROR: {exc}", file=sys.stderr)
        return 2


def default_gpu_block(tp: int, claimed: set[int], gpus: list[GpuInfo], fallback: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if fallback:
        return fallback
    indices = [gpu.index for gpu in gpus]
    for start in indices:
        block = tuple(range(start, start + tp))
        if all(idx in indices and idx not in claimed for idx in block):
            return block
    free = [idx for idx in indices if idx not in claimed]
    return tuple(free[:tp])


def default_port(base: int, slot: int, used: set[int], fallback: int | None = None) -> int:
    if fallback is not None:
        return fallback
    port = base + 20 * slot
    while port in used:
        port += 1
    return port


def prompt_default(prompt: str, default: str) -> str:
    reply = input(f"{prompt} [{default}]: ").strip()
    return reply or default


def parse_gpus_reply(reply: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in reply.split(",") if part.strip())
    except ValueError as exc:
        raise EstateCliError(f"invalid GPU list `{reply}`") from exc


def wizard_instance(slot: int, gpus: list[GpuInfo], existing: InstanceSpec | None, claimed: set[int], used_ports: set[int]) -> InstanceSpec:
    print("")
    print("[estate] Available composes:")
    for name, entry in sorted(COMPOSE_REGISTRY.items()):
        print(f"  {name:28s} model={entry['model']} tp={entry['tp']} port={entry['default_port']}")
    default_compose = existing.compose_name if existing else "llamacpp/default"
    compose = prompt_default("Compose", default_compose)
    if compose not in COMPOSE_REGISTRY:
        raise EstateCliError(f"unknown compose `{compose}`")
    entry = COMPOSE_REGISTRY[compose]
    tp = int(entry["tp"])
    default_gpus = default_gpu_block(tp, claimed, gpus, existing.gpu_indices if existing else None)
    gpu_reply = prompt_default("GPU indices", ",".join(str(gpu) for gpu in default_gpus))
    gpu_indices = parse_gpus_reply(gpu_reply)
    port = int(prompt_default("Port", str(default_port(int(entry["default_port"]), slot, used_ports, existing.port if existing else None))))
    name_default = existing.name if existing else f"{safe_name(compose.replace('/', '-'))}-slot{slot + 1}"
    name = prompt_default("Instance name", name_default)
    return InstanceSpec(name=name, compose_name=compose, gpu_indices=gpu_indices, port=port)


def command_wizard(args: argparse.Namespace) -> int:
    if not sys.stdin.isatty():
        print("[estate] ERROR: --estate wizard needs a TTY. Use --estate-file <path> for non-interactive boot.", file=sys.stderr)
        return 2
    path = estate_path(args.file)
    profiles = load_profiles()
    existing_data: dict[str, Any] | None = None
    existing_instances: list[InstanceSpec] = []
    if path.exists():
        existing_data, existing_instances = parse_estate_yaml(path)
    if args.replace and not existing_instances:
        print(f"[estate] ERROR: --replace {args.replace} needs an existing estate file at {path}", file=sys.stderr)
        return 2

    try:
        raw_gpus = detect_gpus_from_host()
        hardware = [profiles.hardware[gpu.hardware_id] for gpu in raw_gpus]
        nvlink_active, nvlink_pairs = detect_nvlink_pairs()
        instances = list(existing_instances) if (args.append or args.replace) else []
        boot_names: set[str] = set()

        if args.replace:
            target = next((inst for inst in instances if inst.name == args.replace), None)
            if target is None:
                raise EstateCliError(f"--replace target `{args.replace}` not found")
            instances = [inst for inst in instances if inst.name != args.replace]
            claimed = {gpu for inst in instances for gpu in inst.gpu_indices}
            used_ports = {inst.port for inst in instances}
            new_inst = wizard_instance(len(instances), raw_gpus, target, claimed, used_ports)
            instances.append(new_inst)
            boot_names.add(new_inst.name)
        else:
            while True:
                claimed = {gpu for inst in instances for gpu in inst.gpu_indices}
                used_ports = {inst.port for inst in instances}
                new_inst = wizard_instance(len(instances), raw_gpus, None, claimed, used_ports)
                instances.append(new_inst)
                boot_names.add(new_inst.name)
                more = prompt_default("Configure another instance? (y/n)", "n").lower()
                if more not in {"y", "yes"}:
                    break

        result = validate_estate(instances, hardware, profiles, nvlink_active, nvlink_pairs)
        print_validation_summary(instances, result)
        if not result.valid:
            return 1
        write_estate(path, estate_doc(instances, raw_gpus, nvlink_active, existing_data))
        print(f"[estate] wrote {path}")

        boot_now = prompt_default("Boot selected instance(s) now? (y/n)", "y").lower()
        if boot_now not in {"y", "yes"}:
            return 0
        if args.replace:
            target = next(inst for inst in existing_instances if inst.name == args.replace)
            print(f"[estate] replacing {args.replace}: stopping old container after validated estate write")
            run_compose(target, "down")
        selected = select_instances(instances, boot_names)
        for i, inst in enumerate(selected, start=1):
            print(f"[estate] [{i}/{len(selected)}] booting {inst.name}: GPUs={list(inst.gpu_indices)} port={inst.port}")
            run_compose(inst, "up")
            wait_ready(inst, args.timeout)
        return 0
    except (EstateCliError, ProfileError) as exc:
        print(f"[estate] ERROR: {exc}", file=sys.stderr)
        return 2


def command_diagnose(args: argparse.Namespace) -> int:
    path = estate_path(args.file)
    print(f"Estate triage: {path}")
    print("=" * (15 + len(str(path))))
    try:
        data, instances = parse_estate_yaml(path)
        print("[1/6] Estate file parses + schema_version supported")
        print(f"  ✓ schema_version={data.get('schema_version')} accepted; {len(instances)} instance(s) declared")
    except EstateCliError as exc:
        print("[1/6] Estate file parses + schema_version supported")
        print(f"  ✗ {exc}")
        return 2

    missing = [inst.compose_name for inst in instances if inst.compose_name not in COMPOSE_REGISTRY]
    print("[2/6] Each instance compose exists in COMPOSE_REGISTRY")
    if missing:
        for compose in missing:
            print(f"  ✗ {compose} missing from registry")
        return 1
    for inst in instances:
        print(f"  ✓ {inst.compose_name} → registry entry found")

    profiles, _, _, hardware, _, nvlink_active, nvlink_pairs, result = validate_doc(path)
    print("[3/6] Per-instance fits() PASS")
    for inst in instances:
        inst_result = result.per_instance[inst.name]
        marker = "✓" if inst_result.valid else "✗"
        passed = len(inst_result.diagnostics.get("constraints_passed", []))
        failed = len(inst_result.diagnostics.get("constraints_failed", []))
        print(f"  {marker} {inst.name}: passed={passed}, failed={failed}, elapsed={inst_result.diagnostics.get('elapsed_ms')} ms")
        for reason in inst_result.reasons:
            print(f"      - {reason}")

    print("[4/6] Estate cross-checks E1-E4")
    if result.cross_instance_failures:
        for failure in result.cross_instance_failures:
            print(f"  ✗ {failure}")
    else:
        print(f"  ✓ constraints passed: {', '.join(result.diagnostics.get('constraints_passed', []))}")
    for note in result.notes:
        print(f"  ⊘ {note}")

    print("[5/6] Calibration freshness")
    for inst in instances:
        selected = [hardware[idx] for idx in inst.gpu_indices if 0 <= idx < len(hardware)]
        status, row = calibration_status(profiles, inst.compose_name, selected)
        if row:
            print(f"  ✓ {inst.name}: {status}; {row.get('source', 'calibration row present')}")
        else:
            print(f"  ⊘ {inst.name}: {status}; no exact calibration row")

    print("[6/6] Live state (--live only)")
    if not args.live:
        print("  ⊘ skipped")
    else:
        for inst in instances:
            marker = "✓" if endpoint_ready(inst.port) else "✗"
            running = "running" if container_running(inst.name) else "not-running"
            print(f"  {marker} {inst.name}: http://localhost:{inst.port}/v1/models, container={running}")

    print("")
    print(f"Triage summary: {'GREEN' if result.valid else 'RED'}")
    return 0 if result.valid else 1


def command_report_state(args: argparse.Namespace) -> int:
    profiles = load_profiles()
    print("## Profile state")
    print("")
    print("- **Profile schema version:** 1")
    print(
        "- **Profile counts:** "
        f"{len(profiles.hardware)} hardware, {len(profiles.models)} models, "
        f"{len(profiles.workloads)} workloads, {len(profiles.engines)} engines, "
        f"{len(profiles.drafters)} drafters"
    )
    print(f"- **Compose registry:** {len(COMPOSE_REGISTRY)} entries")
    print(f"- **Canonical scenarios:** {len(CANONICAL_SCENARIOS)}")
    if profiles.calibration:
        print("- **Calibration:**")
        for model, cal in sorted(profiles.calibration.items()):
            print(f"  - {model}: {len(cal.rows)} rows")

    path = estate_path(args.file)
    if not path.exists():
        print("- **Active estate:** none (`~/.club3090/estate.yml` not found)")
        return 0
    try:
        _, _, instances, hardware, _, _, _, result = validate_doc(path)
    except EstateCliError as exc:
        print(f"- **Active estate:** present but invalid: {exc}")
        return 0
    claimed = sorted({gpu for inst in instances for gpu in inst.gpu_indices})
    print("- **Active estate:**")
    print(f"  - {len(instances)} instances from `{path}`")
    print(f"  - Validation: {'PASS' if result.valid else 'FAIL'}")
    print(f"  - GPU coverage: {len(claimed)}/{len(hardware)} cards claimed ({claimed})")
    for inst in instances:
        print(f"  - {inst.name}: {inst.compose_name}, GPUs {list(inst.gpu_indices)}, port {inst.port}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="club-3090 estate planner")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--file", default=str(DEFAULT_ESTATE_PATH))
    validate.set_defaults(func=command_validate)

    boot = sub.add_parser("boot")
    boot.add_argument("--file", default=str(DEFAULT_ESTATE_PATH))
    boot.add_argument("--only", default="")
    boot.add_argument("--timeout", type=int, default=int(os.environ.get("READY_TIMEOUT", "600")))
    boot.add_argument("--parallel", action="store_true")
    boot.add_argument("--parallel-jobs", type=int, default=None)
    boot.add_argument("--parallel-stagger", type=float, default=15.0)
    boot.set_defaults(func=command_boot)

    down = sub.add_parser("down")
    down.add_argument("--file", default=str(DEFAULT_ESTATE_PATH))
    down.add_argument("--only", default="")
    down.set_defaults(func=command_down)

    wizard = sub.add_parser("wizard")
    wizard.add_argument("--file", default=str(DEFAULT_ESTATE_PATH))
    wizard.add_argument("--append", action="store_true")
    wizard.add_argument("--replace", default="")
    wizard.add_argument("--timeout", type=int, default=int(os.environ.get("READY_TIMEOUT", "600")))
    wizard.set_defaults(func=command_wizard)

    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("file", nargs="?", default=str(DEFAULT_ESTATE_PATH))
    diagnose.add_argument("--live", action="store_true")
    diagnose.set_defaults(func=command_diagnose)

    report = sub.add_parser("report-state")
    report.add_argument("--file", default=str(DEFAULT_ESTATE_PATH))
    report.set_defaults(func=command_report_state)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
