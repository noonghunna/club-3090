#!/usr/bin/env bash
#
# Formatter for one-row BENCHMARKS.md submissions from results/rebench/<tag>/.
#
# Public functions:
#   bench_row_format <rebench-tag-dir>
#   bench_row_section <rebench-tag-dir>
#   bench_row_fixtures

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale (de_DE.iso88591 and friends) decodes reads, stdout AND argv
# with the locale codec, which crashes the launcher/emit paths (#779). Python
# already auto-enables UTF-8 mode for the C/POSIX locale, so this covers the
# case it does NOT: a genuine non-UTF-8, non-C locale. Exported, so child
# processes and nested scripts inherit it. Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

_BENCH_ROW_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_BENCH_ROW_ROOT="$(cd -- "${_BENCH_ROW_LIB_DIR}/../.." && pwd)"

# Enumerate results/rebench/<tag>/ dirs that carry every artifact the formatter
# requires. DISCOVERY, deliberately not a hardcoded tag list: this used to name
# six specific 2026-05 tags, which only ever existed as UNTRACKED local output
# on the maintainer rig (results/* is gitignored), so every fresh clone found
# zero of them and test-submit-bench.sh was red for every contributor (#776).
# The four -f guards mirror the require_file() calls in _bench_row_python and
# submit-bench.sh's own preflight, so a half-written dir from an interrupted
# rebench is skipped rather than handed downstream to fail mid-format.
bench_row_fixtures() {
  local results_dir="${_BENCH_ROW_ROOT}/results/rebench"
  [[ -d "$results_dir" ]] || return 0
  local dir
  # -L follows symlinks: a bulky results/ tree parked on another volume and
  # symlinked back in is normal practice on big rigs, and a plain `-type d`
  # silently skips those, which would reproduce #776's "found 0" from a
  # different direction. 2>/dev/null swallows the warning from a dangling link
  # (its artifacts then fail the -f guards below and it is skipped anyway).
  while IFS= read -r dir; do
    [[ -f "${dir}/_internal.json" ]] || continue
    [[ -f "${dir}/REPORT.md" ]] || continue
    [[ -f "${dir}/container-config.json" ]] || continue
    [[ -f "${dir}/rig.txt" ]] || continue
    printf '%s\n' "$dir"
  done < <(find -L "$results_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
}

bench_row_section() {
  _bench_row_python section "$1"
}

bench_row_format() {
  _bench_row_python row "$1"
}

bench_row_rig_shortname() {
  _bench_row_python rig-short "$1"
}

_bench_row_python() {
  local mode="$1"
  local tag_dir="$2"
  BENCH_ROW_REPO_ROOT="${_BENCH_ROW_ROOT}" python3 - "$mode" "$tag_dir" <<'PY'
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


MODE = sys.argv[1]
TAG_DIR = Path(sys.argv[2]).resolve()
ROOT = Path(os.environ.get("BENCH_ROW_REPO_ROOT", ".")).resolve()

# Community rigs run non-UTF-8 locales (minimal VMs / containers with LC_ALL=C),
# where a PIPED stdout defaults to ASCII — and every section name carries "×"
# ("2× RTX 3090"), so emitting a row raised UnicodeEncodeError for those users.
# Same #599 class the launcher paths already pin, and the same reason nobody
# caught it: this path had never run outside the maintainer rig (#776).
# Repro: PYTHONUTF8=0 PYTHONCOERCECLOCALE=0 LC_ALL=C (plain LC_ALL=C is coerced).
#
# This fixes the FORMATTER only. submit-bench.sh remains broken on those rigs for
# an unrelated reason (argv decoded with surrogateescape) — tracked in #777.
sys.stdout.reconfigure(encoding="utf-8")


def die(msg: str) -> None:
    print(f"[bench-row] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def require_file(name: str) -> Path:
    path = TAG_DIR / name
    if not path.is_file():
        die(f"missing required artifact: {path}")
    return path


def first_container(blob: Any) -> dict[str, Any]:
    if isinstance(blob, list) and blob:
        return blob[0] if isinstance(blob[0], dict) else {}
    return blob if isinstance(blob, dict) else {}


def flag(cmd: list[str], name: str) -> str:
    try:
        i = cmd.index(name)
        return str(cmd[i + 1])
    except Exception:
        return "?"


def rel(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def infer_compose_path(container_name: str, served: str, tp: str) -> str:
    name = container_name.lstrip("/")
    is_gemma = "gemma" in name or "gemma" in served
    model_root = "models/gemma-4-31b/vllm/compose" if is_gemma else "models/qwen3.6-27b/vllm/compose"

    mapping = {
        "dual-int8-tq3": "dual/autoround-int4/int8-tq3.yml",
        "dual-tq3-mtp-genesis": "dual/autoround-int4/tq3-mtp-genesis.yml",
        "dual-tq3-nomtp": "dual/autoround-int4/tq3-nomtp.yml",
        "dual-tq3-mtp": "dual/autoround-int4/tq3-mtp.yml",
        "dual-int8": "dual/autoround-int4/int8.yml",
        "dual-bf16": "dual/autoround-int4/bf16.yml",
        "dual-dflash-noviz": "dual/autoround-int4/dflash-noviz.yml",
        "dual-dflash": "dual/autoround-int4/dflash.yml",
        "dual-turbo": "dual/autoround-int4/turbo.yml",
        "dual": "dual/autoround-int4/fp8-mtp.yml",
        "minimal": "single/autoround-int4/minimal.yml",
        "tools-text": "single/autoround-int4/tools-text.yml",
        "long-text-no-mtp": "single/autoround-int4/long-text-no-mtp.yml",
        "long-text": "single/autoround-int4/long-text.yml",
        "long-vision": "single/autoround-int4/long-vision.yml",
    }
    for needle, suffix in mapping.items():
        if needle in name:
            return f"{model_root}/{suffix}"
    if tp == "4":
        return "models/qwen3.6-27b/vllm/compose/multi4/autoround-int4/fp8-mtp.yml"
    if tp == "2":
        return f"{model_root}/dual/autoround-int4/fp8-mtp.yml"
    return f"{model_root}/single/autoround-int4/tq3-mtp.yml"


def compose_display(compose_path: str, served: str) -> str:
    path = rel(compose_path).replace("\\", "/")
    try:
        sys.path.insert(0, str(ROOT))
        from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, DEFAULTS

        by_path = {entry["compose_path"]: (key, entry) for key, entry in COMPOSE_REGISTRY.items()}
        hit = by_path.get(path)
        if hit is not None:
            _key, entry = hit
            parts = Path(path).parts
            topology = parts[parts.index("compose") + 1]
            engine_dir = parts[parts.index(entry["model"]) + 1]
            default_key = DEFAULTS.get((entry["model"], engine_dir, topology))
            if default_key and COMPOSE_REGISTRY[default_key]["compose_path"] == path:
                return f"{engine_dir}/default" if topology == "single" else f"{engine_dir}/{topology}/default"
            quant = parts[parts.index("compose") + 2]
            base = Path(path).stem
            if quant == entry.get("weights_variant") == "autoround-int4":
                return f"{topology}-{base}"
            return f"{topology}-{quant}-{base}"
    except Exception:
        pass

    parts = path.split("/")
    base = parts[-1] if parts else path
    parent = parts[-2] if len(parts) >= 2 else ""
    if parent in {"dual", "multi4"} and not base.startswith(f"{parent}-"):
        return f"{parent}-{base}"
    return base


def parse_rig(rig_txt: str) -> dict[str, Any]:
    out: dict[str, Any] = {"gpus": []}
    for raw in rig_txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("hostname:"):
            out["hostname"] = line.split(":", 1)[1].strip()
        elif line.startswith("GPU "):
            gpu = line.split(":", 1)[1].split("(UUID", 1)[0].strip()
            out["gpus"].append(gpu)
        elif line.startswith("power_cap_w:"):
            out["power_cap_w"] = line.split(":", 1)[1].strip()
    return out


def simplify_gpu(name: str) -> str:
    name = re.sub(r"^NVIDIA\s+", "", name)
    name = re.sub(r"^GeForce\s+", "", name)
    name = re.sub(r"^RTX\s+", "", name)
    return name.strip()


def rig_shape(rig: dict[str, Any]) -> str:
    gpus = [simplify_gpu(g) for g in rig.get("gpus") or []]
    if not gpus:
        shape = "rig"
    elif len(set(gpus)) == 1:
        shape = f"{len(gpus)}× {gpus[0]}"
    else:
        shape = " + ".join(gpus)
    power = str(rig.get("power_cap_w") or "").strip()
    if power:
        try:
            power = f"{float(power):.0f} W/card"
        except Exception:
            power = f"{power} W/card"
        return f"{shape}, {power}"
    return shape


def rig_cell(rig: dict[str, Any]) -> str:
    user = os.environ.get("BENCH_ROW_GITHUB_USER", "").strip().lstrip("@") or "your-handle"
    return f"@{user} ({rig_shape(rig)})"


def short_date(tag: str, report: str) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", tag)
    if m:
        return m.group(1)
    m = re.search(r"\*\*Date:\*\*\s*(20\d{2}-\d{2}-\d{2})", report)
    return m.group(1) if m else "—"


def kv_display(raw: str, served: str) -> str:
    raw = (raw or "?").strip("`")
    lowered = raw.lower()
    if lowered in {"turboquant_3bit_nc", "tq3"}:
        return "TQ3"
    if lowered in {"fp8_e5m2", "fp8", "fp8_e4m3"}:
        return "fp8"
    if lowered in {"bfloat16", "bf16"}:
        return "bf16"
    if lowered == "auto":
        return "bf16"
    if lowered == "int8_per_token_head":
        return "int8_per_token_head"
    return raw or "?"


def fmt_ctx(value: str) -> str:
    try:
        n = int(str(value).replace(",", ""))
    except Exception:
        return str(value or "?")
    if n >= 1000:
        return f"{round(n / 1000):.0f}K"
    return str(n)


def fmt_tps(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "?"


def parse_mib(text: str) -> int | None:
    m = re.search(r"([\d.]+)\s*MiB", str(text))
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None


def peak_vram(internal: dict[str, Any], gpu_count: int) -> str:
    vals: list[int] = []
    for g in (((internal.get("bench") or {}).get("gpu_state")) or []):
        mib = parse_mib(g.get("mem_used", ""))
        if mib is not None:
            vals.append(mib)
    if not vals:
        return "TBD"
    suffix = "/card" if gpu_count > 1 else ""
    return f"{max(vals) / 1024:.1f} GB{suffix}"


def pp_display(bench: dict[str, Any]) -> str:
    vals: list[float] = []
    for kind in ("narrative", "code"):
        try:
            val = bench.get(kind, {}).get("pp_tps_mean")
            if val is not None:
                vals.append(float(val))
        except Exception:
            pass
    if vals:
        return f"{sum(vals) / len(vals):.0f}"
    try:
        val = bench.get("prompt_processing", {}).get("pp_tps_mean")
        if val is not None:
            return f"{float(val):.0f}"
    except Exception:
        pass
    return "—"


def parse_jsonish(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def verify_summary(report: str) -> str:
    m = re.search(r"Verify-stress:\s+\*\*(\d+/\d+)\*\*", report)
    if m:
        return m.group(1)
    m = re.search(r"\*\*Overall:\*\*\s+(PASS|FAIL|\?)", report)
    return m.group(1) if m else "?"


def soak_note(soak: dict[str, Any]) -> str:
    verdict = str(soak.get("verdict") or "").upper()
    silent = str(soak.get("silent_empty") or "")
    growth = str(soak.get("max_growth_mib") or "")
    if not verdict:
        return "Soak: —"
    if verdict == "PASS" and (silent.startswith("0 ") or silent.startswith("0/") or silent == "0"):
        return "Soak: ✓ PASS"
    if verdict == "PASS":
        return f"Soak: ⚠ borderline ({silent or growth})"
    return f"Soak: ✗ {verdict}"


def section_name(compose_path: str, served: str, tp: str, container: str) -> str:
    if "gemma" in served or "gemma" in container or "gemma-4-31b" in compose_path:
        return "Gemma 4 31B (community-experimental)"
    path = compose_path.replace("\\", "/")
    if "llama-cpp" in path or "llama-cpp" in container:
        return "Single-card (1× RTX 3090) — llama.cpp"
    if "/multi4/" in path or tp == "4":
        return "Quad-card (4× RTX 3090, TP=4)"
    if "/dual/" in path or tp == "2":
        return "Dual-card (2× RTX 3090, TP=2)"
    return "Single-card (1× RTX 3090) — vLLM"


def load() -> dict[str, Any]:
    if not TAG_DIR.is_dir():
        die(f"tag dir not found: {TAG_DIR}")
    internal = read_json(require_file("_internal.json"))
    if not isinstance(internal, dict):
        die(f"invalid JSON artifact: {TAG_DIR / '_internal.json'}")
    report = read_text(require_file("REPORT.md"))
    config_blob = first_container(read_json(require_file("container-config.json")))
    cfg = config_blob.get("Config") or {}
    labels = cfg.get("Labels") or {}
    cmd = cfg.get("Cmd") or []
    env = cfg.get("Env") or []
    name = str(config_blob.get("Name") or "").lstrip("/")
    served = flag(cmd, "--served-model-name")
    tp = flag(cmd, "--tensor-parallel-size")
    compose = rel(labels.get("com.docker.compose.project.config_files") or "")
    if not compose:
        compose = infer_compose_path(name, served, tp)
    rig = parse_rig(read_text(require_file("rig.txt")))
    tag = TAG_DIR.name
    spec = parse_jsonish(flag(cmd, "--speculative-config"))

    return {
        "tag": tag,
        "report": report,
        "internal": internal,
        "container": {
            "name": name,
            "served": served,
            "tp": tp,
            "compose": compose,
            "kv": flag(cmd, "--kv-cache-dtype"),
            "max_ctx": flag(cmd, "--max-model-len"),
            "max_num_seqs": flag(cmd, "--max-num-seqs"),
            "mem_util": flag(cmd, "--gpu-memory-utilization"),
            "image": cfg.get("Image", ""),
            "spec": spec,
            "genesis": any(str(e).startswith("GENESIS_") for e in env),
        },
        "rig": rig,
        "date": short_date(tag, report),
    }


def format_row(data: dict[str, Any]) -> str:
    c = data["container"]
    internal = data["internal"]
    bench = internal.get("bench") or {}
    narrative = bench.get("narrative") or {}
    code = bench.get("code") or {}
    mtp = bench.get("mtp") or {}
    quality = internal.get("quality") or {}
    aider = internal.get("aider") or {}
    soak = internal.get("soak") or {}
    rig = data["rig"]
    gpu_count = max(len(rig.get("gpus") or []), 1)
    section = section_name(c["compose"], c["served"], c["tp"], c["name"])
    verify = verify_summary(data["report"])
    compose = compose_display(c["compose"], c["served"])
    kv = kv_display(c["kv"], c["served"])
    max_ctx = fmt_ctx(c["max_ctx"])
    tps = f"**{fmt_tps(narrative.get('wall_tps_mean'))} / {fmt_tps(code.get('wall_tps_mean'))}**"
    pp = pp_display(bench)
    peak = peak_vram(internal, gpu_count)
    spec_n = c["spec"].get("num_speculative_tokens")
    notes = [soak_note(soak), f"verify-stress {verify}"]
    if quality.get("total_passed") is not None:
        notes.append(f"quality {quality.get('total_passed')}/{quality.get('total_total')}")
    if aider.get("total_count"):
        notes.append(f"aider {aider.get('passed_count')}/{aider.get('total_count')}")
    if mtp.get("mean_accept_length") is not None:
        n_text = f" n={spec_n}" if spec_n is not None else ""
        notes.append(
            f"MTP{n_text} AL {float(mtp['mean_accept_length']):.2f}, "
            f"accept {float(mtp.get('avg_accept_rate', 0)):.1f}%"
        )
    if c.get("genesis"):
        notes.append("Genesis on")

    note_cell = "; ".join(notes) + f". Report: `results/rebench/{data['tag']}/REPORT.md`."

    if section == "Gemma 4 31B (community-experimental)":
        al = f"{float(mtp['mean_accept_length']):.2f}" if mtp.get("mean_accept_length") is not None else "—"
        per_pos = str(mtp.get("per_position") or "—")
        return (
            f"| `{compose}` | {rig_cell(rig)} | {kv} | {max_ctx} | {tps} | "
            f"{pp} | {al} | {per_pos} | {peak} | {data['date']} | {note_cell} |"
        )

    return (
        f"| `{compose}` | {rig_cell(rig)} | {kv} | {max_ctx} | {tps} | "
        f"{pp} | {peak} | {data['date']} | {note_cell} |"
    )


data = load()
if MODE == "section":
    c = data["container"]
    print(section_name(c["compose"], c["served"], c["tp"], c["name"]))
elif MODE == "row":
    print(format_row(data))
elif MODE == "rig-short":
    print(data["tag"])
else:
    die(f"unknown mode: {MODE}")
PY
}
