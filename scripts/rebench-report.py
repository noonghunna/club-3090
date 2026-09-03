#!/usr/bin/env python3
"""rebench-report.py — synthesize REPORT.md from a results/rebench/<tag>/ dir.

Reads the raw logs and JSON artifacts that scripts/rebench-full.sh writes,
extracts the structured numbers, and produces a single REPORT.md at the top
of the tag dir. Re-runnable standalone — just pass the tag-dir path.

Usage:
    python3 scripts/rebench-report.py results/rebench/<tag>/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# --- file readers -----------------------------------------------------------

def read_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return None


# --- section: meta + config -------------------------------------------------

def parse_container_config(blob: dict | None) -> dict:
    """Pull the relevant bits from `docker inspect <container>`."""
    if not blob:
        return {}
    # docker inspect returns a list with one element
    if isinstance(blob, list) and blob:
        blob = blob[0]
    cfg = blob.get("Config", {}) or {}
    state = blob.get("State", {}) or {}
    mounts = blob.get("Mounts", [])
    cmd = cfg.get("Cmd") or []
    env = cfg.get("Env") or []

    # extract flag values from Cmd list
    def get_flag(name: str) -> str | None:
        try:
            i = cmd.index(name)
            return cmd[i + 1]
        except (ValueError, IndexError):
            return None

    # mounts that look like vendored patches
    patches = []
    for m in mounts:
        src = m.get("Source", "")
        if "/patches/" in src and m.get("Type") == "bind":
            # take the patch dir name
            parts = src.split("/patches/")
            if len(parts) > 1:
                name = parts[1].split("/")[0]
                if name not in patches:
                    patches.append(name)

    return {
        "image": cfg.get("Image"),
        "name": blob.get("Name", "").lstrip("/"),
        "status": state.get("Status"),
        "model": get_flag("--model") or "?",
        "served_model_name": get_flag("--served-model-name") or "?",
        "quantization": get_flag("--quantization") or "?",
        "dtype": get_flag("--dtype") or "?",
        "tensor_parallel_size": get_flag("--tensor-parallel-size") or "?",
        "max_model_len": get_flag("--max-model-len") or "?",
        "gpu_memory_utilization": get_flag("--gpu-memory-utilization") or "?",
        "max_num_seqs": get_flag("--max-num-seqs") or "?",
        "max_num_batched_tokens": get_flag("--max-num-batched-tokens") or "?",
        "kv_cache_dtype": get_flag("--kv-cache-dtype") or "?",
        "speculative_config": get_flag("--speculative-config") or "?",
        "patches": patches,
        "env": env,
    }


def parse_model_config_json(model_dir: Path) -> dict:
    """Read config.json from the served model dir to get quant metadata."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    blob = read_json(cfg_path)
    if not blob:
        return {}
    q = blob.get("quantization_config") or {}
    return {
        "model_type": blob.get("model_type"),
        "architectures": blob.get("architectures") or [],
        "quant_method": q.get("quant_method"),
        "bits": q.get("bits"),
        "group_size": q.get("group_size"),
    }


# --- section: vLLM boot log (KV pool + max concurrency) ---------------------

def parse_vllm_boot(boot_log: str) -> dict:
    """Pull KV cache size and Max concurrency lines."""
    out = {}
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", boot_log)
    if m:
        out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", boot_log)
    if m:
        out["max_concurrency_request_size"] = int(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    m = re.search(r"Available KV cache memory: ([\d.]+) GiB", boot_log)
    if m:
        out["available_kv_cache_gib"] = float(m.group(1))
    m = re.search(r"Model loading took ([\d.]+) GiB memory", boot_log)
    if m:
        out["model_load_gib"] = float(m.group(1))
    return out


# --- section: bench.sh ------------------------------------------------------

def parse_bench(log: str) -> dict:
    """Extract narrative + code TPS summaries from bench.sh output."""
    out: dict[str, Any] = {}
    for kind in ("narrative", "code", "prompt-processing"):
        block = re.search(
            rf"=== summary \[{kind}\] \(n=\d+\) ===\s*\n"
            rf"(?P<body>.*?)(?=\n==========|\n=== GPU state ===|\n=== Last|\Z)",
            log,
            re.DOTALL,
        )
        if not block:
            continue
        body = block.group("body")
        parsed: dict[str, Any] = {}
        m = re.search(r"\s+wall_TPS\s+mean=\s*([\d.]+)\s+std=\s*([\d.]+)\s+CV=\s*([\d.]+)%", body)
        if m:
            parsed["wall_tps_mean"] = float(m.group(1))
            parsed["wall_tps_cv"] = float(m.group(3))
        m = re.search(r"\s+decode_TPS\s+mean=\s*([\d.]+)\s+std=\s*([\d.]+)\s+CV=\s*([\d.]+)%", body)
        if m:
            parsed["decode_tps_mean"] = float(m.group(1))
            parsed["decode_tps_cv"] = float(m.group(3))
        m = re.search(r"\s+TTFT\s+mean=\s*([\d.]+)ms", body)
        if m:
            parsed["ttft_ms_mean"] = float(m.group(1))
        m = re.search(r"\s+PP tok/s\s+mean=\s*([\d.]+)\s+std=\s*([\d.]+)\s+CV=\s*([\d.]+)%", body)
        if m:
            parsed["pp_tps_mean"] = float(m.group(1))
            parsed["pp_tps_cv"] = float(m.group(3))
        if parsed:
            out[kind.replace("-", "_")] = parsed
    # GPU state at end
    gpu_block = re.search(r"=== GPU state ===\s*\n((?:\d.+\n){1,4})", log)
    if gpu_block:
        gpu_lines = gpu_block.group(1).strip().split("\n")
        out["gpu_state"] = []
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                out["gpu_state"].append({
                    "idx": parts[0],
                    "util_pct": parts[1],
                    "mem_used": parts[2],
                    "mem_total": parts[3],
                    "power_w": parts[4],
                    "temp_c": parts[5],
                })
    # MTP last metrics line
    mtp_lines = re.findall(
        r"Mean acceptance length: ([\d.]+), .* "
        r"Per-position acceptance rate: ([^,]+), .* "
        r"Avg Draft acceptance rate: ([\d.]+)%",
        log,
    )
    if mtp_lines:
        last = mtp_lines[-1]
        out["mtp"] = {
            "mean_accept_length": float(last[0]),
            "per_position": last[1].strip(),
            "avg_accept_rate": float(last[2]),
        }
    return out


# --- section: verify-stress -------------------------------------------------

def parse_verify_stress(log: str) -> dict:
    """Extract the 7-check verdict table."""
    checks = []
    # match lines like "[1/7] description ..." followed by ✓ or ✗
    for m in re.finditer(r"\[(\d+/\d+)\] (.+?) \.\.\.\n(.*?)(?=\n\[\d+/\d+\]|\nAll stress|\Z)", log, re.DOTALL):
        idx = m.group(1)
        desc = m.group(2).strip()
        body = m.group(3)
        verdict = "PASS" if "✓" in body else ("FAIL" if "✗" in body else "?")
        checks.append({"idx": idx, "desc": desc, "verdict": verdict})
    overall = "PASS" if "All stress / boundary checks passed" in log else (
        "FAIL" if any(c["verdict"] == "FAIL" for c in checks) else "?"
    )
    result = {"checks": checks, "overall": overall}

    # Ceiling VRAM margin (#184). verify-stress.sh prints free VRAM at the
    # deepest rung and hard-fails below the guard, but a margin that *passes*
    # the guard yet sits close to it can still OOM on a second run / server
    # restart (laurimyllari's marginal-but-passing case). Surface the number so
    # a PASS isn't mistaken for comfortable headroom.
    free = thresh = None
    m_thin = re.search(r"VRAM margin thin at ceiling:\s*(\d+)\s*MB free\s*<\s*(\d+)\s*MB", log)
    if m_thin:
        free, thresh = int(m_thin.group(1)), int(m_thin.group(2))
    else:
        m_ok = re.search(r"VRAM:\s*\d+\s*\S+\s*(\d+)\s*MB[^\n]*margin threshold=(\d+)", log)
        if m_ok:
            free, thresh = int(m_ok.group(1)), int(m_ok.group(2))
    if free is not None and thresh:
        result["ceiling_vram_free_mb"] = free
        result["vram_threshold_mb"] = thresh
        result["vram_status"] = (
            "thin" if free < thresh else "marginal" if free < 2 * thresh else "comfortable"
        )
    return result


# --- section: quality-full --------------------------------------------------

def parse_quality(blob: dict | None) -> dict:
    if not blob:
        return {}
    packs = []
    failure_examples: dict[str, list[str]] = defaultdict(list)
    for p in blob.get("packs", []):
        if p.get("pack_id") == "aider-polyglot-30":
            continue  # rendered separately
        lat = p.get("latency") or {}
        packs.append({
            "pack_id": p.get("pack_id"),
            "passed": p.get("passed"),
            "total": p.get("total"),
            "score_pct": round(100 * (p.get("score") or 0)),
            "p50_latency_s": lat.get("p50") if isinstance(lat, dict) else None,
            "p95_latency_s": lat.get("p95") if isinstance(lat, dict) else None,
            "mean_latency_s": lat.get("mean") if isinstance(lat, dict) else None,
            "status": p.get("status"),
        })
        # gather top-3 failures per pack for the report appendix
        for s in (p.get("scenarios") or []):
            if not s.get("passed", True):
                pack = p.get("pack_id", "?")
                sid = s.get("id") or (s.get("raw_scenario") or {}).get("id", "?")
                fail = s.get("failure_mode") or "?"
                detail = (s.get("detail") or "")[:140]
                if len(failure_examples[pack]) < 3:
                    failure_examples[pack].append(f"{sid}: {fail} — {detail}")
    total_passed = sum(p["passed"] or 0 for p in packs)
    total_total = sum(p["total"] or 0 for p in packs)
    return {
        "packs": packs,
        "total_passed": total_passed,
        "total_total": total_total,
        "total_pct": round(100 * total_passed / max(total_total, 1)),
        "failure_examples": dict(failure_examples),
    }


# --- section: soak ----------------------------------------------------------

def parse_soak(log: str) -> dict:
    out = {}
    for key in ("verdict", "max_growth_mib", "errors", "silent_empty",
                "p50_decode_tps", "p95_ttft_ms", "tps_retention", "boot_vram_mib"):
        m = re.search(rf"\b{key}\b\s+(\S.+)$", log, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    return out


# --- section: aider-polyglot ------------------------------------------------

def parse_aider(blob: dict | None) -> dict:
    """upstream_per_exercise can be either:
    - dict keyed by '<lang>/<exercise>' (current shape from aider's benchmark.py)
    - list of {language, passed} dicts (older or alt shape)
    Handle both.
    """
    if not blob:
        return {}
    for p in blob.get("packs", []):
        if p.get("pack_id") != "aider-polyglot-30":
            continue
        s = (p.get("scenarios") or [{}])[0]
        latency_s = s.get("latency_seconds")
        trace = s.get("verifier_trace", {})
        # Schema v3 nests the trace under .trace; older versions are flat.
        inner = trace.get("trace", trace) or {}
        per_ex = inner.get("upstream_per_exercise") or inner.get("per_exercise") or {}

        per_lang: dict = defaultdict(lambda: [0, 0])
        passed_total = 0
        total = 0

        if isinstance(per_ex, dict):
            # current shape: keyed by "<lang>/<exercise>"
            for path, entry in per_ex.items():
                lang = path.split("/", 1)[0].lower() if "/" in path else "?"
                per_lang[lang][1] += 1
                total += 1
                if entry.get("passed") is True or entry.get("tests_passed") is True \
                        or entry.get("status") == "pass":
                    per_lang[lang][0] += 1
                    passed_total += 1
        elif isinstance(per_ex, list):
            for entry in per_ex:
                lang = (entry.get("language") or "?").lower()
                per_lang[lang][1] += 1
                total += 1
                if entry.get("passed") is True or entry.get("status") == "pass":
                    per_lang[lang][0] += 1
                    passed_total += 1

        # Prefer the top-level fields if they're set; fall back to our tally.
        passed_count = s.get("passed_count") if s.get("passed_count") is not None else passed_total
        total_count = s.get("total_count") if s.get("total_count") is not None else total
        pass_rate = s.get("pass_rate")
        if pass_rate is None and total_count:
            pass_rate = passed_count / total_count

        return {
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "total_count": total_count,
            "wall_seconds": latency_s,
            "per_language": dict(per_lang),
            "aider_commit": inner.get("aider_pinned_commit", ""),
            "polyglot_commit": inner.get("polyglot_pinned_commit", ""),
            "edit_format": inner.get("edit_format", ""),
        }
    return {}


# --- markdown rendering -----------------------------------------------------

def render(report: dict) -> str:
    lines = []
    lines.append(f"# Rebench report — {report.get('tag', '?')}")
    lines.append("")
    lines.append(f"_Generated by `scripts/rebench-report.py` from `{report.get('tag_dir')}`_")
    lines.append("")

    # --- headline TL;DR (auto-computed from data) ---
    tldr = report.get("tldr", [])
    if tldr:
        lines.append("## TL;DR")
        lines.append("")
        for bullet in tldr:
            lines.append(f"- {bullet}")
        lines.append("")

    # --- meta ---
    meta = report.get("meta", {})
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- **Tag:** `{report.get('tag', '?')}`")
    lines.append(f"- **Date:** {report.get('date', '?')}")
    lines.append(f"- **Repo commit:** `{report.get('commit_sha', '?')}`")
    if meta.get("model_config"):
        mc = meta["model_config"]
        lines.append(f"- **Model arch:** {mc.get('model_type', '?')} ({(mc.get('architectures') or ['?'])[0]})")
        lines.append(f"- **Quant:** {mc.get('quant_method', '?')} {mc.get('bits', '?')}-bit, group_size {mc.get('group_size', '?')}")
    if meta.get("container"):
        c = meta["container"]
        lines.append(f"- **Served as:** `{c.get('served_model_name')}` from `{c.get('model')}`")
        lines.append(f"- **vLLM image:** `{c.get('image')}`")
        lines.append(f"- **Container:** `{c.get('name')}`")
    rig = meta.get("rig", {})
    if rig:
        if rig.get("hostname"):
            lines.append(f"- **Rig hostname:** `{rig['hostname']}`")
        if rig.get("gpus"):
            lines.append(f"- **GPUs:** {', '.join(rig['gpus'])}")
        if rig.get("power_cap_w"):
            lines.append(f"- **Power cap:** {rig['power_cap_w']} W/card")
    lines.append("")

    # --- config ---
    lines.append("## Config")
    lines.append("")
    if meta.get("container"):
        c = meta["container"]
        lines.append("| Setting | Value |")
        lines.append("|---|---|")
        lines.append(f"| `--tensor-parallel-size` | {c.get('tensor_parallel_size')} |")
        lines.append(f"| `--max-model-len` | {c.get('max_model_len')} |")
        lines.append(f"| `--gpu-memory-utilization` | {c.get('gpu_memory_utilization')} |")
        lines.append(f"| `--max-num-seqs` | {c.get('max_num_seqs')} |")
        lines.append(f"| `--max-num-batched-tokens` | {c.get('max_num_batched_tokens')} |")
        lines.append(f"| `--kv-cache-dtype` | `{c.get('kv_cache_dtype')}` |")
        lines.append(f"| `--dtype` | `{c.get('dtype')}` |")
        lines.append(f"| `--quantization` | `{c.get('quantization')}` |")
        lines.append(f"| `--speculative-config` | `{c.get('speculative_config')}` |")
        patches = c.get("patches") or []
        lines.append(f"| Patches mounted | {', '.join(f'`{p}`' for p in patches) if patches else 'none'} |")
        # Genesis detection
        genesis_envs = [e for e in (c.get("env") or []) if e.startswith("GENESIS_")]
        lines.append(f"| Genesis | {'on (' + str(len(genesis_envs)) + ' GENESIS_* env vars)' if genesis_envs else 'none'} |")
    lines.append("")

    # --- bench performance ---
    bench = report.get("bench", {})
    lines.append("## Performance — `bench.sh`")
    lines.append("")
    if bench.get("narrative") or bench.get("code"):
        lines.append("| Bench | wall TPS | decode TPS | PP tok/s | TTFT | CV (wall/decode) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for kind in ("narrative", "code"):
            b = bench.get(kind)
            if b:
                pp = f"{b['pp_tps_mean']:.0f}" if b.get("pp_tps_mean") is not None else "n/a"
                lines.append(
                    f"| {kind} | {b['wall_tps_mean']:.2f} | **{b['decode_tps_mean']:.2f}** | {pp} | "
                    f"{b['ttft_ms_mean']:.0f} ms | {b['wall_tps_cv']:.1f}% / {b['decode_tps_cv']:.1f}% |"
                )
        pp_fallback = bench.get("prompt_processing")
        if pp_fallback and pp_fallback.get("pp_tps_mean") is not None:
            lines.append(
                f"| prompt-processing fallback | — | — | **{pp_fallback['pp_tps_mean']:.0f}** | "
                f"{pp_fallback.get('ttft_ms_mean', 0):.0f} ms | — |"
            )
        if bench.get("mtp"):
            m = bench["mtp"]
            lines.append("")
            lines.append(f"**MTP (warm, last metric):** mean accept length {m['mean_accept_length']:.2f}, "
                         f"avg accept rate {m['avg_accept_rate']:.1f}%, per-position {m['per_position']}")
        if bench.get("gpu_state"):
            lines.append("")
            lines.append("**GPU state at bench end:**")
            lines.append("")
            lines.append("| GPU | Util | Mem used / total | Power | Temp |")
            lines.append("|---|---|---|---|---|")
            for g in bench["gpu_state"]:
                lines.append(f"| {g['idx']} | {g['util_pct']} | {g['mem_used']} / {g['mem_total']} | {g['power_w']} | {g['temp_c']} |")
    else:
        lines.append("_(bench artifacts missing)_")
    lines.append("")

    # --- concurrency + VRAM ---
    boot = report.get("vllm_boot", {})
    lines.append("## Concurrency + VRAM")
    lines.append("")
    if boot:
        kv_tokens = boot.get("kv_cache_tokens")
        max_conc = boot.get("max_concurrency")
        req_size = boot.get("max_concurrency_request_size")
        avail = boot.get("available_kv_cache_gib")
        model_load = boot.get("model_load_gib")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        if model_load:
            lines.append(f"| Model load footprint | {model_load:.2f} GiB |")
        if avail:
            lines.append(f"| Available KV cache memory (per card, post-profiling) | {avail:.2f} GiB |")
        if kv_tokens:
            lines.append(f"| GPU KV cache size | **{kv_tokens:,} tokens** |")
        if max_conc and req_size:
            lines.append(f"| Max concurrency @ {req_size:,} tokens/req | **{max_conc:.2f}×** |")
            if req_size > 1:
                for ctx in (100_000, 32_000):
                    practical = max_conc * (req_size / ctx)
                    lines.append(f"| Practical concurrency @ {ctx:,} tokens/req | ~{practical:.1f}× |")
    else:
        lines.append("_(no vLLM boot log captured)_")
    lines.append("")

    # --- verify-stress ---
    stress = report.get("verify_stress", {})
    lines.append("## Verify-stress — 7-check boundary matrix")
    lines.append("")
    if stress.get("checks"):
        lines.append(f"**Overall:** {stress.get('overall')}")
        lines.append("")
        lines.append("| # | Check | Verdict |")
        lines.append("|---|---|---|")
        for c in stress["checks"]:
            lines.append(f"| {c['idx']} | {c['desc']} | {c['verdict']} |")
        if stress.get("ceiling_vram_free_mb") is not None:
            free = stress["ceiling_vram_free_mb"]
            thresh = stress["vram_threshold_mb"]
            note = {
                "thin": " — ⚠ **THIN** (below the sustained-agent guard; verify-stress fails this check)",
                "marginal": " — ⚠ **marginal**: within 2× the guard, so a second run or server restart may OOM. Lower `CTX_SIZE` for sustained agent load.",
                "comfortable": " — ✓ comfortable",
            }[stress["vram_status"]]
            lines.append("")
            lines.append(f"**Ceiling VRAM margin:** {free} MB free (guard {thresh} MB){note}")
    else:
        lines.append("_(verify-stress log missing or unparsed)_")
    lines.append("")

    # --- quality ---
    quality = report.get("quality", {})
    lines.append("## Quality — `quality-test.sh --full`")
    lines.append("")
    if quality.get("packs"):
        lines.append("| Pack | Pass / Total | Score | p50 latency | p95 latency |")
        lines.append("|---|---:|---:|---:|---:|")
        for p in quality["packs"]:
            p50 = f"{p['p50_latency_s']:.2f}s" if p.get("p50_latency_s") is not None else "—"
            p95 = f"{p['p95_latency_s']:.2f}s" if p.get("p95_latency_s") is not None else "—"
            lines.append(f"| {p['pack_id']} | {p['passed']} / {p['total']} | {p['score_pct']}% | {p50} | {p95} |")
        lines.append(f"| **TOTAL** | **{quality['total_passed']} / {quality['total_total']}** | **{quality['total_pct']}%** | | |")
        if quality.get("one_liner"):
            lines.append("")
            lines.append("**Paste-ready compose `Quality:` schema line:**")
            lines.append("```")
            lines.append(quality["one_liner"])
            lines.append("```")
        if quality.get("failure_examples"):
            lines.append("")
            lines.append("**Failure examples (top 3 per pack):**")
            for pack, examples in sorted(quality["failure_examples"].items()):
                lines.append(f"- `{pack}`:")
                for ex in examples:
                    lines.append(f"  - {ex}")
    else:
        lines.append("_(quality JSON missing or unparsed)_")
    lines.append("")

    # --- soak ---
    soak = report.get("soak", {})
    lines.append("## Soak — `soak-test.sh`")
    lines.append("")
    if soak:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k in ("verdict", "silent_empty", "p50_decode_tps", "p95_ttft_ms",
                  "tps_retention", "max_growth_mib", "errors", "boot_vram_mib"):
            if k in soak:
                lines.append(f"| `{k}` | {soak[k]} |")
    else:
        lines.append("_(soak log missing or unparsed)_")
    lines.append("")

    # --- aider-polyglot ---
    aider = report.get("aider", {})
    lines.append("## Aider Polyglot 30 — per-language breakdown")
    lines.append("")
    if aider.get("per_language"):
        lines.append(f"**Total:** {aider.get('passed_count', '?')} / {aider.get('total_count', '?')} "
                     f"({100 * (aider.get('pass_rate') or 0):.1f}%) · "
                     f"wall {aider.get('wall_seconds', 0):.0f}s")
        lines.append("")
        lines.append("| Language | Pass / Total | Score |")
        lines.append("|---|---:|---:|")
        for lang, (passed, total) in sorted(aider["per_language"].items()):
            pct = round(100 * passed / total) if total else 0
            lines.append(f"| {lang} | {passed} / {total} | {pct}% |")
    else:
        lines.append("_(aider-polyglot artifacts missing or no per-exercise trace)_")
    lines.append("")

    # --- phase timings ---
    timings = report.get("timings", {})
    if timings:
        lines.append("## Phase timings")
        lines.append("")
        lines.append("| Phase | Duration |")
        lines.append("|---|---:|")
        total_s = 0
        for phase, secs in timings.items():
            if isinstance(secs, (int, float)) and secs > 0:
                m, s = divmod(int(secs), 60)
                lines.append(f"| {phase} | {m}m {s}s |")
                total_s += secs
        if total_s:
            tm, ts = divmod(int(total_s), 60)
            lines.append(f"| **Total** | **{tm}m {ts}s** |")
        lines.append("")

    # --- reproducer ---
    repro = report.get("reproducer", {})
    if repro:
        lines.append("## Reproduce on your rig")
        lines.append("")
        lines.append("```bash")
        for line in repro.get("commands", []):
            lines.append(line)
        lines.append("```")
        lines.append("")

    # --- delta vs prior tag (when --compare-to given) ---
    delta = report.get("delta", {})
    if delta:
        lines.append(f"## Delta vs `{delta.get('compare_to', '?')}`")
        lines.append("")
        for section, rows in delta.get("sections", {}).items():
            if rows:
                lines.append(f"### {section}")
                lines.append("")
                lines.append("| Metric | this | prior | Δ |")
                lines.append("|---|---:|---:|---:|")
                for r in rows:
                    lines.append(f"| {r['metric']} | {r['this']} | {r['prior']} | {r['delta']} |")
                lines.append("")

    return "\n".join(lines)


def render_discuss(report: dict) -> str:
    """Trimmed variant for posting under a GitHub Discussion comment.
    Drops failure examples + bench raw GPU table; keeps the headline tables."""
    lines = []
    lines.append(f"**Rebench — {report.get('tag', '?')}**")
    lines.append("")
    if report.get("tldr"):
        for b in report["tldr"]:
            lines.append(f"- {b}")
        lines.append("")
    meta = report.get("meta", {})
    c = meta.get("container", {})
    if c:
        # Quant string: prefer model_config bits, fall back to quantization scheme name.
        mc = meta.get("model_config", {}) or {}
        quant_str = f"{mc.get('quant_method', c.get('quantization', '?'))} {mc.get('bits', '?')}-bit" \
            if mc.get("bits") else c.get("quantization", "?")
        lines.append(f"Config: `{c.get('image')}` · {quant_str} · "
                     f"`{c.get('kv_cache_dtype')}` KV · `{c.get('max_model_len')}` ctx · "
                     f"`{c.get('speculative_config')}` · TP={c.get('tensor_parallel_size')}")
        lines.append("")
    # bench
    bench = report.get("bench", {})
    if bench.get("narrative"):
        b_n, b_c = bench["narrative"], bench.get("code", {})
        lines.append("**TPS:**")
        pp_n = f", PP {b_n['pp_tps_mean']:.0f} tok/s" if b_n.get("pp_tps_mean") is not None else ""
        lines.append(f"- Narrative: {b_n['decode_tps_mean']:.1f} TPS decode, {b_n['ttft_ms_mean']:.0f} ms TTFT{pp_n} (CV {b_n['decode_tps_cv']:.1f}%)")
        if b_c:
            pp_c = f", PP {b_c['pp_tps_mean']:.0f} tok/s" if b_c.get("pp_tps_mean") is not None else ""
            lines.append(f"- Code: {b_c['decode_tps_mean']:.1f} TPS decode, {b_c['ttft_ms_mean']:.0f} ms TTFT{pp_c} (CV {b_c['decode_tps_cv']:.1f}%)")
        if bench.get("prompt_processing", {}).get("pp_tps_mean") is not None:
            pp = bench["prompt_processing"]["pp_tps_mean"]
            lines.append(f"- Prompt-processing fallback: {pp:.0f} tok/s")
        lines.append("")
    boot = report.get("vllm_boot", {})
    if boot.get("kv_cache_tokens"):
        lines.append(f"**Concurrency:** GPU KV pool **{boot['kv_cache_tokens']:,} tokens**, "
                     f"**{boot.get('max_concurrency', 0):.2f}×** concurrency @ {boot.get('max_concurrency_request_size', 0):,} tokens/req.")
        lines.append("")
    stress = report.get("verify_stress", {})
    if stress.get("checks"):
        passed = sum(1 for c in stress["checks"] if c["verdict"] == "PASS")
        total = len(stress["checks"])
        lines.append(f"**Stress:** {passed}/{total} checks PASS.")
        lines.append("")
    q = report.get("quality", {})
    if q.get("packs"):
        rows = " · ".join(f"{p['pack_id']} {p['passed']}/{p['total']} ({p['score_pct']}%)" for p in q["packs"])
        lines.append(f"**Quality:** {q['total_passed']}/{q['total_total']} ({q['total_pct']}%) — {rows}")
        lines.append("")
    soak = report.get("soak", {})
    if soak:
        lines.append(f"**Soak:** verdict {soak.get('verdict', '?')}, "
                     f"silent_empty {soak.get('silent_empty', '?')}, "
                     f"p50 decode {soak.get('p50_decode_tps', '?')} TPS.")
        lines.append("")
    aider = report.get("aider", {})
    if aider.get("per_language"):
        per_lang_str = " · ".join(f"{l} {p}/{t}" for l, (p, t) in sorted(aider["per_language"].items()))
        lines.append(f"**Aider-polyglot:** {aider.get('passed_count', '?')}/{aider.get('total_count', '?')} — {per_lang_str}")
        lines.append("")
    return "\n".join(lines)


# --- helpers for the new sections -------------------------------------------

def compute_tldr(report: dict) -> list[str]:
    """Auto-generate 3-5 punchy bullets from the parsed data."""
    bullets = []
    bench = report.get("bench", {})
    if bench.get("narrative") and bench.get("code"):
        pp_vals = [
            b.get("pp_tps_mean")
            for b in (bench.get("narrative", {}), bench.get("code", {}))
            if b.get("pp_tps_mean") is not None
        ]
        pp_text = f", PP {sum(pp_vals) / len(pp_vals):.0f} tok/s" if pp_vals else ""
        bullets.append(
            f"TPS narrative **{bench['narrative']['decode_tps_mean']:.1f}** / "
            f"code **{bench['code']['decode_tps_mean']:.1f}** "
            f"(TTFT {bench['narrative']['ttft_ms_mean']:.0f}/{bench['code']['ttft_ms_mean']:.0f} ms{pp_text})."
        )
    elif bench.get("prompt_processing"):
        pp = bench["prompt_processing"].get("pp_tps_mean")
        if pp is not None:
            bullets.append(f"Prompt processing fallback: **{pp:.0f} tok/s**.")
    boot = report.get("vllm_boot", {})
    if boot.get("kv_cache_tokens") and boot.get("max_concurrency"):
        bullets.append(
            f"KV pool **{boot['kv_cache_tokens']:,} tokens** — "
            f"**{boot['max_concurrency']:.2f}×** concurrency @ {boot.get('max_concurrency_request_size', 0):,} tokens/req."
        )
    stress = report.get("verify_stress", {})
    if stress.get("checks"):
        passed = sum(1 for c in stress["checks"] if c["verdict"] == "PASS")
        total = len(stress["checks"])
        bullets.append(f"Verify-stress: **{passed}/{total}** boundary checks PASS.")
    q = report.get("quality", {})
    if q.get("total_total"):
        bullets.append(
            f"Quality (8 packs, {q['total_total']} scenarios): **{q['total_passed']}/{q['total_total']} ({q['total_pct']}%)**."
        )
    soak = report.get("soak", {})
    if soak.get("verdict"):
        bullets.append(
            f"Soak: **{soak['verdict']}** — silent_empty {soak.get('silent_empty', '?')}, "
            f"p50 decode {soak.get('p50_decode_tps', '?')} TPS."
        )
    a = report.get("aider", {})
    if a.get("total_count"):
        bullets.append(
            f"Aider-polyglot: **{a.get('passed_count', '?')}/{a['total_count']}** "
            f"({100 * (a.get('pass_rate') or 0):.1f}%) — per-lang in section below."
        )
    return bullets


def parse_rig(rig_txt: str) -> dict:
    """Parse rig.txt captured by rebench-full.sh.
    Format: free-form key: value lines from `nvidia-smi -L`, hostname, etc."""
    rig = {"gpus": []}
    for line in rig_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("hostname:"):
            rig["hostname"] = line.split(":", 1)[1].strip()
        elif line.startswith("GPU "):
            # `nvidia-smi -L` format: "GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-xxx)"
            rig["gpus"].append(line.split(":", 1)[1].split("(UUID")[0].strip())
        elif line.startswith("power_cap_w:"):
            rig["power_cap_w"] = line.split(":", 1)[1].strip()
    return rig


def compute_reproducer(meta: dict) -> dict:
    """Render bash commands to re-run this exact config."""
    c = meta.get("container", {})
    cmds = []
    if c.get("image"):
        cmds.append("# Same vLLM nightly + KV class + ctx + MTP n as this run:")
        cmds.append(f"# image: {c['image']}")
    cmds.append("")
    cmds.append("# Bring the model up via gpu-mode (or docker compose -f <compose>.yml up -d):")
    cmds.append(f"# served_model_name = {c.get('served_model_name', '?')}")
    cmds.append("")
    cmds.append("bash scripts/rebench-full.sh")
    cmds.append("")
    cmds.append("# Or run individual phases:")
    cmds.append("bash scripts/bench.sh                      # TPS")
    cmds.append("bash scripts/verify-stress.sh              # boundary")
    cmds.append("bash scripts/quality-test.sh --full        # 8-pack quality")
    cmds.append("bash scripts/soak-test.sh                  # stability")
    cmds.append("bash scripts/quality-test.sh --pack aider-polyglot-30")
    return {"commands": cmds}


def compute_delta(this_report: dict, other_dir: Path) -> dict:
    """Diff this report against another tag dir's REPORT.md numerical fields."""
    other_blob = read_json(other_dir / "_internal.json")
    if not other_blob:
        return {}
    sections = {}

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def diff_row(label, a, b, unit=""):
        an, bn = num(a), num(b)
        if an is None or bn is None:
            return None
        d = an - bn
        sign = "+" if d > 0 else ""
        return {"metric": label, "this": f"{an:.2f}{unit}", "prior": f"{bn:.2f}{unit}",
                "delta": f"{sign}{d:.2f}{unit}"}

    perf_rows = []
    for kind in ("narrative", "code"):
        t = (this_report.get("bench", {}) or {}).get(kind, {})
        o = (other_blob.get("bench", {}) or {}).get(kind, {})
        if t and o:
            r = diff_row(f"{kind} decode TPS", t.get("decode_tps_mean"), o.get("decode_tps_mean"))
            if r: perf_rows.append(r)
            r = diff_row(f"{kind} TTFT", t.get("ttft_ms_mean"), o.get("ttft_ms_mean"), " ms")
            if r: perf_rows.append(r)
    sections["Performance"] = perf_rows

    conc_rows = []
    t = this_report.get("vllm_boot", {})
    o = other_blob.get("vllm_boot", {})
    r = diff_row("GPU KV cache tokens", t.get("kv_cache_tokens"), o.get("kv_cache_tokens"))
    if r: conc_rows.append(r)
    r = diff_row("Max concurrency", t.get("max_concurrency"), o.get("max_concurrency"), "×")
    if r: conc_rows.append(r)
    sections["Concurrency"] = conc_rows

    qual_rows = []
    t = this_report.get("quality", {})
    o = other_blob.get("quality", {})
    r = diff_row("Total passes", t.get("total_passed"), o.get("total_passed"))
    if r: qual_rows.append(r)
    r = diff_row("Total %", t.get("total_pct"), o.get("total_pct"), "%")
    if r: qual_rows.append(r)
    sections["Quality"] = qual_rows

    aid_rows = []
    t = this_report.get("aider", {})
    o = other_blob.get("aider", {})
    r = diff_row("aider-polyglot passes", t.get("passed_count"), o.get("passed_count"))
    if r: aid_rows.append(r)
    sections["Aider Polyglot"] = aid_rows

    return {"compare_to": other_dir.name, "sections": sections}


# --- main -------------------------------------------------------------------

def extract_date(tag_dir: Path) -> str:
    """Tag-dir date: look for YYYY-MM-DD or YYYYMMDD anywhere in the name,
    otherwise fall back to the dir mtime."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{8})", tag_dir.name)
    if m:
        s = m.group(1)
        if len(s) == 8:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
    import datetime
    return datetime.datetime.utcfromtimestamp(tag_dir.stat().st_mtime).strftime("%Y-%m-%d")


def extract_quality_oneliner(qfull_log: str) -> str | None:
    """Grep the paste-ready Quality: line from quality-test.sh stdout."""
    for line in qfull_log.splitlines():
        if line.lstrip().startswith("Quality:"):
            return line.strip()
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("tag_dir", help="results/rebench/<tag>/ directory")
    p.add_argument("--compare-to", default=None,
                   help="another tag dir to compute deltas against")
    p.add_argument("--no-discuss", action="store_true",
                   help="skip writing REPORT-discuss.md")
    args = p.parse_args(argv)

    tag_dir = Path(args.tag_dir).resolve()
    if not tag_dir.is_dir():
        print(f"✗ not a directory: {tag_dir}", file=sys.stderr)
        return 2

    bench_log = read_file(tag_dir / "bench.log")
    stress_log = read_file(tag_dir / "verify-stress.log")
    quality_blob = read_json(tag_dir / "quality-full.json")
    quality_log = read_file(tag_dir / "quality-full.log")
    soak_log = read_file(tag_dir / "soak.log")
    aider_blob = read_json(tag_dir / "aider-polyglot.json")
    container_blob = read_json(tag_dir / "container-config.json")
    boot_log = read_file(tag_dir / "vllm-boot.log")
    timings_blob = read_json(tag_dir / "timings.json") or {}
    rig_txt = read_file(tag_dir / "rig.txt")

    # find model dir for quant config
    model_config = {}
    if container_blob:
        cinfo = parse_container_config(container_blob)
        model_path = cinfo.get("model", "")
        if model_path and model_path.startswith("/root/.cache/huggingface/"):
            host_path = Path("/mnt/models/huggingface") / Path(model_path).name
            model_config = parse_model_config_json(host_path)

    # commit SHA from the club-3090 repo (walk up to find .git)
    sha_dir = tag_dir
    while sha_dir != sha_dir.parent and not (sha_dir / ".git").exists():
        sha_dir = sha_dir.parent
    commit_sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(sha_dir if (sha_dir / ".git").exists() else tag_dir),
        capture_output=True, text=True,
    ).stdout.strip() or "?"

    quality = parse_quality(quality_blob)
    one_liner = extract_quality_oneliner(quality_log)
    if one_liner:
        quality["one_liner"] = one_liner

    report = {
        "tag": tag_dir.name,
        "tag_dir": str(tag_dir),
        "date": extract_date(tag_dir),
        "commit_sha": commit_sha,
        "meta": {
            "container": parse_container_config(container_blob),
            "model_config": model_config,
            "rig": parse_rig(rig_txt),
        },
        "vllm_boot": parse_vllm_boot(boot_log),
        "bench": parse_bench(bench_log),
        "verify_stress": parse_verify_stress(stress_log),
        "quality": quality,
        "soak": parse_soak(soak_log),
        "aider": parse_aider(aider_blob),
        "timings": timings_blob,
    }

    # auto-compute TL;DR
    report["tldr"] = compute_tldr(report)
    report["reproducer"] = compute_reproducer(report.get("meta", {}))

    # optional delta vs prior tag
    if args.compare_to:
        other = Path(args.compare_to).resolve()
        if other.is_dir():
            report["delta"] = compute_delta(report, other)

    # write a sidecar with the parsed numbers so future --compare-to works
    sidecar = {
        "bench": report["bench"],
        "vllm_boot": report["vllm_boot"],
        "quality": {
            "total_passed": report["quality"].get("total_passed"),
            "total_total": report["quality"].get("total_total"),
            "total_pct": report["quality"].get("total_pct"),
            "packs": report["quality"].get("packs"),
        },
        "aider": {
            "passed_count": report["aider"].get("passed_count"),
            "total_count": report["aider"].get("total_count"),
            "per_language": report["aider"].get("per_language"),
        },
        "soak": report["soak"],
    }
    (tag_dir / "_internal.json").write_text(json.dumps(sidecar, indent=2))

    out_path = tag_dir / "REPORT.md"
    out_path.write_text(render(report))
    print(f"wrote {out_path}")

    if not args.no_discuss:
        discuss_path = tag_dir / "REPORT-discuss.md"
        discuss_path.write_text(render_discuss(report))
        print(f"wrote {discuss_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
