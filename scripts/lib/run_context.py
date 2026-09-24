#!/usr/bin/env python3
"""run_context.py — the sampling in effect and the rig, for a quality report.

WHY THIS EXISTS (#1396)
    Quality reports from different rigs could not be compared. vLLM and SGLang
    expose no sampling-defaults endpoint, so a `--sampling-from-server` run read
    "value not exposed by endpoint", and nothing recorded the topology (TP size,
    GPU count and model, NVLink, power cap). Contributors wrote both by hand, and
    in #1396 the sampling turned out to be a main suspect in a cross-rig xhigh
    comparison.

WHAT IT REPORTS — what the engine APPLIES to a chat request, not what it was told:
    * vLLM — the `Default vLLM sampling parameters have been overridden by …`
      line. It is logged AFTER vLLM filters `--override-generation-config` down to
      its allowlist (repetition_penalty, temperature, top_k, top_p, min_p), so a
      compose's presence_penalty is — correctly — absent from it. Reading
      `override_generation_config` from the `non-default args:` dump instead would
      report a presence_penalty the engine never applies.
    * SGLang — the `Using default chat sampling params from model generation
      config:` line, filled out with the chat adapter's OpenAI fallbacks.
      `--preferred-sampling-params` is NOT reported: it is inert on
      /v1/chat/completions (the chat adapter resolves every key eagerly, so the
      request always wins the merge — sglang#39096; the SGLang composes' own
      sampler comment walks through it). A note says so when it would differ.
    * presence_penalty / frequency_penalty — 0.0 on both engines: both chat
      adapters declare them with a hard 0.0 default, so the server cannot change
      them. A client that wants them must send them.
    * llama.cpp — no sampling: benchlocal-cli reads GET /props itself.
    * The rig — GPUs visible to the container (count × model, power cap, PCIe
      gen/width, NVLink) plus TP/PP, quant, KV, spec method, max ctx from the
      engine's own startup dump.

    The ENGINE FAMILY is the caller's input (`--engine`), resolved by
    scripts/lib/engine-kind.sh — this file never classifies (#1282).

OUTPUT
    --json       {"server_defaults": {..}|null, "run_meta": {..}, "notes": [..]}
    --emit-args  one benchlocal-cli argument per line, for `mapfile -t`:
                 `--server-defaults` + JSON (only when resolved) and
                 `--run-meta` + KEY=VALUE pairs.
    Every input can be given as a file (--boot-log / --inspect-json / --smi-csv /
    --nvlink-txt), so the guard test needs neither docker nor a GPU.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys

# The keys each engine lets the SERVER set (vLLM get_diff_sampling_param /
# SGLang get_default_sampling_params — the same allowlist on both).
SERVER_SETTABLE = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
# Declared with a hard 0.0 default by both chat adapters — never server-settable.
REQUEST_ONLY_ZERO = {"presence_penalty": 0.0, "frequency_penalty": 0.0}
# SGLang ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS (v0.5.20 protocol.py) —
# what get_param() falls back to when neither the request nor the model sets a key.
SGLANG_OPENAI_FALLBACK = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                          "repetition_penalty": 1.0}
ENGINES = ("vllm", "sglang", "llamacpp", "exllamav3", "unknown")
SMI_FIELDS = "index,uuid,name,power.limit,pcie.link.gen.max,pcie.link.width.current"


# ---------------------------------------------------------------- parsing (pure)

def dict_after(text: str, marker: str) -> dict | None:
    """The Python-literal dict that starts after `marker` on the same line, or None."""
    i = text.find(marker)
    if i < 0:
        return None
    start = text.find("{", i + len(marker))
    if start < 0 or "\n" in text[i:start]:
        return None
    depth, quote, esc = 0, "", False
    for j in range(start, len(text)):
        ch = text[j]
        if quote:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = ast.literal_eval(text[start:j + 1])
                except (ValueError, SyntaxError):
                    return None
                return value if isinstance(value, dict) else None
        elif ch == "\n":
            return None
    return None


def _numeric(obj: object, keys) -> dict:
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return {}
    if not isinstance(obj, dict):
        return {}
    return {k: obj[k] for k in keys
            if isinstance(obj.get(k), (int, float)) and not isinstance(obj.get(k), bool)}


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def vllm_context(text: str) -> tuple[dict | None, str | None, dict, list[str]]:
    args = dict_after(text, "non-default args:") or {}
    notes: list[str] = []
    override = _numeric(args.get("override_generation_config"),
                        SERVER_SETTABLE + tuple(REQUEST_ONLY_ZERO))
    dropped = {k: v for k, v in override.items() if k in REQUEST_ONLY_ZERO and v != 0}
    if dropped:
        notes.append(f"vLLM ignores {_fmt(dropped)} in --override-generation-config "
                     "(not server-settable) — the served value is 0.0 unless the client sends it")

    effective = None
    m = re.search(r"Default vLLM sampling parameters have been overridden by [^:\n]*: `(\{[^`\n]*\})`", text)
    if m:
        try:
            effective = _numeric(ast.literal_eval(m.group(1)), SERVER_SETTABLE)
        except (ValueError, SyntaxError):
            effective = None
    if effective:
        source = "vLLM effective defaults (generation_config.json"
        source += " + --override-generation-config)" if override else ")"
    elif args.get("generation_config") == "vllm" and override:
        # generation_config=vllm logs nothing; vLLM applies the override, filtered.
        effective = {k: v for k, v in override.items() if k in SERVER_SETTABLE}
        source = "vLLM --override-generation-config (generation_config=vllm)"
    else:
        effective, source = None, None
        if args:
            notes.append("vLLM: no server sampling defaults logged — the engine's neutral defaults apply")
    defaults = {**effective, **REQUEST_ONLY_ZERO} if effective else None

    meta: dict = {}
    if args:
        meta["tp"] = str(args.get("tensor_parallel_size", 1))
        meta["pp"] = str(args.get("pipeline_parallel_size", 1))
        if args.get("quantization"):
            meta["quant"] = str(args["quantization"])
        meta["kv"] = str(args.get("kv_cache_dtype", "auto"))
        if args.get("max_model_len"):
            meta["max_ctx"] = str(args["max_model_len"])
        spec = args.get("speculative_config")
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except ValueError:
                spec = None
        if isinstance(spec, dict) and spec.get("method"):
            n = spec.get("num_speculative_tokens")
            meta["spec"] = f"{spec['method']}" + (f" n={n}" if n else "")
    v = re.search(r"LLM engine \(v([0-9][^)\s]*)\)", text)
    if v:
        meta["engine"] = f"vllm {v.group(1)}"
    return defaults, source, meta, notes


def sglang_context(text: str) -> tuple[dict | None, str | None, dict, list[str]]:
    args = dict_after(text, "server_args=") or {}
    notes: list[str] = []
    if not args:
        return None, None, {}, notes
    model = dict_after(text, "Using default chat sampling params from model generation config:")
    model = _numeric(model, SERVER_SETTABLE) if model else {}
    effective = {**SGLANG_OPENAI_FALLBACK, **model}
    if model:
        source = "SGLang effective defaults (generation_config.json via --sampling-defaults model)"
    else:
        source = "SGLang OpenAI fallbacks (no model generation_config defaults)"
    defaults = {**effective, **REQUEST_ONLY_ZERO}
    preferred = _numeric(args.get("preferred_sampling_params"),
                         SERVER_SETTABLE + tuple(REQUEST_ONLY_ZERO))
    ignored = {k: v for k, v in preferred.items() if defaults.get(k) != v}
    if ignored:
        notes.append(f"SGLang --preferred-sampling-params {_fmt(ignored)} does not apply to "
                     "/v1/chat/completions (sglang#39096) — not reported as in effect")

    meta: dict = {"tp": str(args.get("tp_size", 1)), "pp": str(args.get("pp_size", 1))}
    if args.get("quantization"):
        meta["quant"] = str(args["quantization"])
    meta["kv"] = str(args.get("kv_cache_dtype", "auto"))
    if args.get("context_length"):
        meta["max_ctx"] = str(args["context_length"])
    if args.get("speculative_algorithm"):
        n = args.get("speculative_num_draft_tokens")
        meta["spec"] = f"{args['speculative_algorithm']}" + (f" n={n}" if n else "")
    return defaults, source, meta, notes


def visible_gpu_selectors(inspect: dict) -> list[str] | None:
    """Host GPU indices/UUIDs the container's CUDA sees, in CUDA order; None = all."""
    base: list[str] | None = None
    for req in (inspect.get("HostConfig") or {}).get("DeviceRequests") or []:
        if req.get("DeviceIDs"):
            base = [str(x) for x in req["DeviceIDs"]]
    env = {}
    for entry in (inspect.get("Config") or {}).get("Env") or []:
        key, sep, value = entry.partition("=")
        if sep and value.strip():
            env[key] = value.strip()
    nvd = env.get("NVIDIA_VISIBLE_DEVICES", "")
    if base is None and nvd and nvd not in ("all", "void", "none"):
        base = [v.strip() for v in nvd.split(",") if v.strip()]
    cvd = env.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        picks = [v.strip() for v in cvd.split(",") if v.strip()]
        if base is not None and all(p.isdigit() for p in picks):
            base = [base[int(p)] for p in picks if int(p) < len(base)]
        elif base is None:
            base = picks
    return base


def gpu_meta(smi_csv: str, selectors: list[str] | None, nvlink_txt: str) -> dict:
    rows = []
    for line in smi_csv.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 6:
            rows.append(dict(zip(("index", "uuid", "name", "power", "gen", "width"), cells)))
    if selectors is not None:
        by_key = {**{r["index"]: r for r in rows}, **{r["uuid"]: r for r in rows}}
        rows = [by_key[s] for s in selectors if s in by_key]
    if not rows:
        return {}
    names: dict[str, int] = {}
    for r in rows:
        name = re.sub(r"^NVIDIA (GeForce )?", "", r["name"])
        names[name] = names.get(name, 0) + 1
    meta = {"gpus": ", ".join(f"{n}x {name}" for name, n in names.items())}
    powers = [re.sub(r"\.0+ W$", " W", r["power"]) for r in rows]
    meta["power_cap"] = powers[0] if len(set(powers)) == 1 else " / ".join(powers)
    links = [f"gen{r['gen']} x{r['width']}" for r in rows]
    meta["pcie"] = links[0] if len(set(links)) == 1 else " / ".join(links)
    if len(rows) > 1 and nvlink_txt.strip():
        uuids = {r["uuid"] for r in rows}
        active, current = set(), None
        for line in nvlink_txt.splitlines():
            g = re.search(r"\(UUID: (GPU-[0-9a-f-]+)\)", line)
            if g:
                current = g.group(1)
            elif current and re.match(r"\s*Link \d+: [0-9.]+ GB/s", line):
                active.add(current)
        meta["nvlink"] = "yes" if uuids <= active else "no"
    return meta


def build(engine: str, text: str, inspect: dict, smi_csv: str, nvlink_txt: str) -> dict:
    notes: list[str] = []
    if engine == "vllm":
        defaults, source, meta, notes = vllm_context(text)
    elif engine == "sglang":
        defaults, source, meta, notes = sglang_context(text)
    else:
        defaults, source, meta = None, None, {}
        if engine in ("llamacpp", "exllamav3"):
            notes.append(f"{engine}: sampling is read from the endpoint itself; no engine dump parsed")
    if engine in ("vllm", "sglang") and not meta.get("tp"):
        notes.append(f"{engine}: startup dump not found in the boot log — engine facts not recorded")
    run_meta: dict = {}
    image = str((inspect.get("Config") or {}).get("Image") or "")
    if meta.get("engine"):
        run_meta["engine"] = meta.pop("engine")
    elif engine != "unknown":
        tag = image.rsplit(":", 1)[-1] if ":" in image.rsplit("/", 1)[-1] and "@" not in image else ""
        run_meta["engine"] = f"{engine} {tag}".strip()
    run_meta.update(meta)
    run_meta.update(gpu_meta(smi_csv, visible_gpu_selectors(inspect), nvlink_txt))
    if defaults and source:
        defaults = {**defaults, "source": source}
    return {"server_defaults": defaults, "run_meta": run_meta, "notes": notes}


def emit_args(ctx: dict, with_defaults: bool) -> list[str]:
    out: list[str] = []
    if with_defaults and ctx.get("server_defaults"):
        out += ["--server-defaults", json.dumps(ctx["server_defaults"], separators=(",", ":"))]
    for key, value in (ctx.get("run_meta") or {}).items():
        out += ["--run-meta", f"{key}={value}"]
    return out


# ---------------------------------------------------------------- IO shims

def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _cmd(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _boot_log(container: str) -> str:
    """The container's log from the start (the dumps are printed at boot)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from resolved_config import _boot_log_prefix
        text, _truncated, _err = _boot_log_prefix(container)
        if text:
            return text
    except Exception:  # noqa: BLE001 — fall back to a plain read
        pass
    try:
        p = subprocess.run(["docker", "logs", container], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
        return p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve the sampling in effect and the rig for a quality report (#1396).")
    ap.add_argument("--engine", required=True, choices=ENGINES,
                    help="engine family, from scripts/lib/engine-kind.sh")
    ap.add_argument("--container", help="serving container (live mode)")
    ap.add_argument("--boot-log", help="engine boot log file (instead of docker logs)")
    ap.add_argument("--inspect-json", help="`docker inspect <c>` output file")
    ap.add_argument("--smi-csv", help=f"nvidia-smi --query-gpu={SMI_FIELDS} --format=csv,noheader output")
    ap.add_argument("--nvlink-txt", help="`nvidia-smi nvlink --status` output")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--json", action="store_true", help="print the resolved context as JSON (default)")
    out.add_argument("--emit-args", action="store_true", help="print benchlocal-cli arguments, one per line")
    ap.add_argument("--no-server-defaults", action="store_true",
                    help="with --emit-args: omit --server-defaults (the run is not --sampling-from-server)")
    a = ap.parse_args()
    if not a.container and not a.boot_log:
        ap.error("give --container, or --boot-log (plus the optional file inputs)")

    text = _read(a.boot_log) if a.boot_log else _boot_log(a.container)
    raw_inspect = _read(a.inspect_json) if a.inspect_json else (
        _cmd(["docker", "inspect", a.container]) if a.container else "")
    try:
        inspect = json.loads(raw_inspect)[0] if raw_inspect.strip() else {}
    except (ValueError, IndexError, KeyError):
        inspect = {}
    live = bool(a.container)
    smi = _read(a.smi_csv) if a.smi_csv else (
        _cmd(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader"]) if live else "")
    nvlink = _read(a.nvlink_txt) if a.nvlink_txt else (
        _cmd(["nvidia-smi", "nvlink", "--status"]) if live else "")

    ctx = build(a.engine, text, inspect, smi, nvlink)
    for note in ctx["notes"]:
        print(f"[run-context] note: {note}", file=sys.stderr)
    if a.emit_args:
        print("\n".join(emit_args(ctx, with_defaults=not a.no_server_defaults)))
    else:
        print(json.dumps(ctx, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
