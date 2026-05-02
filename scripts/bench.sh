#!/usr/bin/env bash
#
# Canonical bench against running vLLM service(s).
#   - Runs both the canonical narrative AND code prompts in one invocation.
#     This matches the README's narrative/code TPS pairing.
#   - 3 warmup + N measured batches per prompt (default 5 narrative + 5 code).
#   - Each batch sends CONCURRENCY requests to each target endpoint at once.
#   - Reports aggregate wall TPS per target and combined across targets, plus
#     per-request wall TPS, decode TPS, and TTFT.
#   - Captures sampled per-card peak VRAM during the run.
#   - Writes a timestamped Markdown report for the run.
#
# Why two TPS metrics:
#   - wall_TPS  = "user-perceived speed" (includes prefill cost)
#   - decode_TPS = "model decode rate" (excludes prefill)
#   For long prompts the two can differ a lot. For short prompts they
#   converge. Reporting both keeps comparisons honest across configs.
#
# Why narrative + code:
#   MTP acceptance varies wildly by prompt structure. Code (repetitive,
#   token-predictable) accepts at ~80% per position; prose (semantically
#   rich) at ~50%. Reporting only one half is misleading. README claims
#   like "66 / 85 TPS" pair them; bench should too.
#
# Prereq: stack is running and reports "Application startup complete".
#
# Env vars:
#   URL                Endpoint. Default: http://localhost:8020
#   URLS               Comma-separated endpoints. Overrides URL.
#   MODEL              Served model name. Default: qwen3.6-27b-autoround
#   API_KEY            Optional OpenAI-compatible Bearer token.
#   CONTAINER          Container for log scraping. Default: vllm-qwen36-27b
#   CONTAINERS         Comma-separated containers for log scraping.
#   TARGET_NAMES       Comma-separated labels for URLS.
#   CONCURRENCY        Concurrent requests per target per batch. Default: 1,
#                      or 8 with --quad-pairs.
#   RUNS               Measured batches per prompt. Default: 5
#   WARMUPS            Warm-up batches per prompt. Default: 3
#   PROMPT_NARR        Override narrative prompt
#   PROMPT_CODE        Override code prompt
#   MAX_TOKENS_NARR    Default: 1000
#   MAX_TOKENS_CODE    Default: 800
#   ONLY               Set to "narr" or "code" to skip the other. Default: both
#   QUIET              Set to 1 to skip per-run lines (just print summary)
#   REQUEST_TIMEOUT    Per-request timeout seconds. Default: 900
#   BENCH_OUT_DIR      Markdown report directory. Default: bench-results
#   BENCH_OUT          Exact Markdown report path. Overrides BENCH_OUT_DIR.
#   CLUB3090_NVIDIA_SMI_SUDO
#                      Set to 1 if GPU telemetry requires sudo -n nvidia-smi
#
# Usage:
#   bash scripts/bench.sh
#   ONLY=code bash scripts/bench.sh
#   RUNS=10 bash scripts/bench.sh
#   bash scripts/bench.sh --quad-pairs
#   URLS=http://localhost:8021,http://localhost:8022 CONCURRENCY=8 bash scripts/bench.sh

set -euo pipefail

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } NF == 0 { print; next } { exit }' "$0"
}

QUAD_PAIRS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quad-pairs)
      QUAD_PAIRS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Try: bash scripts/bench.sh --help" >&2
      exit 2
      ;;
  esac
done

if [[ "$QUAD_PAIRS" == "1" ]]; then
  URLS="${URLS:-http://localhost:${PORT:-8021},http://localhost:${PORT_B:-8022}}"
  TARGET_NAMES="${TARGET_NAMES:-pair-a,pair-b}"
  CONTAINERS="${CONTAINERS:-vllm-qwen36-27b-quad-pair-a,vllm-qwen36-27b-quad-pair-b}"
  CONCURRENCY="${CONCURRENCY:-8}"
  API_KEY="${API_KEY:-${OPENAI_API_KEY:-${VLLM_API_KEY:-sk-vllm}}}"
else
  if [[ -n "${URLS:-}" ]]; then
    URLS="${URLS}"
    if [[ -z "${CONTAINERS:-}" && "$URLS" != *,* ]]; then
      CONTAINERS="${CONTAINER:-vllm-qwen36-27b}"
    else
      CONTAINERS="${CONTAINERS:-}"
    fi
  else
    URLS="${URL:-http://localhost:8020}"
    CONTAINERS="${CONTAINERS:-${CONTAINER:-vllm-qwen36-27b}}"
  fi
  TARGET_NAMES="${TARGET_NAMES:-}"
  CONCURRENCY="${CONCURRENCY:-1}"
  API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
fi

MODEL="${MODEL:-qwen3.6-27b-autoround}"
RUNS="${RUNS:-5}"
WARMUPS="${WARMUPS:-3}"
MAX_TOKENS_NARR="${MAX_TOKENS_NARR:-1000}"
MAX_TOKENS_CODE="${MAX_TOKENS_CODE:-800}"
PROMPT_NARR="${PROMPT_NARR:-Write a detailed 800-word essay explaining transformer attention.}"
PROMPT_CODE="${PROMPT_CODE:-Write a Python implementation of quicksort with comments explaining each step.}"
ONLY="${ONLY:-both}"
QUIET="${QUIET:-0}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
BENCH_OUT_DIR="${BENCH_OUT_DIR:-bench-results}"
BENCH_OUT="${BENCH_OUT:-}"
export API_KEY

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not in PATH." >&2; exit 1; }
}
need python3

python3 - "$URLS" "$TARGET_NAMES" "$CONTAINERS" "$MODEL" \
            "$CONCURRENCY" "$WARMUPS" "$RUNS" "$QUIET" "$ONLY" \
            "$PROMPT_NARR" "$MAX_TOKENS_NARR" \
            "$PROMPT_CODE" "$MAX_TOKENS_CODE" \
            "$REQUEST_TIMEOUT" "$BENCH_OUT_DIR" "$BENCH_OUT" \
            "${CLUB3090_NVIDIA_SMI_SUDO:-0}" << 'PYEOF'
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import shutil
import statistics as s
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

(URLS, TARGET_NAMES, CONTAINERS, MODEL,
 CONCURRENCY, WARMUPS, RUNS, QUIET, ONLY,
 PROMPT_NARR, MAX_NARR, PROMPT_CODE, MAX_CODE,
 REQUEST_TIMEOUT, BENCH_OUT_DIR, BENCH_OUT, NVIDIA_SMI_SUDO) = sys.argv[1:]

CONCURRENCY = int(CONCURRENCY)
WARMUPS = int(WARMUPS)
RUNS = int(RUNS)
QUIET = int(QUIET) == 1
MAX_NARR = int(MAX_NARR)
MAX_CODE = int(MAX_CODE)
REQUEST_TIMEOUT = int(REQUEST_TIMEOUT)
NVIDIA_SMI_SUDO = NVIDIA_SMI_SUDO == "1"

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
API_KEY = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

def csv_parts(value, keep_empty=False):
    if value is None or value == "":
        return []
    parts = [p.strip() for p in value.split(",")]
    if keep_empty:
        return parts
    return [p for p in parts if p]

urls = [u.rstrip("/") for u in csv_parts(URLS)]
names = csv_parts(TARGET_NAMES)
containers = csv_parts(CONTAINERS, keep_empty=True)

if not urls:
    raise SystemExit("ERROR: no endpoint URLs configured")
if names and len(names) != len(urls):
    raise SystemExit("ERROR: TARGET_NAMES must have the same count as URLS")
if not names:
    names = ["default"] if len(urls) == 1 else [f"endpoint-{i + 1}" for i in range(len(urls))]
if containers and len(containers) != len(urls):
    raise SystemExit("ERROR: CONTAINERS must have the same count as URLS")
if not containers:
    containers = [""] * len(urls)

targets = [
    {"name": name, "url": url, "container": container}
    for name, url, container in zip(names, urls, containers)
]

if CONCURRENCY < 1:
    raise SystemExit("ERROR: CONCURRENCY must be >= 1")
if WARMUPS < 0 or RUNS < 1:
    raise SystemExit("ERROR: WARMUPS must be >= 0 and RUNS must be >= 1")

def check_endpoint(target):
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{target['url']}/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
    except Exception as exc:
        raise SystemExit(
            f"ERROR: service not reachable at {target['url']}/v1/models: {exc}"
        )

for target in targets:
    check_endpoint(target)

class GpuSampler:
    def __init__(self, sudo=False):
        self.sudo = sudo
        self.stop_event = threading.Event()
        self.thread = None
        self.samples = []
        self.peaks = {}
        self.available = shutil.which("nvidia-smi") is not None

    def start(self):
        if not self.available:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.available:
            return
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _cmd(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        if self.sudo:
            return ["sudo", "-n", *cmd]
        return cmd

    def _run(self):
        while not self.stop_event.is_set():
            self._sample()
            self.stop_event.wait(1)

    def _sample(self):
        try:
            proc = subprocess.run(
                self._cmd(), check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except Exception:
            return
        if proc.returncode != 0:
            return
        now = time.time()
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                idx = int(parts[0])
                util = float(parts[1])
                mem = float(parts[2])
                total = float(parts[3])
                power = float(parts[4])
                temp = float(parts[5])
            except ValueError:
                continue
            sample = {
                "time": now, "index": idx, "util": util, "mem": mem,
                "total": total, "power": power, "temp": temp,
            }
            self.samples.append(sample)
            peak = self.peaks.setdefault(idx, {
                "util": 0.0, "mem": 0.0, "total": total,
                "power": 0.0, "temp": 0.0,
            })
            peak["util"] = max(peak["util"], util)
            peak["mem"] = max(peak["mem"], mem)
            peak["total"] = total
            peak["power"] = max(peak["power"], power)
            peak["temp"] = max(peak["temp"], temp)

    def snapshot(self):
        self._sample()
        latest = {}
        for sample in self.samples:
            latest[sample["index"]] = sample
        return latest

def run_once(target, prompt, max_tokens, prompt_label, phase, batch_id, request_id):
    t_send = time.time()
    result = {
        "target": target["name"],
        "url": target["url"],
        "prompt": prompt_label,
        "phase": phase,
        "batch": batch_id,
        "request": request_id,
        "start": t_send,
        "end": t_send,
        "wall": 0.0,
        "ttft": None,
        "completion_tokens": 0,
        "wall_tps": 0.0,
        "decode_tps": 0.0,
        "error": "",
    }
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{target['url']}/v1/chat/completions", data=body,
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            for line in response:
                line = line.decode("utf-8", errors="ignore").rstrip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content and result["ttft"] is None:
                        result["ttft"] = time.time() - t_send
                usage = chunk.get("usage")
                if usage:
                    result["completion_tokens"] = usage.get(
                        "completion_tokens", result["completion_tokens"]
                    )
    except Exception as exc:
        result["error"] = str(exc)
    t_end = time.time()
    result["end"] = t_end
    result["wall"] = max(t_end - t_send, 1e-6)
    if result["ttft"] is None:
        result["ttft"] = result["wall"]
    decode_t = max(result["wall"] - result["ttft"], 1e-6)
    result["wall_tps"] = result["completion_tokens"] / result["wall"]
    result["decode_tps"] = result["completion_tokens"] / decode_t
    return result

def mean(values):
    return s.mean(values) if values else 0.0

def stdev(values):
    return s.stdev(values) if len(values) > 1 else 0.0

def cv(values):
    m = mean(values)
    return stdev(values) / m * 100 if m > 0 else 0.0

def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"

def summarize_scope(prompt_label, batch_id, scope, results):
    ok = [r for r in results if not r["error"]]
    failed = len(results) - len(ok)
    tokens = sum(r["completion_tokens"] for r in ok)
    if ok:
        window = max(r["end"] for r in ok) - min(r["start"] for r in ok)
        window = max(window, 1e-6)
    else:
        window = 0.0
    return {
        "prompt": prompt_label,
        "batch": batch_id,
        "scope": scope,
        "completed": len(ok),
        "failed": failed,
        "tokens": tokens,
        "window": window,
        "aggregate_wall_tps": tokens / window if window > 0 else 0.0,
        "mean_req_wall_tps": mean([r["wall_tps"] for r in ok]),
        "mean_req_decode_tps": mean([r["decode_tps"] for r in ok]),
        "mean_ttft_ms": mean([r["ttft"] * 1000 for r in ok]),
    }

def run_batch(prompt_label, prompt, max_tokens, phase, batch_id):
    with cf.ThreadPoolExecutor(max_workers=len(targets) * CONCURRENCY) as pool:
        futures = []
        for target in targets:
            for request_id in range(1, CONCURRENCY + 1):
                futures.append(pool.submit(
                    run_once, target, prompt, max_tokens,
                    prompt_label, phase, batch_id, request_id
                ))
        return [future.result() for future in cf.as_completed(futures)]

def batch_summaries(prompt_label, batch_id, results):
    summaries = [summarize_scope(prompt_label, batch_id, "combined", results)]
    for target in targets:
        scoped = [r for r in results if r["target"] == target["name"]]
        summaries.append(summarize_scope(prompt_label, batch_id, target["name"], scoped))
    return summaries

def print_batch_summary(phase, summary_rows):
    combined = next(row for row in summary_rows if row["scope"] == "combined")
    pieces = [
        f"{row['scope']}={row['aggregate_wall_tps']:.2f} TPS"
        for row in summary_rows if row["scope"] != "combined"
    ]
    fail_count = sum(row["failed"] for row in summary_rows if row["scope"] != "combined")
    print(
        f"  {phase:<10s} combined={combined['aggregate_wall_tps']:.2f} TPS "
        f"tokens={combined['tokens']} failed={fail_count} | " + " | ".join(pieces),
        flush=True,
    )

def summary_rows_for(prompt_label, measured_summaries):
    rows = []
    scopes = ["combined", *[target["name"] for target in targets]]
    for scope in scopes:
        matching = [
            row for row in measured_summaries
            if row["prompt"] == prompt_label and row["scope"] == scope
        ]
        if not matching:
            continue
        agg = [row["aggregate_wall_tps"] for row in matching]
        req_wall = [row["mean_req_wall_tps"] for row in matching]
        req_decode = [row["mean_req_decode_tps"] for row in matching]
        ttft = [row["mean_ttft_ms"] for row in matching]
        rows.append({
            "prompt": prompt_label,
            "scope": scope,
            "n": len(matching),
            "aggregate_wall_tps_mean": mean(agg),
            "aggregate_wall_tps_std": stdev(agg),
            "aggregate_wall_tps_cv": cv(agg),
            "aggregate_wall_tps_min": min(agg),
            "aggregate_wall_tps_max": max(agg),
            "mean_req_wall_tps": mean(req_wall),
            "mean_req_decode_tps": mean(req_decode),
            "mean_ttft_ms": mean(ttft),
            "failed": sum(row["failed"] for row in matching),
        })
    return rows

def docker_spec_metrics(container):
    if not container or shutil.which("docker") is None:
        return []
    inspect = subprocess.run(
        ["docker", "inspect", container],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if inspect.returncode != 0:
        return []
    proc = subprocess.run(
        ["docker", "logs", container],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    lines = [line for line in proc.stdout.splitlines() if "SpecDecoding metrics" in line]
    return lines[-3:]

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return out

def render_report(started, finished, report_path, prompts, measured_summaries,
                  final_summaries, sampler, final_gpu, spec_metrics, failures):
    lines = []
    lines.append(f"# club-3090 benchmark run - {started}")
    lines.append("")
    lines.append(f"- Finished: {finished}")
    lines.append(f"- Model: `{MODEL}`")
    lines.append(f"- Concurrency: `{CONCURRENCY}` request(s) per target per batch")
    lines.append(f"- Warmup batches per prompt: `{WARMUPS}`")
    lines.append(f"- Measured batches per prompt: `{RUNS}`")
    lines.append(f"- Temperature/top_p/top_k: `{TEMPERATURE}` / `{TOP_P}` / `{TOP_K}`")
    lines.append(f"- Report path: `{report_path}`")
    lines.append("")
    lines.append("## Targets")
    lines.extend(md_table(
        ["Name", "URL", "Container"],
        [[t["name"], f"`{t['url']}`", f"`{t['container']}`" if t["container"] else "-"]
         for t in targets],
    ))
    lines.append("")
    lines.append("## Prompts")
    lines.extend(md_table(
        ["Prompt", "Chars", "Max tokens"],
        [[label, len(prompt), max_tokens] for label, prompt, max_tokens in prompts],
    ))
    lines.append("")
    lines.append("## Summary")
    rows = []
    for row in final_summaries:
        rows.append([
            row["prompt"], row["scope"], row["n"],
            fmt(row["aggregate_wall_tps_mean"]),
            fmt(row["aggregate_wall_tps_std"]),
            fmt(row["aggregate_wall_tps_cv"]),
            fmt(row["aggregate_wall_tps_min"]),
            fmt(row["aggregate_wall_tps_max"]),
            fmt(row["mean_req_wall_tps"]),
            fmt(row["mean_req_decode_tps"]),
            fmt(row["mean_ttft_ms"], 0),
            row["failed"],
        ])
    lines.extend(md_table(
        ["Prompt", "Scope", "n", "Agg wall TPS mean", "std", "CV %",
         "min", "max", "Mean req wall TPS", "Mean req decode TPS",
         "Mean TTFT ms", "Failures"],
        rows,
    ))
    lines.append("")
    lines.append("## Measured Batches")
    batch_rows = []
    for row in measured_summaries:
        batch_rows.append([
            row["prompt"], row["batch"], row["scope"], row["completed"],
            row["failed"], row["tokens"], fmt(row["window"]),
            fmt(row["aggregate_wall_tps"]), fmt(row["mean_req_wall_tps"]),
            fmt(row["mean_req_decode_tps"]), fmt(row["mean_ttft_ms"], 0),
        ])
    lines.extend(md_table(
        ["Prompt", "Batch", "Scope", "Completed", "Failed", "Tokens",
         "Window s", "Agg wall TPS", "Mean req wall TPS",
         "Mean req decode TPS", "Mean TTFT ms"],
        batch_rows,
    ))
    lines.append("")
    lines.append("## Peak GPU State")
    if sampler.available and sampler.peaks:
        rows = []
        for idx in sorted(sampler.peaks):
            peak = sampler.peaks[idx]
            rows.append([
                idx, fmt(peak["util"], 0),
                f"{fmt(peak['mem'], 0)} / {fmt(peak['total'], 0)}",
                fmt(peak["power"]), fmt(peak["temp"], 0),
            ])
        lines.extend(md_table(
            ["GPU", "Peak util %", "Peak VRAM MiB", "Peak power W", "Peak temp C"],
            rows,
        ))
    else:
        lines.append("No GPU samples captured.")
    lines.append("")
    lines.append("## Final GPU Snapshot")
    if final_gpu:
        rows = []
        for idx in sorted(final_gpu):
            sample = final_gpu[idx]
            rows.append([
                idx, fmt(sample["util"], 0),
                f"{fmt(sample['mem'], 0)} / {fmt(sample['total'], 0)}",
                fmt(sample["power"]), fmt(sample["temp"], 0),
            ])
        lines.extend(md_table(
            ["GPU", "Util %", "VRAM MiB", "Power W", "Temp C"],
            rows,
        ))
    else:
        lines.append("No final GPU snapshot captured.")
    lines.append("")
    lines.append("## SpecDecoding Metrics")
    wrote_metrics = False
    for target in targets:
        metrics = spec_metrics.get(target["name"], [])
        if not metrics:
            continue
        wrote_metrics = True
        lines.append(f"### {target['name']}")
        lines.append("```text")
        lines.extend(metrics)
        lines.append("```")
    if not wrote_metrics:
        lines.append("No SpecDecoding metrics found.")
    if failures:
        lines.append("")
        lines.append("## Failures")
        for failure in failures:
            lines.append(
                f"- {failure['prompt']} {failure['phase']}-{failure['batch']} "
                f"{failure['target']} request {failure['request']}: {failure['error']}"
            )
    lines.append("")
    return "\n".join(lines)

started_dt = dt.datetime.now().astimezone()
started = started_dt.strftime("%Y-%m-%d %H:%M:%S %z")
timestamp = started_dt.strftime("%Y%m%d-%H%M%S")
if BENCH_OUT:
    report_path = Path(BENCH_OUT)
else:
    report_path = Path(BENCH_OUT_DIR) / f"bench-{timestamp}.md"

print("=== club-3090 benchmark ===")
print(f"targets: {', '.join(t['name'] + '=' + t['url'] for t in targets)}")
print(f"concurrency: {CONCURRENCY} request(s) per target")
print(f"warmups: {WARMUPS} batch(es) per prompt; measured: {RUNS} batch(es) per prompt")
print(f"report: {report_path}")

sampler = GpuSampler(sudo=NVIDIA_SMI_SUDO)
all_results = []
measured_summaries = []
prompts = []
if ONLY in ("both", "narr"):
    prompts.append(("narrative", PROMPT_NARR, MAX_NARR))
if ONLY in ("both", "code"):
    prompts.append(("code", PROMPT_CODE, MAX_CODE))
if not prompts:
    raise SystemExit("ERROR: ONLY must be one of: both, narr, code")

sampler.start()
try:
    for label, prompt, max_tokens in prompts:
        print(f"\n========== {label.upper()} (prompt={len(prompt)} chars, max_tokens={max_tokens}) ==========")
        if WARMUPS:
            print(f"=== warmup batches ({WARMUPS}) ===")
        for batch_id in range(1, WARMUPS + 1):
            results = run_batch(label, prompt, max_tokens, "warm", batch_id)
            all_results.extend(results)
            summaries = batch_summaries(label, batch_id, results)
            if not QUIET:
                print_batch_summary(f"warm-{batch_id}", summaries)
        print(f"\n=== measured batches ({RUNS}) ===")
        for batch_id in range(1, RUNS + 1):
            results = run_batch(label, prompt, max_tokens, "run", batch_id)
            all_results.extend(results)
            summaries = batch_summaries(label, batch_id, results)
            measured_summaries.extend(summaries)
            if not QUIET:
                print_batch_summary(f"run-{batch_id}", summaries)
finally:
    sampler.stop()

final_gpu = sampler.snapshot()
final_summaries = []
for label, _, _ in prompts:
    final_summaries.extend(summary_rows_for(label, measured_summaries))

print("\n=== summary ===")
for row in final_summaries:
    print(
        f"  {row['prompt']:<9s} {row['scope']:<12s} "
        f"agg_wall_TPS={row['aggregate_wall_tps_mean']:7.2f} "
        f"std={row['aggregate_wall_tps_std']:6.2f} "
        f"CV={row['aggregate_wall_tps_cv']:4.1f}% "
        f"req_wall_TPS={row['mean_req_wall_tps']:7.2f} "
        f"req_decode_TPS={row['mean_req_decode_tps']:7.2f} "
        f"TTFT={row['mean_ttft_ms']:5.0f}ms "
        f"failures={row['failed']}"
    )

if sampler.available and sampler.peaks:
    print("\n=== peak GPU state ===")
    for idx in sorted(sampler.peaks):
        peak = sampler.peaks[idx]
        print(
            f"  GPU {idx}: util={peak['util']:.0f}% "
            f"mem={peak['mem']:.0f}/{peak['total']:.0f} MiB "
            f"power={peak['power']:.2f} W temp={peak['temp']:.0f} C"
        )

spec_metrics = {
    target["name"]: docker_spec_metrics(target["container"])
    for target in targets
}
if any(spec_metrics.values()):
    print("\n=== Last 3 SpecDecoding metrics ===")
    for target in targets:
        metrics = spec_metrics.get(target["name"], [])
        if not metrics:
            continue
        print(f"[{target['name']}]")
        for line in metrics:
            print(line)

failures = [r for r in all_results if r["error"]]
finished = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
report = render_report(
    started, finished, report_path, prompts, measured_summaries,
    final_summaries, sampler, final_gpu, spec_metrics, failures,
)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")
print(f"\nWrote benchmark report: {report_path}")

if failures:
    raise SystemExit(1)
PYEOF
