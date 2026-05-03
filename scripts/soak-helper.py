#!/usr/bin/env python3
import csv
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to read."}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files under a directory for a text pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "dir": {"type": "string"}},
                "required": ["pattern", "dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a non-destructive shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
]


def base_req(model, messages, max_tokens, temp=0.4, thinking=False, tools=False):
    req = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if tools:
        req["tools"] = TOOLS
        req["tool_choice"] = "auto"
    return req


def fixture(model, session, turn):
    if turn == 1:
        return base_req(
            model,
            [
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": f"Session {session}: give a short checklist for reviewing a small Python patch."},
            ],
            220,
            temp=0.3,
        )
    if turn == 2:
        return base_req(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "You are working inside a repository. Prefer tools when file "
                        "contents or command output would materially change the answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Session {session}: inspect scripts/verify-full.sh and tell me "
                        "whether there is a server reachability check. Use the tools."
                    ),
                },
            ],
            320,
            temp=0.2,
            tools=True,
        )
    if turn == 3:
        block = (
            "src/example.py: def handle_request(payload):\n"
            "    validate(payload)\n"
            "    result = service.call(payload)\n"
            "    return {'ok': True, 'result': result}\n"
            "tests/test_example.py: assert handle_request({'x': 1})['ok'] is True\n"
            "logs/app.log: INFO request completed in 42ms\n"
        )
        payload = (block * 120)[:12000]
        call_id = f"call_soak_{session}"
        return base_req(
            model,
            [
                {"role": "system", "content": "You are a concise coding assistant."},
                {"role": "user", "content": "Read the relevant files and summarize the failure mode."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "grep",
                                "arguments": json.dumps({"pattern": "handle_request", "dir": "."}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "content": payload},
                {"role": "user", "content": "Identify the most likely missing test and keep it under 8 bullets."},
            ],
            700,
            temp=0.35,
            tools=True,
        )
    if turn == 4:
        return base_req(
            model,
            [
                {"role": "system", "content": "You write direct, production-quality code."},
                {
                    "role": "user",
                    "content": (
                        "Implement a Python function parse_size(s) that accepts values like "
                        "'128MiB', '2 GiB', and '4096', returns bytes, and raises ValueError "
                        "for invalid input. Include compact tests."
                    ),
                },
            ],
            900,
            temp=0.45,
        )
    return base_req(
        model,
        [
            {"role": "system", "content": "Solve carefully and show the final answer clearly."},
            {
                "role": "user",
                "content": (
                    f"Session {session}: Cache A grows by 6 MiB per request until it resets every "
                    "11 requests. Cache B grows by 14 MiB on prime-numbered requests and never "
                    "resets during the run. Starting from 21000 MiB used on a 24576 MiB card, "
                    "after 25 requests what is peak used memory, and which request first exceeds 23000 MiB?"
                ),
            },
        ],
        2000,
        temp=0.2,
        thinking=True,
    )


def cmd_model(path):
    with open(path) as f:
        data = json.load(f)
    models = data.get("data") or []
    print(models[0].get("id", "qwen3.6-27b-autoround") if models else "qwen3.6-27b-autoround")


def cmd_baseline(out_dir, container, endpoint, model, sessions, turns, growth):
    out = pathlib.Path(out_dir)
    models = {}
    try:
        models = json.loads((out / "models.json").read_text())
    except Exception:
        pass
    doc = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "container": container,
        "endpoint": endpoint,
        "model": model,
        "soak_sessions": int(sessions),
        "soak_turns": int(turns),
        "soak_max_growth_mib": int(growth),
        "models": models,
    }
    (out / "baseline.json").write_text(json.dumps(doc, indent=2) + "\n")


def cmd_request(model, session, turn, path):
    req = fixture(model, int(session), int(turn))
    pathlib.Path(path).write_text(json.dumps(req) + "\n")


def cmd_run(endpoint, req_path, timeout_s, metrics_path):
    body = pathlib.Path(req_path).read_bytes()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    completion_tokens = 0
    status = 0
    error = ""
    try:
        with urllib.request.urlopen(req, timeout=int(timeout_s)) as resp:
            status = getattr(resp, "status", 200)
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk and not error:
                    error = str(chunk["error"])[:240]
                choices = chunk.get("choices") or []
                if choices and ttft is None:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                        ttft = time.time() - t0
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            error = e.read(500).decode("utf-8", errors="replace")
        except Exception:
            error = str(e)
    except Exception as e:
        status = 0
        error = f"{type(e).__name__}: {e}"

    wall = time.time() - t0
    if ttft is None:
        ttft = wall
    decode_s = max(wall - ttft, 1e-6)
    data = {
        "status": int(status),
        "error": error.replace("\n", " ")[:300],
        "t_ms": round(wall * 1000),
        "ttft_ms": round(ttft * 1000),
        "decode_tps": round((completion_tokens / decode_s) if completion_tokens else 0.0, 3),
        "completion_tokens": completion_tokens,
    }
    pathlib.Path(metrics_path).write_text(json.dumps(data) + "\n")


def cmd_append_log(log_path, session, turn, vram, metrics_path):
    metrics = json.loads(pathlib.Path(metrics_path).read_text())
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow(
            [
                session,
                turn,
                metrics.get("t_ms", 0),
                vram,
                metrics.get("ttft_ms", 0),
                metrics.get("decode_tps", 0),
                metrics.get("status", 0),
                metrics.get("error", ""),
            ]
        )


def cmd_metric(metrics_path):
    m = json.loads(pathlib.Path(metrics_path).read_text())
    print(m.get("status", 0), m.get("t_ms", 0), m.get("ttft_ms", 0), m.get("decode_tps", 0))


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - (k - lo)) + xs[hi] * (k - lo)


def med(xs):
    return statistics.median(xs) if xs else 0.0


def cmd_summary(turn_log, summary_path, boot_vram, growth_limit, timed_out, expected_sessions):
    boot_vram = int(boot_vram)
    growth_limit = int(growth_limit)
    timed_out = int(timed_out) == 1
    expected_sessions = int(expected_sessions)
    rows = []
    with open(turn_log) as f:
        for row in csv.DictReader(f):
            for key in ("session_id", "turn_id", "t_ms", "vram_mib", "ttft_ms", "status"):
                row[key] = int(float(row[key] or 0))
            row["decode_tps"] = float(row["decode_tps"] or 0)
            rows.append(row)

    sessions = sorted({r["session_id"] for r in rows})
    first = sessions[:5]
    last = sessions[-5:]
    tps = [r["decode_tps"] for r in rows if r["decode_tps"] > 0]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"] > 0]
    first_tps = [r["decode_tps"] for r in rows if r["session_id"] in first and r["decode_tps"] > 0]
    last_tps = [r["decode_tps"] for r in rows if r["session_id"] in last and r["decode_tps"] > 0]
    first_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in first and r["ttft_ms"] > 0]
    last_ttft = [r["ttft_ms"] for r in rows if r["session_id"] in last and r["ttft_ms"] > 0]
    max_vram = max([r["vram_mib"] for r in rows] + [boot_vram])
    growth = max_vram - boot_vram
    errors = [r for r in rows if r["status"] != 200 or r["error"]]
    first_med = med(first_tps)
    last_med = med(last_tps)
    tps_retention = last_med / first_med if first_med > 0 else 0.0
    ttft_ratio = med(last_ttft) / med(first_ttft) if med(first_ttft) > 0 else 0.0
    session_max = [max(r["vram_mib"] for r in rows if r["session_id"] == s) for s in sessions]
    oscillation = max([abs(b - a) for a, b in zip(session_max, session_max[1:])] or [0])
    slow_turns = [r for r in rows if r["t_ms"] > 30000]

    warnings = []
    failures = []
    if errors:
        failures.append(f"{len(errors)} request(s) returned non-200 status or stream error.")
    if growth > growth_limit:
        failures.append(f"VRAM grew {growth} MiB > {growth_limit} MiB threshold.")
    if first_med > 0 and tps_retention < 0.80:
        failures.append(f"Decode TPS retention was {tps_retention * 100:.1f}% < 80%.")
    elif first_med == 0 and rows:
        warnings.append("No positive decode TPS samples; retention could not be evaluated.")
    if ttft_ratio > 1.5:
        warnings.append(f"TTFT grew {ttft_ratio:.2f}x from first sessions to last sessions.")
    if slow_turns:
        warnings.append(f"{len(slow_turns)} turn(s) exceeded 30s.")
    if oscillation > 500:
        warnings.append(f"VRAM session-to-session oscillation reached {oscillation} MiB.")
    if sessions and sessions[-1] < expected_sessions:
        warnings.append(f"Only {sessions[-1]} of {expected_sessions} sessions completed.")

    verdict = "INCONCLUSIVE" if timed_out else ("FAIL" if failures else "PASS")
    exit_code = 2 if timed_out else (1 if failures else 0)
    lines = [
        "# Soak test summary",
        "",
        f"- Verdict: **{verdict}**",
        f"- Boot VRAM baseline: {boot_vram} MiB",
        f"- Max VRAM observed: {max_vram} MiB",
        f"- Max growth observed: {growth} MiB",
        f"- Sessions completed: {len(sessions)}",
        f"- Request errors: {len(errors)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| p50 decode TPS | {percentile(tps, 0.50):.2f} |",
        f"| p95 decode TPS | {percentile(tps, 0.95):.2f} |",
        f"| first-5 median TPS | {first_med:.2f} |",
        f"| last-5 median TPS | {last_med:.2f} |",
        f"| TPS retention | {tps_retention * 100:.1f}% |",
        f"| p50 TTFT | {percentile(ttft, 0.50):.0f} ms |",
        f"| p95 TTFT | {percentile(ttft, 0.95):.0f} ms |",
        f"| TTFT first/last ratio | {ttft_ratio:.2f}x |",
        f"| VRAM oscillation | {oscillation} MiB |",
        "",
    ]
    if failures:
        lines += ["## Failures", "", *[f"- {x}" for x in failures], ""]
    if warnings:
        lines += ["## Warnings", "", *[f"- {x}" for x in warnings], ""]
    if errors[:10]:
        lines += ["## First request errors", ""]
        lines += [f"- session {r['session_id']} turn {r['turn_id']}: status={r['status']} error={r['error'][:160]}" for r in errors[:10]]
        lines += [""]
    rec = "Runtime VRAM growth and throughput retention stayed within v1 soak thresholds."
    if verdict == "FAIL":
        rec = "Inspect docker logs and compare turn-log.csv against GPU snapshots to identify the accreting path."
    elif verdict == "INCONCLUSIVE":
        rec = "Re-run with a larger SOAK_TIMEOUT_S or fewer/lighter sessions before treating this config as soak-clean."
    lines += ["## Recommendation", "", f"- {rec}"]
    pathlib.Path(summary_path).write_text("\n".join(lines) + "\n")

    print("")
    print("[soak] summary")
    print(f"[soak]   verdict              {verdict}")
    print(f"[soak]   boot_vram_mib        {boot_vram}")
    print(f"[soak]   max_vram_mib         {max_vram}")
    print(f"[soak]   max_growth_mib       {growth} / {growth_limit}")
    print(f"[soak]   errors               {len(errors)}")
    print(f"[soak]   p50_decode_tps       {percentile(tps, 0.50):.2f}")
    print(f"[soak]   p95_ttft_ms          {percentile(ttft, 0.95):.0f}")
    print(f"[soak]   tps_retention        {tps_retention * 100:.1f}%")
    for label, items in (("failures", failures), ("warnings", warnings)):
        if items:
            print(f"[soak] {label}:")
            for item in items:
                print(f"[soak]   - {item}")
    sys.exit(exit_code)


def main():
    cmd = sys.argv[1]
    args = sys.argv[2:]
    {
        "model": cmd_model,
        "baseline": cmd_baseline,
        "request": cmd_request,
        "run": cmd_run,
        "append-log": cmd_append_log,
        "metric": cmd_metric,
        "summary": cmd_summary,
    }[cmd](*args)


if __name__ == "__main__":
    main()
