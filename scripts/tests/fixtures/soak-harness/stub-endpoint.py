#!/usr/bin/env python3
"""Scriptable OpenAI-compatible SSE stub for the soak-harness tests.

Serves /v1/models and a streaming /v1/chat/completions whose behaviour is
driven by a PLAN file — one directive per chat request, consumed in order,
last line repeating once exhausted. This lets a test drive scripted engine
death, canvas-granularity single-chunk responses, and normal autoregressive
streaming without a GPU, a container, or a real engine.

Usage:
  stub-endpoint.py <port-file> <plan-file> <vram-file> [alive-file]

When <alive-file> is given, /v1/models answers 200 only while that path exists.
Deleting it makes the endpoint stop answering without stopping the process —
which is how the report tests simulate an engine that crashed mid-run.

Plan directive syntax (one per line, '#' comments and blanks ignored):

  ok[:VRAM]        normal autoregressive stream — first content delta, a
                   ~150 ms gap, then the rest + usage. Decode window is
                   comfortably measurable.
  narrow[:VRAM]    autoregressive fast burst — first content delta, ~80 ms
                   gap, then the rest + usage. Window is under the harness's
                   100 ms client-timing floor but nowhere near zero. The #849
                   shape: must NOT be reclassified as canvas, and since #1267
                   must render as `decode_tps=n/a` + the window width rather
                   than a 0.0 indistinguishable from a silent-empty turn. With
                   STUB_TPOT_TPS set it is also the #1268 shape — a turn the
                   engine counter can measure and the SSE window cannot.
  canvas[:VRAM]    canvas granularity — the entire response arrives in ONE
                   chunk carrying content + usage, so the decode window is
                   zero-width. The #809 shape.
  empty[:VRAM]     HTTP 200, stream closes with usage completion_tokens=0.
                   Genuine silent-empty; must stay decode_tps=0.0.
  http:CODE[:VRAM] respond with an HTTP error status.
  dead[:VRAM]      accept the connection then drop it mid-stream (engine death).

The optional trailing VRAM value is written to <vram-file> BEFORE the response
is produced, so the fake nvidia-smi the harness calls after each turn reports
it. That is how a test reproduces "the baseline was taken on a corpse".

Env:
  STUB_TPOT_TPS   When set to a float, the stub also serves a Prometheus
                  /metrics page carrying a `vllm:time_per_output_token_seconds`
                  histogram (plus a decoy histogram and a `_bucket` line that
                  must NOT be mistaken for `_sum`/`_count`). Its totals advance
                  by one request's worth on every chat completion, arranged so
                  the (_count, _sum) DELTA across a turn is exactly this many
                  tokens per second. That is the engine-counter path of #1268.
                  Unset (the default) makes /metrics 404, which is the
                  client-timed fallback path.
"""

import json
import os
import pathlib
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLAN = []
PLAN_IDX = [0]
PLAN_LOCK = threading.Lock()
VRAM_FILE = [""]
ALIVE_FILE = [""]
# (count, sum_seconds) for the fake time_per_output_token_seconds histogram.
TPOT = [0.0, 0.0]


def next_directive():
    with PLAN_LOCK:
        if not PLAN:
            return ("ok", None)
        idx = min(PLAN_IDX[0], len(PLAN) - 1)
        PLAN_IDX[0] += 1
        return PLAN[idx]


def set_vram(value):
    if value is not None and VRAM_FILE[0]:
        pathlib.Path(VRAM_FILE[0]).write_text(str(value) + "\n", encoding="utf-8")


def advance_tpot(completion_tokens):
    """Book `completion_tokens` decode steps at exactly STUB_TPOT_TPS tok/s."""
    tps = float(os.environ.get("STUB_TPOT_TPS") or 0)
    if tps <= 0 or completion_tokens <= 0:
        return
    with PLAN_LOCK:
        TPOT[0] += completion_tokens
        TPOT[1] += completion_tokens / tps


def metrics_page():
    """A Prometheus page shaped like vLLM's, including traps for the parser:
    a `_bucket` line (must not be read as sum/count) and a decoy histogram
    (must not be read as the TPOT one)."""
    with PLAN_LOCK:
        count, total = TPOT[0], TPOT[1]
    return (
        "# HELP vllm:e2e_request_latency_seconds End to end request latency.\n"
        "# TYPE vllm:e2e_request_latency_seconds histogram\n"
        'vllm:e2e_request_latency_seconds_sum{model_name="stub-model"} 9999.0\n'
        'vllm:e2e_request_latency_seconds_count{model_name="stub-model"} 1.0\n'
        "# HELP vllm:time_per_output_token_seconds Inter-token latency.\n"
        "# TYPE vllm:time_per_output_token_seconds histogram\n"
        'vllm:time_per_output_token_seconds_bucket{le="0.01",model_name="stub-model"} 777\n'
        'vllm:time_per_output_token_seconds_sum{model_name="stub-model"} %r\n'
        'vllm:time_per_output_token_seconds_count{model_name="stub-model"} %r\n'
    ) % (total, count)


def sse(payload):
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def chunk(content=None, usage=None):
    body = {
        "id": "chatcmpl-stub",
        "object": "chat.completion.chunk",
        "model": "stub-model",
        "choices": [],
    }
    if content is not None:
        body["choices"] = [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
    else:
        body["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    if usage is not None:
        body["usage"] = usage
    return sse(body)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass  # keep the test output clean

    def do_GET(self):
        if self.path.rstrip("/").endswith("/metrics"):
            if not (os.environ.get("STUB_TPOT_TPS") or "").strip():
                self.send_error(404)
                return
            payload = metrics_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.rstrip("/").endswith("/v1/models"):
            if ALIVE_FILE[0] and not pathlib.Path(ALIVE_FILE[0]).exists():
                # The "engine" has crashed: still listening, no longer serving.
                self.send_error(503, "engine down")
                return
            payload = json.dumps({"data": [{"id": "stub-model", "object": "model"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        kind, vram = next_directive()
        set_vram(vram)

        if kind.startswith("http:"):
            code = int(kind.split(":", 1)[1])
            body = json.dumps({"error": {"message": "stub scripted failure"}}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            if kind == "dead":
                # Engine death: a couple of bytes then the socket goes away.
                self.wfile.write(b": stub\n\n")
                self.wfile.flush()
                self.close_connection = True
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return

            usage = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}

            if kind == "empty":
                # Slow enough to clear cmd_summary's t_ms >= 1000 ms floor, which
                # is what separates a genuine silent-empty turn from a fast no-op.
                time.sleep(1.2)
                usage["completion_tokens"] = 0
                self.wfile.write(chunk(usage=usage))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            advance_tpot(usage["completion_tokens"])

            if kind == "canvas":
                # ONE chunk carrying the whole canvas + usage: the decode window
                # collapses to zero width, which is the #809 signature.
                self.wfile.write(chunk(content="canvas block " * 20, usage=usage))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            gap = 0.08 if kind == "narrow" else 0.15
            self.wfile.write(chunk(content="first "))
            self.wfile.flush()
            time.sleep(gap)
            for _ in range(4):
                self.wfile.write(chunk(content="more tokens "))
            self.wfile.write(chunk(usage=usage))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def main():
    port_file, plan_file, vram_file = sys.argv[1], sys.argv[2], sys.argv[3]
    VRAM_FILE[0] = vram_file
    ALIVE_FILE[0] = sys.argv[4] if len(sys.argv) > 4 else ""
    for raw in pathlib.Path(plan_file).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("http:"):
            parts = line.split(":")
            kind = "http:" + parts[1]
            vram = parts[2] if len(parts) > 2 else None
        else:
            parts = line.split(":")
            kind = parts[0]
            vram = parts[1] if len(parts) > 1 else None
        PLAN.append((kind, vram))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    pathlib.Path(port_file).write_text(str(server.server_address[1]) + "\n", encoding="utf-8")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
