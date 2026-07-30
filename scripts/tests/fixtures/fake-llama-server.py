#!/usr/bin/env python3
"""Fake llama-server for offload-matrix Tier 1 tests.

Emits the exact log lines offload-matrix.sh scrapes and serves the two endpoints
it calls, so the whole sweep-matrix logic can be exercised deterministically with
no GPU, no model and no engine. Scenario is chosen with FAKE_SCENARIO:

  healthy    both devices allocate pools, tokens stream normally
  partial    only CUDA0 allocates a pool (the per-device budget failure that
             still prints "enabled:" -- must be caught as CACHE_PARTIAL)
  nobudget   no device allocates a pool, "no cache budget" logged
  bypass     pools allocated but every decode logs "bypassing cache"
  notokens   endpoints answer but the stream yields zero content deltas
  probefail  the n_layer line NEVER appears, so the layer probe times out -- and
             the process IGNORES SIGTERM, the way a real server does while it is
             still inside model load. Only a probe that escalates past SIGTERM
             reaps this one; anything else leaks a server holding the model's VRAM

Anything the real server prints that the script does NOT scrape is omitted on
purpose: the fixture documents the contract, so an accidental dependency on some
other line shows up as a test failure rather than working by luck.
"""
import json, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

SC   = os.environ.get("FAKE_SCENARIO", "healthy")
NDEV = int(os.environ.get("FAKE_NDEV", "2"))
TOKS = int(os.environ.get("FAKE_TOKENS", "40"))

if "--help" in sys.argv:                      # engine-scope probe
    print("  -ngl,  --n-gpu-layers N\n  -ot,   --override-tensor PATTERN\n         --moe-cache N")
    raise SystemExit(0)

def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

def log(s): print(s, flush=True)

port  = int(arg("--port", "8099"))
n_ctx = int(arg("-c", "32768"))
n_par = int(arg("-np", "1"))

if SC == "probefail":
    # Never print n_layer and never reach "model loaded": the probe must burn its
    # whole PROBE_TIMEOUT. Ignoring SIGTERM models a server that has not installed
    # its shutdown handler yet -- `kill` returns 0 and the process lives on, so a
    # probe that sends SIGTERM and calls `wait` blocks forever on a child that
    # never exits, and is itself orphaned if the operator gives up on it.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log("main: loading model ...")
    while True:
        time.sleep(0.2)

log(f"print_info: n_layer          = 48")
log(f"print_info: n_expert         = 256")
log(f"llama_context: n_ctx_seq     = {n_ctx // max(n_par,1)}")

pooled = {"healthy": NDEV, "partial": 1, "nobudget": 0}.get(SC, NDEV)
if "--moe-cache" in sys.argv:
    if SC == "nobudget":
        for d in range(NDEV):
            log(f"[moe-cache] CUDA{d} has no cache budget after 3072 MiB reserve")
    else:
        if SC == "partial":
            log("[moe-cache] CUDA1 has no cache budget after 3072 MiB reserve")
        log("[moe-cache] enabled: mode=on budget=fixed reserve=1536 MiB min-expert=1024 KiB admit=1/64")
        for d in range(pooled):
            log(f"[moe-cache] CUDA{d} pool[0]: type=iq3_s expert=1320 KiB slots={700+d} total=900 MiB")

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if SC == "bypass":
            log("[moe-cache] bypassing cache: decode batch of 4 tokens exceeds "
                "GGML_CUDA_MOE_CACHE_MAX_BATCH=1 — raise it to at least draft_max+1.")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        n = 0 if SC == "notokens" else TOKS
        for _ in range(n):
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'); self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        log(f"slot print_timing: id  0 | task 1 | n_decoded = {n}, tg = 40.00 t/s")
        if pooled:
            for d in range(pooled):
                log(f"[moe-cache] CUDA{d} hits=500/1000 (50.0%) used=64/64 enqueued=64 filled=64 "
                    f"fill-fail=0 evictions=10 skips=0 admission=0 queue=0 jobs/0 MiB "
                    f"dispatch-fail=0 collect-fail=0")

srv = HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
log("main: server is listening")
try:
    while True: time.sleep(0.2)
except KeyboardInterrupt:
    pass
