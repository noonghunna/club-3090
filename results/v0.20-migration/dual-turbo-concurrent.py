#!/usr/bin/env python3
"""4-stream concurrent throughput on dual-turbo (the variant's selling point)."""
import json, time, urllib.request, statistics, threading, queue

URL = "http://localhost:8011"
MODEL = "qwen3.6-27b-autoround"

PROMPT = "Write a Python implementation of quicksort with comments explaining each step."
MAX_TOKENS = 800

def run_once(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
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
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    return time.time() - t_send, completion_tokens

def worker(out_q, n_runs):
    for _ in range(n_runs):
        try:
            wall, toks = run_once(PROMPT, MAX_TOKENS)
            out_q.put((wall, toks))
        except Exception as e:
            out_q.put(None)

def run_concurrent(n_streams, runs_per_stream):
    print(f"\n=== {n_streams} concurrent streams, {runs_per_stream} run(s) each ===")
    q = queue.Queue()
    threads = []
    t_start = time.time()
    for _ in range(n_streams):
        t = threading.Thread(target=worker, args=(q, runs_per_stream))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t_total = time.time() - t_start
    results = []
    while not q.empty():
        r = q.get()
        if r:
            results.append(r)
    if not results:
        print("no completions")
        return
    walls = [w for w,_ in results]
    toks = [t for _,t in results]
    total_toks = sum(toks)
    aggregate_tps = total_toks / t_total
    per_stream_tps = [t/w for w,t in results]
    print(f"  total_toks={total_toks}  total_wall={t_total:.2f}s  aggregate_TPS={aggregate_tps:.2f}")
    print(f"  per-stream  mean={statistics.mean(per_stream_tps):.2f}  CV={statistics.stdev(per_stream_tps)/statistics.mean(per_stream_tps)*100 if len(per_stream_tps) > 1 else 0:.1f}%")
    return aggregate_tps, statistics.mean(per_stream_tps)

# Warm up: 1 stream, 1 run
print("warmup...")
run_concurrent(1, 1)

# Bench
for n in (1, 2, 3, 4):
    run_concurrent(n, 2)
