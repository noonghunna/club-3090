"""Check the model context boundary through its local API."""

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8144")
parser.add_argument("--model", default="qwen3.8-27b")
args = parser.parse_args()
endpoint = args.url.rstrip("/")
model = args.model
secret = "ORCHID-739125"
prefix = "<|im_start|>user\nRead the notes and remember the verification code.\n"
middle = f"\nThe verification code is {secret}.\n"
suffix = (
    "\nWhat is the verification code? Reply with only the code."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
tokens = []
for part in (prefix, " filler", middle, suffix):
    request = urllib.request.Request(
        endpoint + "/tokenize",
        data=json.dumps({"model": model, "prompt": part, "add_special_tokens": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        tokens.append(json.load(response)["tokens"])

budget = 261000 - len(tokens[0]) - len(tokens[2]) - len(tokens[3])
padding = (tokens[1] * (budget // len(tokens[1]) + 1))[:budget]
prompt = tokens[0] + padding[:budget // 2] + tokens[2] + padding[budget // 2:] + tokens[3]
request = urllib.request.Request(
    endpoint + "/v1/completions",
    data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 64, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"},
)
print(f"prompt_tokens={len(prompt)}", flush=True)
started = time.monotonic()
peaks = [0, 0]
def completed_response():
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    return result, time.monotonic()

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    pending = pool.submit(completed_response)
    while not pending.done():
        sample = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, encoding="utf-8",
        )
        peaks = [max(old, int(value)) for old, value in zip(peaks, sample.stdout.splitlines())]
        time.sleep(5)
    result, finished = pending.result()
text = result["choices"][0]["text"]
print(json.dumps({"seconds": round(finished - started, 3), "usage": result["usage"],
                  "answer": text, "peak_vram_mib": peaks, "recall": secret in text}), flush=True)
assert secret in text, "The model did not recall the code near the context limit"

oversized = urllib.request.Request(
    endpoint + "/v1/completions",
    data=json.dumps({"model": model, "prompt": prompt + padding[:2000], "max_tokens": 1}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(oversized, timeout=30) as response:
        raise AssertionError(f"Oversized request was accepted: {response.status}")
except urllib.error.HTTPError as error:
    print(f"over_limit_http={error.code}", flush=True)
    assert error.code == 400, error.read().decode()
