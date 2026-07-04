# Client examples

Copy-pasteable snippets for talking to the club-3090 endpoint. The default URL is `http://localhost:8020`; the served model name is `qwen3.6-27b` (vLLM) or `qwen3.6-27b` (llama.cpp via the `--alias` flag we set).

All examples assume:

- Server running: `bash scripts/launch.sh` is up
- API endpoint: `http://localhost:8020` (override with `OPENAI_BASE_URL` env var or client-side `base_url`)

The endpoint is **OpenAI-compatible** — anything that speaks OpenAI's `/v1/chat/completions` API works without modification, just point `base_url` at the local endpoint.

---

## `max_tokens` defaults — important if you've enabled thinking

Qwen3.6-27B is a thinking model. The `<think>...</think>` block before the answer routinely runs 2-4K tokens on medium reasoning, 4-8K on harder coding problems, and can exceed 16K on competition-grade problems. If `max_tokens` cuts the response off mid-think, the model never reaches the answer and the request looks like an "empty response" or truncated garbage. We hit this exact trap on our LiveCodeBench v6 baseline ([`docs/STRUCTURED_COT.md`](STRUCTURED_COT.md) caveat section).

**Use these defaults:**

| Scenario | `max_tokens` |
|---|---|
| **FREE thinking (enabled per-request — see note below)** | **8192** minimum. 16384 for hard reasoning / competition-grade problems. |
| **FSM bounded thinking (`bounded-thinking.yml`)** | **4096** is comfortable. The recommended DeepSeek scratchpad grammar uses ~500-1000 think tokens; the andthattoo G/A/E grammar uses ~150. Either fits well below 4096. |
| **`enable_thinking: False`** | Set as tight as the answer needs (50-200 typically). |
| **Tool-using agents (multi-turn)** | 1024-2048 per turn. If a middle turn needs >2K to think, your prompt structure probably needs work. |

The smoke-test examples below use `max_tokens: 200` because they ask short questions where thinking + answer fits comfortably. Real workloads should follow the table above.

> **Thinking is OFF by default on the shipped composes.** Every Qwen3.6 compose sets `--default-chat-template-kwargs '{"enable_thinking": false}'`, so the model answers directly with no `<think>` block unless you opt in. Enable it per-request with `chat_template_kwargs: {"enable_thinking": true}` (no restart) and budget `max_tokens` per the table. The one exception is `bounded-thinking.yml`, which keeps thinking on but bounds its cost via a structured-CoT grammar (see [`docs/STRUCTURED_COT.md`](STRUCTURED_COT.md)).

---

## Quick curl sanity test

```bash
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Capital of France?"}],
    "max_tokens": 200
  }' | jq -r '.choices[0].message.content'
```

Expected response: a sentence containing `Paris`. The shipped composes set `enable_thinking: false` by default, so the model answers directly with no `<think>` block — `max_tokens: 200` is comfortable slack. If you enable thinking per-request (`chat_template_kwargs: {"enable_thinking": true}`), raise `max_tokens` substantially (see the table above) — the model then emits a `<think>...</think>` block first even for simple questions. `verify-full.sh` passes `enable_thinking: false` explicitly.

---

## Python — `openai` SDK (recommended)

```bash
pip install openai
```

### Basic chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Write a haiku about tensor cores."}],
    max_tokens=120,
    temperature=0.6,
    top_p=0.95,
)
print(resp.choices[0].message.content)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Explain attention in 100 words."}],
    max_tokens=300,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="", flush=True)
print()
```

### Tool calling

Works on both engines (vLLM with `--tool-call-parser qwen3_coder` and llama.cpp with `--jinja` — both ship enabled in the default composes):

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=200,
)

msg = resp.choices[0].message
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Call {tc.function.name}({tc.function.arguments})")
else:
    print(msg.content)
```

### Vision (image input)

vLLM and llama.cpp both auto-load the vision tower / mmproj when configured. Send images as base64 or URLs:

```python
import base64
from pathlib import Path

img_b64 = base64.b64encode(Path("photo.png").read_bytes()).decode()

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        }
    ],
    max_tokens=200,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

### Reasoning mode (vLLM with Genesis only)

```python
resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Solve: 7x + 14 = 49. Show your reasoning."}],
    max_tokens=2048,  # FREE thinking on; 2048 fits easy math comfortably. Bump to 8192 for harder reasoning.
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", "") or "(empty)")
print("Answer:   ", msg.content)
```

> **Note:** llama.cpp emits the `<think>...</think>` tokens inline rather than peeling them into a separate `reasoning_content` field. If you need that split, post-process client-side or stick with vLLM.

---

## Python — `requests` (no SDK)

For environments where you can't install the `openai` package:

```python
import requests, json

resp = requests.post(
    "http://localhost:8020/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "qwen3.6-27b",
        "messages": [{"role": "user", "content": "What is 17 × 23?"}],
        "max_tokens": 50,
    },
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])
```

For streaming, use `stream=True` and parse SSE lines:

```python
with requests.post(
    "http://localhost:8020/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={"model": "qwen3.6-27b", "messages": [...], "stream": True, "max_tokens": 200},
    stream=True,
) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = line[6:]
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
```

---

## Python — tool calling (agentic workflow)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8020/v1", api_key="not-needed")

# Define a tool the model can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            },
        }
    }
]

# First turn: model decides to call the tool
resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=512,
)

msg = resp.choices[0].message
print(f"Tool call: {msg.tool_calls}")

# Second turn: feed tool result back
messages = [
    {"role": "user", "content": "What's the weather in Paris?"},
    msg,
    {"role": "tool", "tool_call_id": msg.tool_calls[0].id, "content": "22°C, sunny"},
]

resp2 = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=messages,
    tools=tools,
    max_tokens=512,
)

print(f"Final answer: {resp2.choices[0].message.content}")
```

---

## TypeScript / Node — `openai` SDK

```bash
npm install openai
```

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8020/v1",
  apiKey: "not-needed",
});

const resp = await client.chat.completions.create({
  model: "qwen3.6-27b",
  messages: [{ role: "user", content: "Quicksort in Rust, please." }],
  // Shipped composes default enable_thinking:false → this answers with no <think> block; 4096 is generous.
  // To get reasoning, add extra_body chat_template_kwargs {"enable_thinking": true} and budget 8192+ (800 traps mid-think).
  max_tokens: 4096,
  temperature: 0.6,
  top_p: 0.95,
});

console.log(resp.choices[0].message.content);
```

Streaming:

```ts
const stream = await client.chat.completions.create({
  model: "qwen3.6-27b",
  messages: [{ role: "user", content: "..." }],
  max_tokens: 300,
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
process.stdout.write("\n");
```

---

## Connecting third-party clients

### Open WebUI

Settings → Connections → Add OpenAI Connection:

- **Base URL:** `http://localhost:8020/v1`  *(or `http://<host-ip>:8020/v1` from another machine on your LAN — see [Security](#security-note-network-binding))*
- **API Key:** anything (e.g. `sk-local`) — the server doesn't check it
- **Model:** `qwen3.6-27b`

Vision, tool calling, streaming all work through the WebUI's standard flows.

### Cline / Roo (VS Code agentic coder)

In the Cline settings panel:

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://localhost:8020/v1`
- **API Key:** `sk-local` (any non-empty string)
- **Model ID:** `qwen3.6-27b`

Cline sends large tool returns (file reads, web fetches) up to ~25K tokens. As of 2026-05-02 PM (Genesis v7.69 dev tip + vllm#35975 backport), `vllm/long-text` (180K balanced + MTP K=3) handles these cleanly — 33K AND 50K tool-prefill stress PASS, and **60K single-prompt prefill PASS** (the Cliff 2 wall closed at 60K). For one-shot prompts beyond 60K, switch to `llamacpp/default` (262K vanilla, slower), `llamacpp/mtp` (131K + MTP, ~60 code TPS, single-card, 7/7 verify-stress incl. 91K needle), or `dual-turbo.yml` (262K + 4 streams). See [docs/SINGLE_CARD.md](SINGLE_CARD.md), [docs/CLIFFS.md](CLIFFS.md), and the [VRAM diagram](../models/qwen3.6-27b/README.md#vram-allocation-across-configs).

### Cursor

Settings → Models → Add OpenAI-compatible:

- **Override OpenAI Base URL:** `http://localhost:8020/v1`
- **Verify config:** click "Verify" — should list `qwen3.6-27b`
- **Model name:** `qwen3.6-27b`

Cursor's "Apply" feature works against this model since tool-calling is supported.

### LiteLLM proxy / aider / Continue.dev

All work the same way — OpenAI-compatible endpoint at `http://localhost:8020/v1`, any non-empty API key. Confirmed working with the default compose.

---

## Security note: network binding

The default composes bind to `0.0.0.0:8020` so other machines on your LAN can connect. **If you're on a shared / coffee-shop / coworking network, that exposes your model to anyone who can route to your machine.**

To restrict to localhost-only:

```yaml
# In any docker-compose.*.yml under ports:
ports:
  - "127.0.0.1:8020:8000"   # was: "8020:8000"
```

Or override at run-time with `--host 127.0.0.1` (llama.cpp) / by editing the compose locally.

---

## Switch the served model over HTTP

On a 1–2 GPU rig only one model fits in VRAM at a time. The optional
[`tools/model-switch`](../tools/model-switch/README.md) service wraps
`scripts/switch.sh` behind an HTTP endpoint so an experiment harness can swap
models programmatically (it adds no orchestration — `switch.sh` stays the source
of truth). stdlib-only; run it on the host:

```bash
python3 tools/model-switch/server.py        # or the club3090-model-switch systemd unit
```

```bash
TOKEN=...   # CLUB3090_API_TOKEN (falls back to VLLM_API_KEY); empty = open on loopback
# what's serving now:
curl -s -H "Authorization: Bearer $TOKEN" localhost:8099/status
# switch (blocks until the new model is ready, ~1-2 min):
curl -s -XPOST -H "Authorization: Bearer $TOKEN" localhost:8099/switch -d '{"model":"gemma-4-31b"}'
# list switchable slugs:
curl -s -H "Authorization: Bearer $TOKEN" localhost:8099/models
```

---

## See also

- [`tools/model-switch/README.md`](../tools/model-switch/README.md) — HTTP endpoint to swap the served model
- [`models/qwen3.6-27b/README.md`](../models/qwen3.6-27b/README.md) — variant matrix + VRAM diagram
- [`docs/SINGLE_CARD.md`](SINGLE_CARD.md) and [`docs/DUAL_CARD.md`](DUAL_CARD.md) — workload → recommended compose
- [`scripts/launch.sh`](../scripts/launch.sh) — interactive variant picker
- [`scripts/health.sh`](../scripts/health.sh) — runtime health probe
