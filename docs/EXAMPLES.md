# Client examples

Copy-pasteable snippets for talking to the club-3090 endpoint. The default URL is `http://localhost:8020`; the served model name is `qwen3.6-27b-autoround` (vLLM) or `qwen3.6-27b-autoround` (llama.cpp via the `--alias` flag we set).

All examples assume:

- Server running: `bash scripts/launch.sh` is up
- API endpoint: `http://localhost:8020` (override with `OPENAI_BASE_URL` env var or client-side `base_url`)

The endpoint is **OpenAI-compatible** — anything that speaks OpenAI's `/v1/chat/completions` API works without modification, just point `base_url` at the local endpoint.

---

## Quick curl sanity test

```bash
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b-autoround",
    "messages": [{"role": "user", "content": "Capital of France?"}],
    "max_tokens": 30
  }' | jq -r '.choices[0].message.content'
```

Expected response: a sentence containing `Paris`.

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
    model="qwen3.6-27b-autoround",
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
    model="qwen3.6-27b-autoround",
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
    model="qwen3.6-27b-autoround",
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
    model="qwen3.6-27b-autoround",
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
    model="qwen3.6-27b-autoround",
    messages=[{"role": "user", "content": "Solve: 7x + 14 = 49. Show your reasoning."}],
    max_tokens=400,
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
        "model": "qwen3.6-27b-autoround",
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
    json={"model": "qwen3.6-27b-autoround", "messages": [...], "stream": True, "max_tokens": 200},
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
  model: "qwen3.6-27b-autoround",
  messages: [{ role: "user", content: "Quicksort in Rust, please." }],
  max_tokens: 800,
  temperature: 0.6,
  top_p: 0.95,
});

console.log(resp.choices[0].message.content);
```

Streaming:

```ts
const stream = await client.chat.completions.create({
  model: "qwen3.6-27b-autoround",
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
- **Model:** `qwen3.6-27b-autoround`

Vision, tool calling, streaming all work through the WebUI's standard flows.

### Cline / Roo (VS Code agentic coder)

In the Cline settings panel:

- **API Provider:** OpenAI Compatible
- **Base URL:** `http://localhost:8020/v1`
- **API Key:** `sk-local` (any non-empty string)
- **Model ID:** `qwen3.6-27b-autoround`

Cline sends large tool returns (file reads, web fetches) — at 25K+ tokens these hit Cliff 1 on vLLM single-card 192K configs. Use the `vllm/tools-text` variant (Cliff 1 closed via Genesis PN8 since 2026-04-29) or switch to `llamacpp/default` for cliff-free serving. See [docs/SINGLE_CARD.md](SINGLE_CARD.md) and the [VRAM diagram](../models/qwen3.6-27b/README.md#vram-allocation-across-configs).

### Cursor

Settings → Models → Add OpenAI-compatible:

- **Override OpenAI Base URL:** `http://localhost:8020/v1`
- **Verify config:** click "Verify" — should list `qwen3.6-27b-autoround`
- **Model name:** `qwen3.6-27b-autoround`

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

## See also

- [`models/qwen3.6-27b/README.md`](../models/qwen3.6-27b/README.md) — variant matrix + VRAM diagram
- [`docs/SINGLE_CARD.md`](SINGLE_CARD.md) and [`docs/DUAL_CARD.md`](DUAL_CARD.md) — workload → recommended compose
- [`scripts/launch.sh`](../scripts/launch.sh) — interactive variant picker
- [`scripts/health.sh`](../scripts/health.sh) — runtime health probe
