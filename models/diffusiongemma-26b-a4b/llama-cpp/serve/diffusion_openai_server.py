#!/usr/bin/env python3
# ===========================================================================
# OpenAI-compatible shim for DiffusionGemma-26B-A4B on llama.cpp.
#
# WHY this exists: DiffusionGemma is a block-diffusion LM. llama.cpp support
# lives in draft PR #24423 (see docs/UPSTREAM.md), which ships an interactive
# `llama-diffusion-cli` and a low-level logits microservice — but NO OpenAI
# HTTP server. This shim spawns a *patched* `llama-diffusion-cli` running in
# `DG_NDJSON=1` mode (one JSON request per stdin line -> one JSON response per
# stdout line, model resident, prefix-KV-cache on) and exposes the usual
# /v1/chat/completions + /v1/models so local coding harnesses can use it like
# any other endpoint in this repo.
#
# Status: EXPERIMENTAL / upstream-gated. Single in-flight request (np=1). No
# native tool-call JSON yet (assistant content only). "Streaming" emits the
# final answer as one SSE chunk (diffusion is block-wise, not token-by-token).
#
# Config (env):
#   DG_GGUF      path to the Q4_K_M GGUF                 (required)
#   DG_CLI_BIN   path to patched llama-diffusion-cli     (required)
#   DG_PORT      listen port                             (default 8060)
#   DG_HOST      bind address                            (default 0.0.0.0)
#   DG_NGL       gpu layers                              (default 99)
#   DG_CTX       n_ctx == n_ubatch (prompt+canvas cap)   (default 3072)
#   DG_MAXGEN    launch -n (sets the block ceiling)      (default 1024)
#   DG_MODEL_ID  model id reported to clients            (default diffusiongemma-26b-a4b)
#
# VRAM CEILING: the diffusion forward holds the whole [prompt | canvas] in one
# non-causal ubatch, so VRAM grows with DG_CTX. On a 24 GB 3090, Q4_K_M peaks
# ~22-23 GB already at small ctx — raising DG_CTX for long coding prompts will
# OOM. Start at 3072 and raise cautiously while watching nvidia-smi.
# ===========================================================================
import json
import os
import re
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

DG_GGUF     = os.environ.get("DG_GGUF", "")
DG_CLI_BIN  = os.environ.get("DG_CLI_BIN", "")
DG_PORT     = int(os.environ.get("DG_PORT", "8060"))
DG_HOST     = os.environ.get("DG_HOST", "0.0.0.0")
DG_NGL      = os.environ.get("DG_NGL", "99")
DG_CTX      = os.environ.get("DG_CTX", "3072")
DG_MAXGEN   = os.environ.get("DG_MAXGEN", "1024")
DG_MODEL_ID = os.environ.get("DG_MODEL_ID", "diffusiongemma-26b-a4b")

# --- raw-output parsing ----------------------------------------------------
# DiffusionGemma emits its assistant turn with a "thought" channel that the
# template opens as `<|channel>thought` and closes with `<channel|>`, after
# which the actual answer follows. Turns are closed with `<turn|>`. We split
# the reasoning out and return the answer as `content`.
_CHANNEL_CLOSE = "<channel|>"
_THOUGHT_OPEN  = "<|channel>thought"
_TURN_MARKERS  = ("<turn|>", "<|turn>", "<end_of_turn>", "<eos>", "<end>")


def parse_diffusion_output(raw: str):
    """Return (content, reasoning_or_None) from a raw canvas response.

    Only a genuine `<|channel>thought ... <channel|>` block is treated as
    reasoning. A bare `<channel|>` with no thought opener is just a stray
    marker (the prompt prefills an empty thought channel when thinking is
    off) and must NOT be mistaken for a reasoning boundary — otherwise a
    short answer emitted before it gets dropped.
    """
    text = raw
    reasoning = None
    topen = text.find(_THOUGHT_OPEN)
    if topen != -1:
        close = text.find(_CHANNEL_CLOSE, topen + len(_THOUGHT_OPEN))
        if close != -1:
            reasoning = text[topen + len(_THOUGHT_OPEN):close].strip() or None
            text = text[:topen] + text[close + len(_CHANNEL_CLOSE):]
    # Drop any remaining stray channel / turn / eos markers, keep the rest as content.
    text = text.replace(_CHANNEL_CLOSE, "")
    text = re.sub(r"<\|channel>\w*", "", text)
    for m in _TURN_MARKERS:
        text = text.replace(m, "")
    return text.strip(), reasoning


def est_tokens(s: str) -> int:
    # rough estimate (no tokenizer in-process); ~4 chars/token
    return max(1, len(s) // 4) if s else 0


# --- resident backend subprocess ------------------------------------------
class DiffusionBackend:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.meta = {}

    def start(self):
        if not DG_GGUF or not os.path.exists(DG_GGUF):
            raise RuntimeError(f"DG_GGUF not found: {DG_GGUF!r}")
        if not DG_CLI_BIN or not os.path.exists(DG_CLI_BIN):
            raise RuntimeError(f"DG_CLI_BIN not found: {DG_CLI_BIN!r}")
        cmd = [
            DG_CLI_BIN,
            "-m", DG_GGUF,
            "-ngl", DG_NGL,
            "-c", DG_CTX,
            "-ub", DG_CTX,
            "-n", DG_MAXGEN,
        ]
        env = dict(os.environ, DG_NDJSON="1")
        self.proc = subprocess.Popen(
            cmd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None,  # let the engine's diagnostics flow to our stderr
            text=True, bufsize=1,
        )
        # Wait for the `{"ready":true,...}` banner (model load can take ~15-30s).
        deadline = time.time() + 600
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("backend exited before ready")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("ready"):
                self.meta = obj
                return
        raise RuntimeError("backend did not report ready in time")

    def generate(self, messages, n_predict, temperature, seed):
        req = {"messages": messages}
        if n_predict is not None:
            req["n_predict"] = int(n_predict)
        if temperature is not None:
            req["temperature"] = float(temperature)
        if seed is not None:
            req["seed"] = int(seed)
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("backend not running")
            self.proc.stdin.write(json.dumps(req) + "\n")
            self.proc.stdin.flush()
            # one response line per request (skip any blank lines)
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("backend closed mid-request")
                line = line.strip()
                if line:
                    break
        obj = json.loads(line)
        if "error" in obj:
            raise RuntimeError(obj["error"])
        return obj

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


backend = DiffusionBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    backend.start()
    yield
    backend.stop()


app = FastAPI(title="DiffusionGemma OpenAI shim", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    stream: bool | None = False


def _flatten_content(c):
    # Accept OpenAI content as str or [{type:text,text:...}] parts (text only).
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    parts = []
    for p in c:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


@app.get("/health")
def health():
    alive = backend.proc is not None and backend.proc.poll() is None
    return {"status": "ok" if alive else "down", **backend.meta}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": DG_MODEL_ID, "object": "model", "owned_by": "club-3090"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": _flatten_content(m.content)} for m in req.messages]
    try:
        result = backend.generate(
            messages,
            n_predict=req.max_tokens,
            temperature=req.temperature,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if os.environ.get("DG_DEBUG_RAW"):
        import sys
        print(f"[raw] {result.get('content','')!r}", file=sys.stderr, flush=True)
    content, reasoning = parse_diffusion_output(result.get("content", ""))

    # steps==0 means the backend never ran the denoiser — almost always the prompt + canvas overflowed
    # the launch-time ubatch (n_ctx == DG_CTX). Surface that as a clear 4xx instead of empty content.
    if result.get("steps", 0) == 0 and not content:
        raise HTTPException(
            status_code=413,
            detail=(f"prompt too long: [prompt | canvas] must fit one ubatch (DG_CTX={DG_CTX}). "
                    f"Reduce the prompt, or raise DG_CTX (watch VRAM — the non-causal forward holds "
                    f"the whole sequence on-GPU; ~22-23 GB at ctx 3072 on a 24 GB 3090)."),
        )
    prompt_text = "".join(m["content"] for m in messages)
    usage = {
        "prompt_tokens": est_tokens(prompt_text),
        "completion_tokens": est_tokens(content),
        "total_tokens": est_tokens(prompt_text) + est_tokens(content),
    }
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning  # surfaced for clients that read it

    if req.stream:
        def sse():
            head = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req.model or DG_MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(head)}\n\n"
            body = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req.model or DG_MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(body)}\n\n"
            tail = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": req.model or DG_MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(tail)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": req.model or DG_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": usage,
        "x_diffusion": {  # non-standard: surfaces the diffusion stats for benchmarking
            "steps": result.get("steps"),
            "blocks": result.get("blocks"),
            "time_ms": result.get("time_ms"),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=DG_HOST, port=DG_PORT, log_level="info")
