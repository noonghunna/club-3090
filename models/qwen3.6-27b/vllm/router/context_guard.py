from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from transformers import AutoTokenizer


DEFAULT_TOKENIZER = "/model"
DEFAULT_MAX_CTX = 262144
DEFAULT_MAX_TOKENS = 16384
SERVER_HEADROOM_TOKENS = 20
COMPLETION_CALL_TYPES = {
    "completion",
    "acompletion",
    "text_completion",
    "atext_completion",
}
VLLM_EXTRA_BODY_PARAMS = (
    "chat_template_kwargs",
    "guided_choice",
    "guided_grammar",
    "guided_json",
    "guided_regex",
    "include_stop_str_in_output",
    "ignore_eos",
    "min_p",
    "min_tokens",
    "repetition_penalty",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "structured_outputs",
    "top_k",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _is_completion_call(call_type: Any) -> bool:
    return getattr(call_type, "value", call_type) in COMPLETION_CALL_TYPES


def _messages_for_template(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    templated: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail="each message must be an object")
        role = message.get("role")
        if not isinstance(role, str):
            raise HTTPException(status_code=400, detail="each message must have a string role")
        templated.append(
            {
                "role": role,
                "content": _text_from_content(message.get("content", "")),
            }
        )
    return templated


def _move_vllm_params_to_extra_body(data: dict[str, Any]) -> None:
    extra_body = data.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    moved = False
    for key in VLLM_EXTRA_BODY_PARAMS:
        if key in data:
            extra_body[key] = data.pop(key)
            moved = True

    if moved:
        data["extra_body"] = extra_body


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _set_value(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
        return
    try:
        setattr(obj, key, value)
    except Exception:
        try:
            obj[key] = value
        except Exception:
            pass


def _normalize_message_reasoning(message: Any) -> None:
    if message is None or _get_value(message, "reasoning"):
        return

    reasoning = _get_value(message, "reasoning_content")
    provider_fields = _get_value(message, "provider_specific_fields")
    if not reasoning and provider_fields is not None:
        reasoning = _get_value(provider_fields, "reasoning") or _get_value(
            provider_fields,
            "reasoning_content",
        )

    if reasoning:
        _set_value(message, "reasoning", reasoning)


def _normalize_response_reasoning(response: Any) -> None:
    choices = _get_value(response, "choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        _normalize_message_reasoning(_get_value(choice, "message"))
        _normalize_message_reasoning(_get_value(choice, "delta"))


def _merge_top_level_extra_body(kwargs: dict[str, Any]) -> None:
    extra_body = kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        return

    optional_params = kwargs.get("optional_params")
    if not isinstance(optional_params, dict):
        optional_params = {}
        kwargs["optional_params"] = optional_params

    merged = optional_params.get("extra_body")
    if not isinstance(merged, dict):
        merged = {}
    merged.update(extra_body)
    optional_params["extra_body"] = merged


class Club3090ContextGuard(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer_id = os.environ.get("CLUB3090_TOKENIZER", DEFAULT_TOKENIZER)
        self.max_ctx = _env_int("CLUB3090_MAX_CTX", DEFAULT_MAX_CTX)
        self.headroom_tokens = _env_int("CLUB3090_SERVER_HEADROOM_TOKENS", SERVER_HEADROOM_TOKENS)
        self.default_max_tokens = _env_int("CLUB3090_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        self._tokenizer: Any | None = None
        self._tokenizer_lock = threading.Lock()

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.tokenizer_id,
                        trust_remote_code=True,
                    )
        return self._tokenizer

    def _prompt_tokens(self, data: dict[str, Any]) -> int:
        if "messages" in data:
            messages = _messages_for_template(data.get("messages"))
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return len(self.tokenizer.encode(prompt, add_special_tokens=False))

        prompt = data.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(str(part) for part in prompt)
        return len(self.tokenizer.encode(str(prompt), add_special_tokens=False))

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any] | None:
        if not _is_completion_call(call_type):
            return data

        _move_vllm_params_to_extra_body(data)

        requested_max_tokens = data.get("max_tokens", self.default_max_tokens)
        if requested_max_tokens is None:
            requested_max_tokens = self.default_max_tokens
        try:
            max_tokens = int(requested_max_tokens)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="max_tokens must be an integer") from exc
        if max_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be at least 1")

        prompt_tokens = self._prompt_tokens(data)
        total_tokens = prompt_tokens + max_tokens + self.headroom_tokens
        if total_tokens > self.max_ctx:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Request exceeds club-3090 router context window: "
                    f"prompt_tokens={prompt_tokens}, max_tokens={max_tokens}, "
                    f"headroom_tokens={self.headroom_tokens}, max_ctx={self.max_ctx}"
                ),
            )

        data["max_tokens"] = max_tokens
        metadata = data.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            data["metadata"] = metadata
        metadata["club3090_context"] = {
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "headroom_tokens": self.headroom_tokens,
            "max_ctx": self.max_ctx,
        }
        return data

    async def async_pre_call_deployment_hook(
        self,
        kwargs: dict[str, Any],
        call_type: Any,
    ) -> dict[str, Any] | None:
        if not _is_completion_call(call_type):
            return kwargs

        _move_vllm_params_to_extra_body(kwargs)
        optional_params = kwargs.get("optional_params")
        if isinstance(optional_params, dict):
            _move_vllm_params_to_extra_body(optional_params)
        _merge_top_level_extra_body(kwargs)
        return kwargs

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict[str, Any],
        response: Any,
        call_type: Any,
    ) -> Any:
        if _is_completion_call(call_type):
            _normalize_response_reasoning(response)
        return response

    async def async_post_call_streaming_deployment_hook(
        self,
        request_data: dict[str, Any],
        response_chunk: Any,
        call_type: Any,
    ) -> Any:
        if _is_completion_call(call_type):
            _normalize_response_reasoning(response_chunk)
        return response_chunk


proxy_handler_instance = Club3090ContextGuard()
