# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tool call parser for Google Gemma4 models.

Gemma4 uses a custom serialization format (not JSON) for tool calls::

    <|tool_call>call:func_name{key:<|"|>value<|"|>,num:42}<tool_call|>

Strings are delimited by ``<|"|>`` (token 52), keys are unquoted, and
multiple tool calls are concatenated without separators.

Used when ``--enable-auto-tool-choice --tool-call-parser gemma4`` are set.

For offline inference tool call parsing (direct ``tokenizer.decode()`` output),
see ``vllm.tool_parsers.gemma4_utils.parse_tool_calls``.

----------------------------------------------------------------------------
club-3090 overlay — malformed tool-call recovery + no-swallow fallback
----------------------------------------------------------------------------
Stock behaviour: the streaming parser only understands the canonical
``<|tool_call>call:name{...}<tool_call|>`` (brace) form. Under deep, degraded
agentic context (long multi-round tool loops, repeated tool errors)
DiffusionGemma sometimes emits the tool-call delimiters but with a *malformed*
body — Python-style parentheses and ``key='value'`` / ``key:{json}`` args,
e.g. ``<|tool_call>call:ha_list_entities(domain='media_player')<tool_call|>``.
The stock brace-only regex then matches nothing and ``_handle_tool_call_end``
returns ``None``, so the ENTIRE ``<|tool_call>...<tool_call|>`` block is
silently swallowed — the assistant turn ends with no content AND no tool call
(Hermes records it as ``"(empty)"`` and has to nudge/retry). Reproduced live
2026-06-19 from real Hermes ``state.db`` sessions; see
``patch_docs/diffusionGemma_empty_after_tools.md`` + ``docs/UPSTREAM.md``.

This overlay adds, WITHOUT touching the well-formed brace path:
  1. ``_recover_tool_calls`` — a lenient, delimiter-anchored recovery that
     parses the paren / python-arg form into real tool calls.
  2. a no-swallow fallback — if a ``<|tool_call>...<tool_call|>`` block cannot
     be parsed at all, surface its raw inner text as content instead of
     emitting nothing, so a turn is never silently empty.
  3. plain-quoted string values — the model also drifts toward JSON/python
     quotes *inside* the canonical brace body (``call:fn{domain: "media_player"}``)
     instead of the gemma4 ``<|"|>media_player<|"|>`` delimiters. The stock
     bare-value path kept the surrounding quotes, so every such value came out
     as the string ``"media_player"`` (with literal quotes) → Hermes errors like
     ``Invalid domain format: '"media_player"'``, quoted tool names, and
     ``'arguments' is not valid JSON``. ``_parse_gemma4_args`` /
     ``_parse_gemma4_array`` now scan plain-quoted values quote-aware (so
     embedded commas/braces don't truncate them) and strip the quotes;
     ``_parse_gemma4_value`` keeps a quote-strip safety net. The proper
     ``<|"|>`` delimiter path is unchanged (it's matched first).
Both the streaming (``_handle_tool_call_end``) and non-streaming
(``extract_tool_calls``) paths use the recovery only when the strict regex
finds zero matches, so canonical output is byte-for-byte unchanged.
"""

import ast
import json
from collections.abc import Sequence

import regex as re
from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import find_common_prefix

logger = init_logger(__name__)

# Gemma4 special tokens for tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'


# ---------------------------------------------------------------------------
# Gemma4 argument parser (used by both streaming and non-streaming paths)
# ---------------------------------------------------------------------------


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # club-3090: a value wrapped in matching plain quotes is a string — return
    # its contents (don't re-coerce). The model sometimes emits JSON/python
    # quotes (`key: "value"`) instead of the gemma4 `<|"|>value<|"|>` delimiters;
    # the bare-value path otherwise kept the quotes, so `domain: "media_player"`
    # leaked out as the string '"media_player"' (→ "Invalid domain format",
    # quoted tool names, etc.). Quote-aware scanning in _parse_gemma4_args /
    # _parse_gemma4_array handles the common path; this is a safety net for any
    # quoted value that still reaches here.
    if (
        len(value_str) >= 2
        and value_str[0] in "\"'"
        and value_str[-1] == value_str[0]
    ):
        return value_str[1:-1].replace(STRING_DELIM, "")

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Null
    if value_str.lower() in ("null", "none", "nil"):
        return None

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe).
    # club-3090: at depth the model also leaks the literal <|"|> delimiter
    # *inside* a value (e.g. `"<|"|>music-assistant-mcp<|"|>"`); the delimiter
    # is never legitimate content, so strip any occurrence.
    return value_str.replace(STRING_DELIM, "")


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        # Plain-quoted string value: "..." or '...'  (club-3090)
        # The model sometimes emits JSON/python-style quotes instead of the
        # gemma4 <|"|> delimiters inside a brace body. Scan quote-aware (so
        # embedded commas/braces don't truncate the value) and strip the
        # surrounding quotes — the stock bare-value path kept them, so a value
        # like `domain: "media_player"` leaked out as the string
        # '"media_player"' (→ "Invalid domain format", quoted tool names, etc.).
        elif args_str[i] in ("\"", "'"):
            quote_char = args_str[i]
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                ch = args_str[j]
                if ch == "\\" and j + 1 < n:
                    buf.append(args_str[j + 1])
                    j += 2
                    continue
                if ch == quote_char:
                    closed = True
                    j += 1
                    break
                buf.append(ch)
                j += 1
            if not closed and partial:
                # Incomplete quoted string mid-stream — withhold this key.
                break
            result[key] = "".join(buf).replace(STRING_DELIM, "")
            i = j

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 args parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            if partial:
                raw_val = args_str[val_start:i].strip()
                if raw_val.endswith("."):
                    # Trailing dot means decimal digits may still arrive
                    # (e.g. "108." may become "108.2"). Parsing now would
                    # yield float("108.") == 108.0, whose json repr "108.0"
                    # corrupts the streaming diff when the true digit lands.
                    break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        # String element
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        # Plain-quoted string element: "..." or '...'  (club-3090, see
        # _parse_gemma4_args — same quote-aware strip for array elements.)
        elif arr_str[i] in ("\"", "'"):
            quote_char = arr_str[i]
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                ch = arr_str[j]
                if ch == "\\" and j + 1 < n:
                    buf.append(arr_str[j + 1])
                    j += 2
                    continue
                if ch == quote_char:
                    closed = True
                    j += 1
                    break
                buf.append(ch)
                j += 1
            if not closed and partial:
                break
            items.append("".join(buf).replace(STRING_DELIM, ""))
            i = j

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                logger.warning(
                    "Gemma4 array parser made no progress at position %d; "
                    "aborting on malformed input.",
                    i,
                )
                break
            if partial:
                raw_val = arr_str[val_start:i].strip()
                if raw_val.endswith("."):
                    break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


# ---------------------------------------------------------------------------
# club-3090: lenient recovery for malformed (paren / python-arg) tool calls
# ---------------------------------------------------------------------------

# Matches the body BETWEEN the <|tool_call> ... <tool_call|> delimiters:
# "call:<name><rest>" where <rest> is "{...}" (canonical) or "(...)" (degraded).
_RECOVER_NAME_RE = re.compile(r"\s*call:\s*([\w\-\.]+)\s*(.*)$", re.DOTALL)


def _split_top_level_args(s: str) -> list[str]:
    """Split an argument string on top-level commas only.

    Respects ``()``/``{}``/``[]`` nesting and single/double quoted strings, so
    a dict/list value containing commas is kept intact.
    """
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    prev = ""
    buf: list[str] = []
    for ch in s:
        if quote is not None:
            buf.append(ch)
            if ch == quote and prev != "\\":
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch in "([{":
            depth += 1
            buf.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
        prev = ch
    if buf:
        parts.append("".join(buf))
    return parts


def _coerce_py_value(v: str) -> object:
    """Best-effort coerce a raw arg value (JSON → python-literal → string)."""
    v = v.strip()
    if not v:
        return v
    try:
        return json.loads(v)
    except Exception:
        pass
    try:
        return ast.literal_eval(v)
    except Exception:
        pass
    if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
        return v[1:-1]
    return v


def _parse_python_args(s: str) -> dict:
    """Parse a degraded ``(key='val', key2:{json}, ...)`` argument body.

    Handles the python-ish / mixed form DiffusionGemma emits when it degrades:
    ``key='value'``, ``key="value"``, ``key:value``, and ``key:{...}`` /
    ``key:[...]`` JSON values. Unkeyed/positional fragments are skipped.
    """
    s = s.strip()
    if not s:
        return {}
    out: dict = {}
    for part in _split_top_level_args(s):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"([\w\-\.]+)\s*[:=]\s*(.*)$", part, re.DOTALL)
        if not m:
            continue
        out[m.group(1)] = _coerce_py_value(m.group(2))
    return out


def _iter_tool_call_blocks(text: str) -> list[str]:
    """Return the raw body of every ``<|tool_call> ... <tool_call|>`` block.

    Anchored on the literal delimiters (not a body regex) so nested ``{}``/
    ``()`` in arguments never truncate the block.
    """
    bodies: list[str] = []
    idx = 0
    while True:
        s = text.find(TOOL_CALL_START, idx)
        if s == -1:
            break
        e = text.find(TOOL_CALL_END, s + len(TOOL_CALL_START))
        if e == -1:
            break
        bodies.append(text[s + len(TOOL_CALL_START) : e])
        idx = e + len(TOOL_CALL_END)
    return bodies


def _recover_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Lenient recovery of (name, args_dict) for every completed call block.

    Used ONLY as a fallback when the strict brace regex matched nothing, so it
    never alters canonical parsing. Handles both the brace body (via the
    standard Gemma4 arg parser) and the degraded paren body (via
    ``_parse_python_args``).
    """
    calls: list[tuple[str, dict]] = []
    for body in _iter_tool_call_blocks(text):
        m = _RECOVER_NAME_RE.match(body)
        if not m:
            continue
        name = m.group(1)
        rest = m.group(2).strip()
        args: dict = {}
        if rest.startswith("{"):
            inner = rest[1:]
            if inner.endswith("}"):
                inner = inner[:-1]
            args = _parse_gemma4_args(inner)
        elif rest.startswith("("):
            inner = rest[1:]
            if inner.endswith(")"):
                inner = inner[:-1]
            args = _parse_python_args(inner)
        # else: bare ``call:name`` with no args body → empty args
        calls.append((name, args))
    return calls


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma4 models.

    Handles the Gemma4 function call format::

        <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

    Used when ``--enable-auto-tool-choice --tool-call-parser gemma4``
    are set.

    Streaming strategy: **accumulate-then-parse-then-diff**

    Instead of trying to convert Gemma4's custom format to JSON
    token-by-token (which fails because Gemma4 uses bare keys, custom
    delimiters, and structural braces that differ from JSON), this parser:

    1. Accumulates the raw Gemma4 argument string during streaming
    2. Parses it with ``_parse_gemma4_args()`` into a Python dict
    3. Converts to JSON with ``json.dumps()``
    4. Diffs against the previously-streamed JSON string
    5. Emits only the new JSON fragment as the delta

    This follows the same pattern used by FunctionGemma, Hermes, and Llama
    tool parsers.
    """

    # Gemma4 emits native special-token tool calls, not generic JSON calls.
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        # Token strings
        self.tool_call_start_token = TOOL_CALL_START
        self.tool_call_end_token = TOOL_CALL_END

        # Token IDs
        self.tool_call_start_token_id = self.vocab.get(TOOL_CALL_START)
        self.tool_call_end_token_id = self.vocab.get(TOOL_CALL_END)

        if self.tool_call_start_token_id is None:
            raise RuntimeError(
                "Gemma4 ToolParser could not locate the tool call start "
                f"token '{TOOL_CALL_START}' in the tokenizer!"
            )

        # Regex for non-streaming: extract complete tool calls.
        # Supports function names with letters, digits, underscores,
        # hyphens, and dots (e.g. "get-weather", "module.func").
        self.tool_call_regex = re.compile(
            r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>",
            re.DOTALL,
        )

        # Streaming state — reset per-request via _reset_streaming_state()
        self._reset_streaming_state()

        # Delta buffer for handling multi-token special sequences
        self.buffered_delta_text = ""

    def _reset_streaming_state(self) -> None:
        """Reset all streaming state for a new request."""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tc = request.tool_choice
            if tc == "required" or isinstance(
                tc,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            ):
                # Do NOT call super().adjust_request() for required/named tool
                # choice. The base implementation injects a JSON-array
                # `structured_outputs` schema and forces xgrammar guided
                # decoding, which conflicts with Gemma4's native
                # `<|tool_call>call:...` (non-JSON) tool syntax and crashes
                # EngineCore under MTP spec decode. The streaming/extraction
                # parser already handles the native output, so guided decoding
                # is skipped here (mirrors the GLM4 precedent).
                if request.tool_choice != "none":
                    request.skip_special_tokens = False
                return request
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # Don't skip special tokens — <|tool_call> etc. are needed for
            # the parser to detect tool calls. Apply to BOTH
            # ChatCompletionRequest and ResponsesRequest (the previous
            # isinstance(ChatCompletionRequest) guard caused tool-call
            # delimiters to be stripped on /v1/responses, leaking raw
            # `call:fn{...}` text via output_text.delta).
            request.skip_special_tokens = False
        return request

    # ------------------------------------------------------------------
    # Delta buffering for multi-token special sequences
    # ------------------------------------------------------------------

    def _buffer_delta_text(self, delta_text: str) -> str:
        """Buffer incoming delta text to handle multi-token special sequences.

        Accumulates partial tokens that could be the start of
        ``<|tool_call>`` or ``<tool_call|>`` and only flushes them
        when the complete sequence is recognized or the sequence breaks.

        This prevents partial special tokens (e.g., ``<|tool``) from being
        emitted prematurely as content text.
        """
        combined = self.buffered_delta_text + delta_text

        # Check if combined ends with a complete special token
        if combined.endswith(TOOL_CALL_START) or combined.endswith(TOOL_CALL_END):
            self.buffered_delta_text = ""
            return combined

        # Check if combined ends with a partial prefix of a special token
        for tag in [TOOL_CALL_START, TOOL_CALL_END]:
            for i in range(1, len(tag)):
                if combined.endswith(tag[:i]):
                    self.buffered_delta_text = combined[-i:]
                    return combined[:-i]

        # No partial match — flush everything
        self.buffered_delta_text = ""
        return combined

    # ------------------------------------------------------------------
    # Non-streaming extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            matches = self.tool_call_regex.findall(model_output)
            if matches:
                parsed_calls = [
                    (func_name, _parse_gemma4_args(args_str))
                    for func_name, args_str in matches
                ]
            else:
                # ---- club-3090 recovery: strict (brace) regex matched nothing.
                # The block(s) are present but malformed (paren / python args).
                # Recover them rather than leaking raw `<|tool_call>...` as
                # content (and, in streaming, swallowing to an empty turn).
                parsed_calls = _recover_tool_calls(model_output)
                if not parsed_calls:
                    return ExtractedToolCallInformation(
                        tools_called=False, tool_calls=[], content=model_output
                    )
                logger.debug(
                    "Recovered %d malformed Gemma4 tool call(s) (non-streaming).",
                    len(parsed_calls),
                )

            tool_calls: list[ToolCall] = []
            for func_name, arguments in parsed_calls:
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(arguments, ensure_ascii=False),
                        ),
                    )
                )

            # Content = text before first tool call (if any)
            content_end = model_output.find(self.tool_call_start_token)
            content = model_output[:content_end].strip() if content_end > 0 else None

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error extracting tool calls from Gemma4 response")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # ------------------------------------------------------------------
    # Streaming extraction — accumulate-then-parse-then-diff
    # ------------------------------------------------------------------

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Buffer delta text to handle multi-token special sequences
        delta_text = self._buffer_delta_text(delta_text)
        # Keep current_text from the upstream stream state. The buffered delta
        # is only for emission, and must not be stitched back into the
        # accumulated model text or normal content like "<div>" can be
        # duplicated into "<<div>" when a tool call just ended.

        # If no tool call token seen yet, emit as content
        if self.tool_call_start_token not in current_text:
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        try:
            return self._extract_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
            )
        except Exception:
            logger.exception("Error in Gemma4 streaming tool call extraction")
            return None

    def _extract_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Tag-counting streaming parser.

        Uses the proven approach from FunctionGemma/Hermes: count start/end
        tags in previous vs current text to determine phase, then
        accumulate-parse-diff for arguments.

        Format: ``<|tool_call>call:name{args}<tool_call|>``
        """
        start_count = current_text.count(self.tool_call_start_token)
        end_count = current_text.count(self.tool_call_end_token)
        prev_start_count = previous_text.count(self.tool_call_start_token)
        prev_end_count = previous_text.count(self.tool_call_end_token)

        # Case 1: Not inside any tool call — emit as content
        if (
            start_count == end_count
            and prev_end_count == end_count
            and self.tool_call_end_token not in delta_text
        ):
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # Case 2: One or more new tool calls started in this delta.
        # A single delta can batch several complete calls, so advance the
        # tool id once per newly-seen start token and allocate a tracking
        # slot for each.
        if start_count > prev_start_count:
            num_new = start_count - prev_start_count
            for _ in range(num_new):
                self.current_tool_id += 1
                self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr.append({})
            self.current_tool_name_sent = False
            logger.debug(
                "Started %d new tool call(s); current_tool_id=%d",
                num_new,
                self.current_tool_id,
            )
            # Don't return yet if this delta also contains call payload or
            # the end marker; backends can batch one or more complete tool
            # calls into a single streaming chunk. Only wait for more text
            # when the delta is just the start token itself.
            if start_count > end_count and len(delta_text) <= len(
                self.tool_call_start_token
            ):
                return None

        # Case 3: One or more tool calls just ended (possibly several in a
        # single batched delta) — drain every newly-completed call.
        if end_count > prev_end_count:
            return self._handle_tool_call_end(
                current_text,
                prev_end_count=prev_end_count,
                end_count=end_count,
                start_count=start_count,
            )

        # Case 4: In the middle of a tool call — parse partial content
        if start_count > end_count:
            return self._handle_tool_call_middle(current_text)

        # Default: generate text outside tool calls
        if delta_text:
            text = delta_text.replace(self.tool_call_start_token, "")
            text = text.replace(self.tool_call_end_token, "")
            if text:
                return DeltaMessage(content=text)
        return None

    def _extract_partial_call(self, current_text: str) -> tuple[str | None, str]:
        """Extract function name and raw argument string from partial text.

        Returns (func_name, raw_args_str) or (None, "") if not parseable yet.
        """
        # Get the text after the last <|tool_call> token
        last_start = current_text.rfind(self.tool_call_start_token)
        if last_start == -1:
            return None, ""

        partial_call = current_text[last_start + len(self.tool_call_start_token) :]

        # Strip end token if present
        if self.tool_call_end_token in partial_call:
            partial_call = partial_call.split(self.tool_call_end_token)[0]

        # Expect "call:name{args...}" or "call:name{args...}"
        if not partial_call.startswith("call:"):
            return None, ""

        func_part = partial_call[5:]  # skip "call:"

        if "{" not in func_part:
            # Still accumulating function name, not ready yet
            return None, ""

        func_name, _, args_part = func_part.partition("{")
        func_name = func_name.strip()

        # Strip trailing '}' if present (Gemma4 structural brace)
        if args_part.endswith("}"):
            args_part = args_part[:-1]

        return func_name, args_part

    def _handle_tool_call_middle(self, current_text: str) -> DeltaMessage | None:
        """Handle streaming when we're inside an active tool call.

        Accumulates the raw Gemma4 arguments, parses them into JSON, and
        diffs against the previously-streamed JSON to emit only the new
        fragment.
        """
        func_name, args_part = self._extract_partial_call(current_text)

        if func_name is None:
            return None

        # Step 1: Send function name (once)
        if not self.current_tool_name_sent and func_name:
            self.current_tool_name_sent = True
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": func_name,
                "arguments": {},
            }
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name,
                            arguments="",
                        ).model_dump(exclude_none=True),
                    )
                ]
            )

        # Step 2: Parse and diff arguments
        if self.current_tool_name_sent and args_part:
            return self._emit_argument_diff(args_part)

        return None

    def _handle_tool_call_end(
        self,
        current_text: str,
        prev_end_count: int,
        end_count: int,
        start_count: int,
    ) -> DeltaMessage | None:
        """Handle streaming when one or more tool calls have just completed.

        A single streaming delta can batch several complete tool calls
        (``<|tool_call>...<tool_call|><|tool_call>...<tool_call|>``). Every
        call whose ``<tool_call|>`` end marker arrived in this delta — i.e.
        those with index in ``[prev_end_count, end_count)`` — is drained and
        emitted, with one ``DeltaToolCall`` per call in a single
        ``DeltaMessage`` (this matches the OpenAI streaming wire format, and
        the serving layer iterates over ``delta.tool_calls``).

        Per call:

        * If the function name was already streamed incrementally (the
          token-by-token path), only the remaining argument fragment is
          flushed as a diff.
        * If the call is seen complete for the first time in this delta (the
          batched-complete path), the id + name + full arguments JSON are
          emitted exactly once.
        """
        # Parse the complete tool calls using regex for accuracy.
        all_matches = self.tool_call_regex.findall(current_text)
        if all_matches:
            parsed_calls: list[tuple[str, dict]] = [
                (func_name, _parse_gemma4_args(args_str))
                for func_name, args_str in all_matches
            ]
        else:
            # ---- club-3090 recovery + no-swallow fallback ----
            # Strict (brace) regex matched nothing, but an end marker just
            # arrived — the model emitted a malformed/degraded block (paren or
            # python args). Recover it into a real tool call; if even that
            # fails, surface the raw inner text as content so the turn is never
            # silently empty (the "(empty)" after-tool-calls bug).
            parsed_calls = _recover_tool_calls(current_text)
            if not parsed_calls:
                logger.debug(
                    "Tool call end detected but unparseable; emitting raw "
                    "block as content (no-swallow fallback)."
                )
                return self._malformed_tool_call_fallback(
                    current_text, end_count=end_count, start_count=start_count
                )
            logger.debug(
                "Recovered %d malformed tool call(s) via lenient parser.",
                len(parsed_calls),
            )

        deltas: list[DeltaToolCall] = []
        for idx in range(prev_end_count, end_count):
            if idx >= len(parsed_calls):
                break
            # Ensure the tracking arrays have a slot for this index (defensive;
            # Case 2 normally allocates these when the start token arrives).
            while len(self.prev_tool_call_arr) <= idx:
                self.prev_tool_call_arr.append({})
                self.streamed_args_for_tool.append("")

            func_name, final_args = parsed_calls[idx]
            final_args_json = json.dumps(final_args, ensure_ascii=False)

            # The name is sent exactly once per call. We track that via the
            # per-call entry in prev_tool_call_arr (set either by the middle
            # path or by the batched-complete branch below), which is robust
            # even when several calls are drained in one delta.
            name_already_sent = bool(self.prev_tool_call_arr[idx].get("name"))

            if not name_already_sent:
                # Batched-complete call: emit id + name + full arguments once.
                self.streamed_args_for_tool[idx] = final_args_json
                self.prev_tool_call_arr[idx] = {
                    "name": func_name,
                    "arguments": final_args,
                }
                deltas.append(
                    DeltaToolCall(
                        index=idx,
                        type="function",
                        id=make_tool_call_id(),
                        function=DeltaFunctionCall(
                            name=func_name, arguments=final_args_json
                        ).model_dump(exclude_none=True),
                    )
                )
            else:
                # Incrementally-streamed call: flush the remaining argument
                # tail that was withheld during the middle phase.
                prev_streamed = self.streamed_args_for_tool[idx]
                if len(final_args_json) > len(prev_streamed):
                    diff = final_args_json[len(prev_streamed) :]
                    self.streamed_args_for_tool[idx] = final_args_json
                    self.prev_tool_call_arr[idx]["arguments"] = final_args
                    deltas.append(
                        DeltaToolCall(
                            index=idx,
                            function=DeltaFunctionCall(arguments=diff).model_dump(
                                exclude_none=True
                            ),
                        )
                    )

        # Advance streaming state past the calls completed in this delta. If a
        # further tool call is still being accumulated (start without a
        # matching end), point current_tool_id at it so the middle path can
        # stream its arguments next; otherwise settle on the last completed
        # call.
        if start_count > end_count:
            self.current_tool_id = end_count
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
                self.streamed_args_for_tool.append("")
            self.current_tool_name_sent = bool(
                self.prev_tool_call_arr[self.current_tool_id].get("name")
            )
        else:
            self.current_tool_id = end_count - 1
            self.current_tool_name_sent = True

        if deltas:
            return DeltaMessage(tool_calls=deltas)
        return None

    def _malformed_tool_call_fallback(
        self,
        current_text: str,
        end_count: int,
        start_count: int,
    ) -> DeltaMessage | None:
        """No-swallow fallback for an unparseable ``<|tool_call>`` block.

        club-3090 fix. When a ``<tool_call|>`` end marker arrives but neither
        the strict regex nor the lenient recovery can extract a call, the stock
        parser returned ``None`` — silently dropping the whole block and ending
        the turn with no content and no tool call (Hermes ``"(empty)"``). This
        instead surfaces the raw inner block text (delimiters stripped) as
        content, so the turn is never silently empty and the agent can act on
        it. The well-formed and recoverable paths never reach here.
        """
        first = current_text.find(self.tool_call_start_token)
        raw = current_text[first:] if first != -1 else ""
        fallback = raw.replace(self.tool_call_start_token, "").replace(
            self.tool_call_end_token, ""
        ).strip()

        # Advance streaming state past the unparseable block so it is not
        # reprocessed by later deltas (mirrors _handle_tool_call_end).
        if start_count > end_count:
            self.current_tool_id = end_count
            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
                self.streamed_args_for_tool.append("")
            self.current_tool_name_sent = bool(
                self.prev_tool_call_arr[self.current_tool_id].get("name")
            )
        else:
            self.current_tool_id = max(self.current_tool_id, end_count - 1)
            self.current_tool_name_sent = True

        if fallback:
            return DeltaMessage(content=fallback)
        return None

    def _emit_argument_diff(self, raw_args_str: str) -> DeltaMessage | None:
        """Parse raw Gemma4 arguments, convert to JSON, diff, and emit.

        This is the core of the accumulate-then-parse-then-diff strategy:
        1. Parse ``raw_args_str`` with ``_parse_gemma4_args()``
        2. Convert to JSON string with ``json.dumps()``
        3. Withhold trailing closing characters (``"}``) that may move
           as more tokens arrive
        4. Diff against previously streamed JSON and emit only new chars

        **Why withholding is necessary:**

        Gemma4's custom format produces *structurally incomplete* JSON
        during streaming. For example, when ``<|"|>Paris`` arrives
        without a closing delimiter, ``_parse_gemma4_args`` treats it
        as a complete value and produces ``{"location": "Paris"}``. But
        when ``, France<|"|>`` arrives next, the JSON becomes
        ``{"location": "Paris, France"}``. If we had sent the closing
        ``"}`` from the first parse, the concatenated client output
        would be ``{"location": "Paris"}France"}``, which is garbage.

        The solution: **never send trailing closing chars during
        streaming**. They get flushed by ``_handle_tool_call_end()``
        when the ``<tool_call|>`` end marker arrives.

        Args:
            raw_args_str: The raw Gemma4 argument text accumulated so far
                (without the surrounding ``{`` ``}``).

        Returns:
            DeltaMessage with the argument diff, or None if no new content.
        """
        try:
            current_args = _parse_gemma4_args(raw_args_str, partial=True)
        except Exception:
            logger.debug(
                "Could not parse partial Gemma4 args yet: %s",
                raw_args_str[:100],
            )
            return None

        if not current_args:
            return None

        current_args_json = json.dumps(current_args, ensure_ascii=False)

        # Withhold trailing closing characters that may shift as more
        # tokens arrive. Strip trailing '}', '"', ']' and partial
        # STRING_DELIM fragments ('<', '|', '\\', '>') to get the
        # "safe prefix".
        safe_json = current_args_json
        while safe_json and safe_json[-1] in ("}", '"', "]", "<", "|", "\\", ">"):
            safe_json = safe_json[:-1]

        prev_streamed = self.streamed_args_for_tool[self.current_tool_id]

        if not safe_json or safe_json == prev_streamed:
            return None

        # Use find_common_prefix to handle cases where the value changed
        # structurally (e.g., a string grew).
        if prev_streamed:
            prefix = find_common_prefix(prev_streamed, safe_json)
            sent_len = len(prev_streamed)
            prefix_len = len(prefix)

            if prefix_len < sent_len:
                # Structure changed — we sent too much. Truncate our
                # tracking to the common prefix and wait for the final
                # flush in _handle_tool_call_end.
                self.streamed_args_for_tool[self.current_tool_id] = prefix
                return None

            # Stream the new stable portion
            diff = safe_json[sent_len:]
        else:
            # First emission
            diff = safe_json

        if diff:
            self.streamed_args_for_tool[self.current_tool_id] = safe_json
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = current_args

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        function=DeltaFunctionCall(arguments=diff).model_dump(
                            exclude_none=True
                        ),
                    )
                ]
            )

        return None
