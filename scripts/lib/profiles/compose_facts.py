"""Compose-file facts — what a compose mechanically states about itself.

Lifted out of the cockpit (#1202 P1). It used to live in
``club3090_cockpit.data``, which made it unreachable from the CLI: the local
model layer was write-only from the UI *because the UI owned the only
implementation* of compose->spec derivation (#1153).

Regex over text, stdlib only — matching ``serve_override_defaults``, which must
stay importable on the launcher's no-PyYAML path. No repo imports, so both the
cockpit and ``scripts/`` can consume it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ComposeFacts:
    """What a compose file mechanically tells us (#1153 Route-K).

    A user who already has a working compose is the most common BYOM position,
    and until now the funnel had no door for them: ① takes an HF repo, and the
    only compose that could enter was one c3 itself emitted. This is the read
    half — everything a compose CAN state about itself. Arch dims are absent by
    construction (a compose does not carry hidden_size); those come from the
    weights, or stay as required inline edits in ⑤.

    Regex over text, stdlib only — matching serve_override_defaults, which must
    stay importable on the launcher's no-PyYAML path."""

    path: str = ""
    ok: bool = False
    error: str = ""
    image: str = ""
    engine: str = ""          # vllm | llama-cpp | ik-llama | unknown
    model_path: str = ""      # --model / -m / GGUF_FILE
    served_name: str = ""     # --served-model-name / -a
    port: str = ""            # ${PORT:-N} or a ports: mapping
    max_ctx: str = ""         # --max-model-len / -c
    kv_dtype: str = ""        # --kv-cache-dtype / -ctk
    tp: str = ""              # --tensor-parallel-size / -ts / device count
    status_header: str = ""   # the profile-header Status: line, when present
    service: str = ""


_ENGINE_BY_IMAGE = (
    ("vllm", "vllm"),
    ("llamacpp-club3090", "llama-cpp"),
    ("llama.cpp", "llama-cpp"),
    ("llama-cpp", "llama-cpp"),
    ("ik-llama", "ik-llama"),
    ("ik_llama", "ik-llama"),
)


def derive_compose_facts(text: str, path: str = "") -> ComposeFacts:
    """Read a user-supplied compose (#1153 Route-K). Never raises.

    Flags are read from a TOKEN stream, not by regex-guessing around each name:
    a compose states its command either as a YAML list (``- --model`` / ``- val``
    on separate lines) or as one folded string (``>- -m x -c 65536``), and a
    per-flag regex gets the list form wrong — it captures the next line's ``-``.
    Flatten first, then walk pairs."""
    import re as _re

    f = ComposeFacts(path=path)
    if not (text or "").strip():
        f.error = "empty compose"
        return f

    f.image = ""
    m = _re.search(r"^\s*image:\s*(\S+)", text, _re.M)
    if m:
        f.image = m.group(1).strip().strip('"').strip("'")
    low = f.image.lower()
    for needle, eng in _ENGINE_BY_IMAGE:
        if needle in low:
            f.engine = eng
            break
    else:
        f.engine = "unknown" if f.image else ""

    m = _re.search(r"^services:\s*\n\s{2,}([A-Za-z0-9_.-]+):", text, _re.M)
    if m:
        f.service = m.group(1)

    # ── token stream ────────────────────────────────────────────────────────
    toks: list[str] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("- "):
            ln = ln[2:].strip()
        elif ln == "-":
            continue
        # exec form: command: ["--model=/w/x", "--max-model-len=32768"]
        if ln.startswith(("command:", "entrypoint:")) and "[" in ln:
            ln = ln[ln.index("[") + 1:]
        ln = ln.replace("[", " ").replace("]", " ").replace(",", " ")
        ln = ln.strip('"').strip("'")
        if not ln or ln.endswith(":"):
            continue
        toks.extend(t.strip('"').strip("'") for t in ln.split() if t.strip('"').strip("'"))

    # YAML block-scalar introducers. A flag whose "value" is one of these did not
    # get a value at all — the next line starts a literal block.
    _BLOCK_SCALARS = {"|", ">", "|-", ">-", "|+", ">+"}
    # A short flag directly after a shell is the SHELL's flag, not the engine's:
    # `entrypoint: [bash, -c, |...]` is the standard vLLM compose shape, and its
    # bash -c was being read as llama.cpp's -c (ctx-size), so max_ctx came back
    # as "|" for every such compose.
    _SHELLS = {"bash", "sh", "zsh", "/bin/bash", "/bin/sh"}

    def flag(*names: str) -> str:
        # Alias PRIORITY, not token order: callers list the canonical name first
        # (e.g. "--max-model-len" before "-c"), so an unrelated short flag
        # appearing earlier in the file must not win over the real one.
        for name in names:
            for i, t in enumerate(toks):
                base = t.split("=", 1)[0]
                if base != name:
                    continue
                if i > 0 and toks[i - 1] in _SHELLS and not name.startswith("--"):
                    continue
                if "=" in t:                      # --flag=value
                    v = t.split("=", 1)[1]
                elif i + 1 < len(toks):           # --flag value
                    v = toks[i + 1]
                else:
                    continue
                v = v.strip().strip('"').strip("'")
                if v and not v.startswith("-") and v not in _BLOCK_SCALARS:
                    return v
        return ""

    f.model_path  = flag("--model", "-m", "GGUF_FILE")
    f.served_name = flag("--served-model-name", "-a", "--alias")
    f.max_ctx     = flag("--max-model-len", "-c", "--ctx-size")
    f.kv_dtype    = flag("--kv-cache-dtype", "-ctk", "--cache-type-k")
    f.tp          = flag("--tensor-parallel-size", "-tp", "-ts")
    if not f.tp:
        m = _re.search(r"CUDA_VISIBLE_DEVICES[=:\s]+\"?([0-9,]+)", text)
        if m:
            f.tp = str(len([x for x in m.group(1).split(",") if x.strip()]))

    m = (_re.search(r"\$\{PORT:-([0-9]+)\}", text)
         or _re.search(r"^\s*-\s*\"?([0-9]{2,5}):[0-9]+", text, _re.M))
    if m:
        f.port = m.group(1)

    m = _re.search(r"^#\s*Status:\s*(.+)$", text, _re.M)
    if m:
        f.status_header = m.group(1).strip()

    if not f.image:
        f.error = "no image: found — is this a compose file?"
        return f
    f.ok = True
    return f
