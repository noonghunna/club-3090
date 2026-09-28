#!/usr/bin/env bash
# test-qwen38-template — the vendored Qwen3.8 chat template keeps each club-3090
# change it carries, rendered the way the engines render it.
#
# WHY THIS TEST EXISTS
# --------------------
# models/qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/chat_template.jinja
# is mounted over the checkpoint's own template on every Qwen3.8 and ThinkingCap
# slug, vLLM and SGLang. Its changes (PROVENANCE.md) each fix a failure that
# looks like a healthy server: a Claude-API client's `high` effort 500'd (and a
# Hermes-style `minimal` / `max` 400'd), SGLang
# ignored the request's effort, a reordered tool schema re-prefilled the whole
# conversation, and Claude Code's mid-conversation system message 400'd every
# request on vLLM (#1447). A re-vendor from upstream silently drops all four. This
# renders the file with the same Jinja setup transformers uses (sandboxed,
# trim_blocks + lstrip_blocks, tojson with sort_keys) and checks each one.
# Needs python3 with jinja2; prints SKIP without it.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TPL="$ROOT/models/qwen3.8-27b/vllm/patches/qwen38-reasoning-effort-template/chat_template.jinja"

if ! python3 -c 'import jinja2' 2>/dev/null; then
  echo "SKIP: test-qwen38-template needs python3 with jinja2"
  exit 0
fi

python3 - "$TPL" <<'PY'
import json, sys
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(sys.argv[1], encoding="utf-8").read())

def render(messages, tools=None, gen=True, **kw):
    return tpl.render(messages=messages, tools=tools, add_generation_prompt=gen, bos_token="", eos_token="", **kw)

def raises(messages, tools=None, **kw):
    try:
        render(messages, tools, **kw)
    except jinja2.exceptions.TemplateError as e:
        return str(e)
    return None

TOOLS = [{"type": "function", "function": {"name": "Read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
fails = []
def check(name, ok):
    if not ok:
        fails.append(name)

# 1. a later system message renders where it sits (Claude Code's first request, #1447)
cc = [{"role": "system", "content": "You are Claude Code."},
      {"role": "user", "content": "Find the txt files."},
      {"role": "system", "content": "# Environment\nPlatform: linux"}]
err = raises(cc, TOOLS)
check(f"a system message after the first must render, not raise (got: {err})", err is None)
if err is None:
    out = render(cc, TOOLS)
    check("the later system message must render in place, after the user turn, as its own system turn",
          "<|im_start|>user\nFind the txt files.<|im_end|>\n<|im_start|>system\n# Environment\nPlatform: linux<|im_end|>\n<|im_start|>assistant\n" in out)
    check("the later system message must not be hoisted into the leading system turn",
          out.count("# Environment") == 1 and out.index("# Environment") > out.index("Find the txt files."))

# 2. append-only: each turn's prompt is a prefix of the next turn's
conv = cc + [{"role": "assistant", "content": "", "reasoning_content": "look",
              "tool_calls": [{"type": "function", "function": {"name": "Read", "arguments": {"path": "a.txt"}}}]},
             {"role": "tool", "content": "alpha"},
             {"role": "assistant", "content": "alpha", "reasoning_content": "done"},
             {"role": "system", "content": "reminder"},
             {"role": "user", "content": "again"}]
if err is None:
    for k in range(3, len(conv)):
        a, b = render(conv[:k], TOOLS, gen=False), render(conv[:k + 1], TOOLS, gen=False)
        check(f"turn {k}: the prompt must stay append-only across a later system message", b.startswith(a))

# 3. a later system message keeps the first one's content rules
e = raises([{"role": "user", "content": "q"}, {"role": "system", "content": [{"type": "image", "image": "x"}]}])
check(f"an image in a later system message must still be rejected (got: {e})", e == "System message cannot contain images.")
check("an empty later system message renders nothing",
      raises([{"role": "user", "content": "q"}, {"role": "system", "content": "  "}]) is None
      and render([{"role": "user", "content": "q"}, {"role": "system", "content": "  "}]) == render([{"role": "user", "content": "q"}]))

# 4. the earlier changes are still there
hi, xh = render([{"role": "user", "content": "q"}], reasoning_effort="high"), render([{"role": "user", "content": "q"}], reasoning_effort="xhigh")
check("reasoning_effort 'high' must render as 'xhigh' (Claude-API clients send high)", hi == xh)
check("an unknown effort must still be rejected", raises([{"role": "user", "content": "q"}], reasoning_effort="bogus") is not None)
lo = render([{"role": "user", "content": "q"}], reasoning_effort="low")
check("reasoning_effort 'minimal' must render as 'low' (Hermes / OpenAI-compatible clients send it)",
      render([{"role": "user", "content": "q"}], reasoning_effort="minimal") == lo)
check("reasoning_effort 'max' must render as 'xhigh' (Hermes clamps its `ultra` to max)",
      render([{"role": "user", "content": "q"}], reasoning_effort="max") == xh)
dflt = render([{"role": "user", "content": "q"}], default_reasoning_effort="low")
check("default_reasoning_effort must apply when the request names no effort (SGLang, sglang#38104)",
      dflt == render([{"role": "user", "content": "q"}], reasoning_effort="low") != xh)
check("the request's effort must win over default_reasoning_effort",
      render([{"role": "user", "content": "q"}], reasoning_effort="medium", default_reasoning_effort="low")
      == render([{"role": "user", "content": "q"}], reasoning_effort="medium"))
shuffled = [{"function": {"parameters": {"required": ["path"], "properties": {"path": {"type": "string"}}, "type": "object"},
                          "description": "Read a file", "name": "Read"}, "type": "function"}]
check("tool schemas must render with sorted keys, whatever order the client sends",
      render([{"role": "user", "content": "q"}], TOOLS) == render([{"role": "user", "content": "q"}], shuffled))

for f in fails:
    print(f"✗ {f}", file=sys.stderr)
sys.exit(1 if fails else 0)
PY
rc=$?

# 5. every vLLM compose that mounts this template also NAMES it with --chat-template.
# vLLM v0.30.0's Anthropic /v1/messages decides whether the template can take a
# mid-conversation system message by testing only that flag; unset, it moves every
# such message to the front, and Claude Code (one per turn) re-prefills the whole
# conversation every turn (vllm#58754). The mount alone serves every other endpoint
# correctly, so a compose copied from an older sibling would look fine everywhere
# else. SGLang tests the loaded template and is not checked here.
python3 - "$ROOT" <<'PY' || rc=1
import pathlib, re, sys
root = pathlib.Path(sys.argv[1])
mount = re.compile(r"qwen38-reasoning-effort-template/chat_template\.jinja:([^:\s]+):ro")
checked, bad = 0, []
for f in sorted(root.glob("models/*/vllm/compose/**/*.yml")):
    s = f.read_text(encoding="utf-8")
    m = mount.search(s)
    if not m:
        continue
    checked += 1
    if not re.search(r"\n\s*- --chat-template\n\s*- " + re.escape(m.group(1)) + r"\n", s):
        bad.append(f"{f.relative_to(root)}: mounts {m.group(1)} but does not pass it as --chat-template")
for b in bad:
    print(f"✗ {b}", file=sys.stderr)
if checked == 0:
    print("✗ no vLLM compose mounts the template — the glob or the mount pattern is stale", file=sys.stderr)
sys.exit(1 if bad or checked == 0 else 0)
PY

[ "$rc" -eq 0 ] && echo "test-qwen38-template: ok (later system message in place + append-only, image rule kept, high/max→xhigh, minimal→low, default_reasoning_effort, sorted tool keys, vLLM composes pass --chat-template)"
exit "$rc"
