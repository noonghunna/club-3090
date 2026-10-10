#!/usr/bin/env python3
"""Container half of selftest.sh — runs INSIDE the pinned SGLang image, no GPU.

Order matters: the stock-behaviour control runs before anything is patched, the real install
runs twice, and the glue scenarios import the PATCHED modules in a fresh interpreter.

  1. control      stock ChatCompletionRequest DROPS a top-level thinking_token_budget
  2. negatives    on copies of the three files: anchor removed / duplicated -> refuse, tree
                  untouched; half-patched tree -> refuse; malformed map -> install refuses
                  (positive control: the same copy with a valid map installs)
  3. install x2   the real tree: first run patches, second is a byte-for-byte no-op;
                  install.sh --verify passes; the AST checks pass
  4. scenarios    real ChatCompletionRequest / AnthropicMessagesRequest objects through the
                  REAL patched _convert_to_internal_request and _convert_to_chat_completion_request;
                  the serving objects around them are stubs (no tokenizer, no GPU), and the stub
                  _process_messages replicates v0.5.21's server-default merge (setdefault) and
                  require_reasoning = (enable_thinking is not False)
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALL = HERE / "install.sh"
PATCHER = HERE / "patch_sglang.py"
FILES = ("srt/entrypoints/openai/protocol.py", "srt/entrypoints/openai/serving_chat.py",
         "srt/entrypoints/anthropic/serving.py")
MAP = '{"high":256,"low":64,"max":256,"medium":128,"minimal":64,"xhigh":256}'
fails = 0


def ok(cond: bool, msg: str) -> None:
    global fails
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    fails += not cond


def run(cmd, env=None) -> subprocess.CompletedProcess:
    full_env = {**os.environ, **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=full_env)


def sha(paths) -> dict:
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def pkg_dir() -> Path:
    spec = importlib.util.find_spec("sglang")
    return Path(list(spec.submodule_search_locations)[0])


def copy_tree(dst: Path) -> Path:
    for rel in FILES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkg_dir() / rel, dst / rel)
    return dst


# --------------------------------------------------------------------------- 1. control
CONTROL = r'''
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest as C
r = C(messages=[{"role": "user", "content": "hi"}], thinking_token_budget=7)
print("KEPT" if getattr(r, "thinking_token_budget", None) == 7 else "DROPPED")
'''


def control_stock() -> None:
    print("--- 1. control: stock SGLang drops thinking_token_budget ---", flush=True)
    st = run([sys.executable, str(PATCHER), "state", "--root", str(pkg_dir())]).stdout.strip()
    ok(st == "absent", f"the image's tree starts unpatched (state={st})")
    out = run([sys.executable, "-c", CONTROL])
    ok(out.stdout.strip().endswith("DROPPED"),
       f"stock ChatCompletionRequest drops the unknown key ({out.stdout.strip()[-20:] or out.stderr[-200:]})")


# --------------------------------------------------------------------------- 2. negatives
def negatives() -> None:
    print("--- 2. negative controls (copies of the three files) ---", flush=True)
    import patch_sglang as ps  # noqa: E402  (HERE is on sys.path)

    for edit in ps.EDITS:
        for how in ("removed", "duplicated"):
            with tempfile.TemporaryDirectory() as td:
                root = copy_tree(Path(td))
                f = root / edit.path
                text = f.read_text(encoding="utf-8")
                bad = text.replace(edit.anchor, "") if how == "removed" else text.replace(edit.anchor, edit.anchor * 2, 1)
                f.write_text(bad, encoding="utf-8")
                before = sha(root / r for r in FILES)
                r = run([sys.executable, str(PATCHER), "apply", "--root", str(root)])
                named = edit.name in r.stderr and "Re-anchoring" in r.stderr
                ok(r.returncode != 0 and named and sha(root / x for x in FILES) == before,
                   f"{edit.name} anchor {how}: refuses, names it + README, writes nothing (rc={r.returncode})")

    with tempfile.TemporaryDirectory() as td:
        root = copy_tree(Path(td))
        r1 = run([sys.executable, str(PATCHER), "apply", "--root", str(root)])
        shutil.copy2(pkg_dir() / FILES[1], root / FILES[1])          # un-patch one file
        r2 = run([sys.executable, str(PATCHER), "apply", "--root", str(root)])
        ok(r1.returncode == 0 and r2.returncode != 0 and "partial" in r2.stderr,
           f"half-patched tree: refuses (rc={r2.returncode}: {r2.stderr.strip()[:90]})")

    with tempfile.TemporaryDirectory() as td:
        root, site = copy_tree(Path(td) / "pkg"), Path(td) / "site"
        site.mkdir()
        env = {"CLUB3090_SGLANG_PKG_DIR": str(root), "CLUB3090_SITE_DIR": str(site)}
        bad = run(["bash", str(INSTALL)], {**env, "CLUB3090_REASONING_EFFORT_BUDGETS": '{"low": "lots"}'})
        good = run(["bash", str(INSTALL)], {**env, "CLUB3090_REASONING_EFFORT_BUDGETS": MAP})
        untouched = run([sys.executable, str(PATCHER), "state", "--root", str(root)])
        ok(bad.returncode != 0 and "does not load" in bad.stderr,
           f"malformed map: install refuses before patching (rc={bad.returncode})")
        ok(good.returncode == 0 and good.stdout.startswith("[sglang-effort-thinking-budget] applied")
           and "map: low=64,medium=128,xhigh=256" in good.stdout and untouched.stdout.strip() == "applied",
           f"positive control, same copy + a valid map: installs ({good.stdout.strip() or good.stderr.strip()[:120]})")
        nomod = run(["bash", str(INSTALL)], {**env, "CLUB3090_EFFORT_BUDGET_PY": "/nonexistent/effort_budget.py"})
        ok(nomod.returncode != 0 and "is missing" in nomod.stderr, "effort_budget.py not mounted: install refuses")


# --------------------------------------------------------------------------- 3. install x2
def install_twice() -> None:
    print("--- 3. install twice on the image's real tree ---", flush=True)
    import sysconfig
    site = Path(sysconfig.get_paths()["purelib"])
    watched = [pkg_dir() / r for r in FILES]
    first = run(["bash", str(INSTALL)], {"CLUB3090_REASONING_EFFORT_BUDGETS": MAP})
    print("    " + (first.stdout.strip() or first.stderr.strip()), flush=True)
    ok(first.returncode == 0 and "hooks: patched 3/3" in first.stdout and "modules: 2 refreshed" in first.stdout,
       "first run patches the 3 call sites and installs both modules")
    watched += [site / "club3090_effort_budget.py", site / "club3090_sglang_effort_budget.py"]
    before = sha(watched)
    second = run(["bash", str(INSTALL)], {"CLUB3090_REASONING_EFFORT_BUDGETS": MAP})
    print("    " + (second.stdout.strip() or second.stderr.strip()), flush=True)
    ok(second.returncode == 0 and "hooks: already in place" in second.stdout and "modules: unchanged" in second.stdout
       and sha(watched) == before, "second run is a no-op (5 files byte-identical)")
    lines = [l for l in second.stdout.splitlines() if l.strip()]
    ok(len(lines) == 1, f"install prints exactly one line ({len(lines)})")
    v = run(["bash", str(INSTALL), "--verify"], {"CLUB3090_REASONING_EFFORT_BUDGETS": MAP})
    ok(v.returncode == 0, f"install.sh --verify: {v.stdout.strip().splitlines()[-1:] or v.stderr.strip()}")
    off = run(["bash", str(INSTALL)], {"CLUB3090_REASONING_EFFORT_BUDGETS": ""})
    ok(off.returncode == 0 and "map: off" in off.stdout, "THINKING_BUDGETS=off (map unset): still installs, map: off")


# --------------------------------------------------------------------------- 4. scenarios
SCENARIOS = r'''
import logging, sys, types
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, MessageProcessingResult
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
import club3090_sglang_effort_budget as glue

fails = 0
def ok(cond, msg):
    global fails
    print(("  ✓ " if cond else "  ✗ ") + msg, flush=True)
    fails += not cond

SERVER = {"enable_thinking": True, "default_reasoning_effort": "low"}   # what our composes pass

class StubChat:
    """The serving object around the REAL _convert_to_internal_request (no tokenizer, no GPU)."""
    reasoning_parser = "qwen3"; is_gpt_oss = False; chat_encoding_spec = None
    default_sampling_params = {}
    def __init__(self, defaults):
        self.default_chat_template_kwargs = defaults
        self.template_manager = types.SimpleNamespace(reasoning_config=None)
        self.tokenizer_manager = types.SimpleNamespace(model_config=types.SimpleNamespace(is_multimodal=False))
    def _process_messages(self, request, is_multimodal):
        # replicates v0.5.21 _process_messages' server-default merge (setdefault) ...
        if self.default_chat_template_kwargs:
            ctk = dict(request.chat_template_kwargs or {})
            for k, v in self.default_chat_template_kwargs.items():
                ctk.setdefault(k, v)
            request.chat_template_kwargs = ctk
            if ctk.get("reasoning_effort") is not None and request.reasoning_effort is None:
                request.reasoning_effort = ctk["reasoning_effort"]
        # ... and qwen3's thinking switch
        thinking = (request.chat_template_kwargs or {}).get("enable_thinking") is not False
        return MessageProcessingResult(prompt="", prompt_ids=[1, 2, 3], image_data=None, audio_data=None,
                                       video_data=None, modalities=[], stop=[], require_reasoning=thinking)
    def _engine_prompt(self, pm, is_multimodal): return "input_ids", pm.prompt_ids
    def extract_custom_labels(self, raw): return None
    def extract_routed_dp_rank_from_header(self, raw, rank): return rank
    def _resolve_lora_path(self, model, lora_path): return lora_path
    def _should_return_input_ids(self, request): return False
    def extract_routing_key(self, raw): return None
    def apply_reasoning_enabled(self, request, enabled):          # stub of the qwen3 toggle path
        request.chat_template_kwargs = {**(request.chat_template_kwargs or {}), "enable_thinking": enabled}
    def supports_native_reasoning_history(self): return True
    def wrap_reasoning_history(self, *a, **k): return None

def chat(defaults=SERVER, **body):
    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **body)
    adapted, _ = OpenAIServingChat._convert_to_internal_request(StubChat(defaults), req, None)
    return (adapted.sampling_params.get("custom_params") or {}).get("thinking_budget"), adapted, req

def messages(defaults=SERVER, **body):
    anth = AnthropicServing.__new__(AnthropicServing)
    anth.openai_serving_chat = StubChat(defaults)
    anth._merge_inline_system = False
    areq = AnthropicMessagesRequest(model="m", max_tokens=4096, messages=[{"role": "user", "content": "hi"}], **body)
    creq = AnthropicServing._convert_to_chat_completion_request(anth, areq)
    adapted, _ = OpenAIServingChat._convert_to_internal_request(anth.openai_serving_chat, creq, None)
    return (adapted.sampling_params.get("custom_params") or {}).get("thinking_budget")

glue.reload({"CLUB3090_REASONING_EFFORT_BUDGETS": sys.argv[1]})
b, adapted, req = chat(reasoning_effort="medium")
ok(b == 128 and type(b) is int, f"top-level effort medium -> 128 (got {b!r}), and it reaches GenerateReqInput.sampling_params")
b, _, req = chat(chat_template_kwargs={"reasoning_effort": "xhigh"})
ok(b == 256, f"kwargs-only effort xhigh -> 256 (got {b!r})")
ok(chat()[0] == 64, "no effort -> the server default (default_reasoning_effort=low) -> 64")
ok(chat(defaults={"enable_thinking": True, "reasoning_effort": "medium"})[0] == 128,
   "server default under the plain `reasoning_effort` key -> 128")
ok(chat(reasoning_effort="high")[0] == 256 and chat(reasoning_effort="max")[0] == 256
   and chat(reasoning_effort="minimal")[0] == 64, "aliases: high/max -> xhigh budget, minimal -> low")
ok(chat(reasoning_effort="xhigh", chat_template_kwargs={"enable_thinking": False})[0] is None,
   "enable_thinking=false -> no budget")
ok(chat(reasoning_effort="none")[0] is None, "effort none -> thinking off -> no budget")
ok(chat(defaults={"enable_thinking": False, "default_reasoning_effort": "low"})[0] is None,
   "server default thinking off, request silent -> no budget")
cp = {"thinking_budget": 5, "other": 1}
b, adapted, req = chat(reasoning_effort="xhigh", custom_params=cp)
ok(b == 5 and adapted.sampling_params["custom_params"].get("other") == 1, "explicit custom_params.thinking_budget wins (5), other keys kept")
b, adapted, req = chat(reasoning_effort="xhigh", custom_params={"other": 1})
ok(b == 256 and req.custom_params == {"other": 1, "thinking_budget": 256}, "map budget merged into existing custom_params")
ok(chat(reasoning_effort="xhigh", thinking_token_budget=33)[0] == 33, "thinking_token_budget alias wins over the map (33)")
ok(chat(reasoning_effort="xhigh", thinking_token_budget=33, custom_params={"thinking_budget": 5})[0] == 5,
   "custom_params.thinking_budget wins over the alias")
ok(chat(reasoning_effort="xhigh", thinking_token_budget=33, chat_template_kwargs={"enable_thinking": False})[0] is None,
   "alias with thinking off -> nothing (inert there)")
ok(chat(thinking_token_budget="300")[0] == 300, "alias given as a numeric string -> 300 (pydantic int)")
try:
    chat(thinking_token_budget="lots"); ok(False, "a non-integer alias must be a validation error")
except Exception as exc:
    ok("thinking_token_budget" in str(exc), "a non-integer alias is a request validation error (400), not ignored")
try:
    chat(reasoning_effort="turbo"); ok(False, "an unknown effort string must be refused by SGLang's protocol")
except Exception as exc:
    ok("reasoning_effort" in str(exc), "an unknown effort string never reaches the hook: SGLang's protocol refuses it (400)")
ok(messages(thinking={"type": "enabled", "budget_tokens": 2048}) == 2048,
   "/v1/messages thinking.budget_tokens=2048 -> custom_params.thinking_budget=2048 (explicit wins)")
ok(messages() == 64, "/v1/messages without thinking -> the server default's budget (64)")
ok(messages(thinking={"type": "disabled"}) is None, "/v1/messages thinking disabled -> no budget")
glue.reload({})
ok(chat(reasoning_effort="xhigh")[0] is None, "map off -> no map budget")
ok(chat(reasoning_effort="xhigh", thinking_token_budget=33)[0] == 33, "map off -> the alias is still honoured")
ok(messages(thinking={"type": "enabled", "budget_tokens": 1024}) == 1024, "map off -> budget_tokens still mapped")
sys.exit(1 if fails else 0)
'''


def scenarios() -> None:
    print("--- 4. scenarios: real request objects through the patched call sites ---", flush=True)
    r = run([sys.executable, "-c", SCENARIOS, MAP])
    sys.stdout.write("\n".join(l for l in r.stdout.splitlines() if l.startswith("  ")) + "\n")
    if r.returncode != 0:
        tail = [l for l in r.stderr.splitlines() if l.strip() and "warn" not in l.lower()][-6:]
        print("    stderr: " + "\n    ".join(tail))
    ok(r.returncode == 0, "every scenario passed")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    control_stock()
    negatives()
    install_twice()
    scenarios()
    print(f"[selftest-container] {'PASS' if not fails else f'FAIL ({fails})'}")
    sys.exit(1 if fails else 0)
