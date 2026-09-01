#!/usr/bin/env bash
# test-compose-sampler-profiles.sh — compose ↔ registry sampler drift gate
# (#984/#1014 Layer 2 guard, same spirit as test-compose-status-drift.sh).
#
# Models whose card publishes PER-MODE sampler rows carry them in the registry
# as `_entry(sampler_profiles={"instruct": {...}, "thinking": {...}})` (today:
# qwen3.8-27b, on vLLM AND llama.cpp). Their composes must DERIVE the sampler
# from that data instead of hardcoding a third copy: vLLM composes pick the row
# matching ENABLE_THINKING, llama.cpp composes couple it via THINKING=1 and a
# ${THINKING:+...} argv tail (`:=` defaults either way, so an explicit user env
# still wins).
#
# ⭐ This also retires discussion #993's four-variable ritual — for any model
# covered here (sampler_profiles present), `ENABLE_THINKING=true` (vLLM) or
# `THINKING=1` (llama.cpp) alone now selects the card's thinking sampler.
# Setting TEMP/TOP_P/PRESENCE_PENALTY by hand is redundant (harmless, they
# still win), and the docs saying otherwise are stale.

# Checks, for every registry entry WITH sampler_profiles:
#   (a) shape: exactly instruct+thinking rows, numeric sampler fields,
#       instruct != thinking (a per-mode registry that isn't per-mode is noise);
#   (b) the compose RENDERED with no env ships the row named by the registry's
#       `sampler_default_mode` (absent = "instruct"; qwen3.8-27b = "thinking"
#       since 2026-09-01);
#   (c) the compose RENDERED with the OPT-IN knob for the other mode ships that
#       other row — ENABLE_THINKING=true/false (vLLM), THINKING=1 / INSTRUCT=1
#       (llama.cpp);
#   (d) explicit TEMP/TOP_P/PRESENCE_PENALTY beat both rows (`:=` contract,
#       and last-flag-wins on the llama.cpp tail).
# Rendering runs each compose's real entrypoint under bash with the engine
# stubbed — two argv dialects, picked per compose:
#   vLLM      — `vllm` on PATH; the sampler arrives as one
#               --override-generation-config JSON blob.
#   llama.cpp — /app/llama-server (the image ENTRYPOINT every llama.cpp
#               engine we ship uses); the sampler arrives as --temp/--top-p/
#               --top-k/--min-p/--presence-penalty/--repeat-penalty argv.
#               THINKING=1 flips modes via a ${THINKING:+...} command tail +
#               llama.cpp's deprecated LAST-FLAG-WINS parsing (the compose
#               headers pin this to the b10236 pin; re-test on bump).
# No GPU, no weights, no image.
#
# And for every entry WITHOUT sampler_profiles:
#   (e) untouched — its compose keeps a STATIC sampler (or none): the command
#       list still carries --override-generation-config itself, and the
#       entrypoint grew no sampler-mode logic. Single-row models (qwen3.6
#       family) must stay byte-identical in behavior.
set -euo pipefail

# Force Python's UTF-8 mode (PEP 540) for every python3 this script runs.
# Repo sources are full of unicode (— × → ⚠), and without this a rig on a real
# non-UTF-8 locale decodes reads, stdout AND argv with the locale codec, which
# crashes the launcher/emit paths (#779). Guarded by test-locale-utf8.sh.
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

from scripts.lib.profiles.compose_registry import get_registry

failures = []


def check(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        failures.append(msg)


# --- compose-rendering harness (borrowed from test-spec-toggle-contract.sh) --

def _interp(v, env):
    """docker compose interpolation: `$$` is a literal `$`, `${VAR:-default}`
    resolves (defaults may NEST, e.g. ${TEMP:-${TEMPERATURE:-0.7}}), and
    `${VAR:+alt}` expands to alt only when VAR is set non-empty — the form the
    llama.cpp composes use for their THINKING=1 argv tail."""
    out, i = [], 0
    while i < len(v):
        if v[i:i + 2] == "$$":
            out.append("$"); i += 2; continue
        if v[i:i + 2] == "${":
            depth, j = 1, i + 2
            while j < len(v) and depth:
                if v[j:j + 2] == "${":
                    depth += 1; j += 2
                elif v[j] == "}":
                    depth -= 1; j += 1
                else:
                    j += 1
            m = re.match(r"([A-Za-z_]\w*)(:-|:\+)?", v[i + 2:j - 1])
            name, op = m.group(1), m.group(2)
            rest = v[i + 2 + m.end():j - 1]
            val = env.get(name)
            if op == ":+":
                if val:
                    out.append(_interp(rest, env))
            else:                                     # plain or :- default
                out.append(val if val else (_interp(rest, env) if rest else ""))
            i = j; continue
        out.append(v[i]); i += 1
    return "".join(out)


def _block(lines, key):
    """Return (start, end, indent) of a `key:` block-scalar body, or None."""
    for i, l in enumerate(lines):
        if re.match(rf"^\s*{key}:\s*$", l):
            j = i + 1
            while j < len(lines) and not re.match(r"^\s*-\s*\|", lines[j]):
                if re.match(r"^\s*(command|image|volumes|environment):", lines[j]):
                    j = None; break
                j += 1
            if j is None:
                continue
            ind = len(lines[j]) - len(lines[j].lstrip()) + 2
            k = j + 1
            while k < len(lines):
                if lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) < ind:
                    break
                k += 1
            return j + 1, k, ind
    return None


def container_env(text, host):
    """Reproduce docker's env filtering: a var reaches the container ONLY if the
    compose declares it under `environment:`."""
    lines, out = text.split("\n"), {}
    for i, l in enumerate(lines):
        if not re.match(r"^\s*environment:\s*$", l):
            continue
        ind = len(l) - len(l.lstrip())
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if cur.strip() and (len(cur) - len(cur.lstrip())) <= ind:
                break
            m = re.match(r"^\s*-\s*([A-Za-z_]\w*)(?:=(.*))?$", cur)
            if m:
                name, val = m.group(1), m.group(2)
                if val is None:                       # bare `- NAME` passthrough
                    if name in host:
                        out[name] = host[name]
                else:
                    out[name] = _interp(val.strip().strip("'\""), host)
            j += 1
        break
    return out


def entrypoint_of(text):
    lines = text.split("\n")
    b = _block(lines, "entrypoint")
    if b is None:
        return None
    s, e, ind = b
    return "\n".join(l[ind:] if len(l) > ind else "" for l in lines[s:e])


def _command_list(text, env):
    lines, args = text.split("\n"), []
    for i, l in enumerate(lines):
        if not re.match(r"^\s*command:\s*$", l):
            continue
        ind = len(l) - len(l.lstrip())
        j = i + 1
        while j < len(lines):
            cur = lines[j]
            if not cur.strip():
                j += 1; continue
            if (len(cur) - len(cur.lstrip())) <= ind:
                break
            m = re.match(r"^\s*-\s*(.*)$", cur)
            if m:
                v = m.group(1).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                args.append(_interp(v, env))
            j += 1
        break
    return args


def _command_scalar(text, env):
    """The `command: >-` folded-scalar form (llama.cpp composes): one command
    string, interpolated then word-split. Comments can't live inside the
    scalar, so every body line is real argv text."""
    lines = text.split("\n")
    for i, l in enumerate(lines):
        m = re.match(r"^(\s*)command:\s*>-\s*$", l)
        if not m:
            continue
        ind = len(m.group(1))
        parts = []
        for cur in lines[i + 1:]:
            if cur.strip() and (len(cur) - len(cur.lstrip())) <= ind:
                break
            if cur.strip():
                parts.append(cur.strip())
        return shlex.split(_interp(" ".join(parts), env))
    return None


def argv_under(text, env):
    """Run the compose's entrypoint with its engine stubbed (`vllm` on PATH for
    vLLM composes, /app/llama-server for llama.cpp ones — both print ARG: per
    argv), under the env docker would actually give the container; return the
    argv list."""
    body = entrypoint_of(text)
    if body is None:
        return None, "no block-scalar entrypoint"
    llama = "/app/llama-server" in body
    with tempfile.TemporaryDirectory() as d:
        dp = pathlib.Path(d)
        if llama:
            stub = dp / "app" / "llama-server"
            stub.parent.mkdir(parents=True)
        else:
            stub = dp / "vllm"
        stub.write_text('#!/bin/bash\nfor a in "$@"; do printf "ARG:%s\\n" "$a"; done\nexit 0\n')
        stub.chmod(0o755)
        etc = dp / "etc" / "club3090"
        etc.mkdir(parents=True, exist_ok=True)
        (etc / "detect_nvlink.sh").write_text("_NVLINK_ENABLED=0\n")
        for sub in set(re.findall(r"/etc/club3090/([\w.-]+)/install\.sh", body)):
            (etc / sub).mkdir(parents=True, exist_ok=True)
            (etc / sub / "install.sh").write_text("#!/bin/bash\nexit 0\n")
        script = dp / "ep.sh"
        script.write_text(body.replace("$$", "$")
                          .replace("/etc/club3090", str(etc))
                          .replace("/app/llama-server", str(stub)))
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("SPEC", "NUM_SPEC", "ENABLE_THINKING", "THINKING",
                             "INSTRUCT",
                                  "TEMP", "TEMPERATURE", "TOP_P", "TOP_K", "MIN_P",
                                  "PRESENCE_PENALTY", "REPEAT_PENALTY"))}
        e.update(container_env(text, env))
        e["PATH"] = f"{dp}:{os.environ['PATH']}"
        # llama.cpp composes come in BOTH dialects: the folded `command: >-`
        # form (qwen3.8-27b) and the LIST form (the moe-cache family — GLM,
        # DeepSeek, qwen3.8-flash-next), which is used because comments cannot
        # live inside a folded scalar. Try scalar first, fall back to list.
        cmd = (_command_scalar(text, env) or _command_list(text, env)) if llama \
              else _command_list(text, env)
        if not cmd:
            return None, 'no command: block (neither >- scalar nor list form)'
        r = subprocess.run(["bash", str(script), "--", *cmd],
                           capture_output=True, text=True, env=e, timeout=60)
        args = [l[4:] for l in r.stdout.split("\n") if l.startswith("ARG:")]
        return (args or None), r.stderr.strip()[-300:]


LLAMA_SAMPLER_FLAGS = {
    "--temp": "temperature", "--top-p": "top_p", "--top-k": "top_k",
    "--min-p": "min_p", "--presence-penalty": "presence_penalty",
    "--repeat-penalty": "repetition_penalty",
}


def sampler_payload(args, llama=False):
    """The sampler the engine would receive, or None. vLLM: the
    --override-generation-config JSON blob. llama.cpp: the LAST value of each
    repeated sampler flag (the deprecated last-flag-wins parsing the compose
    headers pin to the b10236 image pin)."""
    if not args:
        return None
    if not llama:
        if "--override-generation-config" not in args:
            return None
        i = args.index("--override-generation-config")
        try:
            return json.loads(args[i + 1])
        except Exception as exc:
            return {"<invalid json>": str(exc)}
    payload, i = {}, 0
    while i < len(args):
        field = LLAMA_SAMPLER_FLAGS.get(args[i])
        if field and i + 1 < len(args):
            payload[field] = float(args[i + 1])
            i += 2
        else:
            i += 1
    return payload or None


FIELDS = ("temperature", "top_p", "top_k", "min_p",
          "presence_penalty", "repetition_penalty")


def rows_agree(payload, row):
    """Every field the compose SHIPS must equal the registry row. A row may
    carry more than the payload (e.g. repetition_penalty=1.0 == the engine
    default on slugs that never emit the key) — absence is agreement."""
    if not isinstance(payload, dict) or "<invalid json>" in payload:
        return False
    return all(
        k in payload and abs(float(payload[k]) - float(row[k])) < 1e-9
        for k in payload if k in row
    ) and all(k in payload for k in FIELDS[:5])


registry = get_registry()
with_profiles = {k: e for k, e in registry.items() if e.get("sampler_profiles")}
without_profiles = {k: e for k, e in registry.items() if not e.get("sampler_profiles")}
llama_with = {k for k in with_profiles if k.startswith("llamacpp/")}
vllm_with = {k for k in with_profiles if k.startswith("vllm/")}
check(len(vllm_with) >= 20,
      f"qwen3.8-27b vLLM entries expose sampler_profiles (got {len(vllm_with)})")
check(llama_with == {"llamacpp/qwen38-27b-single-iq4xs",
                     "llamacpp/qwen38-27b-dual-q8kxl"},
      f"both llama.cpp qwen3.8 slugs gained sampler_profiles (got {sorted(llama_with)})")
check(all(k.split("/", 1)[1].startswith(("qwen38-27b-", "qwen38-flash-next-"))
          for k in with_profiles),
      "only qwen3.8-27b / qwen3.8-flash-next slugs carry sampler_profiles today")

for slug, entry in sorted(with_profiles.items()):
    profiles = entry["sampler_profiles"]
    check(set(profiles) == {"instruct", "thinking"},
          f"{slug}: sampler_profiles has exactly instruct+thinking rows")
    if set(profiles) != {"instruct", "thinking"}:
        continue
    for mode, row in profiles.items():
        check(all(f in row and isinstance(row[f], (int, float)) for f in FIELDS),
              f"{slug}: {mode} row carries all six numeric sampler fields")
    check(profiles["instruct"] != profiles["thinking"],
          f"{slug}: instruct and thinking rows differ (per-mode data)")

    text = pathlib.Path(entry["compose_path"]).read_text(encoding="utf-8")

    # Argv dialect: llama.cpp composes exec /app/llama-server (image ENTRYPOINT)
    # and take the sampler as flags; THINKING=1 drives their ${THINKING:+...}
    # tail. vLLM composes exec `vllm` and flip on ENABLE_THINKING=true.
    # Any llama.cpp engine, not just the one engine id: the moe-cache family
    # ships `llamacpp-club3090-*`. Matching only "llama-cpp-local" silently
    # routed those composes through the vLLM extractor, which looks for
    # --override-generation-config and returned None for every check.
    llama = str(entry.get("engine") or "").startswith(("llama-cpp", "llamacpp"))

    # WHICH row the compose ships by default is registry POLICY, not something
    # read back out of the compose — deriving the expectation from the same file
    # that supplies the evidence would make an accidental flip self-approving.
    # Absent key = "instruct" (the pre-2026-09-01 stack-wide default).
    default_mode = entry.get("sampler_default_mode") or "instruct"
    check(default_mode in ("instruct", "thinking"),
          f"{slug}: sampler_default_mode is instruct|thinking (got {default_mode!r})")
    if default_mode not in ("instruct", "thinking"):
        continue
    other_mode = "thinking" if default_mode == "instruct" else "instruct"

    # The opt-IN knob is whichever one selects the NON-default row, so it flips
    # with the policy: llama.cpp composes carry both ${THINKING:+...} and
    # ${INSTRUCT:+...} argv tails, vLLM composes derive their row from
    # ENABLE_THINKING at boot.
    if llama:
        mode_env = {"THINKING": "1"} if other_mode == "thinking" else {"INSTRUCT": "1"}
        mode_label = "THINKING=1" if other_mode == "thinking" else "INSTRUCT=1"
    else:
        want = "true" if other_mode == "thinking" else "false"
        mode_env = {"ENABLE_THINKING": want}
        mode_label = f"ENABLE_THINKING={want}"

    d_args, d_err = argv_under(text, {})
    d_row = sampler_payload(d_args, llama=llama)
    check(rows_agree(d_row, profiles[default_mode]),
          f"{slug}: default render ships the {default_mode.upper()} row "
          f"(got {d_row}; {d_err})")

    t_args, t_err = argv_under(text, mode_env)
    t_row = sampler_payload(t_args, llama=llama)
    check(rows_agree(t_row, profiles[other_mode]),
          f"{slug}: {mode_label} ships the {other_mode.upper()} row "
          f"(got {t_row}; {t_err})")

    x_args, x_err = argv_under(text, {
        **mode_env, "TEMP": "0.42", "TOP_P": "0.5", "PRESENCE_PENALTY": "0.1",
    })
    x_row = sampler_payload(x_args, llama=llama)
    check(bool(x_row) and float(x_row.get("temperature", -1)) == 0.42
          and float(x_row.get("top_p", -1)) == 0.5
          and float(x_row.get("presence_penalty", -1)) == 0.1,
          f"{slug}: explicit env beats both rows (got {x_row}; {x_err})")

for slug, entry in sorted(without_profiles.items()):
    path = pathlib.Path(entry["compose_path"])
    if not path.exists():
        check(False, f"{slug}: compose exists at {path}")
        continue
    text = path.read_text(encoding="utf-8")
    body = entrypoint_of(text) or ""
    cmd_has_static = "--override-generation-config" in text.split("command:")[-1]
    check(cmd_has_static or "override-generation-config" not in body,
          f"{slug}: no-profiles entry keeps its sampler STATIC "
          "(command-list flag, no entrypoint coupling)")
    check("SAMPLER_ARGS" not in body and "_temp=" not in body,
          f"{slug}: no-profiles entrypoint grew no sampler-mode logic")

if failures:
    raise SystemExit(f"{len(failures)} sampler-profile checks failed")
print(f"test-compose-sampler-profiles: ok "
      f"({len(with_profiles)} coupled + {len(without_profiles)} static entries)")
PY
