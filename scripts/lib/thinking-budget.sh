#!/usr/bin/env bash
#
# thinking-budget.sh — resolve AND VERIFY a cross-engine reasoning budget for
# quality-test.sh --thinking-budget N (club-3090#1383).
#
# WHY THIS FILE EXISTS
# --------------------
# A thinking model can reason until the per-case wall clock: 2 of the first 7
# MiMo-V2.6-9B scenarios hit 900 s and scored `timeout fail`, which says
# nothing about the model. All three engines implement a reasoning budget, but
# with three different shapes, and two of the three ACCEPT the request field
# while doing something other than bounding the run when the server was not
# started for it. So the budget is only worth sending if the harness has
# verified it can take effect — verification is the feature, not an extra.
#
#   engine      server prerequisite               request shape
#   llama.cpp   --reasoning-budget N (boot flag)  none — server-wide
#   vLLM        --reasoning-parser <name>         thinking_token_budget: N
#   SGLang      --enable-custom-logit-processor   custom_logit_processor +
#                                                 custom_params.thinking_budget
#
# What each engine does WITHOUT its prerequisite (read from the pinned images):
#   llama.cpp  nothing to send — the harness cannot set a boot flag per request,
#              and a compose-shipped `--reasoning-budget "${REASONING_BUDGET:--1}"`
#              boots UNBOUNDED (-1) with the flag visibly present. Presence of the
#              flag is therefore NOT evidence; its resolved VALUE is.
#   vLLM       v0.29.0 raises VLLMValidationError per request
#              (vllm/v1/engine/input_processor.py) — every scenario 400s.
#   SGLang     v0.5.20 raises ValueError per request (tokenizer_manager.py) —
#              every scenario 400s. And the per-request processor is a dill
#              pickle of a MODEL-SPECIFIC class (Qwen3…/Glm4Moe…/DeepSeekR1…);
#              the base class has no think-token ids and would fail at sample
#              time. The class is chosen from the server's reasoning parser.
#
# CONTRACT
# --------
#   thinking_budget_engine_kind
#       Prints vllm | llamacpp | sglang | exllamav3 | unknown. Evidence, in
#       order: $CONTAINER name, its docker image, then /v1/models `owned_by`.
#       The DECISION is engine-kind.sh's (#1282) — nothing here classifies.
#   thinking_budget_verify <kind> <N>
#       rc 0  verified — the budget WILL take effect on this server
#       rc 1  refused  — positive evidence it will NOT (no bypass: the evidence
#                        says the run would be unbounded or would 400)
#       rc 2  unverifiable — no evidence either way (no container, no docker,
#                        no server readback). Caller decides; the only
#                        acceptable bypass is an explicit, loud one.
#       Sets THINKING_BUDGET_EVIDENCE (one line, human) and, for SGLang,
#       THINKING_BUDGET_REASONING_PARSER. Prints nothing on stdout.
#   thinking_budget_fix_hint <kind> <N>
#       Prints the fix instruction for a refused/unverifiable verification.
#   thinking_budget_extra_body <kind> <N> [sglang_processor_class]
#       Prints the JSON object benchlocal-cli --extra-body must carry, or
#       nothing for llama.cpp (server-wide, nothing to send).
#   thinking_budget_sglang_processor <reasoning_parser>
#       Prints the SGLang processor class for that parser, or nothing.
#
# Nothing here mutates the server. The only network calls are GETs against
# $URL (/v1/models, /server_info, /get_server_info).
export PYTHONUTF8="${PYTHONUTF8:-1}"

_TB_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=engine-kind.sh
source "${_TB_LIB_DIR}/engine-kind.sh"

# The two packs whose model calls are made by an agent INSIDE the sandbox
# container, not by benchlocal's runner. A per-request budget (vLLM/SGLang)
# never reaches them: the hermes sandbox forwards only temperature/top_p/
# top_k/min_p/repetition_penalty/max_tokens from `sampling`, and aider forwards
# nothing budget-shaped at all. A server-wide budget (llama.cpp) covers them
# for free. See the follow-up on #1383.
THINKING_BUDGET_AGENTIC_PACKS="hermesagent-20 aider-polyglot-30"

thinking_budget_engine_kind() {
  local kind="unknown" c="${CONTAINER:-}" img owned
  if [[ -n "$c" && "$c" != "none" ]]; then
    kind="$(engine_kind_from_container "$c")"
    if [[ "$kind" == "unknown" ]] && command -v docker >/dev/null 2>&1; then
      img="$(docker inspect "$c" --format '{{.Config.Image}}' 2>/dev/null || true)"
      [[ -n "$img" ]] && kind="$(engine_kind_from_image "$img")"
    fi
  fi
  if [[ "$kind" == "unknown" && -n "${URL:-}" ]]; then
    owned="$(curl -sf -m 5 ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} "${URL}/v1/models" 2>/dev/null \
      | python3 -c 'import sys, json
try:
    d = json.load(sys.stdin)
    print((d.get("data") or [{}])[0].get("owned_by") or "")
except Exception:
    pass' 2>/dev/null || true)"
    [[ -n "$owned" ]] && kind="$(engine_kind_from_owned_by "$owned")"
  fi
  echo "$kind"
  return 0
}

# Read the container's boot configuration and answer ONE engine-specific
# question about it. stdin: `docker inspect <container>` JSON. stdout, one
# line:
#   llamacpp  BUDGET <int> | BUDGET none | BUDGET unresolvable <why>
#   vllm      PARSER <value> | PARSER none
#   sglang    CLP true|false PARSER <value|none>
#
# Flags are looked for in Config.Entrypoint and Config.Cmd, as array elements
# (`--flag`, `value` / `--flag=value`) AND inside shell-script strings (the
# shipped composes wrap llama-server in `bash -c '… exec llama-server "$@"
# "${ROW[@]}"'`). A `${VAR:-default}` / `${VAR-default}` / `$VAR` value is
# resolved through Config.Env exactly as the shell would: a bare `VAR` entry
# (no `=`) is UNSET, `VAR=` is EMPTY, and only `:-` treats empty as unset.
# Comment lines inside script strings are skipped. For llama.cpp the LAST
# occurrence wins, matching its parser; the compose pattern appends the ROW
# after "$@", so script-string matches are ordered after Cmd elements.
# LLAMA_ARG_THINK_BUDGET (llama.cpp's env spelling) is consulted only when no
# CLI occurrence exists.
#
# The Python is held in a variable and run with -c: `python3 - <<'PY'` would
# take the SCRIPT from stdin and silently displace the inspect JSON piped in.
_THINKING_BUDGET_FACTS_PY="$(cat <<'PY'
import json, re, sys

mode = sys.argv[1]
try:
    data = json.load(sys.stdin)
except Exception:
    print("ERROR unreadable docker inspect output")
    sys.exit(0)
if isinstance(data, list):
    data = data[0] if data else {}
cfg = (data or {}).get("Config") or {}

env = {}
for item in cfg.get("Env") or []:
    if "=" in item:
        k, v = item.split("=", 1)
        env[k] = v
    else:
        env[item] = None  # declared, unset

_VAR = re.compile(r'^\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?)-([^}]*))?\}$')
_BARE = re.compile(r'^\$([A-Za-z_][A-Za-z0-9_]*)$')

def expand(tok):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
        tok = tok[1:-1]
    m = _VAR.match(tok)
    if m:
        name, colon, default = m.group(1), m.group(2), m.group(3)
        val = env.get(name)
        if val is None or (val == "" and colon == ":"):
            return default if default is not None else ""
        return val
    m = _BARE.match(tok)
    if m:
        return env.get(m.group(1)) or ""
    return tok

def script_lines(s):
    for line in s.splitlines():
        if line.lstrip().startswith("#"):
            continue
        yield line

def is_script(el):
    return "\n" in el or " " in el

def flag_values(flag, elements):
    """Values for a flag: literal array-element matches first (entrypoint,
    then cmd, in order), then matches inside script strings — the shipped
    compose pattern appends its resolved ROW after "$@", so a script-string
    value is the one llama.cpp sees last."""
    literal, scripted = [], []
    pat = re.compile(re.escape(flag) + r'(?:=|[ \t]+)("(?:[^"\\]|\\.)*"|\'[^\']*\'|[^\s"\')]+)')
    n = len(elements)
    for i, el in enumerate(elements):
        if el == flag:
            if i + 1 < n:
                literal.append(expand(elements[i + 1]))
        elif el.startswith(flag + "="):
            literal.append(expand(el[len(flag) + 1:]))
        elif flag in el and is_script(el):
            for line in script_lines(el):
                for m in pat.finditer(line):
                    scripted.append(expand(m.group(1)))
    return literal + scripted

def flag_present(flag, elements):
    pat = re.compile(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])')
    for el in elements:
        if el == flag or el.startswith(flag + "="):
            return True
        if flag in el and is_script(el):
            for line in script_lines(el):
                if pat.search(line):
                    return True
    return False

entry = [e for e in (cfg.get("Entrypoint") or []) if isinstance(e, str)]
cmd = [e for e in (cfg.get("Cmd") or []) if isinstance(e, str)]
ordered = entry + cmd

if mode == "llamacpp":
    vals = flag_values("--reasoning-budget", ordered)
    if not vals:
        envv = env.get("LLAMA_ARG_THINK_BUDGET")
        if envv:
            vals = [envv]
    if not vals:
        print("BUDGET none")
        sys.exit(0)
    last = vals[-1]
    if re.fullmatch(r"-?\d+", last or ""):
        print("BUDGET %d" % int(last))
    else:
        print("BUDGET unresolvable value %r" % last)
elif mode == "vllm":
    vals = [v for v in flag_values("--reasoning-parser", ordered) if v]
    vals += [v for v in flag_values("--reasoning-config", ordered) if v]
    print("PARSER %s" % (vals[-1] if vals else "none"))
elif mode == "sglang":
    clp = flag_present("--enable-custom-logit-processor", ordered)
    vals = [v for v in flag_values("--reasoning-parser", ordered) if v]
    print("CLP %s PARSER %s" % ("true" if clp else "false", vals[-1] if vals else "none"))
else:
    print("ERROR unknown mode %s" % mode)
PY
)"
_thinking_budget_container_facts() {
  python3 -c "$_THINKING_BUDGET_FACTS_PY" "$1"
}

# SGLang's own readback: /server_info (v0.5.20) dumps the resolved ServerArgs;
# /get_server_info is the deprecated alias. Walks the whole document so a
# regrouping of the args (v0.5.20 moved them under sub-groups) cannot hide the
# key. stdout: `CLP true|false PARSER <v|none>` or nothing when unreachable.
_thinking_budget_sglang_server_facts() {
  local ep body
  for ep in /server_info /get_server_info; do
    body="$(curl -sf -m 5 ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} "${URL}${ep}" 2>/dev/null || true)"
    [[ -n "$body" ]] || continue
    python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
hits = {}
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("enable_custom_logit_processor", "reasoning_parser") and k not in hits:
                hits[k] = v
            walk(v)
    elif isinstance(x, list):
        for v in x:
            walk(v)
walk(d)
if "enable_custom_logit_processor" not in hits:
    sys.exit(0)
clp = hits["enable_custom_logit_processor"]
clp = clp is True or str(clp).lower() in ("1", "true", "yes", "on")
rp = hits.get("reasoning_parser") or "none"
print("CLP %s PARSER %s" % ("true" if clp else "false", rp))
' <<<"$body" 2>/dev/null && return 0
  done
  return 0
}

# SGLang ships model-specific ThinkingBudgetLogitProcessor subclasses
# (python/sglang/srt/sampling/custom_logit_processor.py, v0.5.20). Only the
# parsers whose think-token ids are known to match are mapped; anything else
# is a refusal, overridable with THINKING_BUDGET_SGLANG_PROCESSOR=<ClassName>.
thinking_budget_sglang_processor() {
  case "${1:-}" in
    qwen3|qwen3-thinking) echo "Qwen3ThinkingBudgetLogitProcessor" ;;
    glm45)                echo "Glm4MoeThinkingBudgetLogitProcessor" ;;
    deepseek-r1)          echo "DeepSeekR1ThinkingBudgetLogitProcessor" ;;
    *)                    echo "" ;;
  esac
  return 0
}

THINKING_BUDGET_EVIDENCE=""
THINKING_BUDGET_REASONING_PARSER=""

thinking_budget_verify() {
  local kind="$1" want="$2" c="${CONTAINER:-}" facts="" have_container=0
  THINKING_BUDGET_EVIDENCE=""
  THINKING_BUDGET_REASONING_PARSER=""
  if [[ -n "$c" && "$c" != "none" ]] && command -v docker >/dev/null 2>&1 \
     && docker inspect "$c" >/dev/null 2>&1; then
    have_container=1
  fi

  case "$kind" in
    llamacpp)
      if [[ "$have_container" != "1" ]]; then
        THINKING_BUDGET_EVIDENCE="llama.cpp's --reasoning-budget is a boot flag; without the serving container (CONTAINER='${c:-unset}') there is nothing to read it from"
        return 2
      fi
      facts="$(docker inspect "$c" 2>/dev/null | _thinking_budget_container_facts llamacpp)"
      case "$facts" in
        "BUDGET none")
          THINKING_BUDGET_EVIDENCE="container ${c} was booted WITHOUT --reasoning-budget (unbounded)"
          return 1 ;;
        "BUDGET unresolvable"*)
          THINKING_BUDGET_EVIDENCE="container ${c}: --reasoning-budget ${facts#BUDGET unresolvable } — cannot resolve the effective value"
          return 2 ;;
        "BUDGET "*)
          local got="${facts#BUDGET }"
          if [[ "$got" == "$want" ]]; then
            THINKING_BUDGET_EVIDENCE="container ${c} boots llama-server with --reasoning-budget ${got}"
            return 0
          elif [[ "$got" == "-1" ]]; then
            THINKING_BUDGET_EVIDENCE="container ${c} resolves --reasoning-budget to -1 (UNBOUNDED — the flag is present but REASONING_BUDGET was not set at boot)"
            return 1
          else
            THINKING_BUDGET_EVIDENCE="container ${c} resolves --reasoning-budget to ${got}, not ${want} — the server value wins and the harness cannot change it per request"
            return 1
          fi ;;
        *)
          THINKING_BUDGET_EVIDENCE="could not read container ${c}: ${facts:-no output}"
          return 2 ;;
      esac
      ;;
    vllm)
      if [[ "$have_container" != "1" ]]; then
        THINKING_BUDGET_EVIDENCE="vLLM only honours thinking_token_budget with a reasoning parser configured at boot; without the serving container (CONTAINER='${c:-unset}') that cannot be checked"
        return 2
      fi
      facts="$(docker inspect "$c" 2>/dev/null | _thinking_budget_container_facts vllm)"
      case "$facts" in
        "PARSER none")
          # The container's own argv can miss it: an image that starts vLLM through its
          # own launcher (bucko's qwen38-serve adds --reasoning-parser inside the script)
          # shows no parser in `docker inspect`. vLLM prints the arguments it actually
          # resolved in its boot log, so read that before refusing.
          local logged
          logged="$(docker logs "$c" 2>&1 | command grep -m1 'non-default args' \
            | command grep -oE "'reasoning_parser': '[^']+'" | command grep -oE "'[^']+'$" | tr -d "'")"
          if [[ -n "$logged" ]]; then
            THINKING_BUDGET_EVIDENCE="container ${c}: vLLM's boot log reports reasoning_parser=${logged} (set by the image's launcher, not the container argv)"
            THINKING_BUDGET_REASONING_PARSER="$logged"
            return 0
          fi
          THINKING_BUDGET_EVIDENCE="container ${c} was booted WITHOUT --reasoning-parser (container argv and vLLM's boot log both lack it) — vLLM v0.29.0 rejects thinking_token_budget per request (VLLMValidationError), so every scenario would fail"
          return 1 ;;
        "PARSER "*)
          THINKING_BUDGET_EVIDENCE="container ${c} boots vLLM with --reasoning-parser ${facts#PARSER }"
          return 0 ;;
        *)
          THINKING_BUDGET_EVIDENCE="could not read container ${c}: ${facts:-no output}"
          return 2 ;;
      esac
      ;;
    sglang)
      # The server's own readback first — it is the resolved truth and works
      # for hand-rolled / remote servers too; docker is the fallback.
      facts="$(_thinking_budget_sglang_server_facts)"
      local src="/server_info"
      if [[ -z "$facts" && "$have_container" == "1" ]]; then
        facts="$(docker inspect "$c" 2>/dev/null | _thinking_budget_container_facts sglang)"
        src="container ${c}"
      fi
      if [[ -z "$facts" ]]; then
        THINKING_BUDGET_EVIDENCE="SGLang: neither /server_info nor a serving container (CONTAINER='${c:-unset}') is available to check --enable-custom-logit-processor"
        return 2
      fi
      local parser="${facts##*PARSER }"
      THINKING_BUDGET_REASONING_PARSER="$parser"
      case "$facts" in
        "CLP true"*)
          THINKING_BUDGET_EVIDENCE="${src} reports enable_custom_logit_processor=true, reasoning_parser=${parser}"
          return 0 ;;
        "CLP false"*)
          THINKING_BUDGET_EVIDENCE="${src} reports the server was started WITHOUT --enable-custom-logit-processor — SGLang v0.5.20 rejects custom_logit_processor per request (ValueError), so every scenario would fail"
          return 1 ;;
        *)
          THINKING_BUDGET_EVIDENCE="could not read ${src}: ${facts}"
          return 2 ;;
      esac
      ;;
    *)
      THINKING_BUDGET_EVIDENCE="engine family '${kind}' has no known reasoning-budget mechanism"
      return 1
      ;;
  esac
}

thinking_budget_fix_hint() {
  local kind="$1" want="$2"
  case "$kind" in
    llamacpp)
      cat <<EOF
  Fix: the budget is a llama-server BOOT flag — set it and reboot the compose:
       REASONING_BUDGET=${want} bash scripts/switch.sh --force <slug>     # shipped llama.cpp composes
       llama-server ... --reasoning-budget ${want}                        # hand-rolled (env: LLAMA_ARG_THINK_BUDGET=${want})
       Reboot between legs: a running server keeps whatever budget it booted with.
EOF
      ;;
    vllm)
      cat <<EOF
  Fix: start vLLM with a reasoning parser (every shipped thinking compose does):
       --reasoning-parser <name>      e.g. qwen3 / gemma4 — see the compose's command block
EOF
      ;;
    sglang)
      cat <<EOF
  Fix: add --enable-custom-logit-processor to the SGLang server command and reboot
       (no shipped SGLang compose sets it — add it to the compose's command block).
       The per-request processor is chosen from the server's --reasoning-parser
       (qwen3 / qwen3-thinking / glm45 / deepseek-r1); any other parser needs
       THINKING_BUDGET_SGLANG_PROCESSOR=<ThinkingBudgetLogitProcessor subclass>.
EOF
      ;;
    *)
      cat <<EOF
  Fix: serve on llama.cpp (--reasoning-budget), vLLM (--reasoning-parser) or SGLang
       (--enable-custom-logit-processor); set CONTAINER=<name> if the serving container
       was not auto-detected.
EOF
      ;;
  esac
}

# Serialise a reference to an SGLang-side class the way its
# CustomLogitProcessor.from_str expects: json {"callable": <hex of a pickle>}.
# dill pickles an importable class BY REFERENCE (module + qualname); the hand-
# built GLOBAL opcode below is that same reference and loads with dill.loads on
# the server, which has sglang importable. Nothing is executed client-side and
# no sglang install is needed here.
_thinking_budget_sglang_processor_str() {
  python3 - "$1" <<'PY'
import json, sys
module = "sglang.srt.sampling.custom_logit_processor"
name = sys.argv[1]
blob = b"\x80\x04c" + module.encode() + b"\n" + name.encode() + b"\n."
print(json.dumps({"callable": blob.hex()}))
PY
}

thinking_budget_extra_body() {
  local kind="$1" want="$2" proc="${3:-}"
  case "$kind" in
    vllm)
      printf '{"thinking_token_budget": %d}\n' "$want"
      ;;
    sglang)
      [[ -n "$proc" ]] || return 1
      python3 - "$want" "$(_thinking_budget_sglang_processor_str "$proc")" <<'PY'
import json, sys
print(json.dumps({
    "custom_logit_processor": sys.argv[2],
    "custom_params": {"thinking_budget": int(sys.argv[1])},
}))
PY
      ;;
    *)
      : # llama.cpp: server-wide, nothing to send
      ;;
  esac
  return 0
}
