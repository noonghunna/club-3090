#!/usr/bin/env bash
# probe-effort-budget.sh — live check that a server applies the reasoning budget chosen by effort.
#
#   bash scripts/probe-effort-budget.sh <url> [--expect-off | --positive] [--model ID] [--spec-depth N]
#
# The boot probe for BOTH engines (vLLM patches vllm-effort-budget + vllm-default-thinking-budget;
# SGLang strict thinking + its effort hook), and the real check at every engine upgrade. It reads the
# map the server booted with from the container's boot log (the `[effort-budget] v1` line printed by
# scripts/lib/effort_budget.py shell-env), sends real requests, and measures the reasoning each one
# produced. Modes:
#
#   (default)      the server was booted with TINY budgets through the real knob path:
#                    THINKING_BUDGET_LOW=64 THINKING_BUDGET_MEDIUM=128 THINKING_BUDGET_XHIGH=256 \
#                      bash scripts/switch.sh <slug>
#                  Checks: per effort the reasoning stops at about the budget, then the answer ends
#                  finish=stop with content; no effort -> the server default's budget; kwargs-only
#                  effort; an explicit budget wins (vLLM thinking_token_budget; SGLang
#                  custom_params.thinking_budget AND the thinking_token_budget alias); thinking off ->
#                  no reasoning and no stray </think>; /v1/messages with and without budget_tokens;
#                  the Responses API -> the floor; two concurrent requests at different efforts each
#                  get their own budget.
#   --expect-off   negative control: booted with THINKING_BUDGETS=off. The readback must say off and
#                  an xhigh request must reason PAST the tiny xhigh budget (256).
#   --positive     positive control: booted with the REAL budgets (no knobs). A short prompt still
#                  thinks, and its reasoning ends on its own, well under the budget.
#
# "About the budget": reasoning tokens in [budget - 8, budget + slack], slack = drafter depth + 1
# (bonus token) + the forced end sequence (1 token, plain </think>) + 4 for re-tokenizing the text.
# Drafter depth comes from the boot log (--spec-depth overrides). Reasoning length: the usage block's
# reasoning tokens when the engine reports them, else the server's own /tokenize on the reasoning text.
#
# Environment: CONTAINER (default: the container publishing <url>, club_container_for_url), MODEL
# (default: club_served_model_id), API_KEY (sent as a bearer token, never printed). Exit 0 = every
# check passed; 1 = a check failed (one ✗ line each); 2 = could not run (no server, no readback, …).
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/engine-kind.sh
source "$ROOT_DIR/scripts/lib/engine-kind.sh"
# shellcheck source=lib/club-containers.sh
source "$ROOT_DIR/scripts/lib/club-containers.sh"
# shellcheck source=lib/served-model.sh
source "$ROOT_DIR/scripts/lib/served-model.sh"
EB="$ROOT_DIR/scripts/lib/effort_budget.py"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "probe-effort-budget: $*" >&2; exit 2; }

URL_ARG="" MODE="budgets" SPEC_DEPTH="" MODEL="${MODEL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect-off) MODE="off"; shift ;;
    --positive) MODE="positive"; shift ;;
    --model) MODEL="${2:?--model needs an id}"; shift 2 ;;
    --spec-depth) SPEC_DEPTH="${2:?--spec-depth needs a number}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option '$1' (see --help)" ;;
    *) [[ -z "$URL_ARG" ]] || die "one URL only"; URL_ARG="$1"; shift ;;
  esac
done
URL="${URL_ARG:-${URL:-}}"
[[ -n "$URL" ]] || { usage >&2; exit 2; }
URL="${URL%/}"; URL="${URL%/v1}"
[[ -z "$SPEC_DEPTH" || "$SPEC_DEPTH" =~ ^[0-9]+$ ]] || die "--spec-depth takes a whole number"

# -- the container, its engine, the model ---------------------------------------------------------
CONTAINER="${CONTAINER:-$(club_container_for_url "$URL")}"
[[ -n "$CONTAINER" && "$CONTAINER" != none ]] \
  || die "no running container publishes $URL here; the probe reads the map from the boot log — set CONTAINER=<name>"
docker inspect "$CONTAINER" >/dev/null 2>&1 || die "container '$CONTAINER' not found"
KIND="$(engine_kind_from_container "$CONTAINER")"
if [[ "$KIND" == unknown ]]; then
  KIND="$(engine_kind_from_image "$(docker inspect "$CONTAINER" --format '{{.Config.Image}}' 2>/dev/null)")"
fi
if [[ "$KIND" == unknown ]]; then
  owned="$(curl -sf -m 5 "$URL/v1/models" 2>/dev/null | python3 -c 'import json,sys
try: print((json.load(sys.stdin).get("data") or [{}])[0].get("owned_by") or "")
except Exception: pass' 2>/dev/null || true)"
  KIND="$(engine_kind_from_owned_by "$owned")"
fi
[[ "$KIND" == vllm || "$KIND" == sglang ]] || die "engine kind '$KIND' for $CONTAINER — this probe covers vllm and sglang"
[[ -n "$MODEL" ]] || MODEL="$(club_served_model_id "$URL")"
[[ -n "$MODEL" ]] || die "could not resolve the served model at $URL (is it up?) — pass --model"

# -- what the server booted with (logs span restarts; the module reads the LAST v1 line) ----------
LOGS="$(docker logs "$CONTAINER" 2>&1)" || die "docker logs $CONTAINER failed"
readback() { printf '%s\n' "$LOGS" | python3 "$EB" readback-budget --line - ${1:+--effort "$1"}; }
DEFAULT_BUDGET="$(readback)"; rc=$?
[[ $rc -eq 0 ]] || die "no [effort-budget] v1 readback line in the $CONTAINER boot log — the compose is not wired, or predates it"
if [[ "$DEFAULT_BUDGET" == off ]]; then
  LOW=off MEDIUM=off XHIGH=off
else
  LOW="$(readback low)" MEDIUM="$(readback medium)" XHIGH="$(readback xhigh)"
fi
DEFAULT_EFFORT="$(printf '%s\n' "$LOGS" | python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
import effort_budget as m
info = m.parse_readback(sys.stdin.read()) or {}
print(info.get("default_effort") or "")' "$ROOT_DIR/scripts/lib")"

if [[ -z "$SPEC_DEPTH" ]]; then
  SPEC_DEPTH="$(printf '%s\n' "$LOGS" | python3 -c '
import re, sys
text = sys.stdin.read()
# The LAST mention wins: docker logs span restarts. Compose echo, vLLM args, SGLang server_args.
pats = [r"\[spec\] drafter ON - \S+ n=(\d+)", r"num_spec(?:ulative)?_tokens[\"\x27]?\s*[=:]\s*(\d+)",
        r"speculative_num_draft_tokens\s*=\s*(\d+)", r"\[spec\] drafter (OFF)"]
hits = [(m.start(), m.group(1)) for p in pats for m in re.finditer(p, text)]
last = max(hits)[1] if hits else ""
print("0" if last == "OFF" else last)')"
  SPEC_SRC="boot log"
  [[ -n "$SPEC_DEPTH" ]] || { SPEC_DEPTH=8; SPEC_SRC="not in the boot log; assumed"; }
else
  SPEC_SRC="--spec-depth"
fi

echo "[probe-effort-budget] $URL  container=$CONTAINER  engine=$KIND  model=$MODEL  mode=$MODE"
echo "[probe-effort-budget] readback: low=$LOW medium=$MEDIUM xhigh=$XHIGH default_effort=${DEFAULT_EFFORT:-?} default=$DEFAULT_BUDGET  drafter depth=$SPEC_DEPTH ($SPEC_SRC)"

case "$MODE" in
  budgets)
    [[ "$DEFAULT_BUDGET" != off ]] || die "the server booted with THINKING_BUDGETS=off — that is the --expect-off control"
    for b in "$LOW" "$MEDIUM" "$XHIGH"; do
      [[ "$b" =~ ^[0-9]+$ && "$b" -le 2048 ]] || die "default mode needs TINY budgets (got low=$LOW medium=$MEDIUM xhigh=$XHIGH): boot with THINKING_BUDGET_LOW=64 THINKING_BUDGET_MEDIUM=128 THINKING_BUDGET_XHIGH=256, or use --positive for the real ones"
    done ;;
  off)
    [[ "$DEFAULT_BUDGET" == off ]] || die "--expect-off needs a boot with THINKING_BUDGETS=off; the readback says budgets are on" ;;
  positive)
    [[ "$DEFAULT_BUDGET" =~ ^[0-9]+$ ]] || die "--positive needs budgets on (readback: $DEFAULT_BUDGET)" ;;
esac

python3 - "$URL" "$MODEL" "$KIND" "$MODE" "$LOW" "$MEDIUM" "$XHIGH" "$DEFAULT_BUDGET" "${DEFAULT_EFFORT:-}" "$SPEC_DEPTH" <<'PY'
import concurrent.futures as cf
import json
import os
import sys
import urllib.error
import urllib.request

URL, MODEL, KIND, MODE = sys.argv[1:5]
LOW, MEDIUM, XHIGH, DEFAULT, DEFAULT_EFFORT = sys.argv[5:10]
DEPTH = int(sys.argv[10])
END_LEN = 1                      # plain </think>: one token on Qwen3.x (we keep it; vllm#58402/#54467)
SLACK = DEPTH + 1 + END_LEN + 4  # drafts past the crossing + bonus token + end sequence + re-tokenizing
BELOW = 8
TIMEOUT = 300
LONG = ("List every prime number between 1 and 400. Check each candidate carefully, one by one, "
        "then state how many there are.")
SHORT = "What is 7 + 5? Reply with just the number."
fails = 0


def headers():
    h = {"Content-Type": "application/json"}
    key = os.environ.get("API_KEY")
    if key:
        h["Authorization"] = "Bearer " + key
        h["x-api-key"] = key
    return h


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers=headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:300].decode(errors='replace')}") from None


def tokens_of(text):
    """The server's own tokenizer on `text` (both engines serve POST /tokenize)."""
    last = None
    for path in ("/tokenize", "/v1/tokenize"):
        try:
            r = post(path, {"model": MODEL, "prompt": text, "add_special_tokens": False})
            c = r.get("count")
            return c if isinstance(c, int) else len(r.get("tokens") or [])
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"cannot count reasoning tokens: no reasoning-token usage and /tokenize failed ({last})")


def reasoning_len(text, reported):
    """Reported reasoning tokens when the engine gives a meaningful number, else /tokenize.
    SGLang defaults usage.reasoning_tokens to 0, so 0 next to non-empty text is not trusted."""
    text = text or ""
    if isinstance(reported, int) and (reported > 0 or not text.strip()):
        return reported, "usage"
    return (tokens_of(text), "tokenize") if text.strip() else (0, "empty")


def chat(extra, prompt=LONG, max_tokens=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or 2048, "stream": False}
    body.update(extra)
    r = post("/v1/chat/completions", body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    text = msg.get("reasoning") or msg.get("reasoning_content") or ""
    u = r.get("usage") or {}
    reported = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reported is None:
        reported = u.get("reasoning_tokens")
    n, src = reasoning_len(text, reported)
    return {"n": n, "src": src, "finish": ch.get("finish_reason"), "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls"), "text": text}


def report(name, ok, detail):
    global fails
    print(("  ✓ " if ok else "  ✗ ") + name + ": " + detail)
    fails += not ok


def about(name, res, budget):
    budget = int(budget)
    lo, hi = max(0, budget - BELOW), budget + SLACK
    ok_len = lo <= res["n"] <= hi
    ok_end = res["finish"] == "stop" and bool(res["content"].strip())
    why = []
    if res["n"] < lo:
        why.append("reasoning ended under the budget on its own — inconclusive, the prompt did not reach it")
    if res["n"] > hi:
        why.append("reasoning ran past the budget")
    if not ok_end:
        why.append(f"finish={res['finish']} content={len(res['content'])} chars (want stop + content)")
    report(name, ok_len and ok_end,
           f"reasoning {res['n']} tok ({res['src']}) vs budget {budget} [{lo}..{hi}], finish={res['finish']}"
           + ("" if not why else " — " + "; ".join(why)))


def safe(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        report(name, False, f"request failed: {exc}")


def messages(extra, max_tokens):
    body = {"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": LONG}]}
    body.update(extra)
    r = post("/v1/messages", body)
    text = "".join(b.get("thinking") or "" for b in r.get("content") or [] if b.get("type") == "thinking")
    answer = "".join(b.get("text") or "" for b in r.get("content") or [] if b.get("type") == "text")
    n, src = reasoning_len(text, None)
    return {"n": n, "src": src, "finish": "stop" if r.get("stop_reason") == "end_turn" else r.get("stop_reason"),
            "content": answer, "text": text}


def responses(effort):
    r = post("/v1/responses", {"model": MODEL, "input": LONG, "reasoning": {"effort": effort},
                               "max_output_tokens": 2048})
    text, answer = "", ""
    for item in r.get("output") or []:
        if item.get("type") == "reasoning":
            for part in (item.get("content") or []) + (item.get("summary") or []):
                text += part.get("text") or ""
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                answer += part.get("text") or ""
    reported = ((r.get("usage") or {}).get("output_tokens_details") or {}).get("reasoning_tokens")
    n, src = reasoning_len(text, reported)
    return {"n": n, "src": src, "finish": "stop" if r.get("status") == "completed" else r.get("status"),
            "content": answer, "text": text}


print(f"--- slack: +{SLACK} tokens (drafter depth {DEPTH} + bonus 1 + </think> {END_LEN} + tokenize 4), -{BELOW} below")

if MODE == "budgets":
    budgets = {"low": LOW, "medium": MEDIUM, "xhigh": XHIGH}
    for effort, b in budgets.items():
        safe(f"effort {effort} (top-level)", lambda e=effort, b=b: about(f"effort {e} (top-level)", chat({"reasoning_effort": e}), b))
    safe("no effort -> the server default", lambda: about(f"no effort -> the server default ({DEFAULT_EFFORT or '?'})", chat({}), DEFAULT))
    kw = "xhigh" if DEFAULT_EFFORT != "xhigh" else "low"
    safe("kwargs-only effort", lambda: about(f"kwargs-only effort {kw}", chat({"chat_template_kwargs": {"reasoning_effort": kw}}), budgets[kw]))
    explicit = (int(LOW) + int(MEDIUM)) // 2 if int(MEDIUM) - int(LOW) > 2 * SLACK else int(XHIGH) + 3 * SLACK
    shapes = [("thinking_token_budget", {"thinking_token_budget": explicit})]
    if KIND == "sglang":
        shapes.insert(0, ("custom_params.thinking_budget", {"custom_params": {"thinking_budget": explicit}}))
    for label, shape in shapes:
        safe(f"explicit {label}", lambda label=label, shape=shape: about(
            f"explicit {label}={explicit} wins over xhigh's {XHIGH}", chat({"reasoning_effort": "xhigh", **shape}), explicit))

    def thinking_off():
        r = chat({"chat_template_kwargs": {"enable_thinking": False}}, prompt=SHORT, max_tokens=256)
        ok = r["n"] == 0 and not r["text"].strip() and "</think>" not in r["content"] and r["finish"] == "stop" and r["content"].strip()
        report("thinking off", bool(ok), f"reasoning {r['n']} tok, '</think>' in content: {'</think>' in r['content']}, "
                                         f"finish={r['finish']}, content={r['content'].strip()[:40]!r}")
    safe("thinking off", thinking_off)

    safe("/v1/messages without budget_tokens", lambda: about(
        "/v1/messages without budget_tokens -> the server default", messages({}, 2048), DEFAULT))

    def msg_explicit():
        # 1024 is the floor of thinking.budget_tokens: vLLM validates ge=1024 and SGLang 400s below it.
        r = messages({"thinking": {"type": "enabled", "budget_tokens": 1024}}, 4096)
        lo, hi = int(XHIGH) + SLACK, 1024 + SLACK
        ok = lo < r["n"] <= hi and r["finish"] == "stop" and r["content"].strip()
        report("/v1/messages budget_tokens=1024 wins over the map", bool(ok),
               f"reasoning {r['n']} tok ({r['src']}), want ({lo}..{hi}] (past xhigh's {XHIGH}, within 1024), finish={r['finish']}")
    safe("/v1/messages budget_tokens", msg_explicit)

    safe("Responses API -> the floor", lambda: about(
        f"Responses API (reasoning.effort=xhigh) -> the floor ({DEFAULT_EFFORT or '?'}'s {DEFAULT}), not the map",
        responses("xhigh"), DEFAULT))

    def concurrent():
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            fl = ex.submit(chat, {"reasoning_effort": "low"})
            fx = ex.submit(chat, {"reasoning_effort": "xhigh"})
            rl, rx = fl.result(), fx.result()
        about("concurrent: the low request", rl, LOW)
        about("concurrent: the xhigh request", rx, XHIGH)
    safe("concurrent", concurrent)

elif MODE == "off":
    def past():
        r = chat({"reasoning_effort": "xhigh"}, max_tokens=4096)
        report("THINKING_BUDGETS=off: xhigh reasons past 256 (no budget)", r["n"] > 256 + SLACK,
               f"reasoning {r['n']} tok ({r['src']}), want > {256 + SLACK}, finish={r['finish']}")
    safe("off", past)
    def past_default():
        r = chat({}, max_tokens=4096)
        report("THINKING_BUDGETS=off: no effort reasons past 64 (no floor)", r["n"] > 64 + SLACK,
               f"reasoning {r['n']} tok ({r['src']}), want > {64 + SLACK}")
    safe("off default", past_default)

else:  # positive
    def natural():
        r = chat({}, prompt=SHORT, max_tokens=int(DEFAULT) + 512)
        ok = 0 < r["n"] < int(DEFAULT) - BELOW and r["finish"] == "stop" and r["content"].strip()
        report("real budgets: a short prompt thinks and ends on its own", bool(ok),
               f"reasoning {r['n']} tok ({r['src']}) under the default budget {DEFAULT}, finish={r['finish']}, "
               f"content={r['content'].strip()[:40]!r}")
    safe("positive", natural)

print(f"probe-effort-budget: {'FAIL' if fails else 'ok'} ({fails} failure(s), mode={MODE})")
sys.exit(1 if fails else 0)
PY
