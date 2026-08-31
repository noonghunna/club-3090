#!/usr/bin/env bash
# capture.sh — shared measurement-capture primitives for the bench tooling.
#
# WHY THIS EXISTS
#   A bench harness's failure mode is not a crash — it is a PLAUSIBLE WRONG
#   NUMBER. `wall_TPS 0.00` printed next to a healthy-looking TTFT; a cumulative
#   cache hit rate that embeds the cold-fill storm and therefore says a bigger
#   pool is worse; a drafter reporting 0.992 acceptance that fired on 5 of ~20
#   requests. Each of those shipped at least once. The primitives here are the
#   ones that catch that class, factored out so `bench.sh` inherits capture that
#   is already guarded by a test suite instead of growing a drifting copy.
#
# DIVISION OF LABOUR
#   offload-matrix.sh = sweep driver (boots a server per arm)
#   bench.sh          = canonical protocol (measures a server someone else booted)
#   capture.sh        = the common measurement layer
#
#   BOTH scripts consume this lib. offload-matrix.sh adopted it once
#   feat/offload-matrix-prefix-cache landed (#826); its inline copies were DELETED
#   rather than left alongside, because two copies of a scrape is how a fixed
#   defect comes back — the sweep had been carrying the weaker pool-LINE-count
#   detector while this lib counted DEVICES-with-a-pool, and a multi-pool device
#   made the two disagree on whether a run was half-cached.
#
#   Where the two callers genuinely need different statistics, the LIB carries both
#   and each caller picks (e.g. acceptance exposes `last=` for the sweep's per-arm
#   TSV column and `mean=` for bench.sh). Fork the statistic, never the scrape.
#
# CONTRACT FOR EVERY FUNCTION HERE
#   1. GRACEFUL DEGRADATION IS MANDATORY. A missing source (no GPU, no server log,
#      no dmon, a VM with no counters, a build with no expert cache) prints exactly
#      one `capture: <thing> unavailable (<why>)` line and returns non-zero. It
#      never aborts the caller and never prints a fabricated value. A capture that
#      cannot run must be visibly absent, not silently zero.
#   2. NEVER AVERAGE AWAY A PER-DEVICE SPLIT. The CUDA0/CUDA1 hit asymmetry on a
#      layer-split MoE (~48% vs ~58% measured 2026-08-01) is routing-entropy
#      signal. Per-device values stay per-device in the OUTPUT, not just in
#      detection.
#   3. DERIVED IS LABELLED DERIVED. `ram_rd_mbps` is misses x expert size / elapsed.
#      It is NOT a hardware counter and must never be presented as one.
#   4. `command grep`, never bare `grep` — grep is shimmed to ugrep in interactive
#      shells on some rigs and the shim breaks exact-output matching.
#   5. `pgrep -x` / kill-by-pid ONLY. `pgrep -f` / `pkill -f` self-match through an
#      agent harness and have killed the calling shell (exit 144).
#
# USAGE
#   source "${ROOT_DIR}/scripts/lib/capture.sh"
#   cap_init                       # resolve pid / log source / GPU count
#   cap_fingerprint                # always-on config fingerprint
#   ...
#
# Env knobs read here (all optional):
#   SERVER_LOG   path to a bare-metal server log (the CONTAINER=none case)
#   CONTAINER    docker container to scrape with `docker logs` (or "none")
#   SERVER_PID   pin the serving process; "none" disables /proc inspection
#   URL          endpoint base, for /props

[[ -n "${_CAPTURE_SH_LOADED:-}" ]] && return 0
_CAPTURE_SH_LOADED=1

# Python decodes reads/stdout/argv with the LOCALE codec; UTF-8 mode (PEP 540)
# overrides that in one move. Defaulted, not forced (see AGENTS.md).
export PYTHONUTF8="${PYTHONUTF8:-1}"

# ---------------------------------------------------------------------------
# degradation notices — one line per missing thing, deduped
# ---------------------------------------------------------------------------
_CAP_NOTED=" "

# cap_note_unavailable <thing> <why>
# The ONLY sanctioned way to report a missing source. Deduped so a source polled
# in a loop cannot bury the report under its own warning.
cap_note_unavailable() {
  local thing="$1" why="${2:-no reason given}"
  case "$_CAP_NOTED" in
    *" ${thing}|"*) return 0 ;;
  esac
  _CAP_NOTED="${_CAP_NOTED}${thing}| "
  echo "capture: ${thing} unavailable (${why})"
  return 0
}

cap_have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# resolution: serving process, log source, GPU count
# ---------------------------------------------------------------------------
CAP_PID=""          # serving process pid, or "" when unknown
CAP_LOG=""          # path to a readable log snapshot, or "" when unavailable
CAP_LOG_SOURCE=""   # "server-log" | "docker" | ""
CAP_NGPU=0
CAP_TMPDIR=""

cap_gpu_count() {
  if cap_have nvidia-smi; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | command grep -c . || echo 0
  else
    echo 0
  fi
}

# cap_server_pid — pid of the local serving process.
#   SERVER_PID=<n>     use it verbatim
#   SERVER_PID=none    disable /proc inspection entirely (hermetic tests, docker)
#   unset              pgrep -x for the known bare-metal server names
# `pgrep -x` matches the executable NAME exactly. `pgrep -f` would match this
# script's own command line and has killed the calling shell before — never use it.
cap_server_pid() {
  local pinned="${SERVER_PID:-}"
  if [[ "$pinned" == "none" ]]; then echo ""; return 1; fi
  if [[ -n "$pinned" ]]; then echo "$pinned"; return 0; fi
  local p name
  for name in llama-server ik_llama-server sglang vllm; do
    p="$(pgrep -x "$name" 2>/dev/null | head -1 || true)"
    [[ -n "$p" ]] && { echo "$p"; return 0; }
  done
  echo ""
  return 1
}

# cap_init [tmpdir] — resolve everything the later captures depend on.
# Safe to call more than once. Never fails the caller.
cap_init() {
  CAP_TMPDIR="${1:-${CAP_TMPDIR:-$(mktemp -d 2>/dev/null || echo /tmp)}}"
  CAP_NGPU="$(cap_gpu_count)"
  CAP_PID="$(cap_server_pid || true)"

  CAP_LOG=""; CAP_LOG_SOURCE=""
  if [[ -n "${SERVER_LOG:-}" ]]; then
    if [[ -r "${SERVER_LOG}" ]]; then
      CAP_LOG="${SERVER_LOG}"; CAP_LOG_SOURCE="server-log"
    else
      cap_note_unavailable "server log" "SERVER_LOG=${SERVER_LOG} is not readable"
    fi
  elif [[ -n "${CONTAINER:-}" && "${CONTAINER:-}" != "none" ]] && cap_have docker \
       && docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    CAP_LOG="${CAP_TMPDIR}/container.log"; CAP_LOG_SOURCE="docker"
    cap_refresh_log
  fi
  return 0
}

# cap_refresh_log — re-snapshot the docker log (a file log needs no refresh).
cap_refresh_log() {
  [[ "$CAP_LOG_SOURCE" == "docker" ]] || return 0
  docker logs "${CONTAINER}" > "$CAP_LOG" 2>&1 || true
}

# cap_log_available <thing> — guard + one degradation notice.
cap_log_available() {
  if [[ -n "$CAP_LOG" && -r "$CAP_LOG" ]]; then return 0; fi
  cap_note_unavailable "${1:-log scrape}" \
    "no server log — pass SERVER_LOG=<path> on bare metal, or CONTAINER=<name> under docker"
  return 1
}

# ---------------------------------------------------------------------------
# argv / env fingerprint (item 11e — solves the config fingerprint without
# asking the operator for anything)
# ---------------------------------------------------------------------------

# cap_proc_argv [pid] — the serving process's own command line, space-joined.
cap_proc_argv() {
  local pid="${1:-$CAP_PID}"
  [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null
}

# cap_proc_env <var> [pid] — one variable out of the process's environment.
# /proc/<pid>/environ is only readable for our own uid; that is the common case
# for a bare-metal server started by the same operator.
cap_proc_env() {
  local var="$1" pid="${2:-$CAP_PID}"
  [[ -n "$pid" && -r "/proc/$pid/environ" ]] || return 1
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
    | command grep -m1 -E "^${var}=" | cut -d= -f2-
}

# cap_argv_flag "<argv>" <flag> [<flag2> ...] — value of the LAST occurrence of
# any of the given flags. `-ct q8_0` and `--cache-type-k q8_0` are the same fact.
cap_argv_flag() {
  local argv="$1"; shift
  local f v=""
  for f in "$@"; do
    # match "<flag> <value>" and "<flag>=<value>"
    v="$(printf '%s ' "$argv" | command grep -oE -- "(^| )${f}[ =][^ ]+" | tail -1 \
         | sed -E "s/^ *${f}[ =]//" || true)"
    [[ -n "$v" ]] && { printf '%s' "$v"; return 0; }
  done
  return 1
}

# cap_argv_has "<argv>" <flag> — presence test (for valueless / prefix flags).
cap_argv_has() {
  local argv="$1" flag="$2"
  printf '%s ' "$argv" | command grep -qE -- "(^| )${flag}([ =]|$)"
}

# cap_argv_fingerprint "<argv>" — the serving flags that change what is measured.
# Deliberately a SUBSET: -m/-ot/-t/-ub/-ct/-ngl/-ts. A full argv echo is noise and
# on some rigs leaks paths into pasted output.
cap_argv_fingerprint() {
  local argv="$1" out="" v
  for pair in "model:-m|--model" "offload:-ot|--override-tensor" "threads:-t|--threads" \
              "ubatch:-ub|--ubatch-size" "kv:-ct|-ctk|--cache-type-k" \
              "ngl:-ngl|--n-gpu-layers" "split:-ts|--tensor-split"; do
    local key="${pair%%:*}" flags="${pair#*:}"
    # shellcheck disable=SC2086
    v="$(cap_argv_flag "$argv" ${flags//|/ } || true)"
    [[ -n "$v" ]] && out="${out}${key}=${v} "
  done
  [[ -n "$out" ]] || return 1
  printf '%s\n' "${out% }"
}

# ---------------------------------------------------------------------------
# /props (llama.cpp-family) — served ctx, slots, model id, drafter
# ---------------------------------------------------------------------------

# cap_props_json [url] — cached /props body, empty when the endpoint has none.
CAP_PROPS=""
CAP_PROPS_TRIED=0
cap_props_json() {
  local url="${1:-${URL:-}}"
  if (( CAP_PROPS_TRIED )); then printf '%s' "$CAP_PROPS"; [[ -n "$CAP_PROPS" ]]; return; fi
  CAP_PROPS_TRIED=1
  [[ -n "$url" ]] || return 1
  CAP_PROPS="$(curl -sf -m 5 "${url}/props" 2>/dev/null || true)"
  [[ -n "$CAP_PROPS" ]] || return 1
  printf '%s' "$CAP_PROPS"
}

# cap_props_get <dotted.path> — one scalar out of /props. vLLM has no /props;
# the caller degrades on a non-zero return.
cap_props_get() {
  local path="$1" body
  body="$(cap_props_json || true)"
  [[ -n "$body" ]] || return 1
  printf '%s' "$body" | CAP_PATH="$path" python3 -c '
import json, os, sys
try:
    obj = json.load(sys.stdin)
except Exception:
    sys.exit(1)
cur = obj
for part in os.environ["CAP_PATH"].split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        sys.exit(1)
if isinstance(cur, (dict, list)):
    sys.exit(1)
sys.stdout.write(str(cur))
' 2>/dev/null
}

# ---------------------------------------------------------------------------
# KV cache type (item 3 — ALWAYS captured, offload or not)
# ---------------------------------------------------------------------------
# Order of authority: argv (what was actually asked for) > /props > boot log.
# Prints "<type> (source: <where>)"; returns 1 and notes unavailability if none
# of the three can answer.
cap_kv_type() {
  local argv="${1:-}" v
  if [[ -n "$argv" ]]; then
    v="$(cap_argv_flag "$argv" -ct -ctk -ctv --cache-type-k --cache-type-v --kv-cache-dtype || true)"
    [[ -n "$v" ]] && { echo "$v (source: argv)"; return 0; }
  fi
  v="$(cap_props_get "default_generation_settings.params.cache_type_k" || true)"
  [[ -n "$v" ]] && { echo "$v (source: /props)"; return 0; }
  if [[ -n "$CAP_LOG" && -r "$CAP_LOG" ]]; then
    v="$(command grep -oE 'type_k *= *[A-Za-z0-9_]+|cache_type_k *= *[A-Za-z0-9_]+' "$CAP_LOG" 2>/dev/null \
         | tail -1 | sed -E 's/.*= *//' || true)"
    [[ -n "$v" ]] && { echo "$v (source: boot log)"; return 0; }
  fi
  # No flag, no /props field, nothing in the log: for llama.cpp-family servers the
  # ENGINE DEFAULT is f16, and "f16 (engine default)" is a fact the reader can act
  # on where "unavailable" is not. Only inferred when the family is identified —
  # /props answering, or llama.cpp markers in the log. Anything else stays honestly
  # unavailable rather than guessing another engine's default.
  if [[ -n "$argv" ]] || [[ -n "$CAP_LOG" ]]; then
    if cap_props_json >/dev/null 2>&1 \
       || { [[ -r "${CAP_LOG:-/dev/null}" ]] \
            && command grep -qE 'llama_context:|llama_model_loader:|llama_kv_cache' "$CAP_LOG" 2>/dev/null; }; then
      echo "f16 (engine default; no -ctk/-ctv in argv)"
      return 0
    fi
  fi
  # NOTE: no cap_note_unavailable here. This function returns a VALUE on stdout,
  # so a notice printed here would be captured by the caller's $(...) and rendered
  # inside its label. The caller emits the notice on a non-zero return.
  return 1
}

# ---------------------------------------------------------------------------
# CPU-offload detection (item 3 — the gate for every moe-cache capture)
# ---------------------------------------------------------------------------
# Two independent signals, either is sufficient:
#   argv     -ot / --override-tensor / --n-cpu-moe / --cpu-moe
#   boot log a "CUDA_Host model buffer" allocation (weights parked in host RAM)
# Echoes the reason; returns 0 when offload is active.
cap_offload_detected() {
  local argv="${1:-}"
  if [[ -n "$argv" ]]; then
    local f
    for f in -ot --override-tensor --n-cpu-moe --cpu-moe -ncmoe; do
      if cap_argv_has "$argv" "$f"; then echo "argv ${f}"; return 0; fi
    done
  fi
  if [[ -n "$CAP_LOG" && -r "$CAP_LOG" ]] \
     && command grep -qE 'CUDA_Host +model buffer|CPU +model buffer size' "$CAP_LOG" 2>/dev/null; then
    echo "boot log (CUDA_Host model buffer)"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# moe-cache CONFIG fingerprint (item 3a)
# ---------------------------------------------------------------------------
# The resulting pool size does NOT identify the arm: the census self-limits below
# the requested cap, so two different `--moe-cache` values can produce the same
# census. Capture what was ASKED FOR alongside what was GOT.
cap_moe_cache_config() {
  local argv="${1:-}" cap="" out="" v k
  [[ -n "$argv" ]] && cap="$(cap_argv_flag "$argv" --moe-cache || true)"
  out="moe_cache_cap=${cap:-<unset>}"
  for k in RESERVE_MB ADMIT_AFTER THROTTLE MAX_BATCH STATS; do
    v="$(cap_proc_env "GGML_CUDA_MOE_CACHE_${k}" || true)"
    [[ -z "$v" ]] && v="$(printenv "GGML_CUDA_MOE_CACHE_${k}" 2>/dev/null || true)"
    out="${out} ${k}=${v:-<unset>}"
  done
  printf '%s\n' "$out"
}

# ---------------------------------------------------------------------------
# VRAM breakdown — where the card's memory actually went
# ---------------------------------------------------------------------------
# Added 2026-08-31 after a DeepSeek-V4-Flash-Vision-Exp run where nvidia-smi read
# 22,612 MiB/card and NOTHING in the bench output said what held it. Two facts
# that only a breakdown surfaces:
#   • `-devd none` (drafter on the HOST) still costs ~7.5 GiB of VRAM across two
#     cards for the DRAFT CONTEXT. "On the host" is not "free".
#   • The moe-cache pool GROWS after boot (5,814 -> 9,248 MiB on CUDA0 over three
#     generations) until it hits the free-minus-reserve floor. A pool figure read
#     at boot is a LOWER BOUND — quoting it understated the warmed pool by 42%.
# One line per device. Components are llama.cpp-family; on other engines those
# lines are absent and only the nvidia-smi total is reported.
cap_vram_breakdown() {                 # $1=log
  local log="${1:-}"
  [[ -r "$log" ]] || return 1
  command -v python3 >/dev/null 2>&1 || return 1
  local _d
  # resolve from BASH_SOURCE, never cwd: a wrong dir here would make the
  # breakdown silently vanish from every run instead of failing loudly
  _d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1
  [[ -r "${_d}/vram_breakdown.py" ]] || return 1
  PYTHONUTF8="${PYTHONUTF8:-1}" python3 "${_d}/vram_breakdown.py" "$log"
}

# ---------------------------------------------------------------------------
# moe-cache log parsing — one python entry point, several modes
# ---------------------------------------------------------------------------
# Line shapes handled (both the fork's current census format and the older one):
#   [moe-cache] enabled: reserve=1536 MiB admit=1/64
#   [moe-cache] CUDA0 pool[0]: type=mxfp4 expert=4352 KiB slots=3038 total=12911 MiB
#   [moe-cache] CUDA1 pool[0]: slots=742 expert=1536 KiB
#   [moe-cache] CUDA0 hits=659183/980562 evictions=81 expert=4352 KiB skips=0 fill-fail=0
#   [moe-cache] MAX_BATCH=8 exceeded, bypassing cache for this batch
#   [moe-cache] CUDA0 has no cache budget after 3072 MiB reserve
# A device can have MULTIPLE pools — slots and bytes are AGGREGATED per device,
# never tailed to the last pool line.
# CAP_LINE_OFFSET windows the scrape to lines written AFTER the offset — set it
# to the log's line count at the start of the measured window and per-request
# scrapes (acceptance, timings, bypass) stop counting the warm-ups and whatever
# the log already held from previous runs. Cumulative COUNTERS deliberately do
# not use it: they are read whole and differenced (see cap_marginal_rates).
cap_moe_parse() {   # $1=mode (census|counters|acceptance|timings|bypass|pooldev) $2=log
  local mode="$1" log="${2:-$CAP_LOG}"
  [[ -n "$log" && -r "$log" ]] || return 1
  CAP_MODE="$mode" python3 - "$log" <<'PY'
import os, re, sys

mode = os.environ.get("CAP_MODE", "")
try:
    off = int(os.environ.get("CAP_LINE_OFFSET", "0"))
except ValueError:
    off = 0
try:
    with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
except OSError:
    sys.exit(1)
if off > 0 and mode in ("acceptance", "timings", "bypass"):
    lines = lines[off:]
# CAP_LINE_LIMIT truncates to the first N lines. Reading the cumulative counters at
# two different limits gives the delta ACROSS that span — which is how a
# decode-window miss count is recovered from a log that carries no timestamps.
try:
    _lim = int(os.environ.get("CAP_LINE_LIMIT", "0"))
except ValueError:
    _lim = 0
if _lim > 0:
    lines = lines[:_lim]

DEV = re.compile(r"\b(CUDA\d+|GPU\d+|ROCm\d+)\b")


def num(pat, s, cast=int):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else None


if mode == "census":
    # per device: pool count, TOTAL slots, TOTAL MiB, mean expert KiB
    per = {}
    for ln in lines:
        if "pool[" not in ln:
            continue
        d = DEV.search(ln)
        if not d:
            continue
        key = d.group(1)
        e = per.setdefault(key, {"pools": 0, "slots": 0, "mib": 0, "expert": [], "type": ""})
        e["pools"] += 1
        s = num(r"slots=(\d+)", ln)
        if s:
            e["slots"] += s
        t = num(r"total=(\d+)\s*MiB", ln)
        if t:
            e["mib"] += t
        x = num(r"expert=(\d+)\s*KiB", ln)
        if x:
            e["expert"].append(x)
        ty = re.search(r"type=(\S+)", ln)
        if ty and not e["type"]:
            e["type"] = ty.group(1)
    if not per:
        sys.exit(1)
    for k in sorted(per):
        e = per[k]
        exp = round(sum(e["expert"]) / len(e["expert"])) if e["expert"] else 0
        print(f"{k} pools={e['pools']} slots={e['slots']} total_mib={e['mib']} "
              f"expert_kib={exp} type={e['type'] or '-'}")

elif mode == "counters":
    # LAST cumulative counter emission per device (hits/total/evictions + health)
    per = {}
    for ln in lines:
        if "hits=" not in ln:
            continue
        d = DEV.search(ln)
        if not d:
            continue
        m = re.search(r"hits=(\d+)/(\d+)", ln)
        if not m:
            continue
        e = per.setdefault(d.group(1), {})
        e["hits"], e["total"] = int(m.group(1)), int(m.group(2))
        for name, pat in (("evictions", r"evictions=(\d+)"), ("skips", r"skips=(\d+)"),
                          ("admission", r"admission=(\d+)"),
                          ("fill_fail", r"fill-fail=(\d+)"),
                          ("dispatch_fail", r"dispatch-fail=(\d+)"),
                          ("collect_fail", r"collect-fail=(\d+)")):
            v = num(pat, ln)
            if v is not None:
                e[name] = v
    if not per:
        sys.exit(1)
    for k in sorted(per):
        e = per[k]
        print(f"{k} hits={e.get('hits',0)} total={e.get('total',0)} "
              f"evictions={e.get('evictions',0)} skips={e.get('skips',0)} "
              f"admission={e.get('admission',0)} "
              f"fill_fail={e.get('fill_fail',0)} dispatch_fail={e.get('dispatch_fail',0)} "
              f"collect_fail={e.get('collect_fail',0)}")

elif mode == "acceptance":
    # Acceptance WITHOUT a fire rate is a deception: measured 2026-08-01, a drafter
    # reported 0.992 acceptance while firing on 5 of ~20 requests (zero on novel
    # content). Report how OFTEN it fired alongside how WELL it did when it fired.
    acc, drafted, accepted = [], 0, 0
    for ln in lines:
        m = re.search(r"draft acceptance(?: rate)?\s*=\s*([0-9.]+)", ln)
        if m:
            acc.append(float(m.group(1)))
        m2 = re.search(r"\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*(?:generated|drafted)", ln)
        if m2:
            accepted += int(m2.group(1))
            drafted += int(m2.group(2))
    if not acc:
        sys.exit(1)
    # `last` is the final acceptance the log carries. It exists because the sweep's
    # TSV `accept` column has always meant exactly that — a per-arm boot ends with
    # its own last value — and swapping in the mean would silently redefine every
    # historical row. bench.sh quotes the mean; the sweep quotes last. Same scrape.
    print(f"fired={len(acc)} last={acc[-1]:.3f} mean={sum(acc)/len(acc):.3f} "
          f"min={min(acc):.3f} max={max(acc):.3f} accepted={accepted} drafted={drafted}")

elif mode == "timings":
    # llama.cpp per-request print_timing. `prompt eval time` is PREFILL; the bare
    # `eval time` is DECODE. Matching the wrong one reports prefill as decode.
    #
    # CAP_MIN_DECODE_TOKENS drops decode entries from very short completions. The
    # prefill probe and the PP fallback ask for ~16 tokens, and a decode rate over
    # 16 tokens is dominated by start-up effects — leaving them in drags the median
    # and makes the client-vs-engine cross-check flag a divergence that is really
    # just two different populations of request.
    try:
        min_dec = int(os.environ.get("CAP_MIN_DECODE_TOKENS", "0"))
    except ValueError:
        min_dec = 0
    # A bench drives two very different prefill populations: the short canonical
    # prompts (tens of tokens) and the deep haystack probes (thousands). Their
    # rates differ by ~4x, so one mean/median over both describes neither. Split
    # at CAP_PREFILL_SPLIT_TOKENS and report each.
    try:
        split_at = int(os.environ.get("CAP_PREFILL_SPLIT_TOKENS", "2000"))
    except ValueError:
        split_at = 2000
    dec, pre, pre_long, dropped = [], [], [], 0
    for ln in lines:
        m = re.search(r"([0-9.]+)\s+tokens per second", ln)
        if not m:
            continue
        v = float(m.group(1))
        ntok = re.search(r"/\s*(\d+)\s+tokens", ln)
        ntok = int(ntok.group(1)) if ntok else None
        if "prompt eval time" in ln:
            if ntok is not None and ntok >= split_at:
                pre_long.append(v)
            else:
                pre.append(v)
        elif "eval time" in ln:
            if min_dec and ntok is not None and ntok < min_dec:
                dropped += 1
                continue
            dec.append(v)
    if not dec and not pre and not pre_long:
        sys.exit(1)

    def emit(name, xs):
        if not xs:
            print(f"{name} n=0")
            return
        xs = sorted(xs)
        mid = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
        print(f"{name} n={len(xs)} mean={sum(xs)/len(xs):.2f} median={mid:.2f} "
              f"min={xs[0]:.2f} max={xs[-1]:.2f}")

    emit("decode", dec)
    emit(f"prefill_short(<{split_at}tok)", pre)
    if pre_long:
        emit(f"prefill_deep(>={split_at}tok)", pre_long)
    if dropped:
        print(f"dropped_short_decode n={dropped} floor={min_dec}")

elif mode == "bypass":
    print(sum(1 for ln in lines if "bypassing cache" in ln))

elif mode == "pooldev":
    # #824: '[moe-cache] enabled:' still prints when only SOME devices got a pool.
    # Ground truth is the pool-line COUNT vs the device count, never the banner.
    devs = {DEV.search(ln).group(1) for ln in lines
            if "pool[" in ln and DEV.search(ln)}
    nobudget = sum(1 for ln in lines if "no cache budget" in ln)
    enabled = 1 if any("[moe-cache] enabled:" in ln for ln in lines) else 0
    print(f"devices={len(devs)} pool_lines={sum(1 for ln in lines if 'pool[' in ln)} "
          f"enabled={enabled} nobudget={nobudget} devlist={','.join(sorted(devs)) or '-'}")

else:
    sys.exit(2)
PY
}

# cap_marginal_rates <snap-start> <snap-end>
# Item 3b. CUMULATIVE hit rates embed the cold-fill phase, so they are NOT
# boot-comparable — a BIGGER pool shows a LOWER cumulative rate on the same
# traffic because it spent longer filling. The MARGINAL rate across the measured
# window is the only number two boots can be compared on. Per device, never
# averaged (item 3d).
cap_marginal_rates() {
  local s0="$1" s1="$2"
  [[ -r "$s0" && -r "$s1" ]] || return 1
  python3 - "$s0" "$s1" <<'PY'
import sys


def load(p):
    out = {}
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.split()
            if not parts:
                continue
            out[parts[0]] = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
if not b:
    sys.exit(1)
for dev in sorted(b):
    e1 = b[dev]
    e0 = a.get(dev, {})
    h1, t1 = int(e1.get("hits", 0)), int(e1.get("total", 0))
    h0, t0 = int(e0.get("hits", 0)), int(e0.get("total", 0))
    dh, dt = max(h1 - h0, 0), max(t1 - t0, 0)
    de = max(int(e1.get("evictions", 0)) - int(e0.get("evictions", 0)), 0)
    cum = f"{h1*100.0/t1:.1f}%" if t1 else "-"
    marg = f"{dh*100.0/dt:.1f}%" if dt else "-"
    print(f"{dev} marginal={marg} cumulative={cum} dhits={dh} dlookups={dt} devictions={de}")
PY
}

# cap_cumulative_rates <counters-file> — lifetime hit rate per device at the
# moment the snapshot was taken. Only useful as the START of a "hits X% -> Y%"
# pair on a card; for comparing BOOTS use the marginal rate (cap_marginal_rates).
cap_cumulative_rates() {
  local snap="$1"
  [[ -r "$snap" ]] || return 1
  awk '{
    dev = $1; hits = 0; total = 0
    for (i = 2; i <= NF; i++) {
      if ($i ~ /^hits=/)  { split($i, a, "="); hits  = a[2] }
      if ($i ~ /^total=/) { split($i, b, "="); total = b[2] }
    }
    if (total > 0) printf "%s cumulative=%.1f\n", dev, hits * 100 / total
  }' "$snap"
}

# cap_health_verdict <counters-file> — the PASS condition is the HARD-FAILURE
# counters ONLY:
#   fill-fail / dispatch-fail / collect-fail  -> real defects; any non-zero FAILS
#   skips / admission                         -> INFORMATIONAL load metrics
# `skips` is the engine's `insert_skips`, whose dominant increment is admission
# THROTTLE / queue exhaustion (inserts_left <= 0 || queue full). Under a low
# ADMIT_AFTER every miss is an insertion candidate and THROTTLE caps insertions per
# step, so non-zero skips is NORMAL steady-state admission pressure. A live run on
# a healthy server showed skips at 0.5% of misses, tracking the per-device miss
# asymmetry exactly. The original spec said "all-zero is the pass condition" — the
# live run disproved it, so this reports skips as a THROTTLE tuning signal rather
# than failing an otherwise healthy run.
cap_health_verdict() {
  local snap="$1"
  [[ -r "$snap" ]] || return 1
  python3 - "$snap" <<'PY'
import sys

HARD = ("fill_fail", "dispatch_fail", "collect_fail")
INFO = ("skips", "admission")
bad, info = [], []
with open(sys.argv[1], encoding="utf-8") as fh:
    for ln in fh:
        parts = ln.split()
        if not parts:
            continue
        kv = dict(x.split("=", 1) for x in parts[1:] if "=" in x)
        nz = {k: v for k, v in kv.items() if k in HARD and v not in ("0", "-")}
        if nz:
            bad.append(f"{parts[0]}: " + " ".join(f"{k}={v}" for k, v in sorted(nz.items())))
        ni = {k: v for k, v in kv.items() if k in INFO and v not in ("0", "-")}
        if ni:
            info.append(f"{parts[0]} " + " ".join(f"{k}={v}" for k, v in sorted(ni.items())))
if bad:
    print("FAIL — " + "; ".join(bad))
else:
    tail = ""
    if info:
        tail = ("  [informational: " + "; ".join(info) + " — admission throttling, expected under "
                "low ADMIT_AFTER + steady-state eviction pressure; a THROTTLE tuning signal, "
                "not a defect]")
    print("PASS (no fill/dispatch/collect failures)" + tail)
PY
}

# ---------------------------------------------------------------------------
# VRAM triplet + leak delta (item 11b) — a free soak-lite leak check per run
# ---------------------------------------------------------------------------

# cap_vram_used — total MiB across all GPUs (one integer).
cap_vram_used() {
  cap_have nvidia-smi || return 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF{s+=$1} END{if(NR) print s+0; else exit 1}'
}

# cap_vram_per_device — "GPU<i> <MiB>" per line.
cap_vram_per_device() {
  cap_have nvidia-smi || return 1
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF{printf "GPU%d %d\n", NR-1, $1}'
}

# cap_vram_peak_start <outfile> — background sampler; echoes its pid.
cap_vram_peak_start() {
  local out="$1"
  cap_have nvidia-smi || return 1
  : > "$out"
  ( local peak=0 v
    while :; do
      v="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk 'NF{s+=$1} END{print s+0}')"
      if [[ -n "$v" ]] && (( v > peak )); then peak="$v"; echo "$peak" > "$out"; fi
      sleep 2
    done ) >/dev/null 2>&1 &
  echo $!
}

cap_stop_pid() {   # kill by pid ONLY — never pkill -f (self-match footgun)
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  return 0
}

# ---------------------------------------------------------------------------
# PCIe: measured used (dmon) + queried available (link state) — item 10
# ---------------------------------------------------------------------------

# cap_dmon_start <outfile> — 1 Hz per-GPU utilisation/PCIe stream, EPOCH-STAMPED
# so the samples can be sliced by phase afterwards. Echoes the pid to stop.
# `$!` on a pipeline is the LAST element (the stamping loop); killing it SIGPIPEs
# nvidia-smi on its next write, so one kill tears down both.
cap_dmon_start() {
  local out="$1"
  # Echoes a PID on stdout — the caller emits the notice (see cap_kv_type).
  cap_have nvidia-smi || return 1
  : > "$out"
  local sb=""
  cap_have stdbuf && sb="stdbuf -oL"
  # shellcheck disable=SC2086
  $sb nvidia-smi dmon -s ut -d 1 2>/dev/null \
    | while IFS= read -r line; do printf '%s %s\n' "$(date +%s)" "$line"; done > "$out" &
  echo $!
}

# cap_dmon_phase <dmon-file> <phase-file> <label>
# Per-phase means AND PEAKS. Decode and prefill PCIe traffic differ by ~100x on
# an offloaded MoE, so a whole-run mean describes neither; and 1 Hz sampling
# under-reads bursts, which is why the peak is reported next to the mean rather
# than instead of it. Per-device rows are kept (item 3d).
#
# The phase file holds MULTIPLE windows per label (narrative + code are both
# decode; each prefill depth is its own window). Only sample times inside one of
# that label's windows are counted, so the warm-ups and the gaps between shapes
# — which are not the phase — cannot dilute it.
cap_dmon_phase() {
  local file="$1" phases="$2" label="${3:-decode}"
  [[ -r "$file" && -r "$phases" ]] || return 1
  awk -v label="$label" '
    # pass 1: this label’s measured windows
    NR == FNR { if ($1 == label) { lo[++k] = $2 + 0; hi[k] = $3 + 0 } ; next }
    # pass 2: stamped dmon rows
    # <epoch> <gpu> <sm> <mem> <enc> <dec> <jpg> <ofa> <rxpci> <txpci>
    $2 ~ /^#/ { next }
    NF < 10   { next }
    {
      ts = $1 + 0; inwin = 0
      for (i = 1; i <= k; i++) if (ts >= lo[i] && ts <= hi[i]) { inwin = 1; break }
      if (!inwin) next
      g = $2 + 0
      sm = ($3 ~ /^[0-9.]+$/) ? $3 + 0 : 0
      mc = ($4 ~ /^[0-9.]+$/) ? $4 + 0 : 0
      rx = ($9 ~ /^[0-9.]+$/) ? $9 + 0 : 0
      tx = ($10 ~ /^[0-9.]+$/) ? $10 + 0 : 0
      n[g]++; srx[g]+=rx; stx[g]+=tx; ssm[g]+=sm; smc[g]+=mc
      if (rx > prx[g]) prx[g] = rx
      if (tx > ptx[g]) ptx[g] = tx
      N++; SRX+=rx; STX+=tx; SSM+=sm; SMC+=mc
      if (rx > PRX) PRX = rx
      if (tx > PTX) PTX = tx
    }
    END {
      if (!N) exit 1
      printf "%s all rx_mean=%.0f rx_peak=%.0f tx_mean=%.0f tx_peak=%.0f sm_mean=%.1f memctl_mean=%.1f samples=%d\n",
             label, SRX/N, PRX, STX/N, PTX, SSM/N, SMC/N, N
      for (g in n)
        printf "%s GPU%d rx_mean=%.0f rx_peak=%.0f tx_mean=%.0f tx_peak=%.0f sm_mean=%.1f memctl_mean=%.1f samples=%d\n",
               label, g, srx[g]/n[g], prx[g], stx[g]/n[g], ptx[g], ssm[g]/n[g], smc[g]/n[g], n[g]
    }' "$phases" "$file"
}

# cap_pcie_link — queried AVAILABLE link state, per device.
# Prints: "GPU<i> gen=<cur>/<max> width=<cur>/<max>".
cap_pcie_link() {
  cap_have nvidia-smi || return 1
  nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
             --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' 'NF>=4 && $1 ~ /[0-9]/ {printf "GPU%d gen=%s/%s width=%s/%s\n", NR-1, $1, $2, $3, $4}'
}

# cap_pcie_link_sampler_start <outfile> — link state sampled UNDER LOAD.
# An IDLE link downshifts to Gen1 x16 by design (ASPM), so an idle reading is a
# false alarm generator. The warn condition (current < max) is only meaningful
# while traffic is flowing, so we sample throughout the measured window and take
# the BEST observed state as "what this link actually negotiates under load".
cap_pcie_link_sampler_start() {
  local out="$1"
  cap_have nvidia-smi || return 1
  : > "$out"
  ( while :; do cap_pcie_link >> "$out" 2>/dev/null; sleep 3; done ) >/dev/null 2>&1 &
  echo $!
}

# cap_pcie_link_under_load <samplefile> — best observed gen/width per device,
# plus a WARN when it never reached the link's own maximum while under load
# (the silently-trained-at-x8 trap, PCIE_P2P.md §8 row 2).
cap_pcie_link_under_load() {
  local f="$1"
  [[ -s "$f" ]] || return 1
  python3 - "$f" <<'PY'
import re, sys

best = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    for ln in fh:
        m = re.match(r"(GPU\d+) gen=(\d+)/(\d+) width=(\d+)/(\d+)", ln.strip())
        if not m:
            continue
        dev, g, gm, w, wm = m.group(1), *map(int, m.groups()[1:])
        cur = best.get(dev)
        if cur is None or (g, w) > (cur[0], cur[2]):
            best[dev] = (g, gm, w, wm)
if not best:
    sys.exit(1)
for dev in sorted(best):
    g, gm, w, wm = best[dev]
    warn = ""
    if g < gm or w < wm:
        warn = (f"  ⚠ WARN: never reached gen{gm} x{wm} under load — link is training "
                f"below its own maximum (check slot/riser/BIOS bifurcation)")
    print(f"{dev} gen={g}/{gm} width={w}/{wm}{warn}")
PY
}

# cap_pcie_practical_mbps <gen> <width>
# Denominator for a utilisation percentage, ALWAYS stated next to the number.
# Per-lane raw throughput after line coding: gen1 250, gen2 500, gen3 984.6,
# gen4 1969, gen5 3938 MB/s. x0.85 for realistic large-transfer efficiency.
cap_pcie_practical_mbps() {
  local gen="${1:-0}" width="${2:-0}"
  awk -v g="$gen" -v w="$width" 'BEGIN{
    split("250 500 984.6 1969 3938 7877", L, " ")
    if (g < 1 || g > 6 || w < 1) exit 1
    printf "%.0f\n", L[g] * w * 0.85
  }'
}

# ---------------------------------------------------------------------------
# CPU + system RAM (items 8, 11d)
# ---------------------------------------------------------------------------
cap_cpu_snap() { awk '/^cpu /{i=$5;t=0;for(j=2;j<=NF;j++)t+=$j;print t,i}' /proc/stat 2>/dev/null; }

# cap_cpu_pct "<tot0 idle0>" "<tot1 idle1>" — busy% across the window.
# cpu_pct is the missing half of offload bottleneck forensics: on a CPU-offloaded
# MoE a low sm% with a high cpu% says the GPU is waiting on the host, not the
# other way round.
cap_cpu_pct() {
  local a="$1" b="$2"
  awk -v a="$a" -v b="$b" 'BEGIN{
    split(a, A, " "); split(b, B, " ")
    dt = B[1] - A[1]; di = B[2] - A[2]
    if (dt <= 0) exit 1
    printf "%.1f\n", (1 - di/dt) * 100
  }'
}

# cap_ram_status [pid] — item 8. Host RAM is the requirement launchers cannot
# detect (the reason CPU-offload slugs land 🐣), and it is two lines from /proc.
# The swap leg is the free integrity item: serving-process pages in swap means
# every number in the run is suspect and nothing else notices.
cap_ram_status() {
  local pid="${1:-$CAP_PID}"
  # Returns a VALUE on stdout — the caller emits the notice (see cap_kv_type).
  [[ -r /proc/meminfo ]] || return 1
  local rss="" vmswap=""
  if [[ -n "$pid" && -r "/proc/$pid/status" ]]; then
    rss="$(command grep -m1 '^VmRSS:' "/proc/$pid/status" 2>/dev/null | awk '{print $2}')"
    vmswap="$(command grep -m1 '^VmSwap:' "/proc/$pid/status" 2>/dev/null | awk '{print $2}')"
  fi
  CAP_RSS_KB="$rss" CAP_SWAP_KB="$vmswap" awk '
    /^MemTotal:/     {tot=$2}
    /^MemAvailable:/ {avail=$2}
    /^SwapTotal:/    {stot=$2}
    /^SwapFree:/     {sfree=$2}
    END {
      rss   = ENVIRON["CAP_RSS_KB"]
      vmsw  = ENVIRON["CAP_SWAP_KB"]
      printf "mem_total_gib=%.1f mem_available_gib=%.1f", tot/1048576, avail/1048576
      if (rss != "") printf " server_rss_gib=%.1f", rss/1048576
      printf " swap_total_gib=%.1f swap_used_gib=%.1f", stot/1048576, (stot-sfree)/1048576
      if (vmsw != "") printf " server_swap_mib=%.0f", vmsw/1024
      printf "\n"
    }' /proc/meminfo
}

# cap_swap_verdict [pid] — PASS/FAIL for the integrity panel.
cap_swap_verdict() {
  local pid="${1:-$CAP_PID}"
  if [[ -n "$pid" && -r "/proc/$pid/status" ]]; then
    local vmswap
    vmswap="$(command grep -m1 '^VmSwap:' "/proc/$pid/status" 2>/dev/null | awk '{print $2+0}')"
    if [[ -n "$vmswap" ]] && (( vmswap > 0 )); then
      echo "FAIL — ${vmswap} kB of the SERVING PROCESS is paged out to swap; every number in this run is suspect"
      return 0
    fi
    echo "PASS (no serving-process pages in swap)"
    return 0
  fi
  local used
  used="$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{print t-f}' /proc/meminfo 2>/dev/null)"
  [[ -n "$used" ]] || return 1
  if (( used > 0 )); then
    echo "UNKNOWN — ${used} kB host-wide swap in use, but the serving pid is unknown (set SERVER_PID=<pid>) so it cannot be attributed"
  else
    echo "PASS (host swap unused)"
  fi
}

# ---------------------------------------------------------------------------
# Derived RAM-read demand (item 9a) — DERIVED, NOT A PERF COUNTER
# ---------------------------------------------------------------------------
# Direct IMC/uncore counters are unreadable in VM guests and vendor/root-gated on
# bare metal, so they are not portable. This mechanises the hand-analysis instead:
# every expert MISS reads one expert's worth of weights out of host RAM, so
#   demand = miss_count x avg_expert_bytes / elapsed
# It is a LOWER BOUND on the miss path only (it ignores KV traffic, activations
# and prefetch). Never present it as a measured bandwidth.
cap_ram_rd_mbps() {
  local misses="${1:-0}" expert_kib="${2:-0}" elapsed="${3:-0}"
  awk -v m="$misses" -v k="$expert_kib" -v e="$elapsed" 'BEGIN{
    if (m <= 0 || k <= 0 || e <= 0) exit 1
    printf "%.0f\n", m * k / 1024 / e
  }'
}

# cap_stream_triad_gbps [seconds] — item 9b. A ~3 s STREAM-triad ceiling on the
# IDLE box, so the derived demand above can be stated as a FRACTION of what this
# host can actually deliver ("~29 of ~99 GB/s"). OPT-IN: it perturbs nothing by
# itself but it is not free, and the contention probe (co-running STREAM during
# decode) is deliberately NOT here — that one perturbs the run by construction
# and belongs in a separate opt-in arm.
cap_stream_triad_gbps() {
  local secs="${1:-3}" out rc
  out="$(python3 - "$secs" 2>/dev/null <<'PY'
import os, sys, time
try:
    import numpy as np
except ImportError:
    sys.exit(3)

secs = float(sys.argv[1])

# MULTI-THREADED on purpose. numpy element-wise ops are single-threaded, and one
# core cannot saturate a modern memory controller — a single-threaded triad reads
# ~4 GB/s on a host whose real ceiling is ~100 GB/s. Reporting that as "the
# ceiling" would make the derived demand look like ~90% utilisation instead of
# ~30%, which is worse than reporting nothing.
try:
    ncpu = len(os.sched_getaffinity(0))
except AttributeError:
    ncpu = os.cpu_count() or 4
workers = max(1, min(ncpu, 16))

# Size the arrays past the LLC but inside a fraction of AVAILABLE RAM: this runs
# next to a loaded serving process, and a calibration that triggers swap would
# poison the very run it is meant to characterise.
avail = 0
try:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("MemAvailable:"):
                avail = int(ln.split()[1]) * 1024
                break
except OSError:
    pass
budget = min(2 * 1024 ** 3, avail // 8) if avail else 512 * 1024 ** 2
per_worker = max(48 * 1024 ** 2, budget // workers)
n = max(1 << 20, int(per_worker // (3 * 8)))     # 3 float64 arrays per worker


def _triad(q, n, secs):
    a = np.zeros(n); b = np.ones(n); c = np.full(n, 2.0)
    a[:] = b + 3.0 * c                            # fault the pages in before timing
    iters, t0 = 0, time.perf_counter()
    end = t0 + secs
    while time.perf_counter() < end:
        a[:] = b + 3.0 * c
        iters += 1
    dt = time.perf_counter() - t0
    # SUSTAINED, not best-iteration: triad moves 24 B per element (read b, read c,
    # write a). Best-of understates the cost of the run it is calibrating.
    q.put(24.0 * n * iters / dt / 1e9 if dt > 0 else 0.0)


try:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_triad, args=(q, n, secs)) for _ in range(workers)]
    for p in procs:
        p.start()
    total = sum(q.get(timeout=secs + 60) for _ in procs)
    for p in procs:
        p.join(timeout=10)
except Exception:
    # No fork (some containers), or the pool failed: one worker still gives a
    # number, it is just a floor on the ceiling. Labelled as such by the caller.
    import queue
    q = queue.Queue()
    _triad(q, n, secs)
    total, workers = q.get(), 1

print(f"{total:.1f} {workers}")
PY
  )"
  rc=$?
  if (( rc == 0 )) && [[ -n "$out" ]]; then
    printf '%s\n' "$out"
    return 0
  fi
  # Returns a VALUE on stdout — the caller emits the notice (see cap_kv_type),
  # distinguishing the numpy case by this return code.
  (( rc == 3 )) && return 3
  return 1
}

# ---------------------------------------------------------------------------
# Status enum (item 11c) — imported wholesale from offload-matrix
# ---------------------------------------------------------------------------
# Richer than "warn when TPS == 0": each state names the specific way the run is
# not measuring what it claims to.
#   OK              nothing below fired
#   NO_TOKENS       zero tokens / zero TPS — the scrape or the run failed silently
#   REQ_ERRORS      one or more requests errored
#   CACHE_DISABLED  a cache was REQUESTED but never allocated (incl. per-device)
#   INVALID_BYPASS  the cache exists but a decode declined it (bypass counter > 0)
# Precedence is deliberate, strongest last:
#   INVALID_BYPASS < CACHE_DISABLED < NO_TOKENS < REQ_ERRORS
# NO_TOKENS outranks the cache states — a run with no tokens has nothing to say
# about a cache. REQ_ERRORS outranks NO_TOKENS in turn, because a run that errored
# EXPLAINS why it produced no tokens; reporting NO_TOKENS there would send the
# reader hunting for a scrape bug that is not the cause. This ordering is the
# sweep's original one, adopted verbatim so per-arm rows keep their meaning.
cap_status_classify() {   # $1=tokens $2=tps $3=req_errors $4=cache_requested $5=cache_ok $6=bypass
  local toks="${1:-0}" tps="${2:-0}" errs="${3:-0}" want="${4:-0}" got="${5:-1}" byp="${6:-0}"
  local status=OK
  if (( byp > 0 )); then status=INVALID_BYPASS; fi
  if [[ "$want" == "1" && "$got" != "1" ]]; then status=CACHE_DISABLED; fi
  case "${toks:-0}" in ''|'-'|0|0.0|0.00) status=NO_TOKENS ;; esac
  case "${tps:-0}" in ''|'-'|0|0.0|0.00) status=NO_TOKENS ;; esac
  # LAST, so it outranks NO_TOKENS: a run that errored explains its own zero.
  if [[ "${errs:-0}" != "0" && -n "${errs:-}" ]]; then status=REQ_ERRORS; fi
  echo "$status"
}

# ---------------------------------------------------------------------------
# Prompt-processing plausibility gate (#817)
# ---------------------------------------------------------------------------
# The `PP tok/s` line in bench.sh's summaries is SCRAPED from the engine's own
# stats log — vLLM's `Avg prompt throughput`, which is averaged over its ~10 s
# logging window (llama.cpp's `prompt eval time` equivalent goes through the same
# path). That is a windowed figure over whatever the engine happened to be doing,
# and on the canonical bench prompts — ~15 tokens — most of the window contains no
# prefill at all. It renders as `PP tok/s  mean=2.00  CV=55.9%` (#769) or
# `mean=6.88  CV=19.7%` (#822): shaped exactly like a measurement, and not one.
# Three community reports have now pasted it as data.
#
# The trustworthy prefill number is the CLIENT-side one bench.sh already prints by
# default: prompt_tokens / TTFT over a cache-busted haystack (`prefill tok/s`,
# CV 0.3-1.5% on the very same runs).
#
# So the scrape only prints when it could be a prefill rate at all. Three
# conditions, each one an observed failure shape rather than a guess:
#   prompt >= 1000 tok   a ~15-token prompt cannot fill a 10 s window (#769, #822)
#   mean   >= 50 tok/s   below that it is window dilution, not prefill compute
#   CV     <= 100%       the windowed variant inside the prefill sections reads
#                        `mean=1127.30 CV=171.9%`, min 13 / max 8044 — a sample
#                        straddling idle and busy windows is not a rate
# ALL THREE must hold. The CV condition is load-bearing on its own: the prefill-
# section variant passes the first two and is still garbage.
#
# The caller prints the reason whenever the gate refuses — a number that vanishes
# without explanation is its own defect, and the next report would just ask where
# `PP tok/s` went.
#
# ONE DEFINITION, ON PURPOSE. bench.sh has two print sites and shells out to this
# function for both rather than reimplementing the thresholds; two copies of a
# rule is how the two callers of a shared scrape drifted apart before (see the
# pool-LINE-count vs DEVICES-with-a-pool split at the top of this file).
CAP_PP_MIN_PROMPT_TOKS="${CAP_PP_MIN_PROMPT_TOKS:-1000}"
CAP_PP_MIN_TPS="${CAP_PP_MIN_TPS:-50}"
CAP_PP_MAX_CV="${CAP_PP_MAX_CV:-100}"

# cap_pp_plausible <mean_tok_s> <cv_pct> <prompt_tokens>
#   exit 0 — plausible; prints nothing
#   exit 1 — suppress; prints the reason on stdout
cap_pp_plausible() {
  awk -v m="${1:-0}" -v cv="${2:-0}" -v pt="${3:-0}" \
      -v minpt="${CAP_PP_MIN_PROMPT_TOKS:-1000}" \
      -v mintps="${CAP_PP_MIN_TPS:-50}" \
      -v maxcv="${CAP_PP_MAX_CV:-100}" 'BEGIN{
    if (pt + 0 < minpt + 0) {
      printf "prompt is %d tok (< %d) - the engine averages prompt throughput over a ~10 s window, and a prompt this short cannot fill one\n", pt, minpt
      exit 1
    }
    if (m + 0 < mintps + 0) {
      printf "scraped mean %.2f tok/s (< %d) - window dilution, not prefill compute\n", m, mintps
      exit 1
    }
    if (cv + 0 > maxcv + 0) {
      printf "scraped CV %.1f%% (> %d%%) - the sample straddles idle and busy windows, which is not a rate\n", cv, maxcv
      exit 1
    }
    exit 0
  }'
}

# cap_divergence_pct <a> <b> — |a-b| / max(a,b) as a percentage; the engine-side
# vs client-observed cross-check (item 1b). Anything over 5% means one of the two
# numbers is not measuring what its label says.
cap_divergence_pct() {
  awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{
    if (a <= 0 || b <= 0) exit 1
    d = a > b ? a - b : b - a
    m = a > b ? a : b
    printf "%.1f\n", d * 100 / m
  }'
}
