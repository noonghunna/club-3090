#!/usr/bin/env bash
# test-compose-draft-sample-method.sh — Qwen3.8 / ThinkingCap vLLM spec-decode composes
# must pass `draft_sample_method` into --speculative-config, from a validated knob.
#
# WHY: vLLM defaults draft sampling to "greedy", which scores every draft as one-hot in
# the rejection test. "probabilistic" samples drafts at the request's temperature and
# keeps their logits, so more drafts are accepted whenever temperature > 0, with the
# same output distribution. Measured 2026-09-26 on ThinkingCap dual-fast / superfast
# (230 W, T=0.6): +5-6% decode. The value is spliced into a hand-built JSON string in
# each entrypoint, so this guard also proves every variant still parses as JSON.
#   - static: every compose with --speculative-config carries the key, declares
#     DRAFT_SAMPLE_METHOD bare (unset must stay unset, not arrive EMPTY), and has the
#     greedy|probabilistic guard; its JSON parses once the shell variables are filled;
#   - delivery: render one MTP and one DFlash compose and run the real guard block:
#     unset -> probabilistic, greedy passes, junk refuses to boot (NEGATIVE CONTROL).
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

out=$(python3 - <<'PY'
import glob, json, re
files = sorted(glob.glob("models/qwen3.8-27b/vllm/compose/**/*.yml", recursive=True)
               + glob.glob("models/thinkingcap-qwen3.8-27b/vllm/compose/**/*.yml", recursive=True))
checked, bad = 0, []
for f in files:
    if "/_archive/" in f:
        continue
    s = open(f, encoding="utf-8").read()
    lines = re.findall(r'SPEC_ARGS=\(--speculative-config "(\{[^\n]*\})"\)', s)
    if "--speculative-config" not in s:
        continue
    checked += 1
    if not lines:
        bad.append(f"{f}: --speculative-config is not built as SPEC_ARGS=(--speculative-config \"{{...}}\")"); continue
    for raw in lines:
        if "draft_sample_method" not in raw:
            bad.append(f"{f}: speculative-config JSON lacks draft_sample_method")
        filled = (raw.replace('\\"', '"').replace("$$_spec_n", "4")
                     .replace("$$_dsm", "probabilistic").replace("$$DRAFTER", "/drafter"))
        try:
            cfg = json.loads(filled)
            if cfg.get("draft_sample_method") != "probabilistic":
                bad.append(f"{f}: draft_sample_method is not wired to $$_dsm")
        except ValueError as e:
            bad.append(f"{f}: speculative-config is not valid JSON once filled ({e})")
    if not re.search(r"^\s*- DRAFT_SAMPLE_METHOD\s*$", s, re.M):
        bad.append(f"{f}: DRAFT_SAMPLE_METHOD is not declared bare in environment:")
    if 'greedy|probabilistic)' not in s:
        bad.append(f"{f}: no greedy|probabilistic guard on DRAFT_SAMPLE_METHOD")
print(f"CHECKED {checked}")
for b in bad:
    print("BAD " + b)
PY
)
fails=0
while IFS= read -r line; do
  [[ "$line" == BAD* ]] && { echo "FAIL: ${line#BAD }" >&2; fails=$((fails+1)); }
done <<<"$out"
checked=$(command grep -oE '^CHECKED [0-9]+' <<<"$out" | cut -d' ' -f2)
[[ "${checked:-0}" -gt 0 ]] || { echo "FAIL: no spec-decode composes found — the search is wrong, not the tree" >&2; exit 1; }

# --- delivery path + NEGATIVE CONTROL ---------------------------------------
if command -v docker >/dev/null 2>&1; then
  for c in models/thinkingcap-qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml \
           models/qwen3.8-27b/vllm/compose/dual/autoround-int4/dflash2.yml; do
    for v in "" greedy banana; do
      res=$(env -u DRAFT_SAMPLE_METHOD MODEL_DIR="${MODEL_DIR:-/tmp}" ${v:+DRAFT_SAMPLE_METHOD=$v} \
        docker compose -f "$c" config --format json 2>/dev/null | python3 -c '
import json, re, subprocess, sys
svc = next(iter(json.load(sys.stdin)["services"].values()))
script = [x for x in svc["entrypoint"] if "SPEC_ARGS" in x][0].replace("$$", "$")
env = {k: v for k, v in (svc.get("environment") or {}).items() if v is not None}
m = re.search(r"^_dsm=.*?^esac$", script, re.S | re.M)
if not m:
    print("NOGUARD"); raise SystemExit
p = subprocess.run(["bash", "-c", m.group(0) + "\necho $_dsm"], capture_output=True, text=True,
                   env=dict(env, PATH="/usr/bin:/bin"))
print(("OK " + p.stdout.strip()) if p.returncode == 0 else "REFUSED")')
      case "${v:-unset}" in
        unset)  [[ "$res" == "OK probabilistic" ]] || { echo "FAIL: $c unset -> '$res', expected probabilistic" >&2; fails=$((fails+1)); } ;;
        greedy) [[ "$res" == "OK greedy" ]]        || { echo "FAIL: $c greedy -> '$res'" >&2; fails=$((fails+1)); } ;;
        banana) [[ "$res" == "REFUSED" ]]          || { echo "FAIL: $c junk value must refuse to boot, got '$res'" >&2; fails=$((fails+1)); } ;;
      esac
    done
  done
else
  echo "NOTE: docker unavailable — static check only; the delivery-path leg did NOT run." >&2
fi

if [[ $fails -gt 0 ]]; then
  echo "$fails draft_sample_method check(s) failed" >&2
  exit 1
fi
echo "PASS: $checked Qwen3.8/ThinkingCap spec-decode composes pass draft_sample_method (valid JSON, bare knob, guard); unset -> probabilistic, greedy honoured, junk refused"
