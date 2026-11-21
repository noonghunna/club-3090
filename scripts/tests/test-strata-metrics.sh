#!/usr/bin/env bash
# Guard: scripts/lib/strata_metrics.py, the one reader of a Strata server's /metrics JSON.
#
# Strata writes its per-request lines to a log file inside the container, not to `docker logs`,
# so the health, verify-full and bench arms for Strata read /metrics instead. A reader that
# returned "0 drafts" for a document it could not read would make a dead drafter and a broken
# probe look the same, so every way of failing to read must exit 3, distinct from a clean read.
#
# Fixtures are served by a local http.server on a free port, so no Strata server is needed:
#   1. recent: sums the newest requests (decode / prefill tok/s, acceptance, hit rate, engine);
#   2. no drafts offered: accept_rate is EMPTY, not 0 (the consumer decides what that means);
#   3. delta between two snapshots, and a restart between them (fewer requests) refused;
#   4. negative controls, each exit 3: not JSON, JSON without engine.version, unreachable;
#   5. soak-helper's Strata log signature: the per-request 'done' line, not the progress line;
#   6. host-ram: the evidence bench.sh reads for Project Maya's RAM tier, and its exit 1 / 3.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
SM="scripts/lib/strata_metrics.py"

rc=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; rc=1; }

work="$(mktemp -d)"
server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null; rm -rf "$work"; }
trap cleanup EXIT

port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
mkdir -p "$work/ok" "$work/nodrafts" "$work/notjson" "$work/foreign" "$work/maya"
cat > "$work/ok/metrics" <<'JSON'
{"engine": {"version": "0.1.43.1", "kv": "int8", "max_context": 131072, "expert_slots": 8746},
 "totals": {"requests": 2, "output_tokens": 300, "decode_ms": 3000, "prompt_tokens": 1100, "reused": 100,
            "prompt_ms": 500, "drafts_offered": 200, "drafts_accepted": 150},
 "requests": [
   {"output_tokens": 200, "decode_ms": 2000, "prompt_read": 600, "prompt_ms": 300,
    "drafts_offered": 150, "drafts_accepted": 120, "hit_rate": 0.9},
   {"output_tokens": 100, "decode_ms": 1000, "prompt_read": 400, "prompt_ms": 200,
    "drafts_offered": 50, "drafts_accepted": 30, "hit_rate": 0.8}]}
JSON
cat > "$work/nodrafts/metrics" <<'JSON'
{"engine": {"version": "0.1.43.1"}, "totals": {"requests": 1},
 "requests": [{"output_tokens": 50, "decode_ms": 500, "prompt_read": 10, "prompt_ms": 10,
               "drafts_offered": 0, "drafts_accepted": 0}]}
JSON
# Project Maya (strata-maya): version in engine.maya_version, no draft counts in /metrics (they are in
# the engine log only), requests carry prompt_tokens + reused instead of prompt_read
cat > "$work/maya/metrics" <<'JSON'
{"engine": {"version": null, "maya_version": "1.1.11", "kv": "fp16", "max_context": 131072, "expert_slots": 5508},
 "totals": {"requests": 2, "output_tokens": 300, "decode_ms": 3000, "prompt_tokens": 1100, "reused": 100, "prompt_ms": 500},
 "requests": [
   {"output_tokens": 200, "decode_ms": 2000, "prompt_tokens": 650, "reused": 50, "prompt_ms": 300, "hit_rate": 0.8},
   {"output_tokens": 100, "decode_ms": 1000, "prompt_tokens": 450, "reused": 50, "prompt_ms": 200, "hit_rate": 0.7}],
 "storage": {"path": "/data/maya-s"}}
JSON
printf '%s\n' "glm stat: decode 20.28 ms/tok" "glm spec: 15 of 16 drafts accepted (93.8%)" \
  "glm kv: KV streaming: 93.81% of 3086 block reads hit VRAM" "glm spec: 120 of 150 drafts accepted (80.0%)" \
  "glm spec: 30 of 50 drafts accepted (60.0%)" > "$work/maya.log"
head -2 "$work/maya.log" > "$work/maya-early.log"
echo '<html>not metrics</html>' > "$work/notjson/metrics"
echo '{"totals": {"requests": 3}}' > "$work/foreign/metrics"

python3 -m http.server "$port" --bind 127.0.0.1 --directory "$work" >/dev/null 2>&1 &
server_pid=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null "http://127.0.0.1:$port/ok/metrics" && break
  sleep 0.1
done
base="http://127.0.0.1:$port"

# 1. recent
out="$(python3 "$SM" recent "$base/ok")"; r=$?
want="requests=2 output_tokens=300 decode_tok_s=100.00 prefill_tok_s=2000.00 drafts_offered=200 drafts_accepted=150 accept_rate=0.7500 hit_rate=0.850 version=0.1.43.1 kv=int8 max_context=131072 expert_slots=8746"
[[ $r -eq 0 && "$(tr '\n' ' ' <<<"$out" | sed 's/ $//')" == "$want" ]] \
  && ok "recent sums the newest requests (tok/s, acceptance, hit rate, engine)" \
  || bad "recent: rc=$r, got: $(tr '\n' ' ' <<<"$out")"
out="$(python3 "$SM" recent "$base/ok" 1)"
command grep -qx 'accept_rate=0.8000' <<<"$out" && command grep -qx 'requests=1' <<<"$out" \
  && ok "recent N takes only the newest N (newest first)" || bad "recent 1: $(tr '\n' ' ' <<<"$out")"

# 2. no drafts offered
out="$(python3 "$SM" recent "$base/nodrafts")"; r=$?
[[ $r -eq 0 ]] && command grep -qx 'accept_rate=' <<<"$out" && command grep -qx 'drafts_offered=0' <<<"$out" \
  && ok "no drafts offered: accept_rate empty, drafts_offered=0 (not a fake 0% or 100%)" \
  || bad "no drafts: rc=$r, got: $(tr '\n' ' ' <<<"$out")"

# 3. delta, and a restart between snapshots
python3 "$SM" snapshot "$base/nodrafts" > "$work/before.json" && python3 "$SM" snapshot "$base/ok" > "$work/after.json"
out="$(python3 "$SM" delta "$work/before.json" "$work/after.json")"; r=$?
command grep -qx 'requests=1' <<<"$out" && command grep -qx 'decode_tok_s=100.00' <<<"$out" \
  && command grep -qx 'accept_rate=0.7500' <<<"$out" \
  && ok "delta between two snapshots" || bad "delta: rc=$r, got: $(tr '\n' ' ' <<<"$out")"
python3 "$SM" delta "$work/after.json" "$work/before.json" >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "delta refuses a restart between snapshots (fewer requests): exit 3" || bad "restart: exit $r, want 3"

# 4. negative controls: every failure to read is exit 3
for case in notjson foreign; do
  python3 "$SM" recent "$base/$case" >/dev/null 2>&1; r=$?
  [[ $r -eq 3 ]] && ok "$case /metrics: exit 3" || bad "$case /metrics: exit $r, want 3"
done
python3 "$SM" recent "http://127.0.0.1:1" >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "unreachable server: exit 3" || bad "unreachable: exit $r, want 3"
python3 "$SM" recent >/dev/null 2>&1; r=$?
[[ $r -eq 2 ]] && ok "bad usage: exit 2" || bad "bad usage: exit $r, want 2"

# 5. Project Maya: its document reads; drafts come from its engine log, given one
out="$(python3 "$SM" recent "$base/maya")"; r=$?
[[ $r -eq 0 ]] && command grep -qx 'version=1.1.11' <<<"$out" && command grep -qx 'drafts_offered=0' <<<"$out" \
  && command grep -qx 'prefill_tok_s=2000.00' <<<"$out" \
  && ok "Maya document: version from engine.maya_version, prefill from prompt_tokens - reused, no drafts without its log" \
  || bad "Maya recent: rc=$r, got: $(tr '\n' ' ' <<<"$out")"
out="$(python3 "$SM" recent "$base/maya" 2 --log-file "$work/maya.log")"
command grep -qx 'drafts_offered=200' <<<"$out" && command grep -qx 'drafts_accepted=150' <<<"$out" \
  && command grep -qx 'accept_rate=0.7500' <<<"$out" \
  && ok "Maya recent N --log-file: the newest N 'glm spec' lines" || bad "Maya recent log: $(tr '\n' ' ' <<<"$out")"
python3 "$SM" snapshot "$base/maya" --log-file "$work/maya-early.log" > "$work/m0.json" \
  && python3 "$SM" snapshot "$base/maya" --log-file "$work/maya.log" > "$work/m1.json" \
  && out="$(python3 "$SM" delta "$work/m0.json" "$work/m1.json")" \
  && command grep -qx 'drafts_offered=200' <<<"$out" && command grep -qx 'accept_rate=0.7500' <<<"$out" \
  && ok "Maya snapshot --log-file: the log's drafts are summed into totals, so delta reads them" \
  || bad "Maya snapshot/delta: $(tr '\n' ' ' <<<"${out:-}")"
python3 "$SM" snapshot "$base/maya" --log-file "$work/no-such.log" >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "Maya log that can't be read: exit 3 (never a silent 'no drafts')" || bad "missing Maya log: exit $r, want 3"
python3 "$SM" snapshot "$base/ok" --log-file "$work/maya.log" > "$work/s.json"
python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["totals"]; sys.exit(0 if t["drafts_offered"] == 200 else 1)' "$work/s.json" \
  && ok "a Strata document keeps its own draft counts even when a log is given" || bad "Strata totals overridden by a log"

# 6. soak-helper reads Strata's per-request docker-logs line, and not its progress line
got="$(python3 - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("sh", "scripts/soak-helper.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
done = "[strata] done: 231 tokens in 3 s (107.7 tok/s) (stop, cancel=False), expert cache 74.7% hit"
progress = "[strata] answering: 58 of max 1500 tokens, 103.0 tok/s, 1 s"
maya = "[maya] done: 207 tokens in 5 s (52.6 tok/s) (stop, cancel=False), expert cache 80.6% hit"
print(m.engine_rate_from_logs(progress + "\n" + done), m.engine_rate_from_logs(progress), m.engine_rate_from_logs(maya))
PY
)"
[[ "$got" == "(107.7, 'strata-log') (None, '') (52.6, 'strata-log')" ]] \
  && ok "soak-helper: Strata's and Project Maya's 'done' lines are engine-reported decode rates; the progress line is not" \
  || bad "soak-helper Strata signature: got $got"

# 7. host-ram: bench.sh's evidence that Project Maya keeps experts in host RAM (#1700). Its
# docker log doesn't say; /metrics does once a request has run, and its engine log always.
mkdir -p "$work/maya-tiers" "$work/maya-tiers0"
python3 - "$work" <<'PY'
import json, sys
w = sys.argv[1]
doc = json.load(open(f"{w}/maya/metrics"))
doc["tiers"] = {"now": {"tok_s": 29.6, "vram_hit": 0.57, "ram_slots": 9799, "ram_used": 9120}, "history": {}}
json.dump(doc, open(f"{w}/maya-tiers/metrics", "w"))
doc["tiers"]["now"]["ram_slots"] = 0
json.dump(doc, open(f"{w}/maya-tiers0/metrics", "w"))
PY
# the log is appended across restarts: an earlier start on one card, then the current one on two
printf '%s\n' "glm fast: CUDA0 RAM tier 10.00 GB pinned, 900 slots" "glm stat: decode 20.28 ms/tok" \
  "glm fast: CUDA0 RAM tier 52.66 GB pinned, 4737 slots" "glm fast: CUDA1 RAM tier 56.91 GB pinned, 5062 slots" > "$work/maya-ram.log"
printf '%s\n' "glm fast: CUDA0 RAM tier 0.00 GB pinned, 0 slots" > "$work/maya-ram0.log"
out="$(python3 "$SM" host-ram "$base/maya-tiers")"; r=$?
[[ $r -eq 0 && "$out" == "/metrics (9799 expert slots in RAM)" ]] \
  && ok "host-ram: expert slots in RAM from /metrics" || bad "host-ram /metrics: rc=$r, got: $out"
out="$(python3 "$SM" host-ram "$base/maya" --log-file "$work/maya-ram.log")"; r=$?
[[ $r -eq 0 && "$out" == "engine log (RAM tier 109.57 GB pinned)" ]] \
  && ok "host-ram: Maya's engine log, the latest line per card (an earlier start's line superseded)" \
  || bad "host-ram log: rc=$r, got: $out"
python3 "$SM" host-ram "$base/maya" --log-file "$work/maya.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: no RAM-tier line, exit 1 (no evidence, not a guess)" || bad "host-ram no line: exit $r, want 1"
python3 "$SM" host-ram "$base/maya-tiers0" --log-file "$work/maya-ram0.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: a RAM tier of 0 slots / 0 GB (all experts in VRAM) is not offload: exit 1" \
  || bad "host-ram zero tier: exit $r, want 1"
python3 "$SM" host-ram "$base/ok" --log-file "$work/maya-ram.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: a Strata document doesn't read a Maya log: exit 1" || bad "host-ram Strata: exit $r, want 1"
# Project Maya's Hy4-Flash engine (maya-hy4) logs no RAM line: its "CPU lane: N of M routed experts in VRAM" line
# (the latest start's: the log is appended across restarts) means M - N are computed on the CPU from the model file
printf '%s\n' "INFO CPU lane: 5125 of 20992 routed experts in VRAM (24%), the rest on 25 CPU threads from the model's file" \
  "maya hy4 serve: MTP kept 40 of 56 drafts" \
  "INFO CPU lane: 11931 of 20992 routed experts in VRAM (57%), the rest on 25 CPU threads from the model's file" > "$work/maya-hy4.log"
printf '%s\n' "INFO CPU lane: 20992 of 20992 routed experts in VRAM (100%), the rest on 25 CPU threads from the model's file" > "$work/maya-hy4-all.log"
out="$(python3 "$SM" host-ram "$base/maya" --log-file "$work/maya-hy4.log")"; r=$?
[[ $r -eq 0 && "$out" == "engine log (CPU lane: 9061 of 20992 routed experts on the CPU, from the model file)" ]] \
  && ok "host-ram: maya-hy4's CPU-lane line, the latest start's (an earlier start's superseded)" || bad "host-ram hy4: rc=$r, got: $out"
python3 "$SM" host-ram "$base/maya" --log-file "$work/maya-hy4-all.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: maya-hy4 with every routed expert in VRAM is not offload: exit 1" \
  || bad "host-ram hy4 all in VRAM: exit $r, want 1"
python3 "$SM" host-ram "$base/maya" --log-file "$work/no-such.log" >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "host-ram: a Maya log that can't be read: exit 3" || bad "host-ram missing log: exit $r, want 3"
python3 "$SM" host-ram "http://127.0.0.1:1" >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "host-ram: unreachable server: exit 3" || bad "host-ram unreachable: exit $r, want 3"

# 8. Project Maya's DeepSeek engine (maya-ds4): its draft line is "maya ds4 serve: drafts N of M kept",
# its /metrics names no storage.path, so the log is the one its run config names (the server's
# --config), and its host-RAM evidence is the CPU lane's experts. A stub docker serves the
# container's files from a folder, so the lookup runs without a container.
mkdir -p "$work/ds4" "$work/ds4-131" "$work/bin" "$work/ctr/proc/1" "$work/ctr/data/ista" "$work/ctr-nocfg/proc/1" \
  "$work/ctr-nolog/proc/1" "$work/ctr-nolog/data/ista" "$work/ctr/data/maya-s" \
  "$work/ctr-glm/proc/1" "$work/ctr-glm/data/maya-s" "$work/ctr-glm-nocfg/proc/1" "$work/ctr-glm-nocfg/data/maya-s"
python3 - "$work" <<'PY'
import json, sys
w = sys.argv[1]
doc = json.load(open(f"{w}/maya/metrics"))
doc["engine"]["maya_version"] = "1.1.28"
doc["storage"] = None                              # what DeepSeek's server sends
json.dump(doc, open(f"{w}/ds4/metrics", "w"))
doc["storage"] = {"path": "/data/ista"}            # what DeepSeek's server sends from v1.1.31 (its quant's folder)
json.dump(doc, open(f"{w}/ds4-131/metrics", "w"))
glm = ["/opt/maya/.venv/bin/python", "/opt/maya/serve/server.py", "--engine", "strata",
       "--config", "/data/maya-s/maya.json", "--host", "0.0.0.0", "--port", "8080"]
open(f"{w}/ctr-glm/proc/1/cmdline", "w").write("\0".join(glm) + "\0")
json.dump({"log": "/data/maya-s/maya.log", "args": []}, open(f"{w}/ctr-glm/data/maya-s/maya.json", "w"))
open(f"{w}/ctr-glm-nocfg/proc/1/cmdline", "w").write("\0".join(glm[:4]) + "\0")
argv = ["/opt/maya/.venv/bin/python", "/opt/maya/serve/server.py", "--engine", "strata",
        "--config", "/data/ista/maya-ds4.json", "--host", "0.0.0.0", "--port", "8080"]
for d in ("ctr", "ctr-nolog"):
    open(f"{w}/{d}/proc/1/cmdline", "w").write("\0".join(argv) + "\0")
open(f"{w}/ctr-nocfg/proc/1/cmdline", "w").write("\0".join(argv[:4]) + "\0")
json.dump({"log": "/data/ista/maya-ds4.log", "args": []}, open(f"{w}/ctr/data/ista/maya-ds4.json", "w"))
json.dump({"args": []}, open(f"{w}/ctr-nolog/data/ista/maya-ds4.json", "w"))
PY
# two starts appended (the first copied less into RAM), then requests; a GLM line must not leak in
printf '%s\n' "ds4: 9000 routed experts on the CPU lane (80.0 GB): 9000 copied into RAM (1 NUMA parts)" \
  "maya ds4 serve: drafts 1 of 1 kept; per position (kept / reached): 1: 1/1 100%" \
  "ds4: 11665 routed experts on the CPU lane (100.4 GB): 11665 copied into RAM (1 NUMA parts), 24 threads" \
  "maya ds4 serve: drafts 40 of 44 kept; per position (kept / reached): 1: 10/10 100%" \
  "maya ds4 serve: drafts 995 of 1181 kept; per position (kept / reached): 1: 510/603 85%" > "$work/ctr/data/ista/maya-ds4.log"
head -2 "$work/ctr/data/ista/maya-ds4.log" > "$work/ds4-early.log"
cp "$work/maya.log" "$work/ctr/data/maya-s/maya.log"
cp "$work/maya.log" "$work/ctr-glm/data/maya-s/maya.log"
cp "$work/maya.log" "$work/ctr-glm-nocfg/data/maya-s/maya.log"
cat > "$work/bin/docker" <<'SH'
#!/bin/sh
# stub: `docker exec <container> cat <path>` reads <path> under $work/<container>
[ "$1" = exec ] && [ "$3" = cat ] || { echo "stub docker: unexpected: $*" >&2; exit 1; }
f="$STUB_ROOT/$2$4"
[ -f "$f" ] && exec cat "$f"
echo "cat: $4: No such file or directory" >&2; exit 1
SH
chmod +x "$work/bin/docker"
sdocker() { PATH="$work/bin:$PATH" STUB_ROOT="$work" "$@"; }

out="$(python3 "$SM" recent "$base/ds4" 2 --log-file "$work/ctr/data/ista/maya-ds4.log")"
command grep -qx 'drafts_offered=1225' <<<"$out" && command grep -qx 'drafts_accepted=1035' <<<"$out" \
  && ok "maya-ds4 recent N --log-file: the newest N 'maya ds4 serve: drafts' lines" || bad "ds4 recent log: $(tr '\n' ' ' <<<"$out")"
out="$(sdocker python3 "$SM" snapshot "$base/ds4" --container ctr)"; r=$?
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if (t["drafts_accepted"], t["drafts_offered"]) == (1036, 1226) else 1)' "$out" 2>/dev/null \
  && ok "maya-ds4 snapshot --container: no storage.path, so the log its --config names; every draft line summed" \
  || bad "ds4 snapshot container: rc=$r, got: $out"
sdocker python3 "$SM" snapshot "$base/ds4" --log-file "$work/ds4-early.log" > "$work/d0.json" \
  && sdocker python3 "$SM" snapshot "$base/ds4" --container ctr > "$work/d1.json" \
  && out="$(python3 "$SM" delta "$work/d0.json" "$work/d1.json")" \
  && command grep -qx 'drafts_offered=1225' <<<"$out" && command grep -qx 'accept_rate=0.8449' <<<"$out" \
  && ok "maya-ds4 delta: the drafts between two snapshots (what verify-full's acceptance check reads)" \
  || bad "ds4 delta: $(tr '\n' ' ' <<<"${out:-}")"
sdocker python3 "$SM" snapshot "$base/ds4" --container ctr-nocfg >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "maya-ds4: a server command line without --config: exit 3" || bad "ds4 no --config: exit $r, want 3"
sdocker python3 "$SM" snapshot "$base/ds4" --container ctr-nolog >/dev/null 2>&1; r=$?
[[ $r -eq 3 ]] && ok "maya-ds4: a run config that names no log: exit 3" || bad "ds4 config without log: exit $r, want 3"
# the run config's log comes first for every Maya engine; <storage.path>/maya.log is GLM's fallback only
out="$(sdocker python3 "$SM" snapshot "$base/ds4-131" --container ctr)"; r=$?
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if (t["drafts_accepted"], t["drafts_offered"]) == (1036, 1226) else 1)' "$out" 2>/dev/null \
  && ok "maya-ds4 from v1.1.31 (/metrics names storage.path /data/ista): still the log its --config names, not <storage.path>/maya.log" \
  || bad "ds4 v1.1.31 storage.path: rc=$r, got: $out"
out="$(sdocker python3 "$SM" snapshot "$base/maya" --container ctr-glm)"
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if t["drafts_offered"] == 216 else 1)' "$out" 2>/dev/null \
  && ok "GLM Maya --container: the log its --config names (maya.log)" || bad "GLM container path: got $out"
out="$(sdocker python3 "$SM" snapshot "$base/maya" --container ctr-glm-nocfg)"
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if t["drafts_offered"] == 216 else 1)' "$out" 2>/dev/null \
  && ok "GLM Maya --container, no --config on its command line: falls back to <storage.path>/maya.log" || bad "GLM fallback path: got $out"
out="$(python3 "$SM" host-ram "$base/ds4" --log-file "$work/ctr/data/ista/maya-ds4.log")"; r=$?
[[ $r -eq 0 && "$out" == "engine log (CPU lane: 100.40 GB of experts)" ]] \
  && ok "host-ram: maya-ds4's CPU lane, the latest start's line" || bad "host-ram ds4: rc=$r, got: $out"
printf '%s\n' "ds4: 0 routed experts on the CPU lane (0.0 GB): 0 copied into RAM" > "$work/ds4-ram0.log"
python3 "$SM" host-ram "$base/ds4" --log-file "$work/ds4-ram0.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: a maya-ds4 CPU lane of 0 GB (every expert in VRAM) is not offload: exit 1" \
  || bad "host-ram ds4 zero lane: exit $r, want 1"

# 9. Project Maya's MiniMax engine (maya-mm3): its draft line is "maya mm3 serve: DSpark kept N of M drafts". It logs no
# RAM line under the server, so its host-RAM evidence is its /metrics: the engine's INFO line (engine.ram_slots) and the
# tiers block (tiers.now.ram_gb), as a v1.1.30 server reported them on the reference rig (2026-10-27).
printf '%s\n' "maya mm3 serve: DSpark kept 1 of 2 drafts" \
  "glm spec: 7 of 9 drafts accepted (77.8%)" \
  "maya mm3 serve: DSpark kept 40 of 44 drafts" \
  "maya mm3 serve: DSpark kept 300 of 400 drafts" > "$work/mm3.log"
mkdir -p "$work/mm3" "$work/mm3-0"
python3 - "$work" <<'PY'
import json, sys
w = sys.argv[1]
doc = json.load(open(f"{w}/ds4/metrics"))
doc["engine"].update({"version": "mm3", "maya_version": "1.1.30", "expert_slots": 320, "ram_slots": 6976, "dspark": 0,
                      "kv": "int8", "cpu_experts": 6976, "ctx": 204800})
doc["tiers"] = {"now": {"vram_used": 320.0, "ram_used": 6976.0, "vram_gb": 4.52, "ram_gb": 97.91}}
json.dump(doc, open(f"{w}/mm3/metrics", "w"))
doc["engine"]["ram_slots"] = 0
doc["tiers"]["now"]["ram_gb"] = 0.0
json.dump(doc, open(f"{w}/mm3-0/metrics", "w"))
PY
out="$(python3 "$SM" recent "$base/ds4" 2 --log-file "$work/mm3.log")"
command grep -qx 'drafts_offered=444' <<<"$out" && command grep -qx 'drafts_accepted=340' <<<"$out" \
  && ok "maya-mm3 recent N --log-file: the newest N 'maya mm3 serve: DSpark kept' lines" || bad "mm3 recent log: $(tr '\n' ' ' <<<"$out")"
out="$(sdocker python3 "$SM" snapshot "$base/ds4" --log-file "$work/mm3.log")"
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if (t["drafts_accepted"], t["drafts_offered"]) == (348, 455) else 1)' "$out" 2>/dev/null \
  && ok "maya-mm3 snapshot: every draft line summed (MiniMax's and the GLM line alike)" || bad "mm3 snapshot: got $out"
out="$(python3 "$SM" host-ram "$base/mm3" --log-file "$work/mm3.log")"; r=$?
[[ $r -eq 0 && "$out" == "/metrics (6976 expert slots in RAM, 97.91 GB)" ]] \
  && ok "host-ram: maya-mm3's RAM tier from /metrics, its log having none" || bad "host-ram mm3: rc=$r, got: $out"
out="$(python3 "$SM" host-ram "$base/mm3")"; r=$?
[[ $r -eq 0 ]] && ok "host-ram: maya-mm3 needs no container or log for it" || bad "host-ram mm3 without a log: rc=$r, got: $out"
python3 "$SM" host-ram "$base/mm3-0" --log-file "$work/mm3.log" >/dev/null 2>&1; r=$?
[[ $r -eq 1 ]] && ok "host-ram: a maya-mm3 RAM tier of 0 (every expert in VRAM) is not offload: exit 1" \
  || bad "host-ram mm3 zero tier: exit $r, want 1"

# 10. Project Maya's Hy4-Flash engine (maya-hy4): its draft line is "maya hy4 serve: MTP kept N of M drafts" (src/hy4/hy4_main.cpp,
# v1.1.68), one a request that drafted.
printf '%s\n' "maya hy4 serve: MTP kept 1 of 2 drafts" \
  "maya mm3 serve: DSpark kept 7 of 9 drafts" \
  "maya hy4 serve: MTP kept 40 of 44 drafts" \
  "maya hy4 serve: MTP kept 300 of 400 drafts" > "$work/hy4.log"
out="$(python3 "$SM" recent "$base/ds4" 2 --log-file "$work/hy4.log")"
command grep -qx 'drafts_offered=444' <<<"$out" && command grep -qx 'drafts_accepted=340' <<<"$out" \
  && ok "maya-hy4 recent N --log-file: the newest N 'maya hy4 serve: MTP kept' lines" || bad "hy4 recent log: $(tr '\n' ' ' <<<"$out")"
out="$(sdocker python3 "$SM" snapshot "$base/ds4" --log-file "$work/hy4.log")"
python3 -c 'import json,sys; t=json.loads(sys.argv[1])["totals"]; sys.exit(0 if (t["drafts_accepted"], t["drafts_offered"]) == (348, 455) else 1)' "$out" 2>/dev/null \
  && ok "maya-hy4 snapshot: every draft line summed (Hy4-Flash's and the MiniMax line alike)" || bad "hy4 snapshot: got $out"

if [[ $rc -eq 0 ]]; then echo "PASS: test-strata-metrics"; else echo "FAIL: test-strata-metrics"; fi
exit $rc
