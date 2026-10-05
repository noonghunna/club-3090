#!/usr/bin/env bash
# test-concurrency-probe — offline guards for scripts/concurrency-probe.sh (the
# #246 Phase 2 soak-validation tool) and scripts/lib/concurrency_probe.py.
# The live probe needs a running server; these check only what can be verified
# without one: syntax, SWEEP-needs-SLUG, SWEEP_DRY reboot plans, --sweep dry
# plans (no SLUG, no reboot), planner clips, and the card renderer.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)

# Force Python UTF-8 mode (PEP 540) before the first python3 call (#779).
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROBE="$ROOT_DIR/scripts/concurrency-probe.sh"
LIB="$ROOT_DIR/scripts/lib/concurrency_probe.py"
fail() { echo "FAIL: $1" >&2; exit 1; }

# --- hermetic docker layer -----------------------------------------------------
# Several cases below run the probe for real, and the probe's SWEEP path installs
# a restore-on-exit trap that calls `switch.sh --force "$SLUG"`. That trap fires at
# process exit — AFTER the last line of output — so a stdout-only assertion of
# "no boot happened" goes green while a real `docker compose up` runs behind it.
#
# That is not hypothetical: on 2026-09-08 this test booted vllm/minimal (20 GB of
# VRAM) on the rig and the guard sweep still reported PASS. It also means the
# suite could tear down whatever was serving at the time.
#
# Stubbing `docker` on PATH for the whole file fixes both: the probe and switch.sh
# run for real, but cannot touch the estate, and any boot attempt is RECORDED so
# the dry-run case can assert on the side effect instead of on stdout.
SHIMDIR="$(mktemp -d)"
trap 'rm -rf "$SHIMDIR"' EXIT
COMPOSE_UP_LOG="$SHIMDIR/compose-up.log"
: > "$COMPOSE_UP_LOG"
cat > "$SHIMDIR/docker" <<SHIM
#!/usr/bin/env bash
# Records any \`compose ... up\`; answers everything else with a benign success.
_c=0
for a in "\$@"; do
  [ "\$a" = "compose" ] && _c=1
  if [ "\$_c" = 1 ] && [ "\$a" = "up" ]; then echo "\$*" >> "$COMPOSE_UP_LOG"; exit 0; fi
done
exit 0
SHIM
chmod +x "$SHIMDIR/docker"
export PATH="$SHIMDIR:$PATH"

# Negative control: the stub must actually record a boot, or every assertion that
# reads COMPOSE_UP_LOG below is vacuous and would pass over a live regression.
docker compose -f /dev/null up -d >/dev/null 2>&1 || true
[[ -s "$COMPOSE_UP_LOG" ]] || fail "docker stub does not record 'compose up' — the boot assertions would be vacuous"
: > "$COMPOSE_UP_LOG"
echo "  ✓ docker stubbed (estate-safe) + stub self-test"

# 1. syntax
bash -n "$PROBE" || fail "bash -n: syntax error"
python3 -m py_compile "$LIB" || fail "py_compile concurrency_probe.py"
echo "  ✓ syntax"

# 2. SWEEP without SLUG must refuse with exit 2 (a reboot target is mandatory —
#    vLLM can't hot-change max-num-seqs, so each N is a boot).
set +e
out="$(SWEEP="4 8" bash "$PROBE" 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "SWEEP without SLUG should exit 2, got $rc"
command grep -q "SWEEP needs SLUG" <<<"$out" || fail "SWEEP-without-SLUG error message missing"
echo "  ✓ SWEEP refuses without SLUG (exit 2)"

# 3. SWEEP_DRY prints the plan for every N and must NOT boot the server (a dry
#    run never touches switch.sh). Assert on output: one [sweep:dry] line per N,
#    no [sweep] boot line, and the knee summary always prints.
out="$(SWEEP="4 8 12" SLUG=vllm/minimal SWEEP_DRY=1 bash "$PROBE" 2>&1)"
[[ "$(command grep -c '\[sweep:dry\]' <<<"$out")" == "3" ]] || fail "SWEEP_DRY should print one plan line per N (3)"
command grep -q '\[sweep\] boot' <<<"$out" && fail "SWEEP_DRY must not boot the server"
command grep -q "sweep knee" <<<"$out" || fail "SWEEP should always print a knee summary"
if [[ -s "$COMPOSE_UP_LOG" ]]; then
  fail "SWEEP_DRY ran 'docker compose up' — the restore-on-exit trap booted the slug: $(tr '\n' ';' < "$COMPOSE_UP_LOG")"
fi
echo "  ✓ SWEEP_DRY plans 3 reboots without booting (asserted at the docker layer)"

# 4. --sweep --dry does NOT need SLUG and must not plan switch.sh reboots.
out="$(bash "$PROBE" --sweep --dry --n 1,2,4,8 --ctx 1k,4k,16k 2>&1)"
[[ $? == 0 ]] || fail "--sweep --dry should exit 0 without SLUG"
command grep -q "switch.sh" <<<"$out" && fail "--sweep --dry must not plan switch.sh reboots"
command grep -q "SWEEP needs SLUG" <<<"$out" && fail "--sweep must not require SLUG"
command grep -q "1K" <<<"$out" || fail "--sweep --dry should list 1K"
command grep -q "4K" <<<"$out" || fail "--sweep --dry should list 4K"
command grep -q "16K" <<<"$out" || fail "--sweep --dry should list 16K"
echo "  ✓ --sweep --dry plans a live matrix without SLUG"

# 5. --sweep + --validate refused
set +e
out="$(bash "$PROBE" --sweep --validate --dry 2>&1)"; rc=$?
set -e
[[ "$rc" == "2" ]] || fail "--sweep --validate should exit 2, got $rc"
command grep -q "cannot combine" <<<"$out" || fail "--sweep --validate message missing"
echo "  ✓ --sweep --validate refused"

# 6. planner clips: slots, max-len, KV pool
plan="$(
  CTX_SWEEP="1k 4k 8k 16k 32k" N_LIST="1 2 4 8 16 32" GEN_TOKENS=256 \
  KV_TOKENS=262144 SERVED_SLOTS=8 SERVED_MAX_LEN=32768 \
  python3 "$LIB" --plan
)"
command grep -q "32" <<<"$plan" && command grep -q "served slots=8" <<<"$plan" \
  || fail "plan should drop N>8 with a slots note"
command grep -q "32K" <<<"$plan" || fail "32K should remain when max-len=32K"
# 32K × N=8 = 8*(32768+256)=264192 > 262144 → skip
command grep -q "32K" <<<"$plan" || true
tsv="$(
  CTX_SWEEP="1k 4k 8k 16k 32k" N_LIST="1 2 4 8 16 32" GEN_TOKENS=256 \
  KV_TOKENS=262144 SERVED_SLOTS=8 SERVED_MAX_LEN=32768 \
  python3 "$LIB" --plan-tsv
)"
command grep -q $'skip\t32768\t8\t' <<<"$tsv" || fail "32K N=8 should skip on KV"
command grep -q $'run\t32768\t4\t' <<<"$tsv" || fail "32K N=4 should run (fits KV)"
command grep -q $'run\t16384\t8\t' <<<"$tsv" || fail "16K N=8 should run"
command grep -q $'skip\t1024\t16\t' <<<"$tsv" && fail "1K N=16 should be dropped by slots, not appear"
echo "  ✓ planner clips slots / max-len / KV"

plan_ml="$(
  CTX_SWEEP="1k 4k 16k 32k" N_LIST="1 2 4" GEN_TOKENS=256 \
  SERVED_MAX_LEN=16384 \
  python3 "$LIB" --plan
)"
command grep -q "32K" <<<"$plan_ml" && command grep -q "max-model-len" <<<"$plan_ml" \
  || fail "32K should be dropped when max-len=16K"
echo "  ✓ planner drops ctx above served max-model-len"

# 7. card renderer: vs 1-stream + SWEET on 16K knee
card="$(python3 "$LIB" --card <<'JSON'
{
  "model": "qwen3.6-27b",
  "slug": "vllm/qwen-27b-dual-fast",
  "spec": "MTP n=3",
  "gpus": "2× RTX 3090",
  "kv_tokens": 210000,
  "slots": 8,
  "served_max_len": 262144,
  "engine": "vllm",
  "gen_tokens": 256,
  "cache": "shared 75%",
  "command": "bash scripts/concurrency-probe.sh --sweep",
  "rows": [
    {"ctx": 1024, "n": 1, "strm": 87.3, "agg": 87, "ttft_s": 0.1, "vram_gb": 38.2, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 1024, "n": 8, "strm": 38.0, "agg": 241, "ttft_s": 0.4, "vram_gb": 39.8, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 1, "strm": 87.3, "agg": 87, "ttft_s": 0.8, "vram_gb": 38.2, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 2, "strm": 51.8, "agg": 104, "ttft_s": 1.4, "vram_gb": 39.1, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 4, "strm": 22.1, "agg": 88, "ttft_s": 3.2, "vram_gb": 40.4, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 8, "strm": 6.8, "agg": 54, "ttft_s": 12.0, "vram_gb": 42.1, "clean": 0, "pass": 0, "skip": null},
    {"ctx": 32768, "n": 8, "skip": "N*(ctx+gen) > KV_TOKENS"}
  ]
}
JSON
)"
command grep -q "club-3090" <<<"$card" || fail "card should say club-3090"
command grep -q "vs 1-stream" <<<"$card" || fail "card should label vs 1-stream"
command grep -q "1.20" <<<"$card" || fail "16K N=2 should be 104/87 ≈ 1.20×"
command grep -q "SWEET" <<<"$card" || fail "card should print SWEET"
command grep -q "N=2 @ 16K" <<<"$card" || fail "SWEET should star the 16K aggregate peak (N=2, 104 tok/s)"
command grep -q "github.com" <<<"$card" && fail "card should not billboard the repo URL"
command grep -q "=== recommend ===" <<<"$card" || fail "card should end with a compose recommendation"
command grep -q "MAX_NUM_SEQS=2" <<<"$card" || fail "keep-full-ctx rec should be MAX_NUM_SEQS=2 (16K peak)"
command grep -q "MAX_NUM_SEQS=8" <<<"$card" || fail "max-agg rec should be MAX_NUM_SEQS=8 (1K peak)"
command grep -q "MAX_MODEL_LEN=1024" <<<"$card" || fail "max-agg rec should set MAX_MODEL_LEN to the short ctx"
command grep -q "keep" <<<"$card" || fail "full-ctx rec should keep served max-model-len"
echo "  ✓ card renderer"

# 8. prompt shapes: cold salts first, shared salts after the prefix
cold="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import prompt_text; print(prompt_text(0,1,1024,"cold",0.75,256)[:40])')"
[[ "$cold" == \[probe\ s0\ r1\]* ]] || fail "cold prompt must start with per-stream/round salt, got: $cold"
shared="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import prompt_text; p=prompt_text(0,1,1024,"shared",0.75,256); print("SALT" if p.lstrip().startswith("[probe") else "PREFIX")')"
[[ "$shared" == "PREFIX" ]] || fail "shared prompt must NOT start with the salt"
echo "  ✓ prompt cache shapes"

# 9. KV log parse
kv="$(python3 -c 'import sys; sys.path.insert(0,"'"$ROOT_DIR"'/scripts/lib"); from concurrency_probe import parse_kv_tokens_text; print(parse_kv_tokens_text("GPU KV cache size: 210,000 tokens"))')"
[[ "$kv" == "210000" ]] || fail "parse_kv_tokens_text, got $kv"
echo "  ✓ KV log parse"

# 10. Live --sweep warm-up gate (WARMUP) — opt-in, dry-run skips it, default is
#     silent, and a non-dry sweep against an unreachable server aborts BEFORE
#     slot detection (fails closed instead of sweeping a dead engine). All
#     offline: point at a dead port; the gate needs no live server to prove.
out="$(WARMUP=1 URL=http://127.0.0.1:1 bash "$PROBE" --sweep --dry --n 1 2>&1)"
command grep -q "WARMUP" <<<"$out" && fail "WARMUP=1 + --dry must not fire the gate (dry precedence)"
set +e
out="$(URL=http://127.0.0.1:1 MODEL=x bash "$PROBE" --sweep --n 1 2>&1)"; rc=$?
set -e
command grep -q "WARMUP" <<<"$out" && fail "gate must stay silent when WARMUP is unset (default off)"
set +e
out="$(WARMUP=1 WARMUP_TIMEOUT=2 URL=http://127.0.0.1:1 MODEL=x bash "$PROBE" --sweep --n 1 2>&1)"; rc=$?
set -e
[[ "$rc" == "1" ]] || fail "WARMUP=1 against unreachable server should exit 1, got $rc"
command grep -q "WARMUP: waiting" <<<"$out" || fail "gate should announce it is waiting"
command grep -q "not ready" <<<"$out" || fail "gate should report the not-ready abort"
command grep -q "FATAL: --sweep cannot detect" <<<"$out" && fail "gate must abort BEFORE slot detection"
echo "  ✓ --sweep WARMUP gate: opt-in, dry-skips, default-silent, fails closed"


# #1502 (1): a skipped cell killed the sweep. emit_row ran `python3 - <<'PY'`, so the row piped into it
# was replaced by the heredoc on stdin, json.loads("") raised and set -e ended the run before the card.
# Run the REAL emit_row on a row, and guard every call site against piping into it again.
EMIT_DIR="$(mktemp -d)"
emit_fn="$(awk '/^  emit_row\(\) \{$/{f=1} f{print} f&&/^  \}$/{exit}' "$PROBE")"
[[ -n "$emit_fn" ]] || fail "emit_row() not found in concurrency-probe.sh"
set +e
out="$(cells_jsonl="$EMIT_DIR/cells.jsonl" bash -euo pipefail -c "$emit_fn"$'\n''emit_row "{\"ctx\":32768,\"n\":8,\"skip\":\"KV pool\"}"; echo "rc=$?"' 2>&1)"
set -e
[[ "$out" == *"rc=0"* ]] || fail "emit_row failed on a skipped-cell row (the #1502 crash): $out"
python3 -c 'import json,sys; r=json.loads(open(sys.argv[1]).read()); assert r=={"ctx":32768,"n":8,"skip":"KV pool"}, r' \
  "$EMIT_DIR/cells.jsonl" 2>/dev/null || fail "emit_row wrote the wrong row: $(cat "$EMIT_DIR/cells.jsonl" 2>/dev/null)"
command grep -nE '\|[[:space:]]*emit_row' "$PROBE" && fail "a call site pipes into emit_row — its stdin is the heredoc (#1502)"
[[ "$(command grep -cE '^[[:space:]]+emit_row "' "$PROBE")" == 3 ]] \
  || fail "expected the 3 skip paths (budget, clipped, early-stop) to pass their row as an argument"
rm -rf "$EMIT_DIR"
echo "  ✓ emit_row records a skipped cell (row as an argument; no call site pipes into it)"

# #1502 (2): VRAM and the GPU label belong to the probed container's GPUs, not the host's. The
# reporter's rig: 3× RTX 3090 (an unrelated service on them) + 1× RTX 3060 for the container.
# Stub nvidia-smi and `docker inspect` (ahead of the estate-safe docker shim) and ask the library.
GPU_STUB="$(mktemp -d)"
cat > "$GPU_STUB/nvidia-smi" <<'SMI'
#!/usr/bin/env bash
cat <<'ROWS'
0, GPU-aaaa0000-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 15000
1, GPU-aaaa1111-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 14938
2, GPU-aaaa2222-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3090, 15000
3, GPU-bbbb3333-0000-0000-0000-000000000000, NVIDIA GeForce RTX 3060, 10381
ROWS
SMI
cat > "$GPU_STUB/docker" <<'DOCK'
#!/usr/bin/env bash
[ "$1" = inspect ] || exit 0
case "$2" in
  pinned-ids)  echo '[{"HostConfig":{"DeviceRequests":[{"Count":0,"DeviceIDs":["3"]}]},"Config":{"Env":[]}}]' ;;
  pinned-uuid) echo '[{"HostConfig":{"DeviceRequests":[{"Count":0,"DeviceIDs":["GPU-bbbb3333-0000-0000-0000-000000000000"]}]},"Config":{"Env":[]}}]' ;;
  pinned-env)  echo '[{"HostConfig":{"DeviceRequests":[{"Count":-1,"DeviceIDs":null}]},"Config":{"Env":["NVIDIA_VISIBLE_DEVICES=3"]}}]' ;;
  pinned-joined) echo '[{"HostConfig":{"DeviceRequests":[{"Count":0,"DeviceIDs":["2,3"]}]},"Config":{"Env":[]}}]' ;;
  sees-all)    echo '[{"HostConfig":{"DeviceRequests":[{"Count":-1,"DeviceIDs":null}]},"Config":{"Env":["NVIDIA_VISIBLE_DEVICES=all"]}}]' ;;
  *) echo "Error: No such object: $2" >&2; exit 1 ;;
esac
DOCK
chmod +x "$GPU_STUB/nvidia-smi" "$GPU_STUB/docker"
scope() {  # scope <container> -> "<label>|<vram MB>|<selectors>"
  PATH="$GPU_STUB:$PATH" python3 - "$ROOT_DIR/scripts/lib" "$1" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import concurrency_probe as c
sel, _ = c.container_gpus(sys.argv[2])
print(f"{c.gpu_label(sel)}|{c.vram_used_mb(sel)}|{sel}")
PY
}
for ctr in pinned-ids pinned-uuid pinned-env; do
  got="$(scope "$ctr")"
  [[ "$got" == "1× GeForce RTX 3060|10381|"* ]] || fail "container $ctr (the RTX 3060 only): got '$got', want 1× GeForce RTX 3060 and 10381 MB"
done
# device_ids: ["${ESTATE_GPUS}"] with ESTATE_GPUS=2,3 reaches docker as ONE entry "2,3" (#1537).
got="$(scope pinned-joined)"
[[ "$got" == "1× GeForce RTX 3090 + 1× GeForce RTX 3060|25381|"* ]] \
  || fail "a comma-joined DeviceIDs entry (\"2,3\") must scope to GPUs 2 and 3: got '$got'"
got="$(scope sees-all)"
[[ "$got" == "3× GeForce RTX 3090 + 1× GeForce RTX 3060|55319|None" ]] \
  || fail "a container that sees every GPU should cover all four, labelled by name: got '$got'"
got="$(scope no-such-container)"
[[ "$got" == *"|55319|None" ]] || fail "an unresolvable container falls back to rig-wide: got '$got'"
got="$(PATH="$GPU_STUB:$PATH" CONTAINER=pinned-ids python3 "$LIB" --gpu-label)"
[[ "$got" == "1× GeForce RTX 3060" ]] || fail "--gpu-label (the card's GPU field) for the 3060 container: got '$got'"
rm -rf "$GPU_STUB"
echo "  ✓ VRAM + GPU label scoped to the container's GPUs (ids, UUIDs, NVIDIA_VISIBLE_DEVICES; rig-wide fallback)"

# #1537: the card's container and spec label. An SGLang container was never found (name
# heuristic `vllm-(qwen|gemma)`), so its card listed every host GPU and "spec ?"; a vLLM compose that
# builds --speculative-config in its entrypoint from SPEC_N read "spec off" with MTP n=4 running.
SPEC_STUB="$(mktemp -d)"
cat > "$SPEC_STUB/docker" <<'DOCK'
#!/usr/bin/env bash
case "$1" in
  ps)
    echo "sglang-qwen38-27b-mtp-single|0.0.0.0:8144->30000/tcp, [::]:8144->30000/tcp"
    echo "other|0.0.0.0:18144->8000/tcp"
    echo "vllm-qwen38-27b-single-fast|0.0.0.0:8117->8000/tcp" ;;
  logs)
    case "$2" in
      vllm-mtp)
        # The engine-config line sits at the HEAD of the log, then 3000 lines of traffic: a
        # --tail 2500 read (what --detect-kv does) would never see it.
        echo "INFO [core.py:123] Initializing a V1 LLM engine (v0.30.0) with config: model='/m', speculative_config=SpeculativeConfig(method='mtp', model='/m', num_spec_tokens=4), tokenizer=/m"
        for i in $(seq 1 3000); do echo "INFO [loggers.py] Engine 000: Avg generation throughput: 80.0 tokens/s, Running: 1 reqs, Waiting: 0 reqs"; done ;;
      vllm-off) echo "INFO Initializing a V1 LLM engine (v0.30.0) with config: model='/m', speculative_config=None, tokenizer=/m" ;;
      *) exit 1 ;;
    esac ;;
  *) exit 0 ;;
esac
DOCK
chmod +x "$SPEC_STUB/docker"
got="$(PATH="$SPEC_STUB:$PATH" URL=http://localhost:8144 python3 "$LIB" --container-for-url)"
[[ "$got" == "sglang-qwen38-27b-mtp-single" ]] || fail "--container-for-url :8144 should find the SGLang container: got '$got'"
got="$(PATH="$SPEC_STUB:$PATH" URL=http://localhost:9999 python3 "$LIB" --container-for-url)"
[[ -z "$got" ]] || fail "--container-for-url on a port nobody publishes must be empty: got '$got'"
got="$(PATH="$SPEC_STUB:$PATH" CONTAINER=vllm-mtp URL=http://127.0.0.1:9 python3 "$LIB" --spec-label)"
[[ "$got" == "MTP n=4" ]] || fail "--spec-label from vLLM's engine-config line at the log head: got '$got', want 'MTP n=4'"
got="$(PATH="$SPEC_STUB:$PATH" CONTAINER=vllm-off URL=http://127.0.0.1:9 python3 "$LIB" --spec-label)"
[[ "$got" == "spec off" ]] || fail "--spec-label for speculative_config=None: got '$got', want 'spec off'"
got="$(PATH="$SPEC_STUB:$PATH" CONTAINER=no-such URL=http://127.0.0.1:9 python3 "$LIB" --spec-label)"
[[ -z "$got" ]] || fail "--spec-label with nothing readable must be empty (the script then falls back to flags): got '$got'"

# SGLang: the label comes from /get_server_info. A local stand-in serves the shape our MTP composes
# report (EAGLE over the target's own checkpoint).
SGL_PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
python3 -c '
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
INFO = {"model_path": "/models/q", "speculative_algorithm": "EAGLE",
        "speculative_draft_model_path": "/models/q", "speculative_num_steps": 4}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/get_server_info"
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(INFO if ok else {}).encode())
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$SGL_PORT" &
SGL_PID=$!
for _ in $(seq 1 50); do curl -s -m 1 "http://127.0.0.1:$SGL_PORT/get_server_info" >/dev/null 2>&1 && break; sleep 0.1; done
got="$(PATH="$SPEC_STUB:$PATH" CONTAINER= URL="http://127.0.0.1:$SGL_PORT" python3 "$LIB" --spec-label)"
kill "$SGL_PID" 2>/dev/null || true
wait "$SGL_PID" 2>/dev/null || true
[[ "$got" == "MTP n=4" ]] || fail "--spec-label from SGLang's server info: got '$got', want 'MTP n=4'"

# The script must USE the two helpers (the library being right is no help if the card never asks it).
command grep -qF -- '--container-for-url' "$PROBE" || fail "concurrency-probe.sh no longer resolves CONTAINER from URL's port"
command grep -qF -- '--spec-label' "$PROBE" || fail "concurrency-probe.sh's _spec_fp no longer asks the engine"
rm -rf "$SPEC_STUB"
echo "  ✓ container found by URL port (any engine); spec label from the engine (vLLM log head, SGLang server info)"

# #1537 follow-up: the KV pool, the slot count's source and the recommendation's knob names.
# xtj7's cards read "KV ?" on both engines, the SGLang sweep header said "slots=4 (undetected)"
# with the count right, and the SGLang recommendation said MAX_NUM_SEQS (a vLLM knob).
got="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import concurrency_probe as c; print(c.parse_kv_tokens_text("[2026-10-05] max_total_num_tokens=1059144, chunked_prefill_size=2048"))' "$ROOT_DIR/scripts/lib")"
[[ "$got" == "1059144" ]] || fail "SGLang's max_total_num_tokens boot line should parse as the KV pool: got '$got'"

KV_STUB="$(mktemp -d)"
cat > "$KV_STUB/docker" <<'DOCK'
#!/usr/bin/env bash
case "$1" in
  logs)
    if [ "$2" = "--tail" ]; then
      # The --tail window holds only traffic: the boot line has scrolled out of it.
      for i in $(seq 1 50); do echo "INFO Engine 000: Running: 1 reqs, Waiting: 0 reqs"; done
    else
      echo "INFO [kv_cache_utils.py] GPU KV cache size: 1,019,004 tokens, Maximum concurrency for 262,144 tokens per request: 3.89x"
      for i in $(seq 1 3000); do echo "INFO Engine 000: Running: 1 reqs, Waiting: 0 reqs"; done
    fi ;;
  *) exit 0 ;;
esac
DOCK
chmod +x "$KV_STUB/docker"
got="$(PATH="$KV_STUB:$PATH" CONTAINER=vllm-long-running URL=http://127.0.0.1:9 python3 "$LIB" --detect-kv)"
[[ "$got" == "1019004" ]] || fail "--detect-kv must fall back to the log head when the boot line left the --tail window: got '$got'"

# SGLang: KV and the slot source both come from /get_server_info. Drive the real script (dry sweep)
# against a local stand-in, so the header line users see is what's asserted.
SGL2_PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
python3 -c '
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
INFO = {"model_path": "/m", "max_running_requests": 4, "max_total_num_tokens": 547147,
        "speculative_algorithm": None}
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/get_server_info"
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(INFO if ok else {}).encode())
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$SGL2_PORT" &
SGL2_PID=$!
for _ in $(seq 1 50); do curl -s -m 1 "http://127.0.0.1:$SGL2_PORT/get_server_info" >/dev/null 2>&1 && break; sleep 0.1; done
got_kv="$(PATH="$KV_STUB:$PATH" CONTAINER= URL="http://127.0.0.1:$SGL2_PORT" python3 "$LIB" --detect-kv)"
hdr="$(URL="http://127.0.0.1:$SGL2_PORT" N_LIST="1 2 4" CTX_SWEEP="1k" SWEEP_DRY=1 bash "$PROBE" --sweep 2>&1 | head -1)"
kill "$SGL2_PID" 2>/dev/null || true
wait "$SGL2_PID" 2>/dev/null || true
rm -rf "$KV_STUB"
[[ "$got_kv" == "547147" ]] || fail "--detect-kv from SGLang's server info: got '$got_kv', want 547147"
[[ "$hdr" == *"slots=4 (server max_running_requests)"* ]] || fail "sweep header should name the slot source: got '$hdr'"
[[ "$hdr" == *"KV=547147"* ]] || fail "sweep header should carry SGLang's KV pool: got '$hdr'"

card_sgl="$(python3 "$LIB" --card <<'JSON'
{
  "model": "qwen3.8-27b", "slug": "sgl/qwen38-27b-single-fast", "spec": "MTP n=4",
  "gpus": "1× CMP 170HX", "kv_tokens": 1059144, "slots": 4, "served_max_len": null,
  "engine": "sglang", "gen_tokens": 256, "cache": "shared 75%",
  "command": "bash scripts/concurrency-probe.sh --sweep",
  "rows": [
    {"ctx": 1024, "n": 1, "strm": 94.1, "agg": 92, "ttft_s": 0.1, "vram_gb": 58.2, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 1024, "n": 4, "strm": 85.0, "agg": 314, "ttft_s": 0.2, "vram_gb": 58.4, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 1, "strm": 84.0, "agg": 81, "ttft_s": 0.1, "vram_gb": 58.4, "clean": 1, "pass": 1, "skip": null},
    {"ctx": 16384, "n": 4, "strm": 69.7, "agg": 261, "ttft_s": 0.2, "vram_gb": 58.4, "clean": 1, "pass": 1, "skip": null}
  ]
}
JSON
)"
command grep -q "MAX_RUNNING_REQUESTS=4" <<<"$card_sgl" || fail "an SGLang recommendation should use MAX_RUNNING_REQUESTS"
command grep -q "CONTEXT_LENGTH=" <<<"$card_sgl" || fail "an SGLang recommendation should use CONTEXT_LENGTH"
command grep -q "MAX_NUM_SEQS\|MAX_MODEL_LEN\|max-model-len" <<<"$card_sgl" && fail "an SGLang recommendation must not name vLLM's knobs"
echo "  ✓ KV pool (SGLang server info; vLLM log head past the --tail window), slot source, SGLang knob names"

# #1537 follow-up: the served context ("max-len"). A grep of the container's flags for
# `max-model-len N` read "?" for vLLM auto-fit (`--max-model-len -1`) and for every SGLang compose
# (`--context-length`, set in the entrypoint); it feeds the header, the planner's ctx clip and
# VALIDATE's default fill. The engine's own number wins now.
ENGINE_STUB_PY='
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
mode, port = sys.argv[1], int(sys.argv[2])
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if mode == "vllm" and self.path == "/v1/models":
            code, body = 200, {"object": "list", "data": [{"id": "qwen3.8-27b", "max_model_len": 262144}]}
        elif mode == "sglang" and self.path == "/get_server_info":
            code, body = 200, {"model_path": "/m", "context_length": 32768, "max_running_requests": 1}
        else:
            code, body = 404, {"detail": "Not Found"}
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
'
free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'; }
wait_up() { for _ in $(seq 1 50); do curl -s -m 1 "http://127.0.0.1:$1/" >/dev/null 2>&1 && return 0; sleep 0.1; done; }

VPORT="$(free_port)"; python3 -c "$ENGINE_STUB_PY" vllm "$VPORT" & VPID=$!
SPORT="$(free_port)"; python3 -c "$ENGINE_STUB_PY" sglang "$SPORT" & SPID=$!
wait_up "$VPORT"; wait_up "$SPORT"
got_v="$(URL="http://127.0.0.1:$VPORT" CONTAINER= python3 "$LIB" --served-max-len)"
got_s="$(URL="http://127.0.0.1:$SPORT" CONTAINER= python3 "$LIB" --served-max-len)"
hdr_v="$(URL="http://127.0.0.1:$VPORT" N_LIST="1 2" CTX_SWEEP="1k" SWEEP_DRY=1 bash "$PROBE" --sweep 2>&1 | head -1)"
hdr_s="$(URL="http://127.0.0.1:$SPORT" N_LIST="1" CTX_SWEEP="1k" SWEEP_DRY=1 bash "$PROBE" --sweep 2>&1 | head -1)"
kill "$VPID" "$SPID" 2>/dev/null || true
wait "$VPID" "$SPID" 2>/dev/null || true
[[ "$got_v" == "262144" ]] || fail "--served-max-len from vLLM's /v1/models max_model_len: got '$got_v', want 262144"
[[ "$got_s" == "32768" ]] || fail "--served-max-len from SGLang's context_length: got '$got_s', want 32768"
[[ "$hdr_v" == *"max-len=262144"* ]] || fail "sweep header should carry vLLM's served context: got '$hdr_v'"
[[ "$hdr_s" == *"max-len=32768"* ]] || fail "sweep header should carry SGLang's served context: got '$hdr_s'"

CTX_STUB="$(mktemp -d)"
cat > "$CTX_STUB/docker" <<'DOCK'
#!/usr/bin/env bash
if [ "$1" = logs ]; then
  echo "INFO Initializing a V1 LLM engine (v0.30.0) with config: model='/m', max_seq_len=81920, speculative_config=None"
  for i in $(seq 1 3000); do echo "INFO Engine 000: Running: 1 reqs, Waiting: 0 reqs"; done
fi
exit 0
DOCK
chmod +x "$CTX_STUB/docker"
got="$(PATH="$CTX_STUB:$PATH" CONTAINER=vllm-x URL=http://127.0.0.1:9 python3 "$LIB" --served-max-len)"
rm -rf "$CTX_STUB"
[[ "$got" == "81920" ]] || fail "--served-max-len should fall back to vLLM's max_seq_len boot line: got '$got'"
echo "  ✓ served context from the engine (vLLM /v1/models, SGLang context_length, vLLM boot line)"

echo "test-concurrency-probe: ok"
