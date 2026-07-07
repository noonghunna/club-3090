#!/usr/bin/env bash
# test-pod-cli.sh — scripts/pod.sh + the estate_cli pod verbs
# (#610 Phase A′). Runs hardware-free via CLUB3090_FAKE_GPUS (kv-calc prices
# by card NAME, no real GPU needed) so CI exercises the full lifecycle:
#   create (with D1 fit) · count!=TP hard-reject · GPU-collision reject ·
#   list --json · status · rm · the index-based estate file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Two fake RTX 3090s so a TP=2 slug fits and a TP=1 slug leaves GPU 0 free.
export CLUB3090_FAKE_GPUS='0:NVIDIA GeForce RTX 3090:24576:8.6,1:NVIDIA GeForce RTX 3090:24576:8.6'
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
EST="$TMP/estate.yml"
CL=(bash scripts/pod.sh)

fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. create a TP=1 pod on GPU 1 — must fit + write the file.
"${CL[@]}" create chat --gpus 1 --slug vllm/minimal --file "$EST" >/dev/null 2>&1 \
  || fail "create chat (TP=1 on GPU 1) should succeed"
[[ -f "$EST" ]] || fail "estate file not written"
grep -q 'name: chat' "$EST" || fail "chat not in estate file"
# Indices stay index-based in the file (UUIDs resolved at boot, #610 Phase A).
grep -qE '^\s*-\s*1\s*$' "$EST" || fail "estate file should store index-based gpus"

# 2. D1: count != compose TP -> hard reject (vllm/dual is TP=2).
if "${CL[@]}" create bad --gpus 1 --slug vllm/dual --file "$EST" >/dev/null 2>&1; then
  fail "create with 1 GPU for a TP=2 slug must be rejected (D1)"
fi
err="$("${CL[@]}" create bad --gpus 1 --slug vllm/dual --file "$EST" 2>&1 || true)"
grep -q 'TP=2' <<<"$err" || fail "D1 rejection should name the TP mismatch (got: $err)"

# 3. GPU collision -> validate_estate reject (GPU 1 already claimed by chat).
if "${CL[@]}" create chat2 --gpus 1 --slug vllm/minimal --file "$EST" >/dev/null 2>&1; then
  fail "second pod on the same GPU must be rejected"
fi

# 4. list --json: one pod, GPU 0 free.
json="$("${CL[@]}" list --json --file "$EST" 2>/dev/null)"
python3 - "$json" <<'PY' || exit 1
import json, sys
d = json.loads(sys.argv[1])
assert len(d["pods"]) == 1, d
assert d["pods"][0]["name"] == "chat", d
assert d["free_gpus"] == [0], d
assert d["claimed_gpus"] == [1], d
print("  list --json ok")
PY

# 5. status (nothing running) -> down, placement unknown, valid --json shape.
sjson="$("${CL[@]}" status --json --file "$EST" 2>/dev/null)"
python3 - "$sjson" <<'PY' || exit 1
import json, sys
d = json.loads(sys.argv[1])
c = d["pods"][0]
assert c["running"] is False, c
assert c["placement"]["placement"] == "unknown", c
assert set(c["placement"]) == {"requested", "actual", "placement"}, c
print("  status --json ok")
PY

# 6. a second, non-colliding pod on GPU 0 succeeds.
"${CL[@]}" create chat0 --gpus 0 --slug vllm/minimal --file "$EST" >/dev/null 2>&1 \
  || fail "create chat0 on the free GPU 0 should succeed"

# 7. rm removes it from the plan.
"${CL[@]}" rm chat0 --file "$EST" >/dev/null 2>&1 || fail "rm chat0 should succeed"
grep -q 'name: chat0' "$EST" && fail "chat0 still in estate file after rm"

echo "PASS: pod.sh create/list/status/rm + D1 count!=TP + GPU-collision rejects (fake-GPU)"
