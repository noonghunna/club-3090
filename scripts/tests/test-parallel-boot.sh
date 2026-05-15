#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

FAKE_GPUS_4X='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6,2:RTX_3090:24576:8.6,3:RTX_3090:24576:8.6'

ESTATE4="${TMP_DIR}/estate4.yml"
cat > "$ESTATE4" <<'YAML'
schema_version: 1
created: 2026-05-15T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 4
  nvlink_active: false
estate:
  - name: one
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: two
    compose: llamacpp/default
    gpus: [1]
    port: 8120
  - name: three
    compose: llamacpp/default
    gpus: [2]
    port: 8130
  - name: four
    compose: llamacpp/default
    gpus: [3]
    port: 8140
YAML

FAKE_ESTATE_HELPER="${TMP_DIR}/estate-helper.py"
cat > "$FAKE_ESTATE_HELPER" <<'PY'
#!/usr/bin/env python3
import sys
print("ARGS " + " ".join(sys.argv[1:]))
PY
chmod +x "$FAKE_ESTATE_HELPER"

out="$(ESTATE_HELPER="$FAKE_ESTATE_HELPER" bash "${ROOT_DIR}/scripts/launch.sh" \
  --no-preflight \
  --estate-file "$ESTATE4" \
  --only one \
  --parallel \
  --parallel-jobs 3 \
  --parallel-stagger 0 2>&1)"
assert_contains "$out" "ARGS boot --file $ESTATE4 --only one --parallel --parallel-jobs 3 --parallel-stagger 0"

out="$(
  cd "$ROOT_DIR"
  HOME="${TMP_DIR}/home-success" \
  CLUB3090_FAKE_GPUS="$FAKE_GPUS_4X" \
  CLUB3090_ESTATE_BOOT_LOG_DIR="${TMP_DIR}/logs-success" \
  python3 - "$ESTATE4" <<'PY'
import argparse
import sys
import threading
import time

from scripts.lib.profiles import estate_cli as ec

state = {"active": 0, "max_active": 0}
ready = []
lock = threading.Lock()


def fake_run_compose(inst, action, log_path=None):
    if log_path is not None:
        ec.append_log(log_path, f"compose {action} {inst.name}")
    with lock:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
    time.sleep(0.05)
    with lock:
        state["active"] -= 1


def fake_wait_ready_quiet(inst, timeout):
    ready.append(inst.name)
    return 1


ec.run_compose = fake_run_compose
ec.wait_ready_quiet = fake_wait_ready_quiet

rc = ec.command_boot(
    argparse.Namespace(
        file=sys.argv[1],
        only="",
        timeout=5,
        parallel=True,
        parallel_jobs=2,
        parallel_stagger=0.0,
    )
)
print(f"rc={rc}")
print(f"max_active={state['max_active']}")
print("ready=" + ",".join(sorted(ready)))
for name in ("one", "two", "three", "four"):
    print(f"log:{name}={ec.instance_log_path(ec.InstanceSpec(name=name, compose_name='llamacpp/default', gpu_indices=(0,), port=8000)).exists()}")
print(f"hard_cap={ec.effective_parallel_jobs(9, 8)}")
PY
)"
assert_contains "$out" "[estate] parallel boot: 4 instance(s), jobs=2, stagger=0s"
assert_contains "$out" "[estate] Summary: 4/4 healthy, 0 failed."
assert_contains "$out" "rc=0"
assert_contains "$out" "max_active=2"
assert_contains "$out" "ready=four,one,three,two"
assert_contains "$out" "log:one=True"
assert_contains "$out" "log:four=True"
assert_contains "$out" "hard_cap=4"

ESTATE2="${TMP_DIR}/estate2.yml"
cat > "$ESTATE2" <<'YAML'
schema_version: 1
created: 2026-05-15T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: good
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: bad
    compose: llamacpp/default
    gpus: [1]
    port: 8120
YAML

out="$(
  cd "$ROOT_DIR"
  HOME="${TMP_DIR}/home-fail" \
  CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6' \
  CLUB3090_ESTATE_BOOT_LOG_DIR="${TMP_DIR}/logs-fail" \
  python3 - "$ESTATE2" <<'PY'
import argparse
import sys

from scripts.lib.profiles import estate_cli as ec


def fake_run_compose(inst, action, log_path=None):
    if log_path is not None:
        ec.append_log(log_path, f"compose {action} {inst.name}")
    if inst.name == "bad":
        raise ec.EstateCliError("mock compose failed")


def fake_wait_ready_quiet(inst, timeout):
    return 2


ec.run_compose = fake_run_compose
ec.wait_ready_quiet = fake_wait_ready_quiet

rc = ec.command_boot(
    argparse.Namespace(
        file=sys.argv[1],
        only="",
        timeout=5,
        parallel=True,
        parallel_jobs=2,
        parallel_stagger=0.0,
    )
)
print(f"rc={rc}")
PY
)"
assert_contains "$out" "[estate] Summary: 1/2 healthy, 1 failed."
assert_contains "$out" "bad ✗ failed"
assert_contains "$out" "Failed instance: bad. See"
assert_contains "$out" "rc=1"

out="$(
  cd "$ROOT_DIR"
  HOME="${TMP_DIR}/home-single" \
  CLUB3090_FAKE_GPUS="$FAKE_GPUS_4X" \
  CLUB3090_ESTATE_BOOT_LOG_DIR="${TMP_DIR}/logs-single" \
  python3 - "$ESTATE4" <<'PY'
import argparse
import sys

from scripts.lib.profiles import estate_cli as ec

events = []


def fake_run_compose(inst, action):
    events.append(f"{action}:{inst.name}")


def fake_wait_ready(inst, timeout):
    events.append(f"ready:{inst.name}")


def fake_wait_ready_quiet(inst, timeout):
    raise AssertionError("single-instance --parallel should use sequential boot")


ec.run_compose = fake_run_compose
ec.wait_ready = fake_wait_ready
ec.wait_ready_quiet = fake_wait_ready_quiet

rc = ec.command_boot(
    argparse.Namespace(
        file=sys.argv[1],
        only="one",
        timeout=5,
        parallel=True,
        parallel_jobs=2,
        parallel_stagger=0.0,
    )
)
print(f"rc={rc}")
print("events=" + ",".join(events))
PY
)"
assert_contains "$out" "[estate] all selected instances are healthy"
assert_contains "$out" "rc=0"
assert_contains "$out" "events=up:one,ready:one"

echo "test-parallel-boot: ok"
