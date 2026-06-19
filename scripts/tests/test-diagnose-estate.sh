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

export CLUB3090_FAKE_GPUS='0:RTX_3090:24576:8.6,1:RTX_3090:24576:8.6'

GOOD="${TMP_DIR}/estate-good.yml"
cat > "$GOOD" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: qwen-left
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: qwen-right
    compose: llamacpp/default
    gpus: [1]
    port: 8120
YAML

out="$(bash "${ROOT_DIR}/scripts/launch.sh" --validate-estate "$GOOD" 2>&1)"
assert_contains "$out" "Estate validation: PASS"
assert_contains "$out" "qwen-left: llamacpp/default GPUs=[0] port=8110"

out="$(bash "${ROOT_DIR}/scripts/diagnose-estate.sh" "$GOOD" 2>&1)"
assert_contains "$out" "[1/6] Estate file parses + schema_version supported"
assert_contains "$out" "[4/6] Estate cross-checks E1-E4"
assert_contains "$out" "Triage summary: GREEN"

out="$(python3 "${ROOT_DIR}/scripts/lib/profiles/estate_cli.py" report-state --file "$GOOD" 2>&1)"
assert_contains "$out" "## Profile state"
assert_contains "$out" "Active estate"
assert_contains "$out" "qwen-left: llamacpp/default, GPUs [0], port 8110"

BOOT_HOME="${TMP_DIR}/boot-home"
out="$(
  cd "$ROOT_DIR"
  HOME="$BOOT_HOME" CLUB3090_FAKE_GPUS="$CLUB3090_FAKE_GPUS" python3 - "$GOOD" <<'PY'
import argparse
import sys
import yaml

from scripts.lib.profiles import estate_cli as ec

events = []


def fake_run_compose(inst, action):
    if not ec.DEFAULT_ESTATE_PATH.exists():
        raise AssertionError("default estate was not persisted before container action")
    events.append(f"{action}:{inst.name}")


def fake_wait_ready(inst, timeout):
    events.append(f"ready:{inst.name}")


ec.run_compose = fake_run_compose
ec.wait_ready = fake_wait_ready

rc = ec.command_boot(argparse.Namespace(file=sys.argv[1], only="club3090-qwen-left", timeout=1))
if rc != 0:
    raise SystemExit(rc)

data = yaml.safe_load(ec.DEFAULT_ESTATE_PATH.read_text(encoding="utf-8"))
print("default estate persisted")
print(data["estate"][0]["name"])
print(",".join(events))
PY
)"
assert_contains "$out" "default estate persisted"
assert_contains "$out" "qwen-left"
assert_contains "$out" "up:qwen-left,ready:qwen-left"

out="$(
  cd "$ROOT_DIR"
  CLUB3090_ESTATE_BOOT_LOG_DIR="${TMP_DIR}/boot-logs" python3 - <<'PY'
import yaml

from scripts.lib.profiles import estate_cli as ec
from scripts.lib.profiles.compat import InstanceSpec

inst = InstanceSpec(name="gemma-dual", compose_name="vllm/gemma-int8-mtp", gpu_indices=(2, 3), port=8032)
override_path = ec.write_compose_override(inst)
data = yaml.safe_load(override_path.read_text(encoding="utf-8"))
service = data["services"]["vllm-gemma-4-31b-mtp-int8"]
print(override_path)
print(service["environment"]["CUDA_VISIBLE_DEVICES"])
print(service["environment"]["NVIDIA_VISIBLE_DEVICES"])
PY
)"
assert_contains "$out" "gemma-dual.override.yml"
assert_contains "$out" "2,3"

out="$(
  cd "$ROOT_DIR"
  CLUB3090_ESTATE_BOOT_LOG_DIR="${TMP_DIR}/boot-logs" python3 - <<'PY'
from scripts.lib.profiles import estate_cli as ec
from scripts.lib.profiles.compat import InstanceSpec

calls = []

class Proc:
    returncode = 0

def fake_run(cmd, **kwargs):
    calls.append(cmd)
    return Proc()

ec.subprocess.run = fake_run
inst = InstanceSpec(name="gemma-dual", compose_name="vllm/gemma-int8-mtp", gpu_indices=(2, 3), port=8032)
ec.run_compose(inst, "up")
cmd = calls[0]
print(" ".join(cmd))
print(f"f_count={sum(1 for part in cmd if part == '-f')}")
PY
)"
assert_contains "$out" "gemma-dual.override.yml"
assert_contains "$out" "f_count=2"

FAKE_BIN="${TMP_DIR}/fakebin"
mkdir -p "$FAKE_BIN"
cat > "${FAKE_BIN}/docker" <<'SH'
#!/usr/bin/env bash
args="$*"
case "${1:-}" in
  info)
    if [[ "$args" == *"--format"* ]]; then
      echo "/tmp/docker-root"
    fi
    exit 0
    ;;
  ps)
    if [[ "$args" == *"{{.Names}}"* && "$args" == *"name=club3090-"* ]]; then
      echo "club3090-llama-gpu0"
    elif [[ "$args" == *"{{.Status}}"* && "$args" == *"name=club3090-llama-gpu0"* ]]; then
      echo "Up 2 minutes"
    elif [[ "$args" == *"{{.Ports}}"* && "$args" == *"name=club3090-llama-gpu0"* ]]; then
      echo "0.0.0.0:8010->8010/tcp"
    elif [[ "$args" == *"{{.Image}}"* && "$args" == *"name=club3090-llama-gpu0"* ]]; then
      echo "ghcr.io/ggml-org/llama.cpp:server-cuda"
    fi
    exit 0
    ;;
  logs)
    echo "build_info: fake llama.cpp"
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
SH
chmod +x "${FAKE_BIN}/docker"

out="$(PATH="${FAKE_BIN}:$PATH" HOME="${TMP_DIR}/report-home" bash "${ROOT_DIR}/scripts/report.sh" --no-redact 2>&1)"
assert_contains "$out" "**Name:** \`club3090-llama-gpu0\`"
assert_contains "$out" "**Engine:** \`llamacpp\`"

GPU_COLLISION="${TMP_DIR}/estate-gpu-collision.yml"
cat > "$GPU_COLLISION" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: qwen-left
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: qwen-right
    compose: llamacpp/default
    gpus: [0]
    port: 8120
YAML

if out="$(bash "${ROOT_DIR}/scripts/launch.sh" --validate-estate "$GPU_COLLISION" 2>&1)"; then
  echo "ASSERTION FAILED: GPU collision estate unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "E1: GPU 0 claimed by qwen-left, qwen-right"

PORT_COLLISION="${TMP_DIR}/estate-port-collision.yml"
cat > "$PORT_COLLISION" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: qwen-left
    compose: llamacpp/default
    gpus: [0]
    port: 8110
  - name: qwen-right
    compose: llamacpp/default
    gpus: [1]
    port: 8110
YAML

if out="$(bash "${ROOT_DIR}/scripts/diagnose-estate.sh" "$PORT_COLLISION" 2>&1)"; then
  echo "ASSERTION FAILED: port collision estate unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "E4: port 8110 claimed by qwen-left, qwen-right"

MISSING_COMPOSE="${TMP_DIR}/estate-missing-compose.yml"
cat > "$MISSING_COMPOSE" <<'YAML'
schema_version: 1
created: 2026-05-14T00:00:00Z
rig:
  hardware_id: rtx-3090
  gpu_count: 2
  nvlink_active: false
estate:
  - name: missing
    compose: vllm/not-real
    gpus: [0]
    port: 8110
YAML

if out="$(bash "${ROOT_DIR}/scripts/diagnose-estate.sh" "$MISSING_COMPOSE" 2>&1)"; then
  echo "ASSERTION FAILED: missing-compose estate unexpectedly passed" >&2
  echo "$out" >&2
  exit 1
fi
assert_contains "$out" "vllm/not-real missing from registry"

echo "test-diagnose-estate: ok"
