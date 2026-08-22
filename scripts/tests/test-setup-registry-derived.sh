#!/usr/bin/env bash
set -euo pipefail
#
# Drift guard: setup.sh's front door must agree with `weights.py catalog --json`
# for EVERY model — usage list, labels, resolved dispatch keys, WEIGHTS= alias
# resolution, and the unknown-model error string. This replaces the hand-written
# bash case blocks whose arms new models silently missed (#910-#914 defect
# class): adding a model yml is now enough, and this test fails if anyone
# re-hardcodes any of it.
#
# Uses SETUP_DUMP_KEYS=1 (setup.sh prints its resolved keys and exits before
# preflight/download), so nothing here touches the network or the GPU.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_READER="${ROOT_DIR}/scripts/lib/profiles/weights.py"
export PYTHONUTF8="${PYTHONUTF8:-1}"

fail() { echo "ASSERTION FAILED: $*" >&2; exit 1; }

catalog_field() { # <model-id> <field>
  python3 -c '
import json, sys
models = json.loads(sys.stdin.read())["models"]
m = next((x for x in models if x["id"] == sys.argv[1]), None)
if m is None:
    raise SystemExit(f"unknown catalog model: {sys.argv[1]}")
v = m[sys.argv[2]]
print(v if isinstance(v, str) else json.dumps(v))
' "$1" "$2" < <(python3 "${WEIGHTS_READER}" catalog --json)
}

mapfile -t CATALOG_IDS < <(python3 "${WEIGHTS_READER}" catalog --json | python3 -c '
import json, sys
for m in json.load(sys.stdin)["models"]:
    print(m["id"])
')
(( ${#CATALOG_IDS[@]} > 0 )) || fail "catalog is empty"

dump_keys() { # <model-id> [ENV=value ...]
  local model="$1"; shift
  SETUP_DUMP_KEYS=1 env "$@" bash "${ROOT_DIR}/scripts/setup.sh" "${model}" 2>&1
}

dump_value() { # <dump-output> <key>
  awk -F= -v k="$2" '$1 == k { print substr($0, length(k) + 2); found=1 } END { if (!found) exit 1 }' <<< "$1"
}

usage_ids="$(bash "${ROOT_DIR}/scripts/setup.sh" --help \
  | sed -n '/^Supported model names/,/^$/p' \
  | grep -E '^  [a-z0-9]' | tr -d ' ')"
derived_ids="$(printf '%s\n' "${CATALOG_IDS[@]}")"
[[ "${usage_ids}" == "${derived_ids}" ]] \
  || fail "usage() model list drifted from the catalog:
--- usage ---
${usage_ids}
--- catalog ---
${derived_ids}"

# --- 2. unknown-model error names exactly the catalog ids --------------------
err_out="$(bash "${ROOT_DIR}/scripts/setup.sh" some-unknown-model 2>&1 || true)"
expected_supported="$(python3 "${WEIGHTS_READER}" catalog --json | python3 -c '
import json, sys
print(", ".join(m["id"] for m in json.load(sys.stdin)["models"]))
')"
[[ "${err_out}" == *"unsupported model 'some-unknown-model'"* ]] \
  || fail "unknown-model error lost its header: ${err_out}"
[[ "${err_out}" == *"Supported: ${expected_supported}"* ]] \
  || fail "unknown-model error's Supported list drifted from the catalog (expected: ${expected_supported})"

# --- 3. per-model dispatch keys match the catalog ----------------------------
for id in "${CATALOG_IDS[@]}"; do
  dump="$(dump_keys "${id}")"
  [[ "$(dump_value "${dump}" model)" == "${id}" ]] || fail "${id}: dump model mismatch"
  [[ "$(dump_value "${dump}" label)" == "$(catalog_field "${id}" display_name)" ]] \
    || fail "${id}: label drifted from display_name"
  [[ "$(dump_value "${dump}" primary)" == "$(catalog_field "${id}" default_key)" ]] \
    || fail "${id}: primary key drifted from default_key ($(dump_value "${dump}" primary) != $(catalog_field "${id}" default_key))"
  for field in always_draft dflash vision prism_eagle3; do
    expected="$(catalog_field "${id}" "${field}")"
    [[ -n "${expected}" ]] && expected="${id}:${expected}"
    actual="$(dump_value "${dump}" "${field}")"
    [[ "${actual}" == "${expected}" ]] \
      || fail "${id}: ${field} key drifted (${actual} != ${expected})"
  done
done

# --- 4. every registered WEIGHTS= alias resolves to its catalog variant ------
alias_rows="$(python3 "${WEIGHTS_READER}" catalog --json | python3 -c '
import json, sys
for m in json.load(sys.stdin)["models"]:
    for alias, variant in sorted((m.get("aliases") or {}).items()):
        extras = " ".join(
            "{}:{}".format(m["id"], x) for x in (m.get("alias_extras") or {}).get(alias, [])
        )
        print("{}\t{}\t{}\t{}".format(m["id"], alias, variant, extras))
')"
if [[ -z "${alias_rows}" ]]; then
  echo "NOTE: no WEIGHTS= aliases registered in any setup block — alias parity vacuous" >&2
fi
while IFS=$'\t' read -r id alias variant extras; do
  dump="$(dump_keys "${id}" "WEIGHTS=${alias}")"
  actual_primary="$(dump_value "${dump}" primary)"
  [[ "${actual_primary}" == "${id}:${variant}" ]] \
    || fail "${id}: WEIGHTS=${alias} drifted (${actual_primary} != ${id}:${variant})"
  actual_extras="$(dump_value "${dump}" extras)"
  [[ "${actual_extras}" == "${extras}" ]] \
    || fail "${id}: WEIGHTS=${alias} extras drifted ('${actual_extras}' != '${extras}')"
done <<< "${alias_rows}"

# --- 6. WITH_ASSISTANT_DRAFT honors assistant_draft, errors otherwise --------
for id in "${CATALOG_IDS[@]}"; do
  assistant="$(catalog_field "${id}" assistant_draft)"
  if [[ -n "${assistant}" ]]; then
    dump="$(dump_keys "${id}" "WITH_ASSISTANT_DRAFT=1")"
    [[ "$(dump_value "${dump}" always_draft)" == "${id}:${assistant}" ]] \
      || fail "${id}: WITH_ASSISTANT_DRAFT=1 drifted (expected ${id}:${assistant})"
  else
    if out="$(dump_keys "${id}" "WITH_ASSISTANT_DRAFT=1")"; then
      fail "${id}: WITH_ASSISTANT_DRAFT=1 unexpectedly succeeded without assistant_draft"
    fi
    [[ "${out}" == *"WITH_ASSISTANT_DRAFT=1"* ]] \
      || fail "${id}: WITH_ASSISTANT_DRAFT error message lost its header"
  fi
done

echo "test-setup-registry-derived: ok"
