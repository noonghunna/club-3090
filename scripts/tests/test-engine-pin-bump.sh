#!/usr/bin/env bash
# Guard for scripts/engine-pin-bump.sh: syntax, refusal paths, --check is
# write-free against the real repo, and a full write-mode run against a
# synthetic PIN_BUMP_ROOT fixture (spec + display_name + functional compose
# edited; deprecated compose untouched).
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
S="${ROOT_DIR}/scripts/engine-pin-bump.sh"

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }

# 1. syntax
bash -n "$S" || fail "bash -n syntax check"

# 2. refusal paths → exit 2
set +e
bash "$S" >/dev/null 2>&1; [[ $? -eq 2 ]] || fail "no-arg should exit 2"
bash "$S" nosuch-engine v1 >/dev/null 2>&1; [[ $? -eq 2 ]] || fail "unknown engine should exit 2"
bash "$S" vllm-stable "$(command grep -m1 '^  spec:' "$ROOT_DIR/scripts/lib/profiles/engines/vllm-stable.yml" | awk '{print $2}' | cut -d: -f2)" >/dev/null 2>&1
[[ $? -eq 2 ]] || fail "no-op (current tag) should exit 2"
bash "$S" beellama-local sometag >/dev/null 2>&1; [[ $? -eq 2 ]] || fail "digest engine w/o digest ref should exit 2"
set -e

# 3. --check against the real repo: exits 0, plans the engine yml, writes nothing
out="$(bash "$S" vllm-stable v999.0.0-guardtest --check 2>&1)" || fail "--check should exit 0"
command grep -q "would edit" <<<"$out" || fail "--check output should say 'would edit'"
command grep -q "engines/vllm-stable.yml" <<<"$out" || fail "--check should plan the engine yml"
command grep -q "v999.0.0-guardtest" <<<"$out" || fail "--check should show the new tag"
if command -v git >/dev/null && git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT_DIR" diff --quiet -- scripts/lib/profiles/engines/vllm-stable.yml \
    || fail "--check must not modify the engine yml"
fi

# 4. write mode against a synthetic fixture root
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scripts/lib/profiles/engines" "$TMP/models/fake/compose" "$TMP/docs" "$TMP/services" "$TMP/c3" "$TMP/scripts/tests"
cat > "$TMP/scripts/lib/profiles/engines/fake.yml" <<'YML'
schema_version: 1
id: fake-engine
display_name: Fake engine v1.0.0 (guard fixture)
install:
  method: docker_image
  spec: example/fake:v1.0.0
YML
cat > "$TMP/models/fake/compose/live.yml" <<'YML'
services:
  fake:
    image: ${FAKE_IMAGE:-example/fake:v1.0.0}
YML
cat > "$TMP/models/fake/compose/dead.yml" <<'YML'
services:
  fake:
    image: ${FAKE_IMAGE:-example/fake:v1.0.0}
YML
cat > "$TMP/scripts/lib/profiles/compose_registry.py" <<'PYF'
COMPOSE_REGISTRY = {
    "fake/live": {"engine": "fake-engine", "compose_path": "models/fake/compose/live.yml",
                  "status": "experimental"},
    "fake/dead": {"engine": "fake-engine", "compose_path": "models/fake/compose/dead.yml",
                  "status": "deprecated"},
}
PYF
out="$(PIN_BUMP_ROOT="$TMP" bash "$S" fake-engine v2.0.0 2>&1)" || fail "fixture write run should exit 0"
command grep -q "spec: example/fake:v2.0.0" "$TMP/scripts/lib/profiles/engines/fake.yml" || fail "spec not bumped"
command grep -q "display_name: Fake engine v2.0.0" "$TMP/scripts/lib/profiles/engines/fake.yml" || fail "display_name not bumped"
command grep -q "example/fake:v2.0.0" "$TMP/models/fake/compose/live.yml" || fail "functional (experimental) compose not bumped"
command grep -q "example/fake:v1.0.0" "$TMP/models/fake/compose/dead.yml" || fail "deprecated compose must stay untouched"
command grep -q "fake/dead" <<<"$out" || fail "deprecated skip should be listed"

echo "PASS test-engine-pin-bump"
