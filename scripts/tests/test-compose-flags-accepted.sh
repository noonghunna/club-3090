#!/usr/bin/env bash
# test-compose-flags-accepted — does the pinned image actually ACCEPT every flag
# its composes pass? (#1370)
#
# THE CLASS THIS CATCHES. Nothing asserted this before, and the whole class was
# invisible: test-compose-status-drift checks headers, the pull gate checks
# weights, verify-full runs only AFTER a successful boot. So when mainline
# removed `--no-mmap` and the llama-cpp-local pin moved to b10920 (2026-09-12),
# three shipped slugs became unbootable — `error: invalid argument: --no-mmap`
# — and the suite stayed green for nine days.
#
# ⚠️ SCOPED TO THE llama.cpp FAMILY ON PURPOSE. Its `--help` prints a clean
# flag column and its parser rejects an unknown flag outright, so a green here
# means something. vLLM/SGLang take `--help=all`, accept many aliases and route
# unknown args differently; a sloppy parse of those would produce false
# positives, which trains people to add allowlist entries. Extend deliberately,
# with a negative control, not by widening the glob.
#
# ⚠️ IMAGES THAT ARE NOT LOCAL CANNOT BE CHECKED. Skipping is honest; skipping
# SILENTLY is the false-clean this repo has been bitten by repeatedly. Every skip
# is printed, and the test FAILS if it ended up checking nothing at all.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAIL=0; CHECKED=0; SKIPPED=0; SKIPPED_FORK=0; DEPRECATED_HITS=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }

command -v docker >/dev/null 2>&1 || { echo "  ⊘ docker unavailable — skipping"; echo "test-compose-flags-accepted: skipped"; exit 0; }

# (engine_id, image, [compose paths...]) for llama.cpp-family engines
mapfile -t ROWS < <(python3 - <<'PY'
import sys, yaml, pathlib
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
by_engine = {}
for slug, e in COMPOSE_REGISTRY.items():
    by_engine.setdefault(e.get("engine"), []).append(e.get("compose_path") or "")
for f in sorted(pathlib.Path("scripts/lib/profiles/engines").glob("*.yml")):
    d = yaml.safe_load(f.read_text())
    inst = d.get("install") or {}
    if d.get("type") != "llama.cpp" or inst.get("method") != "docker_image":
        continue
    paths = [p for p in by_engine.get(d["id"], []) if p]
    if paths:
        print("\t".join([d["id"], str(inst["spec"]), *paths]))
PY
)

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r eid spec rest <<<"$row"
  IFS=$'\t' read -ra composes <<<"$rest"
  if ! docker image inspect "$spec" >/dev/null 2>&1; then
    echo "  ⊘ ${eid}: image not present locally — ${#composes[@]} compose(s) UNCHECKED"
    SKIPPED=$((SKIPPED + ${#composes[@]})); continue
  fi
  # The accepted-flag set, straight from the binary that will run. `--help` exits
  # before any model load, so this costs one container start per ENGINE.
  # ⚠️ --gpus all: the prism images' `--help` ABORTS without libcuda, and the
  # club3090 `-ot` parser only knows CUDA buffer types when a device is visible.
  # Without it the extraction returns zero flags and every compose reads as broken.
  help_out="$(docker run --rm --gpus all --entrypoint /app/llama-server "$spec" --help 2>&1)"
  accepted="$(printf '%s' "$help_out" \
    | command grep -oE '(^|[[:space:],])--?[A-Za-z][A-Za-z0-9-]*' \
    | tr -d ' ,' | sort -u)"
  if [[ -z "$accepted" ]]; then
    bad "${eid}: parsed ZERO flags out of --help — the extraction broke, not the composes"
    continue
  fi
  for c in "${composes[@]}"; do
    [[ -f "$c" ]] || continue
    # Only a LIST `command:` is a clean argv. A bash -c entrypoint hides flags in
    # a script body; those are reported as uncovered rather than half-parsed.
    # ⚠️ ONE ENGINE ID, TWO BINARIES. ik-llama/* slugs register against
    # `llama-cpp-local` but their composes pin ${IK_LLAMA_IMAGE:-…}, a different
    # fork with a different flag set (that split is why #1365 gave the profile
    # `image_env: null`). Checking an ik compose against the MAINLINE binary
    # would be a confident wrong answer in both directions. Today these composes
    # have no list `command:` so they fall out below anyway — luck, not design.
    compose_img="$(command sed -nE 's/^[[:space:]]*image:[[:space:]]*"?\$\{[A-Z_0-9]+:-([^}"]+)\}"?.*/\1/p;s/^[[:space:]]*image:[[:space:]]*"?([^$"[:space:]]+)"?[[:space:]]*$/\1/p' "$c" | head -1)"
    if [[ -n "$compose_img" && "$compose_img" != "$spec" ]]; then
      SKIPPED_FORK=$((SKIPPED_FORK + 1)); continue
    fi
    flags="$(python3 - "$c" <<'PY'
import sys, yaml, re
d = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
svc = next(iter((d.get("services") or {}).values()), {})
cmd = svc.get("command")
if not isinstance(cmd, list):
    sys.exit(0)
for tok in cmd:
    t = str(tok)
    if re.fullmatch(r"--?[A-Za-z][A-Za-z0-9-]*", t):
        print(t)
PY
)"
    [[ -n "$flags" ]] || continue
    CHECKED=$((CHECKED + 1))
    unknown=""
    while read -r fl; do
      [[ -n "$fl" ]] || continue
      # ⚠️ `--` is load-bearing: every $fl starts with a dash, so without it grep
      # parses the FLAG as its own option and errors on each one -- which made
      # every flag look "unknown" and reported all 70 composes as broken.
      command grep -qxF -- "$fl" <<<"$accepted" || unknown="${unknown}${unknown:+ }${fl}"
    done <<<"$flags"
    # ⚠️ ACCEPTED-BUT-DEPRECATED is the state that produced #1370. --no-mmap was
    # accepted by every fork, with a DEPRECATED marker in --help, right up until
    # mainline deleted it and three slugs stopped booting. A guard that only asks
    # "is it accepted?" cannot see that coming, so surface it -- as a WARNING,
    # not a failure: a deprecated flag is a schedule risk, not a broken config,
    # and failing on it would block a legitimate pin bump.
    deprecated=""
    while read -r fl; do
      [[ -n "$fl" ]] || continue
      # ⚠️ `${fl}`, NOT `\${fl}` — the escaped form makes grep search for the
      # LITERAL string "${fl}", matches nothing, and the check reports a clean
      # run forever. Caught only by a positive control.
      command grep -E -- "(^|[[:space:],])${fl}([[:space:],]|$)" <<<"$help_out" \
        | command grep -qi "DEPRECATED" && deprecated="${deprecated}${deprecated:+ }${fl}"
    done <<<"$flags"
    if [[ -n "$deprecated" ]]; then
      echo "  ⚠ $(echo "$c" | sed 's|models/||'): ${eid} marks these DEPRECATED: ${deprecated}"
      DEPRECATED_HITS=$((DEPRECATED_HITS + 1))
    fi
    if [[ -n "$unknown" ]]; then
      bad "$(echo "$c" | sed 's|models/||') passes flag(s) its pin (${eid}) does not accept: ${unknown}"
      echo "       the pinned parser rejects an unknown flag outright — this slug cannot boot" >&2
    fi
  done
done

echo "  checked ${CHECKED} compose(s); ${SKIPPED} skipped (image not local); ${SKIPPED_FORK} skipped (compose pins a different fork than its engine profile)"
if (( DEPRECATED_HITS )); then
  echo "  ⚠ ${DEPRECATED_HITS} compose(s) pass a flag their pin marks DEPRECATED — not a failure, but it is how #1370 happened"
fi
# ⚠️ A run that checked NOTHING is not a pass. This is the guard's own negative
# control: without it, deleting every local image would turn this green.
if (( CHECKED == 0 )); then
  echo "  ✗ checked zero composes — a green here would be meaningless" >&2
  FAIL=1
fi
(( FAIL )) && { echo "test-compose-flags-accepted: FAIL" >&2; exit 1; }
ok "every checked compose's flags are accepted by its own pinned image"
echo "test-compose-flags-accepted: ok"
