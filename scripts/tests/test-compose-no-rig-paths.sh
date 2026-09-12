#!/usr/bin/env bash
# test-compose-no-rig-paths.sh — no compose may use a MAINTAINER-RIG absolute path
# as an environment DEFAULT. The repo ships to strangers; a default is what runs
# when they set nothing, so a default of `/mnt/models/huggingface` silently points
# their container at a directory only the reference rig has.
#
# Why a guard and not a convention: the convention already existed and was followed
# by most composes (the repo-relative `${MODEL_DIR:-../../../../../../models-cache}`
# is the documented shape), yet three shipped composes had drifted onto rig paths
# and a fourth was added on 2026-08-11 before this guard existed. Conventions that
# only live in reviewers' heads decay; this is the same reasoning as
# test-compose-bind-host.sh.
#
# What is CHECKED — three surfaces:
#   (a) `${VAR:-<rig-path>}` environment defaults.
#   (b) bare rig paths on the host side of a volume/command list entry.
#   (c) rig paths ANYWHERE a container actually runs or PRINTS — i.e. inside an
#       `entrypoint:` or `command:` block, comments excluded.
#
# Surface (c) was added 2026-09-12 for #1271. 18 GLM composes printed
#     echo "[spec]   - download it: /opt/ai/hf-download.sh …"
# as the remediation for a missing DFlash2 drafter — a hardened wrapper that lives
# only on the maintainer's rig and is NOT in this repo. Every user but one was
# handed a script they cannot run, in the one place they are guaranteed to be
# reading: having just been refused a boot. The gate was GREEN throughout, because
# a rig path inside an entrypoint `echo` is neither an env default (a) nor a bare
# host-side list entry (b). The blind spot was the checked surface, not the intent
# — the header above already claimed this class. So: remediation text a user is
# told to TYPE is a value the compose USES, and is checked.
#
# What is NOT checked: comments and prose. Documenting the reference rig
# ("measured on /mnt/models/...", "Ledger: /opt/ai/moecache-releases.md") is
# legitimate and stays legal, inside an entrypoint block as well as outside it —
# the failure mode this guards is a value the compose would actually USE or SHOW.
#
# CONTAINER-side absolutes are legal and must never trip this: `/models/...`,
# `/etc/club3090/...`, `/app/llama-server`, `/dev/shm` are all correct inside the
# container. That allowance is structural, not a deny-list exception — RIG_PREFIXES
# only matches absolutes a stranger cannot have, and none of those match it. The
# prefix set is deliberately NOT widened: an over-wide docs gate once produced
# ~200 false positives, so breadth here is bought only with evidence.
#
# If a compose genuinely needs an operator-supplied absolute path, give it a knob
# with a portable default (repo-relative, or another variable), never a rig path.
# If it needs to tell a user how to fetch something, point at an IN-REPO command
# (`bash scripts/setup.sh <model>`), never at a script on the maintainer's rig.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Absolute prefixes that exist only on the maintainer's rig. Keep this list short
# and specific: it is about paths a stranger cannot have, not about all absolutes
# (/dev, /tmp, /usr and friends are fine and common).
RIG_PREFIXES='/opt/ai|/mnt/models|/home/[a-z][a-z0-9_-]*'

# scan_env_and_host <file> — surfaces (a) and (b). Comments stripped first so
# prose about the reference rig stays legal.
scan_env_and_host() {
  sed 's/#.*$//' "$1" \
    | grep -nE "\\\$\{[A-Za-z_][A-Za-z0-9_]*:-(${RIG_PREFIXES})|^[[:space:]]*-[[:space:]]*\"?(${RIG_PREFIXES})/" \
    || true
}

# scan_run_blocks <file> — surface (c). Walks `entrypoint:`/`command:` blocks by
# indentation (a block ends at the next line indented no deeper than its key, with
# blank lines belonging to the block, since these are YAML block scalars holding a
# shell script). Full-line comments inside the block are prose and skipped.
# Emits `<lineno>:<text>` so the caller can report like grep -n.
scan_run_blocks() {
  awk -v rig="(${RIG_PREFIXES})" '
    function key_line() { return $0 ~ /^[[:space:]]*(entrypoint|command)[[:space:]]*:/ }
    key_line() {
      match($0, /^[[:space:]]*/); keyindent = RLENGTH; inblock = 1
      # A value on the key line itself (`command: /opt/ai/x`) is still a value.
      rest = $0; sub(/^[[:space:]]*(entrypoint|command)[[:space:]]*:/, "", rest)
      if (rest ~ /[^[:space:]]/ && rest ~ rig) printf "%d:%s\n", FNR, $0
      next
    }
    !inblock { next }
    /^[[:space:]]*$/ { next }                 # blank lines stay inside the block
    { match($0, /^[[:space:]]*/); if (RLENGTH <= keyindent) { inblock = 0; next } }
    /^[[:space:]]*#/ { next }                 # prose about the reference rig: legal
    $0 ~ rig { printf "%d:%s\n", FNR, $0 }
  ' "$1"
}

# ---------------------------------------------------------------------------
# SELF-TEST. A scanner that matches nothing is indistinguishable from a clean
# tree, which is exactly how surface (c) stayed green over 18 broken composes.
# So prove both directions on synthetic fixtures before trusting the real run:
# the must-fail fixture must be caught, and the must-pass fixture (container
# absolutes + rig paths in comments) must not be. If either flips, this gate
# fails LOUDLY rather than reporting a clean tree.
# ---------------------------------------------------------------------------
selftest_dir="$(mktemp -d)"
trap 'rm -rf "$selftest_dir"' EXIT

cat > "${selftest_dir}/must-fail.yml" <<'FIXTURE'
services:
  llm:
    entrypoint:
      - /bin/bash
      - -c
      - |
        # A rig path in a comment here is legal — /opt/ai/moecache-releases.md.
        if [ ! -f "$$_draft" ]; then
          echo "[spec] download it: /opt/ai/hf-download.sh some/repo" >&2
          exit 1
        fi
        exec /app/llama-server "$$@"
FIXTURE

cat > "${selftest_dir}/must-pass.yml" <<'FIXTURE'
services:
  llm:
    entrypoint:
      - /bin/bash
      - -c
      - |
        # Ledger: /opt/ai/moecache-releases.md — prose, stays legal.
        _mmproj="$${MMPROJ:-/models/glm-5.3-flash-gguf/mmproj/mmproj-F16.gguf}"
        [ -f /etc/club3090/engine.env ] && . /etc/club3090/engine.env
        echo "[spec] fetch it: bash scripts/setup.sh glm-5.3-flash" >&2
        exec /app/llama-server "$$@"
    command:
      - '-m'
      - '/models/glm-5.3-flash-gguf/Q2_K/model-00001-of-00009.gguf'
FIXTURE

st_fail_hits="$(scan_run_blocks "${selftest_dir}/must-fail.yml" | wc -l | tr -d ' ')"
st_pass_hits="$(scan_run_blocks "${selftest_dir}/must-pass.yml" | wc -l | tr -d ' ')"

if [ "$st_fail_hits" -eq 0 ]; then
  echo "FAIL: self-test — the entrypoint scanner did not catch a rig path in an" >&2
  echo "      entrypoint echo. The scanner is broken or inert; a clean result from" >&2
  echo "      this gate would be meaningless. Fix scan_run_blocks." >&2
  exit 1
fi
if [ "$st_pass_hits" -ne 0 ]; then
  echo "FAIL: self-test — the entrypoint scanner flagged legitimate container-side" >&2
  echo "      paths or commented prose (${st_pass_hits} false positive(s)):" >&2
  scan_run_blocks "${selftest_dir}/must-pass.yml" | sed 's/^/        /' >&2
  echo "      Container absolutes (/models, /etc/club3090, /app) must stay legal." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Real run.
# ---------------------------------------------------------------------------
fails=0
checked=0

while IFS= read -r f; do
  checked=$((checked+1))

  # (a) + (b): a rig path used as an env default, or bare on the host side of a
  # volume entry.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    echo "FAIL: $f uses a maintainer-rig path as a value:" >&2
    echo "        $(printf '%s' "$line" | sed 's/^[[:space:]]*//')" >&2
    echo "        Use a portable default (repo-relative, or another variable)." >&2
    fails=$((fails+1))
  done < <(scan_env_and_host "$f")

  # (c): a rig path inside what the container runs or prints.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    echo "FAIL: $f names a maintainer-rig path in its entrypoint/command:" >&2
    echo "        $(printf '%s' "$line" | sed 's/^[0-9]*:[[:space:]]*//')" >&2
    echo "        Nothing under /opt/ai, /mnt/models or a home dir exists for a user" >&2
    echo "        who cloned this repo. Point at an in-repo command (e.g." >&2
    echo "        'bash scripts/setup.sh <model>') or a container-side path." >&2
    fails=$((fails+1))
  done < <(scan_run_blocks "$f")
done < <(find models -name '*.yml' -path '*/compose/*' | sort)

if [ "$fails" -gt 0 ]; then
  echo "$fails rig-path check(s) failed across $checked composes" >&2
  exit 1
fi
echo "test-compose-no-rig-paths: ok ($checked composes, no rig paths used as values;"
echo "                            env defaults, host-side entries and entrypoint/command text)"
