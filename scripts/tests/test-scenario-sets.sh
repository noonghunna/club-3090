#!/usr/bin/env bash
# Guard for scripts/scenario-sets/*.txt: every non-comment line must be a
# pack-qualified scenario ID (KNOWN_PACK/ID), IDs unique per file, and each
# file must carry a provenance header (a probe set without provenance decays
# into a magic list). Also: quality-test.sh must actually expose the
# passthrough flags these files are consumed through.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SETS_DIR="${ROOT_DIR}/scripts/scenario-sets"
QT="${ROOT_DIR}/scripts/quality-test.sh"

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }

KNOWN_PACKS="toolcall-15 instructfollow-15 structoutput-15 dataextract-15 reasonmath-15 bugfind-15 cli-40 hermesagent-20 aider-polyglot-30 humaneval-plus-30 lcb-v6-30"

ls "$SETS_DIR"/*.txt >/dev/null 2>&1 || fail "no scenario-set files in $SETS_DIR"

for f in "$SETS_DIR"/*.txt; do
  base="$(basename "$f")"
  # provenance header: first line must be a comment mentioning where the set came from
  head -1 "$f" | command grep -q '^#' || fail "$base: first line must be a provenance comment"
  command grep -qiE '^#.*(provenance|#66[0-9]|#6[0-9][0-9])' "$f" || fail "$base: header must state provenance"

  seen=""
  while IFS= read -r line; do
    # strip comments/blank
    case "$line" in ''|\#*|' '*'#'*) [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue ;; esac
    [[ "$line" =~ ^[a-z0-9-]+/[A-Z]+-[0-9]+$ ]] || fail "$base: malformed selection line: '$line'"
    pack="${line%%/*}"
    command grep -qw "$pack" <<<"$KNOWN_PACKS" || fail "$base: unknown pack '$pack' in '$line'"
    case " $seen " in *" $line "*) fail "$base: duplicate selection '$line'";; esac
    seen="$seen $line"
  done < "$f"
done

# the wrapper must expose the flags the sets are consumed through
for flag in --scenario --scenarios-file --incremental --resume --allow-partial; do
  command grep -q -- "$flag)" "$QT" || fail "quality-test.sh missing passthrough case for $flag"
done

echo "test-scenario-sets: PASS"
