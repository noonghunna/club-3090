#!/usr/bin/env bash
# Guard for scripts/strata-pin-bump.sh: the one-command bump for the strata and strata-maya engines.
#
# Every case runs on a COPY of the working tree, with stubs for `docker` and `gh` on PATH and for
# engine_config_reads.py, so nothing is pulled, run or booted (the boot cases refuse before it).
# What it holds:
#   - a full --no-boot bump edits exactly the files the pin lives in, with the right values, and
#     records the settings evidence; docs/UPSTREAM.md, which carries no pin, stays byte-identical;
#   - --check writes nothing;
#   - a no-op, a mislabelled image, a changed interface in the image (each check) and a checkout
#     on master are refused or failed with nothing written;
#   - every edit target that doesn't match exactly once refuses BEFORE anything is written, so a
#     changed format can't leave a stale pin line under a successful run;
#   - Strata's boot needs an explicit --family or --slug (the bump rule).
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

fail() { echo "ASSERTION FAILED: $1" >&2; exit 1; }
pass=0
ok() { pass=$((pass + 1)); }

bash -n "$ROOT_DIR/scripts/strata-pin-bump.sh" || fail "bash -n"
python3 -m py_compile "$ROOT_DIR/scripts/lib/strata_pin_bump.py" || fail "py_compile"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
PRISTINE="$TMP/pristine"
mkdir -p "$PRISTINE"
# tracked + new (not ignored) files: the working tree as it stands, uncommitted edits included
(cd "$ROOT_DIR" && git ls-files -co --exclude-standard -z | tar --null -T - -cf -) | tar -xf - -C "$PRISTINE"
cat > "$PRISTINE/scripts/lib/profiles/engine_config_reads.py" <<'PY'
import os, pathlib, sys
pathlib.Path(os.environ["STUB_DIR"], "recorded").write_text(" ".join(sys.argv[1:]))
print("  ✓ stub: recorded")
PY

mkdir -p "$TMP/bin"
cat > "$TMP/bin/docker" <<'SH'
#!/usr/bin/env bash
# stub docker: answers from $STUB_DIR; anything unexpected fails loudly
case "$1" in
  pull)  echo "Digest: $(cat "$STUB_DIR/digest")"; echo "Status: Downloaded newer image for $2" ;;
  image) cat "$STUB_DIR/inspect.json" ;;
  run)   case "$*" in
           *"strata launch --help"*) cat "$STUB_DIR/launch-help.txt" ;;
           *"entrypoint.sh"*)        cat "$STUB_DIR/entrypoint.sh" ;;
           *"--help 2>&1"*)          cat "$STUB_DIR/engine-help.txt" ;;
           *"maya.MODELS"*)          cat "$STUB_DIR/maya-models.json" ;;
           *) echo "stub docker: unexpected run $*" >&2; exit 97 ;;
         esac ;;
  *)     echo "stub docker: unexpected $*" >&2; exit 97 ;;
esac
SH
cat > "$TMP/bin/gh" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *"/commits/"*)  cat "$STUB_DIR/commit" ;;
  *"/releases"*)  cat "$STUB_DIR/releases.json" ;;
  *) echo "stub gh: unexpected $*" >&2; exit 97 ;;
esac
SH
chmod +x "$TMP/bin/docker" "$TMP/bin/gh"

COMMIT=1234567890abcdef1234567890abcdef12345678
DIGEST=sha256:$(printf 'a%.0s' {1..64})
TODAY="$(date +%F)"

# stubs <engine> <version>: a consistent fake release in $STUB_DIR
stubs() {
  local engine="$1" ver="$2" repo
  STUB_DIR="$TMP/stub-$RANDOM$RANDOM"; mkdir -p "$STUB_DIR"; export STUB_DIR
  echo "$DIGEST" > "$STUB_DIR/digest"
  echo "$COMMIT" > "$STUB_DIR/commit"
  if [[ "$engine" == strata ]]; then repo=ghcr.io/niko1221/strata; else repo=ghcr.io/noonghunna/strata-maya-club3090; fi
  cat > "$STUB_DIR/inspect.json" <<JSON
{"RepoDigests": ["$repo@$DIGEST"], "Config": {"Labels": {
  "org.opencontainers.image.version": "$ver", "com.strata.engine": "9.9.8-linux-x86_64-cuda-cuda13",
  "club3090.upstream.revision": "$COMMIT", "org.opencontainers.image.revision": "fedcba9876543210fedcba9876543210fedcba98",
  "club3090.cuda_architectures": "75;80;86;89;120"}}}
JSON
  cat > "$STUB_DIR/launch-help.txt" <<'TXT'
usage: strata launch [-h] [--context SIZE] [--kv FORMAT] [--vision off|on|gpu|cpu]
  --context SIZE        context window: 32k, 64k, 128k, 256k or a number of tokens
  --kv FORMAT           conversation memory format: int8 (default), q4_0 (half
                        the memory), k8v4, fp16
  --vision off|on|gpu|cpu
                        let the model read pictures
TXT
  cat > "$STUB_DIR/engine-help.txt" <<'TXT'
  --kv fp16|int8       KV storage
  --spec N             the MTP drafter's verify window: how many tokens it proposes per check (setup
                       writes 4; --suffix-draft lets it grow by 2, up to 8, where a repeat is likely)
  --suffix-draft       grow the window where a repeat is likely
TXT
  printf '%s\n' 'm="${MODEL:-}"' 'f="$FAMILY"' 'p="${MODEL_PATH:-}"' 'c="${CONTEXT:-}"' 'k="${KV:-}"' \
    'v="${VISION:-}"' 'g="${GPUS:-}"' 'r="${PRESET:-}"' 'a="${API_KEY:-}"' > "$STUB_DIR/entrypoint.sh"
  echo '["EXL3-3.05bpw", "Maya-L", "Maya-M", "Maya-S-v2-IQ2_XXS", "Maya-S24"]' > "$STUB_DIR/maya-models.json"
  local rtag="v${ver#v}"
  cat > "$STUB_DIR/releases.json" <<JSON
[{"tag_name": "$rtag", "published_at": "2099-01-03T00:00:00Z"},
 {"tag_name": "v9.9.7", "published_at": "2099-01-02T00:00:00Z"},
 {"tag_name": "v-draft", "published_at": "2099-01-02T12:00:00Z", "draft": true},
 {"tag_name": "OLDTAG", "published_at": "2099-01-01T00:00:00Z"}]
JSON
}

# fresh <branch>: a copy of the pristine tree, as a git checkout on <branch>; echoes its path
fresh() {
  local d="$TMP/case-$RANDOM$RANDOM"
  cp -a "$PRISTINE" "$d"
  git -C "$d" init -q -b "${1:-pin-test}"
  echo "$d"
}
bump() {  # bump <root> <args...>: runs the script against <root>; output in $OUT, exit in $RC
  local root="$1"; shift
  set +e
  OUT="$(PATH="$TMP/bin:$PATH" STRATA_PIN_BUMP_ROOT="$root" PIN_BUMP_ROOT="$root" \
         bash "$root/scripts/strata-pin-bump.sh" "$@" 2>&1)"
  RC=$?
  set -e
}
changed() { diff -rq "$PRISTINE" "$1" -x .git -x __pycache__ -x .mutant -x .orig | sed -E "s#^Files $PRISTINE/(.*) and .* differ\$#\\1#" | sort; }
old_field() {  # old_field <engine> <n>: field n of the pin-current line in the pristine tree
  sed -n 's/^  # pin-current: //p' "$PRISTINE/scripts/lib/profiles/engines/$1.yml" | awk -F ' · ' -v n="$2" '{print $n}'
}

# ── 1. Strata, full bump without the boot ─────────────────────────────────────────────────────────
OLD_VER="$(old_field strata 1)"; OLD_ENG="$(old_field strata 3 | cut -d' ' -f2)"; OLD_DATE="$(old_field strata 4 | cut -d' ' -f2)"
OLD_SHORT="$(sed -n 's/^  spec: .*@sha256:\(........\).*/\1/p' "$PRISTINE/scripts/lib/profiles/engines/strata.yml")"
stubs strata 9.9.9
sed -i "s/OLDTAG/v$OLD_VER/" "$STUB_DIR/releases.json"
d="$(fresh)"; bump "$d" strata 9.9.9 --no-boot
[[ $RC -eq 0 ]] || fail "strata --no-boot should exit 0 (got $RC): $OUT"
y="$d/scripts/lib/profiles/engines/strata.yml"
command grep -qx "  spec: ghcr.io/niko1221/strata@$DIGEST" "$y" || fail "strata: spec not repointed"
command grep -qx "  source_ref: \"$COMMIT\"" "$y" || fail "strata: source_ref not updated"
command grep -qx "  # pin-current: 9.9.9 · upstream 12345678 · engine 9.9.8 · pinned $TODAY · booted pending" "$y" \
  || fail "strata: pin-current line wrong: $(command grep pin-current "$y")"
command grep -q "^  # pin-earlier: $OLD_VER \`$OLD_SHORT…\` (upstream " "$y" || fail "strata: outgoing pin not first in pin-earlier"
[[ "$(command grep -c "strata@$DIGEST" "$d"/models/qwen3.8-flash-next/strata/compose/*/*/base.yml | awk -F: '{s+=$2} END{print s}')" == 7 ]] \
  || fail "strata: not all 7 compose image defaults repointed"
cmp -s "$PRISTINE/docs/UPSTREAM.md" "$d/docs/UPSTREAM.md" || fail "strata: the bump edited docs/UPSTREAM.md (the pin lives in the profile)"
command grep -qxF -- "- **Pinned:** 9.9.9 (engine 9.9.8), since $TODAY ($OLD_VER, engine $OLD_ENG, from $OLD_DATE)." "$d/docs/engines/STRATA.md" \
  || fail "strata: STRATA.md Pinned line wrong"
[[ "$(cat "$STUB_DIR/recorded")" == "verify strata --record" ]] || fail "strata: settings evidence not re-recorded"
command grep -qxF -- "- **Releases skipped since the last pin:** v9.9.7" <<<"$OUT" || fail "strata: skipped releases wrong (drafts and the two pins don't count): $OUT"
want="$(printf '%s\n' docs/engines/STRATA.md scripts/lib/profiles/engines/strata.yml \
  $(cd "$d" && ls models/qwen3.8-flash-next/strata/compose/*/*/base.yml) | sort)"
[[ "$(changed "$d")" == "$want" ]] || fail "strata: changed files differ from the pin's places:\n$(changed "$d")"
ok

# ── 2. Maya, full bump without the boot ───────────────────────────────────────────────────────────
M_OLD="$(old_field strata-maya 1)"
stubs strata-maya v9.9.9
sed -i "s/OLDTAG/$M_OLD/" "$STUB_DIR/releases.json"
d="$(fresh)"; bump "$d" strata-maya 9.9.9 --no-boot
[[ $RC -eq 0 ]] || fail "maya --no-boot should exit 0 (got $RC): $OUT"
y="$d/scripts/lib/profiles/engines/strata-maya.yml"
command grep -qx "  spec: ghcr.io/noonghunna/strata-maya-club3090@$DIGEST" "$y" || fail "maya: spec not repointed"
command grep -qx "  # pin-current: v9.9.9 · upstream 12345678 · recipe fedcba98 · archs 75;80;86;89;120 · pinned $TODAY · booted pending" "$y" \
  || fail "maya: pin-current line wrong: $(command grep pin-current "$y")"
command grep -q "^  # pin-earlier: $M_OLD \`" "$y" || fail "maya: outgoing pin not first in pin-earlier"
cmp -s "$PRISTINE/docs/UPSTREAM.md" "$d/docs/UPSTREAM.md" || fail "maya: the bump edited docs/UPSTREAM.md (the pin lives in the profile)"
# every compose whose image default is the engine's (GLM's 9 and DeepSeek's), each repointed
m_composes="$(cd "$d" && command grep -l 'STRATA_MAYA_IMAGE:-' models/*/strata-maya/compose/*/*/*.yml)"
[[ "$(wc -l <<<"$m_composes")" -ge 10 ]] || fail "maya: expected the GLM and DeepSeek composes, found: $m_composes"
for c in $m_composes; do
  command grep -q "STRATA_MAYA_IMAGE:-ghcr.io/noonghunna/strata-maya-club3090@$DIGEST}" "$d/$c" || fail "maya: $c not repointed"
done
want="$(printf '%s\n' scripts/lib/profiles/engines/strata-maya.yml $m_composes | sort)"
[[ "$(changed "$d")" == "$want" ]] || fail "maya: changed files differ from the pin's places:\n$(changed "$d")"
ok

# ── 3. --check writes nothing ─────────────────────────────────────────────────────────────────────
stubs strata 9.9.9
d="$(fresh master)"; bump "$d" strata 9.9.9 --check
[[ $RC -eq 0 ]] || fail "--check should exit 0 (got $RC): $OUT"
[[ "$OUT" == *"+  # pin-current: 9.9.9 · upstream 12345678"* ]] || fail "--check should show the planned pin-current line"
[[ -z "$(changed "$d")" && ! -e "$STUB_DIR/recorded" ]] || fail "--check wrote: $(changed "$d")"
ok

# ── 4. refusals and failures write nothing ────────────────────────────────────────────────────────
expect() {  # expect <rc> <what> <root> <args...>
  local want="$1" what="$2" root="$3"; shift 3
  bump "$root" "$@"
  [[ $RC -eq $want ]] || fail "$what: expected exit $want, got $RC: $OUT"
  [[ -z "$(changed "$root")" ]] || fail "$what: wrote files although it stopped: $(changed "$root")"
  ok
}
stubs strata "$OLD_VER"
expect 2 "no-op (already pinned)" "$(fresh)" strata "$OLD_VER" --no-boot
stubs strata 9.9.9; sed -i 's/"9.9.9"/"9.9.6"/' "$STUB_DIR/inspect.json"
expect 2 "image labelled with another version" "$(fresh)" strata 9.9.9 --no-boot
stubs strata 9.9.9; echo "sha256:$(printf 'b%.0s' {1..64})" > "$STUB_DIR/digest"
expect 2 "pulled digest not among the image's repo digests" "$(fresh)" strata 9.9.9 --no-boot
stubs strata 9.9.9; sed -i 's/k8v4, //' "$STUB_DIR/launch-help.txt"
expect 1 "strata launch --kv lost a format" "$(fresh)" strata 9.9.9 --no-boot
stubs strata 9.9.9; sed -i 's/writes 4/writes 3/' "$STUB_DIR/engine-help.txt"
expect 1 "--spec help changed" "$(fresh)" strata 9.9.9 --no-boot
stubs strata 9.9.9; sed -i '/GPUS/d' "$STUB_DIR/entrypoint.sh"
expect 1 "entrypoint stopped reading GPUS" "$(fresh)" strata 9.9.9 --no-boot
stubs strata-maya v9.9.9; echo '["Maya-M", "Maya-S-v2-IQ2_XXS"]' > "$STUB_DIR/maya-models.json"
expect 1 "maya.py lost Maya-L" "$(fresh)" strata-maya v9.9.9 --no-boot
stubs strata 9.9.9
expect 2 "checkout on master" "$(fresh master)" strata 9.9.9 --no-boot

# ── 5. every edit target refuses unless it matches exactly once ───────────────────────────────────
mutate() {  # mutate <what> <file> <sed program>
  local d; d="$(fresh)"
  sed -i "$3" "$d/$2"
  cp "$d/$2" "$PRISTINE/.mutant"; cp "$PRISTINE/$2" "$PRISTINE/.orig"
  cp "$PRISTINE/.mutant" "$PRISTINE/$2"            # the mutant is the baseline: nothing else may change
  stubs strata 9.9.9
  bump "$d" strata 9.9.9 --no-boot
  local diffs; diffs="$(changed "$d")"
  cp "$PRISTINE/.orig" "$PRISTINE/$2"; rm -f "$PRISTINE/.mutant" "$PRISTINE/.orig"
  [[ $RC -eq 2 ]] || fail "mutant '$1': expected a refusal (2), got $RC: $OUT"
  [[ -z "$diffs" ]] || fail "mutant '$1': wrote files before refusing: $diffs"
  [[ "$OUT" == *"$2"* ]] || fail "mutant '$1': the refusal should name $2: $OUT"
  ok
}
Y=scripts/lib/profiles/engines/strata.yml
mutate "no pin-current line"      "$Y" '/^  # pin-current: /d'
mutate "pin-current lacks engine" "$Y" 's/ · engine [^ ]*//'
mutate "two pin-earlier lines"    "$Y" 's/^\(  # pin-earlier: .*\)$/\1\n\1/'
mutate "no source_ref"            "$Y" 's/^  source_ref: /  source_url: /'
mutate "no Pinned line"           docs/engines/STRATA.md '/^- \*\*Pinned:\*\* /d'

# ── 6. Strata's boot needs an explicit slug ───────────────────────────────────────────────────────
stubs strata 9.9.9
d="$(fresh)"; bump "$d" strata 9.9.9
[[ $RC -eq 2 && "$OUT" == *"--family ista | swift"* && "$OUT" == *"last bump didn't boot"* ]] \
  || fail "strata without --family/--slug should refuse with the hint (got $RC): $OUT"
[[ ! -e "$STUB_DIR/recorded" && -z "$(changed "$d")" ]] || fail "the slug refusal must come before any edit"
ok
expect 2 "unknown family" "$(fresh)" strata 9.9.9 --family nosuch
expect 2 "slug of another engine" "$(fresh)" strata 9.9.9 --slug strata-maya/glm53-flash-dual-maya-l
expect 2 "--family on maya" "$(fresh)" strata-maya v9.9.9 --family ista

echo "test-strata-pin-bump: ok ($pass checks)"
