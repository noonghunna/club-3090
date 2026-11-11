#!/usr/bin/env bash
# test-weights-remove — scripts/weights.sh show / remove (#1814) against a FIXTURE
# catalog and a FIXTURE model dir. Real deletion happens only under $T.
#
# Every refusal is shown to fire (nothing deleted, exit code) AND shown to stand
# aside once its one blocking condition is gone (the same removal then deletes, and
# the files beside it survive). A refusal with no positive control could be passing
# because the tool deletes nothing at all.
#
# Cases: a shared folder (deprecated slugs count as users), a drafter shared across
# slugs, a folder with other quants beside the slug's files, a symlinked folder
# (files go at the target; the link only once nothing is left behind it), a running
# container (stub `docker` on PATH — direct mount, parent mount by a user / by an
# unrelated slug / by an unknown container, docker failing), paths that escape the
# model dir (a symlink below the folder, `..` in a recipe, a recipe file that is a
# symlink, a top-level link to a parent of MODEL_DIR), a recipe with no `files:`,
# a stale dry-run digest, no confirmation, a per-slug saved GGUF_FILE override.
# Plus ONE read-only assertion on the real catalog: the shared DFlash2 drafter.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
FAIL=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# ── fixture repo: scripts COPIED, not symlinked ─────────────────────────────────
# weights.py and compose_registry.py find their data through Path(__file__).resolve(),
# so a symlinked scripts/lib would quietly read the REAL catalog.
R="$T/repo"
mkdir -p "$R/scripts"
cp -r "$ROOT/scripts/lib" "$R/scripts/lib"
cp "$ROOT/scripts/preflight.sh" "$ROOT/scripts/weights.sh" "$R/scripts/"
rm -rf "$R/scripts/lib/profiles-local"
find "$R/scripts/lib" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -f "$R/scripts/lib/profiles/models/"*.yml

cat > "$R/scripts/lib/profiles/models/fx.yml" <<'YAML'
id: fx
display_name: Fixture
weights:
  q4:
    hf_repo: fixture/fx-GGUF
    local_subdir: fx-gguf
    path: fx-gguf/Q4
    files: [Q4/fx-Q4-00001-of-00002.gguf, Q4/fx-Q4-00002-of-00002.gguf]
    verify_glob: Q4/*.gguf
  q8:
    hf_repo: fixture/fx-GGUF
    local_subdir: fx-gguf
    path: fx-gguf/Q8
    files: [Q8/fx-Q8.gguf]
    verify_glob: Q8/*.gguf
  drafter:
    hf_repo: fixture/fx-drafter
    kind: draft
    local_subdir: fx-drafter-gguf
    path: fx-drafter-gguf
    files: [fx-drafter.gguf]
    verify_glob: fx-drafter.gguf
  other:
    hf_repo: fixture/fx-other
    local_subdir: fx-other-gguf
    path: fx-other-gguf
    files: [fx-other.gguf]
    verify_glob: fx-other.gguf
  link-a:
    hf_repo: fixture/fx-linked
    local_subdir: fx-linked-gguf
    path: fx-linked-gguf/A
    files: [A/fx-a.gguf]
    verify_glob: A/*.gguf
  link-b:
    hf_repo: fixture/fx-linked
    local_subdir: fx-linked-gguf
    path: fx-linked-gguf/B
    files: [B/fx-b.gguf]
    verify_glob: B/*.gguf
  awq:
    hf_repo: fixture/fx-awq
    local_subdir: fx-awq
    path: fx-awq
  nested:
    hf_repo: fixture/fx-nested
    local_subdir: fx-nested-gguf
    path: fx-nested-gguf/sub
    files: [sub/fx-nested.gguf]
    verify_glob: sub/*.gguf
  dotdot:
    hf_repo: fixture/fx-dotdot
    local_subdir: fx-dd
    path: fx-dd
    files: [../outside-dd/victim.gguf]
    verify_glob: "*.gguf"
  linkfile:
    hf_repo: fixture/fx-linkfile
    local_subdir: fx-lf
    path: fx-lf
    files: [fx-lf.gguf]
    verify_glob: "*.gguf"
  up:
    hf_repo: fixture/fx-up
    local_subdir: fx-up
    path: fx-up/outside-up
    files: [outside-up/victim.gguf]
    verify_glob: outside-up/*.gguf
YAML

entry() {   # <slug> <variant> <compose> [status]
  cat <<YAML
  $1:
    model: fx
    weights_variant: $2
    workload: fast-chat
    engine: llama-cpp-local
    drafter: null
    kv_format: q8_0
    tp: 1
    max_ctx: 32768
    max_num_seqs: 1
    mem_util: null
    compose_path: $3
    default_port: 8020
    status: ${4:-production}
YAML
}
llama_compose() {   # <path under models/> <model file> [drafter file]
  mkdir -p "$R/models/fx/llama-cpp/compose/single/$(dirname "$1")"
  {
    echo "services:"
    echo "  fx:"
    echo "    image: ghcr.io/ggml-org/llama.cpp@sha256:0000000000000000000000000000000000000000000000000000000000000000"
    echo "    volumes:"
    echo "      - \${MODEL_DIR:-../../../../models-cache}:/models:ro"
    echo "    command: >"
    echo "      -m /models/\${GGUF_FILE:-$2}"
    [[ -n "${3:-}" ]] && echo "      --model-draft /models/$3"
    echo "      --port 8080"
  } > "$R/models/fx/llama-cpp/compose/single/$1"
  echo "models/fx/llama-cpp/compose/single/$1"
}
{
  echo "schema: 1"
  echo
  echo "entries:"
  entry llamacpp/fx-q4            q4       "$(llama_compose q4/base.yml fx-gguf/Q4/fx-Q4-00001-of-00002.gguf fx-drafter-gguf/fx-drafter.gguf)"
  entry llamacpp-club3090/fx-q4-old q4     "$(llama_compose q4/old.yml fx-gguf/Q4/fx-Q4-00001-of-00002.gguf)" deprecated
  entry llamacpp/fx-q8            q8       "$(llama_compose q8/base.yml fx-gguf/Q8/fx-Q8.gguf)"
  entry llamacpp/fx-other         other    "$(llama_compose other/base.yml fx-other-gguf/fx-other.gguf fx-drafter-gguf/fx-drafter.gguf)"
  entry llamacpp/fx-link-a        link-a   "$(llama_compose link-a/base.yml fx-linked-gguf/A/fx-a.gguf)"
  entry llamacpp/fx-link-b        link-b   "$(llama_compose link-b/base.yml fx-linked-gguf/B/fx-b.gguf)"
  entry llamacpp/fx-nested        nested   "$(llama_compose nested/base.yml fx-nested-gguf/sub/fx-nested.gguf)"
  entry llamacpp/fx-dotdot        dotdot   "$(llama_compose dotdot/base.yml fx-dd/x.gguf)"
  entry llamacpp/fx-linkfile      linkfile "$(llama_compose linkfile/base.yml fx-lf/fx-lf.gguf)"
  entry llamacpp/fx-up            up       "$(llama_compose up/base.yml fx-up/outside-up/victim.gguf)"
  mkdir -p "$R/models/fx/vllm/compose/single/awq"
  printf 'services:\n  fx:\n    image: vllm/vllm-openai:v0.33.0\n    command: >\n      --model /root/.cache/huggingface/fx-awq\n' \
    > "$R/models/fx/vllm/compose/single/awq/base.yml"
  entry vllm/fx-awq               awq      models/fx/vllm/compose/single/awq/base.yml
  echo
  echo "defaults: {}"
  echo
  echo "engine_preference: {}"
  echo
  echo "recommended_default_models: []"
} > "$R/scripts/lib/profiles/registry.yaml"

# ── fixture model dir + a second "disk" ──────────────────────────────────────────
MD="$T/models"
DISK2="$T/disk2"
OUT="$T/outside"
mk() { mkdir -p "$(dirname "$1")"; printf '%s' "${2:-weights}" > "$1"; }
build_model_dir() {
  rm -rf "$MD" "$DISK2" "$OUT"
  mkdir -p "$MD" "$DISK2" "$OUT"
  mk "$MD/fx-gguf/Q4/fx-Q4-00001-of-00002.gguf"; mk "$MD/fx-gguf/Q4/fx-Q4-00002-of-00002.gguf"
  mk "$MD/fx-gguf/Q8/fx-Q8.gguf"
  mk "$MD/fx-gguf/.cache/huggingface/download/Q8/fx-Q8.gguf.metadata" meta
  mk "$MD/fx-gguf/.cache/huggingface/download/Q4/fx-Q4-00001-of-00002.gguf.metadata" meta
  mk "$MD/fx-gguf/notes.txt" "a user's own file beside the quants"
  mk "$MD/fx-gguf/.setup-verified.tsv" "sha	1	1	main	Q8/fx-Q8.gguf"
  mk "$MD/fx-drafter-gguf/fx-drafter.gguf"
  # folder bookkeeping: goes only once nothing else is left in the recipe's folder
  mk "$MD/fx-drafter-gguf/.setup-verified.tsv" "sha	1	1	main	fx-drafter.gguf"
  mk "$MD/fx-drafter-gguf/.cache/huggingface/CACHEDIR.TAG" tag
  mk "$MD/fx-drafter-gguf/.cache/huggingface/.gitignore" "*"
  mk "$MD/fx-other-gguf/fx-other.gguf"
  # symlinked folder: MODEL_DIR/fx-linked-gguf → another disk
  mk "$DISK2/fx-linked-gguf/A/fx-a.gguf"; mk "$DISK2/fx-linked-gguf/B/fx-b.gguf"
  ln -s "$DISK2/fx-linked-gguf" "$MD/fx-linked-gguf"
  mk "$MD/fx-awq/config.json" '{}'; mk "$MD/fx-awq/model.safetensors"
  # escapes
  # a symlink BELOW the folder, pointing INSIDE MODEL_DIR — so the roots check passes
  # and only the no-nested-symlink rule can catch it
  mkdir -p "$MD/fx-nested-gguf"; mk "$MD/fx-elsewhere/sub/fx-nested.gguf" victim
  ln -s "$MD/fx-elsewhere/sub" "$MD/fx-nested-gguf/sub"
  mk "$MD/fx-dd/x.gguf"; mk "$MD/outside-dd/victim.gguf" victim  # `..` in the recipe
  mk "$OUT/lf-target.gguf" victim; mkdir -p "$MD/fx-lf"
  ln -s "$OUT/lf-target.gguf" "$MD/fx-lf/fx-lf.gguf"              # the recipe file IS a symlink
  mk "$T/outside-up/victim.gguf" victim
  ln -s "$T" "$MD/fx-up"                                          # a link to a PARENT of MODEL_DIR
}

# ── stub docker on PATH: logs every call; containers come from $DOCKER_JSON ──────
mkdir -p "$T/bin"
cat > "$T/bin/docker" <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$DOCKER_LOG"
[[ "${DOCKER_FAIL:-0}" == "1" ]] && { echo "Cannot connect to the Docker daemon" >&2; exit 1; }
case "$1" in
  ps) python3 -c 'import json,sys; [print(c["Id"]) for c in json.load(open(sys.argv[1]))]' "$DOCKER_JSON" ;;
  inspect) cat "$DOCKER_JSON" ;;
  *) exit 2 ;;
esac
EOF
chmod +x "$T/bin/docker"
export DOCKER_LOG="$T/docker.log" DOCKER_JSON="$T/containers.json"
containers() { printf '%s' "${1:-[]}" > "$DOCKER_JSON"; }
container() {   # <id> <mount source> [compose config_files label]
  printf '{"Id":"%s","Name":"/%s","Mounts":[{"Type":"bind","Source":"%s","Destination":"/models"}],"Config":{"Labels":{"com.docker.compose.project.config_files":"%s"}}}' \
    "$1" "$1" "$2" "${3:-}"
}

W() { MODEL_DIR="$MD" PATH="$T/bin:$PATH" bash "$R/scripts/weights.sh" "$@" </dev/null; }
J() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }
# The refusal codes a dry run reports — each hard refusal below is asserted by its
# CODE as well as rc 4, so a refusal for some other reason can't pass for this one.
codes() { W remove "$1" --dry-run --json | J '" ".join(sorted({r["code"] for r in d["refusals"]}))'; }
# An unattended removal, the only way it can run: dry run → its digest → --yes --digest.
RMY() {
  local slug="$1" dig; shift
  dig="$(W remove "$slug" --dry-run --json "$@" | J 'd["digest"]')"
  W remove "$slug" --yes --digest "$dig" "$@"
}

build_model_dir
containers

echo "== the fixture catalog is the one being read"
out="$(W show llamacpp/fx-q8 --json)"; rc=$?
check "a fixture-only slug resolves (rc=$rc)" '[[ $rc == 0 ]]'
W show vllm/minimal >/dev/null 2>&1; rc=$?
check "a real-catalog slug is unknown here (rc=$rc)" '[[ $rc == 1 ]]'

echo "== show: path, symlink resolved, state, size, other users"
out="$(W show llamacpp/fx-q4 --json)"
check "Q4 folder is complete" '[[ "$(J "[f[\"state\"] for f in d[\"folders\"] if f[\"path\"].endswith(\"fx-gguf/Q4\")][0]" <<<"$out")" == complete ]]'
check "Q4 shows its folder size and its own files' size" \
  '[[ "$(J "[(f[\"size_bytes\"] > 0, f[\"recipe_bytes\"] > 0) for f in d[\"folders\"] if f[\"path\"].endswith(\"fx-gguf/Q4\")][0]" <<<"$out")" == "(True, True)" ]]'
check "the deprecated slug is named as a user of Q4" \
  '[[ "$(J "[u[\"slug\"]+\":\"+u[\"status\"] for f in d[\"folders\"] if f[\"path\"].endswith(\"Q4\") for u in f[\"users\"]]" <<<"$out")" == *"llamacpp-club3090/fx-q4-old:deprecated"* ]]'
check "the Q8 slug (other quant, same bucket) is NOT a user of the Q4 folder" \
  '[[ "$(J "[u[\"slug\"] for f in d[\"folders\"] if f[\"path\"].endswith(\"Q4\") for u in f[\"users\"]]" <<<"$out")" != *fx-q8* ]]'
check "the drafter folder is shown, role draft, used by fx-other" \
  '[[ "$(J "[(f[\"role\"], [u[\"slug\"] for u in f[\"users\"]]) for f in d[\"folders\"] if \"drafter\" in f[\"path\"]]" <<<"$out")" == *"draft"*"llamacpp/fx-other"* ]]'
out="$(W show llamacpp/fx-link-a --json)"
check "a symlinked folder shows where its bytes are" \
  '[[ "$(J "d[\"folders\"][0][\"resolved\"]" <<<"$out")" == "$DISK2/fx-linked-gguf/A" && "$(J "d[\"folders\"][0][\"linked\"]" <<<"$out")" == True ]]'
check "a missing folder says missing" \
  '[[ "$(W show llamacpp/fx-q8 --json | J "d[\"folders\"][0][\"state\"]")" == complete ]] && rm -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && [[ "$(W show llamacpp/fx-q8 --json | J "d[\"folders\"][0][\"state\"]")" == missing ]]'
build_model_dir

echo "== folder with other quants beside the slug's files: only the recipe's files go"
: > "$DOCKER_LOG"
out="$(W remove llamacpp/fx-q8 --dry-run --json)"; rc=$?
check "dry run: rc 0, one weight file + its hf record, the Q8 dir would be rmdir'd" \
  '[[ $rc == 0 && "$(J "len(d[\"targets\"]), \"$MD/fx-gguf/Q8\" in d[\"dirs\"], \"$MD/fx-gguf\" in d[\"dirs\"]" <<<"$out")" == "(2, True, False)" ]]'
check "dry run deleted nothing" '[[ -f "$MD/fx-gguf/Q8/fx-Q8.gguf" ]]'
check "the running-container check ran (docker stub called)" '[[ -s "$DOCKER_LOG" ]] && command grep -q "^ps" "$DOCKER_LOG"'
RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "remove --yes: rc 0" '[[ $rc == 0 ]]'
check "Q8 file and its hf record are gone, the emptied Q8 dir too" \
  '[[ ! -e "$MD/fx-gguf/Q8/fx-Q8.gguf" && ! -e "$MD/fx-gguf/.cache/huggingface/download/Q8/fx-Q8.gguf.metadata" && ! -d "$MD/fx-gguf/Q8" ]]'
check "Q4 beside it, its hf record, a user's own file and the folder's verify marker all survive" \
  '[[ -f "$MD/fx-gguf/Q4/fx-Q4-00001-of-00002.gguf" && -f "$MD/fx-gguf/Q4/fx-Q4-00002-of-00002.gguf" && -f "$MD/fx-gguf/.cache/huggingface/download/Q4/fx-Q4-00001-of-00002.gguf.metadata" && -f "$MD/fx-gguf/notes.txt" && -f "$MD/fx-gguf/.setup-verified.tsv" ]]'

echo "== shared folder: refused while another slug needs the files (deprecated counts)"
build_model_dir
out="$(W remove llamacpp/fx-q4 --dry-run --json)"; rc=$?
check "dry run: rc 3 (shared, overridable)" '[[ $rc == 3 ]]'
check "names the deprecated slug for Q4" '[[ "$(J "[s[\"slug\"]+\":\"+s[\"status\"] for r in d[\"refusals\"] for s in r.get(\"slugs\",[])]" <<<"$out")" == *"fx-q4-old:deprecated"* ]]'
check "names fx-other for the shared drafter" '[[ "$(J "[s[\"slug\"] for r in d[\"refusals\"] if \"drafter\" in r[\"folder\"] for s in r[\"slugs\"]]" <<<"$out")" == *"llamacpp/fx-other"* ]]'
RMY llamacpp/fx-q4 >/dev/null; rc=$?
check "remove --yes without --allow-shared: rc 3, nothing deleted" \
  '[[ $rc == 3 && -f "$MD/fx-gguf/Q4/fx-Q4-00001-of-00002.gguf" && -f "$MD/fx-drafter-gguf/fx-drafter.gguf" ]]'
RMY llamacpp/fx-q4 --allow-shared >/dev/null; rc=$?
check "positive control — --allow-shared: rc 0, Q4 + drafter gone" \
  '[[ $rc == 0 && ! -e "$MD/fx-gguf/Q4/fx-Q4-00001-of-00002.gguf" && ! -e "$MD/fx-drafter-gguf/fx-drafter.gguf" ]]'
check "  the drafter's folder, emptied, goes whole — its verify marker and hf tags with it" '[[ ! -e "$MD/fx-drafter-gguf" ]]'
check "  …and fx-other's own weights, Q8 and the user's file survive" \
  '[[ -f "$MD/fx-other-gguf/fx-other.gguf" && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && -f "$MD/fx-gguf/notes.txt" ]]'
build_model_dir
W remove llamacpp-club3090/fx-q4-old --dry-run >/dev/null; rc=$?
check "the deprecated slug's own removal is refused too (fx-q4 needs Q4): rc 3" '[[ $rc == 3 ]]'

echo "== a per-slug saved GGUF_FILE makes that slug a user too"
mkdir -p "$T/cfg"
printf '{"version": 1, "slugs": {"llamacpp/fx-other": {"GGUF_FILE": "fx-gguf/Q8/fx-Q8.gguf"}}}\n' > "$T/cfg/slugs.json"
CLUB3090_CONFIG_DIR="$T/cfg" W remove llamacpp/fx-q8 --dry-run >/dev/null; rc=$?
check "Q8 is refused as shared once fx-other's saved override points at it (rc=$rc)" '[[ $rc == 3 ]]'
W remove llamacpp/fx-q8 --dry-run >/dev/null; rc=$?
check "  positive control — without the saved override it proceeds (rc=$rc)" '[[ $rc == 0 ]]'

echo "== symlinked folder: files go at the target, the link only once nothing is behind it"
build_model_dir
RMY llamacpp/fx-link-a >/dev/null; rc=$?
check "remove link-a: rc 0, A's file gone at the target" '[[ $rc == 0 && ! -e "$DISK2/fx-linked-gguf/A/fx-a.gguf" ]]'
check "  the link stays — B still lives behind it" '[[ -L "$MD/fx-linked-gguf" && -f "$DISK2/fx-linked-gguf/B/fx-b.gguf" ]]'
RMY llamacpp/fx-link-b >/dev/null; rc=$?
check "remove link-b: rc 0, the link is gone now nothing is behind it" '[[ $rc == 0 && ! -e "$MD/fx-linked-gguf" && ! -L "$MD/fx-linked-gguf" ]]'
check "  the target disk's directory itself is left in place" '[[ -d "$DISK2/fx-linked-gguf" ]]'

echo "== running container"
build_model_dir
containers "[$(container c-direct "$MD/fx-gguf/Q8")]"
RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "a container mounting the folder itself: rc 4, nothing deleted" '[[ $rc == 4 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && "$(codes llamacpp/fx-q8)" == container ]]'
containers "[$(container c-user "$MD" "$R/models/fx/llama-cpp/compose/single/q8/base.yml")]"
RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "a container of the slug itself mounting MODEL_DIR: rc 4" '[[ $rc == 4 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && "$(codes llamacpp/fx-q8)" == container ]]'
containers "[$(container c-unknown "$MD")]"
RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "an unidentifiable container mounting MODEL_DIR: rc 4" '[[ $rc == 4 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && "$(codes llamacpp/fx-q8)" == container ]]'
: > "$DOCKER_LOG"
DOCKER_FAIL=1 RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "docker failing: rc 4 (fail closed), and the stub WAS asked" '[[ $rc == 4 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && "$(DOCKER_FAIL=1 codes llamacpp/fx-q8)" == docker ]] && command grep -q "^ps" "$DOCKER_LOG"'
containers "[$(container c-other "$MD" "$R/models/fx/llama-cpp/compose/single/other/base.yml")]"
RMY llamacpp/fx-q8 >/dev/null; rc=$?
check "positive control — a container of an unrelated slug mounting MODEL_DIR: rc 0, deleted" '[[ $rc == 0 && ! -e "$MD/fx-gguf/Q8/fx-Q8.gguf" ]]'
containers

echo "== paths that escape the model directories"
build_model_dir
RMY llamacpp/fx-nested >/dev/null; rc=$?
check "a symlink below the folder: rc 4 escape, the file it points to survives" '[[ $rc == 4 && -f "$MD/fx-elsewhere/sub/fx-nested.gguf" && "$(codes llamacpp/fx-nested)" == escape ]]'
RMY llamacpp/fx-dotdot >/dev/null; rc=$?
check "\`..\` in a recipe file: rc 4 escape, the file it names survives" '[[ $rc == 4 && -f "$MD/outside-dd/victim.gguf" && "$(codes llamacpp/fx-dotdot)" == escape ]]'
RMY llamacpp/fx-linkfile >/dev/null; rc=$?
check "a recipe file that is a symlink: rc 4 symlink, link and target survive" '[[ $rc == 4 && -L "$MD/fx-lf/fx-lf.gguf" && -f "$OUT/lf-target.gguf" && "$(codes llamacpp/fx-linkfile)" == symlink ]]'
RMY llamacpp/fx-up >/dev/null; rc=$?
check "a top-level link to a parent of MODEL_DIR: rc 4 escape, the file survives" '[[ $rc == 4 && -f "$T/outside-up/victim.gguf" && "$(codes llamacpp/fx-up)" == escape ]]'
# positive control: the same nested recipe with a real directory instead of the link
rm "$MD/fx-nested-gguf/sub"; mk "$MD/fx-nested-gguf/sub/fx-nested.gguf"
RMY llamacpp/fx-nested >/dev/null; rc=$?
check "positive control — the nested recipe in a real dir: rc 0, deleted" '[[ $rc == 0 && ! -e "$MD/fx-nested-gguf/sub/fx-nested.gguf" && -f "$MD/fx-elsewhere/sub/fx-nested.gguf" ]]'

echo "== a recipe with no files: is refused — the folder is never deleted whole"
out="$(W remove vllm/fx-awq --dry-run --json)"; rc=$?
check "no-files: rc 4, code no-files" '[[ $rc == 4 && "$(J "[r[\"code\"] for r in d[\"refusals\"]]" <<<"$out")" == *no-files* ]]'
RMY vllm/fx-awq >/dev/null
check "  the folder and its files are untouched" '[[ -f "$MD/fx-awq/model.safetensors" && -f "$MD/fx-awq/config.json" ]]'

echo "== confirmation and the dry-run digest"
build_model_dir
W remove llamacpp/fx-q8 >/dev/null; rc=$?
check "no --yes on a non-terminal: rc 5, nothing deleted" '[[ $rc == 5 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" ]]'
W remove llamacpp/fx-q8 --yes >/dev/null; rc=$?
check "--yes without the dry run's --digest: rc 5, nothing deleted" '[[ $rc == 5 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" ]]'
dig="$(W remove llamacpp/fx-q8 --dry-run --json | J 'd["digest"]')"
mk "$MD/fx-gguf/.cache/huggingface/download/Q8/fx-Q8.gguf.lock" lock    # the plan changes
W remove llamacpp/fx-q8 --yes --digest "$dig" >/dev/null; rc=$?
check "a stale digest: rc 4, nothing deleted" '[[ $rc == 4 && -f "$MD/fx-gguf/Q8/fx-Q8.gguf" && "$(W remove llamacpp/fx-q8 --yes --digest "$dig" --json | J "[r[\"code\"] for r in d[\"refusals\"]]")" == *changed* ]]'
dig="$(W remove llamacpp/fx-q8 --dry-run --json | J 'd["digest"]')"
W remove llamacpp/fx-q8 --yes --digest "$dig" >/dev/null; rc=$?
check "positive control — the current digest: rc 0, deleted" '[[ $rc == 0 && ! -e "$MD/fx-gguf/Q8/fx-Q8.gguf" ]]'

echo "== real catalog, read-only: a drafter shared across two models"
mkdir -p "$T/empty-md"
out="$(MODEL_DIR="$T/empty-md" PATH="$T/bin:$PATH" bash "$ROOT/scripts/weights.sh" show vllm/qwen38-27b-dual-superfast --json </dev/null)"; rc=$?
users="$(J "[u[\"slug\"] for f in d[\"folders\"] if f[\"path\"].endswith(\"qwen3.8-27b-dflash2-w4a16\") for u in f[\"users\"]]" <<<"$out" 2>/dev/null)"
check "the DFlash2 drafter's other users include the ThinkingCap slugs (rc=$rc)" '[[ $rc == 0 && "$users" == *thinkingcap38-27b* ]]'
check "  and the empty model dir was only read (still empty)" '[[ -z "$(ls -A "$T/empty-md")" ]]'

if [[ $FAIL -ne 0 ]]; then
  echo "test-weights-remove: FAIL" >&2
  exit 1
fi
echo "test-weights-remove: ok"
