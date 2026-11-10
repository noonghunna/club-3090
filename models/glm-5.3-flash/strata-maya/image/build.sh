#!/usr/bin/env bash
# Build ghcr.io/noonghunna/strata-maya-club3090 from one Project Maya release.
#
# Project Maya (github.com/mw00/project-maya, MIT) is a fork of Strata for GLM-5.3-Flash. It
# publishes no image and no Linux engine (its setup compiles on the user's PC). From v1.1.53 its
# repository has a recipe for building an image yourself (docker/, docs/DOCKER.md; adapted from
# this one, mw00/project-maya#808), but still no published image, so we build each release with
# our own Dockerfile (next to this script) and publish the result (an image-provenance exception,
# docs/ADDING_ENGINES.md). Maya releases several times a day; the
# maintainer runs this on the reference rig for the release we want (GitHub CI built it until 2026-10-17):
#
#   bash models/glm-5.3-flash/strata-maya/image/build.sh                    # newest release: build + check, no push
#   bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --push
#   bash models/glm-5.3-flash/strata-maya/image/build.sh --tag <vX.Y.Z> --verify <image>   # checks only
#   --jobs N   compile with N jobs (default: half the CPU threads, as maya.py does)
#
# Steps; any failure stops the run:
#   1. resolve the release tag to its commit (git ls-remote, annotated tags peeled) and build
#      from that commit, so a tag moved later can't change what we built;
#   2. skip when <image>:<tag> is already published (--force rebuilds). Only "not found" means
#      unpublished: any other registry error stops the run rather than re-pushing the tag;
#   3. docker build with fixed build args and OCI labels (upstream commit, llama.cpp pin from
#      setup.py, archs, CPU baseline, this recipe's commit);
#   4. checks on the finished image, none needing a GPU:
#      - the CPU settings the engine was configured with: STRATA_PORTABLE and GGML_AVX2 on,
#        GGML_NATIVE and anything above AVX2 off;
#      - the CUDA archs the engine was configured with, and VERSION = the release;
#      - every library resolves except the host driver's (libcuda, libnvidia-*);
#      - serve/server.py and tools/iq_pack.py still offer every flag our launcher passes, and
#        the expert profile the configs name is there;
#      - DeepSeek's engine build/maya-ds4 is there, its libraries resolve, and it and
#        tools/ds4/pack_gguf.py offer every flag our DeepSeek launcher passes; its seed expert
#        profile (data/expert-profile-ds4.bin) is there;
#      - gguf-py imports and the pack builder indexes the release's synthetic GLM-5 model;
#      - every package in the release's setup.py PY_PACKAGES (less cmake / ninja) is installed in
#        the .venv at a version it accepts;
#   5. with --push: push :<tag> and :sha-<commit7>, then print the digest and the facts the pin PR
#      cites. A push needs a clean recipe folder at a commit on origin/master, checked before
#      anything else runs (scripts/lib/image-push-guard.sh), so the recipe labels name the code
#      that built the image. Rig pushes carry no signed provenance attestation (CI's did).
#
# A new image is NOT a new pin. The strata-maya engine profile moves to one only after a rig boot
# and verify-full (engine-pin-bump.sh), so a bad release never reaches a launch.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

REPO_URL="https://github.com/mw00/project-maya"
IMAGE="${STRATA_MAYA_IMAGE:-ghcr.io/noonghunna/strata-maya-club3090}"
# RTX 20/30/40/50 and the A-series, as the Strata image: a fat binary with one cubin per arch. V100
# (sm_70) needs CUDA 12, which this CUDA 13 image can't build; maya.py builds it natively instead.
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;120}"
# What our launcher (models/glm-5.3-flash/strata-maya/launcher/maya-entrypoint.sh) passes. A release
# that drops or renames one would break every launch, so the build fails instead.
REQUIRED_SERVER_FLAGS="${REQUIRED_SERVER_FLAGS:---engine --config --host --port --api-key}"
REQUIRED_PACK_FLAGS="${REQUIRED_PACK_FLAGS:---gguf --out --compat-bf16}"
# What our DeepSeek launcher (the DeepSeek-V4.1-Flash-1017 strata-maya slugs') writes into the engine's config,
# and passes to the pack builder: the same rule, for build/maya-ds4 and tools/ds4/pack_gguf.py. --engram is for a quant
# whose Engram tables are a GGUF set of their own rather than among its shards (Maya-XS, #1983)
REQUIRED_DS4_FLAGS="${REQUIRED_DS4_FLAGS:---pack --max-context --tp --cpu-threads --dspark --vision --serve}"
REQUIRED_DS4_PACK_FLAGS="${REQUIRED_DS4_PACK_FLAGS:---model --out --drafter --engram --shards}"
# ... and our MiniMax launcher (the MiniMax-M3.1-Flash strata-maya slugs'), for build/maya-mm3 and tools/mm3/pack_gguf.py.
# maya-mm3's usage text leaves out flags its parser takes (--profile, --serve), so these are checked in the binary
# itself: each must be one of its strings, exactly (the parser compares the argument against that literal).
REQUIRED_MM3_FLAGS="${REQUIRED_MM3_FLAGS:---pack --max-context --profile --kv-cache --kv-streaming --cpu-threads --dspark --vision --serve}"
REQUIRED_MM3_PACK_FLAGS="${REQUIRED_MM3_PACK_FLAGS:---model --out --companions --shards}"

TAG="" PUSH=0 FORCE=0 VERIFY="" JOBS=""
usage() { awk 'NR > 1 && !/^#/ {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:?--tag needs a value}"; shift 2 ;;
    --push) PUSH=1; shift ;;
    --force) FORCE=1; shift ;;
    --verify) VERIFY="${2:?--verify needs an image}"; shift 2 ;;
    --jobs) JOBS="${2:?--jobs needs a number}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[maya-image] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

say() { echo "[maya-image] $*"; }
die() { echo "[maya-image] ERROR: $*" >&2; exit 1; }
[[ -z "$JOBS" || "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs needs a positive number, not '$JOBS'"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../../scripts/lib/image-push-guard.sh
source "$here/../../../../scripts/lib/image-push-guard.sh"
if [[ "$PUSH" == 1 ]]; then
  why="$(image_push_guard "$here")" || die "not pushing: $why"
fi
# Test seam (scripts/tests/test-image-push-guard.sh): stop once the guard has had its say.
if [[ "${IMAGE_PUSH_GUARD_ONLY:-0}" == 1 ]]; then
  if [[ "$PUSH" == 1 ]]; then say "push guard passed"; else say "push guard not applied (no --push)"; fi
  exit 0
fi
recipe_commit="$(git -C "$here" rev-parse HEAD 2>/dev/null || echo unknown)"

# ── checks on a built image (step 4) ────────────────────────────────────────────────
check_image() {
  local ref="$1" version="$2" evidence archs got missing help absent f
  local run=(docker run --rm --entrypoint)

  evidence="$("${run[@]}" cat "$ref" /opt/club3090/maya-cpu-flags.txt 2>/dev/null)" \
    || die "$ref has no /opt/club3090/maya-cpu-flags.txt: not built by this recipe, so its CPU code may be native to the build machine"
  python3 - "$evidence" <<'PY' || die "the engine is not a portable (AVX2-baseline) build"
import sys
seen = {}
for line in sys.argv[1].splitlines():
    tree, _, setting = line.partition(" ")
    name, _, value = setting.partition("=")
    seen[name.split(":")[0]] = value.strip().upper()
bad = []
if not seen:
    bad.append("no CPU settings recorded")
for name, want in (("STRATA_PORTABLE", "ON"), ("GGML_NATIVE", "OFF"), ("GGML_AVX2", "ON")):
    if seen and seen.get(name) != want:
        bad.append(f"{name}={seen.get(name, '(unset)')}, want {want}")
for name, value in sorted(seen.items()):
    if name.startswith(("GGML_AVX512", "GGML_AVX_VNNI", "GGML_AMX_")) and value != "OFF":
        bad.append(f"{name}={value}, want OFF (above the AVX2 baseline)")
for m in bad:
    print(f"[maya-image]   {m}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  say "check: the engine was configured portable (STRATA_PORTABLE on, GGML_NATIVE off, nothing above AVX2)"

  archs="$("${run[@]}" cat "$ref" /opt/club3090/cuda-architectures.txt 2>/dev/null | tr -d '[:space:]')" \
    || die "$ref has no /opt/club3090/cuda-architectures.txt"
  [[ "$archs" == "$CUDA_ARCHITECTURES" ]] || die "the engine was configured for archs '$archs', not '$CUDA_ARCHITECTURES'"
  got="$("${run[@]}" cat "$ref" /opt/maya/VERSION 2>/dev/null | tr -d '[:space:]')" || die "$ref has no /opt/maya/VERSION"
  [[ "$got" == "$version" ]] || die "/opt/maya/VERSION says '$got', not $version"
  say "check: archs $CUDA_ARCHITECTURES, VERSION $version"

  # libcuda / libnvidia-* come from the host driver at run time, so only those may be missing here
  # the engine is build/maya from v1.1.16 on (the release renamed its programs), build/strata before
  engine="$("${run[@]}" sh "$ref" -c 'for e in /opt/maya/build/maya /opt/maya/build/strata; do [ -x "$e" ] && { echo "$e"; exit 0; }; done; exit 1')" \
    || die "$ref has no engine at /opt/maya/build/maya or /opt/maya/build/strata"
  missing="$("${run[@]}" ldd "$ref" "$engine" | command grep 'not found' \
             | command grep -v -E 'libcuda\.so|libnvidia' || true)"
  [[ -z "$missing" ]] || die "libraries missing from the image:
$missing"
  say "check: every library of $engine resolves except the host driver's"

  help="$("${run[@]}" /opt/maya/.venv/bin/python "$ref" /opt/maya/serve/server.py --help 2>&1)" \
    || die "serve/server.py --help failed:
$help"
  absent=""
  for f in $REQUIRED_SERVER_FLAGS; do
    command grep -q -E -- "(^|[^a-z-])$f([^a-z-]|$)" <<<"$help" || absent="$absent $f"
  done
  [[ -z "$absent" ]] || die "serve/server.py no longer offers:$absent (update launcher/maya-entrypoint.sh before taking this release)"
  help="$("${run[@]}" /opt/maya/.venv/bin/python "$ref" /opt/maya/tools/iq_pack.py --help 2>&1)" \
    || die "tools/iq_pack.py --help failed:
$help"
  for f in $REQUIRED_PACK_FLAGS; do
    command grep -q -E -- "(^|[^a-z-])$f([^a-z-]|$)" <<<"$help" || absent="$absent $f"
  done
  [[ -z "$absent" ]] || die "tools/iq_pack.py no longer offers:$absent (update launcher/maya-entrypoint.sh before taking this release)"
  "${run[@]}" test "$ref" -s /opt/maya/data/expert-profile-glm.bin \
    || die "data/expert-profile-glm.bin is missing (the configs pass --expert-profile)"
  say "check: server.py and iq_pack.py offer every flag our launcher passes; the expert profile is there"

  # DeepSeek-V4.1-Flash-1017's engine and pack builder (its strata-maya slugs)
  local ds4=/opt/maya/build/maya-ds4
  "${run[@]}" test "$ref" -x "$ds4" || die "$ref has no $ds4 (the release's CMakeLists.txt no longer defines maya-ds4?)"
  missing="$("${run[@]}" ldd "$ref" "$ds4" | command grep 'not found' \
             | command grep -v -E 'libcuda\.so|libnvidia' || true)"
  [[ -z "$missing" ]] || die "libraries of $ds4 missing from the image:
$missing"
  # (maya-ds4 prints its usage for an unknown flag; the exit code is not part of the check)
  help="$("${run[@]}" "$ds4" "$ref" --help 2>&1 || true)"
  absent=""
  for f in $REQUIRED_DS4_FLAGS; do
    command grep -q -E -- "(^|[^a-z-])$f([^a-z-]|$)" <<<"$help" || absent="$absent $f"
  done
  [[ -z "$absent" ]] || die "maya-ds4 no longer offers:$absent (update the DeepSeek launcher before taking this release)"
  help="$("${run[@]}" /opt/maya/.venv/bin/python "$ref" /opt/maya/tools/ds4/pack_gguf.py --help 2>&1)" \
    || die "tools/ds4/pack_gguf.py --help failed:
$help"
  for f in $REQUIRED_DS4_PACK_FLAGS; do
    command grep -q -E -- "(^|[^a-z-])$f([^a-z-]|$)" <<<"$help" || absent="$absent $f"
  done
  [[ -z "$absent" ]] || die "tools/ds4/pack_gguf.py no longer offers:$absent (update the DeepSeek launcher before taking this release)"
  "${run[@]}" test "$ref" -s /opt/maya/data/expert-profile-ds4.bin \
    || die "data/expert-profile-ds4.bin is missing (the DeepSeek launcher seeds each pack's profile from it)"
  say "check: maya-ds4 resolves and it and tools/ds4/pack_gguf.py offer every flag the DeepSeek launcher passes; its expert profile is there"

  # MiniMax-M3.1-Flash's engine and pack builder (its strata-maya slugs)
  local mm3=/opt/maya/build/maya-mm3
  "${run[@]}" test "$ref" -x "$mm3" || die "$ref has no $mm3 (the release's CMakeLists.txt no longer defines maya-mm3?)"
  missing="$("${run[@]}" ldd "$ref" "$mm3" | command grep 'not found' \
             | command grep -v -E 'libcuda\.so|libnvidia' || true)"
  [[ -z "$missing" ]] || die "libraries of $mm3 missing from the image:
$missing"
  absent="$("${run[@]}" /opt/maya/.venv/bin/python "$ref" -c '
import sys
data = open(sys.argv[1], "rb").read()
print(" ".join(f for f in sys.argv[2:] if b"\0" + f.encode() + b"\0" not in data))' "$mm3" $REQUIRED_MM3_FLAGS 2>&1)" \
    || die "could not read $mm3's strings:
$absent"
  [[ -z "$absent" ]] || die "maya-mm3 no longer takes: $absent (update the MiniMax launcher before taking this release)"
  help="$("${run[@]}" /opt/maya/.venv/bin/python "$ref" /opt/maya/tools/mm3/pack_gguf.py --help 2>&1)" \
    || die "tools/mm3/pack_gguf.py --help failed:
$help"
  absent=""
  for f in $REQUIRED_MM3_PACK_FLAGS; do
    command grep -q -E -- "(^|[^a-z-])$f([^a-z-]|$)" <<<"$help" || absent="$absent $f"
  done
  [[ -z "$absent" ]] || die "tools/mm3/pack_gguf.py no longer offers:$absent (update the MiniMax launcher before taking this release)"
  "${run[@]}" test "$ref" -s /opt/maya/data/expert-profile-mm3.bin \
    || die "data/expert-profile-mm3.bin is missing (the MiniMax launcher passes it as the engine's --profile)"
  "${run[@]}" test "$ref" -s /opt/maya/tools/mm3/chat_template.jinja \
    || die "tools/mm3/chat_template.jinja is missing (the MiniMax pack builder copies it into every pack)"
  say "check: maya-mm3 resolves and takes every flag the MiniMax launcher passes; tools/mm3/pack_gguf.py offers its flags; the expert profile and chat template are there"

  # the launcher's first start builds a pack. gguf-py must import as the tools find it, and the pack
  # builder must index the release's own 12 MB synthetic GLM-5 model with the flags the launcher passes.
  # That model carries no tokenizer, so the builder's last step (the tokenizer) can't run on it: the
  # check wants the index and the dense weights, and a real model's first start covers the rest.
  packed="$("${run[@]}" sh "$ref" -c 'cd /opt/maya \
    && .venv/bin/python -c "import sys; sys.path.insert(0, \"tools\"); import _paths; _paths.add_gguf_py(); import gguf" \
    && { .venv/bin/python tools/iq_pack.py --gguf data/glm5-synth.gguf --out /tmp/pack --compat-bf16 >/tmp/pack.log 2>&1
         if test -s /tmp/pack/index.txt && test -s /tmp/pack/dense.bin; then echo ok; else tail -5 /tmp/pack.log; echo failed; fi; }' 2>&1)" || true
  [[ "$(tail -1 <<<"$packed")" == ok ]] || die "gguf-py or the pack builder doesn't work in the image:
$packed"
  say "check: gguf-py imports and the pack builder indexes the release's synthetic GLM-5 model"

  # the release's setup.py, read in the image itself (not the list the Dockerfile recorded), so an image
  # whose recipe missed a package fails here: v1.1.18's jsonschema, for the Responses API's structured
  # output, was missing from the hand-kept list the recipe had until then
  pkgs="$(docker run --rm -i --entrypoint /opt/maya/.venv/bin/python "$ref" - 2>&1 <<'PY'
import ast, sys
from importlib import metadata
from pip._vendor.packaging.requirements import Requirement
tree = ast.parse(open("/opt/maya/setup.py", encoding="utf-8").read())
found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "PY_PACKAGES" for t in node.targets)]
if len(found) != 1:
    sys.exit("setup.py has no single PY_PACKAGES list")
bad, n = [], 0
for spec in ast.literal_eval(found[0]):
    req = Requirement(spec)
    if req.name.lower() in ("cmake", "ninja"):
        continue
    n += 1
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        bad.append(f"{spec}: not installed")
        continue
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        bad.append(f"{spec}: {version} installed")
print("\n".join(bad))
print("ok" if n and not bad else "failed")
PY
)" || true
  [[ "$(tail -1 <<<"$pkgs")" == ok ]] || die "the image's .venv lacks packages the release's setup.py installs:
$pkgs"
  say "check: every package in setup.py's PY_PACKAGES (less cmake / ninja) is installed at a version it accepts"
}

# ── 1. the release and its commit ────────────────────────────────────────────────
if [[ -z "$TAG" ]]; then
  [[ -z "$VERIFY" ]] || die "--verify needs --tag (the release the image should be)"
  # newest tag by version order: releases are tagged vMAJOR.MINOR.PATCH
  TAG="$(git ls-remote --tags --refs "$REPO_URL" 'v*' | awk '{sub("refs/tags/","",$2); print $2}' \
         | command grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1)"
  [[ -n "$TAG" ]] || die "no release tags found at $REPO_URL"
fi
[[ "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "'$TAG' doesn't look like a Project Maya release tag (vX.Y.Z)"
version="${TAG#v}"

if [[ -n "$VERIFY" ]]; then
  check_image "$VERIFY" "$version"
  say "$VERIFY passes every check for $TAG"
  exit 0
fi

refs="$(git ls-remote "$REPO_URL" "refs/tags/$TAG" "refs/tags/$TAG^{}")"
commit="$(awk -v t="refs/tags/$TAG^{}" '$2 == t {print $1}' <<<"$refs")"
[[ -n "$commit" ]] || commit="$(awk -v t="refs/tags/$TAG" '$2 == t {print $1}' <<<"$refs")"
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || die "tag $TAG not found at $REPO_URL"
short="${commit:0:7}"
say "release $TAG = commit $commit"

# ── 2. already published? ─────────────────────────────────────────────────────────
if [[ "$FORCE" != 1 ]]; then
  if probe="$(docker buildx imagetools inspect "$IMAGE:$TAG" 2>&1)"; then
    say "$IMAGE:$TAG is already published; nothing to do (--force rebuilds)"
    exit 0
  fi
  command grep -q -i 'not found' <<<"$probe" \
    || die "could not tell whether $IMAGE:$TAG is published (not rebuilding over it):
$probe"
fi

# ── 3. build ────────────────────────────────────────────────────────────────────────
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git -C "$work" init -q
git -C "$work" remote add origin "$REPO_URL"
git -C "$work" fetch -q --depth 1 origin "$commit"
git -C "$work" checkout -q FETCH_HEAD
[[ "$(git -C "$work" rev-parse HEAD)" == "$commit" ]] || die "checked-out commit is not $commit"
[[ "$(tr -d '[:space:]' < "$work/VERSION" 2>/dev/null)" == "$version" ]] \
  || die "the release's VERSION file doesn't say $version"
llama="$(command grep -m1 -E '^LLAMA_CPP_COMMIT *= *"[0-9a-f]{40}"' "$work/setup.py" | command grep -o -E '[0-9a-f]{40}' || true)"
[[ -n "$llama" ]] || die "setup.py has no LLAMA_CPP_COMMIT: upstream changed how it pins llama.cpp"
cmake_pin="$(command grep -m1 -E 'GIT_TAG +[0-9a-f]{40}' "$work/CMakeLists.txt" | command grep -o -E '[0-9a-f]{40}' || true)"
[[ -z "$cmake_pin" || "$cmake_pin" == "$llama" ]] \
  || die "setup.py pins llama.cpp $llama but CMakeLists.txt fetches $cmake_pin: upstream's two pins disagree"

# The checkout's .git differs from one fetch to the next (its index and fetch state) even at the same commit, and the
# Dockerfile copies the whole context, so with it every build missed Docker's layer cache from that COPY on and
# recompiled every engine, a check build and the push build after it alike. The commit is checked above, and the
# image never kept .git (the runtime stage drops it).
rm -rf "$work/.git"
say "building $IMAGE:$TAG (archs $CUDA_ARCHITECTURES, AVX2 baseline, llama.cpp ${llama:0:7}${JOBS:+, jobs $JOBS})"
docker build --pull -f "$here/Dockerfile" \
  --build-arg "CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES" \
  --build-arg "LLAMA_CPP_COMMIT=$llama" \
  ${JOBS:+--build-arg "JOBS=$JOBS"} \
  --label "org.opencontainers.image.title=strata-maya-club3090" \
  --label "org.opencontainers.image.description=Project Maya $TAG (a Strata fork for GLM-5.3-Flash) built by club-3090 at the release commit, text only, ggml's CPU code for an AVX2 baseline" \
  --label "org.opencontainers.image.source=https://github.com/noonghunna/club-3090" \
  --label "org.opencontainers.image.revision=$recipe_commit" \
  --label "org.opencontainers.image.version=$TAG" \
  --label "org.opencontainers.image.licenses=MIT" \
  --label "club3090.upstream.source=$REPO_URL" \
  --label "club3090.upstream.revision=$commit" \
  --label "club3090.recipe=https://github.com/noonghunna/club-3090/tree/$recipe_commit/models/glm-5.3-flash/strata-maya/image" \
  --label "club3090.cuda_architectures=$CUDA_ARCHITECTURES" \
  --label "club3090.cpu_baseline=avx2-portable" \
  --label "club3090.llama_cpp_commit=$llama" \
  -t "$IMAGE:$TAG" -t "$IMAGE:sha-$short" "$work"

# ── 4. checks ──────────────────────────────────────────────────────────────────────
check_image "$IMAGE:$TAG" "$version"

# ── 5. publish ──────────────────────────────────────────────────────────────────────
if [[ "$PUSH" != 1 ]]; then
  say "built and checked $IMAGE:$TAG (not pushed; --push publishes)"
  exit 0
fi
docker push -q "$IMAGE:$TAG" >/dev/null
docker push -q "$IMAGE:sha-$short" >/dev/null
digest="$(docker buildx imagetools inspect "$IMAGE:$TAG" --format '{{json .Manifest.Digest}}' | tr -d '"')"
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "could not read the pushed digest (got '$digest')"
say "pushed $IMAGE:$TAG and :sha-$short = $IMAGE@$digest"
say "for the pin PR: Project Maya $TAG, upstream ${commit:0:8}, recipe ${recipe_commit:0:8}, llama.cpp ${llama:0:7}, archs $CUDA_ARCHITECTURES, text only"
say "not pinned: the strata-maya engine profile moves to it only after a rig boot + verify-full (engine-pin-bump.sh)"
