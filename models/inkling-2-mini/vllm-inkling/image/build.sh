#!/usr/bin/env bash
# Build ghcr.io/noonghunna/vllm-inkling-club3090: Inkling-2-Mini on RTX 3090s (sm_86), from this recipe.
#
# vLLM's own `vllm/vllm-openai:inkling-2-mini` image does not run below Hopper. The Dockerfile next to
# this script rebuilds its two op libraries for sm_86 and applies inkling2-sm86.diff; README.md says
# why this is an image of ours (an image-provenance exception, docs/ADDING_ENGINES.md) and when it ends.
# The maintainer runs this on the reference rig:
#
#   bash models/inkling-2-mini/vllm-inkling/image/build.sh             # build + check, no push
#   bash models/inkling-2-mini/vllm-inkling/image/build.sh --verify <image>   # checks only
#   bash models/inkling-2-mini/vllm-inkling/image/build.sh --push      # push the image built above
#   --jobs N   compile with N jobs (default 16)
#
# The image that gets pushed is the one that was validated. A build is not reproducible bit for bit
# (CMake fetches its third-party sources, apt and pip resolve at build time), so --push does NOT
# rebuild: it pushes the local :<revision> image, and only when that image was built from the commit
# checked out now, that commit is on origin/master and the recipe folder is clean
# (scripts/lib/image-push-guard.sh). The flow: merge the recipe, build from master, boot and verify
# every slug on that image, then --push, then pin the engine profile to the pushed digest.
#
# Checks on a built image (no GPU needed):
#   - the patched vllm/models/inkling2 files match inkling2-sm86.sha256, the files the slugs were
#     validated with;
#   - both op libraries carry sm_86 code, and the image's labels name its base and recipe commit;
#   - Inkling's quantization config admits sm_80+ and the software-E4M3 module imports;
#   - the int8-weights module (r2, VLLM_INKLING_W8) imports and reads its switch;
#   - TORCH_CUDA_ARCH_LIST is 8.6, as in every run behind the published numbers.
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_INKLING_IMAGE:-ghcr.io/noonghunna/vllm-inkling-club3090}"
# The recipe revision. Bump it with any change to the Dockerfile, the diff or the base, so a new image
# never re-uses a tag an engine profile may point at.
REV="r2"
BASE_DIGEST="sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
PKG=/usr/local/lib/python3.14/dist-packages/vllm

die() { echo "build.sh: $*" >&2; exit 1; }

push=0 verify="" jobs=16 force=0
while (($#)); do
  case "$1" in
    --push) push=1 ;;
    --verify) verify="${2:?--verify needs an image}"; shift ;;
    --jobs) jobs="${2:?--jobs needs a number}"; shift ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive number, got '$jobs'"
command -v docker >/dev/null || die "docker not found"

commit="$(git -C "$here" rev-parse HEAD)"
short="${commit:0:7}"

if ((push)); then
  # shellcheck source=../../../../scripts/lib/image-push-guard.sh
  source "$here/../../../../scripts/lib/image-push-guard.sh"
  why="$(image_push_guard "$here")" || die "not pushing: $why"
fi

img="${verify:-$IMAGE:$REV}"
if [[ -z "$verify" && "$push" == 0 ]]; then
  echo "[build] $IMAGE:$REV (+ :sha-$short) from recipe commit $short, $jobs jobs"
  docker build --build-arg RECIPE_COMMIT="$commit" --build-arg MAX_JOBS="$jobs" \
    -t "$IMAGE:$REV" -t "$IMAGE:sha-$short" "$here"
fi
docker image inspect "$img" >/dev/null 2>&1 || die "no local image $img (build it first: bash $0)"

fails=0
check() {  # <what> <command...>: runs the command, reports, counts a failure
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "  ✓ $what"; else echo "  ✗ $what"; fails=$((fails + 1)); fi
}
label() { docker image inspect "$img" --format "{{index .Config.Labels \"$1\"}}"; }
in_image() { docker run --rm --entrypoint bash "$img" -c "$1"; }

echo "[check] $img"
check "patched inkling2 files match inkling2-sm86.sha256" \
  in_image "cd $PKG/models && sha256sum -c --quiet /opt/club3090/inkling2-sm86.sha256"
for lib in _C_stable_libtorch _moe_C_stable_libtorch; do
  check "$lib carries sm_86 code" \
    in_image "/usr/local/cuda/bin/cuobjdump --list-elf $PKG/$lib.abi3.so | command grep -q 'sm_86'"
done
check "the quantization config admits sm_80+ and the E4M3 module imports" \
  docker run --rm --entrypoint python3 "$img" -c "
from vllm.models.inkling2.quant_config import Inkling2QuantConfig as C
assert C.get_min_capability() == 80, C.get_min_capability()
import vllm.models.inkling2.nvidia.ops.e4m3_sw"
check "the W8 module imports and reads VLLM_INKLING_W8" \
  docker run --rm -e VLLM_INKLING_W8=attn,head --entrypoint python3 "$img" -c "
from vllm.models.inkling2.nvidia.ops import w8_gemm
assert w8_gemm.W8 == {'attn', 'head'}, w8_gemm.W8"
check "TORCH_CUDA_ARCH_LIST=8.6" \
  bash -c "docker image inspect '$img' --format '{{json .Config.Env}}' | command grep -q '\"TORCH_CUDA_ARCH_LIST=8.6\"'"
check "base label names the launch image digest" test "$(label org.opencontainers.image.base.digest)" = "$BASE_DIGEST"
if [[ -z "$verify" ]]; then
  check "revision label = the recipe commit checked out ($short)" test "$(label org.opencontainers.image.revision)" = "$commit"
fi
((fails == 0)) || die "$fails check(s) failed on $img"
echo "[check] all passed"

((push)) || exit 0

# Refuse to move a published tag unless asked: an engine profile may pin it. Only "not found" means
# unpublished; any other registry error stops here rather than pushing over it.
if probe="$(docker buildx imagetools inspect "$IMAGE:$REV" 2>&1)"; then
  ((force)) || die "$IMAGE:$REV is already published; bump REV in build.sh (or --force to overwrite)"
elif ! printf '%s' "$probe" | command grep -qiE 'not found|manifest unknown|name unknown'; then
  die "could not check whether $IMAGE:$REV is published: $probe"
fi
docker tag "$img" "$IMAGE:sha-$short"
echo "[push] $IMAGE:$REV and :sha-$short"
docker push -q "$IMAGE:$REV" >/dev/null
docker push -q "$IMAGE:sha-$short" >/dev/null
digest="$(docker buildx imagetools inspect "$IMAGE:$REV" --format '{{json .Manifest.Digest}}' | tr -d '"')"
echo "[push] done: $IMAGE@$digest"
echo "  pin it in scripts/lib/profiles/engines/vllm-inkling.yml (install.spec), then make the GHCR package public"
echo "  (a new GHCR package is private; the catalog requires an anonymous pull)."
