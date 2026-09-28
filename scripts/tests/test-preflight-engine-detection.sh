#!/usr/bin/env bash
# test-preflight-engine-detection — preflight_compose_deps must recognise EVERY
# llama.cpp-family compose we ship, including our own forks (#1247).
#
# THE BUG. The function decided "is this llama.cpp?" with a private image regex:
#
#     image:.*(ggml-org/llama\.cpp|ikawrakow/ik-llama|beellama)
#
# which did not know `ghcr.io/noonghunna/llamacpp-club3090` or `…/llamacpp-prism`.
# 30 slugs therefore SKIPPED the GGUF / drafter / mmproj existence check entirely
# and fell to the vLLM HF-cache path, which found nothing to complain about and
# returned 0. A user with no DFlash2 drafter on disk got an endless CRASH-LOOP
# instead of one clear line naming the missing file — the check that exists to
# prevent exactly that had silently measured nothing.
#
# ⚠️ This is also why the classification now DELEGATES to engine-kind.sh (#1282):
# it was the eighth private classifier, and the tree-wide guard could not see it
# because that guard greps for the PYTHON shape (`startswith("vllm")`), not a
# bash image regex. Arm 3 below closes that gap for this shape.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FAIL=0
ok()  { echo "  ✓ $1"; }
bad() { echo "  ✗ $1" >&2; FAIL=1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
EMPTY="$TMP/no-models"; mkdir -p "$EMPTY"

# shellcheck source=scripts/preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh" >/dev/null 2>&1

# --- 1. every llama.cpp-family slug takes the GGUF path -----------------------
# Against an EMPTY model dir a llama.cpp compose must name its missing weights.
# Silence here is the bug: it means the compose was routed to the HF-cache path.
mapfile -t ROWS < <(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
FAM = ("llamacpp/", "llamacpp-club3090/", "ik-llama/", "beellama/", "llama-cpp-prism")
for slug, e in sorted(COMPOSE_REGISTRY.items()):
    if slug.startswith(FAM) and e.get("compose_path"):
        print(f"{slug}\t{e['compose_path']}")
PY
)
(( ${#ROWS[@]} > 0 )) || bad "no llama.cpp-family slugs found — the probe is broken, not the code"
blind=(); seen=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r slug cp <<<"$row"
  [[ -f "$cp" ]] || continue
  seen=$((seen + 1))
  out="$(MODEL_DIR="$EMPTY" preflight_compose_deps "$cp" 2>&1)"
  command grep -q "llama.cpp GGUF weights" <<<"$out" || blind+=("$slug")
done
if (( ${#blind[@]} )); then
  bad "${#blind[@]} of ${seen} llama.cpp-family slugs SKIP the weights check: ${blind[*]:0:4}"
else
  ok "all ${seen} llama.cpp-family slugs take the GGUF-presence path"
fi

# --- 2. a required drafter is named, not discovered at boot -------------------
# The #1247 slug: block-45 GLM has no MTP head, so the external DFlash2 drafter
# is the ONLY spec path and the compose refuses without it. That refusal must
# happen HERE, as one line, not as a container restart loop.
GLM="models/glm-5.3-flash/llamacpp-club3090/compose/dual/devquasar-q3km/moecache.yml"
if [[ -f "$GLM" ]]; then
  out="$(MODEL_DIR="$EMPTY" preflight_compose_deps "$GLM" 2>&1)"
  command grep -q "speculative drafter GGUF" <<<"$out" \
    && ok "a missing required drafter is named by preflight (#1247's crash-loop)" \
    || bad "the drafter is not named — the user still learns this from a restart loop"
  # …and the remediation must be runnable by someone who is not the maintainer (#1271)
  command grep -qE "/opt/ai|/mnt/" <<<"$out" \
    && bad "preflight remediation leaks a rig-only path" \
    || ok "remediation names no rig-only path"
fi

# --- 3. the classification is not re-implemented privately -------------------
# The regex this replaced is the shape to keep out: a grep over a compose
# `image:` line with a vendor alternation, deciding an engine family. The
# tree-wide classifier guard greps for the PYTHON shape and cannot see this one.
priv="$(command grep -rnE 'image:\.\*\(' "${ROOT_DIR}/scripts" --include='*.sh' 2>/dev/null \
          | command grep -v '/scripts/tests/' \
          | command grep -v '/scripts/lib/engine-kind.sh' \
          | command grep -vE ':[0-9]+:[[:space:]]*#' || true)"
if [[ -n "$priv" ]]; then
  bad "a private image-based engine classifier is back — put the arms in engine-kind.sh"
  printf '     %s\n' "$priv" >&2
else
  ok "no script regexes a compose image: line to decide an engine family"
fi

# --- 4. non-llama.cpp engines must NOT be classified as llama.cpp -------------
# ⚠️ This arm used to assert a SYMPTOM -- that a vLLM slug never prints
# "llama.cpp GGUF weights". A negative control showed that is worthless: force
# the detector to call EVERYTHING llamacpp and the arm still passed, because
# vLLM composes pass `--model <hf-id>` and have no /models/ path to be missing,
# so the message never appears either way. Assert the DECISION instead.
# shellcheck source=scripts/lib/engine-kind.sh
source "${ROOT_DIR}/scripts/lib/engine-kind.sh"
mapfile -t OTHERS < <(python3 - <<'PYX'
import sys, io, re
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
for slug, e in sorted(COMPOSE_REGISTRY.items()):
    if not slug.startswith(("vllm/", "sgl/", "exllamav3/")):
        continue
    cp = e.get("compose_path")
    if not cp:
        continue
    try:
        txt = io.open(cp, encoding="utf-8").read()
    except OSError:
        continue
    m = re.search(r'^\s*image:\s*"?(?:\$\{[A-Z_0-9]+:-)?([^}"\s]+)\}?', txt, re.M)
    if m:
        print(slug + "\t" + m.group(1))
PYX
)
wrong=()
for row in "${OTHERS[@]}"; do
  IFS=$'\t' read -r slug img <<<"$row"
  [[ "$(engine_kind_from_image "$img")" == "llamacpp" ]] && wrong+=("${slug}")
done
if (( ${#wrong[@]} )); then
  bad "${#wrong[@]} non-llama.cpp slugs classify as llamacpp: ${wrong[*]:0:4}"
else
  ok "${#OTHERS[@]} vLLM/SGLang/exl3 images classify as NOT llamacpp"
fi

# ...and an unknown image must stay 'unknown', never default into a family --
# that default is precisely how 'everything becomes llamacpp'.
[[ "$(engine_kind_from_image 'registry.example/some-unknown-server:1')" == "unknown" ]] \
  && ok "an unrecognised image is 'unknown', not defaulted into a family" \
  || bad "an unrecognised image was given a family"
# --- 5. the OVER-correction, which is the same bug mirrored -------------------
# A detector that says 'llamacpp' too eagerly does not merely mis-label: it
# routes vLLM composes off the HF-cache path, and they then pass an empty model
# dir SILENTLY. Measured: forcing every image to llamacpp flips vllm/dual and
# vllm/minimal from rc=1 (correctly refusing) to rc=0. That is the #1247 false
# clean again, on the other engine -- so this arm is a POSITIVE CONTROL that the
# vLLM path is still live, not a style check.
for _v in vllm/dual vllm/minimal; do
  _cp="$(python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
e = COMPOSE_REGISTRY.get('${_v}') or {}
print(e.get('compose_path') or '')" 2>/dev/null)"
  [[ -n "$_cp" && -f "$_cp" ]] || continue
  MODEL_DIR="$EMPTY" preflight_compose_deps "$_cp" >/dev/null 2>&1
  if (( $? == 0 )); then
    bad "${_v} PASSED against an empty model dir — the vLLM HF-cache check is not running"
  else
    ok "${_v} still refuses an empty model dir (vLLM path live)"
  fi
done
(( FAIL )) && { echo "test-preflight-engine-detection: FAIL" >&2; exit 1; }
echo "test-preflight-engine-detection: ok"
