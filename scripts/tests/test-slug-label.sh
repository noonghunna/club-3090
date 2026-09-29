#!/usr/bin/env bash
# test-slug-label — launched containers carry the registry slug they run (club3090.slug).
#
# Two slugs can share one compose file (vllm/dual + vllm/qwen-27b-dual-fast;
# llamacpp/default + llamacpp/mtp), and per-slug launch settings (#1465) make them run
# differently, so "which compose" no longer says "which slug". switch.sh and gpu-mode's
# model modes now add a label override at `up` (scripts/lib/slug_label.py); the #1477
# resolver (measurement records) and c3's running-container match trust it first.
#
#   1. slug_label.py: writes the override for a registered slug (every service labelled),
#      refuses an unknown slug or a compose that isn't the slug's, rewrites only on change.
#   2. docker compose really merges it (render only).
#   3. gpu-mode's 27b mode (run from a scratch copy, docker/sudo shimmed) hands compose
#      the override for vllm/dual.
#   4. measurement_record.resolve_serving: twin slugs on one container name are told apart
#      by the label; without it, registry order decides (the old behaviour).
# switch.sh's own path is covered in test-launch-settings (the rendered container label).
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export CLUB3090_DATA_DIR="$T/data" CLUB3090_CACHE_DIR="$T/cache"   # never your real ones
compose_of() { python3 -c 'import sys; sys.path.insert(0, "."); from scripts.lib.profiles.compose_registry import get_registry; print(get_registry()[sys.argv[1]]["compose_path"])' "$1"; }
label() { python3 scripts/lib/slug_label.py override --slug "$1" --compose "$2"; }

echo "1. slug_label.py"
DUAL_C="$(compose_of vllm/dual)"; FAST_C="$(compose_of vllm/qwen-27b-dual-fast)"
[[ "$DUAL_C" == "$FAST_C" ]] && ok "fixture: vllm/dual and vllm/qwen-27b-dual-fast share $DUAL_C" || bad "fixture: the twins no longer share a compose"
o1="$(label vllm/dual "$DUAL_C")"; o2="$(label vllm/qwen-27b-dual-fast "$DUAL_C")"
[[ "$o1" == "$T/data/compose-labels/vllm-dual.yml" && "$o2" == "$T/data/compose-labels/vllm-qwen-27b-dual-fast.yml" ]] \
  && ok "one override per slug, in the data dir" || bad "override paths: '$o1' '$o2'"
command grep -q 'club3090.slug: "vllm/qwen-27b-dual-fast"' "$o2" && ok "the override labels the service with the slug" || bad "override content: $(cat "$o2" 2>/dev/null)"
[[ -z "$(label vllm/no-such-slug "$DUAL_C")" ]] && ok "an unknown slug: no label" || bad "unknown slug got a label"
[[ -z "$(label vllm/dual "$(compose_of vllm/gemma-31b-dual)")" ]] && ok "a compose that isn't the slug's: no label (never a wrong one)" || bad "mismatched compose got a label"
m1="$(stat -c %Y.%N "$o1" 2>/dev/null || stat -c %Y "$o1")"; sleep 1; label vllm/dual "$DUAL_C" >/dev/null; m2="$(stat -c %Y.%N "$o1" 2>/dev/null || stat -c %Y "$o1")"
[[ "$m1" == "$m2" ]] && ok "rewritten only when it changes" || bad "rewrote an unchanged override"
n="$(python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from pathlib import Path
from scripts.lib.profiles.compose_registry import get_registry
from scripts.lib.slug_label import write_override
reg = get_registry()
print(sum(1 for s, e in reg.items() if Path(e["compose_path"]).is_file() and write_override(s, Path(e["compose_path"])) is None))
PY
)"
[[ "$n" == 0 ]] && ok "every registry slug with a compose on disk gets a label" || bad "$n slug(s) got no label"

echo "2. docker compose merges it"
if docker compose version >/dev/null 2>&1; then
  got="$(cd "$(dirname "$DUAL_C")" && env -u HF_TOKEN -u LITELLM_MASTER_KEY MODEL_DIR=/nonexistent docker compose -f "$(basename "$DUAL_C")" -f "$o2" config 2>/dev/null | command grep -E 'club3090.slug:')"
  [[ "$got" == *"vllm/qwen-27b-dual-fast"* ]] && ok "rendered: club3090.slug: vllm/qwen-27b-dual-fast" || bad "render: '$got'"
else
  echo "  - skipped: no docker compose"
fi

echo "3. gpu-mode's 27b mode passes the override (scratch copy, docker/sudo shimmed)"
S="$T/tree"; mkdir -p "$S" "$T/bin" "$T/home"
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$S"
cat > "$T/bin/docker" <<'MOCK'
#!/usr/bin/env bash
echo "docker $*" >> "$MOCK_LOG"
exit 0
MOCK
cat > "$T/bin/sudo" <<'MOCK'
#!/usr/bin/env bash
while [[ "${1:-}" == *=* ]]; do shift; done
[[ "${1:-}" == docker ]] && exec "$@"
echo "sudo(skipped) $*" >> "$MOCK_LOG"
exit 0
MOCK
for b in nvidia-smi curl systemctl; do printf '#!/usr/bin/env bash\nexit 0\n' > "$T/bin/$b"; done
chmod +x "$T/bin/"*
for b in docker sudo nvidia-smi curl; do
  [ "$(PATH="$T/bin:$PATH" command -v "$b")" = "$T/bin/$b" ] || { echo "✗ $b shim not first on PATH — refusing to run gpu-mode" >&2; exit 1; }
done
export MOCK_LOG="$T/docker.log"; : > "$MOCK_LOG"
env -u HF_TOKEN -u LITELLM_MASTER_KEY PATH="$T/bin:$PATH" HOME="$T/home" CLUB3090_DIR="$S" GPU_MODE_SETTLE_TIMEOUT=0 timeout 180 bash "$S/scripts/gpu-mode.sh" 27b > "$T/gpu-mode.out" 2>&1
up="$(command grep -E 'compose .*-f fp8-mtp\.yml .* up -d' "$MOCK_LOG" | head -1)"
[[ "$up" == *"-f fp8-mtp.yml -f $T/data/compose-labels/vllm-dual.yml"* ]] \
  && ok "compose up gets -f …/compose-labels/vllm-dual.yml" || bad "27b up: '${up:-none}' ($(tail -3 "$T/gpu-mode.out"))"
command grep -E 'compose .* up -d' "$MOCK_LOG" | command grep -v fp8-mtp.yml | command grep -q compose-labels \
  && bad "a service compose got a slug label" || ok "services (litellm, qdrant, …) get none"

echo "4. the resolver tells twin slugs apart by the label"
resolve() { env -u HF_TOKEN -u LITELLM_MASTER_KEY PATH="$T/bin2:$PATH" HOME="$T/home" python3 - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts.lib.profiles.measurement_record import resolve_serving
print(resolve_serving(None))
PY
}
mkdir -p "$T/bin2"
cat > "$T/bin2/docker" <<'MOCK'
#!/usr/bin/env bash
[[ "$1" == ps ]] && printf '%s\n' "${MOCK_PS:-}"
exit 0
MOCK
chmod +x "$T/bin2/docker"
CNAME="$(python3 -c 'import re,sys; t=open(sys.argv[1]).read(); print(re.search(r"container_name:\s*\"?(?:\$\{[^:}]*:-)?([A-Za-z0-9._-]+)", t).group(1))' "$DUAL_C")"
got="$(MOCK_PS="$CNAME|vllm/qwen-27b-dual-fast" resolve)"
[[ "$got" == "('vllm/qwen-27b-dual-fast', '$CNAME')" ]] && ok "labelled vllm/qwen-27b-dual-fast → that slug, not vllm/dual" || bad "labelled twin: $got"
got="$(MOCK_PS="$CNAME|" resolve)"
[[ "$got" == "('vllm/dual', '$CNAME')" ]] && ok "unlabelled (started before this) → registry order, as before" || bad "unlabelled twin: $got"

[ "$fail" -eq 0 ] && echo "test-slug-label: ok" || { echo "test-slug-label: FAIL"; exit 1; }
