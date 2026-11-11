#!/usr/bin/env bash
# Guard: scripts/club3090-env.sh finds the running engine and the model it serves.
#
# It used its own container-name prefix list until 2026-10-13, which had no arm for
# exllamav3's tabbyapi-* containers, so on a rig serving exl3 it printed "No club-3090
# container running." It now asks the shared helpers (club-containers.sh,
# served-model.sh). Stub `docker` and `curl` on PATH stand in for the rig:
#   1. TabbyAPI: found on its internal port 5000, and the model is the one in its
#      /v1/model card, not a folder from /v1/models (#1360);
#   2. vLLM: no card, so the longest /v1/models alias still wins (unchanged);
#   3. the stack's own SearXNG (8088->8080, not ours) is not taken for an engine (#1584);
#   4. a container that also publishes a metrics port: the engine port's mapping wins;
#   5. a container bound to 127.0.0.1: the URL uses that address.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

rc=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*"; rc=1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/bin"

# docker: `ps` prints $FAKE_PS (Names|Ports|Image lines); everything else prints nothing.
cat > "$work/bin/docker" <<'SH'
#!/usr/bin/env bash
[[ "${1:-}" == ps ]] && printf '%s\n' "${FAKE_PS:-}"
exit 0
SH
# curl: .../v1/model answers $FAKE_CARD (or fails like a 404 when unset); .../v1/models
# answers $FAKE_MODELS.
cat > "$work/bin/curl" <<'SH'
#!/usr/bin/env bash
url=""
for a in "$@"; do [[ "$a" == http* ]] && url="$a"; done
case "$url" in
  */v1/model)  [[ -n "${FAKE_CARD:-}" ]] && { printf '%s' "$FAKE_CARD"; exit 0; }; exit 22 ;;
  */v1/models) printf '%s' "${FAKE_MODELS:-}"; exit 0 ;;
esac
exit 7
SH
chmod +x "$work/bin/docker" "$work/bin/curl"

run_env() {  # stdout = export lines, stderr = diagnostics
  PATH="$work/bin:$PATH" ANTHROPIC_BASE_URL="" ANTHROPIC_MODEL="" \
    bash scripts/club3090-env.sh 2>"$work/err"
}

echo "== 1. TabbyAPI (exllamav3)"
out="$(FAKE_PS='tabbyapi-qwen38-flash-next-exl3-305|0.0.0.0:8180->5000/tcp, [::]:8180->5000/tcp|ghcr.io/theroyallab/tabbyapi@sha256:bce9f76f9ff36df82181d6e595052b5c391e744b33ef7795296efb9692821010' \
       FAKE_CARD='{"object":"model","id":"Qwen3.8-Flash-Next-exl3-3.05bpw"}' \
       FAKE_MODELS='{"data":[{"id":"modules"},{"id":"Qwen3.8-Flash-Next-exl3-3.05bpw"},{"id":"an-unrelated-folder-with-a-much-longer-name"}]}' \
       run_env)"
command grep -qx 'export ANTHROPIC_BASE_URL="http://localhost:8180"' <<<"$out" \
  && ok "found on :8180 (internal 5000)" || bad "URL: $out $(cat "$work/err")"
command grep -qx 'export ANTHROPIC_MODEL="Qwen3.8-Flash-Next-exl3-3.05bpw"' <<<"$out" \
  && ok "model from the /v1/model card, not a /v1/models folder" || bad "model: $out"

echo "== 2. vLLM (no model card)"
out="$(FAKE_PS='vllm-qwen36-27b-dual|0.0.0.0:8010->8000/tcp|vllm/vllm-openai:v0.31.0' FAKE_CARD='' \
       FAKE_MODELS='{"data":[{"id":"qwen3.6-27b"},{"id":"qwen3.6-27b-autoround"}]}' run_env)"
command grep -qx 'export ANTHROPIC_BASE_URL="http://localhost:8010"' <<<"$out" \
  && ok "found on :8010" || bad "URL: $out $(cat "$work/err")"
command grep -qx 'export ANTHROPIC_MODEL="qwen3.6-27b-autoround"' <<<"$out" \
  && ok "the longest alias still wins" || bad "model: $out"

echo "== 3. only the stack's SearXNG is up"
out="$(FAKE_PS='searxng|0.0.0.0:8088->8080/tcp|searxng/searxng:2026.9.25-12f8b6515' FAKE_CARD='' FAKE_MODELS='' run_env)"
[[ -z "$out" ]] && command grep -q 'No club-3090 container running' "$work/err" \
  && ok "not taken for an engine, nothing exported" || bad "exported: '$out' / $(cat "$work/err")"

echo "== 4. a metrics port listed before the engine port"
out="$(FAKE_PS='llama-cpp-qwen38-flash-next-q4kxl-moecache|0.0.0.0:9100->9100/tcp, 0.0.0.0:8122->8080/tcp|ghcr.io/noonghunna/llamacpp-club3090@sha256:69b833eea9266707cd9853fc8041348a59e2403a1e763822db5c2a46484db72b' \
       FAKE_CARD='' FAKE_MODELS='{"data":[{"id":"qwen3.8-flash-next"}]}' run_env)"
command grep -qx 'export ANTHROPIC_BASE_URL="http://localhost:8122"' <<<"$out" \
  && ok "the engine port's mapping (8122), not the first one (9100)" || bad "URL: $out $(cat "$work/err")"

echo "== 5. bound to 127.0.0.1"
out="$(FAKE_PS='strata-qwen38-flash-next-iq3s|127.0.0.1:8018->8080/tcp|ghcr.io/noonghunna/strata-club3090@sha256:75deea401784c424810e8c0c40e8798ab3727276ee6d4dd64515042b491d70e1' \
       FAKE_CARD='' FAKE_MODELS='{"data":[{"id":"qwen3.8-flash-next-iq3_s"}]}' run_env)"
command grep -qx 'export ANTHROPIC_BASE_URL="http://127.0.0.1:8018"' <<<"$out" \
  && ok "URL keeps the bound address" || bad "URL: $out $(cat "$work/err")"

if [[ "$rc" -eq 0 ]]; then echo "test-club3090-env: ok"; else echo "test-club3090-env: FAILED" >&2; fi
exit "$rc"
