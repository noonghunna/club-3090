#!/usr/bin/env bash
# Gate: the LiteLLM RUNTIME view follows what is serving, and never eats what
# isn't ours.
#
# Two files, two jobs, and conflating them is the bug this guards:
#   config.yaml          TRACKED catalog view — one canonical slug per model,
#                        port-pinned. Rendered by litellm-emit.sh, gated by
#                        test-litellm-generate.sh. Unchanged by any of this.
#   config.runtime.yaml  GITIGNORED runtime view — what the container mounts,
#                        rendered by litellm-sync.sh from live endpoints.
#
# The catalog view cannot be what a gateway SERVES: a model's route names ONE
# slug's default_port, so running a sibling slug for that model advertises the
# model on a port with nothing behind it, and 13 of 22 models had no route at
# all. Both failures are silent — the model list looks populated either way.
#
# ⚠️ THE DANGEROUS HALF IS THE PRUNE. A sync that removes routes can remove the
# WRONG routes, and the blast radius is the cloud block that backs benchlocal
# quality runs. Most of what follows pins down what must NEVER be touched.
set -euo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

fail=0
ok()  { echo "  ok   — $*"; }
bad() { echo "  FAIL — $*" >&2; fail=1; }

SYNC="scripts/lib/litellm-sync.sh"
RUNTIME="services/litellm/config.runtime.yaml"
BACKUP="$(mktemp)"; [[ -f "$RUNTIME" ]] && cp "$RUNTIME" "$BACKUP"
restore() { [[ -s "$BACKUP" ]] && cp "$BACKUP" "$RUNTIME"; rm -f "$BACKUP"; }
trap 'restore; rm -f "${LOCAL_CFG:-}"' EXIT

# Deterministic: C3_LITELLM_FAKE_LIVE substitutes the socket probe, so the gate
# does not depend on whatever this rig happens to be serving.
# This rig's own routes come from a FIXTURE, never the real (gitignored)
# services/litellm/config.local.yaml, so the gate is the same on every checkout.
LOCAL_CFG="$(mktemp)"
cat > "$LOCAL_CFG" <<'YAML'
model_list:
  # a cloud endpoint this rig uses — not on any registry port
  - model_name: rig-cloud-model
    litellm_params:
      model: openai/rig-cloud-model
      api_base: https://cloud.example.invalid/v1
      api_key: os.environ/RIG_CLOUD_KEY
YAML
export C3_LITELLM_LOCAL_CONFIG="$LOCAL_CFG"
render() { C3_LITELLM_FAKE_LIVE="$1" bash "$SYNC" --no-restart --quiet; }
routes() { python3 -c "
import yaml,io,sys
d=yaml.safe_load(io.open('$RUNTIME',encoding='utf-8'))
print(' '.join(sorted(m['model_name'] for m in (d.get('model_list') or []))))"; }

# --- 1: a live endpoint becomes a route, named by the SERVER not the registry -
render "8182=alpha-model,8091=beta-model"
got="$(routes)"
[[ "$got" == *alpha-model* && "$got" == *beta-model* ]] \
  && ok "live endpoints become routes" || bad "live routes missing (got: $got)"

# --- 2: this rig's own routes (config.local.yaml) are served and SURVIVE -----
# A cloud endpoint is not "dead" because a local GPU is idle. If the prune ever
# eats one, whatever depends on it breaks with no obvious cause.
[[ "$got" == *rig-cloud-model* ]] \
  && ok "this rig's own routes are merged in and survive the prune" || bad "LOCAL ROUTES MISSING (got: $got)"

# --- 3: a registry-owned port that is NOT live gets pruned -------------------
render "8182=alpha-model"
got="$(routes)"
[[ "$got" != *beta-model* ]] \
  && ok "a route whose port stopped serving is pruned" || bad "dead route kept: $got"

# --- 4: NOTHING live → zero local routes, cloud intact -----------------------
render ""
got="$(routes)"
if [[ "$got" == *rig-cloud-model* ]] && ! command grep -q "host.docker.internal" "$RUNTIME"; then
  ok "with nothing serving: no engine routes, this rig's own routes intact"
else
  bad "empty-rig render wrong" "this rig's own routes only" "$got"
fi

# --- 5: a route on a port we do NOT own is never touched ---------------------
# Someone's hand-added local service on an unrelated port is not ours to collect.
tmpl="services/litellm/config.yaml"; cp "$tmpl" "$BACKUP.tmpl"
python3 - <<'PY'
import io
p="services/litellm/config.yaml"; s=io.open(p,encoding="utf-8").read()
s=s.replace("  # === END GENERATED LOCAL BLOCK ===",
"""  # === END GENERATED LOCAL BLOCK ===

  - model_name: someones-own-thing
    litellm_params:
      model: openai/someones-own-thing
      api_base: http://host.docker.internal:59999/v1
      api_key: EMPTY""",1)
io.open(p,"w",encoding="utf-8").write(s)
PY
render ""
got="$(routes)"
cp "$BACKUP.tmpl" "$tmpl"; rm -f "$BACKUP.tmpl"
[[ "$got" == *someones-own-thing* ]] \
  && ok "a route on an unowned port is never pruned" \
  || bad "PRUNED A ROUTE WE DO NOT OWN (got: $got)"

# --- 6: --check reports staleness --------------------------------------------
render "8182=alpha-model"
C3_LITELLM_FAKE_LIVE="8182=alpha-model" bash "$SYNC" --check --quiet \
  && ok "--check passes on a fresh render" || bad "--check called a fresh render stale"
if C3_LITELLM_FAKE_LIVE="9999=something-else" bash "$SYNC" --check --quiet 2>/dev/null; then
  bad "--check passed while the runtime view was stale"
else
  ok "--check detects a stale runtime view"
fi

# --- 7: the runtime file is GITIGNORED ---------------------------------------
# It is rig state. Committed, every checkout conflicts on a file nobody edited.
git check-ignore -q "$RUNTIME" \
  && ok "runtime view is gitignored" || bad "$RUNTIME is NOT gitignored"

# --- 8: the compose mounts the RUNTIME view, not the tracked catalog ---------
# The whole mechanism is inert if the container still mounts config.yaml, and
# nothing else would notice: the gateway would just keep serving the old routes.
if command grep -qE '^\s*-\s*\./config\.runtime\.yaml:/app/config\.yaml' services/litellm/docker-compose.yml; then
  ok "compose mounts config.runtime.yaml"
else
  bad "compose mount" "./config.runtime.yaml:/app/config.yaml" \
      "$(command grep -E 'config.*:/app/config' services/litellm/docker-compose.yml || echo none)"
fi

# --- 10: one route shape for every engine (agent clients) --------------------
# Two LiteLLM behaviours, both verified against a capturing stub 2026-09-27:
#   - the `openai` provider REJECTS a top-level reasoning_effort with HTTP 400
#     (UnsupportedParamsError) unless the route lists it in allowed_openai_params;
#   - the `hosted_vllm` provider drops `reasoning_content` from past assistant
#     turns — the field omp replays — so the model never sees the reasoning its
#     client kept (Qwen3.8's template re-renders it by default).
# So: openai everywhere, reasoning_effort allowed, no hosted_vllm, no drop_params
# (which would silently discard the effort instead).
render "8113=qwen3.8-27b@262144,8142=sgl-live@163840,8020=gguf-live"
route_field() { python3 - "$RUNTIME" "$1" "$2" <<'PY2'
import io, sys, yaml
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
m = next((m for m in d.get("model_list") or [] if m.get("model_name") == sys.argv[2]), None)
cur = m
for k in sys.argv[3].split("."):
    cur = cur.get(k) if isinstance(cur, dict) else None
print("" if cur is None else cur)
PY2
}
for r in qwen3.8-27b sgl-live gguf-live; do
  prov="$(route_field "$r" litellm_params.model)"; allowed="$(route_field "$r" litellm_params.allowed_openai_params)"
  drop="$(route_field "$r" litellm_params.drop_params)"
  if [[ "$prov" == "openai/$r" && "$allowed" == "['reasoning_effort']" && -z "$drop" ]]; then
    ok "$r: openai provider, reasoning_effort allowed, no drop_params"
  else
    bad "$r route shape: model=$prov allowed_openai_params=$allowed drop_params=$drop"
  fi
done

# --- 11: model_info from the LIVE server (what omp's discovery: litellm reads) -
[[ "$(route_field qwen3.8-27b model_info.max_input_tokens)" == "262144" && "$(route_field sgl-live model_info.max_input_tokens)" == "163840" ]] \
  && ok "max_input_tokens comes from each server's own max_model_len" \
  || bad "max_input_tokens: qwen=$(route_field qwen3.8-27b model_info.max_input_tokens) sgl=$(route_field sgl-live model_info.max_input_tokens)"
[[ "$(route_field qwen3.8-27b model_info.max_output_tokens)" == "32768" ]] \
  && ok "max_output_tokens capped at 32768" || bad "max_output_tokens: $(route_field qwen3.8-27b model_info.max_output_tokens)"
[[ "$(route_field qwen3.8-27b model_info.supports_reasoning)" == "True" && -z "$(route_field sgl-live model_info.supports_reasoning)" ]] \
  && ok "supports_reasoning only where a slug declares a thinking profile (unknown is omitted, never false)" \
  || bad "supports_reasoning: qwen=$(route_field qwen3.8-27b model_info.supports_reasoning) sgl='$(route_field sgl-live model_info.supports_reasoning)'"

# --- 15: Claude Code's /v1/messages goes to the engine's own endpoint, where it holds up
# LiteLLM's translation of /v1/messages builds no thinking blocks from vLLM/SGLang
# (the Responses bridge reads only a reasoning summary), so Claude Code never got
# the model's reasoning back. `supported_endpoints` with /v1/messages makes LiteLLM
# forward the request untranslated. Only on routes the probe marked (+messages).
render "8113=qwen3.8-27b@262144,8142=sgl-live@163840+messages,8020=gguf-live"
eps() { route_field "$1" model_info.supported_endpoints; }
if [[ "$(eps sgl-live)" == *"/v1/messages"* && -z "$(eps qwen3.8-27b)" && -z "$(eps gguf-live)" ]]; then
  ok "only a route whose engine serves /v1/messages itself gets the passthrough"
else
  bad "supported_endpoints: sgl-live='$(eps sgl-live)' qwen3.8-27b='$(eps qwen3.8-27b)' gguf-live='$(eps gguf-live)'"
fi
# …and it must never touch supports_reasoning: /model_group/info publishes it, and
# pi-setup.sh turns thinking off on a false one.
[[ -z "$(route_field sgl-live model_info.supports_reasoning)" ]] \
  && ok "the passthrough leaves supports_reasoning alone" \
  || bad "passthrough route carries supports_reasoning=$(route_field sgl-live model_info.supports_reasoning)"

# --- 16: the REAL probe decides it, against stub servers (not the seam) ------------
# The property that matters is whether the endpoint keeps Claude Code's per-turn
# `system` message where it is: an endpoint that moves it to the front re-prefills
# the conversation every turn (vLLM v0.30.0 without --chat-template). The probe
# measures it with count_tokens: moved, the message counts exactly like the same text
# in the top-level system prompt; in place, it adds its own turn markers. Positive
# controls (SGLang and vLLM keeping it in place) and negatives (vLLM moving it, no
# endpoint, llama.cpp), so a probe that always says yes or always no both fail.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "probe: in-place SGLang/vLLM → passthrough; vLLM that moves the message, no count_tokens, llama.cpp → translated" || bad "probe decision wrong (see above)"
import http.server, json, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

def stub(owned_by, inline_system):   # inline_system: "in-place" | "moved" | "absent"
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/v1/models":
                self._send(200, {"data": [{"id": "m", "owned_by": owned_by, "max_model_len": 4096}]})
            else:
                self._send(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            if self.path != "/v1/messages/count_tokens" or inline_system == "absent":
                return self._send(404, {"error": "stub"})
            if body.get("model") != "m":
                return self._send(404, {"error": "unknown model"})
            # Both probe requests carry the same text; kept in place, the inline message
            # adds its own turn markers (4 tokens on Qwen), moved it adds none.
            inline = any(m["role"] == "system" for m in body.get("messages", []))
            self._send(200, {"input_tokens": 50 + (4 if inline and inline_system == "in-place" else 0)})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

cases = [("sglang", "in-place", True), ("vllm", "in-place", True), ("vllm", "moved", False),
         ("sglang", "absent", False), ("llamacpp", "in-place", False)]
bad = 0
for owned_by, behaviour, want in cases:
    srv = stub(owned_by, behaviour)
    got = litellm_sync.probe(srv.server_address[1])
    srv.shutdown()
    flag = got[0][3] if got else None
    if flag is not want:
        print(f"    owned_by={owned_by} inline system {behaviour}: passthrough={flag}, want {want}", file=sys.stderr)
        bad = 1
sys.exit(bad)
PY2

# --- 12: gateway settings survive every render, including a prune -----------
python3 - "$RUNTIME" <<'PY2' && ok "litellm_settings (request_timeout, num_retries: 0) carried into the runtime view" || bad "litellm_settings missing from the runtime view"
import io, sys, yaml
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
s = d.get("litellm_settings") or {}
sys.exit(0 if s.get("request_timeout") and s.get("num_retries") == 0 else 1)
PY2

# --- 13: no local file → nothing added, no error ------------------------------
C3_LITELLM_LOCAL_CONFIG="$LOCAL_CFG.absent" render "8182=alpha-model" \
  && ! command grep -q "THIS RIG'S OWN ROUTES" "$RUNTIME" \
  && ok "without config.local.yaml: no local block, clean render" || bad "a missing config.local.yaml must be a no-op"

# --- 14: the TRACKED catalog carries no route that leaves the machine -----------
# A cloud endpoint (and its workspace URL) is one rig's business: it belongs in the
# gitignored config.local.yaml. In the catalog it ships to every user, and omp's
# discovery then offers it — as the default model when modelRoles is empty.
ext="$(command grep -nE '^[[:space:]]*api_base:[[:space:]]*https?://' services/litellm/config.yaml | command grep -v 'host.docker.internal' || true)"
[[ -z "$ext" ]] && ok "the tracked catalog routes only to local engines" || bad "tracked config.yaml routes off-machine — move it to config.local.yaml: $ext"

# --- 9: the tracked catalog view is untouched by all of this -----------------
if git diff --quiet -- services/litellm/config.yaml; then
  ok "the tracked catalog view is unmodified"
else
  bad "litellm-sync modified the TRACKED config.yaml — it must only write the runtime view"
fi

[[ $fail -eq 0 ]] && echo "test-litellm-sync: ok" || echo "test-litellm-sync: FAIL"
exit $fail
