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
trap restore EXIT

# Deterministic: C3_LITELLM_FAKE_LIVE substitutes the socket probe, so the gate
# does not depend on whatever this rig happens to be serving.
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

# --- 2: the cloud block SURVIVES ---------------------------------------------
# A cloud endpoint is not "dead" because a local GPU is idle, and these back the
# benchlocal quality runs. If the prune ever eats them, quality runs break with
# no obvious cause.
[[ "$got" == *qwen3.8-max* ]] \
  && ok "cloud routes survive the prune" || bad "CLOUD ROUTES WERE PRUNED (got: $got)"

# --- 3: a registry-owned port that is NOT live gets pruned -------------------
render "8182=alpha-model"
got="$(routes)"
[[ "$got" != *beta-model* ]] \
  && ok "a route whose port stopped serving is pruned" || bad "dead route kept: $got"

# --- 4: NOTHING live → zero local routes, cloud intact -----------------------
render ""
got="$(routes)"
if [[ "$got" == *qwen3.8-max* ]] && ! command grep -q "host.docker.internal" "$RUNTIME"; then
  ok "with nothing serving: no local routes, cloud intact"
else
  bad "empty-rig render wrong" "cloud only" "$got"
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

# --- 12: gateway settings survive every render, including a prune -----------
python3 - "$RUNTIME" <<'PY2' && ok "litellm_settings (request_timeout, num_retries: 0) carried into the runtime view" || bad "litellm_settings missing from the runtime view"
import io, sys, yaml
d = yaml.safe_load(io.open(sys.argv[1], encoding="utf-8"))
s = d.get("litellm_settings") or {}
sys.exit(0 if s.get("request_timeout") and s.get("num_retries") == 0 else 1)
PY2

# --- 9: the tracked catalog view is untouched by all of this -----------------
if git diff --quiet -- services/litellm/config.yaml; then
  ok "the tracked catalog view is unmodified"
else
  bad "litellm-sync modified the TRACKED config.yaml — it must only write the runtime view"
fi

[[ $fail -eq 0 ]] && echo "test-litellm-sync: ok" || echo "test-litellm-sync: FAIL"
exit $fail
