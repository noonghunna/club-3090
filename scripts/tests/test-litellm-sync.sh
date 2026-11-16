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
# Isolated runtime path: the real services/litellm/config.runtime.yaml is
# watched by the club-live omp extension, which re-points every live agent
# session at whatever it sees — fixture states there hijack those sessions onto
# slugs the gateway never serves and their turns 400 (2026-11-05, twice). The
# suite renders through the C3_LITELLM_RUNTIME seam and never touches the real
# file, so there is nothing to back up or restore.
RUNTIME="$(mktemp -t c3-runtime-XXXXXX.yaml)"
export C3_LITELLM_RUNTIME="$RUNTIME"
trap 'rm -f "${LOCAL_CFG:-}" "${RUNTIME:-}"' EXIT

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
if [[ "$got" == *rig-cloud-model* ]] && ! command grep -q "host.docker.internal" "$RUNTIME" \
   && [[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "0" ]]; then
  ok "with nothing serving: no engine routes, this rig's own routes intact"
else
  bad "empty-rig render wrong" "this rig's own routes only" "$got"
fi

# --- 5: a route on a port we do NOT own is never touched ---------------------
# Someone's hand-added local service on an unrelated port is not ours to collect.
tmpl="services/litellm/config.yaml"; TMPL_BK="$(mktemp)"; cp "$tmpl" "$TMPL_BK"
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
cp "$TMPL_BK" "$tmpl"; rm -f "$TMPL_BK"
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
# Pinned to the checkout path: a temp file proves nothing about what git ignores.
git check-ignore -q services/litellm/config.runtime.yaml \
  && ok "runtime view is gitignored" || bad "services/litellm/config.runtime.yaml is NOT gitignored"

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

# --- 10b: the `live` alias route (the club-live omp alias, #1949) -------------
# omp's compaction and skill-compression dispatch bypass the extension's
# before_provider_request hook, so a request can reach the gateway still named
# `live`; it must resolve here or those requests 400. The alias mirrors the
# FIRST route (the one the extension names) and must never BE the first route:
# the extension's parser reads only the first route of the generated block.
[[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "1" ]] \
  && ok "a live alias route is rendered when something serves" \
  || bad "live alias route missing or duplicated: $(routes)"
# The mirror must track WHICHEVER route is first — the fixture's sort order is
# an accident; the invariant is "alias == the first route of the block".
first_gen=$(python3 -c "
import io, re
t = io.open('$RUNTIME', encoding='utf-8').read()
b = re.search(r'BEGIN GENERATED LOCAL BLOCK[\s\S]*?END GENERATED LOCAL BLOCK', t).group(0)
print(re.search(r'- model_name:\s*(\S+)', b).group(1))")
[[ "$first_gen" != "live" ]] \
  && ok "the live alias never shadows the first real route (first is $first_gen)" \
  || bad "the live alias is the first route — the extension's parser would name it"
[[ "$(route_field live litellm_params.api_base)" == "$(route_field "$first_gen" litellm_params.api_base)" \
   && "$(route_field live litellm_params.model)" == "openai/$first_gen" ]] \
  && ok "the live alias mirrors the first route's deployment (first is $first_gen)" \
  || bad "live alias: model=$(route_field live litellm_params.model) api_base=$(route_field live litellm_params.api_base) (want a mirror of $first_gen)"

# The count the sync reports counts DEPLOYMENTS; the alias is not one. Round-1
# adjudication: the alias mirrors a deployment, never adds one — these log
# lines are its only consumers. Pin it so a future edit can't silently drift.
rm -f "$RUNTIME"
cnt_out=$(C3_LITELLM_FAKE_LIVE="8113=qwen3.8-27b@262144,8142=sgl-live@163840,8020=gguf-live" bash "$SYNC" --no-restart 2>&1)
[[ "$cnt_out" == *"rendered 3 live route(s)"* ]] \
  && ok "the reported route count excludes the alias (3 real, not 4)" \
  || bad "count pin: $(printf '%s\n' "$cnt_out" | command grep 'live route(s)' | tail -1)"

# --- 10c: a real slug literally named `live` — no alias on top of it ----------
# Two routes named `live` are one LiteLLM load-balancing group: the
# hook-bypassed traffic the alias exists to save would land nondeterministically
# on the real slug or the mirrored route. The real route resolves `live` itself.
render "8020=live"
[[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "1" \
   && "$(route_field live litellm_params.api_base)" == "http://host.docker.internal:8020/v1" ]] \
  && ok "a real slug named live renders once, without the alias duplicating it" \
  || bad "slug named live: routes=$(routes) api_base=$(route_field live litellm_params.api_base)"

# --- 10f: `live` coexisting with other routes — suppression, not just sole ----
# The load-balance hazard is coexistence: one name, one deployment group. With
# `live` sorting SECOND, `first` is another route — the guard must still refuse
# to mirror it, or hook-bypassed `live` traffic lands on the real slug or the
# mirror nondeterministically.
render "8018=aaa,8020=live"
[[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "1" \
   && "$(route_field live litellm_params.api_base)" == "http://host.docker.internal:8020/v1" ]] \
  && ok "a real live beside other routes stays alone — no mirror of the first" \
  || bad "live coexisting: routes=$(routes) api_base=$(route_field live litellm_params.api_base)"

# --- 10d: the alias follows the first SURVIVING route -------------------------
# A saved key no container uses skips its route entirely (section 21). When the
# skipped route would have sorted first, the alias must mirror the survivor —
# the same route the extension's parser reads and names.
STRATA_API_KEY=other-saved-key-999 C3_LITELLM_FAKE_LIVE="8018=aaa-skipped+keyed=seam-key-1794,8020=bbb-live" bash "$SYNC" --no-restart --quiet 2>/dev/null
[[ "$(route_field live litellm_params.api_base)" == "$(route_field bbb-live litellm_params.api_base)" \
   && "$(route_field live litellm_params.model)" == "openai/bbb-live" ]] \
  && ok "with the first route key-skipped, the alias mirrors the survivor" \
  || bad "key-skip mirror: model=$(route_field live litellm_params.model) api_base=$(route_field live litellm_params.api_base)"

# --- 10e: every route key-skipped — marker present, no alias ------------------
# `first` stays None while the live list is non-empty: the block must say so
# (not render silent nothing) and must not alias a deployment that is not there.
STRATA_API_KEY=other-saved-key-999 C3_LITELLM_FAKE_LIVE="8018=aaa-skipped+keyed=seam-key-1794,8020=bbb-skipped+keyed=seam-key-1794" bash "$SYNC" --no-restart --quiet 2>/dev/null
if command grep -q "no local routes right now" "$RUNTIME" \
   && [[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "0" ]] \
   && ! command grep -q "host.docker.internal" "$RUNTIME"; then
  ok "all routes key-skipped: marker present, no alias, no engine routes"
else
  bad "all-key-skipped render: routes=$(routes)"
fi

# --- 10g: a config.local.yaml route named `live` suppresses the alias ---------
# The suppression must cover the user's own block too: one name across the
# merged config is one LiteLLM load-balancing group, and hook-bypassed `live`
# traffic (omp compaction) would split between the mirrored engine and the
# user's own endpoint. The user's route resolves the name on its own.
LOCAL_CFG_BK="$(mktemp)"; cp "$LOCAL_CFG" "$LOCAL_CFG_BK"
cat > "$LOCAL_CFG" <<'YAML'
model_list:
  # the user's own endpoint, named exactly like the omp alias
  - model_name: live
    litellm_params:
      model: openai/users-own-live
      api_base: https://cloud.example.invalid/v1
      api_key: os.environ/RIG_CLOUD_KEY
YAML
render "8020=real-slug"
cp "$LOCAL_CFG_BK" "$LOCAL_CFG"; rm -f "$LOCAL_CFG_BK"
if [[ "$(routes | tr ' ' '\n' | command grep -cx live)" == "1" ]] \
   && [[ "$(route_field live litellm_params.api_base)" == "https://cloud.example.invalid/v1" ]]; then
  ok "a config.local.yaml route named live suppresses the alias — the user's own endpoint resolves it"
else
  bad "cloud live collision: routes=$(routes) api_base=$(route_field live litellm_params.api_base)"
fi

# restore the section-10 fixture for every section below
render "8113=qwen3.8-27b@262144,8142=sgl-live@163840,8020=gguf-live"

# --- 11: model_info from the LIVE server (what omp's discovery: litellm reads) -
[[ "$(route_field qwen3.8-27b model_info.max_input_tokens)" == "262144" && "$(route_field sgl-live model_info.max_input_tokens)" == "163840" ]] \
  && ok "max_input_tokens comes from each server's own max_model_len" \
  || bad "max_input_tokens: qwen=$(route_field qwen3.8-27b model_info.max_input_tokens) sgl=$(route_field sgl-live model_info.max_input_tokens)"
# --- 11b: the advertised ceiling follows what the ENGINE declares -------------
# vLLM / SGLang / llama.cpp never send `max_output_tokens`, and they reject a
# request whose prompt + max_tokens exceeds max_model_len — so an unreported
# ceiling must stay conservative. QEngine sends an explicit null: it truncates at
# ctx - prompt instead of rejecting, and advertising 32768 there cut every long
# generation short downstream (a client asking for 262144 got 32768 back with
# finish_reason=length, so 256K of context was reachable but unusable).
[[ "$(route_field qwen3.8-27b model_info.max_output_tokens)" == "32768" ]] \
  && ok "an engine that reports no output cap keeps the 32768 default" \
  || bad "unreported outcap: $(route_field qwen3.8-27b model_info.max_output_tokens)"
render "8113=qwen3.8-27b@262144+outcap=none,8142=sgl-live@163840+outcap=65536,8020=gguf-live@262144"
[[ "$(route_field qwen3.8-27b model_info.max_output_tokens)" == "262144" ]] \
  && ok "an engine that declares no output cap (null) advertises the whole window" \
  || bad "outcap=none: $(route_field qwen3.8-27b model_info.max_output_tokens)"
[[ "$(route_field sgl-live model_info.max_output_tokens)" == "65536" ]] \
  && ok "an engine's declared output cap is advertised as-is" \
  || bad "outcap=65536: $(route_field sgl-live model_info.max_output_tokens)"
[[ "$(route_field gguf-live model_info.max_output_tokens)" == "32768" ]] \
  && ok "a larger window alone does not raise the ceiling (32768 unless declared)" \
  || bad "no declared outcap: $(route_field gguf-live model_info.max_output_tokens)"
[[ "$(route_field qwen3.8-27b model_info.supports_reasoning)" == "True" \
   && "$(route_field sgl-live model_info.supports_reasoning)" == "True" \
   && -z "$(route_field gguf-live model_info.supports_reasoning)" ]] \
  && ok "supports_reasoning comes from the model's effort block on every engine; a slug that declares nothing is still omitted, never false" \
  || bad "supports_reasoning: qwen=$(route_field qwen3.8-27b model_info.supports_reasoning) sgl='$(route_field sgl-live model_info.supports_reasoning)' gguf='$(route_field gguf-live model_info.supports_reasoning)'"

# --- 11c: the effort ladder the gateway advertises (#1951 phase 2) ----------
# This is what /model_group/info answers as `supported_reasoning_efforts` — the
# rungs a client may name. Three shapes, three rules, one from each #1951 layer:
# the MODEL's ladder, the ENGINE's handling of `none`, the COMPOSE's default.
render "8198=flash-live@262144,8119=glm-live@262144,8094=mimo-live@8192,8084=inkling-live@262144"
[[ "$(route_field flash-live model_info.reasoning_effort_levels)" == "['low', 'medium', 'xhigh']" ]] \
  && ok "a probed ladder is advertised as the rungs a client may name" \
  || bad "levels: $(route_field flash-live model_info.reasoning_effort_levels)"
[[ "$(route_field flash-live model_info.supports_none_reasoning_effort)" == "True" ]] \
  && ok "an engine that folds the none rung onto the ladder says so" \
  || bad "supports_none: $(route_field flash-live model_info.supports_none_reasoning_effort)"
[[ "$(route_field flash-live model_info.default_reasoning_effort)" == "low" ]] \
  && ok "the compose default is advertised as the server's own default" \
  || bad "default: $(route_field flash-live model_info.default_reasoning_effort)"
# litellm's `default_reasoning_effort` Literal (v1.100.3) EXCLUDES max while the
# engine's REASONING_EFFORT includes it: the rung is advertised, the default is
# OMITTED. Downgrading it to `high` would tell a client the server applies a
# choice it does not apply.
[[ "$(route_field glm-live model_info.reasoning_effort_levels)" == "['low', 'high', 'max']" \
   && -z "$(route_field glm-live model_info.default_reasoning_effort)" ]] \
  && ok "a max ladder is advertised whole and its max default is OMITTED, never downgraded" \
  || bad "max rung: levels=$(route_field glm-live model_info.reasoning_effort_levels) default='$(route_field glm-live model_info.default_reasoning_effort)'"
# Where the compose names no knob at all, the PAIR fact is what the server
# applies: the strata route leaves REASONING_EFFORT unset and its launcher runs
# at the template's xhigh, declared under the engine's `by_model` entry for THIS
# model. Before #1951 review r5 that fact sat on the engine block, where the two
# MiniMax and three DeepSeek slugs on the SAME engine (strata-maya) would have
# been defaulted GLM's way once they got a ladder.
render "8198=flash-live@262144,8119=glm-live@262144,8094=mimo-live@8192,8084=inkling-live@262144,8018=strata-live@262144"
[[ "$(route_field strata-live model_info.default_reasoning_effort)" == "xhigh" ]] \
  && ok "with no compose knob, the pair's own default is what gets advertised" \
  || bad "pair default: '$(route_field strata-live model_info.default_reasoning_effort)'"

# An honest `status:` instead of a ladder: the model DOES think (`unprobed` /
# `no-string-ladder` say the rungs are unknown, not absent), so the capability
# goes out and the ladder does not. Rendering nothing here would fall back to
# litellm's off-map `supports_reasoning: false` — the bug measured on this rig
# 2026-11-15, where a client turned thinking off on a thinking model.
for f in mimo-live inkling-live; do
  [[ "$(route_field "$f" model_info.supports_reasoning)" == "True" \
     && -z "$(route_field "$f" model_info.reasoning_effort_levels)" ]] \
    && ok "$f: an unprobed/non-string thinking model advertises the capability, not a ladder" \
    || bad "$f: reasoning=$(route_field "$f" model_info.supports_reasoning) levels='$(route_field "$f" model_info.reasoning_effort_levels)'"
done
# Agree-or-drop, the rule port_facts already uses for ctx and vision: a port
# hosting two ladders advertises NEITHER. litellm intersects a group's levels
# across its deployments anyway, so a partial fact here would advertise a rung
# the served model raises on. No real port disagrees today, so this is the one
# shape only a synthetic pair can pin.
python3 - scripts/lib/litellm_sync.py <<'PY' || fail=1
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ls", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
low = {"port": 1, "effort": {"levels": ["low", "high"], "compose_default": "low"}}
tall = {"port": 1, "effort": {"levels": ["low", "medium", "xhigh"], "compose_default": "low"}}
same = {"port": 1, "effort": {"levels": ["low", "high"], "compose_default": "low"}}
if m.port_facts([low, tall]).get(1, {}).get("effort"):
    print("  FAIL — a port hosting two different ladders still advertised one", file=sys.stderr); sys.exit(1)
if not m.port_facts([low, same]).get(1, {}).get("effort"):
    print("  FAIL — two slugs agreeing on a ladder dropped the fact", file=sys.stderr); sys.exit(1)
if m.port_facts([{"port": 1}]).get(1, {}).get("effort"):
    print("  FAIL — a slug with no effort block invented a fact", file=sys.stderr); sys.exit(1)
# The pair default is a fact like any other: agree-or-drop across the port, and
# it fills the advertise only where the compose names no knob.
pair_x = {"port": 2, "effort": {"levels": ["low", "xhigh"], "default_when_unset": "xhigh"}}
pair_m = {"port": 2, "effort": {"levels": ["low", "xhigh"], "default_when_unset": "max"}}
if (m.port_facts([pair_x, pair_m]).get(2, {}).get("effort") or {}).get("default_when_unset"):
    print("  FAIL — a port with two different pair defaults still advertised one", file=sys.stderr); sys.exit(1)
one = m.port_facts([pair_x]).get(2, {}).get("effort")
if not one or one.get("default_when_unset") != "xhigh":
    print(f"  FAIL — a single pair default did not survive port_facts: {one}", file=sys.stderr); sys.exit(1)
if "default_reasoning_effort: xhigh" not in "\n".join(m.model_info(262144, {"effort": one})):
    print("  FAIL — the pair default never reached the advertise", file=sys.stderr); sys.exit(1)
max_only = m.port_facts([pair_m]).get(2, {}).get("effort")
if "default_reasoning_effort" in "\n".join(m.model_info(262144, {"effort": max_only})):
    print("  FAIL — a max pair default was advertised instead of omitted", file=sys.stderr); sys.exit(1)
# One compose sets the knob, the sibling's does not: the pair default must NOT
# fill the port — the route could boot either compose, and the slug whose
# compose sets low would be lied to (#1951 review r6).
mixed = {"port": 3, "effort": {"levels": ["low", "xhigh"], "compose_default": "low",
                               "default_when_unset": "xhigh"}}
bare = {"port": 3, "effort": {"levels": ["low", "xhigh"], "default_when_unset": "xhigh"}}
got = (m.port_facts([mixed, bare]).get(3, {}).get("effort") or {})
if got.get("default_when_unset") or got.get("compose_default"):
    print(f"  FAIL — a compose knob beside a bare sibling let the pair default "
          f"fill the port: {got}", file=sys.stderr); sys.exit(1)
if not m.port_facts([bare]).get(3, {}).get("effort", {}).get("default_when_unset"):
    print("  FAIL — a bare compose lost the pair default that should fill it",
          file=sys.stderr); sys.exit(1)
print("  ok   — disagreeing ladders or pair defaults advertise nothing; agreeing ones "
      "still do, and a pair default fills an advertise with no compose knob anywhere "
      "on the port")
PY

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
# pi-setup.sh turns thinking off on a false one. "Absent" was the old shape of
# that property (this route declared no thinking profile); with the #1951 effort
# block the route DOES carry one, so the property is now "the passthrough does
# not change it" — same port, rendered with and without +messages.
render "8142=sgl-live@163840"
plain="$(route_field sgl-live model_info.supports_reasoning)"
render "8142=sgl-live@163840+messages"
passthru="$(route_field sgl-live model_info.supports_reasoning)"
[[ -n "$plain" && "$plain" == "$passthru" ]] \
  && ok "the passthrough leaves supports_reasoning alone" \
  || bad "passthrough changed supports_reasoning: $plain -> $passthru"

# --- 16: the REAL probe decides it, against stub servers (not the seam) ------------
# Two properties, both measured on THIS boot, decide the /v1/messages passthrough:
#   - the endpoint answers with a real message. A tiny generation is the only
#     honest test: an empty-body 400-vs-404 probe cannot tell a working endpoint
#     from one that answers with empty content, and empty content is exactly the
#     failure this avoids;
#   - it keeps Claude Code's per-turn `system` message where it is: an endpoint
#     that moves it to the front re-prefills the conversation every turn (vLLM
#     v0.30.0 without --chat-template). count_tokens measures that without
#     generating: moved, the message counts exactly like the same text in the
#     top-level system prompt; in place, it adds its own turn markers. An
#     endpoint without count_tokens (Project Maya 404s it) gets the same
#     question answered by its prompt cache: a follow-up that only APPENDS the
#     message must hit the cache on the front it did not touch.
# The engine's NAME decides nothing: the old allowlist (SGLang, vLLM) silently
# cost QEngine its whole reply and Strata its thinking, and Strata reports no
# owned_by at all. Positives and negatives on every leg, so a probe that always
# says yes or always no both fail.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "probe: serves-itself + in-place → passthrough, owned_by or not; moved, empty content, no endpoint, rewritten front → translated" || bad "probe decision wrong (see above)"
import http.server, json, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

def stub(owned_by, inline_system, serves="real", cache=None):
    """inline_system: "in-place" | "moved" | "absent" (count_tokens behaviour).
    serves: "real" (a message with content) | "empty" (content: []) | "absent" (404).
    cache: (front, cached) — no count_tokens; the cache probe decides."""
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/v1/models":
                data = [{"id": "m", "max_model_len": 4096}]
                if owned_by:
                    data[0]["owned_by"] = owned_by
                self._send(200, {"data": data})
            else:
                self._send(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            if body.get("model") != "m":
                return self._send(404, {"error": "unknown model"})
            if self.path == "/v1/messages":
                if serves == "absent":
                    return self._send(404, {"error": "stub"})
                if serves == "empty":
                    return self._send(200, {"type": "message", "role": "assistant", "content": []})
                if serves == "think":
                    # no text at ANY budget: an endpoint Claude Code cannot use
                    return self._send(200, {"type": "message", "role": "assistant",
                                            "content": [{"type": "thinking", "thinking": "hmm"}]})
                if serves == "budget" and body.get("max_tokens", 8) <= 8:
                    # the 8-token probe: thinking ate the whole budget; the
                    # retry at a real budget must find the text
                    return self._send(200, {"type": "message", "role": "assistant",
                                            "content": [{"type": "thinking", "thinking": "hmm"}]})
                if cache and len(body.get("messages", [])) > 2:
                    # the follow-up: the cache answers the placement question
                    front, cached = cache
                    return self._send(200, {"type": "message", "role": "assistant",
                                            "content": [{"type": "text", "text": "DONE"}],
                                            "usage": {"input_tokens": 14, "cache_read_input_tokens": cached}})
                if cache:
                    front, cached = cache
                    return self._send(200, {"type": "message", "role": "assistant",
                                            "content": [{"type": "text", "text": "OK"}],
                                            "usage": {"input_tokens": front}})
                return self._send(200, {"type": "message", "role": "assistant",
                                        "content": [{"type": "text", "text": "OK"}]})
            if self.path != "/v1/messages/count_tokens" or inline_system == "absent":
                return self._send(404, {"error": "stub"})
            # Both probe requests carry the same text; kept in place, the inline message
            # adds its own turn markers (4 tokens on Qwen), moved it adds none.
            inline = any(m["role"] == "system" for m in body.get("messages", []))
            self._send(200, {"input_tokens": 50 + (4 if inline and inline_system == "in-place" else 0)})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

cases = [  # (owned_by, count_tokens, serves, cache, want)
    ("sglang", "in-place", "real", None, True),
    ("vllm", "in-place", "real", None, True),
    ("vllm", "moved", "real", None, False),
    ("", "in-place", "real", None, True),          # Strata: no owned_by, still passthrough
    ("", "absent", "real", (1562, 1561), True),    # Maya: no count_tokens, cache says in place
    ("", "absent", "real", (1562, 3), False),      # Maya-shaped but the front was rewritten
    ("", "in-place", "budget", None, True),        # thinking ate 8 tokens; retry finds text
    ("", "in-place", "think", None, False),        # no text at any budget: translated
    ("vllm", "in-place", "empty", None, False),    # answers, but with no content
    ("llamacpp", "in-place", "absent", None, False),
]
bad = 0
for owned_by, behaviour, serves, cache, want in cases:
    srv = stub(owned_by, behaviour, serves, cache)
    got = litellm_sync.probe(srv.server_address[1])
    srv.shutdown()
    flag = got[0][3] if got else None
    if flag is not want:
        print(f"    owned_by={owned_by!r} count_tokens={behaviour} serves={serves} cache={cache}: "
              f"passthrough={flag}, want {want}", file=sys.stderr)
        bad = 1
sys.exit(bad)
PY2

# --- 16b: the emit's default regex and the guard's are ONE shape ---------------
# Two copies of the REASONING_EFFORT default regex exist (registry-emit.sh reads
# the compose for the advertise; test-effort-profiles.sh check 3 pins every
# compose against the same shape). They must stay identical: a form one reads
# and the other does not is a default the advertise and the guard disagree on.
python3 - <<'PY' && ok "the emit's REASONING_EFFORT default regex is byte-identical to the guard's" || bad "the two default regexes drifted (see above)"
import re
RAW = r"r'((?:[^'\\]|\\.)*)'"          # a raw string literal, escapes included
def pattern(path, marker):
    src = open(path, encoding="utf-8").read()
    m = re.search(marker + r"[^\n]*\n[ \t]+" + RAW, src)
    return m.group(1) if m else None
emit = pattern("scripts/lib/registry-emit.sh", r"_EFFORT_DEFAULT_RX = _re\.compile\(")
guard = pattern("scripts/tests/test-effort-profiles.sh", r"default_re = re\.compile\(")
assert emit and guard, f"could not extract a pattern: emit={emit!r} guard={guard!r}"
assert emit == guard, f"drift:\n  emit  = {emit!r}\n  guard = {guard!r}"
PY

# --- 17: an engine without /v1/responses gets the chat-completions bridge (#1520)
# omp talks Responses to every route and LiteLLM turns Claude Code's /v1/messages
# into a Responses call; both are forwarded to the engine's /v1/responses as-is.
# tabbyAPI 404s it, and the 404 cools the whole model group down for 5 s.
# `use_chat_completions_api` sends those requests as chat completions instead —
# only on routes the probe marked (+noresponses), never on an engine that has it.
render "8113=qwen3.8-27b@262144,8142=sgl-live@163840+messages,8181=exl3-live+noresponses,8020=gguf-live+messages+noresponses"
cc() { route_field "$1" litellm_params.use_chat_completions_api; }
if [[ "$(cc exl3-live)" == "True" && "$(cc gguf-live)" == "True" && -z "$(cc qwen3.8-27b)" && -z "$(cc sgl-live)" ]]; then
  ok "only a route whose engine lacks /v1/responses gets the chat-completions bridge"
else
  bad "use_chat_completions_api: exl3-live='$(cc exl3-live)' gguf-live='$(cc gguf-live)' qwen3.8-27b='$(cc qwen3.8-27b)' sgl-live='$(cc sgl-live)'"
fi
# flags combine: `+messages+noresponses` keeps the /v1/messages passthrough too
[[ "$(eps gguf-live)" == *"/v1/messages"* && -z "$(eps exl3-live)" ]] \
  && ok "the bridge and the /v1/messages passthrough are independent flags" \
  || bad "flag parsing: gguf-live eps='$(eps gguf-live)' exl3-live eps='$(eps exl3-live)'"
[[ "$(route_field exl3-live litellm_params.allowed_openai_params)" == "['reasoning_effort']" ]] \
  && ok "a bridged route keeps the common route shape" \
  || bad "bridged route lost allowed_openai_params: $(route_field exl3-live litellm_params.allowed_openai_params)"

# --- 18: the REAL Responses probe, against stub servers (not the seam) ----------
# An empty-body POST: an engine with the endpoint rejects the body (SGLang 400,
# FastAPI 422), one without it 404s the path (tabbyAPI). Only a 404 means absent;
# any other answer — or none — keeps today's route, so a flaky probe can never
# move a native engine onto the bridge.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "Responses probe: 404 → bridge; 400/405/422/500/no answer → native" || bad "Responses probe decision wrong (see above)"
import http.server, json, socket, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

def stub(owned_by, responses_status):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": "m", "owned_by": owned_by}]})
            self._send(404, {"detail": "Not Found"})
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            if self.path == "/v1/responses":
                return self._send(responses_status, {"detail": "stub"})
            self._send(404, {"detail": "Not Found"})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

bad = 0
for owned_by, status, want in [("tabbyAPI", 404, False), ("sglang", 400, True), ("vllm", 422, True),
                               ("llamacpp", 405, True), ("unknown", 500, True)]:
    srv = stub(owned_by, status)
    got = litellm_sync.probe(srv.server_address[1])
    srv.shutdown()
    flag = got[0][4] if got else None
    if flag is not want:
        print(f"    owned_by={owned_by} /v1/responses→{status}: serves_responses={flag}, want {want}", file=sys.stderr)
        bad = 1
with socket.socket() as s:          # a port nothing listens on
    s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]
if litellm_sync.serves_responses(dead) is not True:
    print("    no answer: serves_responses must stay True (today's route)", file=sys.stderr)
    bad = 1
sys.exit(bad)
PY2

# --- 18b: the REAL probe reads the engine's declared output ceiling (#1911) -----
# Section 11b drives the ceiling through the seam; this is the parse it stands
# for. QEngine's /v1/models card says `max_output_tokens: null` ("no cap") and the
# whole window becomes the advertised ceiling. A card WITHOUT the field (vLLM,
# SGLang, llama.cpp) must stay "never said", or those engines would advertise
# replies they reject.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "output-ceiling probe: null → no cap, a number → that number, absent/0 → never said" || bad "output-ceiling probe decision wrong (see above)"
import http.server, json, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

def stub(card):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [card]})
            self._send(404, {"detail": "Not Found"})
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self._send(422, {"detail": "stub"})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

# the card QEngine 1.3.4.2 serves (server/app.py limits()), trimmed to the fields that matter
qe = {"id": "qwen3.8-27b", "object": "model", "owned_by": "qengine", "max_model_len": 262144,
      "max_num_seqs": 32, "max_output_tokens": None}
vllm = {"id": "qwen3.8-27b", "object": "model", "owned_by": "vllm", "max_model_len": 262144}
bad = 0
for name, card, want_cap, want_adv in [("qengine null", qe, -1, 262144),
                                       ("field absent", vllm, None, 32768),
                                       ("declared 65536", dict(qe, max_output_tokens=65536), 65536, 65536),
                                       ("declared 0", dict(qe, max_output_tokens=0), None, 32768)]:
    srv = stub(card)
    got = litellm_sync.probe(srv.server_address[1])
    srv.shutdown()
    if not got:
        print(f"    {name}: no route", file=sys.stderr); bad = 1; continue
    r = got[0]
    adv = litellm_sync.output_ceiling(r.ctx, r.out_cap)
    if r.out_cap != want_cap or adv != want_adv:
        print(f"    {name}: out_cap={r.out_cap!r} advertised={adv}, want {want_cap!r} / {want_adv}", file=sys.stderr)
        bad = 1
sys.exit(bad)
PY2

# --- 19: tabbyAPI route names come from /v1/model, not the directory listing ----
# With auth disabled every caller is admin, and an admin GET /v1/models lists every
# directory in --model-dir (other engines' weights, .cache). Each became a route, and
# a request naming a wrong one silently ran the loaded model. /v1/model (singular)
# names only the loaded model. A tabbyAPI without it falls back to /v1/models, and
# other engines never ask for it.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "tabbyAPI: one route from /v1/model; /v1/model missing → /v1/models; other engines untouched" || bad "tabbyAPI route naming wrong (see above)"
import http.server, json, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

LISTING = ["loaded", ".cache", "modules"]

def stub(owned_by, singular, hits):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            hits.append(self.path)
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": i, "owned_by": owned_by} for i in LISTING]})
            if self.path == "/v1/model" and singular is not None:
                return self._send(200, {"id": singular, "owned_by": owned_by})
            self._send(404, {"detail": "Not Found"})
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self._send(404, {"detail": "Not Found"})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv

bad = 0
for label, owned_by, singular, want, want_singular_asked in [
    ("tabbyAPI with /v1/model", "tabbyAPI", "loaded", ["loaded"], True),
    ("tabbyAPI without /v1/model", "tabbyAPI", None, LISTING, True),
    ("vllm", "vllm", "loaded", LISTING, False),
    ("llamacpp", "llamacpp", "loaded", LISTING, False),
]:
    hits = []
    srv = stub(owned_by, singular, hits)
    got = [r[1] for r in litellm_sync.probe(srv.server_address[1])]
    srv.shutdown()
    if got != want or ("/v1/model" in hits) is not want_singular_asked:
        print(f"    {label}: routes={got} (want {want}); asked /v1/model={'/v1/model' in hits} (want {want_singular_asked})", file=sys.stderr)
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

# --- 13b: a render under the test seam never restarts the gateway ------------
# A test's docker shim is not enough on its own: switch.sh --no-wait starts a
# detached post-ready-sync.sh that outlives the test, and once the shim's dir is
# gone its `docker restart` reaches the real daemon. The seam is inherited, so
# the sync itself must refuse. A docker stub records the calls here; the
# positive control (no seam, same stub) shows the stub does see a restart.
stub13b="$(mktemp -d)"; calls13b="$stub13b/calls"
cat > "$stub13b/docker" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$(dirname "$0")/calls"
case "$1" in
  ps) echo litellm ;;
  *)  exit 1 ;;   # inspect: no mount to follow, so the render stays on C3_LITELLM_RUNTIME
esac
STUB
chmod +x "$stub13b/docker"
echo "stale: not a render" > "$RUNTIME"
out13b="$(PATH="$stub13b:$PATH" C3_LITELLM_FAKE_LIVE="8182=alpha-model" bash "$SYNC" 2>&1)"
if ! command grep -q '^restart' "$calls13b" 2>/dev/null && [[ "$out13b" == *"not restarting litellm"* ]]; then
  ok "a changed render under C3_LITELLM_FAKE_LIVE does not restart litellm"
else
  bad "the test seam restarted the gateway (calls: $(tr '\n' ';' < "$calls13b" 2>/dev/null); out: $out13b)"
fi
: > "$calls13b"; echo "stale: not a render" > "$RUNTIME"
env -u C3_LITELLM_FAKE_LIVE PATH="$stub13b:$PATH" bash "$SYNC" --quiet >/dev/null 2>&1
if command grep -qx 'restart litellm' "$calls13b"; then
  ok "positive control: the same changed render without the seam restarts litellm"
else
  bad "positive control: no restart without the seam, so the check above proves nothing (calls: $(tr '\n' ';' < "$calls13b"))"
fi
rm -rf "$stub13b"

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

# --- 20: a keyed engine (Strata 0.2+) — the probe retries with the key ------
# Strata 0.2+ answers nothing but GET /health without a key, so a keyless probe
# 401s and the port rendered no route at all: the model was invisible to the
# gateway and to omp's discovery (club-3090#1794). The 401 triggers a lazy key
# resolution (engine_key_for_port — the repo's own helpers behind the
# C3_LITELLM_FAKE_ENGINE_KEYS seam here) and one retry with it as a Bearer
# header; the key also rides on the /v1/responses probe. A keyless engine is
# probed exactly as before: one keyless request.
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "keyed probe: seam key → the model list; wrong key → no route; no key → no route" || bad "keyed probe wrong (see above)"
import http.server, json, os, sys, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

KEY = "seam-key-1794"

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _authed(self):
        return self.headers.get("authorization") == f"Bearer {KEY}"
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/v1/models":
            if not self._authed():
                return self._send(401, {"error": {"type": "authentication_error",
                                                   "message": "missing or wrong API key"}})
            self._send(200, {"data": [{"id": "keyed-model", "owned_by": "strata", "max_model_len": 4096}]})
        else:
            self._send(404, {})
    def do_POST(self):
        # the server answers nothing without its key — /v1/responses included.
        # With the key it 404s: no Responses endpoint. Only a keyed probe can
        # see the 404, so the responses flag proves the key was threaded.
        if not self._authed():
            return self._send(401, {"error": {"type": "authentication_error",
                                               "message": "missing or wrong API key"}})
        self._send(404, {})

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
bad = 0
os.environ["C3_LITELLM_FAKE_ENGINE_KEYS"] = f"{port}={KEY}"
r1 = litellm_sync.probe(port)
if not (len(r1) == 1 and r1[0][1] == "keyed-model" and r1[0][5] == KEY and r1[0][4] is False):
    print(f"    seam key: expected one route with the key and responses=False, got {r1}", file=sys.stderr); bad = 1
os.environ["C3_LITELLM_FAKE_ENGINE_KEYS"] = f"{port}=wrong-key"
r2 = litellm_sync.probe(port)
if r2:
    print(f"    wrong key: expected no route, got {r2}", file=sys.stderr); bad = 1
os.environ["C3_LITELLM_FAKE_ENGINE_KEYS"] = ""
r0 = litellm_sync.probe(port)
if r0:
    print(f"    no key for this port: expected no route, got {r0}", file=sys.stderr); bad = 1
del os.environ["C3_LITELLM_FAKE_ENGINE_KEYS"]
srv.shutdown()
sys.exit(bad)
PY2

# The same retry through the REAL container lookup — the seam above
# short-circuits before the helpers, so it cannot cover this leg: a docker
# stub on PATH (as in test-engine-auth.sh) makes club_container_for_url find
# a container and club_engine_api_key read its API_KEY. ENGINE_API_KEY —
# engine-auth.sh's override for a remote URL=, which switch.sh loads from
# saved settings into the sync's environment — must NOT reach the probe:
# the probe only ever asks 127.0.0.1, where the container decides.
python3 - "$ROOT/scripts/lib" <<'PY3' && ok "real lookup: the container's key answers the 401, never a saved ENGINE_API_KEY" || bad "ENGINE_API_KEY reached the probe (see above)"
import http.server, json, os, shutil, sys, tempfile, threading
sys.path.insert(0, sys.argv[1])
import litellm_sync

KEY = "container-key-1794b"
DUMMY = "remote-dummy-1794-nope"
seen_auth = []

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _authed(self):
        seen_auth.append(self.headers.get("authorization") or "")
        return self.headers.get("authorization") == f"Bearer {KEY}"
    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path == "/v1/models":
            if not self._authed():
                return self._send(401, {"error": {"type": "authentication_error",
                                                   "message": "missing or wrong API key"}})
            self._send(200, {"data": [{"id": "keyed-model", "owned_by": "strata", "max_model_len": 4096}]})
        self._send(404, {})
    def do_POST(self):
        if not self._authed():
            return self._send(401, {"error": {"type": "authentication_error",
                                               "message": "missing or wrong API key"}})
        self._send(404, {})

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]
bad = 0

# The docker stub: `ps` publishes this leg's ephemeral port on an engine port
# (->8000/tcp, CLUB_ENGINE_PORTS_ANY), so ports_serve_url passes — the stub
# server really listens on 127.0.0.1:port. `inspect` answers a strata image
# (the same one test-engine-auth.sh uses) and the container's API_KEY.
bin = tempfile.mkdtemp()
stub = """#!/usr/bin/env bash
case "$1" in
  ps) printf 'strata-probe-stub|0.0.0.0:%s->8000/tcp\\n' "$CLUB_TEST_PORT" ;;
  inspect)
    [[ "$2" == -f ]] || exit 1
    case "$3" in
      '{{.Config.Image}}') echo 'ghcr.io/niko1221/strata:0.2.1-cuda13' ;;
      *Config.Env*) printf 'PATH=/usr/bin\\nAPI_KEY=%s\\n' "$CLUB_TEST_KEY" ;;
      *) exit 1 ;;
    esac ;;
  *) exit 1 ;;
esac
"""
with open(os.path.join(bin, "docker"), "w", encoding="utf-8") as fh:
    fh.write(stub)
os.chmod(os.path.join(bin, "docker"), 0o755)
old_path = os.environ["PATH"]
os.environ["PATH"] = bin + os.pathsep + old_path
os.environ["CLUB_TEST_PORT"] = str(port)
os.environ["CLUB_TEST_KEY"] = KEY
os.environ["ENGINE_API_KEY"] = DUMMY
os.environ.pop("C3_LITELLM_FAKE_ENGINE_KEYS", None)
r = litellm_sync.probe(port)
if not (len(r) == 1 and r[0][1] == "keyed-model" and r[0][5] == KEY):
    print(f"    real lookup: expected one route keyed with the container's API_KEY, got {r}", file=sys.stderr); bad = 1
if any(DUMMY in a for a in seen_auth):
    print("    the saved ENGINE_API_KEY reached the server", file=sys.stderr); bad = 1
os.environ["PATH"] = old_path
for k in ("ENGINE_API_KEY", "CLUB_TEST_PORT", "CLUB_TEST_KEY"):
    os.environ.pop(k, None)
shutil.rmtree(bin, ignore_errors=True)
srv.shutdown()
sys.exit(bad)
PY3

# --- 21: a keyed route's api_key — never a user's literal key ---------------
# The runtime file is world-readable like the rest of the checkout; a user's
# key lives in 0600 secrets.env. So: the stored setting → an os.environ
# reference; the public default every strata compose ships → literal (it is
# published in the repo); any other unstored key → the reference plus a
# warning telling the user to save it.
tmp21="$(mktemp)"
env -u STRATA_API_KEY C3_LITELLM_FAKE_LIVE="8018=strata-model+keyed=seam-key-1794" bash "$SYNC" --no-restart --quiet 2> "$tmp21"
ak="$(route_field strata-model litellm_params.api_key)"
if [[ "$ak" == "os.environ/STRATA_API_KEY" ]] && command grep -q "not saved in your settings" "$tmp21"; then
  ok "an unstored non-default key: the route references the setting and the sync warns to save it"
else
  bad "unstored key: api_key=$ak (want os.environ/STRATA_API_KEY + a warning); stderr: $(cat "$tmp21")"
fi
env -u STRATA_API_KEY C3_LITELLM_FAKE_LIVE="8018=strata-model+keyed=club-3090-strata-local-key" bash "$SYNC" --no-restart --quiet
ak="$(route_field strata-model litellm_params.api_key)"
if [[ "$ak" == "club-3090-strata-local-key" ]]; then
  ok "the public default key renders literally — it is published in the repo"
else
  bad "public default: api_key=$ak (want the literal default)"
fi
STRATA_API_KEY=seam-key-1794 render "8018=strata-model+keyed=seam-key-1794"
ak="$(route_field strata-model litellm_params.api_key)"
if [[ "$ak" == "os.environ/STRATA_API_KEY" ]]; then
  ok "the stored key is the container's: the route references the setting, not the value"
else
  bad "stored key: api_key=$ak (want os.environ/STRATA_API_KEY)"
fi
# a SAVED key that is NOT the container's: NO route at all — the reference
# would hand the user's secret to a server that does not use it, and an
# EMPTY route would advertise a model that 401s every request.
tmp21b="$(mktemp)"
STRATA_API_KEY=other-saved-key-999 C3_LITELLM_FAKE_LIVE="8018=strata-model+keyed=seam-key-1794" bash "$SYNC" --no-restart --quiet 2> "$tmp21b"
ak="$(route_field strata-model litellm_params.api_key)"
if [[ -z "$ak" ]] && command grep -q "not the saved STRATA_API_KEY" "$tmp21b"; then
  ok "a saved key that differs from the container's: no route, never the wrong secret"
else
  bad "mismatched saved key: api_key='$ak' (want no route + a warning); stderr: $(cat "$tmp21b")"
fi
rm -f "$tmp21b"
rm -f "$tmp21"
# a keyless route keeps EMPTY, as before
render "8182=alpha-model"
ak="$(route_field alpha-model litellm_params.api_key)"
[[ "$ak" == "EMPTY" ]] && ok "a keyless route keeps api_key: EMPTY" || bad "keyless route api_key=$ak (want EMPTY)"

# --- 22: the public default key — the constant must match every strata compose ---
# litellm_sync.py renders the public default as a literal (it is published in the
# repo); a user's own key never does. If a compose's default ever changes, the
# constant must change with it — otherwise the new default renders as "an
# unstored key" (a reference + a save-it warning) instead of the literal, and
# every strata user ends up with a broken route until they save the key.
pykey="$(command grep -oE 'PUBLIC_DEFAULT_ENGINE_KEY = "[^"]+"' scripts/lib/litellm_sync.py | command sed -n 's/.*"\(.*\)"/\1/p' | head -1)"
if [[ -z "$pykey" ]]; then
  bad "PUBLIC_DEFAULT_ENGINE_KEY not found in litellm_sync.py"
else
  n=0; drift=""
  while IFS= read -r l; do
    f="${l%%:*}"; l="${l#*:}"
    dflt="$(command sed -n 's/.*STRATA_API_KEY:-\([^}]*\).*/\1/p' <<<"$l" | head -1)"
    if [[ -z "$dflt" ]]; then
      continue
    fi
    n=$((n+1))
    if [[ "$dflt" != "$pykey" ]]; then
      drift+="  ${f}: ${dflt}"
    fi
  done < <(command grep -rn 'API_KEY: "${STRATA_API_KEY:-' models/*/strata/compose/ 2>/dev/null)
  if [[ $n -eq 0 ]]; then
    bad "no strata compose declares API_KEY — the guard found nothing to check"
  elif [[ -z "$drift" ]]; then
    ok "the public default key matches all $n strata compose defaults"
  else
    bad "a strata compose's default key drifted from the constant:$drift"
  fi
fi

# --- #1945: a mounted config that isn't ours is never rewritten -----------------
# The gateway may mount the user's own file at /app/config.yaml. Rendering over it
# replaced every route the template lacked (16 → 2, a gateway serving 400s for hours).
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "#1945: the mount is followed only to a runtime view; a user's own config is left untouched" || bad "#1945: runtime-file choice wrong (see above)"
import os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import litellm_sync as ls
d = tempfile.mkdtemp()
checkout = os.path.join(d, "config.runtime.yaml"); open(checkout, "w").write("x\n")
ours = os.path.join(d, "other-checkout.yaml")
open(ours, "w").write("model_list:\n" + ls.BEGIN.rstrip() + " — " + ls.RUNTIME_MARKER + " ===\n")
mine = os.path.join(d, "my-own.yaml"); open(mine, "w").write("model_list:\n  - model_name: qwen3.8-max\n")
errs = []
def want(cond, msg):
    if not cond: errs.append(msg)
want(ls.choose_runtime(checkout, None) == (checkout, ""), "no mount -> the checkout's runtime view")
want(ls.choose_runtime(checkout, checkout) == (checkout, ""), "mount == checkout file -> that file")
want(ls.choose_runtime(checkout, ours) == (ours, ""), "another runtime view (worktree case) -> follow it")
path, why = ls.choose_runtime(checkout, mine)
want(path is None, "a mounted file without the marker must not be chosen (#1945)")
want("config.local.yaml" in why and "left untouched" in why, f"the warning must say what to do, got: {why!r}")
path, _ = ls.choose_runtime(checkout, d)            # a directory (docker made one): not ours either
want(path is None, "a mount that isn't a readable file must not be chosen")
for e in errs: print("   ", e, file=sys.stderr)
sys.exit(1 if errs else 0)
PY2

# --- #1945: a port that answers on loopback only gets no route ---------------
# The probe asks 127.0.0.1, the gateway asks host.docker.internal (the bridge). Fixture
# /proc/net/tcp{,6}: 8092 on 127.0.0.1 only, 9000 on ::1 only, 8113 on 0.0.0.0, 8094 on
# 127.0.0.1 AND :: (reachable), 8095 not listening at all (unknown, kept).
python3 - "$ROOT/scripts/lib" <<'PY2' && ok "#1945: loopback-only ports (IPv4 + IPv6) are found; wildcard, mixed and unknown are kept" || bad "#1945: loopback-only detection wrong (see above)"
import os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import litellm_sync as ls
os.environ.pop("C3_LITELLM_FAKE_LIVE", None)
d = tempfile.mkdtemp()
hdr = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
row = lambda addr, port, st="0A": f"   0: {addr}:{port:04X} 00000000:0000 {st} 00000000:00000000 00:00000000 00000000  1000        0 1\n"
open(os.path.join(d, "tcp"), "w").write(hdr + row("0100007F", 8092) + row("00000000", 8113) + row("0100007F", 8094)
                                        + row("0100007F", 8096, st="01"))   # 8096: ESTABLISHED, not LISTEN
open(os.path.join(d, "tcp6"), "w").write(hdr + row("00000000000000000000000001000000", 9000)
                                         + row("00000000000000000000000000000000", 8094))
got = ls.loopback_only_ports([8092, 8113, 8094, 8095, 8096, 9000],
                             (os.path.join(d, "tcp"), os.path.join(d, "tcp6")))
if got != {8092, 9000}:
    print(f"    want {{8092, 9000}}, got {sorted(got)}", file=sys.stderr); sys.exit(1)
PY2
errf="$(mktemp)"
C3_LITELLM_FAKE_LIVE="8182=alpha-model,8091=beta-model" C3_LITELLM_FAKE_LOOPBACK="8091" \
  bash "$SYNC" --no-restart --quiet 2>"$errf"
got="$(routes)"
if [[ "$got" == *alpha-model* && "$got" != *beta-model* ]] && command grep -q ':8091 answers on 127.0.0.1 only' "$errf"; then
  ok "#1945: a loopback-only live port gets no route, with a warning that names it"
else
  bad "#1945: loopback-only port still routed or not warned (routes: $got; stderr: $(tr '\n' ' ' < "$errf"))"
fi
rm -f "$errf"
out="$(C3_LITELLM_FAKE_LIVE="8182=gamma-model" bash "$SYNC" --no-restart 2>/dev/null)"
# The log line is for a person, so the sync prints the file through
# litellm_local.display_path: relative inside the checkout, ~/… under $HOME. A literal
# "-> $RUNTIME" match failed whenever TMPDIR sat under $HOME. Expect the written file
# in the form the sync shows it; the basename check keeps an empty answer from passing.
shown="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import litellm_local
print(litellm_local.display_path(sys.argv[2], sys.argv[3]))' "$ROOT/scripts/lib" "$RUNTIME" "$ROOT")" || shown=""
if [[ -n "$shown" && "$shown" == *"$(basename "$RUNTIME")" && "$out" == *"-> $shown"* ]]; then
  ok "#1945: the render log names the file it wrote"
else
  bad "#1945: render log doesn't name the written file (want '-> $shown'): $out"
fi

# --- 23: the optimistic route — the model mid-boot is advertised before it answers ---
# switch.sh sets LITELLM_BOOTING_PORT to the target's port on the teardown render,
# so the gateway points at the model being loaded (a retryable 500 while it boots)
# instead of going routeless (a terminal 400 on `live`). The name is a registry
# placeholder (served_name → model id → booting-<port>) that the accurate render
# at ready corrects. The route is PREPENDED so the `live` alias (which mirrors the
# first route) points at the model being loaded.
C3_LITELLM_FAKE_LIVE="" LITELLM_BOOTING_PORT=9999 bash "$SYNC" --no-restart --quiet
got="$(routes)"
live_count="$(tr ' ' '\n' <<< "$got" | command grep -cx live || true)"
if [[ "$got" == *booting-9999* && "$live_count" == "1" ]]; then
  ok "a booting model is advertised before it answers, with the live alias"
else
  bad "booting model not advertised (got: $got)"
fi
[[ "$(route_field live litellm_params.api_base)" == "http://host.docker.internal:9999/v1" ]] \
  && ok "the live alias mirrors the booting route's port" \
  || bad "live alias api_base=$(route_field live litellm_params.api_base) (want :9999)"
# Name resolution: served_name → model id → booting-<port>, and booting-<port>
# whenever the port's own rows disagree. Ports are found dynamically so the test
# follows the registry rather than pinning one.
reg_json="$(bash scripts/lib/registry-emit.sh --json 2>/dev/null)"
boot_name() { python3 -c "
import json,collections,sys
d=json.load(sys.stdin)
by_port=collections.defaultdict(list)
for v in d.get('variants',[]):
    if v.get('port'): by_port[v['port']].append(v)
want=sys.argv[1]=='with'
for p,rows in sorted(by_port.items()):
    if len(rows)==1 and bool(rows[0].get('served_name'))==want and rows[0].get('model'):
        r=rows[0]; print(p, r.get('served_name') or r['model']); break
" "$1" <<< "$reg_json"; }
read -r sport sname <<< "$(boot_name with)"
if [[ -n "$sport" ]]; then
  C3_LITELLM_FAKE_LIVE="" LITELLM_BOOTING_PORT="$sport" bash "$SYNC" --no-restart --quiet
  got="$(routes)"
  if [[ "$got" == *"$sname"* ]]; then
    ok "a registered port advertises its served_name ($sname on :$sport)"
  else
    bad "served_name not advertised for :$sport (want $sname, got: $got)"
  fi
else
  echo "  skip — no registry port with a unique served_name"
fi
read -r mport mname <<< "$(boot_name without)"
if [[ -n "$mport" ]]; then
  C3_LITELLM_FAKE_LIVE="" LITELLM_BOOTING_PORT="$mport" bash "$SYNC" --no-restart --quiet
  got="$(routes)"
  if [[ "$got" == *"$mname"* ]]; then
    ok "a port without a served_name advertises the model id ($mname on :$mport)"
  else
    bad "model id not advertised for :$mport (want $mname, got: $got)"
  fi
else
  echo "  skip — no registry port without a served_name"
fi
# A port hosting two DIFFERENT models must never be named after either: the
# registry really has such ports (:8030 serves gemma-4-31b and
# deepseek-v4-flash-0731), so the placeholder is the only honest name.
read -r xport xnames <<< "$(python3 -c "
import json,collections,sys
d=json.load(sys.stdin)
by_port=collections.defaultdict(list)
for v in d.get('variants',[]):
    if v.get('port'): by_port[v['port']].append(v)
for p,rows in sorted(by_port.items()):
    models={v.get('model') for v in rows if v.get('model')}
    if len(models)>1:
        print(p, ' '.join(sorted(models))); break
" <<< "$reg_json")"
if [[ -n "$xport" ]]; then
  C3_LITELLM_FAKE_LIVE="" LITELLM_BOOTING_PORT="$xport" bash "$SYNC" --no-restart --quiet
  got="$(routes)"
  leak=""
  for n in $xnames; do [[ "$got" == *"$n"* ]] && leak="$leak $n"; done
  if [[ -n "$leak" ]]; then
    bad "multi-model port :$xport advertises a model that may not be the one loading:$leak"
  elif [[ "$got" == *"booting-$xport"* ]]; then
    ok "a multi-model port gets the placeholder, never either model's name (:$xport)"
  else
    bad "multi-model port :$xport got no placeholder route (got: $got)"
  fi
else
  echo "  skip — no registry port hosts two different models"
fi
# A garbage hint is a clean no-op (no crash, no route).
C3_LITELLM_FAKE_LIVE="" LITELLM_BOOTING_PORT=abc bash "$SYNC" --no-restart --quiet
got="$(routes)"
if [[ "$got" != *booting-* ]]; then
  ok "a garbage hint is a clean no-op"
else
  bad "garbage hint produced a route (got: $got)"
fi
# Once the port answers, the optimistic route yields to the probed name (no-op).
C3_LITELLM_FAKE_LIVE="9999=real-name" LITELLM_BOOTING_PORT=9999 bash "$SYNC" --no-restart --quiet
got="$(routes)"
if [[ "$got" == *real-name* && "$got" != *booting-9999* ]]; then
  ok "once the port answers, the optimistic route yields to the probed name"
else
  bad "optimistic route not a no-op once answering (got: $got)"
fi
# No hint → no optimistic route (the routeless case, as before).
C3_LITELLM_FAKE_LIVE="" bash "$SYNC" --no-restart --quiet
got="$(routes)"
if [[ "$got" != *booting-* ]]; then
  ok "without the hint, nothing booting is advertised"
else
  bad "optimistic route without the hint (got: $got)"
fi

[[ $fail -eq 0 ]] && echo "test-litellm-sync: ok" || echo "test-litellm-sync: FAIL"
exit $fail
