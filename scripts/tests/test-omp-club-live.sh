#!/usr/bin/env bash
# test-omp-club-live — services/omp/extensions/club-live.ts stays in sync with
# the repo contracts it consumes: litellm_sync's rendered block, the settings
# precedence, and the omp-setup.sh wire policies.
#
# WHY THIS TEST EXISTS
# --------------------
# The extension lives outside the gateway scripts, so a PR that changes the
# rendered route format, the settings precedence, or the compat pins breaks it
# silently — the failure only surfaces inside a running omp session, far from
# the PR that caused it. The parser checks run the extension's REAL parser
# (node --experimental-strip-types) on the REAL renderer's output; the pin
# checks tripwire the pair. Offline: scratch config dir, no gateway, no omp.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROOT
EXT="$ROOT/services/omp/extensions/club-live.ts"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
bad() { echo "✗ $1" >&2; fail=1; }

node --experimental-strip-types -e 'const x: number = 1' 2>/dev/null \
  || { echo "✗ test-omp-club-live: needs node >= 22.6 (--experimental-strip-types)" >&2; exit 1; }

parse_case() {   # parse_case <fixture> <expect-json|NULL> <desc>
  local fixture="$1" expect="$2" desc="$3"
  EXT_PATH="$EXT" FIXTURE="$fixture" EXPECT="$expect" \
  node --experimental-strip-types --input-type=module -e '
import { readFileSync } from "node:fs";
const ext = await import("file://" + process.env.EXT_PATH);
const got = ext.parseRenderedConfig(readFileSync(process.env.FIXTURE, "utf8"));
const want = process.env.EXPECT === "NULL" ? null : JSON.parse(process.env.EXPECT);
const same = got === null ? want === null : want !== null &&
  got.slug === want.slug && got.ctxWindow === want.ctxWindow && got.maxOut === want.maxOut;
if (!same) throw new Error("parseRenderedConfig -> " + JSON.stringify(got) + ", want " + JSON.stringify(want));
' || bad "parser: $desc"
}

# 1a. The parser on the REAL rendered block. render_block sorts routes by model
#     name and writes each one's model_info; the extension must pick the FIRST
#     route's slug and window — the same first entry the gateway serves.
python3 - "$T/runtime.yaml" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "lib"))
import litellm_sync as ls
live = [ls.Live(8091, "qwen3.8-27b", 262144, False, False, ""),
        ls.Live(8018, "qwen3.8-flash-next-iq3_s", 131072, False, True, "")]
block, n = ls.render_block(live)
assert n == 2, f"expected 2 routes, got {n}"
# Wrapped like the runtime view: gateway settings above, block below.
open(sys.argv[1], "w").write("general_settings:\n  master_key: sk-test\nmodel_list:\n" + block)
PY
[ -s "$T/runtime.yaml" ] || bad "fixture render failed"
exp_slug=$(grep -m1 '  - model_name:'     "$T/runtime.yaml" | awk '{print $3}')
exp_ctx=$( grep -m1 'max_input_tokens:'   "$T/runtime.yaml" | awk '{print $2}')
exp_out=$( grep -m1 'max_output_tokens:'  "$T/runtime.yaml" | awk '{print $2}')
[[ "$exp_slug" == "qwen3.8-27b" ]] || bad "fixture sanity: first route should sort to qwen3.8-27b, got '$exp_slug'"
parse_case "$T/runtime.yaml" \
  "{\"slug\":\"$exp_slug\",\"ctxWindow\":$exp_ctx,\"maxOut\":$exp_out}" \
  "first route of the generated block, with its own model_info"

# 1b. Nothing local serving + the user's cloud routes (config.local.yaml is
#     appended AFTER the generated block at the same indent): the alias must
#     stay unresolved, never adopt a cloud route.
python3 - "$T/cloud-only.yaml" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "lib"))
import litellm_sync as ls
block, n = ls.render_block([])
assert n == 0, f"expected empty block, got {n}"
cloud = ("  - model_name: my-cloud-model\n"
         "    litellm_params:\n"
         "      model: openai/gpt-4o\n"
         "      api_base: https://api.openai.com/v1\n"
         "      api_key: EMPTY\n"
         "    model_info:\n"
         "      max_input_tokens: 272000\n"
         "      max_output_tokens: 128000\n")
open(sys.argv[1], "w").write("general_settings:\n  master_key: sk-test\nmodel_list:\n" + block + cloud)
PY
parse_case "$T/cloud-only.yaml" NULL "empty generated block must NOT pick a cloud route"

# 1c. A route whose engine reports no context (ctx None -> no model_info lines)
#     followed by a routed one: the window must NOT leak from the second route
#     onto the first route's slug (it falls back to the extension's default).
python3 - "$T/no-ctx.yaml" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["ROOT"], "scripts", "lib"))
import litellm_sync as ls
live = [ls.Live(8091, "aaa-served-model", None, False, False, ""),
        ls.Live(8018, "qwen3.8-flash-next-iq3_s", 262144, False, True, "")]
block, n = ls.render_block(live)
assert n == 2, f"expected 2 routes, got {n}"
open(sys.argv[1], "w").write("model_list:\n" + block)
PY
parse_case "$T/no-ctx.yaml" \
  '{"slug":"aaa-served-model","ctxWindow":131072,"maxOut":32768}' \
  "ctx-less route keeps the default window, not the next route's"

# 2. The gateway key follows the club_config.py precedence: shell env >
#    club3090.env > secrets.env > legacy repo .env > public default.
#    CLUB3090_ROOT is always the scratch dir: the parser's default checkout
#    must never leak a real install's .env into these checks.
CFG="$T/cfg"; mkdir -p "$CFG" "$T/repo"
key_case() {   # key_case <expect> <descriptions...> — env prefix comes as "$@"
  local expect="$1"; shift
  CLUB3090_ROOT="$T/repo" EXT_PATH="$EXT" EXPECT_KEY="$expect" \
    env "$@" node --experimental-strip-types --input-type=module -e '
const ext = await import("file://" + process.env.EXT_PATH);
const got = ext.resolveGatewayKey();
if (got !== process.env.EXPECT_KEY) throw new Error("resolveGatewayKey -> " + JSON.stringify(got) + ", want " + JSON.stringify(process.env.EXPECT_KEY));
' || bad "gateway-key chain: $*"
}
printf 'LITELLM_MASTER_KEY=from-club3090-env\n'  > "$CFG/club3090.env"
printf "LITELLM_MASTER_KEY='from-secrets-quoted'\n" > "$CFG/secrets.env"
key_case from-club3090-env   CLUB3090_CONFIG_DIR="$CFG"                       # club3090.env wins
rm "$CFG/club3090.env"
key_case from-secrets-quoted CLUB3090_CONFIG_DIR="$CFG"                       # secrets.env, quotes stripped
printf 'export LITELLM_MASTER_KEY=from-export-prefix\n' > "$CFG/secrets.env"
key_case from-export-prefix  CLUB3090_CONFIG_DIR="$CFG"                       # `export ` prefix, as parse_env_file accepts
key_case from-shell          CLUB3090_CONFIG_DIR="$CFG" LITELLM_MASTER_KEY=from-shell
key_case explicit-wins       CLUB3090_CONFIG_DIR="$CFG" LITELLM_MASTER_KEY=from-shell CLUB_LIVE_KEY=explicit-wins
printf 'LITELLM_MASTER_KEY=from-repo-dotenv\n' > "$T/repo/.env"
key_case from-repo-dotenv    CLUB3090_CONFIG_DIR="$T/empty-cfg"               # legacy repo .env
rm "$T/repo/.env"
key_case sk-litellm-master-key CLUB3090_CONFIG_DIR="$T/empty-cfg"             # public compose default

# 2b. The key is RE-READ on every call — a gateway-key rotate lands without
#     restarting the process that resolved it first.
CFG2="$T/cfg2"; mkdir -p "$CFG2"
printf 'LITELLM_MASTER_KEY=original\n' > "$CFG2/secrets.env"
CLUB3090_ROOT="$T/repo" CLUB3090_CONFIG_DIR="$CFG2" EXT_PATH="$EXT" SECRETS="$CFG2/secrets.env" \
node --experimental-strip-types --input-type=module -e '
import { writeFileSync } from "node:fs";
const ext = await import("file://" + process.env.EXT_PATH);
const before = ext.resolveGatewayKey();
writeFileSync(process.env.SECRETS, "LITELLM_MASTER_KEY=rotated-live\n");
const after = ext.resolveGatewayKey();
if (before !== "original" || after !== "rotated-live")
  throw new Error("key not re-read: before=" + before + " after=" + after);
' || bad "gateway-key rotate: resolveGatewayKey must re-read the settings on every call"

# 3. Pin pairs. The extension re-registers its provider with the SAME wire
#    policies omp-setup.sh pins for the `club` provider; if either side moves
#    without the other, thinking stops reaching the engine's chat template.
#    Presence-grep is a drift floor, not a wire proof: it catches a pin being
#    dropped on either side, not a host-side rename of the payload field.
for pin in 'supportsDeveloperRole: false' 'supportsReasoningEffort: true' \
           'qwenTemplateReasoningEffort: true' 'thinking_token_budget: 8192'; do
  grep -qF "$pin" "$EXT"                        || bad "extension lost $pin"
  grep -qF "$pin" "$ROOT/scripts/omp-setup.sh"  || bad "omp-setup.sh moved $pin without the extension's"
done
# Same values, different syntax: the TS side quotes them, the YAML side does not.
grep -q 'thinkingFormat: "qwen-chat-template"'  "$EXT" || bad "extension lost thinkingFormat"
grep -q 'thinkingFormat: qwen-chat-template'    "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh moved thinkingFormat without the extension's"
grep -q 'reasoningDisableMode: "none-effort"'   "$EXT" || bad "extension lost reasoningDisableMode"
grep -q 'reasoningDisableMode: none-effort'     "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh moved reasoningDisableMode without the extension's"
# Reply cap: omp-setup pins maxTokens 32768 for every id it knows; the alias
# pins the same value.
grep -q 'REPLY_CAP = 32_768'    "$EXT"                   || bad "extension lost its 32768 reply cap"
grep -q 'maxTokens: 32768'      "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh moved the 32768 cap without the extension's"
# Effort ladders: the Qwen3.8 ladder and the GLM ladder must match the
# omp-setup.sh pair, and the GLM id list must be the same on both sides.
grep -q 'efforts: \["low", "medium", "xhigh"\]' "$EXT" || bad "extension Qwen ladder drifted"
grep -q 'efforts: \[low, medium, xhigh\]'       "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh Qwen ladder drifted without the extension's"
grep -q 'efforts: \["low", "high", "max"\]'     "$EXT" || bad "extension GLM ladder drifted"
grep -q 'efforts: \[low, high, max\]'           "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh GLM ladder drifted without the extension's"
grep -q '"glm-5.3-flash": true'                 "$EXT" || bad "extension GLM id list drifted"
grep -q 'GLM53_IDS = \["glm-5.3-flash"\]'       "$ROOT/scripts/omp-setup.sh" || bad "omp-setup.sh GLM id list drifted without the extension's"

# 4. Hygiene tripwires: no per-install secret baked in. The public default is
#    fine — it is published in the repo — but a rotated key (gateway-key.sh:
#    sk-club- + token_hex) or any other sk- token is not. No absolute home path.
grep -Eo 'sk-[A-Za-z0-9_-]+' "$EXT" | grep -vx 'sk-litellm-master-key' | grep -q . \
  && bad "extension embeds a per-install gateway key"
grep -q  '/home/'           "$EXT" && bad "extension hardcodes an absolute home path"

[[ $fail -eq 0 ]] && echo "✓ test-omp-club-live"
exit $fail
