#!/usr/bin/env bash
# test-measurement-record-resolve — which slug a measurement record is attributed to (#1477).
#
# Every producer of a per-rig measurement record (bench.sh, verify-full.sh, soak-test.sh,
# quality-test.sh, … and rebench-full.sh's roll-up) asks measurement_record.resolve_serving
# "which slug served this URL?". Two bugs reported in #1477:
#   1. rebench-full.sh carried a private copy of the lookup: core catalog only, no port
#      guard, first match wins — with two models up, a rebench could be recorded under the
#      OTHER one, and a local-layer slug never.
#   2. the resolver only knew each compose's default container name, so a pod / estate
#      instance (`club3090-<name>`) was never recognised and a run against it left no record.
#
# Hermetic: a scratch copy of the tracked tree (with a local-layer slug promoted into it),
# a temp HOME for the estate file, and a docker shim answering `ps` / `port` / `inspect`.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
S="$T/root"; mkdir -p "$S" "$T/home" "$T/bin"
git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$S"

# ── a local-layer slug, promoted the way a user would ────────────────────────
LSLUG="llama-cpp/resolve-probe"; LMID="resolve-probe"
cat > "$T/spec.json" <<JSON
{
  "model_id": "$LMID", "display_name": "Resolve Probe", "family": "dense",
  "arch": {"hidden_size": 2048, "num_hidden_layers": 8, "num_attn_heads": 16, "num_kv_heads": 2,
           "head_dim_attn": 128, "max_ctx_supported": 4096, "attention_k_eq_v": false},
  "weights": {"q4": {"path": "Probe-Q4", "local_subdir": "Probe-Q4", "size_gb": 1.0, "format": "gguf",
                     "status": "incubating", "hf_repo": "org/Probe", "engine": "llama-cpp", "kind": "main",
                     "verify_glob": "*.gguf"}},
  "default_weight_variant": "q4", "vision_capable": false,
  "compose": {
    "path": "scripts/lib/profiles-local/composes/$LMID/llama-cpp/compose/single/q4/base.yml",
    "content": "# Profile (at-a-glance):\n#   Status:    🐣 Incubating\n#   Caveats:   test fixture\n# ---\nservices:\n  probe:\n    image: busybox\n    container_name: \"\${ESTATE_CONTAINER:-resolve-probe-local}\"\n"
  },
  "registry_entry": {"slug": "$LSLUG", "kwargs": {
    "model": "$LMID", "weights_variant": "q4", "workload": "fast-chat", "engine": "llama-cpp-local",
    "drafter": null, "kv_format": "q8_0", "tp": 1, "max_ctx": 4096, "max_num_seqs": 1, "mem_util": 0.9,
    "compose_path": "scripts/lib/profiles-local/composes/$LMID/llama-cpp/compose/single/q4/base.yml",
    "default_port": 20250, "kvcalc_key": "SKIP"}}
}
JSON
if ! (cd "$S" && python3 scripts/lib/profiles/promote.py --spec-file "$T/spec.json" --root "$S" \
        --layer local --yes >"$T/promote.log" 2>&1); then
  echo "✗ fixture: promote.py failed: $(tail -3 "$T/promote.log")" >&2; exit 1
fi

# ── a core slug whose default container name no other slug shares ────────────
read -r CSLUG CNAME < <(cd "$S" && python3 - <<'PY'
import re, sys
from collections import defaultdict
sys.path.insert(0, ".")
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY
by_name = defaultdict(list)
for slug, e in COMPOSE_REGISTRY.items():
    try:
        txt = open(e["compose_path"], encoding="utf-8").read()
    except OSError:
        continue
    m = re.search(r'container_name:\s*"?(?:\$\{[^:}]*:-)?([A-Za-z0-9._-]+)\}?"?', txt)
    if m:
        by_name[m.group(1)].append(slug)
unique = sorted((s[0], n) for n, s in by_name.items() if len(s) == 1)
pref = [u for u in unique if u[0] == "sgl/thinkingcap38-27b-dual-superfast"]   # the #1477 example
print(*(pref or unique)[0])
PY
)
[[ -n "${CSLUG:-}" && -n "${CNAME:-}" ]] || { echo "✗ fixture: no core slug with a unique container name" >&2; exit 1; }

# ── docker shim ──────────────────────────────────────────────────────────────
# MOCK_PS     "name|club3090.slug label;…"   (docker ps --format '{{.Names}}|{{.Label …}}')
# MOCK_PORTS  "name=hostport;…"               (docker port <name>)
# MOCK_IMAGES "name=image;…"                  (docker inspect <name> --format {{.Config.Image}})
cat > "$T/bin/docker" <<'MOCK'
#!/usr/bin/env bash
lookup() { local kv; for kv in ${2//;/ }; do [[ "${kv%%=*}" == "$1" ]] && printf '%s\n' "${kv#*=}"; done; }
case "$1" in
  ps)      # honour the format asked for: names only, or names|slug label
           if [[ "$*" == *Label* ]]; then printf '%s\n' "${MOCK_PS:-}" | tr ';' '\n'
           else printf '%s\n' "${MOCK_PS:-}" | tr ';' '\n' | cut -d'|' -f1; fi ;;
  port)    p="$(lookup "$2" "${MOCK_PORTS:-}")"; [[ -n "$p" ]] && echo "8000/tcp -> 0.0.0.0:$p" ;;
  inspect) lookup "$2" "${MOCK_IMAGES:-}" ;;
esac
exit 0
MOCK
chmod +x "$T/bin/docker"
[ "$(PATH="$T/bin:$PATH" command -v docker)" = "$T/bin/docker" ] || { echo "✗ docker shim not first on PATH" >&2; exit 1; }

py() { env -u HF_TOKEN -u LITELLM_MASTER_KEY PATH="$T/bin:$PATH" HOME="$T/home" python3 - "$S" "$@"; }
resolve() {   # $1 = URL ("" = none) → "slug container" | None
  py "$1" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
from scripts.lib.profiles.measurement_record import resolve_serving
hit = resolve_serving(sys.argv[2] or None)
print(" ".join(hit) if hit else "None")
PY
}

echo "1. default container names (core + local layer), port guard"
export MOCK_PS="$CNAME|" MOCK_PORTS="$CNAME=8142"
[[ "$(resolve '')" == "$CSLUG $CNAME" ]] && ok "core slug by its default container name ($CSLUG)" || bad "core: $(resolve '')"
[[ "$(resolve http://localhost:8142)" == "$CSLUG $CNAME" ]] && ok "…and the URL's port is one it publishes" || bad "core+port: $(resolve http://localhost:8142)"
[[ "$(resolve http://localhost:9999)" == "None" ]] && ok "a URL on a port it doesn't publish is not attributed to it" || bad "port guard: $(resolve http://localhost:9999)"
export MOCK_PS="resolve-probe-local|" MOCK_PORTS="resolve-probe-local=20250"
[[ "$(resolve http://localhost:20250)" == "$LSLUG resolve-probe-local" ]] && ok "a local-layer slug by its default container name" || bad "local: $(resolve http://localhost:20250)"

echo "2. pods (club3090-<name>)"
export MOCK_PS="club3090-small|$LSLUG" MOCK_PORTS="club3090-small=20250"
[[ "$(resolve http://localhost:20250)" == "$LSLUG club3090-small" ]] && ok "a pod by its club3090.slug label" || bad "label: $(resolve http://localhost:20250)"
[[ "$(resolve http://localhost:9999)" == "None" ]] && ok "the port guard applies to the label too" || bad "label port guard: $(resolve http://localhost:9999)"
mkdir -p "$T/home/.club3090"
printf 'schema_version: 1\nestate:\n  - name: small\n    compose: %s\n    gpus: [3]\n    port: 20250\n' "$LSLUG" > "$T/home/.club3090/estate.yml"
export MOCK_PS="club3090-small|"
[[ "$(resolve http://localhost:20250)" == "$LSLUG club3090-small" ]] && ok "a pod booted before the label: by the default estate file" || bad "estate: $(resolve http://localhost:20250)"
rm -f "$T/home/.club3090/estate.yml"
[[ "$(resolve http://localhost:20250)" == "None" ]] && ok "no label, no estate entry: not recognised (no guess)" || bad "unknown pod: $(resolve http://localhost:20250)"

echo "3. two models up — the #1477 rig: a core slug on :8142 and a pod on :20250"
export MOCK_PS="$CNAME|;club3090-small|$LSLUG" MOCK_PORTS="$CNAME=8142;club3090-small=20250"
[[ "$(resolve http://localhost:20250)" == "$LSLUG club3090-small" ]] && ok "the pod's URL → the pod's slug" || bad "pod URL: $(resolve http://localhost:20250)"
[[ "$(resolve http://localhost:8142)" == "$CSLUG $CNAME" ]] && ok "the core slug's URL → the core slug" || bad "core URL: $(resolve http://localhost:8142)"

# The same headline cases through resolve_serving_tag, the name every producer imported
# before this change (and still can) — so this block also runs against the old code.
tag() { py "$1" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
from scripts.lib.profiles.measurement_record import resolve_serving_tag
print(resolve_serving_tag(sys.argv[2] or None))
PY
}
[[ "$(tag http://localhost:20250)" == "$LSLUG" ]] && ok "resolve_serving_tag: a pod's URL → the pod's slug (it used to find nothing)" || bad "resolve_serving_tag pod: $(tag http://localhost:20250)"

echo "4. the fingerprint inspects the container that matched"
export MOCK_IMAGES="club3090-small=ghcr.io/example/pod-image:1;$CNAME=ghcr.io/example/core-image:2"
got="$(py "$LSLUG" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
from scripts.lib.profiles.measurement_record import _detect_serving_fingerprint
print(_detect_serving_fingerprint(sys.argv[2], "club3090-small")[0])
PY
)"
[[ "$got" == "ghcr.io/example/pod-image:1" ]] && ok "engine_pin read from the pod's own container" || bad "pod fingerprint: $got"

echo "5. estate stamps the slug label on the pods it boots"
got="$(cd "$S" && py <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
from scripts.lib.profiles import estate_cli
from scripts.lib.profiles.compat import InstanceSpec
estate_cli.compose_service_names = lambda name: ["svc"]
estate_cli.resolve_gpu_uuids = lambda gpus: None
doc = estate_cli.compose_override_doc(InstanceSpec(name="small", compose_name="vllm/minimal", gpu_indices=(3,), port=20250))
print(doc["services"]["svc"].get("labels", {}).get("club3090.slug"))
PY
)"
[[ "$got" == "vllm/minimal" ]] && ok "compose override labels each service club3090.slug=<slug>" || bad "override label: $got"

echo "6. rebench-full.sh's record step uses the resolver (run for real)"
command grep -q 'COMPOSE_REGISTRY' "$S/scripts/rebench-full.sh" && bad "rebench-full.sh still reads COMPOSE_REGISTRY" || ok "no private core-only lookup left"
mkdir -p "$T/out"
PREFLIGHT_NO_AUTODETECT=1 CONTAINER=none ENGINE_KIND=vllm BENCH_MOCK=1 bash "$S/scripts/bench.sh" > "$T/out/bench.log" 2>/dev/null
awk "/<<'PY_RECORD'/{f=1; next} /^PY_RECORD\$/{f=0} f" "$S/scripts/rebench-full.sh" > "$T/record.py"
export MOCK_PS="$CNAME|;club3090-small|$LSLUG" MOCK_PORTS="$CNAME=8142;club3090-small=20250"
out="$(cd "$S" && env -u HF_TOKEN -u LITELLM_MASTER_KEY PATH="$T/bin:$PATH" HOME="$T/home" OUT_DIR="$T/out" TAG=probe URL=http://localhost:20250 python3 "$T/record.py" 2>&1)"
[[ "$out" == *"(slug $LSLUG)"* ]] && ok "a rebench of the pod is recorded under the pod's slug, not the other one up" || bad "rebench record: $(printf '%s' "$out" | head -3)"
ls "$S/results/measurement-records/" 2>/dev/null | command grep -q "^${LSLUG//\//-}__" && ok "…in the pod slug's corpus file" || bad "no record file for $LSLUG: $(ls "$S/results/measurement-records/" 2>/dev/null | tr '\n' ' ')"
out="$(cd "$S" && env -u HF_TOKEN -u LITELLM_MASTER_KEY PATH="$T/bin:$PATH" HOME="$T/home" OUT_DIR="$T/out" TAG=probe URL=http://localhost:9999 python3 "$T/record.py" 2>&1)"
[[ "$out" == *"record:      skipped — no running container serving this URL"* ]] && ok "a URL nothing up publishes: skipped, not guessed" || bad "rebench skip: $(printf '%s' "$out" | head -3)"

[ "$fail" -eq 0 ] && echo "test-measurement-record-resolve: ok" || { echo "test-measurement-record-resolve: FAIL"; exit 1; }
