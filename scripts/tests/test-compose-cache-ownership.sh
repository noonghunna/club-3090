#!/usr/bin/env bash
#
# Guard: files a root container writes into a host cache directory must stay the
# user's to delete — in the repo tree, and now in ~/.cache and ~/.local/share.
#
# vLLM runs as root. When a compose maps a host path to a container cache path
# (/root/.triton/cache, /root/.cache/vllm/torch_compile_cache), every file it writes
# there is root-owned on the host. `user: "0:${DOCKER_GID:-1000}"` keeps UID 0 —
# which the image needs — while giving the files the invoking user's GROUP, so they
# come out group-writable (with the entrypoint's `umask 0002`) and clean up normally.
#
# Measured cost of getting this wrong: three worktrees on the reference rig
# accumulated 140-406 MB each of root-owned triton/torch_compile cache that the
# owning user could not remove. Setting TRITON_CACHE_DIR does NOT help — these
# composes bind-mount that exact container path, so the env var has no bearing on
# where the files land.
#
# #1466 phase 4 moved those mounts out of the repo, to a per-user directory keyed
# by the engine image (scripts/lib/engine_cache.py):
#     - ${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/<subdir>:<container cache path>
#     - ${KV_OFFLOAD_DIR:-${CLUB3090_DATA_DIR:-../../../../../..}/kv-offload}:/kv-offload
# which adds a second way to leave root-owned files: docker creates a MISSING
# bind-mount source itself, as root:root 0755 — measured on the reference rig, every
# in-repo triton/ and torch_compile/ top folder is root:root for exactly that reason,
# so even group-writable files inside them can't be removed without sudo. Under $HOME
# that would be worse. So the launchers create every mounted directory as the user
# first, and this guard checks the whole chain:
#   1. every cache mount's compose runs with the host GID, UID 0     (unchanged)
#   2. no compose mounts a cache straight into the repo tree any more, and every
#      shared-cache mount falls back to its engine's in-repo cache/ (raw compose)
#   3. every KV disk-tier mount has the data-dir default; the ones whose compose
#      lacks the GID line are a known list that may only shrink
#   4. the launcher helper creates exactly what the compose mounts, owned by you
#
# _archive/ composes are exempt: they are historical records, not launched.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FAIL=0
bad() { echo "FAIL: $1 — expected $2, got $3" >&2; FAIL=1; }
ok()  { echo "  ✓ $1"; }

NEW_CACHE_RE='^\s+- \$\{CLUB3090_ENGINE_CACHE_DIR:-[^}]*\}/[^:]+:/root/'
OLD_CACHE_RE='^\s+- \.\.[^:]*cache[^:]*:/root/'
KV_MOUNT='- ${KV_OFFLOAD_DIR:-${CLUB3090_DATA_DIR:-../../../../../..}/kv-offload}:/kv-offload'
live() { command grep -v '/_archive/' | sort; }

mapfile -t mounts < <(command grep -rlE "$NEW_CACHE_RE|$OLD_CACHE_RE" "${ROOT}/models" --include='*.yml' 2>/dev/null | live)
[[ ${#mounts[@]} -gt 0 ]] || bad "found composes that mount a cache path" "at least one" "none — the scan matched nothing"

# ── 1. host GID, UID 0 ─────────────────────────────────────────────────────────
missing=()
for f in "${mounts[@]}"; do
  command grep -q 'user: "0:' "$f" || missing+=("${f#${ROOT}/}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  bad "every cache-mounting compose sets the host GID" "user: \"0:\${DOCKER_GID:-1000}\" in all ${#mounts[@]}" \
      $'\n'"$(printf '    %s\n' "${missing[@]}")"
else
  ok "all ${#mounts[@]} live cache-mounting composes run with the host GID"
fi

# The UID must stay 0 — the vLLM images expect root. A well-meaning "fix" that
# sets user: "1000:1000" breaks the container instead of the cleanup.
wrong_uid=()
for f in "${mounts[@]}"; do
  if command grep -qE '^\s*user:\s*"[^0]' "$f"; then wrong_uid+=("${f#${ROOT}/}"); fi
done
if [[ ${#wrong_uid[@]} -gt 0 ]]; then
  bad "UID stays 0" "user: \"0:...\"" $'\n'"$(printf '    %s\n' "${wrong_uid[@]}")"
else
  ok "every one keeps UID 0 (the images need root); only the GID is the host's"
fi

# ── 2. the shared location, with the in-repo cache as the raw-compose fallback ──
bare="$(command grep -nE "$OLD_CACHE_RE" "${mounts[@]}" 2>/dev/null | sed "s#^${ROOT}/##")"
if [[ -n "$bare" ]]; then
  bad "no compose mounts a cache straight into the repo tree (#1466 phase 4)" \
      "\${CLUB3090_ENGINE_CACHE_DIR:-../../../cache}/<subdir>" $'\n'"$(sed 's/^/    /' <<<"$bare")"
else
  ok "no compose mounts a cache into the repo tree: every one goes through \${CLUB3090_ENGINE_CACHE_DIR}"
fi
wrong_fb=(); n_shared=0
while IFS=: read -r f _ line; do
  [[ -n "$f" ]] || continue
  n_shared=$((n_shared + 1))
  fb="$(sed -E 's/^\s+- \$\{CLUB3090_ENGINE_CACHE_DIR:-([^}]*)\}.*/\1/' <<<"$line")"
  want="$(cd "$(dirname "$f")/../../.." && pwd)/cache"          # <model>/<engine>/cache
  got="$(cd "$(dirname "$f")" && cd "$fb" 2>/dev/null && pwd)"
  [[ "$got" == "$want" ]] || wrong_fb+=("${f#${ROOT}/}: falls back to '$fb' (→ ${got:-missing}), not its engine's cache/")
done < <(command grep -nE "$NEW_CACHE_RE" "${mounts[@]}" 2>/dev/null)
if [[ ${#wrong_fb[@]} -gt 0 ]]; then
  bad "a raw docker compose up keeps the engine's in-repo cache/" "../../../cache" $'\n'"$(printf '    %s\n' "${wrong_fb[@]}")"
elif [[ $n_shared -eq 0 ]]; then
  bad "shared-cache mounts found" "at least one \${CLUB3090_ENGINE_CACHE_DIR} mount" "none"
else
  ok "all $n_shared shared-cache mounts fall back to their engine's in-repo cache/ when nothing is set (raw compose)"
fi

# ── 3. the KV disk tier: data-dir default; GID-less composes may only shrink ────
# The 4 SGLang dual MTP composes carry no `user:` line, so HiCache's files are
# root:root. Under a launcher-created (user-owned) directory the user can still
# delete flat files in it; files inside subdirectories SGLang creates would need
# sudo. Not changed here (it needs a live SGLang boot); listed so it can't spread.
KV_NO_GID=$(cat <<'EOF'
models/qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml
models/qwen3.8-27b/sglang/compose/dual/fp8/mtp.yml
models/thinkingcap-qwen3.8-27b/sglang/compose/dual/autoround-int4/mtp.yml
models/thinkingcap-qwen3.8-27b/sglang/compose/dual/fp8/mtp.yml
EOF
)
mapfile -t kvs < <(command grep -rlE ':/kv-offload(:|$)' "${ROOT}/models" --include='*.yml' 2>/dev/null | live)
wrong_kv=(); no_gid=()
for f in "${kvs[@]}"; do
  while IFS= read -r line; do
    [[ "$(sed -E 's/^\s+//' <<<"$line")" == "$KV_MOUNT" ]] || wrong_kv+=("${f#${ROOT}/}: $line")
  done < <(command grep -E '^\s+- .*:/kv-offload(:|$)' "$f")
  command grep -q 'user: "0:' "$f" || no_gid+=("${f#${ROOT}/}")
done
if [[ ${#kvs[@]} -eq 0 ]]; then
  bad "KV disk-tier mounts found" "at least one" "none — the scan matched nothing"
elif [[ ${#wrong_kv[@]} -gt 0 ]]; then
  bad "every KV disk-tier mount defaults to the data dir" "$KV_MOUNT" $'\n'"$(printf '    %s\n' "${wrong_kv[@]}")"
else
  ok "all ${#kvs[@]} KV disk-tier mounts: KV_OFFLOAD_DIR, else <data dir>/kv-offload, else the repo's kv-offload/"
fi
new_ng="$(comm -23 <(printf '%s\n' "${no_gid[@]}" | sort) <(sort <<<"$KV_NO_GID") | command grep -v '^$' || true)"
gone_ng="$(comm -13 <(printf '%s\n' "${no_gid[@]}" | sort) <(sort <<<"$KV_NO_GID") | command grep -v '^$' || true)"
[[ -z "$new_ng" ]] || bad "a new KV disk-tier compose runs with the host GID" "user: \"0:\${DOCKER_GID:-1000}\"" $'\n'"$(sed 's/^/    /' <<<"$new_ng")"
[[ -z "$gone_ng" ]] || bad "the GID-less list only shrinks — remove these from KV_NO_GID in this test" "gone from the list" $'\n'"$(sed 's/^/    /' <<<"$gone_ng")"
[[ -z "$new_ng" && -z "$gone_ng" ]] && ok "KV disk-tier composes without the GID line: exactly the $(awk 'NF' <<<"$KV_NO_GID" | wc -l) known SGLang ones"

# ── 4. the launcher creates what the compose mounts, as you ─────────────────────
# Runs the REAL helper the launchers call (engine_cache.py prepare) on a real compose,
# with a stub docker for the image ID and a temporary cache/data dir; renders the
# compose with what it returned (docker compose config — read-only) and checks that
# every host path mounted under the two variables exists and is yours.
if docker compose version >/dev/null 2>&1; then
  T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
  printf '#!/usr/bin/env bash\n[[ "$1 $2" == "image inspect" ]] && echo sha256:0123456789abcdef0123456789abcdef && exit 0\nexit 1\n' > "$T/docker"
  chmod +x "$T/docker"
  F="${ROOT}/models/qwen3.8-27b/vllm/compose/dual/autoround-int4/mtp.yml"
  out="$(env -u KV_OFFLOAD_DIR -u CLUB3090_ENGINE_CACHE_DIR CLUB3090_CACHE_DIR="$T/cache" CLUB3090_DATA_DIR="$T/data" \
         MODEL_DIR=/nonexistent python3 "${ROOT}/scripts/lib/engine_cache.py" prepare --compose "$F" \
         --compose-bin "docker compose" --docker "$T/docker" 2>/dev/null)"
  assigns=(); [[ -n "$out" ]] && mapfile -t assigns <<<"$out"
  srcs="$(env -u KV_OFFLOAD_DIR MODEL_DIR=/nonexistent "${assigns[@]}" docker compose --env-file /dev/null -f "$F" config --format json 2>/dev/null \
          | python3 -c 'import json,sys; [print(v["source"]) for s in json.load(sys.stdin)["services"].values() for v in s.get("volumes",[]) if v["target"] in ("/kv-offload","/root/.triton/cache","/root/.cache/vllm/torch_compile_cache")]')"
  n="$(awk 'NF' <<<"$srcs" | wc -l)"; notmine=()
  while IFS= read -r s; do
    [[ -n "$s" ]] || continue
    [[ "$s" == "$T"/* && -d "$s" && "$(stat -c %u "$s")" == "$(id -u)" ]] || notmine+=("$s")
  done <<<"$srcs"
  if [[ "$n" -ne 3 || ${#notmine[@]} -gt 0 ]]; then
    bad "the launcher helper creates every mounted dir, as you, before compose runs" \
        "3 mounts (triton, torch_compile, kv-offload) under the temp dirs, owned by uid $(id -u)" \
        "$n mount(s); not created / not yours: ${notmine[*]:-none} (helper said: ${out//$'\n'/ })"
  else
    ok "the launcher helper creates all 3 mounted dirs under ~/.cache / ~/.local/share stand-ins, owned by you, before compose runs"
  fi
else
  echo "  - SKIP launcher-helper leg: no docker compose here"
fi

if [[ $FAIL -ne 0 ]]; then echo "FAIL: test-compose-cache-ownership" >&2; exit 1; fi
echo "PASS: test-compose-cache-ownership"
