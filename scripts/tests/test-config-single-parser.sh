#!/usr/bin/env bash
# test-config-single-parser — club-3090 settings are parsed in ONE place (club-3090#1466).
#
# WHY THIS EXISTS
# ---------------
# The repo-root .env was read five different ways (a line parser in switch.sh,
# `set -a; source` in launch.sh/report.sh/setup.sh, repo_dotenv.py, `docker compose
# --env-file`, single-key greps) with five precedence rules. #1466 replaces them with
# scripts/lib/club-config.sh + scripts/lib/club_config.py. The same drift happened to
# the engine classifier (#1282, #1372): once there is one implementation, the next
# copy arrives quietly unless something fails on it. This is that something.
#
# It is a RATCHET. Files that still read .env their own way are listed below with the
# #1466 phase that moves them. A file that starts reading .env and isn't listed fails;
# a listed file that no longer does also fails, so the list can only shrink.
#
# Scope: tracked .sh / .service files for the shell patterns, tracked .py for the
# Python ones; comment lines, tests and the two loaders themselves are excluded.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

# Files allowed to read or write .env privately until their phase lands.
ALLOWLIST=$(cat <<'EOF'
scripts/switch.sh                                   1b loader, 1c --set-default / thinking pin writer
scripts/launch.sh                                   1b loader
scripts/report.sh                                   1b loader
scripts/setup.sh                                    1b loader, 1c MODEL_DIR writer, WSL2 compose-dir .env
scripts/gpu-mode.sh                                 1b docker compose --env-file, single-key reads
scripts/systemd/club3090-model-switch.service       1b EnvironmentFile path
scripts/lib/profiles/repo_dotenv.py                 1b becomes a wrapper over club_config.py
scripts/lib/profiles/estate_cli.py                  1b loader
scripts/lib/profiles/deriver.py                     1b MODEL_DIR read
services/comfyui/comfyui-paths.sh                   1b reads, 1c LANIP / COMFYUI_* writer
services/comfyui/download_director.sh               1b MODEL_DIR read
services/studio/push-pipe-to-owui.sh                1b LANIP read
tools/residency-instrument/run-instrumented-soak.sh 1b loader
tools/serve-cockpit/club3090_cockpit/app.py         1c thinking-pin / director writer
tools/serve-cockpit/club3090_cockpit/services.py    1b reads + docker compose --env-file, 1c writer
EOF
)

# ── the detector ────────────────────────────────────────────────────────────
# `.env` as a path of its own: not preceded by a word character, a dot or a dash,
# so local.env, imagegen.env, club3090.env and secrets.env never count.
E='(^|[^A-Za-z0-9_.-])\.env\b'
SH_PATTERNS=(
  "(source|^[[:space:]]*\\.)[[:space:]]+[^#]*${E}"                      # sourcing it
  '--env-file[= ]+[^/[:space:]]'                                        # docker compose --env-file (not /dev/null)
  'EnvironmentFile='                                                    # systemd
  "(<|>>?)[[:space:]]*\"?[^[:space:]]*${E}"                              # redirecting from or to it
  "\\b(grep|sed|awk|cut|cat|mv|cp|touch|tee)\\b[^|;]*${E}"               # a tool reading or rewriting it
  "^[[:space:]]*(local[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*[[:space:]]+)*[A-Za-z_][A-Za-z0-9_]*=[\"']?[^[:space:]]*[/\"'{]\\.env\\b"  # its path in a variable
)
PY_PATTERNS=(
  "[\"']\\.env[\"']"                                                    # Path(root) / ".env", "--env-file", ".env"
)
# detect <file>... → prints the files that parse .env themselves.
# ⚠️ No `grep | grep -q` here: under pipefail, -q exiting on the first match SIGPIPEs
# the upstream grep on any file big enough to still be writing, the pipeline
# "fails", and a real reader reads as clean. (Caught by the ratchet below on
# report.sh, while the one-line self-test fixtures all passed.) Read once, match
# against the variable.
detect() {
  local f p body
  local -a pats
  for f in "$@"; do
    body="$(command grep -vE '^[[:space:]]*#' "$f" 2>/dev/null || true)"
    case "$f" in *.py) pats=("${PY_PATTERNS[@]}") ;; *) pats=("${SH_PATTERNS[@]}") ;; esac
    for p in "${pats[@]}"; do
      if command grep -qE -e "$p" <<<"$body"; then printf '%s\n' "$f"; break; fi
    done
  done
  return 0
}

# ── self-test: the detector finds readers and ignores look-alikes ───────────
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
w() { printf '%s\n' "$2" > "$T/$1"; }
w r1.sh  'set -a; source "${ROOT_DIR}/.env"; set +a'
w r2.sh  'done < "${ROOT_DIR}/.env"'
w r3.sh  'sudo docker compose --env-file "$DIR/.env" up -d'
w r4.sh  'v=$(grep -E "^LANIP=" "$HERE/../../.env" | cut -d= -f2-)'
w r5.sh  'ENV_FILE="${ROOT_DIR}/.env"'
w r6.sh  'echo "MODEL_DIR=$m" >> "${ROOT_DIR}/.env"'
w r7.service 'EnvironmentFile=-/opt/club-3090/.env'
w r8.py  'text = (Path(root) / ".env").read_text()'
# A large file whose reader line comes first: under the old `grep | grep -q`
# pipeline this one read as clean.
{ echo 'source "${ROOT_DIR}/.env"'; for i in $(seq 1 20000); do echo "echo line $i padding padding padding"; done; } > "$T/r9.sh"
w n1.sh  '# source "${ROOT_DIR}/.env"   (a comment)'
w n2.sh  'docker compose --env-file /dev/null -f x.yml config'
w n3.sh  'cp services/litellm/local.env /tmp/x; echo ok > "$D/imagegen.env"'
w n4.sh  '. "$LIB/club-config.sh"; club_config_load "$ROOT"'
w n5.sh  'echo "[setup] set MODEL_DIR in your config"'
w n6.py  '"""Reads ``<root>/.env`` as a legacy fallback."""'
w n7.py  'p = config_dir() / "club3090.env"'
got="$(cd "$T" && detect r1.sh r2.sh r3.sh r4.sh r5.sh r6.sh r7.service r8.py r9.sh n1.sh n2.sh n3.sh n4.sh n5.sh n6.py n7.py | tr '\n' ' ')"
want="r1.sh r2.sh r3.sh r4.sh r5.sh r6.sh r7.service r8.py r9.sh "
[[ "$got" == "$want" ]] && ok "self-test: detector flags 9 readers (incl. a 20,000-line file) and ignores 7 look-alikes" \
                        || bad "self-test: detector flagged [$got], want [$want]"

# ── the tree ────────────────────────────────────────────────────────────────
cd "$ROOT" || exit 1
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t FILES < <(git ls-files -- scripts tools services | command grep -E '\.(sh|py|service)$')
else
  mapfile -t FILES < <(find scripts tools services -type f \( -name '*.sh' -o -name '*.py' -o -name '*.service' \) | sort)
fi
mapfile -t FILES < <(printf '%s\n' "${FILES[@]}" \
  | command grep -vE '(^|/)tests/|/\.venv/|/node_modules/' \
  | command grep -vxE 'scripts/lib/club-config\.sh|scripts/lib/club_config\.py')
[[ ${#FILES[@]} -gt 100 ]] || bad "only ${#FILES[@]} files scanned — the file list is broken"

found="$(detect "${FILES[@]}" | sort)"
allowed="$(awk 'NF {print $1}' <<<"$ALLOWLIST" | sort)"
new="$(comm -23 <(printf '%s\n' "$found") <(printf '%s\n' "$allowed") | command grep -v '^$' || true)"
gone="$(comm -13 <(printf '%s\n' "$found") <(printf '%s\n' "$allowed") | command grep -v '^$' || true)"
if [[ -n "$new" ]]; then
  bad "new private .env parser(s) — read settings with scripts/lib/club-config.sh (bash) or scripts/lib/club_config.py (Python), and write them with club_config_set / club_config.py set:"
  sed 's/^/      /' <<<"$new" >&2
else
  ok "no new private .env parsers ($(wc -l <<<"$found") known ones left, each with its #1466 phase)"
fi
if [[ -n "$gone" ]]; then
  bad "no longer parses .env itself — remove it from the ALLOWLIST in this test (the list only shrinks):"
  sed 's/^/      /' <<<"$gone" >&2
fi

[[ $fail -eq 0 ]] && echo "test-config-single-parser: ok" || echo "test-config-single-parser: FAIL"
exit $fail
