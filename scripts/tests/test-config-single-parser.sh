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
# Python ones; comment lines, tests and the two loaders themselves are excluded, and
# so is scripts/lib/litellm_local.py, the one reader of the gateway's old local.env.
set -uo pipefail
export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fail=0
ok()  { echo "  ✓ $*"; }
bad() { echo "  ✗ $*" >&2; fail=1; }

# Files allowed to read or write .env privately until their phase lands.
ALLOWLIST=$(cat <<'EOF'
EOF
)

# ── the detector ────────────────────────────────────────────────────────────
# `.env` as a path of its own: not preceded by a word character, a dot or a dash,
# so imagegen.env, club3090.env and secrets.env never count. `local.env` does: the
# gateway's route keys used to live there (services/litellm/local.env), and they are
# settings now (secrets.env, #1466 4c). Only scripts/lib/litellm_local.py reads it,
# through club_config's parser, to migrate it; docker compose loads it itself. A
# script reading it on its own would miss the saved keys, which win over it.
E='(^|[^A-Za-z0-9_.-])(local)?\.env\b'
SH_PATTERNS=(
  "(source|^[[:space:]]*\\.)[[:space:]]+[^#]*${E}"                      # sourcing it
  '--env-file[= ]+[^/[:space:]]'                                        # docker compose --env-file (not /dev/null)
  'EnvironmentFile='                                                    # systemd
  "(<|>>?)[[:space:]]*\"?[^[:space:]]*${E}"                              # redirecting from or to it
  "\\b(grep|sed|awk|cut|cat|mv|cp|touch|tee)\\b[^|;]*${E}"               # a tool reading or rewriting it
  "^[[:space:]]*(local[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*[[:space:]]+)*[A-Za-z_][A-Za-z0-9_]*=[\"']?[^[:space:]]*[/\"'{](local)?\\.env\\b"  # its path in a variable
)
PY_PATTERNS=(
  "[\"']\\.env[\"']"                                                    # Path(root) / ".env", "--env-file", ".env"
  "[\"']([^\"']*/)?local\\.env[\"']"                                    # "services/litellm/local.env"
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
    # The one sanctioned --env-file: the file the loader writes for `sudo docker
    # compose` (club_config_compose_env_file → $CLUB3090_COMPOSE_ENV_FILE).
    body="${body//--env-file \"\$CLUB3090_COMPOSE_ENV_FILE\"/}"
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
# The gateway's old key file (#1466 4c).
w r10.sh 'k=$(grep -E "^MY_KEY=" "$ROOT/services/litellm/local.env" | cut -d= -f2-)'
w r11.py 'keys = open(os.path.join(root, "services/litellm/local.env")).read()'
w r12.sh 'KEYS_FILE="$ROOT/services/litellm/local.env"'
w n1.sh  '# source "${ROOT_DIR}/.env"   (a comment)'
w n2.sh  'docker compose --env-file /dev/null -f x.yml config'
w n3.sh  'echo ok > "$D/imagegen.env"; cp "$D/club3090.env" "$D/secrets.env" "$D/bak/"'
w n9.sh  'echo "route keys go in secrets.env now, not local.env"'
w n10.py 'p = "mylocal.env"'
w n4.sh  '. "$LIB/club-config.sh"; club_config_load "$ROOT"'
w n5.sh  'echo "[setup] set MODEL_DIR in your config"'
w n6.py  '"""Reads ``<root>/.env`` as a legacy fallback."""'
w n7.py  'p = config_dir() / "club3090.env"'
w n8.sh  'sudo docker compose --env-file "$CLUB3090_COMPOSE_ENV_FILE" -f x.yml up -d'
got="$(cd "$T" && detect r1.sh r2.sh r3.sh r4.sh r5.sh r6.sh r7.service r8.py r9.sh r10.sh r11.py r12.sh n1.sh n2.sh n3.sh n4.sh n5.sh n6.py n7.py n8.sh n9.sh n10.py | tr '\n' ' ')"
want="r1.sh r2.sh r3.sh r4.sh r5.sh r6.sh r7.service r8.py r9.sh r10.sh r11.py r12.sh "
[[ "$got" == "$want" ]] && ok "self-test: detector flags 12 readers (incl. a 20,000-line file and three of the gateway's old local.env) and ignores 10 look-alikes (incl. the sanctioned compose env file)" \
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
  | command grep -vxE 'scripts/lib/club-config\.sh|scripts/lib/club_config\.py|scripts/lib/litellm_local\.py')
[[ ${#FILES[@]} -gt 100 ]] || bad "only ${#FILES[@]} files scanned — the file list is broken"

found="$(detect "${FILES[@]}" | sort)"
allowed="$(awk 'NF {print $1}' <<<"$ALLOWLIST" | sort)"
new="$(comm -23 <(printf '%s\n' "$found") <(printf '%s\n' "$allowed") | command grep -v '^$' || true)"
gone="$(comm -13 <(printf '%s\n' "$found") <(printf '%s\n' "$allowed") | command grep -v '^$' || true)"
if [[ -n "$new" ]]; then
  bad "new private .env parser(s) — read settings with scripts/lib/club-config.sh (bash) or scripts/lib/club_config.py (Python), and write them with club_config_set / club_config.py set:"
  sed 's/^/      /' <<<"$new" >&2
else
  left_n="$(awk 'NF' <<<"$found" | wc -l)"
  if [[ "$left_n" -eq 0 ]]; then
    ok "no private .env parsers left anywhere — every setting goes through the one loader (#1466)"
  else
    ok "no new private .env parsers ($left_n known ones left, each with its #1466 phase)"
  fi
fi
if [[ -n "$gone" ]]; then
  bad "no longer parses .env itself — remove it from the ALLOWLIST in this test (the list only shrinks):"
  sed 's/^/      /' <<<"$gone" >&2
fi

# ── tests never read the real settings ──────────────────────────────────────
# Every launcher now reads ~/.config/club-3090/. A test that inherits the maintainer's
# real settings (model-default pins, MODEL_DIR, …) passes or fails for reasons that
# aren't in the repo. Each test file exports CLUB3090_CONFIG_DIR at the top — a path
# that doesn't exist, so the loader finds no files — and tests that exercise the
# config point it at their own temporary directory per call.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t TESTS < <(git ls-files -- 'scripts/tests/test-*.sh' 'scripts/tests/*/test-*.sh' | sort -u)
else
  mapfile -t TESTS < <(find scripts/tests -name 'test-*.sh' | sort)
fi
missing="$(for t in "${TESTS[@]}"; do command grep -qE '^export CLUB3090_CONFIG_DIR=' "$t" || echo "$t"; done)"
if [[ -n "$missing" ]]; then
  bad "test(s) that could read your real settings — add, right after the 'set -…' line:"
  echo "      export CLUB3090_CONFIG_DIR=/nonexistent/club-3090-test-config   # tests never read your real settings (#1466)" >&2
  sed 's/^/      /' <<<"$missing" >&2
else
  ok "all ${#TESTS[@]} test files isolate CLUB3090_CONFIG_DIR"
fi

[[ $fail -eq 0 ]] && echo "test-config-single-parser: ok" || echo "test-config-single-parser: FAIL"
exit $fail
