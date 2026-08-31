#!/usr/bin/env bash
# test-compose-entrypoint-env-declared.sh — an entrypoint may only read env vars
# that docker actually forwards.
#
# THE CLASS: docker forwards ONLY what `environment:` declares. An entrypoint that
# branches on `$${FOO}` when FOO is undeclared reads an ALWAYS-EMPTY value, so the
# knob is inert while the compose still documents it. Nothing errors; the surface
# looks correct. Four instances found so far (THREADS, -np/NPARALLEL,
# shmem_enabled, VLLM_ENFORCE_EAGER) — see club-3090-todo.md 2026-08-30/31.
#
# ⚠️ Declaring it as `- FOO=${FOO:-}` does NOT fix it: that sets FOO to the EMPTY
# STRING, which differs from unset and, for numeric knobs, atoi("") == 0 silently
# DISABLES the feature. Use the bare form `- FOO`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUTF8="${PYTHONUTF8:-1}"

python3 - <<'PY'
import io,re,subprocess,sys

# Known-broken, tracked, NOT yet fixed. This is a DEBT REGISTER, not an excuse
# list: each entry must name the bug. Shrink it; never grow it silently.
KNOWN = {
    # 15 vLLM composes: `$${VLLM_ENFORCE_EAGER:+--enforce-eager}` never fires because
    # the var is undeclared. The header telling users to set it "in compose/.env" is
    # also wrong — .env feeds compose interpolation, not container env.
    "VLLM_ENFORCE_EAGER",
    # GLM moecache: the entrypoint's projector-existence check reads
    # `$${MMPROJ:-<default>}` while the command arg uses compose-level
    # `${MMPROJ:-<default>}`. Override MMPROJ and the guard tests the WRONG path.
    "MMPROJ",
}
RUNTIME = {"PATH","HOME","HOSTNAME","LD_LIBRARY_PATH","PWD","IFS"}

files=[f for f in subprocess.run(['git','ls-files','models/'],capture_output=True,
       text=True,encoding='utf-8').stdout.split()
       if f.endswith('.yml') and '/_archive/' not in f]
fails=[]; debt=0; scanned=0
for f in files:
    raw=io.open(f,encoding='utf-8').read()
    if 'entrypoint:' not in raw: continue
    scanned+=1
    # Strip comment-only lines first: compose headers discuss `$${VAR:-x}` as
    # prose, and scanning them invents knobs that no code ever reads.
    t='\n'.join(l for l in raw.split('\n') if not l.lstrip().startswith('#'))
    read=set(re.findall(r'\$\$\{([A-Z][A-Z0-9_]{2,})[:\-\}]', t))
    env=set(re.findall(r'^\s+-\s+([A-Z][A-Z0-9_]*)(?:=|\s*$)', t, re.M))
    assigned=set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)=', t, re.M)) | \
             set(re.findall(r'export\s+([A-Z][A-Z0-9_]*)=', t))
    missing = read - env - assigned - RUNTIME
    for v in sorted(missing):
        if v in KNOWN: debt+=1
        else: fails.append((f,v))

if scanned == 0:
    print("FAIL: scanned no composes with entrypoints — the search is wrong, not the tree", file=sys.stderr)
    raise SystemExit(1)
for f,v in fails:
    bare = "- " + v
    trap = "- " + v + "=${" + v + ":-}"
    print("FAIL: " + f + " entrypoint reads $" + v + " but it is not declared in environment: "
          "- docker will never forward it, so the knob is inert. "
          "Add the BARE form `" + bare + "` (never `" + trap + "`, which sets it EMPTY).", file=sys.stderr)
if fails:
    print(f"{len(fails)} undeclared entrypoint env var(s)", file=sys.stderr)
    raise SystemExit(1)
print(f"PASS: {scanned} composes with entrypoints; every env var they read is declared "
      f"({debt} known-broken occurrence(s) still on the debt register)")
PY
