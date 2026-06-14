#!/usr/bin/env bash
# #320: resolve-check that each hf_repo in the weights registry actually exists
# on HF (HTTP via the HF API), not just string-matches the profile. Catches the
# renamed / never-existed repo class (#316) that the string-only guard in
# test-model-weights-registry.sh cannot see.
#
# Opt-in + network-gated: runs only when CLUB3090_CHECK_HF_REPOS=1, so the
# default `for t in scripts/tests/*.sh` sweep stays offline-safe and CI-green.
# Run it manually or periodically (HF rate-limits, so not every commit).
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${CLUB3090_CHECK_HF_REPOS:-0}" != "1" ]]; then
  echo "test-hf-repos-resolve: skipped (set CLUB3090_CHECK_HF_REPOS=1 to run the network check)"
  exit 0
fi

# Resolve an HF repo via the API, following redirects the way `hf download` does
# (-L), and report "<final_code> <num_redirects> <effective_url>". 200 exists,
# 401/403 exists-but-gated, 404 missing/renamed (the bug we catch), other =
# network/rate-limit (ignored). A 200 reached only via redirect means the repo
# was renamed/recased upstream and still works today, but is one dropped redirect
# away from the #316 404, so it is worth flagging. Calls curl, so tests stub it by
# prepending a fake curl to PATH.
hf_api_status() {
  curl -sL -o /dev/null -w '%{http_code} %{num_redirects} %{url_effective}' \
    --max-time "${CLUB3090_HF_RESOLVE_TIMEOUT:-15}" \
    "https://huggingface.co/api/models/$1"
}

mapfile -t repos < <(
  python3 - "$ROOT_DIR/scripts/lib/profiles/models" <<'PY'
import glob, os, sys
try:
    import yaml
except Exception:
    sys.exit(0)
repos = set()
for path in glob.glob(os.path.join(sys.argv[1], "*.yml")):
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for meta in (data.get("weights") or {}).values():
        if not isinstance(meta, dict):
            continue
        if meta.get("hf_repo"):
            repos.add(str(meta["hf_repo"]))
        for r in (meta.get("hf_repos") or []):
            repos.add(str(r))
for r in sorted(repos):
    print(r)
PY
)

if [[ "${#repos[@]}" -eq 0 ]]; then
  echo "test-hf-repos-resolve: no hf_repo entries found (PyYAML missing?)" >&2
  exit 1
fi

ok=0 gated=0 redirected=0 warn=0 fail=0
for repo in "${repos[@]}"; do
  read -r code redirects url <<< "$(hf_api_status "$repo" || true)"
  redirects="${redirects:-0}"
  canon="${url#https://huggingface.co/api/models/}"
  case "$code" in
    200)
      if [[ "$redirects" != "0" ]]; then
        printf '  %-58s OK via redirect -> %s (200)\n' "$repo" "$canon"
        redirected=$((redirected + 1))
      else
        printf '  %-58s OK (200)\n' "$repo"; ok=$((ok + 1))
      fi
      ;;
    401|403) printf '  %-58s gated, exists (%s)\n' "$repo" "$code"; gated=$((gated + 1)) ;;
    404)     printf '  %-58s MISSING/RENAMED (404)\n' "$repo" >&2; fail=$((fail + 1)) ;;
    *)       printf '  %-58s network/rate-limit (%s), not counted\n' "$repo" "$code" >&2; warn=$((warn + 1)) ;;
  esac
done

echo "test-hf-repos-resolve: ok=${ok} redirected=${redirected} gated=${gated} warn=${warn} fail=${fail}"
if [[ "$redirected" -ne 0 ]]; then
  echo "test-hf-repos-resolve: ${redirected} repo(s) resolve only via an upstream redirect" >&2
  echo "  (rename/recase). They work today but will 404 if the redirect is dropped." >&2
  echo "  Consider updating hf_repo to the canonical name shown above." >&2
fi
if [[ "$fail" -ne 0 ]]; then
  echo "test-hf-repos-resolve: ${fail} repo(s) do not resolve on HF (404). A renamed or" >&2
  echo "  re-quantized upstream repo will 404 a fresh setup.sh even though the string" >&2
  echo "  still matches the profile. Fix the hf_repo (and pin a revision: per #319)." >&2
  exit 1
fi
