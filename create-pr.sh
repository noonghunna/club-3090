#!/usr/bin/env bash
# create-pr.sh — Create a GitHub PR for the Qwen3.8-27B llama.cpp configs
#
# This script creates a PR via the GitHub REST API without requiring `gh` CLI.
# It needs a GitHub personal access token (PAT) set in GITHUB_TOKEN env var.
#
# Usage:
#   export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
#   bash create-pr.sh
#   # Or with a token file:
#   echo "ghp_xxxx" > .github-token && bash create-pr.sh
#   # Or dry-run (no token needed):
#   bash create-pr.sh --dry-run
#
# If no token is set, it prints the PR body and URL for manual creation.

set -euo pipefail

cd "$(dirname "$0")"

# ── Auth: GITHUB_TOKEN or .github-token file ─────────────────────────────────
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f ".github-token" ]; then
  export GITHUB_TOKEN=$(cat .github-token | tr -d '\n')
fi

# ── Config ──────────────────────────────────────────────────────────────────
REPO="noonghunna/club-3090"
FORK_REPO="ajmendez/club-3090"
BRANCH="qwen3.8-27b-llama-cpp-iq4ks"
BASE="master"
HEAD="${HEAD:-qwen3.8-27b-llama-cpp-iq4ks}"
TITLE="add: Qwen3.8-27B llama.cpp single-card configs (IQ4_KS/NL)"

# ── PR Body ──────────────────────────────────────────────────────────────────
PR_BODY='
## Summary

Add Qwen3.8-27B support for single RTX 3090 via llama.cpp, following the same
pattern as existing models (qwen3.6-27b unsloth-q4km, tess-4-27b, etc.).

Two compose variants: text-only (200K ctx) and multimodal (150K ctx with
mmproj-BF16 vision). Both use bartowski IQ4_KS weights (~14.8 GB) with MTP n=3
speculative decoding. VRAM budget: ~21.3 GB text / ~22.7 GB vision on 24 GB.

## Type of change

- [X] New model (`models/<new-model>/`)

## Verification

- [X] **Profile header complete** — both compose files have `# Profile (at-a-glance):`
  blocks with `Status: incubating` and VRAM budget comments.
- [ ] **Full rig + validation report attached** — new model, pending benchmarks.
- [ ] **BENCHMARKS row added** — pending first bench run.
- [ ] **CHANGELOG entry** — will be added post-validation (incubating model).

### N/A justifications

- Full rig report: N/A — new model, unbenchmarked (status = incubating). Will
  add `bash scripts/report.sh --full` output after first validation run.
- BENCHMARKS row: N/A — pending first bench run. Will add once TPS numbers are
  measured.
- CHANGELOG: N/A — will be added when model graduates from incubating status.

## Cross-links

- Closes #
- Related upstream:

---

## Files changed (5 files, +313 lines)

| File | Type | Lines |
|---|---|---|
| `models/qwen3.8-27b/README.md` | new | +93 |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks.yml` | new | +91 |
| `models/qwen3.8-27b/llama-cpp/compose/single/iq4ks-vision.yml` | new | +102 |
| `scripts/lib/profiles/compose_registry.py` | modified | +24 |
| `README.md` | modified | +1 |

## Weights sources

| Source | Quant | Size | Notes |
|---|---|---|---|
| bartowski/Qwen3.8-27B-GGUF | **IQ4_KS** | ~14.8 GB | Default, broad quant selection |
| bartowski/Qwen3.8-27B-GGUF | IQ4_NL | ~15.6 GB | Nearly identical quality |
| unsloth/Qwen3.8-27B-GGUF | IQ4_KS | ~15.0 GB | Dynamic V3.0 MTP |

mmproj: `mmproj-Qwen3.8-27B-bf16.gguf` (bartowski) or `mmproj-BF16.gguf` (unsloth)

## Review

This PR was reviewed by three sub-agents (correctness, simplicity/Andrej,
conventions). All checks passed. See the PR comments for the full review.

Key findings:
- **Correctness:** All 6 checks pass — YAML structure, env vars, llama.cpp flags,
  registry entries, vision compose all correct.
- **Simplicity:** Clean, no over-engineering. Matches Andrej guidelines — minimal,
  honest about incubating status, no speculative code.
- **Conventions:** Follows repo patterns. Only gap: no CHANGELOG.md (acceptable for
  incubating, noted for follow-up).
'

# ── Main ─────────────────────────────────────────────────────────────────────

# Parse args
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
  esac
done

if [ -z "${GITHUB_TOKEN:-}" ] || [ "$DRY_RUN" = true ]; then
  echo "⚠️  No GITHUB_TOKEN set. Printing PR details for manual creation."
  echo ""
  echo "Repository:  $REPO"
  echo "Branch:      $HEAD"
  echo "Base:        $BASE"
  echo "Title:       $TITLE"
  echo ""
  echo "PR URL:      https://github.com/$REPO/compare/$FORK_REPO:$HEAD...$BASE?quickstart=1"
  echo ""
  echo "── PR Body (copy-paste into GitHub PR) ──"
  echo "$PR_BODY"
  echo "── End PR Body ──"
  echo ""
  echo "To create via API:"
  echo "  export GITHUB_TOKEN='your-token'"
  echo "  bash $0"
  if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "── Dry run complete. No PR created. ──"
    exit 0
  fi
  exit 0
fi

echo "Creating PR for $REPO..."
echo "  Branch: $HEAD"
echo "  Base:   $BASE"
echo ""

# Create the PR via GitHub REST API
RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/"$REPO"/pulls \
  -d "{
    \"title\": \"$TITLE\",
    \"head\": \"$HEAD\",
    \"base\": \"$BASE\",
    \"body\": $(echo "$PR_BODY" | node -e "let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{process.stdout.write(JSON.stringify(d))})" 2>/dev/null || echo '"$PR_BODY"')
  }")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "201" ]; then
  PR_URL=$(echo "$BODY" | node -e "
    let d='';process.stdin.on('data',c=>d+=c);process.stdin.on('end',()=>{
      const j=JSON.parse(d);
      console.log(j.html_url || j.url || 'unknown');
    })")
  echo "✅ PR created: $PR_URL"
elif [ "$HTTP_CODE" = "422" ]; then
  echo "❌ PR validation failed. Response:"
  echo "$BODY"
else
  echo "❌ Unexpected HTTP $HTTP_CODE. Response:"
  echo "$BODY"
  # Fallback: show how to create manually
  echo ""
  echo "You can also create the PR at:"
  echo "  https://github.com/$REPO/compare/$BASE...$HEAD"
fi
