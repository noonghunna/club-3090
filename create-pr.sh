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
mmproj vision). Both target bartowski 4-bit GGUFs with MTP n=3 speculative
decoding. Note: this arch ships **no IQ4_KS** from bartowski — the 4-bit
default is IQ4_NL (~16.3 GB); the `iq4ks` slug names are a quant-label
placeholder, `GGUF_FILE`/`MMPROJ_FILE` env overrides select the actual file.

## Type of change

- [X] New model (`models/<new-model>/`)

## Benchmarks (measured, 2026-08-16, single 3090)

**63.2 narr / 71.9 code TPS** (wall, n=5, CV 2.7%/1.7%), TTFT 150 ms,
boot ~22 GB, `server-cuda-b9246` + MTP n=2 + q4_0 KV. Directional ~+25%
narrative over the Qwen3.6-27B `llamacpp/mtp` Q4_K_M reference (50.27/58.92),
**not a canonical-prompt comparison** (this bench: max_tokens=200 short prompts
vs bench.sh 1000/800) — flagged in the BENCHMARKS row for a bench.sh re-run.

⚠️ **The bench rig is NOT the compose defaults.** It ran **unsloth Q4_K_M
(17.8 GB) + mmproj-F16 at CTX_SIZE=131072** — Q4_K_M OOMs at 200K on 24 GB
(weights 17.8 + mmproj 0.93 + q4_0 KV 3.7 GB ≈ 22.4+ GB). The compose default
200K default only fits with IQ4_NL (16.3 GB) and stays incubating until that
default path is measured. No power-cap A/B, verify-stress, or 8-pack yet.

## Verification

- [X] **Profile header complete** — both compose files have `# Profile (at-a-glance):`
  blocks with `Status: incubating` and VRAM budget comments.
- [ ] **Full rig + validation report attached** — bench.sh full pass pending
  (first data point is in BENCHMARKS.md, non-canonical protocol).
- [X] **BENCHMARKS row added** — first Qwen3.8-27B single-3090 row (2026-08-16),
  honest about rig-vs-compose drift and non-canonical protocol.
- [ ] **CHANGELOG entry** — will be added post-validation (incubating model).

### N/A justifications

- Full rig report: pending — `bash scripts/report.sh --full` after the
  default-config (IQ4_NL) bench; first data point already published.
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
| bartowski/Qwen3.8-27B-GGUF | **IQ4_NL** | ~16.3 GB | 4-bit default (no IQ4_KS for this arch) |
| bartowski/Qwen3.8-27B-GGUF | Q4_K_M | ~17.8 GB | 200K OOMs on 24 GB — 131K cap (measured) |
| unsloth/Qwen3.8-27B-GGUF | Q4_K_M | ~17.8 GB | **Bench-validated rig** — 63/72 TPS @ 131K + vision |

mmproj: `mmproj-Qwen3.8-27B-bf16.gguf` (bartowski) or `mmproj-Qwen3.8-27B-f16.gguf`
(unsloth — bench rig)

## Review

This PR was reviewed by three sub-agents (correctness, simplicity/Andrej,
conventions). All checks passed.

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
