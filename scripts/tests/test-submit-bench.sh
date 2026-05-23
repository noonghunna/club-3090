#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=../lib/bench-row-formatter.sh
source "$ROOT_DIR/scripts/lib/bench-row-formatter.sh"

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "ASSERTION FAILED: expected output to contain: $needle" >&2
    echo "--- output ---" >&2
    echo "$haystack" >&2
    exit 1
  fi
}

assert_columns() {
  local row="$1"
  local expected="$2"
  local cols
  cols="$(awk -F'|' '{print NF - 2}' <<< "$row")"
  if [[ "$cols" != "$expected" ]]; then
    echo "ASSERTION FAILED: expected $expected columns, got $cols" >&2
    echo "$row" >&2
    exit 1
  fi
}

fixtures=()
while IFS= read -r fixture; do
  fixtures+=("$fixture")
done < <(bench_row_fixtures)

if [[ "${#fixtures[@]}" -lt 6 ]]; then
  echo "ASSERTION FAILED: expected at least 6 submit-bench fixtures, found ${#fixtures[@]}" >&2
  exit 1
fi

for dir in "${fixtures[@]}"; do
  section="$(bench_row_section "$dir")"
  row="$(bench_row_format "$dir")"
  [[ "$row" == \|*\| ]] || {
    echo "ASSERTION FAILED: row is not a markdown table row for $dir" >&2
    echo "$row" >&2
    exit 1
  }
  assert_contains "$row" "Report: \`results/rebench/${dir##*/}/REPORT.md\`"
  if [[ "$section" == "Gemma 4 31B (community-experimental)" ]]; then
    assert_columns "$row" 11
  else
    assert_columns "$row" 9
  fi
done

out="$(BENCH_MOCK=1 RUNS=1 WARMUPS=0 bash scripts/bench.sh)"
assert_contains "$out" "PP tok/s"
out="$(ENABLE_THINKING=1 BENCH_MOCK=1 RUNS=1 WARMUPS=0 bash scripts/bench.sh 2>&1)"
assert_contains "$out" "[bench] thinking: enabled"
assert_contains "$out" "PP tok/s"
out="$(BENCH_MOCK=1 PP=1 RUNS=1 WARMUPS=0 bash scripts/bench.sh)"
assert_contains "$out" "summary [prompt-processing]"
assert_contains "$out" "PP tok/s"

tag="qwen-int8-pth-n4-2026-05-10"
rm -f "results/rebench/${tag}/BENCHMARKS-row.md" \
      "results/rebench/${tag}/PR-body.md" \
      "results/rebench/${tag}/ISSUE-body.md" \
      "results/rebench/${tag}/auto-submit-mock.log"

out="$(bash scripts/submit-bench.sh --tag "$tag")"
assert_contains "$out" "Generated BENCHMARKS row for section: Dual-card (2× RTX 3090, TP=2)"
assert_contains "$out" "Wrote: results/rebench/${tag}/BENCHMARKS-row.md"
assert_contains "$out" "1. Issue + maintainer integrates"
assert_contains "$out" "2. Direct PR"
assert_contains "$out" "3. Manual edit"
test -s "results/rebench/${tag}/BENCHMARKS-row.md"

if out="$(bash scripts/submit-bench.sh --tag does-not-exist 2>&1)"; then
  echo "ASSERTION FAILED: missing tag unexpectedly succeeded" >&2
  exit 1
fi
assert_contains "$out" "tag dir not found: results/rebench/does-not-exist"

out="$(GH_MOCK=1 GH_MOCK_USER=octocat bash scripts/submit-bench.sh --tag "$tag" --auto-submit)"
assert_contains "$out" "Issue title: [bench] @octocat ${tag}"
test -s "results/rebench/${tag}/auto-submit-mock.log"
assert_contains "$(cat "results/rebench/${tag}/auto-submit-mock.log")" "gh issue create --title [bench] @octocat ${tag}"
test -s "results/rebench/${tag}/ISSUE-body.md"
assert_contains "$(cat "results/rebench/${tag}/ISSUE-body.md")" "results/rebench/${tag}/REPORT.md"
assert_contains "$(cat "results/rebench/${tag}/ISSUE-body.md")" "Proposed BENCHMARKS.md row"

out="$(GH_MOCK=1 GH_MOCK_USER=octocat bash scripts/submit-bench.sh --tag "$tag" --auto-submit --as-pr)"
assert_contains "$out" "PR title: bench(matrix): @octocat ${tag}"
test -s "results/rebench/${tag}/PR-body.md"
assert_contains "$(cat "results/rebench/${tag}/auto-submit-mock.log")" "gh pr create --title bench(matrix): @octocat ${tag}"
assert_contains "$(cat "results/rebench/${tag}/PR-body.md")" "results/rebench/${tag}/REPORT.md"

tmp_bin="$(mktemp -d)"
trap 'rm -rf "$tmp_bin"; rm -f "results/rebench/${tag}/BENCHMARKS-row.md" "results/rebench/${tag}/PR-body.md" "results/rebench/${tag}/ISSUE-body.md" "results/rebench/${tag}/auto-submit-mock.log"' EXIT
cat > "${tmp_bin}/gh" <<'MOCK_GH'
#!/usr/bin/env bash
if [[ "$1" == "auth" && "$2" == "status" ]]; then
  exit 1
fi
echo "unexpected gh call: $*" >&2
exit 2
MOCK_GH
chmod +x "${tmp_bin}/gh"

if out="$(PATH="${tmp_bin}:${PATH}" bash scripts/submit-bench.sh --tag "$tag" --auto-submit 2>&1)"; then
  echo "ASSERTION FAILED: unauthenticated gh path unexpectedly succeeded" >&2
  exit 1
fi
assert_contains "$out" "not authed with gh. Run: gh auth login"

echo "test-submit-bench: ok"
