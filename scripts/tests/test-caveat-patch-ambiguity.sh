#!/usr/bin/env bash
# Guard for #1147 — a compose caveat that promises an upstream fix, while the
# compose ships OUR OWN patch in the same fault thread.
#
# THE DEFECT THIS CATCHES (worked example: #1146). dual/fp8/mtp.yml's risk block
# was written 2026-08-21 and said the MTP x hybrid-GDN wild write "clears when
# #50021 merges and a pin bump inherits it". Two days later (#1091) that same
# compose started mounting `vllm-gdn-mtp-async-spec-order` — our own fix, in the
# same fault thread, which #50021 does NOT supersede and which does NOT bound
# #50021's index. Nobody re-read the block. For ten days users were told to
# disable async scheduling (a real throughput cost) for a fault the shipped
# compose already partly closed, and to expect a merge that would not have helped.
#
# THE RULE. Flag a (compose, patch) pair when ALL of:
#   1. the compose header promises resolution via upstream PR #N
#      ("clears when #N merges", "#N merges", ...), AND
#   2. the compose mounts a patch whose patches.yml entry is
#      `upstream.status: local-vendored` — i.e. OUR fix, not a vendoring of #N,
#      AND
#   3. that patch's entry references #N (same fault thread)
# ...and the header does NOT name the patch.
#
# Naming the patch in the header IS the fix: if we ship our own fix for the
# thread a user is told to wait on, say so where they read. That forces the
# disambiguation (#1146: "a #50021 merge clears (b) ONLY; it never covered (a)").
#
# WHY NOT the simpler rules that were tried first:
#   - "every mounted patch must be named in a header" -> 93 mounted instances,
#     8 named: 28 composes of churn, most for plumbing that needs no prose.
#   - "flag if any patch mount is newer than the caveat" (git recency) -> fires on
#     unrelated patch edits; a 2026-07-18 chat-template repoint tripped two gemma
#     composes whose promises were fine.
#   `upstream.status` is the discriminator: `open`/`merged`/`open-rebased-local`
#   patches VENDOR the PR, so "clears when it merges" is legitimately true.
#   Only `local-vendored` (ours, distinct from the PR) creates the ambiguity.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 - <<'PY'
import re, sys, pathlib

PATCHES = pathlib.Path("scripts/lib/profiles/patches.yml")
if not PATCHES.exists():
    print("test-caveat-patch-ambiguity: SKIP — patches.yml not found"); sys.exit(0)

# id -> (upstream.status, {referenced PR numbers})
meta = {}
for e in re.split(r"\n(?=  - id: )", PATCHES.read_text(encoding="utf-8")):
    m = re.search(r"  - id: (\S+)", e)
    if not m:
        continue
    up = re.search(r"upstream:\s*\n(?:.*\n)*?\s+status: \"?([\w-]+)\"?", e)
    prs = {a or b for a, b in re.findall(r"vllm#(\d+)|pull/(\d+)", e)}
    meta[m.group(1)] = (up.group(1) if up else "", prs)

# "clears when #N merges" / "#N merges" — a forward-looking resolution promise
PROMISE = re.compile(r"[Cc]lears? when.{0,80}?#(\d+)|#(\d+).{0,40}?merges")
OURS = {"local-vendored"}          # our fix, NOT a vendoring of the promised PR

fails = []
for f in sorted(pathlib.Path("models").glob("*/*/compose/**/*.yml")):
    if "_archive" in str(f):
        continue
    text = f.read_text(encoding="utf-8", errors="replace")
    head = "\n".join(l for l in text.splitlines() if l.startswith("#"))
    promised = {a or b for a, b in PROMISE.findall(head) if (a or b)}
    if not promised:
        continue
    for patch in sorted(set(re.findall(r"/patches/([a-z0-9\-]+):", text))):
        status, prs = meta.get(patch, ("", set()))
        shared = promised & prs
        if status in OURS and shared and patch not in head:
            fails.append((str(f), patch, sorted(shared)))

if fails:
    print("test-caveat-patch-ambiguity: FAIL\n")
    for path, patch, shared in fails:
        prs = ", ".join("#" + p for p in shared)
        print(f"  {path}")
        print(f"    promises {prs} will clear the risk, but ships OUR patch "
              f"'{patch}'")
        print(f"    (patches.yml: upstream.status=local-vendored, same thread as {prs})")
        print(f"    -> the header never names the patch, so a reader cannot tell "
              f"what is already fixed.")
        print(f"    FIX: name '{patch}' in the risk block and state what {prs} "
              f"would still clear.\n")
    print("See #1147 (the guard) and #1146 (the worked example).")
    sys.exit(1)

print("test-caveat-patch-ambiguity: PASS — no compose promises an upstream fix "
      "while silently shipping our own patch for the same thread")
PY
