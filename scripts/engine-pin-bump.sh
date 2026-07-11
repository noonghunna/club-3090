#!/usr/bin/env bash
# engine-pin-bump — the MECHANICAL half of an engine image-pin bump.
#
# Rewrites the places where a pin is pure repetition, and REPORTS (never edits)
# the places where the pin string encodes a validation claim:
#
#   EDITS                                          REPORTS (hand attention)
#   ─────                                          ────────────────────────
#   engines/<engine>.yml  install.spec             arch_patches.yml engine_pin rows (needs a
#   engines/<engine>.yml  display_name (old tag)     new `loads: true` row — that's a CLAIM,
#   every registered FUNCTIONAL compose's            only a live boot can make it)
#     `image:` default carrying the old ref        pin-asserting test fixtures (they assert
#                                                    what the resolver EMITS — update after
#                                                    the resolver is right)
#                                                  docs/UPSTREAM.md engine-pin row
#                                                  prose mentions (engine notes:, compose
#                                                    headers, docs) — history, not config
#
# Deprecated/non-functional composes are skipped (their defaults are dead weight;
# launchers inject the engine-profile image anyway) and listed.
#
# The judgment half of a bump lives in /opt/ai/CLAUDE.md "When upgrading an engine
# pin" — re-validating vendored patches on the real image, live-boot + 3-warm-up
# validation, verify-full, trackers. This script prints that checklist; it cannot
# run it for you.
#
# Usage:
#   scripts/engine-pin-bump.sh <engine-id> <new-tag-or-full-ref> [--check]
#   scripts/engine-pin-bump.sh vllm-stable v0.25.0
#   scripts/engine-pin-bump.sh vllm-stable vllm/vllm-openai:v0.25.0 --check
#   scripts/engine-pin-bump.sh llama-cpp-local server-cuda-b9967 --check
#
#   <engine-id> is the profile `id:` (NOT the filename — llama-cpp-mainline.yml
#   has id `llama-cpp-local`). Digest-pinned engines (beellama) require a full
#   `repo@sha256:...` ref; for those prefer scripts/beellama-pin-bump.sh which
#   also discovers the newest tag.
#
#   --check  dry-run: print the would-be diff and the hand-attention report, write nothing.
#
# Exit: 0 on success/clean check; 2 on refusal (unknown engine, no-op, malformed ref).
set -euo pipefail

ROOT="${PIN_BUMP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

ENGINE_ID="" NEW_REF_ARG="" CHECK=0
for a in "$@"; do
  case "$a" in
    --check) CHECK=1 ;;
    --help|-h) command grep -E '^#( |$)' "${BASH_SOURCE[0]}" | cut -c3-; exit 0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) if [[ -z "$ENGINE_ID" ]]; then ENGINE_ID="$a"; elif [[ -z "$NEW_REF_ARG" ]]; then NEW_REF_ARG="$a"; else echo "too many args" >&2; exit 2; fi ;;
  esac
done
[[ -n "$ENGINE_ID" && -n "$NEW_REF_ARG" ]] || { echo "usage: engine-pin-bump.sh <engine-id> <new-tag-or-ref> [--check]" >&2; exit 2; }

PIN_ROOT="$ROOT" PIN_ENGINE="$ENGINE_ID" PIN_NEW="$NEW_REF_ARG" PIN_CHECK="$CHECK" python3 - <<'PY'
import difflib, io, os, subprocess, sys

ROOT = os.environ["PIN_ROOT"]
ENGINE = os.environ["PIN_ENGINE"]
NEW_ARG = os.environ["PIN_NEW"]
CHECK = os.environ["PIN_CHECK"] == "1"

def die(msg):
    print(f"engine-pin-bump: {msg}", file=sys.stderr)
    sys.exit(2)

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

# ── locate the engine profile by id: (filename != id) ──────────────────────
eng_dir = os.path.join(ROOT, "scripts/lib/profiles/engines")
eng_path, ids = None, []
for name in sorted(os.listdir(eng_dir)):
    if not name.endswith(".yml"):
        continue
    p = os.path.join(eng_dir, name)
    for line in read(p).splitlines():
        if line.startswith("id: "):
            eid = line[4:].strip()
            ids.append(eid)
            if eid == ENGINE:
                eng_path = p
            break
if eng_path is None:
    die(f"no engine profile with id '{ENGINE}' (known: {', '.join(sorted(ids))})")

eng_text = read(eng_path)
spec_lines = [l for l in eng_text.splitlines() if l.lstrip().startswith("spec:")]
if len(spec_lines) != 1:
    die(f"{os.path.relpath(eng_path, ROOT)}: expected exactly one 'spec:' line, found {len(spec_lines)}")
old_ref = spec_lines[0].split("spec:", 1)[1].strip()

# ── resolve the new ref ─────────────────────────────────────────────────────
digest_pin = "@" in old_ref
if digest_pin:
    if "@" not in NEW_ARG:
        die(f"'{ENGINE}' is digest-pinned ({old_ref}); pass a full repo@sha256:... ref "
            f"(or use scripts/beellama-pin-bump.sh which discovers it)")
    new_ref = NEW_ARG
    old_tag = old_ref.split("@", 1)[1]
    new_tag = new_ref.split("@", 1)[1]
else:
    repo, old_tag = old_ref.rsplit(":", 1)
    new_ref = NEW_ARG if ("/" in NEW_ARG or ":" in NEW_ARG) else f"{repo}:{NEW_ARG}"
    if ":" not in new_ref:
        die(f"can't parse new ref '{new_ref}'")
    new_tag = new_ref.rsplit(":", 1)[1]
if new_ref == old_ref:
    die(f"no-op: '{ENGINE}' already pinned to {old_ref}")

print(f"engine : {ENGINE}  ({os.path.relpath(eng_path, ROOT)})")
print(f"old pin: {old_ref}")
print(f"new pin: {new_ref}")
print()

# ── registered composes for this engine ─────────────────────────────────────
sys.path.insert(0, os.path.join(ROOT, "scripts/lib/profiles"))
import compose_registry as cr

def field(e, name, default=None):
    d = e._asdict() if hasattr(e, "_asdict") else (vars(e) if hasattr(e, "__dict__") else e)
    return d.get(name, default)

# Skip ONLY deprecated composes: everything else (production/caveats/experimental/
# preview/incubating/upstream-gated) is a bootable config whose compose default is
# load-bearing for anyone running `docker compose up` directly (no launcher injection).
targets, skipped = [], []
for slug, e in sorted(cr.COMPOSE_REGISTRY.items()):
    if field(e, "engine") != ENGINE:
        continue
    status = field(e, "status", "production")
    if status == "deprecated":
        skipped.append((slug, status))
    else:
        targets.append((slug, field(e, "compose_path"), status))

# ── compute edits (exact string replace; image: lines only in composes) ─────
edits = []      # (relpath, new_text, n_replacements)
warnings = []

def plan(path, new_text, n):
    edits.append((os.path.relpath(path, ROOT), new_text, n))

# engine yml: spec line + display_name tag echo; anything else is prose → report
lines, n_spec, n_disp = eng_text.splitlines(keepends=True), 0, 0
for i, l in enumerate(lines):
    if l.lstrip().startswith("spec:") and old_ref in l:
        lines[i] = l.replace(old_ref, new_ref); n_spec += 1
    elif l.startswith("display_name:") and old_tag in l:
        lines[i] = l.replace(old_tag, new_tag); n_disp += 1
    elif l.startswith("display_name:"):
        warnings.append(f"display_name in {os.path.relpath(eng_path, ROOT)} doesn't carry '{old_tag}' — hand-check it")
plan(eng_path, "".join(lines), n_spec + n_disp)

seen_paths = set()
for slug, rel, status in targets:
    path = os.path.join(ROOT, rel)
    if rel in seen_paths:
        continue
    seen_paths.add(rel)
    if not os.path.exists(path):
        warnings.append(f"{slug}: registered compose missing on disk: {rel}")
        continue
    clines, n = read(path).splitlines(keepends=True), 0
    for i, l in enumerate(clines):
        if "image:" in l and old_ref in l:
            clines[i] = l.replace(old_ref, new_ref); n += 1
    if n == 0:
        warnings.append(f"{slug}: {rel} has no image: line carrying {old_ref} — default drifted? hand-check")
    else:
        plan(path, "".join(clines), n)

# ── apply or diff ────────────────────────────────────────────────────────────
print(f"{'would edit' if CHECK else 'edited'} {len(edits)} file(s):")
for rel, new_text, n in edits:
    print(f"  {rel}  ({n} line{'s' if n != 1 else ''})")
    if CHECK:
        old_text = read(os.path.join(ROOT, rel))
        for dl in difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                       fromfile=f"a/{rel}", tofile=f"b/{rel}"):
            sys.stdout.write("    " + dl if not dl.endswith("\n") else "    " + dl)
    else:
        with open(os.path.join(ROOT, rel), "w", encoding="utf-8") as f:
            f.write(new_text)
if skipped:
    print(f"\nskipped {len(skipped)} deprecated compose(s):")
    for slug, status in skipped:
        print(f"  {slug}  [{status}]")
for w in warnings:
    print(f"\n⚠ {w}")

# ── hand-attention report: every OTHER occurrence of the old pin ────────────
print("\n── hand attention (pin strings that encode claims — NOT auto-edited) ──")
grep = subprocess.run(
    ["grep", "-rnF", "--exclude-dir=.git", old_tag,
     "scripts", "docs", "models", "services", "c3"],
    cwd=ROOT, capture_output=True, text=True)
buckets = {
    "fixtures — update to the new tag IN THIS BUMP (they assert resolver output)": [],
    "arch_patches.yml — add an engine_pin loads:true row AFTER live boot; keep old row": [],
    "docs/UPSTREAM.md — update the engine-pin row in this bump": [],
    "hand-written launcher/setup suggestion strings — repoint if stale": [],
}
PROSE = "prose / history (compose headers, baselines quality_env, docs, registry notes) — usually LEAVE"
prose_files = {}
edited = {rel for rel, _, _ in edits}
n_hits = 0
for line in grep.stdout.splitlines():
    rel, _, content = line.partition(":")
    if rel == "scripts/engine-pin-bump.sh" or rel.startswith("scripts/tests/test-engine-pin-bump"):
        continue
    # in --check mode the edits haven't landed, so the lines we WOULD edit still
    # carry the old pin — don't report those as hand-attention
    if CHECK and rel in edited and \
            any(k in content for k in ("image:", "spec:", "display_name:")):
        continue
    n_hits += 1
    if rel.startswith("scripts/tests/"):
        buckets[next(iter(buckets))].append(line)
    elif rel == "scripts/lib/profiles/arch_patches.yml":
        buckets[list(buckets)[1]].append(line)
    elif rel == "docs/UPSTREAM.md":
        buckets[list(buckets)[2]].append(line)
    elif rel.startswith("scripts/") and not rel.startswith("scripts/lib/profiles/"):
        buckets[list(buckets)[3]].append(line)
    else:
        prose_files[rel] = prose_files.get(rel, 0) + 1
for title, lines_ in buckets.items():
    if lines_:
        print(f"\n  ▸ {title}")
        for h in lines_:
            print(f"      {h}")
if prose_files:
    print(f"\n  ▸ {PROSE}")
    for rel, n in sorted(prose_files.items(), key=lambda kv: -kv[1]):
        print(f"      {rel}  ({n} line{'s' if n != 1 else ''})")
if n_hits == 0:
    print(f"  (no other occurrences of '{old_tag}' under scripts/ docs/ models/ services/ c3/)")

print(f"""
── remaining checklist (judgment half — /opt/ai/CLAUDE.md "When upgrading an engine pin") ──
  1. arch_patches.yml: add an engine_pin row for {ENGINE}@{new_tag} with loads: true
     ONLY after a live boot proves it (keep the {old_tag} row as history).
  2. Re-validate every vendored patch/overlay on the ACTUAL new image (merged→drop,
     drifted→rebase); update patches.yml + engines yml local_status.  ({ENGINE} note:
     if the profile is overlay-free this step is a no-op — say so in the PR.)
  3. Fixtures asserting the literal old pin (see report above) → update to {new_tag}.
  4. Live boot via switch.sh, 3 warm-up requests BEFORE judging output, verify-full
     (pass MODEL=<served-name>), high-ctx NIAH if KV-affecting.
  5. docs/UPSTREAM.md engine-pin row + dated learnings/<model>.md entry.
  6. Targeted guards: test-launch-compat, test-diagnose-profile, test-patch-attribution,
     registry emit/JSON.
""")
PY
