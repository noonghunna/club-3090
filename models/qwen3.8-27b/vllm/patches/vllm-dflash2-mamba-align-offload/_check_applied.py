"""Is this patch's content present in the installed vLLM tree?

verify.sh checks patches with a reverse dry-run, which is exact but cannot see a patch whose
hunks were disturbed by a second patch touching the same file (the DFlash2 pair does that).
This is the fallback: take the substantial lines a patch adds and look for them in the tree.

  python patches/_check_applied.py <patch> <site-packages/vllm>   # exit 0 = present
"""
import os
import sys

patch, root = sys.argv[1], sys.argv[2]
added = []
for line in open(patch, errors="replace").read().splitlines():
    if line.startswith("+") and not line.startswith("+++"):
        text = line[1:].strip()
        if len(text) > 24 and not text.startswith(("#", '"', "'")):
            added.append(text)
if not added:
    sys.exit(1)
added = added[:400]

blob = []
for dirpath, _dirs, files in os.walk(root):
    for f in files:
        if f.endswith(".py"):
            try:
                blob.append(open(os.path.join(dirpath, f), errors="replace").read())
            except OSError:
                pass
blob = "\n".join(blob)
hits = sum(1 for a in added if a in blob)
sys.exit(0 if hits >= max(3, int(0.8 * len(added))) else 1)
