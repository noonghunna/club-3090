import glob, os, sys
root = sys.argv[1]

def exec_indices(line):
    out, stack, i = [], [], 0
    while i < len(line):
        ch = line[i]
        top = stack[-1] if stack else None
        if top == "'":
            if ch == "'": stack.pop()
            i += 1; continue
        if ch == '\\':
            i += 2; continue
        if top == '"':
            if line.startswith('$(', i): stack.append('$('); i += 2; continue
            if ch == '"': stack.pop()
            i += 1; continue
        if line.startswith('$(', i): stack.append('$('); i += 2; continue
        if ch == ')' and top == '$(': stack.pop(); i += 1; continue
        if ch == "'": stack.append("'"); i += 1; continue
        if ch == '"': stack.append('"'); i += 1; continue
        if line.startswith('grep ', i):
            before = line[:i].rstrip()
            if before == '' or before[-1] in '|;&(' or before.endswith('$('):
                out.append(i)
        i += 1
    return out

import re
HEREDOC = re.compile(r"<<-?\s*[\"\']?([A-Za-z_][A-Za-z0-9_]*)[\"\']?")

def scan_file(path):
    """Yield (lineno, line) for shell lines only. Heredoc bodies are skipped:
    they are usually another language (python, yaml, sql) where a bare `grep`
    may be an identifier, not a command. Rewriting those silently produces
    invalid code — this scanner caused exactly that once (a python
    `grep = subprocess.run(...)` became `command grep = ...`)."""
    term = None
    for n, line in enumerate(open(path, encoding='utf-8'), 1):
        if term is not None:
            if line.strip() == term:
                term = None
            continue
        m = HEREDOC.search(line)
        yield n, line
        if m:
            term = m.group(1)

bad = []
for pat in (os.path.join(root,'scripts','*.sh'), os.path.join(root,'scripts','lib','*.sh')):
    for f in sorted(glob.glob(pat)):
        for n, line in scan_file(f):
            for i in exec_indices(line):
                if line[max(0,i-8):i] == 'command ':
                    continue
                bad.append("%s:%d: %s" % (os.path.relpath(f, root), n, line.strip()[:100]))
print("\n".join(bad))
