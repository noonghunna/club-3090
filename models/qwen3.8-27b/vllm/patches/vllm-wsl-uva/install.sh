#!/usr/bin/env bash
# WSL2 fallback for vLLM's host-UVA request buffers. Native Linux keeps the
# pinned host/UVA path; WSL uses a small device mirror instead.
set -euo pipefail
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
python3 - "$VLLM/v1/worker/gpu/buffer_utils.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
if "self.has_uva = is_uva_available()" in s:
    print("[wsl-uva] fallback already applied", file=sys.stderr)
    raise SystemExit(0)
old = '''        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)
'''
new = '''        self.has_uva = is_uva_available()
        self.cpu = torch.zeros(
            size, dtype=dtype, device="cpu", pin_memory=self.has_uva
        )
        self.np = self.cpu.numpy()
        self.uva = (
            get_accelerator_view_from_cpu_tensor(self.cpu)
            if self.has_uva
            else torch.empty(size, dtype=dtype, device="cuda")
        )
'''
if old not in s:
    raise SystemExit("[wsl-uva] constructor anchor missing; refusing boot")
s = s.replace(old, new, 1)
old = '''        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        return buf.uva[:n]
'''
new = '''        n = len(x)
        if buf.has_uva:
            dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
            dst[:n] = x
        else:
            values = torch.as_tensor(x, dtype=buf.uva.dtype, device=buf.uva.device)
            buf.uva[:n].copy_(values)
        return buf.uva[:n]
'''
if old not in s:
    raise SystemExit("[wsl-uva] copy anchor missing; refusing boot")
p.write_text(s.replace(old, new, 1))
print("[wsl-uva] installed", file=sys.stderr)
PY
