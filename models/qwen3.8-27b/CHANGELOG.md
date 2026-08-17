# Qwen3.8-27B — Changelog

## 2026-08-14 — Initial community contribution (single-rig validation)

Added llama.cpp configs for Qwen 3.8 27B on a single RTX 3090, validated on native Windows (llama.cpp b10435, CUDA 13).

**Measured (RTX 3090, sm_86, 230W cap, streaming decode):**
- `mtp-170k`: ~40–44 t/s @ 170K ctx + MTP n=2 + vision (~24 GB VRAM)
- `max-262k`: ~30–33 t/s @ 262K ctx + vision, no MTP (~24 GB VRAM)
- `balanced-128k`: ~33–36 t/s (interpolated)

**Not yet run (N/A justifications):**
- verify-full.sh / verify-stress.sh: N/A — native Windows path has no Docker container to smoke-test; configs validated by sustained agentic use (Hermes Agent multi-turn tool calling, long-context RAG) over multiple days.
- SOAK_MODE=continuous: N/A — same reason; however, the configs have been running continuously for 7+ days as a production agent backend without observed degradation.
- bench.sh canonical run: N/A — script targets Docker containers. Numbers above are streaming decode measurements from `/metrics` during real workloads (n>50 requests per profile).
- BENCHMARKS row: deferred pending maintainer review (single-rig data; can re-run with the repo harness on request).

**Key findings:**
- `--mlock` is critical on native Windows (without it, decode drops from ~40 to ~10 t/s under RAM pressure)
- ubatch sizing is the VRAM cliff lever: 512 at 170K+MTP, 128 at 262K (same mechanism as vLLM Cliff 2, tunable via `-ub`)
- MTP draft KV must stay f16 (quantizing it hangs every request)
- 170K + MTP + vision is the ceiling on 24 GB; 262K requires dropping MTP

**Contributor:** @Isaac-opz (single rig, Windows 10 native)
