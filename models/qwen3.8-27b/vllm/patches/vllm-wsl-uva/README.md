# WSL2 UVA fallback

vLLM's request-state `UvaBuffer` normally requires CUDA UVA. The CUDA/WSL2
runtime on the reference 2x RTX 3090 rig reports UVA unavailable, so the
DFlash2 path cannot boot without this fallback.

`install.sh` keeps the native pinned-host-memory/UVA path when available and
uses a small device tensor mirror only when `is_uva_available()` is false. It
is idempotent and refuses to apply if the vLLM 0.27.1 source anchor moved.

This is a WSL compatibility patch, not a throughput optimization. Native Linux
users can leave it mounted; it is a no-op at runtime when UVA is available.
