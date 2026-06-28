"""AI Studio — Production Agent (v0a).

A CLI/admin, single-flight executor that drives the EXISTING AI-Studio lanes from a
static `ProductionPlanV1` and assembles one MP4. No 4B planner, no OWUI lane, no
Qdrant/SearXNG, no durable queue — those are v0b/v1 (see
/opt/ai/docs/ai-studio-production-agent-design.md).

stdlib-only on purpose (host python has no pydantic/pytest) so the spike runs +
self-tests with zero installs. A `synthetic` backend (ffmpeg lavfi) exercises the
whole path offline; the `live` backend hits ComfyUI (:8188) + Kokoro TTS (:8192).
"""

__version__ = "0a"
