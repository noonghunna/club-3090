# thinkingcap-27b / vLLM compile caches

> **The launchers no longer use this folder (#1466).** `switch.sh`, `launch.sh`, `gpu-mode`,
> the estate planner and c3 mount these caches from `~/.cache/club-3090/<engine image>/`
> (`CLUB3090_CACHE_DIR` moves it), shared by every checkout and every model that runs the
> same image. This folder is only what a plain `docker compose up` falls back to when
> nothing is set. A cache an older version left here is dead weight:
> `bash scripts/settings.sh caches` shows its size and removes it after asking, and prints
> the `sudo rm -rf` line for folders docker created as root.

`torch_compile/` and `triton/` are warm-start JIT caches mounted by the composes
(`../../compose/**/*.yml` → `../../../cache/...`). Regenerated on first boot when a
variant's config changes; **not source** — everything here is gitignored except this
README and `.gitignore`.
