# thinkingcap-27b / vLLM compile caches

`torch_compile/` and `triton/` are warm-start JIT caches mounted by the composes
(`../../compose/**/*.yml` → `../../../cache/...`). Regenerated on first boot when a
variant's config changes; **not source** — everything here is gitignored except this
README and `.gitignore`.
