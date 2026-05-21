# froggeric chat template (v19) — ik_llama copy

Source: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates

- File: `chat_template_v19.jinja`
- Downloaded: 2026-05-21 (fresh pull from the pinned commit, not copied from the vLLM tree)
- Upstream revision: `c31fd393e531dbacd92b6deb99a2037cc949f950`
- Upstream timestamp: 2026-05-16T13:44:07Z
- Upstream release label: **v19**
- SHA256: `4649b3fa3db3fda4d51173ed4ff0175fde7ece8bbceb9d595d04d862020c9746`
  (verified identical to the adopted vLLM-tree snapshot at
  `models/qwen3.6-27b/vllm/patches/froggeric-chat-template/chat_template.jinja`)

## Why a separate ik copy (vs referencing the vLLM tree)

Version pinned in the filename + ik track owns its template — no cross-tree
mount dependency on `vllm/patches/`.

## Why this is safe on ik_llama (where mtp.yml warns it isn't on mainline)

`llama-cpp/.../mtp.yml` documents that on **mainline** llama.cpp the froggeric
template "has no thinking-mode Jinja hook and silently suppresses
`--reasoning off`", so that path ships native. **Empirically re-tested on
ik_llama (cu13-server) 2026-05-21**: froggeric v19 + `--reasoning off`
suppresses thinking cleanly (hard reasoning prompt → answer in `content`, no
`<think>` leak, empty `reasoning_content`) AND renders tool-calls correctly.
So on ik the same `--reasoning off` lever works; we keep it.

## Quality status

The +10pp `hermesagent-20` win was measured on **vLLM** (PR #157). NOT yet
A/B-confirmed on ik_llama — froggeric-vs-native on toolcall-15 / hermesagent-20
is the owed measurement (relevant to the IQ4_KS toolcall-regression question).
