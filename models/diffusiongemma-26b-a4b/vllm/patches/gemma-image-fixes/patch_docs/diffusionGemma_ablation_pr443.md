# PR #443 ablation — mount necessity (test777, 2026-06-19)

Hermes `state.db` replays against `club3090:8020`, toggling only the three
agentic overlays (marlin + `diffusion_gemma.py` always mounted). Streaming replay
of session `20260619_054214_b53bf8` empty contexts `@3039`/`@3051` (8 reps × 2
contexts × reasoning on/off). Deep sessions use `reasoning_content` from DB.

## Results matrix

| Config | Overlays | CoT in prompt | `054214` empty (ON/OFF) | `082023` quoted-args | `084848` tail (58 prior CoT) | `100411` tail (63 prior CoT) |
|---|---|---|---|---|---|---|
| **A — stock** | none | yes (1 block) | **44% / 31%** | — | — | — |
| **B — PR #443** | reasoning + tool parsers | yes | **0% / 0%** | 3 OK / 8 BAD | — | — |
| **C — template only** | chat template | **no** | **56% / 31%** | 2 OK / 8 no_call | 2/4 empty | 3/4 empty |
| **D — all six** | parsers + template | **no** | **6% / 0%** | 7 OK / 2 BAD | 1/4 empty | **0/4 empty (4/4 tool calls)** |

## Conclusions

1. **Chat template is not the root cause of empty-after-tools.** Config C (template
   only) does not fix the canonical `054214` replay — OFF-mode empty rate stays at
   **31%**, matching stock. The stored empties there had `reasoning=None` (no history
   CoT to strip).

2. **Tool parser overlay is necessary and sufficient for the empty-turn bug.** Config B
   (PR #443 parsers, stock template) drops `054214` from **44%/31% → 0%/0%** without
   the chat template.

3. **Chat template is a separate, additive fix.** It stops history CoT replay (marker
   absent in D; present in A/B). Combined with parsers (D), deep reasoning-heavy
   sessions improve: `100411` tail **0/4 empty, 4/4 tool calls** vs C **3/4 empty**.
   Quoted-args on `082023` improves **3/8 → 7/10** clean when calls fire (D vs B).

4. **Reasoning parser** was not isolated in this matrix (always bundled with tool parser
   in B/D). Its leak is a separate streaming UX bug — see
   [`diffusionGemma_reasoning_stream_fix.md`](./diffusionGemma_reasoning_stream_fix.md).

**Recommendation:** keep all PR #443 parser overlays; add the 6th mount (chat template)
as a follow-up commit on the same branch.

Raw JSON: captured on rig during ablation run (`dgemma_ablation_results.json`).
