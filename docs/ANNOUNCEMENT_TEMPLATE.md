# Announcement template

The skeleton for a **"we shipped X" post** in the GitHub **Announcements** discussion category — the kind of post that introduces a newly-shipped compose/model, reports its numbers, and tells people how to run it.

It's the *wrapper* around a [Results Card](RESULTS_CARD.md): the Results Card is the measured-numbers panel (Serving · Quality · Takeaways); the announcement adds the framing, install steps, credits, and the "what'd help" ask around it. Don't duplicate the Results Card spec here — author that panel per [`RESULTS_CARD.md`](RESULTS_CARD.md) and drop it into the middle of this skeleton.

**Reference example:** [discussion #350 — Deckard-40B](https://github.com/noonghunna/club-3090/discussions/350) (the canonical fully-worked post this template is distilled from).

## When to use it

- A new compose / model graduates to the catalog (🧪 or ✅) and you want users to know it exists.
- A meaningful capability or finding ships (a new spec-dec path, a quant tier, a cross-rig-portable result).

For a *reply* answering a specific question (e.g. "where's the X YAML?"), you can lead with a one-line answer to the asker, then drop the announcement body underneath — the post still works as a standalone announcement.

## Section order

| # | Section | Purpose | Skip when |
|---|---|---|---|
| 0 | **Update banner** (italic, above `---`) | Only on edits — "promoted 🧪 → ✅", "added cross-rig numbers". States *what changed and when*. | First publish. |
| 1 | **Intro** — what shipped, one sentence | Name the slug in **bold**, the model, the engine, the topology. Credit the model/quant/drafter authors inline (`@handle` + HF link) — capability that isn't ours gets attributed up front. | Never. |
| 2 | **Headline finding** | The single most interesting result, one line — the reason someone keeps reading. | Never. |
| 3 | **🎴 Results Card** | The measured panel — ① Serving ② Quality (full 8-pack `/150`) ③ Takeaways. Author per [`RESULTS_CARD.md`](RESULTS_CARD.md). | Never — numbers are the point. |
| 4 | **Why / context** | Why this model/config is worth shipping — the workload it serves, the trade it makes. (#350's "Why an uncensored model?" / "Why it's Production".) | A trivially-obvious ship. |
| 5 | **Getting it** | Where the weights live (public HF repo + `hf download`), whether `setup.sh`/`launch.sh` auto-fetch, any engine-version floor. | — |
| 6 | **Run it** | The exact `switch.sh` / `gpu-mode` command, the port + served model name, OWUI wiring. Copy-pasteable. | — |
| 7 | **What'd help** | The concrete cross-rig / follow-up ask. Call out the *portable* finding (what holds beyond our rig). | — |
| 8 | **Credits** | Model author · quant/drafter author · engine — each `@handle` + link. Restates §1 attribution in full. | No third-party work involved. |

## Skeleton

```markdown
<!-- §0 — only on edits -->
*Updated <YYYY-MM-DD>: <what changed — e.g. promoted 🧪 → **✅ Production**; the one blocker (<X>) is resolved (<how>).>*

---

<!-- §1 intro -->
We just shipped **`<engine>/<slug>`** — <one-sentence what-it-is>. It's **<Model>** — [**@<author>**](<hf-link>)'s <description> (<base GGUF/safetensors link>)<, with <quant/drafter> by [**@<author2>**](<link>)>, <topology>. **All credit for the model goes to @<author>; @<author2> made <the X path> possible** — see [Credits](#credits).

<!-- §2 headline -->
The headline: **<the single most interesting finding, one line>.**

<!-- §3 Results Card — author per docs/RESULTS_CARD.md -->
## 🎴 Results Card — <rig>, <engine + version>, <thinking on/off>

### ① Serving
| Config | Spec-dec | KV | ctx | Narr | Code | Accept | VRAM |
|---|---|---|---|--:|--:|--:|--:|
| <baseline / variant> | <none / MTP n=N / DFlash> | <kv> | <ctx> | <tps> | <tps> | <%> | <GB> |

<n-sweep / context-ceiling / lift summary line>

### ② Quality — core 8-pack (/150, <think setting>)
| Pack | <A> | <B> |
|---|--:|--:|
| toolcall-15 | x | x |
| instructfollow-15 | x | x |
| structoutput-15 | x | x |
| dataextract-15 | x | x |
| reasonmath-15 | x | x |
| bugfind-15 | x | x |
| hermesagent-20 | x | x |
| cli-40 | x | x |
| **Total /150** | **x (y%)** | **x (y%)** |

<one-line read of the quality result — tie? regression? lossless spec-dec?>
Plus: **verify-full <n>/8** · **verify-stress <n>/8** · **soak-continuous <PASS/…>**.

### ③ Takeaways
- **<headline verdict>** — the most important finding.
- <comparison / tradeoff>
- <production / stability / topology note>

<!-- §4 why / context -->
## Why <this model / config>?
<the workload it serves, the trade it makes>

<!-- §5 getting it -->
## Getting it
The weights are public: **[<org/Repo>](<hf-link>)**. <auto-fetch note: setup.sh/launch.sh fetch it, or `hf download <org/Repo>`>. You'll want <engine + min version> for <the feature>.

<!-- §6 run it -->
## Run it
\`\`\`bash
bash scripts/switch.sh --owui <engine>/<slug>   # launches on :<port> and wires Open WebUI
\`\`\`
Serves an OpenAI-compatible API on **`:<port>`** (model `<served-name>`). Point any OpenAI client at it.

<!-- §7 ask -->
## What'd help
<cross-rig numbers / long-ctx / behaviour notes>. The portable finding: **<what holds beyond our rig>**.

<!-- §8 credits -->
## Credits
- **The model:** [**@<author>**](<link>) — <what>.
- **The <quant/drafter>:** [**@<author2>**](<link>) — <what>.
- **The engine:** [<engine>](<link>) — <feature / PR #>.
```

## Rules

- **Attribute third-party work up front and in Credits** — model authors, quant/drafter authors, engine PRs. Capability that isn't ours gets named twice (intro + Credits), not buried.
- **Numbers must be measured** — the Results Card is the empirical core; follow [`RESULTS_CARD.md`](RESULTS_CARD.md)'s rules (core 8-pack is exactly `/150`, spec-dec is its own column, state the `n`, reproduce sampling/pin/thinking in the footnote).
- **State the lifecycle honestly** — 🧪 / ✅ / 👁️ — and *why* (a rolling pre-release engine pin keeps a fully-validated compose 🧪; say so).
- **Pre-post checklist** (same as RESULTS_CARD §"Posting to a public discussion/issue"): grep the draft for internal absolute paths / model-store paths / tokens; don't link untracked/experimental composes (they 404 — describe in prose or link the image tag); verify every repo link with `git ls-tree` and every image tag with `docker manifest inspect`.
- **Lead with the ask if it's a reply** — when answering a specific question, give the one-line answer first, then the announcement body.
