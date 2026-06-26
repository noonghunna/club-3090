# Changelog

Auto-generated from commit subjects by [git-cliff](https://git-cliff.org/) on tag
push. Click any commit SHA below to see the full message body (why / how /
validation data) — those live in `git log`, not here. Don't hand-edit; the file
is regenerated on every tag.

**Versioning:** SemVer in `0.x` — treat any minor bump as potentially breaking
until `1.0`. Past CalVer tags (`v2026.05.09`, `v2026.05.10`) are preserved for
history; SemVer takes over from `v0.3.0` onward.

| CalVer tag | SemVer equivalent | Date |
|---|---|---|
| `v2026.05.09` | (≈ v0.1.0) | 2026-05-09 — first tagged release |
| `v2026.05.10` | (≈ v0.2.0) | 2026-05-10 — stack reorg + Gemma 4 INT8 PTH unblock |

---

## v0.10.0 — 2026-06-26


### ⚠️ Cliffs, gotchas, regressions

- c3 Phase R / R3b-1 fixups: temp unlink-on-decline + lane-native [v] regression test ([4ec2b2b](https://github.com/noonghunna/club-3090/commit/4ec2b2b6341fe85ae1eff652692a9ac49f4039b6))


### ✨ Features

- feat(lmcache): add NVLink auto-detection to dual-lmcache compose ([ea833ad](https://github.com/noonghunna/club-3090/commit/ea833ad134c98ef38152fd30ff93e0cc024bcd71))
- feat(vibethinker-3b): add llamacpp/vibethinker-3b-single (Q8, incubating) ([d3cacfa](https://github.com/noonghunna/club-3090/commit/d3cacfad0c1155066c09893b6ab2939ecb4ba6e2))
- feat(catalog): add 🐣 Incubating status tier + VibeThinker-3B (incubating) ([35a6965](https://github.com/noonghunna/club-3090/commit/35a69650fed92376d8f3d2bca920fdd034f2448d))
- feat(qwen3.6-27b): add beellama/carnice-v2-dual-q8-mtp dual compose (#403) ([186ab9b](https://github.com/noonghunna/club-3090/commit/186ab9b2b6155936ef3d7ee365f00cae253710d6))


### 🎯 New models + serving paths

- Add gemma12b single-card gpu-mode scene + litellm route (#469) ([#469](https://github.com/noonghunna/club-3090/pull/469) by @noonghunna)
- Add guard test: conditional compose entrypoints must $$-escape runtime bash ([1b87ebe](https://github.com/noonghunna/club-3090/commit/1b87ebe05e760fdb6c027341ec7210ef827c209b))
- Add llamacpp/qwen27b-pi-reasoning-single (Qwen3.6-27B Pi-style coding agent, mainline llama.cpp + MTP) (#425) ([#425](https://github.com/noonghunna/club-3090/pull/425) by @noonghunna)
- Add opt-in LMCache KV-offload compose (vllm/qwen-27b-dual-lmcache, incubating) (#421) ([#421](https://github.com/noonghunna/club-3090/pull/421) by @noonghunna)


### 🐛 Bug fixes

- fix(lmcache): forward env tuning knobs ([c90e4f2](https://github.com/noonghunna/club-3090/commit/c90e4f2e18ddd455576080ff6ebba8a195a4b6ad))
- fix(nvlink): strip only expandable_segments on custom-AR path ([f5d5dca](https://github.com/noonghunna/club-3090/commit/f5d5dca09a4b907b9c5aafb611c1bb090af11bc3))


### 📝 Documentation

- docs: add PCIE_P2P.md — PCIe topology & enabling P2P (no NVLink) ([df419c9](https://github.com/noonghunna/club-3090/commit/df419c9bd7f4d518ed8ef4fc9199863c50201dfd))
- README: add a getting-started subsection for the c3 TUI ([069ac27](https://github.com/noonghunna/club-3090/commit/069ac27256b0e1f5027a071d139add8536139d1e))
- docs(lmcache): fold cross-rig findings — warm-L2 compute-bound + aborted-prefill caveat ([7b0c4d4](https://github.com/noonghunna/club-3090/commit/7b0c4d4abc12eda7e7db28fa63fc520856b4bdf2))
- docs(c3): first-run Settings step (Model Dir + HF token) + S keybinding ([50bed70](https://github.com/noonghunna/club-3090/commit/50bed7047633df08ea29b5ddc5ca4e061e95bfad))
- docs(ai-studio): Krea 2 is a real image lane (was "dropped") ([7daef3e](https://github.com/noonghunna/club-3090/commit/7daef3ec1ec5d07777416f379da9e762d0aa0175))
- docs: note the image-lane quality ceiling + optional HQ upgrade path ([852820d](https://github.com/noonghunna/club-3090/commit/852820d2c3631e5053c4b45a22a12da81eec723e))
- docs: AI Studio requirements + Wan tuning + director placement ([11984d1](https://github.com/noonghunna/club-3090/commit/11984d19dac66d0222d0aa66f4bdbf462808fff7))
- docs: Z-Image + Wan2.2 lanes (11 lanes); Krea2 dropped (cloud-only) ([5c2ed40](https://github.com/noonghunna/club-3090/commit/5c2ed400d25b40f7ee653e06594c7b606dddf1ef))
- docs: ai-studio consolidation pass (one scene, lanes inside it) ([813ddf9](https://github.com/noonghunna/club-3090/commit/813ddf94d8a22648fee9ccc8cd7cf56a4de25ac4))
- docs(upstream): add ParserEngine tracker rows (#45413, #45588) ([0596283](https://github.com/noonghunna/club-3090/commit/05962838cfeb37436875c43eebb84bb65abd82c4))
- docs(upstream): clarify #40812 is merged-but-insufficient; point at #43923 ([611587d](https://github.com/noonghunna/club-3090/commit/611587dff3912095097af01d5d5a91ca5bcf3b8f))
- docs: mark transformers>=5.8.0 / gemma4_assistant upstream row resolved (#453) ([#453](https://github.com/noonghunna/club-3090/pull/453) by @steamEngineer)
- docs: fix vLLM upstream table render (drop stray blank line) (#452) ([#452](https://github.com/noonghunna/club-3090/pull/452) by @steamEngineer)
- docs(vibethinker-3b): add measured one-shot-coding scores (HE+ 97%, LCB 83%) ([8b3aa13](https://github.com/noonghunna/club-3090/commit/8b3aa13f54b1780b60f1d053512b65735b9efaa8))
- docs: add ANNOUNCEMENT_TEMPLATE.md (Announcements-post skeleton) ([a728832](https://github.com/noonghunna/club-3090/commit/a728832f606631ef0c8e9c17a13cbf77d12c411b))


### 🛠️ Scripts + tooling

- report.sh: surface P2P engagement as a labeled field (#491) ([#491](https://github.com/noonghunna/club-3090/pull/491) by @noonghunna)
- setup.sh: let WEIGHT_KEY bypass the friendly model-name dispatch ([f9d8f8d](https://github.com/noonghunna/club-3090/commit/f9d8f8d485d1e94686b4a4f7dd6a9b2ffc97d1ca))
- weights.py: add `list --json` (download-state source for c3) ([1730e04](https://github.com/noonghunna/club-3090/commit/1730e0405b96da3aa09ac317b5c4db56f5ea997c))


### 🧹 Other

- BENCHMARKS: add @oven1231231234 dual-3090 NVLink-vs-PHB A/B (PCIe 3.0 x16) ([4032645](https://github.com/noonghunna/club-3090/commit/4032645ee45c31e9a37b3164e7ae208294903bf8))
- Standardize 27b served-model-name → qwen3.6-27b-autoround (#490) ([#490](https://github.com/noonghunna/club-3090/pull/490) by @noonghunna)
- gpu-mode: align power-cap default 230W -> 250W (completes #483) (#489) ([#489](https://github.com/noonghunna/club-3090/pull/489) by @noonghunna)
- Add nvidia-power-cap.service for boot-time GPU power cap (#483) (#485) ([#485](https://github.com/noonghunna/club-3090/pull/485) by @alexpolo1)
- BENCHMARKS: add Ornith-1.0 section (9B + 35B rows) ([c9f651e](https://github.com/noonghunna/club-3090/commit/c9f651ecffe0af72e91fb1e11722df2bd1d052fe))
- Add Ornith-1.0-35B experimental slug (ik-llama/ornith35b-dual) (#479) ([#479](https://github.com/noonghunna/club-3090/pull/479) by @noonghunna)
- BENCHMARKS: add @Whamp 4× 3090 TP=4 multi4 rows (#446) ([291fa74](https://github.com/noonghunna/club-3090/commit/291fa740ffce58a450795ca57ffadcadaa76363c))
- BENCHMARKS: add @guybrush01 dual-5090 cross-rig row (Qwen3.6-27B, first Blackwell) ([2c686e9](https://github.com/noonghunna/club-3090/commit/2c686e96b6484f05f913eca6b2cb5ca61fb92395))
- Add Ornith-1.0-9B experimental slug (ik-llama/ornith9b-single) (#477) ([#477](https://github.com/noonghunna/club-3090/pull/477) by @noonghunna)
- c3: adaptive estate poll — fast GPU, regime-gated docker (burst/steady/idle) (#476) ([#476](https://github.com/noonghunna/club-3090/pull/476) by @noonghunna)
- Director placement lever: CPU / GPU0 / GPU1 (c3 Settings) (#473) ([#473](https://github.com/noonghunna/club-3090/pull/473) by @noonghunna)
- Wire SearXNG web search + Qdrant vector DB into Open WebUI (#472) ([#472](https://github.com/noonghunna/club-3090/pull/472) by @noonghunna)
- Lazy-load step-voice + scene-wire it into ai-studio (#471) ([#471](https://github.com/noonghunna/club-3090/pull/471) by @noonghunna)
- Make the OWUI model picker scene-accurate (studio lanes + LLMs) (#470) ([#470](https://github.com/noonghunna/club-3090/pull/470) by @noonghunna)
- gpu-mode: rename model scenes to qwen27b / qwen35b-a3b / gemma-31b ([33a1f71](https://github.com/noonghunna/club-3090/commit/33a1f71c8198b54cfe22af8504893085cc005526))
- gpu-mode: add the 35b-a3b scene (parity with 27b/gemma/deckard) ([9f1f3a9](https://github.com/noonghunna/club-3090/commit/9f1f3a9e37a2b1a907f73b596573b8668293593a))
- gpu-mode: fix `gpu-mode gemma` (undefined fn aborted before start) + litellm port ([2b17aa5](https://github.com/noonghunna/club-3090/commit/2b17aa5070d4040a4a2aa8d86fb543d9d3fed33a))
- c3: distinguish "booting" from "API not reachable" in Doctor (all engines) ([1ce5a5f](https://github.com/noonghunna/club-3090/commit/1ce5a5f717f303ec9b73f0d5ad958085fbe8e64c))
- owui: wire LiteLLM gateway (:4000) as a chat connection ([f5ec5c3](https://github.com/noonghunna/club-3090/commit/f5ec5c3201ebad2e6e6575df3ce91dec903a6144))
- c3: scene-preview box auto-grows to fit all services (not a cap) ([9459536](https://github.com/noonghunna/club-3090/commit/94595368d215983df4a3bfe88a785292a85c7458))
- c3: cap scene-preview services with "+N more" (ai-studio truncation) ([9299e6f](https://github.com/noonghunna/club-3090/commit/9299e6fc1447a8ea36fee357e697049642c0739c))
- serve-cockpit: declare PyYAML dependency (fixes c3 diagnose-estate crash) ([0cc9d9c](https://github.com/noonghunna/club-3090/commit/0cc9d9c5c4cbf64c5ca098c695178d2bd715149e))
- ai-studio: wire the Krea 2 Turbo image lane (validated on v0.26.0) ([978052a](https://github.com/noonghunna/club-3090/commit/978052abfc6768647d52a5588f23eba7054d8113))
- comfyui entrypoint: fix dubious-ownership blocking the pin checkout ([e738c7c](https://github.com/noonghunna/club-3090/commit/e738c7c59b24e0031132e712fd9c6c322436447a))
- Bump ComfyUI pin to v0.26.0 + queue Krea 2 download ([2693d93](https://github.com/noonghunna/club-3090/commit/2693d937f8bbf1d26ccca654d6ff93d188383ff5))
- detect_nvlink: auto-enable PCIe P2P when nvidia-smi reports it OK ([446c7d9](https://github.com/noonghunna/club-3090/commit/446c7d97428f80587a1e70cf887d7cc5f785c676))
- studio: one-command setup-ai-studio.sh + auto-install the OWUI pipe ([aad1e69](https://github.com/noonghunna/club-3090/commit/aad1e69d0989dccf1381394b57588acac9675880))
- c3: make fresh-clone install work + rewrite stale README + repo pointer ([37c680d](https://github.com/noonghunna/club-3090/commit/37c680d5d28f68eed6d4f7bcee7ac0a5fd9b7c80))
- studio: Wan2.2 full parity — recipe fix, 720p valve, i2v, long-clip chaining ([0ff83b8](https://github.com/noonghunna/club-3090/commit/0ff83b8fae0cda90ccebf745d8531ac462f86644))
- studio: download scripts + manifest rows for Z-Image + Wan2.2 ([5fb8fa1](https://github.com/noonghunna/club-3090/commit/5fb8fa1c7f0930c97da411af21b6fea2ea24d439))
- studio: add Z-Image + Wan2.2 uncensored lanes; unify OWUI lane naming ([9b16ef9](https://github.com/noonghunna/club-3090/commit/9b16ef919c6efac4eeaa01060897307c6df36bf3))
- cockpit: ai-studio missing-models modal + video⊕voice GPU guard ([75dbc0a](https://github.com/noonghunna/club-3090/commit/75dbc0a8f52ae6861f863eee93bae0d1287ee2c2))
- studio: add the 10Eros uncensored video lane (A/B vs Sulphur) ([a4230b3](https://github.com/noonghunna/club-3090/commit/a4230b3861746d9a40e72331121b16b610e7049d))
- studio: consolidate image/video into one `ai-studio` gpu-mode scene ([3fc2685](https://github.com/noonghunna/club-3090/commit/3fc26857f200d692effd189fd2c490a48ef275e2))
- cockpit: give report/full-report/cap-sweep their own Doctor cards ([ef8cba8](https://github.com/noonghunna/club-3090/commit/ef8cba83b12ba0322eceb9f4ca5f76793878b26a))
- studio: preflight the models before starting (gpu-mode + cockpit) ([757c1da](https://github.com/noonghunna/club-3090/commit/757c1daba3b4c8223cc9b4687b4637ed5cab1804))
- studio: fetch the director / Kokoro / Step-Audio models for a fresh install ([cece4ec](https://github.com/noonghunna/club-3090/commit/cece4ec4fbb22d3c4e960677ff515f2c50ed3114))
- cockpit/gpu-mode: comfyui engine label, drop power-cap strip + solo scene, list studio sidecars ([cae021d](https://github.com/noonghunna/club-3090/commit/cae021d6ead9e38c59376b95d3ec8cf4a6cee6c5))
- cockpit: show engine + port for comfyui/qdrant (port-range + non-engine ports) ([ee954e1](https://github.com/noonghunna/club-3090/commit/ee954e140495688737517b7cd50558267a6476c4))
- cockpit: guide studio setup when comfyui-local image is missing ([016bf36](https://github.com/noonghunna/club-3090/commit/016bf36fd50982bfc6bd63cc06f950e011dd7016))
- comfyui: parameterize host paths for portability (${COMFYUI_ROOT} / ${MODEL_DIR}) ([ac31c7e](https://github.com/noonghunna/club-3090/commit/ac31c7e91f08489c56ed112052459a9a91c125f7))
- cockpit: fix per-service stop/start to target the real container + project ([c52ab09](https://github.com/noonghunna/club-3090/commit/c52ab09b4885d51f48519ce6d9e696e7da26ccae))
- stack: remove retired ollama service entirely ([9b73a4c](https://github.com/noonghunna/club-3090/commit/9b73a4ccfd652c2bab2330ca83b62b9c39f717aa))
- cockpit: Containers shows service status/engine/port + start; de-dup Orchestration ([727d90b](https://github.com/noonghunna/club-3090/commit/727d90b06fc0896cdcebca55aa02b4a0541d4cb2))
- cockpit: live status bullets on Orchestration services (read-only) ([1b4e2af](https://github.com/noonghunna/club-3090/commit/1b4e2af4bcf1de7467742e96a6886d26ed2a2148))
- cockpit: cluster Orchestration scenes by group (models → studio → ops) ([0b58a48](https://github.com/noonghunna/club-3090/commit/0b58a48e93abecb7fe53b5e2aa19d0ce9bdbfbb3))
- gpu-mode: collapse the gemma scenes to just 'gemma' ([7bca67c](https://github.com/noonghunna/club-3090/commit/7bca67cbde6fb5c8b2b4d6f43837e852bce43d31))
- gpu-mode scenes: regroup (serving→models), drop bigmodel + diffusiongemma ([5fb2252](https://github.com/noonghunna/club-3090/commit/5fb22527c2e8cd38ee6f3df2b8c9f466a2734c9a))
- c3 Orchestration scenes: footer-only confirm, clearer override, real 'off' ([0896bd9](https://github.com/noonghunna/club-3090/commit/0896bd9721373ea5db9e67e81de155b682bad082))
- c3 Orchestration: power-cap menu, remove prune, move cap-sweep to Doctor ([38227ce](https://github.com/noonghunna/club-3090/commit/38227cea93d158acbbb0ddc2ea0af5416440ff0f))
- c3: unify MODEL_DIR convention between the cockpit and setup.sh ([f03b514](https://github.com/noonghunna/club-3090/commit/f03b5146646d5c8018905fde5087dfdbdcd5a451))
- c3 Download: ignore backup cruft (*.bak/*.old/~) in the progress byte-count ([acacfda](https://github.com/noonghunna/club-3090/commit/acacfdaaf1fbfb7e53d3a1cab6105461c92efe12))
- c3 Download: live progress moves + state survives a catalog refresh ([32fadef](https://github.com/noonghunna/club-3090/commit/32fadef1925e078be3c34a28234ae36a1176875a))
- c3 Download: registry-driven companion artifacts (DFlash draft / mmproj) ([639525b](https://github.com/noonghunna/club-3090/commit/639525bd85ade48efbf1e7390e84de6c046db3d4))
- c3 Settings: show [S] in the footer + honor MODEL_DIR / HF_TOKEN env vars ([2aae1d0](https://github.com/noonghunna/club-3090/commit/2aae1d0fcce9068dcfdf84906a6bcba8422c72b0))
- c3 Download UX: [S] Settings (model dir + HF token) + not-set banner ([340896c](https://github.com/noonghunna/club-3090/commit/340896c574a4e89d3a03d4d024c1262a7a895357))
- c3 Download UX: Download-vs-Start pop-up, listing glyph, download worker ([8947d59](https://github.com/noonghunna/club-3090/commit/8947d597df5f638b1309de3a88e6cea13e3d3982))
- profiles: wire hf_repo for 5 verified weight variants ([eb81c5f](https://github.com/noonghunna/club-3090/commit/eb81c5f0af750f176b4da313442081e1301991a7))
- c3 Download UX (service): download plan + progress + disk-fit ([a28d3d1](https://github.com/noonghunna/club-3090/commit/a28d3d13b09890f5f393768da4d42aafddb95f49))
- c3 Download UX (data layer): per-slug weights-on-disk state ([e8d99a5](https://github.com/noonghunna/club-3090/commit/e8d99a553c99e9e774db688ef85b4c88e4084c63))
- c3: fix slug-match prefix shadow, idle-service reconcile, hint overflow, copy, experimental Force-Start ([53a570f](https://github.com/noonghunna/club-3090/commit/53a570f3059bedc48932e0b886620d117440462d))
- c3: [Y] copy-to-clipboard + shift+arrow horizontal scroll ([d256a05](https://github.com/noonghunna/club-3090/commit/d256a058e92da1dbb6a021f98a4e229e5c2bfd07))
- c3 Doctor: "is it serving correctly?" — verify / verify-full reads ([d05f132](https://github.com/noonghunna/club-3090/commit/d05f132b0a879b1139203870b4ddc733f494b667))
- c3 serve pop-up: state-aware Stop/Start/warned-Start; drop Fit column ([8ad9a00](https://github.com/noonghunna/club-3090/commit/8ad9a00d21a86022a65905f914a466c85b71a03b))
- c3 Catalog: topology column, multi-word filter; Modes/tab-bar arrow refinements ([ba22ccd](https://github.com/noonghunna/club-3090/commit/ba22ccd4b0f70dd4ec524027a155e7853769ff07))
- c3 cockpit: stopped-drill clear, arrow-navigable Modes, lane no-yank ([303d9b6](https://github.com/noonghunna/club-3090/commit/303d9b6bdef78c6222ca44c98dd9e5d31ad3186c))
- c3 cockpit: arrow-key focus descent (↓ into content, ↑ back to tab bar) ([0232797](https://github.com/noonghunna/club-3090/commit/02327971adfd2818fcbc2ce0633c1c935f43fdf2))
- c3 cockpit: fix keyboard focus/navigation (footer + tab bar) ([6bca4e6](https://github.com/noonghunna/club-3090/commit/6bca4e6f6120af498384dcb37ce8a91845346c54))
- c3 cockpit UX Tier-1: footer/binding hygiene (Codex review) ([982e034](https://github.com/noonghunna/club-3090/commit/982e034ca7d46b9e62ce22be2d8e7647df6a4db8))
- c3 cockpit: merge Run+Operate; Bring & Validate visible by default ([b3b842a](https://github.com/noonghunna/club-3090/commit/b3b842a6127d05977775f497eec41e06db548de7))
- c3 cockpit live-use fixes: table cursor preserve, curated profile dropdown, disk/RAM in the left rail ([ba984ca](https://github.com/noonghunna/club-3090/commit/ba984cadbccf6f2039586203c54aee29db48271d))
- c3 cockpit UX batch 5: estate telemetry — disk/RAM bars, studio-* services, GPU-VRAM attribution ([24b672f](https://github.com/noonghunna/club-3090/commit/24b672f1468b6a7c297df78c200c5f8d07f584c9))
- c3 cockpit UX batch 4b: inline previews + registry-derived profile dropdown ([171058c](https://github.com/noonghunna/club-3090/commit/171058c4471c5069aee579e745d61156f05e405b))
- c3 cockpit UX batch 4a: navigation & discoverability + an unsafe-gate Enter safety fix ([78facde](https://github.com/noonghunna/club-3090/commit/78facdee867f32c5340e3354c1448bd3dc4fdc49))
- c3 cockpit UX batch 3: honesty + serving actions (fit-vs-live-VRAM, real running config, targeted stop) ([bb2c30e](https://github.com/noonghunna/club-3090/commit/bb2c30ebca5528277853051e84472a5558efe363))
- c3 cockpit UX batch 2: perception loop — live surfaces reflect writes + fail honestly ([049ede8](https://github.com/noonghunna/club-3090/commit/049ede882048f45b1196c3bf306e817063676df7))
- c3 cockpit UX batch 1: serving panel, full service list, calm Containers, power/cap on cards ([dfe9004](https://github.com/noonghunna/club-3090/commit/dfe90043b0659d06e61f90dd179e7d5a92659b02))
- c3 Phase R / R4: in-app Contribute door (runtime surface toggle + persistence) ([ca67481](https://github.com/noonghunna/club-3090/commit/ca67481b02bcb4ce8832bf6908e443159d477dd0))
- c3 Phase R / R3b-2: ④ measure-vs-curated-bar + producer full-validation report ([a6df0c2](https://github.com/noonghunna/club-3090/commit/a6df0c2c0fab759cc3d0e17fe7f3c8c6f5c82124))
- c3 Phase R / R3b-1: producer "Bring & Validate" lane — ①–⑤ stages + ② Serve ([2c3c93d](https://github.com/noonghunna/club-3090/commit/2c3c93d9409a3741e617580459daff1e223dbf47))
- c3 Phase R / R3a: surface-gate the producer lane (consumer = Run + Operate) ([ddd70b6](https://github.com/noonghunna/club-3090/commit/ddd70b6c5431f597bc420496b05f2314035efaea))
- c3 Phase R / R2b: consumer share-back — rig report [R] / submit bench [B] / problem [!] ([2a5397a](https://github.com/noonghunna/club-3090/commit/2a5397ace4379ca3078383bd84f8042500567c6f))
- c3 Phase R / R2a: rename Estate → Operate + move Doctor into Operate ([250cd50](https://github.com/noonghunna/club-3090/commit/250cd50a93b8bc090cf62bf62274bd7046ee9c32))
- c3 Phase R / R1: fold Discover + Serve + Benchmarks into one Run mode ([8762442](https://github.com/noonghunna/club-3090/commit/8762442fc56274ecaf5a9ce218bd8a621be001cd))
- c3 Phase R / R0 fixups: surface gate before _ALWAYS_ON + de-flake gate tests ([e706f64](https://github.com/noonghunna/club-3090/commit/e706f6458733dc9a8ffee4df76c639a91147815a))
- c3 Phase R / R0: surface scaffolding (consumer | producer) ([b04093e](https://github.com/noonghunna/club-3090/commit/b04093e3306d61c9df7f519c56d64189e91c5880))
- Fix NVLink boot crash when compose pre-sets expandable_segments ([a4b8bd1](https://github.com/noonghunna/club-3090/commit/a4b8bd1d12b66fc6025b0455fa4fd93a3e96ef70))
- Fix estate GPU visibility overrides (#447) ([#447](https://github.com/noonghunna/club-3090/pull/447) by @Whamp)
- c3: auto-load container drill detail on select + consistent Esc ([1252921](https://github.com/noonghunna/club-3090/commit/12529211cb5e028d228568665539a6d9bd81fe6a))
- c3 catalog: instant first-paint + batched fit via kv-calc --fit-all ([24c1f19](https://github.com/noonghunna/club-3090/commit/24c1f194d4767b8af88c2d1c5b1b114dc23a10c4))
- Add kv-calc --fit-all: batch fit for all registry slugs in one process ([6653660](https://github.com/noonghunna/club-3090/commit/665366054f82d1077385d4a9b913b1e6e8081d34))
- Speed up c3 catalog load ~10x (parallelize per-slug enrichment) ([65a7f36](https://github.com/noonghunna/club-3090/commit/65a7f3616f5a3e46c99ff588ffaf85269fe5df12))
- Fix LMCache compose entrypoint for Compose v5 interpolation ([#429](https://github.com/noonghunna/club-3090/pull/429) by @steamEngineer)
- Fix c3 cockpit keyboard hotkeys: context-gated footer + focus ([e881738](https://github.com/noonghunna/club-3090/commit/e881738d3e82ee02305c2c113e4a901ad851225b))
- Phase 5 (v2): wire the c3t-Evaluate / promote-to-catalog / optimizer hooks ([b7b4b7d](https://github.com/noonghunna/club-3090/commit/b7b4b7d5fa2235cabf983d88a96a30f1f09ad795))
- Phase 4 (v1.5): wire the Validate mode + Estate write-extras ([581da4f](https://github.com/noonghunna/club-3090/commit/581da4f71d0fcafd0aeab2142660b00a0e272d43))
- Phase 3: wire cockpit panes to real data + reconcile-gated actions ([e5823bc](https://github.com/noonghunna/club-3090/commit/e5823bca0232f960cfa9901f501320fefde78d94))
- Phase 2b: add --json/CLI contracts to 7 stack scripts (additive) ([d09a063](https://github.com/noonghunna/club-3090/commit/d09a063317b1af94eb49084bc2545ee973463aae))
- Phase 2a: extract club3090_tui_core, repoint c3t + cockpit ([8bcbcd1](https://github.com/noonghunna/club-3090/commit/8bcbcd1846bca142a385ae84d67f757912ec1ada))
- Apply Codex Phase-1 review: harden registry shell call + mark bench mock ([450ece0](https://github.com/noonghunna/club-3090/commit/450ece0fa7048da516db208a02f02089660b0c6b))
- Mirror c3t shell: wider rail + persistent Estate status card ([21c66a6](https://github.com/noonghunna/club-3090/commit/21c66a6b13fd4613f5927a908fabeb7c8db34227))
- Enrich Phase-1 cockpit with representative visual mockups ([128c0ed](https://github.com/noonghunna/club-3090/commit/128c0ed8c5cd77e0a457bef3d68a535328e59321))
- Clarify c3 ⏎ action (per-mode) + fix mode-rail truncation ([b966f00](https://github.com/noonghunna/club-3090/commit/b966f005aee214980b4fd10049ee80497df8d5e0))
- Add serve cockpit (c3) — Phase 1 walking skeleton ([b520dfb](https://github.com/noonghunna/club-3090/commit/b520dfb9801bbb4160bc37aa32de3b50251cf0d8))
- Add NVLink auto-config to qwen3.6-35b-a3b dual (Production — rebench-gated) ([491226f](https://github.com/noonghunna/club-3090/commit/491226feadb12f54955539c75ec79c7466f8743c))
- Add NVLink auto-config to gemma-4-26b-a4b + diffusiongemma duals ([7481aa7](https://github.com/noonghunna/club-3090/commit/7481aa7351700067a90ca094eda2c899ee4c7eb2))
- Fix NVLink auto-detect dead on Docker Compose v5.1+ ([9a188b8](https://github.com/noonghunna/club-3090/commit/9a188b816b3d1b977f7f789e7e7dcad0aab16f63))
- Add c3t — the club-3090 test-console TUI (built by Qwen Max from the spec) (#428) ([#428](https://github.com/noonghunna/club-3090/pull/428) by @noonghunna)
- Record measured pi-reasoning rebench-full results + BENCHMARKS row ([79173b1](https://github.com/noonghunna/club-3090/commit/79173b12d5e5221be258efbd67051dde9b0ab1ca))
- quality-test / rebench-full: forward --max-tokens to benchlocal-cli (#426) ([#426](https://github.com/noonghunna/club-3090/pull/426) by @noonghunna)
- Correct pi-reasoning bench framing: 230W power-cap artifact, MTP head ≡ base ([78c4038](https://github.com/noonghunna/club-3090/commit/78c40388502735dbeb795e05a15969dc145aeab6))
- Pull gate: actionable messages for uncurated derives of curated models (#424) ([#424](https://github.com/noonghunna/club-3090/pull/424) by @noonghunna)
- LMCache: in-repo L2 default + LMCACHE_L2=1 toggle, preflight disk-check, corrected sizing (#422) ([#422](https://github.com/noonghunna/club-3090/pull/422) by @noonghunna)
- Correct LMCache L2 docs with measured numbers (fs adapter, ~125 KB/token, 4.8s rehydrate) ([9901129](https://github.com/noonghunna/club-3090/commit/9901129781ed6019aec15209a39c65d10cca46ad))



[Pin: `git checkout v0.10.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.9.0...v0.10.0)
## v0.9.0 — 2026-06-15


### ✨ Features

- feat(weights): optional revision: pin in weights-fetch schema (#319) (#408) ([#408](https://github.com/noonghunna/club-3090/pull/408) by @deucebucket)
- feat(bench-agentic): decouple context ramp from tool-call success (#255) (#396) ([#396](https://github.com/noonghunna/club-3090/pull/396) by @noonghunna)
- feat(rebench): make the 8-pack opt-in via --with-8pack-thinking (default-skip) (#338) (#395) ([#395](https://github.com/noonghunna/club-3090/pull/395) by @noonghunna)
- feat: add Qwen3.6-40B-Deckard to catalog (llamacpp/deckard40B-dual-mtp) ([5dafaa6](https://github.com/noonghunna/club-3090/commit/5dafaa61fea207302aac4b273df667d92162ad65))
- feat(vllm): opt-in offline/air-gap support across vLLM composes (#318) ([edd77cf](https://github.com/noonghunna/club-3090/commit/edd77cfd465e7da038349894fbeb330ed7eeea61))


### 🎯 New models + serving paths

- Add llamacpp/hauhaucs-35ba3b-dual uncensored MTP compose (🧪) (#410) ([#410](https://github.com/noonghunna/club-3090/pull/410) by @noonghunna)
- Add Carnice-V2-27B beellama single-card compose (beellama/carnice-v2-single-q5km-mtp) (#406) ([#406](https://github.com/noonghunna/club-3090/pull/406) by @noonghunna)
- Add experimental DiffusionGemma dual fp8 vLLM compose + patches ([83e8353](https://github.com/noonghunna/club-3090/commit/83e8353ed48963ff8c89fda8b12d54be6b807dd4))
- Add Qwen3.6-27B fast/max tiers across dual + multi4 (4 slugs) (#340) ([#340](https://github.com/noonghunna/club-3090/pull/340) by @noonghunna)
- Add Qwen3-Omni-30B-A3B vLLM-Omni compose (non-registry, 2x3090) (#317) ([#317](https://github.com/noonghunna/club-3090/pull/317) by @noonghunna)


### 🐛 Bug fixes

- fix(gpu-mode): stop gemma-4-12b chat in video-studio + comfyui modes ([#376](https://github.com/noonghunna/club-3090/pull/376) by @noonghunna)
- fix(35b-a3b): bump preview-single gpu-mem 0.92->0.95 for v0.22.0 boot ([195bd47](https://github.com/noonghunna/club-3090/commit/195bd470abb2ab8452061d88fc79d5f02504d155))
- fix(35b-a3b): correct AutoRound INT4 hf_repo (#316) ([121330c](https://github.com/noonghunna/club-3090/commit/121330c95e83600b7b83fc5da974b58808b18cfd))


### 📝 Documentation

- docs(UPSTREAM): #39598 MTP-streaming tool-call drop is un-mitigated on v0.22.0 (P64 retired); #145 closed ([8e38f7a](https://github.com/noonghunna/club-3090/commit/8e38f7a5ecd5b7d25e0c67a533ae021bee2d4d41))
- docs(BENCHMARKS): add author-rig Cerebellum-v3 data point (35B-A3B, #390/#393) ([85087ad](https://github.com/noonghunna/club-3090/commit/85087ad5b55dd8526c6e268f79489847eb655695))
- docs(UPSTREAM): mark #40361 closed-superseded by #45295; track v0.23.0 pin ([18656e7](https://github.com/noonghunna/club-3090/commit/18656e7b8d215e28fc02f6e7ea4ce2b7f198dbaf))
- docs(ARCHITECTURE): drop pinned version on 'surfaces (current)' marker ([fb114bf](https://github.com/noonghunna/club-3090/commit/fb114bfde9e451cb339415b1fd20dce650d34c8b))
- docs(README): refresh 35B-A3B row + drop stale NEW v0.7.3 tags ([a36d2c0](https://github.com/noonghunna/club-3090/commit/a36d2c0911d9d10be2231925b2d2e035987fef33))
- docs(README): un-stale 35B-A3B vLLM-dual row (preview→Production, 262K) ([0689fc4](https://github.com/noonghunna/club-3090/commit/0689fc4875969447b571d760b06c3f4063e29cc2))
- docs(README): correct 35B-A3B llama.cpp ❌→✅ (mainline serves the MoE; #390) ([559a3fe](https://github.com/noonghunna/club-3090/commit/559a3febf72ef11779b470a5fe0223b927a17aca))
- docs(ai-studio): add architecture diagrams to README/image/audio ([81b7157](https://github.com/noonghunna/club-3090/commit/81b71579ca8cd0f908ee07ccc92344ddf0930d9a))
- docs(ai-studio): scope the umbrella to image/video/audio — text is the core stack ([fd81dc2](https://github.com/noonghunna/club-3090/commit/fd81dc26de23e4b6a24cf8f4c6979a4d43e66bda))
- docs: relocate Studio docs into docs/ai-studio/ + add overview & audio (#386) ([#386](https://github.com/noonghunna/club-3090/pull/386) by @noonghunna)
- docs(video-studio): boxed architecture diagram + Quickstart/First-run (match IMAGE_STUDIO) ([96794a0](https://github.com/noonghunna/club-3090/commit/96794a064c8a3264579ab1331ab195eac660e5b7))
- docs(video-studio): add a user-facing 'How to prompt' section ([4abdadd](https://github.com/noonghunna/club-3090/commit/4abdaddf0e6f1f1416f75499c7a4b70993ec98f7))
- docs: reconcile DUAL/MULTI_CARD for the qwen fast/max tiers (#340) ([c51dc23](https://github.com/noonghunna/club-3090/commit/c51dc23dde57fe60a129219cdbb1bd40ff5d252b))
- docs(ADDING_MODELS): quant & arch gotchas checklist (Step 1) ([21d01bd](https://github.com/noonghunna/club-3090/commit/21d01bdaad06b0723518200268ed333d7182d82e))
- docs: quant-fidelity (KLD/QAT) + the Ampere KV-dtype traps (int8-PTH native, fp8-guard) ([3382a47](https://github.com/noonghunna/club-3090/commit/3382a47f40d21fd68635e044fa4db7f1a4c90c49))
- docs(upstream): repoint Gemma engine-pin row to vllm-gemma-stable + add vllm-stable row (#324 fidelity) ([09a4482](https://github.com/noonghunna/club-3090/commit/09a448238d53d1c30573ab612288421a864a742c))
- docs(adding-models): extend coherence rules — patch roles, engine identity/channels, test matrix ([b314ea9](https://github.com/noonghunna/club-3090/commit/b314ea97a7a5dfc96f9897117188647869518954))
- docs(internals): drop stale /opt/ai vLLM-clone path (marlin-pad overlay is vendored) ([4730406](https://github.com/noonghunna/club-3090/commit/47304065f45c10f7bb7ca344ee8e9d3a530f391a))
- docs: drop stale /opt/ai vLLM-clone instructions (marlin-pad is vendored + auto-mounted) ([47f6291](https://github.com/noonghunna/club-3090/commit/47f6291ad2947de6e0dd40814ccdf7d30ff16a93))
- docs(faq,dual-card): image/video gen guidance + 2x3090 multimodal lessons ([924883c](https://github.com/noonghunna/club-3090/commit/924883c560a629762fa5e85edea14002a9ebb1bc))
- docs(qwen3-omni): add image/video gen options, UI recommendations, 2x3090 lessons ([8c1370b](https://github.com/noonghunna/club-3090/commit/8c1370b6879eea22e7a250e33350c8227fb91cc4))
- docs(qwen3-omni): correct text decode to ~164 tok/s (12 was the audio path) ([4eed607](https://github.com/noonghunna/club-3090/commit/4eed6070042a8c469c596967c347c51360a98cdb))
- docs: document the compose filename convention (serving-feature delta) ([edc44c4](https://github.com/noonghunna/club-3090/commit/edc44c4367fe9b423776a803a83fcdf31f69c4c4))
- docs(ADDING_MODELS): document the registry slug-naming convention ([28794da](https://github.com/noonghunna/club-3090/commit/28794dabffddff566fed7ab24fd6e4ba32875b8b))
- docs(UPSTREAM): record gemma4 p-RoPE fix shared on vllm#39914 ([d391f32](https://github.com/noonghunna/club-3090/commit/d391f3272f211811aae9f885edabaee06b8cfeba))
- docs(ADDING_MODELS): accommodate Codex registry-onboarding review ([d080757](https://github.com/noonghunna/club-3090/commit/d080757c2dc29c75b3bc08cd27dcfb5402443edd))
- docs(WSL_SETUP): add 'Expose the API to your local network (LAN)' section ([d11f29d](https://github.com/noonghunna/club-3090/commit/d11f29dcf6765336078740263cc9472388025b10))
- docs(UPSTREAM): track Gemma-4-12B unified 256K large-prefill OOB (vllm#39914) ([960cbf4](https://github.com/noonghunna/club-3090/commit/960cbf4763c936330f6291c74d49062ee73d2800))


### 🛠️ Scripts + tooling

- verify-full: add streaming tool-call check + a reusable streaming probe (#404) ([#404](https://github.com/noonghunna/club-3090/pull/404) by @noonghunna)
- setup.sh: make Genesis opt-in, stop the default stale-pin clone (#182) (#399) ([#399](https://github.com/noonghunna/club-3090/pull/399) by @noonghunna)
- scripts: auto-detect served model from /v1/models in verify/bench (#372) (#388) ([#388](https://github.com/noonghunna/club-3090/pull/388) by @noonghunna)
- setup.sh: add diffusiongemma-26b-a4b dispatch (fetch fp8 weights) ([8ffa46d](https://github.com/noonghunna/club-3090/commit/8ffa46d51dd50099c385b413968eed6af42d8e77))
- bench.sh: FORCE_TOKENS — bench at a fixed/larger output size ([0948e6b](https://github.com/noonghunna/club-3090/commit/0948e6bbea6f692b29a503d651d89ac08eff404f))
- switch.sh --owui: auto-register a launched model in Open WebUI ([111c72b](https://github.com/noonghunna/club-3090/commit/111c72b0eabd741d689dd474c3a4c8ecad83a93b))
- preflight: resolve ${VAR:-default} model paths in compose-deps check ([345f24e](https://github.com/noonghunna/club-3090/commit/345f24ee1d7f83a915b1aac40013cca59684f70b))
- preflight: detect endpoint by engine port, not a model-name allowlist (#310) ([f48a7ff](https://github.com/noonghunna/club-3090/commit/f48a7ff51411cf547a1fed5a1e48851f20e522c3))


### 🧹 Maintenance

- test(weights): opt-in HF-repo resolve guard (#320) (#409) ([#409](https://github.com/noonghunna/club-3090/pull/409) by @deucebucket)
- chore: make setup script work with WEIGHTS=FP8 and qwen3.6-27b (#369) ([#369](https://github.com/noonghunna/club-3090/pull/369) by @hlo-world)
- chore(327): archive the Qwen3.6-27B DFlash path + deprecate vllm-nightly-dflash ([cc3e906](https://github.com/noonghunna/club-3090/commit/cc3e906f8ca8338f8acbb38b43e98b9b0035daad))
- chore(254): deprecate the now-unused Genesis nightly engines + patches [Phase 3+4] ([baac1ac](https://github.com/noonghunna/club-3090/commit/baac1acafdcd07ec36e5fc51c8c3514e3e424b09))
- chore(254): archive 15 Genesis/nightly composes + migrate the cascade [WIP: 5 test fixtures] ([b5ef9e6](https://github.com/noonghunna/club-3090/commit/b5ef9e697fe42982a05ba94f4a8e90b76c0b3ef2))
- chore(254): migrate qwen-a3b-preview-single off purged nightly-clean -> vllm-stable ([36ee7bb](https://github.com/noonghunna/club-3090/commit/36ee7bbd9470f651fca9ecd9b8abee2bb707bc98))
- refactor(vllm): reconcile vLLM engines to v0.22.0 — two-engine split (#254) ([a9ffb53](https://github.com/noonghunna/club-3090/commit/a9ffb532efcea285982d036bfde07b301ca44ffc))


### 🧹 Other

- BENCHMARKS: add DiffusionGemma 26B-A4B section + cross-rig row (#405) ([0542de8](https://github.com/noonghunna/club-3090/commit/0542de8c573d672d6e687d289c715e61c93850fd))
- Fix soak-test container auto-detect: match by engine port, not model allowlist (#405) (#414) ([#414](https://github.com/noonghunna/club-3090/pull/414) by @noonghunna)
- Pin hauhaucs-35ba3b-dual weights to morikomorizz commit 49a080d (#319) (#413) ([#413](https://github.com/noonghunna/club-3090/pull/413) by @noonghunna)
- qwopus-coder compose: sanitize /opt/ai/hf-download.sh leak in Quick-start -> generic hf download ([69f8632](https://github.com/noonghunna/club-3090/commit/69f8632083d58e651097c77002800638fa4b8eaa))
- deckard-40b: anti-loop sampling defaults (rep-penalty 1.1 + repeat-last-n 256) + DRY opt-in (#402) ([#402](https://github.com/noonghunna/club-3090/pull/402) by @noonghunna)
- ik-llama: migrate spec-dec flags to --spec-type + digest-pin the image (#401) ([#401](https://github.com/noonghunna/club-3090/pull/401) by @noonghunna)
- Add curated quality-baseline corpus + auto-diff (#252 Phase 1) (#397) ([#397](https://github.com/noonghunna/club-3090/pull/397) by @noonghunna)
- catalog: add beellama/qwopus-coder (Qwopus3.6-27B-Coder, KVarN-4) — first KVarN compose (#391) ([#391](https://github.com/noonghunna/club-3090/pull/391) by @noonghunna)
- beellama: bump pin v0.3.0 → v0.3.2-preview (commit-pinned) + validate (#389) ([#389](https://github.com/noonghunna/club-3090/pull/389) by @noonghunna)
- setup: fix fp8 weights nits after #369 (typo, help string, manual_note) ([eae0e36](https://github.com/noonghunna/club-3090/commit/eae0e36f81877170f06549fb4017dfa635af1e9a))
- openwebui: pin OWUI task models to the qwen director (not the generation pipe) ([d7c6e6d](https://github.com/noonghunna/club-3090/commit/d7c6e6d7b1aefe8e1f42a642c88b9934285228f3))
- studio(pipe): ignore OWUI internal task prompts — stop rendering them as images ([0dddfa7](https://github.com/noonghunna/club-3090/commit/0dddfa790eb9824fc4202eb245cc1d43f872e2f7))
- studio(pipe): drop the visible <!--SPEC--> refine marker — read 'Prompt used:' instead ([892324f](https://github.com/noonghunna/club-3090/commit/892324f436f35dc850f40aa7c47bc24c7b558242))
- studio: add push-pipe-to-owui.sh — update the installed OWUI function in one command ([d67a067](https://github.com/noonghunna/club-3090/commit/d67a06773c0596b31761aa732ec795e3cd51f7f1))
- studio(pipe): intent-gate the director — don't render on greetings/small-talk (#387) ([#387](https://github.com/noonghunna/club-3090/pull/387) by @noonghunna)
- studio: add 🎙️ Voice lane — Step-Audio-EditX premium clone (isolated service) (#385) ([#385](https://github.com/noonghunna/club-3090/pull/385) by @noonghunna)
- studio: add ✨ HiDream-O1 image lane (top-quality / photoreal) (#384) ([#384](https://github.com/noonghunna/club-3090/pull/384) by @noonghunna)
- studio: add 🔊 SFX lane (Stable Audio Open) — sound effects + ambient (#383) ([#383](https://github.com/noonghunna/club-3090/pull/383) by @noonghunna)
- studio: add 🎵 Music lane (ACE-Step) — songs + instrumentals ([#382](https://github.com/noonghunna/club-3090/pull/382) by @noonghunna)
- studio: integrated voices for video — Kokoro TTS + layer-aware audio mixdown ([#381](https://github.com/noonghunna/club-3090/pull/381) by @noonghunna)
- studio: add Chroma uncensored image lane (the "Sulphur for stills") ([#380](https://github.com/noonghunna/club-3090/pull/380) by @noonghunna)
- studio: fix OWUI native 🖼️ image button via ComfyUI reverse-proxy shim ([#379](https://github.com/noonghunna/club-3090/pull/379) by @noonghunna)
- studio: fix image-lane _min_caption fallback (empty arrays hard-block) ([#378](https://github.com/noonghunna/club-3090/pull/378) by @noonghunna)
- studio: add 🖼️ Image lane (Ideogram-4) with JSON-caption director ([#377](https://github.com/noonghunna/club-3090/pull/377) by @noonghunna)
- studio: in-chat 60s+ via orchestrator (auto-chain segments -> one video) ([#375](https://github.com/noonghunna/club-3090/pull/375) by @noonghunna)
- studio: extend-chain tool for 60s+ video (proven, seamless joins) ([#374](https://github.com/noonghunna/club-3090/pull/374) by @noonghunna)
- studio: video lane (LTX-2.3 / Sulphur) — Director pipe, gallery, gpu-mode, docs ([86f7e5e](https://github.com/noonghunna/club-3090/commit/86f7e5e835441e1d9aad4f445f82d14dc84b5c69))
- comfyui: claim pip cache for container uid so it persists across recreates ([#370](https://github.com/noonghunna/club-3090/pull/370) by @noonghunna)
- soak-helper: count silent-empty by completion_tokens, not decode_tps ([#368](https://github.com/noonghunna/club-3090/pull/368) by @noonghunna)
- quality-test: add --no-thinking (symmetric force-off for reasoning A/B) ([#367](https://github.com/noonghunna/club-3090/pull/367) by @noonghunna)
- soak-test: add beellama- to container auto-detect glob (#362) ([#366](https://github.com/noonghunna/club-3090/pull/366) by @noonghunna)
- UPSTREAM: dgemma #45163 — add promote-to-caveats trigger (hold at experimental for now) ([ffd525a](https://github.com/noonghunna/club-3090/commit/ffd525a7f673fd0ece4d662fd861a3f4d28a0417))
- dgemma: adopt official vllm/vllm-openai:gemma image (drop 120-file sideload) ([0cfd099](https://github.com/noonghunna/club-3090/commit/0cfd099b37f254ef817c7a9462373b0b8f3bbe2b))
- gpu-mode: add `dgemma` mode (DiffusionGemma 26B-A4B dLLM, dual-card) ([f8ad9a0](https://github.com/noonghunna/club-3090/commit/f8ad9a01c1d25e3daa96ff556f225f5b306b7e9f))
- Wire DiffusionGemma into the catalog via stock-nightly sideload ([af1e909](https://github.com/noonghunna/club-3090/commit/af1e909b04ee9c4504d143ab2483c9344615c219))
- gpu-mode: add `deckard` mode (uncensored 40B, dual-card) ([1cee8eb](https://github.com/noonghunna/club-3090/commit/1cee8eb43d600c6929be06dad301f00d5cbc0354))
- test-measurement-record: make BENCH_MOCK capture hermetic (#478) ([71a05fd](https://github.com/noonghunna/club-3090/commit/71a05fd444e9e723eaf3d1ee87eaf93c4e09a737))
- Promote Deckard-40B to ✅ Production; fix arch + slug naming ([93acbf9](https://github.com/noonghunna/club-3090/commit/93acbf979f897383630b68be2e6bf199d16949fa))
- Deckard-40B: fix provenance — GGUF is PiehSoft's, wire hf_repo fetch ([d6725fa](https://github.com/noonghunna/club-3090/commit/d6725faf0259a344277175070557f9609a0ec002))
- Deckard-40B: record soak PASS + final 105/150 (gates all green) ([20e1d6f](https://github.com/noonghunna/club-3090/commit/20e1d6f36272e1cfc844c7b205b2720030f22a59))
- Fix deckard40B compose: pin b9570, disk-count 45, status word ([881a449](https://github.com/noonghunna/club-3090/commit/881a449d4172d32c58d1d1a9e51e1396c5776b83))
- image-studio setup: real preflight (docker/GPU/disk/hf/chat-model) before heavy work ([caca7b8](https://github.com/noonghunna/club-3090/commit/caca7b8cba3a8903f2a6e3d9cb679d6c2a5eaa0b))
- image-studio P1 follow-up: setup UX, LiteLLM route, architecture docs ([18902fa](https://github.com/noonghunna/club-3090/commit/18902fa49571e124d86f50d60a1d2416bff1a996))
- image-studio P1 docs: add IMAGE_STUDIO.md + index/README/FAQ pointers ([c28470c](https://github.com/noonghunna/club-3090/commit/c28470c5f869b7e698ec908c61c2550c806f4ddc))
- image-studio P1: pin ComfyUI commit + drop hardcoded OWUI secret ([092b7c7](https://github.com/noonghunna/club-3090/commit/092b7c7a46f822f3f159dbcdfefa5c5e4118226a))
- Add image-studio bundle P1: Ideogram-4 + gemma-12b chat + gpu-mode mode ([9981b28](https://github.com/noonghunna/club-3090/commit/9981b286980ab0a2733ec4ab77c768aba71877a8))
- Add vllm/qwen-27b-dual-balanced + record 3-way 8-pack A/B (tie) (#343) ([#343](https://github.com/noonghunna/club-3090/pull/343) by @noonghunna)
- Set 31B w4a16 default MTP n=4->3 (n-swept optimum) + update A/B record ([31dc2c4](https://github.com/noonghunna/club-3090/commit/31dc2c47569dd8e152b69f95f7957907731cb21c))
- Record 31B w4a16 8-pack A/B: 109/150 vs autoround 105 (comparable quality, weaker MTP) ([97b678e](https://github.com/noonghunna/club-3090/commit/97b678e4056f334c34222959dfd5e5255bc98aca))
- Experimental Gemma-4 QAT W4A16 vLLM composes + kv-calc int4 fix (#339) ([#339](https://github.com/noonghunna/club-3090/pull/339) by @noonghunna)
- gemma-26ba4b-single: promote INT8-PTH single → ⚠️ Production-w/-caveats (gate PASS) ([3a5ece7](https://github.com/noonghunna/club-3090/commit/3a5ece7bb55e35f950cfd15e66d4ef65767bd84b))
- gemma-26ba4b: add --reasoning-parser gemma4 (single int8 + dual) ([0375d89](https://github.com/noonghunna/club-3090/commit/0375d8977201504a3c16d684261a2177ea66af16))
- Repoint vllm/gemma-26ba4b-single to INT8-PTH long-ctx (#465) ([e2efb75](https://github.com/noonghunna/club-3090/commit/e2efb757a9eef7cae0486747e12487a0d6c4b83c))
- gpu-mode: add power-cap on/off/status controls ([87190f9](https://github.com/noonghunna/club-3090/commit/87190f90fac27fdc28498bb4d103eee23bb9c0b4))
- kv-calc: per-sequence KV-pool floor (capped at 1 GB), fix KV-light false-FAIL ([95e47fe](https://github.com/noonghunna/club-3090/commit/95e47fedc950ab5f7618756f069f95638191d565))
- Enable MTP on gemma-4-26b-a4b single + ladder both to max ctx (#326) ([95e1448](https://github.com/noonghunna/club-3090/commit/95e1448217cb7bc2f5db8c6adb318d4a76b6455c))
- Wire gemma-4-26b-a4b AWQ on stock vLLM v0.22.0; retire AutoRound (#326) ([48eaf82](https://github.com/noonghunna/club-3090/commit/48eaf8283c1e5ed3cd383fe0657383208f83bb04))
- Fix Genesis cleanup test fixtures ([6b26bd7](https://github.com/noonghunna/club-3090/commit/6b26bd7ffdd3f044115510a318f442cfdd31bd34))
- gemma-4-12b: record soak PASS in the 2 GGUF single-card headers ([5387614](https://github.com/noonghunna/club-3090/commit/5387614ad7f1f6bb27cc48652a652b94089dcfb9))
- gemma-4-12b: promote the two vLLM MTP composes to ⚠️ Production w/ caveats ([90c67dc](https://github.com/noonghunna/club-3090/commit/90c67dcc9f2a288359f16636bab23ee79eb201ec))
- gemma-4-12b: normalize gemma-12b slug naming + prune the no-MTP bases ([87b43f1](https://github.com/noonghunna/club-3090/commit/87b43f14d6573f607052d114b59583eed4ca52e3))
- gemma-4-12b int8: add 8-pack quality (105/150) to BENCHMARKS + compose headers ([4c390e7](https://github.com/noonghunna/club-3090/commit/4c390e7e22f6cd048805fbdc70fda3845664874a))
- gemma-4-12b: add single-card int8 MTP variant (n-swept) + register it ([077584a](https://github.com/noonghunna/club-3090/commit/077584ac0ab4580e6cf62b0ae45c82d947df9347))
- gemma-4-12b: add single-card vLLM INT8 compose (full 256K on one 3090) ([b444189](https://github.com/noonghunna/club-3090/commit/b4441892d397eab391a9930d8ec35a00c16de01e))
- gemma-4-12b: drop vendored p-RoPE overlay — upstream config fix supersedes it ([942c5c8](https://github.com/noonghunna/club-3090/commit/942c5c891fe3f310fb9af9c222922358ba923d4f))
- gemma-4-12b: fold rebench-full verdict into vLLM compose headers + BENCHMARKS ([8c4bcab](https://github.com/noonghunna/club-3090/commit/8c4bcab6f3b51c4cf2bb5c69afb547bd0ba9d69d))
- BENCHMARKS: Gemma-4-12B section (256K all-engine; vLLM 256K+MTP via p-RoPE overlay) ([96e9852](https://github.com/noonghunna/club-3090/commit/96e9852f80adcd410fb7f1a121420f91b546468d))
- gemma-4-12b: register single-card slugs beellama/gemma-12b + llamacpp/gemma-12b ([8b459af](https://github.com/noonghunna/club-3090/commit/8b459afa5862bfd4f416f1d96e0d0ee69ab546ec))
- gemma-4-12b vLLM: 256K via vendored gemma4 p-RoPE long-ctx overlay (#39914) ([872bb89](https://github.com/noonghunna/club-3090/commit/872bb898081d4053d7ac6e45974526aa88d327d1))
- gemma-4-12b mainline llama.cpp: single-card 256K (override-kv, q8_0 KV) ([8c1f811](https://github.com/noonghunna/club-3090/commit/8c1f811dd2244900dcce17a7e6ccad8e874c2dd5))
- gemma-4-12b composes: scrub internal /opt/ai/hf-download.sh from quick-start docs ([ab825e5](https://github.com/noonghunna/club-3090/commit/ab825e5ea46a1f8ee2c5dd3e8af32b00185d2985))
- gemma-4-12b beellama single-card: 256K via --override-kv (native p-RoPE) ([a63ccb0](https://github.com/noonghunna/club-3090/commit/a63ccb0195b7dc35864531cbfa53ae24fe7090c3))
- kv-calc: address Codex review notes (gemma4_unified calibration) ([7f5ef72](https://github.com/noonghunna/club-3090/commit/7f5ef72a40e092c982a6f298304128db461e9021))
- kv-calc: calibrate gemma4_unified global-KV per-token to measured anchor ([0f6f861](https://github.com/noonghunna/club-3090/commit/0f6f86195eb47069c3acfe78a00bf9a705fc5a42))
- Wire gemma-4-12b into curated catalog (vLLM gemma4-unified, bf16 + MTP) ([aff9890](https://github.com/noonghunna/club-3090/commit/aff989072089e8afe9b0c8f3466a93fecc3a19c8))



[Pin: `git checkout v0.9.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.7...v0.9.0)
## v0.8.7 — 2026-06-03


### ⚠️ Cliffs, gotchas, regressions

- Retract beellama v0.3.0 prose-DFlash "regression" — net-positive on tok/s ([f2b9dc2](https://github.com/noonghunna/club-3090/commit/f2b9dc2de45203747e7918dcda15d29f19c95da1))


### ✨ Features

- feat(35b-a3b): byteshape IQ4_XS ik-llama single-card preset — validated (#299) ([#299](https://github.com/noonghunna/club-3090/pull/299) by @noonghunna)
- feat(beellama): release v0.3.0 Q8_K_XL dual composes (experimental) + centralize image pin (#296) ([#296](https://github.com/noonghunna/club-3090/pull/296) by @noonghunna)
- feat(gemma): gemma duals → vLLM v0.22.0 (rebase #40391 lean + re-instate #42006) (#287) ([#287](https://github.com/noonghunna/club-3090/pull/287) by @noonghunna)
- feat(beellama): add beellama.cpp DFlash as first-class compose engine (#268) ([#268](https://github.com/noonghunna/club-3090/pull/268) by @noonghunna)
- feat(switch): hardware-filter --list by GPU count; add --all (#267) ([#267](https://github.com/noonghunna/club-3090/pull/267) by @noonghunna)
- feat: model-default resolver + user-pinnable defaults (PR-B) (#266) ([#266](https://github.com/noonghunna/club-3090/pull/266) by @noonghunna)
- feat(registry): slug health/availability flag (#265) ([#265](https://github.com/noonghunna/club-3090/pull/265) by @noonghunna)
- feat(switch): group --list by model · topology (#264) ([#264](https://github.com/noonghunna/club-3090/pull/264) by @noonghunna)
- feat(qwen3.6-35b-a3b): promote dual → 262K + vision Production (vllm/qwen-35b-a3b-dual) (#259) ([#259](https://github.com/noonghunna/club-3090/pull/259) by @noonghunna)


### 🐛 Bug fixes

- fix(nvlink): add NVLINK_MODE=pcie_p2p for PCIe P2P without NVLink (#290) (#291) ([#291](https://github.com/noonghunna/club-3090/pull/291) by @noonghunna)
- fix(report): detect beellama- containers in report.sh auto-detection ([8cb5027](https://github.com/noonghunna/club-3090/commit/8cb50273d9a04636c64c491801549f05227dcbef))
- fix(gemma): wire --reasoning-parser gemma4 on both duals (#289) ([#289](https://github.com/noonghunna/club-3090/pull/289) by @noonghunna)
- fix(kv-calc): shard qwen3-next-moe weights by TP (#260) (#261) ([#261](https://github.com/noonghunna/club-3090/pull/261) by @noonghunna)


### 📝 Documentation

- docs: surface the mandatory compose Profile header to contributors ([8f3ae5c](https://github.com/noonghunna/club-3090/commit/8f3ae5c62faf83a84997eae55cabcaa6e7da97ad))
- docs(BRING_YOUR_OWN): BYO models aren't registered — boot composes directly ([3b8e469](https://github.com/noonghunna/club-3090/commit/3b8e469f9f7dfd3b715656e3cda7bb211b36b9fe))
- docs(BRING_YOUR_OWN): note INT8 KV is a vendored patch, not a stock dtype ([6ddfaca](https://github.com/noonghunna/club-3090/commit/6ddfaca166b256c4ac0f78a6b9e083fc3a1cfa6d))
- docs(BRING_YOUR_OWN): split KV-quant guidance by engine ([a2d0af6](https://github.com/noonghunna/club-3090/commit/a2d0af6053a63e2c6fbc1a1c8a953d5f20734167))
- docs: add BRING_YOUR_OWN — serve/tune/validate your own model ([93e6cd8](https://github.com/noonghunna/club-3090/commit/93e6cd807db9214862a8c36f451acec834976dec))
- docs(CONTRIBUTING): broaden one-per-PR rule to feature/concern scope ([557e615](https://github.com/noonghunna/club-3090/commit/557e6157c440a3f1d7840ac128b9ed0923265266))
- docs(CONTRIBUTING): require one model per compose PR ([497c341](https://github.com/noonghunna/club-3090/commit/497c341a958deec6f05813a3c6303a4880254817))
- docs(FAQ): fix Copilot LLM Gateway entry — vllm/tools-text retired ([aa458bb](https://github.com/noonghunna/club-3090/commit/aa458bb34a66806f50365508726054c7f40249ea))
- docs(FAQ): agent stops mid-task → set client temperature to 0.6 (#232) ([6bc91b5](https://github.com/noonghunna/club-3090/commit/6bc91b5cc50ee3a0c2c8a2e8b09029bb4d1b0c02))
- docs: deprecate vLLM dual-dflash composes, redirect DFlash to beellama (#297) ([b7955ac](https://github.com/noonghunna/club-3090/commit/b7955acab3e3b7914b603b2f1ad0e5b70d3235c3))
- docs(beellama): point gemma-dflash-dual v0.3.0 override at Anbeeld's official image ([533d2c3](https://github.com/noonghunna/club-3090/commit/533d2c320543782dedf36fd70c35b5dd55b0971c))
- docs(bench): @hlo-world PCIe-P2P A/B validates NVLINK_MODE=pcie_p2p (#291/#290) ([55eb500](https://github.com/noonghunna/club-3090/commit/55eb500f63ea8633d197c2c49645556b9c2abb79))
- docs(beellama): correct prose-regression attribution — v0.3.0-wide, not SWA ([44358d1](https://github.com/noonghunna/club-3090/commit/44358d135a0ec3255a06b45d5761f8ceca2c7a97))
- docs(beellama): record v0.3.0 dual-3090 validation — multi-GPU DFlash fixed ([8da27fd](https://github.com/noonghunna/club-3090/commit/8da27fdbdf5c7ec2394e5d5ddf566ce7fa6b12f3))
- docs(gemma): record bf16-mtp 8-pack (reasoning-on 111/150) + cross-config quality summary ([b2d7d8f](https://github.com/noonghunna/club-3090/commit/b2d7d8fb7f1d04584ef5ff5376723478d6851f9a))
- docs(gemma): record int8-mtp v0.22.0 closeout — bench, soak PASS, 8-pack quality + reasoning A/B ([0abba3a](https://github.com/noonghunna/club-3090/commit/0abba3a8757e08b9d9246c73943647a8ba19c499))
- docs(FAQ): clarify model-switch scope — supported models only ([8356a11](https://github.com/noonghunna/club-3090/commit/8356a11e82036938acc42484fe69e0024e8db3f0))
- docs(FAQ): add 'how to switch to / try a different model' entry ([2aeec74](https://github.com/noonghunna/club-3090/commit/2aeec74301fcbf1735e344656f507e66a1d747d7))
- docs(quality-test): document failure-reason reading; fix drifted Failure-breakdown sample ([77c1cd6](https://github.com/noonghunna/club-3090/commit/77c1cd6c012485c642238648527dc70113dab2e1))
- docs(ik-apex-fit-mtp): warn RAM-constrained systems on --cache-ram + MTP OOM (#243) ([c61cfc4](https://github.com/noonghunna/club-3090/commit/c61cfc49b7f064cdcf27a7a7095f8f1cbe5cbfe7))
- docs: document quality-test timeout sizing (QUALITY_TEST.md + agent guide) ([9c333f4](https://github.com/noonghunna/club-3090/commit/9c333f42bb9e60315e645256c63bee2455c7aaf5))
- docs(UPSTREAM): add vLLM #39056 row + cross-reference to #39598 streaming fix (#354) ([0ed228c](https://github.com/noonghunna/club-3090/commit/0ed228c605beb2788112f9373b1f5cd93d87ba40))
- docs(hardware): correct turbo3 "quality-neutral" claim — it's avg-distortion, not tail ([572f1c9](https://github.com/noonghunna/club-3090/commit/572f1c96ca507c386ad1b8c1529d0d6e21205ffe))
- docs(faq): add "which KV-cache quant should I use?" entry ([6b85575](https://github.com/noonghunna/club-3090/commit/6b85575d711ee4e55bcde1af634d91356d99295e))
- docs(cliffs): NIAH certifies retrieval, not KV-quant tail quality ([a307846](https://github.com/noonghunna/club-3090/commit/a307846eb74b6614a09e225216a470956727a164))
- docs(bench): single-card 1x 3090 llama.cpp Gemma-4 rows (mainline FA_ALL_QUANTS + beellama DFlash) ([6b011f6](https://github.com/noonghunna/club-3090/commit/6b011f60909cab2d2e17fb51a4f08859a59d44cc))
- docs: flip CLAUDE.md/AGENTS.md symlink + document localhost requirement for sandboxed agentic packs ([#240](https://github.com/noonghunna/club-3090/pull/240) by @noonghunna)
- docs: make AGENTS.md a symlink to CLAUDE.md (one source, matches the /opt/ai convention) ([ef1d9b8](https://github.com/noonghunna/club-3090/commit/ef1d9b8a1c8c830b20cd6c17750fa593d87386ab))
- docs: link the docs index + ARCHITECTURE from the repo agent entry points ([dc7c613](https://github.com/noonghunna/club-3090/commit/dc7c613262023596991160dab721baaf0852cb4b))
- docs: formalize the add-a-model workflow for the post-refactor architecture + agent discoverability ([37871c6](https://github.com/noonghunna/club-3090/commit/37871c631729018165ce19dd1cdbf1c14e027ccb))


### 🔧 Pin bumps + upstream

- Bump vllm/minimal + vllm/dual to stable v0.22.0 (off nightly) (#277) ([#277](https://github.com/noonghunna/club-3090/pull/277) by @noonghunna)


### 🛠️ Scripts + tooling

- preflight: detect beellama + check spec-draft-model drafter GGUF ([180adb0](https://github.com/noonghunna/club-3090/commit/180adb083eae957e9c4434f2bf70cb54e3874472))
- switch.sh --list: show max context per slug (+ registry<->compose drift) (#283) ([#283](https://github.com/noonghunna/club-3090/pull/283) by @noonghunna)
- gpu-mode.sh: align with dual prune — drop dead 27b modes, gemma -> int8 default (#280) ([#280](https://github.com/noonghunna/club-3090/pull/280) by @noonghunna)
- setup.sh: reject extra positional args instead of silently ignoring them (#273) ([#273](https://github.com/noonghunna/club-3090/pull/273) by @noonghunna)
- quality-test.sh: forward --progress to benchlocal-cli, default ON (#248) ([#248](https://github.com/noonghunna/club-3090/pull/248) by @noonghunna)
- soak-test.sh: add qwen3.6-35b-a3b to container auto-detect glob (#244) ([#244](https://github.com/noonghunna/club-3090/pull/244) by @noonghunna)
- quality-test.sh: stop hardcoding --timeout-per-case 60 (benchlocal-cli #41) (#245) ([#245](https://github.com/noonghunna/club-3090/pull/245) by @noonghunna)


### 🧹 Maintenance

- refactor(gemma): rename dual slugs + ladder gemma-bf16-mtp to 131K (#286) ([#286](https://github.com/noonghunna/club-3090/pull/286) by @noonghunna)
- chore(beellama): default to published ghcr image (sm86-b9459-07ac3ce) (#270) ([#270](https://github.com/noonghunna/club-3090/pull/270) by @noonghunna)
- chore: retire vllm-club3090 image references (stack on stock vLLM) (#269) ([#269](https://github.com/noonghunna/club-3090/pull/269) by @noonghunna)
- chore(vllm): remove redundant nvlink-* dual composes (NVLink auto-detected) (#257) ([#257](https://github.com/noonghunna/club-3090/pull/257) by @noonghunna)
- chore(litellm): prune dead routes, add 35B-A3B dual, drop wildcard (#263) ([#263](https://github.com/noonghunna/club-3090/pull/263) by @noonghunna)
- chore(35b-a3b): drop the built-in-MTP A/B compose (vllm/qwen-a3b-preview-mtp) (#262) ([#262](https://github.com/noonghunna/club-3090/pull/262) by @noonghunna)


### 🧹 Other

- rebench: add fail-fast verify-full preflight + fix runtime skip set (#306) ([#306](https://github.com/noonghunna/club-3090/pull/306) by @noonghunna)
- rebench-full: swap aider step for think-ON 8-pack + live tee (#303) ([#303](https://github.com/noonghunna/club-3090/pull/303) by @noonghunna)
- Repoint stale single-card launcher hints off deprecated vLLM composes ([bb9f9f2](https://github.com/noonghunna/club-3090/commit/bb9f9f23a002ea5e1ae439189872060bf62fb586))
- beellama: refresh stale b9459 image refs to v0.3.0 ([#300](https://github.com/noonghunna/club-3090/pull/300) by @noonghunna)
- Add Results Card doc — standard format for sharing config results ([cabdf12](https://github.com/noonghunna/club-3090/commit/cabdf1286bb744c256ef2c5f19be0738d464de8b))
- Operational robustness (#281): orphan-safe switch.sh · reboot-surviving vLLM · multi-GPU power sweep (#285) ([#285](https://github.com/noonghunna/club-3090/pull/285) by @noonghunna)
- beellama Gemma-4 ctx: single 128K (caveats) + dual 262K parked/upstream-gated (#284) ([#284](https://github.com/noonghunna/club-3090/pull/284) by @noonghunna)
- Prune dual vLLM composes: qwen-27b -> one config; gemma-31b default -> gemma-int8 (#279) ([#279](https://github.com/noonghunna/club-3090/pull/279) by @noonghunna)
- Gemma vLLM -> v0.22.0: bump dual gemma-mtp, deprecate gemma-mtp-tp1 (#278) ([#278](https://github.com/noonghunna/club-3090/pull/278) by @noonghunna)
- Deprecate Genesis vLLM composes; vLLM single default → vllm/minimal (#276) ([#276](https://github.com/noonghunna/club-3090/pull/276) by @noonghunna)
- Align gemma sampling default to model card (top_k -1 -> 64) (#275) ([#275](https://github.com/noonghunna/club-3090/pull/275) by @noonghunna)
- Promote beellama/gemma-dflash to single-card Gemma-4-31B default (#272) ([#272](https://github.com/noonghunna/club-3090/pull/272) by @noonghunna)
- Promote beellama/dflash to single-card Qwen3.6-27B default (#271) ([#271](https://github.com/noonghunna/club-3090/pull/271) by @noonghunna)
- bench-agentic: anchor TTFT-growth to first warm turn (exclude cold-start) ([#114](https://github.com/noonghunna/club-3090/pull/114) by @noonghunna)
- Revive agentic context-depth benchmark ([a9cf70e](https://github.com/noonghunna/club-3090/commit/a9cf70effd756a6eb16d2e07c3d4c42bcb823053))
- Prune Gemma vLLM variants (9->3) and pin survivors to v0.21.0 ([#253](https://github.com/noonghunna/club-3090/pull/253) by @noonghunna)
- Fail loud on hollow measured records in measurement_record (#251) ([#251](https://github.com/noonghunna/club-3090/pull/251) by @noonghunna)
- Add measurement-record producer for bench runs (#249) ([#249](https://github.com/noonghunna/club-3090/pull/249) by @noonghunna)
- BENCHMARKS: flag Gemma mainline-llama.cpp 36.8 TPS row as does-not-reproduce (13.6/15.4 re-bench) ([95da8e1](https://github.com/noonghunna/club-3090/commit/95da8e1f85a1e3db827b107032dbe453922abaee))
- BENCHMARKS: post-PR #38 chain rebench for ik-llama/iq4ks-mtp (8-pack 101→107, aider 19→18 noise) ([8351956](https://github.com/noonghunna/club-3090/commit/8351956d9b9bcfdedde9fccfe995cc9098589769))
- scripts/preflight.sh: surface 4090 / 5090 hint with FAQ deep-link (#247) ([#247](https://github.com/noonghunna/club-3090/pull/247) by @noonghunna)
- docs/README: refresh supported-models table — single-card Gemma 4 31B + 35B-A3B now in production ([285c632](https://github.com/noonghunna/club-3090/commit/285c632442b72227c40231ea84c6731f001e2dc3))
- docs/README: add above-the-fold cross-rig callout for 4090 / 5090 owners ([eea717f](https://github.com/noonghunna/club-3090/commit/eea717fd5b68308fe29f23ab3fe70c192a7d899a))
- docs/FAQ: refresh 4090 / 5090 entries with cross-rig measurements ([2f0a54c](https://github.com/noonghunna/club-3090/commit/2f0a54c99503f1d02b49d06b4563d8807aa85a88))
- Add ik-llama apex-fit-q8q5 variant for Qwen3.6-35B-A3B (#242) (#243) ([#243](https://github.com/noonghunna/club-3090/pull/243) by @noonghunna)



[Pin: `git checkout v0.8.7`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.6...v0.8.7)
## v0.8.6 — 2026-05-26


### ✨ Features

- feat(ik-llama): PRISM-PRO-DQ + APEX-MTP presets, conformed to the <quant>/ layout ([458c473](https://github.com/noonghunna/club-3090/commit/458c473908242aab349a19529d2606feb28b2e3e))


### 🐛 Bug fixes

- fix(profiles): cover the new ik-llama PRISM/APEX presets in the compat catalog ([aa2a965](https://github.com/noonghunna/club-3090/commit/aa2a965bb93ea7c9cf50f6ef16a9bb1a14ce13f0))
- fix: post-PR-A compose-path fixes for gpu-mode.sh + 2 patch READMEs ([b23846e](https://github.com/noonghunna/club-3090/commit/b23846ea36c71ea2d86d3343eb2d367fa5d58b22))
- fix(registry+bench): sync vision defaults to the 2026-05-25 re-tune (#438) ([b116750](https://github.com/noonghunna/club-3090/commit/b116750ef16850ee2472f857cde4dbea6bd09c0a))
- fix(vision): re-tune single-card vision defaults to measured-safe (1M-px + 160K/150K) ([c9b7dd3](https://github.com/noonghunna/club-3090/commit/c9b7dd3999de78cf2847677e60503d1dafec5af9))
- fix(vllm/dual): pin to stable v0.21.0, drop all source overlays (#407 pin-drift) ([cf1f14f](https://github.com/noonghunna/club-3090/commit/cf1f14fbc781ec7fbfca060c791144dd79ab6308))


### 📝 Documentation

- docs+scripts: finish <quant>/ path migration across full repo sweep ([eaa7a8c](https://github.com/noonghunna/club-3090/commit/eaa7a8c1ad4dc36d39d6935944136cc0734e4ef2))
- docs(switch): correct ik-llama/iq4ks-mtp usage comment 262K -> 200K ([2135230](https://github.com/noonghunna/club-3090/commit/2135230f8f02fde978c897526c9b55618c22c7fd))


### 🧹 Maintenance

- refactor(launch): derive launcher tables from the registry + <engine>/default resolver ([a0520e2](https://github.com/noonghunna/club-3090/commit/a0520e2060966a3b63efadf95401f684417b4c62))
- refactor(compose): insert <quant> layer + make the registry the single source of truth ([9821c94](https://github.com/noonghunna/club-3090/commit/9821c94efb4ac837dcaaa6f5805a14b0c8b0a05d))



[Pin: `git checkout v0.8.6`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.5...v0.8.6)
## v0.8.5 — 2026-05-24


### ✨ Features

- feat(llama.cpp): Structured-CoT bounded-thinking compose + grammar-dialect fix (#214) ([#214](https://github.com/noonghunna/club-3090/pull/214) by @noonghunna)
- feat(kv-calc): opt-in --kv-breakdown architecture cache planning layer (#213) ([#213](https://github.com/noonghunna/club-3090/pull/213) by @noonghunna)
- feat(ik-llama/two-stage): ctx default 131072→200000 + promote 🧪→⭐ code option (#212) ([#212](https://github.com/noonghunna/club-3090/pull/212) by @noonghunna)


### 🐛 Bug fixes

- fix(rebench-report): surface ceiling VRAM margin in verify-stress section (#184) ([1055c91](https://github.com/noonghunna/club-3090/commit/1055c9162fe49f96025c21f4508dca8c009e4775))
- fix(switch): shell env wins over .env (MODEL_DIR etc.) + CRLF-tolerant (#425) ([9a27de8](https://github.com/noonghunna/club-3090/commit/9a27de83d7e6aff293d9f423bc926ff9366ed993))
- fix(ik-llama/two-stage): default ngram n_max 64→4 (tuned code-decode optimum) (#210) ([#210](https://github.com/noonghunna/club-3090/pull/210) by @noonghunna)
- fix(#168): scope report.sh kv-calc calibration to the running model ([477873a](https://github.com/noonghunna/club-3090/commit/477873af93e35c6978b42abf11b890935909ae5b))
- fix(#169): distinct default container_name per llama.cpp/ik single variant ([9e5f200](https://github.com/noonghunna/club-3090/commit/9e5f200449ceb87a64838e60e8f074d468326ed3))


### 📝 Documentation

- docs(BENCHMARKS): 4090 ik-two-stage cross-rig row + 4090 ctx-derate note (#184) ([703fa3c](https://github.com/noonghunna/club-3090/commit/703fa3cc119b590198551da7c844f906c00505cb))
- docs(README): direct docker-compose fallback when launch/switch error + default capture to --full ([27e818f](https://github.com/noonghunna/club-3090/commit/27e818f559f125d3ab438a62f4db84bf353dd45c))
- docs(BENCHMARKS): add thinking-on vs no-think code-gen baseline (Qwen3.6-27B) ([d8adf55](https://github.com/noonghunna/club-3090/commit/d8adf55b4b582ff70dffbaf536f7cad8bf0e3d8a))
- docs(README): Windows/WSL2 signpost at top of Quick start ([f954692](https://github.com/noonghunna/club-3090/commit/f954692676107aef38e7794ea3150079db9c8817))
- docs(WSL): add Diagnostics section — suggest pciutils, set lspci/WSL2 expectation ([d36eb63](https://github.com/noonghunna/club-3090/commit/d36eb633c6d7e5ace99faa40dc9725ae0ccf6ad6))
- docs(WSL,FAQ): clarify club-3090 needs WSL2 — native Windows = upstream engine only ([6fa639b](https://github.com/noonghunna/club-3090/commit/6fa639bb5ad85231de55446775139a9851e5c845))
- docs(README,WSL): fold reasoning suite into Benchmarks; add native llama.cpp + overhead-reduction to WSL guide ([80527f9](https://github.com/noonghunna/club-3090/commit/80527f99708a1c0af972b8e8b3507c341faa98c8))
- Document reasoning quality suite ([605f1df](https://github.com/noonghunna/club-3090/commit/605f1df52e4f7bb2a85cd676ebe56827c163ef9d))
- docs: add WSL2/Windows from-scratch setup guide (#187) ([3c1a6e9](https://github.com/noonghunna/club-3090/commit/3c1a6e962a1206f920e28009c4055fa11f9dcc08))
- docs(README): add Benchmarks + Diagnostics sections ([37574fe](https://github.com/noonghunna/club-3090/commit/37574fe4c00a39059631069876e4131842588601))
- docs(diagnostics): redact internal paths in structured-cot-bench.md ([3b270b9](https://github.com/noonghunna/club-3090/commit/3b270b99a35636070cba0f15b25ae1cdec4715c1))
- docs: promote iq4ks-two-stage 🧪→⭐ (code, 200K) + BENCHMARKS row ([7e6eaf8](https://github.com/noonghunna/club-3090/commit/7e6eaf8f005a84ef968ead7a0138267298834706))
- docs(SINGLE_CARD): add measured two-stage TPS (~59/~98, code +35% vs MTP-only) ([b10096c](https://github.com/noonghunna/club-3090/commit/b10096cca016fe6e4782a7d7f83055801dcedf10))
- docs(SINGLE_CARD): mark the #167-blocked vLLM configs in "Pick a config" ([cb19d68](https://github.com/noonghunna/club-3090/commit/cb19d6829c49af606970676b2840fadefdcc36ed))
- docs(README): drop SGLang from the headline engine claims (blocked, not a route) ([678fd00](https://github.com/noonghunna/club-3090/commit/678fd006c5036ab930f3aaf3a32d0c996eb28170))
- docs(README): single-card "recommended" → llamacpp/default (was #167-blocked vllm/default) ([44c08b6](https://github.com/noonghunna/club-3090/commit/44c08b614668771a3cc637c7222197b90ab4b821))
- docs(DUAL_CARD): distinguish decode-concurrent vs long-prefill-overlap (#208) ([36767f1](https://github.com/noonghunna/club-3090/commit/36767f1dff66d1e7dac5e180a8b229144a935f32))
- docs(README): fix stale 'llama.cpp single = full 262K' → 200K in supported-models cell ([2a8357f](https://github.com/noonghunna/club-3090/commit/2a8357f2815423f3ae6ab0919c9c59211f870413))
- docs+registry: surface ik-llama on the single-card front door + fix stale max_ctx ([7f73361](https://github.com/noonghunna/club-3090/commit/7f733618e60899cb6878f16a5ba32b96daabd91c))
- docs(BENCHMARKS): refresh llamacpp/mtp row to the 200K thinking-off rebench ([2eca18d](https://github.com/noonghunna/club-3090/commit/2eca18d23399759cf38d08a28c49cbb4ecffa0cd))
- docs(#169 branch): fix stale 262K→200K cross-refs in single compose headers ([62a0f4b](https://github.com/noonghunna/club-3090/commit/62a0f4be63144dfe5d652861a650b7917cdc2f6f))
- docs: correct single-card llama.cpp/ik_llama ctx 262K -> 200K (shipped default) ([c7fc9ca](https://github.com/noonghunna/club-3090/commit/c7fc9ca638ab6b80a110fffff5637bfc0842aa4c))


### 🧹 Maintenance

- chore: retire the club-3090 pre-built vLLM image (workflow + docs) ([7003e61](https://github.com/noonghunna/club-3090/commit/7003e6141b8399fb2fe14019241f6f5d171e84a9))


### 🧹 Other

- Add profile-backed model weight fetch registry ([5f37ae6](https://github.com/noonghunna/club-3090/commit/5f37ae6d4d299e0ece942c6475d1b8ae7b275a4d))
- Generalize compose model preflight ([99d264b](https://github.com/noonghunna/club-3090/commit/99d264b2c460247091b8f10c4853e988f2a9ff90))
- Expose benchlocal reasoning suite ([caf6fc2](https://github.com/noonghunna/club-3090/commit/caf6fc2b723c46c70a6dcd38d2778f1d5e1e469c))



[Pin: `git checkout v0.8.5`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.4...v0.8.5)
## v0.8.4 — 2026-05-23


### ✨ Features

- feat(verify-stress): capture prefill throughput during NIAH rungs (#199) ([07d478c](https://github.com/noonghunna/club-3090/commit/07d478c1abfb9873cb106708ee05aa94473866f0))
- feat(eval): expose request-level thinking toggles (#196) ([#196](https://github.com/noonghunna/club-3090/pull/196) by @noonghunna)
- feat(scripts): pass --sampling-from-server through quality-test.sh + rebench-full.sh ([dd1f070](https://github.com/noonghunna/club-3090/commit/dd1f07062642e7a583b19c4e977885ea8404f404))
- feat(compose): expose sampling defaults via env (#194) ([#194](https://github.com/noonghunna/club-3090/pull/194) by @noonghunna)
- feat(setup): WEIGHTS=gguf to fetch the llama.cpp GGUF (not just the vLLM model) (#191) ([#191](https://github.com/noonghunna/club-3090/pull/191) by @noonghunna)
- feat(ik-llama): wire iq4ks-mtp + iq4ks-mtp-vision into launch.sh + switch.sh (#189) ([#189](https://github.com/noonghunna/club-3090/pull/189) by @noonghunna)
- feat(models): add ik_llama Qwen3.6-27B IQ4_KS composes — text 262K + vision 160K (#180) ([#180](https://github.com/noonghunna/club-3090/pull/180) by @noonghunna)


### 🐛 Bug fixes

- fix(rebench): basename model id for the aider/litellm step (ik_llama full-path id → 0/30) ([3b20ce3](https://github.com/noonghunna/club-3090/commit/3b20ce3c7cb53f49512f2af098fe175db5683d8b))
- fix(soak,preflight): recognize llama-cpp / ik-llama containers in autodetect (#403) ([d9fdab2](https://github.com/noonghunna/club-3090/commit/d9fdab296103e240527210fb5af131361fe07223))
- fix(compose): ik iq4ks-mtp header — record measured ceiling-ladder result (200K confirmed) ([1d93343](https://github.com/noonghunna/club-3090/commit/1d9334335f278dcef89f0b817815f243e9d42b57))
- fix(compose): lower single-card MTP CTX_SIZE default 262144 → 200000 (llama.cpp + ik_llama) ([2e45928](https://github.com/noonghunna/club-3090/commit/2e45928e4c455115f509555e54c6c0e6223499e4))
- fix(verify-stress): three live-caught bugs in ceiling ladder (#199) ([b84249c](https://github.com/noonghunna/club-3090/commit/b84249c805d7ce9f311e215f227e32049917ce87))
- fix(verify-stress): add CTX_SIZE-scaled ceiling ladder (#199) ([5a825a4](https://github.com/noonghunna/club-3090/commit/5a825a428a43bcd54fa1b6a3e9a91a2767c5be63))
- fix(report): PyYAML/idle-VRAM/P2P/redaction/kv-calc polish + review fixes (#178/#137) ([#192](https://github.com/noonghunna/club-3090/pull/192) by @noonghunna)
- fix(launch): point users at MODEL_DIR/.env when weights aren't found (#190) ([#190](https://github.com/noonghunna/club-3090/pull/190) by @noonghunna)
- fix(llamacpp): pin image to server-cuda-b9246 (rolling tag broke at b9282) (#188) ([#188](https://github.com/noonghunna/club-3090/pull/188) by @noonghunna)
- fix(launch): single-card default suggestion → llamacpp/default (#185) ([#185](https://github.com/noonghunna/club-3090/pull/185) by @noonghunna)
- fix(rebench): always capture sandboxed-pack logs to the per-tag results dir (#179) ([#179](https://github.com/noonghunna/club-3090/pull/179) by @noonghunna)


### 📝 Documentation

- docs: correct ik_llama verdict — ~18-20% FASTER than mainline, not a "tie" (#184) ([b7353da](https://github.com/noonghunna/club-3090/commit/b7353daa52f851e71ea2974b673ed1329372c234))
- docs: add @mgabor3141 X399/TR-1950X dual.yml row + pre-Zen2 CPU-IPC note (#178) ([6e49960](https://github.com/noonghunna/club-3090/commit/6e499609b4566830b603ec024cd311db0bbadcae))
- docs(CLIFFS): document llama.cpp "boots ≠ fills" false ceiling; 200K = max-safe single-card CTX_SIZE ([9be237d](https://github.com/noonghunna/club-3090/commit/9be237da7407478a61a5f6a7212ad686375f86d1))
- docs: QUALITY_TEST.md — fix stale pack-status (sandboxed packs now implemented) ([f6bdc06](https://github.com/noonghunna/club-3090/commit/f6bdc067dc8dc4992f2c67e23d28a8a9ab50a4eb))
- docs: document sampling/temperature eval options (#193/#194 + benchlocal #19/#21) ([9fd634a](https://github.com/noonghunna/club-3090/commit/9fd634a5b5ef13a902b201db60fd5a8bb666cef9))
- docs(single-card): strike Genesis-pinned vLLM rows (blocked by purged pin #167) ([a30bdfd](https://github.com/noonghunna/club-3090/commit/a30bdfd812a8fcd912b3764d49ba1e204351ace7))
- docs(upstream): correct the #40875 row (open tool-call-corruption bug, not "closed coexistence") ([25f130a](https://github.com/noonghunna/club-3090/commit/25f130a2868a1a79329baca8bbe3fca51035e313))
- docs: correct ik_llama claims to the matched-power tie (#184) ([c470d9a](https://github.com/noonghunna/club-3090/commit/c470d9a32dbcbb66b54a214a44abddaf11f8a80f))
- docs: surface WEIGHTS=gguf + switch.sh ik-llama paths (match #189/#191) ([412315d](https://github.com/noonghunna/club-3090/commit/412315d9ccf9fd251dfff3e3d3a88a0347d67315))
- docs(HARDWARE/FAQ): AMD-Vi IOMMU Xid 154 under TP=2 → iommu=pt fix (#178) ([fe86b72](https://github.com/noonghunna/club-3090/commit/fe86b72a73ec94e6a610711f33778465f8c004d1))
- docs: add ik_llama engine page + QUANTIZATION primer; surface IQK quants ([554b85b](https://github.com/noonghunna/club-3090/commit/554b85b8ea3cc7bdb3cb69a0a37f551870e74037))
- docs(BENCHMARKS): @duart dual NVLink Proxmox VFIO-passthrough, stock-upstream no-Genesis (disc #162) ([bc6e20b](https://github.com/noonghunna/club-3090/commit/bc6e20bb8e110372381f5dd930b1e111141e15f2))
- docs(BENCHMARKS): @mgabor3141 dual.yml — Z77/i7-3770K, PCIe 2.0 x4 slowest cross-card link (#178) ([626fa68](https://github.com/noonghunna/club-3090/commit/626fa68a7fa4be0a70f0e53cfe514e86c8e2d7ed))
- docs(mtp-vision): surface the -ub 512 → 192K context recipe in the compose header ([70bf7e7](https://github.com/noonghunna/club-3090/commit/70bf7e74be185a0f20a5d4cacd4c356aa7c4cf08))
- docs: cross-link the -ub vs ctx trade-off into SINGLE_CARD + CLIFFS + FAQ ([035261b](https://github.com/noonghunna/club-3090/commit/035261bc5831090dd2da4190e6262d4679336afd))
- charts: compose names on x-axis + description legend block below ([07c7cd0](https://github.com/noonghunna/club-3090/commit/07c7cd0222dc5ce646703137515a4aad2d0c8908))
- charts: tighten single-card label format (line 1 = variant + ctx, line 2 = modifier) ([9aa8fa7](https://github.com/noonghunna/club-3090/commit/9aa8fa71ab99306ba730d29686e5963a1ef1bebc))


### 🛠️ Scripts + tooling

- scripts: endpoint-first --url/--model/--engine for non-Docker engines (#174) ([#174](https://github.com/noonghunna/club-3090/pull/174) by @noonghunna)
- report.sh: capture image digest + OCI labels (build tag, upstream commit) ([78556f8](https://github.com/noonghunna/club-3090/commit/78556f88158be9e5baeda6fee03e15d5787077b0))


### 🧹 Maintenance

- chore(compose): drop accidentally-committed qwopus3.6-27b-v2 llama.cpp compose ([b8aeb93](https://github.com/noonghunna/club-3090/commit/b8aeb93333e904f2cef01069a0402b34aa8c731e))
- refactor(llamacpp): collapse single-card composes 3→2 (default = mtp alias) (#181) ([#181](https://github.com/noonghunna/club-3090/pull/181) by @noonghunna)


### 🧹 Other

- Fix verify-full to accept reasoning_content ([3a04ae5](https://github.com/noonghunna/club-3090/commit/3a04ae50843c9aa04c7e1f13e37c67de85586554))
- quality-test: respect explicit MODEL/--model, don't clobber from /v1/models (#177) ([#177](https://github.com/noonghunna/club-3090/pull/177) by @noonghunna)
- sglang: park EAGLE-3 path for Qwen3-Next (MTP wins everywhere) (#176) ([#176](https://github.com/noonghunna/club-3090/pull/176) by @noonghunna)
- quality-test: expose --timeout-per-case + bump aider-polyglot-30 to 3600s (#175) ([#175](https://github.com/noonghunna/club-3090/pull/175) by @noonghunna)
- sglang: experimental EAGLE-3 + Qwen3-Next dual-3090 path (Codex-led patch) ([941fa06](https://github.com/noonghunna/club-3090/commit/941fa0644ec3e98ee0fa0a190f87fdec3d923784))
- SINGLE_CARD: refresh Luce DFlash + PFlash watch-list (2026-05-20) ([f9f9640](https://github.com/noonghunna/club-3090/commit/f9f9640a5148df8c451cb99ac3f409a898830bda))
- AGENTS: pin engine images only when we vendor patches ([6810768](https://github.com/noonghunna/club-3090/commit/68107680e092cf04ee4bc3ab086687eed8b433fb))
- llama-cpp: document speed-vs-context trade-off + fix stale ub default ([1b2a76c](https://github.com/noonghunna/club-3090/commit/1b2a76ca897ae8df351989a585d54e9c3940e55e))
- llama-cpp: switch to rolling :server-cuda tag (no patches → no pin needed) ([4a53eda](https://github.com/noonghunna/club-3090/commit/4a53edab43cd649eaa1ac4c80a13bbb5a7b17437))
- llama-cpp: replace orphan llama-cpp:local with upstream pinned image (#170) ([c3e7c7e](https://github.com/noonghunna/club-3090/commit/c3e7c7ed8036810e0c172f8b1cd7332ac33ac924))
- gpu-mode status: probe :8020 + detect engine on :8030 ([db9c5e1](https://github.com/noonghunna/club-3090/commit/db9c5e1bb9bfd01bd37e4c23038c62805db40b26))



[Pin: `git checkout v0.8.4`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.3...v0.8.4)
## v0.8.3 — 2026-05-20


### 📝 Documentation

- docs: add LOCAL_AI_PRIMER.md — plain-English on-ramp for newcomers ([b07b2f9](https://github.com/noonghunna/club-3090/commit/b07b2f99e78809641427900044fa73cd41574c0f))


### 🧹 Other

- BENCHMARKS: add llamacpp/mtp + llamacpp/mtp-vision rows ([37c739b](https://github.com/noonghunna/club-3090/commit/37c739b4f9e37e3c3a5eb0fe51b5dd8e6b011abc))
- llama.cpp single: thinking-off policy alignment + MTP profile family ([ed15071](https://github.com/noonghunna/club-3090/commit/ed1507122cfd20627315de99d8258aa494f8cb4a))



[Pin: `git checkout v0.8.3`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.2...v0.8.3)
## v0.8.2 — 2026-05-19


### ✨ Features

- feat(pull): v0.8.2 STEP V5 — recommend UX + report-a-failed-pull doc + §9-reconciliation ([c5b5e9b](https://github.com/noonghunna/club-3090/commit/c5b5e9b27e24c04a880a0504b7497e266c2c8044))
- feat(nvlink): auto-detect NVLink on N-GPU topologies; add detection to multi4 + gemma-4-26b dual ([8f8ec1c](https://github.com/noonghunna/club-3090/commit/8f8ec1cebe375cf687550ce1724d1545386f36e4))
- feat(pull): v0.8.2 STEP V4 — optional whichllm hw-detect subprocess (CONTRACT-3, hw-detect-only) ([3917728](https://github.com/noonghunna/club-3090/commit/39177282b793549e6e4f2ac960358f0204c9cc5e))
- feat(switch): v0.8.2 STEP V3 — switch.sh ↔ compose_registry parity (CONTRACT-2b-ii) ([e6503bc](https://github.com/noonghunna/club-3090/commit/e6503bc046e569a7d6adac46dc768418bd14f0a8))
- feat(pull): v0.8.2 STEP V3 — arch-registry expansion + chat-template attribution/drift_guard ([999c93f](https://github.com/noonghunna/club-3090/commit/999c93fe8c3629b484d5901de7e58dda84ad39b4))
- feat(pull): v0.8.2 STEP V2 — surface pointer + --submit-last/--submit (gh + gh-less, consented, F5 reuse) ([e1cdcb5](https://github.com/noonghunna/club-3090/commit/e1cdcb53c7c9ff32e5eb101df15944d0f99747d2))
- feat(pull): v0.8.2 STEP V1 — capture-on-hard-block pt1-gate emitter + BaseCaptureBundle protocol lift ([20f1557](https://github.com/noonghunna/club-3090/commit/20f1557d2992fa2d0362e0e95ee9b9264199cc38))
- feat(report): lspci PCIe/P2P diagnostics subsection (LnkSta/ACS/topology) (#148) ([af2e45a](https://github.com/noonghunna/club-3090/commit/af2e45ae96376a781b15f9cf8c6896e1b3ecf5ae))


### 🐛 Bug fixes

- fix(pull): v0.8.2 STEP V5 — recommend must not label a fits-clean model "DOES NOT FIT" ([26949d7](https://github.com/noonghunna/club-3090/commit/26949d7fa845979f2234a961abb53b3fedef3062))
- fix(pull): v0.8.2 STEP V3 — deliver CONTRACT-2's engine-supported broadening (TRC two-class) ([d78b9a9](https://github.com/noonghunna/club-3090/commit/d78b9a94963c581061c01b413827630cd65aafdd))
- fix(pull): v0.8.2 STEP V2 — gh-less issue body must not carry the absolute capture path ([52451ca](https://github.com/noonghunna/club-3090/commit/52451ca0b0b190eb6c194917f9aa7a0dba8155e0))
- fix(launch): force LC_NUMERIC=C so the VRAM-budget printf survives comma-decimal locales (#159) ([186dc93](https://github.com/noonghunna/club-3090/commit/186dc93fae60d7047a790033176f1c3c2e0bd54a))
- fix(deriver): correct stale "GGUF not supported until v0.8.1" message — now misleading post-v0.8.1-ship ([344ab87](https://github.com/noonghunna/club-3090/commit/344ab87dd3723cf0fc30834141c2ccb17f25f507))


### 📝 Documentation

- docs(architecture): bring current-state docs up to v0.8.2 (recommend / submit on-ramp / arch-registry / hwdetect) ([c5c8f46](https://github.com/noonghunna/club-3090/commit/c5c8f469b09ef17252cf2a4fd0a587ae9e9a2adf))
- docs(generator): state plainly that generated-compose capacity is the reference profile's, NOT fit-adapted ([247b1dc](https://github.com/noonghunna/club-3090/commit/247b1dcfe859f0da439d1bc41d21cd3e9efb8970))
- docs(pull): v0.8.2 STEP V6 — correct §9/headline to the true bundled release scope ([b791271](https://github.com/noonghunna/club-3090/commit/b79127176f34f67258e0dc50b1992032fa1c653e))
- docs: fix duplicate MULTI_CARD.md entry in docs index ([966a8d1](https://github.com/noonghunna/club-3090/commit/966a8d142f0c66f9a5ccde3d9adca39b575e5482))
- docs: reorder docsindex (GSD first), add FAQ TOC + promote troubleshooting ladder, add tool-calling example ([a891b39](https://github.com/noonghunna/club-3090/commit/a891b3921f8fd598237d4649b68dd552419583e8))
- docs: add GETTING_STARTED.md, Gemma 4 model READMEs, restructure main README with quick start first ([6368bae](https://github.com/noonghunna/club-3090/commit/6368bae648684c6e2c9645dc96bcf5aa7f5d1b05))
- docs: fix stale NVLINK_MODE comment, INTERNALS.md cliff status, and dead companion repo link ([28bd0e8](https://github.com/noonghunna/club-3090/commit/28bd0e89703b1c5052990dd071d4debb758867ce))
- docs(container-runtimes): Proxmox passthrough — NVLink is the fragile path, not Proxmox (#161) ([3f066a0](https://github.com/noonghunna/club-3090/commit/3f066a044dceb918709fa32a931511980b2cb0fc))
- docs(benchmarks): add @hlo-world dual-3090 PCIe x4 dual-dflash-noviz row (#158) ([135f2c4](https://github.com/noonghunna/club-3090/commit/135f2c48fd25b782a7f5c74b4ca828cb4682f812))
- docs(upstream): froggeric v19 re-eval PASSED — ADOPTED (#150) ([ec1fd65](https://github.com/noonghunna/club-3090/commit/ec1fd652e8b02aa1f752d581f6e8c1fd5fdef0f3))


### 🧹 Maintenance

- chore(chat-template): re-vendor latest froggeric Qwen3.6 template for re-eval (#150) ([8a9ea6c](https://github.com/noonghunna/club-3090/commit/8a9ea6ca45489ad8520d65a097203d4da1d78989))



[Pin: `git checkout v0.8.2`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.1...v0.8.2)
## v0.8.1 — 2026-05-17


### 🐛 Bug fixes

- fix(patch-attribution): register vendored gemma-4-31b pr41800 overlay (follow-up to #153/#154) ([b4b20ff](https://github.com/noonghunna/club-3090/commit/b4b20ff7b64394a4dbf9a4303c8bec9da95fc54f))
- fix(gemma-4-31b): vendor missing vllm-pr41800 overlay into the model tree (closes #153) ([9c79192](https://github.com/noonghunna/club-3090/commit/9c7919253d01615b9e1d54b31eacc08540bbb6fd))
- fix(pull): argparse usage errors exit 64, not 2 — distinguishable from honest hard-stop (#370) ([820eb38](https://github.com/noonghunna/club-3090/commit/820eb3845c2ed442e88be7312efc524e277094e5))


### 📝 Documentation

- docs(examples): correct "thinking on by default" — shipped composes set enable_thinking=false (#372) ([46bb271](https://github.com/noonghunna/club-3090/commit/46bb271737db263ad225b1fb10c3dd36b5aa389f))
- docs(hardware): newer-driver 3090 caps long-text.yml at MAX_MODEL_LEN=105000 (#149) ([b0774f9](https://github.com/noonghunna/club-3090/commit/b0774f953b24da40c9a4f8b8a79aa1c0f0fc735f))
- docs: fix v0.8.0 docs-fidelity gaps (trc-ack first-run heads-up, exit-code honesty, GGUF message claim) ([78a7dee](https://github.com/noonghunna/club-3090/commit/78a7dee2478e1d86805bf90371f6e586eb482e7e))
- docs: cross-link the v0.8.0 universal pull flow from the existing user guides ([afe56f7](https://github.com/noonghunna/club-3090/commit/afe56f763fccec340b693ed765879eaffcbcfd6f))



[Pin: `git checkout v0.8.1`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.8.0...v0.8.1)
## v0.8.0 — 2026-05-17


### ⚠️ Cliffs, gotchas, regressions

- v0.8.0 Pull-Gate P4-fix: price Tier-1 curated via curated-exact kv-calc spec, not generic-dense (+ non-mocked regression test) ([087a8ea](https://github.com/noonghunna/club-3090/commit/087a8ea6929144ca64b58d499161087111a290ab))


### 🐛 Bug fixes

- fix(verify-full): warm engine before scored checks (closes #352) ([c595496](https://github.com/noonghunna/club-3090/commit/c5954964c5ffa68a8997866296ec83aa349d9219))


### 📝 Documentation

- docs(tq3-mtp): add missing 04-gemma-vs-qwen.png chart ([da9ef5e](https://github.com/noonghunna/club-3090/commit/da9ef5eb108ed01aa436444e4708ba329e8e7662))
- docs: UPSTREAM Gemma4 TurboQuant row — exact config.py:101 mechanism + fix-PR set ([fd8695f](https://github.com/noonghunna/club-3090/commit/fd8695f1ab64082be8818bd72a72ea2d3f710cca))
- docs+hygiene: track Gemma4 native-TurboQuant upstream blocker; gitignore new MoE cache dirs ([f812715](https://github.com/noonghunna/club-3090/commit/f812715d3e8afce1f79969fb76e5ed600a337122))
- docs(KERNEL_MATRIX): add Kernel Selection Philosophy section ([287c766](https://github.com/noonghunna/club-3090/commit/287c76617d5a3432f04425e876889ed6e43b7891))


### 🧹 Other

- Merge PR #147: v0.8.0 — Universal pull (evaluate & serve any safetensors HF model) ([#147](https://github.com/noonghunna/club-3090/pull/147) by @noonghunna)
- v0.8.0 [docs] PULL.md Quickstart (command-first, top-of-doc) + ARCHITECTURE one-liner: stage names are internal, users run one command ([bef766d](https://github.com/noonghunna/club-3090/commit/bef766d4ae2c5a03958c5da3d79bfa6a92ab01dd))
- v0.8.0 [review] pre-tag fixes: scrub internal-path leaks from shipped source + make .pull-captures-corpus tests CI-safe (skip-when-absent) ([49d9bb4](https://github.com/noonghunna/club-3090/commit/49d9bb4313a97d3cacd6cafdaebc5c2903fc4bfb))
- v0.8.0 [docs] ARCHITECTURE.md: add the universal pull→gate→emit→loop pipeline to the mental model + scripts tree (current-state, was stale for v0.8.0) ([1cdda19](https://github.com/noonghunna/club-3090/commit/1cdda195a77558a5cdef166fb7d184538b773140))
- v0.8.0 [UX] §7 two doc tracks: docs/PULL.md (user front-door) + docs/README.md (track spine) + README migration nudge ([a0b3b5c](https://github.com/noonghunna/club-3090/commit/a0b3b5c8b448a028f6037efeb610d4a53da18881))
- v0.8.0 [F] F8-fix: widen §6.1 Tier-1 OOM signature + pt3.actual regexes to real vLLM v0.21.0+ KV-cache-too-large phrasing — on-rig F8 caught classic-torch-only regexes miss the common KV-prediction failure ([f92624d](https://github.com/noonghunna/club-3090/commit/f92624d9a78c18f5b92257a553abdf07e8071dcf))
- v0.8.0 [F] F7: docs/LOOP.md contributor doc (Loop phase, grounded in shipped F1–F6) + CONTRACT-5(i) risk note ([a8b30d6](https://github.com/noonghunna/club-3090/commit/a8b30d67072fe13bad7810c7ef751fc45c369c2f))
- v0.8.0 [F] F6: CONTRACT-5 mandatory content-hash kv_calc_version (G2) + G1 topo-verify + L2 fixture sync ([1ac0481](https://github.com/noonghunna/club-3090/commit/1ac048189d02ccf709942b1c066737f5d9c29e8e))
- v0.8.0 [F] F5: §6.3 canonical-tuple-hash dedup + bounded label scheme + collision-safe submit path (CONTRACT-4) ([5de7224](https://github.com/noonghunna/club-3090/commit/5de7224a731bfefb70cda038cdd8bf3d0d48915d))
- v0.8.0 [F] F4: §6.2 inbound-trust pipeline raw→candidate→validated→Tier-1 + CONTRACT-3a derived-deferral (CONTRACT-3) ([d758f08](https://github.com/noonghunna/club-3090/commit/d758f08fce86d81d80370ad037ce12e2edc54339))
- v0.8.0 [F] F3: G6-A 3-part additive [E] touch (pt1.predicted_b_breakdown, pt3.failure_log_excerpt+actual, container-log capture) + §6.1 Tier-1 (CONTRACT-2) ([b100979](https://github.com/noonghunna/club-3090/commit/b1009793f26f38370ba2a013ed93d2b6044e0b59))
- v0.8.0 [F] F2: §6.1 Tier-2 semantic-fingerprint classifier + Appendix A seed DB (CONTRACT-2 Tier-2) ([9f80d29](https://github.com/noonghunna/club-3090/commit/9f80d29fbe63cd9302221a7d182693b5f803ca82))
- v0.8.0 [F] F1: FInput capture-bundle reader + schema-1 validation + key-normalization (CONTRACT-1) ([1491cbc](https://github.com/noonghunna/club-3090/commit/1491cbc7afaa48df924e2ea27446b7dc3b782c42))
- v0.8.0 [E] E-outcome-fix: honest 3-state manifest outcome (partial-success != failed) — §6.2 partial is a capability-scoped success ([71148d6](https://github.com/noonghunna/club-3090/commit/71148d6054b4616ada57654b7c405cb0c0d50cc6))
- v0.8.0 [E] E3/E4-fix: boot lifecycle as context manager (server stays up for smoke+capture, teardown on ctx-exit) — on-rig E5 caught teardown-in-finally-before-smoke ([f7c405a](https://github.com/noonghunna/club-3090/commit/f7c405a06d4c8c2ef349533d5d9e6ee59455a073))
- v0.8.0 [E] E3-fix: smoke probes the real served-model-name (not literal "derived") + capture failure detail — on-rig E5 caught red-smoke-on-healthy-boot ([16a1e4d](https://github.com/noonghunna/club-3090/commit/16a1e4d944992682ac5c8db9b8f7482730e3b9e6))
- v0.8.0 [E] E2-fix-2: verify *.safetensors against HF API lfs.sha256 (not Xet-redirect-fragile HEAD x-linked-etag) — on-rig E5 caught false no-etag ([3ae74bf](https://github.com/noonghunna/club-3090/commit/3ae74bfdcfc9ddc4378003c04180d934282b8482))
- v0.8.0 [E] E2-fix: download via hf CLI subprocess (not huggingface_hub lib-import) — on-rig E5 caught ModuleNotFoundError ([806a298](https://github.com/noonghunna/club-3090/commit/806a2985226cd1c6ab9f726882e5a610ba2da69f))
- v0.8.0 [E] E5(docs): docs/PULL_EMIT_DERIVED.md (+ private ledger/recon-checklist updates) ([d134d5a](https://github.com/noonghunna/club-3090/commit/d134d5a5fcc2b509e7e259430954c192d18bea44))
- v0.8.0 [E] E4: post-[C1] derived-[E] orchestration + trigger semantics + override force-capture (pt5) ([2ed18aa](https://github.com/noonghunna/club-3090/commit/2ed18aad3fd1f648d4a143d91e9e86d97b785f08))
- v0.8.0 [E] E3: derived boot (HF_HOME mount) + 4 §6 capture emitters + manifest + derived smoke floor ([f327887](https://github.com/noonghunna/club-3090/commit/f327887c3926b8c119afd1d4883f64f40e1b83a6))
- v0.8.0 [E] E2: HF download stage (download_set allowlist + x-linked-etag SHA, no-etag fail-closed, atomic staging) ([7a2ec86](https://github.com/noonghunna/club-3090/commit/7a2ec8664052044677c1724cc46ea78a7cd2988e))
- v0.8.0 [E] E1: generate_from_profile + derived-vllm template + EInput + CONTRACT-5 gate ([411c84f](https://github.com/noonghunna/club-3090/commit/411c84fd8ae692acd99824754b292f387ecf9486))
- v0.8.0 Pull-Gate P5: docs/PULL_GATE.md (two-path model, 6-stratum taxonomy, §4.1 [C1], hardware-SM) ([2582438](https://github.com/noonghunna/club-3090/commit/2582438d0202b0082464cca0bbb1dbee46d2725c))
- v0.8.0 Pull-Gate P4: stratum-5 + [C1] §4.1 total fn + stratum-6 [D] dry-run + pull orchestrator + exhaustive test-pull.sh ([adf7a3b](https://github.com/noonghunna/club-3090/commit/adf7a3bf13e424e96ffb510d318faaf016ebf975))
- v0.8.0 Pull-Gate P3: stratum-2 precondition + [C0] engine-support/runtime/hardware gate + [C2a] disk ([4a1d385](https://github.com/noonghunna/club-3090/commit/4a1d3857f25de46c60ff8c0b7f566733f32a2a6f))
- v0.8.0 Pull-Gate P2: transformers deriver + ModelProfile/confidence + variant-scoped hf_repos schema ([818b79c](https://github.com/noonghunna/club-3090/commit/818b79ccb523aca5034fefe4abddebf5c697dab2))
- v0.8.0 Pull-Gate P1: kv-calc generic-dense family + eligibility predicate + raw_verdict adapter ([1bafcfe](https://github.com/noonghunna/club-3090/commit/1bafcfe4a008c8d4e89763d872a99618e3853112))
- v0.8.0: doc generated composes are not relocatable (run with --project-directory) ([a2fc05e](https://github.com/noonghunna/club-3090/commit/a2fc05e1c873ff4a6a3bc97de7df1685b41e5006))
- v0.8.0 STEP 5: COMPOSE_GENERATOR.md + PATCH_POLICY.md (#141 contributor contract) ([9546f99](https://github.com/noonghunna/club-3090/commit/9546f99303bfcd907618d7778cf36237c33b858e))
- v0.8.0 STEP 3+4: compose generator + 5-triple golden-parity test (#141) ([6d7a043](https://github.com/noonghunna/club-3090/commit/6d7a043d907aedab88cc10c8b97bea75a2c4a81d))
- v0.8.0 STEP 2: extract patch_attribution.py (sound body-only reaches(), test imports it) ([60f3983](https://github.com/noonghunna/club-3090/commit/60f39832834051a294deb445bfc62d5ca90d0486))
- v0.8.0 Phase A-prime: enrich patch/profile data for #141 generator (compose_service_template, genesis_equipped, delivery metadata, drift_guards, drafter/model_slug/trc fold-ins) ([9f23736](https://github.com/noonghunna/club-3090/commit/9f23736f014dbbea74b56a4287836019d0315fb9))
- Add v0.8 Phase A patch attribution data ([91a9622](https://github.com/noonghunna/club-3090/commit/91a9622619ba2b1360676a95e412f30d3e975d7c))



[Pin: `git checkout v0.8.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.7.4...v0.8.0)
## v0.7.4 — 2026-05-15


### 🐛 Bug fixes

- fix(vllm-pr35936): make overlay tolerate bf610c2f upstream drift (closes #144) ([421114b](https://github.com/noonghunna/club-3090/commit/421114b9ddb47288365d3c9e27931671a9755a47))


### 📝 Documentation

- docs(BENCHMARKS): @OVDEN13 dual.yml — PCIe Gen 4 x4+x8 asymmetric (#142) ([77802a3](https://github.com/noonghunna/club-3090/commit/77802a3d8ee18bfa72a0a929ba1e22de661d22eb))


### 🧹 Maintenance

- test(launch): re-align engine pin expectations after revert ([e7bca8e](https://github.com/noonghunna/club-3090/commit/e7bca8e035d03240d791626631bdaf7c2c2182fe))


### 🧹 Other

- Reapply "fix(qwen3.6-27b): route non-TQ3 composes to vllm-nightly-clean" ([d780410](https://github.com/noonghunna/club-3090/commit/d7804107c96f04fc0008dbab84332c2438349aa4))
- Revert "test(launch): re-align engine pin expectations after revert" ([0c4260f](https://github.com/noonghunna/club-3090/commit/0c4260f53d5c84da51d758de9c5c2d1d2d99800c))
- Revert "fix(qwen3.6-27b): route non-TQ3 composes to vllm-nightly-clean" ([d32e168](https://github.com/noonghunna/club-3090/commit/d32e168a89f21af0c4c9d447ea196efd0b2e42b6))



[Pin: `git checkout v0.7.4`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.7.3...v0.7.4)
## v0.7.3 — 2026-05-15


### ✨ Features

- feat(report): surface kv-calc calibration verdict ([#143](https://github.com/noonghunna/club-3090/pull/143) by @noonghunna)
- feat(kv-calc): model v0.7.3 MoE architectures ([39e1873](https://github.com/noonghunna/club-3090/commit/39e18733aa8f14c83a1f76d5bed23156fda61568))
- feat(gemma-4-26b-a4b): AWQ + MTP n=4 — +12% narr / +49% code over no-MTP baseline ([6dc9a0d](https://github.com/noonghunna/club-3090/commit/6dc9a0dce1b4ecef787e50808a8a15439cabe209))
- feat(qwen-35b-a3b): preview-MTP compose + bench row — MTP measured SLOWER on MoE ([e1d44bd](https://github.com/noonghunna/club-3090/commit/e1d44bd732cef0757b8d3f0f872b5b0a1a4fe5cc))
- feat(vllm-pr41800): vendor truncate_prompt_tokens overlay across all pre-fix engines (closes #139) ([1d7aad1](https://github.com/noonghunna/club-3090/commit/1d7aad112c097709d954750ab0e725a785ad062a))
- feat(gemma-4-26b-a4b): AWQ path via vLLM PR #40886 overlay ([0053444](https://github.com/noonghunna/club-3090/commit/0053444e84f80ce9f3fb910435c8d627591f59ed))
- feat(estate): add parallel boot mode ([99328b4](https://github.com/noonghunna/club-3090/commit/99328b4cda7ec9efc723a001295525a6d1f86e2a))
- feat(moe): add dual-card composes for Gemma 26B-A4B + Qwen 35B-A3B preview ([2d1b1dc](https://github.com/noonghunna/club-3090/commit/2d1b1dc347c3897f8b902f7ab60c8bd9c97abbeb))
- feat(moe): wire Gemma 4 26B-A4B + Qwen 3.6 35B-A3B composes through fits() ([f7f6f44](https://github.com/noonghunna/club-3090/commit/f7f6f444b9e937396c3de70df7dd6a5b957be80b))
- feat(profiles): split engine-pin policy by Genesis dependency ([15eda8a](https://github.com/noonghunna/club-3090/commit/15eda8a823dd273ee9b238369cba365021b61a29))
- feat(profiles): add Gemma 4 26B-A4B ModelProfile + num_global_kv_heads field ([abf0e32](https://github.com/noonghunna/club-3090/commit/abf0e327f9b877f989fe16c88bda1a457c144e0a))
- feat(profiles): add Qwen 3.6 35B-A3B ModelProfile (MoE schema extensions) ([9378714](https://github.com/noonghunna/club-3090/commit/937871492ae08b58b041a1fded41ab793f0c8549))


### 🐛 Bug fixes

- fix(gpu-mode): mode_off tears down estate-managed instances ([9cd854d](https://github.com/noonghunna/club-3090/commit/9cd854dbcb9b8b215876e0f88443c269072b93b9))
- fix(qwen3.6-27b): route non-TQ3 composes to vllm-nightly-clean ([3b2d940](https://github.com/noonghunna/club-3090/commit/3b2d940d26ec2ffe6daf631e77986898c1d2849d))
- fix(gemma-4-31b): route default/bf16 composes to vllm-nightly-clean ([cf0451a](https://github.com/noonghunna/club-3090/commit/cf0451a7ad5c56bd7b63327aac7178df70cba1fb))
- fix(engines): vllm-nightly-mtp anchors to 01d4d1ad (Sander v7.72.2 PROD pin) ([87f0a0c](https://github.com/noonghunna/club-3090/commit/87f0a0c528473a225f06997bdcb9b79fefa13b04))


### 📝 Documentation

- docs(soak-test): clarify PASS verdict semantics — closes #140 ([9a039d8](https://github.com/noonghunna/club-3090/commit/9a039d8c922d3141102e8244d5468e8de46c6674))
- docs(UPSTREAM): add PR #41800 truncate_prompt_tokens row ([273c017](https://github.com/noonghunna/club-3090/commit/273c017646087fe508626be7ecea5e2106be54be))
- docs(README): add v0.7.3 MoE models to Supported Models table ([e49c939](https://github.com/noonghunna/club-3090/commit/e49c9397481b5fba226323b59c4fa37bdc2aeeab))
- docs(BENCHMARKS): Gemma 4 26B-A4B AWQ first row + AutoRound row demoted ([92b69bd](https://github.com/noonghunna/club-3090/commit/92b69bd220ce8360d2dfd5fdcf89d26ee2264791))
- docs(BENCHMARKS): add v0.7.3 MoE preview section ([bdfb939](https://github.com/noonghunna/club-3090/commit/bdfb939edd98c3872439a7a05dcde13cce7ccaa2))
- docs(HARDWARE): add note on PCIe Gen 3 + older CPU TP=2 headwind ([8cf38b0](https://github.com/noonghunna/club-3090/commit/8cf38b05906e3954405af3db09a822ac87a25ae5))
- docs(KERNEL_MATRIX): add KV Cache Impact subsection ([1a233cd](https://github.com/noonghunna/club-3090/commit/1a233cd9fd05a6ec51adfadae79813bc80cea4b0))
- docs: add KERNEL_MATRIX.md (attention backend + engine support matrix) ([97195fe](https://github.com/noonghunna/club-3090/commit/97195fe443770bc8adc883755fd3baf9933df874))
- docs(kv-math): extend k_v_tensors=N notation to sliding-KV formulas ([b1c68b4](https://github.com/noonghunna/club-3090/commit/b1c68b4fe3211af3442cd3bb7e9fd011e30e5554))
- docs(kv-math): tighten k_v_tensors notation across all 4 formulas ([54d8bf0](https://github.com/noonghunna/club-3090/commit/54d8bf0b3e2b322dbd67e6939ff9f9995c642437))
- docs(kv-math): third-pass Grok polish ([67de3ec](https://github.com/noonghunna/club-3090/commit/67de3eca3eb4b75b43fd0f83d2ed304976b3f1b5))
- docs(kv-math): second-pass Grok polish ([78f94ca](https://github.com/noonghunna/club-3090/commit/78f94cacf006064f68d945122907d30be8723eec))
- docs(kv-math): address Grok review feedback ([3114399](https://github.com/noonghunna/club-3090/commit/31143999837639b5a7d492d2f2b64e164787bcfe))
- docs(kv-math): config-verify Qwen 35B-A3B + Gemma 26B-A4B MoE sections ([6ec6a67](https://github.com/noonghunna/club-3090/commit/6ec6a6761f36cca44ea819434be7205bcb6fdbc6))


### 🧹 Maintenance

- test(launch): align engine pin expectations ([127f4f6](https://github.com/noonghunna/club-3090/commit/127f4f6d8fe104a954b7865a4d7550017a1c629b))



[Pin: `git checkout v0.7.3`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.7.2...v0.7.3)
## v0.7.2 — 2026-05-15


### ✨ Features

- feat(launch): add hardware topology advisor ([d116ba9](https://github.com/noonghunna/club-3090/commit/d116ba9ba3442482e0005bab9067d304b40e5e24))



[Pin: `git checkout v0.7.2`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.7.1...v0.7.2)
## v0.7.1 — 2026-05-15


### ✨ Features

- feat(bench): surface prompt processing throughput ([2a148d7](https://github.com/noonghunna/club-3090/commit/2a148d702b9415129d4c4ec9d3e7d30765927aa4))
- feat(llamacpp): expose batch tuning knobs ([02249ab](https://github.com/noonghunna/club-3090/commit/02249ab1939f354ac062d343efefe32677203174))


### 🐛 Bug fixes

- fix(ci): simplify vllm image workflow, drop smoke-gate (#135) ([ce2617e](https://github.com/noonghunna/club-3090/commit/ce2617e0bc0f56d42caf64e96847d966380be80c))


### 📝 Documentation

- docs(upstream): PR #42102 closed-as-slop; local overlay permanent ([57eb269](https://github.com/noonghunna/club-3090/commit/57eb269cd70935fc3069b85e46ead8f0f0af13dc))



[Pin: `git checkout v0.7.1`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.7.0...v0.7.1)
## v0.7.0 — 2026-05-14


### ✨ Features

- feat(scripts): add diagnose-profile triage ([c2adb39](https://github.com/noonghunna/club-3090/commit/c2adb3970dbdcb3bf227cfc7a1ac58a2de3930a4))
- feat(compose): use profile-sourced vllm image pins ([e6e33ab](https://github.com/noonghunna/club-3090/commit/e6e33ab37235cdfa2d47987e16e00d617564e55d))
- feat(launch): export profile vllm pins ([c306383](https://github.com/noonghunna/club-3090/commit/c3063838bfde3f7811db3edbb9541814aea8f09d))
- feat(profiles): resolve vllm nightly pins ([40f1ef7](https://github.com/noonghunna/club-3090/commit/40f1ef78f8e3ab80a5424ce196c7715df85a0d1a))
- feat(launch): add estate planner orchestration ([c9b153f](https://github.com/noonghunna/club-3090/commit/c9b153f91f3dd4a076395008ad3bfecdd226a452))
- feat(launch): validate single-model profiles ([a142b1c](https://github.com/noonghunna/club-3090/commit/a142b1ce1085c78e8752fa52718af66f7a49f237))
- feat(compat): add profile validator and estate self-test ([6581ccc](https://github.com/noonghunna/club-3090/commit/6581ccca9e8613594dba876ba2472e78d48c9eed))
- feat(compose): accept ESTATE_GPUS and ESTATE_PORT overrides ([a57596e](https://github.com/noonghunna/club-3090/commit/a57596e0764babc6fe91e96af8b6dd746da45e09))
- feat(profiles): ship v0.7.0 data layer ([69825d7](https://github.com/noonghunna/club-3090/commit/69825d7dae0fe1387a04879b31e0d225143b3684))


### 🐛 Bug fixes

- fix(tools): resolve profile image pins in audit ([98535dc](https://github.com/noonghunna/club-3090/commit/98535dc81453e54be03a1f5915bb2367ebab5c27))
- fix(tools): bump engine nightly profiles ([d1acde0](https://github.com/noonghunna/club-3090/commit/d1acde0b28eccb123338d2a9d94e4ef6ff75cc4b))
- fix(ci): keep vllm base arg in image metadata ([1abe65f](https://github.com/noonghunna/club-3090/commit/1abe65fe2810be594d6890ce0f26a3f0a554075f))
- fix(launch): persist estate source of truth ([52e4347](https://github.com/noonghunna/club-3090/commit/52e43470c6ee111dfa97be81e13e229ede715073))


### 📝 Documentation

- docs: document profile-sourced vllm pins ([86445be](https://github.com/noonghunna/club-3090/commit/86445be3e8c24ad829d45c4f1702a0bccf27e4dc))
- docs: document club vllm image pin ([2ae8303](https://github.com/noonghunna/club-3090/commit/2ae8303833497f20d4079a12226ca23c83f71337))
- docs: expand KV_MATH + add ADDING_MODELS workflow ([1f8aaa2](https://github.com/noonghunna/club-3090/commit/1f8aaa2acc72a7cff763fa81ae938d552fffffb2))
- docs(hardware): clarify 3090 stock TDP varies by board SKU ([0d59f94](https://github.com/noonghunna/club-3090/commit/0d59f949e472095e3ecb83ce133eb103d10588d9))


### 🧹 Maintenance

- chore(vllm): use club3090 image in composes ([aebc4f3](https://github.com/noonghunna/club-3090/commit/aebc4f321c3536943bfbd554d2c21ed6824742f9))
- chore(ci): build club vllm image ([e88a2a8](https://github.com/noonghunna/club-3090/commit/e88a2a8d21efd3556739fcb0fb1a26327b7d38cd))
- refactor(kv-calc): consume profile data ([9ccde62](https://github.com/noonghunna/club-3090/commit/9ccde62abe360bfb9170fc88102623dc5e87597e))


### 🧹 Other

- Revert "chore(vllm): use club3090 image in composes" ([c7c40bd](https://github.com/noonghunna/club-3090/commit/c7c40bdf1232ec2a2f8e5b1d98249df33026f46b))



[Pin: `git checkout v0.7.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.6.3...v0.7.0)
## v0.6.3 — 2026-05-14


### ✨ Features

- feat: unify dual-card composes with NVLink auto-detection ([e00626a](https://github.com/noonghunna/club-3090/commit/e00626a50e92fe44855c410b86ae3f33ace82f29))


### 🐛 Bug fixes

- fix(scripts): report.sh shows human-readable version via git describe ([0f0a9c8](https://github.com/noonghunna/club-3090/commit/0f0a9c84c513392d889dbf51d05c5edb1f4ba649))


### 📝 Documentation

- docs: cross-rig data — eddie 3090/3090Ti power-cap + alanspires 6×3090 VFIO ([e41323f](https://github.com/noonghunna/club-3090/commit/e41323ffa9c471c6f4d9b771bbb94ff784235342))


### 🧹 Other

- Merge PR #128: unify dual-card composes with NVLink auto-detection ([#128](https://github.com/noonghunna/club-3090/pull/128) by @noonghunna)



[Pin: `git checkout v0.6.3`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.6.2...v0.6.3)
## v0.6.2 — 2026-05-14


### 🐛 Bug fixes

- fix(launch): project TP greater than four ([98f0406](https://github.com/noonghunna/club-3090/commit/98f0406d0f265767b4a6712a1eba293b1e2dc889))



[Pin: `git checkout v0.6.2`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.6.1...v0.6.2)
## v0.6.1 — 2026-05-14


### ✨ Features

- feat(launch): add hardware-aware launcher ([5882bbe](https://github.com/noonghunna/club-3090/commit/5882bbef6f5ed7e8ceea450fa3a6b167a1bf4926))
- feat(tools): extend kv-calc.py to multi-model (Qwen 3.6 + Gemma 4 31B) ([0d48dac](https://github.com/noonghunna/club-3090/commit/0d48dac818ba889f74b74df1c454bb961a123c37))


### 📝 Documentation

- docs: update launch.sh references for v0.6.1 wizard flow ([e299e70](https://github.com/noonghunna/club-3090/commit/e299e70451c8d146214a6560e582d0e174dd0ebc))


### 🧹 Other

- Merge codex/v0.6.1-launch into master ([056dcb6](https://github.com/noonghunna/club-3090/commit/056dcb643914fee6169b02b89cb420b038c29b0f))



[Pin: `git checkout v0.6.1`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.6.0...v0.6.1)
## v0.6.0 — 2026-05-13


### ✨ Features

- feat(scripts): add hardware-aware setup picker ([12a33fb](https://github.com/noonghunna/club-3090/commit/12a33fbdd1b6422429521f887d9f21c2e3da793d))


### 🐛 Bug fixes

- fix(launch): exit cleanly on stdin EOF in wizard prompts ([e05f196](https://github.com/noonghunna/club-3090/commit/e05f1969bcf57547a17cd963a1f21435de80815b))



[Pin: `git checkout v0.6.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.5.4...v0.6.0)
## v0.5.4 — 2026-05-13


### 🐛 Bug fixes

- fix(scripts): make submit-bench issue-first ([22bf2e9](https://github.com/noonghunna/club-3090/commit/22bf2e9398c7907aae6b62809bde30e111e4a700))



[Pin: `git checkout v0.5.4`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.5.3...v0.5.4)
## v0.5.3 — 2026-05-13


### ✨ Features

- feat(scripts): add submit-bench flow ([ef77032](https://github.com/noonghunna/club-3090/commit/ef770322f43724f612a80393f547e5da218b5bf7))



[Pin: `git checkout v0.5.3`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.5.2...v0.5.3)
## v0.5.2 — 2026-05-13


### 🎯 New models + serving paths

- Add hardware-aware compose preflight ([2698552](https://github.com/noonghunna/club-3090/commit/26985527f75d8da2a32a8a2f985989d5dcf9e89a))



[Pin: `git checkout v0.5.2`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.5.1...v0.5.2)
## v0.5.1 — 2026-05-13


### 🐛 Bug fixes

- fix(qwen): PR #35936 overlay — sidecar pattern to resolve Genesis RO-mount conflict ([6617e1e](https://github.com/noonghunna/club-3090/commit/6617e1e090a6f52708aaf83821a92b612c4ad869))


### 📝 Documentation

- docs: clarify MODEL_DIR — second drive / HF cache / Windows-WSL ([1678ca0](https://github.com/noonghunna/club-3090/commit/1678ca0c8ba43ea09fad9073639a48337f2ff163))
- docs(upstream): correct stale vllm#40807 row + add #40798/#42215 row ([14ffe45](https://github.com/noonghunna/club-3090/commit/14ffe45667fb0aa292839be05ae3ad0d139d3a04))



[Pin: `git checkout v0.5.1`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.5.0...v0.5.1)
## v0.5.0 — 2026-05-12


### ✨ Features

- feat(qwen): ship froggeric chat-template fixes as default-on ([84498d4](https://github.com/noonghunna/club-3090/commit/84498d47aaf7a2fdb7c0203d53bb64414a64b6c1))
- feat(vllm): add PR #35936 required-tool fallback overlay ([28b16b5](https://github.com/noonghunna/club-3090/commit/28b16b5dc9d602a8e1c4b8d5496aa82cbca7f95d))
- feat(qwen-tq3): add CLUB3090_TQ_K1_SKIP_MTP layer-filter for PR #40914 K+1 dispatch ([6b2a7d5](https://github.com/noonghunna/club-3090/commit/6b2a7d553b6164bb4854e80fc0acdf8dcec18a87))


### 🎯 New models + serving paths

- compose(tq3-mtp-genesis): pin to Genesis v7.72.2 known-good vLLM nightly ([570fa71](https://github.com/noonghunna/club-3090/commit/570fa71240a12ab642693c0570851611e933f8c4))


### 📊 Benchmarks + cross-rig data

- bench(matrix): @ygafarov first heterogeneous Ampere + Blackwell eGPU dual ([1770931](https://github.com/noonghunna/club-3090/commit/1770931729a354bad319c8b58bdee143fe6ebce2))


### 📝 Documentation

- docs(dtype-matrix): more polish — RDNA naming, FP8 maturity caveats, AMD detection ([62b3b45](https://github.com/noonghunna/club-3090/commit/62b3b455a9ea5146cbf5576febd0c8b25a8c0fa1))
- docs(dtype-matrix): polish nuances + add Intel and AMD vendor sections ([3d4548c](https://github.com/noonghunna/club-3090/commit/3d4548c50422da07c16fcba2a59d6f42f268355b))
- docs(dtype-matrix): per-arch hardware accelerator matrix for compose optimization ([9c6d3cf](https://github.com/noonghunna/club-3090/commit/9c6d3cfba1de9d9079d6eafc9ff68c352cac7197))
- docs(faq): add 'INT8 PTH doesn't scale at concurrency — is that a bug?' ([df53287](https://github.com/noonghunna/club-3090/commit/df53287b1c26ec83be8a30eec24baf2bddc993eb))
- docs(tq3-mtp): writeup + charts for the Genesis-backed TQ3+MTP path ([c2b1c93](https://github.com/noonghunna/club-3090/commit/c2b1c93872f84fa9afa3bbe41360dc42be28c066))
- docs(qwen-tq3): close round-4 — #40914 not shippable, route to nomtp + Genesis ([9fba037](https://github.com/noonghunna/club-3090/commit/9fba03788e30151f9bf8c85f279260696954d094))
- docs(qwen-tq3): re-tombstone tq3-mtp.yml after round-3 MTP-skip validation ([063d3e9](https://github.com/noonghunna/club-3090/commit/063d3e943ce8da9cc69bd30c68b87426aca6202e))


### 🧹 Maintenance

- refactor(qwen): rename int8-tq3 → tq3-* family + add no-MTP + Genesis variants ([6182922](https://github.com/noonghunna/club-3090/commit/6182922225dfeee1c28084d1ff917bfd25539520))



[Pin: `git checkout v0.5.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.4.0...v0.5.0)
## v0.4.0 — 2026-05-11


### ✨ Features

- feat(rebench-report): close 9 gaps — TL;DR + rig + timings + reproducer + delta + discuss variant ([be7f9aa](https://github.com/noonghunna/club-3090/commit/be7f9aa143e9942bbc331e219542b529e4156512))
- feat(rebench): add REPORT.md synthesizer + container/boot/GPU captures ([18355f4](https://github.com/noonghunna/club-3090/commit/18355f41f68699d76dd50452b32d505119f399d0))
- feat(rebench): halve default soak to 10 sessions × 5 turns (~15-20 min) ([3406894](https://github.com/noonghunna/club-3090/commit/340689427f971dc16856b9d069ff32d4c280da28))
- feat(rebench): one-shot canonical 5-step bench orchestrator ([94a2522](https://github.com/noonghunna/club-3090/commit/94a2522417d514674db78c590c4b899bb3eb9d36))


### 🐛 Bug fixes

- fix(switch): GPU memory pre-flight + widen RUNNING_PATTERN ([4866913](https://github.com/noonghunna/club-3090/commit/4866913a10002e8a3e6d2a2df7e6fda07fb1f953))
- fix(rebench-report): parse aider upstream_per_exercise as dict (not list) ([7c4b310](https://github.com/noonghunna/club-3090/commit/7c4b310cca0c62a55a9603cdd31540763e82191a))


### 📊 Benchmarks + cross-rig data

- bench(head-to-head): matched-config rebench + Qwen INT8 PTH KV compose ([755e519](https://github.com/noonghunna/club-3090/commit/755e5199ffd029336000fbfb80dc342e42c8c6d5))


### 📝 Documentation

- docs(gemma-4-31b): document TQ3 Ampere FA2 head_dim wall + vendor #40108 overlay ([f8c7066](https://github.com/noonghunna/club-3090/commit/f8c706699ca8ae9630c7fda58e23a21183241175))
- docs(benchmarks): Qwen 3.6 27B vs Gemma 4 31B head-to-head on dual 3090 ([edda3b3](https://github.com/noonghunna/club-3090/commit/edda3b3ecac7d460b34081e307ffd843a414b83e))


### 🧹 Maintenance

- chore(composes): bump Qwen pins → 1acd67a7, drop obsolete patch_tolist_cudagraph ([16a1374](https://github.com/noonghunna/club-3090/commit/16a1374053f979363fe4634a73ddd9c90db6061c))
- chore(cliff): skip auto-regen bot commits in changelog parser ([a258e49](https://github.com/noonghunna/club-3090/commit/a258e496bf926892c3a33e6f2f5ee2efb87a9d71))



[Pin: `git checkout v0.4.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.3.3...v0.4.0)
## v0.3.3 — 2026-05-10


### 🧹 Maintenance

- chore(changelog): subject-only rendering (drop commit body verbosity) ([eeb946b](https://github.com/noonghunna/club-3090/commit/eeb946b0a7b90462a968c800ecceb2e519b0e7fd))



[Pin: `git checkout v0.3.3`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.3.2...v0.3.3)
## v0.3.2 — 2026-05-10


### ✨ Features

- feat(quality-test): auto-set BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1 for localhost URLs ([83bf73d](https://github.com/noonghunna/club-3090/commit/83bf73d3ec464f6a366c074a3d43f203ff1e3444))


### 🧹 Maintenance

- chore: trigger v0.3.2 release workflow (GitHub deduped previous tag push) ([255c743](https://github.com/noonghunna/club-3090/commit/255c743dff59149ef83a06a4e63b0e74153c61cb))
- chore(changelog): automate CHANGELOG + release notes from commits via cliff (Option A) ([64b0474](https://github.com/noonghunna/club-3090/commit/64b0474a628d5a91222446d90b4974b14ab3237f))



[Pin: `git checkout v0.3.2`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.3.1...v0.3.2)
## v0.3.1 — 2026-05-10


### 🐛 Bug fixes

- fix(soak-helper): capture `delta.reasoning` alongside `delta.reasoning_content` ([88eb67a](https://github.com/noonghunna/club-3090/commit/88eb67aa18263a5706268a06d66784987ec69069))


### 📝 Documentation

- docs(changelog): v0.3.1 entry for soak-helper delta.reasoning capture ([9db8b26](https://github.com/noonghunna/club-3090/commit/9db8b2603ca2e9638b533f76e9fbc1aa7bf936a3))



[Pin: `git checkout v0.3.1`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v0.3.0...v0.3.1)
## v0.3.0 — 2026-05-10


### ✨ Features

- feat(power-cap-sweep): --include-commit flag stamps club-3090 git SHA in report header (closes #112) ([7d91ac7](https://github.com/noonghunna/club-3090/commit/7d91ac75e05eb01c81a30b35d2aa1290d5dc4b7f))
- feat(qwen3.6-27b): thinking OFF by default across all 21 composes ([29d17ed](https://github.com/noonghunna/club-3090/commit/29d17ed82d5a62191e96284f557686c4baa1cea7))
- feat(setup): interactive MODEL_DIR prompt for fresh TTY users ([3909c2d](https://github.com/noonghunna/club-3090/commit/3909c2d6b826d40a382fe3554cfe068a81145657))


### 🐛 Bug fixes

- fix(qwen3.6-27b): use --default-chat-template-kwargs (not --chat-template-kwargs) ([534d29f](https://github.com/noonghunna/club-3090/commit/534d29f1b1da3ff5f34035e01b496db1c565a81b))
- fix: 4 stale refs missed in 2026-05-10 reorg push (caught by RobH589 #116) ([cf7f195](https://github.com/noonghunna/club-3090/commit/cf7f1959fdd002b4c354aa14ec5ae983aa971c9d))


### 📝 Documentation

- docs(benchmarks): aider-polyglot-30 — Qwen 27B 20/30 (66.7%) > Gemma 4 31B 17/30 (56.7%) ([e08988e](https://github.com/noonghunna/club-3090/commit/e08988e6140ea5afc7be46434f0bd1aa0c02096d))
- docs(recipes): use \$MODEL_DIR placeholder + sensible cross-rig default ([cc3a717](https://github.com/noonghunna/club-3090/commit/cc3a7175243abe43f21a1eb69ca0015b5a10f7f8))
- docs: use \$MODEL_DIR placeholder, not the dev rig's /mnt/models/huggingface/ ([fbf3431](https://github.com/noonghunna/club-3090/commit/fbf343129ccc48d242178a0d5b57d6def7d5651d))


### 🧹 Other

- release: SemVer adoption + v0.3.0 changelog entry ([7080f1f](https://github.com/noonghunna/club-3090/commit/7080f1f89b674a0cc5becd361623a0ab6abdd53d))



[Pin: `git checkout v0.3.0`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v2026.05.10...v0.3.0)
## v2026.05.10 — 2026-05-10


### ✨ Features

- feat(gemma-4-31b): INT8 PTH KV unblocks 262K + AWQ + DFlash compose family ([403b16f](https://github.com/noonghunna/club-3090/commit/403b16f303253430bf58b606da7490d8253b7ef7))


### 🎯 New models + serving paths

- compose: parametrize VLLM_ENFORCE_EAGER, KV_CACHE_DTYPE, P40/P82/PN54 across all variants (#110) ([#110](https://github.com/noonghunna/club-3090/pull/110) by @easel)
- composes: refresh Quality lines with --full sandboxed (8-pack) results ([9dea0eb](https://github.com/noonghunna/club-3090/commit/9dea0ebb7beb72845a6264664bf5da1bc5c65b9f))
- composes: add --full Quality lines on Qwen3.6-27B + Gemma 4 31B duals ([26ff0e5](https://github.com/noonghunna/club-3090/commit/26ff0e58643dae75904f4b1ea4705a5fa32007bc))


### 🐛 Bug fixes

- fix: BIND_HOST opt-in + localhost script fixes (#109) ([#109](https://github.com/noonghunna/club-3090/pull/109) by @easel)


### 📝 Documentation

- docs: WSL2 budget formula + Cliff 3 (DeltaNet SSM-state non-cacheable) ([6e12700](https://github.com/noonghunna/club-3090/commit/6e12700f9a91c92c3c28c35d25be798745016ee0))


### 🛠️ Scripts + tooling

- quality-test.sh: --sandboxed-only passthrough ([7020d96](https://github.com/noonghunna/club-3090/commit/7020d965bdbcc1b4d3bfeadd0c85b030488c1dfd))
- quality-test.sh: --help, --pack passthrough, align with benchlocal-cli v0.5 ([1be02d2](https://github.com/noonghunna/club-3090/commit/1be02d22719e78913470e6ee98a17a2b2d46f152))
- ci: replace Release Drafter with git-cliff for commit-based release notes ([7002e6b](https://github.com/noonghunna/club-3090/commit/7002e6b550fe357859e34bb6ff15d1e9ae493de4))


### 🧹 Other

- reorg: services/ consolidation + gpu-mode under git + ComfyUI + pin tracker + path updates ([00366a5](https://github.com/noonghunna/club-3090/commit/00366a58d761ca797581c1411b06c4f6ab98654c))
- encourage-soak: template dropdown + script ergonomics + report reminder + Notes convention ([c298b60](https://github.com/noonghunna/club-3090/commit/c298b60f762309d3829a0ce3a45b17ac3ca9224a))
- BENCHMARKS: add @ygafarov Strix-Halo + oculink-eGPU x4-PCIe single-3090 row (#113) ([a589058](https://github.com/noonghunna/club-3090/commit/a5890587610404d69819512a32290d87deeb5460))



[Pin: `git checkout v2026.05.10`] · [Full diff](https://github.com/noonghunna/club-3090/compare/v2026.05.09...v2026.05.10)
## v2026.05.09 — 2026-05-09


### ⚠️ Cliffs, gotchas, regressions

- Merge v7.69-cliff2-test: ship Cliff 2 closure recipes (Balanced MTP + Max-context) ([15b84df](https://github.com/noonghunna/club-3090/commit/15b84df717d1a7b193946a1cb8de0945d7f2693d))
- v7.69 + #35975 + Codex P103 gate fix — Cliff 2 closure recipes ([f6613c8](https://github.com/noonghunna/club-3090/commit/f6613c869abd6260825cf3fde17956a867316d43))
- docs + charts: v7.66 + Cliff 1 mech B closed across all 4 TQ3 composes ([ae4846f](https://github.com/noonghunna/club-3090/commit/ae4846fd6345ee414b933a6aa272ee1fdf8c3adc))
- PN30 dst-shaped temp fix: close DS conv state regression class on long-text ([9af1a52](https://github.com/noonghunna/club-3090/commit/9af1a5245adf6ac740b9e2baa7ac515379e158f4))
- PN25 v3: close Cliff 1 mech B (club-3090#16) on long-text via setup-time Genesis backport ([a62ad78](https://github.com/noonghunna/club-3090/commit/a62ad78a4e8a7ce62aec4d11ae967382b188df46))
- walk back: Cliff 1 mech B reproduces on real IDE-agent prompts (club-3090#16) ([b62b6b1](https://github.com/noonghunna/club-3090/commit/b62b6b1de46cab098718d6f671b12480d687ab4c))
- Ship verified Cliff 1 closure on long-text 205K + long-vision 192K ([287de1c](https://github.com/noonghunna/club-3090/commit/287de1c6e8f1500dd9144dce9a9b9fbf601d7d67))
- Cliff 1 P104 + P101 anchor fix outcomes (built on cliff1-fa-clamp branch) ([e6570a7](https://github.com/noonghunna/club-3090/commit/e6570a7d1139c6bf61c91569f55a4844971a7ecb))
- Cliff 1 dual-mechanism: P101+P103 cross-rig test reveals FFN buffer cliff ([573a377](https://github.com/noonghunna/club-3090/commit/573a377690a760f7fed29b80b7633b5526af2095))
- Cliff 1 root cause revised: FA2 softmax_lse sized by max_seqlen ([2d6b69d](https://github.com/noonghunna/club-3090/commit/2d6b69dd50eaa4764f12d6c30cc7d91c4355881b))


### ✨ Features

- feat(preflight): compose-dependency + HF_TOKEN + KV-format checks (#37, #47, #219) ([b6c8708](https://github.com/noonghunna/club-3090/commit/b6c870820978fe641771e631700e11e17bf475d2))
- feat(tools): kv-calc.py — predict per-card VRAM budget for Qwen3.6-27B (#226) ([4e89c6a](https://github.com/noonghunna/club-3090/commit/4e89c6aa40856126ab5803159ad69211a03a1a9a))
- feat(bounded-thinking): Phase 3 grammar A/B complete; DeepSeek scratchpad is the new recommended grammar ([b956c85](https://github.com/noonghunna/club-3090/commit/b956c85477f0f4e9dd00e20c0dfc68bc30ad20b4))
- feat(report.sh): --stress + --soak flags, --full now the canonical "everything" pass ([8a29b95](https://github.com/noonghunna/club-3090/commit/8a29b95da10f7ab45151fba866aa526f2a1796c2))
- feat(qwen3.6-27b/vllm): add dual4 + dual4-dflash composes (TP=4, 4×3090, #44) ([#44](https://github.com/noonghunna/club-3090/pull/44) by @Whamp)
- feat(grammar-eval): land harness for Holiday tagline grammar A/B ([7be8ecc](https://github.com/noonghunna/club-3090/commit/7be8ecc9e0977ae2d27397811a1cbb24b51b4b68))
- feat(soak-test): continuous-mode v2 fixtures + reproduces Cliff 2 at 25K accumulated context ([8d5bfd8](https://github.com/noonghunna/club-3090/commit/8d5bfd85f99a6a7c2c91cf70241c1e46c56ae957))
- feat(scripts): add soak-test.sh — runtime VRAM accretion validation (closes gap from #41) ([563a39e](https://github.com/noonghunna/club-3090/commit/563a39e0d3e7acbc15bb3c743d81cdcbef442dd4))
- feat: detect repo drift in preflight + add scripts/update.sh ([43fe2a4](https://github.com/noonghunna/club-3090/commit/43fe2a4e20ac8eb5c3f406f25ca0588a0989fb21))
- feat(launch/switch): register vllm/dual-nvlink as a known variant ([75de7c9](https://github.com/noonghunna/club-3090/commit/75de7c95dbcbc0c2c87f8130960340d204956224))
- feat(preflight): warn when Genesis tree out of sync with setup.sh's declared pin ([d552ed9](https://github.com/noonghunna/club-3090/commit/d552ed92166a07ce5fc3c289d9b42745f8479da5))
- feat(scripts/report.sh): capture per-GPU PCIe lane width + Gen + bus ID ([535be29](https://github.com/noonghunna/club-3090/commit/535be29df4520095bdb07cfe093031ca076f65b2))
- feat(scripts/report.sh): capture container-internal Python/CUDA versions ([e491e07](https://github.com/noonghunna/club-3090/commit/e491e07972275e2155421182a7f2df830170cd21))
- feat(scripts): add report.sh — paste-ready triage report ([31982f0](https://github.com/noonghunna/club-3090/commit/31982f0e6b029498b7244273217d448250aeeb20))
- push long-text/bounded-thinking back to 185K + 0.975; long-vision stays 140K + 0.95 ([df91d64](https://github.com/noonghunna/club-3090/commit/df91d641c443e1dd308cde8cc06175655dcfcd8e))
- feat(vllm): structured-CoT bounded-thinking compose (cross-rig port) ([3d151b9](https://github.com/noonghunna/club-3090/commit/3d151b9edc2621cfa45a7f38f8f5c05962fe924e))
- Push verified ceilings: long-text 218K, long-vision 198K ([f3e5b52](https://github.com/noonghunna/club-3090/commit/f3e5b5217c93d1476062caf5c94c1bfe93029dba))
- Verify 256K single-prompt prefill on dual.yml (Sandermage cross-rig) ([5270d94](https://github.com/noonghunna/club-3090/commit/5270d9400adfa07687ebd1beaa24cb812b10f1ea))


### 🎯 New models + serving paths

- composes: formalize Status enum + Caveats field (100% coverage) ([e1137d6](https://github.com/noonghunna/club-3090/commit/e1137d6889bbb6c7d7ddb584c942488f654c3b86))
- composes: rename dual4 → multi4 to align topology prefix with MULTI_CARD.md framing ([d33e6f8](https://github.com/noonghunna/club-3090/commit/d33e6f82da84e3e402bd2f1e0904c56cbcdb656a))
- composes: complete profile-schema header rollout (8 more composes) ([fca643d](https://github.com/noonghunna/club-3090/commit/fca643d6437bfc79522da57278a7001d1b268887))
- compose: extend VLLM_ENFORCE_EAGER hook to dual / dual-nvlink / dual4 + HARDWARE.md docs ([5ec40c6](https://github.com/noonghunna/club-3090/commit/5ec40c65ffd47b805026a9e65023a3bfe0a1dbd4))
- compose: VLLM_ENFORCE_EAGER env hook + WSL2 .env docs ([#99](https://github.com/noonghunna/club-3090/pull/99) by @easel)
- Add dual-nvlink-dflash-noviz compose variant (NVLink + DFlash N=5, 200K ctx, no vision) ([63ab224](https://github.com/noonghunna/club-3090/commit/63ab224c570516f158eee13cde22afcd9a4ba944))
- Add docker-compose.dual-nvlink-dflash.yml (#92) ([#92](https://github.com/noonghunna/club-3090/pull/92) by @danbedford)
- composes: PYTORCH_CUDA_ALLOC_CONF env-override knob + WSL2 boot-crash docs (#84) ([#84](https://github.com/noonghunna/club-3090/pull/84) by @easel)
- Add Gemma 4 + DFlash compose (vLLM PR #41703 Codex-rebased overlay) (#81) ([#81](https://github.com/noonghunna/club-3090/pull/81) by @noonghunna)
- composes: env-override knobs MAX_MODEL_LEN + GPU_MEMORY_UTILIZATION (#79) ([#79](https://github.com/noonghunna/club-3090/pull/79) by @noonghunna)
- add Gemma 4 31B + Google MTP drafter (first Ampere data) (#68) ([#68](https://github.com/noonghunna/club-3090/pull/68) by @noonghunna)
- Add dual NVLINK Docker Compose setup for Qwen3.6-27B ([1350450](https://github.com/noonghunna/club-3090/commit/135045002464a440381c8f77d4385a0045e754de))
- Add llama.cpp compose + perf chart + Q3_K_XL bench data ([39692c9](https://github.com/noonghunna/club-3090/commit/39692c98e5f2ea963da8274815b24fb3da70ecfd))
- Add long-vision + long-text composes (formalize R3' / R3''' bench rows) ([b641719](https://github.com/noonghunna/club-3090/commit/b641719eb815de978f01ddb6c2caed978cb45408))


### 🐛 Bug fixes

- fix: verify-full.sh broken pipe + llama-cpp DISABLE_THINKING env hook ([8f103f3](https://github.com/noonghunna/club-3090/commit/8f103f33ec8ed42c293e12b6bb39e73f736b1946))
- fix(preflight): catch missing llama.cpp GGUF before container boot (#63) ([#63](https://github.com/noonghunna/club-3090/pull/63) by @noonghunna)
- fix: remove thinking prompt from Carnice chat template + JSON tool format ([3729144](https://github.com/noonghunna/club-3090/commit/3729144107c587be2576171bad039aa453afceb2))
- fix: missing pipe in DUAL_CARD table row ([a28ba38](https://github.com/noonghunna/club-3090/commit/a28ba387dcdf3b6bb735872109e86062a5a5eb71))
- fix(soak): flag silent-empty turns (HTTP 200 + 0 tokens) as warnings ([f32d8a6](https://github.com/noonghunna/club-3090/commit/f32d8a69721059537557dcaeaaf522732798339a))
- fix: 3 issues from community feedback ([2f8ed19](https://github.com/noonghunna/club-3090/commit/2f8ed197ce8fb1ec1ce91a1d7647847bbca996f8))
- fix(soak-test, switch): calibration + boot-progress UX from first cross-rig runs ([8e9cf70](https://github.com/noonghunna/club-3090/commit/8e9cf70d9997d04336ead0eb7664cac608b64b3c))
- fix(dual-nvlink): rename to avoid collision + vendored Marlin path ([147f2e3](https://github.com/noonghunna/club-3090/commit/147f2e33ee8e8c021ab70f000928927326e9b4e8))
- fix(default compose): swap P65 (cudagraph workaround) → P67 (proper Triton kernel fix) ([620d918](https://github.com/noonghunna/club-3090/commit/620d918db3c0a346ae245ffaf78b7ecf0d789f70))
- fix(long-text-no-mtp): drop P65 + P85 — missed in a26e30b ([22e6549](https://github.com/noonghunna/club-3090/commit/22e654989b5658b09af989508584ce635826398e))
- fix(composes): drop GENESIS_ENABLE_P65 + P85 — out of sync with v7.69 dispatcher v2 ([a26e30b](https://github.com/noonghunna/club-3090/commit/a26e30b279d5dacbdaf7489c5dfb8d1203219166))
- fix(setup.sh): auto-clone vllm-src Marlin patched fork (was manual step) ([2e934ad](https://github.com/noonghunna/club-3090/commit/2e934ad18ad70d98e2e314ca3a96759935393949))
- fix(docs): replace dead luce-spec/llama-cpp-dflash links with Luce-Org/lucebox-hub ([e9c658c](https://github.com/noonghunna/club-3090/commit/e9c658cbc6ca453e9502b20a6fa6e682b2e5f752))
- fix(scripts): register vllm/long-text-no-mtp in switch.sh + launch.sh ([1f09a05](https://github.com/noonghunna/club-3090/commit/1f09a059d59138288bf630881c6870dacc91d9ed))
- fix(dflash): close the docs+setup gap that hit @lolren on club-3090#18 ([eb54cf4](https://github.com/noonghunna/club-3090/commit/eb54cf4e68da057a884b8d5fa8d26c8fd04969a0))
- fix(verify): drop tail buffer on Genesis check 2 anchor (refines 95b0905) ([f2c1433](https://github.com/noonghunna/club-3090/commit/f2c143326ea710cd1b724407dff5367930f14534))
- fix(verify+docs): close two items from troymroberts cross-rig validation (#25) ([95b0905](https://github.com/noonghunna/club-3090/commit/95b090567c646e7bbfcada5a05ec622dc9936ca2))
- fix(launch): pass per-variant URL + CONTAINER to verify-full.sh (#20) ([77ca576](https://github.com/noonghunna/club-3090/commit/77ca5767f5bf27ba0774edf1ded301c826d66277))
- fix(docs): bump curl smoke-test max_tokens 30 → 200 (#14) ([2f8bade](https://github.com/noonghunna/club-3090/commit/2f8bade82c53886f2525e0a5cb0efe748b1135e9))
- fix(vllm): fail fast when Genesis patches volume is empty (#13) ([0df8f74](https://github.com/noonghunna/club-3090/commit/0df8f743192809dbdcda942887b625b0f48699f2))
- fix: address open issues #1, #4, #7 ([ebacba1](https://github.com/noonghunna/club-3090/commit/ebacba1efd052fa3eda7ee3b05f2f8479a2fb2ff))


### 📊 Benchmarks + cross-rig data

- results: re-bench dual.yml + dual-dflash + dual-dflash-noviz on v0.20 ([0bdcb69](https://github.com/noonghunna/club-3090/commit/0bdcb69fa3e446442c8bc744dcb588e1697cba55))
- results: dual-turbo re-bench with corrected env vars (PN22 / PN26 naming fix) ([077228e](https://github.com/noonghunna/club-3090/commit/077228e81b06aebca3401fc03456f4a3eb55e227))


### 📝 Documentation

- docs: add Discord invite to README + FAQ + issue template ([c18257f](https://github.com/noonghunna/club-3090/commit/c18257f4399e0e7a5ee6bc38e80b10ff97ed2a8e))
- docs: refresh 4090 cross-rig knee with @laurimyllari's richer 38-cap sweep ([20ca297](https://github.com/noonghunna/club-3090/commit/20ca29737897df1d52c7c31ad6674a9d08aca54e))
- AGENTS.md: codify why patches/cache stay engine-level (not under a topology) ([9fbce96](https://github.com/noonghunna/club-3090/commit/9fbce961202965aeccd6851799b7abbdc806d355))
- AGENTS.md: capture compose naming + profile schema + experimental-compose conventions ([62e636c](https://github.com/noonghunna/club-3090/commit/62e636c0525156e6a98faaf318ce50c85a44de6b))
- docs+composes: align Gemma 4 compose names to Qwen's <topology>-<feature>.yml convention ([fe86b48](https://github.com/noonghunna/club-3090/commit/fe86b48c2134df8b333ff8445f40a7c6205aad0b))
- docs: surface Gemma 4 31B + add at-a-glance profile schemas to canonical composes ([4d7356a](https://github.com/noonghunna/club-3090/commit/4d7356aac69fbe46e87cf44083ee84965287ca19))
- docs: add Community projects section pointing at VykosX/club-3090-server ([cd48764](https://github.com/noonghunna/club-3090/commit/cd487648c9bc42c9807ec458b09e21d2ce238a5e))
- docs: laptop EC-managed power + TQ3 vs fp8 KV naming-trap; verify-stress: auto-bump curl timeout under VLLM_ENFORCE_EAGER ([fe23eff](https://github.com/noonghunna/club-3090/commit/fe23eff8f0dcac9647dffa198959b89e42e9fe33))
- docs + compose: ship Phase 2 INT8 PTH validation results — 262K Gemma 4 unblocked ([1e1886a](https://github.com/noonghunna/club-3090/commit/1e1886a3a95825f387bcc33badba3a52409e7778))
- docs(hardware): add Qwen3.6-35B-A3B (MoE) 3090 power-cap charts + comparison ([ec27d75](https://github.com/noonghunna/club-3090/commit/ec27d7594eadcad6a343fe4b4e3246f195b87eef))
- docs(img): reposition freq-cap chart annotations to clear right margin ([2f7eb44](https://github.com/noonghunna/club-3090/commit/2f7eb44a15b1383e90cfd8b8245193a9363cd255))
- docs(hardware): add 5090 clock-lock chart + Blackwell freq-cap section ([119a5fa](https://github.com/noonghunna/club-3090/commit/119a5fa00db659ac5283540d70cea65257c4cbaa))
- docs(hardware): regen 3090 power-cap charts with SM clock + plateau evidence ([9f77be7](https://github.com/noonghunna/club-3090/commit/9f77be791bced574965276d8271eaa28dc58f5a9))
- docs(hardware): reconcile 230W vs 290W vs 330W sweet-spot story ([a7a1d59](https://github.com/noonghunna/club-3090/commit/a7a1d591d5154fb8c0822738ff66536f85ae9fb0))
- docs(hardware): @apnar prefill-heavy 5090 sweep — proves per-workload power ceiling ([d5ef8c8](https://github.com/noonghunna/club-3090/commit/d5ef8c89d09c35b37ce10b6a85db63d374ebcf17))
- docs(hardware): correct 3090 cooling class — air, not water ([1f94478](https://github.com/noonghunna/club-3090/commit/1f94478179f9fe4d1daff644a1cf1eb5f06cc0cc))
- docs(hardware): embed 3090 + Qwen3.6 + llama.cpp power-cap chart ([42afdbb](https://github.com/noonghunna/club-3090/commit/42afdbbff8ac552f716c9d61383b9a80e9f8280b))
- docs(hardware): embed 4090 + Qwen3.6 + llama.cpp power-cap chart ([e70258c](https://github.com/noonghunna/club-3090/commit/e70258c47f6a134f5f4f503a06e074f1a7d54758))
- docs(hardware): embed 5090 + Gemma 4 power-cap efficiency chart ([8b1d51a](https://github.com/noonghunna/club-3090/commit/8b1d51ab3c5b4d976cea1f1c5c74ba8c1f3b4a42))
- docs(engines): more honest vLLM GGUF status ([2d9aa14](https://github.com/noonghunna/club-3090/commit/2d9aa1483ccdc446a6e3609e86eb4010c08474fe))
- docs(engines): fix 12 corrupted table separators from ik_llama.cpp column add ([8f5b924](https://github.com/noonghunna/club-3090/commit/8f5b92491a44e06873c9c8a252b6602589550043))
- docs(engines): add ik_llama.cpp as 5th column to comparison matrix ([0959206](https://github.com/noonghunna/club-3090/commit/09592065f5261f17d55efc2e053690552d148fbb))
- docs(hardware): add 5090 + Gemma 4 + MTP cross-rig anchor rows (apnar disc #86) ([bef5701](https://github.com/noonghunna/club-3090/commit/bef57012a725fe8c5f0975ff531508826c5eb45e))
- docs: add INFERENCE_ENGINES.md feature matrix (vLLM/llama.cpp/SGLang/ktransformers) ([dfceccb](https://github.com/noonghunna/club-3090/commit/dfceccbf5577c8fb44d1a746b2808803ca659a46))
- docs: codify canonical power-cap-sweep command for cross-rig anchors ([886b619](https://github.com/noonghunna/club-3090/commit/886b619cc45d3a5f26adeebb3ec78508ce856388))
- docs: BENCHMARKS rows + CHANGELOG entry for danbedford NVLink+DFlash variants ([b893d60](https://github.com/noonghunna/club-3090/commit/b893d60f43a02aea8b4032003004d855eb97980e))
- docs(benchmarks): @apnar 5090 Gemma 4 MTP + DFlash rows (disc #67) ([98b0601](https://github.com/noonghunna/club-3090/commit/98b0601b6888b86116312f2b9df38e9f871b7521))
- docs(benchmarks): three cross-rig rows from 2026-05-07 reports ([76aacdc](https://github.com/noonghunna/club-3090/commit/76aacdc22043e224e6a080f23dccad4146d01cf5))
- docs(benchmarks): add @aaronlockhartdev patched-P2P driver row (#91, disc #70) ([4eea837](https://github.com/noonghunna/club-3090/commit/4eea837da808bf7f6bfd2c944bc7c19fecb79726))
- docs(upstream): note we filed cross-rig validation on vLLM PR #40391 ([e46f1e8](https://github.com/noonghunna/club-3090/commit/e46f1e8e03de6b0d882a95581fd73163594796d6))
- docs(gemma-4): int8_per_token_head on Ampere — Codex investigation verdict ([1c2c156](https://github.com/noonghunna/club-3090/commit/1c2c156003550b5b1b004c329461f7d7ba2588bb))
- docs: surface host-build contributor flow + power-cap-sweep in README + CONTRIBUTING ([9aa6cb2](https://github.com/noonghunna/club-3090/commit/9aa6cb2b0e93c8b45d33b00b368532d7055086ca))
- docs(benchmarks): add @lamentofhighborne 1× 3090 llama.cpp MTP row (#85) ([68dbfaf](https://github.com/noonghunna/club-3090/commit/68dbfafe94148806a97025c1758bd6864454c1af))
- docs(hardware): add @apnar's 5090 power-cap anchor + compute-saturation note ([60d4df6](https://github.com/noonghunna/club-3090/commit/60d4df6ce7a960c7ff6a38c78c6be85a55e4608c))
- docs(upstream): correct Gemma 4 per-token-head KV row — upstream PR exists ([eb9f955](https://github.com/noonghunna/club-3090/commit/eb9f9552832931d4cc5e2bdd0a9d21f18f739e80))
- docs(gemma-4): document fp8 + int8 KV exploration on Ampere — both blocked ([bb07eb5](https://github.com/noonghunna/club-3090/commit/bb07eb59aec1c70f1748a43be51acbaccc1e2364))
- docs(gemma-4): empirical ctx ceilings + PR #41745 merge status ([1038e5f](https://github.com/noonghunna/club-3090/commit/1038e5fc77f72f28c0900cf53a947e90f008b002))
- docs(power): add cooling caveat — 388W stock requires liquid cooling ([b15c5e1](https://github.com/noonghunna/club-3090/commit/b15c5e1f33bee67ed17c9301e630f54af6d81827))
- docs(power): revise default cap 230W → 330W per @syangsao cross-rig data ([2fe017f](https://github.com/noonghunna/club-3090/commit/2fe017f88dee7036184c6ee0e048c91c59754cea))
- docs(benchmarks): correct V100 row VRAM 14.6→15.6 GB/card per @efschu ([d7bffec](https://github.com/noonghunna/club-3090/commit/d7bffeccb74e330f070cf3461019a387b6e766ff))
- docs(benchmarks): add @efschu 2× Tesla V100 16GB row (first sm_70 Volta data) ([9212c60](https://github.com/noonghunna/club-3090/commit/9212c606e07c0cb4f52ea5bcf259e987cb184630))
- docs(benchmarks): @danbedford 2× 3090 cross-rig matrix (6 benches, controlled PCIe vs NVLink) ([6e57215](https://github.com/noonghunna/club-3090/commit/6e572156883fe2d5a291922307346b526b930525))
- docs(benchmarks): add @laurimyllari 4090 single-card vllm/long-text row ([461c4d4](https://github.com/noonghunna/club-3090/commit/461c4d4d3d9cadbd88ad6bd39437958238ff7c62))
- docs(benchmarks): add @lolren 2× 3090 + Ryzen 5950X cross-rig rows (3 variants) ([34a2348](https://github.com/noonghunna/club-3090/commit/34a23486c4bc10b1a54c6eccdc6a793ddeff1fe7))
- docs(benchmarks): add @apriori dual-dflash row (EPYC 7302P + Arch + 2× 3090) ([344e595](https://github.com/noonghunna/club-3090/commit/344e595cb915f2168d48b26492624c5c555c4832))
- docs(upstream): track llama.cpp MTP PR #22673 + non-adoption rationale (#64) ([#64](https://github.com/noonghunna/club-3090/pull/64) by @noonghunna)
- docs(contributing): clarify issues-vs-discussions routing (#61) ([#61](https://github.com/noonghunna/club-3090/pull/61) by @noonghunna)
- docs: add Carnice BF16MTP to DUAL_CARD, vllm README, and CHANGELOG ([fbd3531](https://github.com/noonghunna/club-3090/commit/fbd3531960894e4f1dac1c29e55fdd99224c4f6c))
- docs(runtimes): tighten Proxmox section — native venv works (#49) ([a51202c](https://github.com/noonghunna/club-3090/commit/a51202c7b32e755afa2739d35574efa41fbcf17c))
- docs(hardware): note SM86 structural ~70% TG drop at 131K (cross-rig) ([eb5cd70](https://github.com/noonghunna/club-3090/commit/eb5cd708c3899953ecd9bb9852bd5825609b0799))
- docs: capture environmental footnotes — WSL2 TDR + Proxmox uvloop (#49, #50) ([224ca71](https://github.com/noonghunna/club-3090/commit/224ca71b1910d5ca4eb8f2379aa353c4ce0f3156))
- docs(cliffs): add rig-class caveat — "known good" is rig-specific (#49) ([53d5c6b](https://github.com/noonghunna/club-3090/commit/53d5c6b02f4dac7fab16f53aee37391ca77bc67c))
- docs(multi-card): topology-aware pair selection on awkward GPU counts (#49) ([8e60539](https://github.com/noonghunna/club-3090/commit/8e605397e05f00ee6eca2c825e49e6b2a02a7250))
- docs(benchmarks): walk back PFlash "shippable" framing — TTFT + NIAH ≠ full validation (#230, #231) ([ccac1ff](https://github.com/noonghunna/club-3090/commit/ccac1ff1751ce5b80c1d635e6e88d0a8294b0a6e))
- docs(benchmarks): PFlash long-context bench — 131K source ceiling on 1× 3090 (#230) ([ebca0c8](https://github.com/noonghunna/club-3090/commit/ebca0c8921293b156fffc338c3aee708621384ac))
- docs(benchmarks): K8V4 result + P2P-CNS finding on lucebox-hub dual-GPU (#229) ([e78eaa1](https://github.com/noonghunna/club-3090/commit/e78eaa1148af83566753d2ffb1c66fc1cbf2aba5))
- docs(benchmarks): add lucebox-hub DFlash dual-GPU bench — no-op on 24 GB cards (#229) ([cb089e1](https://github.com/noonghunna/club-3090/commit/cb089e1e36c3cc37db45da630331d0a5bfa10215))
- docs(benchmarks): add @JusefPol's 2× 3090 + NVLink dual-nvlink row (#29, #31) ([017d0d2](https://github.com/noonghunna/club-3090/commit/017d0d2c2009ddd15ec50395e1947d52562e4747))
- docs(lucebox): record PRs #78 + #80 — dual-GPU PFlash + DFlash split shipped (May 2026) ([dec0f22](https://github.com/noonghunna/club-3090/commit/dec0f22dac051c96b38a31b7245020f0736025c9))
- docs(sglang): refresh per-engine + comparison pages — DFlash + MTP native upstream as of May 2026 ([ecc2d74](https://github.com/noonghunna/club-3090/commit/ecc2d747aea7d7c0e06aaacae4e6bbbfbe1c3b40))
- docs(structured-cot): soften Phase 3 framing per Codex v2-prompt validation ([011d4cc](https://github.com/noonghunna/club-3090/commit/011d4cc37111dcb857338684faaa57a7bce2a717))
- docs(cliffs/hardware): ground Cliff 2 + TQ3 explanations in published literature ([9b370f5](https://github.com/noonghunna/club-3090/commit/9b370f5ab1ab24bfc0ff6a5a96022cdc3e093606))
- docs: cross-reference TQ3→fp8 KV swap from CLIFFS, DUAL_CARD, dual-turbo.yml + CHANGELOG record (#47) ([129a4f4](https://github.com/noonghunna/club-3090/commit/129a4f42f4e3a1eab64abcdf6a444937b566c353))
- docs(hardware): 20 GB Ampere TP=2 needs fp8_e5m2 KV, not TQ3 (#47) ([124f08c](https://github.com/noonghunna/club-3090/commit/124f08c7ceaf400965622c3038e851a1d62365b4))
- docs(benchmarks): add @snoby's 2× 4090 dual-dflash-noviz row (#46) ([fc4c061](https://github.com/noonghunna/club-3090/commit/fc4c061be1298124e2c144c7143b308eee4f57bb))
- docs: align bug-report + FAQ + MULTI_CARD with report.sh --full / --soak ([b859630](https://github.com/noonghunna/club-3090/commit/b85963033aa62465a0c75a911d7ce394be52a098))
- docs(benchmarks): add Rig column for cross-rig contributions ([d8e7f73](https://github.com/noonghunna/club-3090/commit/d8e7f73eb281ea62771b38458429d14271bd6427))
- docs: add BENCHMARKS.md + extend grammar harness for full-bench mode ([9043678](https://github.com/noonghunna/club-3090/commit/90436788fbf54258e98a206da4260d6fb94fbb40))
- docs+gates: PR template, soak-continuous gate, Phase 2 grammar A/B ([85a6ea8](https://github.com/noonghunna/club-3090/commit/85a6ea8c48d44558476f09d2bc5a1a6c9fa93d2f))
- docs: UPSTREAM tracker + SINGLE_CARD polish — close the cliff-2b research thread ([451b9f3](https://github.com/noonghunna/club-3090/commit/451b9f37ab645fb2c9b3540017518a6ea32179df))
- docs: surface Cliff 2b multi-turn envelope + WHY TP=2 / llama.cpp escape ([04764c5](https://github.com/noonghunna/club-3090/commit/04764c5f28c3c21ad0621d9693356a3712918514))
- docs(UPSTREAM): sync 3 upstream changes + add next-week revisit queue ([4327fd3](https://github.com/noonghunna/club-3090/commit/4327fd30247c87bda7bac43672635356bfa2847e))
- docs: surface scripts/update.sh + repo-drift detection ([bca5a06](https://github.com/noonghunna/club-3090/commit/bca5a063c964b0029189c46aeec6cdce6323b72a))
- docs(vllm-marlin-pad/README): add sanity-check procedure before image-bump syncs ([1bb85fa](https://github.com/noonghunna/club-3090/commit/1bb85fadb82f92b80216a6a1dfa4eff87e25b03d))
- docs: add MULTI_CARD.md for 3+ GPU users (derived, untested locally) ([75a64a6](https://github.com/noonghunna/club-3090/commit/75a64a694e0789d2ec3ed74c64eda96c789a2de8))
- docs(FAQ): add WSL2 RAM-constraint failure mode to troubleshooting ([3bf7da7](https://github.com/noonghunna/club-3090/commit/3bf7da7501e3bc0e909cf7eeb55acb425697e7e3))
- docs: surface triage ladder at issue-filing time + add at-a-glance table ([f55b0a7](https://github.com/noonghunna/club-3090/commit/f55b0a734f259f031da18e7d823ef7eb44dbbca5))
- docs(FAQ): add 5-step triage ladder before symptom-matching ([9560efd](https://github.com/noonghunna/club-3090/commit/9560efd1f7616bd91b3d753d5ffc961ad3d3641d))
- docs: add PFlash integration feasibility memo (Codex audit, 2026-05-02) ([90a83a3](https://github.com/noonghunna/club-3090/commit/90a83a3fb5e4a931380a3920b30047ab062a23a2))
- docs: route bug + bench templates through scripts/report.sh ([b9a1305](https://github.com/noonghunna/club-3090/commit/b9a13056cc04a45f828c4d6e3809ce2f5c979c60))
- docs(dual-card): substrate refs from v7.65/v7.66 → v7.69 ([95b2c3b](https://github.com/noonghunna/club-3090/commit/95b2c3b1fa36600f88815b90d66eda86e20eb22c))
- docs: full sync to v7.69 + Cliff 2 60K closure recipes ([f8c9c36](https://github.com/noonghunna/club-3090/commit/f8c9c365e06ffdfa0d938b0e7c6f6557a8e8f1f1))
- docs(UPSTREAM): track Pflash (Luce-Org prefill accelerator) — flagged by @troymroberts (#25) ([e0e1752](https://github.com/noonghunna/club-3090/commit/e0e1752b3a0299671a4c5fec610fa3e321caed95))
- docs(CLIFFS): note v7.68 cross-rig test outcome — 3 regressions, master stays on v7.66 ([ae1b92f](https://github.com/noonghunna/club-3090/commit/ae1b92fef33b3873464b45cfab30a73733064621))
- docs: Genesis #14/#15 fixes shipped on Sandermage dev (P38B/P15B/PN25 pending v7.65) ([60d7b02](https://github.com/noonghunna/club-3090/commit/60d7b02c55ae3c4b85d65977508068ab2e25ed27))
- docs(upstream): refresh tracker for v0.20 blockers, P38/FA varlen filings, v7.64 closures ([f633fdb](https://github.com/noonghunna/club-3090/commit/f633fdbe17d91ee6c56af3d7d20ea7e11e337bac))
- docs+composes: refresh long-text/long-vision/bounded-thinking headers + max_tokens guidance ([cc4f083](https://github.com/noonghunna/club-3090/commit/cc4f0835e6b0214b5775ffd3bb638a0b6e8cf0d7))
- docs + bounded-thinking: roll new context defaults across user-facing surfaces ([d803278](https://github.com/noonghunna/club-3090/commit/d803278ebc78028a58172a6a9cfc976c7bbbc0ea))
- docs(compose): document Cliff 1 mech B real-workload gap + escape hatches (#16) ([6bff99a](https://github.com/noonghunna/club-3090/commit/6bff99a3f2a2bc17caaf72c7e23b465799094d27))
- charts: add tweet-asset variant (single-card vLLM only, 2 bars) ([f754669](https://github.com/noonghunna/club-3090/commit/f754669562ea2be8f83215423a112b2f4b0af433))
- charts: combined width 18 + 2-line group labels + dual VRAM title says vLLM ([24c8a62](https://github.com/noonghunna/club-3090/commit/24c8a629fd7c632286ba99e8f5b8f166b41f9012))
- charts: fix layout overlap with Luce DFlash 7th bar ([1ce7dc4](https://github.com/noonghunna/club-3090/commit/1ce7dc4512a292c906c5f85df27e7851f879369f))
- docs+charts: add Luce DFlash bench + watch entry; cautions in single-card chart ([cf71feb](https://github.com/noonghunna/club-3090/commit/cf71feb3e809cf618930959c3f5a6198a81c4cbc))
- docs: demote 48K/tools-text/minimal to fallback; lead with long-* + llama.cpp ([48f93e5](https://github.com/noonghunna/club-3090/commit/48f93e550fc3232783b516e6d22bf484defb6f5b))
- docs: fix stale chart ref in HARDWARE.md + delete obsolete vram-budget.svg ([cc02699](https://github.com/noonghunna/club-3090/commit/cc026993da6b2bfb3dd3a2322970317524f6399d))
- docs: catch remaining stale 192K/205K refs in long-text.yml header ([f00f279](https://github.com/noonghunna/club-3090/commit/f00f279d381885e979b0e3e46428df5c40171fd8))
- docs: final cleanup pass on stale 192K/205K refs ([a1fc225](https://github.com/noonghunna/club-3090/commit/a1fc22556a26c7dbbdfe8995364361ccbb93a5ff))
- docs+scripts+charts: propagate new ceilings (long-vision 198K, long-text 218K) ([427d2f8](https://github.com/noonghunna/club-3090/commit/427d2f8aa9f47931ea2a5258b59936f32b1bb7fe))
- docs: record verified ceilings and bisection in CLIFFS + CHANGELOG ([26e5f65](https://github.com/noonghunna/club-3090/commit/26e5f65975eea982ae2babec8a2cbfb32e05ae5a))
- docs: note revised Cliff 1 diagnosis posted on Sandermage issue #11 ([8d8968b](https://github.com/noonghunna/club-3090/commit/8d8968b034e11af6ad76b2ffb952c76d7846b357))
- docs: link PN12 PR #13 + record independent validation pass ([5e38365](https://github.com/noonghunna/club-3090/commit/5e383657f988e320ab2e15e44d1a509e5f839d95))
- docs: revise Cliff 1 analysis (PN12 anchor drift was the real bug) ([13d325b](https://github.com/noonghunna/club-3090/commit/13d325b1eddf8062341c5723503516389548074a))
- Document Cliff 1 205K closure ([9f6182e](https://github.com/noonghunna/club-3090/commit/9f6182edc65691fe17e83df7cc88559db084e25a))
- changelog: link P101 PR #12 in 2026-04-30 entry ([90a03ce](https://github.com/noonghunna/club-3090/commit/90a03ce2775aa08c8e972df3c6eeaf23067ace1c))
- docs: link P101 PR #12 in UPSTREAM and CLIFFS ([d0d79b1](https://github.com/noonghunna/club-3090/commit/d0d79b1c25ab334f78f1ae4530f3f2ebbed91e76))
- CLIFFS.md: post-2026-04-30 architectural-wall conclusion ([8580dc6](https://github.com/noonghunna/club-3090/commit/8580dc612dfe4be605552671448b34777993a914))
- CLIFFS.md: refine clamp formula + implementation shape (ChatGPT review) ([da6393b](https://github.com/noonghunna/club-3090/commit/da6393bb79da167a018834dff71843cff2337853))
- Add docs/CLIFFS.md — comprehensive prefill-cliff synopsis ([b0eed46](https://github.com/noonghunna/club-3090/commit/b0eed46ff7e86f54c3ca6b19803cbd996a6f83fd))
- LLAMA_CPP.md: add structural explanation of why prefill cliffs don't fire ([17aff4c](https://github.com/noonghunna/club-3090/commit/17aff4ce05d14ed56515928d31caf7ccfce72c07))
- Add docs/UPSTREAM.md + AGENTS.md (consolidate upstream tracking) ([53d811d](https://github.com/noonghunna/club-3090/commit/53d811d82b0fc09b2f5ebf2aa557718a97d9c18b))
- Add docs/COMPARISONS.md — self-host vs cloud and other local options ([297a982](https://github.com/noonghunna/club-3090/commit/297a9821f5c2f0a0489bb86490d9a21055b1357c))
- Add docs/FAQ.md — common questions answered for tweet click-throughs ([1b9374b](https://github.com/noonghunna/club-3090/commit/1b9374b8b5fd47541d168241642cddec8c3b6f04))
- Add docs/EXAMPLES.md — client snippets + IDE / Open WebUI connection ([91b817f](https://github.com/noonghunna/club-3090/commit/91b817fa72d22d54bd9ab029711d8866b008763b))
- README: lead with two-routes framing (matches launch tweet) ([710def5](https://github.com/noonghunna/club-3090/commit/710def58e421aaa2629b19495bef1f1f3bdb76cb))


### 🔧 Pin bumps + upstream

- bump Genesis pin 753344b → fc89395 (v7.66 dev tip) ([7a7efbe](https://github.com/noonghunna/club-3090/commit/7a7efbea0dfd6694abe0bcfcbdb570d85fb9a884))
- v0.20 migration + Genesis v7.65 dev tip + cold-start cache + env-var alignment ([5aa97a2](https://github.com/noonghunna/club-3090/commit/5aa97a25d910012c1a614978665e57fee53934d0))
- Genesis v7.62.x + PN8 on FP8 paths (closes Cliff 1 on tools-text) ([51a4001](https://github.com/noonghunna/club-3090/commit/51a4001af78e7e3cc7d8a0f10fe276c8f10afee6))


### 🛠️ Scripts + tooling

- ci: add Release Drafter for CalVer release notes ([c49db50](https://github.com/noonghunna/club-3090/commit/c49db508e64bb2332822b15fe54239845de7f9a2))
- power-cap-sweep: also sum delta.reasoning (third field-path) ([1528b59](https://github.com/noonghunna/club-3090/commit/1528b591c3900fde4c5fffb6af137063053947b5))
- power-cap-sweep: sum delta.reasoning_content alongside delta.content ([71e5954](https://github.com/noonghunna/club-3090/commit/71e5954ea987ece658ecfa1f232ccf3506a41fbf))
- power-cap-sweep: clamp prefill calibration to model context window ([32f924c](https://github.com/noonghunna/club-3090/commit/32f924c1c4889f26113dcf0f3ae8677339a95a08))
- power-cap-sweep: plateau auto-detection + multi-mode chain docs ([fd11ae6](https://github.com/noonghunna/club-3090/commit/fd11ae64c5dfb29ae0941e29ce96364d85d490cd))
- power-cap-sweep: add SM/mem clock + throttle% + pstate sampling ([ab2796d](https://github.com/noonghunna/club-3090/commit/ab2796dd6eaba09086f03847fa4e8c6cd9d55852))
- power-cap-sweep: time-bounded prefill-heavy + decode-concurrent (Codex round 2) ([1ede998](https://github.com/noonghunna/club-3090/commit/1ede998ced854f970018380dbf829132b3f9fe5f))
- power-cap-sweep: time-bounded streaming bench (Codex Option A redesign) ([7877c04](https://github.com/noonghunna/club-3090/commit/7877c047b55eb80a2941e6da1490389ffbb5aff3))
- power-cap-sweep: 4 cross-card portability fixes ([652103f](https://github.com/noonghunna/club-3090/commit/652103f07405535bea9c1dd659ab3610e4f86443))
- power-cap-sweep: env-overridable bench shape for decode-single mode ([c638c30](https://github.com/noonghunna/club-3090/commit/c638c305873e2a6b109302b2cae1f99fcad5ce43))
- setup.sh: auto-create .env for WSL2 boot-crash workaround (#60) ([4861ee7](https://github.com/noonghunna/club-3090/commit/4861ee7b6bfb6e337de541e44cecc30e3343b9c4))
- power-cap-sweep: --concurrency-stretch N flag for probing headroom past plateau pick ([3991ecc](https://github.com/noonghunna/club-3090/commit/3991ecc5b945fad1eaba1165e7d9a85dc6d80701))
- power-cap-sweep: plateau-detection auto-calibration (saturate headroomy GPUs) ([29e7de5](https://github.com/noonghunna/club-3090/commit/29e7de5a90ea70807c32d33ab4ebca9814f74e55))
- report.sh: engine-aware Active container probes (vllm + llamacpp) ([6fa66d2](https://github.com/noonghunna/club-3090/commit/6fa66d2f97e23563f2c5f3e7965473f3b6672d31))
- report.sh: capture recently-exited containers' boot logs (#60) ([cd980f6](https://github.com/noonghunna/club-3090/commit/cd980f64b7e7276016eae9c4cd47557ed9211ee9))
- setup.sh: add gemma-4-31b model support (#89) ([dd3bccc](https://github.com/noonghunna/club-3090/commit/dd3bcccb05f3027ca7dda6cabca962d4b7a75c5c))
- power-cap-sweep: --concurrency auto for workload-calibrated sweeps (Codex) ([f811457](https://github.com/noonghunna/club-3090/commit/f811457fffaad61c834d6c2b2b32ecca66376fa2))
- power-cap-sweep: --bench-runs N for variance mitigation (Codex) ([f99fad3](https://github.com/noonghunna/club-3090/commit/f99fad3361a577e8360c4c7a1d36531d80216c97))
- power-cap-sweep: document decode-concurrent n=1 variance caveat ([18c74de](https://github.com/noonghunna/club-3090/commit/18c74def0a6b987eecac26883bb13b2f602fe791))
- power-cap-sweep: load-mode flag + concurrent/prefill modes (Codex iteration) ([f387622](https://github.com/noonghunna/club-3090/commit/f38762251b31333a4bde463c3c1af2796bcec28c))
- verify-stress: engine-aware diagnostic hints (closes #87) ([4f01abb](https://github.com/noonghunna/club-3090/commit/4f01abb2d5fb318ee8862db51b19045257f4a69a))
- power-cap-sweep: make CONTAINER optional for host engine builds (#85, #87) ([2bb3cf7](https://github.com/noonghunna/club-3090/commit/2bb3cf72170b57de56358ca4ce4893346ebbabf0))
- scripts(verify-full, soak-test): decouple from docker/vLLM assumptions (#85, #87) ([a8606e3](https://github.com/noonghunna/club-3090/commit/a8606e34398d1c207fcfb6e6c989c2f94e766296))
- power-cap-sweep: fix stale summary footer + add compute-saturation note ([8c26c4b](https://github.com/noonghunna/club-3090/commit/8c26c4b56af0414ff8c0a499076250c6c38cefe3))
- power-cap-sweep: reduce per-cap bench to ~30s for faster sweeps ([a413321](https://github.com/noonghunna/club-3090/commit/a413321ad95667b9c767a8e56ee267657cdc690d))
- power-cap-sweep: 10W default increment + under-load median power sampling ([6d70b72](https://github.com/noonghunna/club-3090/commit/6d70b7287084cfa40dfdcf7470a8234cf8b5dbba))
- power-cap-sweep: auto-derive cap range from card's min/max power limits ([e5c7a34](https://github.com/noonghunna/club-3090/commit/e5c7a34e91e8577f61133693878f1625f92e99fc))
- Add scripts/power-cap-sweep.sh — automated cross-rig power-cap A/B (#83) ([#83](https://github.com/noonghunna/club-3090/pull/83) by @noonghunna)
- scripts: auto-detect running container + port in verify / bench (closes #52 promise) ([29718ca](https://github.com/noonghunna/club-3090/commit/29718cac994a7060e994fc85b898ffbec8fd73c1))
- verify-stress: add 3 probes to cover the bug shapes we missed ([5e745c5](https://github.com/noonghunna/club-3090/commit/5e745c5c85547c028a86fe2bcf83376d61b6c8b5))
- Add scripts/health.sh — operational health check for running server ([e7780c5](https://github.com/noonghunna/club-3090/commit/e7780c556a55f98038b287fca9314a35c88ec1a5))
- Split verify-full.sh → verify-full.sh (fast functional) + verify-stress.sh (boundary) ([5060e22](https://github.com/noonghunna/club-3090/commit/5060e22a6c25320104edf10fe799b4c2c31a2296))


### 🧹 Maintenance

- restructure: promote topology to a directory level (single/dual/multi4) ([acd7ffb](https://github.com/noonghunna/club-3090/commit/acd7ffb67c07a1df4b34ec11a7ec52087f249d96))
- Drop vllm-gemma4-mtp overlay tree (merged upstream as #41745, validated) ([aa99173](https://github.com/noonghunna/club-3090/commit/aa99173e7ad9577e6d95032b67c596c254d4ee13))
- chore(gitignore): allow results/lucebox-*/ — evidence for BENCHMARKS lucebox row ([030f780](https://github.com/noonghunna/club-3090/commit/030f780f24d5c14777ccd8e19d2788c58b7afcd8))
- chore(tools): commit residency-instrument as research tool with framing README (#41, #217) ([ed05d1c](https://github.com/noonghunna/club-3090/commit/ed05d1c35ca804a369108e18915af9e5db5bc4a1))
- chore(results): commit grammar bench evidence + gitignore investigation artifacts (#217) ([d82e898](https://github.com/noonghunna/club-3090/commit/d82e89807aefbcdf9d6b1bcf1ed83ad5f9552617))
- refactor: vendor vllm#40361 Marlin patched files in-repo (drops /opt/ai/vllm-src/ host dep) ([d8b341f](https://github.com/noonghunna/club-3090/commit/d8b341fa8cea9a7dec47c26a5f3afc81fd7e08d2))
- chore: untrack docs/diagnostics/, gitignore the path ([3f18053](https://github.com/noonghunna/club-3090/commit/3f18053659f2f2997574ed6da7317f7e5872f20e))
- Remove no-genesis-mtp.yml (research artifact, not user-facing) ([f4a28b1](https://github.com/noonghunna/club-3090/commit/f4a28b19eb4466d504dcf4e7c532e0d4fac5e967))
- Remove fast-chat.yml; extend P68/P69 disable to default ([37a4895](https://github.com/noonghunna/club-3090/commit/37a4895f6dae510e42729c6502d23e83599894a8))
- Restructure docs around hardware axis: SINGLE_CARD.md + DUAL_CARD.md ([26ac811](https://github.com/noonghunna/club-3090/commit/26ac8118de51480e6c0ecce6c8dc0ceccabc93fb))
- Audit + reconcile dual-card compose headers, patches README, setup output ([0f33561](https://github.com/noonghunna/club-3090/commit/0f33561b6bd85f890cd36d5ada7bfb489e10c6d7))


### 🧹 Other

- benchmarks: add JDWarner #107 TB3 dual-eGPU + mixed-arch row ([fa9df49](https://github.com/noonghunna/club-3090/commit/fa9df49ef2cd55f088db94dba67b3533702e9baa))
- Rename gemma-mtp-fp8.yml → gemma-mtp-int8.yml to match Ampere reality ([160e8fc](https://github.com/noonghunna/club-3090/commit/160e8fce8b5158c9870e4714f8e05bf63fa9460f))
- Two regressions caught + reframe Phase 2 around INT8 PTH (Ampere reality) ([119f296](https://github.com/noonghunna/club-3090/commit/119f2965401c19132f9c654420b742c9eb684b63))
- gemma-mtp-fp8: vendor rebased PR #40391 + stacked tool-parser fixes (#42006 + #41991) ([f93d312](https://github.com/noonghunna/club-3090/commit/f93d31215e20b195e07d4c30d2ad854852c9b7dc))
- gemma-mtp: drop PR #41745 overlay + bump to post-merge nightly ([595be8f](https://github.com/noonghunna/club-3090/commit/595be8fb8eb0442916a3570494ec5b90fc3e33f3))
- llama.cpp: --reasoning-format none default (opencode unblock, #97) ([af00ab7](https://github.com/noonghunna/club-3090/commit/af00ab7bef911ed6127ac900ecc081f9e5293ddc))
- Set dual-nvlink-dflash-noviz --max-model-len default to 188000 ([89c6862](https://github.com/noonghunna/club-3090/commit/89c686288e482a5b3529afd3af01d86157232c51))
- patches: qwen3coder tool-parser deferred-commit sidecar (#72) ([2e00b6d](https://github.com/noonghunna/club-3090/commit/2e00b6d718ea3e30fb0a0a380eaff095c39c098f))
- TQ3 composes: propagate PN34 to remaining 4 (follow-up to #82 audit) ([ab69f65](https://github.com/noonghunna/club-3090/commit/ab69f656910aa00a77da5c06aea9cd9f03041750))
- vllm/default: also enable P98 (belt+suspenders with PN34, follow-up to #82) ([2c7efe6](https://github.com/noonghunna/club-3090/commit/2c7efe61086181dfd6777bd2c1a2e55e2957d2c5))
- vllm/default: add GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1 (#82) ([3167497](https://github.com/noonghunna/club-3090/commit/3167497fef643267e42df7009272bb6c062c13fe))
- add dual-nvlink-turbo variant (rebased on v7.72.2 master, sibling-table edits dropped) (#65) ([#65](https://github.com/noonghunna/club-3090/pull/65) by @noonghunna)
- release(v7.72.2-uplift): Genesis pin + vLLM pin + sidecar consolidation (#59) ([#59](https://github.com/noonghunna/club-3090/pull/59) by @noonghunna)
- carnice-bf16mtp: restore original template + qwen3_xml parser ([d57579c](https://github.com/noonghunna/club-3090/commit/d57579c31ae6b4a42d0bf604e0e86364a2e94b85))
- carnice-bf16mtp: JSON tool format + empty think block, no reasoning parser ([a350df7](https://github.com/noonghunna/club-3090/commit/a350df7c911b8c8935980e328a7b4c57f374c70b))
- carnice-bf16mtp: add HF model URL to header ([5da50ec](https://github.com/noonghunna/club-3090/commit/5da50ec8978d64ebcefbf11637921955a6edd0e7))
- carnice-bf16mtp: formal narrative + code bench results ([7fef94f](https://github.com/noonghunna/club-3090/commit/7fef94f600a906687d60f4c4a9da85e3659d9da7))
- carnice-bf16mtp: 2 streams at 262K confirmed + formal bench numbers ([66d42c7](https://github.com/noonghunna/club-3090/commit/66d42c7940cacd011ad2468fcb1320f5384766c0))
- carnice-bf16mtp: 65K context was config choice, not VRAM ceiling — bumped to 262K ([1cf0cb2](https://github.com/noonghunna/club-3090/commit/1cf0cb288e046af486f3ace1ef2295f4944989b2))
- Carnice-V2-27B + BF16 MTP overlay — new compose variant ([bc28542](https://github.com/noonghunna/club-3090/commit/bc28542c5571c929ecee3e6371f30457855e9618))
- extend PN25 v3 + PN30 dst-shaped temp fix to all 4 TQ3 composes ([b875624](https://github.com/noonghunna/club-3090/commit/b875624f2d9ea41d6ead3a8563f9ef37ffbdb59c))
- Genesis pin d89a089 → 753344b + cross-rig validation of Sander's PN30/PN31 ([2b5ab4d](https://github.com/noonghunna/club-3090/commit/2b5ab4d0cf761e895772290ecaf45573727a0553))
- cliffs: v0.20 unblock recipe + 50K-stress-PASSES finding ([9506561](https://github.com/noonghunna/club-3090/commit/9506561ba8fdc35ef515a17275a2c94a5bec1e69))
- cliffs: document P38 silently no-op'd on TurboQuant KV path ([91355b8](https://github.com/noonghunna/club-3090/commit/91355b8fd577fc172f3f6d2af035d06b5ce08ec7))
- long-text/long-vision/bounded-thinking: middle-ground recovery 130K → 175K / 120K → 140K ([383b5cc](https://github.com/noonghunna/club-3090/commit/383b5cc38197d2ffa68a001c1e0c4c60877d99fe))
- long-text/long-vision: enable P37 + back off context for activation headroom ([1a931b4](https://github.com/noonghunna/club-3090/commit/1a931b4042090f182266aa150fa0e31d15afdbd5))
- genesis: bump pin v7.62 → v7.64 + add compile-safe FFN sidecar (#16) ([53d0663](https://github.com/noonghunna/club-3090/commit/53d0663a50c91f0a929917ae6390303842508e9e))
- Add local FA max seqlen clamp sidecar ([9f06a0f](https://github.com/noonghunna/club-3090/commit/9f06a0fe79e8e4f55b438c2e2427b8738e72d1cf))
- Fix local PN12 activation pool anchor ([41eabac](https://github.com/noonghunna/club-3090/commit/41eabac17b0b8b558121e213be1498855be314d6))
- CLIFFS: document PN12-is-partial finding (full stack still hits wall) ([537875a](https://github.com/noonghunna/club-3090/commit/537875a7fd1235ef21006ef1c5f5c2d5aabee74a))
- Add genesis #11 row to UPSTREAM.md ([bb406f9](https://github.com/noonghunna/club-3090/commit/bb406f90f16d2c59a0512b060f7af3028140e02a))
- Add Max ctx column to TL;DR + perf-summary tables on both pages ([e94c2e7](https://github.com/noonghunna/club-3090/commit/e94c2e78c4176704f4dffe93137af6f9d56f201a))
- DUAL_CARD: promote perf chart to top, parallel to SINGLE_CARD ([19fb8e7](https://github.com/noonghunna/club-3090/commit/19fb8e730f5f009f73ceef0e31ced393a2f9b2c2))
- Disable P68/P69 on long-vision, long-text, dual-turbo too ([f0cbcc6](https://github.com/noonghunna/club-3090/commit/f0cbcc6a9c9333af4dfaf6a7267ff8a93767a157))
- Disable Genesis P68/P69 in shipped composes (silent-stop bugfix) ([aab8ff4](https://github.com/noonghunna/club-3090/commit/aab8ff4a0e1eddc4c53ab24ef31ff748c54f84c8))
- Split charts per GPU-count page; chart sources land in tools/charts/ ([3742244](https://github.com/noonghunna/club-3090/commit/3742244e4d98960732569551490a64c62a23d1d8))
- Move performance chart into docs/img/ alongside vram-budget-dual ([2e3ae0c](https://github.com/noonghunna/club-3090/commit/2e3ae0c7861b4a5f74bdec0bff1b8cdca399a837))
- UX polish: pre-flight checks + cards-first wizard + PNG embeds ([abc06c3](https://github.com/noonghunna/club-3090/commit/abc06c3e3317ab2472242f8410a0d105f3631a16))
- FAQ: add VS Code Copilot LLM Gateway entry ([f275bf5](https://github.com/noonghunna/club-3090/commit/f275bf502a9ac914f011d750c1f63486ce981436))
- Add CONTRIBUTING.md — what kind of PRs land cleanly ([0c261eb](https://github.com/noonghunna/club-3090/commit/0c261ebcdfaf8bd057195d007eee28dfc5922085))
- CHANGELOG: capture post-launch polish day in cross + per-model logs ([1cc6ee6](https://github.com/noonghunna/club-3090/commit/1cc6ee6e24f991bed4a63eb084c82bea0a17bf21))
- Add launch.sh wizard + switch.sh stateless variant switcher ([4b77ed5](https://github.com/noonghunna/club-3090/commit/4b77ed5eb1eee64bbf913a332f38a54746c17d05))
- Add per-card VRAM allocation diagram + reference from model README ([88523b3](https://github.com/noonghunna/club-3090/commit/88523b3d05985476ae5bcbaeee05aab4d091319e))
- Cite Kaitchup Qwen3.6-27B GGUF eval as quant-quality lens ([b7ef91f](https://github.com/noonghunna/club-3090/commit/b7ef91f9518d5b8f521d2b850570722322c4ea3b))
- Pin Genesis to exact tested commit + add .env.example + issue templates ([ec704e4](https://github.com/noonghunna/club-3090/commit/ec704e4e2e86d8ff2ec990d8d544427f3e957dcd))
- Dual-card re-bench on club-3090 substrate + fix dual-turbo mount path ([c701474](https://github.com/noonghunna/club-3090/commit/c70147426dd241e894186f40bb3207a53f43c8df))
- dual-turbo: switch kv-cache-dtype k8v4 → 3bit_nc to align with test findings ([3e1f5f6](https://github.com/noonghunna/club-3090/commit/3e1f5f61c0475474461008a71d54ca39ddc908b5))
- Pin Genesis version + fix MODEL_DIR defaults + clean stale headers ([7f00e52](https://github.com/noonghunna/club-3090/commit/7f00e5214072491637ba7b02cdec2c9e8135b445))
- Fix .gitignore + add the entire models/ tree (initial commit was incomplete) ([2511a98](https://github.com/noonghunna/club-3090/commit/2511a981109b983d05247a485db8bdf999c6d38e))
- Initial commit — club-3090: model-agnostic LLM serving recipes for RTX 3090 ([3fa3333](https://github.com/noonghunna/club-3090/commit/3fa33332ce12b042c171fc98ad21fe412c0f92a0))



[Pin: `git checkout v2026.05.09`]

