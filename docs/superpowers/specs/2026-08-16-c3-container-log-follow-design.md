# c3 Containers log-follow (`[f]`) — design

Date: 2026-08-16
Status: approved in brainstorming (user, 2026-08-16) — ready for an implementation plan
Surface: `tools/serve-cockpit` (c3) + `tools/tui-core` (`LivePane`)

## Problem

In c3's Operate · Containers view, the Logs drill (`#drill-logs` `LivePane`) shows a
**one-shot** `docker logs --tail 200` snapshot: `action_container_logs()` →
`stream_container_logs()` (app.py:10452 / 10551) reads the tail exactly once. New log
lines only appear if the user re-presses `[l]`. There is no way to watch a container's
log live, and no way to pause a live view.

## Decisions (and why)

| Question | Decision | Rationale |
|---|---|---|
| Stream transport | **Periodic poll** of the existing `container_logs(name, tail=200)` service read, ~2s cadence | Reuses the injected one-shot `Runner` (fully `FakeRunner`-mocked in tests → headless tests stay subprocess-free); no long-lived subprocess lifecycle across pause / container-switch / tab-switch / app-quit; transient docker errors can't wedge a follow process. 2s granularity is imperceptible for a log-tail. True `docker logs -f` (Approach B) was considered and rejected on process-lifecycle surface (single-flight `SubprocessRunner` needs a 3rd instance, `start_raw` assumes a parser, new streaming test fake, orphan risk on quit). |
| Default state | **Snapshot first, opt-in to stream** | `[l]` keeps today's semantics exactly; streaming is armed deliberately. Protects against log-spammy containers. |
| Toggle key | **`[f]`** (follow, matching `docker logs -f`) | Lowercase `f` is free at app level; the `f` = force-start binding (app.py:1976) lives only on the staged-write **modal** screen, which owns its keys — no conflict. |
| Stream scope | **Follows the selection** | Follow is a property of the Logs pane, not of a container. Moving the cursor re-snapshots (existing nav path, app.py:11992) and polling continues on the new container. |
| New-line marking | **Persistent subtle tint: brighter default foreground** | Lines arriving on a poll tick render in a brighter default-foreground style; snapshot/tail lines stay normal. Copy stays clean for free: `LivePane.append_line` buffers `Text.from_markup(line).plain` for `[Y]` (live_pane.py:266–277), so markup is stripped from the copy. Decaying-flash and leading-marker options were considered; flash needs re-styling off-screen `RichLog` lines + timers, marker needs a new display-vs-copy API. |
| Disarm on tab/mode leave | Yes, silently | No background polling; preserves the invariant "a periodic poll never starts docker logs for a container the user didn't actively select" (app.py:8273). |

## Behavior & state

`[f]` is a toggle, active **only** in mode 0 · Containers tab (same context gate as
`l`/`t`). Footer binding shown in that context; palette entry added.

State (on `CockpitApp`) — three distinct states: **off** (never armed / disarmed),
**following**, **paused**. Paused is *not* the same as off: it keeps the anchor and the
"this pane was a live view" context, and resumes incrementally.

- `_log_follow_armed: bool` — True in following **and** paused.
- `_log_follow_paused: bool` — True only in paused.
- `_log_follow_timer` — the `set_interval` handle (or `None`; None while paused).
- `_log_follow_anchor: Optional[str]` — last displayed line of the current tail.
- `_log_follow_name: str` — container the anchor belongs to.

**Arming (`[f]` while off):**

1. No container selected, or selected service stopped → warn notify (same as `[l]`), no state change.
2. Activate the Logs drill tab (`#drill-tabs` → `drill-tab-logs`).
3. `_log_follow_armed = True`; `_log_follow_name = name` — set **before** the snapshot
   so the snapshot path's rebase (below) applies.
4. Load the initial `--tail 200` snapshot via the existing
   `stream_container_logs(name)` (`@work(group="container-logs", exclusive=True)`) —
   normal color; on success it rebases the anchor to the last line.
5. `_log_follow_timer = self.set_interval(2.0, self._log_follow_tick)`.
6. Pane title → live indicator (see UI below); notify once: `Log follow armed — [f] to pause` (timeout ~3s).

**Pausing (`[f]` while following):** stop the timer, `_log_follow_timer = None`,
`_log_follow_paused = True` (anchor/name kept), pane title → paused indicator, append
one dim **display-only** note (`buffer=False`, so it never enters the `[Y]` copy
buffer): `follow paused`.

**Resuming (`[f]` while paused):** `_log_follow_paused = False`; run one poll
**immediately, incrementally** against the stored anchor — a quiet log shows no change;
lines emitted during the pause arrive tinted — then restart the interval, title → live.

**"Disarm"** (used throughout): stop the timer if running, `_log_follow_armed = False`,
`_log_follow_paused = False`, clear the anchor/name, title → `Live`.

**Container change while armed:** the existing row-highlight → `_load_active_drill_tab`
path (app.py:11992) re-snapshots the new running container; `stream_container_logs`
rebases the anchor (see below), so the next tick dedupes against the new container.

**A stopped row is selected while armed.** The pane shows the existing "stopped · no
live logs/stats" placeholder (`_clear_drill_for_stopped`); the mode **stays armed** —
ticks no-op while the selection is stopped or unselected, and follow resumes when the
next running row is selected. **Exception:** the container that was *being followed*
itself stopped or vanished (selected row stopped/absent **and**
`selection.name == _log_follow_name`) → dim note + **disarm** — without this the live
indicator would lie about a dead container (`docker logs` on a stopped container still
succeeds, just with no new lines).

**Leaving the Containers tab / switching modes:** the app disarms on leave (timer
stopped, state reset). Re-entering starts in OFF.

## Polling & dedupe

- Cadence constant: `_LOG_FOLLOW_PERIOD = 2.0` (module level, next to the other poll constants).
- The interval callback is a **sync** `def _log_follow_tick(self)` that calls an
  `async def _log_follow_poll()` decorated
  `@work(group="container-logs", exclusive=True)` — same group as
  `stream_container_logs`, so a tick and a manual `[l]` can never interleave (exclusive
  cancels the in-flight), and the existing `workers.cancel_group(self, "container-logs")`
  paths (app.py:12029) cover it with no change. (Matches the established idiom: sync
  interval callback → `@work` coroutine, e.g. the estate poll.)

**Tick guards** (local, free, in order): `_log_follow_armed` → `_active_mode == 0` →
active operate tab == `tab-containers` → active drill tab == `drill-tab-logs` → a
container is selected. Any miss → return.

**Local liveness check** (per Behavior): the selected `ContainerInfo.status` (from the
periodic estate poll — no extra docker call) is `stopped`, or the selected container is
no longer listed →

- selection differs from `_log_follow_name` (user browsed to a stopped row) → no-op;
  the mode stays armed;
- selection **is** `_log_follow_name` (the followed container died) → dim display-only
  note (`▸ <name> — stopped · follow stopped`) + **disarm**.

**Dedupe — single-line anchor.** After any successful full-tail render, set
`anchor = lines[-1]` (or `None` if the tail is empty). Each tick:

1. `res = await self._data.container_logs(name, tail=200)` — existing service method
   (services.py:2228); **no new service surface**.
2. `res["error"]` → dim display-only note `follow stopped: <err>` + **disarm**.
3. Empty `lines` → no-op (keep anchor).
4. `_log_follow_name != name` (shouldn't happen — navigation rebases — but guard) → resync.
5. **Anchor found** in the new tail (last occurrence) → append only the lines after it,
   each tinted; anchor = new tail's last line.
6. **Anchor missing** (≥200 new lines, container restart, in-place `\r` progress-bar
   line) → **resync**: `clear_log()`, re-render the fresh tail in **normal** color,
   rebase anchor. Resync is "a new look at the tail" — the same semantic as a snapshot.

The anchor is deliberately a single line: minimal state that makes quiet logs
append-zero and active logs append-exactly-the-new-lines, with a bounded, visible
resync as the only failure mode.

**Self-heal:** the exclusive `container-logs` group means a tick can cancel an
in-flight navigation snapshot (or a manual `[l]` cancel a tick). Either way the
pane ends up showing a consistent tail and the next tick either dedupes normally or
hits the name-guard/resync (step 4) and rebases — no stuck or half-rendered state.

**Snapshot path gets one awareness:** in `stream_container_logs`, after a successful
load, rebase anchor + name **when `_log_follow_armed` is True** (covers the arm
snapshot — armed is set before the call — navigation snapshots, and manual `[l]`
while following). Consequences:

- manual `[l]` while following = natural "resync now" (clear + re-tail, normal color, polling continues);
- navigation snapshots (app.py:11992) rebase for free.

Reconsidered and rejected: `docker logs --since` + timestamp cursor (changes on-screen
line format; host/container clock dependence) and true `docker logs -f` (see Decisions).

## UI

**Pane title** (`#drill-logs`'s `.live-title` `Label`; this pane never calls
`set_run_header`, so `update_elapsed_timer` never fights it):

- off (never armed / disarmed by leaving) → `Live` (unchanged)
- armed → `Live  ●  following`
- paused → `Live  …  paused` (dim)

Glyph/spacing exact forms at implementation; no U+FE0F variation-selector emoji (c3
convention), `●` is a text-default glyph.

**Hint line** (`#containers-hint`): static copy gains `\ [f] follow` next to the existing
`\ [l] logs   \ [t] top …`.

**Toast:** arm only (`notify(..., title="Logs", timeout=3)`); pause is legible from the
title + in-pane note.

**Scroll-follow (tui-core `LivePane`).** Today `append_line` auto-scrolls to the bottom
on every append whenever the widget's `_follow` flag is True — and nothing ever turns it
off (the `toggle_follow()` at live_pane.py:335 is dormant). With a live stream, the user
could not scroll up to read history. Add to `LivePane`:

- on user scroll of `#live-log`: not at the bottom → auto-follow **off**; at the bottom
  → auto-follow **on**.
- No new key; standard tail-reader behavior; improves every `LivePane` consumer (c3
  Run/Gate output, c3t). While follow is **paused**, nothing appends, so the pane is
  readable at any scroll position; this only matters while actively streaming.

## Keybindings / palette

- `Binding("f", "container_follow", "Follow", show=False)` at app level, context-restricted
  to `{tab-containers}` in mode 0 (same context-map pattern as `container_logs`).
- Palette entry: `("container_follow", "Container log follow", "Containers — [f] arm/pause the live log tail for the selected container")`.
- The modal's `f` (force-start) is unaffected: modal screens own their keys.

## Testing

**tui-core** (`tools/tui-core/tests/test_live_pane.py`):

- scroll-up disables auto-follow; scroll-to-bottom re-enables; append while scrolled up
  does not move the scroll offset.

**c3** (`tools/serve-cockpit/tests/test_app_headless.py`) — all headless; `conftest`
already blocks real subprocesses; ticks driven by direct `_log_follow_tick()` calls (no
real 2s waits):

1. `[f]` arms: timer set, title indicator, snapshot loaded; wrong tab/mode → no-op.
2. Tick appends **only** new lines (fake: L1–L3, then L1–L5 → L4, L5 exactly once).
3. Quiet log (same tail again) → zero new lines.
4. Anchor-missing → resync: clear + full tail, no duplicated tail content.
5. `[f]` pauses: timer cleared, no further runner calls, `follow paused` note absent from
   `tail_text()`.
6. Resume: one immediate incremental poll + timer re-armed (quiet log → no change;
   paused-window lines arrive tinted).
7. Navigation to another running container while armed → snapshot re-bases; ticks stream
   the new container.
8. The followed container stops/vanishes (selection == `_log_follow_name`) → note +
   disarm; browsing to a *different* stopped row while armed → stays armed, tick no-ops.
9. Runner error → note + disarm.
10. Copy stays clean: tick-appended tinted lines' `tail_text()` returns unmarked plain
    text.
11. Tint markup present on-screen for tick lines only (snapshot lines un-tinted).

Both suites run (CLAUDE.md: the c3 suite is not in `scripts/tests/*.sh`):
`tools/serve-cockpit/.venv/bin/python -m pytest tools/serve-cockpit/tests/ -q` plus
tui-core's suite.

## Docs

- `tools/serve-cockpit/README.md` keybindings table: add an `f` row, noting the context
  split (`f` = log-follow on the Containers tab; `f` = force-start only inside the
  staged-write modal, where it shadows app keys).
- No new top-level doc.

## Open at implementation (decide, don't re-debate)

- **Tint token:** brighter default-foreground as a single Rich style — candidate
  `bold` (portable "brighter") vs `bright_white` (hue-neutral; zero effect on themes
  whose default fg is already bright white). Pick whichever reads as "brighter default"
  in both a light and a dark terminal; record the choice in the commit message.
- Exact title glyphs/spacing (above: `●` vs alternatives, dim treatment of "paused").

## Out of scope (explicit)

- c3t views (no container-log drill there).
- True `docker logs -f` streaming.
- Tinting the initial snapshot; changing `[Y]` copy semantics.
- The pre-existing `from_markup` edge for docker lines containing `[brackets]`
  (both display and copy paths parse the raw line today; wrapping in a color tag is no
  worse — logged, not fixed here).
