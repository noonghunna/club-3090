# Launch-knob catalogue

`launch-knobs.json` describes the launch settings a user may persist (#1465): what
values each one takes, which engines read it, how it depends on other settings, and
why. It is phase 3a of the #1465 plan. The resolver (3b) and the per-slug store (3c)
read it — see [Resolving a slug's settings](#resolving-a-slugs-settings-3b3c) below;
the c3 form (3d) will too.

| File | Role |
|---|---|
| `launch-knobs.json` | the catalogue (hand-written, JSON so the stdlib-only launcher path can read it) |
| `launch_knobs.py` | loads and shape-checks the catalogue; validates values and dependencies; scans a compose for the knobs it reads |
| `launch_knobs_check.py` | the checks behind the guard |
| `scripts/tests/test-launch-knobs.sh` | the guard, including negative controls that prove each check can fail |
| `scripts/lib/registry-emit.sh --json` | exposes each slug's knobs as `variants[].knobs` |
| `scripts/lib/launch_settings.py` | the resolver: effective value + source per knob, validation, delivery (3b) |
| `scripts/lib/slug_settings.py` | the per-slug store, `slugs.json` (3c) |
| `scripts/tests/test-launch-settings.sh`, `test-slug-settings-store.sh` | their guards |

## Which slugs read a knob: derived, never declared

Consumption comes from the compose text. A value set for a slug that doesn't read it
does nothing, with no error (#1465 gotcha 4). A compose **reads** a knob when docker
can deliver the host value into the container. There are two ways:

- the knob is declared under a service's `environment:` (`- NAME`, `- NAME=${NAME…}`,
  `NAME:` or `NAME: ${NAME…}`);
- compose interpolates it (`${NAME}`, `${NAME:-x}`, `$NAME`, i.e. an odd run of `$`)
  anywhere outside that entry.

A container-side read (`$${NAME}`, an even run of `$`) does not count on its own: the
container sees the value only if the knob is also forwarded. A compose that reads
`$${NAME}` without forwarding it has a dead setting, and the guard fails on it.
Comments are ignored, whether YAML comments or shell comments inside a `|` block;
many composes mention knobs only in comments. Flow-style `environment:`, YAML
aliases and merge keys are refused rather than guessed at. No shipped compose uses
them.

The scan is a claim, and the guard proves each claim (#1465 gotcha 5). It renders
every registered compose with `docker compose config`, with every catalogued knob set
to a sentinel value. It does this twice, once for each way the launchers hand
settings to docker: the process environment (`switch.sh` / `launch.sh`) and an
`--env-file` (`gpu-mode.sh`). A claimed knob's sentinel must reach the rendered
environment, command or entrypoint. An unclaimed knob's sentinel must not.

## Schema

Knob (`knobs.<ENV_NAME>`):

| Field | |
|---|---|
| `status` | `first-batch` or `candidate` |
| `description` | one line |
| `type` | `enum` · `bool` · `int` · `size_gb` · `string` |
| `unit` | optional (`GiB`) |
| `unset` | what the compose does when the knob is unset |
| `engines` | engine kinds whose composes read it. The guard checks this equals what the scan finds. Kinds come from `scripts/lib/engine-kind.sh`. |
| `variants` | the value domain, per engine and model (below). Every reader must match exactly one. |
| `requires` | `{when, knob, then, why, source}`. `when`/`then` is `"set"` or `{"equals": v}`. Example: `KV_OFFLOAD_DISK` `{"equals":"1"}` requires `KV_OFFLOAD_GB` `"set"`. |
| `interacts` | `{knob, effect, source}`: other settings that change or override this one |
| `caveats` | `{text, source}` |

Variant:

| Field | |
|---|---|
| `match` | `{engine?: [kind], model?: [id], compose_contains?: "literal"}`; `{}` matches every reader |
| `enforced` | where an out-of-domain value is caught: `boot` (the compose refuses and the container exits), `request` (it boots, then every request without its own value fails), `none` (nothing refuses it), `unverified` (the compose passes it through and the repo records no evidence of what happens next) |
| `values` | accepted strings (`enum` / `bool`); a `bool` also has `spelling: {on, off}` |
| `aliases` | accepted value → the value it means (`high` → `xhigh`) |
| `pattern` / `min` / `max` / `above` | numeric domain. `int` and `size_gb` imply `[0-9]+` and `[0-9]+(\.[0-9]+)?` |
| `default` | the compose's fallback, as the compose writes it (`""` = unset/off). Omit it when it varies per compose. The guard checks it against every reader. |
| `every_consumer_contains` | literals every reader's compose must contain, e.g. its validation line. They catch a compose that drifts away from the domain. |
| `source` | `{file, what}`: where the rule comes from and what that code does |
| `evidence` | `[{file, contains}]`: text the domain relies on outside the compose (e.g. the patched chat template) |
| `checks` | one or more executable proofs against real composes (below) |

Check: `{kind, compose, accept, reject, stricter?, env?, …}`. It runs values through
the compose's own code:

- `shell`: runs the entrypoint lines from the `from` regex to the `to` regex (plus
  `to_plus` lines), with `$$` → `$`, under the script's own `set -e…` line, in a
  clean `env -i` shell. File-touching commands (`rm`, `curl`, …) are stubbed. The
  compose accepts a value when the fragment exits 0; with `emit` + `json`, the emitted
  text must also parse as JSON.
- `json_line`: interpolates the matched line (e.g. the `--default-chat-template-kwargs`
  item) and requires valid JSON; `strip_prefix` removes `NAME=` from an env entry.
- `passthrough`: the compose interpolates the value into that line unchecked.

A probe is a value, or a `{NAME: value}` map for dependency cases, merged over `env`.

| list | catalogue | compose, `enforced: boot` | compose, otherwise |
|---|---|---|---|
| `accept` | valid | accepts | accepts |
| `reject` | invalid | refuses | accepts (the resolver has to catch it) |
| `stricter` | invalid | accepts: stricter on purpose, e.g. `KV_OFFLOAD_GB=0` | n/a |

## Adding a knob

1. Read how **every** compose that uses the variable handles it (`command grep -rn NAME models/*/*/compose`,
   ignoring comments). Write each domain down from that code, not from memory, one
   variant per distinct handling. Put the file and what it does in `source`.
2. Add the entry, then run `python3 scripts/lib/profiles/launch_knobs.py --check`
   (shape) and `python3 scripts/lib/profiles/launch_knobs_check.py` (coverage, domains,
   delivery). Every reader must be covered, and every variant needs at least one
   `check` against a real compose with `accept` and `reject` probes.
3. If the new knob is ever forwarded as `- NAME=${NAME:-}`, the ratchet in
   `test-launch-knobs.sh` counts it. Prefer a bare `- NAME`, so that unset stays unset.
4. `bash scripts/tests/test-launch-knobs.sh` and `bash scripts/tests/test-registry-json.sh`.

**A compose that starts reading a catalogued knob** (a new slug, or a new model on an
engine) fails the guard until its reader is covered. If its handling matches an
existing variant, widen that variant's `match`. Otherwise add a variant with its own
check. That failure is the point: it is where a model's different domain would
otherwise go unnoticed.

## Resolving a slug's settings (3b/3c)

`scripts/lib/launch_settings.py` is the one resolver; `switch.sh` calls it and nothing
reimplements it. For each catalogued knob a slug's compose reads (the scan above), the
value comes from the first layer that sets it:

| # | Layer | Where |
|---|---|---|
| 1 | `shell` | exported for this launch. Only a key the settings loader did **not** export counts: `switch.sh` passes the loader's `CLUB3090_CONFIG_SOURCE` as `--loaded`, and `launch.sh` drops the knobs its own loader exported before it runs `switch.sh` |
| 2 | `this slug` | `<config dir>/slugs.json`, written by `switch.sh --set <slug> KEY=VALUE` |
| 3 | `model pin` | `CLUB3090_THINKING_<MODEL>`: `on` → `ENABLE_THINKING=true`, `off` → `false`; `inherit` or anything else adds nothing. The only pin today |
| 4 | `club3090.env` · `secrets.env` · `repo .env` | the global settings, in the loader's own order |
| 5 | `compose default` | nothing sets it; the compose's `${KNOB:-x}` applies |

An empty value from the shell or a settings file means unset: every catalogued knob is
read as `${KNOB:-x}`, so empty is the compose default. An empty shell value still masks
the saved layers, which is how `KV_OFFLOAD_GB= bash scripts/switch.sh <slug>` turns a
saved RAM tier off for one launch. `slugs.json` never holds an empty value.

**Validation**, in `check_variant`, before the running slug is torn down (#1464). Only
values a layer supplied are checked, never a compose default:

| Refused | Rule |
|---|---|
| a value outside the slug's domain | the variant matching the slug's engine kind, model and compose. With `--force`, a domain marked `enforced: unverified` or `none` only warns; `boot` and `request` always refuse |
| an unmet `requires` rule | e.g. `KV_OFFLOAD_DISK=1` without `KV_OFFLOAD_GB`, `KV_OFFLOAD_DISK_GB` without `KV_OFFLOAD_DISK=1`. Skipped when the compose defaults alone break it |
| a RAM tier the host can't hold | `KV_OFFLOAD_GB × factor + 28 GiB > MemTotal` (GiB). `factor` is 1.0 on vLLM, which pins exactly the tier, and 74/64 on SGLang (host use measured at 74 GiB for a 64 GiB tier). The 28 GiB is the headroom `preflight_lmcache_ram` budgets for a 27B TP=2 serving process plus the OS, the shape of every compose that reads the knob. MemTotal, not MemAvailable, because the running slug still holds its memory at this point. Applies with `--force` too. There is no disk-tier check: the disk tier is uncapped by default (discussion #1419) |
| an unreadable `slugs.json` | bad JSON, wrong shape, or a newer `version`: the saved values can't be known |

Warned, never refused: a saved per-slug or global value, or a thinking pin, for a knob
the slug doesn't read (it does nothing there); a per-slug key that isn't a catalogued
knob; a pin that isn't `on|off|inherit`.

**Delivery.** Right before `docker compose up`, `switch.sh` exports every value that came
from this slug, the model pin or a settings file, overriding a global value the loader
already exported and never touching the shell's. `--explain <slug>` shows each knob's
value, its source and what it overrides.

**The store**, `slugs.json`: `{"version": 1, "slugs": {"<slug>": {"KEY": "value"}}}`.
Values follow the global writer's literal rules (`club_config.check_value`). Credential-
looking keys are refused, so secrets stay in the 0600 `secrets.env`. Writes are atomic
(temp file + `os.replace`) under the config dir's `.lock`, the loader's lock file.
`switch.sh --set` also refuses a knob the slug doesn't read and a value outside its domain;
`--unset` can always remove a key that is stored, so the warning above has a way out.
