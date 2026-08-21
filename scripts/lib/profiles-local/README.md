# profiles-local — your LOCAL model layer

This directory holds models **you** added to this checkout — community models,
fine-tunes, experiments. Nothing here is ever committed: the curated catalog in
`scripts/lib/profiles/` stays maintainer-only, and `git pull` / branch switches
can never conflict with your local additions.

Everything here is gitignored except this README and the `.gitignore` itself.

## Layout

```
scripts/lib/profiles-local/
├── README.md            ← this file (committed)
├── .gitignore           ← keeps the contents below out of git (committed)
├── models.d/<id>.yml    ModelProfile YAML — same schema as scripts/lib/profiles/models/
├── composes/<id>/...    your compose files (same profile-schema layout as models/<id>/)
└── registry.local.json  registry entries: { "<slug>": { …_entry kwargs… } }
```

## The easy way: c3 → ⑤ Promote

In the cockpit's Bring & Validate lane, run a ① Bring fit-check on your repo,
then press **P** (⑤ Promote). The scaffold pre-fills every arch fact the
deriver knows; you fill `display_name` + `family`, confirm, and the gated write
plan runs:

```
python3 scripts/lib/profiles/promote.py --layer local --spec-env C3_PROMOTE_SPEC \
  && bash scripts/diagnose-profile.sh <slug> \
  && bash scripts/preflight-add-model.sh <slug>
```

That writes all three artifacts above and prints `PROMOTE_OK <slug>`.

## The manual way

1. Write `models.d/<id>.yml` (copy the shape of an existing
   `scripts/lib/profiles/models/*.yml`; see `docs/ADDING_MODELS.md`).
2. Put your compose under `composes/<id>/<engine>/compose/<topology>/<quant>/base.yml`
   with a `# Profile (at-a-glance):` header carrying `Status: 🐣 Incubating`.
3. Append an entry to `registry.local.json`:

   ```json
   {
     "local/my-model-dual-autoround-int4": {
       "model": "my-model",
       "weights_variant": "autoround-int4",
       "workload": "long-ctx-single",
       "engine": "vllm-stable",
       "drafter": null,
       "kv_format": "fp8_e5m2",
       "tp": 2,
       "max_ctx": 131072,
       "max_num_seqs": 2,
       "mem_util": 0.92,
       "compose_path": "scripts/lib/profiles-local/composes/my-model/vllm/compose/dual/autoround-int4/base.yml",
       "default_port": 20230,
       "status": "incubating"
     }
   }
   ```

   Keys are exactly the `compose_registry._entry(...)` kwargs; defaults are
   applied the same way. Validate with `bash scripts/diagnose-profile.sh <slug>`.

## Rules

- **Slug namespace:** local slugs MUST start with `local/`. The loader refuses
  anything else, and refuses any slug or model id that collides with a core
  catalog row (loudly — a broken `registry.local.json` fails the launch rather
  than silently shrinking the catalog).
- **Status:** local entries start at `"incubating"` — hidden from
  `switch.sh --list`, launched only with `--force` (see the Status enum in
  `compose_registry.py`). Promote up the enum locally as your model validates.
- **Never a default:** local entries are visible to every runtime lookup
  (`launch.sh`, `switch.sh`, `diagnose-profile.sh`, the cockpit catalog) via
  `get_registry()`, but the curated-default tables (`DEFAULTS`,
  `ENGINE_PREFERENCE`, `RECOMMENDED_DEFAULT_MODELS`) stay core-only — a local
  model can never become someone's auto-default.
- **No core writes:** adding a model to the CURATED catalog is a maintainer PR.
  `promote.py --layer core` exists for that workflow and is double-gated
  (`--layer core` **and** `C3_ALLOW_CORE_PROMOTE=1`); it edits
  `compose_registry.py` directly and should only ever run in a maintainer
  checkout on a clean branch (git is the rollback).

## Uninstall

Delete the file(s): `rm scripts/lib/profiles-local/models.d/<id>.yml`, the
compose dir, and the JSON entry. There is nothing else to revert — no core file
was ever touched.
