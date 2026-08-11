# Plan — #423: tag Dagster assets with the layer they belong to

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/423>
Branch: `claude/sleepy-agnesi-5d27ea`

## Verdict

**Worth implementing, roughly as described.** The mechanism the issue names (an asset tag) is the
right one, and the payoff is real: an operator can type one selection string into the Dagster UI
and see only the assets the live service runs. It is also the queryable form of the fail-fast /
fail-operational asymmetry that
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#rd-fails-the-other-way)
already argues but currently leaves implicit.

### Departures from the issue

1. **The tag values cannot be the literal strings "R&D" and "production".** Dagster rejects `&` in
   both tag keys and tag values. Verified against the installed Dagster 1.13.17:

   ```text
   {"layer": "R&D"} -> DagsterInvalidDefinitionError: Invalid tag value: R&D, for key: layer.
                       Allowed characters: alpha-numeric, '_', '-', '.'. Must have <= 63 characters.
   {"R&D": ""}      -> DagsterInvalidDefinitionError: Found invalid tag keys: ['R&D'].
   ```

   So the vocabulary has to be chosen, not copied from the issue title.

2. **One key with two values, not two bare tags.** The issue reads as "tag it `R&D` *or*
   `production`", which suggests two independent marker tags. A single `layer` key with two values
   is better: every asset carries exactly one, so an untagged asset is visibly a gap rather than
   an ambiguity, and the two selections are symmetric.

3. **`promotable_model_runs` and `promoted_model` are classified as production, not R&D**, despite
   reading MLflow. Reasoning and the counter-argument are in [Ambiguous cases](#ambiguous-cases) —
   this is the one classification Jack should confirm.

## The mechanism: tags, not a group and not a kind

Dagster offers three things that could carry this classification. Findings below are from a
throwaway Dagster project run against this repo's pinned Dagster 1.13.17, with the UI open.

**`tags` — chosen.** Verified:

- `AssetSelection.from_string('tag:"layer"="production"')` parses and resolves to exactly the
  tagged assets. The same string is what the Dagster UI's asset-selection box accepts, what the
  catalog page takes as its `?asset-selection=` query parameter, and what
  `dg launch --assets` / `dagster asset materialize --select` take on the command line.
- The asset detail page renders a **Definition → Tags** panel showing `layer: production`.
- The GraphQL API exposes `tags { key value }` per asset node, so anything built on the UI's API
  can filter on it too.
- `AssetSelection.tag("layer", "production")` exists as the Python API, so a future job or
  schedule can select by layer without restating a list of asset names.

**`group_name` — rejected.** A group is one-per-asset and exclusive, which suits a partition, but
it is Dagster's *primary* structural axis: it lays out the global lineage graph into labelled
boxes and appears as a column in the asset catalog. Spending it on the R&D/production split gives
up the ability to group by anything else later (ingest / features / CV / serving), and this repo
has not yet spent it — every asset is in `default` today. Groups also cannot express an asset that
is legitimately both, whereas a second tag key could.

**`kinds` — rejected.** Dagster implements kinds as reserved `dagster/kind/<name>` tags (confirmed
in the GraphQL output) and renders them as technology badges on the graph node. They are for
naming the tool an asset uses — `python`, `s3`, `xgboost` — and there is a hard limit of three per
asset. Using one for a lifecycle classification would misuse a namespace Dagster owns.

**Bulk application in `definitions.py` — rejected.** `load_assets_from_modules` accepts
`group_name` but **not** `tags` (verified: `TypeError: got an unexpected keyword argument 'tags'`),
so the group-per-module trick has no tag equivalent. `dagster.map_asset_specs` could do it, but it
is more machinery than eleven decorator arguments, and it would tie an asset's classification to
which file it happens to live in — exactly the coupling that breaks for the two ambiguous assets
in `production_assets.py`.

### Vocabulary

| | Value |
|---|---|
| Tag key | `layer` |
| Production value | `production` |
| R&D value | `rnd` |

`rnd` is the ASCII spelling of R&D that Dagster's character set permits; `r-and-d` and `research`
are the alternatives, and `env` was rejected as a key because both layers run inside the same
Dagster deployment, so `env` would read as the deployment environment.

## What changes, file by file

### New: `src/nged_substation_forecast/_asset_tags.py`

A small module beside the existing `_sentry.py`, holding the vocabulary once so the strings never
appear at a decorator:

- `LAYER_TAG_KEY: Final[str] = "layer"`
- `PRODUCTION_LAYER_TAGS` and `RND_LAYER_TAGS` — ready-to-splat mappings, so a decorator reads
  `@asset(tags=PRODUCTION_LAYER_TAGS)`.

The two mappings are `MappingProxyType`, not plain `dict`. This is not cosmetic: Dagster stores
the mapping it is handed **by reference**, so one shared `Final[dict]` used by eleven decorators
would let a mutation anywhere change every asset's tags. Verified — mutating the source dict after
definition changed the resulting asset's tags; `MappingProxyType` is accepted by `@asset` and
makes that impossible.

Placed at the package root rather than inside `defs/` because `_sentry.py` sets that precedent for
a small shared module, and because `defs/` is the directory `dg` treats specially.

### `src/nged_substation_forecast/defs/assets.py` — decorator lines only

| Asset | Tag |
|---|---|
| `power_time_series_and_metadata` | `production` |
| `h3_grid_weights` | `production` |
| `ecmwf_ens` | `production` |

All three are inputs the live service needs. Two of them (`power_time_series_and_metadata`,
`ecmwf_ens`) already run on scheduled jobs carrying the Sentry failure hook, which is the existing
implicit form of exactly this classification. `h3_grid_weights` is not scheduled but `ecmwf_ens`
depends on it, so production cannot run without it. That CV also reads all three does not make
them R&D: the tag says which layer needs the asset, and production needing it is the binding
constraint.

`@asset` becomes `@asset(tags=PRODUCTION_LAYER_TAGS)` for the first two; `ecmwf_ens` gains one
`tags=` argument inside its existing argument list.

### `src/nged_substation_forecast/defs/production_assets.py` — decorator lines only

| Asset | Tag |
|---|---|
| `promotable_model_runs` | `production` |
| `promoted_model` | `production` |
| `live_forecasts` | `production` |

### `src/nged_substation_forecast/defs/cv_assets.py` — decorator lines only

| Asset | Tag |
|---|---|
| `eligible_time_series` | `rnd` |
| `effective_capacity` | `rnd` |
| `trained_cv_model` | `rnd` |
| `cv_power_forecasts` | `rnd` |
| `metrics` | `rnd` |

All five exist to compare candidate models on the leaderboard. None runs on the production box and
none is on a schedule. `effective_capacity` is the closest call — it computes the NMAE denominator,
which is an evaluation concern, and its only consumer is `metrics`.

### Deliberately out of scope

- **Asset checks** (`defs/checks.py`), **jobs** and **schedules** are not tagged. The issue is
  about the asset graph, both existing checks are already production-only, and the Sentry hook
  already marks the three production jobs.
- **No asset body is touched.** Issue #505 owns the `ecmwf_ens` body and issues #486 and #488 own
  bodies in `production_assets.py`, all in flight in parallel. This diff is decorator lines,
  imports, one new module, one new test file, and docs — chosen so the merges do not collide.
  If implementation finds it needs to change a body, stop and ask.

## Ambiguous cases

**`promoted_model` and `promotable_model_runs`.** Recommended: `production`.

*For production*: promotion is an operations task, not a research one. It changes what the live
service serves, `docs/live_service/operations.md` documents it as steps 1–2 of *operating the live
service*, and `production-deployment.md` argues promotion-as-an-asset precisely so that changing
production has an audit trail. Tagging it `rnd` would hide the promotion step from the operator who
performs it.

*Against*: both assets need a reachable MLflow tracking server, and the production box has none —
the model arrives there baked into the Docker image, as `live_forecasts`' own `deps` comment
explains. An operator filtering to `layer=production` on the box would see two assets that can
never be materialised there.

I judge the first argument stronger, because the filter's job is "what concerns whoever operates
the service", not "what this particular host can execute". A third value (`promotion`) was
considered and rejected: it breaks the issue's binary for one edge case and gives the operator a
third string to remember. **Jack's call.**

## Design-philosophy check

This change adds no runtime code path, so it cannot fail open or closed. It adds no asset check.

It is the queryable form of rule 9 in
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
— "fail in the direction where being wrong is cheapest to recover from" — whose *R&D fails the
other way* section already names this issue as the missing mechanism. It serves
[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/)
(a service that mostly runs itself) by making "what does production actually need" answerable in
the UI rather than by reading three modules.

No design principle is traded away.

## Tests

New file `tests/test_asset_layer_tags.py` — new rather than added to `tests/test_assets.py`,
because that module is `pytest.mark.integration` and materialises assets, while these are fast
definition-time assertions, and because a new file cannot collide with the parallel sessions
editing the existing test modules.

| Test | Assertion | Why it fails on `main` today |
|---|---|---|
| `test_every_asset_declares_its_layer` | Every asset key in `defs.get_repository_def().asset_graph` has a `layer` tag whose value is `production` or `rnd` | No asset carries any tag on `main`, so this fails on the first asset |
| `test_assets_are_classified_by_layer` | `live_forecasts`, `ecmwf_ens` and `power_time_series_and_metadata` resolve under `AssetSelection.from_string('tag:"layer"="production"')`, and `trained_cv_model` and `metrics` under the `rnd` equivalent | Both selections resolve to the empty set on `main` |

The first is the one with ongoing value: it fails the moment someone adds an asset without
classifying it. It asserts a property of every asset rather than an exact list of names, so it does
not become a merge conflict when a parallel session adds an asset.

Resolving through `AssetSelection.from_string` rather than reading `.tags` directly means the test
exercises the same parser the UI and the CLI use, which is what the issue actually promises.

Neither test needs network, a trained model, or wall-clock time — they construct `Definitions` and
resolve selections. `test_definitions_resolve` in `tests/test_assets.py` shows the existing pattern
for loading the repository definition in a test (including the `env` fixture it takes).

## Docs to update

1. **`docs/architecture/production-deployment.md`** — a short new section (working title: "Mark
   which assets production needs with a `layer` tag") recording the classification, the selection
   string, and why a tag rather than a group or a kind. This page is where production-side design
   decisions live, and the classification is a statement about the architecture.
2. **`docs/design-philosophy/inherent-stability.md`**, the *R&D fails the other way* section
   (around line 487). It currently forward-references this issue: "the natural mechanism is a
   strict-mode flag on the feature and validation layer, plus asset tagging ([#423])". Rewrite in
   the present tense to say the tag exists and what it is — per CLAUDE.md's "write about the
   present, not the past", the issue reference goes away rather than becoming a history note. The
   strict-mode-flag half of that sentence stays forward-looking, since it is still unbuilt.
3. **`docs/live_service/operations.md`** — one sentence under *Prerequisites*, giving the operator
   the selection string to paste into the Dagster UI. This is the issue's actual payoff, so it
   belongs on the page the operator reads.

No roadmap item completes here, so there is no ship-time triage.

## Verification commands

```bash
uv run ruff check . --fix && uv run ruff format .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus, specific to this change:

```bash
uv run dagster asset list -m nged_substation_forecast.definitions --select 'tag:"layer"="production"'
uv run dagster asset list -m nged_substation_forecast.definitions --select 'tag:"layer"="rnd"'
```

Those two must between them list every asset, with nothing in both and nothing in neither — the
end-to-end check that the tag does what the issue asked, through the real CLI parser rather than
the test's in-process one.

`mkdocs build --strict` is on the list because all three doc edits add cross-page links; read the
rendered HTML for the new `production-deployment.md` section, since the repo's
`mkdocs-authoring` skill documents several ways a page renders wrong while both linters pass.

## Risks and open questions

1. **Is `promoted_model` / `promotable_model_runs` production or R&D?** Recommendation:
   `production`, reasoning above. This is the only judgement call in the change and it is a
   one-word edit either way.
2. **Is `layer` / `production` / `rnd` the right vocabulary?** Recommendation: yes.
   `rnd` is the least-bad spelling Dagster's character set allows; `r-and-d` is the alternative if
   `rnd` reads badly on a UI chip.
3. **Merge collisions.** Low but real: #505 touches `ecmwf_ens` and #486/#488 touch
   `production_assets.py`. Adding one argument to a decorator conflicts only if another session
   edits the same decorator. Worth rebasing on `main` immediately before opening the PR.

## Review findings — first pass (simplicity)

*To be filled in.*

## Review findings — second pass (correctness and testability)

*To be filled in.*
