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

- `AssetSelection.from_string("tag:layer=production")` parses and resolves to exactly the tagged
  assets. The same string is what the Dagster UI's asset-selection box accepts, what the catalog
  page takes as its `?asset-selection=` query parameter, and what `dg launch --assets` /
  `dagster asset list --select` take on the command line.
- The asset detail page renders a **Definition → Tags** panel showing `layer: production`.
- The GraphQL API exposes `tags { key value }` per asset node, so anything built on the UI's API
  can filter on it too.
- `AssetSelection.tag("layer", "production")` is the Python API equivalent, which the test uses.

**`kinds` — rejected.** Dagster implements kinds as reserved `dagster/kind/<name>` tags (confirmed
in the GraphQL output) and renders them as technology badges on the graph node. They are for
naming the tool an asset uses — `python`, `s3`, `xgboost` — and there is a hard limit of three per
asset. Using one for a lifecycle classification would misuse a namespace Dagster owns.

**`group_name` — rejected, but see [open question 2](#risks-and-open-questions).** A group is
one-per-asset and exclusive, and it is Dagster's *primary* structural axis: it lays out the global
lineage graph into labelled boxes and is a column in the asset catalog table (verified — the
catalog's column header reads "Code location / Asset group"; there is no tag column). That makes
groups genuinely better on one axis: the operator sees the split with nothing to type. Tags win on
three others — the issue asks for a tag; tags are additive, so grouping by pipeline stage stays
available later, whereas a group spent on this is spent; and a tag can later be applied to an
asset that is legitimately needed by both layers. The repo uses `group_name` nowhere today.

**Bulk application in `definitions.py` — rejected.** `load_assets_from_modules` accepts
`group_name` but not `tags` (verified: `TypeError: got an unexpected keyword argument 'tags'`);
`dagster.map_asset_specs` would do it in about eight lines, applying one layer per source module.
Rejected because the module boundary is not the classification: `defs/production_assets.py`'s own
docstring says it exists because "`defs/cv_assets.py` is already ~900 lines". Deriving an asset's
layer from which file it happens to sit in means the next split-a-long-file refactor silently
re-classifies assets, and no test can catch it — whereas with per-asset tags a new untagged asset
fails the test in [Tests](#tests). The classification is also worth reading at the asset itself.

### Vocabulary

| | Value |
|---|---|
| Tag key | `layer` |
| Production value | `production` |
| R&D value | `rnd` |

`rnd` is the ASCII spelling of R&D that Dagster's character set permits; `research` and `r-and-d`
are the alternatives. `env` was rejected as a key because both layers run inside the same Dagster
deployment, so `env` would read as the deployment environment.

**Write the selection string unquoted — `tag:layer=production`, not `tag:"layer"="production"`.**
Both parse to the same selection normally, but the quoted form is silently broken whenever Python
warnings are errors, which includes this repo's own pytest run (`filterwarnings = ["error", …]`,
`pyproject.toml:339`). Verified: under `pytest`, `AssetSelection.from_string('tag:"layer"="production"')`
returns `tag:""layer""=""production""` — quotes retained, resolving to **zero** assets — because a
`BetaWarning` raised inside Dagster's ANTLR parse path drops it into a fallback parser that does
not strip quotes. `key:"…"` is unaffected (which is why `tests/test_asset_selection_parses.py`
passes today) and `group:"…"` raises outright. The unquoted form works everywhere.

## What changes, file by file

### New: `src/nged_substation_forecast/defs/_tags.py`

A small private module holding the vocabulary once, so the strings never appear at a decorator:

- `LAYER_TAG_KEY: Final[str] = "layer"`
- `PRODUCTION_LAYER_TAGS` and `RND_LAYER_TAGS` — plain `Final[dict[str, str]]`, ready to splat, so
  a decorator reads `@asset(tags=PRODUCTION_LAYER_TAGS)`.

Sharing one dict across eleven decorators is safe: Dagster normalises the mapping into a fresh
`dict` at definition time (verified — `spec.tags is shared` is `False`, and mutating the source
dict afterwards does not change the asset's tags), so no defensive copy or `MappingProxyType` is
needed.

Its own module rather than a home in one of the three asset modules, because all three import it
and `production_assets.py` already imports `cv_assets.py` — adding more cross-imports between the
asset modules to share two strings is the worse trade. Inside `defs/` rather than beside
`_sentry.py` at the package root, because only `defs/*` consumes it; `dg` does not scan `defs/`
for components here (`[tool.dagster] module_name = "nged_substation_forecast.definitions"` names
the entry point explicitly, and `definitions.py` lists the asset modules by hand).

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

One new test in a new file `tests/test_asset_layer_tags.py` — new rather than added to
`tests/test_assets.py`, because that module is `pytest.mark.integration` and materialises assets
while this is a fast definition-time assertion, and because a new file cannot collide with the
parallel sessions editing the existing test modules.

`test_every_asset_is_classified_as_production_or_rnd` resolves both selections against
`defs.get_repository_def().asset_graph` and asserts:

| Assertion | Why it fails on `main` today |
|---|---|
| `production \| rnd == all asset keys` — every asset carries a legal `layer` | Both selections are empty on `main`, so the union misses all eleven assets |
| `production & rnd == set()` — no asset carries both | (Holds trivially on `main`; it is the guard against a future second tag) |
| `live_forecasts` and `ecmwf_ens` are in `production`; `trained_cv_model` and `metrics` are in `rnd` | Both sets are empty on `main` |

The union assertion is the one with ongoing value: it fails the moment someone adds an asset
without classifying it, and it is a property of every asset rather than a list of names, so it
does not become a merge conflict when a parallel session adds one.

Resolve through `AssetSelection.tag(LAYER_TAG_KEY, …)` rather than `AssetSelection.from_string`,
and add one assertion that the operator-facing string `tag:layer=production` parses to the same
selection — with a comment recording the quoted-form breakage under `filterwarnings = ["error"]`
documented above, so nobody "tidies" the string back into the quoted form.

The test needs no network, no trained model and no wall-clock time. `test_definitions_resolve` in
`tests/test_assets.py` shows the existing pattern for loading the repository definition in a test,
including the `env` fixture it takes.

## Docs to update

1. **`docs/design-philosophy/inherent-stability.md`**, the *R&D fails the other way* section
   (around line 487). It currently forward-references this issue: "the natural mechanism is a
   strict-mode flag on the feature and validation layer, plus asset tagging ([#423])". Rewrite in
   the present tense to say the tag exists and what it is — per CLAUDE.md's "write about the
   present, not the past", the issue reference goes away rather than becoming a history note. The
   strict-mode-flag half of that sentence stays forward-looking, since it is still unbuilt.
2. **`docs/architecture/production-deployment.md`** — a short new section (working title: "Mark
   which assets production needs with a `layer` tag") recording the classification, the two
   selection strings, and one clause on why a tag rather than a group. Keep it to a paragraph plus
   the two lists; this is a decorator-level change, not a control-plane decision, so it does not
   warrant the treatment the longer sections on that page get.
3. **`docs/live_service/operations.md`** — one sentence under *Prerequisites*, giving the operator
   `tag:layer=production` to paste into the Dagster UI. This is the issue's actual payoff, so it
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
uv run dagster asset list -m nged_substation_forecast.definitions --select tag:layer=production
uv run dagster asset list -m nged_substation_forecast.definitions --select tag:layer=rnd
```

Those two must between them list every asset, with nothing in both and nothing in neither — the
end-to-end check that the tag does what the issue asked, through the real CLI parser rather than
the test's in-process one. (`dagster asset list --select` is verified to work on this repo today.)

`mkdocs build --strict` is on the list because all three doc edits add cross-page links; read the
rendered HTML for the new `production-deployment.md` section, since the repo's `mkdocs-authoring`
skill documents several ways a page renders wrong while both linters pass.

## Risks and open questions

1. **Is `promoted_model` / `promotable_model_runs` production or R&D?** Recommendation:
   `production`, reasoning above. This is the only judgement call in the change and it is a
   one-word edit either way.
2. **Tag or group?** Recommendation: tag. A group would show the split in the catalog's "Asset
   group" column and as labelled boxes in the lineage graph, with nothing for the operator to
   type — a real advantage. It is rejected because the issue asks for a tag, because a group is
   exclusive and would spend Dagster's one structural axis on this rather than on pipeline stage,
   and because a tag can later be added to an asset both layers need. Both could be applied
   together for about eight extra lines if Jack wants the visual split as well.
3. **Is `layer` / `production` / `rnd` the right vocabulary?** Recommendation: yes. `rnd` is the
   shortest spelling Dagster's character set allows; `research` and `r-and-d` are the alternatives
   if `rnd` reads badly on a UI chip.
4. **Merge collisions.** Low but real: #505 touches `ecmwf_ens` and #486/#488 touch
   `production_assets.py`. Adding one argument to a decorator conflicts only if another session
   edits the same decorator. Worth rebasing on `main` immediately before opening the PR.

## Review findings — first pass (simplicity)

### Accepted

1. **The `MappingProxyType` justification was factually wrong.** The plan claimed Dagster stores
   the tags mapping by reference and cited a probe as evidence; re-running the probe shows Dagster
   copies it (`spec.tags is shared` → `False`). I had misread my own output. Plain `Final` dicts
   now, and the false claim is gone.
2. **Two tests collapsed into one.** A `production | rnd == all keys` assertion subsumes both the
   "every asset has the tag" test and the "these five are classified correctly" test, while staying
   a property of every asset rather than a name list.
3. **Use the unquoted `tag:layer=production` in docs, tests and verification commands.** The
   reviewer offered this as ergonomics; checking it turned up something worse than that — the
   quoted form silently resolves to zero assets under this repo's own pytest `filterwarnings`
   setting. Written up under [Vocabulary](#vocabulary), with a test comment to stop it regressing.
4. **The constants module moves from `src/nged_substation_forecast/_asset_tags.py` to
   `defs/_tags.py`,** since only `defs/*` consumes it. Kept as its own module rather than inlined,
   because all three asset modules import it.
5. **Cut the "a future job or schedule could select by layer" rationale** — no such job is proposed
   here or in the issue.
6. **`research` added to the vocabulary alternatives**, and the `production-deployment.md` section
   cut down to a paragraph.

### Rejected

1. **Apply the tags in bulk in `definitions.py` via `map_asset_specs`, one layer per source
   module.** Genuinely fewer lines and it would sidestep the merge-collision risk, but it makes an
   asset's layer a function of which file it sits in — and `production_assets.py` exists because
   `cv_assets.py` grew past ~900 lines, not because of the layer split, so the next long-file split
   would silently re-classify assets with no test able to catch it.
2. **Cut the `docs/architecture/` section entirely.** Whether the architecture docs should record
   the classification is a question Jack explicitly asked to be answered; the answer is yes, but
   short. Shrunk rather than cut.
3. **Use `group_name` instead of `tags`.** The reviewer's evidence is good and the point is real,
   so this is promoted to open question 2 for Jack rather than silently dismissed — but the
   recommendation stands, for the reasons recorded there.
4. **Drop `LAYER_TAG_KEY`.** Rejected: the test imports it, so it has a consumer.

## Review findings — second pass (correctness and testability)

*To be filled in.*
