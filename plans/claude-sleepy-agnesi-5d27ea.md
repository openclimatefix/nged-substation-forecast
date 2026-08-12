# Plan — #423: tag Dagster assets with the layer they belong to

Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/423>
Branch: `claude/sleepy-agnesi-5d27ea`

**What is missing.** The Dagster asset graph holds eleven assets and nothing on them says which
ones the live forecasting service needs. Six do — the three ingest assets, model promotion, and
6-hourly inference — and five exist only to compare candidate models on the cross-validation
leaderboard. All eleven sit in one undifferentiated group in the UI, so whoever operates the
service has to already know which is which, and cannot filter the experiment assets out of the
catalog or the lineage graph. The same split is argued at length in the design docs, where
production degrades rather than raising and R&D fails fast, but it is expressed nowhere in the code
— the closest thing is a Sentry failure hook that happens to be attached to the three scheduled
production jobs.

**What the plan does.** Tag every asset with `layer`, valued `production` or `rnd`, so the split
becomes a thing you can query rather than a thing you have to know. The vocabulary is defined once
in a new `defs/_tags.py` and applied as one extra argument on each of the eleven `@asset`
decorators, touching no asset body. An operator then types `tag:layer=production` into the Dagster
UI's selection box — or `dagster asset list --select tag:layer=production` — and sees exactly the
six assets the service runs. One new test asserts that every asset carries exactly one of the two
values, so a future asset added without a layer fails the suite, and it pins the four
classifications that are contestable. Three doc pages record the classification and give the
operator the string. Two decisions are left open for Jack: whether to apply the tags per-decorator
or in bulk in `definitions.py`, and whether the two promotion assets count as production.

## Verdict

**Worth implementing, roughly as described.** The mechanism the issue names (an asset tag) is the
right one, and the payoff is real: an operator can type one selection string into the Dagster UI
and see only the assets the live service runs. It also makes the R&D side of the fail-fast /
fail-operational asymmetry — argued in
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#rd-fails-the-other-way)
but nowhere expressed in code — enumerable in one query. It does not make the *posture* itself
queryable; see [What the tag means](#what-the-tag-means), which the doc wording depends on.

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

## What the tag means

**`layer` records which side of the system needs the asset, not what failure posture its code
has.** The distinction matters because the issue contains both framings: the body asks to see
"which assets are required on production", while the comment calls the tag "the mechanism behind
the fail-fast / fail-operational asymmetry". The two are correlated but not congruent, and they
disagree on exactly the two assets flagged under [Ambiguous cases](#ambiguous-cases):
`promoted_model` and `promotable_model_runs` are needed to operate the service, yet both fail fast
(unguarded `mlflow` calls, an unguarded `read_text` at `production_assets.py:147`).

The "needed by" reading is the one to implement, because it is what the issue body asks for and
what an operator filtering the UI wants. The posture asymmetry stays a *property* of the layers
argued in `inherent-stability.md` — the tag makes the R&D side easy to enumerate, which is the
useful half — but the tag does not itself promise that everything marked `production` degrades
rather than raising. Rule 1 already carves out our-own-bug states from that promise, and the
failure-modes table now carries **three** rows about `promoted_model` refusing a promotion, all in
the *Production* column: the model being empty or unloadable, a saved config this code can no
longer rebuild, and a run id naming no model at all. Two of them spell out that the asset raising
*is* the degrading behaviour — "the outgoing champion stays and keeps forecasting". The doc edit in
[Docs to update](#docs-to-update) has to be worded to match.

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
- `AssetSelection.tag(key="layer", value="production")` is the Python API equivalent, which the
  test uses.

**`kinds` — rejected.** Dagster implements kinds as reserved `dagster/kind/<name>` tags (confirmed
in the GraphQL output) and renders them as technology badges on the graph node. They are for
naming the tool an asset uses — `python`, `s3`, `xgboost` — and there is a hard limit of three per
asset. Using one for a lifecycle classification would misuse a namespace Dagster owns.

**`group_name` — not chosen, but it is genuinely close; see
[open question 3](#risks-and-open-questions).** A group is one-per-asset and exclusive, and it is
Dagster's *primary* structural axis: it lays out the global lineage graph into labelled boxes and
is a column in the asset catalog table (verified — the catalog's column header reads "Code location
/ Asset group"; there is no tag column). That makes groups better on the axis the issue cares most
about: the operator sees the split with nothing to type.

The one argument for tags that survives review is that a group is exclusive and this is not the
only way we will want to slice the graph. The natural grouping for these eleven assets is by
pipeline stage — `ingest` / `cv` / `serving` — and under *that* grouping "what does production
need" is `ingest + serving`, which is a query, not a group. Spending `group_name` on the
production/R&D split forecloses the grouping we are more likely to want, whereas a tag composes
with it. (The plan previously also argued that a tag could later mark an asset needed by *both*
layers; that is dropped, because the test below asserts the two selections are disjoint. It cannot
be both an argument for tags and a thing the tests forbid.)

**Bulk application in `definitions.py`.** `load_assets_from_modules` accepts `group_name` but not
`tags` (verified: `TypeError: got an unexpected keyword argument 'tags'`); `dagster.map_asset_specs`
does it in about six lines, applying one layer per source module, and produces an asset graph
byte-identical to this plan's. This is the biggest live alternative to the whole plan and it is
[open question 1](#risks-and-open-questions) — it is not rejected on the merits here. The plan
continues with per-asset decorators because that is the file surface the work was scoped to, and
because it keeps the classification readable at the asset and testable per-asset.

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

Sharing one dict across six decorators and the other across five is safe: Dagster normalises the
mapping into a fresh `dict` at definition time (verified — `spec.tags is shared` is `False`, and
mutating the source dict afterwards does not change the asset's tags), so no defensive copy or
`MappingProxyType` is needed.

Its own module rather than a home in one of the three asset modules, because all three import it
and `production_assets.py` already imports `cv_assets.py` — adding more cross-imports between the
asset modules to share two strings is the worse trade. Inside `defs/` rather than beside
`_sentry.py` at the package root, because only `defs/*` consumes it; `dg` does not scan `defs/`
for components here (`[tool.dagster] module_name = "nged_substation_forecast.definitions"` names
the entry point explicitly, and `definitions.py` lists the asset modules by hand).

The constants buy one place to change the vocabulary, and nothing more — in particular they do
**not** buy typo-safety, since the test would share any typo they carried. What catches a
mistyped tag is the union assertion in [Tests](#tests), which works just as well against inline
literals. Eleven inline `{"layer": "production"}` dicts would be the honest alternative; the
module wins on reading, not on safety.

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
  about the asset graph, all four asset checks already hang off production assets, and the Sentry hook
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
performs it. `inherent-stability.md`'s failure-modes table also treats `promoted_model` as a
production concern in its own right: it now has three rows for the ways a promotion is refused,
and two of them say the refusal leaves "the outgoing champion … forecasting" — which is
fail-operational reasoning applied to this asset, not R&D fail-fast.

*Against*: both assets need a reachable MLflow tracking server, and the production box has none —
the model arrives there baked into the Docker image, as `live_forecasts`' own `deps` comment
explains. An operator filtering to `layer=production` on the box would see two assets that can
never be materialised there.

I judge the first argument stronger, because the filter's job is "what concerns whoever operates
the service", not "what this particular host can execute" — which is [what the tag
means](#what-the-tag-means). A third value (`promotion`) was considered and rejected: it breaks the
issue's binary for one edge case and gives the operator a third string to remember. **Jack's
call**, and note that answering it the other way also changes how doc item 1 has to be worded.

## Design-philosophy check

This change adds no runtime code path, so it cannot fail open or closed. It adds no asset check.

It names the two sides that rule 9 in
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
— "fail in the direction where being wrong is cheapest to recover from" — is stated in terms of,
and whose *R&D fails the other way* section names this issue as the missing mechanism for. It does
not encode the postures themselves: `layer` says which side needs an asset, and an asset can be
`production` and still fail fast where rule 1 permits it, which is the case for `promoted_model`.
It serves
[H1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/)
(a service that mostly runs itself) by making "what does production actually need" answerable in
the UI rather than by reading three modules.

No design principle is traded away.

## Tests

One new test in a new file `tests/test_asset_layer_tags.py`. The reason for a new file is merge
collisions, not speed: `main` changed `tests/test_assets.py` by 243 lines in the last 65 commits
and more sessions are editing it now, whereas a new file cannot conflict. (The `integration` marker
on that module is *not* a reason — verified, it is declared in `pyproject.toml:355` but nothing
deselects it, so a plain `uv run pytest` collects all 24 of its tests.)

`test_every_asset_is_classified_as_production_or_rnd` resolves both selections against
`defs.get_repository_def().asset_graph` and asserts:

| Assertion | Why it fails on `main` today |
|---|---|
| `production \| rnd == asset_graph.executable_asset_keys` — every asset carries a legal `layer` | Both selections are empty on `main`, so the union misses all eleven assets |
| `production & rnd == set()` — no asset carries both | (Holds trivially on `main`; it is the guard against a future second tag) |
| The four assets whose classification is contestable land where this plan puts them: `promoted_model`, `promotable_model_runs` and `h3_grid_weights` in `production`, `effective_capacity` in `rnd`. Plus `live_forecasts` and `trained_cv_model` as the two obvious anchors | Both sets are empty on `main` |

The union assertion is the one with ongoing value: it fails the moment someone adds an asset
without classifying it, and it is a property of every asset rather than a list of names, so it
does not become a merge conflict when a parallel session adds one.

**The spot-checks must name the contestable assets, not the obvious ones.** Union and disjointness
are value-agnostic — they demand that every asset carry exactly one of the two tags, not *which*.
So a spot-check list of `live_forecasts` / `ecmwf_ens` / `trained_cv_model` / `metrics` leaves the
suite green if `promoted_model` flips to `rnd`, which is precisely the decision
[open question 2](#risks-and-open-questions) escalates. Whatever Jack answers has to be pinned by
an assertion, so that changing it later is a deliberate edit to a test rather than a silent
one-word change. `h3_grid_weights` (production but unscheduled) and `effective_capacity` (the
closest call on the R&D side) belong in the list for the same reason.

**Compare against `executable_asset_keys`, not `get_all_asset_keys()`.** The two are identical
today (eleven keys, no source assets), but a typo in any `deps=["…"]` string silently creates an
external asset key, which lands in `get_all_asset_keys()` and can never carry a tag. Verified: a
`deps=["typo_upstream"]` probe puts `typo_upstream` in `get_all_asset_keys()` and not in
`executable_asset_keys`. Using the former would make an unrelated dep typo fail this test with a
message about layer tags — the exact false attribution `test_definitions_resolve`'s docstring
already warns about, and which that test is the right place to catch.

Resolve through `AssetSelection.tag(key=LAYER_TAG_KEY, value=…)` rather than
`AssetSelection.from_string` — by keyword, per the *Calling functions* rule that landed in
`docs/architecture/code-style.md` on 2026-08-11 —
and add one assertion that the operator-facing string `tag:layer=production` parses to the same
selection — with a comment recording the quoted-form breakage under `filterwarnings = ["error"]`
documented above, so nobody "tidies" the string back into the quoted form. That comment should say
what the assertion does *not* cover: under `filterwarnings = ["error"]` every `tag:` string takes
the fallback parser, so the ANTLR parser the UI and CLI actually use is never exercised in CI —
the two `dagster asset list` commands below are the only check of it, and they are manual.

The test needs no network, no trained model, no wall-clock time and **no fixture**. It does not
need the `env` fixture that `test_definitions_resolve` takes: that fixture is module-local to
`tests/test_assets.py:135` and would be `fixture 'env' not found` in a new file, and building
`Definitions` never touches a data path anyway. Importing `nged_substation_forecast.definitions`
at test time is safe because the root `conftest.py`'s `pytest_configure` forces `SENTRY_DSN=""`
before collection, so the import-time `init_sentry` is a no-op.

## Docs to update

1. **`docs/design-philosophy/inherent-stability.md`**, the *R&D fails the other way* section
   (around line 497). It currently forward-references this issue: "the natural mechanism is a
   strict-mode flag on the feature and validation layer, plus asset tagging ([#423])". Rewrite in
   the present tense to say the tag exists and what it is — per CLAUDE.md's "write about the
   present, not the past", the issue reference goes away rather than becoming a history note. The
   strict-mode-flag half of that sentence stays forward-looking, since it is still unbuilt.

   **Word it as "records which layer needs the asset", and leave the posture claim attached to the
   sentence about the Sentry hook.** Saying the tag *is* the fail-fast/fail-operational mechanism
   would make that section promise something the tag does not deliver — see
   [What the tag means](#what-the-tag-means).
2. **`docs/architecture/overview.md`** — two or three sentences on the *Orchestration* bullet under
   *Core Components*, naming the `layer` tag, the two selection strings, and which assets fall each
   side. Jack asked whether `docs/architecture/` should record the classification; the answer is
   yes, and this is the page for it — `overview.md` describes what is built, in bullets of exactly
   this size.

   **Not a new section in `production-deployment.md`.** That page's sections are control-plane
   design decisions — running the control plane on a VM, baking the model into the image, promoting
   via an asset — and by this plan's own words the change is "a decorator-level change, not a
   control-plane decision". A new top-level section there would be the wrong weight; the
   tag-versus-group reasoning goes in a one-line comment above the constants in `defs/_tags.py`,
   where an editor tempted to change it will be standing.
3. **`docs/live_service/operations.md`** — a sentence under *Prerequisites*, giving the operator
   `tag:layer=production` to paste into the Dagster UI. This is the issue's actual payoff, so it
   belongs on the page the operator reads.

   **It must say what the selection includes, not just the string.** Six assets match, and two of
   them — `promoted_model` and `promotable_model_runs` — only ever run on a laptop: the same page
   already says at line 50 that "Promotion (this step and the next) always happens **on your
   laptop**, whichever environment serves the forecasts", and
   `docs/design-philosophy/design-principles.md:555` says the production box "has no MLflow and
   never runs promotion". A bare selection string would tell an operator on the AWS box that six
   assets are production when four of them are what that box runs. This is the same tension as
   [open question 2](#risks-and-open-questions), surfacing in the docs.
4. **`tests/test_asset_selection_parses.py`** — its module docstring says "nothing in the test
   suite or the Dagster UI parses a selection string — only the CLI does". This change makes the
   first half false (the new test parses one). The second half is *already* false: the webserver
   bundle ships its own ANTLR asset-selection grammar with a `TagAttributeExpr` rule, which is what
   makes the UI selection box work at all. One-line rewrite, in the same commit.

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

`mkdocs build --strict` is on the list because the doc edits add cross-page links; read the
rendered HTML for the new `overview.md` bullet, since the repo's `mkdocs-authoring`
skill documents several ways a page renders wrong while both linters pass.

## Risks and open questions

1. **Eleven decorators, or six lines in `definitions.py`?** The simplicity review proposed applying
   the layer in bulk, one per source module, via `map_asset_specs` — and built it, confirming the
   resulting asset graph is identical to this plan's on every axis it checked (asset keys, parents,
   checks, partitions, kinds), with `dagster asset list --select` returning the same 6/5 split.
   That version deletes `defs/_tags.py`, all eleven decorator edits, three imports, and open
   question 5 below.

   My first-pass reason for rejecting it does not hold, and I have withdrawn it. I argued the
   module boundary is incidental, citing `production_assets.py`'s "New file (`defs/cv_assets.py` is
   already ~900 lines)" — but that parenthetical is the history note this plan already flags for
   deletion, and the docstrings' actual first lines read "The ingestion Dagster assets",
   "**Production** Dagster assets" and "**Cross-validation** Dagster assets". The modules *are*
   named for the layers. Nor is a future misfiling silent: a new module has to be named in one of
   the two `load_assets_from_modules` calls, on the line that says which layer it is.

   **Recommendation: keep the decorators**, on two grounds that survive. A new asset added to an
   *existing* module inherits that module's layer with nothing to notice; under decorators it has
   no tag and the test fails. And answering open question 2 the other way for one asset is a
   one-word edit here, versus an exception carved out of a bulk rule there. But this is close, the
   bulk version is genuinely simpler, and the decorator surface is what the work was scoped to —
   so **Jack's call**, and the answer changes roughly half this plan.
2. **Is `promoted_model` / `promotable_model_runs` production or R&D?** Recommendation:
   `production`, reasoning above. A one-word edit either way, and it changes the wording of doc
   edit 1.
3. **Tag or group?** Recommendation: tag, but this survived two reviews as the weakest call in the
   plan. A group shows the split in the catalog's "Asset group" column and as labelled boxes in the
   lineage graph with nothing for the operator to type, which is closer to what the issue asks for,
   and the repo spends `group_name` on nothing today. The argument that keeps me on tags is that
   `ingest` / `cv` / `serving` is the grouping we are more likely to want, and under it "what
   production needs" is a two-group query rather than a group. Both mechanisms can be applied
   together for a few extra lines if Jack wants the visual split now.
4. **Is `layer` / `production` / `rnd` the right vocabulary?** Recommendation: yes. `rnd` is the
   shortest spelling Dagster's character set allows; `research` and `r-and-d` are the alternatives
   if `rnd` reads badly on a UI chip.
5. **Merge collisions.** Low but real: #505 touches `ecmwf_ens` and #488 touches
   `production_assets.py` (#486 has landed). Adding one argument to a decorator conflicts only if
   another session edits the same decorator. Worth merging `main` immediately before opening the
   PR. Answering open question 1 the other way removes this risk entirely.

## Re-checked against `main` at 92feca64

The branch has been merged up twice while this plan was being written — 65 commits, then a further
64. Nothing in either range changes the plan's substance:

- **Still exactly eleven assets**, same names, same three modules, and `get_all_asset_keys()` still
  equals `executable_asset_keys`. Still exactly four asset checks, all of them hanging off
  production assets (`ecmwf_ens` ×2, `live_forecasts`, `power_time_series_and_metadata`). Only line
  numbers moved, and the plan's citations match the current tree.
- **Every empirical claim re-verified on the merged tree** against the same Dagster 1.13.17,
  including the quoted-selection-string breakage under `pytest`, and `tests/test_assets.py` still
  collecting 24 tests in a plain run.
- **#486 and #549 both landed**, and together they strengthen the `production` classification of
  `promoted_model` rather than weakening it. Promotion now refuses a model whose saved config this
  code cannot rebuild, and one whose run holds no model at all — in both cases before replacing the
  model on disk. The failure-modes table records all three refusal paths in its *Production*
  column, twice noting that the outgoing champion keeps forecasting. That is fail-operational
  reasoning applied to this asset.
- **#505 and #488 have not landed**, so the merge-collision risk on `ecmwf_ens` and
  `production_assets.py` still stands.
- **A new house rule landed** — *Calling functions* in `docs/architecture/code-style.md`: pass
  arguments by keyword wherever the callee allows. The test spec above now says
  `AssetSelection.tag(key=…, value=…)`. The decorator change is unaffected: `tags=` is already a
  keyword.

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
   cut down to a paragraph. *(Superseded: the second simplicity pass moved the architecture record
   to `docs/architecture/overview.md` instead.)*

### Rejected

1. **Apply the tags in bulk in `definitions.py`, one layer per source module.** Rejected on the
   first pass; **that rejection has since been withdrawn** — see the second simplicity pass below
   and open question 1.
2. **Cut the `docs/architecture/` record entirely.** Whether the architecture docs should record
   the classification is a question Jack explicitly asked to be answered; the answer is yes, but
   short. Shrunk rather than cut.
3. **Use `group_name` instead of `tags`.** The reviewer's evidence is good and the point is real,
   so this is promoted to an open question for Jack rather than silently dismissed.
4. **Drop `LAYER_TAG_KEY`.** Rejected: the test imports it, so it has a consumer.

## Review findings — simplicity, second pass

The `plan-issue` skill changed after the first pass: the simplicity reviewer is now told it is not
confined to the plan's scope and may propose a different architecture outright. A fresh reviewer
ran under that brief against the merged tree, and it landed harder than the first.

### Accepted

1. **My rejection of bulk application was built on the wrong evidence, and is withdrawn.** I had
   cited `production_assets.py`'s "New file (…already ~900 lines)" as proof the module split is
   incidental — while elsewhere in this same plan flagging that sentence as a history note that
   should be deleted. The docstrings' actual first lines name the layers. The proposal is now open
   question 1, with its case made and a recommendation, rather than a rejected finding.
2. **The "a tag could later mark an asset needed by both layers" argument is self-contradicting**
   and has been cut: the plan's own test asserts the two selections are disjoint. The tags-over-
   groups case now rests on one argument, stated where it can be judged.
3. **`_tags.py` does not buy typo-safety.** The constants and the test would share any typo. What
   catches a mistyped tag is the union assertion. Justification corrected; the module stays, on
   readability alone, with the inline alternative named.
4. **The new test file's justification was wrong.** `integration` is declared at
   `pyproject.toml:355` but nothing deselects it — a plain `uv run pytest` collects all 24 tests in
   `tests/test_assets.py`. The real reason for a separate file is merge collisions, and that is now
   what the plan says.
5. **No new section in `production-deployment.md`.** The reviewer quoted my own sentence back at me
   — "a decorator-level change, not a control-plane decision" — which is exactly the test that page
   fails. The architecture record moves to a bullet in `docs/architecture/overview.md`, which still
   answers Jack's question and at the right weight.

### Rejected

1. **Cut the selection-string round-trip assertion as duplicating
   `tests/test_asset_selection_parses.py`.** That test guards the ANTLR *runtime version*; this one
   pins the unquoted form so nobody tidies it back into the quoted form the docs must not use.
   Different failure, one line.
2. **Fold the test into `test_definitions_resolve`.** The marker argument was wrong, but the
   collision argument stands on its own — `main` has changed that file by 243 lines since this
   branch started.
3. **Drop "What the tag means" and the ambiguity escalation as answering a question the issue did
   not ask.** The two framings really do disagree, and the disagreement changes how the
   `inherent-stability.md` sentence must be worded — a finding the correctness reviewer raised
   independently. Jack also asked for genuinely ambiguous assets to be flagged rather than guessed
   at. Kept, and the section is short.

## Review findings — second pass (correctness and testability)

The reviewer re-ran every empirical claim in the plan and all of them held, including the
quoted-selection-string breakage. It also applied the eleven tags exactly as specified and ran the
suite — 110 passed, `test_definitions_resolve` unaffected — and confirmed the eleven asset names
and the 3 / 5 / 3 module split are right.

### Accepted

1. **The test must not take the `env` fixture.** That fixture is module-local to
   `tests/test_assets.py:135`, so a new file asking for it errors with `fixture 'env' not found`.
   It is also unnecessary — building `Definitions` touches no data path, and the root
   `conftest.py` already forces `SENTRY_DSN=""` before collection.
2. **Compare the union against `executable_asset_keys`, not `get_all_asset_keys()`.** A typo in any
   `deps=["…"]` string silently creates an external asset key that lands in the latter and can
   never carry a tag, so the layer test would fail on an unrelated fault, in the exact way
   `test_definitions_resolve`'s docstring warns about.
3. **The tag's meaning had to be pinned down before the doc edit.** The issue body ("which assets
   are required on production") and the issue comment ("the mechanism behind the fail-fast /
   fail-operational asymmetry") are two different definitions, and they disagree on
   `promoted_model`. New [What the tag means](#what-the-tag-means) section settles it on the body's
   reading, and doc item 1 is now worded so the rewritten passage does not promise a posture
   guarantee the tag cannot deliver.
4. **The test's round-trip assertion was over-claimed.** Under `filterwarnings = ["error"]` every
   `tag:` string takes the fallback parser, so CI never exercises the ANTLR parser the UI and CLI
   use. Kept the assertion (it pins the fallback, which is what the comment is about) and recorded
   the gap.
5. **The new architecture section must land before line 378**, ahead of *Considered but rejected
   designs*. *(Superseded: there is no longer a `production-deployment.md` section — the record
   moved to `docs/architecture/overview.md`, which has no such heading.)*
6. **"Sharing one dict across eleven decorators"** — it is two dicts, across six and five.

### Rejected

Nothing. The reviewer raised no finding I could not reproduce.

### Noted, not acted on

`production_assets.py`'s module docstring is itself written as a history note ("New file
(`defs/cv_assets.py` is already ~900 lines)"), which the repo's "write about the present" rule
forbids. Out of scope for this issue — worth a separate one-line fix some time.

## Review findings — correctness, second pass

Re-run against the merged tree after the rescoped simplicity review, because the plan had changed
substantially. Every empirical claim and every line citation re-verified and held; the reviewer
also wrote the test, ran it red on the current tree, applied the eleven tags, and ran the full
suite green (607 passed). Five real defects, all accepted.

1. **Two accepted findings from the first correctness pass had gone stale** — they still described
   a `production-deployment.md` section that the later simplicity pass moved to `overview.md`. Both
   are now marked superseded, so an implementer reading the findings lists cannot edit the wrong
   page.
2. **The Verdict and the *Design-philosophy check* contradicted *What the tag means*.** Both said
   the tag *is* the queryable form of the fail-fast asymmetry; the later section says it is not, and
   the doc wording depends on that. All three now agree: the tag makes the two *sides* enumerable,
   not the postures.
3. **The test did not pin any of the contestable classifications.** Union and disjointness are
   value-agnostic, and the spot-check list named only obvious assets — so flipping `promoted_model`
   to `rnd`, the very decision escalated to Jack, left the suite green. The spot-checks now name
   `promoted_model`, `promotable_model_runs`, `h3_grid_weights` and `effective_capacity`.
4. **Doc edit 3 would have misled an operator on the AWS box**, telling them six assets are
   production when two of them only ever run on a laptop — a tension the plan had already recorded
   under *Ambiguous cases* and then failed to carry into the doc wording.
5. **`tests/test_asset_selection_parses.py`'s docstring becomes false** ("nothing in the test suite
   or the Dagster UI parses a selection string"). Added as doc edit 4. Its second clause is already
   false — the webserver bundle ships its own ANTLR grammar with a `TagAttributeExpr` rule.

### Rejected

Nothing.

### Noted

"Both existing checks are already production-only" undercounts: there are four check keys, the two
standalone ones plus `ecmwf_ens`'s two `check_specs`. All four hang off production assets, so the
conclusion is unchanged and the sentence is merely imprecise. Separately, the reviewer saw one
intermittent failure of `tests/test_metrics.py::test_metrics_no_filter_scores_every_group` that did
not reproduce; this change adds no runtime code, so do not attribute a single red run to it.
