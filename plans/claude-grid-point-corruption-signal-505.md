# Plan — Move the upstream ECMWF-corruption signal to grid-point level (#505)

Branch: `claude/grid-point-corruption-signal-505`.
Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/505>.

**The problem.** We answer the question "is Dynamical's ECMWF feed broken?" with the number of null
H3 cells in an ingested run, published as the `nwp_has_no_unexpected_nulls` asset check. That number
has just gone nearly blind. Since #496 landed, each H3 cell is renormalised over
the grid points that actually supplied a value, so a corrupt grid point costs only its own share of
its cell instead of nulling the whole thing — which is the entire point of #496, but it means the
upstream corruption is absorbed before it reaches anything we count. On the production grid, upstream
scatter now has to get roughly 230× worse before the check reports what it reported before. The
corruption itself is unchanged; only our ability to see it has gone.

**The plan.** Count the nulls where they live — on the raw 0.25° grid, before the H3 aggregation
touches it. A new pure function, `assess_upstream_grid_point_nulls`, makes one pass over the
downloaded `xr.Dataset` and returns four scalars: how many grid points were null beyond lead-0, out
of how many, across how many `(variable, member, step)` slices, and which variables. Their ratio,
`null_nwp_grid_point_fraction`, is published on the same WARN check and on every materialisation, so
the trend is plottable in the Dagster timeline rather than only visible on runs bad enough to warn.
The cell count stays, because "how much did the model lose" and "how broken is the feed" are
different questions — and every key on the check is renamed to carry the population it counts
(`n_null_h3_cells` beside `n_null_nwp_grid_points`), so an operator reading the Dagster UI cannot
mistake one for the other. The check stays WARN and non-blocking, and the new call sits inside the
guard that already stops a warning path failing the run.

## Verdict

**Worth implementing, roughly as described.** The issue's premise checks out against the code on
`main`:

- `assess_nwp_quality` (`packages/contracts/src/contracts/weather_schemas.py:587`) counts nulls in
  the *stored H3 cells*, via `_deaccumulated_null_breakdown` at line 492. Its input is a validated
  `pt.DataFrame[Nwp]`, which by construction is post-aggregation, so it cannot see a grid point.
- #496 has landed. `_aggregate_grid_points_to_h3_cells`
  (`packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py:135`) renormalises each
  numeric variable over its own contributing weight, and the `> 0` guard at line 211 makes a cell
  with no contributing point null rather than `0.0` — so a cell is null only when *every*
  contributing grid point is null. The amplification that made `n_null_cells` a usable provider
  proxy is gone by design.
- The code already admits the gap in prose: `NwpQualityReport`'s docstring
  (`weather_schemas.py:532`) says "Read this as 'how much did we lose', not as 'how corrupt was the
  feed'", and `docs/live_service/operations.md:243` tells the operator the same thing. Both point at
  this issue. So the change closes a hole the codebase has already documented.

The provider channel in
[Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#three-audiences-three-channels)
wants a number the provider would act on. The grid-point null rate is that number; the cell count is
that rate convolved with our H3 resolution, our grid spacing and our aggregation policy.

## Departures from the issue body

1. **Measure with a separate pass over the downloaded `xr.Dataset`, not by changing the converter's
   return type.** The issue names both options. The deciding fact is that counting inside the
   converter puts the counting *outside* the guard that stops a warning path failing the run.
   See [Why a separate pass](#why-a-separate-pass).

2. **Report scalars, not a per-slice breakdown frame.** The issue asks for
   `(variable, ensemble_member, valid_time)` granularity, "to keep the whole-slice-versus-scatter
   distinction that report already draws". The distinction is worth keeping; a second breakdown frame
   is not the way to get it, because the cell-level report already draws it. If a slice is null at
   every grid point then every cell in that slice has zero contributing weight, so every cell comes
   out null, and the cell-level `n_whole_null_h3_slices` names it exactly. So the upstream report carries
   a *count* of affected slices — which separates "one bad slice" from "a hundred", and is not
   derivable from the cell-level numbers once #496 absorbs the scatter — and leaves the naming of
   wholly-null slices to the report that already does it well.

   The two counts are not strictly redundant in both directions: a grid point the H3 weights name but
   the dataset does not carry nulls cells with no upstream null behind them at all
   (`docs/architecture/ecmwf-ens-known-issues.md:262`). That divergence runs the other way and does
   not weaken the argument, but the plan should not claim the two are equivalent.

   What this gives up: naming *which* slices carried scatter the aggregation then absorbed. Nothing
   acts on that today, and the provider question ("is your feed broken, and since when?") is answered
   by a rate on a timeline.

3. **The new number does not drive the check's `passed`.** It is published as metadata and read as
   a trend. See [Metadata, not a gate](#metadata-not-a-gate).

4. **No second metadata table.** `_nwp_quality_check_result` already writes a capped affected-slices
   table for the cell-level report. The upstream signal is a rate; a second ~100-row table on every
   materialisation buys little for its event-log weight.

## What changes, file by file

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/upstream_nulls.py` (new module)

**`UpstreamNullRate`** — a frozen dataclass, four fields and two properties, no Polars frame:

- `n_null_nwp_grid_points: int` — null grid points in the counted variables, beyond lead-0.
- `n_total_nwp_grid_points: int` — the denominator: every counted variable × ensemble member ×
  forecast step beyond lead-0 × grid point in the downloaded box.
- `n_affected_nwp_slices: int` — `(variable, ensemble_member, lead_time)` slices with ≥1 null
  grid point.
- `affected_nwp_variables: tuple[str, ...]` — sorted.
- `null_nwp_grid_point_fraction: float` — **returns `0.0` when `n_total_nwp_grid_points == 0`**, with a
  comment saying why: a run carrying no step beyond lead-0 has nothing to measure, and a
  `ZeroDivisionError` here would be a warning path failing a run. This is reachable, not defensive —
  see [the zero-denominator case](#the-zero-denominator-case).
- `is_healthy: bool`.

Its docstring must say plainly what it measures and over what population, because that is the point
of Jack's comment on the issue: nulls on the raw 0.25° grid Dynamical sent, before any H3
aggregation, over the downloaded lat/lon box, beyond lead-0.

**`assess_upstream_grid_point_nulls(ds: xr.Dataset, variables: Collection[str]) -> UpstreamNullRate`**
— pure and Dagster-free, mirroring `assess_nwp_quality`'s shape. Select the steps beyond lead-0, then
for each name in `sorted(variables)` reduce `isnull()` over `latitude`/`longitude` to give a null
count per `(ensemble_member, lead_time)`; sum those for `n_null_nwp_grid_points`, count the non-zero
ones for `n_affected_nwp_slices`, and take `n_total_nwp_grid_points` from the selected array's
`.size`.

`variables` is a parameter rather than a hard-coded `Nwp.deaccumulated_var_names` purely so the
[follow-on check](#follow-on-work) for the instantaneous variables can reuse this function unchanged.
The asset passes `Nwp.deaccumulated_var_names`. This is the one piece of generality the plan builds
for a caller that does not exist yet, and it is justified because that caller is already agreed and
scheduled, not hypothetical.

**Name the lead-0 predicate explicitly: `ds.lead_time > np.timedelta64(0)`.** Not "skip the first
chunk", which is wrong for a run whose steps do not start at 0, and not an integer comparison, which
would silently include lead-0 and put `3 × n_lat × n_lon × n_members` guaranteed nulls into the
numerator. `lead_time` is a `timedelta64` in both the fixture (`[ns]`,
`packages/dynamical_data/tests/conftest.py:78`) and the committed real slice (`[us]`), and
`np.timedelta64(0)` compares correctly across units.

Excluding lead-0 is not optional: the de-accumulated variables are null there by design in every run,
everywhere.

### `docs/api/dynamical_data/index.md`

Add `::: dynamical_data.ecmwf_ens.upstream_nulls` alongside `download` and `convert_to_polars`.

### `packages/contracts/src/contracts/weather_schemas.py`

Docstring only, no code. `NwpQualityReport`'s "measuring that rate where it lives is tracked in #505"
sentence must go — it would otherwise point at a closed issue — replaced by a pointer to
`UpstreamNullRate` as the sibling number answering the provider question.

No new type and no shared base class here. The two reports share no code once the upstream one is
scalars, and `contracts` is the home of Patito data schemas; a report describing an
`xr.Dataset`-derived count belongs next to the code that computes it, as `PowerFreshnessResult`,
`MissedNwpRuns` and `LiveForecastHealthResult` already do in `defs/checks.py`.

### `src/nged_substation_forecast/defs/assets.py`

- Call `assess_upstream_grid_point_nulls(ds, Nwp.deaccumulated_var_names)` **inside the existing
  `try` block** at line 308 that
  already wraps `assess_nwp_quality` and `assess_nwp_run_completeness`. The guard, the ordering
  before `write_nwp`, and `_degraded_nwp_check_result` all already exist and the new call inherits
  them unchanged. Positional `ds` is right here under
  [Calling functions](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/#calling-functions)'
  third exception — one argument whose role the function name states — and matches the sibling
  `assess_nwp_quality(nwp)` on the line above.
- `_nwp_quality_check_result` gains an `upstream: UpstreamNullRate` parameter. Its call site becomes
  `_nwp_quality_check_result(report=quality, upstream=upstream)` — two arguments now, so the
  keyword-argument rule applies and the existing positional call must change with it.
- **Rename every metadata key on this check so it names the population it counts**, per
  [Naming the two populations](#naming-the-two-populations). Six of the eleven are renames of
  existing keys, which is a deliberate breaking change to the check's metadata; nothing consumes
  these keys programmatically (no training filter, no metric, no alert — `operations.md` says so
  explicitly), so the cost is the docs and tests listed below.

  | H3 cells, after aggregation | Raw NWP grid, before aggregation |
  |---|---|
  | `n_null_h3_cells` *(was `n_null_cells`)* | `n_null_nwp_grid_points` |
  | — | `n_total_nwp_grid_points` |
  | — | `null_nwp_grid_point_fraction` |
  | `n_affected_h3_slices` *(was `n_affected_slices`)* | `n_affected_nwp_slices` |
  | `affected_h3_variables` *(was `affected_variables`)* | `affected_nwp_variables` |
  | `n_whole_null_h3_slices`, `n_scattered_h3_slices` *(renames)* | — |
  | `affected_h3_slices` *(was `affected_slices`, the table)* | — |

  `affected_nwp_variables` is load-bearing, not decoration: after #496 the H3 variable list is
  exactly the one that goes blind, so a run whose `precipitation_surface` scatter is fully absorbed
  publishes `affected_h3_variables: []` while the NWP list names the variable. Which variable is
  corrupt is the first thing in a mail to Dynamical.

  No `n_whole_null_nwp_slices` (it would duplicate `n_whole_null_h3_slices`, departure 2) and no
  `n_scattered_nwp_slices` (the difference of two keys already present).

  The `TableColumn` names *inside* the `affected_h3_slices` table (`_NWP_NULL_SLICES_SCHEMA`,
  `n_null_cells` / `n_total_cells`) stay as they are: the table's own key already names the
  population, so repeating it in every column would be noise. Worth stating so the implementer does
  not churn on it.
- **Give the check a `description` on its `AssetCheckSpec`.** `AssetCheckSpec` accepts one and
  neither NWP check sets it today, so the Checks view in the Dagster UI shows no standing
  explanation of what the check means — only the per-run `AssetCheckResult.description`. This is
  where the two populations get explained once, permanently, to whoever is reading the UI: that the
  `nwp_` keys count grid points on the raw 0.25° grid Dynamical sent, that the `h3_` keys count the
  cells we store after area-weighted aggregation, that the first answers "is the feed broken" and
  the second "how much did we lose", and that the two are not comparable as rates. The `ecmwf_ens`
  asset docstring — which the UI shows as the asset description — gains one sentence pointing at it.
- **Rewrite the per-run description to always name both levels**, one clause each, with no case
  analysis. Today it reads "No unexpected nulls in the de-accumulated NWP variables." whenever no
  cell is null (`assets.py:413`) — which after #496 is the *usual* state even when upstream sent a
  corrupt run, so the string claims health it did not measure. Always emitting both numbers removes
  the failure mode rather than adding a branch to keep in sync with `passed`.
- Add `null_nwp_grid_point_fraction` and `n_null_nwp_grid_points` to the materialisation metadata
  next to `**shape_metadata`, so the trend is plottable in the Dagster asset timeline on every run
  and not only the ones that warn — following `_nwp_run_shape_metadata`'s precedent. Like the shape
  keys they are absent from a materialisation whose assessment raised, since they come from the same
  guarded block; the operations runbook already documents that absence and needs one sentence
  extending it.

### Naming the two populations

**Dagster has no grouping primitive for metadata**, so the key string is the only lever.
`AssetCheckResult.metadata` and `MaterializeResult.metadata` are both a flat
`dict[str, MetadataValue]` — verified against the pinned Dagster 1.13.17 — with no nesting, no
sections and no display ordering we control. A `MetadataValue.table` *could* group the two
populations into one two-row table, but a table cannot be plotted on the asset timeline, and the
trend plot is the point of publishing the fraction at all.

So every key carries its population in its own name: `nwp_grid_point` for the raw 0.25° grid
Dynamical sent us, `h3_cell` for what we store after aggregation. Two properties fall out, both
worth having:

- **A key is unambiguous read alone**, in a Dagster event-log row, a screenshot or a runbook, with
  no legend and no memory of which check emitted it.
- **The comparable pair sorts adjacent.** `n_null_h3_cells` and `n_null_nwp_grid_points` land next
  to each other alphabetically, as do `affected_h3_variables` and `affected_nwp_variables` — so if
  the UI sorts keys, the reader gets the comparison for free, and if it does not, nothing is lost.

The names are deliberately longer than `n_null_cells`. That is the trade: this check exists to be
read by an operator deciding whether to mail Dynamical, and a wrong reading of which population a
number describes is exactly the failure this issue was filed about.

### Why a separate pass

Both options appear in the issue. Counting inside the converter looked attractive — the counts are a
byproduct of a loop it already runs, on a frame (`nwp_df`, line 123) that is exactly one row per
downloaded grid point with no H3 geometry in it. One fact decides it:

**Counting inside the converter puts the counting outside the guard.** The converter is called in the
earlier `try` at `assets.py:287`, which catches only `NwpRunNotYetAvailable` and
`NwpVariableWhollyMissing` — so a raise anywhere in it costs the partition. Everything else in this
plan is a warning path, and a warning path that can fail the run it is warning about is the rule-7
hazard #509 was filed about. The zero-denominator case below shows it is not theoretical: the obvious
implementation turns a lead-0-only run from "lands and WARNs" into a hard failure. Guarding it inside
the converter would mean wrapping library code in a `try` that exists solely for the asset's benefit,
which puts the guard in the wrong place. The separate pass sits inside the guard that already exists,
and inherits it for free.

The converter option would also touch 28 call sites against the separate pass's five. **That is not
why**, and it would not change the conclusion if the numbers were reversed — mechanical test edits
are cheap and the guard is not. It is recorded only so the implementer is not surprised into
reopening the question when they see how small the converter diff looks.

What the separate pass costs: the five download stubs in `tests/test_assets.py` (lines 378, 434, 480,
533, 607) return a bare `object()`, so they need a small synthetic `xr.Dataset` instead — roughly a
dozen lines, once, at module level in that file, since root `tests/` cannot import
`packages/dynamical_data/tests/conftest.py`. And it reads `ds`'s `latitude`/`longitude` dimension
names and `lead_time`'s dtype, which `convert_to_polars.py` and `download.py` already assume
throughout.

### The zero-denominator case

`n_total_nwp_grid_points` is zero for any run carrying no step beyond lead-0, and that is reachable in
this repo today: the committed real ECMWF slice
(`packages/dynamical_data/tests/data/ecmwf_ens_real_slice.nc`, converted at
`test_ecmwf_ens_cached.py:83`) has `lead_time == [0]`, and 18 datasets in `test_convert_to_polars.py`
are built with `lead_time_hours=(0,)`. In production it is the shape of a partial upstream
publication — exactly what `_ECMWF_ENS_MAX_RETRIES` exists for. An unguarded division would turn a
run that lands and WARNs today into a hard partition failure, which is the rule-7 inversion this
whole workstream exists to prevent. Hence the explicit `0.0`, and test 6.

### Metadata, not a gate

**Decided: the fraction is published as metadata and does not drive `passed`.** Two reasons:

1. `passed` answers "did this run land damaged", which is the cell-level question. The upstream rate
   answers "is the feed getting worse", which is a question about a *sequence* of runs and belongs on
   a timeline, not in a boolean.
2. Shipping the measurement first is what makes a future gate designable on measured rather than
   inferred numbers. Escalating a badly-degraded run is already tracked in
   [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501), and this metric is
   the input it needs.

One argument deliberately *not* used, because it is false and the plan should not rest on it: that a
zero threshold would warn constantly because the grid-point base rate is unknown. The archive says
otherwise. `docs/architecture/ecmwf-ens-known-issues.md:98` gives per-grid-point rates directly
(0.014% of `precipitation_surface`'s grid points on 2025-06-04 00Z, the worst run in the archive),
and lines 111–113 record that only 12 of 862 archived runs carry any de-accumulated null beyond
lead-0 at all. Since the pre-#496 aggregation let one null point null its cell, that 12/862 ≈ 1.4%
bounds the runs carrying any null among the grid points our cells use, so a zero-threshold gate would
fire on order 1% of runs. Gating is therefore affordable; we are declining it because it answers the
wrong question on this check, not because it would be noisy.

The issue's constraint holds: one check, still `WARN`, still `blocking=False`.

## Design-philosophy check

- **Rule 6**: the check keeps `AssetCheckSeverity.WARN` and `blocking=False`. No `ERROR`-severity
  check is added.
- **Rule 7**: the new call sits inside the `except BaseException` guard added for #509, which turns
  any failure into two degraded WARN results plus a Sentry event and lets the run land — and the
  guard sits before `write_nwp`, so a raise cannot duplicate rows either. The one arithmetic hazard
  that would otherwise live *outside* any guard is the zero denominator, handled explicitly above.
- **Rule 2** (liberal about missing, strict about malformed) is untouched: `Nwp.validate` remains the
  only gate, and this change adds no rejection path.
- **H1 / T1.1.** This serves the operability claim directly. T1.1 counts interventions *by cause*,
  and the intervention log can only attribute a cause it can see; today a worsening Dynamical feed is
  invisible until it is bad enough to null whole cells.
- **No principle is traded away.** Nothing is fabricated or imputed, so
  [principle 9](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#9-provenance-travels-with-the-forecast-data)
  is unaffected; the change adds a measurement and no new failure path.

## Tests

Offline in every case — no network, no wall clock, no trained model.

**`packages/dynamical_data/tests/test_upstream_nulls.py`** (new), using the existing
`make_ens_dataset` and `make_h3_grid` fixtures. Every one of these fails on `main` because the
function does not exist; what matters is the assertion each pins:

1. *Clean run, exact denominator* — no NaN beyond lead-0 gives `is_healthy`,
   `n_null_grid_points == 0`, `null_grid_point_fraction == 0.0`, and `n_total_grid_points ==
   3 × n_lat × n_lon × n_members × n_steps_beyond_lead0` on the default `lead_time_hours=(0, 6, 12)`
   fixture. The denominator assertion is load-bearing: it is what a corrupt-slices-only denominator
   gets wrong, and that is the trap `_deaccumulated_null_breakdown`'s `.filter(n_null > 0)` leads an
   implementer into.
2. *Lead-0 nulls are excluded* — de-accumulated variables entirely NaN at lead-0 and clean beyond it
   still gives `is_healthy` and the same denominator. Catches both the skip-the-first-chunk and the
   integer-comparison bugs named above.
3. *Exact count on a known scatter* — NaN at a known set of `(member, lead_time, lat, lon)` positions
   **beyond lead-0** in `precipitation_surface` gives exactly that `n_null_nwp_grid_points`, the exact
   `null_nwp_grid_point_fraction` (compared with `pytest.approx`), `n_affected_nwp_slices` equal to the
   of distinct `(member, lead_time)` pairs touched, and
   `affected_variables == ("precipitation_surface",)`.
4. *The blindness this issue is about* — the load-bearing test. Convert a dataset whose scatter the
   H3 aggregation fully absorbs, and assert the returned `Nwp` frame has **no null cells** for that
   variable while `assess_upstream_grid_point_nulls` on the same dataset reports
   `n_null_grid_points > 0`. That is the exact condition under which `main`'s signal reads clean.

   **`default_h3_grid` cannot express this test** — it maps one grid point per cell at
   `proportion=1.0` (`conftest.py:150`), so any null point it names nulls its cell. Build a grid with
   two points per cell at 0.5/0.5 via `make_h3_grid`, as `test_convert_to_polars.py:713` already does.
5. *Instantaneous variables are out of scope* — NaN in `temperature_2m` does not move
   `n_null_nwp_grid_points`. Same multi-point grid needed: on a single-point cell a null `temperature_2m`
   nulls the cell, and that column is non-nullable, so `Nwp.validate` raises
   `DataFrameValidationError` and the test asserts nothing.
6. *Zero denominator* — a dataset with `lead_time_hours=(0,)` gives `n_total_grid_points == 0`,
   `null_grid_point_fraction == 0.0` and `is_healthy`, and does **not** raise. Pins the guard that
   keeps `test_ecmwf_ens_cached.py:83`'s real-slice path working.

**`tests/test_assets.py`**:

- **Test 7** — *The keys are plumbed onto the check and the materialisation* — a run whose stubbed converter
  returns a frame with no null cells, paired with a dataset carrying upstream scatter, publishes all
  five new keys on the check and `n_null_nwp_grid_points` / `null_nwp_grid_point_fraction` on the
  `MaterializeResult`, while the check still `passed`. **Fails on `main`:** none of those keys exist
  (`assets.py:431`, `339`).
- **Test 8** — *The description cannot claim health it does not have* — for that same run the description does
  **not** contain "No unexpected nulls" and does name the upstream fraction. **Fails on `main`:**
  `assets.py:413` emits exactly that phrase for this run.
- **Test 9** — *Degradation* — monkeypatch `assess_upstream_grid_point_nulls` to raise; the run still lands, both
  checks come back as WARN `_degraded_nwp_check_result`s, one Sentry event is reported, and the
  shape and upstream metadata keys are absent. Extends
  `test_ecmwf_ens_lands_the_run_when_an_assessment_fails` (line 643), which already asserts
  `"n_ensemble_members" not in materialisation.metadata` at line 679, so the mirror assertion drops
  straight in. This is what pins that the new call really is inside the guard.

New `monkeypatch` calls take keyword arguments — `monkeypatch.setattr(target=assets,
name="assess_upstream_grid_point_nulls", value=_raise)` — matching every call in the file.

Dropped as duplicates: a cancellation test (`test_ecmwf_ens_re_raises_a_cancelled_run_without_writing`,
`tests/test_assets.py:710`, already pins that branch) and
a `contracts` refactor guard (there is no refactor left to guard).

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "Write about the present, not the past".

- **`docs/architecture/ecmwf-ens-known-issues.md`** — the "Nulls in the de-accumulated variables
  (tolerated)" section gains a short paragraph naming the upstream rate as the measured provider
  signal and the cell count as the loss measure; the "Spatial aggregation is where a grid point's
  null is resolved" section stops implying the upstream rate is knowable only by offline analysis.
  **Do not present the existing "0.014% of `precipitation_surface`'s grid points" figure as what
  `null_nwp_grid_point_fraction` would read** — that number is per-variable and the metric pools all
  three de-accumulated variables, so the same run reads roughly a third of it. Either say so, or keep
  the doc's per-variable figure and describe the metric separately.
- **`docs/live_service/operations.md`** — "Reading the NWP check" currently tells the operator that
  measuring the feed "is tracked in issue #505" and that `n_null_cells` and `n_scattered_slices` stay
  small even when corruption is heavy. Rewrite: name `null_nwp_grid_point_fraction` as the number to read
  for the provider question and `affected_nwp_variables` as what to name in a mail to
  Dynamical, `n_null_h3_cells` for "what did we lose", keep the `n_whole_null_h3_slices` guidance, and drop
  the #505 pointer. Extend the existing "the shape metadata is absent from that materialisation"
  sentence to cover the two new materialisation keys.
- **`packages/contracts/src/contracts/weather_schemas.py`** — `NwpQualityReport`'s docstring, as
  above.
- **Every doc that names a renamed metadata key.** The rename in
  [Naming the two populations](#naming-the-two-populations) touches six existing keys, and
  `operations.md` and `ecmwf-ens-known-issues.md` both name several of them in prose. Grep for the
  old names (`n_null_cells`, `n_affected_slices`, `n_whole_null_slices`, `n_scattered_slices`,
  `affected_variables`, `affected_slices`) across `docs/` and the docstrings before finishing, since
  a doc naming a key that no longer exists is worse than one naming none.
- **`docs/design-philosophy/inherent-stability.md`** — add the upstream null rate to the
  provider-channel row of the "Three audiences, three channels" table (line 322). One cell, but it is
  the table the issue cites as its justification.
- No roadmap ship-time triage: this issue is not a roadmap item and completes no milestone banner.

## Verification commands

The green-before-push set from `implement-issue`:

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
```

Docs were touched, so also:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md && uv run mkdocs build --strict
```

And, because the new code reads the real dataset's coordinate structure:

```bash
uv run pytest --run-network -m network
```

`convert_to_polars.py` and `download.py` both carry explicit comments saying to run the network-gated
test after touching the code that reads `ds`, because the offline fixtures share their assumptions
with the code and cannot catch a mismatch with the live catalog. The new module reads `lead_time`'s
dtype and the `latitude`/`longitude` dimension names, so it is in that category.

## Decisions taken

All four open questions are settled; they are recorded here because each shaped the plan above.

**1. The fraction is metadata, not a gate.** See [Metadata, not a gate](#metadata-not-a-gate).

**2. The measured population is the downloaded lat/lon box**, not the grid points the H3 weights
name. The box is the bounding rectangle of the H3 grid's `nwp_lat`/`nwp_lon` extremes
(`open_ecmwf_ens_run`), so it includes corner points no H3 cell uses. That is what the provider
sent, it needs no join, and restricting to used points would put our H3 geometry back into the
number. The consequence the docs must state rather than paper over: this fraction and
`n_null_h3_cells` are **not** comparable as rates — different units over different populations. Only
the slice filter matches (both exclude lead-0), so the two agree on which slices are in scope and
nothing more. `NwpQualityReport` publishes no fraction at all, so there is no cell-level rate to
compare against.

**3. The instantaneous variables get their own check, in a follow-on PR.** See
[Follow-on work](#follow-on-work).

**4. #506 gets a triage note, not work here.** Once `null_nwp_grid_point_fraction` and
`n_null_h3_cells` are both published, #506's contributing-weight fraction is bracketed by the two
and may no longer earn its own metric. Worth re-reading #506 before starting it in wave 5.

## Follow-on work

**A second PR, after this one merges, adding a separate check for scattered nulls in the
*instantaneous* variables.** Not folded into this PR, and not a stretch goal for it.

`ecmwf-ens-known-issues.md:195` says out loud what #496 cost: "scattered corruption in a variable
that should never carry any is now mostly invisible at ingest", because a scattered null in
`temperature_2m` is absorbed by a cell's other points and never reaches `Nwp.validate`. The
machinery this plan builds restores that detector for roughly one extra line — the variable list it
counts over — but it must not share this check, for two reasons. A null in an instantaneous variable
is *anomalous* rather than expected, so it deserves a `passed` that means something, which
`nwp_has_no_unexpected_nulls` cannot give it while also reporting tolerated corruption. And pooling
both into one `null_nwp_grid_point_fraction` would average over variables with opposite null
semantics, producing a number that means nothing.

So: a third `AssetCheckSpec` on `ecmwf_ens`, with its own `passed` (plausibly "no instantaneous
grid-point nulls at all", since any is anomalous), its own metadata keys under the same
`nwp_grid_point` naming scheme, and its own paragraph in the operations runbook.

It reuses `assess_upstream_grid_point_nulls` with a different variable list, so that second PR stays
small **provided this one takes the variable list as a parameter** rather than hard-coding
`Nwp.deaccumulated_var_names` inside the counting loop. That is the one thing this PR does to
prepare for it, and it costs nothing here.

## Findings rejected

- *Publish the two trend keys on the materialisation only, not on the check as well.* Rejected: the
  duplication follows the established `_nwp_run_shape_metadata` / `_nwp_completeness_check_result`
  precedent, and an operator reading the check should not have to leave it to see the rate.
- *Merge #505 and #506 by reusing the `__contributing_weight` columns the aggregation already
  computes.* Rejected — and the reviewer that raised it rejected it too, for the right reason: that
  weight is weighted by H3 `proportion` and restricted to the points a cell uses, which puts our H3
  geometry back into the number, and removing it is the entire point of this issue.
- *Fold the whole-null-slice count into the upstream report for symmetry.* Rejected: departure 2 —
  the cell-level report already measures that case exactly, and duplicating it would add a key that
  means the same thing under a different name, which is what Jack's comment asks us to avoid.

## Out of scope

- **#506** (report the contributing-weight fraction from the H3 aggregation) is wave 5 and
  deliberately waits on this. Not folded in; see decision 4.
- `src/nged_substation_forecast/defs/checks.py` and `defs/production_assets.py` are owned by other
  parallel sessions and are not touched. Nothing in this plan needs them.
- `write_nwp`'s append-only behaviour and partition replacement
  ([#476](https://github.com/openclimatefix/nged-substation-forecast/issues/476)) are untouched.
