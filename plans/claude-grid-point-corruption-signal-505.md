# Plan — Move the upstream ECMWF-corruption signal to grid-point level (#505)

Branch: `claude/grid-point-corruption-signal-505`.
Issue: <https://github.com/openclimatefix/nged-substation-forecast/issues/505>.

## Verdict

**Worth implementing, roughly as described.** The issue's premise checks out against the code on
`main`:

- `assess_nwp_quality` (`packages/contracts/src/contracts/weather_schemas.py:589`) counts nulls in
  the *stored H3 cells*, via `_deaccumulated_null_breakdown` at line 494. Its input is a validated
  `pt.DataFrame[Nwp]`, which by construction is post-aggregation, so it cannot see a grid point.
- #496 has landed. `_aggregate_grid_points_to_h3_cells`
  (`packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py:135`) now renormalises
  each numeric variable over its own contributing weight, so a cell is null only when *every*
  contributing grid point is null. The amplification that made `n_null_cells` a usable provider
  proxy is gone by design.
- The code already admits the gap in prose: `NwpQualityReport`'s docstring says "Read this as 'how
  much did we lose', not as 'how corrupt was the feed'", and `docs/live_service/operations.md`
  tells the operator the same thing. Both point at this issue. So the change closes a hole the
  codebase has already documented rather than one this plan has invented.

The provider channel in
[Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#failure-modes)
wants a number the provider would act on. The grid-point null rate is that number; the cell count
is that rate convolved with our H3 resolution, our grid spacing and our aggregation policy.

## Departures from the issue body

1. **Compute it from the downloaded `xr.Dataset`, in its own pass — do not change the converter's
   return type.** The issue leaves this open ("a return-type change or a separate pass"). The
   separate pass wins on three counts: `convert_nwp_xarray_dataset_to_polars_dataframe` keeps a
   single return value and its five monkeypatch sites in `tests/test_assets.py` keep working
   unchanged; the measurement is a pure function of `ds` that can be unit-tested against the
   existing `_build_ens_dataset` fixture with no H3 grid at all; and the population it measures is
   the grid points Dynamical actually sent us, rather than the `(cell, grid-point)` pairs the join
   produces, which would re-introduce exactly the H3-geometry convolution the issue is removing (a
   point feeding 4.92 cells would otherwise count 4.92 times).

   The cost is one extra reduction over the downloaded arrays. That is ~1,200 grid points × 51
   members × 85 steps × 3 variables of `float32`, so a `~60 MB` `isnull().sum()` — negligible
   beside the download and the per-chunk aggregation loop it sits next to.

2. **The new number does not drive the check's `passed`.** See
   [Why it is metadata, not a gate](#why-it-is-metadata-not-a-gate). The issue does not ask for a
   pass/fail change, but it is the obvious thing an implementer would reach for, so the plan states
   the decision rather than leaving it to be discovered.

3. **No second metadata table.** `_nwp_quality_check_result` already writes a capped
   affected-slices table for the cell-level report. The upstream signal is a *rate*, and a second
   ~100-row table on every materialisation buys little for its event-log weight. Named variables
   plus the counts are enough.

## What changes, file by file

### `packages/contracts/src/contracts/weather_schemas.py`

The two reports count the same thing at two stages, so factor the arithmetic once:

- **New private base `_NullBreakdownReport`** — frozen dataclass holding `affected: pl.DataFrame`
  (columns `variable`, `init_time`, `ensemble_member`, `valid_time`, `n_null`, `n_total`, one row
  per slice carrying at least one null) with the unit-neutral properties: `n_null`, `n_total`,
  `null_fraction`, `n_affected_slices`, `n_whole_null_slices`, `n_scattered_slices`, `is_healthy`,
  `affected_variables`. The bodies move off `NwpQualityReport` unchanged.
- **`NwpQualityReport`** becomes a subclass. It keeps `n_null_cells` as a one-line alias for
  `n_null`, because that name is what makes the Dagster metadata self-describing — the point of
  Jack's comment on the issue. Its docstring loses the "measuring that rate where it lives is
  tracked in #505" paragraph and instead points at the new report as the sibling number.
- **New `UpstreamGridPointNullReport`** — the other subclass, with `n_null_grid_points` and
  `n_total_grid_points` aliases. Docstring says plainly what it measures: nulls in the
  de-accumulated variables on the raw 0.25° grid Dynamical sent, beyond lead-0, before any H3
  aggregation, over the downloaded lat/lon box.

`null_fraction` is new to both. It is the number that is comparable across runs, grids and box
sizes, and it is the one worth plotting.

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/upstream_nulls.py` (new module)

`assess_upstream_grid_point_nulls(ds: xr.Dataset) -> UpstreamGridPointNullReport`. Pure, Dagster-free,
mirroring `assess_nwp_quality`'s shape. For each name in `Nwp.deaccumulated_var_names` (sorted, as
`_deaccumulated_null_breakdown` does): reduce `ds[var].isnull()` over `latitude`/`longitude` to give
`n_null` per `(ensemble_member, lead_time)`, take `n_total` from the two spatial dimension sizes,
map `lead_time` to `valid_time` through the dataset's `valid_time` coordinate, drop lead-0, and
concatenate into one Polars frame keeping only rows with `n_null > 0`.

Dropping lead-0 is not optional: the de-accumulated variables are null there by design in every
run, everywhere, so including it would put a large constant in the numerator and hide the signal.
This matches `_deaccumulated_null_breakdown`'s `valid_time > init_time` filter, so the two reports
count over the same population and their fractions are directly comparable.

Two things the implementer must confirm against the real dataset rather than assume, both covered
by the network-gated test: that `download_ecmwf_ens_data`'s rebuilt `xr.Dataset(data_arrays)` still
carries the `valid_time` coordinate (it is a non-dimension coordinate on each `DataArray`, so it
should survive, but it is rebuilt rather than sliced), and that `init_time` is scalar by this point
(`open_ecmwf_ens_run` does `ds.sel(init_time=...)`, and the fixture models that with
`init_time_as_dim=False`).

### `docs/api/dynamical_data/index.md`

Add `::: dynamical_data.ecmwf_ens.upstream_nulls` so the new module reaches the published API
reference alongside `download` and `convert_to_polars`.

### `src/nged_substation_forecast/defs/assets.py`

- Import the new function and report type.
- Call `assess_upstream_grid_point_nulls(ds)` **inside the existing `try` block** that already wraps
  `assess_nwp_quality` and `assess_nwp_run_completeness` (line 308). This is the whole of the
  rule-7 story: the guard, the ordering before `write_nwp`, and the degraded-result fallback all
  already exist and the new call inherits them.
- `_nwp_quality_check_result` gains an `upstream: UpstreamGridPointNullReport` parameter. Its
  metadata gains the mirror-image key set, so the two levels read as a pair:

  | Stored cells (existing) | Upstream grid points (new) |
  |---|---|
  | `n_null_cells` | `n_null_grid_points` |
  | — | `n_total_grid_points` |
  | — | `null_grid_point_fraction` |
  | `n_affected_slices` | `n_upstream_affected_slices` |
  | `n_whole_null_slices` | `n_upstream_whole_null_slices` |
  | `n_scattered_slices` | `n_upstream_scattered_slices` |
  | `affected_variables` | `upstream_affected_variables` |

- **Rewrite the description**, which is the part that would otherwise become a lie. Today it reads
  "No unexpected nulls in the de-accumulated NWP variables." whenever no cell is null — which after
  #496 is the *usual* state even when upstream sent us a corrupt run. Three cases:
    - clean at both levels: no nulls upstream or in the stored cells;
    - upstream nulls fully absorbed: name the fraction and the variables, and say explicitly that
      the H3 aggregation absorbed all of them so no stored cell is null;
    - nulls at both levels: both numbers, each labelled with its unit.
- Add `null_grid_point_fraction` and `n_null_grid_points` to the materialisation metadata, next to
  the existing `**shape_metadata`, so the trend is plottable in the Dagster asset timeline on every
  run and not only on the ones that warn. This follows `_nwp_run_shape_metadata`'s existing
  precedent of publishing on every materialisation. They are absent — like the shape keys — on a
  materialisation whose assessment raised, for the same reason: there is no report to read them
  from, and a key whose metadata *type* varies between runs breaks the timeline plot.

### Why it is metadata, not a gate

`passed` stays driven by the cell-level report. Three reasons, in order of weight:

1. There is no honest threshold. The upstream corruption is known, recurring and WONTFIX
   ([dynamical-org/reformatters#722](https://github.com/dynamical-org/reformatters/issues/722)), so
   `passed = (no null grid points)` would WARN on a large and unknown fraction of runs — the archive
   figures in `ecmwf-ens-known-issues.md` count *cells*, so we genuinely do not know the grid-point
   base rate inside the GB box until this ships and measures it. A check that warns constantly
   carries no information. The alternative, a rate threshold, is the magic number
   `ecmwf-ens-known-issues.md`'s "the gate needs no magic thresholds" and the missed-runs argument
   in `inherent-stability.md` both refuse.
2. `passed` answers "did this run land damaged", which is the cell-level question. The upstream rate
   answers "is the feed getting worse", which is a question about a *sequence* of runs and belongs
   on a timeline, not in a boolean.
3. Shipping the measurement first is what makes a future gate designable: after a few months of
   `null_grid_point_fraction` on the timeline there is a base rate to set an escalation on.
   Escalating a badly-degraded run is already tracked separately in
   [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501), and this metric is
   the input it needs.

This keeps the issue's constraint exactly: one check, still `WARN`, still `blocking=False`, still
inside a body that cannot raise.

## Design-philosophy check

- **Production path, so degrade rather than raise.** `ecmwf_ens` is production ingest. The new call
  is inside the `except BaseException` guard added for #509, which turns any failure into two
  degraded WARN results plus a Sentry event and lets the run land — so
  [rule 7](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
  (a warning path may never fail the thing it warns about) holds by construction. The guard sits
  *before* `write_nwp`, so a raise cannot duplicate rows either.
- **Rule 6**: the check keeps `AssetCheckSeverity.WARN` and `blocking=False` in its
  `AssetCheckSpec`. No `ERROR`-severity check is added.
- **Rule 2** (liberal about missing, strict about malformed) is untouched: `Nwp.validate` remains the
  only gate, and this change adds no rejection path.
- **H1 / T1.1.** This serves the operability claim directly. T1.1 counts interventions *by cause*,
  and the intervention log can only attribute a cause it can see; today a worsening Dynamical feed
  is invisible until it is bad enough to null whole cells. The provider channel in
  [Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#failure-modes)
  requires a number the provider would act on, and this is it.
- **No principle is traded away.** Nothing is fabricated or imputed, so
  [principle 9](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#9-provenance-travels-with-the-forecast-data)
  is unaffected; the change adds a measurement and no new failure path.

## Tests

Each of these fails on `main` today. Offline in every case — no network, no wall clock, no trained
model.

**`packages/dynamical_data/tests/test_upstream_nulls.py`** (new), using the existing
`make_ens_dataset` fixture:

1. *Clean run* — no NaN anywhere beyond lead-0 gives `is_healthy`, `n_null_grid_points == 0`,
   `null_fraction == 0.0`, and an empty `affected` frame. **Fails on `main`:** the function does not
   exist.
2. *Lead-0 nulls are excluded* — a dataset whose de-accumulated variables are entirely NaN at
   lead-0 and clean beyond it still reports `is_healthy`. This is the assertion that would catch the
   most likely implementation bug, and it fails on `main` for the same reason.
3. *Scattered scatter is counted exactly* — NaN at a known set of `(member, lead_time, lat, lon)`
   positions in `precipitation_surface` gives exactly that `n_null_grid_points`, the right
   `n_total_grid_points` (`n_lat × n_lon × n_members × n_steps_beyond_lead0` for the one affected
   variable's slices), the exact `null_fraction`, and `affected_variables == ("precipitation_surface",)`.
4. *Whole-slice versus scattered* — one wholly-NaN `(member, lead_time)` slice plus one partly-NaN
   slice gives `n_whole_null_slices == 1` and `n_scattered_slices == 1`.
5. *Instantaneous variables are out of scope* — NaN in `temperature_2m` does not appear in the
   report. This pins the deliberate scope decision so a later change to it is a visible test edit.

**`tests/test_assets.py`**:

6. *The blindness this issue is about* — the load-bearing test. Drive `ecmwf_ens` with a dataset
   carrying scattered grid-point NaN that the H3 aggregation fully absorbs, and assert both halves
   at once: `n_null_cells == 0` and the check `passed`, **and** `n_null_grid_points > 0` with
   `null_grid_point_fraction` at its exact expected value. **Fails on `main`:** those two keys do not
   exist, and `main`'s check reports the run as entirely clean. The existing scattered-null test at
   `tests/test_assets.py:422` is the closest thing on `main` and shows the shape to follow.
7. *Materialisation metadata carries the trend keys* — `null_grid_point_fraction` and
   `n_null_grid_points` are present on the `MaterializeResult` of a **healthy** run too, not only a
   degraded one. **Fails on `main`:** absent.
8. *The description cannot claim health it does not have* — for the run in test 6, the description
   does **not** contain "No unexpected nulls" and does name the upstream fraction. **Fails on
   `main`:** `main` emits exactly that phrase for this run.
9. *Degradation* — monkeypatch `assess_upstream_grid_point_nulls` to raise; assert the run still
   lands, both checks come back as WARN `_degraded_nwp_check_result`s, one Sentry event is
   reported, and the shape and grid-point metadata keys are absent. Mirrors the existing
   `assess_nwp_quality`-raises tests at `tests/test_assets.py:637` and `:671`, and pins that the new
   call really is inside the guard rather than above it. **Fails on `main`:** the patch target does
   not exist.
10. *Cancellation still cancels* — the same monkeypatch raising `KeyboardInterrupt` propagates,
    matching `tests/test_assets.py:697`. Cheap, and it stops the guard from being widened by
    accident.

**`packages/contracts/tests/test_weather_schemas_validation.py`**: extend the existing
`assess_nwp_quality` tests with `null_fraction` on a known breakdown, and assert `n_null_cells` still
reads the same value after the base-class refactor. The latter passes on `main` by definition — it
is a refactor guard, and is called out as such rather than claimed as a test of this change.

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "Write about the present, not the past".

- **`docs/architecture/ecmwf-ens-known-issues.md`** — the "Nulls in the de-accumulated variables
  (tolerated)" section gains a short paragraph naming the upstream rate as the measured provider
  signal and the cell count as the loss measure, and the "Spatial aggregation is where a grid
  point's null is resolved" section stops implying the upstream rate is only knowable by offline
  analysis. The measured-run figures there (0.014% of `precipitation_surface`'s grid points on
  2025-06-04 00Z) become the worked example of what `null_grid_point_fraction` reads.
- **`docs/live_service/operations.md`** — "Reading the NWP check" currently tells the operator that
  measuring the feed "is tracked in issue #505" and that `n_null_cells` and `n_scattered_slices`
  stay small even when corruption is heavy. Rewrite: name `null_grid_point_fraction` as the number
  to read for the provider question, `n_null_cells` for the "what did we lose" question, keep the
  `n_whole_null_slices` guidance, and drop the #505 pointer. Also extend the "shape metadata is
  absent when the assessment failed" note to cover the two new materialisation keys.
- **`packages/contracts/src/contracts/weather_schemas.py`** — `NwpQualityReport`'s docstring, as
  above. Its "measuring that rate where it lives is tracked in #505" sentence must go, or the docs
  will point at a closed issue.
- **`docs/design-philosophy/inherent-stability.md`** — the provider-channel row of the
  "Three audiences, three channels" table lists the channels that answer "is your feed broken";
  add the upstream null rate to it. One cell, but it is the table the issue cites as its
  justification.
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

And specific to this change:

```bash
uv run pytest --run-network -m network
```

**This one is not optional here.** The new code reads `ds`'s dimension names, its `valid_time`
coordinate and its `lead_time` units directly, which is exactly the convention-sensitivity that
`convert_to_polars.py` and `download.py` both carry an explicit comment about. The offline fixture
shares those assumptions with the code, so only the network-gated test can catch a mismatch with the
real Dynamical catalog.

## Risks and open questions

**1. Should the measurement also cover the instantaneous variables?** Recommendation: **not in this
issue, but it is worth its own.** `ecmwf-ens-known-issues.md` says out loud what #496 cost:
"scattered corruption in a variable that should never carry any is now mostly invisible at ingest",
because a scattered null in `temperature_2m` is absorbed by a cell's other points and never reaches
`Nwp.validate`. The machinery this plan builds would restore that detector for roughly one extra
line — the variable list it unpivots. It is excluded here because it changes what the check *means*:
a null in an instantaneous variable is anomalous rather than expected, so it warrants its own check
with its own `passed`, not a line item in a report about tolerated corruption. Folding it in would
also make the single `null_grid_point_fraction` an average over variables with opposite null
semantics. Jack's call.

**2. `passed` stays cell-level** ([above](#why-it-is-metadata-not-a-gate)). Recommendation: keep it,
and revisit once there is a measured base rate. Flagged because it is the one place a reader might
expect the issue to have changed behaviour and it has not.

**3. The measured population is the downloaded lat/lon box, not the grid points the H3 weights
name.** The box is the bounding rectangle of the H3 grid's `nwp_lat`/`nwp_lon` extremes
(`open_ecmwf_ens_run`), so it includes points in the rectangle's corners that no H3 cell uses.
Recommendation: keep the box. It is what the provider sent, it needs no join, and restricting to
used points would put our H3 geometry back into the number. Worth knowing when comparing this
fraction against a figure computed some other way — the denominators differ.

**4. Is the shared `_NullBreakdownReport` base worth it?** It removes ~35 lines that would otherwise
be duplicated across two dataclasses in the same file. Recommendation: keep it, since it has two
real call sites today rather than a hypothetical second one — but it is the most arguable structural
choice in this plan and the natural thing for a reviewer to attack.

## Out of scope

- **#506** (report the contributing-weight fraction from the H3 aggregation) is wave 5 and
  deliberately waits on this. It is the same question from the other end, and is not folded in.
- `src/nged_substation_forecast/defs/checks.py` and `defs/production_assets.py` are owned by other
  parallel sessions and are not touched. Nothing in this plan needs them.
- `write_nwp`'s append-only behaviour and partition replacement
  ([#476](https://github.com/openclimatefix/nged-substation-forecast/issues/476)) are untouched.
