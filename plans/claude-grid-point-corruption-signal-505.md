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
  much did we lose', not as 'how corrupt was the feed'", and `docs/live_service/operations.md` tells
  the operator the same thing. Both point at this issue. So the change closes a hole the codebase
  has already documented rather than one this plan has invented.

The provider channel in
[Three audiences, three channels](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#failure-modes)
wants a number the provider would act on. The grid-point null rate is that number; the cell count is
that rate convolved with our H3 resolution, our grid spacing and our aggregation policy.

## Departures from the issue body

1. **Count inside the converter and change its return type**, rather than making a separate pass
   over the downloaded `xr.Dataset`. The issue leaves this open. See
   [Why the converter, not a second pass](#why-the-converter-not-a-second-pass) — the deciding facts
   are that the count is a free byproduct of a loop the converter already runs, and that it needs no
   xarray API surface at all.

2. **Report scalars, not a per-slice breakdown frame.** The issue asks for
   `(variable, ensemble_member, valid_time)` granularity, "to keep the whole-slice-versus-scatter
   distinction that report already draws". The distinction is worth keeping; a second breakdown
   frame is not the way to get it, because the cell-level report already draws it *exactly*. If a
   slice is null at every grid point, then every cell in that slice has zero contributing weight, so
   every cell comes out null (`_aggregate_grid_points_to_h3_cells`), and the existing
   `n_whole_null_slices` names it. The two can only diverge if corruption spares precisely the
   box-corner points no H3 cell uses. So the upstream report carries a *count* of affected slices —
   which distinguishes "one bad slice" from "a hundred", and is not derivable from the cell-level
   numbers once #496 absorbs the scatter — and leaves the naming of wholly-null slices to the
   report that already does it well.

   What this gives up: naming *which* slices carried scatter that the aggregation then absorbed.
   Nothing acts on that today, and the provider question ("is your feed broken, and since when?") is
   answered by a rate on a timeline.

3. **The new number does not drive the check's `passed`.** See
   [Why it is metadata, not a gate](#why-it-is-metadata-not-a-gate). The issue does not ask for a
   pass/fail change, but it is the obvious thing an implementer would reach for, so the plan states
   the decision rather than leaving it to be discovered.

4. **No second metadata table.** `_nwp_quality_check_result` already writes a capped affected-slices
   table for the cell-level report. The upstream signal is a rate; a second ~100-row table on every
   materialisation buys little for its event-log weight.

## What changes, file by file

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py`

**New frozen dataclass `UpstreamNullRate`** — the upstream-corruption signal, four fields and two
properties, no Polars frame:

- `n_null_grid_points: int` — null grid points across the de-accumulated variables, beyond lead-0.
- `n_total_grid_points: int` — the denominator: every de-accumulated variable × ensemble member ×
  forecast step beyond lead-0 × grid point in the downloaded box.
- `n_affected_slices: int` — `(variable, ensemble_member, valid_time)` slices carrying at least one
  null grid point.
- `affected_variables: tuple[str, ...]` — sorted.
- `null_grid_point_fraction: float` and `is_healthy: bool` as properties.

Its docstring must say plainly what it measures and over what population, because that is the whole
point of Jack's comment on the issue: nulls on the raw 0.25° grid Dynamical sent, before any H3
aggregation, over the downloaded lat/lon box, beyond lead-0.

**`_process_chunk_for_1_lead_time_and_1_ens_member` returns its per-chunk counts alongside the
aggregated frame.** The counts come off `nwp_df` — the raw-grid frame that already exists at line
123, after `fill_nan(None)` has normalised NaN to null and *before* the H3 join — so the population
is the grid points Dynamical sent, with no H3 geometry in it. Per de-accumulated variable that is
one `is_null().sum()` and `nwp_df.height`.

**`convert_nwp_xarray_dataset_to_polars_dataframe` returns `tuple[pt.DataFrame[Nwp], UpstreamNullRate]`**,
accumulating the per-chunk counts over the loop it already runs at lines 52–66 and skipping lead-0
chunks. Its docstring documents both return values; the name still describes its primary job.

Skipping lead-0 is not optional: the de-accumulated variables are null there by design in every run,
everywhere, so including it would put a large constant in the numerator and hide the signal. This
matches `_deaccumulated_null_breakdown`'s `valid_time > init_time` filter, so the two reports count
over the same population and their fractions are comparable.

The denominator is the *whole* run, not the corrupt slices. Stated explicitly because the obvious
implementation — mirror `_deaccumulated_null_breakdown`, keep only rows with `n_null > 0`, then sum
both columns — yields "nulls ÷ grid points in slices that already had a null", which is not a rate:
a run with one 2%-corrupt slice and a run with a hundred of them would publish the same number.

### `packages/contracts/src/contracts/weather_schemas.py`

Docstring only, no code. `NwpQualityReport`'s "measuring that rate where it lives is tracked in
#505" paragraph must go — it would otherwise point at a closed issue — and is replaced by a pointer
to `UpstreamNullRate` as the sibling number that answers the provider question.

No new type and no shared base class here. The two reports have no code in common once the upstream
one is scalars, and `contracts` is the home of Patito data schemas; a report describing an
`xr.Dataset`-derived count belongs next to the code that computes it, as `PowerFreshnessResult`,
`MissedNwpRuns` and `LiveForecastHealthResult` already do in `defs/checks.py`.

### `src/nged_substation_forecast/defs/assets.py`

- Unpack the converter's two return values at line 294.
- `_nwp_quality_check_result` gains an `upstream: UpstreamNullRate` parameter and four metadata
  keys, named so the two levels cannot be confused:

  | Stored H3 cells (existing) | Upstream grid points (new) |
  |---|---|
  | `n_null_cells` | `n_null_grid_points` |
  | — | `n_total_grid_points` |
  | — | `null_grid_point_fraction` |
  | `n_affected_slices` | `n_upstream_affected_slices` |
  | `n_whole_null_slices`, `n_scattered_slices`, `affected_variables`, `affected_slices` | — |

  Four keys, not seven: `n_upstream_whole_null_slices` would duplicate the existing
  `n_whole_null_slices` (departure 2), and `n_upstream_scattered_slices` is then the difference of
  two keys already present.
- **Rewrite the description to always name both levels**, in one sentence each, with no case
  analysis. Today it reads "No unexpected nulls in the de-accumulated NWP variables." whenever no
  cell is null — which after #496 is the *usual* state even when upstream sent a corrupt run, so the
  string claims health it did not measure. Always emitting both numbers removes the failure mode
  rather than adding a branch to handle it.
- Add `null_grid_point_fraction` and `n_null_grid_points` to the materialisation metadata next to
  `**shape_metadata`, so the trend is plottable in the Dagster asset timeline on every run and not
  only on the ones that warn — following `_nwp_run_shape_metadata`'s existing precedent. Unlike the
  shape keys these are always present, because they come from the converter rather than from the
  guarded assessment block.

`ecmwf_ens`'s `AssetCheckSpec`s, the guard, the ordering before `write_nwp` and
`_degraded_nwp_check_result` are all untouched.

### Why the converter, not a second pass

Both options appear in the issue. The converter wins on evidence, not taste:

- **The second pass would break five existing tests, not zero.** Every site in `tests/test_assets.py`
  stubs the download as `lambda ds: object()` (lines 364, 420, 464, 517, and `_stub_ecmwf_download`
  at 574). A function taking `ds` would receive a bare `object()`, raise inside the assessment
  guard, and degrade *both* checks — breaking every assertion those tests make about `n_null_cells`,
  `n_whole_null_slices` and the shape metadata. Fixing that needs a real synthetic `xr.Dataset` in
  root `tests/`, which cannot import `packages/dynamical_data/tests/conftest.py`. The converter
  option changes the same five stubs to return a 2-tuple and needs no xarray there at all.
- **It adds no new xarray coupling.** Counting on `nwp_df` uses Polars only. A second pass would
  read `ds`'s dimension names, its `valid_time` coordinate and its `lead_time` units — new
  convention-sensitivity in exactly the area `convert_to_polars.py` and `download.py` both carry
  explicit warnings about, and a new thing that can drift against the real catalog.
- **It is free.** The counts come off a frame the loop has already materialised, instead of a second
  reduction over ~60 MB of downloaded arrays.

What it costs is a two-valued return on the main converter. See risk 1 for the one thing that buys
the second pass.

### Why it is metadata, not a gate

`passed` stays driven by the cell-level report. Three reasons, in order of weight:

1. There is no honest threshold. The upstream corruption is known, recurring and WONTFIX
   ([dynamical-org/reformatters#722](https://github.com/dynamical-org/reformatters/issues/722)), so
   `passed = (no null grid points)` would WARN on a large and unknown fraction of runs — the archive
   figures in `ecmwf-ens-known-issues.md` count *cells*, so we genuinely do not know the grid-point
   base rate inside the GB box until this ships and measures it. A check that warns constantly
   carries no information. The alternative, a rate threshold, is the magic number
   `ecmwf-ens-known-issues.md`'s "the gate needs no magic thresholds" and the missed-runs argument in
   `inherent-stability.md` both refuse.
2. `passed` answers "did this run land damaged", which is the cell-level question. The upstream rate
   answers "is the feed getting worse", which is a question about a *sequence* of runs and belongs on
   a timeline, not in a boolean.
3. Shipping the measurement first is what makes a future gate designable: after a few months of
   `null_grid_point_fraction` on the timeline there is a base rate to set an escalation on.
   Escalating a badly-degraded run is already tracked separately in
   [#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501), and this metric is
   the input it needs.

This keeps the issue's constraint exactly: one check, still `WARN`, still `blocking=False`.

## Design-philosophy check

- **Rule 6**: the check keeps `AssetCheckSeverity.WARN` and `blocking=False`. No `ERROR`-severity
  check is added.
- **Rule 7 — the one that needs an argument, because the counting moves into the ingest path.** The
  warning *function* still cannot raise: `_nwp_quality_check_result` runs inside the guard added for
  #509, before `write_nwp`, so a failure there still degrades to two WARN results and lets the run
  land. What changes is that the *counting* now sits in the converter, inside the earlier
  `RetryRequested` try, where a raise costs the partition. That is acceptable here because the
  counting cannot fail independently of the conversion it rides on: it is
  `nwp_df[var].is_null().sum()` over the same frame, the same columns and the same null semantics
  that `_aggregate_grid_points_to_h3_cells` immediately uses for its contributing weights
  (`pl.col("proportion").filter(pl.col(var).is_not_null()).sum()`, line 169). Any state that breaks
  the count breaks the aggregation two lines later. Rule 7 forbids a warning path that can fail a
  healthy run; this one cannot, because there is no healthy run in which it fails. Risk 1 records
  the alternative if that argument is not accepted.
- **Rule 2** (liberal about missing, strict about malformed) is untouched: `Nwp.validate` remains the
  only gate, and this change adds no rejection path.
- **H1 / T1.1.** This serves the operability claim directly. T1.1 counts interventions *by cause*,
  and the intervention log can only attribute a cause it can see; today a worsening Dynamical feed
  is invisible until it is bad enough to null whole cells.
- **No principle is traded away.** Nothing is fabricated or imputed, so
  [principle 9](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#9-provenance-travels-with-the-forecast-data)
  is unaffected; the change adds a measurement and no new failure path.

## Tests

Offline in every case — no network, no wall clock, no trained model.

**`packages/dynamical_data/tests/test_convert_to_polars.py`**, using the existing `make_ens_dataset`
and `make_h3_grid` fixtures:

1. *Clean run* — no NaN beyond lead-0 gives `is_healthy`, `n_null_grid_points == 0`,
   `null_grid_point_fraction == 0.0`, and `n_total_grid_points` equal to
   `3 × n_lat × n_lon × n_members × n_steps_beyond_lead0`. **Fails on `main`:** the converter returns
   one value, so the tuple unpack raises. The `n_total_grid_points` assertion is the one that pins
   the whole-run denominator — it is what a corrupt-slices-only denominator gets wrong.
2. *Lead-0 nulls are excluded* — de-accumulated variables entirely NaN at lead-0 and clean beyond it
   still gives `is_healthy` and the same denominator. Catches the most likely implementation bug.
3. *Exact count on a known scatter* — NaN at a known set of `(member, lead_time, lat, lon)` positions
   in `precipitation_surface` gives exactly that `n_null_grid_points`, the exact
   `null_grid_point_fraction`, `n_affected_slices` equal to the number of distinct
   `(member, lead_time)` pairs touched, and `affected_variables == ("precipitation_surface",)`.
4. *The blindness this issue is about* — the load-bearing test. Scatter that the H3 aggregation fully
   absorbs: assert the returned `Nwp` frame has **no null cells** for that variable *and*
   `n_null_grid_points > 0`. This is the exact condition under which `main`'s signal reads clean, in
   one assertion pair.
5. *Instantaneous variables are out of scope* — NaN in `temperature_2m` does not move
   `n_null_grid_points`. Pins the deliberate scope decision, so changing it is a visible test edit.

**`tests/test_assets.py`**:

6. *The keys are plumbed onto the check and the materialisation* — a run whose stubbed converter
   returns a non-zero `UpstreamNullRate` alongside a frame with no null cells publishes
   `n_null_grid_points`, `n_total_grid_points`, `null_grid_point_fraction` and
   `n_upstream_affected_slices` on the check, and the first and third on the `MaterializeResult`,
   while the check still `passed`. **Fails on `main`:** none of those keys exist.
7. *The description cannot claim health it does not have* — for that same run the description does
   **not** contain "No unexpected nulls" and does name the upstream fraction. **Fails on `main`:**
   `main` emits exactly that phrase for this run.
8. *The materialisation keys survive a degraded assessment* — when `assess_nwp_quality` raises, the
   shape metadata is absent but `null_grid_point_fraction` is still published, because it comes from
   the converter rather than from the guarded block. **Fails on `main`:** the key does not exist.
   Extends the existing raising test at `tests/test_assets.py:637` rather than adding a new one.

Dropped as duplicates: a cancellation test (`tests/test_assets.py:697` already pins that line, and a
second patch target tests the same branch) and a `contracts` refactor guard (there is no refactor
left to guard).

## Docs to update

Written to describe how the code works now, per CLAUDE.md's "Write about the present, not the past".

- **`docs/architecture/ecmwf-ens-known-issues.md`** — the "Nulls in the de-accumulated variables
  (tolerated)" section gains a short paragraph naming the upstream rate as the measured provider
  signal and the cell count as the loss measure. The "Spatial aggregation is where a grid point's
  null is resolved" section stops implying the upstream rate is knowable only by offline analysis;
  its measured figures (0.014% of `precipitation_surface`'s grid points on 2025-06-04 00Z) become
  the worked example of what `null_grid_point_fraction` reads.
- **`docs/live_service/operations.md`** — "Reading the NWP check" currently tells the operator that
  measuring the feed "is tracked in issue #505" and that `n_null_cells` and `n_scattered_slices` stay
  small even when corruption is heavy. Rewrite: name `null_grid_point_fraction` as the number to read
  for the provider question, `n_null_cells` for "what did we lose", keep the `n_whole_null_slices`
  guidance, and drop the #505 pointer. Note that the two upstream keys are present even on a
  materialisation whose assessment degraded, unlike the shape keys.
- **`packages/contracts/src/contracts/weather_schemas.py`** — `NwpQualityReport`'s docstring, as
  above.
- **`docs/design-philosophy/inherent-stability.md`** — add the upstream null rate to the
  provider-channel row of the "Three audiences, three channels" table. One cell, but it is the table
  the issue cites as its justification.
- No new `docs/api` entry: `UpstreamNullRate` lives in `convert_to_polars.py`, which
  `docs/api/dynamical_data/index.md` already renders.
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

And, because `convert_to_polars.py` changed:

```bash
uv run pytest --run-network -m network
```

That module carries an explicit comment saying to run the network-gated test after changing it,
since the offline fixtures share their assumptions with the code and cannot catch a mismatch with
the live catalog. This change adds no new xarray surface, so the risk is lower than it would have
been for a second pass over `ds` — but the instruction is unconditional and the run is cheap.

## Risks and open questions

**1. The counting moves into the ingest path, where a raise costs the partition.** Recommendation:
**accept**, on the argument in the design-philosophy section — the count cannot fail independently
of the conversion it rides on. Flagged because rule 7 is the constraint the issue names explicitly
and #509 was filed about exactly this hazard, so it deserves Jack's eye rather than a silent
judgement. The fallback if the argument is rejected: compute from the downloaded `xr.Dataset`
instead, inside the existing assessment guard, at the cost of ~15 lines of synthetic `xr.Dataset` in
root `tests/`, five stub-site edits, a second reduction over the downloaded arrays, and new
convention-sensitivity to the real catalog's coordinate names.

**2. Should the measurement also cover the instantaneous variables?** Recommendation: **not in this
issue, but it is worth its own.** `ecmwf-ens-known-issues.md` says out loud what #496 cost:
"scattered corruption in a variable that should never carry any is now mostly invisible at ingest",
because a scattered null in `temperature_2m` is absorbed by a cell's other points and never reaches
`Nwp.validate`. The machinery here would restore that detector for roughly one extra line — the
variable list it counts over. It is excluded because it changes what the check *means*: a null in an
instantaneous variable is anomalous rather than expected, so it warrants its own check with its own
`passed`, not a line item in a report about tolerated corruption. Folding it in would also make a
single `null_grid_point_fraction` an average over variables with opposite null semantics.

**3. `passed` stays cell-level** ([above](#why-it-is-metadata-not-a-gate)). Recommendation: keep, and
revisit once there is a measured base rate. Flagged because it is the one place a reader might expect
this issue to have changed behaviour and it has not.

**4. The measured population is the downloaded lat/lon box, not the grid points the H3 weights name.**
The box is the bounding rectangle of the H3 grid's `nwp_lat`/`nwp_lon` extremes
(`open_ecmwf_ens_run`), so it includes corner points no H3 cell uses. Recommendation: keep the box.
It is what the provider sent, it needs no join, and restricting to used points would put our H3
geometry back into the number. Worth knowing when comparing this fraction against a figure computed
some other way — the denominators differ.

**5. Triage note for #506, not work for this issue.** Once `null_grid_point_fraction` (upstream) and
`n_null_cells` (post-aggregation) are both published, #506's contributing-weight fraction is
bracketed by the two and may no longer earn its own metric. Worth re-reading #506 before starting it
in wave 5.

## Simplicity review — findings rejected

- *Publish the two trend keys on the materialisation only, not on the check as well.* Rejected: the
  duplication follows the established `_nwp_run_shape_metadata` / `_nwp_completeness_check_result`
  precedent, and an operator reading the check should not have to leave it to see the rate.
- *Merge #505 and #506 by reusing the `__contributing_weight` columns the aggregation already
  computes.* Rejected — and the reviewer that raised it rejected it too, for the right reason: that
  weight is weighted by H3 `proportion` and restricted to the points a cell uses, which puts our H3
  geometry back into the number, and removing it is the entire point of this issue.

## Out of scope

- **#506** (report the contributing-weight fraction from the H3 aggregation) is wave 5 and
  deliberately waits on this. Not folded in; see risk 5.
- `src/nged_substation_forecast/defs/checks.py` and `defs/production_assets.py` are owned by other
  parallel sessions and are not touched. Nothing in this plan needs them.
- `write_nwp`'s append-only behaviour and partition replacement
  ([#476](https://github.com/openclimatefix/nged-substation-forecast/issues/476)) are untouched.
