# Known ECMWF ENS data-quality issues

This page and [NWP variable conventions](nwp-variable-conventions.md) split the subject between
them:

- **This page is about *upstream*.** It covers the ways Dynamical.org's data arrives damaged —
  corrupt source accumulations, null grid points, incomplete runs — and the ingest policy that
  decides which of those we reject and which we tolerate. Its subject is data we did not create and
  cannot fix, so a defect described here is a reason to talk to Dynamical.

- **That page is about *us*.** It covers how the data that arrives intact is meant to be
  interpreted, and what our own code does with it. Everything there is a choice this project made,
  or failed to make deliberately, so a defect described there is a reason to change our code.

We ingest ECMWF IFS ENS from [Dynamical.org](https://dynamical.org). Its data carries a few
known, recurring quality quirks. This page records what they are, how we tell them apart, and the
policy the `ecmwf_ens` ingest applies to each. The policy is implemented in
[`contracts.weather_schemas`](../api/contracts/index.md): `Nwp.validate` is the fatal ingest gate,
and non-fatal reporters sit behind the asset's three WARN checks — `assess_nwp_quality` behind
`nwp_has_no_unexpected_nulls`, `assess_upstream_grid_point_nulls` behind
`nwp_instantaneous_variables_have_no_nulls`, and `assess_nwp_run_completeness` behind
`nwp_run_is_complete`.

## The guiding principle

We reject data that is *structurally* wrong and tolerate gaps a model can absorb, surfacing those
as a warning rather than throwing away an otherwise-good run. Which case we are in is decided by
the *variable*, so the gate needs no magic thresholds:

- An **instantaneous** variable (temperature, winds, pressures) is never legitimately null, so any
  null in one is anomalous, and any null is fatal.
- A **de-accumulated** variable (precipitation, the two radiation fluxes) is null by design at
  lead-0 and carries known upstream corruption beyond it, so its nulls are in-distribution. Every
  null pattern in one of these is tolerated, with exactly one exception: it is fatal when that
  variable is null in **every single** `(ensemble_member, valid_time)` slice beyond lead-0 of the
  run being ingested — that is, when the column holds no weather anywhere in the run. The test is
  literally "is anything left?", so a run with 4283 of one variable's 4284 slices empty still
  lands (with a warning); only 4284 of 4284 fails.

The gate judges the **H3 cells we store**, not the raw grid points upstream sent us, and the two
differ because a cell is the area-weighted mean of the ~2.9 grid points that overlap it — see
[the next section](#spatial-aggregation-is-where-a-grid-points-null-is-resolved). For the numeric
variables a cell is null only when *no* grid point contributed a value to it, since a null at one
of its points is absorbed by the others. So "any null is fatal" is a statement about a cell with
nothing behind it, and a scattered grid-point null in an instantaneous variable usually never
reaches the gate at all. One kind of cell is exempt: a cell fed by a single grid point, where that
point *is* the cell and one null is still fatal; the V1 grid has 10 such cells out of 1671.

`categorical_precipitation_type_surface` reaches the same rule by a different route, since a
category cannot be averaged: its cell takes whichever category covers the most of the cell's area,
and points that supplied no category are excluded from that ranking rather than competing in it. So
it too is null only when *no* point supplied a value, and a null there is fatal for any `init_time`
after 2024-11-12.

## Spatial aggregation is where a grid point's null is resolved

We do not store the raw 0.25° grid. `convert_nwp_xarray_dataset_to_polars_dataframe` aggregates it
onto H3 cells, each of which is the area-weighted mean of the grid points overlapping it, so a
grid point's null is resolved before anything on this page ever sees it. On the V1 grid (H3
resolution 5 over the GB boundary) a cell averages **2.93 grid points**, and only 10 of its 1671
cells are fed by a single point.

Each numeric variable is renormalised over the points that actually supplied a value: the weighted
sum is divided by the *contributing* weight rather than by 1.0. A null grid point therefore costs
its own share of each of the ~4.9 cells it feeds, and nothing more. The alternative — dividing by
1.0 regardless — silently treats an absent point as contributing zero, so the cell comes out low in
proportion to how much of it went missing. That is the imputation the
[never zero-fill rule](../design-philosophy/inherent-stability.md#missingness-in-learned-models)
warns about, which is worth saying plainly because it inverts the intuition: renormalising
*replaces* an implicit zero-fill with an available-case area mean rather than adding a fill on top
of clean data, so it leaves the provenance guarantees of
[principle 9](../design-philosophy/design-principles.md#9-provenance-travels-with-the-forecast-data)
intact. Nothing is fabricated from another time step, another run or another cell.

Each variable gets its *own* denominator, rather than one shared across the cell. A shared
denominator would be simpler and is wrong: it would let a single corrupt variable null every other
variable in the same cell, which is the amplification described below, applied across variables
instead of across cells. What per-variable denominators cost is that two variables in one cell can
end up averaged over different sub-areas of the hexagon. That matters only where a pair is later
recombined — `wind_u_*` with `wind_v_*`, in `_calc_wind_speed` and `_calc_wind_direction` — and
upstream corruption has so far always been co-located across variables, so it is theoretical today.

`categorical_precipitation_type_surface` cannot be averaged, so the `proportion` weights are summed
per category and the category covering most of the cell's area wins. Points that supplied no
category are excluded from that ranking rather than competing in it — the distinction matters,
because a mode that counts null as a candidate value nulls a cell whenever the missing points
outnumber each surviving category, which is a far weaker condition than losing the cell.

An exact tie is broken on the lowest category code. That is arbitrary, but it has to be
deterministic: ties are reachable, because two grid points in one cell can carry identical
`proportion` weights — 185 of the V1 grid's 1671 cells contain such a pair — and an order-dependent
answer would drift with Polars' internals. Be aware which way the bias runs: code 0 is "no
precipitation", so a tied cell leans dry.

The renormalisation is worth most where the corruption is scattered rather than blocky, which is
exactly the shape the de-accumulated variables take. Dividing by 1.0 lets a single bad point null
its whole cell, and because a grid point feeds 4.92 cells on average, that *amplifies* the
corruption on its way in.

Measured on 2025-06-04 00Z — the worst run in the archive by this measure — where 0.014% of
`precipitation_surface`'s grid points and 0.009% of
`downward_short_wave_radiation_flux_surface`'s arrived null beyond lead-0:

| | Null cells in the ingested run |
|---|---:|
| Dividing by 1.0 | 4,394 |
| Renormalised | 339 |

The 339 that remain are cells where *every* contributing point was corrupt — the corruption is
spatially clustered, so a scattered null rate of one part in ten thousand still wipes out whole
cells occasionally. No cell became null that was not null before.

Keep the magnitude in proportion: across the whole archive (862 runs, 6.24 billion rows) only 12
runs carry any de-accumulated null beyond the lead-0 floor at all, totalling 6,550 cells. Most of
the upstream scatter documented above lands outside the small GB box we download. This is
therefore a correctness fix — the estimator was wrong, and a wholly-uncovered cell was a false
zero — rather than a change that recovers much data.

A cell where *no* point contributed is a different case, and it yields **null**, never `0.0`. That
distinction is the whole reason the contributing weight is computed rather than assumed: Polars
sums an all-null group to `0.0`, which for weather is a physically plausible, in-bounds lie
(0 °C, 0 Pa, 0 m s⁻¹) that would pass every check on this page.

## Nulls in the de-accumulated variables (tolerated)

The three de-accumulated variables — `precipitation_surface`,
`downward_short_wave_radiation_flux_surface`, and `downward_long_wave_radiation_flux_surface` —
carry nulls beyond the first forecast step. Dynamical de-accumulates these from ECMWF's cumulative
source fields to instantaneous rates, and the root cause is corrupt source accumulation: some
`(ensemble_member, forecast_step)` fields report physically-impossible *negative* accumulation,
which the de-accumulation step correctly leaves as null rather than silently clamping corrupt data
to zero. This is documented and WONTFIX upstream in
[dynamical-org/reformatters#722](https://github.com/dynamical-org/reformatters/issues/722); a
looser clamp threshold would only convert visibly-null corrupt data into invisibly-zeroed corrupt
data.

Usually the corruption is scattered per-pixel, a few percent of a slice, and most of that is
absorbed by the spatial aggregation above rather than reaching the stored table as nulls at all.
Occasionally a whole
`(ensemble_member, valid_time)` slice arrives null across the grid: the 2026-08-09 00Z run had
`downward_short_wave_radiation_flux_surface` null worldwide for ensemble member 34 at the 354-hour
and 360-hour steps — 2 of that variable's 4284 `(member, step)` slices (51 ensemble members × the
84 steps beyond lead-0).

Both patterns are tolerated at ingest, for two reasons. First, a tolerated gap is a small,
isolated part of one member's trajectory, and it is absorbed the same way the scattered corruption
already is. Be careful with the tempting shorter version of that argument — "these variables are
null at lead-0 anyway, so models handle their nulls" — because it does not quite transfer. Lead-0
nulls reach the model *as nulls* only because they are *leading*, and `_upsample_nwp_to_half_hourly`
interpolates only *interior* nulls; an interior wholly-null slice is bridged from its neighbouring
steps, so the model sees a fabricated value instead. Note the span that bridges: losing one native
step means interpolating between the steps *either side* of it, so 6 hours in the 3-hourly part of
the horizon and 12 in the 6-hourly part. That is acceptable, but it is a different claim from "the
model sees a null and copes" — and it is the reason a *spatial* average over the same step's
neighbouring grid points, which is what the aggregation above does to the scattered nulls, is the
better of the two.

Which of those two a slice gets depends on where it sits in the horizon, and the dswrf example
above is the awkward case: 360 h is the *last* of the 85 steps, and `interpolate()` leaves trailing
nulls alone exactly as it leaves leading ones. So that slice is neither absorbed nor bridged — it
reaches the model as a null, which is benign for a variable that is already null at lead-0 in every
run, but it is not the "absorbed" case. Second, the
run that failing would discard is overwhelmingly good. Take that 2026-08-09 run as the worked example: 0.05% of one
already-nullable variable is not worth the other 4282 slices of that same variable, nor the twelve
other variables that arrived complete, and rejecting it leaves the live forecast on a run 24 hours
older. That is exactly the trade
[principle 7](../design-philosophy/design-principles.md#7-strict-contracts-at-every-boundary)
warns against, in its own words: throwing away an otherwise-good NWP run converts a tolerable
problem into an outage.

`Nwp.validate` therefore permits both patterns, and the `nwp_has_no_unexpected_nulls` asset check
reports them (WARN, non-blocking), naming the affected `(variable, ensemble_member, valid_time)`
slices and counting the wholly-null ones separately from the scattered ones — the two warrant
different responses even though neither fails the run.

### Two populations, counted separately

`nwp_has_no_unexpected_nulls` counts nulls at both stages of ingest, and the metadata keys say which
stage each number came from. The `nwp_grid_point` keys count the raw 0.25° grid, before the
aggregation ([`assess_upstream_grid_point_nulls`](../api/dynamical_data/index.md)); the `h3_cell`
keys count the cells we store afterwards (`assess_nwp_quality`). The same grid-point keys appear on
`nwp_instantaneous_variables_have_no_nulls`, counting the same raw grid over
[the other null population](#the-instantaneous-variables-scattered-nulls-counted-on-the-raw-grid) —
the check name says which is which.

Both are needed because the aggregation deliberately breaks the link between them. Renormalising
over the contributing grid points is what keeps a scattered upstream null out of the stored cells —
and it is therefore also what stops a cell count from measuring the feed. A cell count is the
grid-point rate convolved with our H3 resolution, our grid spacing and our aggregation policy, so a
change to any of those three moves it without anything upstream having changed. Only the grid-point
rate answers the provider question in
[Three audiences, three channels](../design-philosophy/inherent-stability.md#three-audiences-three-channels).

The two are not comparable as rates: different units over different populations. The grid-point
denominator is the whole downloaded lat/lon box, including the corner points no H3 cell uses, which
is what keeps our geometry out of the number. What they do share is the slice filter — both ignore
lead-0. And `null_nwp_grid_point_fraction` pools the three de-accumulated variables, so it does not
equal any single variable's rate: the 2025-06-04 figures above are per-variable, and that run's
pooled fraction is roughly 0.008%, a little over half the 0.014% quoted for
`precipitation_surface`.

Only the cell count drives the check's `passed`. The upstream rate is published on every
materialisation instead, because "is the feed degrading?" is a question about the trend across runs
that no single run can answer, and the archive offers no threshold that separates a healthy feed
from a worsening one.

## A wholly-missing variable, and instantaneous nulls (fatal)

Two null patterns *do* fail ingest:

- **A null in any instantaneous variable** (temperature, dew point, winds, pressures,
  geopotential height). These are never legitimately null, so any null is an anomalous structural
  gap. They stay non-nullable in the `Nwp` contract, so base validation rejects them. This is the
  pattern behind the 2026-07-14 run, where a whole forecast step went missing for 50 of 51
  ensemble members across every variable — reported upstream as
  [dynamical-org/reformatters#765](https://github.com/dynamical-org/reformatters/issues/765).

    A cell reaches this state only when *every* grid point feeding it is missing, because of the
    renormalisation described [above](#spatial-aggregation-is-where-a-grid-points-null-is-resolved).
    A blocky failure like 2026-07-14's is caught exactly as before — the whole grid is gone, so
    every cell is empty — while a *scattered* grid-point null in an instantaneous variable is
    absorbed by the cell's other points, unless it lands on one of the 10 single-point cells, which
    have no others. That is a deliberate trade rather than an oversight: an instantaneous
    variable's nulls have only ever arrived as whole-step dropouts, and losing an entire run over
    one bad pixel is the outage
    [principle 7](../design-philosophy/design-principles.md#7-strict-contracts-at-every-boundary)'s
    granularity clause exists to prevent. What it costs is that the *gate* cannot see scattered
    corruption in a variable that should never carry any, so a separate detector counts it on the
    raw grid instead: the `nwp_instantaneous_variables_have_no_nulls` check, described
    [below](#the-instantaneous-variables-scattered-nulls-counted-on-the-raw-grid).

- **A de-accumulated variable null in *every* slice beyond lead-0 of a run** — the column is absent
  rather than degraded, so `Nwp._check_no_wholly_missing_deaccumulated_variable` raises
  `NwpVariableWhollyMissing`. This is the one place where landing the run is worse than not landing
  it: an all-null weather column would train and serve silently for the full 15-day horizon,
  whereas the previous run is stale but complete. The judgement is made per `init_time`, so a run
  with an empty column is caught even in a frame that also holds healthy runs.

A run that fails ingest writes nothing (validation runs before the Delta write), so there are no
partial partitions to clean up.

### The instantaneous variables: scattered nulls, counted on the raw grid

The gate above judges cells, so it sees an instantaneous variable's corruption only when a whole
cell goes. The `nwp_instantaneous_variables_have_no_nulls` check counts the same corruption where it
arrives, on the raw 0.25° grid, over the nine downloaded variables that are never legitimately null.
Its `passed` is false on a single null grid point — a zero threshold, unlike
`nwp_has_no_unexpected_nulls`, whose nulls are expected — and it still only WARNs, because by the
time it runs the aggregation has already absorbed what it counted. What is at stake is whether we
raise the run with Dynamical.org, not whether we keep it.

The variable set comes from the download list rather than from the `Nwp` contract, because the two
name the winds differently: we download `wind_u_10m`/`wind_v_10m` and derive
`wind_speed_10m`/`wind_direction_10m` from them, so a set drawn from the contract would name four
variables the downloaded dataset does not carry.

Unlike the de-accumulated count, this one includes **lead-0**: these variables are not null there by
design, so a null at lead-0 means what a null at any other step means.

What it cannot see is a null that reached a stored cell, because `Nwp.validate` rejects that run
before any check runs. That covers a blocky failure, where every grid point of a cell goes at once,
and also a scattered null that happens to land on one of the 10 single-point cells. So a run that
lands with this check red is telling you about absorbed scatter — a pattern this project has never
yet seen in an instantaneous variable, whose nulls have only ever arrived as whole-step dropouts.

### A wholly-missing variable is retried, not failed outright

`NwpVariableWhollyMissing` is its own exception type because the `ecmwf_ens` asset **retries** it,
on the same ladder as a run that is not in the catalog yet: every 30 minutes, up to 8 times. (That
is *at least* four hours of waiting — this failure is only detectable after downloading, so each
of those attempts pays for a download too.)
Both mean "the upstream run is not ready yet"; they just say it at different points.

That is worth doing because Dynamical.org publishes each 00Z run as roughly 40 separate Icechunk
commits between 08:05 and 08:20 UTC, one per worker. A run being written is therefore genuinely
readable and genuinely incomplete: a variable whose worker has not committed yet reads as
fill-value null across every member and step, which is precisely this fatal pattern. That covers
less ground than it sounds, and the limit is worth stating: only 3 of the `Nwp` contract's 13
weather variables reach this check at all. Nine are instantaneous and non-nullable, so a
half-published run also missing one of *those* is rejected by base Patito validation first and
fails immediately, with no retry; the thirteenth, `categorical_precipitation_type_surface`, is
nullable but has its own historical invariant that rejects an all-null column just as fast. So the
retry engages when the variables still unwritten are the de-accumulated ones. A *defective*
run also gets republished — the 2026-08-09 00Z run was repaired by a second sweep at 11:45 UTC,
3 hours 25 minutes after its first publication, and well inside the retry budget.

The retry stays deliberately narrow: it covers these two exceptions and nothing else, so a genuine
bug still fails immediately rather than retrying for four hours. The partition simply stays
unmaterialised if every retry is exhausted, until the upstream data is fixed and it is re-run.

## An incomplete run (tolerated, and reported)

The checks above ask whether the rows we received are usable. A separate question is whether we
received *all* the rows: a complete ECMWF ENS run is the full cartesian product of **51 ensemble
members × 85 native forecast steps × every H3 cell in the H3 grid weights**. A run missing an
ensemble member, or stopping short of the 15-day horizon, is short in a way `Nwp.validate` cannot
see — every row it *does* contain is perfectly well-formed — and would otherwise only surface much
later as strange training data.

`assess_nwp_run_completeness` therefore compares the ingested run against that expected shape and
returns an `NwpRunCompletenessReport`. The `ecmwf_ens` asset publishes the report as the
`nwp_run_is_complete` asset check, naming exactly which members and which lead times are absent.

### What it can and cannot detect

The two detections that bite are a **missing ensemble member** and a **missing forecast step**,
because both arrive as a short coordinate on the source dataset:
`convert_nwp_xarray_dataset_to_polars_dataframe` loops over `ds.ensemble_member` and `ds.lead_time`,
so a short coordinate becomes absent rows and the check names it. An off-grid `valid_time` — the
upstream step structure changing under us — is caught the same way.

The **cell count and the total row count cannot fire through today's converter**, and are there as
defence-in-depth rather than as live detections. That converter left-joins the NWP values onto the
H3 grid and then groups by `h3_index`, so its output always carries exactly the cells the grid
weights name, and always as a dense cross-product. The row count is what would catch a *ragged*
run — every member, step and cell present, but some (member, step, cell) combinations absent —
which the three marginal counts all miss. Both would start to matter if that converter were ever
replaced by one that can emit such a frame.

A dropped *grid point* is not covered by this check either, and is worth knowing about, because it
is handled by the aggregation rather than by any check. A point the H3 grid weights name but the
source dataset does not carry misses the left join, and is then excluded from its cell's
contributing weight exactly as an upstream null is — so the cell is renormalised over whatever else
feeds it, and only a cell that loses *every* one of its points comes out null. That takes either a
dropped point feeding one of the 10 single-point cells, or enough neighbouring points dropping
together, which is what a whole missing coordinate slice would look like. The resulting null is
fatal for the instantaneous variables, so it costs the whole partition and shows up as a missed
run — a blunt response to a small cause. Relaxing it to a warning and an absent row is tracked in
[issue #478](https://github.com/openclimatefix/nged-substation-forecast/issues/478).

### Where the expected shape comes from

- **Ensemble members** — `ECMWF_ENS_ENSEMBLE_MEMBERS`, the 51 members (control plus 50 perturbed,
  indexed 0–50) that are a fixed property of the ECMWF IFS ENS configuration.
- **Forecast steps** — `ECMWF_ENS_LEAD_TIME_HOURS`, the 85 native steps: 3-hourly from lead-0 out
  to 144 h, then 6-hourly out to 360 h (= `init_time` + 15 days). Expressed as the step *structure*
  rather than a bare count of 85, so the constant explains itself and a gap can be named in hours.
- **H3 cells** — **not** a constant. The asset passes the distinct cell count of the
  `h3_grid_weights` parquet it has already loaded, so the expectation tracks whatever grid we are
  actually running on. (For the V1 trial area that is 1671 cells, which is why a complete V1
  partition is ~7.24M rows — see [Performance and Scale](performance.md).)

### Why it warns instead of failing the run

An incomplete run is **absent input, not malformed input**, and the two get opposite treatment
under [Inherent Stability](../design-philosophy/inherent-stability.md#the-rules) — rule 1 says never
raise because an input is absent, rule 2 says be liberal about missing inputs and strict about
malformed ones, and rule 6 says asset checks warn rather than block. Concretely, failing the
partition would discard the 50 members we did get; the live forecast would then fall back on
*yesterday's* NWP run, which is a strictly worse degradation than forecasting from a slightly short
run today. So the run lands, the check WARNs, and the missing pieces are named.

One asymmetry is worth naming, because it means an *instantaneous* variable's nulls are judged by
the shape they arrive in rather than by how much data is lost. The 2026-07-14 outage lost a
forecast step for 50 of 51 members, but the rows still existed carrying null temperatures —
*malformed*, so `Nwp.validate` rejects that and the day becomes a missed run. The same loss
arriving as absent *rows* is this check's territory: a whole `ensemble_member` or `lead_time`
coordinate short on the source dataset, where the rest of the run is kept and the gap is named.
The de-accumulated variables do not work that way — their nulls are tolerated by volume, up to the
point where the column is empty — so the asymmetry is confined to the variables that are never
legitimately null at all. Both postures are on the
[degradation ladder](../design-philosophy/inherent-stability.md#failure-modes).

### Completeness is not part of `Nwp.validate`

`assess_nwp_run_completeness` is called from the asset and never from `Nwp.validate`. Validation
runs on arbitrary frames — filtered test fixtures, partition-pruned scans, the single-member reads
that training does — and completeness is false for all of them by construction. It is a property of
one whole ingested run, so it is asked once, at the one place that holds one.

### Shape metadata on every materialisation

The observed `n_ensemble_members`, `n_valid_times`, `n_h3_cells`, `valid_time_min` and
`valid_time_max` are attached to the materialisation metadata on *every* run, not only the failing
ones, so slow drift in the upstream dataset is visible in the Dagster UI's timeline before it ever
becomes a warning.

## `categorical_precipitation_type_surface` before 2024-11-13 (historical)

This variable is all-null for init times on and before 2024-11-12 and populated from the
2024-11-13 00Z run onwards. `Nwp.validate` enforces that split as a hard invariant — it is a fixed
historical fact about the dataset, not a quality quirk, so a violation is always fatal.
