# Known ECMWF ENS data-quality issues

We ingest ECMWF IFS ENS from [Dynamical.org](https://dynamical.org). Its data carries a few
known, recurring quality quirks. This page records what they are, how we tell them apart, and the
policy the `ecmwf_ens` ingest applies to each. The policy is implemented in
[`contracts.weather_schemas`](../api/contracts/index.md): `Nwp.validate` is the fatal ingest gate,
and two non-fatal reporters sit behind the asset's two WARN checks — `assess_nwp_quality` behind
`nwp_has_no_unexpected_nulls`, and `assess_nwp_run_completeness` behind `nwp_run_is_complete`.

## The guiding principle

We fail a run's ingest only when a weather column carries **no data at all**, and we tolerate every
smaller gap — surfacing it as a warning rather than throwing away an otherwise-good run. What
decides the case is the *variable*: an instantaneous variable is never legitimately null, so any
null in one is fatal, while the de-accumulated variables are null by design at lead-0 and are
tolerated right up to the point where the column is empty. So the fatal gate needs no magic
thresholds.

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

Usually the corruption is scattered per-pixel, a few percent of a slice. Occasionally a whole
`(ensemble_member, valid_time)` slice arrives null across the grid: on 2026-08-09,
`downward_short_wave_radiation_flux_surface` was null worldwide for ensemble member 34 at the
354-hour and 360-hour steps — 2 of that run's 4284 `(variable, member, step)` slices.

Both patterns are tolerated at ingest, for the same two reasons. First, all three variables are
already legitimately null at lead-0 (the de-accumulation has no previous step to difference
against), so every model must handle their nulls regardless — the nulls are in-distribution.
Second, the run that failing would discard is overwhelmingly good: rejecting the 2026-08-09
partition over 0.05% of one already-nullable variable cost the live forecast the other 4282 slices,
all 13 variables and all 51 ensemble members, leaving it to forecast from a run 24 hours older.
That is exactly the trade
[principle 7](../design-philosophy/design-principles.md#7-strict-contracts-at-every-boundary)
warns against, in its own words: throwing away an otherwise-good NWP run converts a tolerable
problem into an outage.

`Nwp.validate` therefore permits both patterns, and the `nwp_has_no_unexpected_nulls` asset check
reports them (WARN, non-blocking), naming the affected `(variable, ensemble_member, valid_time)`
slices and counting the wholly-null ones separately from the scattered ones — the two warrant
different responses even though neither fails the run.

## A wholly-missing variable, and instantaneous nulls (fatal)

Two null patterns *do* fail ingest, because in both the column carries no weather at all:

- **A null in any instantaneous variable** (temperature, dew point, winds, pressures,
  geopotential height). These are never legitimately null, so any null is an anomalous structural
  gap. They stay non-nullable in the `Nwp` contract, so base validation rejects them. This is the
  pattern behind the 2026-07-14 run, where a whole forecast step went missing for 50 of 51
  ensemble members across every variable — reported upstream as
  [dynamical-org/reformatters#765](https://github.com/dynamical-org/reformatters/issues/765).

- **A de-accumulated variable null in *every* slice beyond lead-0** — the column is absent rather
  than degraded, so `Nwp._check_no_wholly_missing_deaccumulated_variable` raises
  `NwpVariableWhollyMissing`. This is the one place where landing the run is worse than not landing
  it: an all-null weather column would train and serve silently for the full 15-day horizon,
  whereas the previous run is stale but complete.

A run that fails ingest writes nothing (validation runs before the Delta append), so there are no
partial partitions to clean up.

### A wholly-missing variable is retried, not failed outright

`NwpVariableWhollyMissing` is its own exception type because the `ecmwf_ens` asset **retries** it,
on the same ladder as a run that is not in the catalog yet: every 30 minutes for up to 4 hours.
Both mean "the upstream run is not ready yet"; they just say it at different points.

That is worth doing because Dynamical.org publishes each 00Z run as roughly 40 separate Icechunk
commits between 08:05 and 08:20 UTC, one per worker. A run being written is therefore genuinely
readable and genuinely incomplete: a variable whose worker has not committed yet reads as
fill-value null across every member and step, which is precisely this fatal pattern. A *defective*
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

A dropped *grid point* is not covered by this check at all, and is worth knowing about: because the
left join misses and the weighted `sum` over an all-null group returns `0.0`, a dropped point lands
as a plausible-looking all-zero cell (0 °C, 0 Pa, 0 m s⁻¹) rather than as an absent one. That is
inside physical bounds, so `Nwp.validate` accepts it and neither non-fatal check sees it.

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

One asymmetry survives, and it is worth naming. When the 2026-07-14 outage above lost a forecast
step for 50 of 51 members, the rows still existed carrying null temperatures — *malformed*, so
`Nwp.validate` rejected them and the day became a missed run. Had the same outage arrived as absent
rows instead, this check would have kept the run and merely warned. An instantaneous variable's
nulls are therefore still judged by the shape they arrive in rather than by how much data is lost;
what is no longer true is that the de-accumulated variables work that way, since they are now
tolerated up to the point where the column is empty. This check covers the absent-rows shape: a
whole `ensemble_member` or `lead_time` coordinate short on the source dataset, and the rest of the run
is kept. The two postures are deliberate, and the fatal one is already recorded as such on the
[degradation ladder](../design-philosophy/inherent-stability.md) ("a whole ECMWF slice corrupt …
manifests downstream as a missed run").

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
