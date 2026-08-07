# Known ECMWF ENS data-quality issues

We ingest ECMWF IFS ENS from [Dynamical.org](https://dynamical.org). Its data carries a few
known, recurring quality quirks. This page records what they are, how we tell them apart, and the
policy the `ecmwf_ens` ingest applies to each. The policy is implemented in
[`contracts.weather_schemas`](../api/contracts/index.md): `Nwp.validate` is the fatal ingest gate,
and two non-fatal reporters sit behind the asset's two WARN checks — `assess_nwp_quality` behind
`nwp_has_no_unexpected_nulls`, and `assess_nwp_run_completeness` behind `nwp_run_is_complete`.

## The guiding principle

We fail a run's ingest only when its data is *structurally* broken, and we tolerate *localised*
corruption that a model can absorb — surfacing it as a warning rather than throwing away an
otherwise-good run. The variable that is affected turns out to be a clean signal for which case we
are in, so the fatal gate needs no magic thresholds.

## Scattered per-pixel nulls in the de-accumulated variables (tolerated)

The three de-accumulated variables — `precipitation_surface`,
`downward_short_wave_radiation_flux_surface`, and `downward_long_wave_radiation_flux_surface` —
sometimes carry scattered, per-pixel nulls beyond the first forecast step. Dynamical de-accumulates
these from ECMWF's cumulative source fields to instantaneous rates, and the root cause is corrupt
source accumulation: some `(ensemble_member, forecast_step)` fields report physically-impossible
*negative* accumulation, which the de-accumulation step correctly leaves as null rather than
silently clamping corrupt data to zero. This is documented and WONTFIX upstream in
[dynamical-org/reformatters#722](https://github.com/dynamical-org/reformatters/issues/722); a
looser clamp threshold would only convert visibly-null corrupt data into invisibly-zeroed corrupt
data.

We tolerate these at ingest for two reasons. First, all three variables are already legitimately
null at lead-0 (the de-accumulation has no previous step to difference against), so every model
must handle their nulls regardless. Second, the corruption is genuinely scattered — empirically a
few percent of a slice at most, never a whole slice — so the run remains overwhelmingly usable.

`Nwp.validate` therefore permits scattered nulls in these variables, and the
`nwp_has_no_unexpected_nulls` asset check reports them (WARN, non-blocking) with the affected
`(variable, ensemble_member, valid_time)` slices, so the quirk stays visible without failing the
run.

## Whole-slice and instantaneous nulls (fatal)

Two null patterns *do* fail ingest, because both mean the data is structurally missing rather than
locally corrupt:

- **A null in any instantaneous variable** (temperature, dew point, winds, pressures,
  geopotential height). These are never legitimately null, so any null is an anomalous structural
  gap. They stay non-nullable in the `Nwp` contract, so base validation rejects them. This is the
  pattern behind the 2026-07-14 run, where a whole forecast step went missing for 50 of 51
  ensemble members across every variable — reported upstream as
  [dynamical-org/reformatters#765](https://github.com/dynamical-org/reformatters/issues/765).

- **A whole-slice null in a de-accumulated variable** — an entire `(ensemble_member, valid_time)`
  slice null across the grid beyond lead-0. Unlike the scattered case above, a wholesale-missing
  field is a structural outage, so `Nwp._check_no_whole_null_deaccumulated_slices` fails it.

A run that fails ingest writes nothing (validation runs before the Delta append), so there are no
partial partitions to clean up; the partition simply stays unmaterialised until the upstream data
is fixed or the partition is re-run.

## An incomplete run (tolerated, and reported)

The checks above ask whether the rows we received are usable. A separate question is whether we
received *all* the rows: a complete ECMWF ENS run is the full cartesian product of **51 ensemble
members × 85 native forecast steps × every H3 cell in the
[H3 grid weights](overview.md)**. A run missing an ensemble member, stopping short of the 15-day
horizon, or dropping a grid cell is short in a way `Nwp.validate` cannot see — every row it *does*
contain is perfectly well-formed — and would otherwise only surface much later as strange training
data.

`assess_nwp_run_completeness` therefore compares the ingested run against that expected shape and
returns an `NwpRunCompletenessReport`. The `ecmwf_ens` asset publishes the report as the
`nwp_run_is_complete` asset check, naming exactly which members and which lead times are absent.

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

The report also compares the total row count against the full grid, which catches a *ragged* run
that the three marginal counts miss: every member, step and cell present, but some
(member, step, cell) combinations absent.

### Why it warns instead of failing the run

An incomplete run is **absent input, not malformed input**, and the two get opposite treatment
under [Inherent Stability](../design-philosophy/inherent-stability.md#the-rules) — rule 1 says never
raise because an input is absent, rule 2 says be liberal about missing inputs and strict about
malformed ones, and rule 6 says asset checks warn rather than block. Concretely, failing the
partition would discard the 50 members we did get; the live forecast would then fall back on
*yesterday's* NWP run, which is a strictly worse degradation than forecasting from a slightly short
run today. So the run lands, the check WARNs, and the missing pieces are named.

That is why the same real-world upstream failure can be fatal or tolerated depending on how it
manifests. When the 2026-07-14 outage below dropped a forecast step, the rows still existed with
null temperatures — malformed, so `Nwp.validate` rejected them. Had the rows simply been *absent*,
this check would have landed the rest of the run and warned. The difference is deliberate, and it
is the missing/malformed split, not an inconsistency.

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
