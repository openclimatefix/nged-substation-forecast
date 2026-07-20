# Known ECMWF ENS data-quality issues

We ingest ECMWF IFS ENS from [Dynamical.org](https://dynamical.org). Its data carries a few
known, recurring quality quirks. This page records what they are, how we tell them apart, and the
policy the `ecmwf_ens` ingest applies to each. The policy is implemented in
[`contracts.weather_schemas`](../api/contracts/index.md): `Nwp.validate` is the fatal ingest gate,
and `assess_nwp_quality` is the non-fatal reporter behind the `nwp_has_no_unexpected_nulls` asset
check.

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

## `categorical_precipitation_type_surface` before 2024-11-13 (historical)

This variable is all-null for init times on and before 2024-11-12 and populated from the
2024-11-13 00Z run onwards. `Nwp.validate` enforces that split as a hard invariant — it is a fixed
historical fact about the dataset, not a quality quirk, so a violation is always fatal.
