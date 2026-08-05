# Could this codebase forecast another country?

> **Status: Thought experiment — not planned work.** This page records an assessment made on
> 2026-08-05 (against `main` at `737fb86`) while OCF was preparing a pitch for an innovation
> project in India. There is no GitHub issue for any of it, no roadmap entry, and **no intention to
> refactor this codebase for portability**. We would not start any of this work unless we won that
> bid. Nothing here is a commitment, and no current design decision should be taken "so that India
> would be easier later" — see [Why we are not doing any of this now](#why-we-are-not-doing-any-of-this-now).

This page exists in the same spirit as
[Why Dagster, not Airflow?](why-dagster-not-airflow.md): a question was asked, we did the analysis,
and the reasoning is worth keeping auditable even though the answer is "not now".

## Summary

*This section is written for a reader who understands the energy system but does not read code.*

We were asked what it would take to point this system at a different country — concretely, an
Indian project with 100,000 secondary substations reporting every 15 minutes, where we would
forecast net demand for each substation and *also* estimate how much rooftop solar sits behind it.
None of that solar is metered, and there is no register of how much has been installed.

The answer splits into three quite different pieces.

**Moving to a different country is easy.** This surprised us. The weather data we already use is a
global product, so India needs no new weather source at all. The parts of the system that handle
spatial gridding, storage, and feature building never learned that they were in Britain. What *is*
British is a thin, well-contained layer: a map outline, a permitted latitude and longitude range,
the names of British substation categories, and the assumption that readings arrive every half
hour rather than every 15 minutes. Turning those from hard-coded facts into settings is a few
weeks of work, not a rewrite.

**Handling 100,000 substations is a real engineering project, but an ordinary one.** It is roughly
80 times the scale we are currently building towards for NGED. The main consequence is that we
would have to stop training a separate model for each substation and instead train one model that
learns across all of them at once. That is already on our NGED roadmap for its own reasons, and at
100,000 sites it stops being a compromise and becomes strictly better — far more data supporting
each part of the model. Storage layouts would also need reworking, because a year of forecasts at
that scale runs to tens of terabytes.

**Estimating unmetered solar is the research bet — and it is the same bet we are already making
for NGED.** Separating rooftop solar from underlying demand, with no generation meters and no
capacity register, is exactly the problem described in
[Net-demand disaggregation](../roadmap/disaggregation.md). The method there is designed for
precisely this: it treats a substation's reading as demand minus solar generation, models the solar
physically from sunlight, and infers the installed capacity as an unknown that only ever grows.
It does not need a capacity register.

Two honest qualifications belong alongside that. In Britain we plan to use the *metered* solar
farms we can see to calibrate and sanity-check our estimates of the *unmetered* ones; in India
there would be no metered solar to anchor against, so we would lean harder on satellite-derived
sunlight measurements and on pooling information across sites. Against that, 100,000 sites
reporting every 15 minutes would be a substantially better dataset for this method than the 32
sites it was designed against — more sites means the shared parts of the model are far better
constrained, and 15-minute data separates a solar signal from a demand signal more cleanly than
half-hourly data does. There are also confounders India has and Britain does not: load shedding,
diesel backup generation, and unmetered agricultural pumping all break the assumption that
underlying demand moves smoothly with the weather. Load shedding in particular would need explicit
handling, because it looks like demand collapsing for no meteorological reason.

**For the bid**, the defensible claim is not "we can forecast substations" — many people can. It is
that OCF already has a designed, written-down method for recovering *unmetered* rooftop solar from
net substation flow without a capacity register, together with a published protocol for proving
whether it actually works ([Evaluating disaggregation](../techniques/disaggregation-evaluation.md)).
The Indian dataset would be a better proving ground for that method than the one it was built for.

**Rough size of the job:** about one engineer for four to five months to have India forecasting net
demand at scale, with the solar-disaggregation research running alongside and shared with the NGED
work rather than duplicated. That assumes the Indian data arrives in a sane bulk format; if it has
to be polled per-substation across 100,000 sites, that is a separate workstream we have not costed.

## The scenario

The assessment assumed the following brief:

- ~100,000 secondary substations, each reporting power flow every 15 minutes.
- Forecast **net demand** per substation.
- Additionally forecast **PV** per substation.
- **No PV metering anywhere**, and **no prior on installed capacity** at any site.

## What is already geography-neutral

More than we expected. None of the following would need to change for India:

- **The weather source.** `dynamical_data` reads the
  `ecmwf-ifs-ens-forecast-15-day-0-25-degree` catalogue
  ([`download.py:62`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py)),
  which is global. The spatial bounds are not hard-coded: they are derived at runtime from the
  minimum and maximum latitude/longitude of whatever H3 grid is passed in, so changing the boundary
  changes the download automatically. The one stated limitation — the slice fails across the
  anti-meridian — does not affect India.
- **H3 gridding.** `geo.h3.compute_h3_grid_weights_for_boundary` takes any Shapely geometry. There
  is no Great Britain anywhere in `geo/h3.py`.
- **Storage.** `delta_store` is indifferent to geography; the NWP table is keyed by `h3_index`, so
  its size depends on the area covered, not on which country it is.
- **Feature engineering.** The tabular pipeline is vectorised across time series — `time_series_id`
  is only a join and grouping key — and, importantly, **lags and rolling windows are expressed as
  durations, not as counts of half-hour periods** (`pl.duration(hours=…)`,
  `rolling(period="…h")`). That single decision removes most of what would otherwise make a change
  of reporting interval painful.
- **The model interface.** `BaseForecaster` already documents that an implementation may hold "one
  sub-model per series, a single model spanning many series, or anything in between", so moving to a
  global model needs no base-class change.

## What is hard-wired to Great Britain

All of it sits in a thin layer, and most of it sits in `contracts`.

| Assumption | Where | Consequence for India |
|---|---|---|
| Latitude bounded to 49–61°N, longitude to −9–2°E | `contracts/power_schemas.py:154-162` | Validation **hard-fails** on any Indian coordinate. |
| `licence_area` is `Enum(["EMids"])` | `contracts/power_schemas.py:136` | The tightest single lock. |
| `substation_type` is the GB DNO voltage taxonomy (`BSP`, `GSP`, `Primary`, …) | `contracts/power_schemas.py:148` | Indian secondary substations do not map onto it. |
| `units` is `Enum(["MW", "MVA"])` | `contracts/power_schemas.py:131` | Probably fine, but should be checked against the Indian feed. |
| `LIST_OF_TIME_SERIES_TYPES` — 22 NGED categories, re-exported as the `AllFeatures` enum | `contracts/power_schemas.py` | Propagates into the ML schema. |
| Power bounded to ±1000 MW; `max_mw_threshold` / `min_mw_threshold` sized to GB primaries | `contracts/power_schemas.py`, `contracts/settings.py` | Secondary substations are far smaller; thresholds are meaningless as set. |
| The GB outline | `geo/great_britain/load.py` | Add a sibling region loader; swap one import in `defs/assets.py`. |
| `"Europe/London"` as a bare string literal in the feature engineer | `ml_core/features/tabular_feature_engineer.py:350` | Drives every local-time feature in the champion feature set. |
| `nged_s3_bucket_url` / `_access_key` / `_secret` are **required** settings with no defaults | `contracts/settings.py` | `Settings()` raises for any deployment with no NGED bucket. |

The exercise also surfaced **one genuine latent bug**. `local_utc_offset` is computed as
`(base_utc_offset + dst_offset).dt.total_seconds() // 3600` cast to `Int8`
([`tabular_feature_engineer.py:358`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/ml_core/src/ml_core/features/tabular_feature_engineer.py)).
Britain only ever has whole-hour offsets, so this is correct today and correct for as long as we
only forecast Britain. It would silently truncate Indian Standard Time (UTC+5:30) to +5, and would
be equally wrong for Nepal and parts of Australia.

## What is hard-wired to half-hourly

Narrower than it first appears, because of the duration-based lag design noted above. The complete
list:

| Assumption | Where |
|---|---|
| `validate()` **raises** unless every timestamp has `minute ∈ {0, 30}` | `contracts/power_schemas.py:48` |
| Field descriptions declaring a "30-minute observation period" | `contracts/power_schemas.py:18,25` |
| `stuck_window_periods = 48` (i.e. 24 hours at 30 minutes) | `contracts/settings.py` |
| NWP upsampled to `interval="30m"` | `ml_core/features/_nwp.py:121` |
| The live forecast spine built at `interval="30m"` | `ml_core/_production_helpers.py:115` |
| A row-count guard assuming "51 members × 14 days × 48 half-hours" | `dashboard/forecast_chart.py` |

The one piece of real design work here is the **feature grammar**: lags are parsed from strings
like `power_lag_24h` into an integer number of hours, so there is currently no way to express a
15-, 30- or 90-minute lag. Generalising that from integer hours to durations is the substantive
change; everything else in the table is a constant.

## The real work is scale, not geography

100,000 series at 15-minute resolution is roughly **80× the V2 design point** of ~2,500 series at
half-hourly — and V2 is itself ~78× V1. Three things break.

**One model per substation stops working.** `XGBoostForecaster.train` collects the whole population
into memory and then loops over `group_by("time_series_id")` in Python, holding every booster in
RAM and writing one `.ubj` file per series
([`forecaster.py:124`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py)).
That is fine for 32 series and already strained at 2,500. At 100,000 it is a non-starter, which
forces the **global model** — already planned as
[Global model per `time_series_type`](../roadmap/xgboost-improvements.md) (issue
[#104](https://github.com/openclimatefix/nged-substation-forecast/issues/104)), described there as
"the stepping stone to V2 scale". Its stated prerequisites — per-series target normalisation,
static per-series features, batched training — are exactly what an Indian deployment would need
anyway.

**The storage partitioning needs rework.** `power_time_series` partitions by `time_series_id`
([`assets.py:115`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/src/nged_substation_forecast/defs/assets.py)),
which would mean 100,000 Hive directories and a small-file explosion on every append; it would need
a date-based partition instead. `power_forecasts` partitions only by `(experiment_name, fold_id)`,
with no time or series axis. At the brief's scale a single full-ensemble run is roughly 100,000
series × 51 members × 14 days × 96 steps ≈ **6.9 billion rows per run**, or of order 10 trillion
rows per year — around 18 TB per year at the measured ~1.8 bytes per row. Storable on S3; painful
to read. The obvious mitigation is to persist the agreed delivery quantiles rather than raw
ensemble members, which is roughly a 5× reduction on its own.

**Polars' 32-bit row-index cap stops being an edge case.** As documented in
[Architecture Overview](overview.md#the-other-hard-ceiling-polars-32-bit-row-index), row counts
silently wrap past 2³² rows. At V2 this affects one code path (the `metrics` asset's whole-fold
collect); at the scale in this brief it would be a routine constraint on almost every aggregate.

NWP volume, by contrast, scales with **area, not site count**. India is roughly 15× the land area
of Great Britain, so a full ECMWF ensemble archive would be of order 600 GB per year against
Britain's measured ~40 GB — an estimate by area, not a measurement. The bigger loss is that the
`h3_index` pruning described in
[Architecture Overview](overview.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
stops helping: with 100,000 sites spread across the country, the cells the sites occupy *are* the
whole grid.

## PV disaggregation without capacity priors

This is the part the bid actually turns on, and it is the part that is already designed.

[`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode)
models exactly this object: an aggregate, unmetered solar fleet behind one substation, whose
installed capacity is unknown and is represented as a cumulative sum of non-negative weekly
increments (installations only ever add capacity) with a sparsity penalty, because installs happen
in bursts. It needs no capacity register. The
[convex dictionary baseline](../roadmap/disaggregation.md#the-convex-dictionary-baseline) — fit a
sparse, non-negative, monotonically-growing amount of each of a menu of candidate panel
orientations — needs no capacity prior either, and would be the right first deliverable.

What is genuinely **harder** in India:

- **No metered PV anywhere.** The NGED plan uses verified metered generators
  ([Capacity estimation](../roadmap/capacity-estimation.md)) to anchor the harder unmetered
  inference. That anchor does not exist here. Compensating levers are satellite-derived irradiance
  — Meteosat IODC and INSAT-3D both cover India — and cross-site pooling.
- **Confounders with no British analogue.** Load shedding, diesel gensets, and unmetered
  agricultural pumping all violate the assumption that latent demand is smooth and
  weather-driven. Load shedding is the dangerous one: it resembles a demand collapse uncorrelated
  with weather, and an unguarded optimiser would explain it with phantom solar. Explicit regime
  detection would need to be budgeted, not bolted on.

What is genuinely **easier**:

- **100,000 sites instead of 32.** The design's cross-site strength comes from hierarchical
  parameter sharing — universal basis shapes plus a small per-site style vector. That structure
  improves markedly with more sites; 32 is close to the worst case for it.
- **15-minute data instead of half-hourly.** Finer sampling separates the solar shape from the load
  shape more cleanly, particularly around sunrise and sunset ramps.

## How we would structure it

Recorded for completeness. This is what we would do *if* we won the bid; it is not what we are
doing.

**One monorepo, not a fork.** Forking would mean doing the disaggregation research twice, which is
the single most expensive mistake available here. Instead, promote geography to an explicit seam:

- A **`RegionProfile`** in `contracts` carrying the latitude/longitude bounds, the four enums, the
  sampling interval, the timezone, and the power thresholds — injected rather than hard-coded.
  This is the largest single edit, and it pays for itself by making the British assumptions
  *visible* rather than implicit.
- `nged_data` becomes one of several ingest packages behind a small **`PowerIngest`** protocol whose
  whole contract is "emit `PowerTimeSeries` and `TimeSeriesMetadata`". That boundary is already
  clean — only two modules import `nged_data` today — so this is mostly a renaming exercise.
- `geo/great_britain/` becomes a small region registry.
- Two Dagster code locations over one set of shared packages.

Indicative sizing, and how much is shared with NGED's own V2 work:

| Workstream | Effort | Shared with NGED V2? |
|---|---|---|
| Region seam, 15-minute support, Indian ingest | 4–6 weeks | Seam yes; ingest no |
| Global model, replacing per-series XGBoost | 6–10 weeks | **Yes — needed for V2 regardless** |
| Storage partitioning and metrics chunking at 80× | 6–8 weeks | Mostly |
| Convex dictionary disaggregator | 8–12 weeks | **Yes — it is the V2 baseline** |
| Full differentiable-physics PV engine | 6–12 months | **Yes** |

## Why we are not doing any of this now

Speculative generality is not free. A `RegionProfile` seam introduced today is a layer of
indirection that every NGED contributor pays for, on every change, in service of a project we may
not win. The correct move is to leave the British assumptions hard-coded and *legible* — this page
is a large part of what makes them legible — and to pay the refactoring cost only once there is a
second consumer to amortise it against.

The exception is the `local_utc_offset` truncation described above. That is an ordinary latent bug
rather than a portability concern, and it can be fixed on its own merits whenever it is convenient.

**What would change our mind:**

- **Winning the Indian bid.** The obvious trigger, and the only one that justifies the full seam.
- **Any second DNO or DSO engagement**, British or otherwise. A second British licence area would
  exercise most of the same seam — the enums and the required NGED settings — without the
  resolution or scale work.
- **The scale work becoming necessary on NGED's own merits.** The global model and the storage
  re-partitioning are already on the V2 path. If they land for NGED, the marginal cost of a
  geographic port drops sharply, and this assessment should be re-run rather than trusted.

## See also

- [Net-demand disaggregation](../roadmap/disaggregation.md) — the method that does most of the work
  in the Indian scenario, and the reason a fork would be the wrong structure.
- [Differentiable Physics](../techniques/differentiable-physics.md) — `UniversalSolarFleetNode` and
  the monotone capacity representation.
- [Evaluating disaggregation](../techniques/disaggregation-evaluation.md) — the protocol that turns
  the disaggregation claim into something testable.
- [Architecture Overview](overview.md) — the memory and row-index ceilings that set the scale limits
  quoted here.
- [Why Dagster, not Airflow?](why-dagster-not-airflow.md) — the same genre of assessment, reaching
  the same "not now, and here is what would change that" conclusion.
