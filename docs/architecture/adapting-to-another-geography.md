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

What would it take to point this system at a different country — concretely, an
Indian project with 100,000 secondary substations reporting every 15 minutes, where we would
forecast net demand for each substation and *also* estimate how much rooftop solar sits behind it?
None of that solar is metered, and there is no register of how much has been installed.

The answer splits into three quite different pieces.

**Moving to a different country is easy.**  The weather data we already use is a
global product, so India needs no new weather source at all. The parts of the system that handle
spatial gridding, storage, and feature building never learned that they were in Britain. What *is*
British is a thin, well-contained layer: a map outline, a permitted latitude and longitude range,
the names of British substation categories, and the assumption that readings arrive every half
hour rather than every 15 minutes. Turning those from hard-coded facts into settings is a few
weeks of work, not a rewrite.

**Handling 100,000 substations is a real engineering project, but an ordinary one.** It is 40 times
as many substations as we are currently building towards for NGED, and — because the readings
arrive twice as often — around 80 times the volume of forecast data. The main consequence is that
we would have to stop training a separate model for each substation and instead train one model
that learns across all of them at once. That is already on our NGED roadmap for its own reasons,
and at 100,000 sites it becomes clearly the favourable choice, because each part of the model is
supported by far more data. Storage layouts would also need reworking, because a year of forecasts
at that scale runs to tens of terabytes.

**Estimating unmetered solar is the research bet — and it is the same bet we are already making
for NGED.** Separating rooftop solar from underlying demand, with no generation meters and no
capacity register, is exactly the problem described in
[Net-demand disaggregation](../roadmap/disaggregation.md). The method there is designed for
precisely this: it treats a substation's reading as demand minus solar generation, models the solar
physically from sunlight, and infers the installed capacity as an unknown that only ever grows.
It does not need a capacity register.

Qualifications belong alongside that, and they cut both ways. In Britain we plan to use the
*metered* solar farms we can see to calibrate and sanity-check our estimates of the *unmetered*
ones. The Indian brief offers no metered solar inside the dataset, so that anchor would have to
come from outside it — India does have a large, metered, utility-scale solar fleet with published
output, but using it would be new work rather than something we already have. The physical
background is also harder: Indian rooftop panels lose a substantial fraction of their output to
dust between monsoons and recover when the rain washes them, and our method currently assumes
installed capacity only ever grows, so it has no way to express a loss that reverses. High
atmospheric dust also biases the satellite and forecast estimates of sunlight that the whole method
leans on.

Against that, 100,000 sites reporting every 15 minutes would be a far larger and finer-grained
dataset than the 32 sites the method was designed against — more sites means the shared parts of
the model are much better constrained, and 15-minute data separates a solar signal from a demand
signal more cleanly than half-hourly data does. There are also confounders India has and Britain
does not: load shedding and diesel backup generation both break the assumption that underlying
demand moves smoothly with the weather. Load shedding is the dangerous one, because it looks like
demand collapsing for no meteorological reason. Agricultural pumping is a special case worth
calling out as an opportunity rather than a problem: Indian agricultural feeders are largely
segregated and run to a published supply schedule, so a load that would otherwise be invisible is
partly *known in advance*.

**For the bid**, the defensible claim is not "we can forecast substations" — many people can. It is
that OCF already has a designed, written-down method for recovering *unmetered* rooftop solar from
net substation flow without a capacity register, together with a published protocol for proving
whether it actually works ([Evaluating disaggregation](../techniques/disaggregation-evaluation.md)).
The Indian dataset would be a larger and finer-grained proving ground for that method than the one
it was built for, against a harder physical background.

**Rough size of the job:** about one engineer for four to five and a half months to have India
forecasting net demand at scale, with the solar-disaggregation research running alongside and
shared with the NGED work rather than duplicated. That assumes the Indian data arrives in a sane
bulk format; if it has to be polled per-substation across 100,000 sites, that is a separate
workstream we have not costed.

## The scenario

The assessment assumed the following brief:

- ~100,000 secondary substations, each reporting power flow every 15 minutes.
- Forecast **net demand** per substation.
- Additionally forecast **PV** per substation.
- **No PV metering anywhere**, and **no prior on installed capacity** at any site.

## What is already geography-neutral

More is geography-neutral than we expected. None of the following would need to change for India:

- **The weather source.** `dynamical_data` reads the
  `ecmwf-ifs-ens-forecast-15-day-0-25-degree` catalogue
  ([`download.py:62`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py)),
  which is global. The spatial bounds are not hard-coded: they are derived at runtime from the
  minimum and maximum latitude/longitude of whatever H3 grid is passed in, so changing the boundary
  changes the download automatically. The one stated limitation — the slice fails across the
  anti-meridian — does not affect India. Two documented caveats do carry over and bite harder
  there, though (see [Data sources](../roadmap/data-sources.md)): the archive only extends back to
  2024-04-01, which is thin history for a 100,000-site training set, and its radiation is global
  short-wave only with no direct component — a bigger problem for PV disaggregation under heavy
  aerosol load than it is for Britain.
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
| `DISPLAY_TIME_ZONE = "Europe/London"`, asserted in the dashboard's axis titles | `dashboard/forecast_chart.py:40` | Display only, but it is a second hard-coded timezone. |
| H3 resolution 5 (~253 km² per cell) chosen for GB, and reached for via a **private** import from the ingest package | `defs/assets.py:40,141` | The NWP grid resolution currently lives inside `nged_data`; see the `PowerIngest` note [below](#how-we-would-structure-it). |
| `nged_s3_bucket_url` / `_access_key` / `_secret` are **required** settings with no defaults | `contracts/settings.py` | `Settings()` raises for any deployment with no NGED bucket. |

The exercise also surfaced a **latent correctness wart**, though a milder one than it first looks.
`local_utc_offset` is computed as
`(base_utc_offset + dst_offset).dt.total_seconds() // 3600` cast to `Int8`
([`tabular_feature_engineer.py:358`](https://github.com/openclimatefix/nged-substation-forecast/blob/main/packages/ml_core/src/ml_core/features/tabular_feature_engineer.py)),
so it can only ever represent whole-hour offsets. Note that `//` floors rather than truncates, so a
negative fractional offset moves *away* from zero.

In any single-timezone deployment this costs nothing: the feature is constant across the dataset,
so mapping UTC+5:30 to `5` discards no information a model could have used. The genuine failure
mode is **collision** in a mixed-offset deployment — India (+5:30) and Nepal (+5:45) both land on
`5`, silently merging two distinct zones — and, more immediately, legibility: neither the `// 3600`
nor the `Int8` states the whole-hour assumption it depends on. Tracked as
[issue #431](https://github.com/openclimatefix/nged-substation-forecast/issues/431).

## What is hard-wired to half-hourly

The list is narrower than it first appears, because of the duration-based lag design noted above:

| Assumption | Where |
|---|---|
| `validate()` **raises** unless every timestamp has `minute ∈ {0, 30}` | `contracts/power_schemas.py:48` |
| Field descriptions declaring a "30-minute observation period" | `contracts/power_schemas.py:18,25` |
| `stuck_window_periods = 48` (i.e. 24 hours at 30 minutes) | `contracts/settings.py` |
| NWP upsampled to `interval="30m"` | `ml_core/features/_nwp.py:121` |
| The live forecast spine, both its start offset and its step | `ml_core/_production_helpers.py:112,115` |
| A row-count guard assuming "51 members × 14 days × 48 half-hours" | `dashboard/forecast_chart.py` |

The one piece of real design work here is the **feature grammar**: lags are parsed from strings
like `power_lag_24h` into an integer number of hours, so there is currently no way to express a
15-, 30- or 90-minute lag. Generalising that from integer hours to durations is the substantive
change; everything else in the table is a constant.

## The real work is scale, not geography

Two different multipliers matter here, and it is worth keeping them apart. On **series count** —
the axis that governs how many models we train — 100,000 sites is **40×** the V2 design point of
~2,500, which is itself ~78× V1's 32. On **forecast-row volume**, the 15-minute sampling doubles it
again, so the storage and query pressure is around **80×**. Three things break.

**One model per substation stops working.** This is the 40× axis. `XGBoostForecaster.train`
collects the whole population into memory and then loops over `group_by("time_series_id")` in
Python, holding every booster in RAM; `save()` then writes one `.ubj` file per series
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
series × 51 members × 14 days × 96 steps/day ≈ **6.9 billion rows per run**. At the current
6-hourly cadence (4 runs/day) that is of order 10 trillion rows per year — around 18 TB per year at
the measured ~1.8 bytes per row. That is storable on S3, but painful to read. The obvious
mitigation is to persist the thirteen agreed
[delivery quantiles](../roadmap/delivery-tables.md#representation-2-percentiles) rather than raw
ensemble members: one row per `valid_time` instead of 51, so a 51× reduction in rows and roughly 4×
fewer stored values.

**Polars' 32-bit row-index cap stops being an edge case — it becomes a write-path blocker.** As
documented in
[Architecture Overview](overview.md#the-other-hard-ceiling-polars-32-bit-row-index), row counts
silently wrap past 2³² rows, and materialising a single frame of ≥2³² rows is unsupported outright.
At V2 the cap affects one code path (the `metrics` asset's whole-fold collect). Here, **a single
run's 6.9 billion forecast rows exceed the 4.29-billion cap on their own**, so the output of one
inference run could not be materialised as one frame at all. Chunking the write is not an
optimisation at this scale; it is a precondition.

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

- **No metered PV inside the dataset.** The NGED plan uses verified metered generators
  ([Capacity estimation](../roadmap/capacity-estimation.md)) to anchor the harder unmetered
  inference. The brief offers no such anchor. Note the careful wording: India certainly *has*
  metered solar — a large utility-scale fleet with published output — so an external anchor may
  well be obtainable. It would simply be new work, not something the method already assumes.
- **Reversible soiling, which the capacity prior cannot express.** Indian rooftop PV loses a
  substantial share of its output to dust between monsoons and recovers sharply once rain washes
  the panels. But
  [`UniversalSolarFleetNode`](../techniques/differentiable-physics.md#scaling-to-aggregate-fleets-universalsolarfleetnode)
  represents installed capacity as a cumulative sum of non-negative increments — **non-decreasing
  by construction** — precisely because installations only ever add. A loss that reverses is
  structurally inexpressible in that prior. In the GB design the mechanism that absorbs this kind
  of variation is *effective*-capacity tracking, which
  [scopes itself to metered generators](../roadmap/capacity-estimation.md) — the very anchor the
  previous bullet says is missing. The two gaps compound rather than being independent, and
  closing them is real research, not a parameter change.
- **Aerosol and monsoon bias in the irradiance itself.** The Indo-Gangetic Plain carries among the
  world's highest aerosol optical depth, which systematically biases satellite- and NWP-derived
  irradiance, and monsoon convection is poorly resolved at 0.25°. Because installed capacity is
  inferred *from* irradiance, a systematic irradiance bias becomes a systematic capacity bias.
  Worse, the planned high-resolution irradiance source — CM SAF SARAH-3, see
  [Data sources](../roadmap/data-sources.md) — is a Meteosat prime-disc product that does not
  reach India. A replacement would have to be sourced: Meteosat IODC (at 45.5°E, though eastern
  India sits at an oblique viewing angle) or the INSAT-3D series (3D/3DR/3DS).
- **Confounders with no British analogue.** Load shedding and diesel gensets both violate the
  assumption that latent demand is smooth and weather-driven. Load shedding is the dangerous one:
  it resembles a demand collapse uncorrelated with weather, and an unguarded optimiser would
  explain it with phantom solar. Explicit regime detection would need to be budgeted, not bolted
  on. Unmetered agricultural pumping is the happier case — Indian agricultural feeders are largely
  segregated and supplied on a published schedule, which makes a large unmetered load partly
  *observable exogenous information* rather than a pure confounder.

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
  sampling interval, the timezone, the H3 resolution, and the power thresholds — injected rather
  than hard-coded. This is the largest single edit, and it pays for itself by making the British
  assumptions *visible* rather than implicit.
- `nged_data` becomes one of several ingest packages behind a small **`PowerIngest`** protocol whose
  contract is "emit `PowerTimeSeries` and `TimeSeriesMetadata`". Only three modules import
  `nged_data` today, so the boundary is nearly clean already — with one wrinkle worth fixing on its
  own merits: `defs/assets.py:40` imports the **private** `_H3_RESOLUTION` out of the ingest package
  and feeds it to the H3 grid builder, so the NWP spatial resolution currently lives inside the
  DNO-specific ingest code. That constant belongs in the `RegionProfile`.
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

The two exceptions are the `local_utc_offset` whole-hour assumption and the private
`_H3_RESOLUTION` import, both described above. Neither is really a portability concern — they are
ordinary code-quality items that happened to surface here — so both can be fixed on their own
merits whenever convenient, independently of anything on this page.

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
