# Which weather product best describes past sunshine?

Several parts of this project read an estimate of weather that has already happened, not a forecast.
Capacity estimation infers a generator's size from how its output tracks sunshine. Training targets
and pre-training need a long history of sunshine to learn from. Historical features give a model the
weather of hours already past. Disaggregation separates hidden solar generation from demand at a
substation. Each of these consumers has to choose one weather product to read. This page measures
how well six products describe past sunshine at six metered solar farms, and says which product each
consumer should read.

**A satellite retrieval describes past sunshine far better than any weather model tested.** The
Copernicus Atmosphere Monitoring Service's retrieval (CAMS) has a mean absolute error of 5.05% of
each generator's capacity, against 7.71% for the best weather model and 8.98% for ERA5, the
reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF). Every error on this
page is a mean absolute error as a percentage of each generator's own 99th percentile of output,
which the page calls its capacity. **Among weather models, the German weather service's ICON-D2
(from its Icosahedral Nonhydrostatic model family) is the best as served, and its advantage fades
within hours of each run.** **ICON-EU beats ICON global as served, by 0.09 points, and most of that
gap sits in the hours ICON global is served further ahead.** **ICON-EU and the Met Office's UK
variable-resolution model (UKV) cannot be told apart once UKV's hour is rebuilt from its own
snapshots.** The evidence is six metered solar farms inside one 25 km by 23 km box in Lincolnshire,
and 79,384 generator-hours from December 2022 to September 2026.

## The six products

**The products differ in how far ahead each value was forecast and in what area they cover, as well
as in accuracy, and both properties matter to a consumer.** A weather model's archived value for an
hour is a forecast made a few hours earlier. How many hours earlier is the served lead. ICON-D2 does
not cover South West England or South Wales, which are inside NGED's licence area.

| Product | What it is | Served lead | Covers all of Great Britain? | Grid | Usable from | Available after |
|---|---|---|---|---|---|---|
| CAMS | Satellite retrieval from Meteosat | no forecast step | yes | point values | 2004 | about 1 day |
| ERA5 | ECMWF reanalysis | 1 to 12 hours | yes | about 31 km | 1940 | about 5 days |
| UKV | Met Office weather model | T+0, the analysis | yes | 2 km | March 2022, a backfill from an unnamed source until August 2024 | about 4 hours |
| ICON-D2 | German weather service (DWD) model, central Europe | 1 to 3 hours | no | 2 km | December 2022 | about 1.5 hours |
| ICON-EU | DWD model, Europe | 1 to 3 hours | yes | 7 km | November 2022 | about 3.5 hours |
| ICON global | DWD model, global | 1 to 6 hours | yes | 11 km | November 2022 | about 3.5 hours |

**ICON-D2's western edge runs from about 2°W on the south coast to about 2.5°W in the Midlands.**

**The latencies come from each service's own documentation:** CAMS's [radiation-service
notes](https://confluence.ecmwf.int/x/jOLjDw), the [ERA5 dataset
page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview),
Open-Meteo's [UKV documentation](https://open-meteo.com/en/docs/ukmo-api), and the publication times
of the German weather service's own open-data files.

Each served lead comes from how the product's archive is built:

- **ERA5's hourly radiation is not an analysis.** It comes from the reanalysis's own short
  forecasts, started at 06 and 18 UTC, at steps of 1 to 12 hours ([ERA5
  documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)).
- **Open-Meteo's weather-model archive keeps, for each hour, the freshest run that covers it.** For
  UKV, which runs every hour, that is the T+0 analysis. `verify_ukv_lineage.py` matched it against
  the Met Office's own files to within 0.55 W m⁻², over the period since August 2024.
- **For ICON-EU, the freshest run is measured.** `verify_icon_lineage.py` reconstructed each hour
  from 08 to 16 UTC on one day, at one place, from every run the German weather service still
  published. At all nine hours the freshest run reproduced Open-Meteo's served value to within
  1 W m⁻².
- **For ICON-D2 the same check does not confirm the mapping on its own.** The freshest run was the
  closest match at 7 of 9 hours, but still differed by up to 44 W m⁻². The 3-hour pattern in
  ICON-D2's own errors, described below, is the stronger evidence.
- **ICON global's lead is inferred from its 6-hourly cycle.** It is published only on a grid the
  check cannot read.

**SARAH-3, the Surface Solar Radiation Data Set from the Satellite Application Facility on Climate
Monitoring, is not here.** It is a second satellite retrieval, available only by manual order from
the European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT), which
[#806](https://github.com/openclimatefix/nged-substation-forecast/issues/806) tracks.

## How the comparison was made

**Every product is scored on the same hours, with the same model and the same features, so a
difference between two products is a difference between their irradiance alone.**

- **One row set.** An hour is kept only if all six products cover it. An hour holding a zero
  half-hour of metered power is dropped, whatever any product says about it. So no product's own
  values decide which hours are scored. CAMS is read in full, not only on the hours it rates as
  reliable.
- **One model per generator.** A gradient-boosted tree (XGBoost) is fitted per generator on one
  product's global horizontal irradiance plus sun position, season, time of day, and ERA5's air
  temperature. Every product is shown the same temperature.
- **Folds that respect an upgrade.** The Met Office upgraded UKV on 21 January 2026. The record is
  cut into five blocks of whole months separately before and after that date, so every held-out
  block is scored by a model trained on both versions. Every model is also told which side of the
  date an hour falls. The 11 days from 21 to 31 January, which carry the pre-upgrade month label,
  are dropped.
- **One normalisation.** Each hour's error is divided by its own generator's capacity before any
  mean or difference.
- **Intervals from whole months.** The six generators share their weather, so each 95% interval
  comes from resampling whole calendar months, and one of three fitting seeds, 2,000 times.
- **Four contrasts named before the run.** The ranking rests on four contrasts, fixed before any
  result was seen: CAMS against ICON-D2, ICON-EU against ICON-D2, ICON-EU against UKV, and ICON
  global against ICON-EU. Every other figure on this page is exploratory, and the UKV snapshot
  comparisons were added after the first run.

**The machinery is `studies.cross_validation` and `studies.bootstrap`, both tested.**

## CAMS describes past sunshine best, by a wide margin

**The satellite retrieval beats the best weather model by 2.65 points [2.42, 2.88], and the gap
holds at every generator, in every season, and in each calendar year from 2023 to 2026.**

| Product | Mean absolute error, % of capacity |
|---|---|
| CAMS | 5.05 |
| ICON-D2 | 7.71 |
| ICON-EU | 8.30 |
| ICON global | 8.39 |
| UKV | 8.79 |
| ERA5 | 8.98 |

Across the six generators, CAMS's lead over ICON-D2 ranges from 2.28 to 3.36 points. By season it
ranges from 1.80 points in winter to 3.02 in autumn, and in each calendar year from 2023 to 2026
(2026 to 10 September) it lies between 2.58 and 2.84.

**The gap is not an artefact of lead or of reading CAMS in full.** On the
hours ICON-D2 is served 1 hour after its run started, CAMS still beats it by 2.23 points. On the
hours CAMS rates as reliable, its lead over ERA5 widens from 3.92 to 4.36 points, which is why the
comparison reads CAMS in full.

## ICON-D2 is the best weather model as served, and its advantage fades within hours of each run

**ICON-D2 beats ICON-EU across the record by 0.60 points [0.46, 0.71], but by 0.98 points in the
first hour after each run and by only 0.30 points in the third hour.** ICON-D2 and ICON-EU run on
the same 3-hourly cycle, so at any given hour both are served at the same lead, and their contrast
at that hour is lead-matched.

| Hour (UTC) | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|
| Served lead of both | 3 h | 1 h | 2 h | 3 h | 1 h | 2 h | 3 h | 1 h |
| ICON-D2 − ICON-EU (points) | −0.35 | −1.27 | −0.69 | −0.23 | −1.44 | −0.72 | −0.43 | −0.94 |

Hours 09 to 16 UTC are shown; the lead figures above use 07 to 19 UTC.

**The time of day does not explain the pattern around noon.** The hours 12 and 13 UTC sit on either
side of solar noon, yet
ICON-D2's advantage is 0.23 points at 12 UTC, 3 hours into a run, against 1.44 points at 13 UTC, 1
hour in. ICON-EU shows a much weaker pattern with lead. The mechanism is not identified. The six
generators sit roughly 150 to 250 km inside ICON-D2's western boundary, where air arriving on a
westerly wind carries ICON-EU's weather into the ICON-D2 domain within hours, so the rate of decay
may not hold elsewhere. By a 3-hour lead only 0.30 points of the advantage remain, so the as-served
advantage should not be assumed to hold a day ahead;
[#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810) measures that.

## ICON-EU against ICON global and UKV

**Most of ICON global's 0.09-point deficit to ICON-EU sits in the hours it is served further
ahead.** ICON global runs every 6 hours, so half its hours are served 4 to 6 hours after the run
started, where ICON-EU is at 1 to 3 hours. On those hours ICON global is 0.18 points behind. On the
hours where the two leads are equal ICON global is 0.02 points behind [−0.03, +0.08]. So no effect
of the coarser grid is detected, though an effect as large as 0.08 points is not excluded.

**ICON-EU beats UKV's served hourly value by 0.48 points [0.30, 0.66], and the whole gap disappears
once UKV's hour is rebuilt from its own snapshots.** ICON serves each hour as a mean over the hour.
UKV publishes a snapshot each hour, and Open-Meteo builds UKV's hourly value from the snapshot at
the hour's end, rescaled by the change in the sun's angle. Averaging UKV's snapshots at both ends of
the hour cuts its error by 0.60 points, which leaves ICON-EU 0.12 points behind [−0.06, +0.30].

**Showing a model the neighbouring hours helps both products by about 0.2 points, which is more than
the gap between them.** A model shown ICON-EU's hourly mean for the hour before and after, as well
as the hour itself, improves by 0.18 points. The same context improves UKV's rebuilt hour by 0.21
points. With context on both, ICON-EU is 0.14 points behind [−0.03, +0.31]. Every UKV construction
scores a little ahead of ICON-EU, but UKV is served at T+0 and ICON-EU 1 to 3 hours ahead, so equal
leads would if anything favour ICON-EU. On the 8 months after UKV's upgrade alone, ICON-EU is 0.01
points behind rebuilt UKV [−0.36, +0.36].

**UKV against ERA5 is unresolved.** Across the record UKV is 0.19 points ahead [0.03, 0.36], but UKV
is served at T+0 and ERA5 at 1 to 12 hours, so the shorter-lead product wins, and equalising the
leads could narrow the gap. On the hours since Open-Meteo's own UKV downloader started in August
2024, UKV is 0.11 points ahead [−0.07, +0.32]. After the January 2026 upgrade the evidence
conflicts. The pooled model scores UKV 0.18 points worse than ERA5 [+0.05, +0.42], while models
fitted on the post-upgrade months alone find UKV 0.11 points ahead [−0.21, +0.45]. Eight months
cannot settle which is better after the upgrade.

## A product's own direct beam adds little

**Every product except ERA5 improves by 0.03 to 0.11 points when shown its own published direct beam
instead of a separation model's estimate from its own total.** The separation model is the [Erbs et
al. (1982)](https://doi.org/10.1016/0038-092X(82)90302-4) correlation, applied to each product's own
global irradiance. CAMS improves by 0.08 points and UKV by 0.11. The three ICON products improve by
0.03 to 0.05 points each. ERA5 shows no detectable effect. These effects are smaller than every
difference between products except ICON global against ICON-EU. For CAMS and ERA5 they agree with
[the beam/diffuse study](beam-diffuse-split.md), which tested whether the published beam carries
information or merely encodes the total differently.

## The ranking holds for a generator trained on its neighbours

**A model trained on five generators and applied to the sixth ranks the products the same way, at a
cost of 0.06 to 0.18 points.** Each held-out generator is predicted by a model that never saw that
generator and never saw the held-out months at any generator. The neighbours share their weather,
so letting a model train on the same months elsewhere would let it learn each day's outcome rather
than transfer. The prediction is converted back to megawatts with the generator's own capacity, so
this arm assumes the capacity is known and says nothing about capacity estimation itself. The
training generators all sit within 25 km of the held-out one, so the cost here is a lower bound for
a generator elsewhere. For disaggregation, the result supports the ranking but not the size of the
error.

## Implied capacity from month to month

**Once the seasonal cycle is removed, CAMS's implied capacity is the steadiest from month to month,
but CAMS swings the most with the seasons.** A month's implied capacity is the ratio of metered
output to what a fixed south-facing panel at 30° tilt predicts per megawatt from each product. With
each calendar month's average removed, CAMS's month-to-month spread is 6.5%, against 7.9% to 10.0%
for the weather models and ERA5. For each of them the interval on the difference excludes zero. In
December, however, CAMS implies a capacity 23% below its annual mean, against 4% to 11% for the
weather models and ERA5. This study cannot say which product is right about December.

## Which product each consumer should read

**These recommendations rest on regional evidence from one part of Lincolnshire.** They weigh
accuracy against availability and coverage.

- **Capacity estimation: CAMS, fitted over at least a year of history or with November to January
  left out.** Every accuracy figure on this page follows a per-generator recalibration that capacity
  estimation cannot make, so the evidence for this consumer is the implied-capacity measure alone.
  Before any production decision, the project's own capacity estimator should be run with each
  product over rolling windows and scored against known capacity.
- **Training targets and pre-training: CAMS for history, with a caveat.** CAMS gives the most
  accurate description of past sunshine of the six products. A model pre-trained on CAMS and then
  served on a weather forecast meets a different product at inference, and this page does not
  measure what that mismatch costs.
- **Historical features in the live service: ICON-EU or UKV, with UKV's two snapshots averaged, and
  with the neighbouring hours included.** CAMS arrives a day late and ERA5 about 5 days late, so
  neither can supply the last few hours at run time. ICON-D2 is more accurate where it covers, so
  ICON-D2 at generators east of its edge and ICON-EU elsewhere is an option, at the cost of mixing
  products across generators. UKV's history spans a backfill from an unnamed source, a live product,
  and the January 2026 upgrade, so a model trained on it mixes three versions.
- **Disaggregation: CAMS where its one-day delay allows, otherwise ICON-EU or rebuilt UKV.** The
  ranking holds for a generator predicted from its neighbours.

## What this does not show

- **The evidence is regional.** All six generators sit in one 25 km by 23 km box, inside ICON-D2's
  domain and in only two ERA5 grid cells. The ranking is untested elsewhere in Great Britain. Wind
  is measured separately on [Which weather product best describes past
  wind?](weather-products-for-past-wind.md).
- **The comparison is not lead-equal.** The served lead is part of what a consumer receives, so the
  as-served ranking answers the consumer's question. A comparison at a held-equal lead is a forecast
  comparison, and belongs to #810.
- **Every accuracy figure is recalibrated per generator.** A product with a large but stable bias
  scores well here. The implied-capacity measure is the only evidence on raw bias.
- **The figures rest on today's capacity table.** Each generator's capacity is its 99th percentile
  of output from the `effective_capacity` table as rebuilt in September 2026.
  [#825](https://github.com/openclimatefix/nged-substation-forecast/issues/825) may move the
  absolute figures, though not the contrasts between products.

## Reproducing the figures

Build every product's dataset, then run the study:

```bash
for source in open-meteo ukv icon-d2 icon-eu icon-global cams; do
  uv run python studies/beam_diffuse_split/build_dataset.py --source $source
done
uv run python studies/beam_diffuse_split/build_dataset.py --source cams \
  --min-cams-reliability 0 --suffix _allhours
uv run python studies/beam_diffuse_split/weather_products.py
```

The report lands in `data/studies/beam_diffuse_split/beam_diffuse_weather_products/report.md`. The
served-lead check runs with `uv run --with cfgrib python
studies/beam_diffuse_split/verify_icon_lineage.py --model icon-eu`, against the runs the German
weather service still publishes, which cover about one day.
