# Which weather product best describes past sunshine?

Several parts of this project read an estimate of weather that has already happened, not a
forecast. Capacity estimation infers a generator's size from how its output tracks sunshine.
Training targets and pre-training need a long history of sunshine to learn from. Historical features
give a model the weather of hours already past. Disaggregation separates hidden solar generation from
demand at a substation. Each of these consumers has to choose one weather product to read. This page
measures how well six products describe past sunshine at six metered solar farms, and says which
product each consumer should read.

**A satellite retrieval describes past sunshine far better than any weather model tested.** The
Copernicus Atmosphere Monitoring Service's retrieval (CAMS) leaves 5.05% of each generator's output
unexplained, against 7.71% for the best weather model and 8.98% for the reanalysis. Every error on
this page is a mean absolute error, expressed as a percentage of each generator's own 99th percentile
of output. **Among weather models, the German weather service's ICON-D2 is best as served, and part
of its lead comes from being re-run every 3 hours.** **ICON-EU, the Met Office's UKV, and ICON global
cannot be told apart once each is compared on equal terms.** Equal terms means UKV's hour built the
same way as ICON's, and ICON global at the same forecast lead as ICON-EU. The evidence is 6 metered
solar farms inside one 25 km by 23 km box in Lincolnshire, and 79,384 hours from December 2022 to
September 2026.

## The six products

**The products differ in how far ahead each value was forecast and in what area they cover, as well
as in accuracy, and both properties matter to a consumer.** A weather model's archived value for an
hour is a forecast made a few hours earlier. How many hours earlier is the served lead. ICON-D2 does
not cover South West England or South Wales, which are inside NGED's licence area.

| Product | What it is | Served lead | Covers all of Great Britain? | Grid | Usable from | Available after |
|---|---|---|---|---|---|---|
| CAMS | Satellite retrieval from Meteosat | no forecast step | yes | point values | 2004 | about 1 day |
| ERA5 | ECMWF reanalysis | 1 to 12 hours | yes | about 31 km | 1940 | about 5 days |
| UKV | Met Office weather model | T+0, the analysis | yes | 2 km | March 2022 | about 4 hours |
| ICON-D2 | German weather service (DWD) model, central Europe | 1 to 3 hours | no: stops near 2.5°W | 2 km | December 2022 | about 1.5 hours |
| ICON-EU | DWD model, Europe | 1 to 3 hours | yes | 7 km | November 2022 | about 3.5 hours |
| ICON global | DWD model, global | 1 to 6 hours | yes | 11 km | November 2022 | about 3.5 hours |

Each served lead comes from how the product's archive is built:

- **ERA5's hourly radiation is not an analysis.** It comes from the reanalysis's own short forecasts,
  started at 06 and 18 UTC, at steps of 1 to 12 hours
  ([ERA5 documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)).
- **Open-Meteo's weather-model archive keeps, for each hour, the freshest run that covers it.** For
  UKV, which runs every hour, that is the T+0 analysis. `verify_ukv_lineage.py` matched it against
  the Met Office's own files to within 0.55 W m⁻².
- **For ICON-EU, the freshest run is measured.** `verify_icon_lineage.py` reconstructed each daytime
  hour from every run the German weather service still published. The freshest run reproduced
  Open-Meteo's served value to within 1 W m⁻² at every hour from 08 to 15 UTC.
- **For ICON-D2 the same check was consistent but not exact.** The freshest run was the closest
  match at 7 of 9 hours. The 3-hour pattern in ICON-D2's own errors, described below, corroborates
  it across the whole record.
- **ICON global's lead is inferred from its 6-hourly cycle.** It is published only on a grid the
  check cannot read.

The table's latencies come from the upstream services and from Open-Meteo's documentation; the
sources are in the study's pull request. SARAH-3, a second satellite retrieval, is not here: it is
available only by manual order from EUMETSAT, which
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
- **Folds that respect an upgrade.** The Met Office upgraded UKV on 21 January 2026. The record is cut
  into five blocks of whole months separately before and after that date, so every held-out block is
  scored by a model trained on both versions. Every model is also told which side of the date an hour
  falls. The ten days after the upgrade, which are labelled with the pre-upgrade month, are dropped.
- **One normalisation.** Each hour's error is divided by its own generator's capacity before any mean
  or difference.
- **Intervals from whole months.** The six generators share their weather, so each 95% interval comes
  from resampling whole calendar months, and one of three fitting seeds, 2,000 times.
- **Four contrasts named before the run.** The recommendations rest on four contrasts, fixed before
  any result was seen. Every other figure on this page is exploratory.

The machinery is `studies.cross_validation` and `studies.bootstrap`, both tested. The study script
is `studies/beam_diffuse_split/weather_products.py`.

## CAMS describes past sunshine best, by a wide margin

**The satellite retrieval beats the best weather model by 2.65 points [2.42, 2.88], and the gap holds
at every generator, in every season, and in every year.**

| Product | Error, global irradiance only |
|---|---|
| CAMS | 5.05 |
| ICON-D2 | 7.71 |
| ICON-EU | 8.30 |
| ICON global | 8.39 |
| UKV | 8.79 |
| ERA5 | 8.98 |

The gap is not a matter of CAMS being served without a forecast lead. On the hours ICON-D2 is served
at its shortest lead of 1 hour, CAMS still beats it by 2.23 points. Restricting the comparison to the
hours CAMS rates as reliable widens CAMS's lead over ERA5 from 3.92 to 4.36 points, which is why the
comparison reads CAMS in full.

## ICON-D2 is the best weather model, partly because it is re-run every 3 hours

**ICON-D2 beats ICON-EU at every matched lead, but by much more at its freshest hours: 0.98 points at
a 1-hour lead, falling to 0.30 at 3 hours.** ICON-D2 and ICON-EU run on the same 3-hourly cycle, so
at any given hour both are served at the same lead, and their contrast is lead-matched. Across the
record, ICON-D2 beats ICON-EU by 0.60 points [0.46, 0.71].

The pattern shows in each hour's error against ERA5. At 10, 13, and 16 UTC, ICON-D2 is served 1 hour
after its run started and beats ERA5 by 1.76, 2.47, and 2.22 points. At the 3-hour-lead hours
before each of those, 09, 12, and 15 UTC, it beats ERA5 by 0.47, 1.12, and 1.75 points. ICON-EU shows
no such pattern. A 3-hour cycle in ICON-D2's error means ICON-D2's advantage decays within hours of
each run. That decay matters to a consumer: the advantage an archive shows does not carry over to a
forecast made a day ahead, which is
[#810](https://github.com/openclimatefix/nged-substation-forecast/issues/810)'s question.

## ICON-EU, UKV and ICON global cannot be told apart on equal terms

**ICON-EU beats UKV's served hourly value by 0.48 points [0.30, 0.66], and the gap comes from how
UKV's hour is built rather than from UKV's weather.** ICON serves each hour as a mean over the hour.
UKV publishes a snapshot each hour, and Open-Meteo builds UKV's hourly value from the snapshot at the
hour's end, rescaled by the change in the sun's angle. Averaging UKV's own snapshots at both ends of
the hour cuts UKV's error by 0.60 points. Against that average, ICON-EU is no better: +0.12
[−0.06, +0.30]. So a consumer reading UKV should average its two snapshots, and on that footing
neither ICON-EU nor UKV is the better model for the whole of Great Britain.

Showing the model both UKV snapshots separately does better still, beating ICON-EU by 0.23 points
[0.05, 0.41]. That result was found after the first run and is exploratory.

**ICON global's 0.09-point deficit to ICON-EU is its longer lead, not its coarser grid.** ICON global
runs every 6 hours, so half its hours are served at a lead of 4 to 6 hours, where ICON-EU is at 1 to
3 hours. On the hours where the two leads are equal, ICON global is +0.02 points [−0.03, +0.08]
behind ICON-EU, which is no detectable difference. On the longer-lead hours it is 0.18 points behind.

**UKV beats ERA5 across the record by 0.19 points [0.03, 0.36], but the comparison cannot rule out
the lead.** UKV is served at T+0 and ERA5 at 1 to 12 hours, so the shorter-lead product wins, and
equalising the leads could narrow the gap. After UKV's upgrade in January 2026, fitting on the
post-upgrade months alone gives no detectable difference between the two: −0.11 [−0.45, +0.21].
That post-upgrade span is eight months, so its intervals rest on only eight independent blocks.

## A product's own direct beam adds little

**Every product except ERA5 improves by 0.03 to 0.11 points when shown its own published direct beam
instead of a separation model's estimate from its own total.** The separation model is the 1982
Erbs correlation, applied to each product's own global irradiance. CAMS improves by 0.08 points and
UKV by 0.11. The three ICON products improve by 0.03 to 0.05 points each. ERA5 shows no detectable
effect. These effects are one to two orders of magnitude smaller than the differences between
products. For CAMS and ERA5 they agree with
[the beam/diffuse study](beam-diffuse-split.md), which settled whether the published beam carries
information or merely encodes the total differently.

## The ranking holds for a generator with no metered history

**A model trained on five generators and applied to the sixth ranks the products the same way, at a
cost of 0.06 to 0.18 points.** Each held-out generator is predicted by a model that never saw that
generator and never saw the held-out months at any generator. The neighbours share their weather,
so letting a model train on the same months elsewhere would let it learn each day's outcome rather
than transfer. The prediction is a fraction of capacity, converted back to megawatts with the
generator's own capacity, so this arm assumes the capacity is known. That is the situation
disaggregation and capacity estimation are in, except for the capacity itself.

## Stability over time

**CAMS's ratio of power to irradiance is the steadiest from month to month.** Its median monthly
change is 8.0%, against 8.9% to 9.9% for the weather models. A consumer reading irradiance with no
model fitted to the generator, as capacity estimation does, needs a product whose bias holds still.
Season moves every product's ratio, so these figures compare products rather than measure a
product's absolute stability.

## Which product each consumer should read

These recommendations weigh accuracy against availability and coverage. They are regional evidence
from one part of Lincolnshire.

- **Capacity estimation: CAMS.** It is the most accurate, the steadiest, and has history back to 2004.
  Its one-day delay does not matter to an estimate made from weeks of history.
- **Training targets and pre-training: CAMS for history, with a caveat.** CAMS gives the truest
  description of past sunshine. A model pre-trained on CAMS and then served on a weather forecast
  meets a different product at inference, and this page does not measure what that mismatch
  costs.
- **Historical features in the live service: ICON-EU or UKV, with UKV's two snapshots averaged.**
  CAMS arrives a day late and ERA5 about five, so neither can supply the last few hours at run time.
  ICON-D2 is more accurate but cannot serve generators west of about 2.5°W. ICON-EU and UKV cover the
  whole of Great Britain and are indistinguishable on equal terms.
- **Disaggregation: CAMS where its latency allows, otherwise ICON-EU or averaged UKV.** The ranking
  holds for a generator with no metered history, which is disaggregation's situation.

## What this does not show

- **The evidence is regional.** All six generators sit in one 25 km by 23 km box, inside ICON-D2's
  domain and in only two ERA5 grid cells. The ranking is untested elsewhere in Great Britain, and
  untested for wind, which [#826](https://github.com/openclimatefix/nged-substation-forecast/issues/826)
  covers.
- **The comparison is not lead-equal.** The served lead is part of what a consumer receives, so the
  as-served ranking answers the consumer's question. A comparison at a held-equal lead is a forecast
  comparison, and belongs to #810.
- **Every error is recalibrated per generator.** A product with a large but stable bias scores well
  here. The stability table is the only evidence on raw bias.
- **The figures rest on today's capacity table.** Each generator's capacity is its 99th percentile of
  output from the `effective_capacity` table as rebuilt in September 2026.
  [#825](https://github.com/openclimatefix/nged-substation-forecast/issues/825) may move the absolute
  figures, though not the contrasts between products.

## Reproducing the figures

Build every product's dataset, then run the study:

```bash
for source in open-meteo ukv icon-d2 icon-eu icon-global; do
  uv run python studies/beam_diffuse_split/build_dataset.py --source $source
done
uv run python studies/beam_diffuse_split/build_dataset.py --source cams \
  --min-cams-reliability 0 --suffix _allhours
uv run python studies/beam_diffuse_split/build_dataset.py --source cams
uv run python studies/beam_diffuse_split/weather_products.py
```

The report lands in `data/studies/beam_diffuse_split/beam_diffuse_weather_products/report.md`. The
served-lead check runs with `uv run --with cfgrib python
studies/beam_diffuse_split/verify_icon_lineage.py --model icon-eu`, and only against the runs the
German weather service still publishes, which cover about one day.
