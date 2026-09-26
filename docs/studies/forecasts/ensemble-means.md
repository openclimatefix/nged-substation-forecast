# How do Open-Meteo's ensemble-mean products compare for solar and wind power?

> **This study uses only Flexpectation's own data, and its purpose is to inform Flexpectation's
> choices.** The study scores each weather product only at the generators in Flexpectation's trial
> area in Lincolnshire. The study does not compare results across many regions or climates, so a
> result on this page may not hold elsewhere.

**At 6 solar farms and 3 wind farms in Lincolnshire, over 88 days from 25 June to 20 September 2026,
an XGBoost model given ICON-D2-EPS's ensemble mean had the lowest solar power error of the four
ensemble-mean products that Open-Meteo serves, and tied for the lowest wind power error at 10 m.**
The other three products are the ensemble means of
MOGREPS-UK, ICON-EU-EPS, and ECMWF IFS ENS at 0.25 degrees. Every error on this page is a mean
absolute error as a percentage of each generator's effective capacity. For solar power,
ICON-D2-EPS's
mean gave 7.4%, against 7.7% for ECMWF ENS's mean, 8.1% for ICON-EU-EPS's mean, and 8.7% for
MOGREPS-UK's mean. An XGBoost model given the satellite retrieval CAMS gave 4.9%, and a model given
no weather at all gave 12.6%. For wind power with every product's 10 m wind speed, ICON-D2-EPS's
mean gave 5.75% and ECMWF ENS's mean 5.79%, a gap of 0.04 points that this page treats as a tie. The
ERA5 reanalysis gave 6.28%, ICON-EU-EPS's mean 6.72%, and no weather 12.60%.

**The comparison is descriptive.** The window is one summer, the scoring uses 12 held-out weeks, and
the page states no interval and no significance test. Each figure is the mean over the same
generator-hours for every product, from an XGBoost model per generator, at the setting named in the
text.

**The products' leads are not matched, and the page does not measure them.** Each ensemble mean is a
stitched series, so each hour comes from a run of unknown age. ICON-D2-EPS starts a new run every 3
hours and ECMWF IFS ENS less often, so ICON-D2-EPS's hours probably come from shorter leads. Equal
leads would probably narrow ICON-D2-EPS's advantage, and the size of the narrowing is not known.

![Figure 1: ICON-D2-EPS's ensemble mean had the lowest solar error and tied for the lowest wind
error
at 10 m](../assets/ensemble_means_mae.svg)

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5). A Claude Sonnet review of the script and a Claude
> Sonnet review of the pull request have run. No Opus scientific-validity review has run, and no
> human has reviewed the page.

## What to use

**For a solar model that reads an ensemble mean, read ICON-D2-EPS's where ICON-D2 covers the
generator, and ECMWF ENS's elsewhere.** ICON-D2-EPS's mean scored 0.3 points better than ECMWF ENS's
mean, in both hyperparameter settings. ICON-D2 does not cover South West England or South Wales, and
this page cannot say how ICON-D2-EPS would score there, because every generator here is inside its
domain. All four Open-Meteo ensemble means scored 2.3 to 3.8 points worse than CAMS across both
settings, and 3.8 to
5.2 points better than no weather at all. CAMS is a satellite retrieval available only after the
hour, so it
sets the target that a forecast of past sunshine could reach and is not a competing forecast.

**For a wind model, the evidence favours ICON-D2-EPS's mean given its 100 m and 10 m speeds, with
two
caveats that stop this page recommending it without reservation.** Given those speeds, ICON-D2-EPS's
mean scored 5.08%, against 5.71% for ECMWF ENS's mean and 6.11% for ERA5, and it beat ERA5 in all 12
held-out weeks. The first caveat is the unmatched leads described above, which probably favour
ICON-D2-EPS. The second is that Open-Meteo's 100 m wind for ICON-D2's deterministic forecast is the
120 m speed multiplied by 0.98, as the [past-wind study](../past-weather/wind.md) found, and this
study has not checked whether the ensemble mean's 100 m wind is built the same way. The 5.08% is
therefore a score for a 100 m speed of unchecked origin. Open-Meteo serves no 100 m wind for
MOGREPS-UK's or ICON-EU-EPS's mean (the `README.md` in each product's folder lists the field as all
null, which the script also checks), so neither could be scored at hub height. At 10 m,
ICON-EU-EPS's mean scored 0.45 points worse than ERA5 in both settings, and does not
suit a wind model.

**These verdicts would change with a longer window or a lead-resolved comparison.** The window holds
one summer, and none of its wind is from autumn or winter. The ensemble means are stitched series
that carry no run time, so the page says nothing about which product forecasts best at a stated lead
such as day-ahead.

## Data and methods

**The rows are the hours that every product covers, 6,663 solar generator-hours and 5,930 wind
generator-hours.** Solar rows are the hours with the sun above the horizon. The rows drop multi-day
zero runs and metering spikes above 1.5 times capacity, and the solar rows also drop hours where the
meter reads exactly zero while ERA5 reports more than 100 W/m². The wind rows drop every hour
holding
an exactly-zero half-hour. Solar power is the mean of the hour ending at the label and wind power is
the mean centred on the label, the conventions of the earlier past-weather studies. Rows with the
generator's export limit in force (49 solar rows at one generator) are scored but not trained on.

**Each product is one stitched series.** For each hour, Open-Meteo returns the mean of the ensemble
members from the newest run that covers the hour, with no run time attached. Leads are therefore
short and mixed. This repository's own ECMWF ENS table gives one more series, labelled "local": the
mean of the 51 members, the newest run for each valid time from the three lead bands with 3-hourly
steps (the script's
`LOCAL_ENS_HORIZONS`, leads up to 69 hours), radiation held
over its 3-hour step, and wind interpolated linearly between steps. The local series takes its
00:00 UTC valid time from lead 24 of the previous day's run.

**Each arm is an XGBoost model per generator given one product's values.** Solar arms read five
shared columns (solar zenith, solar azimuth, extraterrestrial irradiance, ERA5 air temperature, and
hour of day) and one column of the product's global irradiance. Wind arms read hour of day and the
product's wind speed at 10 m. The hub-height design reads the 100 m and the 10 m speeds, for the
four
products that carry a 100 m speed. Wind direction is left out, because Open-Meteo's ensemble mean of
a direction is a naive average of member directions, which is wrong near north. Every arm of a
design has the same number of columns. The baseline "no weather" has only the shared columns. The
models use XGBoost 3.4.1 on the CPU at two fixed settings, with three seeds each, and the tables
report the first setting unless they say otherwise.

**The folds are leave-one-week-out.** The 88 days form 12 blocks of 7 days, the last block holding
the 11 final days. Every arm, generator, and setting shares the folds. Because the days beside a
scored week stay in the training rows, the scheme measures how well a model interpolates between
weather episodes within one summer, not how well it forecasts into a new season.

## Results

**Each ensemble mean's error in irradiance follows its error in solar power.** Against CAMS's
irradiance, which averaged 354 W/m² on these rows, ICON-D2-EPS's mean had a mean absolute error of
50 W/m², ECMWF ENS's 54, ICON-EU-EPS's 59, MOGREPS-UK's 71, and the local ECMWF ENS series 86. The
power errors have the same order. MOGREPS-UK's mean reads 33 W/m² below CAMS on average.

**Wind speed agreement with ERA5 does not follow wind power error.** At 10 m, the local ECMWF ENS
series differs from ERA5 by 2.4 km/h, ECMWF ENS's mean by 3.0, ICON-EU-EPS's by 3.0, MOGREPS-UK's by
3.8, and ICON-D2-EPS's by 4.1. ICON-D2-EPS's mean has the largest difference and the lowest power
error. ERA5 is a 31 km reanalysis and not a measurement, so a large difference from ERA5 does not
show that a product is wrong.

| Design and product | First setting (% of capacity) | Second setting (% of capacity) | Held-out weeks beating the reference (first setting) |
|---|---|---|---|
| Solar, CAMS (reference) | 4.89 | 5.16 | not applicable |
| Solar, ICON-D2-EPS mean | 7.36 | 7.42 | 0 of 12 |
| Solar, ECMWF ENS mean (Open-Meteo) | 7.68 | 7.75 | 0 of 12 |
| Solar, ICON-EU-EPS mean | 8.13 | 8.18 | 0 of 12 |
| Solar, MOGREPS-UK mean | 8.69 | 8.66 | 0 of 12 |
| Solar, ECMWF ENS mean (local) | 8.89 | 8.95 | 0 of 12 |
| Solar, no weather | 12.60 | 12.47 | 0 of 12 |
| Wind 10 m, ICON-D2-EPS mean | 5.75 | 5.83 | 10 of 12 |
| Wind 10 m, ECMWF ENS mean (Open-Meteo) | 5.79 | 5.87 | 11 of 12 |
| Wind 10 m, ECMWF ENS mean (local) | 6.00 | 6.07 | 9 of 12 |
| Wind 10 m, MOGREPS-UK mean | 6.03 | 6.06 | 8 of 12 |
| Wind 10 m, ERA5 (reference) | 6.28 | 6.30 | not applicable |
| Wind 10 m, ICON-EU-EPS mean | 6.72 | 6.75 | 3 of 12 |
| Wind 10 m, no weather | 12.60 | 12.60 | 0 of 12 |
| Wind 100 m and 10 m, ICON-D2-EPS mean | 5.08 | 5.04 | 12 of 12 |
| Wind 100 m and 10 m, ECMWF ENS mean (Open-Meteo) | 5.71 | 5.73 | 9 of 12 |
| Wind 100 m and 10 m, ECMWF ENS mean (local) | 6.00 | 5.95 | 8 of 12 |
| Wind 100 m and 10 m, ERA5 (reference) | 6.11 | 6.07 | not applicable |
| Wind 100 m and 10 m, no weather | 12.60 | 12.60 | 0 of 12 |

**The two settings give the same order, with one swap.** At 10 m for wind, MOGREPS-UK's and the
local ECMWF ENS series swap places, and their gap is at most 0.04 points in both settings. The three
seeds moved each error by less than 0.04 points, so gaps smaller than that are noise.

**The local ECMWF ENS series scores worse than Open-Meteo's ECMWF ENS mean.** The gap is 1.2 points
for solar, 0.2 for wind at 10 m, and 0.3 for wind at 100 m. The two series come from the same
ensemble on different grids and with different stitching and interpolation, so the gap may reflect
the pipeline. The study did not isolate the cause.

## Limitations

**The window is short and holds one summer.** The 88 days give 12 held-out weeks. The window does
not sample autumn or winter wind, and it samples only summer cloud regimes.

**The generators share weather.** Among the 6 solar generators, MOGREPS-UK's and ICON-D2-EPS's means
serve 5 distinct series, ICON-EU-EPS's 4, and ECMWF ENS's 2, because neighbouring generators fall in
one grid cell. Among the 3 wind generators, ECMWF ENS's mean serves 2 distinct series and the other
products 3. The effective sample is smaller than the number of generator-hours.

**The scoring is interpolation.** Training rows include the days on both sides of each scored week.
A model scored on a later season would show larger errors.

**The ensemble means are stitched series with unmatched leads.** They carry no run time, so their
leads are short, mixed, and different between products, and the page has not measured them.
ICON-D2-EPS
probably has the shortest leads, which probably flatters it. The page says nothing about day-ahead
skill or about how the error grows with lead.

**Products differ in coverage.** ICON-D2 does not cover South West England or South Wales, and every
generator here lies inside its domain. MOGREPS-UK's and ICON-EU-EPS's means have no 100 m wind, and
the origin of ICON-D2-EPS's 100 m wind
is unchecked.

**The capacity denominator is the latest effective capacity per generator in
`data/effective_capacity`
when the study ran on 26 September 2026.**

## Scope

**The study does not cover** ensemble spread or probabilistic scores, leads beyond the stitched
series, other seasons, generators outside Lincolnshire, wind direction, other Open-Meteo ensembles,
or the beam and diffuse split of radiation. The study does not test whether any difference is
statistically significant at the 5% level.

## Data and code availability

**The four Open-Meteo ensemble means, CAMS, and ERA5 are public downloads, and the metered power is
private.** The Open-Meteo means are in `data/studies/weather/OPEN-METEO-ENSEMBLE-MEANS/`. The local
ECMWF ENS series is built from `data/studies/weather/ENS/`, which reads this project's own ENS Delta
table. NGED's power readings are private. Every output carries only the anonymised labels `A` to `F`
and `W1` to `W3`. The code is in `studies/open_meteo_ensemble_means/` and
`packages/studies/`, at the commit that adds this page.

## Reproducing the figures

```bash
uv run python studies/open_meteo_ensemble_means/ensemble_means_mae.py
uv run python studies/open_meteo_ensemble_means/ensemble_means_chart.py
npx svgo@4 --multipass --precision=1 --final-newline docs/studies/assets/ensemble_means_mae.svg
```

The first command writes the tables and `report.md` under `data/studies/open_meteo_ensemble_means/`
and takes about 15 minutes on the CPU. The script refuses to overwrite, so move the existing files
to
a `superseded/` subfolder first. `--report-only` rebuilds the tables from the saved losses.
