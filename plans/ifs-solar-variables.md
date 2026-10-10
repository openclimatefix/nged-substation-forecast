# Plan: which IFS forecast variables help a day-ahead to nine-day-ahead solar PV forecast?

Status: draft, written before any IFS result exists, and revised after one Opus plan review. It
follows
[PR #1097 (Plan: which ERA5 variables help explain solar PV output and CAMS
irradiance)](https://github.com/openclimatefix/nged-substation-forecast/pull/1097),
whose ERA5 screen asks the same question of a reanalysis. The IFS data is fetched and validated.

## Question

**If an XGBoost model (a gradient-boosted tree library) is given the extra variables that
Open-Meteo serves from ECMWF's high-resolution forecast (IFS), does it predict solar photovoltaic
(PV) output better than an XGBoost model given only the minimal IFS set, at forecast lead times of
1 to 9 days?** IFS is the Integrated Forecasting System of the European Centre for Medium-Range
Weather Forecasts (ECMWF). The study repeats, on real past forecasts, the question the ERA5 study
asks of a reanalysis. ERA5 is not a forecast, so a positive ERA5 result does not guarantee a gain
at a lead time, and a negative ERA5 result may reflect ERA5's age. The follow-up asks the question
of the forecast data that the NGED service would receive.

**Three decisions depend on the answer, and the plan states a rule for each.** The third is the
main one: whether engineering effort should go into ingesting the variables that only ECMWF's MARS
archive serves, or into ingesting another forecast.

- **Which variables to ask Dynamical.org to add.** Dynamical.org supplies IFS ensemble forecasts
  to the production pipeline, and it can add only fields that ECMWF's free open-data feed carries.
  That feed carries total column water vapour, convective available potential energy, and skin
  temperature among the solar-relevant fields, and it carries no cloud layers, direct radiation, or
  boundary-layer height (checked 2026-10-09; recheck on a recent run before the page assigns
  classes).
- **Whether the production forecast needs a second IFS feed.** A second feed such as Open-Meteo is
  needed only if the groups that help are ones the open-data feed does not carry.

- **Whether to ingest the MARS-only variables or another forecast.** The study cannot test the 12
  MARS-only variables, because Open-Meteo does not serve them. The ERA5 study's planned contrast P4
  is the only evidence on those, and ERA5 is a reanalysis, so P4 is an upper bound on the gain at
  any lead. The comparison section below measures what a second forecast adds on the same
  target, and the page sets the two gains side by side.

## Why this design

- **Open-Meteo's Single Runs archive is the quickest source of past IFS forecasts with these
  variables among the sources we compared.** It holds the 00 UTC run of every day from 2024-03-14,
  with hourly values out to 240 hours, at one request per run. The fetch is done: 933 runs, six
  solar farms, 20 variables, in `data/studies/downloads/NWP/ECMWF-IFS-SINGLE-RUNS-SOLAR/`. Eight run
  days are missing: five in August 2025, two in June 2026, and one in September 2026.
- **The archive is 2.6 years, not the six of the ERA5 study.** Intervals are wider. Before any fit
  the report prints the width that a planned interval would have, from the ERA5 losses cut to the
  same months, so the page states the smallest gain the study can rule out. If that width exceeds
  0.2 points of capacity, the page says the study can rule out only gains larger than the width.
- **The decision time is the morning of the day before the valid day.** ECMWF publishes the 00 UTC
  high-resolution run around 06 to 07 UTC and the ensemble around 08 UTC, before the GB day-ahead
  gate closures at about 09:20 to 09:50 UTC. The 00 UTC run is therefore the run a morning
  day-ahead decision uses, and "lead day 1" means the 24 to 47 hour leads of that run.
- **Open-Meteo builds some hourly values from coarser output.** IFS output steps coarsen beyond 90
  hours, so values at lead days 5 to 9 are interpolated. Open-Meteo labels the runs before
  2024-11-12 as cycle 49r1 hindcasts, so the 2024-11 boundary is a change of source as well as of
  cycle.

## Arms: a shortened ladder from the free IFS variables

Every arm is shown the shared geometry and calendar columns of the ERA5 study
(`studies.era5_ladder.SHARED_FEATURES`), so a variable cannot win by supplying sun position. The
ladder is shorter than ERA5's, because IFS gives fewer variables. Each rung adds one physical idea.

| Rung | Adds | IFS variables (Open-Meteo names) |
|---|---|---|
| F0 minimal | the minimal set of the ERA5 study | `shortwave_radiation`, `temperature_2m` |
| F1 total cloud | cloud amount | `cloud_cover` |
| F2 cloud layers | where the cloud is | `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` |
| F3 direct beam | the direct share of the sunlight | `direct_radiation` |
| F4 humidity and column water | haze and cloud-forming moisture | `dew_point_2m`, `total_column_integrated_water_vapour`, `boundary_layer_height` |
| F5 convection and visibility | convective cloud and haze | `cape`, `convective_inhibition`, `visibility` |
| F6 the rest | surface and weather state | `surface_temperature`, `snow_depth`, `snowfall`, `precipitation`, `surface_pressure`, `wind_speed_10m`, `wind_gusts_10m` |

F6 is the full set, with 20 variables. Convective inhibition is missing wherever it is undefined
(about 92% of hours), and XGBoost reads a missing value natively.

**A production reference arm, FP, is added for the second decision.** The production ensemble feed
already carries `dew_point_2m`, `surface_pressure`, `precipitation`, and `wind_speed_10m`, so FP is
F0 plus those four variables. FP is a reference only: it shows what the production set already
earns, and F2, F3, and F6 are read against it as exploratory contrasts. The production feed also
carries downward long-wave radiation, a strong cloud signal that Open-Meteo does not serve, so the
page states that cloud gains here may overstate the gain over the production set.

**Column counts and subsampling.** Column subsampling is off (`colsample_bytree=1`). The arms of a
planned contrast differ by 1, 2, and 12 columns, and the study skill asks for equal counts. The
negative control (below) has the same columns as F6, so each planned contrast is also read against
it, and the report states the size of difference that extra columns produce from nothing. P3 is
read as F6 minus F2 and as F6 minus the negative control.

## Targets

- **PV output** (primary): the farm's hourly power divided by its effective capacity, from the
  same cleaned hourly series the ERA5 study uses, with the same filters (daylight, no outage runs,
  no commissioning ramp, export cap honoured).
- **CAMS clearness index** (second target, exploratory): CAMS global horizontal irradiance over the
  top-of-atmosphere flux, so a gain that the PV target shows can be separated from curtailment.

Both targets are scored with `absolute_error_capped_fraction_of_capacity`, as in the ERA5 study.

## Rows, lead days, folds, and the metric

- **The row set comes from the ERA5 study's dataset.** The study reads
  `dataset_main_through_g9.parquet` for each (farm, valid hour), and joins the IFS forecast whose
  run began `L` days before the valid day, for each lead day `L`. Lead day `L` of a valid hour at
  hour `h` of its day is the run's lead of `24 L + h` hours. The lead days are 1, 2, 3, 5, 7, and
  9. Lead day 10 has one hour in the run and is not used.
- **Within one lead day, every arm is scored on the same rows.** An hour is kept only if every
  IFS variable of the full set is present (convective inhibition excepted), and a missing run day
  drops that valid day at that lead day only. Each lead day is its own model, one XGBoost model
  per farm and fold and seed.
- **Eras and folds.** IFS cycle 49r1 went live on 2024-11-12 and cycle 50r1 on 2026-05-12. The
  study drops the two straddling part-months (2024-11 and 2026-05), adds an era feature, and drops
  the third era (2026-06 onwards), which is too short to form folds. Two eras remain: 2024-04 to
  2024-10 (7 months) and 2024-12 to 2026-04 (17 months). Folds come from
  `studies.cross_validation.cut_eras`, with the fold offsets found by `search_fold_offsets` and
  confirmed by `raise_on_uncovered_months`, so that no calendar month is absent from the training
  rows of a fold that holds it out. **The folds are assigned once, on all rows, before any
  lead-day filter, and the build asserts that every lead day carries the same fold for the same
  valid hour.**
- **Seeds and intervals.** Three seeds, and a paired bootstrap resampling whole months and one
  seed (`studies.bootstrap`), with 10,000 resamples for planned contrasts.
- **Smallest effects of interest:** 0.1 percentage points of capacity for PV and 0.01 for the
  clearness index, as in the ERA5 study.
- **Second hyperparameter setting** (`SENSITIVITY_HYPER_PARAMETERS`) on every planned arm.
- **Radiation of exactly −1 W/m² is clipped to 0**, as the dataset's README warns.

## Planned contrasts (written before any result exists)

Two lead days (1 and 3) and three contrasts on the PV target make six planned contrasts, each a
treatment minus a reference, so a negative difference is a gain.

1. **P1:** F1 minus F0. Does total cloud help beyond the radiation and temperature a forecast
   already supplies?
2. **P2:** F2 minus F1. Do the cloud layers help beyond total cloud?
3. **P3:** F6 minus F2. Does any other free variable help beyond the cloud layers?

The family-wise level of 5% becomes 0.833% per contrast (a 99.17% interval). A contrast "rules out
a gain larger than the smallest effect" when its lower bound lies above minus the smallest effect.
Every other number is exploratory: the other lead days (2, 5, 7, and 9), the rungs F3 to F5 one by
one, the contrasts against FP, the CAMS target, the regime and season splits, and the drop-one
runs. The page labels them exploratory and does not correct them.

**Decision rule for the variables to request, fixed before any result.** A contrast is a gain when,
at lead days 1 and 3 on the PV target at both hyperparameter settings, its whole adjusted interval
lies below minus the smallest effect. It is no gain when the whole interval lies above minus the
smallest effect at both lead days. Otherwise it is unresolved.

- **A gain in P1** recommends asking Dynamical.org to ingest total cloud, which the open-data feed
  already carries and the production pipeline currently leaves out.
- **A gain in P2 or a gain in F3 (direct beam) read as an exploratory check** recommends a second
  IFS feed, because the open-data feed carries no cloud layers or direct radiation.
- **A gain in P3** names the groups that carry it through the drop-one runs. The page lists them,
  and marks each group as available from the open-data feed or not.
- **No gain, or unresolved:** the production forecast changes nothing, and the page says how much
  more history would resolve the contrast.

## Controls

- **Negative control:** F2 plus a permuted copy of the F3 to F6 variables
  (`studies.blending.climatology_permutation`, over farm, month, and hour of day), at lead days 1
  and 3. It shows what the extra columns produce from nothing. It is not strictly information
  free, because it keeps each month-and-hour mean. The report prints the control's difference from
  F2, and a planned gain smaller than that difference is not read as a gain.
- **Positive control:** F2 plus CAMS global horizontal irradiance at the valid hour, which is the
  observation. It must show a gain larger than 1 point of capacity. If it does not, the page states
  that the instrument cannot detect a large effect, and reads every null as unresolved.
- **Drop-one-group runs** from F6 (exploratory), at lead days 1 and 3.

## Comparison with the ERA5 study

The page draws the ERA5 contrast and the IFS contrast side by side for each rung that both studies
share, at lead day 1. The comparison is like for like only if the ERA5 arms are refit on the IFS
study's rows, folds, and eras, with the same feature set apart from the variables themselves
(ERA5's minimal set includes a clearness-index column that the IFS minimal set lacks), so the ERA5
arms G0, G1, G2, and G9 are refit that way and the two sets of contrasts are labelled exploratory.
The comparison shows whether the ERA5 screen and the forecast agree on these months. It does not
prove that ERA5 results transfer to other months.

## Comparison with a second forecast: all IFS variables, or minimal IFS plus another forecast

**This comparison asks whether the extra IFS variables are worth more than a second weather forecast.** The
[blends page](../docs/studies/forecasts/blends-with-ens.md) found that adding one product to ECMWF's
ensemble mean sometimes lowered the error. The production service can follow either route: ask
Dynamical.org for more IFS fields, or add a second forecast to the minimal IFS set it already
holds. The three arms below put the two routes on the same rows.

| Arm | Columns | Reads as |
|---|---|---|
| F0 | the minimal IFS set | the starting point |
| FB (blend) | F0 plus the partner's `shortwave_radiation` and `temperature_2m` at the same lead day | the minimal IFS set plus another forecast |
| F6 | all 20 IFS variables | all the relevant IFS variables |
| F6B | F6 plus the partner's two variables | the ceiling that uses both routes |

- **The partner is AIFS Single** (ECMWF's machine-learned forecast), read from the Dynamical.org
  store the blends page used. It supplies downward shortwave radiation and 2 m temperature from
  the same 00 UTC run, so the lead day matches IFS's, and its first run day is 2024-04-01. Its
  radiation is a 6-hour mean, so each valid hour reads the 6-hour window that contains it, and the
  page says so.
- **ICON-EU is an optional second partner at lead days 1 and 3 only**, read from the Previous Runs
  files, if the files cover the IFS months at whole-day offsets. If they do not, the page says that
  ICON-EU was not tested.
- **The rows are the intersection of the IFS rows and the partner's rows**, so F0 and F6 are
  refit on them. Their errors can differ from the main ladder's, and the report prints both.
- **FB is given the same number of columns as the partner adds to F0 (2), and a control gives F0 two
  permuted copies of the partner's variables** (`studies.blending.climatology_permutation`), so a
  blend gain is read against what two extra columns produce from nothing.
- **The contrasts are exploratory, at lead days 1 and 3, on the PV target, at both hyperparameter
  settings:** FB minus F0 (what the partner adds), F6 minus F0 (what the IFS variables add),
  F6 minus FB (which route is better), and F6B minus F6 (whether the partner still adds anything
  once every IFS variable is present). The page states the row count, the era restriction, and that
  the verdict is for AIFS Single at these lead days, not for forecasts in general.

**The decision rule for the third decision is fixed before any result.** The page draws three
gains in points of capacity, each with its 95% interval: the ERA5 study's P4 (MARS-only variables,
an upper bound), F6 minus F0 (free IFS variables), and FB minus F0 (a second forecast), the last two
at lead days 1 and 3. The page states that the gains come from different rows and so are compared
only as sizes. **Another forecast is the better use of engineering effort when FB minus F0 is a gain
(its whole interval lies below minus the smallest effect at both lead days) and P4's upper-bound
point estimate is no larger than that gain.** **The MARS-only variables are worth pricing when P4's
upper-bound gain is larger than the partner's gain and the partner's gain is not a gain.**
Otherwise the page says the evidence does not separate the routes. The page does not state costs
as findings: the cost of ECMWF dissemination is unverified, and engineering effort is not measured
here.

## Page structure and figures

The page follows the study skill's figure-led form. It opens with the title, a bottom line of a few
sentences, and the headline figure above the fold. The take-home message is general (which IFS
variables help a solar PV forecast, and how far ahead), and it says what it means for the
Flexpectation project.

1. **Headline.** The planned contrasts (P1 to P3) at lead days 1 and 3, with the adjusted and 95%
   intervals, and the smallest effect marked.
2. **What the forecasts look like.** Three days chosen by a stated rule (the clearest, the most
   variable, and the dullest by the CAMS clear-sky index of the valid day), at one farm: IFS cloud
   covers and radiation at lead days 1 and 7 against the CAMS observation. A series of one farm's
   weather carries no calendar date on its axis (days are counted 1 to 3, with the month and year
   in the text) and no farm label, and the marks have `aria=False`.
3. **How skill falls with lead.** The error of F0 and F6 at every lead day, on the rows common to
   every lead day.
4. **The models work.** Predictions against measured output on held-out months at lead day 1, for
   the same three stated-rule days at every anonymised farm, with each arm's mean absolute error
   per farm.
5. **Controls.** The negative and positive controls at lead days 1 and 3, beside F0 and F6.
6. **The leaderboard.** Every rung's error by lead day, with intervals.
7. **ERA5 against IFS.** The contrasts that both studies share, side by side.
8. **Regimes and seasons.** The difference in error between F6 and F0 by ERA5 cloud regime (clear,
   broken, overcast) and by season, exploratory.
9. **Drop-one-group.** Error added when each group is removed from F6, exploratory.
10. **Variables or another forecast.** The three gains of the third decision's rule beside each
    other, then the error of F0, FB, F6, and F6B at lead days 1 and 3 with
    intervals, and the contrasts FB minus F0, F6 minus F0, and F6 minus FB, exploratory.

Each figure carries the text it needs to be read alone, and its text is reviewed (a first-stumble
reader, then a one-rule-per-pass sweep) before the figure is first rendered.

## Data and code

- **Inputs.** The ERA5 study's dataset (rows, targets, geometry, export cap, ERA5 cloud regime),
  `ECMWF-IFS-SINGLE-RUNS-SOLAR.parquet` (validated: the four repeated variables equal the
  `ECMWF-IFS-SINGLE-RUNS` product's exactly, no duplicate keys, no value outside its range), and
  the CAMS irradiance already in the dataset. Whether Open-Meteo's `direct_radiation` is native
  IFS output or derived is unchecked
  (`studies.served_column_checks.check_direct_is_not_a_separation_model`
  is run on it before the fit), and the page states the result.
- **Code.** A new folder `studies/ifs_solar_variables/` holds `ifs_ladder_arms.py`,
  `ifs_ladder_build_dataset.py`, `ifs_ladder_fit.py`, `ifs_ladder_report.py`,
  `ifs_ladder_charts.py`, and a README. Code both studies need goes into
  `packages/studies/src/studies/` with tests (the matched-lead join, the era labelling, and the
  lead-day rows). Each script has at least one fresh Opus review before it runs.
- **Compute.** The fits run as the ERA5 fits do: a slot from the study coordinator, GPU, resumable
  checkpoints. A lead day is about 60,000 rows over all farms, so the whole ladder is a fraction
  of the ERA5 fit.
- **Outputs.** `data/studies/per_study/ifs_solar_variables/`, and the page
  `docs/studies/forecasts/ifs-solar-variables.md`.

## What the study cannot say

- It does not test the 12 MARS-only IFS variables. Only the ERA5 study speaks to them.
- It uses one run a day (00 UTC) of the deterministic 9 km forecast. The production feed is the
  ensemble on a 0.25° grid with 3-hourly output, so the study says nothing about ensemble members,
  the regridding, or the output step. Resolution is the one property that transfers, because the
  ensemble has run at 9 km since cycle 48r1.
- The history is 2.6 years with two model cycles, and the two eras differ in source as well as in
  cycle, so intervals are wide and a result can change with a new cycle. An exploratory check
  scores the newest cycle on 2026-06, the one month the targets cover after 2026-05, with a model
  trained on the second era.
- Open-Meteo's hourly values beyond 90 hours are interpolated, not native model output.
- Downward long-wave radiation, which the production feed carries, is not available here, so cloud
  gains may be overstated against the production set.

## The five complexity triggers (for sizing)

- **Changes what is stored:** yes. A published study page and its results.
- **Touches the production serving path:** no.
- **Touches a degradation rule:** no.
- **Admits more than one defensible design:** yes. The ladder, the lead days, and the eras can
  each be cut differently.
- **Spans code whose callers one could not name without searching:** yes, because it shares
  `packages/studies`.

The study is complex. It gets the plan and all four reviews: two Opus plan reviews (the first is
done), two Opus science reviews, the diff review, the mutation pass (`packages/studies` changes),
and the prose and persona reviews. No review is shortened.

## Order of work

The IFS fetch is done and validated. Next: this plan and a second Opus plan review, the build
script and its review, the fits (after the ERA5 fits release their slot), the report, the first
Opus science review, charts with their text reviewed before the first render, the page, the second
Opus science review, the diff review, the mutation pass, the first-stumble and one-rule-per-pass
prose reviews, the persona reviews, and merge. The pull request stays a draft until the reviews
are done.
