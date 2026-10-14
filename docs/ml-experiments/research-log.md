# Research log of the ML experiments

**This page lists every finished ml-experiment in the order it was started, what it set out to find,
and what it found.** A reader who wants the headline of each experiment starts here and follows the
link to the experiment's `result.md` for the numbers. The experiments that are planned or proposed
are on the [research plan](research-plan.md). Each experiment carries a production status. The
maintainer reviews the candidates every one to two weeks and ports the best into a new model
release; the [release log](../live_service/model-releases.md) records which experiments each release
adopted.

## Candidates for the next release

**These experiments are flagged `candidate` and are not yet adopted.** The flags are proposals for
the maintainer to confirm.

| Id | What would change in production | Note |
|---|---|---|
| [00001](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Train the weather-to-power model on ERA5 and apply it to ENS members | Needs the staged design, and helps only up to day 6 |
| [00002](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Pool the ERA5-trained quantiles over the ENS members | Stage 2 itself did not help |
| [00006](00006-stage1a-demand-input-feature-sweep/result.md) | Use the clock, a COVID indicator, and recency weights as the stage 1a demand inputs | The Christmas column needs its 14-day window, a 2023 start needs a fresh year, and windchill can go |
| [00014](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) | Map the ENS shortwave onto CAMS (a stage 0 quantile mapping) before a CAMS-trained stage 1a reads it | The year's mean is corrected but individual months are up to 18.6% off, and the member spread widens by 27% |
| [00016](00016-pooled-mixture-calibration-solar-wind/result.md) | Scale the solar mixture's quantiles about the median by a factor near 0.95, refitted monthly | Fixes 94% to 95% coverage at no change in CRPS; wind needs a spread factor on two lead groups, a post hoc arm |

## The experiments so far

| Id | Question | Verdict | Production status | Merged |
|---|---|---|---|---|
| [00001](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Does a weather-to-power model trained on ERA5 beat the same model trained on ENS weather? | Supported up to day 6 | Candidate | 2026-10-14 |
| [00002](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Does a stage 2 calibrator beat the pooled stage 1a mixture? | Partly supported | Candidate | 2026-10-14 |
| [00003](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md) | Does the ERA5-trained model beat the ENS-trained model on demand series? | Not supported | Not for production | 2026-10-14 |
| [00005](00005-lag-placement-stage1b-vs-stage2-demand/result.md) | Where should lagged power enter a two-stage demand forecast: stage 1b, stage 2, or both? | Refuted: stage 2 is better | Not for production | 2026-10-14 |
| [00006](00006-stage1a-demand-input-feature-sweep/result.md) | Which non-lag inputs are best for the ERA5-trained demand model (stage 1a)? | Partly supported | Candidate | 2026-10-14 |
| [00007](00007-cams-static-correction-stage1a-residual/result.md) | Does a static CAMS correction fitted on one year remove most of the next year's daytime stage 1a residual? | Inconclusive | Not for production | 2026-10-14 |
| [00008](00008-lag-summaries-in-stage2-calibrator-demand/result.md) | Do summaries of recent power (minimum, mean, and maximum of the last 24 hours and of the same weekday) help the stage 2 calibrator? | Inconclusive: no gain on top of raw lags | Not for production | 2026-10-14 |
| [00009](00009-cams-trained-stage1a-vs-era5-trained-demand/result.md) | Does a stage 1a trained on CAMS irradiance beat one trained on ERA5 shortwave, when both are applied to ENS members? | Refuted: a gain of 0.035 points, not 0.2 | Not for production | 2026-10-14 |
| [00010](00010-elexon-price-lags-stage1a-demand/result.md) | Does the lagged electricity price lower the stage 1a demand error at substations with batteries? | Inconclusive | Not for production | 2026-10-14 |
| [00011](00011-school-term-and-return-to-work-columns-stage1a-demand/result.md) | Do columns for the return to work after New Year and for Lincolnshire's school terms lower the stage 1a demand error after Christmas, at Easter, and in half-term holidays? | Refuted by the rule as written (inconclusive as intended): no gain in any window | Not for production | 2026-10-14 |
| [00012](00012-solar-wind-stage1a-inputs-cams-and-training-start/result.md) | Which inputs and which training start work best for the solar and wind stage 1a model? | Inconclusive: no change reached the 0.15-point gain | Not for production | Not yet merged |
| [00014](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) | Does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained demand model? | Inconclusive: a gain of 0.067 points, but monthly mismatches remain | Candidate | Not yet merged |
| [00016](00016-pooled-mixture-calibration-solar-wind/result.md) | Does a correction bring the 5% to 95% band of the pooled mixture to 90% coverage for solar and wind? | Solar supported, wind refuted (a post hoc arm for wind reaches 90%) | Candidate | 2026-10-14 |

### 00001: ERA5-trained stage 1a against ENS-trained, for solar and wind

**Stage 1a trained on ERA5 forecasts solar and wind power better than the same columns trained on
ENS weather, up to a lead of 6 days.** The experiment applied an XGBoost model trained on ERA5 and
CAMS to each of the 51 ENS members, averaged the members, and compared the mean absolute error with
an ENS-trained model. It used 6 solar farms and 3 wind farms on one held-out year (53 weekly runs,
row set `fold_a_tuesdays_v1`).

- **Headline result:** the error fell by 1.70 points of capacity for wind (interval −2.33 to −1.08)
  and 0.57 for solar (−0.83 to −0.32). Beyond day 6 the gain is not material.
- **The gain comes from training on ERA5 weather,** not from the length of the history: a model
  trained on only the ENS months had the same error.
- **A likely reason (not tested directly):** a model trained on forecast weather learns the
  forecast's errors, which flattens the weather-to-power curve, most of all for wind, where power
  rises steeply with wind speed. A model trained on ERA5 learns the sharper curve.
- **Wind peaks gain most.** In the top decile of observed wind power the gain is 6.14 points.
- **A flaw in the first run:** the production-recipe control memorised calendar position, which
  inflated the first result by 1.13 points for solar and 3.40 for wind. Every arm now omits calendar
  columns, and every later experiment includes a climatology row.
- **Write-up and pull request:** [result](00001-era5-trained-stage1a-vs-ens-trained/result.md),
  [PR #1184 (Study: stage 1 pilot, an ERA5-trained weather-to-power model against the ENS-trained
  control)](https://github.com/openclimatefix/nged-substation-forecast/pull/1184).

![Planned contrasts by lead band](00001-era5-trained-stage1a-vs-ens-trained/figure_2.svg)

### 00002: a stage 2 calibrator against the pooled stage 1a mixture

**Pooling the ERA5-trained stage 1a quantiles over the ENS members beats an ENS-trained quantile
model, and a stage 2 calibrator with a learned blend did not improve on the pooled mixture.** The
experiment scored the continuous ranked probability score (CRPS) on the same 9 generator series.

- **Headline result:** the pooled mixture beat the ENS-trained quantile model by 0.62 points of
  capacity for wind (−0.99 to −0.26) and 0.24 for solar (−0.36 to −0.06). Stage 2 against the
  mixture was not material (solar +0.40, wind +0.26).
- **Training on the long ERA5 history** changed the CRPS by only 0.01 to 0.09 points against
  training on the ENS months.
- **A likely reason stage 2 did not help:** it had little to correct. The mixture is already close
  to calibrated for one product, and the calibrator was trained on about 15 months of ENS data for 9
  series.
- **What the test did not stress:** it used one weather product and no lagged power, so it did not
  test the two jobs stage 2 is meant to do.
- **Write-up and pull request:** [result](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md),
  [PR #1189 (Stage 2 pilot: shared calibrator and learned blend against the pooled stage 1a
  mixture)](https://github.com/openclimatefix/nged-substation-forecast/pull/1189).

![CRPS by lead band](00002-stage2-calibrator-and-mixture-vs-ens-quantile/figure_1.svg)

### 00003: ERA5-trained stage 1a against ENS-trained, for demand series

**For demand series, the ERA5-trained model forecasts worse than the ENS-trained model.** The
experiment repeated 00001 on 22 series that are not wind or solar farms, on a new row set of 73 runs
(every fifth day) so that every weekday is represented in every lead band.

- **Headline result:** with weather only, the ERA5-trained error is larger by 1.37 points of capacity
  (+0.68 to +2.17). With calendar columns and lagged power the two are equal (+0.10, −0.26 to
  +0.50).
- **The likely cause is drift:** the error falls steadily as the training start moves later, from 2019
  to 2023, which suggests that the customer mix and the embedded generation changed.
- **Write-up and pull request:** [result](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md),
  [PR #1192 (Experiment 00003: ERA5-trained stage 1a against ENS-trained, for demand
  series)](https://github.com/openclimatefix/nged-substation-forecast/pull/1192).

![Contrasts by lead band](00003-era5-trained-stage1a-vs-ens-trained-demand/figure_2.svg)

### 00005: where lagged power enters a two-stage demand forecast

**For Primary substations, recent power helps more in the stage 2 calibrator than in stage 1b.** The
experiment gave both placements identical inputs and calibrated both, on 20 substation series and
the 73 runs of the validation year. Every lagged arm is a ceiling, because the recent residuals use
ERA5 weather at past times.

- **Headline result:** the primary contrast, stage 1b with a calibrator minus the calibrator with
  lags, was +0.84 points of capacity for leads up to 48 hours (99% interval +0.28 to +1.50, 9 of 12
  months favouring the calibrator). Lagged power in the calibrator scored 5.31 against 6.14 for
  stage 1b and 7.11 for no lags. Using both placements lowered the quantile score by at most 0.12 points.
- **Finding for later experiments:** a lag-free calibrator lowered the quantile score by about 1
  point of capacity at short leads and raised the median's error. Stage 1a alone scored no better
  than day-type climatology. The grid supply points' error is dominated by drift.
- **Write-up and pull request:** [result](00005-lag-placement-stage1b-vs-stage2-demand/result.md),
  [PR #1195 (Experiment 5: where lagged power enters a two-stage demand
  forecast)](https://github.com/openclimatefix/nged-substation-forecast/pull/1195).

![Lagged power in the calibrator beats lagged power in stage 1b at every lead](00005-lag-placement-stage1b-vs-stage2-demand/figure_2.svg)

### 00006: which inputs work best for the ERA5-trained demand model (stage 1a)

**The clock is the largest single input for Primary substations, but it lowers the error by only
about 0.5 points of capacity, and the choice of training window matters as much.** The experiment
trained one XGBoost model per substation on ERA5 weather and scored held-out ERA5 months (it does not
forecast), for 14 scored Primary substations, two bulk supply points, and two grid supply points.
It chose inputs on 2024-07 to 2025-06 and scored them once on 2025-07 to 2026-06.

- **Headline result:** hour and weekday lower the Primary error by 0.55 points (99.4% interval
  -0.71 to -0.41) on the selection year and 0.46 (-0.61 to -0.30) on the confirmation year, against
  1.5 predicted. The chosen Primary set adds a COVID indicator and the days to Christmas, which
  lowers the error by a further 0.29 on the confirmation year, supported at 95% and not at the
  planned level, and the Christmas column gains little over a year and a lot at Christmas, where it
  lowers the Primary error by 1.62 points over 15 December to 7 January. For grid supply points the
  COVID indicator beats the clock.
- **Training history and lockdown handling matter about as much as any column:** recency weights
  lower the confirmation-year error by 0.46 more. A 2023-09 start lowers it by 0.86 more, but it
  did not help on the selection year and needs a fresh year before anyone adopts it.
- **Time of year adds nothing, because the weather block already carries the season.** Windchill
  changes the error by less than 0.01 points, and trailing temperature means do not help.
- **Different substation types need different sets:** bulk supply points gain from holiday flags,
  and grid supply points gain from the COVID indicator, and have errors about four times larger.
- **Write-up and pull request:** [result](00006-stage1a-demand-input-feature-sweep/result.md),
  [PR #1202 (Experiment 00006: which input features work best for the ERA5-trained demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202).

![Error on the confirmation year by arm and substation type](00006-stage1a-demand-input-feature-sweep/figure_1.svg)

### 00012: inputs and training start for the solar and wind stage 1a model

**No input change lowered the solar or wind error by the 0.15 points the plan counted as a gain.**
The experiment tested a CAMS-trained model, a stage 0 model that calibrates ENS irradiance to CAMS, a
clear-sky-ratio target (solar), a power-curve base margin (wind), a smaller weather block, and two
later training starts, on 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** CAMS-trained minus ERA5-trained for solar is -0.140 points of capacity (99.5%
  interval -0.277 to -0.044), largest within 24 h at -0.35. The stage 0 model made the 51-member
  average 0.160 higher (-0.017 to +0.382), although it improved the control member alone from
  8.34 to 7.82. The clear-sky-ratio target (-0.048) and the base margin (-0.030) were equivalent to
  the benchmark.
- **Finding for later experiments:** training starts in 2023-09 and 2024-04 were within 0.3 points
  of 2019-09 for solar and wind, and a gain of 0.15 points is not ruled out.
- **Write-up and pull request:** [result](00012-solar-wind-stage1a-inputs-cams-and-training-start/result.md),
  [PR #1217 (Plan experiment 12: inputs and training start for the solar and wind stage 1a
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1217).

![Planned contrasts](00012-solar-wind-stage1a-inputs-cams-and-training-start/figure_2.svg)

### What the experiments taught us so far

- **Check for calendar memorisation.** When the same valid time repeats across many ENS runs, a model
  with calendar columns can memorise it. Give each experiment a climatology row, and group folds by
  valid time.
- **Choose a row set that covers every weekday.** The 53 Tuesday runs of 00001 and 00002 put every
  case of the first day on a Tuesday or Wednesday, which does not suit demand.
- **Training on ERA5 weather helps solar and wind, but a longer history does not add to it, and for
  demand a long history hurt.** The solar and wind gain appeared even when the ERA5-trained model
  saw only the ENS-era months, so the gain comes from the weather it was trained on and not from the
  years. For demand, training from 2019 was worse than training from 2023, probably because the
  customer mix and embedded generation drifted.
- **Put recent power in the stage 2 calibrator, and calibrate before judging it.** For Primary
  substations, the calibrator with recent-origin inputs beat a stage 1b correction, and a lag-free
  calibrator is worth about 1 point of capacity at short leads (00005).
- **The term and return-to-work columns showed no detectable gain on top of the Christmas column for
  Primary demand.** Over the three validation years, the return and term columns moved the Primary
  error by +0.06 points from 2 to 20 January (99.2% interval -0.17 to +0.30), +0.20 at Easter (-0.19
  to +0.59), and -0.29 over the half terms (-0.68 to +0.07), and Both was not 0.2 points better than
  a calendar shifted by 17 days (00011). The arm with the full history and the COVID indicator
  scored 0.26 points higher on the whole year than the base arm.

### 00007: does a static CAMS correction remove the stage 1a daytime residual?

**The cheap first test for P2 and P4 was inconclusive.** A correction with fixed coefficients on
ERA5 shortwave and CAMS irradiance, fitted per substation on 2024-07 to 2025-06 and scored on
2025-07 to 2026-06, removed 46.6% of the daylight squared residual at 13 Primary substations
(98.33% interval 38.4% to 53.1%), just under the planned 50%. A fixed daytime shift alone removed
39.1%, CAMS given ERA5 added a partial R squared of 0.11 (0.08 to 0.13), and the stability ratio was
0.84. One substation sets the pooled figure: without it the share is 18.2%. A CAMS-trained stage 1a
(the second stage of P2) is worth its own test, and the slope statistics of P2 wait for it. See
[`result.md`](00007-cams-static-correction-stage1a-residual/result.md) and [PR #1210 (Experiment 7:
does a static CAMS correction remove the stage 1a daytime
residual?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1210).

![Share removed by substation](00007-cams-static-correction-stage1a-residual/figure_1.svg)

### 00008: summaries of recent power in the stage 2 calibrator

**Adding the minimum, mean, and maximum of the last 24 hours and of the same weekday to the
calibrator with raw lags changed the score by no resolvable amount.** The experiment reused 00005's
stage 1 statistics on the same 73 runs, and every arm with ERA5 residuals is a ceiling.

- **Headline result:** C1 (summaries added to L3) was -0.02 points of capacity for leads up to 48
  hours (98.75% interval -0.14 to +0.13). Replacing the raw power lags with summaries was worse
  (+0.29).
- **Finding for later experiments:** summaries alone, without ERA5 residuals, scored 1.18 points
  better than no recent power up to 48 hours. They are available at forecast time, but the stage 1
  statistics beneath them come from an ERA5-trained stage 1a.
- **Write-up and pull request:** [result](00008-lag-summaries-in-stage2-calibrator-demand/result.md),
  [PR #1209 (Experiment 00008: summaries of recent power in the stage 2
  calibrator)](https://github.com/openclimatefix/nged-substation-forecast/pull/1209).

![Summaries alone clearly beat no recent power](00008-lag-summaries-in-stage2-calibrator-demand/figure_2.svg)

### 00009: a stage 1a trained on CAMS irradiance, for demand

**Training stage 1a on CAMS irradiance in place of ERA5 shortwave lowered the Primary quantile score
by 0.035 points of capacity, a sixth of the 0.2 points the plan required.** The contrast over all
leads at the 16 Primary substations is -0.035 (97.5% interval -0.070 to +0.001), and -0.043 (-0.081
to -0.004) without the substation that set experiment 00007's pooled figure. The score falls by 0.07
to 0.08 points in the first 2 days, no gain is resolved from day 7, and the gain appears in daylight
only. At the observed weather, where the CAMS-trained arm also reads CAMS, the daylight error of the
median falls by 0.88 points, so the smaller gain at ENS is probably a mismatch at application: the
ENS members' shortwave averages 7% above ERA5's, which this experiment did not test. The paired
interval was far narrower than the plan guessed (0.035 points either side, not 0.2). See
[`result.md`](00009-cams-trained-stage1a-vs-era5-trained-demand/result.md) and [PR #1213 (Experiment
9: does a stage 1a trained on CAMS irradiance beat one trained on ERA5
shortwave?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1213).

![Contrast of the CAMS-trained arm by scope](00009-cams-trained-stage1a-vs-era5-trained-demand/figure_1.svg)

### 00014: a stage 0 for the CAMS-trained demand model

**Mapping the ENS shortwave onto CAMS lowered the Primary quantile score of the CAMS-trained stage
1a by 0.067 points of capacity, but the verdict is inconclusive.** The contrast over all leads at
the 16 Primary substations is -0.067 (97.5% interval -0.094 to -0.039), in 12 of 12 months and
entirely in daylight, so the CAMS-trained arm with stage 0 is 0.102 points better than the
ERA5-trained arm. The pre-registered rule also required the corrected shortwave to stay within 5%
of CAMS in every month, and it was up to 18.6% off in December although the year's mean matched
(ratio 1.000). The gain is smallest in the first day, whose interval reaches zero. The mapping
widened the member spread by 27%, against a predicted 10%, and a post hoc monthly mapping did no
better. See [`result.md`](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) and
[PR #1219 (Experiment 14: does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained
demand
model?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1219).

![Ratio of ENS daylight shortwave to CAMS](00014-cams-trained-stage1-with-stage0-ens-correction-demand/figure_1.svg)

### 00010: the lagged electricity price in the stage 1a demand model

**Adding the Elexon price from 7 days earlier to the ERA5-trained demand model did not reliably
help at the substation hosting the largest battery, and the experiment could not test a battery
response.** The
price is Elexon's Market Index Data (the `APXMIDP` half-hourly index, free, 98.5% of half-hours
covered from 2019). The comparison arms were demand and wind at the same lag, and a price taken from
an earlier week.

- **Headline result:** at B2, the only listed series with enough history in the selection year, the
  price lowered the error by 0.92 points of capacity (98.3% interval -2.10 to +0.07), and by 0.06
  (-0.38 to +0.23) in the confirmation year. The pre-registered contrast C4, the gain at the listed
  series minus the gain at the others, was -0.55 (-1.43 to +0.20) and then +0.47 (+0.06 to +0.91),
  the wrong sign.
- **Finding for later experiments:** NGED's register lists battery storage of at least 5% of
  capacity at only 2 of the 20 series (B2 and P4), and P4 has too little history for the selection
  year, so this series set cannot test a battery response. The shuffled price from an earlier week
  gave about five sixths of the selection-year gain.
- **Write-up and pull request:** [result](00010-elexon-price-lags-stage1a-demand/result.md),
  [PR #1215 (Experiment 10: lagged electricity price in the stage 1a demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1215).

![Planned contrasts with 98.3% intervals](00010-elexon-price-lags-stage1a-demand/figure_2.svg)

### 00011: school-term and return-to-work columns in the stage 1a demand model

**The return-to-work and school-term columns did not lower the Primary demand error after Christmas,
at Easter, or over the half-term holidays.** All 20 series lie in a box that is essentially
Lincolnshire, so the experiment used Lincolnshire County Council's recommended term calendar for
2021-22 to 2025-26, collected by hand and validated by script. The base arm was the XGBoost arm of
experiment 00006 with the 14-day Christmas column, trained from 2021-09-01.

- **Headline result:** for Primary substations, the return and term columns together changed the
  error by +0.06 points of capacity from 2 to 20 January (99.2% interval -0.17 to +0.30), +0.20 over
  Easter (-0.19 to +0.59), and -0.29 over the half terms (-0.68 to +0.07), against 7.18 for the base
  arm over the whole year. Both was not 0.2 points better than either calendar shifted by 17 days,
  so C6 failed, and C5 failed because its interval reaches -0.11.
- **Verdict wording:** the rule as pre-registered gives "refuted". Its refutation clause was looser
  than intended. Read as intended, the verdict is inconclusive for Easter and the half terms,
  because the data do not rule out a gain of 0.15 points there.
- **Finding for later experiments:** the base arm's Primary error averages 8.4 points from 2 to 20
  January, and the new columns change it by +0.06. The arm with the full history and the COVID
  indicator scored 0.26 points higher on the whole year than the base arm (99.2% interval 0.12 to
  0.43).
- **Write-up and pull request:**
  [result](00011-school-term-and-return-to-work-columns-stage1a-demand/result.md), [PR #1214
  (Experiment 11: school-term and return-to-work columns for the stage 1a demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1214).

![Change in error by window and column set](00011-school-term-and-return-to-work-columns-stage1a-demand/figure_1.svg)

### 00016: calibration of the pooled mixture for solar and wind

**A scalar spread factor brings the pooled mixture's 5% to 95% band for solar from about 95% to
about 90% coverage, with no change in the CRPS.** The experiment corrected the pooled mixture of
00002's ERA5-trained stage 1a with a spread factor, a spread factor per lead band, a
location-and-scale calibrator, and a free quantile calibrator, each fitted on earlier validation
months, for 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** uncorrected solar covers 94.3% to 95.3% of non-trivial cases in every lead
  band, and `spread` gives 89.5% to 91.8% with a CRPS change of +0.006 points of capacity (99.4%
  interval -0.004 to +0.024).
- **Wind:** the uncorrected band covers 95.3% and 94.4% in the first two bands and about 90% after
  that. No planned arm brings the first two bands within 2.5 points of 90, and the fit per lead band
  existed for only 3 months of the first band. A post hoc spread factor on two lead groups
  does, with a CRPS change of -0.021 (-0.042 to -0.002), a small gain.
- **Finding for later experiments:** a location-and-scale calibrator per lead band undershoots
  90% in two wind bands (86.7% and 83.9%) and its CRPS interval is 0.36 points wide.
- **Write-up and pull request:** [result](00016-pooled-mixture-calibration-solar-wind/result.md),
  [PR #1221 (Experiment 16: calibration of the pooled mixture for solar and
  wind)](https://github.com/openclimatefix/nged-substation-forecast/pull/1221).

![Coverage of the 5% to 95% band by lead band](00016-pooled-mixture-calibration-solar-wind/figure_1.svg)
