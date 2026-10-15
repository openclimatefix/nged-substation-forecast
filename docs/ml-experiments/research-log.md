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
| [00017](00017-training-window-and-retraining-rule-drift-demand/result.md) | Train the stage 1a demand model on the last 12 months or with a 0.5-year half-life, retrain monthly, and add a 14-day error correction | The correction needs live power 2 days old, and every gain is on ERA5 weather |
| [00019](00019-primary-demand-stack-vs-manual-heuristic/result.md) | Add a 14-day error correction and a spread factor of 1.15 to the single-stage ENS-trained quantile model N1 for Primary demand | The correction supplies nearly all of the gain (-0.99); it is tested on forecasts 5 days apart and on a model that is never retrained |
| [00016](00016-pooled-mixture-calibration-solar-wind/result.md) | Scale the solar mixture's quantiles about the median by a factor near 0.95, refitted monthly | Fixes 94% to 95% coverage at no change in CRPS; wind needs a spread factor on two lead groups, a post hoc arm |
| [00020](00020-solar-wind-stack-cams-stage0-spread-margin/result.md) | Add a power-curve base margin to the wind stage 1a quantile model | Lowers the wind CRPS by 0.055 points (0.007 to 0.110); the stack of margin and spread factor gives 0.071; nothing detectably helped solar |

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
| [00013](00013-primary-demand-against-stored-baselines/result.md) | Where does Primary substation demand stand: does an upgraded ENS-trained quantile model (N1) beat `manual_heuristic`, and why does the production recipe lose to it? | Inconclusive: N1 is level with `manual_heuristic`; the production recipe trails it by 1.69 points | Not for production | 2026-10-14 |
| [00014](00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md) | Does a stage 0 that maps ENS irradiance onto CAMS help the CAMS-trained demand model? | Inconclusive: a gain of 0.067 points, but monthly mismatches remain | Candidate | Not yet merged |
| [00017](00017-training-window-and-retraining-rule-drift-demand/result.md) | Do a recent training window, recency weights, monthly retraining, or a 14-day error correction lower the demand error? | Supported | Candidate | 2026-10-14 |
| [00016](00016-pooled-mixture-calibration-solar-wind/result.md) | Does a correction bring the 5% to 95% band of the pooled mixture to 90% coverage for solar and wind? | Solar supported, wind refuted (a post hoc arm for wind reaches 90%) | Candidate | 2026-10-14 |
| [00015](00015-same-time-price-stage1-demand/result.md) | Does the same-time electricity price help a stage 1 demand model explain power across all fitted series? | Inconclusive | Not for production | 2026-10-14 |
| [00018](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/result.md) | Does the staged chain, with lag anomalies from the weather sources serving has, beat N1 and `manual_heuristic` on Primary substation demand? | Inconclusive: the chain is level with N1 (+0.08) and trails `manual_heuristic` (+0.52) | Not for production | 2026-10-14 |
| [00019](00019-primary-demand-stack-vs-manual-heuristic/result.md) | Does stacking a 14-day error correction, a 0.5-year half-life, and a spread factor of 1.15 on N1 beat `manual_heuristic` for Primary substation demand? | Supported: the stack beats `manual_heuristic` (-0.56) and N1 (-1.00); the correction supplies nearly all of it | Candidate | Not yet merged |
| [00020](00020-solar-wind-stack-cams-stage0-spread-margin/result.md) | Do the candidate changes for solar and wind (CAMS-trained stage 1a, a stage 0 quantile mapping, a power-curve base margin, a spread factor) add up to a lower CRPS than the pooled ERA5-trained mixture? | Partly supported: wind gains 0.071 points, solar does not | Candidate (wind) | Not yet merged |
| [00021](00021-error-correction-variants-bsp-gsp-demand/result.md) | Does the recent-error correction help BSP and GSP demand, and which variant works best? | Inconclusive for BSP and GSP; the 14-day additive form is the best variant | Not for production | 2026-10-15 |
| [00022](00022-era5-weather-ceiling-primary-demand/result.md) | How much would Primary demand gain from ERA5 weather in place of ENS weather at serving time, as a ceiling for better weather products? | Refuted: a gain of 0.10 points, not 0.3 | Not for production | 2026-10-15 |
| [00023](00023-lagged-power-and-error-correction-solar-wind/result.md) | Does recent observed power in the stage 2 calibrator, or a 14-day error correction, lower the CRPS of solar and wind forecasts? | Inconclusive: no sign of a benefit | Not for production | Not yet merged |

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

### 00021: the error correction on BSP and GSP demand, and its variants

**The 14-day error correction did not lower the BSP and GSP quantile score, and no variant beat the
plain 14-day additive form.** The experiment post-processed N1's stored forecasts with power that is
2 days old, on 73 forecast runs with ENS weather.

- **Headline result:** the correction minus N1 on the 4 BSP and GSP series was +0.36 points of
  capacity over all leads (99.375% interval -0.19 to +0.98). On the 16 Primary substations it was
  -0.99 (-1.82 to -0.18), the same as with power 3.5 hours old.
- **Absolute scores (all leads, BSP and GSP):** N1 13.66, corrected 14.02, `manual_heuristic` 14.70,
  `climatology` 15.11.
- **Variants (Primary, minus the 14-day additive form):** 7-day window +0.19, 28-day window -0.07
  (-0.15 to +0.01), multiplicative +2.28, per-half-hour +0.02, recency-weighted +0.04.
- **Finding for later experiments:** N1's BSP series have no bias (+0.01), so a bias correction has
  nothing to remove there; the GSP series are biased low by 5.2 points, the correction removes 3.2,
  and the score still does not improve.
- **Write-up and pull request:** [result](00021-error-correction-variants-bsp-gsp-demand/result.md),
  [PR #1228 (Experiment 21: which error correction works best, and does it help BSP and GSP
  demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1228).

![The eight planned contrasts](00021-error-correction-variants-bsp-gsp-demand/figure_1.svg)

### 00022: the weather ceiling for Primary demand

**ERA5 weather in place of ENS weather at serving time lowered the Primary demand quantile score by
only 0.10 points, so better weather products have little to offer Primary demand.** The experiment
used experiment 19's full stack on the 16 Primary substations and 73 forecast runs. ERA5 arrives
about 5 days late, so the arm is a ceiling and no serving system could use it.

- **Headline result:** the all-variable ERA5 arm minus the ENS arm was -0.10 points of capacity over
  all leads (99.38% interval -0.16 to -0.04); the ENS arm scored 5.32 and the ERA5 arm 5.22.
- **By variable:** shortwave radiation alone gave -0.08 (-0.13 to -0.04); temperature gave -0.01
  (-0.04 to +0.01).
- **By lead:** the gain was -0.08 at 3 to 6 days and -0.15 at 11 to 15 days, and ERA5 was slightly
  worse (+0.03) under 24 hours.
- **Finding for later experiments:** if a second weather product is tried for Primary demand, test
  its shortwave radiation first.
- **Write-up and pull request:** [result](00022-era5-weather-ceiling-primary-demand/result.md), [PR
  #1227 (Experiment 22: how much would Primary demand gain from ERA5 weather in place of ENS
  weather?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1227).

![Each ERA5 arm minus the ENS arm](00022-era5-weather-ceiling-primary-demand/figure_1.svg)

### What the experiments taught us so far

- **A 14-day error correction put N1 below `manual_heuristic` for Primary demand.** The correction
  lowered the quantile score by 0.99 points on N1, a model frozen at its training end, and the 0.5-year
  half-life added nothing to that frozen model (00019).
- **Use the 14-day additive error correction for Primary demand, with power 2 days old, and do not
  yet use it for BSP and GSP.** It scored the same with 2-day-old power as with 3.5-hour-old power,
  a 7-day window and a ratio form were worse, and on 4 BSP and GSP series it did not lower the
  quantile score (00021).
- **A weather product as accurate as ERA5 would lower the Primary demand score by 1.9%.** ERA5 weather
  in place of ENS weather lowered the quantile score by 0.10 points, nearly all from shortwave
  radiation and none up to 2 days ahead (00022).
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
- **A change that helps the point forecast of solar or wind need not help the pooled mixture's
  CRPS.** CAMS-trained solar lowered the member-average error by 0.14 points (00012) and the
  mixture's CRPS by 0.009 (00020), and a stage 0 mapping widened the member spread by a quarter. A
  power-curve base margin lowered the wind CRPS by 0.055 (00020).

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

### 00013: Primary substation demand against the stored baselines

**The upgraded single-stage model N1 did not beat `manual_heuristic`, and the production recipe trails
`manual_heuristic` by 1.69 points of capacity at every lead.** The experiment scored `manual_heuristic`,
`climatology`, the production recipe, a stage 1 trained on ERA5, and N1 (an XGBoost quantile model
trained on the ENS control member, with lagged power measured from the forecast origin) on the 16
Primary substations and the 73 runs of `fold_a_every5d_v1`.

- **Headline result:** N1 minus `manual_heuristic` was +0.22 points of capacity over all leads (99%
  interval -0.54 to +0.97) and -0.57 up to 48 hours (-1.33 to +0.27) in the mean absolute error of the
  median. The production recipe minus `manual_heuristic` was +1.69 (+0.98 to +2.59).
- **Absolute scores (all leads):** `manual_heuristic` 8.23, N1 8.45, production 9.92, `climatology`
  10.33 points of capacity.
- **Why the production recipe loses:** about 0.6 points is a level offset (every arm forecast the
  validation year 2 to 3 points too low, `manual_heuristic` by 1.3), and the rest matches the lack of
  recent power, which lagged power from the origin recovers (1.15 points, C4).
- **Finding for later experiments:** compare Primary arms against `manual_heuristic`, and test a
  drift correction in P8.
- **Write-up and pull request:** [result](00013-primary-demand-against-stored-baselines/result.md),
  [PR #1218 (Experiment 13: Primary substation demand against the stored
  baselines)](https://github.com/openclimatefix/nged-substation-forecast/pull/1218).

![The five planned contrasts](00013-primary-demand-against-stored-baselines/figure_2.svg)

### 00020: stacking the candidate changes for solar and wind

**For wind, a power-curve base margin and a spread factor together lower the pooled mixture's CRPS
by 0.071 points of capacity, and the base margin carries the gain; for solar, nothing helped.** The
experiment stacked a CAMS-trained stage 1a, a stage 0 quantile mapping, the base margin (wind only),
and the spread factor against the production recipe's pooled ERA5-trained mixture, and dropped each
component in turn, for 6 solar farms and 3 wind farms (row set `fold_a_tuesdays_v1`).

- **Headline result:** wind full stack minus recipe, -0.071 (99.58% interval -0.133 to -0.014); solar
  full stack minus the production recipe, +0.035 (-0.075 to +0.116).
- **Components:** the wind base margin alone gives -0.055 (-0.110 to -0.007). For solar, CAMS-trained
  gives -0.009, stage 0 +0.036, and the spread factor +0.006, each with an interval that includes
  zero.
- **Finding for later experiments:** the stage 0 quantile mapping widens the member spread by about a
  quarter and leaves monthly ratios to CAMS from 0.83 to 1.10, and it did not lower the solar CRPS
  in this experiment.
- **Write-up and pull request:** [result](00020-solar-wind-stack-cams-stage0-spread-margin/result.md),
  [PR #1226 (Experiment 20: do the candidate changes for solar and wind add up to a better pooled
  mixture?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1226).

![Planned contrasts](00020-solar-wind-stack-cams-stage0-spread-margin/figure_1.svg)

### 00018: the staged chain against the single-stage model for Primary demand

**The staged chain, built only from weather that serving has and with NGED's real telemetry delay,
had a quantile score not distinguishable from N1's and a point estimate above
`manual_heuristic`'s.** The experiment gave the stage 2 calibrator lag anomalies from ERA5T, CAMS,
and the ENS control member's early leads, with the newest usable power 7 hours before the 09:00 UTC
origin.

- **Headline result:** the chain minus N1 was +0.08 points of capacity over all leads (99.17%
  interval -0.11 to +0.25), -0.27 up to 48 hours (-0.62 to +0.03). The chain minus
  `manual_heuristic` was +0.52 (-0.25 to +1.25).
- **Absolute scores (all leads):** `manual_heuristic` 5.89, N1 6.33, chain 6.41.
- **What the serving sources cost:** 0.13 points against final ERA5, and 3.5 more hours of telemetry
  delay cost another 0.13. Lag anomalies lowered the score by 0.04 against raw lagged power.
- **Finding for later experiments:** a spread factor of 1.15 lifts the 5% to 95% coverage from 86%
  to 90%, and every arm under-forecast the validation year by 1.3 to 3.5 points.
- **Write-up and pull request:** [result](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/result.md),
  [PR #1224 (Experiment 18: does the staged chain beat the single-stage model for Primary
  substation demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1224).

![The six planned contrasts](00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/figure_2.svg)

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

### 00017: training window, recency, and retraining rule for demand

**Question.** Does training the stage 1a demand model on recent rows, retraining it monthly, or
correcting it by its last 14 days of error lower the Primary substation demand error, as drift
suggests?

**Verdict: supported.** On 16 Primary substations, with monthly rolling origins from January 2022 to
June 2026 on ERA5 weather, all-history training retrained monthly scores 7.62 points of capacity.
The 14-day correction lowers it by 1.52 points (99.44% interval 1.13 to 1.95), a half-life of 0.5
years by 0.92 (0.64 to 1.22), and a 12-month window by 0.84 (0.53 to 1.19). Retraining yearly instead
of monthly adds 0.52 points, and a model never retrained adds 3.80.

**What changes later experiments.** The bias of the all-history model runs from forecast too high
(2022 to 2025) to too low (2026), so a window change needs several years of rolling origins to be
judged. The correction stacks with a 0.5-year half-life (5.82 points, post hoc).

![Contrasts against the all-history model](00017-training-window-and-retraining-rule-drift-demand/figure_1.svg)

Details: [result.md](00017-training-window-and-retraining-rule-drift-demand/result.md),
[PR #1223](https://github.com/openclimatefix/nged-substation-forecast/pull/1223).

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

### 00019: stacking small gains on N1 for Primary demand

**Stacking a 14-day error correction, a recency half-life, and a spread factor on N1 beat
`manual_heuristic` on the quantile score, and the correction supplied nearly all of the gain.** The
experiment used the 16 Primary substations, ENS weather, and 73 forecast runs.

- **Headline result:** the stack minus `manual_heuristic` was -0.56 points of capacity over all leads
  (99.44% interval -0.93 to -0.26), and the stack minus N1 was -1.00 (-1.88 to -0.14).
- **Absolute scores (all leads):** stack 5.32, `manual_heuristic` 5.89, N1 6.33.
- **Components:** dropping the correction from the stack raised the score by 0.97 (+0.15 to +1.81);
  the 0.5-year half-life moved it by -0.01 (-0.03 to +0.02); dropping the spread factor raised it by
  0.03 (+0.005 to +0.05).
- **Finding for later experiments:** N1 forecast the validation year 2.6 points too low, and the
  correction cut that bias to 0.5. The correction was tested on forecasts 5 days apart and on a model
  that is never retrained.
- **Write-up and pull request:** [result](00019-primary-demand-stack-vs-manual-heuristic/result.md),
  [PR #1225 (Experiment 19: does stacking small gains beat manual_heuristic for Primary
  demand?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1225).

![The nine planned contrasts](00019-primary-demand-stack-vs-manual-heuristic/figure_2.svg)

### 00015: the same-time electricity price in a stage 1 demand model

**The price at the same half-hour as the target lowered the pooled demand error in the selection
year, but a price from another week did almost as well, and the gain did not repeat in the
confirmation year.** The stage 1 demand model maps the observed price and ERA5 weather at time t to
power at time t
on 17 series (18 in the confirmation year), scored at observed values.

- **Headline result:** price against no price, pooled: -0.52 points of capacity (97.5% interval
  -0.88 to -0.11) in the selection year and -0.18 (-0.50 to +0.08) in the confirmation year. Price
  against a price from another week: -0.07 (-0.17 to +0.03) and -0.06 (-0.18 to +0.07).
- **Finding for later experiments:** national demand and wind at time t did no worse than the price
  in either year (pooled intervals include zero; 0.43 points lower in the confirmation year), so the
  price may be a proxy for national conditions. The gain was largest from 12:00 to 20:00 and at the
  series with the largest errors, and the other-week price reproduced most of it there.
- **Write-up and pull request:** [result](00015-same-time-price-stage1-demand/result.md), [PR #1220
  (Experiment 15: same-time price in a stage 1 demand
  model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1220).

![Pooled contrasts with 97.5% intervals](00015-same-time-price-stage1-demand/figure_1.svg)

### 00023: recent observed power for solar and wind

**Recent power did not help the stage 2 calibrator for solar or wind, and a 14-day correction built
from the mean error made the forecast worse.** The experiment gave the calibrator lagged power and
summaries of the latest day for 6 solar farms and 3 wind farms, and shifted the pooled mixture by
its own mean error over the previous 14 days (row set `fold_a_tuesdays_v1`).

- **Headline result:** lagged power added to the calibrator, solar -0.012 (99.58% interval -0.122 to
  +0.088), wind +0.012 (-0.090 to +0.146). The 14-day correction on the mixture, solar +0.437
  (+0.215 to +0.664), wind +0.413 (-0.003 to +0.924).
- **Finding for later experiments:** a calibrator trained on the ENS control member alone scored 0.27
  (solar) and 0.31 (wind) points worse than the pooled mixture before any lag was added. Shifting by
  the median error (post hoc) instead of the mean error moved the CRPS by +0.010 (solar) and -0.064
  (wind), both with intervals that include zero.
- **Write-up and pull request:** [result](00023-lagged-power-and-error-correction-solar-wind/result.md),
  [PR #1229 (Experiment 23: does recent observed power help solar and wind forecasts?)](https://github.com/openclimatefix/nged-substation-forecast/pull/1229).

![Contrasts](00023-lagged-power-and-error-correction-solar-wind/figure_1.svg)
