# Research log of the ML experiments

**This page lists every ml-experiment in the order it was started, what it set out to find, and what
it found.** A reader who wants the headline of each experiment, or wants to know what is planned
next, starts here and follows the link to the experiment's `result.md` for the numbers. Each
experiment carries a production status. The maintainer reviews the candidates every one to two weeks
and ports the best into a new model release; the release log records which experiments each release
adopted.

## Candidates for the next release

**These experiments are flagged `candidate` and are not yet adopted.** The flags are proposals for
the maintainer to confirm.

| Id | What would change in production | Note |
|---|---|---|
| [00001](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Train the weather-to-power model on ERA5 and apply it to ENS members | Needs the staged design, and helps only up to day 6 |
| [00002](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Pool the ERA5-trained quantiles over the ENS members | Stage 2 itself did not help |

## The experiments so far

| Id | Question | Verdict | Production status | Merged |
|---|---|---|---|---|
| [00001](00001-era5-trained-stage1a-vs-ens-trained/result.md) | Does a weather-to-power model trained on ERA5 beat the same model trained on ENS weather? | Supported up to day 6 | Candidate | 2026-10-14 |
| [00002](00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md) | Does a stage 2 calibrator beat the pooled stage 1a mixture? | Partly supported | Candidate | 2026-10-14 |
| [00003](00003-era5-trained-stage1a-vs-ens-trained-demand/result.md) | Does the ERA5-trained model beat the ENS-trained model on demand series? | Not supported | Not for production | 2026-10-14 |

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

## Planned experiments

Each planned experiment has a `result.md` with status `planned`, committed before any code runs. An
idea that has no folder yet is marked "proposed".

| Id | Question | State |
|---|---|---|
| 00005 | Where should lagged power enter a two-stage demand forecast: stage 1b, stage 2, or both? | Planned, in [PR #1195 (Experiment 00005 (planned): where lagged power enters a two-stage demand forecast)](https://github.com/openclimatefix/nged-substation-forecast/pull/1195) |
| 00006 | Which input features are best for the demand weather-to-power model (stage 1a), trained on ERA5, for Primary, BSP, and GSP substations? | Running. Runs first, and its chosen columns feed 00005 |
| Proposed | Do statistical summaries of lagged power (the minimum, mean, and maximum of the most recent 24 hours, and of the most recent day on the same weekday) help, in the placement that 00005 finds best? | Waits for 00005 |
