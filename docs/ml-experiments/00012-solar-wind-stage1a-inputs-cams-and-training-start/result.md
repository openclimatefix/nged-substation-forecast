---
id: "00012"
slug: solar-wind-stage1a-inputs-cams-and-training-start
status: planned
size: complex
hypothesis: >-
  For the ERA5-trained stage 1a XGBoost model of the 6 solar farms and 3 wind farms, applied to the
  51 ENS members and averaged, at least one change to the inputs (CAMS irradiance in place of ERA5
  shortwave as the training input, a stage 0 model that calibrates ENS irradiance to CAMS, a
  clear-sky-ratio target for solar, or a power-curve base margin for wind) lowers the mean absolute
  error of the member average.
prediction: >-
  Against the benchmark arm `w` (the 00001 recipe, trained from 2019-09), the clear-sky-ratio
  target lowers the solar error by at least 0.3 points of capacity and the power-curve base margin
  lowers the wind error by at least 0.3 points. CAMS-trained stage 1 moves the solar error by less
  than 0.15 points either way (experiment 00009 found -0.035 for demand). The stage 0 model lowers
  the solar error by a further 0.15 points or more against plain CAMS-trained. A training start of
  2023-09 or 2024-04 moves the error of either technology by less than 0.3 points, and the reduced
  weather blocks move it by less than 0.15 points.
decision_rule: >-
  A contrast is a gain when its 99.5% interval (95% with a Bonferroni correction across the ten
  planned contrasts, resampling whole months and averaging over three seeds) lies wholly below zero
  and its point estimate is at most -0.15 points. It is a harm on the mirror-image condition. Two
  arms are equivalent when the interval lies inside -0.3 to +0.3. Supported if at least one of C1 to
  C4 is a gain. Refuted if none of C1 to C4 is a gain and every upper bound is above -0.15, so that
  no interval rules out the predicted size. Inconclusive otherwise. C5 to C10 are descriptive, and
  report equivalence or a gain per technology.
benchmark: >-
  The ERA5-trained arm `w` (the same recipe as experiment 00001's `s1_memberavg`: 9 columns for a
  solar farm and 6 for a wind farm, trained from 2019-09), a month-by-half-hour climatology row, and
  an `era5_ceiling` row.
row_set_id: fold_a_tuesdays_v1
metric: >-
  mean absolute error of the 51-member average, points of capacity (power over effective capacity,
  times 100)
---

# Which inputs and which training start work best for the solar and wind stage 1a model?

**Status: planned.** This page is the plan. It fixes the arms, the contrasts, and the decision rule
before any fit. It is proposal P17 in the [research plan](../research-plan.md).

## Size

**The size is complex, because the experiment adds a new method (a stage 0 model, a clear-sky-ratio
target, and a base margin) and admits more than one defensible design.**

- **What gets stored:** private predictions only. No schema in `packages/contracts/` changes.
- **Production serving path:** not touched. The winners are candidates for a later model release.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The definition of clear-sky power, the form of the power
  curve, and the inputs of the stage 0 model all admit alternatives. This page fixes one.
- **Code whose callers cannot be named without searching:** no. The fitting code copies the
  patterns of `studies/stage1_pilot/stage1_fit_stage1a.py` and of experiment 00009.

## Which earlier advice this plan follows

- **Followed:** every arm gets the same columns unless the arm is the contrast (00001); the
  benchmark's absolute score and a climatology row are reported beside every contrast (00001); a
  CAMS-trained stage 1 is tested against ERA5-trained with irradiance applied from ENS (00006,
  00007).
- **Departs from:** 00006 selected on one year and confirmed on the next, using ERA5 weather as the
  input. This experiment scores once, on fold A applied to ENS members, because no earlier ENS year
  exists for the farms (the ENS-era data starts in 2024-04, and the pilot scored fold A only). The
  contrasts are pre-registered and each is scored once, so nothing is selected on the scoring year.
  Winners are not combined, so no combination effect is claimed.
- **Cannot follow:** the production recipe's windchill needs temperature at the wind farms' ERA5
  cells, which is not on disk, so windchill is not tested for wind. For solar, windchill is not a
  physical input.

## Setup

**Rows and series.** The 6 solar farms (labels A to F) and 3 wind farms (W1 to W3) of
`fold_a_tuesdays_v1`: 53 runs on Tuesdays, valid times from 2025-07-01 to 2026-06-30. Anonymised
labels only appear in the write-up.

**Training.** One XGBoost model per series and seed (0, 1, 2) with 00005's hyperparameters, no
calendar column and no power lag. Training uses cleaned power (the pilot's settling rule) at valid
times before 2025-06-17, 14 days before the first run. The arm's start differs only in the start
arms.

**Weather at training and at forecast time.** Training reads ERA5 at the farm's cell (and, for CAMS
arms, CAMS irradiance at the farm). Application reads ENS members. CAMS is therefore needed only
for training, and no weather source is read at lag times because no arm has a lag.

## Arms

| Arm | Technology | What differs from `w` |
|---|---|---|
| `w` | both | Benchmark: solar 9 columns, wind 6 columns, from 2019-09 |
| `start_2023` | both | Training from 2023-09-01 |
| `start_2024` | both | Training from 2024-04-01, the ENS-era months |
| `small_block` | solar | 4 columns: `shortwave`, `temperature_2m`, `solar_zenith`, `solar_azimuth` |
| `speed_block` | wind | `wind_speed_100m` and the two 100 m bearing columns only |
| `csr` | solar | Target is power over clear-sky power, weighted by clear-sky power; the forecast is the predicted ratio times clear-sky power |
| `base_margin` | wind | XGBoost `base_margin` from a monotone power curve of 100 m wind speed fitted on the arm's training rows |
| `cams` | solar | CAMS irradiance replaces ERA5 shortwave in training; ENS shortwave is applied |
| `cams_stage0` | solar | `cams`, with each ENS member's shortwave first mapped to CAMS by the stage 0 model |

**Clear-sky power.** Clear-sky irradiance comes from `pvlib`'s simplified Solis model at the farm's
location. Clear-sky power is a farm-specific constant times that irradiance, set to the 99th
percentile of power over clear-sky irradiance, over training rows with clear-sky irradiance of at
least 300 W/m2.

**Power curve.** Mean power fraction in 0.5 m/s bins of 100 m wind speed, made non-decreasing by a
running maximum, and interpolated linearly. The same curve is added at prediction time to each
member's 100 m speed.

**Stage 0.** A small XGBoost model (depth 4, 200 rounds, learning rate 0.05, pooled over the 6 solar
farms, no farm identifier) maps ENS shortwave, longwave, 2 m temperature, 2 m dew point, solar
zenith and azimuth, and lead hours to CAMS irradiance. It trains on the control member's rows from
2024-04-01 to before 2025-06-17, daylight rows only, and applies to every member. The control member
alone trains it, because the ENS training frame holds that member only.

## Planned contrasts

Each is the arm's mean absolute error of the member average minus the reference arm's, in points of
capacity, over all leads. The level is 99.5%.

| Id | Technology | Arm | Reference |
|---|---|---|---|
| C1 | solar | `cams` | `w` |
| C2 | solar | `cams_stage0` | `cams` |
| C3 | solar | `csr` | `w` |
| C4 | wind | `base_margin` | `w` |
| C5, C6 | solar, wind | `start_2023` | `w` |
| C7, C8 | solar, wind | `start_2024` | `w` |
| C9 | solar | `small_block` | `w` |
| C10 | wind | `speed_block` | `w` |

Every table also gives the scores per technology and per lead band (0 to 24 h, 1 to 2 days, 3 to 6
days, 7 to 10 days, 11 to 15 days), with a note that the claims rest on 6 solar and 3 wind farms.

## Checks before the contrasts are read

- **Reproduction.** Arm `w` must reproduce experiment 00001's `s1_memberavg` score to within 0.05
  points per technology. If not, the pipeline has changed and the experiment stops.
- **Leakage review.** A reviewer reads the code for leakage, for every fit and for stage 0 stopping
  before 2025-06-17, and for the clear-sky constant and power curve using training rows only.
- **Stage 0 sanity.** Its error on a held-out part of the training months is printed, and it is
  compared with the ENS shortwave's error against CAMS.

## Resolution limit

Nine series in two weather regions cannot resolve gains under about 0.3 points of capacity.
Intervals are wide, and an equivalence is reported only when the interval lies inside -0.3 to +0.3.
