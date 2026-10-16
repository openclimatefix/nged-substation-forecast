---
id: "00054"
slug: product-trained-xgboost-wind-farms
status: planned
size: complex
hypothesis: >-
  For the three metered wind farms, an XGBoost quantile model trained on ENS and given UKV's wind as
  extra input columns scores lower on the quantile score at lead days 1 and 2 than the same model
  trained on ENS alone, with every arm trained on the same window, rows, and seeds. UKV is read from
  the CEDA archive, which is not the live product, so the contrast is a ceiling read.
prediction: >-
  The ENS-plus-UKV arm scores -0.2 to -0.5 points of capacity against the ENS-only arm at the lead
  hours 24 to 54 after the ENS run's 00 UTC (the UKV-CEDA blends study found -0.2 to -0.3 at lead
  days 1 and 2 for an ENS mean plus UKV winds). The UKV-alone arm scores worse than the ENS-only arm.
  Every arm scores worse than the ERA5-trained best guess, as the ERA5-trained wind model beat the
  ENS-trained one by 1.70 points in experiment 1. Experiment 52's interval on the same farms had a
  half-width of about 0.14 points at 95%, so a `supported` verdict (interval wholly below -0.15)
  needs a point estimate near -0.30 or better, and `inconclusive` is possible.
decision_rule: >-
  One decisive contrast, C1: the ENS-plus-UKV arm minus the ENS-only arm on the quantile score for
  cases with a valid time 24 to 54 hours after the 00 UTC start of the ENS run's date (lead day 1,
  and the first 6 hours of lead day 2), read at 95% with whole months and the three seeds
  resampled, 50,000 resamples. The interval is conditional on the three farms. The threshold is 0.15
  points, because an added weather feed and an added model input column are a pipeline step each,
  and a smaller gain does not justify them. C1 is `supported` when its interval lies wholly below
  -0.15, `small gain` when it lies wholly below zero but not wholly below -0.15, `refuted` when it
  lies wholly above zero, and `inconclusive` otherwise. Because the CEDA archive is not the live
  UKV product, a `supported` C1 is reported as `small gain` for production and `supported` as a
  ceiling, and the verdict line says both. Every other read (the UKV-alone arm, per farm, per lead
  day, against the ERA5-trained best guess) is exploratory, labelled so, and read at 95%.
benchmark: >-
  The ENS-trained XGBoost arm with the same training protocol (arm A), on the same cases. The
  ERA5-trained best guess of experiment 33 is a reference row, not the benchmark.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; coverage of the 5% to 95% band; share of cases above the 0.95
  quantile
production_status: not-for-production
adopted_in: none
code: ml-experiments/00054-product-trained-xgboost-wind-farms/
---

# Experiment 54: does UKV, learned from its own history, help the wind-farm forecast?

**Status: planned. No code has run.** [Experiment 52](../00052-more-nwp-products-for-wind-farms/result.md)
fed ICON-EU and IFS 0.25 degree into a wind-power model trained on ERA5, an estimate of past
weather, and could not tell whether they help. That design favours ENS, because the ERA5-trained
model never learned either product's own errors. This experiment trains every model on a forecast
product's own history, with one training protocol for every arm, and the product is UKV, at the
maintainer's request, relayed by the manager. ICON-EU, IFS HRES 9 km, and GFS are not run.

## The three arms

**Each arm is an XGBoost model with nine quantile levels, trained per farm and per seed with the
same window, the same recency weights, and the same lagged-power inputs.** The training protocol is
that of experiment 33's ENS-trained model (N1): the ENS control member of every daily 00 UTC run
with valid times from 2024-04-01 to 2025-06-30, a recency weight with a half-life of 2 years, the
wind columns, `lead_hours`, and three lags of power from the forecast origin (09:00 UTC).

- **Arm A, ENS only:** the six ENS wind columns (speed and direction at 10 m and 100 m, direction as
  sine and cosine).
- **Arm B, ENS plus UKV:** the six ENS columns and four UKV columns (10 m speed, the sine and cosine
  of the 10 m direction, and the speed at 925 hPa, about 750 m above sea level; UKV-CEDA archives no
  100 m wind). At prediction the model runs once per ENS member with the UKV columns fixed, and the
  51 member quantile sets are pooled, as arm A does.
- **Arm C, UKV alone:** the four UKV columns. At prediction the model gives one set of nine
  quantiles per case.
- **Reference row:** experiment 33's ERA5-trained best guess (`bg_full`) and its ENS-trained N1,
  scored on the same cases. They are not the control.

**All three arms train and score the same rows: valid times 0 to 54 hours after the 00 UTC start of
the ENS run's date, and only the 24 to 54 hour cases are scored.** UKV's leads stop at 54 hours, so
the scored cases are lead day 1 (24 to 48 hours) and the first 6 hours of lead day 2 (48 to 54
hours). The page says "days 1 and 2" for these cases and states the 54-hour limit wherever it quotes
a number. Training on the lead hours 0 to 54 gives the models more rows than the scored range alone
and does not use any validation data.

## UKV, as read, and what the data on disk supports

**UKV is the CEDA archive's 00 UTC run of the ENS run's date, aligned with the ENS run.** The
archive (`UKV-CEDA` and its later parts) holds runs at 00, 06, 12, and 18 UTC at 2 km, hourly to 48
hours and 3-hourly to 54 hours. A valid time at an hour that has no archived step (49, 50, 52, and 53
hours) interpolates linearly between the steps either side, and a valid time at :30 takes the
average of the east-west and north-south wind components of the hours either side. The wind
farms' nearest 2 km cell is read, with the cell-matching code of the matched-lead study.

**The archive covers the whole training window and nearly the whole validation year.** A count on
disk shows 455 of the 456 daily 00 UTC runs of 2024-04-01 to 2025-06-30 with a complete wind set,
and 52 of the 53 Tuesday validation runs; the 2026-06-30 run is missing and its cases are dropped
from every arm.

**No matched live history exists for the wind farms, so the result is a labelled ceiling.**
Open-Meteo's UKV history at the three wind farms holds only `previous_day1` (a forecast issued at
least 1 day before the hour); `previous_day2` to `previous_day7` are empty. `previous_day1` starts
on 2024-08-06 and holds 7,769 of the 10,944 hourly values of the ENS window at one farm (71%), so
it could support lead day 0 only. The CEDA archive is not the live product: its licence is
non-commercial, the [UKV-CEDA blends study](../../studies/forecasts/ukv-ceda-blends.md) found it
statistically different from the live feed, and UKV was upgraded on 2026-01-21, inside the
validation year. A UKV-CEDA gain therefore says what a matched UKV might add, not what a deployable
one would.

**The latency UKV has live is stated, not measured.** The service reads ENS at 09:00 UTC. The
experiment assumes the 00 UTC UKV run is available within 8 hours of its start, from the Met
Office's published schedule, and does not measure when it publishes.

## Caveats

- **ENS-trained models are weaker for wind.** Experiment 1 found the ERA5-trained wind model beat
  the ENS-trained model by 1.70 points, so arm A and arm B sit above the best-guess reference row.
- **UKV's history is short.** Training uses 455 daily runs (about 15 months).
- **A UKV upgrade lies inside the validation year** (2026-01-21), so part of the validation year
  runs on a version the models never saw in training.
- **The cases are the same Tuesday runs and three farms as experiment 52's**, so the contrast is
  not a confirmation on fresh data.
- **Arm B's pooled mixture may be too narrow.** Its spread comes only from the six ENS columns, so the
  more the trees lean on UKV, the narrower the pooled mixture. No spread factor corrects this; the
  report's coverage and above-0.95-quantile share show it.
- **The scored cases cover lead day 1 and 6 hours of lead day 2 only**, because UKV stops at 54
  hours. A result at days 1 and 2 of ENS-based forecasts in other experiments is not comparable.

## Sizing

- **What gets stored:** private prediction tables under `data/ml-experiments/`. Not a trigger.
- **Serving path:** untouched. A `supported` verdict would add a weather feed and a model input.
- **Degradation rule:** untouched. A missing UKV run would fall back to arm A.
- **Defensible designs:** more than one (UKV as extra columns, as a second model in a blend, or as a
  calibrator input). The plan follows the maintainer's request for arms A, B, and C.
- **Callers:** the experiment copies experiment 33's N1 recipe and the matched-lead study's UKV
  reading, which it may not import.
- **Size: complex** (two triggers). One Sonnet plan review, one Sonnet code review, one Sonnet prose
  review, and one Opus result review only if the result needs it.

## Design details

- **Cases:** the half-hourly cases of experiment 33's stored `bg_full` table (three farms, three
  seeds, the Tuesday runs), restricted to valid times 24 to 54 hours after the 00 UTC start of the
  ENS run's date. A case where UKV is absent is dropped from every arm, so C1 is scored on the
  52 Tuesday runs that have a complete UKV run, not on all 53 stored runs.
- **Training rows:** every 00 UTC run of 2024-04-01 to 2025-06-30 for which the UKV run is complete,
  per farm, using the ENS control member's half-hourly valid times 0 to 54 hours after the run's
  00 UTC. The `lead_hours`, lag, and recency-weight definitions are experiment 33's. Arms A, B, and
  C train on the same rows.
- **Fits:** the nine-level quantile loss, `eta` 0.05, depth 6, 500 rounds, on the GPU, as in
  experiment 33; seeds 0, 1, and 2.
- **Spread factor and blend:** none. Every arm is scored as the model outputs it.
- **Missing values:** a case or training row with a non-finite value in any UKV column (the 925 hPa
  speed is NaN where the level lies below the ground) is dropped from all three arms alike.
- **Units and cells:** UKV-CEDA speeds are in m/s, as ENS's are. Each farm reads its nearest 2 km
  cell (the three farms lie 0.9 to 1.1 km from one), by `studies.grid_sampling.nearest_cells`; the
  store reader is copied from the study script that already reads it.
- **Exploratory splits:** C1 is also read before and after UKV's upgrade of 2026-01-21, and without
  the first validation month.
- **Alignment assertion:** every run read has `init_time` at 00:00 on the ENS run's date.
- **Scores reported:** the absolute quantile score of every arm per farm and per lead-day band (lead
  day 1 and the first 6 hours of lead day 2); the coverage of the 5% to 95% band; the share of cases
  above the 0.95 quantile; UKV's mean 10 m wind speed on the validation year against ENS's and
  ERA5's; the number of training rows and runs per arm. Cases at lead day 0 (0 to 24 hours) are
  scored and reported as exploratory, labelled as outside the contrast.
- **Placebo:** arm B with the UKV columns shuffled among rows of the same farm, month, lead hour of
  day, and lead day, in training and in scoring. It shows how much of any gain comes from the extra
  columns' mere presence.

## Advice followed and departures

- **Followed:** experiment 52's advice to rerun with the product aligned to the ENS run, and not to
  fit a spread factor on blends made under other weights.
- **Departed from:** experiment 52's advice to correct each product's wind speed. The trees learn
  the product's bias from its own history, so no separate correction is fitted.
- **Departures from earlier plans:** no ERA5-trained model, blend, or spread step is used; ICON-EU,
  IFS HRES 9 km, and GFS are dropped; the contrast is arm B minus arm A.

## Compute

Slot E: 6 cores, GPU at most 12 GB. Each seed fits arms A, B, and C for three farms (9 fits), so 27
fits in all, plus 9 for the placebo, each on about 40,000 rows (455 runs times about 89
half-hourly valid times). The XGBoost thread count is set to 6.
No download.
