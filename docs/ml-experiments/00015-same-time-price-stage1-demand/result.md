---
id: "00015"
slug: same-time-price-stage1-demand
status: planned
size: medium
hypothesis: >-
  Adding the intra-day price at the same half-hour as the target (Elexon Market Index Data, observed
  at time t, not forecast and not lagged) to the ERA5-trained stage 1 demand model, which already
  sees the weather at time t, lowers its error at observed power across all fitted series pooled,
  and by more than a price taken from another week does.
prediction: >-
  Pooled over the fitted series, PRICE lowers the mean absolute error of the median against BASE by
  0.1 to 0.3 points of capacity, the gain is larger in the evening peak (16:00 to 20:00 local) than
  overnight, and the gain is larger at B2, the series with the largest battery, than at the pooled
  median series.
decision_rule: >-
  Supported if C1 (PRICE − BASE) and C2 (PRICE − SHUFFLED), pooled over fitted series, each have a
  point estimate of at most −0.1 points and a 97.5% month-resampled interval (95% familywise,
  Bonferroni across C1 and C2) that excludes zero in the selection year, and in the confirmation
  year, scored once, both intervals also exclude zero with the same sign. Refuted if the C1
  interval lies wholly above −0.1 points in both years. Otherwise inconclusive. C3, C4, and the
  residual breakdowns are descriptive and do not enter the rule.
benchmark: >-
  The experiment 00006 Primary input set with recency weights (arm BASE), the SHUFFLED price
  control, and a month-by-half-hour climatology row.
row_set_id: era5_val_2024_07_to_2025_06_v1
metric: >-
  mean absolute error of the median, points of capacity (power over effective capacity, times 100)
verdict: null
best_arm: null
lessons: null
advice: []
production_status: null
adopted_in: null
code: ml-experiments/00015-same-time-price-stage1-demand/
pull_request: null
---

# Does the same-time electricity price help a stage 1 demand model explain power?

**Status: planned.** Written before any fit. Follows [experiment 10](../00010-elexon-price-lags-stage1a-demand/result.md),
which tested the price from 7 days earlier at two battery sites and was inconclusive.

## Purpose

**The model is a tool for a later experiment, not a forecaster.** It maps the observed price and
weather at time t to power at time t. A later experiment could feed it the prices of the past to
estimate the power that should have been metered, and so measure how unusual the actual lagged power
was. That use needs two facts from this experiment: whether the price adds information beyond the
weather and the clock, and in which hours and at which series it adds it. The ERA5 weather and the
price at time t are both observed, so every score here is at observed values and none is a forecast
skill.

## Design

**The base is the experiment 00006 Primary input set**: the weather block (ERA5 at the series'
nearest grid cell), hour and weekday, the COVID indicator, and the days to Christmas inside a 14-day
window, with recency weights (half-life 2 years). Folds, XGBoost settings (no column subsampling),
purge, and the series fitted are those of experiment 10. A series needs 17,520 training half-hours
before the purge to be fitted in a fold, so the count fitted is reported for each fold.

| Arm | Adds to BASE |
| --- | --- |
| BASE | nothing |
| PRICE | `price_now`: the `APXMIDP` price of the half-hour ending at `valid_time`; missing if the half-hour is missing or has zero volume |
| SHUFFLED | `price_now` taken from another week of the same weekday and half-hour, 1 to 4 weeks away in either direction, drawn with a fixed seed (a negative control) |
| CONTROL | GB national demand outturn (`INDO`) and wind output (`FUELHH`) at the same half-hour, the non-price national signals the price follows |
| CLIMATOLOGY | not a fit: the month-by-half-hour median of the fold's training window, as `ClimatologyForecaster` computes it |

**Weather and price source.** The weather is ERA5, which arrives about 5 days late, so the model
cannot run at forecast time. That is intended, because the model scores past power. The price is
the experiment 10 pull of `APXMIDP`, 98.5% complete and fully complete from 2024-04-14. Training
rows before 2021 can have a missing price, which XGBoost handles natively.

**Calendar memorisation.** BASE has hour and weekday, so the price column could stand in for the
calendar. SHUFFLED keeps the price's calendar structure and removes its information, and C2 is the
contrast that controls for it.

## Planned contrasts

All are differences in mean absolute error, in points of capacity, over the selection year (2024-07
to 2025-06) and then, once, the confirmation year (2025-07 to 2026-06), with a month-resampled
interval, pooled over the fitted series and the 3 seeds.

- **C1 (verdict):** PRICE − BASE.
- **C2 (verdict):** PRICE − SHUFFLED.
- **C3 (descriptive):** PRICE − CONTROL.
- **C4 (descriptive):** C1 by local hour of day, in 4-hour blocks.

**Residual breakdowns, descriptive.** For each local hour and each series, the report gives the
mean absolute error of BASE and PRICE, and the share of BASE's squared residual that PRICE removes.
This is the number the later experiment needs. The B2 gain is reported beside the pooled gain.

## Sizing

**Sizing, one line per `plan-issue` trigger.** Stores: private predictions only. Serving path: not
touched. Degradation rule: not touched. More than one defensible design: yes (which price columns,
which controls), fixed above. Code whose callers could not be named without searching: no, the code
reuses experiment 10's. This is medium: one Sonnet plan review, one Sonnet code review, one Sonnet
write-up review.

**Resources.** Slot B, 8 CPU cores, one thread per fit, no GPU.

**Earlier advice.** Follows experiment 10's advice to keep a shuffled-price control and a
climatology row. Departs from its advice to test on series with large batteries, because the
maintainer's design pools all series and reports B2 beside the pool.
