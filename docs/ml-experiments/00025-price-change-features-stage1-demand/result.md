---
id: "00025"
slug: price-change-features-stage1-demand
status: planned
size: complex
hypothesis: >-
  Giving the ERA5-trained stage 1 demand model the change in the intra-day price over the last
  half-hour, 1 hour, 2 hours, and 6 hours, in addition to the price at the target half-hour, lowers
  its error at observed power for Primary substations, and by more than the same columns built from
  a price taken from another week. Batteries and other dispatchable generation respond to how the
  price moves, not only to its level.
prediction: >-
  For Primary substations, PRICE_CHANGE lowers the mean absolute error of the median against
  PRICE_LEVEL by 0.15 to 0.4 points of capacity, the gain is largest from 16:00 to 20:00 local and on
  the days with the largest absolute half-hourly price change, and PRICE_CHANGE does no worse than
  NATIONAL. BSP and GSP substations show a smaller gain, if any.
decision_rule: >-
  Verdict contrasts, Primary substations only: C1 (PRICE_CHANGE - PRICE_LEVEL) and C2 (PRICE_CHANGE
  - SHUFFLED_CHANGE). Supported if both have a point estimate of at most -0.1 points and a 97.5%
  month-resampled interval (95% familywise, Bonferroni across two) that excludes zero in the
  selection year, and in the confirmation year, scored once, both intervals also exclude zero with
  the same sign. Small gain if the intervals exclude zero in both years but a point estimate is
  above -0.1. Refuted if the C1 interval lies wholly above -0.1 in both years. Otherwise
  inconclusive. The threshold of 0.1 points (about 1% of the Primary error) is set by the cost of a
  live Market Index Data feed in the pipeline that uses this model; the model needs the price at the
  target half-hour, so it cannot run as a forecast. C3 to C7 and the BSP and GSP results are
  descriptive.
benchmark: >-
  Arm BASE (experiment 00006's Primary input set, recency half-life 0.5 years), arm PRICE_LEVEL (the
  experiment 00015 arm), the other-week controls, and a month-by-half-hour climatology row.
row_set_id: era5_val_2024_07_to_2025_06_v1
metric: >-
  mean absolute error of the median, points of capacity (power over effective capacity, times 100)
verdict: null
best_arm: null
headline: null
lessons: null
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00025-price-change-features-stage1-demand/
pull_request: null
---

# Does the change in the intra-day price help a stage 1 demand model explain power?

**Status: planned.** This page is committed before any code runs. Follows
[experiment 15](../00015-same-time-price-stage1-demand/result.md), which found that the price level
at the target half-hour mostly acted as a week-level proxy, and
[experiment 10](../00010-elexon-price-lags-stage1a-demand/result.md).

## Design

**Stage 1 here is an explanatory model, not a forecaster.** It maps the ERA5 weather, the clock, and
information about the price at time t or earlier to power at time t. Every column uses only values
at or before the target half-hour. The price at t is published only after the half-hour, so no
score is a forecast skill.

**The base is experiment 00006's Primary input set with the 0.5-year recency half-life of
experiment 00017.** Folds, purge, XGBoost settings (no column subsampling), the 20 series (16
Primary, 2 BSP, 2 GSP), the seeds (0, 1, 2), and the minimum training history are those of
experiment 00015. The 14-day error correction and the rest of experiment 00019's stack are stage 2
steps and are not part of stage 1, so they are left out.

| Arm | Adds to BASE |
| --- | --- |
| BASE | nothing |
| PRICE_LEVEL | `price_now`: Market Index Data (`APXMIDP`) at the target half-hour |
| PRICE_CHANGE | `price_now`, and `d1`, `d2`, `d4`, `d12`: price(t) minus price(t-k half-hours) |
| PRICE_SHAPE | `price_now`, price minus its trailing 24-hour mean, rank within the trailing 24 hours, half-hours since the last spike (price above the trailing-24-hour mean plus 2 standard deviations, capped at 48), and the 6-hour range |
| SHUFFLED_CHANGE | PRICE_CHANGE's five columns read at another week's same weekday and half-hour |
| SHUFFLED_SHAPE | PRICE_SHAPE's five columns read at another week's same weekday and half-hour |
| NATIONAL | GB demand and wind at t, their `d1`, and demand's `d4` (6 columns) |
| SYSTEM (descriptive) | `price_now`, the system price, the net imbalance volume, and the one-half-hour changes of both |
| CLIMATOLOGY | not a fit: month-by-half-hour median of the fold's training window |

**Rolling columns use only the past.** The rolling mean, standard deviation, rank, and range are
trailing windows that end at t. The report checks that no column changes when the future rows are
removed from the price series.

**A shuffled arm takes the real derived columns from the shifted time.** The shift is 1 to 4 weeks
either way, drawn per week with a fixed seed, so the arm keeps the week's price level and the
calendar structure and loses the match to the target half-hour. A training row whose shifted time
would fall after the training end shifts backwards instead, so no price from the purge or validation
weeks enters training.

**Prices.** Market Index Data is the experiment 10 pull, 2019 to 2026, with a zero volume treated as
missing. The system price (single price since 2015) and the net imbalance volume come from Elexon
DISEBSP, fetched free for 2019-09-01 to 2025-08-31 with `studies/market_downloads/fetch_gb_prices.py`
and joined to the 2025-09 to 2026-09 pull already on disk. Market Index Data is a proxy for the
intra-day price: it is an index of short-term trading, published after the period, and it is not
the tick-by-tick intraday price in which an interconnector trip shows. The tick price is paid for
and is not used. Weather is ERA5, which arrives about 5 days late, so no arm can run at forecast
time. The system price is a second, free signal of fault stress.

## Planned contrasts

All are differences in mean absolute error in points of capacity, over the selection year (2024-07
to 2025-06) and then, once, the confirmation year (2025-07 to 2026-06), with month-resampled
intervals, pooled over the series of one substation type and the 3 seeds. The types are scored
separately and never pooled.

- **C1 (verdict, Primary):** PRICE_CHANGE - PRICE_LEVEL.
- **C2 (verdict, Primary):** PRICE_CHANGE - SHUFFLED_CHANGE.
- **C3:** PRICE_CHANGE - NATIONAL.
- **C4:** PRICE_SHAPE - PRICE_CHANGE, and PRICE_SHAPE - SHUFFLED_SHAPE.
- **C5:** PRICE_LEVEL - BASE (the experiment 00015 check).
- **C6:** SYSTEM - PRICE_LEVEL.
- **C7:** C1 by 4-hour block of local time, in the 16:00 to 20:00 peak, and on the days with the top
  5% absolute half-hourly price change.
- **BSP and GSP:** C1 to C6 for each type, labelled provisional because each type has two series.

## Sizing

- **Stores:** private predictions and a private download of system prices; nothing is committed.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (which price columns, which controls), fixed above.
- **Code whose callers could not be named without searching:** no, the code copies experiment
  00015's.

This is complex because it adds a data source and more than one defensible design. Reviews: one
Sonnet plan review, one Sonnet code review, one Sonnet write-up review. Opus is not used, by the
manager's instruction on credits.

**Resources.** Slot A: 12 cores, one process per fit with one thread, no GPU.

**Earlier advice.** Follows 00015's advice to keep an other-week price control and to try national
demand and wind, and 00017's advice to use a 0.5-year half-life. Departs from 00019's stack because
this is stage 1 only.
