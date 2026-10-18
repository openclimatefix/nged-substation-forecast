---
id: "00094"
slug: explicit-heating-offset-primary-demand
status: planned
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, adding a per-series heating term to the XGBoost
  model's base margin lowers the forecast error in cold hours (ENS 2 m temperature below 3 C),
  because the term extrapolates linearly beyond the coldest training hour and cannot be damped at
  long leads, whereas trees predict a flat response there. The term is the heating degree-hours
  (15.5 C minus the temperature, floored at zero) times a per-series, per-hour-of-day coefficient
  fitted by least squares on the training window only.
prediction: >-
  In hours below 3 C at leads up to 48 hours, the quantile score falls by at least 0.3 points of
  capacity and the share of actuals above the q95 forecast falls by at least 0.03 (from about 0.18),
  while the all-rows score changes by no more than +0.01. At leads beyond 48 hours the heating term
  cannot repair the ENS temperature forecast being too warm, so the long-lead cold-hour bias falls
  by under half. A monotone-decreasing constraint on temperature (arm `mono`) gains at most a third
  as much as the offset, because a tree stays flat beyond the coldest training hour whatever the
  constraint.
decision_rule: >-
  Four contrasts, each the treatment minus `ref` on the quantile score (1 point is 1% of the
  series' effective capacity), read at 98.75% (a 95% interval split across four) from 20,000
  resamples of whole ISO weeks of valid time, paired on series, run, and valid time. C1 is `ht`
  minus `ref` in cold hours at leads up to 48 hours. C2 is `ht` minus `ref` over all rows and leads.
  C3 is `mono` minus `ref` in cold hours at leads up to 48 hours. C4 is `mono` minus `ref` over all
  rows. The verdict is `supported` when the upper end of C1 is below -0.3 and the upper end of C2 is
  at most +0.01; `small gain` when the upper end of C1 is below 0 but not below -0.3 and the upper
  end of C2 is at most +0.01; `refuted` when the lower end of C1 is above -0.3 and the upper end of
  C1 is not below 0, or when the lower end of C2 is above +0.01; `inconclusive` otherwise. The
  threshold of 0.3 is set by the cost of porting: the offset adds a per-series coefficient fit to
  training and a base margin to serving, and a gain in 5% of rows is worth that only if it is
  material in those rows. The interval for cold hours rests on a handful of independent cold weeks
  (about 9 days carry the coldest hours), and the write-up says how many weeks each interval holds.
  Everything else (tail calibration at q05 and q95, bias by lead, per-series reads, other
  thresholds) is exploratory and labelled so.
benchmark: >-
  `ref`: the stored zero-start monthly P-roll arm `pm0` of experiment 00085, which covers all 365
  daily runs of the validation year (July 2025 to June 2026) for the 16 Primaries. Experiment 33's
  and experiment 64's stored predictions are also scored on the 73 runs of `fold_a_every5d_v1` for
  context. `manual_heuristic` and `climatology` as stored.
row_set_id: fold_a_all_daily_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity, spread factor 1.15, no 14-day correction, as in experiment 00085.
  Tail reads are the pinball loss and the exceedance share at q05 and q95. The model's top level is
  q95, so q01 and q99 are not available from these fits.
code: ml-experiments/00094-explicit-heating-offset-primary-demand/
---

# Experiment 94: does an explicit heating term lower the cold-hour error? (HT1)

Planned. The write-up replaces this page once the fits have run.

## Plan

- **Arms (three fits, all monthly refits on the latest 12 months, 150 rounds, `eta` 0.15, seed 0):**
  `ref` is experiment 85's stored `pm0`, which starts every quantile level from zero as `am` does.
  `ht` adds the heating term as the base margin. `mono` is `ref` with a decreasing monotone
  constraint on `temperature_2m`.
- **The solar offset of experiment 64's `am` is left out of every arm.** That offset acts only in
  April to September at five series, and the cold slice is December to February, so the contrast
  of interest is unchanged. The all-rows contrast is therefore `ht` against a zero-start model with
  no solar term. This is a deliberate deviation from the brief's base.
- **The heating term.** For each series and each refit, the coefficients `b_h` come from a weighted
  least-squares fit on the control-member training rows (the latest 12 months): power over capacity
  against a month-by-hour-of-day intercept and `b_h * H`, where `H` is `max(0, 15.5 - T)` and `T` is
  the row's ENS 2 m temperature (already the series' area mean). Coefficients are floored at zero.
  The margin is `b_h * H` alone, with the intercepts discarded, and is the same for all nine
  quantile levels. At prediction, each member row gets its own `H`. The fit reads only rows the
  refit is allowed to see, so there is no leakage. Hour of day is a 24-bucket local hour that is
  not a calendar position, so the margin cannot memorise the date.
- **Temperature at forecast time.** ENS 2 m temperature serves at forecast time; ERA5 is not used.
- **Row set.** All 365 daily runs, because the cold hours are few. Cold means the row's ENS member
  temperature below 3 C for the primary slice; a second slice uses ERA5 temperature as a check.
- **Compute.** Monthly P-roll: 12 refits x 16 series x 2 new arms, 8 cores, GPU at most 12 GB.
- **The optional arm scaled by NGED's Low Carbon Technology counts is skipped.** It needs a portal
  account, which needs the maintainer's agreement.
- **Sizing (the five triggers).** Stored: private predictions only. Serving path: untouched.
  Degradation rule: untouched. More than one defensible design: yes (balance point, hour shape), so
  the page reports the one design chosen and nothing else. Code with unnamed callers: none, since
  the scripts are copied from experiment 85. Medium: one Sonnet code review, one Sonnet write-up
  review, one Sonnet prose review.
- **Earlier advice followed:** monthly P-roll for contrasts of 0.05 or more (experiment 85); a
  zero-start twin as the reference (experiment 78). **Departed from:** a single seed, because
  experiment 85 found repeat fits move the score by 0.000 to 0.001.
