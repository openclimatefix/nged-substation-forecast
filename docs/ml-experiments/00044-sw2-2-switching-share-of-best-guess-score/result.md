---
id: "00044"
slug: sw2-2-switching-share-of-best-guess-score
status: planned
size: simple
hypothesis: >-
  Half-hours in labelled switching events, and half-hours whose lagged power falls inside one, carry
  a larger share of the best guess's quantile score and tail score than their share of half-hours,
  and a perfect switching detector could lower the Primary quantile score or tail score by 3% or
  more.
prediction: >-
  Half-hours inside trustworthy-band events (moved load of 25% or more of the receiving load,
  lasting over 1 day) are under 3% of the 16 Primary series' half-hours. Their share of the quantile
  score is at least 1.5 times their share of half-hours. Adding the half-hours whose lagged power
  falls inside an event raises the flagged share of half-hours to 5% to 15%. A perfect detector would
  lower the quantile score by 1% to 5%.
decision_rule: >-
  `supported` when the 95% interval of the perfect-detector reduction (trustworthy band, (a-after)
  plus (b), metered view, series-mean counterfactual) has a lower end of 1% or more, and its point
  estimate is 3% or more, for the quantile score or the tail score. `refuted` when the upper end of that
  interval is under 3% for both scores. `inconclusive` otherwise. The 3% threshold is set by the
  cost of the change it would justify: a switching detector adds a model and a pipeline step to
  the serving path, so it needs a gain well above zero. The intervals resample whole months, so
  they are wide when the events fall in a few months, and `inconclusive` is a likely result.
benchmark: >-
  The same cases scored outside the flagged half-hours (the unflagged mean loss of the same series).
row_set_id: >-
  fold_a_every5d_v1 (experiment 00033's 73 runs for the 16 Primary series, July 2025 to June 2026)
metric: >-
  Quantile score (points of capacity) and threshold-weighted CRPS at the P99 threshold (twCRPS), as
  `case_metrics` in experiment 00033 computes them, summed over cases and averaged over the 3 seeds.
verdict: ""
best_arm: ""
lessons: ""
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00044-sw2-2-switching-share-of-best-guess-score/
---

# Experiment 44: how much do visible switching events hurt the best guess?

*Planned; this page is the plan and will be rewritten as the write-up.*

## Plan

**Score experiment 00033's best guess again, split by whether a half-hour is in a labelled
switching event, and fit nothing.** The best guess for Primary demand is N1 (an XGBoost model per
substation, trained on the ECMWF ensemble forecast, with recent power as inputs) plus the 14-day
error correction and a spread factor of 1.15. Its per-case scores are in
`data/ml-experiments/00033-*/work/metrics/Primary/n1_cs.parquet`. The predictions are not in
`experiment_store`, because experiment 00033 kept its arms in its own work folder; this experiment
reads the per-case scores read-only and re-uses its row set. The labels are experiment 41's
`labelled_set.parquet` (SW2-0b), which cover the 16 Primary substation series only, so BSP, GSP,
solar, and wind are not scored.

### Which half-hours are flagged

- **Category (a), the target is in an event.** The case's valid time lies between a labelled event's
  onset and end, for the donor or any named receiver in that series. It splits in two by whether a
  detector could have known: (a-before) the forecast origin is before the onset, so no detector can
  anticipate the event, because NGED plans switching in its own systems and not in the telemetry;
  (a-after) the origin is at or after the onset.
- **Category (b), the lagged power is in an event.** At least one of these stamps lies in an event of
  the case's series: the newest usable observation, the same time on the latest earlier day, the
  same time on the latest earlier week (the three lags N1 reads, computed as `origin_lags` does),
  or any half-hour of the 14-day window the correction averages. The report gives the three lags
  (b1), the correction window (b2), and their union (b), and counts a case in category (b) only
  when it is not in category (a).
- **Trustworthy band (headline).** Visible events (conserved or one-sided) with a metered receiver
  that moved load of 25% or more of the receiving load, lasting over 24 hours. Experiment 41 found
  Sonnet's recall in this band to be 0.83.
- **Sensitivities.** All visible events of any size and duration; and the trustworthy band without
  the 13 conserved operations that rest on one substation with over half its event readings missing,
  which are one-sided at best. Visible alarms count as events; sentinel and placebo cards do not.
- **Window.** The write-up states how many visible events overlap the scored year, how many are in
  the trustworthy band, and the months they fall in. Sentinel and placebo cards are excluded from
  every count.

### What is reported

- **Share of half-hours, share of the quantile score, and share of twCRPS at P99** for each category
  and for the union, with the ratio of score share to half-hour share, overall and per series.
- **The mean loss inside and outside the flags**, in points of capacity.
- **The upper bound for a perfect detector, under two views of the target.** Under the metered view
  (the forecast is scored against the metered power) the detector can help in (a-after) and (b), so
  the headline bound counts those. Under the normal running arrangement (NRA) view, a forecast in (a)
  is scored against a target NGED does not want, so only (b) counts, scored on the half-hours
  outside every event. The looser ceiling that also counts (a-before) is reported beside the
  headline. Each counted case is given the series' mean unflagged loss, and the bound is the
  resulting fall in the summed loss as a share of the total. A second counterfactual uses the same
  series' mean unflagged loss at the same lead and hour of day, because event half-hours cluster in
  particular hours. The bound ignores that a real detector has to find the events, and that a
  forecast made with the events removed still has an error.
- **Intervals** come from resampling the 12 months by initialisation time with replacement (2,000
  resamples) and recomputing each statistic on each resample. Months are resampled whole because
  events last days and neighbouring series share weather.

### Planned figures

Raw first: the actual power of a Primary series with its events shaded, and the best guess's band
against the actual for each trustworthy event. Then the shares against the half-hour share, the
excess ratio with intervals, the loss by lead inside and outside the flags, a per-series chart, the
perfect-detector bound with intervals, and the sensitivity comparison.

## Size

- **Stores:** nothing durable; a private results table under `data/ml-experiments/00044-*/`.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, for the lag definition; the plan reports both lag
  definitions (b1 and b2) rather than choosing one.
- **Code spanning unknown callers:** no; the scripts read one score table and one label file.

Simple, so one Sonnet reviewer reads the code, one checks the write-up against the printed report,
and one runs the prose review.

## Earlier advice followed

- Experiment 41: treat bands under 25% of the receiving load, under 24 hours, or without a metered
  receiver as don't-care, so the headline uses only the trustworthy band and the rest is a
  sensitivity.
- Experiment 38: the end of a detected interval is where the damage happens, which is why category
  (b) is scored separately from category (a).
