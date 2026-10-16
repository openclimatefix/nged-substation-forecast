---
id: "00044"
slug: sw2-2-switching-share-of-best-guess-score
status: done
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
verdict: refuted
best_arm: >-
  Not applicable (a re-scoring of one forecast). In the trustworthy band, 1.35% of half-hours (95%
  interval 0.04% to 3.22%) carry 3.4% of the quantile score (0.04% to 8.68%), and a perfect
  detector would lower the quantile score by 0.42% (-0.47% to 1.96%).
lessons: >-
  In the year scored, switching in the trustworthy band touches 3 of the 16 Primary series and
  few half-hours, and a perfect detector would lower the quantile score by under 2%. The tail
  score is not lowered at all in this band. The all-visible sensitivity shows a much larger tail
  share (55%) that rests on a few short events and has an interval from 2% to 70%. The stored
  `moved_share_of_receiver` is NaN for 62 of 81 events, partly because the labeller wrote some
  series ids as strings.
advice:
  - >-
    Do not build a switching detector to lower the Primary forecast error alone: in the trustworthy
    band a perfect detector would lower the quantile score by 0.42% (-0.47% to 1.96%; figure 4).
  - >-
    Judge a detector's value for the normal-running-arrangement forecast by a different test, because
    this experiment scores against the metered power, and lagged power inside an event is the only
    part a detector can help (the NRA bound, figure 4).
  - >-
    Before scoring anything against the 00041 labelled set, recompute `moved_share_of_receiver`:
    62 of 81 stored values are NaN, partly because series ids were written as strings (caveats
    section).
  - >-
    Treat the all-visible tail-score share (55%, 2% to 70%) as a lead for the first detector
    experiment, since a few short events hold much of the tail loss, and test it on more than 22
    events (figures 2 and 4).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00044-sw2-2-switching-share-of-best-guess-score/
---

# Experiment 44: how much do visible switching events hurt the best guess?

We asked how much of a demand forecast's error falls on the half-hours that engineers' records show
were switching events, when load was moved from one substation to its neighbour. We took the
forecast that experiment 33 found best for the 16 primary substations, which covers July 2025 to
June 2026, and split its errors three ways: half-hours inside a labelled switching event,
half-hours outside any event whose recent power readings (the ones the forecast reads as inputs)
fall inside one, and everything else. In the best-defined group of events, the larger transfers
that lasted over a day, these half-hours were 1.35% of the total and carried 3.4% of the error. We
estimate that a perfect detector would lower the forecast error by under 2%.

**Refuted: in the trustworthy band, a perfect detector would lower the quantile score by 0.42%
(95% interval -0.47% to 1.96%) and the tail score by no measurable amount, under the 3% threshold
of the plan.** The *quantile score* is the average pinball loss over nine quantiles, in points of
capacity. The *tail score* is the threshold-weighted continuous ranked probability score (twCRPS) at
each series' 99th percentile of power. The *trustworthy band* is the visible events that moved at
least 25% of the receiving substation's load and lasted over 24 hours, the band where experiment
41's labeller found 0.83 of injected transfers. The *perfect-detector bound* gives each flagged
half-hour its series' mean score outside every flag, so it is an upper bound.

| Trustworthy band | Share of half-hours | Share of quantile score | Mean quantile score |
|---|---|---|---|
| Target in an event, onset after the forecast was made | 0.12% (0.00 to 0.28) | 0.78% (0.00 to 2.04) | 35.8 (17.7 to 53.3) |
| Target in an event, onset before the forecast was made | 0.03% (0.00 to 0.07) | 0.15% (0.00 to 0.37) | 25.9 (16.1 to 37.6) |
| Target outside events, lagged power inside one | 1.20% (0.02 to 2.87) | 2.49% (0.02 to 6.26) | 11.0 (3.8 to 13.5) |
| Any of the three | 1.35% (0.04 to 3.22) | 3.43% (0.04 to 8.68) | 13.5 (4.4 to 17.3) |
| Outside every flag | 98.65% | 96.57% | 5.21 (4.76 to 5.63) |

The prediction held for the shares and failed for the size. Flagged cases are under 3% of
half-hours (1.35%), and the score share is 2.5 times the half-hour share (the 1.5 times the plan
asked for), but the flagged share did not reach 5% to 15% and the bound did not reach 1% to 5%.
Intervals resample whole months of forecast origin and are wide, because only 6 of the 22
trustworthy events fall in the scored year, on 3 series (10, 12, and 13), with onsets in July,
September, and October 2025 and June 2026. The 13 other series have no trustworthy event in the
year, so the result says little about them.

![Power and the best guess around each trustworthy event](figure_1_raw_events.svg)

![Shares of half-hours and of each score](figure_2_shares.svg)

![Perfect-detector bound, by view of the target](figure_4_bound.svg)

## What a detector could and could not help

**Only the half-hours with lagged power inside an event, and events that started before the
forecast was made, are open to a detector.** The plan's headline bound counts those two groups
(the metered view), and the normal running arrangement (NRA) view counts lagged power only, scored
outside every event. The three bounds for the quantile score, with the series-mean counterfactual,
are 0.42% (-0.47% to 1.96%) for the metered view, 0.34% (-0.51% to 1.79%) for the NRA view, and
0.96% (-0.30% to 3.42%) for the ceiling that also counts events no detector can anticipate. The
counterfactual that also matches the lead and the hour of day gives the same figures to two
decimals. For the tail score every bound is slightly negative (-0.85% metered), because the
flagged half-hours score below average on the tail.

**The all-visible sensitivity, with 22 events of any size or length, gives a larger and far less
certain figure.** Flagged half-hours are 7.4% of the total (3.8% to 11.3%) and carry 11.6% of the
quantile score (4.8% to 18.6%), and the bounds are 2.35% (-0.22% to 5.73%) metered and 4.11% (0.16%
to 8.34%) for the ceiling. Their tail-score share is 55% (2% to 70%) and the metered tail bound is
32.7% (-6.0% to 61.1%), which rests on a few short events.
Removing the 13 operations that rest on one substation with over half its readings missing
leaves 2 trustworthy events in the year, and every bound's point estimate is then between -0.3% and -0.1%.

![Mean quantile score, flagged against unflagged](figure_3_mean_score.svg)

## Caveats

- **Few events.** The verdict follows the pre-registered rule, but 6 events on 3 series cannot
  support a statement about the other 13 series, and the month-resampling interval understates
  that uncertainty.
- **Stored shares are NaN.** The labelled set's `moved_share_of_receiver` is NaN for 62 of the 81
  visible events that touch a Primary series (82 in all). The causes are that the labeller wrote
  some donor or receiver ids as strings and the share code dropped them, and that 20 events name no
  metered receiver. The share is recomputed here from the moved MW and the named receivers' mean
  absolute power in the 28 days before onset. It agrees with the stored value where both exist (for
  example 0.33 against 0.34), but it ignores the split of load between receivers and so understates
  the share when a receiver takes only part of it. The stored values give a share for 19 of the 81
  events, and the recomputed share exists for 61, of which 22 are 0.25 or more with a duration over
  24 hours.
- **Detectability.** An event counts as known to a detector once the origin passes its onset, but
  telemetry arrives 3 hours late, so events starting under 3.5 hours before the origin are
  flattered slightly. The category (b) split gives nested figures, because the lags lie inside the
  correction window, so the two lag definitions are the same set to the second decimal (1.20%).
- **Only Primary series are scored.** The labels cover the 16 Primary series, so BSP, GSP, solar,
  and wind have no figure.
- **Predictions not in the store.** Experiment 33 kept its forecasts in its own work folder, so
  this experiment reads its per-case scores from there.

![Share of half-hours and of the quantile score, per series](figure_5_series.svg)

![Mean quantile score by lead time](figure_6_lead.svg)

## Lessons for later experiments

- A switching detector is worth building for the normal running arrangement forecast or for an
  event table, and it is not worth building for the forecast error alone.
- A first detector experiment should look at the few short events that hold much of the all-visible
  tail loss before it looks at the whole year.
- The labelled set's receiver shares need recomputing, as the advice says.

Numbers come from `e00044_score.py` and `e00044_report.py`; figures from `e00044_charts.py`.
Pull request: [PR #1261 (Experiment 44: how much do visible switching events hurt the best guess?
(SW2-2 plan))](https://github.com/openclimatefix/nged-substation-forecast/pull/1261).
