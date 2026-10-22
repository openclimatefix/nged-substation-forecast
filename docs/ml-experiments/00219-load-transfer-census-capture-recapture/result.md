---
id: "00219"
slug: load-transfer-census-capture-recapture
title: >-
  How many large load transfers exist, estimated without labels by capture-recapture between
  detectors?
theme: switching-detectors
also_relevant_to: []
status: done
size: >-
  medium. Stored: private tables under
  data/ml-experiments/00219-load-transfer-census-capture-recapture/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  (the matching rule and the size cut-off are fixed in advance and varied as sensitivities).
  Callers: this folder's scripts only, which copy experiment 148's interval and end-rule code and
  read
  stored alarm tables of experiments 148, 156, 195 and 217.
hypothesis: >-
  Three detectors that read the same demand-model residual find overlapping but not identical sets
  of
  large load transfers. Capture-recapture on those sets, stratified by event size, estimates the
  number of large transfers and each detector's recall without any switching log.
prediction: >-
  The label-free estimate of RO-NC's recall is within 25% (relative) of its recall on the 35 logged
  transfers (0.51 by count, 0.657 weighted by MW moved), and the injected-event arm recovers the
  known
  injected count within 25% in at least two of the three size strata. It fails if the dependence
  between detectors (all read the same residual) makes the estimate far off.
decision_rule: >-
  Verdict `supported` when both parts hold, `partly supported` when exactly one holds, `refuted`
  when
  neither holds. The 35 logged transfers are read only after the estimates are frozen, and no
  threshold is tuned on them. Nothing is ported, so no effect-size threshold applies. The page names
  the strata in which the estimate is unstable and reports bounds for them.
benchmark: >-
  Recall of RO-NC, the pair alarm and the learned score on the 35 logged transfers, and the known
  injected count in experiment 195's held-out injected worlds.
row_set_id: none (event counts, not forecasts)
metric: >-
  Estimated number of large events, events per Primary-year, and each detector's estimated recall,
  with 95% intervals that resample whole calendar months.
verdict: refuted
best_arm: >-
  None. At size 40% of a Primary's 99th percentile of demand or more, the three detectors find 10
  events, an estimated 12.7 in all (95% interval 9.8 to 21; 0.2 per Primary-year). RO-NC's recall
  there is 0.79 (interval 0.40 to 0.99).
headline: >-
  Over all large events the three detectors share too few events for capture-recapture to work. The
  three-list estimate is 1,650 to 2,590 events (27 to 42 per Primary-year) against 300 seen, and
  RO-NC's
  recall comes out at 0.05 to 0.09 against 0.51 measured. On injected events the estimate
  undercounts
  by 23% to 59%.
lessons: >-
  Capture-recapture needs detectors that look at the same events independently. These three do not.
  The pair alarm looks at smaller, paired events than RO-NC, and the learned score re-ranks RO-NC's
  and the pair alarm's own candidates, so it is nearly nested in them.
take_home_message: >-
  We cannot count large load transfers, or measure how many each detector misses, by comparing the
  detectors with each other. Across all large events, the three detectors disagree too much for the
  estimate to mean anything, and on injected events with a known count it undercounts by 23% to 59%.
  Only for the biggest events (40% of a substation's 99th percentile of demand or more) do the
  detectors agree, and there the estimate is about 0.2 events per Primary-year from 10 events seen.
advice:
  - >-
    Do not estimate the number of large transfers, or a detector's recall, by capture-recapture
    between RO-NC, the pair alarm, and the learned score; the estimate failed the known-answer test
    (figure 3) and disagrees with the measured recall (figure 4).
  - >-
    Count large transfers by size band and quote only the 40%-of-p99-or-more band, where the
    detectors agree; the 10 events seen there give 0.2 per Primary-year (figure 2).
  - >-
    For a label-free recall estimate in version 2, use detectors that read different data, such as a
    meter-based list of switching actions, or a hand-checked random sample of alarms; the three
    residual-based detectors are too dependent (figure 1).
production_status: not-for-production
adopted_in: none
---

**Take-home: comparing the three detectors with each other cannot tell us how many large load
transfers exist or how many each detector misses.** The three detectors share too few events, and
the estimate fails the known-answer test. We asked whether the number of large load transfers (a
block of demand moved from one Primary substation to another for 3 hours or more) can be estimated
without a switching log. The method is capture-recapture: if two detectors search independently, the
share of one detector's events that the other also finds says how many events both missed. We ran it
on three detectors that read the same demand-model residual, over 15 Primary substations and 61.8
Primary-years. The detectors found about 300 events, 4.9 per Primary-year, but they overlap so
little that the estimate is 1,650 to 2,590 events (27 to 42 per Primary-year) and RO-NC's recall
comes out at 0.05 to 0.09. Experiment 146 measured a recall of 0.51 for RO-NC on the 35 logged
transfers. On injected transfers whose count is known, the same estimator undercounts by 23% to 59%.
The method works only for events of 40% of a Primary's 99th percentile of demand or more, where 10
events were seen. The result narrows the search for a version 2 recall estimate to detectors that
read different data, or to a hand-checked sample of alarms.

**The method is capture-recapture, and the detectors are the three of experiments 148, 156, and
195.** We use the standard Lincoln-Petersen estimate of two lists, the no-three-way-interaction
estimate for three lists, and Chao's lower bound ([Chao, 1987](https://doi.org/10.2307/2531532)).
RO-NC is the refit-offset neighbour-count detector (experiment 148, with its 6-hour-lookahead end
rule and 24-hour merge); the pair alarm raises an alarm when a neighbouring substation steps the
opposite way ([experiment
156](../00156-switching-faster-onset-alarm-from-opposite-steps/result.md)); the learned score
re-ranks their candidate alarms ([experiment
195](../00195-switching-neural-plus-detector-features/result.md), one alarm set per held-out fold).
A large event is an interval of at least 3 hours whose step is at least 10% of either Primary's 99th
percentile of demand. Two intervals are the same event when they share a Primary and overlap by at
least half of the shorter. Size, duration, and "metered partner seen" define the strata. No
switching log is read; the 35 logged transfers are compared at the end only.

## Figures

![Events by the detectors that found them](figure_1_overlaps.svg)

![Events per Primary-year, seen and estimated](figure_2_events_per_primary_year.svg)

![Estimated against known injected count](figure_3_injected_known_count.svg)

![Label-free recall against recall on the 35 logged transfers](figure_4_recall_against_logged.svg)

## Headline numbers

- **Prediction fails on both parts.** RO-NC's label-free recall is 0.08 (median of three folds; 0.05
  to 0.09), outside 25% of 0.51 (0.38 to 0.64). Even Chao's lower bound on the event count gives
  0.27, which is also outside. The injected arm is within 25% in 1 of 3 size strata (large third:
  0.77 of the known count; middle third 0.71; small third 0.41).
- **Events seen.** RO-NC 141, the pair alarm 136, and the learned score 125 to 144 large events;
  297 to 308 distinct events in all (4.8 to 5.0 per Primary-year). RO-NC and the pair alarm share 27
  events, RO-NC and the learned score 64 to 72, and the pair alarm and the learned score 31 to 34
  (figure 1).
- **Estimates over all large events (three learned-score folds).** Lincoln-Petersen: 710 (RO-NC and
  pair alarm), 256 to 282 (RO-NC and learned), and 516 to 576 (pair alarm and learned). Three-list:
  1,650 to 2,590 events (26.7 to 41.9 per Primary-year; interval of the median fold 777 to 7,235
  events, from resampling whole calendar months). Chao lower bound: 506 to 556 (8.2 to 9.0 per
  Primary-year).
- **By size.** 10-20% of p99: 196 seen, 696 estimated (310 to 3,060), Chao 324. 20-40%: 96 seen,
  1,600
  estimated (134 to 5,690), Chao 187. 40% or more: 10 seen, 12.7 estimated (9.8 to 21), 0.2 per
  Primary-year.
- **By duration and partner.** Under 3 days: 171 seen, three-list estimate 987. 3 to 14 days: 119
  seen, 738. 14 days or more: 12 seen, 16 (Chao). Events with a metered partner seen (the pair alarm
  supplies every partner): three-list undefined, Chao 334. Events without one: 214 seen, 935.
  Unstable strata: every stratum except size 40% or more and duration 14 days or more.
- **Sensitivity.** The 5% and 20% size cut-offs and the unmerged end rule change the three-list
  estimate to between 17 and 57
  events per Primary-year and never bring RO-NC's recall above 0.12 (median over folds).
- **Injected arm (experiment 195's held-out injected transfers of 24 hours or longer, 419 events).**
  Seen 261 (62%), estimated 263 (interval 236 to 291), known 419. True recall: RO-NC 0.575, learned
  score 0.592, pair alarm 0.215. Estimated recall: RO-NC 0.92, learned score 0.94, pair alarm 0.34.
  By size third (known 140, 139, 140): estimates 57, 99, 108, which is 41%, 71%, and 77% of the
  known count.
- **Recall by size (label-free).** At 40% of p99 or more: RO-NC 0.79 (0.40 to 0.99), the pair alarm
  0.32, the learned score 0.47, against 0.51, 0.14, and 0.64 on the 35 logged transfers.

## Why the estimate fails

- **The detectors do not search independently.** On the injected events the learned score finds
  nearly the same events as RO-NC (248 and 241 of 419, with 233 found by both, from the stored
  flags), because it re-ranks RO-NC's candidates. The union of detectors is then barely larger than
  one detector, so the estimator sees almost no unseen events and gives 263 of 419.
- **On real data the opposite fault shows.** The pair alarm's median step is 0.36 MW against 0.91 MW
  for RO-NC, so each detector's events are mostly its own. A few shared events and many single-list
  events inflate the three-list estimate. Alarms that are false positives add to single-list events
  too, and the method counts every alarm as an event.
- **The estimate is stable only where every detector can see the event.** At 40% of p99 or more all
  three detectors find 4 of 10 events and RO-NC finds all 10.

## Checks

- Median minus mean forecast, and mean-forecast error beside quantile-score contrasts: not
  applicable, because the experiment forecasts nothing.
- **Triggers fired:** trigger 1 (a known-answer test fails: the injected arm misses its 25% band in
  2
  of 3 strata) and trigger 2 (RO-NC's label-free recall of 0.08 contradicts the 0.51 measured in
  experiment 146; the explanation above is tested by the injected arm and by figure 1). Audit:
  requested from the manager; findings and triage will be added here.
- No data-processing finding to add to the data-processing page.

## Caveats

- The detectors' alarm sets were tuned on switching labels (the 80 unexplained odd-month alarms
  operating point of experiment 195, and the label-fitted learned score). The census reads only
  detector outputs, but the detectors are not label-free.
- The injected arm's found flags come from matching to the known event, so the arm tests the
  estimator and not the blind matching rule. Real-data matching is blind.
- The 35 logged transfers are transfers over 1 day of any size, and the large events are defined by
  step size. The two sets differ, so the recall comparison is approximate.
- The 10% of p99 threshold, 50% overlap, and the strata edges were fixed before the run; the 5% and
  20% cut-offs are sensitivities.

## What the next experiment should answer

- Can a hand-checked random sample of alarms (precision) plus a meter-based list (recall) give the
  count that the detectors cannot?
- How many events of 40% of p99 or more, 0.2 per Primary-year, does the logged record of the
  trial area contain?
