---
id: "00249"
slug: lsc-self-training-on-consistent-events
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private tables, weights and alarm tables under
  data/ml-experiments/00249-lsc-self-training-on-consistent-events/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for the
  consistency rule, the pseudo-label weight, and the hard negatives, so all three are fixed below
  before any real transfer is scored. Callers: this folder's scripts only; they copy the modules of
  experiments 195 and 203 and read experiment 195's stored worlds and candidate table.
hypothesis: >-
  The learned load-shift score of experiment 195 is trained only on injected transfers. If it is
  retrained on injected transfers plus the real events that pass a label-free consistency rule
  (balanced opposite steps at a metered neighbour, a high round 1 score, and a clean return step),
  it finds more of the 35 real transfers lasting over 1 day than the round 1 score, at the same 80
  unexplained odd-month alarms, because the pseudo-positives show it what real events look like.
prediction: >-
  Written before any real transfer is scored. Round 2 (pseudo-positive weight 0.25) finds at least 3
  more of the 35 held-out transfers than round 1 (experiment 195's re-ranker: 24 of 35) at 80
  unexplained odd-month alarms, with a 95% interval from resampling whole Primary pairs whose lower
  end is above zero, and the no-harm interval of the gated mask on ordinary days is no worse than
  round 1's. The prediction fails if round 2 only re-finds what the pair alarm already finds
  (confirmation bias), which the held-out 35 show; the page says so first.
decision_rule: >-
  Primary contrast: found count of R2 (re-ranker, weight 0.25, headline) minus R1 (re-ranker, round
  1), over the 35 transfers lasting over 1 day, with a 95% interval from resampling whole Primary
  pairs (20,000 draws). `supported` when the difference is at least 3, the lower end of the
  interval is above zero, and the no-harm contrast (gated mask minus no mask on ordinary days,
  experiment 171's rows, rows resampled by calendar month) has a point estimate no more than 0.01
  points above round 1's and an upper limit no more than 0.02 points above round 1's. `partly
  supported` when the difference is at least 3 and either of the other two conditions fails, or
  when the difference is 1 to 2 with a lower end above zero. `refuted` when the difference is 0 or
  below. `inconclusive` otherwise or when the downstream harness cannot run. The 3 events are the
  brief's bar; round 2 adds a retraining loop to the R&D pipeline only (no serving-path change), so
  no larger porting threshold is set. Reported beside it, not bars: the same contrast for the
  logistic regression (F), the second weight (0.5), pseudo-positives from even months only (so the
  odd-month alarm threshold never sees a pseudo-positive), a series-level hold-out, the strict
  tier, a post hoc heavy weight (pseudo-positives holding 25% of the injected positives' weight),
  and a third round.
benchmark: >-
  Round 1 re-ranker (experiment 195: 24 of 35, MW-weighted recall 0.752), round 1 logistic
  regression (22 of 35), RO-NC (19 of 35), and the pair alarm (5 of 35), all at 80 unexplained
  odd-month alarms; the gated mask of experiments 171 and 203.
row_set_id: >-
  none (scores alarms against the 35 real transfers of experiment 41 and freshly injected
  transfers; the downstream test uses experiment 171's rows)
metric: >-
  Real transfers found out of 35 (start within 2 hours, end within the larger of 6 hours and 10% of
  the duration) at 80 unexplained odd-month alarms, MW-weighted recall, recall on injected
  transfers, and the masked-forecast score on ordinary days.
code: ml-experiments/00249-lsc-self-training-on-consistent-events/
verdict: refuted
best_arm: >-
  None. Round 2 with pseudo-positive weight 0.25 finds 24 of the 35 real transfers over 1 day, the
  same 24 as round 1 (difference 0, 95% interval 0 to 0, resampling whole Primary pairs).
headline: >-
  Retraining the learned score on 47, 38, and 45 real events per fold (56 distinct events in all)
  that pass a label-free consistency rule changed no alarm that matters: round 2 finds 24 of 35
  transfers, the same as round 1, and the arms that changed the score lost 0 to 2.
lessons: >-
  The real record holds too few events that pass a consistency rule to move a score trained on
  thousands of injected transfers. Giving the pseudo-positives a quarter of the injected positives'
  weight (post hoc) moved the logistic regression by one transfer (+1, interval 0 to +3) and the
  re-ranker by none. The strict balanced-pair tier passes only 14 to 16 candidates per fold.
take_home_message: >-
  Retraining the learned load-shift score on its own consistent real events found no extra real
  transfers: it still finds 24 of 35, because only about 50 real events per fold pass the
  consistency rule, against thousands of injected transfers. Self-training needs many more
  consistent events than 4.5 years of 16 Primary substations provide, so the next gains must come
  from better injected transfers or from more data, not from relabelling the existing record.
advice:
  - >-
    Avoid self-training the learned load-shift score on this record's consistent events: about 50
    real events per fold pass the rule and round 2 found no extra transfer (figure 1).
  - >-
    Check how many events pass a pseudo-label rule before building a training loop on it: the first
    draft of the balanced-pair rule passed 2 candidates per fold (design bullets).
  - >-
    Consider giving pseudo-positives a large share of the weight only on a record with several
    hundred consistent events; a quarter of the injected weight moved the logistic regression by
    one transfer (heavy arm).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
---

# Experiment 249: retraining the learned load-shift score on its own consistent events finds no more real transfers

**Retraining the learned score on real events that pass a label-free consistency check did not find
any extra real transfer.** Round 2 finds 24 of the 35 real load-shift transfers lasting over 1 day,
the same 24 as round 1 (difference 0, 95% interval 0 to 0, resampling whole Primary pairs). The
prediction was at least 3 more, so the hypothesis is refuted. The cause is a shortage of examples:
only 47, 38, and 45 real events per fold (56 distinct events in all) pass the rule, against
thousands of injected transfers, so the extra rows could not move the score. A heavier weight, a
stricter rule, even-month events only, and a third round changed the count by 0 to 2 transfers
either way.

**What we asked.** Version 2 of the forecast will have no NGED labels, so the learned score of
[experiment 195](../00195-switching-neural-plus-detector-features/result.md) must be trained on
injected transfers alone. We asked whether the score finds more real transfers after it is retrained
on its own most consistent real events. Such self-training follows the usual pseudo-label recipe.
We fixed the consistency rule before scoring any transfer. A real event counts as consistent when
the forecast error steps by at least 2.5 noise units over 24 hours, a neighbouring Primary
substation steps the opposite way, the logistic regression scores it in the top quarter, and the
error returns to its old level at the estimated end. The remaining design, including the hold-out of
each Primary pair's own transfers, is below.

![Re-ranker after self-training minus before, in transfers found](figure_1_found_minus_round_1.svg)

![The 56 real events that pass the rule, by size and month](figure_2_pseudo_positive_mix.svg)

## Results

- **Headline (re-ranker, 35 transfers over 1 day, 80 unexplained odd-month alarms).** Round 1
  finds 24 (MW-weighted recall 0.752, 95% interval 0.585 to 0.865). Round 2 at weight 0.25 finds 24
  (0.752). Round 2 at weight 0.5, with even-month events only, and with the strict tier each find
  22 (−2, interval −4.7 to 0). The series-level hold-out finds 23. The third round finds 24. The
  heavy weight finds 24.
- **Logistic regression.** Round 1 finds 22. Every round 2 and round 3 arm finds 22 except the heavy
  arm, which finds 23 (+1, interval 0 to +3): the one transfer gained is 48 hours long and moves 0.4
  MW, and neither the pair alarm nor RO-NC finds it.
- **Reproduction.** The retrained round 1 re-finds experiment 195's 24 (re-ranker) and 22
  (logistic regression), RO-NC's 19, and the pair alarm's 5, so the setup matches.
- **Confirmation bias.** Round 2 gained no transfer, so it did not re-find what the pair alarm
  finds. Of the 56 distinct pseudo-positives, 12 lie within 2 hours of a labelled transfer's onset
  (counted after scoring, for this page only). The rest are real events nobody logged or labelled
  or are false alarms; the rule cannot tell which.
- **Pseudo-positive mix.** 99 of the 130 fold-level rows step under 1 MW and 31 step 1 to 3 MW;
  none steps over 3 MW. They sit on 13 of the 16 Primary series; series 24 has 23 of the 130 rows.
  Each fold holds 16 to 19 events in odd months, where the alarm threshold is chosen.
- **Injected-world recall.** Rounds 1 and 2 are the same to within 0.01 on both injected sets
  (held-out pairs, over-1-day recall 0.595 for round 1 and 0.588 for round 2 at weight 0.25), so
  the pseudo-labels did not make the score overfit to them.
- **Downstream test not run.** The decision rule marks the experiment refuted when the count
  difference is 0 or below, and the headline arm finds exactly the transfers that round 1 finds, so
  the gated-mask replay of experiment 203 would repeat its result.
- **Tail calibration, curtailment cost, and forecast quantiles** do not apply: the experiment
  scores alarms, not a forecast.

## Checks

- **Median minus mean forecast per site:** not applicable; no forecast is produced.
- **Mean-forecast error beside each contrast:** not applicable, for the same reason.
- **Triggers:** trigger 2 fired (the result contradicts the prediction). The explanation, too few
  consistent events, was tested in the same run with the heavy arm, which gave pseudo-positives a
  quarter of the injected positives' weight and still moved the count by 0 to +1, so no audit was
  run. Trigger 1 did not fire: the reproduction of experiment 195 is the positive control. Triggers
  3 to 7 do not apply or did not fire.

## Caveats and next steps

- **Lab shortcuts:** the logged closures and labelled transfers define the hold-out, the
  explained-alarm flag in the threshold, and the negatives, as in experiment 195.
- **Schedule:** a single fit per arm and fold, three seeds averaged in each network; no weekly
  refits, because no forecast is trained.
- **Rule changes:** the first draft of the strict tier passed 2 candidates per fold, so the
  thresholds were relaxed and the loose rule added before any transfer was scored.
- **Next step.** Gains must come from a richer injector or from a longer record, not from
  relabelling this record. A pooled model across the 16 Primaries or the full 2019 history, if
  disaggregation allows it, would give more consistent events.

## Pre-registered design

- **Round 1** is experiment 195's re-ranker and logistic regression, retrained here from the same
  stored injected worlds with the same seeds.
- **Pseudo-positive rule** (label-free; evaluated on every real candidate of every source, over all
  months). Every pseudo-positive is outside the D8 drops and the 17 block days, has a
  logistic-regression score above the 75th percentile of that fold's scores over all real candidates
  (the re-ranker scores only RO-NC's candidates, so the logistic regression is the gate), and has a
  clean return. A clean return means the estimated end lies 48 to 176 half-hours after the step (so
  that the 12 half-hours after the end lie inside the 4-day window) and the own residual falls across
  the end by at least 50% of the step (mean of the 12 half-hours before the end minus mean of the 12
  after, against the step measured the same way at the onset). The **loose rule** (headline) needs a
  24-hour step of at least 2.5 noise units and an opposite step at any neighbour. The **strict
  tier** needs an own 6-hour step of at least 3.0, an opposite 6-hour step of at least 2.0 at a
  metered neighbour within 4 hours, and an opposite-to-own ratio between 0.3 and 2.0. The first
  draft of the strict tier (opposite step 3.0, ratio 0.5 to 1.5, return 70%, 90th percentile) passed
  2 candidates per fold, so the thresholds were relaxed and the loose rule added **before any real
  transfer was scored**. Counts recorded before scoring: strict 14, 15, and 16 candidates in folds
  0, 1, and 2; loose 49 in every fold, which falls to 47, 38, and 45 after the event hold-out
  (below), of which 19, 18, and 16 lie in odd months. The real record holds 2,980 candidates in all.
- **Hold-out.** A pseudo-positive that lies within 48 half-hours before the onset to 48 half-hours
  after the end of a labelled transfer of fold k, on a series of that transfer, is dropped from fold
  k's round 2 training. The round 1 score that gates fold k's pseudo-positives is fold k's own
  model, trained without fold k's injector pairs. The series-level hold-out (drop every
  pseudo-positive on a series of fold k's pairs) is reported as a robustness check.
- **Hard negatives.** The label-0 candidates of the injected worlds (as in experiment 195), plus
  real candidates that fail the rule and are either on an England bank holiday or isolated (no
  other candidate on any series within 7 days). A candidate is itself an alarm, so "quiet stretch"
  means an isolated alarm. D8 drops and block days are excluded from positives and negatives.
- **Round 2** retrains the same networks with injected positives at weight 1, pseudo-positives at
  weight 0.25 (headline) or 0.5, and the negatives at weight 1. A round 3 is run only if round 2
  finds events that round 1 missed and the pair alarm did not.
- **Scoring.** Operating point at 80 unexplained odd-month alarms, as in experiment 195; the 35
  transfers are scored last. The downstream test copies experiment 203's harness with round 2's
  alarms.

## Size (five triggers)

- **Stored:** private tables and weights only.
- **Serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (rule thresholds, weights, negatives); all fixed above.
- **Callers not nameable without searching:** no; this folder's scripts only.

## Sources

The injector, the held-out-by-pair
scoring, and the learned score are from experiments
[189](../00189-switching-neural-onset-end-block-detector/result.md) and
[195](../00195-switching-neural-plus-detector-features/result.md); the downstream harness is from
experiments [171](../00171-switching-gated-mask-no-harm-pooled-months/result.md) and
[203](../00203-switching-replay-learned-score-in-masked-forecast/result.md).
