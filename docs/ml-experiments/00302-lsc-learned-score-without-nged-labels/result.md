---
id: "00302"
slug: lsc-learned-score-without-nged-labels
title: >-
  Does the learned LSC score keep its matches without NGED labels in training and thresholds?
theme: lsc-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private world, score, and alarm tables under
  data/ml-experiments/00302-lsc-learned-score-without-nged-labels/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for a label-free threshold
  (a global alarm rate, or a per-Primary rate), so both are pre-registered arms. Callers: this
  folder's scripts only, which copy the modules of experiments 195 and 227 and read the stored
  scores of experiment 227 and the Primary statistics of experiment 217. No Opus review (credits).
hypothesis: >-
  Experiment 227's carried alarm source (the re-ranker of experiment 195 at a budget of 20
  unexplained odd-month candidates, "LR_b20") uses NGED labels twice: its training negatives drop
  every logged or labelled span, every D8 drop, and the 17 block days, and its threshold counts
  "unexplained" candidates, which means no logged closure within 24 hours. A re-ranker trained on
  every real candidate as a negative, and thresholded by a rule that reads no label, still matches
  about as many of the 35 logged load shifts lasting over 1 day at about the same number of alarms
  per match, with ordinary-day harm inside the limit.
prediction: >-
  Written before any score is read. L3 (label-free negatives and a per-Primary alarm rate predicted
  from the share of misfit days) matches at least 18 of the 35 logged load shifts (L0, the stored
  labelled source, matches 21) at no more than 6.5 alarms per match, and its harm on months it did
  not train on has a 95% interval inside +/-0.05 points. It fails if L3 matches 17 or fewer, or the
  harm interval leaves +/-0.05; the page then says first that the learned score needs labels, and
  reports whether the loss comes from the negatives (L1 against L2) or from the threshold (L2
  against L3).
decision_rule: >-
  Decisive contrast: L3 minus L1 in matched count (paired by load shift, resampling whole Primary
  pairs) and in harm (paired by whole calendar month). `supported` when L3 is within 3 matches of L1
  and the prediction holds; `partly supported` when L3 meets the prediction but is 4 or more
  matches below L1, or when L2 meets it and L3 does not; `not supported` otherwise. The
  contributions are read as L2 minus L1 (negatives) and L3 minus L2 (threshold). The limit of 0.05
  points and the 18-match bar come from experiment 227; they stay because a label-free score that
  matches fewer or harms more cannot replace LR_b20. Three contrasts are planned, so intervals are
  quoted at 95% and the Bonferroni level (98.3%). The primary stratum is the long-and-deep load
  shifts (at least 2 days and at least 10% of load) with the whole 35 as the second row.
benchmark: >-
  L0: the stored re-ranker of experiment 195 at a budget of 20 unexplained odd-month candidates
  (experiment 227: 21 of 35 matched at 5.0 alarms per match, harm -0.007 with interval -0.039 to
  +0.014 points). Reproducing 21 matches is the positive control.
row_set_id: >-
  Experiment 227's real-record rows (ordinary rows, cached daily 09:00 UTC forecasts, six Primaries)
  restricted to the months each arm did not train on; not comparable with the leaderboard
metric: >-
  Logged load shifts matched out of 35, alarms per matched load shift, MW-weighted recall, onset
  error, alarms per Primary-year, injected-load-shift recall on held-out Primary pairs, and
  ordinary-day harm (arm minus no-mask K0 mean quantile score, in points of effective capacity).
code: ml-experiments/00302-lsc-learned-score-without-nged-labels/
---

# Experiment 302: does the learned LSC score keep its matches without NGED labels in training and thresholds?

*Planned. The write-up replaces this page after the fits finish.*

## Plan

**Question.** If the re-ranker of experiment 195 is trained with label-free negatives and
thresholded by a label-free alarm-rate rule, does it still match about as many of the 35 logged load
shifts lasting over 1 day at about 5 alarms per match, with ordinary-day harm inside the limit?
Flexpectation version 2 will have no operator labels, so neither label use of LR_b20 can run there.
[Experiment 249](../00249-lsc-self-training-on-consistent-events/result.md) added label-free
positives and found nothing to add. Nobody has yet tested whether the score survives losing the
labels it already uses.

**Arms.** All arms use the re-ranker architecture, features, and injected positives of [experiment
195](../00195-switching-neural-plus-detector-features/result.md), and the three folds of Primary
pairs it holds out.

- **L0 (reference, labelled).** The stored re-ranker of experiment 195, trained on even months,
  with a threshold that leaves 20 unexplained odd-month candidates (LR_b20 of [experiment
  227](../00227-lsc-fewer-false-alarms-and-softer-use-part-2/result.md)).
- **L1 (reference, labelled, months swapped).** The same recipe with odd and even months exchanged:
  injected load shifts and real negatives in odd months, a threshold that leaves 20 unexplained
  even-month candidates. L1 is an out-of-sample-in-time reference for L2 and L3.
- **L2 (label-free negatives, global rate).** Trained on odd months like L1, with every real
  candidate a negative unless it lies inside an injected span. The logged or labelled spans, the D8
  drops, and the 17 block days stay in the pool. The threshold leaves as many flagged candidates
  over the whole record as L0 does in the same fold.
- **L3 (label-free negatives, per-Primary rate).** L2's model with one threshold per Primary. The
  alarm count of Primary p is the fold's L0 total times p's share of predicted rates. The predicted
  log2 alarm rate is a straight-line fit on log2(share of misfit days + 0.01), where the share of
  misfit days is the statistic of [experiment 217](../00217-switching-label-free-thresholds-in-noise-units/result.md)
  and the fitted rate is RO-NC's alarm rate in that experiment. The line for Primary p is fitted
  on the other 14 Primaries. The rule is fixed here, before any match is counted.
- **L3 at 5 alarms per Primary-year (label-free throughout).** L3's rule with the total fixed at 5
  merged alarms per Primary-year, so the pin to L0 is not used.
- **L4 (unmasked demand model).** Skipped. Experiment 212's unmasked fits (arm A) cover 5 of the 15
  Primaries, and a 15-series residual panel for them is not stored, so L4 would need a new
  demand-model fit. The page says that the demand model behind every arm was trained with the
  labelled load shifts masked, so one label use remains.

**Score.** Final check only; no model, threshold, or rule is chosen on the 35 logged load shifts.

- **Matches and alarms.** Matches of the 35 (experiment 195's rule: start within 2 hours, end within
  the larger of 6 hours and 10% of the duration) with each load shift scored by the fold that
  held out its Primary pair. Alarms are counted as merged alarms (candidates on one series within
  24 hours merge), summed over the 15 Primaries and the whole record, as the mean over the three
  folds. Also reported: MW-weighted recall, median onset error, alarms per Primary-year by Primary
  (D8 and D9 shown separately), and recall on injected load shifts of held-out pairs (swapped months).
- **Strata.** Long-and-deep load shifts (at least 2 days and at least 10% of load) are the primary
  stratum, since they carry about 88% of the curtailment benefit (experiment 296). The neighbour-
  confirmed variant keeps only alarms with at least one opposite step at a metered neighbour.
- **Harm.** Experiment 227's harness (cached daily runs, six Primaries, experiment 171's ordinary
  rows) on the months an arm did not train on: odd months for L0, even months for the others, with
  95% intervals that resample whole calendar months. Each Primary's harm averages the three folds'
  re-rankers.
- **Label contamination count.** For L2, the number of real training negatives (unique series and
  stamp, odd months) inside logged or labelled spans, D8 drops, and block days.

**Leak and bias traps.**

1. No arm, threshold, or rule is chosen on the 35 logged load shifts.
2. L3's rule for Primary p is fitted without p.
3. Pinning L2 and L3 to L0's alarm count uses one labelled budget as a matched-rate comparison. The
   5-alarms-per-Primary-year point uses none.
4. Unexplained-alarm counts need labels. They are reported for reading only.
5. Harm is compared only on months an arm did not train on.
6. The held-out Primary pairs of a fold can share a Primary with another fold's training pairs
   (as in experiment 195).

## Checks (to fill in after the run)

## Next steps (to fill in after the run)
