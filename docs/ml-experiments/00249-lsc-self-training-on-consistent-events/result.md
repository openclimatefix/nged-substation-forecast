---
id: "00249"
slug: lsc-self-training-on-consistent-events
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
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
  odd-month alarm threshold never sees a pseudo-positive), a series-level hold-out, and a second
  round.
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
---

# Experiment 249: does retraining the learned LSC score on its own consistent events find more real load shifts?

**Planned; no result yet.** Version 2 will have no NGED labels. Experiment 195's learned score is
trained on injected transfers only. This experiment retrains it on injected transfers plus real
events that pass a consistency check built from the data alone, then asks whether the retrained
score finds more of the 35 labelled real transfers.

## Pre-registered design

- **Round 1** is experiment 195's re-ranker and logistic regression, retrained here from the same
  stored injected worlds with the same seeds.
- **Pseudo-positive rule** (label-free; evaluated on every real candidate of every source, over all
  months): not in a D8 drop and not on one of the 17 block days; own 6-hour step and the opposite
  step at the best neighbour both at least 3.0 noise units; at least 1 metered opposite
  neighbour; opposite-to-own step ratio between 0.5 and 1.5; opposite step within 2 hours of the
  own step; round 1 score above the 90th percentile of that fold's round 1 scores over all real
  candidates; and a clean return: the estimated end lies 48 to 192 half-hours after the step and
  the own residual falls by at least 70% of the step across the end (mean of the 12 half-hours
  before the end minus mean of the 12 after, against the step measured the same way at the onset).
  The count is recorded in this file before any real transfer is scored.
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
