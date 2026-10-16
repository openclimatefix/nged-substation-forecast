---
id: "00058"
slug: sw2-8b-donor-step-recut-and-stronger-veto
status: planned
size: simple
hypothesis: >-
  (1) The gap in recall between injected load transfers with a metered receiver (0.50) and without
  one (0.17), seen in experiment 55, comes mostly from the injection giving the second group a
  smaller donor step, so the gap shrinks to under half its size when the two groups are compared at
  the same donor step in noise units. (2) A neighbour veto stronger than experiment 55's strongest
  (1.5 noise units), at 1.25 or 1.0, lets the 24-hour threshold fall further and finds more
  injected transfers than no veto.
prediction: >-
  (1) Experiment 55 reports a median donor step of 16% of the donor's load without a metered
  receiver and 55% with one. Within bins of equal donor step in noise units, the recall gap falls
  from 0.33 (0.50 minus 0.17) to 0.05 to 0.15. (2) The odd-month gain over no veto kept rising as
  the strength fell (0.000 at 3.0, +0.013 at 2.0, +0.030 at 1.5), so odd months pick 1.25 or 1.0.
  On even months the 24-hour gain over no veto (experiment 55's C1, +0.008) rises to +0.01 to
  +0.03, and the matched 24-hour threshold falls to 3.4 to 3.6 from 3.70.
decision_rule: >-
  Three planned contrasts, read at 98.33% (Bonferroni across C1 to C3) from 20,000 resamples of
  whole calendar months, over experiment 55's 620 headline-band transfers that start in even months
  (24-hour window). C1 is the recall gap, metered minus unmetered receiver, within matched bins of
  the largest step in noise units over the donor and the metered receivers (a transfer counts as
  found when an alarm on any of those series matches it, so a transfer with a metered receiver has
  more chances to alarm at one donor step) (four bins of equal count over the range both groups share, with the
  bin gaps averaged with the unmetered count as weight). The raw gap is 0.33. The step size
  explains at least half the gap when the upper end of the C1 interval lies below 0.165, and
  explains less than half when the lower end lies above 0.165; otherwise the result is mixed.
  C2 is D2 at the veto strength chosen on odd months among 1.0, 1.25, and 1.5, minus D0M, in the
  share of transfers found, paired by transfer, exactly as experiment 55's C1. `supported` when the
  C2 interval lies wholly above zero, the C2 point value is at least +0.01, and D2's even-month
  unexplained-alarm count is no more than 10% above D0M's; `refuted` when the upper end lies below
  +0.01; `inconclusive` otherwise. The threshold of +0.01 is the one experiment 55 used: the veto
  adds a pipeline step, and a smaller gain would not pay for it. C3 is the same contrast on the
  transfers with no metered receiver, reported without changing the verdict. The scoring design
  for SW2-7 follows from C1, as the write-up states.
benchmark: >-
  D0M, which is D0 at experiment 43's thresholds (the matching search returned them in experiment
  55), and experiment 55's D2 at 1.5 noise units, re-run here through the same code so that the
  comparison uses one set of draws.
row_set_id: >-
  none (this experiment scores detector alarms against injected transfers, as experiments 43 and 55
  do, not forecasts)
metric: >-
  The share of injected transfers found (start within 2 hours and end within the larger of 6 hours
  and 10% of the duration), by donor step in noise units and by metered receiver; the even-month
  unexplained-alarm count.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00058-sw2-8b-donor-step-recut-and-stronger-veto/
---

# Experiment 58: is the recall gap without a metered receiver a step-size effect, and does a stronger veto help?

**Planned.** This page is the plan; the write-up replaces it when the run finishes.

## Plan

- **Code.** The scripts are copies of experiment 55's, with the experiment id changed in each
  filename, and are otherwise unchanged. The harness check (experiment 43's alarm counts and the
  outcome of each of its 600 transfers) runs first and stops the run if it fails.
- **Step in noise units.** Two versions are written: the donor's, and the largest over the donor and
  the metered receivers (the pre-registered C1 uses the second; the first is a reported check).
  The donor version: For each injected transfer, the injected donor load over the
  first 24 hours after the onset (the mean of the injection's `delta` on the donor), divided by the
  noise scale of the 24-hour level-shift statistic for the donor series (the `scale` that
  `prepare` returns from odd months). The injection is re-drawn from experiment 55's seeds, so the
  steps belong to the same transfers experiment 55 scored.
- **Part 1 (re-cut).** Recall of D0M and of D2 at 1.5 noise units by donor-step bin, split by
  metered receiver, on the 620 headline-band transfers that start in even months. A figure shows
  recall against donor step for both groups, with the overlap of the two step distributions.
- **Part 2 (stronger veto).** Tune the matched 24-hour and 4-hour thresholds for veto strengths 1.0
  and 1.25 on odd months exactly as experiment 55 did; choose the strength among 1.0, 1.25, and 1.5
  on odd-month transfers; score the chosen strength on even months.
- **Checks and disclosures.** The step script stops if the re-drawn injection does not cover the
  scored events. The bin edges are fixed from the pooled transfers, not redrawn inside each
  resample, so the C1 interval is slightly narrow. The per-bin counts are printed. Noise units are
  odd-month noise units. The 620 even-month transfers were seen in experiment 55, which is where
  the 0.17 and 0.50 came from, so C1's design is informed by those data. A veto at 1.0 or 1.25
  falls below the 2.0 noise units that count a neighbour as a partner, so the write-up reports the
  transfers C2 loses and gains.
- **Compute.** CPU only, slot C, 4 workers.

## Sizing

- **What gets stored:** private parquet tables of scored transfers; nothing the production code reads.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **More than one design:** the re-cut could use bins or a regression of found on step; bins are
  pre-registered, and a logistic regression of found on step and metered receiver is reported as a
  check.
- **Callers:** the code is experiment 55's, whose callers are its own scripts.

The size is simple: both parts re-use existing code and data with settings already in the pipeline.

## What the next experiment follows from

The advice items of experiment 55 that this plan follows: re-cut recall by donor step before
building a new step statistic, and test a stronger veto. No item is departed from.
