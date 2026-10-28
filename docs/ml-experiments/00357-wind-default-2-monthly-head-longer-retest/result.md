---
id: "00357"
slug: wind-default-2-monthly-head-longer-retest
title: >-
  Does retraining Wind farms research default 2's mixture head every month still beat default 2 on
  many more runs?
theme: generator-forecasting
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: change
date: 2026-10-28
size: >-
  medium. Stored: monthly heads' quantiles under data/ml-experiments/00357-wind-default-2-monthly-head-longer-retest/
  only. Serving path: not touched (the page says what serving needs). Degradation rule: not
  touched. Designs: one defensible design (u3 is fixed by experiment 356). Callers: this folder's
  scripts, which import experiments 356, 347, and the folders 356 imports. Sonnet code review and
  Sonnet prose review, no Opus.
hypothesis: >-
  Retraining Wind farms research default 2's mixture head on the first day of every month (experiment
  356's arm u3) gains over default 2 at true days 1 to 10, because a recent head tracks the farms'
  drift. Experiment 356 measured -0.038 points on 53 Tuesday runs (interval -0.104 to +0.024) and
  -0.120 on about 16 Tuesday runs of 2026-07 onwards.
prediction: >-
  Written before any score is read. On the 440 kept daily runs of 2025-07 to 2026-09-13, u3 gains
  between 0.03 and 0.10 points over default 2 at true days 1 to 10 and its month-block interval
  excludes zero. The head retrained on stale months (the placebo) gains less than half of u3's gain.
  u3 leaves at least 1.8% of observations above q99 (default 2: about 2.15%). The gain is larger on
  the runs of 2026-07 onwards than on the runs of 2025-07 to 2026-06.
decision_rule: >-
  The five adoption conditions of the research-default page for a changed setting (minimum gain 0),
  one arm, interval at the 99% month-block level. Condition 1 is read on the 53 Tuesday runs of
  `fold_a_tuesdays_v1` (the brief's primary scope) and, beside it, on every kept daily run of the
  scored year (2025-07 to 2026-06, about 360 runs). Condition 2: the one-sided 95% upper bound of
  the harm to the headline is at most 0.02 and to the 0.01 and 0.99 pinball losses at most 0.05,
  against default 2. Condition 3: the replicate is every kept daily run of 2026-07-01 to 2026-09-13
  (about 75 runs, fixed now, scored once), with the same sign; the Tuesday-only replicate of 356 is
  reported beside it. Condition 4: reproduction gate (default 2 within 0.01 of 10.518 and the
  Tuesday u3 within 0.02 of 356's 10.480 on the scored year), five seeds for u3, the stale-head
  placebo, the leakage check, and no audit trigger unresolved. Condition 5: the page says what
  serving needs. Nothing is adopted without the maintainer's decision.
benchmark: >-
  Wind farms research default 2 (experiment 347's y2, 10.518 points at true days 1 to 10 on the
  kept rows of the 53 Tuesday runs). Beside it, descriptive: Wind farms research default 1 (10.752).
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06), kept rows by experiment 210's mask, as
  the primary scope; every kept daily run of 2025-07 to 2026-09-13 (440 runs) as the all-weekdays
  scope; the kept daily runs of 2026-07-01 to 2026-09-13 as the replicate.
metric: >-
  Quantile score over 11 levels in points of capacity, headline scope true days 1 to 10 (lead 24 to
  240 hours), with days 1 to 2, days 3 to 10, and every lead beside; tolerance 0.02 for the headline
  and 0.05 for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00357-wind-default-2-monthly-head-longer-retest/
---

Planned; the write-up replaces this page.

## Plan

- **Arm.** u3 only: default 2's head retrained on the first day of each month on all finished runs
  from 2024-04 (valid times at least 3.5 hours before the month start), early stopping on the last
  2 months, as in experiment [356](../00356-wind-default-2-blend-and-monthly-refit-retest/result.md).
  Five seeds, mean and spread reported.
- **Runs.** 356 scored the head only on Tuesday runs. Here the same heads predict every kept daily
  run of the month, from the stored case table of experiment 356. No fetch is needed.
- **Placebo.** For each month, the head fitted at a random earlier month (drawn from the months at
  least 3 months before it; the first 3 months keep the first head), so the head is stale but
  otherwise the same.
- **Tails.** Share above q99 and 0.99 pinball loss for u3 and default 2, quantile score at days 1 to
  10 with days 1 to 2 and 3 to 10 beside, and harm to the 0.01 and 0.99 pinball losses.
- **Compute.** Five processes, one per seed, on the GPU, about 10 minutes each in 356.
