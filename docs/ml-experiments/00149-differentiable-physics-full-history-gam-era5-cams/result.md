---
id: "00149"
slug: differentiable-physics-full-history-gam-era5-cams
title: >-
  Differentiable physics: with the solar term explicit, does training the convex model on all
  history back to 2019 beat a 24-month window?
part: n/a
theme: differentiable-physics-long-history-and-drift
also_relevant_to: [retraining-and-training-history, differentiable-physics-embedded-generation]
status: planned
size: >-
  simple. Stored: private files only. Serving path and degradation rules: not touched. Designs: one,
  because it re-runs experiment 107's rolling-origin design with two training windows and two drift
  settings. Callers: this folder's scripts, which copy experiment 107's and call the reviewed
  `studies.dp_*` modules without changing them.
hypothesis: >-
  When the convex model's solar term is explicit (a pinned capacity times a unit solar shape in
  both arms), training on every year of a substation's history back to 2019 gives a lower quantile
  score than training on the last 24 months, once a drift term lets the model follow the slow
  level change. The mechanism is that more years pin down the temperature, calendar, and weekday
  terms better, and the drift term absorbs the level change that made all-years training unstable
  in experiments 104, 106, and 107.
prediction: >-
  Written before any fit. With the drift term in both arms, all history minus 24 months (C2) is
  between -0.30 and -0.05 points pooled, below zero at 8 or more of the 12 origins, and larger in
  the winter-quarter origins than the summer-quarter ones. Without the drift term (C1) the sign
  varies by origin, as in experiment 107. The drift term lowers the all-history score by 0.2 to 0.5
  points (C3), as experiment 107 found, and by less on 24 months (C4), because 24 months has less
  level change to follow. Dropping the substation that moves the result most (D10 or D16, as in
  experiment 107) changes C2 by under 0.15 points.
decision_rule: >-
  The verdict rests on C2 (all history plus drift minus 24 months plus drift), pooled over the
  substation-origin pairs that pass the history rule below, with a month-block bootstrap interval
  at 98.75% (a Bonferroni correction over the four planned contrasts). The threshold is zero,
  because a training start costs nothing to port. `supported`: the interval lies wholly below zero,
  at least 9 of the 12 origins lie below zero outside the tie band (one week-block standard error),
  at most 1 origin lies above zero outside the tie band, and the sign survives leaving out each
  substation in turn. `partly supported`: the interval lies wholly below zero but the origin or
  leave-one-site-out condition fails, or C2 is below zero in winter-quarter origins only. `small
  gain`: not applicable, because the threshold is zero. `refuted`: the interval lies wholly above
  zero. `inconclusive`: otherwise. C1, C3, and C4 are reported with the same rule and never decide
  the verdict. The verdict is capped at `inconclusive` if a blocking check fails.
benchmark: >-
  Arm A24+D1 (24 months, drift term) on the same rows; the 12-month arm of experiment 107 as a
  reference only; climatology and manual_heuristic as stored by experiment 107
row_set_id: >-
  dp_mw_v1 (experiment 103); 12 quarterly origins 2023-07-01 to 2026-04-01, each scored on the
  following 3 months, as experiment 107
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored substation;
  tail coverage and pinball loss at q01, q05, q95, and q99
code: ml-experiments/00149-differentiable-physics-full-history-gam-era5-cams/
production_status: not-for-production
adopted_in: none
---

Plan only; the write-up follows the runs.

## Plan

**Question.** Experiments 104, 106, and 107 gave answers to "does more history help?" that changed
sign with the substation and the year. The convex demand model has since gained an explicit solar
term, and the plan of this experiment asks whether the answer is stable once that term is fixed in
both arms. The aim is a feel for whether full-history training works, not a forecast: the experiment
trains and scores on past weather only (ERA5 temperature and CAMS irradiance), so every score is a
ceiling for a forecast made on weather forecasts, and no ensemble forecast (ENS) is used.

**Builds on** [experiment
107](../00107-differentiable-physics-rolling-origin-history-and-tails/result.md), whose 12 quarterly
refits, rows, spread, and scoring this experiment reuses, and [experiment
106](../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md) and [experiment
104](../00104-differentiable-physics-2-simple-structural-model/result.md), which found the sign
changes. The solar term is the pinned capacity of `studies.dp_pinned_capacity`, the interface that
[experiment 139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) feeds an
as-of capacity through.

**Arms.** All four arms are the convex model of `studies.dp_gam` at experiment 104's tuned
penalties, with the pinned solar capacity, on the same scored rows.

| Arm | Training window | Drift term |
|---|---|---|
| A24 | last 24 months | none |
| A24+D1 | last 24 months | D1 |
| Aall | every year from the substation's first usable date | none |
| Aall+D1 | every year from the substation's first usable date | D1 |

**Planned contrasts** (treatment minus reference, negative favours the treatment): C1 is Aall minus
A24; C2 (the verdict) is Aall+D1 minus A24+D1; C3 is Aall+D1 minus Aall; C4 is A24+D1 minus A24.
Each contrast is reported per refit origin as well as pooled, because experiment 107 showed that a
pooled number hid sign changes between origins.

**The history rule.** The substation list is the five long-history Primaries of experiment 107 (D5,
D10, D11, D14, and D16), each starting at its first usable date after the early-calibration cuts of
[experiment 103](../00103-differentiable-physics-1-data-look/result.md) (D11 2019-09-17, D10
2019-12-01, D5 2020-02-01, D14 2020-07-01, and D16 2021-02-01). **A substation-origin pair is scored
only when the substation has at least 36 months of usable history before the origin**, so that the
all-history window holds at least 12 months more than the 24-month window. D11, D10, D5, and D14
pass at every origin. D16 passes from the 2024-04-01 origin. The other Primaries are not used: D9
and D18 are too short, and the MVA Primaries need the dark-hours method of experiment 103. The first
months before each substation's first usable date are cut and not trained on, as in experiment 103.

**Rows and refits.** Experiment 107's 12 scored origins (2023-07-01 to 2026-04-01, each frozen and
scored on the next 3 months) after its four burn-in origins, which only supply the first spread
pools. The spread is experiment 107's baseline spread S0, fitted on each arm's own residuals over the
12 months before each origin. The retraining schedule is a quarterly refit of the whole
fit, which is the schedule of experiment 107. Contrasts could depend on cadence: experiment
85 found quarterly refits score 0.38 points worse than weekly refits, and this experiment does not
test the interaction.

**Checks that block the verdict.** The no-lookahead checks of experiment 107 (every training row
before the cut, no drift knot after the cut, no spread-pool row after the origin), the reproduction
of experiment 107's G-all and G-all+D1 scores by Aall and Aall+D1 to the last digit where the
windows and capacity match (they use the same pinned capacity), and the history rule's counts of
substation-origin pairs per origin.

**Leave-one-site-out and splits.** C1 to C4 are repeated leaving out each substation, and split by
winter-quarter and summer-quarter origins and by origin year, because experiment 107's contrasts
changed with each. A post hoc diagnosis of any driver is labelled.

**Outputs.** One figure per contrast across the 12 origins (headline first), absolute scores per arm
and substation, leave-one-site-out sign counts, tail coverage, a week of forecasts for each arm, and
the mean forecast error per arm. The `Checks` list follows the `ml-experiment` skill: the median
minus the mean forecast per site, the mean-forecast error beside each score contrast, and which
audit triggers fired. NGED's metered generators are not in this experiment (demand substations
only), and substations are labelled D5 and so on.

**Possible follow-up, not part of this plan.** If experiment 139 finds the as-of capacity at least
as good as the pin, rerun Aall+D1 and A24+D1 with the as-of capacity as a labelled sensitivity.

## Size and reviews

- **Stored:** private files under the experiment's folder in `data/ml-experiments/` only.
- **Serving path:** not touched. **Degradation rule:** not touched.
- **Designs:** one defensible design, experiment 107's, with a different window length and a drift
  switch.
- **Callers:** this folder's scripts only.
- **Reviews (Sonnet only):** one code review for leakage and for the fits stopping at the cut, one
  write-up review against the printed report, and one prose review.
- **Compute:** CPU only, at most 8 cores, one thread per worker, inside `systemd-run --user --scope
  -p MemoryMax=16G -p OOMPolicy=continue`. Experiment 107's nine arms took about 15 minutes, so four
  arms take less.
