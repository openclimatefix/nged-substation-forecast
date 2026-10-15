---
id: "00021"
slug: error-correction-variants-bsp-gsp-demand
status: planned
size: medium
hypothesis: >-
  A recent-error correction of N1's quantiles, measured with power that is 2 days old, lowers the
  quantile score of the 4 bulk supply point (BSP) and grid supply point (GSP) demand series as it
  does for the 16 Primary substations, and some variant of the correction (a 7-day or 28-day
  window, multiplicative, per-half-hour-of-day, or weighting recent days more) beats the plain
  14-day additive correction.
prediction: >-
  The 14-day additive correction (A14) lowers the BSP and GSP quantile score over all leads by at
  least 0.5 points of capacity, because N1 forecast every series too low over the validation year.
  With the 2-day delay, A14 lowers the Primary quantile score by 0.6 to 0.9 points, less than the
  0.99 of experiment 00019, which used a 3.5-hour delay. No variant beats A14 by more than 0.2
  points; the 28-day window and the per-half-hour correction are the likeliest to lose, because
  their estimates are noisier or staler.
decision_rule: >-
  Eight planned contrasts on the quantile score over all leads, read at 99.375% (Bonferroni across
  eight), whole validation months and the three seeds resampled. C1 (primary) is A14 minus N1 on the
  4 BSP and GSP series. C2 is A14 minus `manual_heuristic` on the same series. C3 is A14 minus N1 on
  the 16 Primary substations (a replication with the 2-day delay). C4 to C8 are, on the Primary
  substations, a 7-day window, a 28-day window, the multiplicative form, the per-half-hour form,
  and the recency-weighted form, each minus A14. The correction is supported for BSP and GSP when
  C1's interval is wholly below zero and its point estimate is at most -0.10 points, the threshold
  that 00019 set for a new pipeline step that reads observed power. It is a small gain when the
  interval is wholly below zero and the estimate is above -0.10, refuted when the interval is wholly
  above zero, and inconclusive otherwise. A variant replaces A14 when its contrast's interval is
  wholly below zero (threshold 0, because the variants change a number or a formula in a step that
  exists anyway); when no variant qualifies, A14 stays. C2 decides no verdict. The same variant
  contrasts on BSP and GSP, the 3.5-hour delay, and every other contrast are exploratory at 95%.
benchmark: >-
  N1, `manual_heuristic`, and `climatology` (stored by 00013, re-scored on the row set).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels) and mean absolute error
  of the median, in points of capacity; mean signed error reported separately as bias
production_status: not-for-production
adopted_in: none
code: ml-experiments/00021-error-correction-variants-bsp-gsp-demand/
---

# Experiment 21: which error correction works best, and does it help BSP and GSP demand?

## Plan: sizing

- **Stored data:** prediction sets only, under `data/ml-experiments/`; no production table changes.
- **Production serving path:** not touched.
- **Degradation rule:** not touched. A run with under 3 days of history gets no correction.
- **Defensible designs:** more than one. The variants are the question; each is fixed below.
- **Callers:** the scripts copy 00019's `corrections_for` and scoring, which the author could name
  without searching.

Size: medium. No model is fitted, because every arm post-processes the stored predictions of N1
from experiment 00013, which covers all 20 series. One Sonnet review of the plan, one of the code
(leakage and the delay), and one of the write-up.

## Plan: arms

All arms are scored on the cases of `fold_a_every5d_v1` that every arm holds, with an observed value
and a valid time before 2026-07-01. The corrected arms apply to N1's three seeds. A correction
shifts or scales all 9 quantiles of a run and re-sorts them. No spread factor is applied, so that
the correction is isolated.

**The correction.** For each run with origin `T`, the window is the `W` days that end 2 days before
`T`. For each half-hour in the window, the forecast is the median of the latest earlier run that
began before it (the rule of 00019, which checks that no forecast comes from a run that began after
its half-hour). The error is the observed power over capacity minus that median. A run with under 3
days of paired history gets no correction.

- **A14:** add the mean error over a 14-day window to every quantile (00019's correction, with the
  delay changed from 3.5 hours to 2 days).
- **A7, A28:** the same with a 7-day and a 28-day window.
- **M14:** multiply every quantile by the ratio of the window's mean observed power to its mean
  forecast median, clipped to 0.5 to 2.0; no correction when the mean median is under 0.01 of
  capacity.
- **H14:** add, to each valid time, the mean error of the window's half-hours with the same UTC
  half-hour of day; a cell with fewer than 3 pairs gets the window's overall mean error.
- **R14:** the 14-day mean error with weights `0.5 ** (age_days / 3.5)`, age counted back from the
  window's end, so the newest days weigh most.
- **A14-fast:** A14 with 00019's 3.5-hour delay, to measure what the 2-day delay costs
  (exploratory).
- **References:** N1, `manual_heuristic`, `climatology`, all uncorrected. A stored arm has no seed,
  so it is repeated once per seed.

**Delay.** A run at origin `T` uses only power observed up to `T - 2 days`, as the brief for this
experiment requires. This is stricter than 00013's lags, which use power 3.5 hours old.

**Bias** is the mean signed error of the median, reported by arm and by group (Primary, BSP, GSP),
beside the quantile score and the mean absolute error of the median.

**Groups.** BSP and GSP have 2 series each. The planned contrasts pool the 4 and the tables also
show each type. A resolved interval for either type alone is not expected.

## Plan: past weather

No past weather is used; the correction reads observed power only, and N1's weather is ENS in
training and scoring.
