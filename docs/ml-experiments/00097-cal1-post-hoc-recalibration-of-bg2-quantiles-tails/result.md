---
id: "00097"
slug: cal1-post-hoc-recalibration-of-bg2-quantiles-tails
status: planned
size: medium
hypothesis: >-
  A post-hoc recalibration of the stored quantiles of the best-guess forecast (BG2, experiment 87),
  fitted on earlier months only, brings the coverage of the 0.01, 0.05, 0.95, and 0.99 quantiles to
  within 0.01 of nominal in the 12 to 36 hour lead band of each class, without making the centre of
  the distribution worse.
prediction: >-
  The best guess already covers the Primary tails better than the older forecasts did: on the whole
  year the unadjusted q05 covers 0.039 and q95 covers 0.938 (nominal 0.05 and 0.95), so the lower
  tail is too wide and the upper tail slightly too narrow. We therefore predict that no single
  global scale (arm 1) reaches the 0.01 band at both ends, that a per-level, per-lead-band, per-class
  adjustment (arms 3 to 5) does, and that arm 6's absolute gap from nominal at 0.5% and 99.5% is smaller than arm 0's and arm 5's,
  which read those levels off a normal extrapolation of q01 and q05.
decision_rule: >-
  An arm passes the tail test when, in the 12 to 36 hour lead band of each class (Primary, and
  bulk or grid supply point), the coverage of q05, q95, q01, and q99 is each within 0.01 of nominal.
  An arm passes the centre test when the upper end of its 95% month-resampled interval for the
  change in centre score against arm 0 is at most 1% of arm 0's centre score. The centre score is
  the mean pinball loss over the levels 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, and 0.90. The best arm
  is the passing arm with the lowest quantile score over all 11 delivered levels. No arm passing
  both tests gives the verdict `refuted`. Where several arms pass, the best arm is a candidate for a
  separate issue; the maintainer decides whether any adjustment is ported, because a stored
  adjustment table is new serving logic. A cell with fewer than 30 days holding an exceedance is
  reported as "not testable" and is neither a pass nor a fail. The decision cells are the 12 to 36
  hour band of each class; every other cell is exploratory.
benchmark: >-
  Arm 0, the stored best guess weekly-retrained forecasts (`p_s4_weekly` for Primary substations,
  `b_weekly` for bulk and grid supply points), and arm 1, the global spread factor 1.15.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Coverage and pinball loss at the levels 0.01, 0.05, 0.95, and 0.99 (and coverage at 0.005 and
  0.995 with the number of exceedances), the mean pinball loss over all 11 delivered levels, and the
  node curtailment cost at the scaled limit of 0.9 (energy proxy at GBP 2,000 per MWh, node-only
  mode, Curtailment Limit 0), reported descriptively because 2 nodes hold too few events to rank
  arms.
code: ml-experiments/00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/
production_status: not-for-production
adopted_in: none
---

# Experiment 97: which post-hoc recalibration of the best guess fixes its tails

**Status: planned. This page is the plan and is rewritten as the write-up once the scoring has
run.** The sections below are fixed before any recalibrated number exists.

## Question

NGED uses these forecasts to decide when to curtail generators, so the outer quantiles matter most.
Earlier experiments found the tails of the older forecasts too narrow, and the global spread factor
of 1.15 is one number for every level, lead, and series. This experiment asks which recalibration of
the best guess's stored quantiles gives the best tail coverage without hurting the centre. It reuses
stored predictions only, so nothing is refitted and the experiment runs on 4 CPU cores.

## Data and rows

- **Stored sets:** `p_s4_weekly` (16 Primary substations, 292 daily runs, 2025-07-02 to 2026-06-30)
  and `b_weekly` (2 bulk supply points and 2 grid supply points, same runs), from experiment 87
  (BG2). Both hold q01, q05 to q95, and q99 as fractions of capacity, plus `actual` and
  `lead_hours`. `p_s2_stack` is a second Primary set and runs through the same code as a
  robustness check. BG2's solar and wind sets had not landed when this plan was written; they join
  as further classes when they exist, and the write-up says which classes it covers.
- **Rows dropped:** rows whose `actual` is NaN (11,189 Primary and 17,311 supply-point rows in
  the sets read), because a NaN measurement has no coverage.
- **Weekly retraining** produced the stored sets, so absolute scores are production-style.
- **Lead bands:** under 12 hours, 12 to 36 hours (the curtailment band), 36 to 96 hours, and 96
  hours or more.
- **Classes:** Primary, and bulk or grid supply point (4 series, pooled for calibration).

## Rolling origin, no peeking

The scored months are October 2025 to June 2026 (9 months, by month of `power_fcst_init_time`).
For a scored month M, every adjustment is fitted on the rows whose `valid_time` is before the
first instant of M, so a run made in an earlier month whose valid times reach into M never leaks
its measurements into M. The window expands: it holds every earlier row, so a fit for a winter month
includes summer rows, and the write-up reports coverage by scored month to show any drift. The
first three months (July to September 2025) only supply calibration data. A scored run is one whose
`power_fcst_init_time` is in M. The stored rows are out of sample by construction, because weekly
retraining fits each run's model on data before the run's initialisation time. Every arm is scored
on the same rows.

## Arms

All arms keep `q50` as stored, sort the output quantiles so they never cross, and fit per class and
lead band unless stated.

0. **As stored.**
1. **Global spread factor 1.15** about the median, for every level.
2. **Scale factor per quantile level:** the adjusted quantile is `q50 + k (q_level - q50)`, on both
   sides of the median, and `k` is the value that makes the calibration coverage of the adjusted
   quantile equal its nominal level (the quantile of `(actual - q50) / (q_level - q50)` at that
   level for levels above 0.5, and at one minus that level below 0.5; rows where the stored
   quantile equals `q50` are skipped). One `k` per level, pooled over lead bands and series within
   a class, limited to 0.5 to 5; the write-up counts how often the limit binds. Stored quantiles are
   sorted before fitting.
3. **Scale factor per level, lead band, and class.**
4. **Level remapping:** isotonic regression of calibration coverage on nominal level, inverted to
   find the stored level whose empirical coverage equals each target (the lowest level of a flat
   segment); the stored quantile function is interpolated linearly in the level, and extrapolated beyond q01 and q99 with the normal
   spacing of q01 and q05.
5. **Split-conformal shift:** for each level, the additive shift equal to the calibration quantile
   of `actual - q_level` (with the usual `ceil((n+1)p)/n` correction).
6. **Stored body plus fitted tails:** q10 to q90 stay as stored. At each end, a generalised Pareto
   distribution is fitted to the calibration excesses beyond the stored q95 (and q05), scaled by
   the stored spread, pooled over the series of a class, with the exceedance probability `p_u`
   taken from the calibration share of rows beyond the stored q95 rather than assumed to be 0.05.
   A level with tail probability `1 - p` below `p_u` is `q95 + spread * GPD quantile at 1 -
   (1 - p) / p_u`, which gives q95, q99, q99.5, and q99.9 (and the mirrored lower levels); a level
   with `1 - p` at or above `p_u` keeps its stored value. The shape parameter is limited to -0.25 to
   0.5.
7. **Fully parametric:** a Student-t (7a) and a skew-normal (7b) fitted to the calibration
   residual `(actual - q50)` divided by the stored interquartile spread over 1.349, subsampled to
   at most 20,000 rows per fit with a fixed seed. Logit-normal for bounded generator output is not
   run, because no generator sets exist yet.

Levels beyond q01 and q99 (0.5%, 99.5%, and 0.1%, 99.9%) come from the normal extrapolation of
arms 0 to 5 and from the fitted tails of arms 6 and 7. A comparison at those levels therefore
compares extrapolation rules as well as calibration methods. The 0.1% and 99.9% levels are
reported with their exceedance counts and are not part of the decision rule.

## Scoring

- **Tail table:** coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99 by arm, lead band, and
  class, with a 95% interval from resampling whole months (`studies.tail_calibration`). Coverage at
  0.005 and 0.995 is reported with the number of exceedances and an interval from the same month
  resampling, which is wide because 9 months hold few exceedances. Every cell also reports the
  number of days holding an exceedance, as the count of independent events. A 9-block month
  resampling gives indicative intervals only.
- **Coverage by scored month** for the 12 to 36 hour band, and the centre score by lead band, so a
  hidden worsening or a seasonal drift is visible.
- **Quantile score** over the 11 delivered levels, and the centre score defined above, with
  paired month-resampled intervals against arm 0.
- **Curtailment cost:** the two bulk supply points (Boston and Skegness) scored through
  `studies.node_curtailment` in node-only mode, at the scaled limit of 0.9 times the maximum export
  before the scored period, with the sweep at 0.75 and 0.5, the event count first, and the perfect
  forecast as reference. Bicker Fen and Walpole are not scored because their export signs are
  unconfirmed. The two supply points hold few events (19 half-hours at 0.9 over a full year), so the
  result shows the arithmetic and cannot rank arms.
- **Leakage check:** the report script asserts, before any number prints, that the largest
  calibration `valid_time` is before the first scored `power_fcst_init_time` of each scored month.

## Size and process

- **Stored:** nothing outside `data/ml-experiments/`; no schema changes.
- **Serving path:** untouched; the experiment reads stored predictions.
- **Degradation rule:** untouched.
- **More than one design:** yes (eight arms are the designs), which is why the arms are all run.
- **Callers across code:** none; the code is one folder under `ml-experiments/` plus the existing
  `studies` machinery.

The size is medium: one Sonnet plan review, Sonnet implementation, one Sonnet code review for
leakage, and one Sonnet prose review.

## Earlier advice followed

- Report tail coverage and pinball at 0.01, 0.05, 0.95, and 0.99 next to the headline score
  (maintainer, 2026-10-17).
- Headline the scaled limit of 0.9 for curtailment and show the real limit only as reference
  (curtailment reporting rule, 2026-10-18).
- Experiment 49's advice to keep the spread factor at 1.15 is the baseline arm 1, not an assumption.
