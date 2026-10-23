---
id: "00308"
slug: lsc-two-forecast-gap-as-a-detector
title: >-
  Is the gap between a no-lag forecast and a forecast with lags a detector of LSC events?
theme: lsc-detectors
also_relevant_to: []
status: planned
size: >-
  medium. Stored: private score tables under data/ml-experiments/00308-lsc-two-forecast-gap-as-a-detector/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible for the alarm episode, the match rule, and the neighbour check, so each is fixed
  below before any score is read. Callers: this folder's scripts only, which read the stored
  harness of experiment 258 and the stored worlds of experiment 141. Asset graph: not touched.
  The maintainer's brief limits the work to Sonnet reviewers.
hypothesis: >-
  The gap between the forecast with lags (K0, the Kalman filter on recent power) and the forecast
  without lags (GAM_ONLY, which forecasts the normal running arrangement) at a forecast origin
  already works as a load shift detector. NGED cannot see the true switching state, so this gap is
  the only signal it has. This is part 1 of the brief
  briefs/sw_lsc_two_forecasts_divergence_and_selection_brief.md.
prediction: >-
  Written before any score is read. Variant (a), the plain gap, or variant (b), the gap with the
  neighbour check, matches at least as many of the 35 real load shifts over 1 day as the learned
  score (LR_b20) at LR_b20's false alarm rate, and at least as many as RO-NC at RO-NC's false
  alarm rate. The prediction fails if the gap is dominated by weather error or misfit days
  (holidays, clock changes, D8 and D9), in which case the page names which.
decision_rule: >-
  Fixed before any score is read. Rows: experiment 258's harness (15 Primary series, whole
  history from 2022-01 to 2026-06, an origin every 6 hours, the frozen past-weather GAM of
  experiment 130 as GAM_ONLY, K0 as in experiment 258). Gap scores at an origin, in units of the
  series' noise (1.4826 times the median absolute deviation of the GAM residual in odd months):
  the mean of K0 minus GAM_ONLY over leads 0 to 6 hours (in MW as well), and the mean residual of
  the readings that ended at least 3.5 hours before the origin, over the last 6, 24, and 72 hours.
  An origin is flagged when the absolute score exceeds a threshold k. Consecutive flagged
  origins within 48 hours form one alarm episode; the episode opens at the first flagged origin.
  The threshold is chosen on odd months of ordinary rows only, without labels, as the smallest k
  at which the false alarm rate equals a target rate, and is applied unchanged to even months.
  The target rates are the false alarm rates, measured by the same rule on the same odd months,
  of LR_b20, RO-NC (RO_b80), and the pair alarm (PG_g70), whose stored interval tables are
  scored by the same match rule. Variants: (a) plain; (b) with the neighbour check: the episode is
  kept when at least one connected metered neighbour has a gap of the opposite sign and at least
  30% of the node's size in MW, and the sum of the node's gap and its opposite-sign neighbours'
  gaps is at most half the node's gap; (c) a minimum duration: flagged at two consecutive
  origins (6 hours) or at five consecutive origins (24 hours). A real shift is matched when an
  alarm episode opens between 2 hours before its onset and its end, on either Primary of its
  pair. The lag is the origin's data cutoff minus the onset. A false alarm is an episode that
  opens in an ordinary row: outside every logged closure that touches the series or a
  connected neighbour, and the 72 hours after it, and outside D8's multi-day drops and the 17
  block days (the clean set). D8 and D9 are reported separately. Scoring sets: the 35 real
  shifts over 1 day (matched count and MW-weighted share, alarms per match, lag); the 411
  shifts injected into 44 worlds of experiment 141 (the neighbour check does not apply, because
  the receivers are unmetered); D8's drops and the 17 block days, separately; false alarms per
  Primary-year. Intervals: whole shifts resampled for matches, whole calendar months for
  false alarm rates, 95%, 20,000 draws. Support: variant (a) or (b) matches at least as many of
  the 35 as LR_b20 and as RO-NC at their respective false alarm rates, in the point estimate;
  the interval is reported beside. Verdict `supported` when both comparisons hold,
  `partly supported` when one holds, `not supported` when neither holds. The page also reports
  which of weather error, holidays and clock changes, D8, D9, and misfit days hold the false
  alarms, because the prediction names them as the failure case.
benchmark: >-
  LR_b20 (the carried learned score), RO-NC, and the pair alarm, scored on the same rows by the
  same match rule at their own false alarm rates.
row_set_id: >-
  Experiment 258's harness rows (15 Primary series, 6-hourly origins); not comparable with the
  leaderboard
metric: >-
  Matched shifts of the 35 real shifts over 1 day, alarms per match, lag in hours, false alarms per
  Primary-year
code: ml-experiments/00308-lsc-two-forecast-gap-as-a-detector/
---

**Planned.** The plan is the front matter above; the write-up replaces this body when the
experiment finishes.
