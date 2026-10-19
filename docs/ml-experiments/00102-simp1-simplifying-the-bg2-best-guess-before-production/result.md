---
id: "00102"
slug: simp1-simplifying-the-bg2-best-guess-before-production
status: planned
size: complex
hypothesis: >-
  Most of the parts of the BG2 best guess can be removed with a loss below the cost of keeping them in
  production: a single weekly-refitted XGBoost model without the three power lags, the solar offset,
  the 14-day correction, or the blend scores within 0.15 points of capacity of BG2 at the 16 Primary
  substations, within 0.30 at the 4 bulk and grid supply points, within 0.10 at the 6 solar farms, and
  within 0.30 at the 3 wind farms. Separately, the solar offset helps most where it should, at midday
  in summer at the 5 Primary substations that have one.
prediction: >-
  Dropping the three power lags raises the score by 0.05 to 0.30 points at Primaries, 0.10 to 0.50 at
  supply points, 0.10 to 0.40 at solar farms, and 0.10 to 0.50 at wind farms, because earlier
  experiments found lagged power helps most in the first day. The simplest Primary candidate scores
  0.20 to 0.40 above BG2 (the weekly model alone is already 0.095 above, experiment 87). The best
  2-member blend is within 0.05 of BG2. The solar offset lowers the score by 0.05 to 0.15 at the 5
  offset Primaries in midday summer rows, and by less than 0.01 at night.
decision_rule: >-
  Five verdict contrasts, each read at 99% (5% divided by five, Bonferroni), whole origin months
  resampled, 20,000 resamples. C1 to C4 are the simplest candidate of each target class minus BG2 on
  the same runs. The simplest candidate is, for Primary, the single weekly-refitted N1 without lags,
  offset, correction, or blend; for supply points, the weekly N1 without lags; for solar and wind
  farms, BG2's ENS-trained N1 without lags. It is safe to take to production when the interval's
  upper end lies below the threshold: +0.15 (Primary), +0.30 (supply points), +0.10 (solar farms),
  +0.30 (wind farms). The Primary and supply point thresholds are about half of BG2's own gain over
  the stack in experiment 87 (0.283 and 0.634), because a simplification that gives back half of a
  gain has not earned its place; the farm thresholds are experiment 87's non-inferiority margins. C5
  is BG2 minus BG2 without the offset (stored arms p_s6_bg2 and p_s5_blend_no_offset) on the 5 offset
  Primaries, for valid times from 10:00 to 14:00 UK time in April to September; the offset helps when
  the interval lies wholly below zero (threshold zero: the offset is already built). The overall
  verdict is `supported` when C1 to C4 are all safe and C5 helps, `partly supported` when at least
  one holds, and `refuted` when none does. Every other arm, every single-lag or single-part
  simplification, and every comparison with `manual_heuristic` is read at 95% and decides no
  verdict; each is labelled safe when its upper end lies below its class's threshold.
benchmark: >-
  BG2 (experiment 87: `p_s6_bg2`, `b_weekly`, `g_ens_weekly`), `manual_heuristic`, the stack
  (`p_s2_stack`, `b_stack`), `climatology`, and the production-style `xgboost_baseline`, all stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss) in points of capacity over 11 levels
  (0.01, 0.05 to 0.95, 0.99). Reported and deciding nothing: mean absolute error of the median, and
  coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99.
code: ml-experiments/00102-simp1-simplifying-the-bg2-best-guess-before-production/
---

# Experiment 102: SIMP1, how much of the BG2 best guess can be removed before production

**Status: planned. This page is the plan; the results replace it.**

We ask which parts of the BG2 best guess (experiment 87) earn their place in production. BG2 forecasts
the 16 Primary substations with a blend of five XGBoost models, a solar offset at 5 of them, and a
14-day error correction in one model. Every model gets three recent-power lags. We remove each part
in turn, and then all parts together, and score each result against BG2 on the same 292 daily runs.

## Design

- **Row set and scoring.** The 292 runs of `fold_a_daily_gap_v1` (July 2025 to June 2026), the quantile
  score in points of capacity, whole origin months resampled, as in experiment 87.
- **Schedule.** Monthly refits on the rolling window of BG2 for every contrast that needs a refit
  (SPD1, experiment 85); a weekly refit only for the quoted absolute score of the simplest
  candidate at Primaries and supply points. The no-lag generator models refit monthly, like BG2's
  generator arms, which were monthly.
- **Primary arms (reuse stored member predictions, no refit).** Each of the five members alone (W, M,
  AN, PT, ST, each as in BG2 with the offset at the 5 offset series), every 2-member and 3-member
  blend, the best pair and best triple chosen on the 73 runs of `fold_a_every5d_v1` and scored on the
  292 gap runs, BG2 with ST uncorrected, BG2 without the offset (stored), and BG2 without the spread
  factor (stored).
- **Primary arms (refit).** N1 without the three lags: monthly refit (contrast against M) and
  weekly refit (contrast against W). The simplest candidate is the weekly no-lag N1, no offset, no
  correction, with the spread factor of 1.15 (and a variant without it).
- **Supply point arms.** Weekly N1 with lags (BG2, stored) and without (refit).
- **Farm arms.** BG2's ENS-trained N1 with lags (stored) and without (refit), plus the stored arms with
  the fitted spread factor and, for wind, with the power-curve margin.
- **Midday summer.** Rows with valid time 10:00 to 14:00 UK time in April to September, at the 5
  offset Primaries and at all 16; night (23:00 to 03:59 UK time, all year) as a control where
  the offset cannot help; every other row as the remainder. For farms, the same split shows what the
  lags are worth when the sun is up.
- **Extra outputs.** Example-forecast weeks with quantile bands for every configuration (a summer sunny
  week, a winter cold week, a week with large errors), the five blend members' output beside the
  blend over several weeks at a few Primaries, and the blend weights per month. Generators carry
  anonymous labels only.
- **Reporting.** Tails (coverage and pinball loss at 0.01, 0.05, 0.95, 0.99) for the simplest
  candidates, and the position of each simplification against `manual_heuristic` and the stack. A
  production change is not made; the write-up says which simplifications are safe.

## Size

- **What is stored.** New arms in `data/ml-experiments/00102-*/predictions`; no production table.
- **Serving path.** Not touched.
- **Degradation rule.** Not touched. Removing the lags changes which inputs a model needs, but no
  degradation behaviour is edited here.
- **More than one design.** The single-part drops are fixed by the brief; only the choice of best pair
  and triple is a design choice, made on the disjoint 73 runs.
- **Callers.** Copied from experiment 87's own folder; callers are known.

Reviews: Sonnet for the plan, the code, and the write-up (the maintainer's brief; Opus only if a
verdict could change). Earlier advice followed: monthly refits for contrasts (SPD1), the offset tested
on daylight rows (experiment 87), and a farm lag ablation (experiment 23 tested lags only in the
stage 2 calibrator). Departed from: none.
