---
id: "00102"
slug: simp1-simplifying-the-bg2-best-guess-before-production
status: planned
size: complex
hypothesis: >-
  A single weekly-refitted XGBoost model without the three power lags, the solar offset, the 14-day
  correction, or the blend scores within 0.15 points of capacity of BG2 at the 16 Primary
  substations, within 0.30 at the 4 bulk and grid supply points, within 0.10 at the 6 solar farms, and
  within 0.30 at the 3 wind farms. We expect the Primary claim to fail and the other three to be
  close calls (see the prediction). Separately, the solar offset helps most where it should, at midday
  in summer at the 5 Primary substations that have one.
prediction: >-
  Dropping the three power lags raises the score by 0.05 to 0.30 points at Primaries, 0.10 to 0.50 at
  supply points, 0.10 to 0.40 at solar farms, and 0.10 to 0.50 at wind farms, because earlier
  experiments found lagged power helps most in the first day. The simplest Primary candidate scores
  0.20 to 0.40 above BG2 (the weekly model alone is already 0.095 above, experiment 87), so we expect
  C1 to fail its 0.15 threshold; the ranges for supply points and wind straddle their 0.30
  thresholds, so C2 and C4 are genuinely open. The best
  2-member blend is within 0.05 of BG2. The solar offset lowers the score by 0.05 to 0.15 at the 5
  offset Primaries in midday summer rows, and by less than 0.01 at night.
decision_rule: >-
  Five verdict contrasts, each read at 99% (5% divided by five, Bonferroni), whole origin months
  resampled, 20,000 resamples. C1 to C4 are the simplest candidate of each target class minus BG2 on
  the same runs. The simplest candidate is, for Primary, the single weekly-refitted N1 without lags,
  offset, correction, or blend and with BG2's spread factor of 1.15; for supply points, the weekly N1
  without lags and with no spread factor (as in BG2's `b_weekly`, which is weekly); for solar and wind
  farms, BG2's ENS-trained N1 without lags, refitted monthly and with no spread factor, like the
  stored arm `g_ens_weekly`, which experiment 87 also refitted monthly. It is safe to take to production when the interval's
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
- **Schedule.** The weekly no-lag refit feeds C1 and C2 (against BG2's weekly arms) and the
  quoted absolute scores. The monthly no-lag refit feeds the lag contrast at Primaries (against
  the stored monthly member) and is cheap (SPD1, experiment 85). The no-lag farm models refit monthly,
  like BG2's farm arms. Every contrast says which schedule it used.
- **Primary arms (reuse stored member predictions, no refit).** The five blend members are the weekly
  model (W), the monthly model (M), the analogue model (AN), the model pooled by type (PT), and the
  frozen model with the 14-day correction (ST). The arms are each member alone (as in BG2, with the
  offset at the 5 offset series), every 2-member and 3-member blend, BG2 with ST uncorrected, BG2
  without the offset (stored), and BG2 without the spread factor (stored). The best pair and best
  triple are chosen on the 73 runs of `fold_a_every5d_v1` and scored on the 292 gap runs. Experiment
  87's member predictions cover all 365 daily runs, so the choice needs no refit. The two sets of runs
  interleave in time, so the choice is exploratory and decides no verdict. Each blend re-chooses its
  own monthly weights, so the offset contrast C5 also differs in its weights.
- **Primary arms (refit).** N1 without the three lags: monthly refit (contrast against M) and
  weekly refit (contrast against W). The simplest candidate is the weekly no-lag N1, no offset, no
  correction, with the spread factor of 1.15 (and a variant without it).
- **Supply point arms.** Weekly N1 with lags (BG2, stored) and without (refit).
- **Farm arms.** BG2's ENS-trained N1 with lags (stored) and without (refit), plus the stored arms with
  the fitted spread factor and, for wind, with the power-curve margin.
- **Midday summer.** Every arm is scored on rows with valid time 10:00 to 14:00 UK time in April to
  September (six origin months), at the 5 offset Primaries and at all 16 Primaries; on night rows
  (23:00 to 03:59 UK time, all year), where the offset cannot help; and on every other row. The
  offset-minus-no-offset interval is given in the midday-summer rows and in the night rows. For
  farms, the same split shows what the lags are worth when the sun is up.
- **Extra outputs.** Example-forecast weeks with quantile bands for every configuration (a summer sunny
  week, a winter cold week, a week with large errors), the five blend members' output beside the
  blend over several weeks at a few Primaries, and the blend weights per month. Generators carry
  anonymous labels only.
- **Reporting.** For each simplest candidate, the coverage and the pinball loss at the quantiles 0.01,
  0.05, 0.95, and 0.99. Every simplification is also placed against `manual_heuristic` and the stack
  (`p_s2_stack`, `b_stack`). The write-up also quotes the candidate against the weekly model with
  lags (`p_s4_weekly`), which shows what the lags alone are worth. No production change is made; the
  write-up says which simplifications are safe. The example-forecast charts read the stored quantiles
  of every arm, so the refit arms keep their quantiles.

## Size

**The experiment is sized complex because it stores new arms, but it has no serving path.** It stores
new arms in `data/ml-experiments/00102-*/predictions` and no production table. It does not touch the
serving path or a degradation rule. The brief fixes the single-part drops, so the only open design
choice is the best pair and triple. The code is copied from experiment 87's folder, so its callers are
known. Sonnet reviews the plan, the code, and the write-up; Opus is used only if a verdict could
change. The plan follows the earlier advice to use monthly refits for contrasts (experiment 85), to
test the offset on daylight rows (experiment 87), and to ablate the lags in the farm models
(experiment 23 tested lags only in the stage 2 calibrator). It departs from no earlier advice.
