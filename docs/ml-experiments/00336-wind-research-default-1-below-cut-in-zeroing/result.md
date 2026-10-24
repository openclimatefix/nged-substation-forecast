---
id: "00336"
slug: wind-research-default-1-below-cut-in-zeroing
title: >-
  Does setting the wind forecast to zero where the wind speed is below the turbines' cut-in speed
  improve Wind farms research default 1?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: change
date: 2026-10-24
size: >-
  complex by the research-default rule, because the experiment adds three arms to a family that
  adopts at most one. Stored: private mixtures, tables, and scored quantiles under
  data/ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing/ only, plus the default
  plus x forecasts of the adopted arm if there is one. Serving path: not touched (arm x1 would add
  a rule on the ensemble-mean wind speed to the serving path, which the page says and does not do).
  Degradation rule: not touched. Designs: several are defensible for what "zeroed" means (the
  maintainer's proposal), so the plan fixes three interpretations and their variants before any
  score is read. Callers: this folder's scripts, which import experiments 326, 303, 285, 259, and
  230 and call no reviewed module beyond `studies.*`. One Sonnet code review and one Sonnet prose
  review (maintainer's brief), no Opus.
hypothesis: >-
  The maintainer proposes that, for wind, rows where the wind speed is below the turbines' cut-in
  speed carry only noise, so zeroing them lowers the quantile score of Wind farms research default 1
  on the kept rows of the three metered wind farms W1 to W3 at 1 to 10 day leads. Three readings
  are tested, each added alone: x1 sets the forecast to zero power whenever the ensemble-mean 100 m
  speed is below the farm's cut-in speed, x2 drops the ERA5 training rows below cut-in from the
  XGBoost model and the offset table, and x3 sets the training target to zero on those rows. At
  most one is adopted, in the order x1, x2, x3.
prediction: >-
  Written before any score. x1 raises the all-half-hours score by under 0.1 (worse) and lowers the
  kept-rows score by under 0.05 (a hard zero removes the noise floor that the mask tracks but also
  removes real small outputs). x2 changes the kept-rows score by under 0.1 in either direction with
  ERA5 as the weather, and raises it by 0.03 to 0.1 (worse) on the ensemble, as experiment 276 found
  for the 210 mask. x3 is no better than x2. The speed mask covers under half of the rows of
  experiment 210's idle band. The gain of c1 (scoring the default on the rows outside the speed mask)
  disappears when all half-hours are scored. No arm raises the 0.01 or 0.99 pinball loss by more
  than 0.05.
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x1, x2, x3, and the
  first arm that passes all five is adopted (at most one). Condition 1, gain: the 99.67% month-block
  interval of (default plus x, minus default) on the kept-row quantile score, scored at 1 to 10
  days, lies wholly below zero (level 1 - 0.01/3 for three arms), and the point estimate is at least
  0 (every arm is a changed setting, minimum gain 0). Condition 2, no harm: the one-sided 95% upper
  bound of the harm to the pooled 0.01 pinball loss and to the pooled 0.99 pinball loss is at most
  0.05 points each; each farm on its own, rows with wind above rated, and lead days 1 to 2 are
  reported without a pass or fail. Condition 3, replication: the replicate period of 2025-01 to
  2025-06 gives the same sign, and a t test on the six monthly paired differences does not
  contradict it. Condition 4: the reproduction gate, the shuffled-mask placebo, the known-answer
  test, and the leakage check pass, and no audit trigger is unresolved. Condition 5: every mask reads
  only ensemble weather available at forecast time, and the page says if the port needs a
  serving-path change. A change that wins on condition 1 and fails another stays a candidate. The
  variants x1m (per-member masks) and x2e (also drop the 210 mask rows) and the evaluation-only
  variants c1 and c2 are descriptive and are never adopted.
benchmark: >-
  The default: Wind farms research default 1, the ERA5-trained XGBoost of experiment 33 on the 51
  members with the spread factor, plus experiment 259's offset table (experiment 303's P1, 11.352
  points on the kept rows, as experiment 326 reproduced). Also reported beside it: climatology, the
  persistence on hours 0 to 6, and the frozen XGBoost reference (the default without the table).
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all rows beside). Replicate: the daily hindcast runs of 2025-01-01 to 2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline and 0.05 for the 0.01 and 0.99 pinball
  losses.
code: ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing/
take_home_message: >-
  Planned: the result will say whether zeroing the wind forecast below the turbines' cut-in speed
  helps, and whether the rows below cut-in are the noisy rows that experiment 210 found.
---

**Plan, committed before any score is read.** We test the maintainer's proposal that wind rows
below the turbines' cut-in speed (the wind speed at which a turbine starts to turn) are noise and
should be zeroed, in training and in evaluation, on Wind farms research default 1 (the
ERA5-trained XGBoost quantile model on the 51 ensemble members, plus experiment 259's offset
table), scored at 1 to 10 days.

## The five plan items

1. **Family and version.** Wind farms research default 1, rows `fold_a_tuesdays_v1`, kept rows
   (the mask of [experiment 210](../00210-wind-idle-noise-mva-artefact/result.md)) as the headline
   and all half-hours beside, lead range 1 to 10 days (days 1 to 2 and days 3 to 10 separately),
   quantile score over 11 levels in points of capacity, tolerances 0.02 and 0.05. The XGBoost
   models are fitted once on ERA5 before 2024-04-01, as in [experiment
   326](../00326-wind-research-default-1-change-arms/result.md), not refitted quarterly.
2. **Arms.** The default (experiment 326's stored default, with a reproduction gate: the unmasked
   pipeline rebuilt here must match the stored kept-row score of 11.352 within 0.01, and the
   default without the table must stay 0.345 worse, as in [experiment
   259](../00259-wind-plant-plus-quantile-offset-head/result.md)) and default plus x for x in this
   pre-registered order, at most one adopted, interval at 1 - 0.01/3:
    - **x1, a forecast rule:** all 11 quantiles are set to zero power whenever the MEAN mask is set
      (the 51 members' mean 100 m speed is below the farm's cut-in speed). The training is
      unchanged. Variant x1m applies each member's own speed (a member below cut-in contributes
      zero power before the 51 members are pooled; the spread factor and the offset table then run
      unchanged). Variant x1a uses the ALL mask (every member below cut-in).
    - **x2, a training rule:** ERA5 rows whose 100 m speed is below cut-in are dropped from the
      training of the XGBoost model and from the offset table's hindcast cases (judged by the ERA5
      speed at the case's valid time). The scoring rows are unchanged. Variant x2e also drops the
      rows where the 210 mask is set.
    - **x3, a training target rule:** the training target is set to zero where the ERA5 speed is
      below cut-in (rows kept, offset table unchanged).
    - **Evaluation-only variants, descriptive:** c1 scores the default on the rows outside the
      ERA5 mask and outside the MEAN mask (a different row set); c2 scores the default against a
      target set to zero where the ERA5 mask is set (a different target). They are the proposal's
      literal reading and are never adopted.
3. **Tier.** x1, x2, and x3 are changed settings (minimum gain 0). x2 and x3 act inside the fitted
   stage (tier 2, code imported from experiment 326 and its imports); x1 acts on stored forecasts
   and member quantiles.
4. **Predictions that can fail:** see the front matter.
5. **Controls.** A shuffled-mask placebo (the same share of rows masked at random within farm and
   month) for x1, x2, and x3; column subsampling off in every XGBoost fit; a known-answer test (a
   synthetic series with a known share of noisy near-zero rows below and above cut-in: the mask must
   flag every below-cut-in row and no above-cut-in row); a leakage check that the MEAN, ALL, and
   MEMBER masks read only ensemble member weather, and that the ensemble-mean speed equals
   experiment 307's.

## Cut-in speeds and masks

- **Cut-in speed per farm** is the cut-in of the parametric power curve of [experiment
  230](../00230-wind-physics-core-part-1/result.md)'s physics core, fitted on the ERA5 training
  window: W1 2.69 m/s, W2 2.85 m/s, W3 3.38 m/s. All three lie in the 2 to 5 m/s range, so none is
  replaced by 3.0 m/s. Sensitivity arms use each cut-in plus and minus 1 m/s, and a generic 3.0
  m/s.
- **The speed** is the 100 m wind speed. ERA5 at the valid time sets the training masks (and is an
  oracle in the evaluation-only variants). The forecast-time masks use the ensemble members' 100 m
  speed only. The implied wind of [experiment 307](../00307-wind-farm-power-as-wind-sensor/result.md)
  is never used for a mask.

## Replicate and reporting

As in experiment 326: the offset table (and the spread factor's cases) fitted on hindcast cases
with valid times before 2025-01-01, the daily hindcast runs of 2025-01 to 2025-06-16 scored, and a
t test on the monthly paired differences. Intervals resample whole months and the three seeds
(50,000 resamples), farms not resampled. The write-up reports five row classes (all half-hours,
kept rows, speed-masked rows by each of ERA5, MEAN, and ALL, the 210 idle-band rows, and the rest)
with the share of rows in each, results by lead band and farm, the 0.01 and 0.99 pinball loss, and
coverage; the default against climatology, persistence, and the default without the table; and the
physics core with the adapter as a reported alternative family. Compute: CPU plus under 1 GPU hour,
cap 6 hours.
