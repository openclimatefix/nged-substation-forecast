---
id: '00318'
slug: metered-farm-mdn-head-more-features
title: >-
  Does giving the mixture-density head of the metered solar-farm forecast more inputs, especially the
  forecast lead, lower its quantile score?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
kind: change
date: 2026-10-23
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00318-metered-farm-mdn-head-more-features/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: the arms are fixed by the brief, so no design choice is
  open. Callers: this folder's scripts, which import experiment 310's feature and fit modules and
  experiment 313's pure columns, and no new reviewed module. Asset graph: not touched. One Sonnet
  code review and one Sonnet prose review, no Opus.
hypothesis: >-
  Giving the mixture-density head of Metered solar farms research default 1 the forecast lead (lead
  in whole days), the hour of day and day of year as sine and cosine pairs, and the ensemble spread
  of the shortwave irradiance lowers its quantile score, mostly beyond 36 hours and in the lower
  tail, and one head trained on all lead bands beats one head per band.
prediction: >-
  Written before any score is read. P1: lead features lower the score by at least 0.02 points at
  leads beyond 36 hours (days 3 to 10) and change it by under 0.02 points within 12 to 36 hours (day
  1), in own history. P2: the full feature set lowers the 0.99 pinball loss by at least 3% in own
  history. P3: the shuffled-feature control scores within 0.02 points of the default, so any gain is
  attributable to the features. P4: lead features move the coverage of the 0.01 quantile to within
  0.5 points of 1% at days 3 to 10 (experiment 310: 3.3% to 3.5% for the blend). P5 (the pure
  mixture-density network with lead features, trained on all lead bands) closes experiment 313's
  gap to the plant-based head at days 8 to 10 to under 1.5%. P6 (the pure network given the plant
  forecast as an extra column) beats both the pure network and the default by under 0.5%, so the
  plant is redundant. Each can fail. "Beyond 36 hours" means the bands day 3, days 4 to 5, days 6 to 7, and days 8 to 10; P1, P2, and P4 hold when
  the point estimate meets the stated size in every one of those four bands (P2 and P4: pooled over them).
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months, at 99.17% (Bonferroni across six planned contrasts per band: lead minus default,
  full minus default, full pooled minus default, full pooled minus full, pure with lead (pooled)
  minus pure pooled, and the pure network with the plant column minus pure). The research-default rule sets
  the tier: a new input, so a minimum gain of 0.02 points, a no-harm tolerance of 0.02 points on the
  pooled headline and 0.05 on the 0.01 and 0.99 pinball tails, and a replicate on the scored year's
  second half. At most one arm is adopted. The adoption test is on the mixture-density head alone;
  the blend is not rescored, so the best outcome is a candidate. Stop rule: if no arm beats the
  default by 0.02 points at any band, report that the head's inputs are not limiting and add no
  arms.
benchmark: >-
  Metered solar farms research default 1's mixture-density head ("default", five seeds refitted
  here), experiment 313's pure network, experiment 310's `climatology` and `manual_heuristic` rows
row_set_id: >-
  Experiment 310's rows (farms A to F, 2025-09 to 2026-08, day bands 1 to 10)
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00318-metered-farm-mdn-head-more-features/
---

**Plan only: this page is the pre-registration, committed before any fit runs.** The finished
write-up replaces it.

## Question and arms

Metered solar farms research default 1 ([experiment
300](../00300-metered-farm-best-guess-assembly/result.md), scored to 10 days by [experiment
310](../00310-metered-farm-long-lead-scoring/result.md)) corrects the plant's quantiles with a
three-component Gaussian mixture on the residual. [Experiment
313](../00313-metered-farm-pure-xgboost-and-mdn-vs-plant/result.md) found that a mixture network
with no plant is within 0.5% of the plant-based head at day 1 and 3.4% worse at days 8 to 10. This
experiment asks whether giving the head more inputs, and one head for all leads, lowers its score.
Every arm is the same network as experiment 310's `mdn` arm (copied from its fitter), refitted
quarterly on the same rows, with five seeds averaged in own history and three in zero history.

- **default.** Experiment 310's head inputs: plant forecast, plant spread, availability ratio,
  lagged power, lag-missing flag, clear-sky index of the ensemble irradiance, and lead hours.
- **lead.** default plus the lead in whole days (`ceil(lead_hours / 24)`).
- **full.** lead plus the hour of day and the day of year as sine and cosine pairs, and the ensemble
  standard deviation of shortwave irradiance (the only cloud-related spread the store holds; cloud
  cover itself is not stored). The clear-sky ratio is already a default input (`csi`).
- **full_nolag.** full without lagged power, its missing flag, and the availability ratio. This is
  the named control for "the previous day's plant error": experiment 310 found lagged power adds
  nothing, so it is not in the feature set.
- **full_shuffled.** full with the added columns shuffled across rows (control for P3).
- **default_pooled and full_pooled.** One head per (setting, farm, refit) trained on the rows of
  all six lead bands, and scored band by band. default_pooled separates pooling from features.
- **pure_pooled and pure_lead_pooled.** Experiment 313's pure columns (no plant anywhere), then
  those plus the lead in whole days, each trained on all six bands. Experiment 313's per-band
  `pure_mdn` is the comparison row.
- **pure_plant.** The pure columns plus the plant forecast as an extra column, trained per band.

## Compute and schedule

- Quarterly refits on all earlier months (screening schedule), 4 processes with one thread each,
  `MemoryMax=16G`. Zero history runs only default and full per band (three seeds), because its
  training set is five farms; the pooled arms run in own history only. Absolute scores could shift
  slightly under weekly refits; the contrasts were not tested for dependence on the schedule.
- Rows, scoring, and the month-block bootstrap copy experiment 310's scorer.
