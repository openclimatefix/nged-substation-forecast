---
id: "00288"
slug: pooled-head-on-public-wind-farms
title: >-
  Does an 11-quantile correction head trained on dozens of public Great Britain wind farms lower the
  quantile score of the forecasts for the three metered wind farms, and does more history help the
  XGBoost model more than the physics model?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: finished
date: 2026-10-23
size: >-
  complex. Stored: private farm tables, ERA5 and ENS extracts at the farms' cells, trained weights,
  and predictions under data/ml-experiments/00288-pooled-head-on-public-wind-farms/ and
  data/studies/downloads/market_wind00288/ only. Serving path: not touched (the experiment decides
  whether a later port is worth proposing). Degradation rule: not touched. Designs: several are
  defensible (which farms, how to drop curtailed half-hours, which base model the head corrects,
  how folds avoid sharing weather), so this plan fixes one before any fit. Callers: this folder's
  scripts, which copy experiment 259's, and `studies.*` helpers only. One plan review, one code
  review, one write-up review, and one prose review, each by a fresh Sonnet sub-agent; an audit
  runs if a trigger fires.
hypothesis: >-
  A neural head that adds 11 quantile offsets to the XGBoost wind forecast (R0) and was trained on
  dozens of public onshore wind farms (metered output from Elexon's Balancing Mechanism Reporting
  Service) lowers the quantile score of the three metered wind farms (W1 to W3) against R0 alone,
  and a head trained on the pool beats the same head trained on W1 to W3 only, because experiment
  259's head had too few farm-years to learn a correction that transfers. A second hypothesis:
  XGBoost keeps improving as the training history grows from 12 to 48 months while the physics
  core is flat after 24 months, and a pool of public farms replaces the missing history.
prediction: >-
  Written before any fit. C1, P2 minus P1 on W1 to W3, kept rows, lead days 1 and 2: at most -0.10
  points of capacity (pooling repairs the small-sample failure). C2, P3 minus P0: inside -0.30 to
  0.00 points, with P3's 0.99 pinball loss at least 5% lower than P0's. C3, P2 minus X2 (the
  non-neural pooled arm): at least 0.03 points lower, which is the threshold for keeping the
  network. Blind share, (BL2 minus P0) divided by (P2 minus P0): under 0.5. Held-out farms: a head
  scored on a held-out fifth of the public farms scores within 5% of the same farms scored by a head
  that trained on them. Learning curve: R0's quantile score on W1 to W3 falls by at least 0.15
  points from 24 to 48 months of history, the physics core's changes by less than 0.05, and a
  base XGBoost model trained on the public farms alone scores within 0.30 points of R0 trained on
  12 months of the farm's own history.
decision_rule: >-
  Three decisive contrasts, C1 to C3, each paired per case on the quantile score on W1 to W3's
  kept rows at lead days 1 and 2 (rows kept by experiment 210's mask), read at 98.33% (Bonferroni
  across three), resampling whole scored months and the three seeds, 50,000 resamples. The
  threshold for porting the pooled head is C2 wholly below zero, because the head adds a trained
  network and an Elexon feed to the training pipeline (not the serving path). The tail score T (mean
  of the 0.01 and 0.99 pinball losses) is reported beside the headline and is not a decisive test.
  C3 decides whether the network is kept over the non-neural stage; a C3 interval that does not lie
  wholly below -0.03 means the non-neural stage is preferred. Learning curves, the blind share, the
  held-out-farm check, all leads, and everything per farm are descriptive. The intervals describe
  these three farms.
benchmark: >-
  P0, the XGBoost recipe of experiment 230's A0r (R0) fitted on ERA5 before 2024-04-01 and run on
  51 ensemble members with the spread factor. Experiment 259's stored R0 rows serve W1 to W3.
  Reference rows: month-by-half-hour climatology and experiment 230's stored A0r.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3). Headline on the rows kept by experiment 210's mask; all rows descriptive.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of
  capacity; pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; the tail score T; coverage of the 5% to
  95% and 1% to 99% bands; by lead band
production_status: not-for-production
adopted_in: none
code: ml-experiments/00288-pooled-head-on-public-wind-farms/
verdict: not supported
best_arm: >-
  P1, the head trained on W1 to W3 alone, 6.706 points on the kept rows at lead days 1 and 2; the
  best arm that uses the public farms is X3 (XGBoost stage trained on the pool plus W1 to W3),
  6.740; R0 scored 7.091.
headline: >-
  The head trained on the public farms and applied with no fine-tune scored 1.15 points worse than
  R0 (8.239 against 7.091); fine-tuning on W1 to W3 recovered a 0.28-point gain over R0, no better
  than an offset table fitted on W1 to W3 alone.
lessons: >-
  The public farms are almost all in Scotland, and a head trained on them does not carry a
  correction to farms in England and Wales. Even within the pool, a head that has not seen a farm
  beat R0 on only 11 of 40 farms. After fine-tuning, the pooled head gained what any correction
  fitted on W1 to W3's own runs gains, and its 0.99 pinball loss did not fall. R0's score on the
  public farms did not change with 12 to 54 months of history.
take_home_message: >-
  Training the correction on 40 public wind farms did not help the forecasts for the three NGED
  wind farms. Applied directly, it made them worse, because the public farms are nearly all in
  Scotland. After a short fine-tune on the three farms' own data, it matched what a simple offset
  table fitted on those farms gives, so the public farms added nothing.
advice:
  - >-
    Do not apply a correction head trained on Scottish wind farms to farms in England and Wales
    without fine-tuning (C1, C3).
  - >-
    Fit the correction on the target farms' own runs; a pooled head fine-tuned on them scored no
    better than an offset table per lead day and quantile (D2).
  - >-
    Look for public wind farms in England and Wales before pooling again; the register held 1
    English and no Welsh farm that passed the checks (region table).
  - >-
    Do not apply the 5% physical-notification drop to wind farms' half-hours; the notification is a
    forecast that exceeds metered output in 35% of half-hours (data stage).
---

**Training a correction head on 40 public wind farms did not help the three metered wind farms
(W1 to W3).** Applied with no fine-tune, the pooled head scored 8.239 points of capacity at lead
days 1 and 2, worse than the XGBoost model alone (R0, 7.091). After a fine-tune of up to 3 epochs on
W1 to W3's own training runs, it scored 6.812, a 0.28-point gain over R0 that a table of quantile
offsets per lead day fitted on W1 to W3 alone nearly matches (T1, 6.876). The public farms added
nothing that W1 to W3's own runs did not already supply. The verdict is `not supported`.

Contains BMRS data © Elexon Limited copyright and database right 2026.

## Results

![Quantile score of every arm on W1 to W3](figure_1.svg)

![Contrasts with 98.33% intervals](figure_2.svg)

**The three decisive contrasts all fail their predictions.** Each interval is the 98.33% interval
from resampling whole months and the three seeds, on W1 to W3's kept rows at lead days 1 and 2.

| Contrast | Prediction | Estimate | Interval | Mean-forecast error beside it |
|---|---|---|---|---|
| C1: P2 minus P1 | at most -0.10 | +1.534 | +1.010 to +2.075 | +2.040 (+1.243 to +2.851) |
| C2: P3 minus P0 | -0.30 to 0.00, and 0.99 pinball loss 5% lower | -0.279 | -0.441 to -0.123 | -0.083 (-0.334 to +0.166) |
| C3: P2 minus X2 | at most -0.03 | +1.016 | +0.603 to +1.442 | +1.646 (+0.989 to +2.323) |

- **C1:** the pooled head with no fine-tune is worse than the head trained on W1 to W3 alone, by
  more than the interval's width.
- **C2:** the score gain falls inside the predicted range, but the 0.99 pinball loss did not fall.
  P3's 0.401 against P0's 0.402 is a difference of -0.001 (interval -0.008 to +0.007), not 5%.
  Because the second condition fails, C2 does not clear the porting threshold.
- **C3:** the zero-shot network is worse than the zero-shot XGBoost stage (X2), so the non-neural
  stage is preferred.
- **Mean forecast against the median:** the quantile score judges the whole distribution, and the
  forecast's median sits below its mean by 1.6 points for R0 and 1.0 for P3 (kept rows, lead days 1
  and 2). C2's gain on the mean-forecast error is -0.083 with an interval that includes zero, so the
  median improved and the mean did not.

**Fine-tuning recovers only what W1 to W3's own runs would give.** P3 minus T1 is -0.064 (interval
-0.240 to +0.101), P3 minus P1 is +0.106 (-0.017 to +0.232), and X3, the XGBoost stage trained on
the pool plus W1 to W3's rows, scored 6.740 against P3's 6.812 (X3 minus P3 -0.072, -0.209 to
+0.069). The pool's mean offset therefore did not help beyond the offset table. P1 minus P0 is
-0.385 (-0.552 to -0.227), reproducing experiment 259's gain from a head on the three farms.

| Arm | Hours 12 to 36 | Days 1 and 2 | Days 3 to 6 | Days 7 and later | All leads |
|---|---|---|---|---|---|
| P0 (R0 alone) | 6.63 | 7.09 | 10.44 | 13.89 | 11.70 |
| T1 (offset table on W1 to W3) | 6.43 | 6.88 | 10.02 | 13.21 | 11.17 |
| P1 (head on W1 to W3) | 6.21 | 6.71 | 10.03 | 13.28 | 11.17 |
| P2 (pooled head, zero shot) | 7.95 | 8.24 | 10.76 | 13.86 | 11.98 |
| P3 (pooled head, fine-tuned) | 6.31 | 6.81 | 10.07 | 13.29 | 11.20 |
| X2 (XGBoost stage on the pool) | 6.80 | 7.22 | 10.20 | 13.57 | 11.47 |
| X3 (X2 plus W1 to W3 rows) | 6.21 | 6.74 | 9.94 | 13.20 | 11.11 |
| BL2 (blind pooled head) | 7.03 | 7.48 | 10.72 | 14.24 | 12.03 |
| Climatology | 12.87 | 13.12 | 13.88 | 13.48 | 13.55 |

Quantile score in points of capacity on the kept rows; smaller is better. Experiment 230's stored
A0r scored 7.028 at lead days 1 and 2.

| Arm | q01 | q05 | q50 | q95 | q99 | 5% to 95% coverage | 1% to 99% coverage |
|---|---|---|---|---|---|---|---|
| P0 | 0.259 | 1.128 | 5.004 | 1.460 | 0.402 | 0.942 | 0.982 |
| P1 | 0.248 | 1.046 | 4.735 | 1.430 | 0.396 | 0.906 | 0.973 |
| P2 | 0.279 | 1.208 | 5.942 | 1.640 | 0.435 | 0.914 | 0.984 |
| P3 | 0.261 | 1.063 | 4.795 | 1.443 | 0.401 | 0.906 | 0.980 |
| X2 | 0.259 | 1.128 | 4.993 | 1.572 | 0.418 | 0.956 | 0.984 |

Pinball loss at each level (points of capacity) and coverage of the 5% to 95% and 1% to 99% bands,
kept rows, lead days 1 and 2. The tails barely move between arms. Every head's 5% to 95% band
(0.906 to 0.914) is narrower than R0's (0.942), and all of them sit above the 0.90 target.

**The blind share is not interpretable.** BL2 minus P0 is +0.394 and P2 minus P0 is +1.149, a share
of 0.34, under the 0.5 threshold. The share is a fraction of a loss, not of a gain, so the
threshold does not apply.

## The public farms

![Held-out head minus R0 per public farm](figure_3.svg)

- **Region:** of the 40 kept farms, 39 are in Scotland and 1 is in England. The register held no
  Welsh farm that passed the checks. C1 to C3 therefore test transfer from Scotland to England and
  Wales, and the plan's region arm (at least 3 kept farms in England or Wales) did not apply.
- **Size:** the kept farms span 4.7 to 181 MW of capacity and turbine ratings of 0.66 to 4.48 MW;
  27 of 40 have a hub height predicted from turbine rating, because REPD gives none. W1 to W3's
  capacities and ratings are private and were not compared.
- **Selection:** 65 candidates, 52 qualified, 40 kept (the longest histories). The plan's stop rules
  passed: far more than 15 farms qualified, and R0 scored worse than climatology at lead days 1 and
  2 on 1 of 40 farms (mean 9.29 against 15.17).
- **Held-out fifth:** a head trained on the other four spatial blocks scored 9.731 on its held-out
  farms, 12.8% worse than the same farms scored by the all-farm head (8.627) and worse than R0
  (9.283). The plan predicted within 5%. The head beat R0 on 11 of 40 held-out farms. The
  all-farm score is in sample for the pool, so the fair comparison is the held-out score against
  R0.
- **Curtailment drop:** the plan dropped half-hours with a bid-offer acceptance and half-hours
  where the final physical notification exceeds metered output by more than 5% of capacity. The
  second rule removed 35% of half-hours at the median farm (range 20% to 85% over the 65
  candidates), because wind farms' notifications are forecasts that overshoot. Dropping on it also
  selects half-hours where the farm outproduced its own forecast. The primary clean set therefore
  drops acceptance half-hours only (median 3.6%, at most 10.4% on the kept farms), and
  the notification rule is written to a separate file for a sensitivity run that was not made.

![R0's score on the public farms against months of history](figure_4.svg)

**More history did not improve R0 on the public farms.** R0 refitted on the last 12, 24, 36, and 48
months scored 9.285, 9.305, 9.298, and 9.266 at lead days 1 and 2, against 9.283 on all available
history. The manager's learning-curve prediction (a gain of at least 0.15 points from 24 to 48
months) was written for W1 to W3, and this curve is the public farms' only.

## What was not done

- **W1 to W3's learning curve and the core-only arm** were not run, so the claim that XGBoost
  keeps improving to 48 months of history while the physics core is flat is untested here. The
  public farms' curve is flat.
- **The public-farm base model applied to W1 to W3, and the no-drop and notification-rule
  sensitivity runs,** were not run. All three would have tested the same transfer failure from a
  different side, and the result of every arm that uses the pool already answers the question.
- **Five head seeds in the pool:** the pool's held-out check used seed 0 only.

## Checks

- **Median minus mean forecast (kept rows, days 1 and 2):** the median forecast sits below the mean
  forecast by 1.62 points for P0, 0.90 for P1, 1.03 for P3, and 0.84 for P2. Mean-forecast error is
  beside each quantile contrast above.
- **Audit triggers fired:** the result contradicts its own prediction (C1, C3, and the held-out
  check), and a nested arm beats the arm that contains it (the blind head BL2 scores 0.755 points
  better than the full head P2, D10). Within Scotland, the held-out check fails too, so the region
  does not explain the failure alone. The audit findings are triaged below.
- **Retraining schedule:** every R0 model is fitted once on ERA5 before 2024-04-01; heads are
  trained once on runs before 2025-04-16. No rolling retraining.
- **Tail calibration:** coverage and pinball losses at 0.01, 0.05, 0.95, and 0.99 are in the table
  above.

## What the next experiment should answer

**Does a head trained on public wind farms in England and Wales transfer to W1 to W3?** The public
register held no such farm that passed the checks, so the transfer question stays open for farms
in the same region. A cheaper question is whether more of W1 to W3's own history helps the XGBoost
model, which the unrun learning curve on W1 to W3 would answer.
