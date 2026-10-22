# Problems and the current best forecast

**The experiments serve three problems: demand at Primary substations, the output of metered
generators, and curtailment with its tails.** This page states what NGED asks for, the two baselines
every forecast must beat, the staged design, the best-guess recipes and their ablation results, and
what production does today. Each bullet ends with an evidence tag
([update rule](index.md#update-rule)), and the linked page owns the evidence.

## What NGED asks for

- **The project must give NGED new information about forecasting for its assets, and a negative
  result counts** ([requirements](../../background/requirements.md#core-objectives)) (docs).
- **The forecasts are probabilistic, half-hourly, 14 days ahead, and updated every 6 hours.** Users
  mostly act on 1 to 10 days ahead, so skill in that band matters most
  ([requirements](../../background/requirements.md#core-objectives)) (docs).
- **The day-ahead forecast must be available by 11:00 and cover midnight to 23:59 of the next day**
  ([requirements](../../background/requirements.md#core-objectives)) (docs).
- **The forecast is used for two decisions: buying flexibility ahead of a high-demand risk, and
  curtailing generators ahead of a high-export risk.** The question users ask is how likely net
  demand is to cross a limit, so the tails of each forecast matter most
  ([requirements](../../background/requirements.md#the-worst-case-matters-most-forecasting-threshold-exceedance))
  (docs).

## Problem 1: demand at Primary substations

- **Version 1 covers 32 time series: 16 Primaries, 6 solar farms, 3 wind farms, 2 grid supply points,
  2 bulk supply points, and 3 other generators or storage** ([NGED power
  telemetry](nged-power-telemetry.md#the-series)) (docs).
- **`manual_heuristic` is the baseline NGED recognises, and beating it is the core deliverable.** It
  forecasts each half-hour from 13 analogues: the same weekday and time in the last 6 weeks and in
  weeks 49 to 55 back ([manual heuristic](../../background/manual-heuristic-forecast.md);
  [glossary](../glossary.md#manual_heuristic)) (docs).
- **`climatology` is the second baseline.** It forecasts from past power in the same series, local
  month, half-hour, and weekday-or-weekend cell, and uses no weather
  ([glossary](../glossary.md#climatology)) (docs).
- **On 292 daily runs at the 16 Primaries, the quantile score was 6.182 for `manual_heuristic` and
  7.368 for `climatology`.** The production-style `xgboost_baseline` scored 10.536, and the stack
  scored 5.543 ([experiment
  87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)) (exp 87; 16 Primaries,
  one year; provisional).
- **At the 4 bulk and grid supply points the same three baselines scored 14.506, 14.883, and
  20.859.** The stack scored 13.559 ([experiment
  87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)) (exp 87; 4 supply points,
  one year; provisional).

## Problem 2: generator output

- **The generator experiments score 6 solar farms and 3 wind farms** ([row sets and
  scoring](row-sets-and-scoring.md#series-sets)) (exp 1; fold A; established).
- **On the 292 runs, the two baselines scored 5.718 and 5.708 at solar farms, and 13.906 and 13.461
  at wind farms.** The production-style `xgboost_baseline` scored 8.784 and 17.171 ([experiment
  87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)) (exp 87; 6 solar farms, 3
  wind farms, one year; provisional).
- **`climatology` is level with `manual_heuristic` at solar farms, so a solar forecast shows skill
  only by beating both** ([experiment
  87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)) (exp 87; 6 solar farms,
  one year; provisional).

## Problem 3: curtailment and tails

- **Curtailment is the capping of generators beneath a substation when its export would pass a
  limit.** The scoring code plans caps from a forecast of the substation's export, and charges the
  energy lost to the caps plus a penalty for export that stays over the limit
  ([curtailment cost theme](../themes/curtailment-cost/index.md)) (docs).
- **The two bulk supply points of the pilot hold too few events to rank forecasts at the headline
  limit, so no forecast has yet been shown to curtail less** ([experiment
  84](../00084-substation-limit-curtailment-cost-worked-example/result.md)) (exp 84; 2 bulk supply
  points; provisional).
- **Tail calibration is reported beside the headline score in every forecasting experiment**
  ([row sets and scoring](row-sets-and-scoring.md#tails-and-curtailment)) (docs).
- **The production-style 0.95 quantile was exceeded in 49% of Primary cases, against 7% for the best
  guess of experiment 33** ([experiment
  33](../00033-best-guess-architecture-vs-production/result.md)) (exp 33; 16 Primaries, 73 runs;
  provisional).

## The staged design

- **The design under test splits the forecast into stages.** Stage 0 maps ensemble weather towards
  the analysis weather. Stage 1a is an XGBoost quantile model trained on analysis weather with no
  power lags. Stage 2 calibrates, can use lagged power, and can blend weather products
  ([staged probabilistic forecast](../../roadmap/staged-probabilistic-forecast.md);
  [glossary](../glossary.md#stages-0-1a-and-2)) (docs).
- **N1 is a single-stage model.** It is trained on the ensemble control member with recent
  power as inputs, and its quantiles are pooled over the 51 members ([glossary](../glossary.md#n1))
  (docs).

## The best-guess recipes

- **The best guess of experiment 33 beat the production-style forecast by 4.89 points at Primaries,
  3.81 at solar farms, and 6.27 at wind farms.** The recipe is N1 with the 14-day correction and a
  spread factor of 1.15 for Primaries, and an ERA5-trained mixture over the 51 members for farms
  ([experiment 33](../00033-best-guess-architecture-vs-production/result.md);
  [glossary](../glossary.md#best-guess)) (exp 33; 16 Primaries, 6 solar farms, 3 wind farms, one year;
  provisional).
- **BG2 scored 5.260 at the 16 Primaries and 12.925 at the 4 supply points, against 5.543 and 13.559
  for the stack.** The Primary gain is 0.283 points (98.75% interval -0.425 to -0.146) ([experiment
  87](../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md)) (exp 87; 16 Primaries,
  4 supply points, one year; provisional).
- **BG2 for Primaries is a five-member quantile blend of weekly-refitted and monthly-refitted
  XGBoost models, an analogue model, a pooled-by-type model, and the stack, with a solar offset at 5
  of 16 Primaries** ([glossary](../glossary.md#bg2)) (exp 87; 16 Primaries, one year; provisional).
- **Refitting weekly instead of monthly lowers the Primary score by 0.123 points, and the blend by
  another 0.086.** Removing the blend from BG2 raises the score by 0.091 (exp 87; 16 Primaries, one
  year; provisional).
- **The solar offset lowers the Primary score by 0.114 points on the rows it can touch, which are the
  daylight rows from April to September at 5 of 16 Primaries.** Over all rows the gain is 0.010 (exp
  87; 5 Primaries, one year; provisional).
- **Adding the 14-day correction to the supply-point model raises the score by 0.305 points.** BG2
  serves supply points without it (exp 87; 4 supply points, one year; provisional).
- **For farms, the ensemble-trained model scored 0.050 (solar) and 0.228 (wind) points worse than the
  ERA5-trained mixture, and neither interval rules out a loss beyond the accepted margin.** No farm
  recipe is ranked (exp 87; 6 solar farms, 3 wind farms, one year; provisional).
- **Removing the three recent-power inputs from the best-guess XGBoost model worsens the Primary
  score by 0.36 points once one substation's commissioning months are cut, and the supply-point score
  by 0.28.** Farm scores barely change ([experiment
  102](../00102-simp1-simplifying-the-bg2-best-guess-before-production/result.md)) (exp 102; 16
  Primaries, one year; provisional).
- **The physics-based demand line reached the same level as BG2 at 5 Primaries.** The convex model
  with a Kalman filter on recent errors (GAM-K) scored 2.800 against 2.780 for BG2 (difference +0.020,
  interval -0.109 to +0.157, not like for like) ([experiment
  111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)) (exp 111; 5
  Primaries, one year; provisional).
- **Adding a correction network and a two-number solar summary to the physics-based best guess
  lowered its score from 2.752 to 2.616 (C1 -0.136, 99% interval -0.253 to -0.047).** The two gains
  add (ratio 0.99) ([experiment
  220](../00220-dp-bg-stage4-assembly-with-drop-one/result.md)) (exp 220; 5 Primaries, one year;
  provisional).

## Superseded positions

- **A 14-day error correction on a frozen model is no longer the default for Primary demand.** The
  position of experiments 17, 19, and 21 is superseded by weekly refitting without the correction
  (experiments 34 and 48), and by the Kalman filter on recent errors (experiments 106 and 111) ([retraining
  theme](../themes/retraining-and-training-history/index.md)) (exp 34, 48, 106, 111; Primary demand;
  provisional).
- **"Keep ERA5 for stage 1a until a CAMS arm has a bias-corrected ensemble" is superseded.**
  Experiment 14 paired a CAMS-trained stage 1a with a stage 0 correction, and scored 0.067 points
  better than raw ensemble input (97.5% interval -0.094 to -0.039); its verdict is inconclusive
  because the daylight bias check failed in single months ([experiment
  14](../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)) (exp 14; 16
  Primaries, one year; provisional).
- **"Train demand stage 1a from 2023-09" is superseded.** Each Primary now starts at its first usable
  date ([NGED power telemetry](nged-power-telemetry.md#gaps-ramp-up-and-regime-changes)), and whether
  old years help is unsettled ([long history and drift
  theme](../themes/differentiable-physics-long-history-and-drift/index.md)) (exp 103, 104, 106,
  107; 5 Primaries; provisional).
- **"ERA5-trained" in experiments 1, 3, 9, and 22 assumes ERA5 is the only estimate of past weather.**
  Name the weather source at lag times in every write-up ([weather
  inputs](weather-inputs-by-product.md#availability-at-forecast-time)) (docs).

## What production does today

- **No numbered model release exists, so the live service has no adopted experiment winner**
  ([model releases](../../live_service/model-releases.md)) (docs).
- **The production-style forecast loses to `manual_heuristic` at Primaries.** In experiment 13 it
  scored 1.69 points of capacity worse (99% interval +0.98 to +2.59) ([experiment
  13](../00013-primary-demand-against-stored-baselines/result.md)) (exp 13; 16 Primaries, 73 runs;
  provisional).
- **The origin of the stored production-style table is unconfirmed**, so its scores may not match
  the deployed forecast ([glossary](../glossary.md#xgboost_baseline)) (exp 87; stored table;
  provisional).

## Which results are provisional

- **Every score above rests on one validation year (July 2025 to June 2026), and several settings
  were chosen on that year.** The gains are probably optimistic
  ([production reference and best guess](../themes/production-reference-and-best-guess/index.md))
  (docs).
- **Scores on different row sets, quantile levels, or series counts are not comparable.** The 16
  Primary scores near 5 and the 5-Primary scores near 2.7 come from different rows ([row sets and
  scoring](row-sets-and-scoring.md)) (docs).

Last checked against experiment 225.
