---
id: "00087"
slug: bg2-best-guess-assembled-with-drop-one-ablations
theme: production-reference-and-best-guess
also_relevant_to: []
status: done
size: complex
hypothesis: >-
  Assembling the best ideas of the last experiments into one forecast beats the previous best forecast
  of each target class on the 292 daily forecast runs of the validation year (July 2025 to June
  2026). For the 16 Primary substations the assembly is a five-member quantile blend of
  weekly-refitted and monthly-refitted XGBoost models, an analogue model, a pooled-by-type model, and
  the frozen stack, with an explicit solar offset at the 5 substations that have a clean solar
  capacity. For the 4 bulk and grid supply points it is the weekly-refitted XGBoost model with no
  error correction. For the 6 solar farms and 3 wind farms it is the weekly-refitted XGBoost model
  trained on the ensemble forecast's own history.
prediction: >-
  Primary: BG2 scores about 5.30 points of capacity on the quantile score, against 5.64 for the stack
  (experiment 48) and 6.26 for `manual_heuristic`; the blend adds -0.05 to -0.10 to weekly
  refitting and the offset adds -0.10 to -0.15. Bulk and grid supply points: weekly refitting scores
  0.2 to 0.7 below the frozen model with the correction (experiment 48, C7). Solar and wind farms:
  the ensemble-trained model is within 0.1 (solar) and 0.3 (wind) points of experiment 33's
  ERA5-trained mixture, which is the non-inferiority margin of decision D1.
decision_rule: >-
  Four verdict contrasts, one per target class, each the quantile score of BG2 minus the class's
  previous best forecast on the same runs, read at 98.75% (Bonferroni across four), whole origin
  months resampled, 20,000 resamples, seed 0. The previous best is the stack for Primary, the frozen
  XGBoost model with the 14-day correction for bulk and grid supply points, and experiment 33's
  ERA5-trained mixture for solar and for wind. Primary and bulk and grid supply points are supported
  when the interval lies wholly below zero (threshold zero: the weekly refit and the blend replace
  existing scheduled jobs and one weight vector per month, and the offset needs one capacity table).
  Solar and wind are supported when the interval's upper end lies below +0.10 (solar) and +0.30
  (wind), because the swap to the ensemble's own history is worth making whenever it costs less than
  that. Every ablation, every staircase stage, and every comparison with `manual_heuristic` is read
  at 95% and decides no verdict. Clarification added at write-up: the overall verdict is `supported`
  when all four contrasts are supported, `partly supported` when at least one and fewer than four
  are, and `refuted` when none is.
benchmark: >-
  The stack, `manual_heuristic`, `climatology`, the production-style `xgboost_baseline` (decision
  D4), and a perfect forecast, each scored on the same 292 runs.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss) in points of capacity, over 11 levels
  (0.01, 0.05 to 0.95, 0.99); the mean absolute error of the median, tail coverage and pinball loss at
  0.01, 0.05, 0.95, and 0.99, and the threshold-weighted CRPS are reported and decide nothing.
verdict: partly supported
best_arm: >-
  Primary substations: the five-model blend with the solar offset (BG2), 5.260 points of capacity on
  the quantile score, against 5.543 for the stack and 6.182 for `manual_heuristic`.
headline: >-
  BG2 beat the previous best forecast at the 16 Primary substations by 0.283 points (98.75%
  interval -0.425 to -0.146) and at the 4 bulk and grid supply points by 0.634 (-1.111 to -0.279).
  For the 6 solar farms and 3 wind farms, the forecast trained on the ensemble's own history scored
  0.050 (-0.125 to +0.168) and 0.228 (-0.178 to +0.653) worse than the ERA5-trained mixture, and
  neither interval rules out a loss larger than the margin.
lessons: >-
  Weekly refitting and the blend carry the Primary gain: refitting weekly instead of monthly lowers
  the score by 0.123 points and the blend by another 0.086. The solar offset lowers it by 0.114
  points (0.075 to 0.164) on the daylight April-to-September rows of the 5 Primaries that carry
  one, inside the -0.10 to -0.15 predicted, and by 0.010 points over all rows of all 16 Primaries,
  because only 8.5% of those rows can carry an offset. At the supply points,
  adding the 14-day correction raised the score by 0.305 points. The generator models cannot yet be
  ranked, and the wind power-curve margin lowered the wind score by 0.057 points.
production_status: candidate
adopted_in: none
arms: >-
  Quantile score over 11 levels on the 292 runs, in points of capacity, with 95% intervals where
  the arm is a headline. Primary: S1 frozen 6.927 (6.294 to 7.507); S2 stack 5.543 (5.275 to 5.800);
  S3 monthly 5.478; S4 weekly 5.355; S5 blend without offset 5.270; S6 BG2 5.260 (4.987 to 5.534);
  A2 no blend 5.351; A3 monthly in the weekly slot 5.288; A4 correction on every member 5.327; A5
  spread factor 1.0 5.277; A6 capacity held 5.261; A7 three members 5.258; `manual_heuristic` 6.182;
  `climatology` 7.368; production-style `xgboost_baseline` 10.536. Supply points: frozen 13.369;
  stack 13.559; monthly 13.085; weekly (BG2) 12.925 (11.569 to 14.177); weekly plus correction
  13.230; `manual_heuristic` 14.506; `climatology` 14.883; `xgboost_baseline` 20.859. Solar farms
  (monthly refits): ERA5-trained mixture 4.962 (3.687 to 6.441); ensemble-trained frozen 5.042;
  BG2 5.012 (3.788 to 6.392); BG2 with the fitted spread factor 5.014; `manual_heuristic` 5.718;
  `climatology` 5.708; `xgboost_baseline` 8.784. Wind farms (monthly refits): ERA5-trained mixture
  10.804 (9.451 to 12.041); ensemble-trained frozen 11.131; BG2 11.032 (9.641 to 12.303); plus
  power-curve margin 10.975; with the fitted spread factor 11.016; `manual_heuristic` 13.906;
  `climatology` 13.461; `xgboost_baseline` 17.171.
decisions: >-
  D1 ensemble-trained generators, run as stated. D2 wind margin as an add-one arm only. D3 curtailment
  labelled "calibration to history, not a saving". D4 production-style row read from the local
  `data/power_forecasts` table. D5 offset at 5 of 16 Primaries. D6 R11 shown only as a labelled side
  panel. D7 twCRPS copied into the experiment's own folder.
advice:
  - >-
    Refit Primary and bulk and grid supply point models weekly rather than monthly, because weekly
    scored 0.123 points (0.051 to 0.202) below monthly at Primaries and 0.160 (0.066 to 0.247) below
    at supply points (staircase steps).
  - >-
    Blend five models (weekly, monthly, analogue, pooled by type, and the stack) rather than serve
    one, because dropping the blend raises the Primary score by 0.091 points (0.048 to 0.138)
    (ablation A2).
  - >-
    Do not add the 14-day correction to the bulk and grid supply point model, because adding it
    back raises the score by 0.305 points (0.060 to 0.594) (ablation, supply points).
  - >-
    Keep the solar offset at the 5 Primaries that carry one, because on their daylight rows from
    April to September it lowers the score by 0.114 points (0.075 to 0.164), and by 0.273 points at
    leads up to 24 hours (S6 minus S5, rows gated as in experiment 64). The offset gate was chosen on
    this validation year, so confirm the gain on a second year.
  - >-
    Add the power-curve margin to the ensemble-trained wind model, because it lowers the wind score
    by 0.057 points (0.024 to 0.098) against BG2 (add-one arm D2).
  - >-
    Rank solar and wind forecasts on more than 6 and 3 farms before choosing between training on the
    ensemble's history and training on ERA5, because the intervals of both contrasts include the
    margin (verdict contrasts).
  - >-
    Recalibrate the solar forecast bands before relying on them, because the 0.05 and 0.95 quantiles
    cover 0.028 (0.022 to 0.035) and 0.973 (0.964 to 0.979) of the actuals (tail table).
code: ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/
---

# Experiment 87: BG2, the best guess so far and what each part contributes

**Take-home message: weekly refitting and a blend of five models lower the forecast error at
Primary substations and at bulk and grid supply points, while the solar and wind farm forecasts are
still too close to call.** A Primary substation steps 33 kV or 66 kV down to 11 kV. A bulk or grid
supply point is a substation where the transmission network feeds the distribution network. BG2 is
the second "best guess" recipe, assembled from the ideas that scored best in earlier experiments.
The score is the quantile score, which is twice the mean pinball loss over 11 quantiles, in points
of capacity; one point is 1% of a series' capacity, and a lower score is better. At the 16 Primary
substations BG2 scores 5.260, against 5.543 for the previous best, the stack: an XGBoost model
trained once, with a correction from the last 14 days of errors. At the 4 supply points BG2 scores
12.925, against 13.559 for the stack. The solar offset (the forecast output of the unmetered solar
panels behind the substation, experiments 64 and 78) lowers the score by 0.114 points (0.075 to
0.164) on the rows it can touch: daylight from April to September at the 5 Primaries that carry
an offset, 8.5% of the Primary rows. That is inside the prediction of 0.10 to 0.15. Averaged over
all rows of all 16 Primaries the gain dilutes to 0.010.
For the solar and wind farms, training on the archived
ECMWF ensemble forecasts scored 0.050 and 0.228 points worse than experiment 33's mixture trained
on ERA5 weather. With 6 solar and 3 wind farms the experiment cannot tell whether the loss is under
the 0.10 (solar) or 0.30 (wind) points we would accept. The verdict is `partly supported`.

We asked whether the ideas that scored best in earlier experiments still work when they are
combined in one forecast, and which idea gives which part of the gain. We refitted each XGBoost
model weekly or monthly, as its idea requires, forecast 292 daily runs from July 2025 to June 2026,
and scored each target class against its previous best forecast and against NGED's
`manual_heuristic`. Each drop-one ablation removes one idea at a time from the best guess and
scores the result again. Solar and wind farms are anonymised in every figure.

![Mean absolute error against forecast day](figure_1_headline.svg)

![The score after each added idea](figure_3_staircase.svg)

![How much each ablation worsens the score](figure_4_ablations.svg)

![Curtailment cost compared with NGED's method](figure_8_curtailment.svg)

The other figures are in this folder: the stages and ablations as lines (`figure_2_*`), error by
lead band, hour, month, season, and sunshine (`figure_5_*`), error by series
(`figure_6_by_series`), tail calibration (`figure_7*`), and the side panels of experiments 81 and
54 (`figure_9_side_panels`). In the figures and below, S1 to S6 are the stages of the Primary
staircase (frozen, stack, monthly, weekly, blend, and blend plus offset). A2 to A7 are drop-one
ablations of BG2: no blend, monthly in the weekly slot, the correction on every member, no spread
factor, capacity held at its value before the validation year, and three members. The spread factor
widens or narrows the quantiles about the median. Experiment 81 (R11) is a replication on other
forecast runs.

## Results

Scores are quantile scores in points of capacity on the 292 runs, with 95% intervals from
resampling whole months. Negative differences mean BG2 scores lower, which is better.

- **BG2 beats the stack at the 16 Primary substations by 0.283 points.** BG2 scores 5.260 (4.987 to
  5.534). The verdict contrast is -0.283 (98.75% interval -0.425 to -0.146), supported. Against
  `manual_heuristic` the difference is -0.920 (-1.078 to -0.767). The mean absolute error of the
  median is 7.432 (7.063 to 7.814), against 8.695 for `manual_heuristic`.
- **BG2 beats the stack at the 4 supply points by 0.634 points.** BG2 scores 12.925 (11.569 to
  14.177). The contrast is -0.634 (-1.111 to -0.279), supported. The stack is the frozen XGBoost
  model with the 14-day correction. Against `manual_heuristic` the difference is -1.578 (-1.951 to
  -1.251). The mean absolute error is 18.666 (16.484 to 20.682), against 20.876.
- **The solar farm forecast is 0.050 points worse than the ERA5-trained mixture, but the interval
  includes no change.** BG2 scores 5.012 (3.788 to 6.392) against 4.962; the contrast is +0.050
  (-0.125 to +0.168). The upper end exceeds the +0.10 margin, so non-inferiority is not shown.
  Against `manual_heuristic` the difference is -0.703 (-0.966 to -0.474). The mean absolute error
  is 7.444 (5.610 to 9.510), against 8.533.
- **The wind farm forecast is 0.228 points worse, with a wide interval.** BG2 scores 11.032 (9.641
  to 12.303) against 10.804; the contrast is +0.228 (-0.178 to +0.653), and non-inferiority within
  0.30 is not shown. Against `manual_heuristic` the difference is -2.871 (-3.568 to -2.234). The
  mean absolute error is 15.915 (13.745 to 17.895), against 19.607. The power-curve margin lowers
  the wind score by 0.057 (0.024 to 0.098) against BG2, and the fitted spread factor lowers it by
  0.016 (0.009 to 0.025).
- **Weekly refitting and the blend produce the Primary gain.** The staircase scores are: frozen
  6.927; the stack lowers the score by 1.384 (0.943 to 1.785); monthly refitting by 0.065 (-0.117
  to 0.244); weekly refitting by 0.123 (0.051 to 0.202); the blend by 0.086 (0.038 to 0.135); the
  offset by 0.010 (0.000 to 0.019), which is the all-row dilution of 0.114 (0.075 to 0.164) on the
  rows the offset can touch. The ablation that removes the blend from the final blend raises
  the score by 0.091 (0.048 to 0.138), which differs from the staircase step because the two
  remove the blend from different forecasts.
- **Other ablations.** At the supply points, adding the 14-day correction raises the score by 0.305
  (0.060 to 0.594) and moving from weekly to monthly refitting by 0.160 (0.066 to 0.247). At
  Primaries, monthly in the weekly slot raises it by 0.028 (0.003 to 0.058), the correction on every
  member by 0.067 (-0.039 to 0.160), no spread factor by 0.017 (0.003 to 0.033), and capacity held
  by 0.001 (0.000 to 0.002). The three-member blend is 0.002 below BG2 (-0.006 to 0.000), so the
  experiment found no difference between three and five members.
- **Solar and wind ablations.** Refitting monthly instead of freezing the ensemble-trained model
  lowers the score by 0.030 (-0.015 to 0.077) at solar farms and 0.098 (0.027 to 0.171) at wind
  farms. Swapping in the ERA5-trained mixture changes the score by -0.050 (-0.148 to +0.085) at
  solar farms and -0.228 (-0.560 to +0.095) at wind farms.
- **The noise floor is 0.001 points.** A repeat fit of the monthly Primary model at 4 Primaries moved
  the score by -0.001 (-0.003 to +0.001).
- **The check on the 73 runs of `fold_a_every5d_v1` is not independent.** The blend and the offset
  were chosen on those runs. BG2 minus the previous best is -0.285 (-0.402 to -0.171) at Primaries,
  -0.652 (-1.057 to -0.332) at supply points, +0.037 (-0.093 to +0.149) at solar farms, and
  +0.323 (-0.042 to +0.736) at wind farms.

| BG2 tail check | Quantile | Primary | Supply points | Solar farms | Wind farms |
|---|---|---|---|---|---|
| Share of actuals below the quantile (95% interval) | 0.01 | 0.007 (0.004 to 0.010) | 0.004 (0.002 to 0.005) | 0.007 (0.004 to 0.010) | 0.003 (0.001 to 0.006) |
| | 0.05 | 0.042 (0.033 to 0.051) | 0.039 (0.031 to 0.050) | 0.028 (0.022 to 0.035) | 0.038 (0.029 to 0.047) |
| | 0.95 | 0.937 (0.913 to 0.955) | 0.944 (0.911 to 0.965) | 0.973 (0.964 to 0.979) | 0.951 (0.935 to 0.966) |
| | 0.99 | 0.986 (0.978 to 0.993) | 0.995 (0.992 to 0.998) | 0.997 (0.995 to 0.999) | 0.987 (0.981 to 0.993) |
| Pinball loss | 0.01 / 0.05 | 0.280 / 0.901 | 0.731 / 2.389 | 0.180 / 0.782 | 0.270 / 1.267 |
| | 0.95 / 0.99 | 1.206 / 0.414 | 2.403 / 0.594 | 0.934 / 0.547 | 2.709 / 0.698 |
| Threshold-weighted CRPS | above p99 / above p95 / below p05 | 0.119 / 0.779 / 0.251 | 0.276 / 0.967 / 0.666 | 0.041 / 0.510 / 0.008 | 0.019 / 0.361 / 0.004 |

- **The Primary tails match their nominal levels, and the other classes' tails are mostly too
  wide.** At Primaries every interval includes its nominal level. At supply points the 0.01 and
  0.99 quantiles cover 0.004 and 0.995 of the actuals, and at wind farms the 0.01 quantile covers
  0.003, so those bands are slightly wide. At solar farms the 0.05, 0.95, and 0.99 quantiles cover
  0.028, 0.973, and 0.997, so the bands are too wide.
- **Curtailment cost is a calibration to history, not a saving.** Curtailment cost is the cost in
  GBP per MW of generator capacity per day under node limits set at 0.9 times the highest export
  seen before the scored year. The 3 scored supply points saw 19 export half-hours beyond that
  limit out of 40,714 scored half-hours, too few to rank forecasts. A fourth supply point (Walpole)
  is excluded because the sign of its flow is unknown. NGED's method is the maintainer's unconfirmed
  understanding of it: every generator assumed to run at capacity. The assumed risk level is a
  breach ratio r of 20 (planning to the 0.95 quantile).
    - **Headline, f = 0.9 and r = 20:** BG2 minus NGED's method is -32.85 GBP per MW per day (-99.24
      to +0.00), or -11,989 GBP per MW per year pro-rated from 292 days, not statistically
      significant at the 5% level. The previous best is +6.64 and the perfect forecast -57.39.
    - **Risk level:** r = 4, 10, and 100 give -0.81, -13.87, and -166.89 GBP per MW per day.
    - **Limit:** f = 0.75 gives -87.03 (-298.28 to +34.62) with 118 events, and f = 0.5 gives -418.99
      (-1242.54 to +129.46) with 853 events. The published thermal limit has 1 event and is a
      reference only.
    - **Primaries (node only):** 0 events at f = 0.9 and 0.75, and 12 at f = 0.5, among 104,141
      half-hours of the 6 scored series.

## Deviations, caveats, and next steps

- **The generator rows refit monthly.** The planned weekly refits of each of the 9 farm models did
  not fit the 16 GPU-hour budget, which the manager raised to 20. The arms `g_ens_weekly` and
  `g_a_weekly_to_frozen` in the stored predictions are monthly refits. The hypothesis above
  described weekly refits at the farms. The noise-floor repeat used the monthly schedule too.
  Total GPU use was 14.4 hours of wall time.
- **Schedule for the rest:** the full production recipe, with 500 rounds, a learning rate of 0.05,
  and a half-life of 0.5 years. Primary and supply-point models refit weekly.
- **The offset follows experiment 74's mechanism, not the design's.** The weights come from
  experiment 64's fit before the validation year, where the design says to refit them. The capacity
  per run comes from experiment 78's vintages, and the offset models start from the median of the
  training labels plus the offset. Ablation A6 holds the capacity at experiment 64's value, which
  was estimated before July 2025. The offset lowers the score by 0.114 on its daylight
  April-to-September rows at the 5 offset Primaries (0.010 over all rows of all 16 Primaries).
  An audit found no flaw that dilutes or
  cancels the effect: the base margin, blend weights, gate, and member twins are consistent between
  training and prediction. Using the held capacity at prediction instead of the vintage capacity
  gives 0.105 (ablation A6), so the vintage capacity helps slightly. The spread factor is a fixed
  1.15 around the median and the 14-day correction uses the stack member's own errors, so neither
  carries a replica-year bias into the offset.
- **The pooled member** refits monthly on the full history with 500 rounds, not under P-roll (the
  rolling-window schedule of experiment 49).
- **References:** `climatology` and `manual_heuristic` are recomputed at 11 levels. The
  production-style row is the stored `xgboost_baseline` from the local `data/power_forecasts`
  table. Whether the local Dagster cross-validation assets wrote it, and not S3, was not confirmed.
  Its band is too narrow, so its tail scores are noise.
- **Ablation A3** replaces the weekly member with the monthly one, so the blend holds a duplicate.
  **Ablation A7** (three members) is exploratory and not in the best guess.
- **Selection bias:** the spread factor 1.15, the blend scheme, the offset gate, and the half-life
  were chosen or scored on this validation year, so the Primary and supply-point gains are
  optimistic.
- **Not run:** corrected lags, analogues on top of the offset, pooling by type on cleaned targets,
  and any UKV, ICON-EU, or IFS blend. The side panels of figure 9 repeat numbers from experiments
  81 and 54, labelled control-only and "ceiling, not available live".
- **Next:** a second validation year for the offset, and more farms or a second year to rank the
  farm models. The solar bands need recalibration, because the tail check
  shows them too wide; the other classes' tails need no tail-specific spread factor on this
  evidence.
