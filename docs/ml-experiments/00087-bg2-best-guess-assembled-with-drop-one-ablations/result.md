---
id: "00087"
slug: bg2-best-guess-assembled-with-drop-one-ablations
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
  existing scheduled jobs and one fixed weight vector, and the offset needs one capacity table).
  Solar and wind are supported when the interval's upper end lies below +0.10 (solar) and +0.30
  (wind), because the swap to the ensemble's own history is worth making whenever it costs less than
  that. Every ablation, every staircase stage, and every comparison with `manual_heuristic` is read
  at 95% and decides no verdict.
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
  Weekly refitting and the blend carry the Primary gain: refitting weekly instead of monthly gains
  0.123 points and the blend gains another 0.086. The solar offset adds only 0.010 points over the
  whole year because it acts on 5 of 16 substations in daylight from April to September. At the
  supply points the 14-day correction hurts. The generator models cannot yet be ranked.
production_status: candidate
adopted_in: none
arms: >-
  Primary (quantile score over 11 levels, 292 runs): S1 frozen 6.927; S2 stack 5.543; S3 monthly
  5.478; S4 weekly 5.355; S5 blend without offset 5.270; S6 BG2 5.260; A2 no blend 5.351; A3
  monthly in the weekly slot 5.288; A4 correction on every member 5.327; A5 spread factor 1.0
  5.277; A6 capacity held 5.261; A7 three members 5.258; `manual_heuristic` 6.182; `climatology`
  7.368; production-style `xgboost_baseline` 10.536. Supply points: frozen 13.369; stack 13.559;
  monthly 13.085; weekly (BG2) 12.925; weekly plus correction 13.230. Solar farms: ERA5-trained
  mixture 4.962; ensemble-trained frozen 5.042; weekly (BG2) 5.012. Wind farms: ERA5-trained mixture
  10.804; ensemble-trained frozen 11.131; weekly (BG2) 11.032; plus power-curve margin 10.975.
decisions: >-
  D1 ensemble-trained generators, run as stated. D2 wind margin as an add-one arm only. D3 curtailment
  labelled "calibration to history, not a saving". D4 production-style row read from the local
  `data/power_forecasts` table. D5 offset at 5 of 16 Primaries. D6 R11 shown only as a labelled side
  panel. D7 twCRPS copied into the experiment's own folder.
advice:
  - >-
    Refit Primary and bulk and grid supply point models weekly rather than monthly, because weekly
    scored 0.123 points (0.051 to 0.202) below monthly at Primaries and 0.160 (0.070 to 0.250) below
    at supply points (staircase steps).
  - >-
    Blend five models (weekly, monthly, analogue, pooled by type, and the stack) rather than serve
    one, because dropping the blend costs 0.091 points (0.048 to 0.138) (ablation A2).
  - >-
    Do not add the 14-day correction to the bulk and grid supply point model, because adding it
    back costs 0.305 points (0.060 to 0.594) (ablation, supply points).
  - >-
    Judge the solar offset on daylight rows from April to September at the 5 offset substations,
    because over all rows of all 16 Primaries it adds only 0.010 points (-0.019 to 0.000) (S6 minus
    S5).
  - >-
    Rank solar and wind forecasts on more than 6 and 3 farms before choosing between training on the
    ensemble's history and training on ERA5, because the intervals of both contrasts include the
    margin (verdict contrasts).
  - >-
    Widen the upper tail of the Primary forecast before calibrating, because coverage at the 0.95
    quantile is 0.937 (0.913 to 0.955) and at 0.05 is 0.042 (0.033 to 0.051) (tail table).
code: ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/
---

# Experiment 87: BG2, the best guess so far and what each part contributes

**Take-home message: weekly refitting and a blend of five models make the Primary-substation and
supply-point forecasts clearly better than the previous best, while the farm forecasts are still too
close to call.** The previous best was a model trained once, with a 14-day correction. At the 16
Primary substations the best guess (BG2) scores 5.26 points of capacity against 5.54, and at the 4
bulk and grid supply points 12.93 against 13.56. One point is 1% of a series' capacity, and a lower
score is better. The solar offset, which was the headline of earlier experiments, adds only 0.01
points over the whole year. For the solar and wind farms, training on the ensemble forecast's own
history scored slightly worse than the ERA5-trained mixture, and nine farms cannot show whether the
loss is smaller than 0.10 (solar) or 0.30 (wind) points. The verdict is `partly supported`.

We asked whether the best ideas of the last experiments still work when combined in one forecast,
and which idea each part of the gain comes from. We refitted every model on the schedule the idea
needs, forecast 292 daily runs from July 2025 to June 2026, and scored each target class (Primary
substations, bulk and grid supply points, solar farms, wind farms) against its previous best forecast
and against NGED's `manual_heuristic`. Each drop-one ablation removes one idea from the
best guess. The solar farms and wind farms are anonymised in every figure.

![Mean absolute error against forecast day](figure_1_headline.svg)

![The score after each added idea](figure_3_staircase.svg)

![What each ablation costs](figure_4_ablations.svg)

![Curtailment cost compared with NGED's method](figure_8_curtailment.svg)

The other figures are in this folder: the stages and ablations as lines (`figure_2_*`), error by lead
band, hour, month, season and sunshine (`figure_5_*`), error by series (`figure_6_by_series`), tail
calibration (`figure_7*`), and the side panels of experiments 81 and 54 (`figure_9_side_panels`).

## Results

- **Primary (16 substations):** BG2 5.260 (95% interval 4.987 to 5.534). Verdict contrast against the
  stack: -0.283 (98.75% interval -0.425 to -0.146), supported. Against `manual_heuristic`: -0.920
  (-1.078 to -0.767).
- **Supply points (4):** BG2 12.925 (11.569 to 14.177). Verdict contrast against the frozen model with
  the correction: -0.634 (-1.111 to -0.279), supported.
- **Solar farms (6):** BG2 5.012 against 4.962 for the ERA5-trained mixture, contrast +0.050 (-0.125
  to +0.168). Non-inferiority within 0.10 is not shown.
- **Wind farms (3):** BG2 11.032 against 10.804, contrast +0.228 (-0.178 to +0.653). Non-inferiority
  within 0.30 is not shown. The power-curve margin scores 10.975 (an add-one arm).
- **Largest ablation per class:** Primary, no blend, +0.091 (0.048 to 0.138). Supply points,
  correction added back, +0.305 (0.060 to 0.594). Solar, weekly to frozen, +0.030 (-0.015 to +0.077).
  Wind, weekly to frozen, +0.098 (0.027 to 0.171).
- **Staircase, Primary:** frozen 6.927; stack -1.384; monthly -0.065; weekly -0.123; blend -0.086;
  offset -0.010 (-0.019 to 0.000).
- **Noise floor:** a repeat fit of the monthly model at 4 Primaries moved the score by -0.001 (-0.003
  to +0.001).
- **Check on the 73 runs of `fold_a_every5d_v1`:** BG2 minus the stack is -0.285 (-0.402 to -0.171),
  the same as on the 292 runs.

| Tail check, BG2 | Quantile | Primary | Supply points | Solar | Wind |
|---|---|---|---|---|---|
| Share of actuals below the quantile | 0.01 | 0.007 | 0.004 | 0.007 | 0.003 |
| | 0.05 | 0.042 | 0.039 | 0.028 | 0.038 |
| | 0.95 | 0.937 | 0.944 | 0.973 | 0.951 |
| | 0.99 | 0.986 | 0.995 | 0.997 | 0.987 |

- **Curtailment (calibration to history, not a saving):** with the limit at 0.9 times the past
  maximum export, 19 of 40,714 scored half-hours at the 3 scored supply points were beyond the limit,
  so the count is too small to rank forecasts. BG2 minus NGED's method (the maintainer's unconfirmed
  understanding of it) is -32.85 GBP per MW per day (-99.24 to +0.00), not statistically significant
  at the 5% level. At 0.75 there are 118 events, -87.03 (-298.28 to +34.62); at 0.5 there are 853,
  -418.99 (-1242.54 to +129.46). The published thermal limit holds 1 event and is a reference only.

## Deviations, caveats, and next steps

- **Schedule:** full production recipe (500 rounds, learning rate 0.05, half-life 0.5 years), weekly
  refits for the best guess. The generator models refit monthly, not weekly, because the 16 GPU-hour
  budget (raised to 20) did not hold 52 weekly refits of each of the 9 farm models. The monthly refit is what the generator
  rows show. The noise-floor repeat also used the monthly schedule. GPU use was 14.4 hours of wall
  time.
- **Offset:** weights come from experiment 64's fit before the validation year (the design says to
  refit them), the capacity per run comes from experiment 78's vintages, and offset models start
  from the median of the training labels plus the offset. Ablation A6 holds the capacity at its
  pre-validation value.
- **Pooled member:** it refits monthly on the full history with 500 rounds, not under P-roll.
- **References:** `climatology` and `manual_heuristic` are recomputed at 11 levels. The
  production-style row is the stored `xgboost_baseline` from the local `data/power_forecasts` table;
  its band is far too narrow, so its tails are noise.
- **Ablation A3** replaces the weekly member with the monthly one, so the blend holds a duplicate.
  **Ablation A7** (three members) scores 0.002 below BG2, so two members are not needed; it is
  exploratory and not in the best guess.
- **Selection bias:** the spread factor 1.15, the blend scheme, the offset gate, and the half-life
  were chosen or scored on this validation year.
- **Not run:** corrected lags, analogues on top of the offset, pooling by type on cleaned targets,
  and any UKV, ICON-EU or IFS blend. The side panels of figure 9 repeat numbers from experiments 81
  and 54 and are labelled control-only and "ceiling, not available live".
- **Next:** a tail-specific spread factor, since the upper-tail coverage at 0.95 is 0.937; weekly
  refits at the farms with more farms or a second year; and a second validation year for the offset.
