---
id: '00289'
slug: summer-gap-weather-input-discriminator
title: >-
  Does the ensemble sunshine forecast cause the April to June gap between the best-guess stack and
  BG2?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
  - solar-disaggregation
status: done
size: >-
  medium. Stored: private files under
  data/ml-experiments/00289-summer-gap-weather-input-discriminator/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: two are defensible (a zero-fit re-forecast with realised
  sunshine, or a refit with realised sunshine as the driver); this plan pre-registers the zero-fit
  one and names the refit as the fallback if the result is undecided. Callers: this folder's
  scripts, which call reviewed `studies.*` modules and copy experiment 111's and 136's forecasting
  code without changing it. No refits, no GPU, no downloads.
hypothesis: >-
  The best-guess stack's weaker April to June score than BG2 at the decision Primaries comes from
  the ensemble's sunshine forecast (or its ratio calibration), and not from demand absorbing the
  solar term or from the error-correction layers.
prediction: >-
  Written before any forecast is re-made. (P0, gate) Re-forecasting the base engine with the stored
  ensemble weather reproduces its stored mean to within 1e-6 points at every row. (P1) On daylight
  half-hours in April to June at leads of 0 to 36 hours at D5, D9, D10, and D16, arm D's
  quantile-score gap to BG2 shrinks by at least half when the ensemble's sunshine is replaced by
  realised (CAMS) sunshine. (P2) The mean absolute error of the mean forecast shrinks by at least a
  third of its gap to BG2 on the same rows. (P0b) The irradiance that the engine builds from CAMS
  agrees with the raw CAMS hourly values to within 1 W/m² on average, with no half-hour shift in the
  hour-of-day profile. If P1 fails, the gap stays at least half as large and the cause is demand
  absorbing the solar term or the error-correction layers.
decision_rule: >-
  The gap is arm D minus BG2 in quantile score on April to June daylight half-hours at leads of 0 to
  36 hours, at D5, D9, D10, and D16, averaged over the four Primaries so that no one Primary
  decides. C2 (primary) is arm D with ensemble sunshine minus arm D with realised sunshine in mean
  absolute error of the mean forecast, on the same rows; C1 (secondary) is the same contrast in
  quantile score. Intervals resample whole ISO weeks of forecast initialisation time (about 13
  weeks) at the 97.5% level (Bonferroni over two contrasts). The ensemble sunshine is implicated if
  C2 is at least a third of arm D's mean absolute error gap to BG2 and its interval excludes zero,
  and C1 is at least half of the quantile-score gap. It is cleared if C2's upper bound is below a
  third of that gap. Anything else is undecided and the page names the refit as the next step. If C1
  and C2 disagree the page reports "undecided" and says which one moved. C1 holds the stored
  quantile widths fixed, so it cannot show that realised sunshine would narrow the spread; a cleared
  verdict says only that the level of the mean is not the cause. Realised sunshine is a ceiling, so
  every gain is an upper bound for a better forecast. The threshold is a fraction of the gap,
  because a better sunshine input would add a data source, so zero is not the right threshold. A
  negative control repeats the swap in July to September, where the stack beats BG2: the swap should
  not help there by more than it helps in April to June.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228, 251, and 266 (six
  substations, 2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside
  hand-filtered load shifts).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  and mean absolute error of the mean forecast, in points.
code: ml-experiments/00289-summer-gap-weather-input-discriminator/
verdict: partly supported
best_arm: >-
  Not applicable: a diagnostic. Arm D with realised sunshine scores 3.387 points against 3.790 with
  ensemble sunshine and 3.223 for BG2 (April to June, daylight, 0 to 36 hours ahead).
headline: >-
  Realised sunshine closes 71% of arm D's April to June gap to BG2, but it lowers arm D's error by
  almost as much in July to September, when there is no gap, so only about 14% of the gap particular
  to April to June is explained by the sunshine input.
lessons:
  - >-
    A better sunshine forecast would improve arm D in every season (+0.327 points in July to
    September), so a gain in April to June is not evidence that the sunshine input causes the gap.
  - >-
    Gap minus July-to-September gap: +0.541 points with ensemble sunshine and +0.466 with realised
    sunshine, so most of the April to June excess stays (post hoc).
  - >-
    At D5, the Primary with the largest gap (+1.437 points), realised sunshine gains only +0.248
    points (interval -0.145 to +0.675).
production_status: not-for-production
adopted_in: none
date: 2026-10-23
take_home_message: >-
  Giving the best-guess stack the sunshine that actually happened would close about 70% of its April
  to June gap to BG2, but the same change helps the stack nearly as much in July to September, when
  it already matches BG2. Most of the gap particular to April to June therefore stays, which points
  at the way demand absorbs the solar term, or at the error-correction layers, and not at the
  ensemble sunshine forecast. A refit with realised sunshine as the solar driver, and a check of D5
  and D9's solar term in April to June, are the next steps.
advice:
  - >-
    Do not read a gain from realised sunshine as the cause of a gap to BG2 without the same swap in
    a season with no gap (figure 3).
  - >-
    Look for the April to June excess in the solar term at D5 and D9, and in the error-correction
    layers, rather than in the ensemble sunshine input (figures 3 and 4).
  - >-
    Treat any gain from realised sunshine at leads over 72 hours as a ceiling no forecast can reach
    (figure 5).
---

**If the best-guess stack were given the sunshine that actually happened, its April to June gap to
BG2 would shrink by about 70%. The same change helps almost as much in July to September, so the
sunshine input explains little of what is particular to April to June.** BG2 is the production-style
XGBoost forecast. The stack's base engine (arm D) predicts demand from an ensemble weather forecast.
We re-ran that engine with realised sunshine (the CAMS satellite irradiance estimate) in place of
the ensemble's sunshine at the forecast hours, and measured how much of arm D's gap to BG2
disappears. Nothing was refitted. Realised sunshine is not available at forecast time, so the result
is a ceiling for a better sunshine input.

Scores are quantile scores in points of effective capacity (a substation's rated demand), and lower
is better. "Arm D minus BG2" is positive where arm D is worse. [Experiment
266](../00266-summer-gap-diagnostics-solar-term/result.md) located the gap at D5, D9, and D10 and
cleared the ensemble sunshine only by an agreement test; this experiment tests it directly. The
method of adding a per-row shift to arm D's quantiles follows [experiment
251](../00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md)'s clip arm.

![Figure 1. Scores of BG2 and arm D under three sunshine inputs](figure_1_scores.svg)

![Figure 2. Gap to BG2 before and after realised sunshine](figure_2_gap_closure.svg)

![Figure 3. April to June gap beyond July to September's (post hoc)](figure_3_seasonal_excess.svg)

![Figure 4. Gap and gain by substation](figure_4_by_site.svg)

![Figure 5. Gap and gain by lead band](figure_5_by_lead.svg)

## Results

- **Verdict `partly supported`.** The pre-registered rule says the ensemble sunshine is implicated:
  the gain from realised sunshine is 71% of the quantile-score gap (P1 held) and 70% of the
  mean-absolute-error gap (P2 held), both with intervals above zero. The negative control, which the
  plan added after review, weakens that reading, so the verdict stops short of `supported`.
- **Headline (April to June, daylight, 0 to 36 hours ahead, D5, D9, D10, and D16 averaged, 97.5%
  intervals over 14 ISO weeks).** BG2 scores 3.223 (2.954 to 3.523) and arm D with ensemble sunshine
  3.790 (3.484 to 4.077), a gap of +0.567 (+0.384 to +0.733). Arm D with realised sunshine scores
  3.387 (3.186 to 3.583), a gap of +0.164 (-0.116 to +0.378).
- **C1, quantile score: ensemble minus realised sunshine is +0.403 (+0.154 to +0.680).** C2, mean
  absolute error of the median forecast: +0.715 (+0.260 to +1.217), against a gap of +1.021 (+0.721
  to +1.285) points of effective capacity, which is 70%. The median and the mean forecast of arm D
  are identical, so the median stands for the mean.
- **Negative control (July to September, same rows).** The gap to BG2 is +0.026 (-0.099 to +0.152),
  and realised sunshine still lowers arm D's score by +0.327 (+0.176 to +0.470), 81% of its April to
  June gain. With realised sunshine arm D beats BG2 by 0.301 points there (-0.476 to -0.123).
- **Post hoc: the gap particular to April to June.** The April to June gap minus the July to
  September gap is +0.541 (+0.317 to +0.758) with ensemble sunshine and +0.466 (+0.147 to +0.763)
  with realised sunshine. The swap removes +0.076 of it (-0.149 to +0.324), 14%; the same reading on
  mean absolute error gives 15%.
- **Calibration versus input.** Setting the irradiance ratio and the solar shrink to 1 on the
  ensemble raises the score by 0.066 (interval -0.010 to +0.137), so the calibration helps slightly;
  the whole gain comes from the realised input.
- **By substation (0 to 36 hours, 95% intervals).** The gap and the gain are D5 +1.437 and +0.248
  (-0.145 to +0.675), D9 +0.464 and +0.285 (+0.038 to +0.515), D10 +0.143 and +0.527, D16 +0.225 and
  +0.551, D11 +0.066 and +0.258, and D14 +0.222 and +0.511. Realised sunshine helps least at D5,
  where the gap is largest.
- **By lead band (decision Primaries).** The gap and the gain are +0.453 and +0.309 at 0 to 24
  hours, +0.565 and +0.530 at 24 to 72 hours, and +0.603 and +0.686 at 72 to 168 hours. The gain
  grows with lead and exceeds the gap at long leads, so it is a ceiling there.
- **Tail calibration (April to June, daylight, 0 to 36 hours, decision Primaries).** Coverage and
  pinball loss, BG2 then arm D with ensemble sunshine then arm D with realised sunshine, are 0.015,
  0.019, and 0.012 at the 0.01 quantile (pinball 0.203, 0.263, 0.236); 0.052, 0.075, and 0.049 at
  0.05 (0.676, 0.924, 0.808); 0.976, 0.922, and 0.932 at 0.95 (0.985, 1.031, 0.938); and 0.993,
  0.969, and 0.977 at 0.99 (0.340, 0.324, 0.289). Arm D's upper tail is under-covered under both
  inputs. The curtailment-cost metric was not run, because it needs the node and generator forecasts
  that this swap does not produce.
- **Data and schedule.** The 561,207 rows of experiment 251, with experiment 136's monthly fits (no
  refit) and weekly-block intervals; D11 and D14 are reported beside the decision Primaries. The
  ensemble temperature is unchanged in every arm. CAMS sunshine was missing at no scored half-hour.
- **What this cannot tell.** The shift keeps arm D's stored quantile widths, which came from
  ensemble residuals, so the quantile score understates what realised sunshine would do to the
  spread. The monthly fits were trained on calibrated ensemble sunshine, so realised sunshine is
  slightly out of distribution, and the offset tables and recipe R of arm D are not re-simulated.
  One year gives 14 ISO weeks per season.
- **What the next experiment should answer.** Refit arm D with realised sunshine as the solar driver
  in April to June, to see whether the excess is in the solar term itself, and look at D5's and D9's
  solar term in those months.

## Checks

- **Gates.** P0 held: the engine with ensemble weather reproduces experiment 136's stored mean
  exactly (largest difference 0.0 at all six substations). Arm D minus BG2 in April to June over all
  rows and six substations is +0.322 (+0.199 to +0.438), the +0.319 of experiment 266 within 0.005.
  P0b held on the mean: the engine's irradiance from CAMS differs from the cache's by -0.002 W/m² on
  average over April to June 2026 at D5, with half-hour differences up to 24 W/m² (mean irradiance
  221 W/m²) from re-spreading each hour by the sun's height.
- **Median minus mean forecast.** 0.0 at all six substations, because arm D's quantiles are
  symmetric about the mean.
- **Error of the median beside the quantile score.** The mean absolute error contrast (+0.715) and
  the quantile-score contrast (+0.403) agree in sign in every subset reported above.
- **Triggers.** Trigger 5 fired: the July to September control, which should change nothing, moved
  the score by +0.327, more than its half-width of 0.15. The page reads this as the finding that
  realised sunshine improves every season. The audit's findings are triaged in the pull request.
  Triggers 1 to 4 and 6 did not fire. Trigger 7 applies if a later design depends on this reading.

## Plan

- **Weather swap.** Arm D's base engine is experiment 136's A2 (monthly fits, ensemble-driven, with
  the Kalman filter). Re-forecast it with the ensemble's shortwave replaced by realised CAMS
  irradiance at the forecast's target half-hours only (the analysis proxy and the filter's past
  residuals stay as stored), with the irradiance ratio set to 1 and the solar shrink to 1. The
  monthly fits were trained on calibrated ensemble irradiance, so the realised input is slightly out
  of distribution; a second arm with ensemble irradiance and the same identity calibration
  separates the calibration effect from the input effect. Temperature stays the ensemble's, so only
  sunshine changes. Nothing is refitted.
- **Arm D with realised sunshine.** Add the per-row change in the base engine's mean to every
  quantile of stored arm D, the method experiment 251 used for its clip arm. The offset tables,
  winter night offset, and recipe R of arm D are not re-simulated; this is a caveat.
- **Rows.** The six stored Primaries, April to June daylight half-hours (clear-sky irradiance above
  20 W/m²); the decision Primaries D5, D9, D10, and D16 are the pooled headline, and D11 and D14
  are reported beside them.
- **Gate (P0).** The re-forecast with ensemble weather must reproduce experiment 136's stored A2
  mean.
- **Stratification.** Results are reported by lead band (0 to 24, 24 to 72, 72 to 168 hours) and
  per substation; the decision uses 0 to 36 hours. A lead of 72 hours or more is an upper bound
  only, since no forecast has that sunshine.
- **Reported:** absolute quantile score and mean absolute error for BG2, arm D with ensemble
  sunshine, arm D with realised sunshine, and the base engine alone; tail coverage and pinball loss
  at 0.01, 0.05, 0.95, and 0.99; per-substation results.
- **Caveat.** CAMS sunshine exists only after the fact, so arm D with realised sunshine is a
  ceiling for an improved sunshine input, not a deployable forecast.
- **Checks.** Median minus mean forecast per site, the mean-forecast error beside the quantile
  score, and the audit triggers, recorded in the write-up.
- **Earlier advice followed.** Judge any change on April to June at D5, D9, and D10 as well as the
  whole year (experiment 266).
