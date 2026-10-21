---
id: '00135'
slug: differentiable-physics-n10-joint-kalman-and-demand-fit
part: N10
theme: differentiable-physics-long-history-and-drift
also_relevant_to:
- differentiable-physics-forecast-engine
- lagged-power-and-error-correction
status: done
size: 'complex. Stored: private fits, predictions, and scores under data/ml-experiments/00135-differentiable-physics-n10-joint-kalman-and-demand-fit/
  only. Serving path: not touched (the change is in how the convex model is fitted, which is training-loop
  code). Degradation rule: not touched. Designs: more than one is defensible (what the loss is, whether
  the filter is trained or held fixed, how the shapes are tied to the two-stage fit), and this plan
  fixes one of each. Callers: this folder''s scripts only, which call the reviewed `studies.dp_gam`,
  `studies.dp_assimilation`, `studies.dp_filter_grid`, and `studies.dp_ens` modules without changing
  them.'
hypothesis: Fitting the convex demand model (GAM-K) so that its coefficients minimise the day-ahead
  error after the Kalman correction, instead of the plain error before it, lowers the day-ahead quantile
  score at 12 to 36 hours below the two-stage fit (fit the demand model, then filter its residuals),
  without moving the fitted shapes to explain recent-level structure that the filter already tracks.
prediction: C1 (J1 minus TS, quantile score at 12 to 36 hours, scored year, five decision substations)
  is -0.05 to +0.02 points, most likely about zero. C2 (J2 minus TS) is -0.03 to +0.01 points. J1's
  fitted shapes move to explain the injected level change and PV-shaped change by more than TS's do
  (the absorption control fails for J1), and J2's do not. The fitted shapes of J1 differ from TS's by
  a root mean square of 5% to 15% of each term's own root mean square, and J2's by under 5%.
decision_rule: 'Two planned contrasts (C1 J1 minus TS, C2 J2 minus TS), each read as a two-sided percentile
  interval of the month-resampled paired bootstrap (12 calendar months of the scored year, 20,000 resamples)
  at the 1.25% and 98.75% percentiles, the Bonferroni level for two contrasts at 5%. The verdicts apply
  in this order. (1) The verdict is capped at `inconclusive` if a blocking control fails: P1 (reproduction),
  P2 (the linear filter equals the filter), or P3 (the solved loss equals the loss of the filter run
  on the fitted residuals). (2) A joint arm that lowers the score and fails the absorption control is
  reported as an identification failure (`not supported`), whatever its score. (3) A joint arm is `supported`
  when its contrast interval lies wholly below zero, at least 4 of the 5 decision substations improve
  by at least 0.005 points in both the scored year and the replicate months, the score over all leads
  0.5 to 168 hours has an interval upper bound of at most +0.02, and it passes the absorption control.
  (4) It is `small gain` when the interval lies wholly below zero but a condition of (3) fails. (5)
  It is `refuted` when the interval lies wholly above zero. (6) Otherwise it is `inconclusive`, which
  is the expected outcome for a true effect of a few hundredths of a point, because 12 months pooled
  over 5 substations give wide intervals, and is not evidence of no effect. The threshold is zero, because
  a joint fit replaces the fitting routine inside the training loop and adds no model, data source,
  or serving step. The shape distance is reported, not binding: it feeds the substation cards and the
  discussion of whether to port. The experiment tests only whether the coefficients should be fitted
  through a filter with fixed noise settings, so a null result does not rule out learning the noise
  settings and the coefficients together.'
benchmark: TS, the two-stage GAM-K with the learned-noise filter settings of experiment 133 (its GRID
  arm); GAM-K with experiment 111's fixed settings (FIX) as reference; climatology and manual_heuristic
  as in experiment 111 for reference
row_set_id: fold_a_daily_gap_v1
metric: Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points
  of effective capacity, on rows outside the hand-filtered switching events, at the five decision substations
  D5, D10, D11, D14, and D16 (D9 beside them), at leads 12 to 36 hours (D+1) with every lead beside
  it, then winter (November to March), winter nights, cold nights, and the Christmas and bank-holiday
  periods broken out; coverage and pinball loss of the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00135-differentiable-physics-n10-joint-kalman-and-demand-fit/
verdict: not supported
best_arm: TS, the two-stage fit, scored 2.488 points at leads 12 to 36 hours in the scored year (J2
  2.484, J1 7.907). The tied joint fit J2 minus TS is -0.0045 (97.5% interval -0.013 to +0.003).
headline: Fitting the demand model through its Kalman filter with no tie to the two-stage fit (J1) raised
  the next-day quantile score by 5.4 points of capacity at leads 12 to 36 hours (97.5% interval +3.3
  to +8.5), and fitting it with a ridge penalty towards the two-stage coefficients (J2) changed the
  score by -0.0045 points of capacity (-0.013 to +0.003).
lessons: The filter's level absorbs a constant shift in the slow terms of the demand model. The joint
  loss then leaves those terms to the penalty alone, and the untied joint fit's weekly profile drifts
  by tens of points of capacity. A tie to the two-stage coefficients removes that freedom, and the tuning
  months then prefer the strongest tie on the grid, which returns the two-stage fit. The shape distance
  and the known-answer control support this explanation, and no variant of the loss tested it.
take_home_message: 'We fitted the demand model so that its coefficients minimise the error left after
  a Kalman filter (a running correction of recent errors) has adjusted the forecast. That made the day-ahead
  forecast much worse, because the filter can absorb a shift in the slow parts of the demand model,
  so the fit barely constrains those parts. Tying the fit to the two-stage coefficients removed the
  damage and also removed any gain: the quantile score was within 0.013 points of the two-stage score,
  in either direction. The two-stage design (fit the demand model, then filter its residuals) stays.'
advice:
- 'Do not fit the convex demand model through a filter that tracks a recent level without a tie to the
  two-stage coefficients: the untied fit scored 5.4 points of capacity worse at 12 to 36 hours (C1,
  figure 1).'
- 'Keep the two-stage fit (fit GAM-K, then filter its residuals) for the day-ahead forecast: the tied
  joint fit changed the quantile score by -0.0045 points of capacity (-0.013 to +0.003), inside the
  noise (C2, figure 1).'
- 'When a loss passes a demand model through a filter that tracks a level, count which terms the filter
  can absorb and tie those terms to a reference fit: the untied fit''s weekly profile moved by a median
  of 19 points of capacity (up to 95) when the training cut-off moved back by 7 days, against 1 point
  for the two-stage fit (Checks, absorption control).'
- 'Run the known-answer control on any new joint fit before the real data: on a simulated stream generated
  from the two-stage curves plus a level shift, the untied fit recovered the curves with 3.3 times the
  error of the two-stage fit (P4).'
- 'Widen the grid of tie strengths beyond 3 before reading the tied fit as a null: the tuning months
  preferred the strongest tie at five of six substations, so the best tie may lie past the grid (Results,
  third bullet).'
production_status: not-for-production
adopted_in: none
---

# Experiment 135: does fitting the demand model through the Kalman filter beat fitting it first and filtering after?

**Fitting the demand model so that it is judged after the Kalman filter's correction made the
day-ahead forecast much worse, and tying it to the ordinary fit removed the damage and added no
gain.**
The quantile score at leads 12 to 36 hours (our error score for a forecast with uncertainty bands,
in points of capacity; smaller is better) rose by 5.4 points without the tie (arm J1; 97.5% interval
+3.3 to +8.5). With a ridge penalty that ties the fit to the two-stage coefficients (arm J2), the
score changed by -0.0045 points (-0.013 to +0.003), which cannot be told from zero. The verdict is
`not supported`. The two-stage design stays: fit the demand model, then filter its residuals.

We asked whether the convex demand model forecasts the next day better when its coefficients are
chosen to minimise the error that remains after the Kalman filter has corrected the forecast. The
demand model is GAM-K, the convex additive model with a Kalman filter on its residuals from
[experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md). The
filter tracks a slowly moving level and a quickly decaying transient of the demand model's error in
each of the 4 hour bands of the day, with the noise settings that [experiment
133](../00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/result.md)
chose. In the two-stage design, the demand model is fitted first and the filter then runs on its
residuals. For fixed noise settings the filter is linear in the residuals, so the joint fit is an
exact bounded quadratic problem. We solved it on a graphics processing unit (GPU) with no gradient
descent. The idea follows [de Vilmarest and Goude (2022)](https://arxiv.org/abs/2110.00334), who run
a state-space correction on the residuals of a generalised additive model for electricity load. They
correct the residuals after fitting, and this experiment fits through the correction.

The experiment compared three arms on `fold_a_daily_gap_v1` at the five decision substations (D5,
D10, D11, D14, and D16, with D9 beside them):

- **TS:** the two-stage fit, as in experiment 111, with experiment 133's filter settings (its GRID
  arm, which chose the settings per substation by next-day score).
- **J1:** the joint fit with no tie to TS.
- **J2:** the joint fit plus a penalty (a ridge) that pulls the coefficients towards TS's
  coefficients, with a weight chosen per substation from {0.1, 0.3, 1, 3} by the score on the tuning
  months (odd months of 2024-07 to 2025-06). The choice was 3 at five substations (D5, D9, D10, D11,
  and D14) and 0.3 at D16.

![Figure 1: J1 and J2 minus the two-stage fit's quantile score, by lead band](headline_contrasts.svg)

![Figure 2: J2 minus the two-stage fit's quantile score at 12 to 36 hours, by substation](c2_per_site.svg)

![Figure 3: how far the joint fits move each fitted curve from the two-stage curve](shape_distance.svg)

## Results

- **J1 scores worse at every lead and at every substation.** C1 (J1 minus TS) at 12 to 36 hours is
  +5.419 points (97.5% interval +3.330 to +8.465) in the scored year (2025-07 to 2026-06), and none
  of the five substations improve. In the replicate months (even months of 2024-07 to 2025-06) it is
  +4.323 (+2.988 to +6.281). The gap grows with lead, from +3.96 at 0 to 24 hours to +7.56 at days 4
  to 7 (figure 1). The absolute quantile score at 12 to 36 hours is 7.907 for J1 against 2.488 for
  TS. Over all leads (0.5 to 168 hours), J1 scores 9.563 and TS scores 2.783. D9, scored beside the
  decision substations, shows the same pattern (J1 6.009 against 2.151 at 12 to 36 hours).
- **J2 is indistinguishable from TS.** C2 (J2 minus TS) at 12 to 36 hours is -0.0045 points (-0.0132
  to +0.0025) in the scored year, with three of five substations improving, and -0.0002 (-0.0027 to
  +0.0018) in the replicate months, with three of five. The interval rules out a worsening of the
  score by more than 0.003 points and an improvement by more than 0.013 points. Over all leads, C2
  is -0.0015 (-0.0106 to +0.0048), within the non-inferiority bound of +0.02. D10 contributes most
  of the scored-year difference (-0.0144, figure 2), and leaving D10 out leaves a mean of about
  -0.002.
- **The tuning months prefer the strongest tie on the grid.** The mean tuning-month score is 2.7136
  for ridge weight 0.1, 2.7067 for 0.3, 2.7028 for 1, and 2.7020 for 3, against 2.7028 for TS (J1 is
  8.200). The score falls as the tie tightens and reaches TS's, so the tuning months prefer the
  two-stage fit. The best weight lies at the edge of the grid at five of six substations.
- **J1's fitted curves move far from the two-stage curves, and J2's barely move.** For each term, we
  took the root-mean-square difference between a joint fit's curve and TS's curve, as a share of the
  spread of TS's curve, and then the median over the 27 monthly refits and five substations. For J1
  the shares are 444% for the weekly profile, 46% for the tent (a term for the temperature
  response), 56% for the harmonics (the yearly cycle), 27% for heating, 41% for the holiday term,
  and 33% for lighting. For J2 at weight 3 they are 2%, 7%, 3%, 4%, 9%, and 3% (figure 3). Most of
  J1's weekly-profile difference is a constant offset: a median of 51 points of capacity as it
  stands, against 10 points once the offset is removed. The filter's level absorbs that offset.
- **Breakouts, at 12 to 36 hours in the scored year (descriptive).** C1 and C2 are:
    - Winter (November to March): +6.53 (+2.17 to +12.93) and -0.007 (-0.023 to +0.004).
    - Winter nights: +1.85 (+0.05 to +4.26) and -0.004 (-0.025 to +0.007).
    - Cold nights: +2.44 (+0.64 to +5.80) and -0.017 (-0.048 to +0.013).
    - Christmas and bank holidays: +4.32 (+3.88 to +5.15) and -0.061 (-0.209 to +0.023).
- **Tails.** J2's coverage matches TS's to within 0.001 at every quantile shown, so the cold-night
  shortfall of the 0.95 quantile that [experiment
  125](../00125-differentiable-physics-n8-ens-members-through-gam-k-and-the-spread-around-them/result.md)
  and experiment 133 found is unchanged. The values for TS, J1, and J2 are:
    - 0.95 quantile: 0.942, 0.886, and 0.941 over all hours, and 0.864, 0.945, and 0.864 on cold
      nights.
    - 0.01 quantile: 0.020, 0.046, and 0.020.
    - 0.05 quantile: 0.064, 0.078, and 0.064.

## Checks

- **P1, reproduction (blocking): passed.** TS's scored-year mean forecasts equal experiment 111's
  stored GAM-K rows, and its mean and median forecasts equal experiment 133's GRID rows, with a
  maximum absolute difference of 0 at all five decision substations.
- **P2, the linear filter equals the filter: passed.** Unit tests in
  `packages/studies/tests/test_dp_joint_fit.py` match the corrected loss to `run_kalman` and
  `correction_at` to a relative 1e-9 in float64, with gaps in the readings and a binding variance
  floor. They also match `fit_gam` to 1e-6 when the filter is switched off.
- **P3, the solved loss is the true loss: passed.** At all five substations, the loss of the filter
  run on the joint solution's residuals (the summed squared error after correction) is lower than
  the loss at a perturbed set of coefficients and lower than the loss at TS's coefficients. At D10
  the three values are 1,120,985, 1,121,407, and 1,150,502. The fit does minimise the loss it was
  given, so J1's poor score comes from that loss and not from the solver.
- **P4, known answer: failed for J1, passed for J2.** The target was simulated from TS's own curves
  plus a slow level, a decaying day anomaly, and white noise. The summed centred root-mean-square
  error of the recovered curves over the five substations was 36.6 points of capacity for J1, 11.2
  for TS, and 10.4 for J2. J1 loses even when the assumptions of the joint design hold. The weekly
  profile accounts for 27.3 of J1's 36.6.
- **Absorption control: J1 failed in 4 of 15 substation-cut pairs, and J2 in none.** A term fails
  the control only if it moves by more than twice its noise floor, which is how far it moves when
  the training cut-off moves back by 7 days. J1's noise floors are large, so J1 can move a long way
  without failing. J1's weekly-profile noise floor has a median of 19 points of capacity and a
  maximum of 95, against 1 point for J2 and TS. The J1 failures are D10 at the 2025-01, 2025-07, and
  2026-01 cuts and D5 at the 2026-01 cut. A term had a noise floor above 1 point (unidentified) in
  14 of 15 pairs for J1 and 10 of 15 for J2. The holiday term has a noise floor above 1 point in 11
  of 15 pairs for J1 and 10 of 15 for J2 and TS, because few holidays fall in a training window.
- **Predictions against results.** C1 was predicted at -0.05 to +0.02 points, and the result of +5.4
  is more than 100 times the top of that range. C2 was predicted at -0.03 to +0.01 and came out at
  -0.0045, inside the range. J1's curves were predicted to differ from TS's by 5% to 15% of each
  term's spread, and they differ by 27% to 444%. J2's were predicted to differ by under 5%, and that
  held for four terms but failed for the tent (7%) and the holiday term (9%). The absorption
  prediction (J1 fails, J2 does not) held.
- **Error of the mean forecast beside each contrast (12 to 36 hours, scored year).** The mean
  absolute error is 4.374 for TS, 11.027 for J1, and 4.367 for J2, in points of capacity. The C2
  contrast on the mean forecast's error is -0.0073 (-0.0207 to +0.0032), the same sign as the
  quantile-score contrast, so the audit trigger for a disagreement did not fire. The median forecast
  minus the mean forecast is -0.156 for TS at D5 and -0.154 for J2, against -0.319 for J1, and is
  within 0.05 for TS and J2 at the other four substations.
- **Audit triggers.** Trigger 1 (a control fails) fired, because P4 failed for J1. Trigger 2 (a
  result against its own prediction) fired for C1. The shape distance, the noise floors, and P4
  support the explanation that the filter absorbs a shift of the slow terms, so the loss leaves them
  to the penalty alone. No variant of the loss was run to test that explanation. Trigger 4 (one
  substation drives the contrast) did not change a verdict: C1 is positive at every substation, and
  C2's full interval spans zero while the mean of the four substations other than D10 is -0.0019 (an
  interval for those four was not computed). No Opus audit was run, by agreement with the experiment
  manager, so triggers 1 and 2 are unresolved caveats.
- **Biases stated in advance.** At the joint fit, the filter reads in-sample residuals from the
  training weather (the fifth-generation European Centre reanalysis, ERA5, and the Copernicus
  Atmosphere Monitoring Service, CAMS). At forecast time it reads out-of-sample residuals from the
  analysis proxy, the stand-in for past weather. The noise settings were chosen on TS's own residual
  stream, which favours TS. J2 is a family of fits, with the tie weight chosen on the tuning months.
  These biases favour TS, or tie J2 to TS, so they cannot explain J1's poor score. They do limit
  what a null for J2 says about a joint fit with retuned noise settings.
- **Retraining schedule and cost.** All arms refit the convex model monthly, as in experiment 111.
  The joint fits of all five weights (J1 and four J2) took 0.33 GPU hours in total, about 3.3
  minutes per substation for 27 refits. The fits, tuning, forecasts, scoring, and controls together
  took about an hour and a half of wall-clock time.

## What the next experiment should answer

- **Whether learning the filter's noise settings together with the coefficients helps.** This
  experiment held the settings fixed at experiment 133's values, so a null for J2 does not rule out
  that joint learning helps. That experiment would need a loss in which the slow terms stay
  identified, and J1's failure suggests tying them to a reference fit.
- **Whether a ridge weight above 3 changes J2.** The tuning months preferred the strongest tie at
  five of six substations, so a wider grid would show whether a tie stronger than weight 3 gives any
  improvement.
