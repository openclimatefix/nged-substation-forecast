---
id: "00126"
slug: differentiable-physics-s1-cheap-screens-added-to-gam-k
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - differentiable-physics-forecast-engine
  - lagged-power-and-error-correction
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00126-<slug>/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one is defensible for every arm (knot
  positions, decay rates, regime cut-offs), and this plan pre-registers one each. Callers: this
  folder's scripts only; no change to packages/. Asset graph: not touched.
hypothesis: >-
  Each of six families of cheap additions to the day-ahead convex model GAM-K lowers its score
  after the Kalman filter, winter first: a free or monotone temperature curve (A-free, A-mono),
  exponentially smoothed temperatures (S-ewma) or smoothed daily extremes (S-mm), wind chill
  (C-wind) or apparent temperature (C-hw) inside the heating hinge, a school-term term (D-sch),
  lighting from solar elevation (E-sol), and a filter whose noise differs by regime (F-mix). One
  hypothesis per arm, in "Arms".
prediction: >-
  Written before any fit, see "Predictions". Headline: most arms tie (within 0.02 points); E-sol is
  the likeliest gain, at -0.15 to -0.03 points on dark winter evenings; F-mix moves cold-snap q95
  and q99 coverage 1 to 3 points towards nominal.
decision_rule: >-
  Each arm gets its own label and there is no aggregate verdict. gain: 99.44% interval (Bonferroni
  over 9) on the primary rows wholly below zero, estimate at most -M (0.03 points on winter rows,
  0.05 on event or cold-snap rows), replicate-year estimate below zero, at least 3 of 5 sites
  improve in the scored year, and non-inferiority holds on the other rows. The threshold is
  nonzero because a gain only justifies the later ENS confirmation if it exceeds the screening noise.
  Gates (known-answer power, placebo, filter-fit control) can downgrade a label to not powered or
  inconclusive. See "Decision rule".
benchmark: >-
  T-all-K, the decision form of experiment 114 (GAM-K), re-read from 114's stored predictions;
  B0-K and MH-K (experiment 121) as descriptive references
row_set_id: >-
  dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and D16; tuned 2023-07 to 2024-06, replicate
  2024-07 to 2025-06, scored 2025-07 to 2026-06
metric: >-
  Quantile score (twice the mean pinball loss over experiment 87's 11 quantiles) in points of
  effective capacity, on rows outside the hand-filtered switching events, at day D+1; tail coverage
  and pinball at q01, q05, q95, and q99; for the gates, power and false-positive rate over
  synthetic draws
production_status: not-for-production
---

# Differentiable physics, S1: do six families of cheap additions to GAM-K lower the day-ahead score?

**Status: planned.** No fit has run. This page holds the pre-registered plan; the write-up replaces
it after the results. The full plan, with the Sonnet plan review and its triage, is the private
brief `data/ml-experiments/briefs/dp_screens.md`; the sections below are the parts the results
depend on.

## Question

**Do cheap additions to the day-ahead GAM-K (a generalised additive model fitted by convex
optimisation) lower its score once the Kalman filter has run, judged on November-to-March rows
first?** Nine arms in six families each add one term to T-all-K, the decision form of
[experiment 114](../00114-differentiable-physics-n3-n4-interactions-and-heating-threshold/result.md).
The ideas come from GAM load forecasting: smoothed temperatures and smoothed daily extremes
([Obst et al. (2021)](https://arxiv.org/abs/2009.06527); [Fasiolo et al.
(2021)](https://arxiv.org/abs/1707.03307)), wind chill and apparent temperature (North American
wind-chill index; [Steadman (1984)](https://doi.org/10.1175/1520-0450(1984)023%3C1674:AUSOTA%3E2.0.CO;2)),
and a Kalman filter on the model's residuals as in earlier experiments of this series.

## Arms

| Arm | Hypothesis | Added to T-all |
|---|---|---|
| A-mono | A monotone heating curve that may flatten in deep cold lowers the winter score. | I-spline block (non-negative increments, not forced convex) on the 24-hour smoothed temperature in 4 hour bands, knots at 2, 6, 10, 14, 18, 22 C |
| A-free | A free temperature curve (no sign bound) lowers the winter score. | natural cubic spline block, same knots and bands, difference penalty (lambda 1), ridge 0.1 |
| S-ewma | Exponentially smoothed temperatures capture thermal inertia better than T-all's smoothings. | 15.5 C hinge on exponentially weighted temperature, time constants 1.5 and 4 days, 4 bands, non-negative |
| S-mm | The smoothed daily minimum and maximum temperature add information. | 15.5 C hinge on the previous 24 hours' minimum and maximum and their 7-day means, 4 bands, non-negative |
| C-wind | Wind chill inside the hinge lowers the winter score. | T-all's hinge on T plus `WC(min(T, 10), v) - min(T, 10)` (North American formula, v in km/h floored at 4.8) |
| C-hw | Humidity and wind (apparent temperature) inside the hinge lowers the winter score. | T-all's hinge on `T + 0.33 (e - e_mean) - 0.70 (w - w_mean)`, e from the 2 m dew point (Magnus), w the 10 m wind in m/s, means from the tuning year |
| D-sch | School holidays shift weekday demand. | 48 half-hour deviations on school-holiday weekdays (Lincolnshire term table of experiment 00011; one authority's dates, not NGED's footprint), ridge 0.1 |
| E-sol | Darkness set by solar elevation explains the evening peak better than a fixed clock curve. | 5 B-spline functions of solar elevation (-12 to +10 degrees) times a 15:00 to 21:00 local indicator, weekday and weekend apart |
| F-mix | A filter whose noise differs by regime (cold snap, mild, holiday) widens the tails correctly. | one observation and step variance per regime in the filter |

No night-time or Economy 7 arm (experiment 121 owns that), and no onboarding or short-history arm.

## Rows, years, and winter-first rules

- **Years:** tuning 2023-07-01 to 2024-06-30, replicate 2024-07-01 to 2025-06-30, scored 2025-07-01
  to 2026-06-30. One frozen fit per arm and cut, one seed. Nothing scores a valid time on or after
  2026-07-01.
- **Sites:** D5, D10, D11, D14, and D16 decide; D9 is descriptive in the scored year. D15 and D18 are
  not fitted.
- **Units:** points of effective capacity (each series' own 99th percentile of cleaned demand).
- **Winter first:** every arm's primary score is on November-to-March rows. Two winters decide
  (2024-25, 2025-26); 2023-24 is the tuning winter. A winter loss beside an annual gain holds the
  arm back.
- **One cold-snap definition:** a day whose mean ERA5 temperature at the site's cell is below the
  10th percentile of the 2023-24 winter's daily means, fixed in degrees now and printed per site.
  Fewer than 7 such days in a winter prints "too few days".
- **Event-day rule:** D-sch is judged on weekdays inside school holidays, E-sol on 15:30 to 19:00
  local half-hours from November to February, each with non-inferiority on all other rows.

## Known-answer tests and gates (all can fail)

1. **Reproduction (blocking):** each arm with its block forced to zero reproduces T-all's fitted
   values to 1e-5 points; T-all-K reproduces experiment 114's stored scored-year score to 1e-4;
   F-mix with equal regime pairs reproduces the plain filter; no training row after its cut;
   E-sol's elevation matches the `studies` routine; T-all's heating slope can fall to zero.
2. **Pilot half-widths:** bootstrap B0-K minus T-all-K on 114's stored tuning-year predictions at
   99.44%, per primary and non-inferiority row subset, written to `decisions.json` before any arm
   result is read.
3. **Injected-effect gate:** an effect sized so the oracle gain equals M is injected into T-all's
   real 2023-07-01 fit and passed through the filter with experiment 113's real-noise rule. Pass:
   the paired mean is at most -M/2 and the 99.44% interval over 100 draws lies below zero.
   Otherwise the arm is `not powered`.
4. **Placebo gate (arms reaching `gain` or `small gain`):** experiment 114's audit construction,
   each added column split by an odd/even parity with no time-of-day or weekly structure. Pass: the
   placebo's gain is under one third of the arm's gain, or its 99.44% interval includes zero.
   Failure labels the arm `inconclusive`.
5. **Curve coverage (A, descriptive):** the 80% month-block bootstrap interval for the ratio of
   the fitted response at 5 C to that at 12 C must cover 70 to 90% in gate 3's draws, or no curve
   summary is drawn.

## Planned contrasts

Nine, Bonferroni (99.44% intervals), arm minus T-all-K, resampling whole calendar months, paired,
three seeds, day D+1, after the filter: X1 A-mono and X2 A-free, X3 S-ewma and X4 S-mm, X5 C-wind
and X6 C-hw (all on November to March); X7 D-sch (school-holiday weekdays); X8 E-sol (dark winter
evenings); X9 F-mix (cold-snap days, with a tails table). Every arm is also run with T-all-K's
stored filter hyperparameters (`-b0`); a label that differs between `-K` and `-b0` is
`inconclusive`. Non-inferiority margin NI is the larger of 0.02 points and the pilot half-width.

## Predictions

- A-mono, A-free: -0.03 to +0.01 on November to March; `tie` likeliest.
- S-ewma, S-mm: within +-0.02 of zero; added columns correlate above 0.9 with T-all's.
- C-wind, C-hw: -0.05 to 0 on cold-snap days, within +-0.02 on November to March.
- D-sch: -0.05 to 0 on school-holiday weekdays (experiment 00011 found a null for XGBoost).
- E-sol: -0.15 to -0.03 on dark winter evenings, largest in late February against December.
- F-mix: cold-snap q95 and q99 coverage moves 1 to 3 points towards nominal; annual score within
  +-0.01.
- About one in four arms is `not powered` at gate 3.

## Decision rule

**Labels, per arm:** `gain`, `small gain` (interval below zero, estimate between -M and zero,
replicate below zero), `not replicated`, `worse`, `worse on other rows`, `tie` (interval includes
zero and half-width at most M), `undecidable` (half-width above M), plus the gate labels. Only a
`gain` becomes a candidate, and only after an ENS confirmation with its own plan. Every arm uses
past weather, so every number is a ceiling for ENS. With nine contrasts, one `gain` is weaker
evidence than one contrast with one `gain`.

## Privacy and stage rule

D-labels only; D15 and D18 never appear. Experiment 114 (PR 1397) is unmerged: until it merges,
only code copying, known-answer test 1, tuning-year fits, and gates on the 2022-07-01 and
2023-07-01 cuts run; no replicate-year or scored-year row is fitted or read.
