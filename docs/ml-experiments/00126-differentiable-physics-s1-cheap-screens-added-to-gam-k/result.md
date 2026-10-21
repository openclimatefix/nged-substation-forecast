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
  lighting from solar elevation (E-sol), and a filter whose noise differs by regime (F-mix). A
  second family of four exploratory screens: a switching Kalman filter with three hidden regimes
  (G-skf), a learned periodic basis for the year cycle (W-basis), a soft cold/mild mixture of
  heating slopes (W-mix), and a cross-substation anomaly from weather-station residuals (W-stn).
  One hypothesis per arm, in "Arms" and "Four exploratory screens".
prediction: >-
  Written before any fit, see "Predictions". Headline: most arms tie (within 0.02 points); E-sol is
  the likeliest gain, at -0.15 to -0.03 points on dark winter evenings; F-mix moves cold-snap q95
  and q99 coverage 1 to 3 points towards nominal.
decision_rule: >-
  Each arm gets its own label and there is no aggregate verdict. gain: 99.44% interval (Bonferroni
  over 9; 98.75% over 4 for X10 to X13) on the primary rows wholly below zero, estimate at most -M
  (0.03 points on winter rows, 0.05 on event or cold-snap rows), replicate-year estimate below zero,
  at least 3 of 5 sites improve in the scored year, and non-inferiority holds on the other rows. The
  threshold is nonzero because a gain only justifies the later ENS confirmation if it exceeds the
  screening noise. Gates (known-answer power, placebo, filter-fit control) can downgrade a label to
  not powered or inconclusive; a failed G-skf gate caps its label at inconclusive. See "Decision
  rule".
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

# Differentiable physics, S1: do cheap additions to GAM-K lower the day-ahead score?

**Status: planned.** No fit has run. This page holds the pre-registered plan; the write-up replaces
it after the results. The full plan, with the Sonnet plan review and its triage, is the private
brief `data/ml-experiments/briefs/dp_screens.md`; the sections below are the parts the results
depend on.

## Question

**Do cheap additions to the day-ahead GAM-K (a generalised additive model fitted by convex
optimisation) lower its score once the Kalman filter has run, judged on November-to-March rows
first?** Nine arms in six families, and then four exploratory screens in a second family, each add
one term to T-all-K, the decision form of [experiment 114](../00114-differentiable-
physics-n3-n4-interactions-and-heating-threshold/result.md). The ideas come from GAM load
forecasting: smoothed temperatures and smoothed daily extremes ([Obst et al.
(2021)](https://arxiv.org/abs/2009.06527); [Fasiolo et al.
(2021)](https://arxiv.org/abs/1707.03307)), wind chill and apparent temperature (North American
wind-chill index; [Steadman
(1984)](https://doi.org/10.1175/1520-0450(1984)023%3C1674:AUSOTA%3E2.0.CO;2)), and a Kalman filter
on the GAM-K residuals as in earlier experiments of this series.

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

Nine in the first family, Bonferroni (99.44% intervals), arm minus T-all-K, resampling whole
calendar months, paired, three seeds, day D+1, after the filter: X1 A-mono and X2 A-free, X3 S-ewma
and X4 S-mm, X5 C-wind and X6 C-hw (all on November to March); X7 D-sch (school-holiday weekdays);
X8 E-sol (dark winter evenings); X9 F-mix (cold-snap days, with a tails table). Every arm is also
run with T-all-K's stored filter hyperparameters (`-b0`); a label that differs between `-K` and
`-b0` is `inconclusive`. Non-inferiority margin NI is the larger of 0.02 points and the pilot half-
width.

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

## Four exploratory screens (a second family, added before any fit)

These four screens form a family of their own with Bonferroni over 4 (98.75% intervals), separate
from the nine above. Thirteen contrasts exist in all, each family holds its own error rate, and a
`gain` in either family carries the same caveat about multiple screens. No night-time or onboarding
arm. **No paper on switching Kalman filters, expectation-maximisation, or the generalised
pseudo-Bayesian (GPB) approximation is cited, because none has been read in full.** The run order
below includes a reading step, and until it is done the page claims no novelty and cites none.

| Arm | Hypothesis | Design (fixed now) |
|---|---|---|
| G-skf | A Kalman filter with three hidden regimes (normal; load transfer or level shift; bad meter or outlier) detects the hand-filtered events earlier, recovers faster after a transfer ends, and lowers the winter score, with no more false alarms than the plain filter. | One Kalman filter per regime on the out-of-sample residuals of the frozen GAM-K, mixed by regime probability with the first-order GPB collapse (GPB1) at each step, diagonal noise. Normal: the plain filter's level and noise. Shift: level step variance 100 times the normal one. Outlier: observation variance 100 times the normal one and no level update. The two multipliers are fixed now. The 3 by 3 transition matrix and the normal regime's noise are fitted by expectation-maximisation (EM) as described below. |
| W-basis | A periodic basis learned from the data fits the year cycle better than T-all's fixed day-of-year harmonics. | Read the harmonic count K from experiment 114's code at step 0. The basis is shared by all sites. Its columns are the leading K left singular vectors of the matrix of smoothed day-of-year demand profiles (365 days by site-years), after removing T-all's weather terms with a fit whose year-cycle columns are zero, from training rows before the cut only, made continuous across 31 December. If K exceeds the matrix rank, the remaining columns are T-all's own higher harmonics, and the count of each kind is printed. |
| W-mix | Heating slopes that differ between cold and mild weather lower the winter score. | T-all's heating block multiplied by w(T) and by 1 - w(T), with w a logistic of the 24-hour smoothed temperature centred on 8 C with scale 3 C: two non-negative blocks. W-mix overlaps A-free, so the contrast against A-free is descriptive. |
| W-stn | The weather station's observed temperature minus ERA5's, over the day before the forecast, carries information on the next day's demand. | One column per hour band: the mean over the 24 hours ending 06:00 UTC on day D of (MIDAS Open hourly air temperature at the nearest station with hourly data minus ERA5 at the site's cell), so the last hour used is 3 hours before the 09:00 UTC origin. |

**W-stn's data limits.** MIDAS Open is a historical archive, released months late, and the copy on
disk covers 2017 to 2025. The assumed latency for a live feed is 3 hours, and this plan names no
live feed that delivers it, so any W-stn gain is a ceiling for production. The scored year runs to
2026-06-30, so W-stn and T-all-K are compared on the rows to 2025-12-31 only, which holds two
winter months of the scored year. W-stn therefore cannot be labelled `gain` and its best label is
`small gain (partial scored year)`. Step 0 checks station coverage per site and year, and a site
with a station more than 50 km away or with a gap over 10% of hours drops from this arm. The
data are already on disk under the licence recorded in the download's `lineage.json`. A longer
record would need a new download, which this plan does not request.

**G-skf, the EM and thresholds.** For each cut c in {2023-07-01, 2024-07-01, 2025-07-01}:

- **Residuals:** EM and the detection threshold use the out-of-sample residuals of the replica GAM-K
  fitted 12 months before c, over [c - 12 months, c); these are the residuals the plain filter
  already tunes on. Rows inside the hand-filtered events (padded 24 hours) are masked, so EM sees
  no event label and no injection.
- **Held-out year:** injections go into the residuals of the GAM-K fitted at c over [c, c + 12
  months), a year that neither EM nor the threshold has seen. At c = 2023-07-01 this is the tuning
  year, so all of it runs before the merge. Recall on injected events is measured there only.
- **Real events:** the 66 hand-filtered events (35 lasting over a day, 31 shorter) that fall in
  the replicate and scored years are the test set, scored with the cuts 2024-07-01 and 2025-07-01.
  Events in the tuning year are in sample for experiment 101 and descriptive here.
- **Alarm rule common to all detectors:** the statistic exceeds k for 3 consecutive half-hours,
  where k is set so the detector raises 1 alarm per 100 non-event days on [c - 12 months, c).
  Statistics: the standardised innovation for the plain filter; the sum of the shift and outlier
  probabilities for G-skf; edge-flow's own stored statistic from experiment 67 at its own
  operating point (its coverage count of the 66 events is printed, and its comparison is
  descriptive when it covers fewer than 20). KEZ-DT, the day-type variant of the Kalman filter on
  effects from experiment 106, zeroes updates on holidays and is not a detector, so it is
  compared on holiday and Christmas scores only.

**G-skf injection harness.** The harness of experiment 43 (`e00043_inject.py`, copied) injects
conserved load transfers, and its metering-error types (a stuck run of 24 hours, and scale
errors of 30 and 90 days, as in experiment 124's plan) at 10%, 25%, and 40% of the injected
site's median demand. Each cell has 100 seeds per site. Two placebos run beside them: the clean
stream (zero amplitude), and the cold-snap days of the clean stream, which are a legitimate
demand rise.

**G-skf's failable rules (all measured on the held-out year at c = 2023-07-01 before the merge,
again at later cuts after it).**

| Claim | Pass rule | Prediction |
|---|---|---|
| Detection | recall on injected transfers at least the plain filter's plus 0.10 | +0.05 to +0.20 |
| Delay | median hours from event start to alarm no longer than the plain filter's, for 25% and 40% transfers | 2 to 6 hours shorter |
| Recovery | median hours until the filtered residual is back within 1.5 normal standard deviations after a transfer ends, at least 6 hours shorter than the plain filter's | 6 to 24 hours shorter |
| Real events | recall on the real events at least the plain filter's plus 0.05, counts reported | 8 to 16 of the 35 long events, 0 to 3 of the 31 short |
| False alarms | zero-amplitude and cold-snap placebos each under 2 per 100 non-event days; winter outside Christmas under 2 per 100 days; the Christmas fortnight under 3 alarm site-days per year | within these limits |
| Power gate | on a synthetic stream with a known regime path drawn from the fitted model (100 draws), G-skf recovers the regime on at least 80% of half-hours and its score is below the plain filter's, 98.75% interval wholly below zero | passes |
| Reproduction | with both multipliers at 1, G-skf reproduces the plain filter to 1e-6 | passes |
| Leakage | a test asserts that EM, the threshold, and the transition matrix see no row at or after c | passes |

**A failed row caps G-skf's X10 label at `inconclusive`.** The score contrast is still printed.

**Other arms' gates.** W-basis and W-mix are linear in a fixed basis, so the injected-effect gate
sized at M (pass: paired mean at most -M/2, because the oracle gain is M and a fit that recovers half
of it still shows the screen can see the effect) and the parity placebo apply to them. The parity
placebo suits D-sch and E-sol because a parity of half-hours or days has no daily or weekly
structure for a calendar term to find. W-mix's reproduction test sets the two blocks' coefficients
equal, which must reproduce T-all to 1e-5. W-basis has a leakage test covering the singular-value
decomposition: no row at or after the cut enters the basis.

**Planned contrasts X10 to X13:** G-skf-K, W-basis-K, W-mix-K, and W-stn-K minus T-all-K, all on
November to March, with M = 0.03 points and the same replicate and site rules. G-skf also reports
its score on the 48 hours after each of the 66 events ends, with non-inferiority 0.02 elsewhere.
Before any of the four arms is read, the pilot half-width for November to March at 98.75% is
printed beside M and written to `decisions.json`. **Power:** the nine arms' pilot half-width is
about 0.03 points, equal to M, so a contrast whose half-width exceeds M is `undecidable` whatever it
scores. The predictions are conditional on the printed half-width.

**Predictions for X10 to X13.** G-skf: a winter score within +-0.03 of T-all-K and a cold-snap
score improvement of -0.05 to 0 points. W-basis and W-mix: within +-0.02, `tie` or `undecidable`.
W-stn: -0.05 to 0 points on November to March, largest on cold-snap days.

## Figures added by the exploratory screens

- **Weights over time.** The nine arms and their `-b0` runs share the plain filter, so one plot
  shows the plain filter's gain over the coldest week and the Christmas fortnight per D-labelled
  site, with holidays and cold-snap spells shaded as bands and the filtered level's 80% interval
  drawn. An arm gets its own plot only where its mean absolute gain differs from T-all-K's by more
  than 10%. G-skf shows its regime probabilities and its effective gain.
- **Detection curves** for G-skf, the plain filter, and edge-flow on injected and real events
  (recall against false alarms per 100 days), and a recovery-time chart in hours.
- **X10 to X13 join the headline chart, the absolute-score figure with Part 4's references, and the
  whole-day (00:00 to 24:00 local) coldest-week figure,** where the W-stn and G-skf curves sit
  beside the 01:00 step that no arm models.

## Compute and run order for the exploratory screens

- **CPU only:** the GPB1 filter (a mixture over regimes at every step, not a batched plain
  filter), EM, the SVD, and the W-stn build. Plain-filter fits may use the GPU after the batched
  filter of PR 1419 has merged. The injection grid is 3 error types and transfers, 3 amplitudes,
  100 seeds, and 5 sites: 4,500 streams of 17,520 half-hours with 3 regimes. The loop runs over
  time with the streams as one numpy axis, in chunks of 500 streams, so a chunk holds about 70
  MB per array and each script stays under 8 GB. Estimated cost: 17,520 steps by 10 chunks by
  3 regimes by 2 cuts, about 1 hour. EM adds 20 iterations over 5 sites and 3 cuts, about 10
  minutes. **The total for the experiment becomes about 4 hours, budgeted at 5.**
- **Reading step:** before the write-up, read in full at least one switching Kalman filter paper
  and one GPB or interacting-multiple-model reference, and cite them from the read copies.

## Privacy and stage rule

D-labels only; D15 and D18 never appear, no topology beyond interconnected pairs is shown (the edge-
flow comparison shows counts only), temperatures are in degrees Celsius and delays in hours. Until
experiment 114 has merged, only code copying, known-answer test 1, tuning-year fits, and gates on
the 2022-07-01 and 2023-07-01 cuts run; no replicate-year or scored-year row is fitted or read.