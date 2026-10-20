---
id: "00113"
slug: differentiable-physics-n5-convex-identification-at-realistic-noise
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - differentiable-physics-pooling-across-substations
  - differentiable-physics-demand-calendar-and-weather
  - solar-disaggregation
status: done
size: >-
  medium: experiment scripts only, no module change (every variant fit is a column subset of one
  Gram matrix per substation, solved with studies.dp_gam's solver). See "Size triggers".
hypothesis: >-
  On synthetic substations built from the real calendar and weather, with noise as large and as
  autocorrelated as the convex model's real residuals, the convex model's 80% month-block bootstrap
  intervals cover the truth 70 to 90% of the time, separately for each hour band's heating slope
  and for the latest season-year's PV capacity. Labelled separately, not in the verdict: the
  day-of-year harmonics and the drift term make those intervals wide, by absorbing the seasonal
  cycles of temperature and clear-sky PV.
prediction: see "Predictions"
decision_rule: see "Decision rule"
benchmark: >-
  The posterior interval and the sandwich variance of the penalised fit; an unstratified
  month-block bootstrap; experiment 112's pooled VP gate errors (different noise, see below);
  experiment 45's frozen vintages and Part 5's K-base bootstrap ranges on real data
row_set_id: >-
  Synthetic targets on the training rows of dp_mw_v1 (experiment 103) at D5, D10, D11, D14, and
  D16, cut 2024-07-01; real dp_mw_v1 at the 2025-07-01 cut, replicate 2024-07-01
metric: >-
  Coverage of 80% intervals over 100 synthetic substations, with site-clustered bootstrap intervals,
  for each hour band's total heating slope and the latest season-year's PV capacity; median width
  in points and as a share of the truth; variance ratios and effective sample size
verdict: refuted
production_status: not-for-production
take_home_message: >-
  At the noise the real fits actually have, the convex model's 80% bootstrap intervals for heating
  slopes and for rooftop PV capacity contain the true value 84% of the time, so they can go on
  substation cards, but the heating slopes are weak (the intervals are about 38% of the slope wide)
  and the posterior interval of the same fit is not usable (20%). The pre-registered verdict is
  refuted, because in the low-noise control the capacity interval missed the truth 48% of the
  time, because the penalty's shrinkage bias of about 0.35 points is small against a real-noise
  interval 2.8 points wide, and large against a low-noise one. Pooled estimates stay weakly
  identified, because the seasonal terms absorb about 80% of a heating slope's variance.
advice:
  - >-
    Report the 80% month-block bootstrap with stratified quarters and the
    square-root-of-m-over-m-minus-1 inflation for the convex model's heating slopes and capacity,
    never the posterior interval, which covered 20% at real noise (figure 1).
  - >-
    Treat a convex heating slope as weakly identified at real noise, since the 80% interval is
    about 38% of the slope wide, because the day-of-year harmonics absorb about 80% of the variance
    (figure 3, figure 4).
  - >-
    Score a penalised fit's interval against the noise-free fit's value as well as the truth, and
    print the bias over the width, because the ridge shrinks capacity by about 0.35 points (the
    low-noise capacity row of figure 1).
  - >-
    Do not remove the harmonics to tighten heating slopes unless the heating-slope bias is checked:
    a fit without harmonics on a truth with them overstates slopes by 66% and covers 27% (G-0 on
    T-HD).
  - >-
    Test the pooled model's identification with the same real-residual noise before reading any
    pooled capacity or slope, since a pooled fit with no drift and harmonics is the G-0 case here.
date: 2026-10-20
pr: 1393
adopted_in: none
code: ml-experiments/00113-differentiable-physics-n5-convex-identification-at-realistic-noise/
---

# Differentiable physics, N5: does the convex model identify heating slopes and PV capacity at realistic noise?

**At the noise the real fits have, the convex model's 80% month-block bootstrap intervals contain
the true heating slope and the true PV capacity 84% of the time each, but a heating slope is only
known to within about 38% of its size, and the pre-registered verdict is `refuted` because the
capacity interval failed the low-noise control.** The failure is a deterministic shrinkage bias of
the ridge penalty (about 0.35 points of capacity). It is small against a real-noise interval 2.8
points wide and large against a low-noise one 0.69 points wide. Against the noise-free fit's own
value the same intervals cover 85%, so the interval method works and the bias is the problem. The
posterior interval of the same fit covers only 20% at real noise and must not go on a card. The
cause of the weak heating slopes is plain: the day-of-year harmonics and the drift term can
reproduce the seasonal cycles of temperature and clear-sky PV, and so they absorb about 80% of a
heating slope's variance. Experiment 112's failure to identify slopes and capacities at real noise
is therefore expected of any fit with those seasonal terms, not a flaw of the pooled optimiser.

We built 100 synthetic substations on the real calendar, weather, and PV shape of 5 Primary
substations (20 each), put a known heating slope per hour band and a known PV capacity into each,
added either real residuals borrowed from other substations (the decision noise, standard deviation
4.97 points, autocorrelation 0.83 at one half-hour) or independent noise (standard deviation 5),
and asked how often each 80% interval contains the truth. No forecast is scored, so there is no
quantile score, tail table, or curtailment number. "Identified" means the data fix a quantity
within a narrow range. A "calibrated" interval contains the truth as often as its nominal 80%. The
"effective sample size" is the number of independent half-hours that carry the same information as
the autocorrelated rows.

## Figures

**Figure 1 shows the month-block bootstrap (I1) covering the truth 84% of the time at real noise,
and the posterior interval (I2) 20%.** Under independent noise the posterior interval is fine for
heating slopes (80%), and the capacity row shows the shrinkage bias: I1 covers the truth 52% of the
time, and the noise-free fit's value 85%.

![Coverage of the 80% intervals](figure_1_coverage.svg)

**Figure 2 shows the estimates scattering around the truth with a median signed relative error of
-1.4% for band slopes and -1.3% for capacity.**

![Estimate against truth](figure_2_recovery.svg)

**Figure 3 shows the seasonal terms widening the heating-slope intervals.** Where the truth has no
harmonics or drift, the fit that omits them has an I1 interval about 0.55 times as wide (relative
width 0.21 against 0.38; real noise, band slopes), and an I2 interval 0.44 times as wide, matching
the exact variance ratio's prediction of 0.44.

![Interval width by arm](figure_3_widths.svg)

**Figure 4 shows which term absorbs the information.** The harmonics carry the heating slopes
(removing them leaves 26 to 32% of the variance); the drift carries part of the capacity (removing
it leaves 73%). At the looser drift penalty of experiment 106 (step 30) the drift absorbs less of
the heating slopes and more of the capacity.

![Variance absorbed by each term](figure_4_absorption.svg)

**Figure 5 shows the real fits at the 2025-07-01 cut.** The fitted latest capacity is 0 to 20% below
the weekly vintage's season-year mean at the 4 substations that have one, and I1 is about 5 times as
wide as I2.

![Real latest capacity per substation](figure_5_real_capacity.svg)

## Headline numbers

- **Design:** convex model of `studies.dp_gam` with fitted capacity and D1 drift (ridge 0.1, drift
  step 3,000), one design per substation, cut 2024-07-01, all years, lockdown out. Every fit is a
  column subset of that design; control 1 shows it reproduces `fit_gam`. 20 draws at each of D5,
  D10, D11, D14, and D16; truth heating totals U(0.7, 1.3) times the real fit's (floor 2 points per
  10 °C); capacity U(5, 25) points then U(1, 5) increments; truth T-HD keeps the real fit's
  harmonics and drift, T-0 zeroes them. Real noise: whole months of other decision substations'
  residuals, other years, random sign; 4.7% of slots were empty (set to zero).
- **Decision rule, per family (I1 coverage of the truth in the decision cell and the
  independent-noise cell):** band slopes `calibrated` (84% with site-clustered 95% interval 76 to
  90%, and 83%, 78 to 88%); capacity `overconfident` (84%, 76 to 91%, and 52%, 30 to 74%). Verdict
  `refuted`.
- **Width, decision cell:** band slopes 3.3 points per 10 °C, 38% of the truth, so `weak`; capacity
  2.8 points, 12% of the truth, so `precise`. Under independent noise 8% and 3%.
- **Posterior interval (I2) at real noise:** 20% (bands) and 19% (capacity). The
  bootstrap-to-sandwich variance ratio is 24 (bands) and 21 (capacity), so the effective sample
  size is 0.25% (bands) and 0.84% (capacity) of the rows. I1 is 5.0 and 4.5 times as wide as I2.
- **Penalty bias:** median 0.35 points for capacity; over width 0.51 at independent noise and 0.13
  at real noise. Heating slopes: 0.07 points, 0.02 of the width at real noise. Fitting the full
  model to the real noise alone returns 5.1 to 5.9% of the true slope and 3.6% of the true capacity.
- **Absorption (G-0 against G-full, drift step 3,000):** r is 0.18 to 0.22 for band slopes and 0.67
  for capacity; at step 30 it is 0.17 to 0.20 and 0.60. The capacity ratio is above the 0.56 limit,
  so the label is `absorption partial` at both penalties, though the heating slopes alone would be
  `absorption large`. The Monte Carlo width ratios (0.44 and 0.83 at real noise) match the square
  roots of r (0.44 and 0.82).
- **Other arms (I1 coverage, real noise, T-HD unless stated):** G-full-30 83% and 80%; capacity
  pinned to the truth 84%; pinned to 0.8 times the truth 83% (the median slope error moves by 1.2
  percentage points from the pinned-truth arm); no bounds 84% and 79% (95% of fits have an active
  heating bound); G-0 on T-HD 27% and 0%, with
  slopes overstated by 66% and capacity by 35%; the 2-year window 85% and 81%, only 1.2 times as
  wide for slopes and 0.93 times for capacity. On T-0: G-0 73% and 71%, G-H 84% and 75%, G-D 75%
  and 80%.
- **Noise:** autocorrelation 0.83, 0.44, and 0.21 at lags of 1, 48, and 336 half-hours; standard
  deviation 4.97 points.

## Predictions against outcome

- **Held:** decision-cell I1 coverage 70 to 90% for both families; band slopes `weak`; capacity
  width under 25%; I2 under 60% at real noise; τ between 3 and 30; I1 and I2 both 70 to 90% for
  band slopes under independent noise; r at most 0.36 for every band slope; Monte Carlo width ratio
  within 0.1 of √r; G-0 on T-HD under 50%; the pinned-0.8 arm within 10%; an active bound in over
  30% of fits; unbounded coverage within 5 points; spectral share over 70% in G-full; I1 2 to 5
  times I2; the night-band slope excluding zero at all 5 real sites.
- **Failed:** I1 and I2 both 70 to 90% for capacity under independent noise (52% and 39%); the
  penalty bias under a quarter of the width at independent noise (0.51); the noise-only bias under
  5% of the true slope (5.1 to 5.9%); r at most 0.56 for capacity (0.67);
  drift absorbing more of the night band than harmonics (the harmonics absorb it); G-0's spectral
  share under 50% (70 to 83%); 2-year widths 1.3 to 2 times the all-years widths (1.2 and 0.93).
- **Real data (scored cut):** I1 capacity intervals are 2.7 to 5.9 points wide (D5 4.2, D10 5.9, D11
  3.1, D14 2.7, D16 4.2); they overlap the vintage's ±20% band at all 4 sites that have a vintage
  (D14's stops in 2023). The night-band slope excludes zero at 5 of 5 sites.

## Substation cards for the five decision substations

Provisional: fitted to past weather on a screening run, fit up to 2025-07-01, 80% month-block
bootstrap (I1), points of effective capacity (heating: per 10 °C). Heating slopes carry the label
"80% month-block bootstrap; calibrated for noise on in-family synthetic data (N5); excludes
PV-shape bias; weak". The capacity intervals carry no N5 label, because the capacity family failed
its gate, and are printed with "overconfident in the low-noise control (experiment 113)". Pooled or
variable-projection numbers carry "weakly identified at real noise levels (experiment 112)".

| Substation | Night | Morning | Midday | Evening | Capacity, 2024 season-year |
|---|---|---|---|---|---|
| D5 | 7.9 to 10.5 | 9.7 to 13.3 | 9.6 to 13.0 | 8.3 to 10.9 | 56.3 to 60.5 |
| D10 | 15.5 to 19.8 | 17.8 to 22.5 | 18.8 to 23.7 | 16.1 to 20.2 | 39.3 to 45.2 |
| D11 | 6.9 to 9.7 | 9.2 to 12.1 | 8.9 to 12.2 | 6.7 to 9.9 | 25.1 to 28.2 |
| D14 | 5.2 to 7.0 | 7.8 to 9.2 | 9.3 to 10.7 | 7.0 to 9.0 | 29.4 to 32.2 |
| D16 | 4.3 to 6.6 | 5.5 to 8.0 | 6.1 to 8.9 | 3.4 to 5.8 | 29.8 to 34.0 |

## Checks

- **Median forecast minus mean forecast, per site:** not applicable; no forecast is made. The
  estimates' own bias is the median signed relative error: -1.4% (bands), -1.3% (capacity).
- **Mean-forecast error beside the quantile score:** not applicable for the same reason.
- **Triggers fired:** (2) the convex model's capacity is `precise` at real noise where experiment
  112's pooled fit failed, and the diagnostics that explain why (the seasonal terms absorb
  information, and the in-family truth excludes PV-shape bias) are recorded in this page but not
  yet tested by a fresh agent; (3) the capacity family's coverage fails at independent noise with a
  bias 0.51 of the width, so coverage and bias point in different directions; (7) the cards, Part
  5's capacity reading, and the next pooled test depend on the result. Not fired: (1) all five
  controls pass (the largest relative difference from `fit_gam` is 4.9e-10, noise-free recovery is
  exact to 1.6e-12 over 235 checks, I2 covers 79.5% of the noise-free value over 500
  independent-noise draws, I1's coverage on the cell's 100 draws is 78% and 81%, every resample
  keeps every quarter's size, and no row is past its cut); (4) the gate labels do not change with
  any one site left out (the capacity row stays `overconfident` and the band row `calibrated`); (5)
  G-full never beats the true smaller G-0 on T-0 in error (3.2% against 1.6% for band slopes at
  independent noise); (6) Part 5's K-base ranges were not read, so the trigger is unchecked. A
  fresh Opus audit and numbers review are requested.

## Caveats and what to try next

- **Caveats:** ERA5 and CAMS flatter the solar term; in-family truths cannot show PV-shape bias
  (experiment 112's gate test 1); the donors are the other decision substations; the m over m minus
  1 inflation is an approximation (uninflated I1 covers 83% and 80%); N5's noise is not 112's
  (whole months of
  the same season from another year, not weeks from any season); capacity's supporting months, as
  counted, are 11 and not the planned 6; the 2024 season-year column is dropped (see the
  data-processing page).
- **Next:** a test of the pooled model with the real-residual noise before reading any pooled
  capacity or slope; a capacity interval centred on the noise-free fit (bias-corrected) would put
  the capacity family through the gate again, which needs a new pre-registration; Part 5's capacity
  rule may use I1's range at real noise, but the gate says not to print it as calibrated.

## Reproduce

```bash
uv run python ml-experiments/00113-*/e00113_grid.py reals
uv run python ml-experiments/00113-*/e00113_grid.py control
uv run python ml-experiments/00113-*/e00113_controls.py static
uv run python ml-experiments/00113-*/e00113_controls.py noisy
uv run python ml-experiments/00113-*/e00113_grid.py main
uv run python ml-experiments/00113-*/e00113_analytic.py
uv run python ml-experiments/00113-*/e00113_real.py
uv run python ml-experiments/00113-*/e00113_summary.py
uv run python ml-experiments/00113-*/e00113_report.py
uv run python ml-experiments/00113-*/e00113_charts.py
```

Every number above is printed in `report.txt` next to this page.
