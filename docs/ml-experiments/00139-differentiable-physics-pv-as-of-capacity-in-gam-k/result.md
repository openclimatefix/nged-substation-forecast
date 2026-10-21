---
id: "00139"
slug: differentiable-physics-pv-as-of-capacity-in-gam-k
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - differentiable-physics-forecast-engine
  - solar-disaggregation
status: finished
size: >-
  medium. Stored: private files under data/ml-experiments/00139-differentiable-physics-pv-as-of-capacity-in-gam-k/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because the convex model already subtracts a capacity times a unit solar shape and only the source
  of the capacity changes. Callers: this folder's scripts, which copy experiments 111, 129, and 136
  and call the reviewed `studies.dp_*` modules without changing them.
hypothesis: >-
  Feeding the convex model with the Kalman filter (GAM-K, experiment 111) the as-of solar capacity
  of experiment 129 in place of experiment 45's pinned capacity lowers the quantile score on the
  sunniest third of half-hours, and does not raise it on the darkest third. The mechanism is that
  the pinned capacity is a hindsight growth prior, and the as-of capacity is read from each
  Primary's own power before the forecast origin.
prediction: >-
  Written before any fit. The as-of arm scores 0.03 to 0.10 points better than the pin on the
  sunniest third, and within 0.01 points on the darkest third. Over all half-hours the gain is under
  0.1 points, because experiment 129's ranges are 1.2 to 1.6 MW wide and the pin lies inside them at
  most Primaries. The hindsight arm beats the as-of arm by under 0.03 points. The no-PV arm is worse
  than the pin by 0.2 to 0.6 points on the sunniest third. The pin and as-of capacities differ most
  at D10 and D16, where capacity grew (held-forward error 0.30 and 0.70 MW).
decision_rule: >-
  Contrast C1 is the as-of arm minus the pin, read on the sunniest third of half-hours (by CAMS
  irradiance) and on the darkest third, at the 98.75% level of the month-block bootstrap on paired
  differences, as experiment 111, pooled over the five Primaries (D9 has no as-of arm). The
  hypothesis is `supported` when the sunniest-third difference is at most -0.05 points with an
  interval that excludes zero, and the darkest-third difference is at most +0.02 points. These are
  the thresholds the maintainer agreed on 2026-10-21. The 0.05 points is not zero because the as-of
  capacity adds a fitting step (a capacity fit per origin) that the pin does not need. It is `partly
  supported` when the sunniest-third interval lies wholly below zero but the difference is smaller
  than 0.05, or the darkest-third difference is above +0.02. It is `refuted` when the sunniest-third
  interval lies wholly above zero. It is `inconclusive` otherwise. The hindsight arm is a labelled
  ceiling and never a result. The no-PV
  arm is a floor. The verdict is capped at `inconclusive` if the known-answer stage fails.
benchmark: >-
  Arm 1, the pin (experiment 45's `frozen` vintage, as experiment 111), on the same rows; absolute
  skill of climatology and manual_heuristic as stored by experiment 111
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events at the scored substation,
  at D5, D10, D11, D14, and D16 with D9 beside them (D11 also alone), over leads 0.5 to 168 hours
  from the 09:00 UTC `power_fcst_init_time`; sliced by the sunniest and darkest third of half-hours,
  by season, and by lead band; tail calibration at the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00139-differentiable-physics-pv-as-of-capacity-in-gam-k/
verdict: refuted
date: 2026-10-21
pr: 1441
headline: >-
  The as-of capacity scored 0.110 points worse than the pin on the sunniest third of daylight
  half-hours (98.75% interval 0.038 to 0.181) in the scored year, and 0.042 points better on the
  darkest third; the hindsight capacity matched the pin (0.029, interval -0.020 to 0.085).
best_arm: >-
  the pin (experiment 45's real-time vintage), 2.840 points over all half-hours at the five
  Primaries in the scored year, level with the hindsight arm (2.840) and ahead of the as-of arm (2.852)
take_home_message: >-
  Replacing the pinned solar capacity with a capacity read from each substation's own power did not
  improve the forecast, and made sunny half-hours worse by 0.11 points of capacity. A more accurate
  capacity (the hindsight arm) only matched the pin, so the accuracy of the capacity number is not
  what limits the solar part of the forecast. Whatever errs on sunny days lives elsewhere, in the
  ensemble's cloud forecast or the solar shape.
lessons:
  - >-
    A capacity fitted once a year and held flat lags a growing fleet, and the pin's weekly update
    beats it on sunny half-hours (D16: 3.0 MW as-of against 3.9 to 4.7 MW pinned).
  - >-
    A known-answer test that passes (0.78 to 0.88 recovered at 5 of 5 Primaries) does not predict a
    forecast gain: the plumbing was right and the capacity still did not matter.
advice:
  - >-
    Keep experiment 45's pinned capacity in GAM-K; do not replace it with a once-a-year as-of
    capacity (C1, figure 1).
  - >-
    Look for the sunny-day error in the ensemble mean's cloud forecast and in the solar shape, not
    in the capacity, because hindsight capacity gave no gain over the pin (C2).
  - >-
    Do not scale the as-of capacity up (x 1.15) or use the convex model's own units to repair it;
    both made sunny half-hours worse (post hoc arms).
  - >-
    Feed any future capacity source through a per-cut `PinnedCapacity` that passes the
    reproduction check against experiment 111's fits to the last digit (wiring check).
production_status: not-for-production
adopted_in: none
---

# Experiment 139: does an as-of solar capacity in GAM-K lower the forecast error on sunny half-hours?

**Take-home message: replacing the pinned solar capacity with a capacity read from each substation's
own power did not help, and made sunny half-hours worse by 0.11 points of capacity.** A more accurate
capacity (the hindsight arm, which sees the future) only matched the pin. The accuracy of the solar
capacity is therefore not what limits the solar part of the forecast, and the sunny-day error must
come from the ensemble's cloud forecast or from the fixed solar shape. The verdict is `refuted`
under the pre-registered rule: the interval of the as-of arm minus the pin on the sunniest third
lies wholly above zero.

We asked whether the solar capacity subtracted by the convex demand model with a Kalman filter
(GAM-K, [experiment 111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md))
should come from [experiment 129](../00129-differentiable-physics-pv-capacity-known-answer-test/result.md)'s
capacity read from the substation's own power, fitted only on data before each 1 July, instead of
[experiment
45](../00045-pvdis-4-weekly-capacity-asymmetric-l1/result.md)'s pinned capacity. Both feed the same
convex model, driven by the mean of the European Centre's weather ensemble as in [experiment 136][e136].
Only the capacity changed. The known-answer test passed and the forecast got worse.

![Paired contrasts of the quantile score, scored year](figure_01_contrasts_scored.svg)

![The capacity each arm subtracts at each monthly cut](figure_03_capacity.svg)

![Paired contrasts of the quantile score, replicate months](figure_02_contrasts_replicate.svg)

![Known-answer recovery](figure_04_recovery.svg)

## Arms and rows

- **Pin (control):** experiment 45's `frozen` vintage, passed through the adaptor. It reproduces
  experiment 111's fits (largest coefficient difference 0.0 at all six Primaries) and experiment 136's
  `a0` forecasts at D5 exactly.
- **As-of (scored arm):** experiment 129's arm 2, fitted on rows before the latest 1 July origin
  (2023, 2024, or 2025) and held for the following season-year, divided by a recovery divisor
  rebuilt from known-answer fits cut at the same origin. The rebuilt estimates for 2025-07-01 agree
  with experiment 129's published ones within 0.05 MW (leakage check). The origin is 1 July and not
  the design note's 1 September, because the convex model's season-years run July to June.
- **Hindsight (ceiling, never a result):** experiment 129's published estimates, fitted on data to
  2026-07-31. **No PV (floor):** capacity zero.
- **Post hoc, not pre-registered:** as-of x 1.15 (one over the known-answer recovery of 0.87) and
  as-of in the convex model's own units (the raw capacity, before the divisor).
- **Rows:** `fold_a_daily_gap_v1`, 09:00 UTC `power_fcst_init_time`, leads 0.5 to 168 hours, outside
  the hand-filtered switching events. D5, D10, D11, D14, and D16 are pooled. D9 has no experiment 129
  table, so it has the pin and no-PV arms only. Monthly refits on a 24-month window, filter noise as
  experiment 111, 4 to 8 seconds per Primary for a whole year of fits and forecasts.
- **Slices:** the thirds are per Primary, over daylight half-hours (CAMS-driven unit solar shape above
  0.01), by that shape. Night half-hours are in no third.

## Headline numbers

Quantile score in points of effective capacity, five Primaries, scored year (2025-07 to 2026-06).
Intervals are 98.75% intervals from the month-block bootstrap with 20,000 resamples.

| Arm | All | Sunniest third | Darkest daylight third |
|---|---|---|---|
| Pin | 2.840 | 4.452 | 3.000 |
| As-of | 2.852 | 4.562 | 2.959 |
| Hindsight | 2.840 | 4.481 | 2.963 |
| No PV | 3.389 | 5.502 | 3.983 |
| As-of x 1.15 (post hoc) | 2.860 | 4.615 | 2.910 |
| As-of, model units (post hoc) | 2.874 | 4.569 | 3.050 |

- **C1, as-of minus pin:** all half-hours +0.013 (-0.003 to 0.028); sunniest third **+0.110
  (0.038 to 0.181)**; darkest third -0.042 (-0.076 to -0.003). The error of the mean forecast agrees:
  sunniest third +0.221 (0.114 to 0.359).
- **C2, hindsight minus pin:** sunniest third +0.029 (-0.020 to 0.085); all half-hours +0.001
  (-0.011 to 0.013). The ceiling gives no gain over the pin.
- **C3, hindsight minus as-of:** sunniest third -0.081 (-0.125 to -0.043). The as-of arm loses to
  the hindsight arm mainly because it holds one value for a year while the fleet grows (figure 2).
- **No PV minus pin:** all half-hours +0.549 (0.407 to 0.727); sunniest third +1.051 (0.467 to 1.650).
  At D9, +0.275 (0.062 to 0.460). An explicit solar term is worth about 0.55 points, and the choice of
  capacity moves the score by at most 0.11.
- **Post hoc arms:** as-of x 1.15 minus pin on the sunniest third +0.163 (0.100 to 0.218); as-of in
  model units minus pin +0.117 (-0.024 to 0.255). Neither repairs the as-of arm.
- **Replicate months (even months of 2024-07 to 2025-06, origin 2024-07-01):** the sign is the same.
  As-of minus pin on the sunniest third +0.072 (-0.030 to 0.115) and on the darkest third -0.028
  (-0.070 to 0.014). Absolute scores: pin 2.765, as-of 2.774, hindsight 2.768, no PV 3.265. The stop
  rule (as-of worse than the pin on the sunniest third after the replicate, interval wholly above
  zero) did not fire, because the interval includes zero.
- **By Primary:** leaving out any one Primary leaves the sunniest-third sign positive (+0.060 without
  D16 to +0.153 without D14); the interval includes zero when D10 or D16 is left out.

## Known-answer test

- **Passed at 5 of 5 Primaries for all three plant classes** (real farm, east-west, south-west):
  raising the capacity by half the as-of capacity removed 0.86 to 0.88 (real farm), 0.84 to 0.87
  (east-west), and 0.78 to 0.80 (south-west) of the injected output on the sunniest third.
  These are medians over five real farms for the real-farm class. The pre-registered band was 0.75
  to 1.25.
- **The injection enters the scored rows only,** so no fit or filter sees it. Recovery divided by the
  Primary's divisor is 0.92 to 1.11.
- **The calibrated-fake-ensemble gate of experiment 111 was not rerun.** This experiment uses the
  stored calibration and ensemble of experiment 111 and 136.

## Tail calibration and checks

| Arm | Below q01 | Below q05 | Below q95 | Below q99 | Pinball q01 | q05 | q95 | q99 |
|---|---|---|---|---|---|---|---|---|
| Pin | 0.0216 | 0.0632 | 0.9465 | 0.9828 | 0.225 | 0.747 | 0.743 | 0.231 |
| As-of | 0.0208 | 0.0629 | 0.9463 | 0.9824 | 0.226 | 0.757 | 0.741 | 0.233 |

- **Triggers:** the second fired (the result contradicts the prediction of a 0.03 to 0.10 gain). The
  session tested two explanations: staleness (the hindsight arm, which closes 0.081 of the 0.110
  gap) and units or damping (the two post hoc arms, which do not). The seventh fires, because the
  full-history arm, the calibration experiments, the partly pooled GAM, and the card rebuild wait for
  this headline. The manager decides on an audit. The first, third, fourth, and fifth did not fire.
- **Median minus mean forecast:** 0.0 at every Primary and arm, because the spread table is centred
  by construction, so it checks nothing here.
- **Curtailment cost saving against the manual heuristic:** not computed. No node-plus-generators
  metric exists for these five Primaries.
- **Weather and availability:** the ensemble mean is available at forecast time. The as-of capacity
  uses data before each 1 July origin only, and `e00139_asof_table.py` rebuilds the divisor per
  origin.

## Caveats

- **The scored arm holds one capacity for a year,** so C1 includes the staleness of the as-of
  capacity. A live service could refit monthly. The hindsight arm (also one value per season-year)
  bounds what accuracy alone gives, and it equals the pin.
- **Why the pin beats the as-of arm on sunny half-hours is partly unexplained.** At four of five
  Primaries the pin is below the as-of capacity for most of the year, and the pin still wins. The
  post hoc arms rule out a simple unit or damping fix but do not identify the cause.
- **Replicate months cover six months only,** so the replicate intervals are wide.
- **The adaptor lives in this folder** (`e00139_adaptor.py`). Moving it into `packages/studies`
  would be a module change that needs the maintainer's review.

## What to try next

- **Keep the pin.** The capacity is not the limiting input; the next solar experiment should target
  the sunny-day forecast of the ensemble mean (per-member solar, or the cloud damping of the
  irradiance calibration) or the fixed solar shape.
- **If a capacity source is still wanted,** test one refitted monthly and compared with the pin
  weekly path, since a flat annual value loses to the hindsight value.

[e136]: https://github.com/openclimatefix/nged-substation-forecast/pull/1437
