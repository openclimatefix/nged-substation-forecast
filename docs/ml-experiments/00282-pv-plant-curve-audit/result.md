---
id: '00282'
slug: pv-plant-curve-audit
title: >-
  Does our solar plant model turn sunshine into power the way the six metered solar farms do?
theme: metered-solar-farms
also_relevant_to:
- differentiable-physics-embedded-generation
- solar-disaggregation
status: finished
size: >-
  medium. Stored: private tables under data/ml-experiments/00282-pv-plant-curve-audit/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one defensible
  design for the departure test (binning by irradiance or by elevation), so the bins and the
  departure threshold are fixed here before any table is built. Callers: this folder's scripts,
  which copy experiments 218 and 269 and call the reviewed `studies.pv_*` modules without editing
  them. Asset graph: not touched.
hypothesis: >-
  The curve that turns sunshine into power in our solar models departs from the metered farms'
  measured curve by more than 5% of capacity somewhere in the clear-sky range, and the departure
  is the flat-top level (the inverter limit), not the panel tilt or the temperature loss.
prediction: >-
  Written before any table is built. At 4 or more of the 6 farms the engine's curve, in both
  engines (the plant fit of experiment 218 and the solar term of the best-guess stack, experiment
  139's pinned capacity times the fleet shape), departs from the binned measured curve by more
  than 5% of capacity in at least one clear-sky irradiance bin of the scored year. The departure
  is largest in the top three irradiance bins, where the measured curve flattens at a different
  level from the engine's. Tilt, azimuth, and temperature residuals are each under 3% of capacity.
decision_rule: >-
  A farm "departs" when, in some irradiance bin holding at least 100 half-hours of the scored year,
  the mean of measured minus modelled output, in points of capacity, has a 95% month-block interval
  excluding zero and an absolute value above 5. The prediction is supported when 4 or more of 6
  farms depart in both engines and the largest departure at those farms is in the top three
  irradiance bins. Stop rule: if no bin departs by more than 3 points at 5 of 6 farms, report that
  the curve is sound and stop before check 7. Check 7 (the pooled empirical shape in place of the
  engine's curve) is a fix test and needs a gain of at least 0.02 in the fraction of output fitted
  on the scored year to be worth a new fitted object in production, the threshold experiment 218
  used for the same reason.
benchmark: >-
  The fleet shape with one fitted scale per farm (experiment 218's arm 1) and the per-farm plant
  (arm 2), scored on experiment 218's rows (fold 1, scored 2025-09 to 2026-08, tuned before)
row_set_id: >-
  Experiment 218's fold 1 rows: daylight half-hours (sun more than 5 degrees above the horizon)
  with a null `drop_reason`, CAMS irradiance and ERA5 air temperature
metric: >-
  Mean of measured minus modelled output in points of capacity per irradiance bin, season, hour
  half, elevation band, and temperature band; fitted flat-top level over capacity; best tilt and
  azimuth per farm; fraction of output fitted
code: ml-experiments/00282-pv-plant-curve-audit/
verdict: not supported
best_arm: >-
  Not applicable: an audit. Fraction of output fitted, median over the six farms: per-farm plant
  0.953, fleet shape 0.950, stack's solar term 0.950, pooled shape of the other five farms 0.944
headline: >-
  The engines' curves depart by more than 5 points of capacity at 3 of 6 farms (A, B, and E), not
  the 4 predicted; the common error across all six farms is the stack's 35 degree tilt, which
  over-predicts December clear midday output by about a third.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The curve that turns sunshine into power is sound at 3 of the 6 metered farms and departs at the
  other 3 in different directions, so no single flat-top fix repairs it. The stack's solar term
  also assumes panels tilted 35 degrees, and the farms fit best at about 20 degrees, which makes the
  stack over-predict winter and under-predict summer by up to a third. Swapping in the pooled shape
  of the other farms does not help at the farms, so the next experiment should try the 20 degree
  tilt inside the stack.
advice:
  - >-
    Test a 20 degree south-facing fleet tilt in the stack's solar shape in place of 35 degrees,
    because all six farms fit best at 15 to 20 degrees and the 35 degree shape over-predicts
    December clear midday output by about a third (Check 6, figure 5).
  - >-
    Do not replace the engine's curve with the pooled shape of other farms for forecasting at a
    single farm; it scores 0.006 below the fleet shape (Check 7, figure 6).
  - >-
    Do not blame the spring capacity bias of experiment 266 on the shape's seasonal error; at the
    farms the shape under-predicts April to July, the opposite sign (Check 6, figure 5).
  - >-
    Expect the stack's hard clip at the pinned capacity to cut the top of the curve 7 to 18 points
    below the measured 99th percentile at every farm; test a higher or softer clip (Check 2, figure 2).
---

---

**Our solar curve is sound at three of the six metered farms and wrong at the other three in
different directions, so no single flat-top fix repairs it; the one error common to all six farms is
the stack's 35 degree panel tilt.** NGED meters the output of six solar farms, labelled A to F. We
compared the curve that two of our engines use to turn sunshine into power (the per-farm plant fit
of [experiment 218](../00218-metered-solar-farm-plant-fit/result.md), and the solar term of the
best-guess demand stack, [experiment
139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md)) with the farms' measured
output on the 12 months after the engines were tuned. We predicted a departure above 5 points of
capacity at 4 or more of the 6 farms; it happened at 3 (A, B, and E), so the prediction is not
supported. The stop rule (stop if no departure exceeds 3 points at 5 of 6 farms) did not fire,
because 5 or 6 farms exceed 3 points in some bin, so check 7 ran.

![Figure 1: residual by irradiance bin per farm](figure_1_residual_by_irradiance.svg)

![Figure 2: flat-top level](figure_2_flat_top.svg)

![Figure 3: low sun, clear midday, temperature](figure_3_sun_and_temperature.svg)

![Figure 4: best tilt per farm](figure_4_orientation.svg)

![Figure 5: seasonal shape error](figure_5_seasonal_shape.svg)

![Figure 6: the pooled shape against the engines](figure_6_fix_test.svg)

## Which curve each engine uses

- **Plant fit (experiment 218).** Plane-of-array irradiance is the direct beam projected on the
  plane, plus isotropic sky diffuse, plus ground reflection (albedo 0.2). Cell temperature is air
  temperature plus 25 K times the plane's irradiance in units of 1000 W/m². DC power is a fitted
  scale times that irradiance times one minus 0.4% per kelvin above 25 C (fleet shape: tilt 35
  degrees, facing south) or a fitted loss per farm. A soft clip (a p-norm minimum, exponent 6) with
  a fitted AC limit follows.
- **The stack's solar term.** The shape is the fleet plant (tilt 35 degrees, facing south, DC:AC
  1.15) with a hard clip at 1.0 per MW of AC capacity. The term is a pinned capacity times that
  shape times the same temperature loss, applied after the clip. Here the capacity is one fitted
  scale per farm.

## Results

- **Check 1, residual by irradiance (figure 1).** Departures above 5 points whose 95% interval
  excludes zero: farm A (engine too high by 7 to 11 points above 800 W/m² on the fleet plane), farm
  E (too high by 6 to 10 points from 500 W/m²), and farm B (engine too low by 5 to 10 points above
  600 W/m²). Farms C, D, and F stay within 5 points in every bin. The stack and the per-farm plant
  depart at the same three farms. The signs differ by farm, so one clip level cannot fix them.
- **Check 2, flat top (figure 2).** The measured 99th percentile of output over capacity above 900
  W/m² is 1.05 to 1.13. The stack's term reaches 0.95 to 0.99 (7 to 18 points lower), the per-farm
  plant 0.98 to 1.07, and the fleet shape 1.09 to 1.15.
- **Check 3, low and high sun (figure 3).** At sun elevation under 15 degrees, both engines
  over-predict at all six farms, by 2.7 to 3.6 points (per-farm plant) and 4.0 to 5.3 points
  (stack), with intervals that exclude zero. At clear midday the per-farm plant is off by 8 points
  at A and E (too high) and 5 points at B (too low).
- **Check 4, temperature (figure 3).** At 700 to 900 W/m², the stack's residual rises from the
  coolest to the warmest third of air temperatures by about 13 points at A, 10 at C, and 12 at E.
  Winter and cool weather coincide in the data, so this repeats the seasonal tilt effect of check 6
  and is not an independent temperature error.
- **Check 5, orientation (figure 4).** The best fit is south-facing at every farm, at a tilt of 20
  degrees at five farms and 15 degrees at E, against the 35 degrees both engines assume as the fleet
  shape. An east-west pair at tilt 15 fits worse than the best south-facing tilt at every farm.
  Using the best tilt in place of 35 changes the fraction of output fitted by -0.002 (A) to +0.011
  (E). The public register of renewable energy projects states location and AC capacity but not tilt
  or azimuth, so there is no register value to compare with.
- **Check 6, seasonal shape (figure 5).** On clear half-hours at 400 to 900 W/m², the ratio of
  measured to the stack's output, relative to each farm's whole year (median over farms), is 0.66 in
  December, 0.76 in January, 0.81 in November, and 1.08 in May. The per-farm plant (tilt 20 degrees)
  is within 5% of 1 in every month except December (0.81) and January (0.92), so the winter error is
  largely the tilt. Experiment 266 found the pinned capacity too high in April to June at D5 (-41%),
  D9 (-14%), and D16 (-10%). At the farms the stack under-predicts April to July by 5 to 8%, the
  opposite sign, so the spring capacity bias is not the same departure as the seasonal shape error.
  The Primaries' own solar output is not metered, so this test is indirect.
- **Check 7, pooled shape (figure 6).** The pooled leave-one-farm-out shape, fitted to the other
  five farms' tuning months with one scale per farm, scores a median fraction of output fitted of
  0.944 against 0.950 for the fleet shape (difference -0.006, 95% interval -0.013 to +0.002) and
  0.953 for the per-farm plant (difference -0.010, -0.017 to -0.005). The mean 11-quantile pinball
  loss over the six farms is 4.91 points for the pooled shape, against 4.69 for the fleet shape,
  4.70 for the stack, and 4.46 for the per-farm plant. The threshold of a 0.02 gain is not met.

## Checks

- **Median minus mean forecast per site:** not applicable, because every arm is a point forecast
  with residual-quantile bands from the tuning months. The mean-forecast error is the fraction of
  output fitted reported above.
- **Reproduction:** the fleet shape (0.950) and per-farm plant (0.953) reproduce experiment 218's
  fold 1 numbers.
- **Triggers fired:** trigger 2 (the result contradicts the prediction of 4 of 6 farms): the
  departures sit at A, B, and E and their signs differ, so the explanation tested is that the
  departures are farm-specific and not a common flat-top error. Trigger 7: experiment 280 scores a
  pooled six-farm shape inside arm D; this result found no gain at the farms, which experiment 280's
  session is told by message. No leave-one-farm-out check flips the sign of the check 7 contrast:
  five of six farms show the pooled shape below the fleet shape. No audit was run.
- **Caveats:** the irradiance axis uses the 35 degree fleet plane, which the farms do not share.
  Bins hold 100 to 400 half-hours of the scored year. Farm E has 10 tuning months. Irradiance is
  CAMS and temperature is ERA5, neither available at forecast time. The pooled shape's penalty was
  chosen at the grid's lowest values. The pinball losses have no interval.

## What the next experiment should answer

- Does a 20 degree fleet tilt in the stack's solar shape lower the stack's winter error and its
  quantile score?
- Does a higher or softer clip level in the stack's shape fix the 7 to 18 point shortfall at the top
  of the curve?
