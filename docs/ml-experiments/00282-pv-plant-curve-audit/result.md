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
  The solar models' curves depart by more than 5 percentage points of capacity at 3 of 6 farms (A,
  B, and E), not the 4 predicted. All six farms fit best at a 15 to 20 degree tilt against the
  stack's 35 degrees; that error is large in winter and small in the annual fit (0.002 below to
  0.010 above).
production_status: not-for-production
adopted_in: none
take_home_message: >-
  The curve that turns sunshine into solar power stays within 5 percentage points of capacity at 3
  of the 6 metered solar farms and departs from their measured output at the other 3 in different
  directions, so no single fix to the inverter limit repairs it. The best-guess demand stack assumes
  panels tilted 35 degrees and the farms fit best at about 20 degrees, which makes the stack
  over-predict output on clear half-hours in December (measured is 0.66 of the stack, median over
  farms) and under-predict it from April to July by 5 to 8%. Swapping in the pooled shape of the
  other farms does not help, so the next experiment should try the 20 degree tilt inside the stack.
advice:
  - >-
    Test a 20 degree south-facing fleet tilt in the stack's solar shape in place of 35 degrees,
    because all six farms fit best at 15 to 20 degrees and the 35 degree shape over-predicts
    output on clear December half-hours (measured is 0.66 of the stack) (Check 6, figure 5).
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

**Our solar curve stays within 5 percentage points of capacity at three of the six metered farms and
departs by more at the other three, in different directions, so no single fix to the inverter limit
repairs it.** The one error common to all six farms is the panel tilt of 35 degrees that the
best-guess demand stack assumes. National Grid Electricity Distribution (NGED) meters the output of
six solar farms, labelled A to F. We compared the farms' measured output over September 2025 to
August 2026 with the curve that turns irradiance (sunshine, in W/m²) into power in two of our solar
models. The two models are the per-farm plant fit of [experiment
218](../00218-metered-solar-farm-plant-fit/result.md) and the solar term of the best-guess demand
stack (the "stack") of [experiment
139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md). Both were tuned on
earlier months. A "departure" is a mean error above 5 percentage points of the farm's capacity whose
95% interval excludes zero. We predicted departures at 4 or more of the 6 farms; they occurred at 3
(A, B, and E), so the prediction is not supported. The count is fragile to the bin edges: farm D's
fleet-shape error reaches 4.95 points in one bin. The stop rule would have ended the audit before
the pooled-shape test (check 7) if fewer than 5 of 6 farms had a bin with a mean error above 3
points. Five farms had such a bin in the per-farm plant fit and six in the stack, so check 7 ran.

![Figure 1: measured minus modelled output by irradiance bin, one panel per farm](figure_1_residual_by_irradiance.svg)

![Figure 2: the 99th percentile of output at high irradiance, measured and modelled](figure_2_flat_top.svg)

![Figure 3: residuals at low sun, clear midday, and by air temperature](figure_3_sun_and_temperature.svg)

![Figure 4: best-fitting tilt per farm against the assumed 35 degrees](figure_4_orientation.svg)

![Figure 5: measured over modelled output by calendar month](figure_5_seasonal_shape.svg)

![Figure 6: fraction of output fitted per farm for each shape](figure_6_fix_test.svg)

## Which curve each solar model uses

- **Plant fit (experiment 218).** Plane-of-array irradiance (the irradiance on the tilted panel
  surface) is the direct beam projected on the plane, plus isotropic sky diffuse, plus ground
  reflection (albedo 0.2). Cell temperature is air temperature plus 25 K times the plane's
  irradiance in units of 1000 W/m². DC power is a fitted scale times that irradiance times one minus
  0.4% per kelvin above 25 °C (fleet shape: tilt 35 degrees, facing south). The per-farm plant
  instead fits its own loss for each farm. A soft clip (a smooth cap on output, a p-norm minimum
  with exponent 6) with a fitted limit on alternating-current (AC) power follows.
- **The stack's solar term.** The shape is the fleet plant (tilt 35 degrees, facing south, a
  direct-current to AC rating ratio of 1.15) with a hard clip (an abrupt cap) at 1.0 per MW of AC
  capacity. The term is a pinned capacity times that shape times the same temperature loss, which is
  applied after the clip. Here the capacity is one fitted scale per farm.

## Results

- **Check 1, residual by irradiance (figure 1).** Departures above 5 percentage points of capacity
  whose 95% interval excludes zero: farm A (solar model too high by 7 to 11 points in the 800 and
  900 W/m² bins on the fleet plane), farm E (too high by 6 to 12 points, in the 500 to 900 W/m² bins
  for the per-farm plant and in the 800 and 900 bins for the fleet shape), and farm B (solar model
  too low by 5 to 10 points, at 600 to 900 W/m² for the stack and in fewer bins for the other two).
  Farms C, D, and F stay within 5 points in every bin. The stack and the per-farm plant depart at
  the same three farms. The signs differ by farm, so one clip level cannot fix them.
- **Check 2, flat top (figure 2).** The measured 99th percentile of output over capacity above 900
  W/m² is 1.05 to 1.13. The stack's term reaches 0.95 to 0.99 (7 to 18 points lower), the per-farm
  plant 0.98 to 1.07, and the fleet shape 1.09 to 1.15.
- **Check 3, low and high sun (figure 3).** At sun elevation under 15 degrees, both solar models
  over-predict at all six farms, by 2.7 to 3.6 points (per-farm plant) and 4.0 to 5.3 points
  (stack), with intervals that exclude zero. At clear midday the per-farm plant is off by 8 points
  at A and E (too high) and 5 points at B (too low).
- **Check 4, temperature (figure 3).** At 700 to 900 W/m², the stack's residual rises from the
  coolest to the warmest third of air temperatures by about 13 points at A, 10 at C, and 12.5 at E.
  The same rise appears at the other three farms (5 to 8 points). Cool air and winter coincide in
  the data and no table here separates them, so the temperature sub-prediction (residuals under 3
  points) is not answered.
- **Check 5, orientation (figure 4).** The best fit is south-facing at every farm, at a tilt of 20
  degrees at five farms and 15 degrees at E, against the 35 degrees both solar models assume as the
  fleet shape. An east-west pair at tilt 15 fits worse than the best south-facing tilt at every
  farm. Using the best tilt in place of 35 changes the fraction of output fitted (both fitted here
  with no heat loss) by -0.002 (A) to +0.010 (E). The public register of renewable energy projects
  states location and AC capacity but not tilt or azimuth, so there is no register value to compare
  with.
- **Check 6, seasonal shape (figure 5).** On clear half-hours at 400 to 900 W/m², the ratio of
  measured to the stack's output, relative to each farm's whole year (median over farms), is 0.66 in
  December (0.53 to 0.71 across farms), 0.76 in January, 0.81 in November, and 1.08 in May. The
  per-farm plant (best tilt 20 degrees, fitted heat loss and soft clip) is within 5% of 1 in every
  month except November (0.95), December (0.81), and January (0.92). No arm isolates the tilt, so
  the winter error is partly the tilt; the fit may also absorb error in the irradiance data.
  Experiment 266 found the stack's pinned capacity too high in April to June at three of NGED's
  primary substations (D5 -41%, D9 -14%, D16 -10%). At the farms the stack under-predicts April to
  July by 5 to 8%, the opposite sign, so the spring capacity bias is not the same departure as the
  seasonal shape error. The primary substations' own solar output is not metered, so this test is
  indirect.
- **Check 7, pooled shape (figure 6).** The pooled leave-one-farm-out shape (fitted on the other
  five farms and scored on the sixth, for each farm in turn), fitted to the other five farms' tuning
  months with one scale per farm, scores a median fraction of measured output explained (one minus
  the squared error over the squared output) of 0.944 against 0.950 for the fleet shape (difference
  -0.006, 95% interval -0.013 to +0.002) and 0.953 for the per-farm plant (difference -0.010, -0.017
  to -0.005). The pinball loss (a quantile-forecast score; lower is better) averaged over 11
  quantiles and the six farms is 4.91 points for the pooled shape, against 4.69 for the fleet shape,
  4.70 for the stack, and 4.46 for the per-farm plant. The threshold of a 0.02 gain is not met.

## Checks

- **Median minus mean forecast per site:** not applicable, because every arm is a point forecast
  with residual-quantile bands from the tuning months. The mean-forecast error is the fraction of
  output fitted reported above.
- **Reuse of experiment 218:** the fleet shape (0.950) and per-farm plant (0.953) use experiment
  218's stored fold 1 predictions, so their numbers match by construction.
- **Triggers fired:** trigger 2 (the result contradicts the prediction of 4 of 6 farms): the
  departures sit at A, B, and E and their signs differ, so the explanation tested is that the
  departures are farm-specific and not a common flat-top error. Trigger 7 (a later experiment
  depends on this result): experiment 280 scores a pooled six-farm shape inside arm D; its session
  has been told by message that the pooled shape gave no gain at the farms. The pooled shape scores
  below the fleet shape at five of the six farms (no farm-dropping check was run). No audit was run.
- **Caveats:** the irradiance axis uses the 35 degree fleet plane, which the farms do not share.
  The bins used for flagging hold 147 to 1,608 half-hours of the scored year, and check 6 uses all
  years of data.
  Farm E has 10 tuning months. Irradiance is from the Copernicus Atmosphere Monitoring Service
  (CAMS) and air temperature is from the ERA5 reanalysis, and neither is available at forecast time.
  The pooled shape's penalty took the lowest values of its search range, so a weaker penalty might
  score differently. The pinball losses have no interval, and the pooled shape's quantile offsets
  come from residuals of a shape it was not fitted to, while the other shapes' offsets are
  in-sample.

## What the next experiment should answer

- Does a 20 degree fleet tilt in the stack's solar shape lower the stack's winter error and its
  quantile score?
- Does a higher or softer clip level in the stack's shape fix the 7 to 18 point shortfall at the top
  of the curve?
