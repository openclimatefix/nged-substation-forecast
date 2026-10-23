---
id: '00282'
slug: pv-plant-curve-audit
title: >-
  Does our solar plant model turn sunshine into power the way the six metered solar farms do?
theme: metered-solar-farms
also_relevant_to:
- differentiable-physics-embedded-generation
- solar-disaggregation
status: planned
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
---

**This is a plan; the write-up replaces it.** NGED meters the output of six solar farms, labelled A
to F. Our solar models turn sunshine into power with one assumed curve. We ask whether that curve
matches the six farms' measured curve, and if not, where it departs.

## Which curve each engine uses

- **The plant fit (experiment 218).** Plane-of-array irradiance is the direct beam projected onto
  the plane, plus isotropic sky diffuse, plus ground reflection at albedo 0.2. Cell temperature is
  air temperature plus 25 K times the plane's irradiance in units of 1000 W/m². DC power is a
  fitted scale times that irradiance times one minus 0.4% per kelvin above 25 C (fleet shape, tilt
  35 degrees, facing south) or a fitted loss per farm. A soft clip (a p-norm minimum, exponent 6)
  with a fitted AC limit follows. Night and sun lower than 5 degrees give zero.
- **The stack's solar term (experiment 139 and `studies.dp_gam`).** The unit shape is the same
  plane-of-array irradiance for the fleet plant (tilt 35, facing south, DC:AC 1.15) with a hard
  clip at 1.0 per MW of AC capacity, driven by CAMS global irradiance at the substation. The solar
  term is a pinned capacity times that shape times `1 - 0.004 (T_cell - 25)`, with the same cell
  temperature rise. Differences from the plant fit: a hard clip at DC:AC 1.15 instead of a fitted
  soft clip, the temperature loss applied to the clipped shape, and one capacity per season-year.

## Checks

Each check is one figure; farms appear by letter only. All tables use fold 1 of experiment 218:
engine parameters tuned on the months before 2025-09, residuals on the 12 months after, with
intervals from resampling those 12 months as blocks (10,000 draws, seed 282). The irradiance axis
is the plane-of-array irradiance of the fleet plane (CAMS). Bins are 100 W/m² wide.

1. Measured output against irradiance, with both engines' curves overlaid, by season (March to
   May, June to August, September to November, December to February) and by morning against
   afternoon (before and after solar noon). Residual by bin, with intervals.
2. Clipping: the 99th percentile of measured output over capacity in the top irradiance bins,
   against the engine's AC limit (the fitted limit; 1.0 for the stack's shape scaled by capacity).
3. Low sun (elevation under 15 degrees) and clear-sky midday (clear-sky index above 0.9, elevation
   above 40 degrees): residual against the engines.
4. Temperature: residual against air temperature in fixed irradiance bins (700 to 900 W/m²), in
   terciles of air temperature.
5. Orientation: tilt 0 to 60 degrees in steps of 5 and azimuth 90 to 270 degrees in steps of 10,
   fitted freely per farm with no heat loss and the soft clip, against the assumed 35 degrees south
   and experiment 218's coarse-grid choice. An east-west pair at tilt 15 is added as a candidate.
   The public register of renewable energy projects states location and AC capacity but not tilt
   or azimuth, so there is no register value to compare with, and the page says so.
6. The stack's solar term: the engine's shape times each Primary's pinned capacity, for D5, D9, and
   D16, by month. The Primaries' own solar output is not metered, so the test is indirect: the
   measured-over-modelled ratio of the farms by month at fixed irradiance gives the seasonal shape
   error, which is multiplied by the pinned capacity and compared with experiment 266's capacity
   bias (-41%, -14%, -10% in April to June).
7. Fix test (runs only if the stop rule does not fire): replace the engine's curve with the pooled
   leave-one-farm-out empirical shape of experiment 269 (a table of output over irradiance and
   elevation from the other five farms), tuned only on the months before 2025-09, and report the
   fraction of output fitted and the pinball loss at 11 quantiles (residual quantiles of the
   tuning months added to both curves) on the scored year. Experiment 280 (session
   `session_01QkF8sGfTpHcdb8oqWSTmjg`) scores the same shape inside arm D of the stack; this
   experiment scores it at the farms only and does not duplicate that stage.

## Weather

Irradiance is CAMS (about 2 days late) and temperature is ERA5 (about 5 days late). Neither exists
at forecast time, so every number is a ceiling for the forecast and a clean test of the curve.

## Schedule

Four CPU cores, one thread per process, no GPU, no seeds beyond the bootstrap's.
