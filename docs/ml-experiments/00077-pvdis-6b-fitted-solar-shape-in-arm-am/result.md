---
id: "00077"
slug: pvdis-6b-fitted-solar-shape-in-arm-am
status: planned
size: medium
hypothesis: >-
  At the 5 Primary substations (33/11 kV or 66/11 kV substations in the distribution network) of
  experiment 00064, replacing the all-south solar shape of arm `am` with a fitted solar shape (a
  blend of east-, south-, and west-facing output, one set of shares per Primary, fitted on the
  training window only and held fixed across training years) lowers the quantile score of the
  forecast, and the gain comes from the timing of the forecast sunshine and not from the shape
  alone.
prediction: >-
  The fitted shape lowers the score of `am` by 0.2% to 0.8% of its score (0.01 to 0.04 points of
  capacity), because experiment 00073 found a 1.3% lower held-out fit error on the same Primaries
  and the solar term accounts for roughly a quarter of the improvement that `am` brings over
  `d0` (0.152 points of 4.620). The shuffled-shares placebo is expected to score within 0.01 points
  of the fitted shape, because the shares are not identified (experiment 00073) and the gain is
  expected to come from having some east and west output. The placebo twin (forecast sunshine of
  another date) is expected to score 0.2 to 0.4 points worse, as in experiment 00064. Probability
  that the upper end of C1 falls below -0.022 (the `supported` condition on C1): 0.35.
decision_rule: >-
  Three verdict contrasts, each the quantile score of the first arm minus that of the second, in
  points of capacity, at a 98.33% interval (a Bonferroni level for three contrasts) from 50,000
  resamples of whole ISO weeks of forecast start and of seeds, on experiment 00064's rows
  (`fold_a_every5d_v1`, seeds 0, 1, 2, the 5 verdict series, daylight April to September). C1 is
  `am_blend` minus experiment 00064's `am`. C2 is `am_blend` minus `am_s25`, the all-south shape at
  the blend's tilt of 25 degrees, which separates the effect of mixing orientations from the effect
  of the tilt change. C3 is `am_blend` minus `am_blend_twin`, the same arm with the forecast
  sunshine of another date (experiment 00064's twin construction). The threshold is 0.5% of `am`'s
  score of 4.467, which is 0.022 points. `supported` when the upper end of C1 is below -0.022, the
  upper end of C2 is below 0, and the upper end of C3 is below 0. `small gain` when the upper end
  of C1 is below 0 and the rest of the `supported` rule is not met. `refuted` when the lower end of
  C1 is above -0.022 and the upper end is at most +0.022, so that a gain of 0.5% is excluded and a
  loss of 0.5% is not shown. `worse` when the lower end of C1 is above +0.022. `inconclusive`
  otherwise. The three-contrast level is conservative for the `supported` rule, which is an
  intersection of three reads. The paired contrast of `am` against the input-column arm `a` in
  experiment 00064 had a half-width of 0.036 points at 98.75%, so C1 resolves a gain of about 0.04
  points, and a true gain of 0.022 would most likely read as `inconclusive`; a power calculation
  from experiment 00064's stored predictions is printed before the fits and does not change the
  rule. The 0.5%
  threshold is a judgement: porting the fitted shape adds a table of three shares per Primary and
  a three-orientation calculation to the capacity and solar-shape steps, and a smaller gain than
  the seed-and-week noise of a half-year contrast would not justify the table. Every other read is
  exploratory, labelled so, at 95%: the shuffled-shares contrast `am_blend` minus `am_shuffled`
  (the shares of another Primary, a fixed cyclic permutation of the 5 series), the contrast by lead
  band and by valid month, the per-series contrasts, and the evening-west check below. The result
  is read against the separate check that experiment 00064's gain beyond 10 days (-0.082 points
  where the solar weight is zero) is not leakage; a gain that sits only at leads beyond 10 days is
  not reported as a gain of the fitted shape.
benchmark: >-
  Experiment 00064's `am` (stored predictions, no refit), `am_twin`, `d0`, `climatology`, and
  `manual_heuristic` on the same rows and seeds.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity, after the 14-day correction and the spread factor, as in experiment
  00064.
code: ml-experiments/00077-pvdis-6b-fitted-solar-shape-in-arm-am/
---

# Experiment 77: does the fitted solar shape lower the error of arm am? (PVDIS-6b)

**Status: planned. The page is the plan, committed before any code runs.**

We asked whether the forecast of experiment 64 (PVDIS-5) improves when its solar term is built from
a blend of east-, south-, and west-facing panel output instead of from south-facing panels only. A
Primary substation is a 33/11 kV or 66/11 kV substation in the electricity distribution network.
Experiment 73 (PVDIS-6) found that a blend of three shapes fits the half-hourly power of the 8
Primaries in MW 1.3% better than south-facing panels, although the shares of the blend are not
identified. We rebuild the solar term of experiment 64's best arm `am` with the blend, and score the
new forecast on the same rows. The page has no result yet.

## What is held fixed and what changes

**One set of shares per Primary, fitted on the training window only, replaces the all-south shape.**
Experiment 73's pooled fit (one set of east, south, and west shares per Primary, shared by every
season-year) is refitted on summer half-hours up to 2025-06-30 only, on experiment 73's
plane-of-array shapes at a tilt of 25 degrees and a DC to AC ratio of 1.15. Experiment 73's own
shares used rows up to 2026 and are not reused. The shares are held fixed across training years,
which removes the year-to-year swing of 0.24 that experiment 73 found.

**The blend replaces the all-south shape everywhere that the solar term is built.** The CAMS shape,
the CAMS and ENS climatologies, the ENS anomaly and its calibration slope per lead band, the
capacity, and the weight `w` fitted on the training window are all rebuilt for the blend, by the
code of experiments 46 and 64 with the plant output replaced by the blend of the three orientations'
output. The per-member ENS anomaly is computed from the shortwave radiation of each of the 51
members, as in experiment 46. The weight `w` absorbs a scale error in the capacity, so the capacity
keeps experiment 45's recovery ratio of the fleet shape; the effect of that choice is a caveat of
the write-up.

## Arms

- `am_blend`: arm `am` with the fitted shape.
- `am_s25`: arm `am` with all-south panels at a tilt of 25 degrees, the control for the tilt change.
  The script prints the AC capacity of `am_s25` and of `am_blend` for each series, and the
  weights `w`, so that a difference in level between the arms is visible.
- `am_blend_twin`: `am_blend` with the forecast sunshine taken from another date, by experiment 64's
  permutation of runs within series, month, half-hour, and lead band. The donor run is the same for
  the east, south, and west anomalies, so the twin keeps their mutual timing.
- `am_shuffled`: `am` with the shares of another Primary (a fixed cyclic permutation of the 5
  series), refitted capacity and weight. This is exploratory.
- Reference rows, read from experiment 64's stored predictions: `am`, `am_twin`, `d0`, `d1l`,
  `climatology`, and `manual_heuristic`.

Each XGBoost arm has 5 series and 3 seeds, 15 fits, with experiment 64's settings and column list.
Each model is trained on its own history.

## No fitted quantity sees the validation period

**Every fitted quantity is read with a valid time up to 2025-06-30 23:30 UTC, and the scripts assert
it.** The quantities are the shares, the AC capacity, the CAMS and ENS climatologies, the
calibration slope per lead band, the weight `w`, and the 14-day correction and spread factor, which
are refitted per arm on training rows only. Each script prints the latest timestamp of every table
it fits. The shares are fitted on rows of `shapes.parquet` and the frame up to that time. The ENS
members' half-hour irradiance is read from the archive and passed through the plane-of-array
calculation for the east, south, and west orientations at 25 degrees, since the stored ENS tables
hold only the fleet shape. The reliable-hour mask of the CAMS shapes must give the same scored rows
as experiment 64's; where it does not, all arms are restricted to the intersection, and `am` is
rescored on it.

## The evening-west check

**A west-leaning share at some Primaries may be evening demand and not solar.** Experiment 73's
advice recommends a check before the west share is read as solar. For each Primary, using training
rows only, we remove the calendar cell and heating degree-days from the half-hours after dark
(solar zenith above 90 degrees, 18:00 to 23:30 local time), and regress the remainder on that
evening's mean cloud cover and on its mean temperature. If evening demand were the source of a west
share, the after-dark residual would rise in dull weather, so a positive cloud slope is the
expected sign. We tabulate, for each of the 5 Primaries, the slope as a share of the Primary's
capacity beside its west share. With 5 Primaries, no rank correlation has power, so the table is
descriptive, and no verdict uses it.

## Size

- **What gets stored:** prediction sets under `data/ml-experiments/` (private, uncommitted); nothing
  in production.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes, in how the shares are fitted (fixed, or under a
  year-to-year penalty) and at which tilt. The plan fixes the shares across years because that
  choice removes the instability experiment 73 found, and uses `am_s25` to separate tilt from mix.
- **Callers not nameable without searching:** the code is copied from experiments 46 and 64, so the
  callers are known.
- **Size: medium.** One Sonnet plan review, one Sonnet code review, one Sonnet write-up review, and
  one Sonnet prose review.

## Changes made after the plan was committed and before any arm was scored

- **The flat-timing control `am_blend_flat` is dropped.** A reviewer of the code showed that a
  single scale on the south shape cancels: the calibration slope, the capacity, and the weight `w`
  absorb it, so the arm would equal `am_s25`. A per-cell version was not built.
- **The evening-west check regresses on the day's mean daytime cloud and heating degree-days.**
  The frame holds no cloud measure after dark, because irradiance is zero then, so the cloud
  variable of the plan's "that evening" is the same day's daylight cloud.
- **The fallback capacity of the series with no 2024/25 constant (D14) is scaled by the shape.** It
  is experiment 64's capacity times the median ratio of the shape's capacity to experiment 64's at
  the other four series.
- **Exploratory contrasts against `am0` and `am_sy` are added,** at the request of the manager,
  because experiment 78 found that a zero base margin alone lowers `d0`'s score. `am0` is
  experiment 76's `am` without the solar term, and `am_sy` is experiment 78's `am` with a
  season-year capacity vintage.

## Earlier advice followed and departed from

- Followed: use `am` (a starting offset) and not an input column or a shift after the fit
  (experiment 64); describe the blend as a fitted solar shape and not as a measurement of
  orientation, and do not publish shares (experiment 73); hold the shares fixed across training
  years (experiment 73); check the west lean against after-dark demand (experiment 73).
- Departed from: none.
