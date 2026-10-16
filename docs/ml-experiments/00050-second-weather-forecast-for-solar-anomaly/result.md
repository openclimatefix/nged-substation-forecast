---
id: "00050"
slug: second-weather-forecast-for-solar-anomaly
status: planned
size: complex
hypothesis: >-
  At the five Primary substations metered in MW in experiment 00046's verdict set (D9, D10, D11,
  D14, and D16), an anomaly forecast that blends the NOAA GEFS ensemble mean with the ECMWF ENS
  ensemble mean, with weights fitted on the training window only, tracks the CAMS solar anomaly more
  closely than ENS alone. Laid into experiment 00046's solar term with its fitted lead-dependent
  coefficient, the blended anomaly lowers the quantile score by at least 0.1 points of capacity
  against the same term built from ENS alone.
prediction: >-
  GEFS alone correlates with the CAMS anomaly less well than ENS in every lead band: about 0.68 at 0
  to 24 hours (ENS 0.79), 0.63 at 1 to 2 days (ENS 0.75), 0.50 at 2 to 6 days (ENS 0.58), and 0.20 at
  6 to 10 days (ENS 0.27). The blend gains +0.01 to +0.03 in correlation at 0 to 6 days and about 0 at
  6 to 10 days. In the demand forecast, C7 is -0.02 (-0.06 to +0.01) and C7s is -0.04 (-0.10 to
  0.00). The likely verdict is `refuted`, decided at stage 1. If the demand stage runs, a C7s
  interval of -0.10 to 0.00 would read `inconclusive`, and one lying wholly below 0 would read
  `small gain`. The basis: in multi-model irradiance comparisons ECMWF usually leads, and adding a
  weaker model helps only where the two models' errors are not correlated.
decision_rule: >-
  The verdict is set in two stages. Stage 1, the anomaly selection gate, uses the training window
  only. Fifteen-fold month-block cross-validation, one fold per training valid month from April 2024
  to June 2025, fits every calibration on the other 14 months (including each source's climatology,
  the daylight mask, the slopes, and the blend weights) and predicts the held-out month. It scores
  the mean squared error of each of the three calibrated anomalies against the CAMS anomaly on
  daylight rows at leads of 0 to 240 hours, pooled over the five verdict series. The arm with the
  lowest error is the best anomaly. The blend or GEFS replaces ENS only if its error is at least 1%
  lower than ENS's. If ENS wins, the demand stage is not run, the verdict is `refuted`, and the
  validation anomaly table is the evidence. Stage 2, the demand contrasts. C7 (all scored rows) and
  C7s (scored rows at leads of 0 to 48 hours) each compare the selected arm's solar term with the
  ENS solar term, are read at 97.5% (Bonferroni across two), and decide the verdict. The verdict is
  `supported` when either interval lies wholly below -0.1, `small gain` when either interval lies
  wholly below 0 but neither lies wholly below -0.1, `refuted` when both intervals lie wholly above
  -0.1 and neither lies wholly below 0, and `inconclusive` otherwise. The verdict is also reported
  without D9, as in experiment 00046. The threshold of 0.1 points (about 2% of `d0`'s 4.620) is set
  by the cost of adopting the blend, which adds a second NWP ingest from Dynamical.org with its own
  Delta table, schedule, and degradation path for a late GEFS run, plus a blend table to refit. The
  threshold matches PVDIS-5's non-inferiority margin.
benchmark: >-
  `d1l_ens_shift_fitted` of experiment 00046: the same `d1l` boosters, the same member spread, and
  the same coefficient, with the ENS central anomaly.
row_set_id: fold_a_every5d_v1
metric: >-
  Stage 1: Pearson correlation, calibration slope, and mean-squared-error skill of each calibrated
  anomaly against the CAMS anomaly, by lead band. Stage 2: experiment 00046's metric, the quantile
  score (twice the mass-weighted mean pinball loss over nine levels) in points of capacity after the
  14-day correction and the spread factor, on its scored rows (daylight April-to-September rows with
  a CAMS unit PV shape and an observed power)
production_status: not-for-production
adopted_in: none
code: ml-experiments/00050-second-weather-forecast-for-solar-anomaly/
---

# Experiment 50: does a second weather forecast give a better forecast of the solar anomaly than ENS?

**We ask whether a second weather forecast, from the American NOAA Global Ensemble Forecast System
(GEFS), predicts how much sunnier or cloudier than usual a day will be better than the European
ECMWF ensemble forecast (ENS) that the demand forecast uses today.** Experiment 46 found that a
correction for embedded solar panels lowered the demand forecast's error by about 20% when it used
the true sunshine, taken from the satellite-based CAMS data, and by almost nothing when it used the
ENS forecast of sunshine. The sunshine forecast, not the panels' capacity, limits the gain. We
first score the two forecasts, and a weighted blend of them, against the CAMS sunshine at the five
substations metered in MW, on the April-to-September forecasts of the validation year that
experiment 46 scored. We then
lay the best of the three into experiment 46's correction, without refitting any demand model, and
check whether the demand forecast's error falls by at least 0.1 points (a point is 1% of the
substation's capacity). We expect ENS to stay at least as good, because ECMWF usually leads in
multi-model comparisons of irradiance. The experiment has not run yet.

## Plan

**Sizing.** Each of the five triggers gets one line.

- **What gets stored:** private intermediate tables and about five prediction sets under
  `data/ml-experiments/`; no production table changes.
- **The production serving path:** not touched. Adoption would add an ingest, which a later
  production issue would own.
- **A degradation rule:** not touched. Adoption would need a rule for a late or missing GEFS run,
  which a later issue would own.
- **More than one defensible design:** yes. The choices are the product, the blend form, same-day or
  previous-day runs, the climatology window, and how the member spread is handled. This plan fixes
  each.
- **Code whose callers could not be named without searching:** no. The scripts copy named
  experiment 46 helpers, call `_upsample_nwp_to_half_hourly` as experiment 46 does, and call
  `geo.h3.compute_h3_grid_weights`.

The size is complex. The process is a plan review, a code review for leakage (training-window
cut-offs, the selection running before the predictions, and the choice of GEFS run), a run and
write-up with a prose review, and a review of the result.

**Earlier advice followed.** Experiment 46's advice is followed in four places: fit the solar
coefficient by lead band instead of fixing it at 1; limit any gain test to leads of 0 to 48 hours
(C7s); check the sign of a contrast in every valid month before reading its pooled interval; and
compare arms that share one fitted booster. The experiment departs from none of it.

**The second product is GEFS, with GFS as the one fallback.** GEFS (31 members, 00 UTC runs, 0.25
degrees) is on disk for 2020-10 onward, comes from Dynamical.org like the ENS ingest, and was the
same operational GEFS v12 product throughout. NOAA's deterministic GFS is the fallback if GEFS
fails the latency check or a data-validation check; its radiation windows run from 1 to 6 hours and
need a running-mean de-accumulation, which this plan does not implement unless the fallback is
used. ECMWF's AIFS model is ruled out: its shortwave field is missing before 2025-02-24, and two
model versions fall inside the validation year. The CEDA UKV archive and the Open-Meteo archives
are ruled out for different reasons: the CEDA UKV differs from the live UKV, and the Open-Meteo
archives are held at the generator meter points and reach at most 5 days. No download is needed.

**GEFS radiation is a 6-hour mean at 00, 06, 12, and 18 UTC and a 3-hour mean at 03, 09, 15, and 21
UTC, so the means are de-accumulated before use.** The 3-hour mean ending at a 6-hourly label `L`
is `2 * v(L) - v(L - 3 h)`. Means below zero are clipped to zero and counted (the source is
quantised, and the first look found 5.2% of 3-hour means below -1 W/m²). Beyond a lead of 144 hours
pairs of 3-hour means are averaged into 6-hour means, which gives the same schedule as ENS's native
radiation windows, so the conversion passes the production resampler's schedule check. Each
Primary's H3 cell is area-weighted over the GEFS grid cells; for D16 only 0.898 of the cell lies
inside the downloaded crop, so its weights are renormalised, and the write-up says so.

**The run used depends on a latency check made before any score.** Production's forecast origin is
the 00 UTC run plus 9 hours. `e00050_latency.py` records, on three or more dates between 07:00 and
09:00 UTC, whether the store already holds that day's 00 UTC GEFS run. If it always does, the arms
use the same-day run. Otherwise they use the previous day's run, and lead bands stay measured from
the forecast origin. The decision is written to `results/latency_decision.json` before any GEFS
input is built.

### Stage 1: the anomaly

**Three calibrated anomalies are scored against the CAMS anomaly.** The anomaly of a source is its
unit PV shape `u` minus its own month-by-half-hour climatology over the training window. The `ens`
arm is slope times the ENS ensemble-mean anomaly (experiment 46's). The `gefs` arm is slope times
the GEFS ensemble-mean anomaly. The `blend` arm regresses the CAMS anomaly on both ensemble-mean
anomalies with two weights and no intercept. Slopes and weights are fitted per site and lead band;
a negative weight is kept and flagged. An equal-weight average of the two calibrated anomalies is an
exploratory fourth arm and takes no part in the selection.

**The gate and the scores use different data.** The selection gate is the cross-validation of the
decision rule and uses the training window only. The validation-year scores fit once on the whole
training window and report, by lead band and pooled over the five verdict series: the correlation
(A1), the calibration slope (A2), and the mean-squared-error skill against a zero anomaly (A3).
The difference in correlation between each arm and ENS has a 95% interval from resampling the 7
initialisation months of the runs. These reads are exploratory and decide nothing. A second
exploratory read repeats A1 on 3-hour mean anomalies, free of any difference in resampling shape.

### Stage 2: the demand contrasts

**Every demand arm is a shift of experiment 46's saved `d1l` boosters, so nothing is refitted.**
The shift of ENS member `m` is `-K(site, band) * C * (central + (a_m - mean(a))) / capacity`, where
`C` is experiment 40's 2023/24 capacity, `K` is experiment 46's fitted lead-band coefficient (1 in
bands where none was fitted), `central` is the arm's calibrated anomaly, and `a_m - mean(a)` is
the raw ENS member's deviation from the ENS mean. The arms differ only in `central`. The 14-day
correction and the spread factor 1.15 are applied to each arm as in experiment 46.

| Id | Contrast | Read | Predicted |
|---|---|---|---|
| C7 | selected arm's shift minus ENS shift, all scored rows | decides, 97.5% | -0.02 (-0.06 to +0.01) |
| C7s | the same, scored rows at leads of 0 to 48 hours | decides, 97.5% | -0.04 (-0.10 to 0.00) |
| C3b | selected arm's shift minus `d1l` | exploratory, 95% (experiment 46's fitted C3 was -0.143) | -0.16 |
| C6b | selected arm's shift minus `d0`, all leads and 0 to 48 hours | exploratory, 95% | +0.02, and -0.05 at 0 to 48 hours |

**C7 compares two shifts of one booster, so it has no GPU fit noise but also little contrast.** The
difference between the two central anomalies is expected to have about a third of the standard
deviation of the ENS shift. Scaling experiment 46's C3 intervals by that third gives an expected
97.5% half-width of 0.03 to 0.05 points for C7 and 0.05 to 0.10 for C7s. C7s can therefore clear
-0.1 only if the true gain is about 0.15 to 0.2, roughly three times the predicted gain. The
statistics script prints the realised half-widths. PVDIS-5's test is a gain test at 0 to 48 hours
only if C6b is wholly below zero there.

**Exploratory reads (95%, labelled so):** C7 by lead band, by valid month (the sign must match in
every month before a pooled read is believed), by series, and without D9; the other arms' shifts,
including the equal-weight blend; the coverage of the 0.05 and 0.95 quantiles in sunny and cloudy
rows; and a week-block bootstrap.

### Checks before any read

**Data validation of GEFS** (`e00050_validate_gefs.py`, with each number printed):

1. Every 00 UTC run from 2024-04-01 to 2026-06-30 is present with 31 members, the expected lead
   times, and 9 grid cells, with no duplicate keys; every `NaN` shortwave value after lead 0 is
   listed.
2. De-accumulation: clipping removes under 1% of the positive energy, windows that are dark
   throughout hold under 10 W/m², and twice a 6-hour mean equals the sum of its two 3-hour halves.
3. Each native window's mean equals the mean of its half-hours to within the 13-bit quantisation
   step.
4. The centre of mass of the hour-of-day profile of GEFS irradiance in the first 48 hours lies
   within one half-hour of CAMS's, which would fail if windows were labelled by their start.
5. The true cell weights give a higher cloud-driven correlation with CAMS than the
   latitude-mirrored mapping at every site whose mapping the mirroring changes.
6. CAMS irradiance, presented as GEFS windows and passed through the GEFS chain, reproduces the
   3-hour block means of the CAMS anomaly (correlation 0.99 or more), and the half-hour
   correlation is printed against the planned 0.95.
7. The 99th percentile of the clear-sky index is at most 1.3, and the monthly ratio of mean GEFS
   to mean CAMS irradiance has no step over 15% between neighbouring months.

**Experiment checks:** (1) the rebuilt `d1l_ens_shift_fitted` matches experiment 46's stored arm to
within 1e-5 on every quantile and key; (2) a blend with weights (ENS slope, 0) reproduces the `ens`
central anomaly bit for bit; (3) all arms hold the same scored keys; (4) every climatology, slope,
weight, and fold uses valid times up to 2025-06-30 23:30 only, which the code asserts; (5) the
latency decision is recorded before stage 1 runs; (6) stage 1's selection is written before any
script reads an anomaly for the demand stage.

**Stop rule.** The experiment stops without a verdict if experiment check 1, 2, 5, or 6 fails and
one revision does not fix it. Nothing is tuned after the scores or any contrast is read: the bands,
the folds, the 1% selection margin, `K`, `C`, and the threshold are fixed here.

### Compute and amendments

**The run fits in 4 cores, and the GPU is optional.** Reading and converting GEFS takes minutes
(four worker processes, one Polars thread each). Prediction from the boosters runs on the CPU, one
thread per worker, at about 1.5 to 3 minutes per series-and-seed job; the shifts and the bootstrap
take minutes.

**Amendments from the plan review, accepted:** clip the de-accumulated means at zero and state the
clip; add "neither wholly below zero" to `refuted`, so that `small gain` and `refuted` cannot both
hold; fit the ENS calibration, each climatology, and the blend weights inside each fold; check the
publication time of the GEFS run; and verify the D16 crop coverage in the private store. **Rejected:**
skipping the demand stage, because the brief requires repeating experiment 46's C3 and C6, and a
fixed 50/50 blend as the only blend, because the fitted blend is one of the three pre-registered
arms (the fixed blend is added as an exploratory arm). **Open:** the Bonferroni correction across
C7 and C7s widens both intervals; both stay at 97.5% unless the maintainer rules otherwise.

**Implementation notes.** Check 6 gates on the 3-hour block means because averaging CAMS to 3-hour
windows discards its within-window cloud detail, which caps the half-hour correlation near 0.94 to
0.95 whatever the chain does. The night-window bound of check 2 is 10 W/m², because the archive's
3-hour and 6-hour windows are not exactly consistent near dusk, and the PV model returns 0 when the
sun is below the horizon.
