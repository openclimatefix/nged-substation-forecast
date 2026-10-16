---
id: "00053"
slug: ukv-vs-ens-solar-anomaly
status: done
size: medium
hypothesis: >-
  At the five Primary substations metered in MW that decide experiment 00046's verdict (D9, D10,
  D11, D14, and D16), the Met Office UKV forecast of the solar anomaly (the output of a generic
  solar fleet minus its own month-by-half-hour mean), alone or blended with the ECMWF ensemble mean
  (ENS) using weights fitted on the training window, tracks the CAMS solar anomaly more closely than
  ENS alone does within the UKV's lead range of 120 hours.
prediction: >-
  Not recorded before the run. The scores were computed before this page was written, so the page
  has no prediction that the run could have falsified. The design (arms, bands, gate, and margin)
  follows the brief for PVDIS-8b and the stage 1 gate of the PVDIS-8 plan.
decision_rule: >-
  Stage 1 gate, on the training window only (April 2024 to June 2025): 15-fold month-block
  cross-validation, one fold per local calendar month of valid time, fitting each forecast's
  climatology, the CAMS climatology, and the calibration on the other months. The scored rows are
  daylight half-hours pooled over the five verdict series. The blend or the UKV replaces ENS only if
  its cross-validated mean squared error against the CAMS anomaly is at least 1% lower. If neither
  the UKV nor the blend clears 1% in any lead band within 0 to 54 hours, the verdict is `refuted`
  and the demand contrasts are not run. If the gate clears, the experiment stops and reports; the
  demand contrasts (C3 and C6 of experiment 00046 at these leads) are decided by the manager. The
  1% margin is a screening threshold for adding a second weather product, not a threshold on
  demand-forecast error, which this experiment does not measure.
benchmark: >-
  ENS alone: the slope through the origin of the CAMS anomaly on the ENS ensemble-mean anomaly, fitted
  per series and lead band on the training window, as in experiment 00046 but with this page's lead
  bands.
row_set_id: >-
  The scored rows of `fold_a_every5d_v1`: experiment 00046's rows with a finite observed power,
  daylight, local months April to September, valid 2025-07-01 to 2026-06-30 (72 forecast runs, one
  every 5 days).
metric: >-
  Pearson correlation, calibration slope (slope through the origin of the CAMS anomaly on the
  calibrated forecast anomaly; ideal 1), and mean squared error of the calibrated forecast anomaly
  against the CAMS anomaly, by lead band, with 95% intervals from resampling whole calendar months
  of the forecast origin (50,000 resamples).
code: ml-experiments/00053-ukv-vs-ens-solar-anomaly/
verdict: small gain
best_arm: >-
  The ENS and UKV blend, with a cross-validated mean squared error 6.8% below ENS's over 0 to 54
  hours (4.4% to 11.6% in each band). UKV alone is worse than ENS in every band.
headline: >-
  The blend raises the correlation with the CAMS anomaly by 0.02 to 0.03 within 24 hours (0.020,
  0.002 to 0.036 at 0 to 6 hours) and by about 0.01 over 0 to 54 hours (0.009, -0.008 to +0.025),
  and cuts the cross-validated mean squared error by 4.4% to 11.6% in every band within 54 hours.
  UKV alone lowers the correlation by 0.112 over 0 to 54 hours (-0.190 to -0.049). Beyond 54 hours
  the blend gains 0.005 (-0.015 to +0.024).
lessons: >-
  The UKV forecast of the solar anomaly is worse than ENS's but its errors are partly independent
  of ENS's, so a fitted blend improves on ENS by a small margin within 2 days and by nothing beyond
  54 hours. The finer time steps of the UKV are not what limits it: forming 3-hour window means from
  the UKV, as ENS delivers its flux, scores slightly better than the native hourly values. The only
  UKV with a matched history is the CEDA archive, which is not the live product and is licensed for
  non-commercial use, so the result is a ceiling and does not justify adoption.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not add the CEDA UKV to the serving path, because it is a different product from the live
    UKV and the licence is non-commercial, so the blend's 6.8% lower mean squared error within 54
    hours is a ceiling and not an estimate for production (stage 1 gate table).
  - >-
    Do not replace ENS with the UKV in the solar term at any lead from 0 to 120 hours, because the
    UKV's correlation with the CAMS anomaly is lower by 0.032 to 0.359 in every band within 54
    hours and by 0.187 beyond (differences table).
  - >-
    Do not expect a second weather forecast to help the solar term beyond 54 hours, because the
    blend gains 0.005 in correlation (-0.015 to +0.024) at 54 to 120 hours and misses the 1%
    cross-validation margin there, at 0.8% lower (stage 1 gate table).
  - >-
    Decide the demand contrasts (C3 and C6 of experiment 00046 at 0 to 48 hours) only if a live-matched
    UKV history of at least a year becomes available, because the blend's gain in correlation within
    48 hours is 0.019 to 0.029 and has an interval excluding zero in 3 of 4 bands (differences table).
  - >-
    Interpolate instantaneous model irradiance in clear-sky-index space and keep model steps hourly or
    finer where they exist, but expect no gain from it, because 3-hour window means formed from the
    UKV score 0.021 higher in correlation over 0 to 54 hours than the native hourly interpolation
    (0.013 to 0.031), not lower (control arms).
---

# Experiment 53: does the UKV forecast the solar anomaly better than ENS?

We asked whether a weather model with a finer grid, the Met Office UKV (a 2 km model of the United
Kingdom that runs to 2 to 5 days ahead), predicts how sunny each half-hour will be better than the
European Centre for Medium-Range Weather Forecasts ensemble forecast (ENS) that the forecast uses
today. We converted each forecast's sunshine into the output of a generic solar fleet, subtracted
that forecast's own monthly average for each half-hour of the day (the "anomaly"), and compared the
result with the anomaly of satellite-based sunshine estimates from the Copernicus Atmosphere
Monitoring Service (CAMS) at 5 Primary substations, on 72 forecast runs from July 2025 to June 2026.
The UKV alone was worse than ENS at every lead time. A mix of ENS and UKV, with weights fitted on
the earlier 15 months, was slightly better than ENS within 2 days (the correlation with CAMS rose by
about 0.02, and the squared error in cross-validation fell by 4% to 12%) and no better beyond 54
hours. The only UKV with a long matching history is the CEDA archive, which differs from the live
UKV and is licensed for non-commercial use, so the gain is an upper bound.

**The verdict is `small gain` for the blend, and no gain for the UKV alone.** The blend clears the
1% cross-validation margin of the stage 1 gate in every lead band within 54 hours, and the UKV
alone clears it in none (relative mean squared error +14% to +32% against ENS). The gate therefore
does not stop the experiment as `refuted`. As the brief says, the experiment stops there: the demand
contrasts were not run, and the manager decides whether to run them. The plan did not commit a
prediction before the run, and this page was written after the scores existed.

All figures are for 5 Primary substations (D9, D10, D11, D14, and D16, anonymised as in experiment
46), daylight half-hours of April to September, valid from July 2025 to June 2026, on 72 forecast
runs (one every 5 days; the half-hours of 1,000 to 4,714 rows per lead band come from 7 calendar
months of forecast origin, so the intervals are probably somewhat narrow).

## What was compared, and when each forecast would have been available

**Lead is counted from the forecast origin that the production service has, which is the 00 UTC ENS
run plus 9 hours, 09:00 UTC.** Each arm uses the freshest forecast available at that origin.

- **ENS** is the daily 00 UTC run, as in experiment 46.
- **UKV, `t54`** is the 06 UTC run of the same day (leads to 54 hours from the run, so to 51 hours
  from the origin), assuming a run is available 3 hours after its initialisation. The assumption of
  3 hours is ours and unverified: the Met Office's delivery time of the live UKV was not measured.
- **UKV, `t120`** is the 03 UTC run of the same day, whose T120 files reach 120 hours from the run
  (117 hours from the origin).
- **UKV, `t54_late`** is the 00 UTC run of the same day, a sensitivity check that assumes the 06 UTC
  run is not yet available (leads to 45 hours from the origin).
- **CAMS** is the reference: experiment 39's half-hour irradiance for the substation, kept where
  CAMS reliability is at least 0.9.

The five arms of each table are ENS alone, the UKV alone (native steps), the blend, and two controls
that separate the time resolution of the UKV from its skill. **Control `ukv3h`** subsamples the UKV
to ENS's 3-hourly steps and passes the instantaneous values through the production ENS resampler,
labelled as if they were ENS's 3-hour window means, as the maintainer asked. **Control `ukv3w`**
first forms 3-hour window means from the UKV (the trapezoid mean of the instantaneous values inside
each window) and then uses the same resampler. The second control is added because the first has a
half-window timing error built in: the hour-of-day figure below shows the `ukv3h` curve peaking
early.

The calibration is a slope through the origin of the CAMS anomaly on a forecast's anomaly, fitted per
series and lead band on the training window (valid times to 2025-06-30). The blend regresses the CAMS
anomaly on the ENS and UKV anomalies, with two coefficients and no intercept. Fitted on the whole
training window, the ENS weight is 0.61 to 0.84 per series and band and the UKV weight is at least
0.18, with no negative weight (`t54`).

## Shortwave conventions and how each was aligned with CAMS

**CAMS and ENS are half-hour or window means labelled at the period end, and the UKV and MOGREPS-UK
are instantaneous values, so only they need the interpolation.**

| Source | Shortwave convention | Valid-time stamp | Alignment with CAMS |
|---|---|---|---|
| CAMS (experiment 39) | Half-hour mean irradiance | End of the half-hour | None needed |
| ENS | Mean over the native window (3 hours to 144 hours, 6 hours after) | End of the window | The production resampler `_upsample_nwp_to_half_hourly` shapes each window by the clear-sky index and conserves its energy, giving half-hour means ending at the stamp |
| CEDA UKV, `shortwave_down` | Instantaneous flux, not a mean | The instant, init time plus step; hourly steps to 48 hours, then 3-hourly | The clear-sky index (flux over the Haurwitz clear-sky flux at the same instant) at each step is interpolated linearly in time to the midpoint of each half-hour, then multiplied by the clear-sky flux at that midpoint |
| MOGREPS-UK, `shortwave_total` | Instantaneous, as the archive notes (no cell methods or time bounds in the files) | The instant; hourly steps | The same as the UKV |

The same fleet chain then turns each half-hour irradiance into `u`. A clear-sky index is only formed
where the clear-sky flux is at least 20 W/m2, and a midpoint more than 3 hours from a usable step
takes the nearest step's index, so nothing is interpolated across a night. Beyond 48 hours the
native UKV steps are 3-hourly, so the interpolation there spans 3 hours. The index is capped at
1.5. Each site's UKV value is the cell whose centre is nearest (the largest distance is 1.09 km).

## Raw data first

**The UKV anomaly follows the CAMS anomaly, with a wider scatter than ENS.** The panels show the
calibrated forecast anomaly against the CAMS anomaly at leads under 24 hours. The `ukv3h` control is
visibly the worst, with a flat top.

![Each calibrated forecast anomaly against the CAMS anomaly at leads under 24 hours](figure_1_scatter.svg)

**All sources peak at 12 UTC, as CAMS does, so no label is shifted, except the `ukv3h` control.**
The UKV's mean `u` is 0.875 of CAMS's at leads under 24 hours and ENS's is 1.023. A multiplicative
bias of the UKV does not affect its anomaly, which is taken against its own climatology.

![Mean u by hour of day for each source at leads under 24 hours](figure_3_hour_of_day.svg)

## Results by lead band

Correlation with the CAMS anomaly (95% interval). Bands up to 48 to 54 hours use `t54`, and the 54
to 120 hour band uses `t120`. The 48 to 54 hour band of `t54` holds leads of 48 to 51 hours only.

| Lead | Rows | ENS | UKV alone | Blend | `ukv3h` control | `ukv3w` control |
|---|---|---|---|---|---|---|
| 0-6 h | 1,970 | 0.837 (0.781-0.880) | 0.797 (0.752-0.819) | **0.857** (0.805-0.890) | 0.657 (0.577-0.692) | 0.809 (0.766-0.832) |
| 6-12 h | 1,556 | 0.810 (0.731-0.877) | 0.753 (0.699-0.804) | **0.830** (0.758-0.891) | 0.599 (0.459-0.710) | 0.784 (0.724-0.838) |
| 12-24 h | 1,000 | 0.762 (0.691-0.825) | 0.730 (0.623-0.811) | **0.791** (0.726-0.846) | 0.632 (0.516-0.752) | 0.752 (0.655-0.832) |
| 24-48 h | 4,714 | 0.734 (0.689-0.783) | 0.633 (0.550-0.705) | **0.754** (0.711-0.795) | 0.510 (0.401-0.584) | 0.660 (0.585-0.723) |
| 48-54 h | 1,225 | **0.732** (0.594-0.823) | 0.373 (0.023-0.670) | 0.685 (0.474-0.815) | 0.282 (0.016-0.561) | 0.367 (0.017-0.670) |
| 0-54 h | 10,465 | 0.764 (0.736-0.791) | 0.651 (0.580-0.718) | **0.773** (0.751-0.801) | 0.529 (0.442-0.601) | 0.673 (0.610-0.733) |
| 54-120 h | 10,851 | 0.651 (0.615-0.685) | 0.464 (0.244-0.595) | **0.656** (0.619-0.687) | 0.373 (0.126-0.519) | 0.469 (0.239-0.607) |

![Correlation with the CAMS anomaly by lead band, with 95% intervals](figure_2_correlation_by_band.svg)

**The blend's gain is small, and its interval excludes zero in about half of the bands.** The
difference in correlation, blend minus ENS, is +0.020 (0.002 to 0.036) at 0 to 6 hours, +0.020
(0.014 to 0.027) at 6 to 12 hours, +0.029 (0.003 to 0.058) at 12 to 24 hours, +0.019 (-0.005 to
+0.042) at 24 to 48 hours, -0.047 (-0.130 to +0.006) at 48 to 54 hours, and +0.005 (-0.015 to
+0.024) at 54 to 120 hours. The UKV alone minus ENS is negative in every band, from -0.032 (-0.122
to +0.052) at 12 to 24 hours to -0.359 (-0.598 to -0.132) at 48 to 54 hours.

**Time resolution is not what limits the UKV.** The `ukv3w` control, which forms 3-hour window means
from the hourly UKV as ENS delivers its flux, scores 0.021 higher in correlation than the native
steps over 0 to 54 hours (0.013 to 0.031) and has a lower mean squared error by 5.3% (3.7% to 6.7%).
The literal `ukv3h` control scores 0.122 lower than the native steps over 0 to 54 hours, because the
instantaneous values stand in for window means and are shifted by about 1.5 hours.

Calibration slope (ideal 1) and mean squared error of the calibrated anomaly, same rows:

| Lead | ENS slope, MSE | UKV slope, MSE | Blend slope, MSE |
|---|---|---|---|
| 0-6 h | 0.95, 0.0257 | 0.98, 0.0339 | 0.94, 0.0244 |
| 6-12 h | 1.00, 0.0074 | 0.97, 0.0094 | 0.97, 0.0067 |
| 12-24 h | 1.21, 0.0078 | 1.16, 0.0084 | 1.20, 0.0070 |
| 24-48 h | 1.04, 0.0196 | 0.89, 0.0258 | 1.02, 0.0183 |
| 48-54 h | 1.04, 0.0318 | 0.58, 0.0614 | 0.98, 0.0357 |
| 54-120 h | 1.04, 0.0228 | 1.03, 0.0312 | 1.04, 0.0226 |

## The stage 1 gate on the training window

**The blend clears the 1% margin in every band within 54 hours and the UKV alone in none.**
Cross-validated mean squared error, relative to ENS, pooled over the five verdict series (`t54`):

| Lead | ENS (mean squared error) | UKV alone | Blend | `ukv3w` control |
|---|---|---|---|---|
| 0-6 h | 0.0255 | +21.6% | -11.6% | +14.0% |
| 6-12 h | 0.0082 | +21.6% | -8.3% | +12.9% |
| 12-24 h | 0.0078 | +14.5% | -4.4% | +12.2% |
| 24-48 h | 0.0196 | +31.9% | -5.1% | +24.9% |
| 48-54 h | 0.0360 | +19.8% | -4.4% | +23.5% |
| 0-54 h pooled | 0.0206 | +25.3% | -6.8% | +20.7% |

For `t120` the blend is 2.9% lower pooled over 0 to 120 hours and 0.8% lower at 54 to 120 hours,
which misses the 1% margin. If the 06 UTC run were not available at the origin (`t54_late`), the UKV
alone is worse by 18% to 42% in the bands to 48 hours, and the blend is still lower by 3.5% to
6.7% in every band.

## The check on MOGREPS-UK

**MOGREPS-UK, the live ensemble of the Met Office, is no better than ENS on 14 days of overlap with
CAMS, and an equal-weight blend is no better either.** MOGREPS-UK has 3 members and about 7 weeks of
archive (27 August to 16 October 2026), so nothing was calibrated or fitted; its anomaly is its
member-mean `u` minus the CAMS climatology of experiment 46, and the ENS anomaly is against ENS's own
climatology. CAMS ends on 11 September 2026, so the reference covers only 14 forecast origin days
(27 August to 10 September), outside the validation year. Correlation with the CAMS anomaly, 95%
interval from resampling whole origin days:

| Lead | Rows | ENS | MOGREPS-UK | Equal-weight blend |
|---|---|---|---|---|
| 0-24 h | 1,551 | 0.741 (0.584-0.847) | 0.627 (0.416-0.771) | 0.742 (0.538-0.845) |
| 24-48 h | 1,501 | 0.690 (0.502-0.795) | 0.438 (0.204-0.631) | 0.637 (0.421-0.764) |
| 54-120 h | 3,063 | 0.393 (0.178-0.593) | 0.310 (0.176-0.447) | 0.394 (0.217-0.565) |

Over all 49 origin days from 27 August 2026, the MOGREPS-UK anomaly correlates with the ENS
anomaly at 0.768 within 24 hours, 0.699 at 24 to 54 hours, and 0.545 at 54 to 120 hours. That
agreement is not a skill score, because neither anomaly is the reference. MOGREPS-UK's mean `u` is
0.303 against ENS's 0.373 and CAMS's 0.378 on those rows.

## What these data could not show

- **The live UKV is not the CEDA UKV.** The CEDA archive differs statistically from the live feed,
  its licence is non-commercial, and the live product has no history at the Primaries, so the blend's
  gain is a ceiling.
- **The 3-hour delivery delay is assumed.** The `t54_late` check shows that a 9-hour delay lowers
  the blend's gain but does not remove it, but the true delay was not measured.
- **The demand-forecast effect is untested.** A 0.02 gain in correlation of the anomaly need not
  carry through experiment 46's solar term, whose coefficient the residual fitted below 1.
- **The 48 to 54 hour band is thin.** It holds 1,225 rows from leads of 48 to 51 hours and its
  intervals are wide.
- **Calibration and the blend use 15 months.** Each source's climatology is fitted on the same 15
  months, with neighbouring local months pooled as in experiment 46.

## What the next experiment should answer

The manager decides whether to run experiment 46's C3 and C6 contrasts at 0 to 48 hours with the
blend's anomaly. The scores here predict a change in that contrast of the order of the blend's 0.02
gain in correlation, which experiment 46's intervals (half-widths of 0.13 to 0.32 points of
capacity at 0 to 48 hours) could not resolve, so a non-live product with a licence that forbids
production is a weak case for the spend.
