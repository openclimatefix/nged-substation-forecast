---
id: "00007"
slug: cams-static-correction-stage1a-residual
status: planned
size: medium
hypothesis: >-
  Most of the daytime residual of the stage 1a demand model at Primary substations is a fixed
  error in the ERA5 irradiance input and not a growing embedded-solar capacity, so a correction
  with fixed coefficients on irradiance from the Copernicus Atmosphere Monitoring Service (CAMS),
  fitted on one year of out-of-fold residuals, removes most of the residual in the next year.
prediction: >-
  Fitted on the 2024-07 to 2025-06 residuals and scored on the 2025-07 to 2026-06 residuals,
  ERA5 shortwave plus CAMS irradiance removes at least 50% of the daylight mean squared residual
  of the median forecast at the Primary substations that have CAMS values, and CAMS adds at least
  20% on top of ERA5 shortwave alone (partial R squared). The stability ratio, the out-of-sample share removed over the share removed when the same
  correction is fitted on the scored year itself, is at least 0.8.
decision_rule: >-
  Share removed is one minus the sum of squared daylight residuals after the correction over the
  sum before it (uncentred, pooled over the 14 substations), so a fixed bias counts as error. The
  gate points to P4 (train stage 1a on CAMS irradiance) when all four hold: the share removed by
  the ERA5-plus-CAMS correction is at least 50%; the partial R squared of CAMS given ERA5 is above
  zero; the daylight mean absolute error of the median falls by at least 0.3 points of capacity
  against the fixed-shift arm; and the stability ratio is at least 0.8. The three quantities with
  intervals (share removed, partial R squared, MAE gain) must have 98.33% month-block intervals
  (Bonferroni across the three) that exclude the zero of their own contrast, and the share
  removed's interval must also lie above 25%. The gate points to the capacity-change statistics of
  P2 when the share removed is 20% or lower, or when the stability ratio is below 0.5. Every other
  outcome is inconclusive.
benchmark: >-
  The uncorrected median forecast of 00006's chosen arm (weather block, hour, weekday, COVID
  indicator, days to Christmas, recency weights), and the same forecast with a fixed daylight
  mean shift only.
row_set_id: era5_val_2025_07_to_2026_06_v1
metric: >-
  Share of the daylight mean squared residual removed (points of capacity squared), and the mean
  absolute error of the median, points of capacity (power over effective capacity, times 100)
verdict: null
production_status: not-for-production
adopted_in: none
---

# Experiment 7: does a static CAMS correction remove the stage 1a daytime residual?

**This is the cheap first test that gates P4 and the rest of P2 in the
[research log](../research-log.md#p2-embedded-solar-capacity-from-lagged-residuals).** The result
says whether the daytime residual of the stage 1a demand model tracks a fixed error in the ERA5
irradiance input, or a change in embedded solar capacity that a correction with fixed coefficients
cannot follow. The page is a plan now and becomes the write-up after the run.

## Size

- **What gets stored:** residual tables and figures only, under this experiment's private folder.
  Nothing enters a Delta table.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Designs that would test the hypothesis defensibly:** two. A linear correction of the residuals
  answers the question cheaply and is the planned design. Retraining the XGBoost model with CAMS as
  a feature is the later P4.
- **Callers that need searching for:** none, because the scripts read 00006's stored predictions and CAMS parquet and nothing imports them. The scripts read 00006's stored predictions and the
  CAMS download from [PR #1201 (Download and validate CAMS irradiance at the demand series'
  locations)](https://github.com/openclimatefix/nged-substation-forecast/pull/1201).

The experiment is medium: it uses a new data source and a new method, but touches nothing stored
for production. It gets one Sonnet review of the plan, one of the code, and one of the write-up.

## Design

**The residuals come from 00006's out-of-fold fits.** The residual is the metered power over
effective capacity minus the median forecast of the chosen arm, times 100, averaged over the three
seeds of the `finals` stage. Each validation year was forecast by a model trained on data ending
a week before that year starts, so no residual comes from a fit that saw its own row. The two
years are 2024-07 to 2025-06 (year A, fold `select_2024`) and 2025-07 to 2026-06 (year B, fold
`confirm_2025`). Half-hours are averaged to the hour ending at the CAMS time stamp, as in
`e00006_solar_residual.py`, and only daylight hours (solar zenith below 80 degrees) are kept.

**The correction is fitted on year A and scored on year B, per substation.** Each correction is an
ordinary least-squares fit of the hourly residual on the columns below, with its coefficients
frozen after year A. Scoring on year B is the walk-forward test of a static correction. The
substations are the Primary series that have CAMS values (14).

| Arm | Fitted columns | Role |
|---|---|---|
| `none` | nothing | The uncorrected forecast |
| `shift` | intercept | A fixed daylight bias |
| `era5` | intercept, ERA5 shortwave | What stage 1a could already learn |
| `cams` | intercept, CAMS irradiance | CAMS alone |
| `era5+cams` | intercept, ERA5 shortwave, CAMS irradiance | The headline arm |
| `cams-era5` | intercept, CAMS minus ERA5 shortwave | One coefficient on the difference |

**Two planned contrasts, both on year B.** The first is the share of the daylight mean squared
residual removed by `era5+cams` against `none`. The second is the partial R squared of CAMS given
ERA5, that is `1 - SSE(era5+cams) / SSE(era5)`. A third quantity, the mean absolute error of the
median, is read for the same arms because it is the metric the other experiments report. The
stability ratio is the out-of-sample share removed divided by the share removed when the same
correction is fitted on year B itself. A ratio near 1 means the relation is stable. A low ratio
means the coefficients drift between the years.

**The interval resamples whole months, each with all 14 substations.** Weather is shared across
substations and hours within a day, so the block is a month of year B with every substation in it.
The 98.33% interval (Bonferroni across the three interval conditions) comes from 10,000 resamples
of the 12 months, rescored from the frozen coefficients. With 12 blocks the tails are coarse, and
the write-up says so. The write-up also reports the share removed for each substation, and the
share removed with each substation left out in turn. A secondary figure repeats the scoring with a
leave-one-month-out fit inside year B, which keeps the within-year seasonality in both halves.

**The stability ratio is pooled.** Its numerator is the out-of-sample share removed and its
denominator is the share removed by the same correction fitted on year B. A denominator below 5%
makes the ratio undefined, and the outcome is then inconclusive.

| Share removed | Stability ratio | Outcome |
|---|---|---|
| 50% or more | 0.8 or more | P4 gate, if the three interval conditions also hold |
| 50% or more | 0.5 to 0.8 | Inconclusive |
| 20% to 50% | 0.5 or more | Inconclusive |
| 20% or less, or any share | below 0.5 | P2 capacity-change statistics |

**CAMS is a lag-time input only.** CAMS is observed after the fact and has no forecast, so a
stage 1a trained on it could use it only for the past. Its latency at forecast time is not checked
here. A positive gate is a necessary condition for P4 and does not show that the input is available
in time. ERA5 arrives about 5 days late, so the ERA5 column is also a lag-time input in this test.

**Year B's residuals come from a model trained on year A.** The `confirm_2025` model saw year A, so
it may already have absorbed part of a fixed ERA5 error, and the stability ratio then measures that
model change as much as any drift. Year B is also 00006's confirmation fold, and the July 2025
exploration in `e00006_solar_residual.py` has already looked at one month of it. The arm and the
seeds of the chosen forecast were selected on folds that include year A. The write-up repeats all
three points. The expected sign of the CAMS coefficient is negative, as at P12 in that exploration.
CAMS and ERA5 shortwave are collinear, so the write-up reports the partial R squared and not the
two coefficients. A linear fit on irradiance cannot capture a response that is nonlinear in cloud,
so a low share removed is weaker evidence than a high one.

**Confounds the write-up has to discuss.** A fixed error in ERA5's cloud field and a fixed
embedded-solar capacity both give a static relation, so a positive gate says the relation is stable
over one year and does not exclude capacity. The stability ratio is the only evidence against
capacity drift in this test.

## Results

To be written after the run.
