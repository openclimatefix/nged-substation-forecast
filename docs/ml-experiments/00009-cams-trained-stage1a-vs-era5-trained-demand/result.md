---
id: "00009"
slug: cams-trained-stage1a-vs-era5-trained-demand
status: planned
size: medium
hypothesis: >-
  A stage 1a XGBoost demand model trained with irradiance from the Copernicus Atmosphere
  Monitoring Service (CAMS) in place of ERA5 shortwave, and applied to the shortwave of ECMWF
  ensemble (ENS) members, has a lower quantile score at Primary substations than the same model
  trained on ERA5 shortwave, because the satellite-derived irradiance is closer to the irradiance
  that the substations' embedded solar generation responded to.
prediction: >-
  On the 73 ENS runs of `fold_a_every5d_v1`, the CAMS-trained arm lowers the Primary quantile
  score, over all leads, by 0.2 to 0.5 points of capacity against the ERA5-trained arm (C1). The
  gain is smaller without the substation that sets 00007's pooled figure (label P2 in experiments
  00006 and 00007), between 0.1 and 0.3 points (C2), and it is concentrated in daylight hours.
decision_rule: >-
  C1 is the quantile score of the CAMS-trained arm minus that of the ERA5-trained arm, Primary
  substations, all leads, mean of the three seeds. C2 is the same difference without the substation
  that sets 00007's pooled figure. Intervals resample whole validation months and the three seeds,
  at 97.5% (Bonferroni across the two planned contrasts), with 10,000 resamples. Supported if C1 is
  at most -0.2 points with its interval wholly below zero, and C2 is at most -0.1 points with its
  interval wholly below zero. Partly supported if C1 meets its condition and C2 does not (the gain
  rests on one substation). Refuted if C1's interval lies wholly above -0.2 (the data exclude a gain
  of the planned size). Inconclusive otherwise.
benchmark: >-
  The ERA5-trained arm refitted here with identical settings (it must reproduce 00005's `r0_from2023`
  to within rounding), and 00005's month-by-half-hour climatology `r2`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score: twice the mass-weighted mean pinball loss over 9 quantile levels, an estimate of
  the continuous ranked probability score (CRPS), in points of capacity (power over effective
  capacity, times 100); and the mean absolute error of the median
production_status: not-for-production
adopted_in: none
code: ml-experiments/00009-cams-trained-stage1a-vs-era5-trained-demand/
---

# Does a stage 1a trained on CAMS irradiance beat one trained on ERA5 shortwave, for demand?

**This is the second stage of P2 in the [research plan](../research-plan.md).** Experiment 00007
found that a fixed correction on CAMS irradiance removes 46.6% of the next year's daylight squared
residual of the ERA5-trained stage 1a (98.33% interval 38.4% to 53.1%), that a fixed bias alone
removes 39.1%, and that one substation sets the pooled figure (18.2% without it). The result was
inconclusive. This experiment trains stage 1a on CAMS at the training times and scores it on ENS.

## Size

- **What gets stored:** prediction sets and tables under this experiment's private folder. Nothing
  enters a Delta table.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **Designs that would test the hypothesis defensibly:** two. Replacing ERA5 shortwave by CAMS at
  the training times, planned here, is the simpler. Keeping ERA5 for training and quantile-mapping
  the ENS shortwave onto CAMS at application would also be defensible, and is left to a later
  experiment.
- **Callers that need searching for:** none. The scripts copy the patterns of
  `ml-experiments/00005-*` and `ml-experiments/00006-*` and read their inputs.

The experiment is medium: it adds a data source to training and a method nobody has run here, and
touches nothing stored for production. It gets a Sonnet review of the plan, one of the code, and one
of the write-up. The earlier `advice` lists were read: this plan follows "Train a demand stage 1a
from 2023-09" (00005), "Use the weather block, hour, weekday, and the COVID indicator" (00006),
"Report a pooled share with and without the substation that has the largest residual" (00007), and
"Compare irradiance read at the substation's own location, such as CAMS, with ERA5 shortwave" (00006).
It departs from "Weight recent years more" (00006), so that the ERA5-trained arm matches 00005's
`r0_from2023`.

## Design

**Two arms differ in one column.** Both are one XGBoost model per series, with 00005's settings
(500 rounds, depth 6, learning rate 0.05, squared error, three seeds), trained on valid times from
2023-09-01 to before 2025-06-17. The columns are 00006's chosen set, as frozen in 00005: ERA5
shortwave, longwave, temperature, dew point, surface pressure, precipitation, and 10 m wind speed;
solar zenith and azimuth; local hour and weekday; the COVID indicator; and the days to Christmas.
The `era5` arm reads ERA5 shortwave at the series' H3 cell. The `cams` arm reads CAMS irradiance at
the series' own location instead (the download in [PR #1201 (Download and validate CAMS irradiance
at the demand series' locations)](https://github.com/openclimatefix/nged-substation-forecast/pull/1201)).
At validation, both arms read the ENS member's shortwave. Every other weather column stays ERA5 in
training and ENS in validation.

**CAMS is hourly and the grid is half-hourly.** A CAMS value is the mean over the hour ending at its
stamp. Both half-hours of that hour take the value. This differs from the half-hourly construction
of ERA5 shortwave, and the write-up repeats the point.

**Each arm gives 9 quantiles from the 51 ENS members.** The model is applied to every member and
the quantiles are the empirical quantiles of the 51 forecasts, as in 00005's `r0` arms. There is no
calibrator, no stage 1b, and no lagged power, so no weather has to exist at lag times. CAMS is used
in training only and is never needed at forecast time.

**The scored rows are those of 00005.** They are the 73 runs of `fold_a_every5d_v1` for the 20
series, restricted to valid times with an observed power and before 2026-07-01. The 16 Primary
series are the headline. The 2 bulk supply point and 2 grid supply point series are reported
separately and decide nothing. Intervals resample whole validation months and the three seeds. The
climatology row is 00005's `r2`.

**Two planned contrasts, both on Primary substations over all leads.** C1 compares the arms
(CAMS-trained minus ERA5-trained). C2 repeats C1 without the substation that sets 00007's pooled
figure. The 0.2-point threshold of the research plan is kept for C1. In 00005, a contrast between
two stage 1a variants on Primary substations over all leads had a 95% interval about 0.2 points
wide on each side (E11: +0.075, -0.060 to +0.265), so 0.2 is the size of the noise and not below it.
C2 uses half that size because removing a substation removes much of the gain 00007 saw. The 97.5%
level is a Bonferroni correction across the two.

**Exploratory contrasts are read at 95%.** They are: C1 by lead band and for daylight and night
valid times separately (the gain should come in daylight); C1 for the bulk supply points and the
grid supply points; C1 for each Primary series; C1 leaving out each Primary series in turn; and the
quantile score of the ERA5-trained arm against the climatology. A deterministic diagnostic at the
observed weather also compares the two arms at the validation hours, with the ERA5-trained arm
reading ERA5 and the CAMS-trained arm reading CAMS, as in 00007's residuals. It shows what training
on CAMS gains when CAMS is also the input, which no forecast can supply.

**Leakage and calendar checks.** Both arms train on valid times before 2025-06-17, 14 days before
the first validation run, so no fit sees a validation hour. There is no early stopping. Both arms
carry the same calendar columns, so the contrast cannot come from calendar memorisation, and in
00005 removing the calendar columns changed the stage 1a score by -0.009 points (E12).

**Known limits.** CAMS and ERA5 shortwave are measured on different cloud fields, and the
CAMS-trained arm sees ENS shortwave, a forecast with its own bias and spread, which is the mismatch
the experiment tests. Validation is one year (2025-07 to 2026-06), which is also 00006's
confirmation year, so 12 months bound the interval's resolution. The 20 series and 73 runs are
00005's choice and were not re-selected.
