---
id: "00028"
slug: pvlive-forecast-in-stage2-gsp-demand
status: planned
size: medium
hypothesis: >-
  For the two grid supply points (GSPs) Walpole and Bicker Fen, giving the stage 2 calibrator a
  forecast of the GSP's solar generation at the target time, from a simple XGBoost model of
  Sheffield Solar's PV_Live estimate applied to ECMWF ensemble (ENS) weather, lowers the quantile
  score below that of the same calibrator without a solar forecast.
prediction: >-
  The solar forecast lowers the GSP quantile score over all leads by 0.15 points of capacity
  (plausible range 0 to 0.5), less than the true PV_Live value at the target time, which lowers it
  by 0.5 (plausible range 0.1 to 1.5; the true value is an oracle, not a bound). The true value adds
  information that no weather forecast holds. The stage 2 calibrator sees only summary statistics of
  stage 1a's demand forecast, not the weather, so the solar forecast adds a PV_Live-trained summary
  of the weather; the two models share the shortwave, longwave, temperature, dew point, pressure,
  precipitation, wind, and sun-position inputs. The gain may be larger at Bicker Fen than at
  Walpole, because experiment 00027 found that PV_Live at the valid time helps stage 1a at Bicker
  Fen only; that comparison is exploratory.
decision_rule: >-
  C1, the arm with the solar forecast (B1) minus the arm without (B0), on the GSP quantile score
  over all leads, read at 95% (one deciding contrast, so no multiplicity correction), resampling
  whole validation months and the three seeds. Supported when C1's interval is
  wholly below zero and its estimate is at most -0.30 points, because the change adds a model and a
  data source to the serving path. Refuted when C1's lower bound is above -0.30 and its interval is
  not wholly below zero. A small gain when the interval is wholly below zero and the arm is not
  supported. Inconclusive otherwise. The threshold is 0.30 points because the change adds a model
  and a second published data source to the serving path; it equals experiment 00027's threshold
  for the same reason, not by copy. C2, the arm with the true PV_Live value (B2) minus B0, decides
  no verdict; it says how much the true value adds. Every other contrast, including B1 minus B2 and
  each GSP alone, is exploratory at 95%.
benchmark: >-
  B0, the stage 2 calibrator of experiments 00005 and 00027 without PV_Live (its score is checked
  against 00027's A0), and a month-by-half-hour climatology of PV_Live for the solar forecast's own
  score.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity,
  and mean absolute error of the median; for the solar forecast, mean absolute error of the
  ensemble mean and quantile score, in points of PV_Live's own capacity
---

# Does a forecast of the GSP's PV_Live solar generation help the stage 2 demand forecast?

## Plan, written before any code ran

### Plan: sizing

- **Stored data:** prediction sets and the solar forecast's statistics, under
  `data/ml-experiments/`; no production table changes. - **Production serving path:** not touched. -
  **Degradation rule:** not touched. A missing solar forecast would be a missing input. -
  **Defensible designs:** more than one (the solar model's inputs and the summary given to stage 2).
  Both are fixed below. - **Callers:** the scripts copy experiment 00005's archive reader and
  calibrator and experiment 00027's PV_Live loader, which the author could name without searching.

Size: medium. One Sonnet review of the plan, one of the code, and one of the write-up.

### Plan: the solar forecast

**Target:** the latest revision of PV_Live's `generation_percent_of_capacity` divided by 100 (the
share of PV_Live's own capacity), because the share stays comparable as capacity grows. The chain's
15-day gap exceeds any settling time, and the scores below are against the revised values.
**Model:** one XGBoost model per GSP and seed, with experiment 00005's stage 1a hyperparameters,
trained on ERA5 weather at the valid time. **Inputs (weather only):** `shortwave`, `longwave`,
`temperature_2m`, `dew_point_temperature_2m`, `pressure_surface`, `precipitation`, `wind_speed_10m`,
and the solar zenith and azimuth, which are geometry and not weather. No calendar column and no
power lag. **Training window:** 2019-09-01, the start of the ERA5 and PV_Live grid, to before
2025-06-17 for the final model (the cut-off of experiment 00005's stage 1a). The solar model starts
earlier than the demand model's 2023-09 because its relationship is physical and has no demand
drift, and a window from 2023-09 would leave the first chain models without a summer. Calibration
runs use a forward chain: the model for each quarter from 2024-04 is trained on valid times that end
15 days before the quarter starts, so no calibration case sees a model trained on its own period.

**Applied to ENS as in experiment 00012:** each of the 51 members is run through the model at the
valid time, on the same half-hourly resampling that experiment 00005's statistics use. The summary
given to stage 2 is the member mean and standard deviation of the forecast share at the target time;
two columns keep the stage 2 inputs few. The forecast is known at the forecast origin, because it
uses ENS weather only.

**The solar forecast's own score**, before any demand effect, on the validation runs: the mean
absolute error of the ensemble mean and the quantile score of the 51 member quantiles, in points of
PV_Live's capacity, by lead band, against a month-by-half-hour climatology of PV_Live's share
trained on the same window. The report adds the member mean's bias against PV_Live by lead band
and its error by chain quarter, to show the ERA5-trained model's behaviour on ENS weather.

### Plan: arms

Series, calibrator, rows, seeds, and hyperparameters are those of experiment 00027's A0.

- **B0:** stage 2 without PV_Live.
- **B1:** B0 plus the solar forecast's mean and standard deviation at the target time.
- **B2 (oracle):** B0 plus the true PV_Live share at the target time, which is not known at the
  forecast origin. B2 has one added column and B1 has two, so B2 need not score below B1.

**Past weather.** The delta at lag times uses ERA5 weather as in experiments 00005 and 00027, so
every arm is a ceiling for the lagged inputs; the solar forecast uses ENS weather only, and the
PV_Live training target is a measurement that exists at training time.

**Rows.** Every arm is scored on the cases that have a PV_Live value at the target time and a solar
forecast, and the report states the number of cases per GSP.

**Resolution.** Two series, so the intervals describe these two GSPs. Experiment 00027's intervals
were 0.2 to 0.35 points wide at 98.33%, and the same width is expected here.

Attribution: Sheffield Solar PV_Live, CC BY 4.0.
