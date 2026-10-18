---
id: "00098"
slug: det-bg-solar-and-heating-offsets-in-switching-detector-residual
status: planned
size: medium
hypothesis: >-
  The refit-offset neighbour-count switching detector (RO-NC, experiment 82's XL) subtracts a
  lagless XGBoost demand forecast (experiment 17's H0.5, trained on ERA5) from each Primary
  substation's measured power. Giving that forecast the best guess's two explicit offsets, an
  embedded-solar term (capacity times a solar anomaly) and a heating degree-hour term, removes
  structured weather error from the residual, so the detector finds more of the moved MW at the
  same number of unexplained alarms.
prediction: >-
  The solar offset lowers the median absolute residual in summer midday hours (10:00 to 14:00 UTC,
  April to September) at the 5 Primaries that have a solar capacity by 5% to 15%, and RO-NC's
  MW-weighted recall (pooled over the 16 Primaries) rises by +0.01 (-0.01 to +0.03) for the better
  of the ERA5 and CAMS solar arms. The change is small because 11 of the 16 series have no solar
  term and the model already sees ERA5 shortwave as an input. The heating offset adds +0.00 to
  +0.01. CAMS and ERA5 solar arms differ by under 0.01 in recall.
decision_rule: >-
  Four planned contrasts, each the paired difference in MW-weighted recall of RO-NC, pooled with
  equal weight over injection shapes S2, S3, and S4, on the draws of experiment 82 (same seeds),
  for transfers that start in even months, intervals at 98.75% (a 95% interval split across four)
  from 20,000 resamples of whole calendar months. C1 is arm `era5_solar` minus the control `ctrl`.
  C2 is arm `cams_solar` minus `ctrl`. C3 is arm `heat` (the better solar arm plus the heating
  offset, where better means the smaller median absolute summer-midday residual on odd months at
  the 5 solar series) minus `ctrl`. C4 is `heat` minus its solar parent. Every arm is held to 80
  unexplained odd-month alarms. The verdict is `supported` when the lower end of the best of C1 to
  C3 is above zero and its point value is at least +0.02, `small gain` when that lower end is above
  zero but the point value is under +0.02, `refuted` when the upper ends of C1, C2, and C3 are all
  under +0.02, and `inconclusive` otherwise. The +0.02 threshold is experiment 70's: the offsets
  add a capacity table, an irradiance feed, and a coefficient fit to the detector's residual
  model. The residual noise, forecast MAE of H0.5, recall by donor step, and the same detectors
  PF14 and D0 are descriptive.
benchmark: >-
  `ctrl`: H0.5 refitted with a zero base margin under the refit schedule below. `h05_stored`
  (experiment 17 and 71's stored fits) is the replication check: RO-NC must reproduce experiment
  82's 0.770 (0.665 to 0.857) MW-weighted recall and PF14's 0.656.
row_set_id: >-
  none (this experiment scores detector alarms against injected and labelled real transfers, as
  experiments 69 to 71 and 82 do, not forecasts)
metric: >-
  MW-weighted recall at 80 unexplained odd-month alarms (headline), pooled over S2, S3, and S4,
  even months; median absolute residual by hour band and season; forecast MAE of H0.5 itself;
  recall by donor step in noise units.
code: ml-experiments/00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual/
---

# Experiment 98: do the best guess's solar and heating offsets sharpen the switching detector?

*Planned; the write-up replaces this section after the run.*

## Plan

- **Question.** Does a cleaner H0.5 residual let RO-NC find more of the moved MW?
- **Lab shortcuts (stated, not fixed).** ERA5 weather in the residual model, the 24-hour hindsight
  end rules of experiment 82, and solar capacities fitted with data after some scored dates.
- **Arms (6).**
    - `h05_stored`: experiment 82's stored panels (replication check only).
    - `ctrl`: H0.5 refitted here with a zero base margin (the control for every contrast).
    - `era5_solar`: `ctrl` plus a base margin `-(capacity / effective capacity) * (u - clim)` at
      the 5 series that have a capacity (series 7, 8, 10, 16, 18), where `u` is the unit PV shape
      from ERA5 shortwave and `clim` is its month-by-half-hour mean over 2019-09 to 2021-12.
    - `cams_solar`: the same with CAMS `u` and its own climatology over the same months.
    - `heat`: the better solar arm plus experiment 94's heating term for all 16 series.
    - `heat_cal`: optional, `heat` plus experiment 47's bank-holiday column.
- **Weight on the solar margin.** Fixed at -1 (net power falls one-for-one with solar output),
  because reanalysis anomalies lack the forecast noise that pulled experiment 64's fitted weights
  to -0.3 to -0.6. A printed diagnostic regresses the control's training residual on the margin.
- **Capacity.** Experiment 64's held capacity before 2025-07-01 and experiment 78's season-year
  value from 2025-07-01 (one refresh per season-year).
- **Schedule.** Two H0.5 models per arm, refitted on the 1st and the 16th of each month (experiment
  71's refit-offset layout), 150 rounds, eta 0.15, otherwise experiment 17's settings, one thread
  per worker. A cheaper schedule than experiment 17's 500 rounds at eta 0.05; every refit arm
  shares it, so the contrasts are clean, and `h05_stored` checks the detector reproduces.
- **Detector.** Experiment 82's code with frozen settings; the per-arm noise scale, thresholds, and
  logistic score are re-estimated on odd months from the arm's own residual panels.
- **Advice followed.** Experiment 94: add the heating term to the solar term, not instead of it.
  Experiment 64: attach the solar term as a base margin. Experiment 47: the bank-holiday column
  lowers forecast error without changing detector recall (so `heat_cal` is optional).

**Size (five triggers).** Storage: private residual panels only. Serving path: untouched. Degradation
rule: untouched. More than one defensible design: yes (weight, climatology), fixed above. Callers
hard to name: no, the code is copied from experiments 71, 82, 94, and 64.
