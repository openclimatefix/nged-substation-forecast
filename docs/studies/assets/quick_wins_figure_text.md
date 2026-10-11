# Quick-wins figure text

## Figure 1

Plots: Fold mid_2025_to_mid_2026: CRPS and MAE (% of capacity) of every arm, all series types, all leads, with 95% month-resampled intervals; baselines and production as lines; a third panel compares calibrated and uncalibrated CRPS.

Title: Figure 1: The best arm's CRPS is 10.0% of capacity, against 11.3% for production with cleaning, but still above the manual heuristic's 7.2%

Subtitle lines:
- Each dot is an arm's mean score over 31 series (solar, wind, demand, battery, and biofuel) and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Each score is a share of the series' own effective capacity.
- Each line is the 95% interval from resampling whole months. Month-to-month swings are shared by all arms, so overlapping intervals can hide a real gap (see Figure 2).
- Every arm is trained on one XGBoost model per series from the same weather; arms differ only in the one change in the label.
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stopping is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, and pooled is one model per series type with a site identifier.

Panel titles:
- CRPS of each arm, best first (scores the whole ensemble)
- MAE of each arm, in the same order (scores the ensemble mean only)
- Calibrated against uncalibrated, on the 45 runs after the first 8 (not comparable with the panels above; no interval shown)

Axis titles:
- Continuous ranked probability score, CRPS (% of capacity; smaller is better)
- Mean absolute error of the ensemble mean, MAE (% of capacity; smaller is better)
- CRPS on the 45 later forecast runs only (% of capacity; smaller is better)

Legend entries:
- Production (reference)
- Round 0: cleaning
- Round 1: columns shown
- Round 1: training
- Round 2: combinations
- Spread calibration: uncalibrated (blue)
- Spread calibration: calibrated (green)

Annotations:
- Dashed line, Manual heuristic: 7.2%
- Dashed line, Climatology: 8.9%
- Dashed line, Production as stored: 11.6%
- The MAE panel carries the same three dashed lines, at their MAE.
- Manual heuristic and climatology are scored on the rows they share with production with cleaning; they are drawn as lines without intervals.

## Figure 2

Plots: Paired differences in CRPS and MAE (percentage points of capacity) between each round 1 arm and production with cleaning, with 95% month-resampled intervals.

Title: Figure 2: 7 of 12 round 1 ideas lower CRPS reliably; the best, per type, with site identifier, lowers it by 0.8 points of capacity

Subtitle lines:
- Each dot is the paired difference between an arm and production with cleaning in mean score, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Zero means no change.
- Each line is the 95% interval from resampling whole months with the same months drawn for both arms. A line that excludes zero is a reliable difference.
- (post hoc) marks an arm or rule chosen after seeing earlier results; a bold (planned) row was written into the study plan beforehand.

Panel titles:
- CRPS: scores the whole ensemble
- MAE: scores the ensemble mean only

Axis titles:
- Change in CRPS from production with cleaning (percentage points of capacity; more negative means better than production with cleaning)
- Change in MAE from production with cleaning (percentage points of capacity; more negative means better than production with cleaning)

Legend entries:
- Kind of change: Columns shown
- Kind of change: Training

Annotations:
- same as production with cleaning
- ← better than production with cleaning
- Row labels ending (planned) are in bold; rows ending (post hoc) were chosen later.

## Figure 3

Plots: Paired difference in CRPS (percentage points of capacity) from production with cleaning, for the main ideas and the best round 2 arm, one panel per series type.

Title: Figure 3: The gains come mostly from wind farms: abs. loss, stopping, lead, wind, pooled lowers their CRPS by 3.6 points of capacity

Subtitle lines:
- Each dot is the paired difference in mean CRPS between an arm and production with cleaning, within one type of series, over 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Zero means no change.
- Each line is the 95% interval from resampling whole months. Battery storage (1 series) and biofuel (1 series) are in the all-series figures but not drawn here.
- The best round 2 arm combines absolute-error loss, early stopping, the lead-time column, lean wind columns, and one model per series type with a site identifier.
- (post hoc) marks an arm chosen after seeing earlier results; unmarked arms were planned.

Panel titles:
- Solar farms (6 series)
- Wind farms (3 series)
- Demand, disaggregated (16 series)
- Raw flow (4 series)

Axis titles:
- Change in CRPS from production with cleaning (percentage points of capacity; more negative means better than production with cleaning)

Legend entries:
- Kind of change: Columns shown
- Kind of change: Training
- Kind of change: Combination

Annotations:
- same as production with cleaning
- ← better than production with cleaning
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stopping is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, and pooled is one model per series type with a site identifier.

## Figure 4

Plots: Paired differences in CRPS and MAE (percentage points of capacity) between each round 2 arm and production with cleaning, for all leads and for two lead bands.

Title: Figure 4: Combining ideas lowers CRPS by up to 1.4 points of capacity, at every lead

Subtitle lines:
- Each dot is the paired difference between an arm and production with cleaning in mean score, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Zero means no change.
- Each line is the 95% interval from resampling whole months. Days 0 to 2 are forecasts up to 72 hours after the run starts; days 3 to 10 are 72 to 264 hours.
- Every arm here was assembled after seeing the round 1 results, so every row is post hoc. The first row, absolute-error loss, is the best round 1 arm; each later row adds ideas.
- The horizontal axis has the same range as Figure 2.
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stopping is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, and pooled is one model per series type with a site identifier.

Panel titles:
- CRPS: scores the whole ensemble
- MAE: scores the ensemble mean only

Axis titles:
- Change in CRPS from production with cleaning (percentage points of capacity; more negative means better than production with cleaning)
- Change in MAE from production with cleaning (percentage points of capacity; more negative means better than production with cleaning)

Legend entries:
- All leads
- Days 0 to 2
- Days 3 to 10

Annotations:
- same as production with cleaning
- ← better than production with cleaning

## Figure 5

Plots: Paired differences in CRPS and MAE (percentage points of capacity) between each training-data cleaning rule and production, and below them the per-series change for the chosen rule.

Title: Figure 5: Dropping each new meter's first 6 weeks lowers CRPS by 0.3 points of capacity, and only 2 series change

Subtitle lines:
- Each dot is the paired difference between a cleaning rule and production, which uses the minimal cleaning already in the stored power table, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Rules change only the training data. Zero means no change.
- Each line is the 95% interval from resampling whole months. Dropping the first 6 months of each series was also tried but removes a whole series, so it was not scored.
- The chosen rule, drop new-meter settling, removes the 6 weeks after a demand meter first reads live. It was written after seeing the results of the other rules.
- (post hoc) marks an arm or rule chosen after seeing earlier results; a bold (planned) row was written into the study plan beforehand.

Panel titles:
- CRPS: scores the whole ensemble
- MAE: scores the ensemble mean only
- Change for each of the 2 series whose forecast changed (29 other series are exactly unchanged)

Axis titles:
- Change in CRPS from production (percentage points of capacity; more negative means better than production)
- Change in MAE from production (percentage points of capacity; more negative means better than production)
- Change from production for one series (percentage points of that series' capacity) (smaller is better)

Legend entries:
- Change in CRPS (blue)
- Change in MAE (orange)

Annotations:
- same as production
- ← better than production
- Demand 13 (2,722 half-hours removed)
- Demand 16 (2,025 half-hours removed)

## Figure 6

Plots: Share of outcomes inside the ensemble's 10% to 90% band before and after spread calibration (a), and the paired change in CRPS from calibrating, by series type (b), for production with cleaning and the best round 2 arm.

Title: Figure 6: Rescaling the ensemble spread lifts 80% band coverage from 11% to 60% and lowers CRPS by 1.9 points of capacity

Subtitle lines:
- Spread calibration (post hoc) multiplies how far each ensemble member sits from the ensemble mean by a factor chosen per series from earlier forecast runs only, so the ensemble mean and the MAE do not change.
- Scored only on the 45 forecast runs after the first 8, which are left out while history builds up; 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; seed 0. Panels a use the two arms named in the panel titles.
- In panel a, a dot is the share of outcomes that fell inside the ensemble's 10% to 90% band; a well-spread ensemble sits on the dashed 80% line. In panel b, a dot is the paired difference in CRPS (calibrated minus uncalibrated), and a line is the 95% interval from resampling whole months; zero means no change.

Panel titles:
- a. 10% to 90% band coverage, Production with cleaning
- a. 10% to 90% band coverage, Abs. loss, stopping, lead, wind, pooled
- b. Change in CRPS from calibrating, by series type

Axis titles:
- Share of outcomes inside the ensemble's 10% to 90% band (%; 80% is ideal)
- Change in CRPS from calibrating the spread (percentage points of capacity; more negative means better than uncalibrated)

Legend entries:
- Before calibration (blue)
- After calibration (green)
- Production with cleaning
- Abs. loss, stopping, lead, wind, pooled

Annotations:
- Dashed line: 80% band coverage if the ensemble were well spread
- same as uncalibrated
- ← better than uncalibrated
