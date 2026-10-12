# Quick-wins figure text

## Figure 1

Plots: Fold mid_2025_to_mid_2026: CRPS and MAE (% of capacity) of every arm, all series types, all leads, with 95% month-resampled intervals; baselines and production as lines; a third panel compares calibrated and uncalibrated CRPS.

Title: Figure 1: The lowest CRPS is 7.8% of capacity ("quantile, lead, wind, pooled"), against 11.3% for production with cleaning, but still above the manual heuristic's 7.2%

Subtitle lines:
- Each dot is an arm's mean score over 31 series (solar, wind, demand, battery, and biofuel) and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; average of seeds 0, 1, 2. Each score is a share of the series' own effective capacity.
- Each line is the 95% interval from resampling whole months. Month-to-month swings are shared by all arms, so overlapping intervals can hide a real gap (see Figure 2).
- Every arm is trained on one XGBoost model per series from the same weather; arms differ only in the one change in the label.
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stop is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, pooled is one model per series type with a site identifier, and quantile is training on nine quantile levels with the pinball loss (the quantile arms' CRPS is scored on 51 members x 9 levels samples).

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
- Quantile training (post hoc)
- Round 2: combinations
- Spread calibration: uncalibrated (black)
- Spread calibration: calibrated (green)

Annotations:
- Dashed line, Manual heuristic: 7.2%
- Dashed line, Climatology: 8.9%
- Dashed line, Production as stored: 11.6%
- The MAE panel carries the same three dashed lines, at their MAE.
- Manual heuristic and climatology are scored on the rows they share with production with cleaning; they are drawn as lines without intervals.

## Figure 2

Plots: Paired differences in CRPS and MAE (percentage points of capacity) between each round 1 arm and production with cleaning, with 95% month-resampled intervals.

Title: Figure 2: 9 of 14 round 1 ideas lower CRPS reliably; the largest, "Quantile training, 9 levels", lowers it by 2.4 points of capacity

Subtitle lines:
- Each dot is the paired difference between an arm and production with cleaning in mean score, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; average of seeds 0, 1, 2. Zero means no change.
- Each line is the 95% interval from resampling whole months with the same months drawn for both arms. A line that excludes zero is a reliable difference.

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
- Added by the figure template: Planned: one of the comparisons written into the study plan before any result existed. Every other row is exploratory or, where marked, post hoc.
- Row labels ending (planned) are in bold; rows ending (post hoc) were chosen later.

## Figure 3

Plots: Paired difference in CRPS (percentage points of capacity) from production with cleaning, for the main ideas and the best round 2 arm, one panel per series type.

Title: Figure 3: The largest gains are in raw flow: quantile, lead, wind, pooled lowers their CRPS by 6.0 points of capacity

Subtitle lines:
- Each dot is the paired difference in mean CRPS between an arm and production with cleaning, within one type of series, over 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; average of seeds 0, 1, 2. Zero means no change.
- Each line is the 95% interval from resampling whole months. Battery storage (1 series) and biofuel (1 series) are in the all-series figures but not drawn here.
- The best round 2 arm, "Quantile, lead, wind, pooled", is quantile training with the lead-time column, lean wind columns, and pooling.
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
- Kind of change: Quantile training
- Kind of change: Combination

Annotations:
- same as production with cleaning
- ← better than production with cleaning
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stop is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, pooled is one model per series type with a site identifier, and quantile is training on nine quantile levels with the pinball loss (the quantile arms' CRPS is scored on 51 members x 9 levels samples).

## Figure 4

Plots: Paired differences in CRPS and MAE (percentage points of capacity) between each round 2 arm and production with cleaning, for all leads and for two lead bands.

Title: Figure 4: Combining ideas lowers CRPS by up to 3.5 points of capacity, at every lead

Subtitle lines:
- Each dot is the paired difference between an arm and production with cleaning in mean score, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; average of seeds 0, 1, 2. Zero means no change.
- Each line is the 95% interval from resampling whole months. Days 0 to 2 are forecasts up to 72 hours after the run starts; days 3 to 10 are 72 to 264 hours.
- The first row, absolute-error loss, is the best round 1 arm; each later row adds ideas.
- The horizontal axis has the same range as Figure 2.
- Round 2 labels list the ideas combined: abs. loss is the absolute-error loss, stop is early stopping, lead is the forecast lead-time column, wind is the lean wind-farm columns, pooled is one model per series type with a site identifier, and quantile is training on nine quantile levels with the pinball loss (the quantile arms' CRPS is scored on 51 members x 9 levels samples).

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
- Each dot is the paired difference between a cleaning rule and production, which uses the minimal cleaning already in the stored power table, over 31 series and 53 weekly ECMWF ensemble (ENS) forecast runs, scored from July 2025 to June 2026; average of seeds 0, 1, 2. Rules change only the training data. Zero means no change.
- Each line is the 95% interval from resampling whole months. Dropping the first 6 months of each series was also tried but removes a whole series, so it was not scored.
- The chosen rule, drop new-meter settling, removes the 6 weeks after a demand meter first reads live. It was written after seeing the results of the other rules.

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
- Added by the figure template: Planned: one of the comparisons written into the study plan before any result existed. Every other row is exploratory or, where marked, post hoc.
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
- a. 10% to 90% band coverage, Abs. loss, stop, lead, wind, pooled
- b. Change in CRPS from calibrating, by series type

Axis titles:
- Share of outcomes inside the ensemble's 10% to 90% band (%; 80% is ideal)
- Change in CRPS from calibrating the spread (percentage points of capacity; more negative means better than uncalibrated)

Legend entries:
- Before calibration (black)
- After calibration (green)
- Production with cleaning
- Abs. loss, stop, lead, wind, pooled

Annotations:
- Dashed line: 80% band coverage if the ensemble were well spread
- same as uncalibrated
- ← better than uncalibrated

## Figure 7

Title: Figure 7: Forecasts of a PV farm follow the daily solar cycle but often miss the height of each day's peak

Subtitle:
- Black line: actual power of PV farm A. Coloured lines: each forecast's ensemble mean (the average of the 51 ensemble members), as a percentage of the series' effective capacity. Zero is no power.
- Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to June 2026), chosen by the actual power alone and not by forecast error.
- Panels run from the run with the highest mean actual power (top) to the lowest (bottom). The series is the lower-middle one of its type when ranked by the cleaned production forecast's mean normalised CRPS (continuous ranked probability score) over the 53 weekly runs.

Panel titles:
- PV farm A: Run with the highest daily energy in its first 7 days
- PV farm A: Run with the median daily energy in its first 7 days
- PV farm A: Run with the lowest daily energy in its first 7 days

X axis title: Days since the run's initialisation time (day 1 is the first 24 hours)
Y axis title: Power (% of effective capacity)

Legend entries:
- Actual power
- Production as stored
- Production, cleaned training data
- Best round 2 arm (post hoc)

Other text:
- Models were trained on one random seed (seed 0).

## Figure 8

Title: Figure 8: Forecasts of a wind farm follow the broad swings in power but miss its fast changes and highest peaks

Subtitle:
- Black line: actual power of Wind farm W3. Coloured lines: each forecast's ensemble mean (the average of the 51 ensemble members), as a percentage of the series' effective capacity. Zero is no power.
- Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to June 2026), chosen by the actual power alone and not by forecast error.
- Panels run from the run with the highest mean actual power (top) to the lowest (bottom). The series is the lower-middle one of its type when ranked by the cleaned production forecast's mean normalised CRPS (continuous ranked probability score) over the 53 weekly runs.

Panel titles:
- Wind farm W3: Run with the highest mean actual power in its first 7 days
- Wind farm W3: Run with the median mean actual power in its first 7 days
- Wind farm W3: Run with the lowest mean actual power in its first 7 days

X axis title: Days since the run's initialisation time (day 1 is the first 24 hours)
Y axis title: Power (% of effective capacity)

Legend entries:
- Actual power
- Production as stored
- A3w: lean wind columns (post hoc)
- Best round 2 arm (post hoc)

Other text:
- Models were trained on one random seed (seed 0).

## Figure 9

Title: Figure 9: Forecasts of a demand series follow its daily pattern but can sit well above or below its level

Subtitle:
- Black line: actual power of Demand 1. Coloured lines: each forecast's ensemble mean (the average of the 51 ensemble members), as a percentage of the series' effective capacity. Zero is no power.
- Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to June 2026), chosen by the actual power alone and not by forecast error.
- Panels run from the run with the highest mean actual power (top) to the lowest (bottom). The series is the lower-middle one of its type when ranked by the cleaned production forecast's mean normalised CRPS (continuous ranked probability score) over the 53 weekly runs.

Panel titles:
- Demand 1: Run with the highest mean actual power in its first 7 days
- Demand 1: Run with the median mean actual power in its first 7 days
- Demand 1: Run with the lowest mean actual power in its first 7 days

X axis title: Days since the run's initialisation time (day 1 is the first 24 hours)
Y axis title: Power (% of effective capacity)

Legend entries:
- Actual power
- Production as stored
- C4: absolute-error loss
- D3: pooled with site identifier (post hoc)
- Best round 2 arm (post hoc)

Other text:
- Models were trained on one random seed (seed 0).

## Figure 10

Title: Figure 10: Forecasts of a Raw Flow series follow its daily swing but miss longer departures from it

Subtitle:
- Black line: actual power of Raw Flow 1. Coloured lines: each forecast's ensemble mean (the average of the 51 ensemble members), as a percentage of the series' effective capacity. Zero is no power.
- Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to June 2026), chosen by the actual power alone and not by forecast error.
- Panels run from the run with the highest mean actual power (top) to the lowest (bottom). The series is the lower-middle one of its type when ranked by the cleaned production forecast's mean normalised CRPS (continuous ranked probability score) over the 53 weekly runs.

Panel titles:
- Raw Flow 1: Run with the highest mean actual power in its first 7 days
- Raw Flow 1: Run with the median mean actual power in its first 7 days
- Raw Flow 1: Run with the lowest mean actual power in its first 7 days

X axis title: Days since the run's initialisation time (day 1 is the first 24 hours)
Y axis title: Power (% of effective capacity)

Legend entries:
- Actual power
- Production as stored
- C4: absolute-error loss
- D3: pooled with site identifier (post hoc)
- Best round 2 arm (post hoc)

Other text:
- Models were trained on one random seed (seed 0).

## Figure 11

Title: Figure 11: The raw ensemble band is too narrow, and rescaling the spread widens it

Subtitle:
- Black line: actual power as a percentage of effective capacity. Blue shade: the band holding the middle 80% of the 51 ensemble members (10% to 90%) of the cleaned production forecast. Purple shade: the same band after the post hoc B1 spread calibration, which rescales each series' members about their mean using only past outcomes. The ensemble mean, and so the mean absolute error, is unchanged.
- A well-calibrated band holds the actual power 80% of the time. Each panel shows one forecast run from the weekly runs of the validation year (July 2025 to June 2026): the run with the median mean actual power in its first 7 days among the runs the calibration scored, chosen by the actual power alone.

Panel titles:
- PV farm A: median run (by daily energy); this series' spread was multiplied by 8.0
- Wind farm W3: median run (by mean actual power); this series' spread was multiplied by 8.0

X axis title: Days since the run's initialisation time (day 1 is the first 24 hours)
Y axis title: Power (% of effective capacity)

Legend entries:
- Actual power
- 10% to 90% member band, production (cleaned)
- 10% to 90% member band after spread calibration (post hoc)

Other text:
- PV, all runs after the first 8: share of actual values inside the band 16% before and 70% after calibration (nominal 80%).
- Wind, all runs after the first 8: share of actual values inside the band 16% before and 65% after calibration (nominal 80%).

## Figure 12

Title: Figure 12: The best round 2 arm has a lower error than production at every lead day, for all four series types

Subtitle:
- Mean absolute error of the ensemble mean (the average of the 51 ensemble members) as a percentage of effective capacity; smaller is better, zero is a perfect forecast. Each point averages every half-hourly forecast of the type's series, over all 53 weekly runs of the validation year (July 2025 to June 2026).
- Day 1 is the first 24 hours after a run's initialisation time. Leads of 11 days and more are left out. The best round 2 arm was chosen after seeing round 1 (post hoc).

Panel titles:
- Mean over the 6 PV farms
- Mean over the 3 Wind farms
- Mean over the 16 Disaggregated Demand series
- Mean over the 4 Raw Flow series

X axis title: Lead day (day 1 is the first 24 hours after the run's initialisation)
Y axis title: Mean abs. error (% of capacity)

Legend entries:
- Production as stored
- Best round 2 arm (post hoc)

Other text:
- Models were trained on one random seed (seed 0).

## Figure 13

Title: Figure 13: XGBoost beats the manual heuristic at days 0 to 2 for PV, wind, and Raw Flow, and is usually behind it from day 3

Subtitle:
- Each dot is the mean over the series of a type of each series' mean score, as a percentage of the series' own effective capacity; zero is a perfect forecast. The last column averages all 31 series. The scores come from 53 weekly ECMWF ensemble forecast runs, July 2025 to June 2026 (fold A), on the rows that all five models scored. XGBoost models were trained with one random seed (seed 0). No intervals are shown.
- Lead is the time from a run's initialisation to the valid time: days 0 to 2 is under 72 hours, days 3 to 10 is 72 to 263 hours, and 11+ is 264 hours and more. 
- The best arms were chosen after seeing earlier results (post hoc). The quantile arm trains on 9 quantile levels and is scored on its 51 members at each of 9 levels (459 values).

Panel titles:
- Mean absolute error of the ensemble mean, MAE (% of capacity; smaller is better)
- Continuous ranked probability score, CRPS, of the whole ensemble (% of capacity; smaller is better)
- Column: PV farms (6 series)
- Column: Wind farms (3 series)
- Column: Demand series (16 series)
- Column: Raw Flow series (4 series)
- Column: All types (31 series)

X axis title: Lead (days)
Y axis title: (none: the panel title carries the metric, unit and direction)

Legend entries:
- Manual heuristic
- Climatology
- XGBoost, production with cleaning
- XGBoost, best point-forecast arm (post hoc)
- XGBoost, best quantile-trained arm (post hoc)

Other text:
- In 78% of the 40 comparisons of an XGBoost arm with the heuristic at days 3 to 10 and 11+ (both arms, both metrics, five columns), the heuristic's error is lower.

## Figure 14

Title: Figure 14: Over all series types, more months of training data lower the CRPS, and the last 3 months add the least

Subtitle:
- Each dot is the continuous ranked probability score (CRPS) of a model trained on only the last 6, 9, or 12 months of the 15-month training window, minus the CRPS of the production arm with cleaning, which uses all 15 months. Zero (dashed line, and the 15-month dot by definition) is no change; above zero is a worse forecast.
- Each vertical line is a 95% interval from resampling whole months of the validation year, July 2025 to June 2026 (53 weekly forecast runs, fold A), with the same months drawn for both arms. Models were trained with one random seed (seed 0). A shorter window drops whole seasons from the training data. The first column also includes the one battery and the one biofuel series.

Panel titles:
- CRPS of a model trained on only the last N months, minus the CRPS of the 15-month production arm with cleaning (points of capacity; smaller is better)
- Column: All types
- Column: PV farms
- Column: Wind farms
- Column: Demand series
- Column: Raw Flow series

X axis title: Months of training data
Y axis title: (none: the panel title carries the unit and direction)

Legend entries:
- (none: one colour)

Other text:
- All-types differences from the 15-month arm, in points of capacity: 6 months +0.48, 9 months +0.31, 12 months +0.08, 15 months +0.00.
