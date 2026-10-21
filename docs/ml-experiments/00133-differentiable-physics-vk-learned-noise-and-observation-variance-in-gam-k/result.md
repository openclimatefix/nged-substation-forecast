---
id: "00133"
slug: differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k
part: "VK"
theme: differentiable-physics-forecast-engine
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/
  only (grid scores, chosen settings, prediction sets). Serving path: not touched. Degradation
  rule: not touched. Designs: more than one is defensible (the score the grid maximises, whether a
  setting is shared across hour bands, how the online variance enters the tails), and the plan
  pre-registers one of each. Callers: this folder's scripts only, which copy experiment 111's
  scripts and call the reviewed `studies.dp_filter_grid`, `studies.dp_assimilation`, `studies.dp_ens`,
  and `studies.dp_gam` modules without changing them.
hypothesis: >-
  On GAM-K's filter (a level and a transient per hour band, per substation), noise settings chosen
  per substation by the day-ahead score on a grid lower the day-ahead quantile score below the
  fixed settings of experiment 111, which were chosen by marginal likelihood on 184 non-contiguous
  days. An online observation variance in the manner of VIKING (de Vilmarest and Wintenberger,
  2022) then improves the coverage of the 0.95 quantile on cold nights, which experiment 125 found
  at 73% to 78%, because the spread widens when recent errors have been large.
prediction: >-
  C1 (grid-chosen per substation minus fixed, quantile score at 12 to 36 hours, scored year, five
  decision substations) is -0.02 to -0.08 points, with the replicate gain at most 60% of the
  tuning-month gain. C2 (one shared dimensionless setting scaled per substation, minus the
  per-substation grid) is within +/-0.03. C3 (online observation variance with the tail scaling,
  minus the best fixed arm) is -0.01 to -0.04 points in the quantile score, and lifts the
  coverage of the 0.95 quantile on cold nights by 3 to 8 points, to between 78% and 85%, still
  short of 95%. The fixed arm reproduces experiment 111's GAM-K rows to within 1e-4 points.
decision_rule: >-
  C1 is the headline; C2 and C3 are planned contrasts, three in all, each read at the 99% level
  of the month-resampled paired bootstrap (12 calendar months of the scored year, 20,000
  resamples). C1 is `supported` when its scored-year interval lies wholly below zero, its
  replicate estimate has the same sign, and at least 4 of 5 decision substations improve in both
  periods. The threshold for C1 is zero: choosing a setting per substation on a grid adds no
  pipeline step beyond the likelihood fit it replaces, and the extra filter runs are cheap. C3
  needs a larger gain, because an online variance adds a state to the filter and a rule to the
  spread: it is `supported` only when its interval is wholly below -0.03 points at 12 to 36 hours
  or its cold-night coverage of the 0.95 quantile rises by at least 5 points with the pooled
  quantile score no worse. Otherwise `small gain` (interval below zero, estimate under the bar),
  `refuted` (interval wholly above zero or the estimate above +0.02 with the interval wholly above
  zero), or `inconclusive`. The verdict is capped at `inconclusive` if the reproduction control
  fails. No weight is claimed to be identified: the experiment compares forecast scores only.
benchmark: >-
  GAM-K as stored by experiment 111 (2.800 points over leads 0.5 to 168 hours at the five decision
  substations in the scored year), reproduced; climatology and manual_heuristic as in experiment
  111 for reference
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, at the five decision
  substations D5, D10, D11, D14, and D16 (D9 beside them), at leads 12 to 36 hours (D+1) with
  every lead beside it, then winter (November to March), winter nights, cold nights, and the
  Christmas and bank-holiday periods broken out; coverage and pinball loss of the 0.01, 0.05,
  0.95, and 0.99 quantiles
code: ml-experiments/00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/
---

# Experiment 133: do learned noise settings beat the fixed ones in GAM-K's filter?

**Planned; this page is the pre-registration and is rewritten when the experiment finishes.**

We ask whether the Kalman filter of the convex demand model with the filter (GAM-K, [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md)) forecasts the
next day better when its four noise settings are chosen by how well they forecast the next day,
instead of by likelihood, and whether letting the observation variance follow recent errors
improves the tails of the forecast. The filter tracks, for each of four hour bands, a slowly
moving level and a quickly decaying transient of the convex model's error. Its four settings are
the level's variance per day (`q`), the transient's decay factor (`phi`) and innovation standard
deviation (`sigma_z`), and the observation standard deviation (`sigma_eps`). Experiment 111 fitted
them per substation by marginal likelihood on the 184 days of the odd months of 2024-07 to
2025-06. Experiments [118](../00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/result.md)
and 128 found that this kind of likelihood is flat and scatters about 10 times between fits, so
the experiment avoids it. It runs the filter on a grid of 4,032 settings per substation on the
GPU through `studies.dp_filter_grid`, scores each setting on the day-ahead forecasts of the odd
months, and judges the chosen settings on the even months and on the scored year.

The experiment re-implements published methods: the variance tracking is a scalar version of
[de Vilmarest and Wintenberger (2022)](https://arxiv.org/abs/2104.10777)'s VIKING, applied as
[de Vilmarest and Goude (2022)](https://arxiv.org/abs/2110.00334) did to a state-space model of
electricity load. The scale-and-share pooling check follows the pooling arms of experiments 107
and 111.

## Pre-registered design

- **Sizing, one line per trigger.** What gets stored: private grid scores, chosen settings, and
  prediction sets only. Serving path: not touched. Degradation rule: not touched. More than one
  defensible design: yes, and the three choices below are fixed here. Callers: this folder's
  scripts only.
- **Rows.** `fold_a_daily_gap_v1`'s 292 scored daily runs at D5, D10, D11, D14, and D16, with D9
  beside them and D18 excluded because it has no readings in the tuning months. Tuning is the odd
  months of 2024-07 to 2025-06, replicate is the even months, and the scored year is
  2025-07-01 to 2026-06-30. The state runs warm through every month; only readings and cases of
  the tuning months count towards a score used to choose a setting.
- **Forecasts.** Every arm forecasts the ECMWF ensemble (ENS) weather through experiment 111's
  stored monthly fits of the convex model, calibration, and solar shrink (1.0 in every band).
  Only the filter's trace changes between arms. Each arm refits its spread tables on its own
  tuning-month residuals and takes a spread factor per lead band from {0.9, 1.0, 1.15, 1.3} by
  the tuning score at the decision substations, as experiment 111's `e00111_tune.py` does.
  The filter's weather is the analysis proxy, which exists at forecast time (the control member of
  the latest ENS run), so nothing here uses ERA5 at lag times.
- **The grid.** `log q` in {-4.0, -3.5, ..., 3.5} (16 values), `logit phi` in {1.0, 1.4, 1.8, 2.2,
  2.6, 3.0, 4.0}, `log sigma_z` in {0.0, 0.35, 0.7, 1.05, 1.4, 1.75}, and `log sigma_eps` in
  {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5}, one shared setting across the four hour bands. The grid
  brackets every fixed value of experiment 111 (`q` 0.4 to 9.5, `phi` 0.81 to 0.90, `sigma_z` 2.0
  to 4.1, `sigma_eps` 0.22 to 0.75). A substation whose best cell is on the grid's edge is
  reported.
- **The grid's score.** For each setting, the mean absolute error of the proxy-driven residual
  stream after the filter's correction at day-ahead cases: the daily 09:00 UTC origin of each
  tuning-month run, targets 12 to 36 hours ahead, state at the newest reading that ended 3.5 hours
  before the origin. The score uses no ensemble weather and no spread. This is a choice among
  defensible scores, fixed before any fit.

### Arms

| Arm | Settings |
|---|---|
| FIX (control) | Experiment 111's stored values per substation |
| GRID | The setting with the lowest day-ahead score per substation, on the odd months |
| POOL | One setting in units of each substation's residual standard deviation (`q` times the variance, the others times the standard deviation), chosen by the sum of the substations' scores relative to FIX |
| VK | GRID's `q`, `phi`, and `sigma_z`, with `sigma_eps` replaced by an online estimate; the same stream, with the tail scaling below |

**VK's observation variance follows the scalar variant of VIKING.** The log observation variance
`a_t` is a Gaussian random walk with innovation variance `rho_a`, with a variational posterior
`N(a_hat, s)`. At each reading the filter runs `N = 2` rounds: the Kalman update with observation
variance `exp(a_hat - s/2)`, then Equation 6 for `s` and Equation 7 for `a_hat` of the paper (with
`x = (1, 1)` and `x' P x` the posterior variance of level plus transient). The start is GRID's
`log sigma_eps`, with `s` at 1. `rho_a` is chosen on the tuning months from {e^-9, e^-6, e^-3}.
The paper's learning of the noise matrix (its steps 5, 8, and 9) is not run: the grid already picks
the two diagonal process-noise entries (the level's `q` and the transient's `sigma_z`) by score.

**VK's tail rule.** Each run's spread table is multiplied by
`clip(sqrt(exp(a_hat_cutoff) / median(exp(a_hat) over the tuning months)), 0.8, 1.5) ^ kappa`
where `a_hat_cutoff` is the estimate after the newest reading allowed at the run's cutoff, and
`kappa` is chosen on the tuning months from {0, 1}. `kappa = 0` is VK with the mean correction
only. The factor changes the spread around the mean, not the mean.

### Planned contrasts

| Contrast | Meaning | Prediction |
|---|---|---|
| C1 | GRID minus FIX | -0.02 to -0.08, replicate gain at most 60% of the tuning gain |
| C2 | POOL minus GRID | within +/-0.03 |
| C3 | VK minus the lower-scoring of GRID and POOL on the tuning months | -0.01 to -0.04; cold-night coverage of the 0.95 quantile +3 to +8 points |

### Controls and the gate

- **P1, reproduction (blocking).** FIX's scored-year rows match experiment 111's stored GAM-K
  rows to within 1e-4 points in the median forecast.
- **P2, selection on known answers.** On streams simulated from the filter's own model with
  settings that equal each decision substation's fixed values, plus a level shift, the grid's
  choice by day-ahead score reaches a day-ahead error within 2% of the true setting's. This tests
  the selection, not the recovery of the parameters: the experiment makes no claim to identify
  any weight.
- **Tracking gate on real residuals and AR(1) noise.** A level shift shaped like a rise in solar
  capacity (30% of the substation's midday PV effect over 180 days) is added to each decision
  substation's real tuning-year residuals (12 circular month-block draws) and to AR(1) noise at
  the filter's own settings (6 draws), paired with the same noise unshifted. The report gives, for
  FIX and for each chosen setting, the share of draws whose filtered midday level reaches 50% of
  the injected shift on the read day, next to experiment 128's figures for GAM-K with the fixed
  values (0.22 on real residuals, 0.80 on AR(1) noise). It is a diagnostic on how much a setting
  chosen for the day-ahead score lets the level move; the experiment's own re-implementation of
  the gate is not the same code as experiment 128's, which is not on `main`. A chosen setting
  whose tracking falls more than 0.10 below FIX's is flagged and listed with the caveats.
- **Overfitting guard.** The tuning-month gain of each arm beside its replicate gain and its
  scored-year gain, per substation.

### Reporting

- **Headline figure:** quantile score at 12 to 36 hours by arm with 99% month-resampled
  intervals, then every lead band beside it, per substation (the five decision substations).
- **Breakouts, descriptive:** winter, winter nights, cold nights (hour bands 0 and 3, ERA5
  temperature at or below 3 degrees C), and the Christmas and bank-holiday periods.
- **Tail table:** coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, and the coverage of the
  0.95 quantile on cold nights, for FIX, GRID, POOL, and VK.
- **Fitted values** per substation next to the fixed ones, and the grid's score surface in `q`
  and `sigma_eps` at the best `phi` and `sigma_z`.
- **Checks bullets:** median minus mean forecast per site, the mean-forecast error beside each
  quantile-score contrast, and which audit triggers fired.
- **Not reported:** the curtailment-cost saving against the manual heuristic, because it needs
  the node and generator forecasts of a node, and this experiment compares filter settings on
  demand forecasts of single Primaries.
- Retraining schedule: monthly refits of the convex model, as in experiment 111. The contrast
  between filter settings does not depend on the refit schedule of the fits, because every arm
  reads the same stored fits.

## Compute

GPU: `run_kalman_grid` over 5 to 6 streams and 4,032 settings, about 1 GPU hour for the grid and
the controls on one job at a time under 24 GB. CPU: the arms' forecasts take about 35 s per arm
and substation, run in at most 6 processes. Memory: `systemd-run --user --scope -p MemoryMax=16G
-p OOMPolicy=continue`.
