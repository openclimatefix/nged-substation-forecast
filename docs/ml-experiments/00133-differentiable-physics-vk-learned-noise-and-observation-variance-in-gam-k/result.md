---
id: '00133'
slug: differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k
part: VK
theme: differentiable-physics-forecast-engine
also_relevant_to:
- lagged-power-and-error-correction
- calibration-and-uncertainty
status: planned
size: 'medium. Stored: private files under data/ml-experiments/00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/
  only (grid scores, chosen settings, and prediction sets). Serving path: not touched. Degradation
  rule: not touched. Designs: more than one is defensible (the score the grid maximises, whether
  a setting is shared across hour bands, how the online variance enters the tails), and the
  plan pre-registers one of each. Callers: this folder''s scripts only, which copy experiment
  111''s scripts and call the reviewed `studies.dp_filter_grid`, `studies.dp_assimilation`,
  `studies.dp_ens`, and `studies.dp_gam` modules without changing them.'
hypothesis: On GAM-K's filter (a level and a transient per hour band, per substation), noise
  settings chosen per substation by day-ahead score lower the day-ahead quantile score below
  the fixed settings of experiment 111, which were chosen by marginal likelihood on 184 non-contiguous
  days. An online observation variance in the manner of VIKING (de Vilmarest and Wintenberger,
  2022), used to scale the spread, then improves the coverage of the 0.95 quantile on cold nights,
  which experiment 125 found at 73% to 78%, because recent large errors widen the spread. The
  mechanism under test is that the size of recent proxy-driven residuals predicts the size of
  the next day's error on cold nights.
prediction: C1 (GRID minus FIX, quantile score at 12 to 36 hours, scored year, five decision
  substations) is -0.02 to -0.08 points, with the replicate gain at most 60% of the tuning gain.
  C2 (POOL minus GRID) is within +/-0.03. C3a (VK with the online variance only minus GRID)
  is within +/-0.02. C3b (VK with the tail scaling minus GRID) is -0.01 to -0.04 points, and
  lifts the coverage of the 0.95 quantile on cold nights by 3 to 8 points over FIX's coverage,
  which is still short of 95%. The FIX arm reproduces experiment 111's GAM-K rows to within
  1e-4 points.
decision_rule: 'Four planned contrasts (C1, C2, C3a, C3b), each read at the 99% level of the
  month-resampled paired bootstrap (12 calendar months of the scored year, 20,000 resamples);
  the 99% level is the multiplicity correction, and each write-up row gives the number of the
  12 months with a negative difference. C1 is the headline. It is `supported` when its scored-year
  interval lies wholly below zero, its replicate estimate has the same sign, at least 4 of 5
  decision substations improve in both periods, and the score over all leads 0.5 to 168 hours
  has an interval upper bound of at most +0.02. It is `small gain` when the scored-year interval
  lies wholly below zero but fewer than 4 of 5 substations improve or the replicate sign differs.
  It is `refuted` when the scored-year interval lies wholly above zero. It is `inconclusive`
  otherwise, which includes a tuning-month gain with no scored-year gain (the scored year decides,
  the replicate months are the check). The threshold for C1 is zero: choosing a setting per
  substation by score replaces the likelihood fit and adds no pipeline step. C2 is `equivalent`
  when its 99% interval lies within +/-0.03, and otherwise its sign decides. C3a and C3b add
  a state or a rule to the filter, so each is `supported` only when its interval is wholly below
  -0.03 points at 12 to 36 hours, or when the coverage of the 0.95 quantile on cold nights rises
  by an amount whose month-resampled 99% interval lower bound is above zero and the all-lead
  quantile score has an interval upper bound of at most +0.01. Otherwise the same `small gain`,
  `refuted`, and `inconclusive` rules apply to them. The verdict is capped at `inconclusive`
  if the reproduction control P1 fails. No weight is claimed to be identified: the experiment
  compares forecast scores only.'
benchmark: GAM-K as stored by experiment 111 (2.800 points over leads 0.5 to 168 hours at the
  five decision substations in the scored year), reproduced; climatology and manual_heuristic
  as in experiment 111 for reference
row_set_id: fold_a_daily_gap_v1
metric: Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87)
  in points of effective capacity, on rows outside the hand-filtered switching events, at the
  five decision substations D5, D10, D11, D14, and D16 (D9 beside them), at leads 12 to 36 hours
  (D+1) with every lead beside it, then winter (November to March), winter nights, cold nights,
  and the Christmas and bank-holiday periods broken out; coverage and pinball loss of the 0.01,
  0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k/
---

# Experiment 133: do learned noise settings beat the fixed ones in GAM-K's filter?

**Planned; this page is the pre-registration and is rewritten when the experiment finishes.**

**The experiment asks whether the filter of the convex demand model forecasts the next day better
when its noise settings are chosen by next-day score instead of by likelihood.** The model is GAM-K,
the convex demand model with the Kalman filter of [experiment
111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md). The filter
tracks, for each of four hour bands, a slowly moving level and a quickly decaying transient of the
convex model's error. Its settings are the level's variance per day (`q`), the transient's decay
factor per half-hour (`phi`) and innovation standard deviation (`sigma_z`), and the observation
standard deviation (`sigma_eps`). Experiment 111 fitted them per substation by marginal likelihood
on the 184 days of the odd months of 2024-07 to 2025-06.

**The likelihood is avoided because it is flat.** [Experiment
118](../00118-differentiable-physics-kfe-kalman-filter-on-the-gam-effects-on-ens/result.md) and
experiment 128 found that the likelihood of a filter's noise settings on 184 non-contiguous days
scatters about 10 times between fits. This experiment instead runs the filter on a grid of settings
per substation on the GPU through `studies.dp_filter_grid`, ranks the cells by next-day error on the
odd months, and judges the chosen settings on the even months and on the scored year. It also asks
whether an online observation variance improves the tails of the forecast.

**The experiment re-implements published methods.** The online observation variance is the scalar
variant of VIKING, from [de Vilmarest and Wintenberger (2022)](https://arxiv.org/abs/2104.10777),
and the use of a state-space model with a noise-adaptive filter for electricity load follows [de
Vilmarest and Goude (2022)](https://arxiv.org/abs/2110.00334). The pooling arm follows the pooled
arms of experiments 107 and 111. The cold-night coverage baseline is [experiment
125](../00125-differentiable-physics-n8-ens-members-through-gam-k-and-the-spread-around-them/result.md).

## Pre-registered design

- **Sizing, one line per trigger.** What gets stored: private grid scores, chosen settings, and
  prediction sets only. Serving path: not touched. Degradation rule: not touched. More than one
  defensible design: yes, and the choices below are fixed here. Callers: this folder's scripts only.
- **Rows.** `fold_a_daily_gap_v1`'s 292 scored daily runs at D5, D10, D11, D14, and D16, with D9
  beside them. D18 is excluded because it has no readings in the tuning months. Tuning is the odd
  months of 2024-07 to 2025-06, replicate is the even months, and the scored year is 2025-07-01 to
  2026-06-30.
- **What is tuned on the odd months only.** Experiment 111's monthly convex-model fits train on rows
  that end before each cut, and its calibration, solar shrink, and spread factors were fitted on the
  tuning months, so the replicate months and the scored year are out of sample for every arm. The
  filter is causal, so it runs warm through every month. Only cases whose run belongs to a tuning
  month count towards a score that chooses a setting.
- **Forecasts.** Every arm forecasts the ECMWF ensemble (ENS) weather through experiment 111's
  stored fits, calibration, and solar shrink. Only the filter's trace and the spread differ between
  arms. Each arm refits its spread tables on its own tuning-month residuals and takes a spread
  factor per lead band from {0.9, 1.0, 1.15, 1.3} by the tuning score at the decision substations,
  as `e00111_tune.py` does. The filter's weather is the analysis proxy, which exists at forecast
  time (the control member of the latest ENS run), so no ERA5 enters at lag times.
- **Excluded by the brief.** The 2-nat sparse-`Q` rule and a day-level shared anomaly are not used,
  because experiment 128's gate failed on both. The learned diagonal process noise of the proposal
  is replaced by the grid: the two-state filter has two process-noise entries (`q` and `sigma_z`),
  and the grid chooses both by score.
- **`sigma_eps` is held at 1 for the grid arms.** The correction depends on `q`, `sigma_z`^2, and
  `sigma_eps`^2 only through their ratios (apart from the fixed initial level variance of 100), so a
  score on the corrected mean cannot identify `sigma_eps` separately. The grid therefore chooses `q`
  and `sigma_z` in units of `sigma_eps`, and fitted values are reported as `q / sigma_eps^2` and
  `sigma_z / sigma_eps`, for FIX as well.
- **The grid.** `log q` in {-4.0, -3.5, ..., 6.0} (21 values), `logit phi` in {1.0, 1.4, 1.8, 2.2,
  2.6, 3.0, 4.0}, and `log sigma_z` in {-0.5, 0.0, ..., 3.5} (9 values), shared across the four hour
  bands: 1,323 cells. Each substation's FIX setting, converted to these units, is added as a cell,
  so C1 does not include a loss from grid resolution. A substation whose best cell is on the grid's
  edge is reported.
- **Choosing a cell in two steps.** First, the day-ahead error of each cell on the proxy-driven
  residual stream: the mean absolute error of the corrected residual at the day-ahead cases of the
  tuning-month runs, with cutoffs from `usable_cutoff_positions` on `setup.origins` and
  `DEFAULT_DELAY_HOURS`, targets 12 to 36 hours ahead. This step needs no ensemble weather. Second,
  the 20 cells with the lowest error per substation are re-scored by the quantile score at 12 to 36
  hours of the full ENS forecast (spread refitted for each cell) on the tuning months, and the
  lowest wins. The report gives the rank correlation between the two scores, and the winner's curse
  is bounded by the replicate and scored-year results.

### Arms

| Arm | Settings |
|---|---|
| FIX (control) | Experiment 111's stored values per substation |
| GRID | The cell chosen per substation by the two-step rule |
| POOL | One cell in units of each substation's residual standard deviation (`q` times the variance, `sigma_z` times the standard deviation), chosen by the sum over substations of the relative two-step score |
| VK | GRID's cell with `sigma_eps` replaced by an online estimate, as `kappa` = 0 (mean correction only, row VK0) and `kappa` = 1 (with the tail scaling, row VK1) |
| TRAIL (descriptive) | GRID's cell with the tail scaling taken from the trailing 7-day root mean square of the filter's one-step innovations instead of the online variance, so VIKING is kept only if VK1 beats it |

**VK's observation variance follows the scalar variant of VIKING.** For each hour band, the log
observation variance `a` is a Gaussian random walk, with a variational posterior `N(a_hat, s)`.
Before each reading, `s` becomes `s + rho_a x gap`, with `gap` the half-hours since the band's
previous reading and `rho_a` chosen from {e^-9, e^-7, e^-5} per half-hour on the tuning months. The
reading's update runs `N = 2` rounds. Each round does a Kalman update with observation variance
`exp(a_hat - s/2)`, takes `s` from Equation 6 and `a_hat` from Equation 7 of the paper (with `Ma` =
3 times the prior `s`, and both equations on the updated residual `y - theta_hat' x` and `x' P x`,
the posterior variance of level plus transient, with `x` = (1, 1)). The start is `log sigma_eps` = 0
and `s` = 1. The paper's learning of the noise matrix (its steps 5, 8, and 9) is not run. The filter
is a new sequential loop in this folder, because `run_kalman_grid` has a constant observation
variance, and its unit test checks that VK with `rho_a` = 0 and `s` = 0 reproduces GRID's trace.

**The tail scaling is applied to each row's spread.** Each row's factor is `clip(sqrt(exp(a_hat) /
median(exp(a_hat))), 0.8, 1.5)` with `a_hat` the estimate of the target half-hour's hour band after
the newest reading allowed at the run's cutoff, and the median over the tuning-month cases of the
same band. The spread tables are fitted on residuals divided by the same factor, and each quantile
is then taken as `mean + factor x (q - mean)`. The report logs the correlation between `a_hat` at
the cutoff and the later absolute error, which is the mechanism under test.

### Planned contrasts

| Contrast | Meaning | Prediction |
|---|---|---|
| C1 | GRID minus FIX | -0.02 to -0.08, replicate gain at most 60% of the tuning gain |
| C2 | POOL minus GRID | within +/-0.03 |
| C3a | VK0 minus GRID | within +/-0.02 |
| C3b | VK1 minus GRID | -0.01 to -0.04; cold-night coverage of the 0.95 quantile +3 to +8 points over FIX's |

### Controls and the gate

- **P1, reproduction (blocking).** FIX's scored-year rows match experiment 111's stored GAM-K rows
  to within 1e-4 points in the median forecast.
- **P2, selection on known answers.** On simulated streams with settings equal to each decision
  substation's FIX values, in two forms (the filter's own state-space model plus a level shift, and
  a misspecified form of a level plus a seasonal wobble plus AR(1) noise), the two-step choice
  reaches a day-ahead error within 2% of the true setting's. This tests the selection, not the
  recovery of any parameter.
- **Tracking gate on real residuals and AR(1) noise.** A level shift shaped like a rise in solar
  capacity (30% of the substation's midday PV effect over 180 days) is added to each decision
  substation's real tuning-year residuals (12 circular month-block draws per substation, 60 in all)
  and to AR(1) noise at the filter's own settings (12 draws per substation, 60 in all), each paired
  with the same noise unshifted. The report gives, for FIX and for each chosen setting, the share of
  draws whose filtered midday level reaches 50% of the injected shift on the read day, next to
  experiment 128's figures for GAM-K with the fixed values (0.22 on real residuals, 0.80 on AR(1)
  noise). The gate here is a re-implementation, because experiment 128's code is not on `main`, so
  its pass rate for FIX is reported as measured here and is not expected to equal experiment 128's.
  A chosen setting whose tracking falls more than 0.10 below FIX's is listed with the caveats.
- **Overfitting guard.** The tuning-month gain of each arm beside its replicate gain and its
  scored-year gain, per substation.

### Reporting

- **Headline figure:** quantile score at 12 to 36 hours by arm with 99% month-resampled intervals,
  then every lead band beside it, per decision substation.
- **Non-inferiority:** the all-lead score (0.5 to 168 hours) of every contrast, because the settings
  are tuned for 12 to 36 hours and applied at every lead.
- **Breakouts, descriptive:** winter (November to March), winter nights, cold nights (hour bands 0
  and 3 with ERA5 temperature at or below 3 degrees C), and the Christmas and bank-holiday periods.
- **Tail table:** coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, and the coverage of the
  0.95 quantile on cold nights with a month-resampled interval, for FIX, GRID, POOL, VK0, VK1, and
  TRAIL.
- **Fitted values** per substation next to the fixed ones, in the ratio units above, and the grid's
  error surface in `q` and `sigma_z` at the best `phi`.
- **Checks bullets:** median minus mean forecast per site, the mean-forecast error beside each
  quantile-score contrast, and which audit triggers fired.
- **Not reported:** the curtailment-cost saving against the manual heuristic, because it needs the
  forecasts of a node and its generators, and this experiment compares filter settings on the demand
  forecasts of single Primaries.
- **Retraining schedule:** monthly refits of the convex model, as in experiment 111. Every arm reads
  the same stored fits, so the contrasts between filter settings do not depend on that schedule.

## Compute

The grid takes about 1 GPU hour at most on one job at a time under 24 GB (`run_kalman_grid` over 6
streams and 1,323 cells, and the gate). The ENS forecasts take about 35 s per arm and substation,
run in at most 6 processes on the CPU, and the 20-cell re-scoring adds about 20 tuning-month
forecasts per substation. Memory: `systemd-run --user --scope -p MemoryMax=16G -p
OOMPolicy=continue`.
