# Differentiable physics: the forecast engine: research plan

**Part 6 of the series was the first experiment of this theme, and its follow-ups wait on the
maintainer.** The rows below are proposals, not agreed work, and the priorities are a suggested
order for the maintainer to change.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Margin arms at weekly cadence (proposed) | Does the DP's forecast as the starting point of lagged XGBoost help when the XGBoost model refits weekly, as the production recipe does, rather than monthly? | 3 | The maintainer's GPU budget (about 60 GPU-hours at the measured speed) | Proposed |
| ENS history of 2021 to 2024 (proposed) | Do three more years of ensemble history give XGBoost a fair training history, and a second validation year for the DP against XGBoost? | 2 | The maintainer's approval of a fetch from Source Cooperative | Proposed |
| Filter in the production serving path (proposed) | Does a Kalman filter on the recent residuals of the production XGBoost blend lower its day-ahead score as it does for the convex model? | 1 | Nothing; a candidate for the next release review | Proposed |
| W1, a diurnal-shape state | Does a second filter state for the shape of the day lower the convex model's day-ahead error? | Running | Nothing | Running (experiment 115) |
| N8, ensemble members through the DP | Does running each ensemble member through the DP give better-calibrated spread than residual quantiles? | 4 | Part 6 merged, and a free GPU | Proposed |
| N7, paired residuals at switching events | Do the donor's and the receiver's residuals sum to roughly zero where both sides are metered? | 5 | Nothing | Proposed |
| Tail quantile regression on filtered residuals | Does a quantile regression on the Kalman residuals calibrate the tails better than residual quantiles? | 6 | CAL, if the maintainer restarts it | Proposed |
| Conditional Pareto tails | Does a Pareto distribution with covariate-dependent scale calibrate the 1% and 99% quantiles? | 7 | CAL, and the quantile regression above | Proposed |
| Learned observation variance | Does tracking the filter's observation variance online beat a fixed value? | 8 | Nothing | Proposed |
| CAL, tail calibration and coherence | Can the convex model's tails on ensemble weather be calibrated per lead band, and do its summed ensembles match the observed sum across substations? | Postponed | The maintainer's decision to restart (postponed on 2026-10-20) | Postponed |

## CAL: tail calibration and a coherence pass

**The maintainer postponed CAL on 2026-10-20.** The question is whether the tails of the convex
model with the Kalman filter (GAM-K) on ECMWF ensemble weather are calibrated in every lead band,
and which re-spreading of the same mean forecast fixes the bands that are not. Experiments 107 and
111 found the pooled tails close to the nominal 95% and 99%, so the work is in particular cells: some
substations in winter, the night hours, and one or two lead bands. Curtailment depends on the upper
tail, so those cells matter more than the pooled figure.

**The mean forecast stays fixed, and only the spread changes.** Five methods are candidates, each
reusing the stored fits of experiment 111 so that no new model is fitted:

- a rolling conformal offset per substation, lead band, and quantile;
- the Kalman filter's own predictive variance, scaled by a table of standardised residuals;
- the same variance widened by the spread across the ensemble members;
- a model-error spread placed around each member, estimated from forecasts driven by the analysis
  proxy so that the weather-forecast error is not counted twice;
- separate lower and upper widening factors per lead band.

**A cell counts as calibrated when its coverage lies within a binomial band set by the number of
distinct substation-days in the cell.** A substation-day is the unit because the errors of one
substation's half-hours within a day move together. Before any method is scored, a blocking
known-answer test checks that the band does not flag a forecast that is calibrated by construction,
and that the band and the methods detect and remove a miscalibration of the size that matters. The
decision rule asks for a method that passes in the replicate months and in the scored year
without raising the quantile score by more than 0.02 points of effective capacity.

**The coherence pass tests whether the ensemble sums across substations are calibrated against the
observed sum.** Calibrated marginals for each substation do not make the sum of several substations
calibrated, because the sum depends on how the errors of different substations move together. The
test turns each substation's calibrated quantiles back into ensemble members with ensemble copula
coupling (ECC), or with the Schaake shuffle, which orders the members by a template taken from
history. It then sums the members across substations and scores the summed ensemble against the
observed sum, and scores the joint distribution with the variogram score. The staged probabilistic
forecast already specifies [ECC and the coherence
scores](../../../roadmap/staged-probabilistic-forecast.md) for the production forecast. The pass asks
whether the convex model's spread, once calibrated, keeps the cross-substation dependence that
those scores reward. It runs after the per-substation calibration, on the same rows.

## W1: a diurnal-shape state

**Experiment 115 is running.** It tests whether a second filter state holding the deviation of the
day's shape from the model's day-type profile lowers the day-ahead error of the convex model with the
Kalman filter. The idea comes from the remaining day-ahead error being mostly in the shape of the
day, and not in its level. The result goes in the experiment's own write-up.

## N8: ensemble members through the DP

**Running each of the 50 perturbed ensemble members through the DP lets weather uncertainty enter
physically, through heating and solar output.** The comparison arms are member propagation,
residual quantiles, and the two combined, plus the Kalman filter run on ensemble-driven residuals by
lead. The fit is unchanged, so the work is feature preparation, which is CPU and disk bound, plus
about one GPU hour. CAL's model-error arm is a cheaper version of the same question, so N8 waits for
CAL's result if the maintainer restarts it.

## N7: paired residuals at switching events

**Where both sides of a hand-filtered switching event are metered, the donor's and the receiver's
residuals should be anti-correlated and sum to roughly the moved power.** The experiment reports the
share of events where the two residuals sum to within 20% of the moved power. A high share would show
that a later model can treat a transfer as a conserved flow between two nodes. Only a minority of the
events have both sides metered, so the result describes those events and does not generalise to the
rest. The run takes minutes on the CPU.

## Tail quantile regression on filtered residuals

**The proposal fits a quantile regression on the Kalman filter's residuals, with the model's
prediction, its square, and the hour of day as covariates.** [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) did this for net load in the 14 GB grid supply
point groups, and the regression gave a small gain over Gaussian quantiles on the same residuals.
They also adapted the regression online by gradient descent on the pinball loss. In their results
nearly all of the improvement came from filtering the mean, and the quantile step added 2 to 6%.
Their data are regional and national load, so the gain at Primary substations is untested. The
method replaces the residual-quantile spread of the convex model, and runs in CPU seconds.

## Conditional Pareto tails

**The proposal models the quantiles beyond 1% and 99% with a generalised Pareto distribution whose
scale depends on covariates.** [Browell and Fasiolo
(2021)](https://arxiv.org/abs/2103.10335) found that plain quantile regression is not calibrated
beyond those probabilities. A conditional Pareto tail cut the upward reserve requirement by up to
24.6% and the downward reserve by up to 10.8%, against a static Pareto tail. Their fit uses the R
package `mgcv`, so a Python version needs a small custom maximum-likelihood fit, which takes CPU
minutes per substation. The proposal serves the curtailment goal, which depends on the extreme upper
quantiles.

## Learned observation variance

**The proposal lets the Kalman filter track its own observation variance online instead of using a
fixed value.** [de Vilmarest and Wintenberger (2021)](https://arxiv.org/abs/2104.10777) call the
method VIKING. In [de Vilmarest and Goude (2022)](https://arxiv.org/abs/2110.00334), it matched a
hand-set large process noise on a linear model (mean absolute error 11.5 MW against 11.2 MW) and
beat a dynamic model with fixed noise (14.9 MW), so the main benefit is less tuning. It would give
the filter a variance that adapts to noisy periods, which also feeds the filter-variance arm of CAL.
The run takes CPU seconds.

## Planned experiments

<!-- planned-table -->
