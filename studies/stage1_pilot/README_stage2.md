# Stage 2 pilot (study 2b): does a shared calibrator plus a learned blend beat the pooled stage 1a mixture and the ENS-trained quantile model?

This is study 2b of the staged probabilistic forecast plan ([PR #1183 (Plan: document a staged
probabilistic forecast, and schedule the studies that test it)](https://github.com/openclimatefix/nged-substation-forecast/pull/1183)).
It builds on study 2a ([PR #1184 (Stage 1 pilot: ERA5-trained weather-to-power model against the
ENS-trained control)](https://github.com/openclimatefix/nged-substation-forecast/pull/1184)) and uses
its inputs, feature lists, folds, series, and seeds: the 9 generator series (6 solar farms, 3 wind
farms), fold A of the quick-wins study, seeds 0, 1, 2. The scripts sit beside study 2a's
(`stage2_*.py`) because they import its helpers. Outputs are under
`data/studies/per_study/stage2_pilot/`. It is exploratory and autonomous, changes no production
code, and the demand series are out of scope.

This README was written before the first stage 2 fit. Everything not listed under "Planned
contrasts" is exploratory.

## Question

**Does stage 2 (one shared calibrator, then a learned blend) fed by the stage 1a quantiles give a
better probabilistic forecast than the pooled stage 1a mixture (no second model) and than a
single-stage XGBoost quantile model trained on ENS weather?**

## Arms

Every arm outputs 9 quantile levels of power over effective capacity per case (levels 0.05, 0.1,
0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95; the median is the point forecast). All stage 1a, ENS-trained,
and calibrator models use the 2a feature lists (no calendar column, no power lag).

| Arm | What it is |
| --- | --- |
| `a_ens_quantile` | Arm (a). The `ens_trained` recipe as a quantile model: stage 1a's columns, trained on the ENS control member 2024-04 to 2025-06, 9 quantile levels, applied to each of the 51 members and pooled as a mixture. |
| `b_mixture_full` | Arm (b). Stage 1a refitted with the 9 quantile levels on ERA5 from 2019-09 to 2025-06-16, applied to each member's weather, member quantiles pooled as a mixture. No second model. |
| `b_mixture_short` | As `b_mixture_full` with stage 1a trained on ERA5 from 2024-04 only (the ENS months). |
| `c_stage2_full` | Arm (c). Stage 2 given `b_mixture_full`'s stage 1a quantiles: a shared per-technology calibrator on each member, a pooled mixture, then a learned blend. |
| `c_stage2_short` | As `c_stage2_full` with the short-trained stage 1a. |
| `*_blp` | The same arm after the learned blend (see below). Every arm is reported with and without it, so the blend's effect is its own contrast. |
| `climatology_q` | Reference. Empirical quantiles of power by local month and half-hour of day, 2019-09 to 2025-06-16. |

## Stage 2

- **Calibrator.** One XGBoost model pair per technology, pooled over that technology's series.
  Inputs: the member's 9 stage 1a quantiles and the lead (valid time minus forecast initialisation
  time). A location model (quantile loss at level 0.5) and a scale model (the
  absolute residual, scale at least 0.01 of capacity). The calibrated quantile at level `t` is the
  location plus the scale times a fixed shape `z_t`, clipped to 0 to 1 and sorted.
- **Shape.** `z_t` is the empirical level-`t` quantile of the standardised out-of-fold residuals
  `(y - location) / scale` for the technology and lead band.
- **Out-of-fold.** The stage 1a inputs on the ENS training months are forward-chained out-of-fold
  as in 2a (one model per month, trained on valid times up to 14 days before the month starts).
  The training months are 2024-07 to 2025-06, because the short-trained stage 1a has too little
  history before 2024-07. The calibrator's own out-of-fold values (for the shape) are
  forward-chained over 2024-10 to 2025-06 with the same 14-day purge. The final calibrator is fitted
  on all 12 training months.
- **One ENS product.** With ENS the only product, the shared calibrator (Design 2) and
  per-product columns (Design 1) are the same model, and the blend has one component. Design 1 as a
  control is untestable here and is not run. The study 3 plan tests it with other products.
- **Members in training.** The training frames hold only the control member's weather, so the
  calibrator is fitted on single-member rows and applied to each of the 51 members, and the
  across-member spread and percentiles of the plan's calibrator inputs are not used. The ENS control
  member is also the unperturbed member, so its error is somewhat smaller than a perturbed member's.
- **Blend.** The mixture of the 51 members' distributions has the cumulative distribution `F`. The
  learned blend is the beta transform `B(F)` ([Gneiting and Ranjan (2013)](https://doi.org/10.1214/13-EJS823)),
  with the two shape parameters per technology and lead band chosen from a grid
  (0.7, 0.85, 1.0, 1.2, 1.5 each) to minimise the CRPS estimate below. The 51-member rows exist only
  in the validation year, so the parameters for each validation month are fitted on earlier
  validation months, forward-chained with a 14-day purge. A month with less than one full month of
  earlier data uses both parameters equal to 1 (no transform), which is the plain linear pool.

## Mixture and scores

- **Mixture.** Each member's 9 quantiles are treated as atoms with the probability masses 0.075,
  0.10, 0.15, 0.125, 0.10, 0.125, 0.15, 0.10, 0.075 (the interval between the mid-points to the
  neighbouring levels, so the masses sum to 1), each member weighted 1/51. The mixture quantile
  function interpolates the weighted atoms linearly at their mass mid-points on a 199-level grid.
  **Tail rule:** a mixture quantile below the lowest atom or above the highest atom takes that
  atom's value, so tails never extrapolate, and a level between two atoms interpolates linearly.
  Output quantiles are read from the 199-level grid.
- **CRPS estimate.** Twice the mass-weighted mean pinball loss over the 9 levels, with the masses
  above as weights. It approximates the continuous ranked probability score from 9 levels. The two
  tails (below 0.075 and above 0.925) are carried by the 0.05 and 0.95 levels.
- **MAE of the median.** Absolute error of the 0.5 quantile, points of capacity.
- **Coverage.** The share of cases with the observation inside the closed 10 to 90 and 5 to 95
  quantile intervals (the nominal values are 80% and 90%). Where the interval has zero width
  (solar at night, both ends 0) an observation of 0 counts as covered.
- All errors are normalised by effective capacity per row before any mean, and each level is the
  mean over series of each series' mean, as in 2a. Intervals resample whole validation months and one
  fitting seed (2,000 resamples, stream seed `20261013`), the same draw for every arm.

## Planned contrasts

Each is a CRPS difference in points of capacity, per lead band and technology (lead bands as in 2a).
A planned contrast is material if the mean paired difference is at most -0.15 points, the 95%
month-resampled interval is wholly below zero, and at least 9 of 12 months are negative.

- **Q1 = `c_stage2_full_blp` minus `b_mixture_full`.** Does stage 2 beat the pooled mixture?
- **Q2 = `c_stage2_full_blp` minus `a_ens_quantile`.** Does stage 2 beat the single-stage ENS
  quantile model?
- **Q3 = `b_mixture_full` minus `a_ens_quantile`.** Does the pooled stage 1a mixture alone beat the
  single-stage ENS quantile model?

Every other contrast is exploratory: the learned blend against none (`*_blp` minus the same arm
without it), the short-trained against the full-history stage 1a for the mixture and for stage 2
(does the long ERA5 history matter for generators, with and without stage 2), the MAE of the median
for each planned pair, and coverage. A planned contrast is read at the CRPS and also reported for the
MAE of the median.

## Untested alternatives

- **Quantile-mapping ENS onto ERA5** before stage 1a (the plan's "stage 0") is not run.
- **Design 1 against Design 2**, the across-member spread inputs, lag inputs, and ECC coupling are
  not run (single product, single-member training rows, no lags).
- **Per-series calibrators** are not run (the calibrator pools the series of one technology).

## Reproduce

```bash
uv run python studies/stage1_pilot/stage2_fit_stage1a_quantile.py
uv run python studies/stage1_pilot/stage2_fit_arm_a.py
uv run python studies/stage1_pilot/stage2_fit_calibrator.py
uv run python studies/stage1_pilot/stage2_score.py
uv run python studies/stage1_pilot/stage2_report.py
```

Every script skips outputs that exist.

## Results

All numbers are printed by `stage2_report.py` into
`data/studies/per_study/stage2_pilot/results/report.md`. Scores are in points of capacity (smaller
is better) on 175,795 solar and 107,121 wind half-hours per arm and seed. Intervals resample whole
months, as in study 2a. Only Q1 to Q3 are planned contrasts; every other number is exploratory.

**Verdict: the pooled stage 1a mixture is the best arm for wind and for the first six days of solar.
Stage 2 did not beat it, and the learned blend added nothing reliable.**

- **Q1 (stage 2 with blend minus the pooled mixture) is not material for either technology.**
  Solar CRPS is worse by 0.40 points (interval +0.26 to +0.61; 0 of 12 months better). Wind CRPS is
  worse by 0.26 points (interval -0.17 to +0.64).
- **Q2 (stage 2 with blend minus the ENS-trained quantile arm) is material for wind only.** Wind
  CRPS is better by 0.36 points (interval -0.59 to -0.12; 10 of 12 months). Solar is not
  better overall (+0.17; interval -0.03 to +0.50). Solar is better at 0 to 24 h (-0.43) and 1 to 2
  days (-0.32) and worse at 7 to 15 days.
- **Q3 (the pooled mixture minus the ENS-trained quantile arm) is material for both.** Solar CRPS is
  better by 0.24 points (interval -0.36 to -0.06; 11 of 12 months) and wind by 0.62 points
  (interval -0.99 to -0.26; 10 of 12 months). The MAE of the median agrees: -0.40 for solar and
  -0.99 for wind, both material.

### Absolute scores

CRPS over all leads, in points of capacity:

| Arm | Solar | Wind |
| --- | --- | --- |
| `climatology_q` | 5.65 | 13.03 |
| `a_ens_quantile` | 5.29 | 11.73 |
| `b_mixture_full` | 5.05 | 11.11 |
| `c_stage2_full_blp` | 5.45 | 11.37 |

- **Beyond day 6 no arm beats climatology by much.** At 7 to 10 days the wind arms score 13.0 to
  13.5 against climatology's 12.9, and the solar arms score 5.5 to 6.2 against 5.84. Stage 2 is the
  worst solar arm there (6.13 to 6.23).
- **Gains are concentrated in the first six days.** For wind, Q3 is -1.77 points at 0 to 24 h,
  -2.32 at 1 to 2 days, and -0.96 at 3 to 6 days. At 7 to 10 days and 11 to 15 days it is positive
  but its interval includes zero.

### Does the long ERA5 history matter for generators?

- **Mostly not at the CRPS.** Exploratory contrasts X5 to X7 compare stage 1a trained on ERA5 from
  2019-09 against ERA5 from 2024-04 only. The differences are 0.01 to 0.09 points for solar and 0.01
  to 0.03 for wind, and the largest interval (X7, solar, +0.02 to +0.16) hardly clears zero. In
  study 2a the same history made a clear difference to the point forecast of the member-averaged
  model; here the pooled mixture and the calibrator close most of that gap.
- **The short history gives narrower intervals.** `b_mixture_short` covers 85.0% (solar) and 77.9%
  (wind) of cases with its 10 to 90 interval, against 88.6% and 81.9% for the full history
  (nominal 80%). The full history is slightly better at the 0.9 and 0.95 pinball levels (0.04 to
  0.06 points, intervals above zero for wind).

### The tails

- **The ERA5-trained gain for wind is not larger in the tail than elsewhere in the distribution.**
  The CRPS is twice the mean pinball loss over the levels, so half the CRPS gain is the typical
  per-level gain. Over all leads, Q3 gives -0.31 of that typical gain against -0.24 (interval -0.53
  to +0.06) at level 0.9 and -0.18 (-0.35 to -0.02) at level 0.95. At 0 to 24 h the tail gains
  (-1.10 and -0.76) bracket the typical -0.88. Both are material at 0 to 6 days and absent beyond
  day 6.
- **The exceedance forecasts do improve for wind.** The Brier scores of exceeding the 90th and 95th
  training percentiles of wind power give Q3 of -0.50 (interval -0.85 to -0.21; 12 of 12 months)
  and -0.27 (-0.42 to -0.12; 10 of 12 months).
- **For solar the tails show no ERA5-trained gain.** Q3 is -0.02 at level 0.9 and 0.00 at level
  0.95, and its Brier scores have intervals that include zero (-0.22 and -0.04).
- **The ENS-trained arm under-forecasts wind exceedances.** In the reliability table for exceeding
  the series' 90th percentile, cases where it forecasts a probability of 0.20 to 0.30 exceed
  the threshold 48% of the time, and cases forecast at 0.30 to 0.50 exceed it 70% of the time. The
  pooled mixture's matching bins are 0.24 forecast against 0.24 observed, and 0.38 against 0.44. The
  same pattern holds at the 95th percentile.
- **Stage 2 is not more reliable in the tail.** At the 90th percentile of wind its bins above 0.30
  are under-forecast (0.58 forecast against 0.83 observed at 0.50 to 0.70). Q1 on the exceedance
  Brier scores is worse for both technologies, with every interval wholly above zero (solar +0.75
  and +0.38, wind +0.28 and +0.17).

### Coverage and the blend

- **The 10 to 90 interval over-covers for most arms.** It covers 83% to 89% (nominal 80%), apart from
  stage 2 without the blend for solar (83.9%) and the short mixtures. For wind the ENS-trained arm
  under-covers beyond day 6 (79.4% to 79.6%).
- **The blend did not help.** For stage 2, X1 is +0.00 (solar) and +0.08 (wind) points of CRPS,
  neither clearly different from zero. For the mixture and the ENS-trained arm, the blend helps wind
  by 0.13 to 0.19 points (intervals reach or cross zero) and hurts solar by 0.04 to 0.08 points
  (intervals above zero).
- **Calibrator without blend (X4) is worse than the mixture it takes as input** for solar (+0.40)
  and not distinguishable for wind (+0.18).

### Limits

- One fold and one forecast product. The ERA5-trained stage 1a never sees the ENS bias at training
  time, and the quantile-mapping ("stage 0") alternative is untested.
- Single-member training rows, no power lags, and no per-series calibrators, so stage 2 here is a
  small test of Design 2, not of the whole plan.
- Only three wind series and six solar series, in two ERA5 cells for solar. The intervals resample
  12 months, and 9 of 12 is the smallest agreement that counts.
- Pinball at level 0.99 cannot be scored with 9 levels, so "the tail" means the 0.9 and 0.95 levels
  and the 90th and 95th percentile exceedances.
