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

- **Mixture.** Each member's 9 quantiles are treated as atoms with the probability masses
  0.05, 0.10, 0.15, 0.125, 0.10, 0.125, 0.15, 0.10, 0.05 (half the gap to each neighbouring level),
  each member weighted 1/51. The mixture quantile function interpolates the weighted atoms linearly
  at their mass mid-points on a 199-level grid. **Tail rule:** a mixture quantile below the lowest
  atom or above the highest atom takes that atom's value, so tails never extrapolate, and a level
  between two atoms interpolates linearly. Output quantiles are read from the 199-level grid.
- **CRPS estimate.** Twice the mass-weighted mean pinball loss over the 9 levels, with the masses
  above as weights. It approximates the continuous ranked probability score from 9 levels, and the
  two tail masses (0 to 0.05 and 0.95 to 1) are carried by the 0.05 and 0.95 levels.
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

Not yet run.
