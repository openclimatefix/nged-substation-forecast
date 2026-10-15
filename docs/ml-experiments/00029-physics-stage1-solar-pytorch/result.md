---
id: "00029"
slug: physics-stage1-solar-pytorch
status: planned
size: complex
hypothesis: >-
  For the 6 metered solar farms, replacing the XGBoost stage 1 with a differentiable PyTorch physics
  model (priors on tilt, azimuth, DC capacity, and the AC-to-DC ratio, fitted to metered power with
  CAMS irradiance and ERA5 temperature), applied to each ENS member and followed by a per-farm
  quantile XGBoost stage 2, gives a lower continuous ranked probability score (CRPS) than the pooled
  XGBoost mixture of experiment 00020.
prediction: >-
  The physics stage 1 alone is worse than the XGBoost stage 1, by 0.3 to 1.0 points of capacity
  (the beam-diffuse study found a fitted physical model 1.05 MAE points behind XGBoost on CAMS,
  and a calibration layer closed most of that gap). The hybrid (physics output plus the nine weather
  columns) is level with the XGBoost stage 1, within 0.05 points. The PyTorch model recovers
  synthetic tilt to within 5 degrees, azimuth to within 10 degrees, and DC capacity to within 5%.
decision_rule: >-
  Four planned contrasts on mean CRPS over all leads (arm minus comparison, points of capacity),
  read at 98.75% (a 5% level with a Bonferroni correction across four, two-sided), resampling whole
  months, with the three XGBoost seeds averaged. C1 `torch` minus `base`, C2 `torch_hybrid` minus
  `base`, C3 `torch` minus `scipy`, C4 `scipy` minus `base`. The threshold for a gain is 0.05
  points, because the change adds a fitted physics model and its refit to the pipeline. Supported
  when C1 or C2 has an interval wholly below zero and an estimate of at most -0.05. Refuted when
  the lower bounds of C1 and C2 are both above -0.05. Inconclusive otherwise. C3 and C4 are
  descriptive. The synthetic test passes when every parameter's 90% posterior interval covers the
  truth in at least 80% of the synthetic plants.
benchmark: >-
  `base`: the pooled ERA5-trained XGBoost mixture of experiment 00020 (read from its stored
  predictions, 5.03 to 5.05 points for solar), and a month-by-half-hour climatology row.
row_set_id: fold_a_tuesdays_v1
metric: >-
  CRPS estimated from 9 quantile levels, points of capacity (power over effective capacity, times
  100), solar farms only, by lead band
verdict: null
best_arm: null
headline: null
lessons: null
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00029-physics-stage1-solar-pytorch/
pull_request: null
---

# Does a differentiable physics stage 1 beat the XGBoost stage 1 for solar farms?

**Status: planned.** This page is proposal P31 in the [research plan](../research-plan.md). The
write-up replaces this text when the experiment finishes. CRPS is the continuous ranked probability
score, ENS the ECMWF ensemble, ERA5 the ECMWF reanalysis, and CAMS the Copernicus Atmosphere
Monitoring Service's satellite irradiance estimate. Farms appear only under anonymous labels A to F.

## Survey of the earlier PV fits (step 1)

*Filled in at write-up.*

## Arms

| Arm | Stage 1 | Stage 2 |
|---|---|---|
| `base` | 00020's XGBoost quantile stage 1 on nine ERA5 weather columns | Pooled mixture over 51 members |
| `scipy` | Fitted PV model from `studies.pv_fit` (free orientation, SciPy), run on each member's shortwave | XGBoost quantile model on the physics power only, then pooled |
| `torch` | New PyTorch physics model with priors | The same layer as `scipy` |
| `torch_hybrid` | The `torch` output | XGBoost quantile model on the physics power plus the nine weather columns |
| `climatology` | None | Month-by-half-hour climatology |

**Fitting.** Physics parameters are fitted once per farm to metered power at rows before
2025-06-17, using CAMS shortwave and ERA5 temperature (CAMS exists for only part of the history,
so rows without it are left out). The stage 2 layer trains on physics power driven by ERA5
shortwave (`base`'s weather source), and is applied to ENS members' shortwave. CAMS arrives about 2
days late and ERA5 about 5 days, but neither is read at forecast time, because no arm uses a lag.

**PyTorch model.** Erbs split of global into beam and diffuse, isotropic-sky transposition onto a
plane of fitted tilt and azimuth, a temperature derate, a DC capacity (a multiple of effective
capacity), and a soft clip at DC capacity over an AC-to-DC ratio. Parameters have normal priors
and a mean-field normal posterior fitted by stochastic variational inference (Adam, GPU).

## Synthetic test (step 2)

Six plants at the farms' coordinates with known parameters (tilt 10 to 45 degrees, azimuth 150 to
210 degrees, DC capacity 0.9 to 1.3, AC-to-DC ratio 0.7 to 1.0) are driven by the real training
irradiance and temperature, with 3% multiplicative and 2% additive noise. The fit is repeated for
20 noise draws. The write-up gives each parameter's error with a 90% interval and the fit's MAE.

## Checks before the contrasts are read

- **Leakage.** Every fit, including the physics parameters, stops before 2025-06-17. A reviewer
  reads the code for this.
- **No calendar columns** in any stage 2 layer; the hybrid's nine columns are `base`'s.
- **Reproduction.** `base` is read from 00020's stored predictions, and its solar CRPS is checked
  against 5.05 within 0.05.

## Size

**Complex, because the experiment adds a new method and admits more than one defensible design.**

- **What gets stored:** private predictions only.
- **Production serving path:** not touched.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes (the transposition model, the prior widths, whether the
  stage 2 layer reads the weather).
- **Code whose callers cannot be named without searching:** no; the code copies experiment 00020's
  patterns and calls `studies.pv_physics`, `studies.pv_fit`, and `studies.experiment_store`.

## Which earlier advice this plan follows

- **Followed:** `base`'s absolute score and a climatology row sit beside every contrast (00012,
  00020); no calendar columns (skill rule); a contrast is read against a threshold set by the cost
  of the change (00020).
- **Departs from:** 00012's advice to test CAMS-trained solar on more farms; this experiment keeps
  the 6 farms the brief names, so it cannot resolve gains under about 0.1 points.

## Resolution limit

Six farms in one region and one validation year resolve gains of roughly 0.1 points of capacity.
