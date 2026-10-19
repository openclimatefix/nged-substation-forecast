# Calibration and uncertainty

**The summary of this theme is in progress.** This theme covers whether the forecast's quantiles and
ensemble members are well calibrated, and how to widen or sharpen them. The
[Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home
message, and the [Research plan](plan.md) lists what is planned next.

## What would change in production

- **[00002 - Stage 2 calibrator for solar and
  wind](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md):** Pool the
  ERA5-trained quantiles over the members of the ECMWF ensemble (ENS). The stage 2 calibrator itself
  did not help.
- **[00016 - Mixture calibration, solar and
  wind](../../00016-pooled-mixture-calibration-solar-wind/result.md):** Scale the solar mixture's
  quantiles about the median by one fitted spread factor. The solar 5% to 95% band covered 94.3% to
  95.3% of cases before the scaling and 89.5% to 91.8% after it, and the continuous ranked
  probability score (CRPS) did not change detectably. For wind, a spread factor on two lead groups,
  chosen after seeing the results, brought every lead band to 90% coverage.
- **[00020 - Solar and wind
  stack](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md):** Add a power-curve base
  margin to the wind stage 1a quantile model. The margin lowers the wind CRPS by 0.055 points
  (99.58% interval for the change -0.110 to -0.007), and the margin with a spread factor lowers it
  by 0.071. Nothing detectably helped solar.
- **[00035 - Quantile blend of stored
  models](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md):** Serve a
  fixed-weight average of the quantiles of the monthly-retrained N1, `a7_cs`, and the stack for
  Primary demand. N1 is the single-stage XGBoost quantile model trained on ENS weather. The blend
  lowers the quantile score by 0.10 points of capacity (98.33% interval -0.19 to -0.03) against the
  monthly-retrained N1. The blend needs three models, one of them refitted monthly, and the solar
  and wind gains are unresolved.
