# Calibration and uncertainty

**The summary of this theme is in progress.** Whether the forecast's quantiles and ensemble members are well calibrated, and how to widen or sharpen them. The [Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home message, and the [Research plan](plan.md) lists what is planned next.

## What would change in production

**These experiments are flagged `candidate`, and none is adopted yet.** The flags are proposals for the maintainer to confirm.

- **[00002 - Stage 2 calibrator for solar and wind](../../00002-stage2-calibrator-and-mixture-vs-ens-quantile/result.md):** Pool the ERA5-trained quantiles over the ENS members. Stage 2 itself did not help.
- **[00016 - Mixture calibration, solar and wind](../../00016-pooled-mixture-calibration-solar-wind/result.md):** Scale the solar mixture's quantiles about the median by a factor near 0.95, refitted monthly. Fixes 94% to 95% coverage at no change in CRPS; wind needs a spread factor on two lead groups, a post hoc arm.
- **[00020 - Solar and wind stack](../../00020-solar-wind-stack-cams-stage0-spread-margin/result.md):** Add a power-curve base margin to the wind stage 1a quantile model. Lowers the wind CRPS by 0.055 points (0.007 to 0.110); the stack of margin and spread factor gives 0.071; nothing detectably helped solar.
- **[00035 - Quantile blend of stored models](../../00035-lead-weighted-quantile-blend-of-stored-models/result.md):** Serve a fixed-weight average of the quantiles of the monthly-retrained N1, `a7_cs`, and the stack for Primary demand. Lowers the quantile score by 0.10 points (-0.19 to -0.03) against the monthly-retrained N1; needs three models, one of them refitted monthly; solar and wind gains are unresolved.
