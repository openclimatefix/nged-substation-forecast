# Hierarchy and pooling

**The summary of this theme is in progress.** This theme covers forecasts that share information
across substations or across levels of the electricity network hierarchy. The
[Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home
message, and the [Research plan](plan.md) lists what is planned next.

## What would change in production

**These experiments are flagged `candidate`, and none is adopted yet.** The flags are proposals for
the maintainer to confirm.

- **[00024 - Pooled demand models, part
  1](../../00024-pooling-groupings-demand-xgboost/result.md):** Train one XGBoost model per
  substation type (or per size band), with the series identifier as a categorical column, instead of
  one per series. Costs 0.15 points on Primary demand after the error correction and saves 16 of 20
  models; needs the series identifier, and the test covered 20 series.
