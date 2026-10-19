# Hierarchy and pooling

**The summary of this theme is in progress.** This theme covers forecasts that share information
across substations or across levels of the electricity network hierarchy. The
[Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home
message, and the [Research plan](plan.md) lists what is planned next.

## What would change in production

**This experiment is flagged `candidate`.**

- **[00024 - Pooled demand models, part
  1](../../00024-pooling-groupings-demand-xgboost/result.md):** Train one XGBoost model per
  substation type (or per size band), with the series identifier as a categorical column, instead of
  one model per series. After the 14-day error correction, the quantile score of Primary demand
  rises by 0.15 points of capacity (99.44% interval +0.03 to +0.26), and the number of XGBoost
  models falls from 20 to 4. The pooled models need the series identifier, and the test covered 20
  series.
