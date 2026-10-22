# Metered solar farms: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| – | No experiment is queued; the proposals below are not yet agreed | – | The maintainer | Open |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### Better irradiance for the metered farms

**Experiment 218 found that the plant model is not the limit: per-farm parameters fit 0.003 better
than the fleet shape.** The next question is whether an irradiance product other than CAMS, or a
blend, lifts the fraction of output fitted on the same plants. The prediction that could fail is a
gain of at least 0.02 for the best blend over CAMS alone.

### XGBoost with the plant's mean and the members' spread

**Experiment 224 found that XGBoost wins the middle of the day-ahead distribution and the plant wins
the 0.99 quantile.** The proposal is to give XGBoost the plant's mean response and the members'
spread as input columns, refit weekly. The prediction that could fail is a quantile-score gain of at
least 2% over plain XGBoost, with the 0.99 pinball loss at or below the plant's.
