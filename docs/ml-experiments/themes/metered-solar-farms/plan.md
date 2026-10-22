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

**[Experiment 218](../../00218-metered-solar-farm-plant-fit/result.md) found that the plant's shape
is not what limits the fit: giving each farm its own plant parameters raised the fraction of output
explained by only 0.003 over one shape shared by all farms.** The next question is whether a
solar-irradiance product other than the Copernicus Atmosphere Monitoring Service (CAMS), or a blend
of products, raises the fraction of the six farms' output explained by the plant model. The
experiment fails if the best blend raises that fraction by less than 0.02 over CAMS alone.

### XGBoost with the plant's mean and the members' spread

**[Experiment 224](../../00224-metered-farm-dayahead-two-number/result.md) found that an XGBoost
model scores lower pinball loss at the 0.01 to 0.95 quantiles and the plant at the 0.99 quantile.**
The proposal is to give an XGBoost model two extra input columns, the plant's mean response and the
standard deviation of the 51 plant responses, and to refit it weekly. The experiment fails if the
quantile score improves by less than 2% over plain XGBoost, or if the 0.99 pinball loss stays above
the plant's.
