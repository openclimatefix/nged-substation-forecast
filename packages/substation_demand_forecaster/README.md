# substation_demand_forecaster

**This package computes the step-5 neural network's demand forecast for the 16 Primaries, from an
exported bundle and production's own `nwp` and `cleaned_power_time_series` tables.** The neural
network is trained in the research repository,
[nged-ml-research](https://github.com/openclimatefix/nged-ml-research), which exports one bundle
per monthly refit. A bundle is a directory holding the weights of each seed, the standardisers, the
column order, every constant of the recipe, and golden fixtures. The bundle's own `README.md`
describes its layout. The package imports nothing from the research repository: the forward pass is
a line-by-line port of the functional reference that each bundle ships as `reference_check.py`.

**The modules run in the order of the forecast:**

- `bundle`: reads a bundle into a frozen `Bundle`, and raises `ValueError` on a format version,
  a recipe, or a constant that the code does not implement.
- `ens_summary`: the ENS member mean and population standard deviation of each run, per Primary
  and half-hour.
- `features`: the 60 input columns of one forecast, and the expected-weather rows that the lag
  branch of the neural network reads.
- `neural_network`: the forward pass of every seed, and the quantiles read out of each row's
  Gaussian mixture.

**The golden tests compare every stage with a bundle's fixtures, and run only when
`PRIMARY_SUBSTATION_BUNDLE_PATH` names a bundle directory.** The golden tests read production's
Delta tables, so they need `DATA_PATH_INTERNAL` set as well:

```bash
PRIMARY_SUBSTATION_BUNDLE_PATH=/path/to/bundle \
    uv run pytest tests/test_substation_demand_forecaster_golden.py
```
