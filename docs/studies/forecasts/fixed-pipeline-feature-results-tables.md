# Supporting tables: lean inputs and individual additions

These aggregate tables support the [fixed-pipeline study](fixed-pipeline-lean-features.md).
The main lean and first-solar comparisons are complete; alternate settings and the remaining
feature/control tables are still being added. Negative paired differences favour the recipe
named before `minus`. Losses and differences use percentage points of frozen effective capacity.

The full-input control is the historical cleaned point-model recipe, rather than the public
production configuration. All comparisons below have 24 training columns, including deterministic
duplicates of retained inputs where needed. The untargeted type rows deliberately retain the
full control's predictions; the private unaffected-type audit checked their exact equality.
Absolute-score intervals differ from paired-difference intervals. The primary bootstrap draws
whole months and one paired fitting seed; the separate equal-month t-interval averages seeds
first and weights months equally. Neither method estimates transfer to new generators.

The tables below are the unaltered output of the committed publication-table exporter, selecting
saved aggregate results and rounding their original values directly to three decimals. The
selection specification and source hashes are retained with the private publication artifact;
their public configuration inventory is still being finalised.

## lean-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| lean_pv | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| lean_wind | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |

## lean-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.206 | 0.044 | 0.393 | 0.107 | -0.170 | 0.384 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -3.330 | -3.989 | -2.739 | -3.309 | -4.026 | -2.592 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## lean-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| lean_pv | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| lean_wind | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |

## lean-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.204 | 0.070 | 0.337 | 0.200 | 0.044 | 0.355 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -3.744 | -4.849 | -2.678 | -3.725 | -4.923 | -2.527 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| solar_add_wind10 | PV | 8.070 | 5.940 | 10.402 | 3 | 12 | 6 | 175795 | False |

## solar-01-solar_add_wind10-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | -0.025 | -0.063 | 0.022 | -0.021 | -0.037 | -0.005 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| solar_add_wind10 | PV | 8.821 | 7.120 | 10.385 | 3 | 15 | 6 | 253003 | False |

## solar-01-solar_add_wind10-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | 0.005 | -0.038 | 0.051 | 0.006 | -0.032 | 0.043 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
