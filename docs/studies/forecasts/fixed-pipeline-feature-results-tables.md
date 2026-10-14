# Supporting tables: lean inputs and individual additions

These aggregate tables support the [fixed-pipeline study](fixed-pipeline-lean-features.md).
The original lean, first-solar, individual wind-addition and control comparisons are complete
for both training settings and folds. Negative paired differences favour the recipe
named before `minus`. Losses and differences use percentage points of frozen effective capacity.

The full-input control is the historical cleaned point-model recipe, rather than the public
production configuration. Equal-padding comparisons below have 24 training columns, including deterministic
duplicates of retained inputs where needed. The separately labelled unpadded comparisons retain
only their real input columns. The untargeted type rows deliberately retain the
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

## lean-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False |
| lean_pv | PV | 7.881 | 5.854 | 10.119 | 3 | 12 | 6 | 175795 | False |
| lean_pv | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False |
| lean_wind | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False |
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |

## lean-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.118 | -0.009 | 0.264 | 0.097 | -0.077 | 0.270 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -2.992 | -3.724 | -2.362 | -2.968 | -3.756 | -2.179 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

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

## lean-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False |
| lean_pv | PV | 8.752 | 7.082 | 10.307 | 3 | 15 | 6 | 253003 | False |
| lean_pv | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False |
| lean_wind | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False |
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |

## lean-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.138 | 0.023 | 0.242 | 0.133 | 0.012 | 0.254 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -3.632 | -4.756 | -2.606 | -3.606 | -4.793 | -2.419 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| solar_add_wind10 | PV | 8.070 | 5.940 | 10.402 | 3 | 12 | 6 | 175795 | False |

## solar-01-solar_add_wind10-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | -0.025 | -0.063 | 0.022 | -0.021 | -0.037 | -0.005 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 7.881 | 5.854 | 10.119 | 3 | 12 | 6 | 175795 | False |
| solar_add_wind10 | PV | 7.859 | 5.841 | 10.091 | 3 | 12 | 6 | 175795 | False |

## solar-01-solar_add_wind10-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | -0.022 | -0.047 | -0.004 | -0.016 | -0.033 | 0.001 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| solar_add_wind10 | PV | 8.821 | 7.120 | 10.385 | 3 | 15 | 6 | 253003 | False |

## solar-01-solar_add_wind10-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | 0.005 | -0.038 | 0.051 | 0.006 | -0.032 | 0.043 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## solar-01-solar_add_wind10-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_pv | PV | 8.752 | 7.082 | 10.307 | 3 | 15 | 6 | 253003 | False |
| solar_add_wind10 | PV | 8.692 | 7.048 | 10.223 | 3 | 15 | 6 | 253003 | False |

## solar-01-solar_add_wind10-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_add_wind10 minus lean_pv | PV | -0.060 | -0.090 | -0.033 | -0.060 | -0.090 | -0.031 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_categorical_precipitation_type_surface-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 14.141 | 12.772 | 15.351 | 3 | 12 | 3 | 107121 | False |

## wind_add_categorical_precipitation_type_surface-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.320 | 0.082 | 0.604 | 0.331 | 0.015 | 0.647 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_categorical_precipitation_type_surface-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 14.182 | 12.821 | 15.410 | 3 | 12 | 3 | 107121 | False |

## wind_add_categorical_precipitation_type_surface-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.331 | 0.111 | 0.601 | 0.342 | 0.043 | 0.640 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_categorical_precipitation_type_surface-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 13.101 | 12.024 | 14.115 | 3 | 15 | 3 | 135921 | False |

## wind_add_categorical_precipitation_type_surface-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.225 | -0.072 | 0.572 | 0.224 | -0.015 | 0.462 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_categorical_precipitation_type_surface-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 12.960 | 11.814 | 14.050 | 3 | 15 | 3 | 135921 | False |

## wind_add_categorical_precipitation_type_surface-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | -0.085 | -0.321 | 0.120 | -0.081 | -0.268 | 0.105 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_dew_point_temperature_2m | Wind | 13.799 | 12.236 | 15.205 | 3 | 12 | 3 | 107121 | False |

## wind_add_dew_point_temperature_2m-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | -0.022 | -0.089 | 0.048 | -0.024 | -0.098 | 0.049 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_dew_point_temperature_2m | Wind | 13.842 | 12.308 | 15.255 | 3 | 12 | 3 | 107121 | False |

## wind_add_dew_point_temperature_2m-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | -0.009 | -0.064 | 0.045 | -0.011 | -0.066 | 0.043 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_dew_point_temperature_2m | Wind | 12.883 | 11.732 | 13.964 | 3 | 15 | 3 | 135921 | False |

## wind_add_dew_point_temperature_2m-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | 0.007 | -0.067 | 0.086 | 0.006 | -0.047 | 0.060 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_dew_point_temperature_2m | Wind | 13.027 | 11.871 | 14.107 | 3 | 15 | 3 | 135921 | False |

## wind_add_dew_point_temperature_2m-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | -0.018 | -0.056 | 0.019 | -0.018 | -0.055 | 0.018 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 13.848 | 12.264 | 15.268 | 3 | 12 | 3 | 107121 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | 0.027 | -0.028 | 0.091 | 0.026 | -0.031 | 0.083 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 13.850 | 12.298 | 15.287 | 3 | 12 | 3 | 107121 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | -0.001 | -0.046 | 0.041 | -0.002 | -0.046 | 0.042 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 12.931 | 11.788 | 14.027 | 3 | 15 | 3 | 135921 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | 0.055 | -0.013 | 0.127 | 0.054 | -0.013 | 0.122 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 13.062 | 11.910 | 14.148 | 3 | 15 | 3 | 135921 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | 0.017 | -0.022 | 0.065 | 0.017 | -0.031 | 0.064 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week | Wind | 14.168 | 12.579 | 15.643 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.347 | 0.200 | 0.497 | 0.347 | 0.183 | 0.510 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week | Wind | 14.102 | 12.517 | 15.575 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.251 | 0.137 | 0.355 | 0.253 | 0.128 | 0.378 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week | Wind | 13.164 | 12.017 | 14.230 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.288 | 0.164 | 0.393 | 0.294 | 0.182 | 0.406 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week | Wind | 13.250 | 12.080 | 14.353 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.205 | 0.125 | 0.286 | 0.208 | 0.113 | 0.303 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_cos | Wind | 14.069 | 12.494 | 15.513 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_cos-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.248 | 0.114 | 0.386 | 0.249 | 0.095 | 0.403 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_cos | Wind | 14.027 | 12.451 | 15.481 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_cos-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.176 | 0.077 | 0.281 | 0.179 | 0.056 | 0.301 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_cos | Wind | 13.118 | 11.990 | 14.189 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_cos-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.242 | 0.131 | 0.347 | 0.244 | 0.144 | 0.343 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_cos | Wind | 13.234 | 12.073 | 14.327 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_cos-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.189 | 0.108 | 0.268 | 0.188 | 0.100 | 0.277 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_sin | Wind | 14.139 | 12.588 | 15.567 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_sin-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.318 | 0.179 | 0.461 | 0.318 | 0.167 | 0.470 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_sin | Wind | 14.062 | 12.512 | 15.488 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_sin-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.211 | 0.103 | 0.314 | 0.213 | 0.092 | 0.333 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_sin | Wind | 13.127 | 11.988 | 14.199 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_sin-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.251 | 0.160 | 0.331 | 0.256 | 0.165 | 0.348 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_sin | Wind | 13.206 | 12.058 | 14.275 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_sin-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.161 | 0.088 | 0.231 | 0.163 | 0.086 | 0.241 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_cos | Wind | 15.778 | 13.915 | 17.484 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_cos-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 1.957 | 1.566 | 2.357 | 1.955 | 1.493 | 2.416 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_cos | Wind | 15.379 | 13.538 | 17.074 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_cos-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 1.529 | 1.174 | 1.890 | 1.530 | 1.100 | 1.961 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_cos | Wind | 15.253 | 13.665 | 16.863 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_cos-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 2.377 | 1.566 | 3.211 | 2.378 | 1.449 | 3.308 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_cos | Wind | 15.088 | 13.506 | 16.652 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_cos-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 2.043 | 1.282 | 2.793 | 2.041 | 1.209 | 2.874 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_sin | Wind | 16.238 | 14.205 | 18.143 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_sin-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.417 | 1.563 | 3.244 | 2.379 | 1.382 | 3.375 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_sin | Wind | 15.907 | 13.955 | 17.749 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_sin-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.057 | 1.280 | 2.839 | 2.025 | 1.110 | 2.940 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_sin | Wind | 15.617 | 13.960 | 17.082 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_sin-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.741 | 1.960 | 3.491 | 2.752 | 1.901 | 3.604 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_sin | Wind | 15.424 | 13.806 | 16.856 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_sin-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.379 | 1.673 | 3.055 | 2.393 | 1.615 | 3.171 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## precip-era-after_precip-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.465 | 12.137 | 14.742 | 3 | 8 | 3 | 66457 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 13.497 | 12.156 | 14.796 | 3 | 8 | 3 | 66457 | False |

## precip-era-after_precip-fold_b; main; padding equal — descriptive availability-era paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.032 | -0.008 | 0.085 | 0.029 | -0.003 | 0.061 | 1 | 8 | 3 | False | 8 | 8 | 0 | 0 | 0 |

## precip-era-before_precip-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.313 | 10.799 | 13.899 | 3 | 8 | 3 | 69464 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 12.721 | 11.206 | 14.230 | 3 | 8 | 3 | 69464 | False |

## precip-era-before_precip-fold_b; main; padding equal — descriptive availability-era paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.409 | -0.161 | 0.953 | 0.371 | -0.116 | 0.858 | 2 | 8 | 3 | False | 8 | 8 | 0 | 0 | 0 |

## guarded-windchill-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| production_cleaned_guarded_windchill | PV | 7.897 | 5.810 | 10.147 | 3 | 12 | 6 | 175795 | False |
| production_cleaned_guarded_windchill | Wind | 17.133 | 15.530 | 18.669 | 3 | 12 | 3 | 107121 | False |
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| lean_pv | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| lean_pv_guarded_windchill | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| lean_pv_guarded_windchill | Wind | 17.133 | 15.530 | 18.669 | 3 | 12 | 3 | 107121 | False |
| lean_wind | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| lean_wind_guarded_windchill | PV | 7.897 | 5.810 | 10.147 | 3 | 12 | 6 | 175795 | False |
| lean_wind_guarded_windchill | Wind | 13.820 | 12.257 | 15.207 | 3 | 12 | 3 | 107121 | False |

## guarded-windchill-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_cleaned_guarded_windchill minus production_cleaned | PV | 0.008 | -0.029 | 0.058 | 0.008 | -0.013 | 0.030 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill minus production_cleaned | Wind | -0.017 | -0.106 | 0.066 | -0.017 | -0.068 | 0.035 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus production_cleaned_guarded_windchill | PV | 0.198 | 0.028 | 0.394 | 0.099 | -0.195 | 0.392 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus production_cleaned_guarded_windchill | Wind | -3.314 | -3.953 | -2.730 | -3.293 | -3.994 | -2.592 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus lean_pv | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus lean_wind | Wind | -0.001 | -0.036 | 0.036 | -0.001 | -0.018 | 0.016 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## guarded-windchill-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False |
| production_cleaned_guarded_windchill | PV | 7.768 | 5.778 | 9.933 | 3 | 12 | 6 | 175795 | False |
| production_cleaned_guarded_windchill | Wind | 16.840 | 15.087 | 18.525 | 3 | 12 | 3 | 107121 | False |
| lean_pv | PV | 7.881 | 5.854 | 10.119 | 3 | 12 | 6 | 175795 | False |
| lean_pv | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False |
| lean_pv_guarded_windchill | PV | 7.881 | 5.854 | 10.119 | 3 | 12 | 6 | 175795 | False |
| lean_pv_guarded_windchill | Wind | 16.840 | 15.087 | 18.525 | 3 | 12 | 3 | 107121 | False |
| lean_wind | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False |
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False |
| lean_wind_guarded_windchill | PV | 7.768 | 5.778 | 9.933 | 3 | 12 | 6 | 175795 | False |
| lean_wind_guarded_windchill | Wind | 13.860 | 12.321 | 15.288 | 3 | 12 | 3 | 107121 | False |

## guarded-windchill-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_cleaned_guarded_windchill minus production_cleaned | PV | 0.006 | -0.008 | 0.023 | 0.017 | -0.008 | 0.042 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill minus production_cleaned | Wind | -0.003 | -0.061 | 0.044 | -0.003 | -0.038 | 0.032 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus production_cleaned_guarded_windchill | PV | 0.113 | -0.009 | 0.257 | 0.079 | -0.092 | 0.250 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus production_cleaned_guarded_windchill | Wind | -2.980 | -3.703 | -2.348 | -2.956 | -3.735 | -2.177 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus lean_pv | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus lean_wind | Wind | 0.009 | -0.006 | 0.020 | 0.009 | 0.002 | 0.016 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## guarded-windchill-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| production_cleaned_guarded_windchill | PV | 8.627 | 6.994 | 10.151 | 3 | 15 | 6 | 253003 | False |
| production_cleaned_guarded_windchill | Wind | 16.701 | 15.013 | 18.349 | 3 | 15 | 3 | 135921 | False |
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| lean_pv | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| lean_pv_guarded_windchill | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| lean_pv_guarded_windchill | Wind | 16.701 | 15.013 | 18.349 | 3 | 15 | 3 | 135921 | False |
| lean_wind | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| lean_wind_guarded_windchill | PV | 8.627 | 6.994 | 10.151 | 3 | 15 | 6 | 253003 | False |
| lean_wind_guarded_windchill | Wind | 12.890 | 11.782 | 13.963 | 3 | 15 | 3 | 135921 | False |

## guarded-windchill-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_cleaned_guarded_windchill minus production_cleaned | PV | 0.014 | -0.017 | 0.042 | 0.014 | -0.002 | 0.030 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill minus production_cleaned | Wind | 0.081 | -0.111 | 0.314 | 0.082 | 0.022 | 0.142 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus production_cleaned_guarded_windchill | PV | 0.190 | 0.046 | 0.325 | 0.186 | 0.033 | 0.338 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus production_cleaned_guarded_windchill | Wind | -3.811 | -4.896 | -2.766 | -3.793 | -4.959 | -2.627 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus lean_pv | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus lean_wind | Wind | 0.014 | -0.013 | 0.041 | 0.014 | -0.005 | 0.032 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## guarded-windchill-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False |
| production_cleaned_guarded_windchill | PV | 8.615 | 6.991 | 10.134 | 3 | 15 | 6 | 253003 | False |
| production_cleaned_guarded_windchill | Wind | 16.594 | 14.900 | 18.304 | 3 | 15 | 3 | 135921 | False |
| lean_pv | PV | 8.752 | 7.082 | 10.307 | 3 | 15 | 6 | 253003 | False |
| lean_pv | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False |
| lean_pv_guarded_windchill | PV | 8.752 | 7.082 | 10.307 | 3 | 15 | 6 | 253003 | False |
| lean_pv_guarded_windchill | Wind | 16.594 | 14.900 | 18.304 | 3 | 15 | 3 | 135921 | False |
| lean_wind | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False |
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False |
| lean_wind_guarded_windchill | PV | 8.615 | 6.991 | 10.134 | 3 | 15 | 6 | 253003 | False |
| lean_wind_guarded_windchill | Wind | 13.028 | 11.905 | 14.092 | 3 | 15 | 3 | 135921 | False |

## guarded-windchill-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| production_cleaned_guarded_windchill minus production_cleaned | PV | 0.000 | -0.025 | 0.025 | -0.000 | -0.008 | 0.008 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill minus production_cleaned | Wind | -0.083 | -0.227 | 0.062 | -0.082 | -0.143 | -0.021 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus production_cleaned_guarded_windchill | PV | 0.138 | 0.025 | 0.241 | 0.133 | 0.011 | 0.255 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus production_cleaned_guarded_windchill | Wind | -3.566 | -4.626 | -2.561 | -3.541 | -4.705 | -2.377 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv_guarded_windchill minus lean_pv | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill minus lean_wind | Wind | -0.017 | -0.043 | 0.000 | -0.017 | -0.031 | -0.004 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## negative-controls-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| without_local_utc_offset_minutes | PV | 7.873 | 5.788 | 10.134 | 3 | 12 | 6 | 175795 | False |
| without_local_utc_offset_minutes | Wind | 17.171 | 15.549 | 18.774 | 3 | 12 | 3 | 107121 | False |
| without_local_day_of_week | PV | 7.907 | 5.808 | 10.175 | 3 | 12 | 6 | 175795 | False |
| without_local_day_of_week | Wind | 17.218 | 15.593 | 18.830 | 3 | 12 | 3 | 107121 | False |

## negative-controls-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| without_local_utc_offset_minutes minus production_cleaned | PV | -0.016 | -0.043 | 0.008 | -0.024 | -0.047 | -0.001 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_utc_offset_minutes minus production_cleaned | Wind | 0.021 | -0.072 | 0.127 | 0.020 | -0.031 | 0.070 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | PV | 0.017 | -0.011 | 0.061 | 0.031 | -0.006 | 0.068 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | Wind | 0.067 | -0.043 | 0.185 | 0.068 | -0.027 | 0.162 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## negative-controls-fold_a; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False |
| without_local_utc_offset_minutes | PV | 7.754 | 5.763 | 9.921 | 3 | 12 | 6 | 175795 | False |
| without_local_utc_offset_minutes | Wind | 16.852 | 15.078 | 18.552 | 3 | 12 | 3 | 107121 | False |
| without_local_day_of_week | PV | 7.754 | 5.763 | 9.923 | 3 | 12 | 6 | 175795 | False |
| without_local_day_of_week | Wind | 16.920 | 15.115 | 18.650 | 3 | 12 | 3 | 107121 | False |

## negative-controls-fold_a; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| without_local_utc_offset_minutes minus production_cleaned | PV | -0.008 | -0.037 | 0.017 | -0.001 | -0.025 | 0.023 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_utc_offset_minutes minus production_cleaned | Wind | 0.010 | -0.063 | 0.095 | 0.010 | -0.021 | 0.041 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | PV | -0.009 | -0.028 | 0.013 | 0.011 | -0.033 | 0.054 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | Wind | 0.078 | -0.004 | 0.171 | 0.077 | 0.002 | 0.152 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## negative-controls-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| without_local_utc_offset_minutes | PV | 8.624 | 6.981 | 10.144 | 3 | 15 | 6 | 253003 | False |
| without_local_utc_offset_minutes | Wind | 16.733 | 15.003 | 18.360 | 3 | 15 | 3 | 135921 | False |
| without_local_day_of_week | PV | 8.607 | 6.973 | 10.125 | 3 | 15 | 6 | 253003 | False |
| without_local_day_of_week | Wind | 16.742 | 14.991 | 18.450 | 3 | 15 | 3 | 135921 | False |

## negative-controls-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| without_local_utc_offset_minutes minus production_cleaned | PV | 0.012 | -0.079 | 0.094 | 0.011 | 0.001 | 0.022 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_utc_offset_minutes minus production_cleaned | Wind | 0.112 | -0.233 | 0.650 | 0.112 | 0.047 | 0.177 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | PV | -0.005 | -0.035 | 0.029 | -0.003 | -0.037 | 0.031 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | Wind | 0.121 | -0.017 | 0.262 | 0.124 | -0.014 | 0.262 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## negative-controls-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False |
| without_local_utc_offset_minutes | PV | 8.612 | 6.985 | 10.126 | 3 | 15 | 6 | 253003 | False |
| without_local_utc_offset_minutes | Wind | 16.667 | 14.997 | 18.371 | 3 | 15 | 3 | 135921 | False |
| without_local_day_of_week | PV | 8.606 | 6.982 | 10.121 | 3 | 15 | 6 | 253003 | False |
| without_local_day_of_week | Wind | 16.741 | 15.024 | 18.470 | 3 | 15 | 3 | 135921 | False |

## negative-controls-fold_b; sensitivity; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| without_local_utc_offset_minutes minus production_cleaned | PV | -0.003 | -0.023 | 0.026 | -0.003 | -0.015 | 0.008 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_utc_offset_minutes minus production_cleaned | Wind | -0.010 | -0.148 | 0.108 | -0.009 | -0.032 | 0.014 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | PV | -0.008 | -0.029 | 0.014 | -0.009 | -0.029 | 0.011 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| without_local_day_of_week minus production_cleaned | Wind | 0.064 | -0.101 | 0.234 | 0.072 | -0.074 | 0.218 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## unpadded-lean-fold_a; main; padding none — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False |
| lean_pv | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False |
| lean_wind | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False |
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |

## unpadded-lean-fold_a; main; padding none — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.206 | 0.044 | 0.393 | 0.107 | -0.170 | 0.384 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -3.330 | -3.989 | -2.739 | -3.309 | -4.026 | -2.592 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## unpadded-lean-fold_b; main; padding none — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False |
| lean_pv | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False |
| lean_wind | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False |
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |

## unpadded-lean-fold_b; main; padding none — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lean_pv minus production_cleaned | PV | 0.204 | 0.070 | 0.337 | 0.200 | 0.044 | 0.355 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind minus production_cleaned | Wind | -3.744 | -4.849 | -2.678 | -3.725 | -4.923 | -2.527 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## precip-era-after_precip-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.720 | 12.322 | 15.056 | 3 | 8 | 3 | 66457 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 13.732 | 12.333 | 15.065 | 3 | 8 | 3 | 66457 | False |

## precip-era-after_precip-fold_b; sensitivity; padding equal — descriptive availability-era paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.012 | -0.017 | 0.043 | 0.013 | -0.007 | 0.033 | 3 | 8 | 3 | False | 8 | 8 | 0 | 0 | 0 |

## precip-era-before_precip-fold_b; sensitivity; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.399 | 10.927 | 13.976 | 3 | 8 | 3 | 69464 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 12.222 | 10.681 | 13.775 | 3 | 8 | 3 | 69464 | False |

## precip-era-before_precip-fold_b; sensitivity; padding equal — descriptive availability-era paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | -0.177 | -0.593 | 0.216 | -0.210 | -0.677 | 0.257 | 5 | 8 | 3 | False | 8 | 8 | 0 | 0 | 0 |

## Requested minimal-feature comparisons

**The requested sets use the exact columns listed in the study's feature mapping.**
`solar_minimal_requested` is PV2, `solar_expanded_requested` is PV11,
`wind_minimal_requested` is Wind5, `wind_minimal_uv_requested` is Wind9 u/v, and
`wind_minimal_circular_requested` is Wind7 circular. The
[ordered input lists](fixed-pipeline-lean-features.md#requested-minimal-feature-extension)
identify every real column, each reference recipe, and the duplicate padding. Full input
(`production_cleaned`) means the historical Part 1b point control; `lean_wind` means the original
16-input wind set. Neither label denotes the production configuration or pooled quick-wins winner.

These post hoc tables cover the forward Fold A and retrospective Fold B. Equal-padding scopes
train with 24 columns in both settings. Main-setting `none` scopes use the requested sets' real
columns without duplicates. Within a `none` report, a `__padded` suffix identifies the corresponding
equal-padding fit, so a padded-minus-unpadded row compares the same real input set. Guarded
reference labels change the windchill domain rule in a reference that contains windchill; the
five requested sets omit windchill. Untargeted series retain the full control's inputs.

Fair CRPS is the deciding loss for these matched 51-value ensembles. Every difference is candidate
minus reference in capacity percentage points; negative favours the candidate. These are frozen
radiation `0f094c3` results, rather than an evaluation of merged `01ddf53`. The comparison-class
column distinguishes deciding, exploratory, guard and padding checks. A displayed zero is rounded
and does not establish exact equality. All table cells below come from the unchanged exporter.

### Post hoc fold_a_main_equal: PV absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.089 | 4.436 | 7.896 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 7.331 | 5.436 | 9.399 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | PV | 7.897 | 5.810 | 10.147 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_main_equal: Wind absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.572 | 12.153 | 14.889 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.624 | 12.234 | 14.927 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.600 | 12.192 | 14.910 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | Wind | 17.133 | 15.530 | 18.669 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill | Wind | 13.820 | 12.257 | 15.207 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_main_equal: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -1.800 | -2.296 | -1.310 | -2.409 | -3.785 | -1.033 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.559 | -0.839 | -0.308 | -0.835 | -1.474 | -0.196 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.241 | 0.864 | 1.641 | 1.574 | 0.731 | 2.417 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.579 | -4.269 | -2.927 | -3.568 | -4.329 | -2.807 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | -0.249 | -0.554 | 0.026 | -0.259 | -0.598 | 0.081 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.052 | 0.015 | 0.095 | 0.054 | 0.008 | 0.099 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.028 | -0.002 | 0.064 | 0.029 | -0.010 | 0.067 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -2.006 | -2.617 | -1.443 | -2.516 | -3.779 | -1.254 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.764 | -1.080 | -0.478 | -0.942 | -1.462 | -0.423 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.527 | -4.233 | -2.869 | -3.514 | -4.290 | -2.739 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | -0.197 | -0.494 | 0.080 | -0.205 | -0.541 | 0.131 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.551 | -4.257 | -2.891 | -3.539 | -4.312 | -2.767 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | -0.221 | -0.516 | 0.048 | -0.230 | -0.563 | 0.103 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | -0.024 | -0.047 | -0.000 | -0.025 | -0.048 | -0.002 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_minimal_requested | production_cleaned_guarded_windchill | PV | -1.808 | -2.306 | -1.330 | -2.418 | -3.801 | -1.034 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_expanded_requested | production_cleaned_guarded_windchill | PV | -0.566 | -0.845 | -0.316 | -0.844 | -1.490 | -0.197 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | production_cleaned_guarded_windchill | Wind | -3.561 | -4.221 | -2.937 | -3.551 | -4.291 | -2.812 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | production_cleaned_guarded_windchill | Wind | -3.509 | -4.183 | -2.880 | -3.498 | -4.252 | -2.744 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | production_cleaned_guarded_windchill | Wind | -3.534 | -4.206 | -2.904 | -3.523 | -4.274 | -2.771 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | lean_wind_guarded_windchill | Wind | -0.248 | -0.549 | 0.022 | -0.258 | -0.594 | 0.078 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | lean_wind_guarded_windchill | Wind | -0.196 | -0.483 | 0.080 | -0.204 | -0.536 | 0.128 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | lean_wind_guarded_windchill | Wind | -0.220 | -0.510 | 0.045 | -0.229 | -0.558 | 0.099 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_sensitivity_equal: PV absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.138 | 4.467 | 7.961 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 7.252 | 5.411 | 9.257 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | PV | 7.763 | 5.772 | 9.924 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv | PV | 7.881 | 5.854 | 10.119 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | PV | 7.768 | 5.778 | 9.933 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_sensitivity_equal: Wind absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.666 | 12.275 | 14.961 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.709 | 12.338 | 14.997 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.703 | 12.302 | 15.007 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | Wind | 16.843 | 15.085 | 18.527 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind | Wind | 13.851 | 12.308 | 15.275 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | Wind | 16.840 | 15.087 | 18.525 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind_guarded_windchill | Wind | 13.860 | 12.321 | 15.288 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_sensitivity_equal: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -1.625 | -2.048 | -1.235 | -2.111 | -3.297 | -0.926 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.510 | -0.739 | -0.291 | -0.656 | -1.057 | -0.255 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.114 | 0.778 | 1.479 | 1.455 | 0.550 | 2.360 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.177 | -3.941 | -2.463 | -3.160 | -4.018 | -2.302 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | -0.185 | -0.467 | 0.089 | -0.192 | -0.520 | 0.136 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.044 | 0.016 | 0.069 | 0.044 | 0.013 | 0.075 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.037 | 0.013 | 0.064 | 0.037 | 0.007 | 0.066 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -1.743 | -2.237 | -1.296 | -2.208 | -3.409 | -1.007 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.628 | -0.866 | -0.416 | -0.752 | -1.141 | -0.364 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.133 | -3.896 | -2.417 | -3.116 | -3.975 | -2.258 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | -0.141 | -0.421 | 0.129 | -0.148 | -0.477 | 0.180 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.140 | -3.892 | -2.434 | -3.123 | -3.973 | -2.273 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | -0.148 | -0.420 | 0.112 | -0.155 | -0.473 | 0.163 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | -0.006 | -0.038 | 0.025 | -0.007 | -0.042 | 0.028 | 8 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_minimal_requested | production_cleaned_guarded_windchill | PV | -1.630 | -2.057 | -1.235 | -2.128 | -3.337 | -0.920 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_expanded_requested | production_cleaned_guarded_windchill | PV | -0.516 | -0.740 | -0.302 | -0.673 | -1.089 | -0.257 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | production_cleaned_guarded_windchill | Wind | -3.174 | -3.926 | -2.480 | -3.157 | -4.003 | -2.311 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | production_cleaned_guarded_windchill | Wind | -3.130 | -3.896 | -2.429 | -3.113 | -3.960 | -2.267 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | production_cleaned_guarded_windchill | Wind | -3.137 | -3.878 | -2.434 | -3.120 | -3.958 | -2.282 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | lean_wind_guarded_windchill | Wind | -0.194 | -0.476 | 0.080 | -0.201 | -0.531 | 0.129 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | lean_wind_guarded_windchill | Wind | -0.150 | -0.433 | 0.122 | -0.157 | -0.488 | 0.174 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | lean_wind_guarded_windchill | Wind | -0.157 | -0.432 | 0.105 | -0.164 | -0.484 | 0.156 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_main_none: PV absolute diagnostic fair CRPS; includes equal-padding aliases for direct padding checks

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.089 | 4.436 | 7.896 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 7.331 | 5.436 | 9.399 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | PV | 7.889 | 5.802 | 10.144 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| lean_pv | PV | 8.095 | 5.953 | 10.420 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested__padded | PV | 6.089 | 4.436 | 7.896 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested__padded | PV | 7.331 | 5.436 | 9.399 | 3 | 12 | 6 | 175795 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_main_none: Wind absolute diagnostic fair CRPS; includes equal-padding aliases for direct padding checks

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.572 | 12.153 | 14.889 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.624 | 12.234 | 14.927 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.600 | 12.192 | 14.910 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| production_cleaned | Wind | 17.151 | 15.507 | 18.710 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested__padded | Wind | 13.572 | 12.153 | 14.889 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested__padded | Wind | 13.624 | 12.234 | 14.927 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested__padded | Wind | 13.600 | 12.192 | 14.910 | 3 | 12 | 3 | 107121 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_main_none: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -1.800 | -2.296 | -1.310 | -2.409 | -3.785 | -1.033 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.559 | -0.839 | -0.308 | -0.835 | -1.474 | -0.196 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.241 | 0.864 | 1.641 | 1.574 | 0.731 | 2.417 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.579 | -4.269 | -2.927 | -3.568 | -4.329 | -2.807 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | -0.249 | -0.554 | 0.026 | -0.259 | -0.598 | 0.081 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.052 | 0.015 | 0.095 | 0.054 | 0.008 | 0.099 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.028 | -0.002 | 0.064 | 0.029 | -0.010 | 0.067 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -2.006 | -2.617 | -1.443 | -2.516 | -3.779 | -1.254 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.764 | -1.080 | -0.478 | -0.942 | -1.462 | -0.423 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.527 | -4.233 | -2.869 | -3.514 | -4.290 | -2.739 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | -0.197 | -0.494 | 0.080 | -0.205 | -0.541 | 0.131 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.551 | -4.257 | -2.891 | -3.539 | -4.312 | -2.767 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | -0.221 | -0.516 | 0.048 | -0.230 | -0.563 | 0.103 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | -0.024 | -0.047 | -0.000 | -0.025 | -0.048 | -0.002 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested__padded minus solar_minimal_requested | post_hoc_padding_diagnostic | solar_minimal_requested__padded | solar_minimal_requested | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested__padded minus solar_expanded_requested | post_hoc_padding_diagnostic | solar_expanded_requested__padded | solar_expanded_requested | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested__padded minus wind_minimal_requested | post_hoc_padding_diagnostic | wind_minimal_requested__padded | wind_minimal_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested__padded minus wind_minimal_uv_requested | post_hoc_padding_diagnostic | wind_minimal_uv_requested__padded | wind_minimal_uv_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested__padded minus wind_minimal_circular_requested | post_hoc_padding_diagnostic | wind_minimal_circular_requested__padded | wind_minimal_circular_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_b_main_equal: PV absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.465 | 5.298 | 7.578 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 8.108 | 6.548 | 9.579 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | PV | 8.627 | 6.994 | 10.151 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_main_equal: Wind absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.419 | 12.180 | 14.602 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.449 | 12.214 | 14.646 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.437 | 12.198 | 14.636 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | Wind | 16.701 | 15.013 | 18.349 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill | Wind | 12.890 | 11.782 | 13.963 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_main_equal: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -2.147 | -2.769 | -1.538 | -2.142 | -2.849 | -1.435 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.505 | -1.006 | 0.016 | -0.476 | -1.064 | 0.112 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.642 | 1.004 | 2.349 | 1.666 | 0.881 | 2.452 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.201 | -4.289 | -2.201 | -3.182 | -4.348 | -2.016 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | 0.543 | 0.304 | 0.805 | 0.543 | 0.263 | 0.823 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.030 | 0.005 | 0.057 | 0.030 | 0.004 | 0.056 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.018 | -0.005 | 0.043 | 0.019 | -0.007 | 0.045 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -2.351 | -2.981 | -1.701 | -2.342 | -3.100 | -1.584 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.709 | -1.256 | -0.168 | -0.675 | -1.325 | -0.025 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.171 | -4.258 | -2.170 | -3.152 | -4.321 | -1.982 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | 0.573 | 0.322 | 0.849 | 0.573 | 0.277 | 0.869 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.183 | -4.264 | -2.183 | -3.163 | -4.330 | -1.996 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | 0.561 | 0.313 | 0.835 | 0.562 | 0.269 | 0.854 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | -0.012 | -0.027 | 0.003 | -0.011 | -0.027 | 0.004 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_minimal_requested | production_cleaned_guarded_windchill | PV | -2.161 | -2.774 | -1.552 | -2.156 | -2.864 | -1.448 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_expanded_requested | production_cleaned_guarded_windchill | PV | -0.519 | -1.006 | -0.012 | -0.490 | -1.076 | 0.097 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | production_cleaned_guarded_windchill | Wind | -3.282 | -4.304 | -2.302 | -3.263 | -4.407 | -2.120 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | production_cleaned_guarded_windchill | Wind | -3.252 | -4.284 | -2.274 | -3.233 | -4.380 | -2.087 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | production_cleaned_guarded_windchill | Wind | -3.264 | -4.300 | -2.289 | -3.245 | -4.388 | -2.101 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | lean_wind_guarded_windchill | Wind | 0.529 | 0.288 | 0.790 | 0.529 | 0.245 | 0.814 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | lean_wind_guarded_windchill | Wind | 0.559 | 0.308 | 0.837 | 0.560 | 0.259 | 0.860 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | lean_wind_guarded_windchill | Wind | 0.547 | 0.301 | 0.825 | 0.548 | 0.251 | 0.846 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_sensitivity_equal: PV absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.522 | 5.341 | 7.640 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 7.839 | 6.416 | 9.195 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | PV | 8.615 | 6.985 | 10.146 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv | PV | 8.752 | 7.082 | 10.307 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | PV | 8.615 | 6.991 | 10.134 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_sensitivity_equal: Wind absolute diagnostic fair CRPS; equal-padding controls

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.562 | 12.308 | 14.789 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.580 | 12.320 | 14.818 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.593 | 12.333 | 14.829 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | Wind | 16.677 | 15.030 | 18.373 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind | Wind | 13.045 | 11.913 | 14.117 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned_guarded_windchill | Wind | 16.594 | 14.900 | 18.304 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind_guarded_windchill | Wind | 13.028 | 11.905 | 14.092 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_sensitivity_equal: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -2.092 | -2.703 | -1.501 | -2.094 | -2.797 | -1.392 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.776 | -1.219 | -0.296 | -0.757 | -1.299 | -0.216 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.316 | 0.797 | 1.884 | 1.337 | 0.697 | 1.977 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.115 | -4.235 | -2.114 | -3.086 | -4.293 | -1.880 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | 0.517 | 0.296 | 0.754 | 0.519 | 0.265 | 0.774 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.018 | -0.002 | 0.037 | 0.018 | -0.003 | 0.040 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.031 | 0.010 | 0.055 | 0.033 | 0.007 | 0.059 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -2.230 | -2.849 | -1.614 | -2.227 | -2.947 | -1.507 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.914 | -1.437 | -0.383 | -0.890 | -1.503 | -0.277 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.097 | -4.234 | -2.094 | -3.068 | -4.283 | -1.853 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | 0.535 | 0.304 | 0.781 | 0.538 | 0.273 | 0.802 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.084 | -4.208 | -2.087 | -3.054 | -4.258 | -1.850 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | 0.549 | 0.331 | 0.787 | 0.552 | 0.294 | 0.810 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | 0.013 | -0.009 | 0.037 | 0.014 | -0.012 | 0.041 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_minimal_requested | production_cleaned_guarded_windchill | PV | -2.093 | -2.716 | -1.504 | -2.094 | -2.796 | -1.392 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | solar_expanded_requested | production_cleaned_guarded_windchill | PV | -0.776 | -1.232 | -0.292 | -0.757 | -1.299 | -0.215 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | production_cleaned_guarded_windchill | Wind | -3.032 | -4.126 | -2.050 | -3.004 | -4.174 | -1.835 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | production_cleaned_guarded_windchill | Wind | -3.014 | -4.114 | -2.026 | -2.986 | -4.163 | -1.809 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | production_cleaned_guarded_windchill | Wind | -3.001 | -4.089 | -2.026 | -2.972 | -4.138 | -1.805 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_requested | lean_wind_guarded_windchill | Wind | 0.534 | 0.320 | 0.766 | 0.537 | 0.284 | 0.789 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_uv_requested | lean_wind_guarded_windchill | Wind | 0.552 | 0.330 | 0.793 | 0.555 | 0.293 | 0.817 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind_guarded_windchill | post_hoc_guard_diagnostic | wind_minimal_circular_requested | lean_wind_guarded_windchill | Wind | 0.566 | 0.349 | 0.805 | 0.569 | 0.313 | 0.825 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_main_none: PV absolute diagnostic fair CRPS; includes equal-padding aliases for direct padding checks

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested | PV | 6.465 | 5.298 | 7.578 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested | PV | 8.108 | 6.548 | 9.579 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | PV | 8.612 | 6.976 | 10.134 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| lean_pv | PV | 8.816 | 7.103 | 10.392 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested__padded | PV | 6.465 | 5.298 | 7.578 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested__padded | PV | 8.108 | 6.548 | 9.579 | 3 | 15 | 6 | 253003 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_main_none: Wind absolute diagnostic fair CRPS; includes equal-padding aliases for direct padding checks

| label | group | Fair CRPS (capacity points) | Whole-month/seed bootstrap 95% lower | Whole-month/seed bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_minimal_requested | Wind | 13.419 | 12.180 | 14.602 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested | Wind | 13.449 | 12.214 | 14.646 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested | Wind | 13.437 | 12.198 | 14.636 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| production_cleaned | Wind | 16.620 | 14.859 | 18.291 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested__padded | Wind | 13.419 | 12.180 | 14.602 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested__padded | Wind | 13.449 | 12.214 | 14.646 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested__padded | Wind | 13.437 | 12.198 | 14.636 | 3 | 15 | 3 | 135921 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_main_none: paired diagnostic fair CRPS, left minus right; classes shown explicitly

| contrast | Post hoc comparison class | left | right | group | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower | Paired whole-month/seed bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | -2.147 | -2.769 | -1.538 | -2.142 | -2.849 | -1.435 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | -0.505 | -1.006 | 0.016 | -0.476 | -1.064 | 0.112 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | 1.642 | 1.004 | 2.349 | 1.666 | 0.881 | 2.452 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | -3.201 | -4.289 | -2.201 | -3.182 | -4.348 | -2.016 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | 0.543 | 0.304 | 0.805 | 0.543 | 0.263 | 0.823 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | 0.030 | 0.005 | 0.057 | 0.030 | 0.004 | 0.056 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | 0.018 | -0.005 | 0.043 | 0.019 | -0.007 | 0.045 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus lean_pv | post_hoc_exploratory | solar_minimal_requested | lean_pv | PV | -2.351 | -2.981 | -1.701 | -2.342 | -3.100 | -1.584 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus lean_pv | post_hoc_exploratory | solar_expanded_requested | lean_pv | PV | -0.709 | -1.256 | -0.168 | -0.675 | -1.325 | -0.025 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_uv_requested | production_cleaned | Wind | -3.171 | -4.258 | -2.170 | -3.152 | -4.321 | -1.982 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus lean_wind | post_hoc_exploratory | wind_minimal_uv_requested | lean_wind | Wind | 0.573 | 0.322 | 0.849 | 0.573 | 0.277 | 0.869 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus production_cleaned | post_hoc_exploratory | wind_minimal_circular_requested | production_cleaned | Wind | -3.183 | -4.264 | -2.183 | -3.163 | -4.330 | -1.996 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus lean_wind | post_hoc_exploratory | wind_minimal_circular_requested | lean_wind | Wind | 0.561 | 0.313 | 0.835 | 0.562 | 0.269 | 0.854 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_uv_requested | post_hoc_exploratory | wind_minimal_circular_requested | wind_minimal_uv_requested | Wind | -0.012 | -0.027 | 0.003 | -0.011 | -0.027 | 0.004 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested__padded minus solar_minimal_requested | post_hoc_padding_diagnostic | solar_minimal_requested__padded | solar_minimal_requested | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested__padded minus solar_expanded_requested | post_hoc_padding_diagnostic | solar_expanded_requested__padded | solar_expanded_requested | PV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested__padded minus wind_minimal_requested | post_hoc_padding_diagnostic | wind_minimal_requested__padded | wind_minimal_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested__padded minus wind_minimal_uv_requested | post_hoc_padding_diagnostic | wind_minimal_uv_requested__padded | wind_minimal_uv_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested__padded minus wind_minimal_circular_requested | post_hoc_padding_diagnostic | wind_minimal_circular_requested__padded | wind_minimal_circular_requested | Wind | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## Requested comparisons by forecast lead and loss metric

These supplementary tables retain all seven deciding feature comparisons in both folds and
settings, for all leads together and each of the three fixed lead bands. Fair CRPS (`crps_norm`)
is the deciding loss. Mean absolute error (`abs_error_norm`) and 19-level pinball loss
(`pinball19_norm`) are secondary diagnostics; their lead slices do not select additional winners.
The paired month-and-seed bootstrap describes the equal-series estimate. The seed-averaged
equal-month estimate and its t interval address a separate estimand and can disagree.

The saved provisional flag reflects fitting-seed replication, rather than certifying complete
monthly support. Read the minimum and maximum supported months, missing-month series and draws
dropping a series for each particular comparison. Complete all-lead support does not establish
complete support within every lead band. Every fixed row remains visible regardless of its sign,
interval or support. These post hoc slices have no correction for multiple comparisons and retain
the same frozen-source limitation as the primary tables.

### Post hoc fold_a_main_equal: seven fixed deciding families, all four lead bands and three loss metrics; left minus right in capacity points

| contrast | Post hoc comparison class | left | right | group | Lead band | Loss metric | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower (capacity points) | Paired whole-month/seed bootstrap 95% upper (capacity points) | Equal-month difference (capacity points) | Equal-month t 95% lower (capacity points) | Equal-month t 95% upper (capacity points) | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | crps_norm | -1.800 | -2.296 | -1.310 | -2.409 | -3.785 | -1.033 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | abs_error_norm | -0.867 | -1.154 | -0.588 | -1.387 | -2.683 | -0.091 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | pinball19_norm | -1.766 | -2.253 | -1.290 | -2.368 | -3.733 | -1.004 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -1.819 | -2.405 | -1.299 | -1.827 | -2.410 | -1.244 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -1.302 | -1.776 | -0.907 | -1.280 | -1.730 | -0.830 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -1.798 | -2.378 | -1.282 | -1.805 | -2.381 | -1.228 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -1.772 | -2.304 | -1.263 | -2.408 | -3.811 | -1.005 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.819 | -1.156 | -0.519 | -1.360 | -2.680 | -0.041 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -1.737 | -2.261 | -1.236 | -2.366 | -3.757 | -0.975 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | crps_norm | -1.847 | -2.479 | -1.295 | -1.954 | -2.707 | -1.202 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.657 | -1.052 | -0.220 | -0.625 | -1.301 | 0.051 | 10 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -1.806 | -2.424 | -1.267 | -1.907 | -2.649 | -1.165 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | crps_norm | -0.559 | -0.839 | -0.308 | -0.835 | -1.474 | -0.196 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | abs_error_norm | -0.428 | -0.672 | -0.213 | -0.706 | -1.317 | -0.095 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | pinball19_norm | -0.552 | -0.830 | -0.304 | -0.825 | -1.457 | -0.193 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -0.323 | -0.664 | -0.020 | -0.339 | -0.717 | 0.038 | 9 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -0.337 | -0.659 | -0.046 | -0.354 | -0.719 | 0.010 | 9 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -0.322 | -0.662 | -0.020 | -0.338 | -0.715 | 0.039 | 9 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -0.569 | -0.882 | -0.288 | -0.858 | -1.538 | -0.178 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.448 | -0.728 | -0.211 | -0.738 | -1.389 | -0.088 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -0.562 | -0.873 | -0.282 | -0.847 | -1.520 | -0.174 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | crps_norm | -0.709 | -0.985 | -0.472 | -0.737 | -1.095 | -0.379 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.453 | -0.687 | -0.264 | -0.456 | -0.753 | -0.158 | 11 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -0.697 | -0.970 | -0.463 | -0.724 | -1.080 | -0.369 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | crps_norm | 1.241 | 0.864 | 1.641 | 1.574 | 0.731 | 2.417 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | abs_error_norm | 0.439 | 0.081 | 0.763 | 0.681 | -0.180 | 1.541 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | pinball19_norm | 1.214 | 0.845 | 1.608 | 1.544 | 0.703 | 2.384 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | crps_norm | 1.496 | 1.037 | 1.964 | 1.488 | 0.966 | 2.010 | 0 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | abs_error_norm | 0.965 | 0.587 | 1.337 | 0.926 | 0.473 | 1.380 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | pinball19_norm | 1.476 | 1.024 | 1.937 | 1.467 | 0.949 | 1.984 | 0 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | crps_norm | 1.203 | 0.819 | 1.640 | 1.550 | 0.705 | 2.396 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | abs_error_norm | 0.371 | 0.013 | 0.749 | 0.622 | -0.236 | 1.480 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | pinball19_norm | 1.175 | 0.799 | 1.608 | 1.519 | 0.676 | 2.362 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | crps_norm | 1.139 | 0.699 | 1.642 | 1.217 | 0.572 | 1.863 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | abs_error_norm | 0.203 | -0.289 | 0.624 | 0.170 | -0.609 | 0.948 | 3 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | pinball19_norm | 1.109 | 0.673 | 1.608 | 1.183 | 0.537 | 1.828 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | crps_norm | -3.579 | -4.269 | -2.927 | -3.568 | -4.329 | -2.807 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | abs_error_norm | -2.043 | -2.796 | -1.329 | -2.033 | -2.860 | -1.207 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | pinball19_norm | -3.502 | -4.185 | -2.860 | -3.492 | -4.249 | -2.735 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | crps_norm | -1.821 | -2.645 | -1.095 | -1.831 | -2.665 | -0.997 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | abs_error_norm | -1.410 | -2.214 | -0.717 | -1.415 | -2.231 | -0.599 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | pinball19_norm | -1.792 | -2.609 | -1.067 | -1.801 | -2.631 | -0.971 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | crps_norm | -3.508 | -4.271 | -2.865 | -3.497 | -4.295 | -2.700 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | abs_error_norm | -1.956 | -2.734 | -1.245 | -1.942 | -2.794 | -1.090 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | pinball19_norm | -3.429 | -4.188 | -2.790 | -3.418 | -4.212 | -2.625 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | crps_norm | -5.204 | -6.020 | -4.293 | -5.135 | -6.185 | -4.085 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | abs_error_norm | -2.765 | -3.698 | -1.695 | -2.711 | -3.914 | -1.507 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | pinball19_norm | -5.094 | -5.918 | -4.186 | -5.026 | -6.077 | -3.976 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | crps_norm | -0.249 | -0.554 | 0.026 | -0.259 | -0.598 | 0.081 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | abs_error_norm | 0.277 | -0.045 | 0.555 | 0.267 | -0.089 | 0.622 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | pinball19_norm | -0.222 | -0.525 | 0.052 | -0.232 | -0.570 | 0.106 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | crps_norm | 2.033 | 0.929 | 2.958 | 1.972 | 0.778 | 3.167 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | abs_error_norm | 1.935 | 0.790 | 2.909 | 1.873 | 0.620 | 3.126 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | pinball19_norm | 2.036 | 0.932 | 2.963 | 1.975 | 0.778 | 3.173 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | crps_norm | -0.402 | -0.641 | -0.174 | -0.416 | -0.697 | -0.136 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | abs_error_norm | 0.048 | -0.189 | 0.251 | 0.036 | -0.231 | 0.303 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | pinball19_norm | -0.375 | -0.612 | -0.150 | -0.390 | -0.667 | -0.112 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | crps_norm | -1.817 | -2.184 | -1.462 | -1.811 | -2.227 | -1.395 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | abs_error_norm | -0.602 | -0.883 | -0.351 | -0.609 | -0.915 | -0.303 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | pinball19_norm | -1.770 | -2.134 | -1.425 | -1.765 | -2.173 | -1.357 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | crps_norm | 0.052 | 0.015 | 0.095 | 0.054 | 0.008 | 0.099 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.026 | -0.004 | 0.062 | 0.027 | -0.012 | 0.066 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.049 | 0.013 | 0.089 | 0.050 | 0.006 | 0.094 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.114 | 0.010 | 0.253 | 0.114 | -0.030 | 0.258 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.095 | -0.002 | 0.234 | 0.097 | -0.041 | 0.234 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.112 | 0.007 | 0.251 | 0.112 | -0.033 | 0.256 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.035 | 0.001 | 0.067 | 0.036 | -0.001 | 0.073 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | 0.005 | -0.016 | 0.030 | 0.005 | -0.018 | 0.029 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.031 | -0.002 | 0.061 | 0.032 | -0.003 | 0.067 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.037 | 0.009 | 0.065 | 0.039 | 0.008 | 0.070 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | 0.015 | -0.006 | 0.039 | 0.014 | -0.013 | 0.041 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.035 | 0.007 | 0.061 | 0.036 | 0.007 | 0.065 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | crps_norm | 0.028 | -0.002 | 0.064 | 0.029 | -0.010 | 0.067 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.017 | -0.008 | 0.048 | 0.017 | -0.016 | 0.050 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.027 | -0.002 | 0.063 | 0.028 | -0.010 | 0.066 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.071 | -0.011 | 0.174 | 0.071 | -0.039 | 0.180 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.064 | -0.018 | 0.163 | 0.064 | -0.038 | 0.167 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.071 | -0.013 | 0.175 | 0.071 | -0.040 | 0.182 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.009 | -0.015 | 0.036 | 0.011 | -0.019 | 0.040 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | -0.001 | -0.020 | 0.020 | -0.001 | -0.023 | 0.021 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.008 | -0.015 | 0.034 | 0.010 | -0.019 | 0.038 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.031 | 0.011 | 0.053 | 0.033 | 0.010 | 0.056 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | 0.017 | -0.005 | 0.039 | 0.018 | -0.004 | 0.039 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.031 | 0.011 | 0.052 | 0.032 | 0.010 | 0.054 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_a_sensitivity_equal: seven fixed deciding families, all four lead bands and three loss metrics; left minus right in capacity points

| contrast | Post hoc comparison class | left | right | group | Lead band | Loss metric | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower (capacity points) | Paired whole-month/seed bootstrap 95% upper (capacity points) | Equal-month difference (capacity points) | Equal-month t 95% lower (capacity points) | Equal-month t 95% upper (capacity points) | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | crps_norm | -1.625 | -2.048 | -1.235 | -2.111 | -3.297 | -0.926 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | abs_error_norm | -0.821 | -1.125 | -0.534 | -1.255 | -2.442 | -0.067 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | pinball19_norm | -1.595 | -2.013 | -1.216 | -2.077 | -3.257 | -0.897 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -1.560 | -1.949 | -1.195 | -1.554 | -1.965 | -1.142 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -1.157 | -1.491 | -0.852 | -1.134 | -1.506 | -0.762 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -1.543 | -1.930 | -1.181 | -1.535 | -1.944 | -1.126 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -1.610 | -2.073 | -1.183 | -2.119 | -3.332 | -0.906 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.780 | -1.139 | -0.456 | -1.230 | -2.439 | -0.021 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -1.580 | -2.040 | -1.163 | -2.084 | -3.291 | -0.877 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | crps_norm | -1.704 | -2.264 | -1.202 | -1.785 | -2.483 | -1.086 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.668 | -1.079 | -0.195 | -0.624 | -1.350 | 0.102 | 10 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -1.667 | -2.221 | -1.166 | -1.743 | -2.435 | -1.051 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | crps_norm | -0.510 | -0.739 | -0.291 | -0.656 | -1.057 | -0.255 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | abs_error_norm | -0.490 | -0.725 | -0.272 | -0.671 | -1.102 | -0.240 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | pinball19_norm | -0.508 | -0.737 | -0.289 | -0.652 | -1.051 | -0.253 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -0.199 | -0.551 | 0.129 | -0.219 | -0.621 | 0.183 | 8 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -0.321 | -0.663 | -0.006 | -0.351 | -0.738 | 0.036 | 10 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -0.202 | -0.553 | 0.126 | -0.222 | -0.623 | 0.178 | 8 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -0.527 | -0.772 | -0.293 | -0.682 | -1.122 | -0.243 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.513 | -0.759 | -0.284 | -0.703 | -1.171 | -0.235 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -0.524 | -0.768 | -0.290 | -0.678 | -1.115 | -0.241 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | crps_norm | -0.701 | -0.941 | -0.501 | -0.719 | -1.017 | -0.421 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.565 | -0.809 | -0.367 | -0.572 | -0.862 | -0.282 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -0.695 | -0.935 | -0.495 | -0.713 | -1.010 | -0.416 | 12 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | crps_norm | 1.114 | 0.778 | 1.479 | 1.455 | 0.550 | 2.360 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | abs_error_norm | 0.332 | -0.055 | 0.684 | 0.584 | -0.369 | 1.537 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | pinball19_norm | 1.087 | 0.758 | 1.448 | 1.425 | 0.520 | 2.330 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | crps_norm | 1.361 | 0.942 | 1.777 | 1.335 | 0.844 | 1.826 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | abs_error_norm | 0.835 | 0.478 | 1.179 | 0.783 | 0.337 | 1.228 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | pinball19_norm | 1.341 | 0.928 | 1.753 | 1.313 | 0.826 | 1.800 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | crps_norm | 1.083 | 0.744 | 1.486 | 1.437 | 0.528 | 2.345 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | abs_error_norm | 0.268 | -0.125 | 0.663 | 0.527 | -0.430 | 1.484 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | pinball19_norm | 1.055 | 0.720 | 1.454 | 1.406 | 0.498 | 2.314 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | crps_norm | 1.003 | 0.591 | 1.458 | 1.066 | 0.450 | 1.681 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | abs_error_norm | 0.103 | -0.415 | 0.519 | 0.052 | -0.750 | 0.854 | 3 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | pinball19_norm | 0.973 | 0.562 | 1.420 | 1.030 | 0.414 | 1.646 | 1 | 12 | 3 | False | 11 | 12 | 1 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | crps_norm | -3.177 | -3.941 | -2.463 | -3.160 | -4.018 | -2.302 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | abs_error_norm | -1.686 | -2.423 | -0.983 | -1.672 | -2.505 | -0.840 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | pinball19_norm | -3.104 | -3.862 | -2.401 | -3.088 | -3.937 | -2.238 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | crps_norm | -1.317 | -2.324 | -0.424 | -1.308 | -2.318 | -0.297 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | abs_error_norm | -1.159 | -2.107 | -0.326 | -1.147 | -2.103 | -0.191 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | pinball19_norm | -1.299 | -2.305 | -0.416 | -1.290 | -2.295 | -0.284 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | crps_norm | -3.179 | -4.032 | -2.480 | -3.164 | -4.062 | -2.267 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | abs_error_norm | -1.660 | -2.470 | -0.971 | -1.644 | -2.504 | -0.783 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | pinball19_norm | -3.103 | -3.953 | -2.409 | -3.088 | -3.978 | -2.199 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | crps_norm | -4.726 | -5.491 | -3.892 | -4.656 | -5.637 | -3.674 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | abs_error_norm | -2.183 | -3.021 | -1.250 | -2.134 | -3.200 | -1.068 | 9 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | pinball19_norm | -4.616 | -5.379 | -3.773 | -4.547 | -5.522 | -3.572 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | crps_norm | -0.185 | -0.467 | 0.089 | -0.192 | -0.520 | 0.136 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | abs_error_norm | 0.391 | 0.114 | 0.650 | 0.383 | 0.072 | 0.695 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | pinball19_norm | -0.158 | -0.439 | 0.116 | -0.166 | -0.491 | 0.160 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | crps_norm | 2.388 | 1.573 | 3.122 | 2.346 | 1.449 | 3.243 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | abs_error_norm | 2.107 | 1.286 | 2.860 | 2.064 | 1.135 | 2.993 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | pinball19_norm | 2.384 | 1.564 | 3.116 | 2.341 | 1.443 | 3.239 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | crps_norm | -0.334 | -0.596 | -0.073 | -0.347 | -0.655 | -0.038 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | abs_error_norm | 0.154 | -0.098 | 0.379 | 0.144 | -0.139 | 0.426 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | pinball19_norm | -0.307 | -0.567 | -0.049 | -0.321 | -0.627 | -0.014 | 10 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | crps_norm | -2.008 | -2.367 | -1.667 | -2.004 | -2.407 | -1.602 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | abs_error_norm | -0.521 | -0.796 | -0.279 | -0.529 | -0.821 | -0.238 | 11 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | pinball19_norm | -1.954 | -2.308 | -1.618 | -1.951 | -2.346 | -1.556 | 12 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | crps_norm | 0.044 | 0.016 | 0.069 | 0.044 | 0.013 | 0.075 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.033 | 0.005 | 0.062 | 0.033 | -0.000 | 0.067 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.044 | 0.017 | 0.069 | 0.044 | 0.014 | 0.074 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.129 | 0.017 | 0.261 | 0.126 | -0.019 | 0.270 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.139 | 0.022 | 0.280 | 0.137 | -0.013 | 0.288 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.129 | 0.017 | 0.262 | 0.126 | -0.019 | 0.270 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.023 | -0.002 | 0.047 | 0.023 | -0.004 | 0.050 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | 0.010 | -0.009 | 0.029 | 0.009 | -0.010 | 0.029 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.024 | -0.001 | 0.047 | 0.023 | -0.003 | 0.049 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.018 | -0.004 | 0.039 | 0.019 | -0.007 | 0.045 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | -0.003 | -0.020 | 0.017 | -0.003 | -0.025 | 0.019 | 7 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.018 | -0.002 | 0.037 | 0.019 | -0.004 | 0.043 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | crps_norm | 0.037 | 0.013 | 0.064 | 0.037 | 0.007 | 0.066 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.030 | 0.005 | 0.058 | 0.029 | -0.002 | 0.060 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.038 | 0.014 | 0.065 | 0.038 | 0.008 | 0.067 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.101 | 0.027 | 0.184 | 0.098 | 0.010 | 0.186 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.105 | 0.035 | 0.183 | 0.103 | 0.017 | 0.188 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.102 | 0.027 | 0.186 | 0.099 | 0.011 | 0.188 | 3 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.020 | -0.008 | 0.051 | 0.020 | -0.014 | 0.054 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | 0.010 | -0.018 | 0.040 | 0.009 | -0.024 | 0.041 | 6 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.022 | -0.006 | 0.052 | 0.022 | -0.012 | 0.055 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.022 | 0.010 | 0.034 | 0.023 | 0.009 | 0.037 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | 0.013 | -0.005 | 0.035 | 0.014 | -0.009 | 0.037 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.021 | 0.009 | 0.033 | 0.022 | 0.008 | 0.036 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

### Post hoc fold_b_main_equal: seven fixed deciding families, all four lead bands and three loss metrics; left minus right in capacity points

| contrast | Post hoc comparison class | left | right | group | Lead band | Loss metric | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower (capacity points) | Paired whole-month/seed bootstrap 95% upper (capacity points) | Equal-month difference (capacity points) | Equal-month t 95% lower (capacity points) | Equal-month t 95% upper (capacity points) | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | crps_norm | -2.147 | -2.769 | -1.538 | -2.142 | -2.849 | -1.435 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | abs_error_norm | -1.092 | -1.596 | -0.609 | -1.091 | -1.668 | -0.513 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | pinball19_norm | -2.108 | -2.725 | -1.503 | -2.103 | -2.805 | -1.402 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -1.903 | -2.641 | -1.152 | -1.888 | -2.718 | -1.058 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -1.382 | -2.062 | -0.714 | -1.364 | -2.138 | -0.589 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -1.880 | -2.615 | -1.134 | -1.865 | -2.692 | -1.037 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -2.231 | -2.851 | -1.636 | -2.216 | -2.929 | -1.502 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -1.068 | -1.573 | -0.611 | -1.064 | -1.636 | -0.491 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -2.188 | -2.806 | -1.599 | -2.173 | -2.880 | -1.465 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | crps_norm | -2.186 | -2.855 | -1.565 | -2.245 | -3.043 | -1.447 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.897 | -1.424 | -0.409 | -0.945 | -1.602 | -0.289 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -2.143 | -2.811 | -1.525 | -2.201 | -2.993 | -1.410 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | crps_norm | -0.505 | -1.006 | 0.016 | -0.476 | -1.064 | 0.112 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | abs_error_norm | -0.125 | -0.612 | 0.399 | -0.098 | -0.677 | 0.482 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | pinball19_norm | -0.488 | -0.991 | 0.032 | -0.459 | -1.048 | 0.129 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | crps_norm | 0.204 | -0.473 | 0.899 | 0.178 | -0.631 | 0.986 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | 0.347 | -0.349 | 1.056 | 0.322 | -0.485 | 1.129 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | 0.212 | -0.465 | 0.907 | 0.186 | -0.623 | 0.994 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -0.591 | -1.092 | -0.081 | -0.556 | -1.135 | 0.023 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.169 | -0.643 | 0.341 | -0.139 | -0.706 | 0.428 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -0.573 | -1.076 | -0.060 | -0.538 | -1.117 | 0.042 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | crps_norm | -0.927 | -1.416 | -0.402 | -0.919 | -1.498 | -0.341 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.433 | -0.895 | 0.081 | -0.420 | -0.972 | 0.132 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -0.906 | -1.395 | -0.382 | -0.898 | -1.475 | -0.321 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | crps_norm | 1.642 | 1.004 | 2.349 | 1.666 | 0.881 | 2.452 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | abs_error_norm | 0.967 | 0.336 | 1.657 | 0.993 | 0.215 | 1.770 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | pinball19_norm | 1.620 | 0.983 | 2.329 | 1.644 | 0.859 | 2.430 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | crps_norm | 2.107 | 1.293 | 2.991 | 2.065 | 1.121 | 3.010 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | abs_error_norm | 1.730 | 0.915 | 2.632 | 1.685 | 0.745 | 2.626 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | pinball19_norm | 2.092 | 1.277 | 2.976 | 2.050 | 1.106 | 2.994 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | crps_norm | 1.640 | 1.022 | 2.340 | 1.660 | 0.878 | 2.442 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | abs_error_norm | 0.898 | 0.284 | 1.578 | 0.925 | 0.161 | 1.688 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | pinball19_norm | 1.615 | 0.994 | 2.318 | 1.635 | 0.852 | 2.417 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | crps_norm | 1.259 | 0.730 | 1.868 | 1.325 | 0.612 | 2.039 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | abs_error_norm | 0.464 | -0.119 | 1.091 | 0.525 | -0.218 | 1.268 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | pinball19_norm | 1.237 | 0.707 | 1.847 | 1.303 | 0.588 | 2.018 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | crps_norm | -3.201 | -4.289 | -2.201 | -3.182 | -4.348 | -2.016 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | abs_error_norm | -1.622 | -2.756 | -0.613 | -1.603 | -2.799 | -0.406 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | pinball19_norm | -3.132 | -4.219 | -2.134 | -3.112 | -4.279 | -1.946 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | crps_norm | -1.152 | -2.203 | -0.278 | -1.185 | -2.248 | -0.121 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | abs_error_norm | -0.713 | -1.828 | 0.203 | -0.750 | -1.854 | 0.355 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | pinball19_norm | -1.123 | -2.180 | -0.249 | -1.156 | -2.222 | -0.090 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | crps_norm | -3.510 | -4.571 | -2.488 | -3.493 | -4.652 | -2.334 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | abs_error_norm | -1.955 | -3.033 | -0.941 | -1.941 | -3.105 | -0.777 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | pinball19_norm | -3.439 | -4.496 | -2.414 | -3.422 | -4.581 | -2.263 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | crps_norm | -4.224 | -5.870 | -2.809 | -4.161 | -5.801 | -2.521 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | abs_error_norm | -1.643 | -3.412 | -0.190 | -1.565 | -3.267 | 0.138 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | pinball19_norm | -4.125 | -5.779 | -2.711 | -4.060 | -5.701 | -2.419 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | crps_norm | 0.543 | 0.304 | 0.805 | 0.543 | 0.263 | 0.823 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | abs_error_norm | 0.984 | 0.730 | 1.261 | 0.985 | 0.696 | 1.274 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | pinball19_norm | 0.560 | 0.319 | 0.823 | 0.560 | 0.278 | 0.842 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | crps_norm | 3.467 | 2.731 | 4.151 | 3.452 | 2.660 | 4.245 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | abs_error_norm | 3.335 | 2.538 | 4.073 | 3.319 | 2.454 | 4.183 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | pinball19_norm | 3.469 | 2.727 | 4.157 | 3.454 | 2.659 | 4.250 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | crps_norm | 0.254 | -0.076 | 0.588 | 0.245 | -0.138 | 0.629 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | abs_error_norm | 0.578 | 0.253 | 0.898 | 0.569 | 0.198 | 0.940 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | pinball19_norm | 0.270 | -0.061 | 0.606 | 0.262 | -0.123 | 0.647 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | crps_norm | -1.250 | -1.579 | -0.979 | -1.277 | -1.619 | -0.934 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | abs_error_norm | -0.075 | -0.396 | 0.171 | -0.082 | -0.396 | 0.232 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | pinball19_norm | -1.220 | -1.548 | -0.949 | -1.246 | -1.588 | -0.905 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | crps_norm | 0.030 | 0.005 | 0.057 | 0.030 | 0.004 | 0.056 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.004 | -0.017 | 0.025 | 0.004 | -0.019 | 0.027 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.028 | 0.003 | 0.054 | 0.028 | 0.002 | 0.054 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.038 | -0.010 | 0.089 | 0.037 | -0.009 | 0.083 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.036 | -0.020 | 0.096 | 0.035 | -0.026 | 0.095 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.036 | -0.012 | 0.088 | 0.035 | -0.011 | 0.082 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.031 | 0.006 | 0.057 | 0.031 | 0.004 | 0.059 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | -0.002 | -0.020 | 0.016 | -0.001 | -0.020 | 0.017 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.029 | 0.004 | 0.055 | 0.029 | 0.002 | 0.056 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.020 | -0.002 | 0.041 | 0.020 | -0.002 | 0.042 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | -0.011 | -0.031 | 0.009 | -0.011 | -0.030 | 0.008 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.019 | -0.004 | 0.039 | 0.019 | -0.003 | 0.041 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | crps_norm | 0.018 | -0.005 | 0.043 | 0.019 | -0.007 | 0.045 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | abs_error_norm | -0.011 | -0.029 | 0.007 | -0.010 | -0.030 | 0.010 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.016 | -0.006 | 0.040 | 0.017 | -0.008 | 0.043 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.018 | -0.033 | 0.067 | 0.018 | -0.035 | 0.071 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.001 | -0.053 | 0.052 | 0.001 | -0.055 | 0.057 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.018 | -0.034 | 0.066 | 0.018 | -0.035 | 0.071 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.019 | -0.005 | 0.044 | 0.019 | -0.007 | 0.046 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | -0.012 | -0.029 | 0.005 | -0.011 | -0.030 | 0.007 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.017 | -0.006 | 0.042 | 0.018 | -0.008 | 0.044 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.016 | -0.001 | 0.034 | 0.017 | 0.001 | 0.033 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | -0.020 | -0.036 | -0.005 | -0.019 | -0.034 | -0.005 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.014 | -0.004 | 0.031 | 0.015 | -0.001 | 0.031 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

### Post hoc fold_b_sensitivity_equal: seven fixed deciding families, all four lead bands and three loss metrics; left minus right in capacity points

| contrast | Post hoc comparison class | left | right | group | Lead band | Loss metric | Equal-series difference (capacity points) | Paired whole-month/seed bootstrap 95% lower (capacity points) | Paired whole-month/seed bootstrap 95% upper (capacity points) | Equal-month difference (capacity points) | Equal-month t 95% lower (capacity points) | Equal-month t 95% upper (capacity points) | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | crps_norm | -2.092 | -2.703 | -1.501 | -2.094 | -2.797 | -1.392 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | abs_error_norm | -1.159 | -1.658 | -0.698 | -1.166 | -1.746 | -0.586 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | all | pinball19_norm | -2.058 | -2.668 | -1.470 | -2.060 | -2.757 | -1.363 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -1.842 | -2.532 | -1.156 | -1.821 | -2.589 | -1.052 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -1.428 | -2.063 | -0.792 | -1.405 | -2.123 | -0.688 | 13 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -1.824 | -2.511 | -1.139 | -1.802 | -2.568 | -1.036 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -2.174 | -2.801 | -1.582 | -2.166 | -2.884 | -1.448 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -1.137 | -1.655 | -0.692 | -1.141 | -1.727 | -0.556 | 14 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -2.136 | -2.759 | -1.544 | -2.128 | -2.840 | -1.415 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | crps_norm | -2.141 | -2.799 | -1.568 | -2.210 | -3.003 | -1.418 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.976 | -1.509 | -0.539 | -1.040 | -1.694 | -0.386 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_minimal_requested minus production_cleaned | post_hoc_deciding | solar_minimal_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -2.103 | -2.755 | -1.538 | -2.172 | -2.958 | -1.386 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | crps_norm | -0.776 | -1.219 | -0.296 | -0.757 | -1.299 | -0.216 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | abs_error_norm | -0.508 | -0.935 | -0.064 | -0.493 | -0.999 | 0.013 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | all | pinball19_norm | -0.765 | -1.207 | -0.287 | -0.746 | -1.287 | -0.205 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | crps_norm | -0.123 | -0.664 | 0.421 | -0.135 | -0.779 | 0.508 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | abs_error_norm | -0.094 | -0.616 | 0.444 | -0.105 | -0.738 | 0.528 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 0 to 2 | pinball19_norm | -0.121 | -0.659 | 0.422 | -0.133 | -0.776 | 0.511 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | crps_norm | -0.884 | -1.338 | -0.414 | -0.856 | -1.399 | -0.314 | 13 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | abs_error_norm | -0.575 | -0.990 | -0.145 | -0.553 | -1.048 | -0.057 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 3 to 10 | pinball19_norm | -0.872 | -1.323 | -0.402 | -0.844 | -1.385 | -0.302 | 13 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | crps_norm | -1.109 | -1.621 | -0.576 | -1.127 | -1.740 | -0.513 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | abs_error_norm | -0.725 | -1.197 | -0.237 | -0.744 | -1.315 | -0.173 | 10 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus production_cleaned | post_hoc_deciding | solar_expanded_requested | production_cleaned | PV | days 11 onward | pinball19_norm | -1.093 | -1.602 | -0.561 | -1.110 | -1.722 | -0.499 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | crps_norm | 1.316 | 0.797 | 1.884 | 1.337 | 0.697 | 1.977 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | abs_error_norm | 0.650 | 0.147 | 1.195 | 0.673 | 0.048 | 1.298 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | all | pinball19_norm | 1.294 | 0.774 | 1.859 | 1.314 | 0.675 | 1.954 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | crps_norm | 1.719 | 1.064 | 2.436 | 1.685 | 0.919 | 2.452 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | abs_error_norm | 1.335 | 0.705 | 2.030 | 1.301 | 0.557 | 2.044 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 0 to 2 | pinball19_norm | 1.703 | 1.047 | 2.419 | 1.670 | 0.904 | 2.435 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | crps_norm | 1.290 | 0.791 | 1.834 | 1.310 | 0.683 | 1.937 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | abs_error_norm | 0.563 | 0.076 | 1.103 | 0.589 | -0.017 | 1.194 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 3 to 10 | pinball19_norm | 1.264 | 0.763 | 1.806 | 1.284 | 0.657 | 1.910 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | crps_norm | 1.031 | 0.566 | 1.541 | 1.083 | 0.477 | 1.689 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | abs_error_norm | 0.251 | -0.264 | 0.778 | 0.296 | -0.337 | 0.929 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| solar_expanded_requested minus solar_minimal_requested | post_hoc_deciding | solar_expanded_requested | solar_minimal_requested | PV | days 11 onward | pinball19_norm | 1.010 | 0.544 | 1.521 | 1.062 | 0.455 | 1.668 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | crps_norm | -3.115 | -4.235 | -2.114 | -3.086 | -4.293 | -1.880 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | abs_error_norm | -1.562 | -2.686 | -0.556 | -1.534 | -2.752 | -0.315 | 11 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | all | pinball19_norm | -3.046 | -4.163 | -2.046 | -3.017 | -4.224 | -1.811 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | crps_norm | -0.860 | -1.734 | 0.032 | -0.895 | -1.883 | 0.093 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | abs_error_norm | -0.647 | -1.552 | 0.255 | -0.686 | -1.706 | 0.334 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 0 to 2 | pinball19_norm | -0.840 | -1.718 | 0.056 | -0.876 | -1.865 | 0.114 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | crps_norm | -3.475 | -4.615 | -2.411 | -3.451 | -4.693 | -2.208 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | abs_error_norm | -1.952 | -3.108 | -0.876 | -1.930 | -3.179 | -0.682 | 12 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 3 to 10 | pinball19_norm | -3.405 | -4.537 | -2.337 | -3.381 | -4.622 | -2.139 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | crps_norm | -4.196 | -5.820 | -2.824 | -4.130 | -5.777 | -2.482 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | abs_error_norm | -1.460 | -3.157 | -0.045 | -1.381 | -3.062 | 0.300 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus production_cleaned | post_hoc_deciding | wind_minimal_requested | production_cleaned | Wind | days 11 onward | pinball19_norm | -4.090 | -5.716 | -2.719 | -4.023 | -5.669 | -2.376 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | crps_norm | 0.517 | 0.296 | 0.754 | 0.519 | 0.265 | 0.774 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | abs_error_norm | 1.004 | 0.766 | 1.251 | 1.007 | 0.736 | 1.277 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | all | pinball19_norm | 0.535 | 0.313 | 0.772 | 0.538 | 0.281 | 0.794 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | crps_norm | 3.649 | 2.919 | 4.360 | 3.623 | 2.819 | 4.427 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | abs_error_norm | 3.337 | 2.578 | 4.068 | 3.309 | 2.486 | 4.132 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 0 to 2 | pinball19_norm | 3.642 | 2.910 | 4.353 | 3.617 | 2.812 | 4.421 | 0 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | crps_norm | 0.252 | -0.033 | 0.546 | 0.244 | -0.085 | 0.574 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | abs_error_norm | 0.611 | 0.326 | 0.887 | 0.603 | 0.283 | 0.923 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 3 to 10 | pinball19_norm | 0.269 | -0.017 | 0.565 | 0.262 | -0.069 | 0.592 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | crps_norm | -1.502 | -1.802 | -1.223 | -1.527 | -1.859 | -1.194 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | abs_error_norm | -0.069 | -0.322 | 0.152 | -0.077 | -0.357 | 0.203 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_requested minus lean_wind | post_hoc_deciding | wind_minimal_requested | lean_wind | Wind | days 11 onward | pinball19_norm | -1.462 | -1.766 | -1.186 | -1.486 | -1.816 | -1.156 | 15 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | crps_norm | 0.018 | -0.002 | 0.037 | 0.018 | -0.003 | 0.040 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | abs_error_norm | 0.001 | -0.017 | 0.018 | 0.001 | -0.019 | 0.021 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.017 | -0.003 | 0.037 | 0.017 | -0.004 | 0.039 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.040 | -0.007 | 0.087 | 0.040 | -0.010 | 0.090 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.045 | -0.005 | 0.097 | 0.045 | -0.012 | 0.102 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.040 | -0.007 | 0.087 | 0.040 | -0.011 | 0.090 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.011 | -0.013 | 0.035 | 0.010 | -0.016 | 0.036 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | -0.012 | -0.033 | 0.006 | -0.012 | -0.034 | 0.009 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.010 | -0.013 | 0.033 | 0.010 | -0.015 | 0.035 | 7 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.016 | -0.004 | 0.036 | 0.016 | -0.006 | 0.037 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | -0.008 | -0.026 | 0.009 | -0.008 | -0.028 | 0.011 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_uv_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_uv_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.013 | -0.006 | 0.033 | 0.013 | -0.008 | 0.035 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | crps_norm | 0.031 | 0.010 | 0.055 | 0.033 | 0.007 | 0.059 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | abs_error_norm | -0.001 | -0.021 | 0.020 | -0.000 | -0.024 | 0.024 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | all | pinball19_norm | 0.030 | 0.009 | 0.052 | 0.031 | 0.005 | 0.057 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | crps_norm | 0.049 | 0.005 | 0.106 | 0.049 | -0.006 | 0.105 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | abs_error_norm | 0.025 | -0.023 | 0.081 | 0.026 | -0.032 | 0.083 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 0 to 2 | pinball19_norm | 0.048 | 0.005 | 0.105 | 0.048 | -0.007 | 0.104 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | crps_norm | 0.028 | -0.000 | 0.055 | 0.028 | -0.003 | 0.060 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | abs_error_norm | -0.007 | -0.035 | 0.019 | -0.007 | -0.038 | 0.024 | 8 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 3 to 10 | pinball19_norm | 0.026 | -0.002 | 0.052 | 0.026 | -0.005 | 0.057 | 6 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | crps_norm | 0.026 | 0.007 | 0.044 | 0.026 | 0.006 | 0.046 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | abs_error_norm | -0.008 | -0.024 | 0.008 | -0.008 | -0.026 | 0.009 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
| wind_minimal_circular_requested minus wind_minimal_requested | post_hoc_deciding | wind_minimal_circular_requested | wind_minimal_requested | Wind | days 11 onward | pinball19_norm | 0.024 | 0.004 | 0.043 | 0.024 | 0.003 | 0.045 | 4 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |
