# Supporting tables: lean inputs and individual additions

These aggregate tables support the [fixed-pipeline study](fixed-pipeline-lean-features.md).
The main lean, first-solar and individual wind-addition comparisons are complete; alternate
settings and control tables are still being added. Negative paired differences favour the recipe
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

## wind_add_categorical_precipitation_type_surface-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 14.141 | 12.772 | 15.351 | 3 | 12 | 3 | 107121 | False |

## wind_add_categorical_precipitation_type_surface-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.320 | 0.082 | 0.604 | 0.331 | 0.015 | 0.647 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_categorical_precipitation_type_surface-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_categorical_precipitation_type_surface | Wind | 13.101 | 12.024 | 14.115 | 3 | 15 | 3 | 135921 | False |

## wind_add_categorical_precipitation_type_surface-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_categorical_precipitation_type_surface minus lean_wind | Wind | 0.225 | -0.072 | 0.572 | 0.224 | -0.015 | 0.462 | 3 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_dew_point_temperature_2m | Wind | 13.799 | 12.236 | 15.205 | 3 | 12 | 3 | 107121 | False |

## wind_add_dew_point_temperature_2m-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | -0.022 | -0.089 | 0.048 | -0.024 | -0.098 | 0.049 | 5 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_dew_point_temperature_2m-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_dew_point_temperature_2m | Wind | 12.883 | 11.732 | 13.964 | 3 | 15 | 3 | 135921 | False |

## wind_add_dew_point_temperature_2m-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_dew_point_temperature_2m minus lean_wind | Wind | 0.007 | -0.067 | 0.086 | 0.006 | -0.047 | 0.060 | 9 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 13.848 | 12.264 | 15.268 | 3 | 12 | 3 | 107121 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | 0.027 | -0.028 | 0.091 | 0.026 | -0.031 | 0.083 | 4 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_downward_short_wave_radiation_flux_surface | Wind | 12.931 | 11.788 | 14.027 | 3 | 15 | 3 | 135921 | False |

## wind_add_downward_short_wave_radiation_flux_surface-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_downward_short_wave_radiation_flux_surface minus lean_wind | Wind | 0.055 | -0.013 | 0.127 | 0.054 | -0.013 | 0.122 | 5 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week | Wind | 14.168 | 12.579 | 15.643 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.347 | 0.200 | 0.497 | 0.347 | 0.183 | 0.510 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week | Wind | 13.164 | 12.017 | 14.230 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week minus lean_wind | Wind | 0.288 | 0.164 | 0.393 | 0.294 | 0.182 | 0.406 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_cos | Wind | 14.069 | 12.494 | 15.513 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_cos-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.248 | 0.114 | 0.386 | 0.249 | 0.095 | 0.403 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_cos-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_cos | Wind | 13.118 | 11.990 | 14.189 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_cos-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_cos minus lean_wind | Wind | 0.242 | 0.131 | 0.347 | 0.244 | 0.144 | 0.343 | 2 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_day_of_week_sin | Wind | 14.139 | 12.588 | 15.567 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_day_of_week_sin-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.318 | 0.179 | 0.461 | 0.318 | 0.167 | 0.470 | 2 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_day_of_week_sin-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_day_of_week_sin | Wind | 13.127 | 11.988 | 14.199 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_day_of_week_sin-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_day_of_week_sin minus lean_wind | Wind | 0.251 | 0.160 | 0.331 | 0.256 | 0.165 | 0.348 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_cos | Wind | 15.778 | 13.915 | 17.484 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_cos-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 1.957 | 1.566 | 2.357 | 1.955 | 1.493 | 2.416 | 0 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_cos-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_cos | Wind | 15.253 | 13.665 | 16.863 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_cos-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_cos minus lean_wind | Wind | 2.377 | 1.566 | 3.211 | 2.378 | 1.449 | 3.308 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_a; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 13.821 | 12.261 | 15.203 | 3 | 12 | 3 | 107121 | False |
| wind_add_local_time_of_year_sin | Wind | 16.238 | 14.205 | 18.143 | 3 | 12 | 3 | 107121 | False |

## wind_add_local_time_of_year_sin-fold_a; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.417 | 1.563 | 3.244 | 2.379 | 1.382 | 3.375 | 1 | 12 | 3 | False | 12 | 12 | 0 | 0 | 0 |

## wind_add_local_time_of_year_sin-fold_b; main; padding equal — absolute fair CRPS (capacity points)

| label | group | Fair CRPS (capacity points) | Bootstrap 95% lower | Bootstrap 95% upper | seeds | months | series | rows_per_seed | provisional |
|---|---|---|---|---|---|---|---|---|---|
| lean_wind | Wind | 12.876 | 11.768 | 13.944 | 3 | 15 | 3 | 135921 | False |
| wind_add_local_time_of_year_sin | Wind | 15.617 | 13.960 | 17.082 | 3 | 15 | 3 | 135921 | False |

## wind_add_local_time_of_year_sin-fold_b; main; padding equal — planned paired fair CRPS (left minus right)

| contrast | group | Equal-series difference (capacity points) | Paired bootstrap 95% lower | Paired bootstrap 95% upper | Equal-month difference (capacity points) | Equal-month t 95% lower | Equal-month t 95% upper | Months with lower loss | months | seeds | provisional | min_supported_months | max_supported_months | series_with_missing_months | bootstrap_draws_dropping_series | max_dropped_series_per_draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wind_add_local_time_of_year_sin minus lean_wind | Wind | 2.741 | 1.960 | 3.491 | 2.752 | 1.901 | 3.604 | 1 | 15 | 3 | False | 15 | 15 | 0 | 0 | 0 |

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
