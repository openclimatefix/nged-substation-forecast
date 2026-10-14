# Supplementary calibration diagnostic tables

> **Supporting evidence.** These ten tables are copied verbatim from the completed aggregate
> publication extract. Scientific and prose reviews are complete; reproduction records are published.
> Human review has not been completed.

Further Part 2 implementation and experiments are on hold while the maintainer revises the plan.
The tables document completed diagnostics. No calibration correction or method for temporal or
spatial coherence was evaluated.

**The tables extend the [main study](calibration-diagnostics-1156.md) with descriptive empirical-CDF,
lead-band and November-era diagnostics.** `production` denotes the production-configuration refit;
`r2_e1_c1_a3w_d3` denotes the historical quick-wins quantile recipe. Both use corrected wind and
radiation inputs. `PV` means solar, `Wind` means wind, and `all` includes the other series types.
Fold B is retrospective. The tables select saved aggregates without recalculating statistics.

**Coverage and CDF-bin shares are percentages of cases; loss, width, bias and spread are
percentages of frozen effective capacity.** `pinball19` is twice mean pinball loss at the nineteen
midpoint probabilities defined in the main study. `coverage80`/`coverage90` refer to the stated
interval labels; `width80`/`width90` give their widths. Bias is prediction mean minus observed
power, so negative bias means underprediction. Spread is the standard deviation of delivered
values. Full-history capacity enters only as the common retrospective scoring denominator.

**The CDF bins describe a deterministic midrank empirical CDF, without a uniform-reference
assumption.** At observation `y`, CDF is `(count(values < y) + 0.5 * count(values == y)) / n`.
Bin index is `floor(CDF * 10)`, clipped to 0 through 9; bin 9 includes CDF equal to 1. Bins
include their lower endpoint. For continuous independent identically distributed samples and an
independent outcome from the same distribution, the no-tie reference has equal mass at the
discrete points `k/n`, for `k = 0, …, n`, rather than a continuous uniform distribution. Neither
recipe is asserted to satisfy those assumptions, and the structured quantile pool has no such
reference guarantee. Histogram shape cannot identify the cause of miscalibration.

**Support counts describe overlapping forecast cases rather than distinct observed hours.** A
case pairs a series, forecast issuance and valid time; weather members do not add cases. Series
receive equal weight after their supported cases are averaged, then the three fitting seeds
receive equal weight. “Seed spread” is the maximum minus minimum of the three seed-specific
aggregate values, in percentage points; the spread is not a confidence interval. Mean coverage
has no reported confidence interval. “Supported months” counts represented calendar months in
the table's scope and does not guarantee complete month support for every series.

**Lead and era summaries remain descriptive.** Lead bands are below 72 hours, 72 to below
264 hours and 264 hours onward. November source eras use weather initialisation time around
13 November 2024 at 00:00 UTC, the verified archive categorical-precipitation-type availability
boundary; differing
seasons, target support and training exposure confound era comparisons. The archive grouping
does not evaluate the operational upgrade cycle at 12 November 2024 at 06:00 UTC. That exact
operational-cycle comparison has not been evaluated, and the separate comparison around
12 May 2026 at 06:00 UTC remains pending. No table identifies a causal upgrade effect, supplies coherent
joint scenarios or evaluates a calibrator. The [main report's public inventory](calibration-diagnostics-1156.md#data-and-code-availability)
links the published selection configurations and reproduction records.

## Fold A, main: delivered empirical-CDF bin shares (% of cases)

| Recipe | Series type | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|
| production | all | cdf_bin_00 | 40.801 | 0.204 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_01 | 1.759 | 0.059 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_02 | 1.382 | 0.049 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_03 | 1.236 | 0.047 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_04 | 1.208 | 0.020 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_05 | 1.229 | 0.035 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_06 | 1.311 | 0.044 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_07 | 1.491 | 0.047 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_08 | 2.037 | 0.083 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_09 | 47.545 | 0.196 | 3 | 31 | 12 | 1070930 |
| production | PV | cdf_bin_00 | 40.744 | 0.682 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_01 | 3.037 | 0.269 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_02 | 2.377 | 0.207 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_03 | 2.056 | 0.256 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_04 | 2.041 | 0.117 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_05 | 2.055 | 0.158 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_06 | 2.061 | 0.102 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_07 | 2.245 | 0.158 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_08 | 3.077 | 0.302 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_09 | 40.306 | 0.745 | 3 | 6 | 12 | 175795 |
| production | Wind | cdf_bin_00 | 49.616 | 0.185 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_01 | 1.924 | 0.082 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_02 | 1.591 | 0.093 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_03 | 1.580 | 0.050 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_04 | 1.550 | 0.121 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_05 | 1.658 | 0.075 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_06 | 1.866 | 0.061 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_07 | 2.227 | 0.035 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_08 | 2.958 | 0.224 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_09 | 35.029 | 0.394 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_00 | 11.992 | 0.060 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_01 | 8.942 | 0.094 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_02 | 7.699 | 0.138 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_03 | 7.475 | 0.065 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_04 | 8.737 | 0.193 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_05 | 13.573 | 0.212 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_06 | 7.600 | 0.093 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_07 | 8.291 | 0.079 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_08 | 9.300 | 0.075 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_09 | 16.391 | 0.065 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_00 | 8.768 | 0.096 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_01 | 6.136 | 0.070 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_02 | 5.344 | 0.043 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_03 | 5.056 | 0.022 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_04 | 12.502 | 0.864 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_05 | 37.334 | 0.748 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_06 | 5.135 | 0.146 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_07 | 5.313 | 0.035 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_08 | 5.741 | 0.074 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_09 | 8.671 | 0.070 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_00 | 11.854 | 0.028 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_01 | 9.521 | 0.044 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_02 | 8.634 | 0.033 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_03 | 8.480 | 0.054 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_04 | 8.597 | 0.032 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_05 | 9.138 | 0.031 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_06 | 9.493 | 0.040 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_07 | 9.419 | 0.051 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_08 | 9.589 | 0.035 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_09 | 15.274 | 0.029 | 3 | 3 | 12 | 107121 |

## Fold A, main: delivered diagnostics by lead band (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Lead band | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| production | all | days 0 to 2 | pinball19 | 10.870 | 0.050 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | bias | -1.871 | 0.085 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | coverage80 | 9.755 | 0.091 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | coverage90 | 12.796 | 0.198 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | width80 | 3.593 | 0.067 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | width90 | 4.745 | 0.094 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | spread | 1.483 | 0.027 | 3 | 31 | 12 | 217328 |
| production | all | days 3 to 10 | pinball19 | 11.491 | 0.065 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | bias | -2.128 | 0.101 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | coverage80 | 12.035 | 0.381 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | coverage90 | 15.727 | 0.531 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | width80 | 4.245 | 0.087 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | width90 | 5.559 | 0.121 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | spread | 1.741 | 0.037 | 3 | 31 | 12 | 586691 |
| production | all | days 11 onward | pinball19 | 11.955 | 0.048 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | bias | -1.999 | 0.106 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | coverage80 | 12.301 | 0.591 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | coverage90 | 16.361 | 0.680 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | width80 | 4.065 | 0.060 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | width90 | 5.338 | 0.085 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | spread | 1.661 | 0.026 | 3 | 31 | 12 | 266911 |
| production | PV | days 0 to 2 | pinball19 | 7.812 | 0.027 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | bias | 0.145 | 0.072 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | coverage80 | 14.040 | 0.714 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | coverage90 | 18.570 | 1.254 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | width80 | 3.495 | 0.016 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | width90 | 4.614 | 0.024 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | spread | 1.436 | 0.007 | 3 | 6 | 12 | 33153 |
| production | PV | days 3 to 10 | pinball19 | 7.920 | 0.032 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | bias | 0.942 | 0.083 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | coverage80 | 19.724 | 1.567 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | coverage90 | 25.878 | 1.926 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | width80 | 4.273 | 0.118 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | width90 | 5.549 | 0.153 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | spread | 1.737 | 0.049 | 3 | 6 | 12 | 96597 |
| production | PV | days 11 onward | pinball19 | 8.313 | 0.013 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | bias | 0.936 | 0.015 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | coverage80 | 20.863 | 2.204 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | coverage90 | 28.169 | 2.563 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | width80 | 4.258 | 0.149 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | width90 | 5.576 | 0.203 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | spread | 1.745 | 0.062 | 3 | 6 | 12 | 46045 |
| production | Wind | days 0 to 2 | pinball19 | 12.867 | 0.094 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | bias | -1.610 | 0.410 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | coverage80 | 23.309 | 0.462 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | coverage90 | 29.765 | 1.013 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | width80 | 11.392 | 0.238 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | width90 | 15.019 | 0.367 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | spread | 4.704 | 0.107 | 3 | 3 | 12 | 22098 |
| production | Wind | days 3 to 10 | pinball19 | 17.448 | 0.093 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | bias | -2.285 | 0.135 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | coverage80 | 15.152 | 0.106 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | coverage90 | 19.254 | 0.159 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | width80 | 10.478 | 0.225 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | width90 | 13.802 | 0.324 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | spread | 4.356 | 0.094 | 3 | 3 | 12 | 58642 |
| production | Wind | days 11 onward | pinball19 | 20.305 | 0.073 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | bias | -1.575 | 0.221 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | coverage80 | 9.154 | 0.340 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | coverage90 | 12.200 | 0.575 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | width80 | 7.190 | 0.135 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | width90 | 9.466 | 0.133 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | spread | 2.947 | 0.055 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | pinball19 | 6.842 | 0.013 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | bias | -0.631 | 0.025 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage80 | 72.556 | 0.178 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage90 | 75.852 | 0.213 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width80 | 24.477 | 0.288 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width90 | 26.725 | 0.306 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | spread | 8.229 | 0.063 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | pinball19 | 7.752 | 0.006 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | bias | -0.883 | 0.016 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage80 | 71.282 | 0.128 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage90 | 75.377 | 0.154 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width80 | 26.446 | 0.218 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width90 | 29.209 | 0.247 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | spread | 9.021 | 0.054 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | pinball19 | 8.518 | 0.018 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | bias | -0.712 | 0.036 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage80 | 71.562 | 0.026 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage90 | 75.656 | 0.019 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width80 | 29.075 | 0.107 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width90 | 31.974 | 0.104 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | spread | 9.892 | 0.028 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | pinball19 | 4.869 | 0.016 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | bias | -0.500 | 0.002 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage80 | 82.397 | 0.073 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage90 | 86.340 | 0.053 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width80 | 19.205 | 0.095 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width90 | 21.586 | 0.087 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | spread | 6.791 | 0.036 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | pinball19 | 5.356 | 0.009 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | bias | 0.025 | 0.014 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage80 | 82.273 | 0.045 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage90 | 86.897 | 0.094 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width80 | 20.342 | 0.009 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width90 | 22.829 | 0.055 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | spread | 7.292 | 0.015 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | pinball19 | 5.767 | 0.012 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | bias | -0.008 | 0.019 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage80 | 83.278 | 0.130 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage90 | 87.671 | 0.228 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width80 | 21.963 | 0.085 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width90 | 24.552 | 0.091 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | spread | 7.872 | 0.018 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | pinball19 | 7.004 | 0.014 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | bias | -0.299 | 0.053 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage80 | 78.708 | 0.164 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage90 | 87.282 | 0.203 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width80 | 32.281 | 0.080 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width90 | 39.930 | 0.071 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | spread | 12.390 | 0.030 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | pinball19 | 11.936 | 0.002 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | bias | -4.233 | 0.009 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage80 | 72.110 | 0.038 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage90 | 80.462 | 0.029 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width80 | 43.978 | 0.015 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width90 | 52.387 | 0.028 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | spread | 16.660 | 0.003 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | pinball19 | 13.876 | 0.002 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | bias | -5.610 | 0.009 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage80 | 69.688 | 0.042 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage90 | 76.960 | 0.102 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width80 | 45.984 | 0.049 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width90 | 53.518 | 0.020 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | spread | 16.997 | 0.006 | 3 | 3 | 12 | 26381 |

## Fold A, sensitivity: delivered empirical-CDF bin shares (% of cases)

| Recipe | Series type | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|
| production | all | cdf_bin_00 | 40.923 | 0.178 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_01 | 1.802 | 0.056 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_02 | 1.426 | 0.029 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_03 | 1.311 | 0.047 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_04 | 1.271 | 0.049 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_05 | 1.299 | 0.059 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_06 | 1.385 | 0.047 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_07 | 1.576 | 0.030 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_08 | 2.092 | 0.036 | 3 | 31 | 12 | 1070930 |
| production | all | cdf_bin_09 | 46.915 | 0.399 | 3 | 31 | 12 | 1070930 |
| production | PV | cdf_bin_00 | 42.107 | 0.601 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_01 | 3.117 | 0.115 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_02 | 2.340 | 0.101 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_03 | 2.125 | 0.190 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_04 | 2.045 | 0.198 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_05 | 2.023 | 0.258 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_06 | 2.000 | 0.139 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_07 | 2.100 | 0.028 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_08 | 2.791 | 0.063 | 3 | 6 | 12 | 175795 |
| production | PV | cdf_bin_09 | 39.353 | 0.656 | 3 | 6 | 12 | 175795 |
| production | Wind | cdf_bin_00 | 49.183 | 0.120 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_01 | 1.849 | 0.076 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_02 | 1.591 | 0.049 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_03 | 1.632 | 0.088 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_04 | 1.588 | 0.034 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_05 | 1.747 | 0.035 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_06 | 1.990 | 0.030 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_07 | 2.364 | 0.037 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_08 | 2.963 | 0.060 | 3 | 3 | 12 | 107121 |
| production | Wind | cdf_bin_09 | 35.093 | 0.113 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_00 | 11.479 | 0.056 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_01 | 8.551 | 0.027 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_02 | 7.655 | 0.033 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_03 | 7.604 | 0.051 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_04 | 9.825 | 0.087 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_05 | 13.201 | 0.232 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_06 | 8.309 | 0.264 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_07 | 8.533 | 0.031 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_08 | 9.723 | 0.063 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_09 | 15.121 | 0.064 | 3 | 31 | 12 | 1070930 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_00 | 7.600 | 0.035 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_01 | 6.108 | 0.083 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_02 | 5.551 | 0.062 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_03 | 5.453 | 0.067 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_04 | 16.684 | 0.057 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_05 | 33.247 | 1.002 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_06 | 6.409 | 0.998 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_07 | 5.629 | 0.024 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_08 | 5.868 | 0.018 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_09 | 7.451 | 0.049 | 3 | 6 | 12 | 175795 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_00 | 11.202 | 0.023 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_01 | 9.192 | 0.051 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_02 | 8.640 | 0.035 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_03 | 8.522 | 0.022 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_04 | 8.940 | 0.046 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_05 | 9.271 | 0.020 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_06 | 9.797 | 0.032 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_07 | 9.776 | 0.033 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_08 | 9.977 | 0.039 | 3 | 3 | 12 | 107121 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_09 | 14.681 | 0.022 | 3 | 3 | 12 | 107121 |

## Fold A, sensitivity: delivered diagnostics by lead band (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Lead band | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| production | all | days 0 to 2 | pinball19 | 10.639 | 0.060 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | bias | -1.788 | 0.093 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | coverage80 | 9.932 | 0.274 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | coverage90 | 12.980 | 0.369 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | width80 | 3.832 | 0.024 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | width90 | 5.050 | 0.029 | 3 | 31 | 12 | 217328 |
| production | all | days 0 to 2 | spread | 1.576 | 0.010 | 3 | 31 | 12 | 217328 |
| production | all | days 3 to 10 | pinball19 | 11.275 | 0.053 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | bias | -1.998 | 0.105 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | coverage80 | 12.555 | 0.333 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | coverage90 | 16.262 | 0.399 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | width80 | 4.630 | 0.038 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | width90 | 6.017 | 0.043 | 3 | 31 | 12 | 586691 |
| production | all | days 3 to 10 | spread | 1.886 | 0.013 | 3 | 31 | 12 | 586691 |
| production | all | days 11 onward | pinball19 | 11.785 | 0.044 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | bias | -1.921 | 0.079 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | coverage80 | 13.037 | 0.272 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | coverage90 | 17.117 | 0.307 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | width80 | 4.539 | 0.043 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | width90 | 5.911 | 0.045 | 3 | 31 | 12 | 266911 |
| production | all | days 11 onward | spread | 1.840 | 0.013 | 3 | 31 | 12 | 266911 |
| production | PV | days 0 to 2 | pinball19 | 7.568 | 0.038 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | bias | -0.095 | 0.046 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | coverage80 | 12.706 | 0.359 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | coverage90 | 16.770 | 0.598 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | width80 | 3.792 | 0.033 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | width90 | 4.994 | 0.043 | 3 | 6 | 12 | 33153 |
| production | PV | days 0 to 2 | spread | 1.551 | 0.013 | 3 | 6 | 12 | 33153 |
| production | PV | days 3 to 10 | pinball19 | 7.746 | 0.048 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | bias | 0.866 | 0.053 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | coverage80 | 19.252 | 1.255 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | coverage90 | 25.045 | 1.101 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | width80 | 4.840 | 0.068 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | width90 | 6.232 | 0.076 | 3 | 6 | 12 | 96597 |
| production | PV | days 3 to 10 | spread | 1.953 | 0.024 | 3 | 6 | 12 | 96597 |
| production | PV | days 11 onward | pinball19 | 8.154 | 0.024 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | bias | 0.936 | 0.044 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | coverage80 | 21.248 | 0.900 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | coverage90 | 28.038 | 1.004 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | width80 | 5.032 | 0.087 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | width90 | 6.493 | 0.086 | 3 | 6 | 12 | 46045 |
| production | PV | days 11 onward | spread | 2.041 | 0.030 | 3 | 6 | 12 | 46045 |
| production | Wind | days 0 to 2 | pinball19 | 12.488 | 0.070 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | bias | -1.851 | 0.196 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | coverage80 | 25.544 | 0.103 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | coverage90 | 32.498 | 0.285 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | width80 | 11.999 | 0.171 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | width90 | 15.795 | 0.214 | 3 | 3 | 12 | 22098 |
| production | Wind | days 0 to 2 | spread | 4.925 | 0.074 | 3 | 3 | 12 | 22098 |
| production | Wind | days 3 to 10 | pinball19 | 17.163 | 0.034 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | bias | -2.511 | 0.118 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | coverage80 | 15.393 | 0.082 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | coverage90 | 19.364 | 0.040 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | width80 | 10.310 | 0.018 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | width90 | 13.461 | 0.010 | 3 | 3 | 12 | 58642 |
| production | Wind | days 3 to 10 | spread | 4.254 | 0.003 | 3 | 3 | 12 | 58642 |
| production | Wind | days 11 onward | pinball19 | 19.865 | 0.092 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | bias | -2.173 | 0.134 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | coverage80 | 8.246 | 0.144 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | coverage90 | 10.842 | 0.083 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | width80 | 6.601 | 0.052 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | width90 | 8.585 | 0.059 | 3 | 3 | 12 | 26381 |
| production | Wind | days 11 onward | spread | 2.679 | 0.015 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | pinball19 | 6.814 | 0.005 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | bias | -0.581 | 0.017 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage80 | 74.351 | 0.055 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage90 | 77.302 | 0.075 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width80 | 26.242 | 0.002 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width90 | 28.371 | 0.007 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | spread | 8.787 | 0.006 | 3 | 31 | 12 | 217328 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | pinball19 | 7.720 | 0.004 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | bias | -0.873 | 0.011 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage80 | 73.170 | 0.103 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage90 | 77.020 | 0.075 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width80 | 28.157 | 0.025 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width90 | 30.916 | 0.025 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | spread | 9.573 | 0.009 | 3 | 31 | 12 | 586691 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | pinball19 | 8.686 | 0.018 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | bias | -0.528 | 0.021 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage80 | 73.093 | 0.221 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage90 | 77.107 | 0.187 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width80 | 30.903 | 0.103 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width90 | 33.897 | 0.113 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | spread | 10.448 | 0.029 | 3 | 31 | 12 | 266911 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | pinball19 | 4.688 | 0.002 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | bias | -0.337 | 0.020 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage80 | 84.072 | 0.044 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage90 | 87.952 | 0.108 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width80 | 20.232 | 0.008 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width90 | 22.683 | 0.015 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | spread | 7.173 | 0.012 | 3 | 6 | 12 | 33153 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | pinball19 | 5.160 | 0.006 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | bias | 0.005 | 0.014 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage80 | 84.880 | 0.039 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage90 | 89.404 | 0.056 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width80 | 21.756 | 0.025 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width90 | 24.500 | 0.029 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | spread | 7.877 | 0.015 | 3 | 6 | 12 | 96597 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | pinball19 | 5.600 | 0.001 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | bias | -0.108 | 0.010 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage80 | 85.722 | 0.046 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage90 | 90.282 | 0.021 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width80 | 23.494 | 0.028 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width90 | 26.498 | 0.020 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | spread | 8.573 | 0.012 | 3 | 6 | 12 | 46045 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | pinball19 | 6.932 | 0.006 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | bias | -0.454 | 0.014 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage80 | 80.057 | 0.113 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage90 | 88.371 | 0.036 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width80 | 33.466 | 0.002 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width90 | 41.204 | 0.009 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | spread | 12.855 | 0.006 | 3 | 3 | 12 | 22098 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | pinball19 | 11.935 | 0.001 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | bias | -4.416 | 0.002 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage80 | 73.272 | 0.051 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage90 | 81.280 | 0.056 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width80 | 44.595 | 0.027 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width90 | 53.019 | 0.014 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | spread | 16.917 | 0.001 | 3 | 3 | 12 | 58642 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | pinball19 | 13.823 | 0.000 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | bias | -5.747 | 0.008 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage80 | 71.030 | 0.049 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage90 | 78.054 | 0.015 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width80 | 46.955 | 0.083 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width90 | 54.346 | 0.037 | 3 | 3 | 12 | 26381 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | spread | 17.304 | 0.012 | 3 | 3 | 12 | 26381 |

## Fold B (retrospective), main: delivered empirical-CDF bin shares (% of cases)

| Recipe | Series type | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|
| r2_e1_c1_a3w_d3 | all | cdf_bin_00 | 17.744 | 0.005 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_01 | 9.068 | 0.087 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_02 | 7.918 | 0.108 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_03 | 7.658 | 0.069 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_04 | 9.033 | 0.230 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_05 | 13.716 | 0.242 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_06 | 7.456 | 0.074 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_07 | 7.503 | 0.046 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_08 | 8.071 | 0.049 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_09 | 11.832 | 0.007 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_00 | 10.385 | 0.028 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_01 | 5.606 | 0.045 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_02 | 5.021 | 0.101 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_03 | 4.730 | 0.127 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_04 | 12.131 | 1.217 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_05 | 36.953 | 1.323 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_06 | 4.842 | 0.234 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_07 | 4.988 | 0.100 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_08 | 6.169 | 0.085 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_09 | 9.173 | 0.029 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_00 | 17.320 | 0.102 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_01 | 9.470 | 0.058 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_02 | 9.597 | 0.048 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_03 | 10.005 | 0.010 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_04 | 9.515 | 0.015 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_05 | 9.063 | 0.029 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_06 | 8.705 | 0.058 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_07 | 8.082 | 0.042 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_08 | 7.631 | 0.058 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_09 | 10.611 | 0.040 | 3 | 3 | 15 | 135921 |
| production | all | cdf_bin_00 | 51.211 | 0.113 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_01 | 1.742 | 0.014 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_02 | 1.330 | 0.027 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_03 | 1.191 | 0.010 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_04 | 1.136 | 0.007 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_05 | 1.133 | 0.007 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_06 | 1.220 | 0.011 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_07 | 1.403 | 0.030 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_08 | 1.838 | 0.051 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_09 | 37.797 | 0.199 | 3 | 31 | 15 | 1326430 |
| production | PV | cdf_bin_00 | 48.593 | 1.475 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_01 | 2.994 | 0.151 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_02 | 2.220 | 0.127 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_03 | 1.961 | 0.038 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_04 | 1.811 | 0.086 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_05 | 1.775 | 0.038 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_06 | 1.828 | 0.041 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_07 | 2.005 | 0.235 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_08 | 2.507 | 0.302 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_09 | 34.307 | 1.769 | 3 | 6 | 15 | 253003 |
| production | Wind | cdf_bin_00 | 56.937 | 1.388 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_01 | 2.173 | 0.154 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_02 | 1.714 | 0.021 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_03 | 1.608 | 0.084 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_04 | 1.533 | 0.116 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_05 | 1.524 | 0.102 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_06 | 1.708 | 0.151 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_07 | 2.015 | 0.120 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_08 | 2.564 | 0.151 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_09 | 28.223 | 0.915 | 3 | 3 | 15 | 135921 |

## Fold B (retrospective), main: delivered diagnostics by lead band (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Lead band | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | pinball19 | 7.623 | 0.018 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | bias | 1.618 | 0.021 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage80 | 71.683 | 0.067 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage90 | 74.782 | 0.053 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width80 | 23.285 | 0.053 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width90 | 25.399 | 0.040 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | spread | 7.815 | 0.013 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | pinball19 | 8.870 | 0.013 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | bias | 1.910 | 0.018 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage80 | 69.865 | 0.019 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage90 | 74.068 | 0.025 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width80 | 26.473 | 0.038 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width90 | 29.436 | 0.056 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | spread | 9.124 | 0.019 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | pinball19 | 9.413 | 0.018 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | bias | 2.460 | 0.023 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage80 | 70.548 | 0.035 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage90 | 74.747 | 0.025 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width80 | 28.630 | 0.020 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width90 | 31.778 | 0.053 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | spread | 9.837 | 0.011 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | pinball19 | 4.469 | 0.009 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | bias | 0.159 | 0.018 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage80 | 82.889 | 0.059 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage90 | 86.425 | 0.097 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width80 | 16.134 | 0.022 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width90 | 18.453 | 0.030 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | spread | 5.658 | 0.011 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | pinball19 | 6.250 | 0.004 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | bias | -0.065 | 0.004 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage80 | 78.991 | 0.058 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage90 | 83.926 | 0.051 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width80 | 20.496 | 0.024 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width90 | 23.489 | 0.016 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | spread | 7.364 | 0.010 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | pinball19 | 6.023 | 0.005 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | bias | -0.759 | 0.007 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage80 | 81.248 | 0.103 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage90 | 85.713 | 0.054 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width80 | 20.098 | 0.029 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width90 | 22.983 | 0.021 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | spread | 7.260 | 0.013 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | pinball19 | 6.175 | 0.003 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | bias | 1.254 | 0.010 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage80 | 75.218 | 0.043 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage90 | 84.106 | 0.139 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width80 | 28.041 | 0.016 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width90 | 34.667 | 0.025 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | spread | 10.834 | 0.005 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | pinball19 | 10.376 | 0.007 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | bias | 0.413 | 0.026 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage80 | 71.663 | 0.077 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage90 | 80.047 | 0.139 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width80 | 44.020 | 0.089 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width90 | 53.211 | 0.076 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | spread | 16.903 | 0.026 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | pinball19 | 12.119 | 0.006 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | bias | 1.134 | 0.016 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage80 | 70.347 | 0.071 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage90 | 78.010 | 0.199 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width80 | 49.982 | 0.119 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width90 | 59.229 | 0.026 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | spread | 18.780 | 0.020 | 3 | 3 | 15 | 33597 |
| production | all | days 0 to 2 | pinball19 | 11.358 | 0.047 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | bias | 2.541 | 0.034 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | coverage80 | 8.545 | 0.061 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | coverage90 | 11.191 | 0.079 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | width80 | 3.260 | 0.037 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | width90 | 4.308 | 0.045 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | spread | 1.351 | 0.014 | 3 | 31 | 15 | 276283 |
| production | all | days 3 to 10 | pinball19 | 12.324 | 0.037 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | bias | 2.579 | 0.025 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | coverage80 | 11.379 | 0.128 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | coverage90 | 14.921 | 0.149 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | width80 | 4.298 | 0.024 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | width90 | 5.681 | 0.030 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | spread | 1.769 | 0.009 | 3 | 31 | 15 | 718201 |
| production | all | days 11 onward | pinball19 | 12.563 | 0.036 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | bias | 2.598 | 0.044 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | coverage80 | 12.195 | 0.170 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | coverage90 | 16.012 | 0.311 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | width80 | 4.176 | 0.027 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | width90 | 5.540 | 0.037 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | spread | 1.703 | 0.011 | 3 | 31 | 15 | 331946 |
| production | PV | days 0 to 2 | pinball19 | 7.349 | 0.076 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | bias | 0.883 | 0.018 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | coverage80 | 11.688 | 0.361 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | coverage90 | 15.481 | 0.331 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | width80 | 2.884 | 0.008 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | width90 | 3.789 | 0.011 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | spread | 1.193 | 0.004 | 3 | 6 | 15 | 55505 |
| production | PV | days 3 to 10 | pinball19 | 9.198 | 0.115 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | bias | 0.793 | 0.100 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | coverage80 | 17.581 | 0.501 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | coverage90 | 23.199 | 0.690 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | width80 | 4.248 | 0.139 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | width90 | 5.573 | 0.182 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | spread | 1.743 | 0.063 | 3 | 6 | 15 | 130890 |
| production | PV | days 11 onward | pinball19 | 8.910 | 0.161 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | bias | 0.314 | 0.232 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | coverage80 | 20.664 | 0.767 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | coverage90 | 26.931 | 1.395 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | width80 | 4.286 | 0.175 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | width90 | 5.684 | 0.199 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | spread | 1.754 | 0.069 | 3 | 6 | 15 | 66608 |
| production | Wind | days 0 to 2 | pinball19 | 13.065 | 0.395 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | bias | 4.367 | 0.699 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | coverage80 | 17.906 | 0.769 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | coverage90 | 22.917 | 1.065 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | width80 | 9.412 | 0.232 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | width90 | 12.484 | 0.267 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | spread | 3.920 | 0.084 | 3 | 3 | 15 | 27972 |
| production | Wind | days 3 to 10 | pinball19 | 17.112 | 0.269 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | bias | 3.464 | 0.645 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | coverage80 | 15.042 | 0.511 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | coverage90 | 19.497 | 0.866 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | width80 | 10.402 | 0.098 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | width90 | 13.866 | 0.147 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | spread | 4.327 | 0.048 | 3 | 3 | 15 | 74352 |
| production | Wind | days 11 onward | pinball19 | 18.501 | 0.271 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | bias | 2.558 | 0.675 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | coverage80 | 11.840 | 0.333 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | coverage90 | 15.633 | 0.518 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | width80 | 7.960 | 0.158 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | width90 | 10.603 | 0.188 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | spread | 3.257 | 0.051 | 3 | 3 | 15 | 33597 |

## Fold B (retrospective), main: delivered descriptive diagnostics by November source era (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Source era | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | pinball19 | 8.415 | 0.012 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | bias | 2.560 | 0.016 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | coverage80 | 72.717 | 0.012 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | coverage90 | 76.842 | 0.021 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | width80 | 26.927 | 0.024 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | width90 | 29.765 | 0.060 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | spread | 9.210 | 0.019 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | pinball19 | 7.592 | 0.007 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | bias | -0.442 | 0.005 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | coverage80 | 70.141 | 0.024 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | coverage90 | 74.113 | 0.053 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | width80 | 25.072 | 0.016 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | width90 | 27.975 | 0.010 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | spread | 8.671 | 0.002 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | pinball19 | 4.977 | 0.009 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | bias | -1.440 | 0.010 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | coverage80 | 82.315 | 0.056 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | coverage90 | 86.700 | 0.020 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | width80 | 17.167 | 0.019 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | width90 | 19.610 | 0.016 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | spread | 6.146 | 0.012 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | pinball19 | 6.582 | 0.003 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | bias | 0.990 | 0.015 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | coverage80 | 78.657 | 0.033 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | coverage90 | 83.275 | 0.111 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | width80 | 21.587 | 0.018 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | width90 | 24.759 | 0.052 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | spread | 7.738 | 0.002 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | pinball19 | 10.431 | 0.006 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | bias | 0.525 | 0.024 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | coverage80 | 73.314 | 0.096 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | coverage90 | 81.510 | 0.175 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | width80 | 44.868 | 0.090 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | width90 | 53.602 | 0.045 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | spread | 16.946 | 0.020 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | pinball19 | 9.475 | 0.005 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | bias | 0.993 | 0.014 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | coverage80 | 70.877 | 0.072 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | coverage90 | 79.295 | 0.108 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | width80 | 39.658 | 0.070 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | width90 | 48.281 | 0.059 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | spread | 15.326 | 0.018 | 3 | 3 | 8 | 69464 |
| production | all | after November 2024 boundary | pinball19 | 11.764 | 0.021 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | bias | 3.341 | 0.042 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | coverage80 | 11.378 | 0.219 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | coverage90 | 14.857 | 0.237 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | width80 | 3.940 | 0.047 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | width90 | 5.223 | 0.064 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | spread | 1.626 | 0.020 | 3 | 31 | 8 | 663954 |
| production | all | before November 2024 boundary | pinball19 | 10.862 | 0.067 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | bias | -0.375 | 0.106 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | coverage80 | 11.140 | 0.207 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | coverage90 | 14.679 | 0.204 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | width80 | 4.238 | 0.051 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | width90 | 5.595 | 0.065 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | spread | 1.735 | 0.020 | 3 | 29 | 8 | 662476 |
| production | PV | after November 2024 boundary | pinball19 | 7.429 | 0.134 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | bias | -1.036 | 0.131 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | coverage80 | 19.515 | 1.552 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | coverage90 | 25.330 | 1.973 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | width80 | 3.501 | 0.081 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | width90 | 4.603 | 0.092 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | spread | 1.438 | 0.031 | 3 | 6 | 8 | 123295 |
| production | PV | before November 2024 boundary | pinball19 | 9.941 | 0.115 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | bias | 2.332 | 0.099 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | coverage80 | 14.803 | 0.800 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | coverage90 | 19.785 | 0.900 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | width80 | 4.394 | 0.158 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | width90 | 5.789 | 0.203 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | spread | 1.803 | 0.072 | 3 | 6 | 8 | 129708 |
| production | Wind | after November 2024 boundary | pinball19 | 17.579 | 0.098 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | bias | 2.613 | 0.019 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | coverage80 | 14.578 | 0.253 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | coverage90 | 18.917 | 0.044 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | width80 | 9.585 | 0.043 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | width90 | 12.834 | 0.038 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | spread | 4.014 | 0.020 | 3 | 3 | 8 | 66457 |
| production | Wind | before November 2024 boundary | pinball19 | 15.707 | 0.641 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | bias | 4.204 | 1.317 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | coverage80 | 15.090 | 1.169 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | coverage90 | 19.560 | 1.513 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | width80 | 9.603 | 0.139 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | width90 | 12.718 | 0.209 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | spread | 3.945 | 0.073 | 3 | 3 | 8 | 69464 |

## Fold B (retrospective), sensitivity: delivered empirical-CDF bin shares (% of cases)

| Recipe | Series type | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|
| production | all | cdf_bin_00 | 50.585 | 0.039 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_01 | 1.739 | 0.016 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_02 | 1.368 | 0.036 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_03 | 1.225 | 0.030 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_04 | 1.197 | 0.038 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_05 | 1.250 | 0.041 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_06 | 1.339 | 0.002 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_07 | 1.506 | 0.009 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_08 | 1.957 | 0.019 | 3 | 31 | 15 | 1326430 |
| production | all | cdf_bin_09 | 37.834 | 0.103 | 3 | 31 | 15 | 1326430 |
| production | PV | cdf_bin_00 | 47.382 | 1.206 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_01 | 3.008 | 0.109 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_02 | 2.350 | 0.113 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_03 | 2.035 | 0.149 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_04 | 1.933 | 0.172 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_05 | 2.031 | 0.139 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_06 | 2.095 | 0.041 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_07 | 2.102 | 0.019 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_08 | 2.544 | 0.164 | 3 | 6 | 15 | 253003 |
| production | PV | cdf_bin_09 | 34.520 | 0.928 | 3 | 6 | 15 | 253003 |
| production | Wind | cdf_bin_00 | 58.809 | 0.533 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_01 | 1.883 | 0.050 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_02 | 1.493 | 0.029 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_03 | 1.379 | 0.046 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_04 | 1.367 | 0.018 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_05 | 1.476 | 0.011 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_06 | 1.631 | 0.039 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_07 | 1.913 | 0.034 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_08 | 2.453 | 0.077 | 3 | 3 | 15 | 135921 |
| production | Wind | cdf_bin_09 | 27.596 | 0.491 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_00 | 16.152 | 0.087 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_01 | 9.011 | 0.096 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_02 | 8.485 | 0.032 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_03 | 8.421 | 0.040 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_04 | 9.668 | 0.135 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_05 | 13.817 | 0.101 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_06 | 7.728 | 0.062 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_07 | 7.710 | 0.029 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_08 | 8.078 | 0.003 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | all | cdf_bin_09 | 10.931 | 0.010 | 3 | 31 | 15 | 1326430 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_00 | 9.119 | 0.031 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_01 | 5.942 | 0.128 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_02 | 5.156 | 0.196 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_03 | 5.866 | 0.217 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_04 | 12.881 | 0.639 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_05 | 36.239 | 0.406 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_06 | 5.439 | 0.143 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_07 | 5.532 | 0.044 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_08 | 5.725 | 0.036 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | PV | cdf_bin_09 | 8.101 | 0.016 | 3 | 6 | 15 | 253003 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_00 | 16.872 | 0.041 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_01 | 9.445 | 0.024 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_02 | 9.524 | 0.004 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_03 | 10.026 | 0.053 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_04 | 9.744 | 0.063 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_05 | 9.219 | 0.039 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_06 | 9.013 | 0.010 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_07 | 8.369 | 0.004 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_08 | 7.700 | 0.016 | 3 | 3 | 15 | 135921 |
| r2_e1_c1_a3w_d3 | Wind | cdf_bin_09 | 10.089 | 0.014 | 3 | 3 | 15 | 135921 |

## Fold B (retrospective), sensitivity: delivered diagnostics by lead band (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Lead band | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| production | all | days 0 to 2 | pinball19 | 11.191 | 0.039 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | bias | 2.402 | 0.055 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | coverage80 | 8.852 | 0.019 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | coverage90 | 11.539 | 0.073 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | width80 | 3.402 | 0.039 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | width90 | 4.496 | 0.047 | 3 | 31 | 15 | 276283 |
| production | all | days 0 to 2 | spread | 1.413 | 0.015 | 3 | 31 | 15 | 276283 |
| production | all | days 3 to 10 | pinball19 | 12.221 | 0.037 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | bias | 2.568 | 0.071 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | coverage80 | 12.047 | 0.114 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | coverage90 | 15.638 | 0.185 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | width80 | 4.565 | 0.059 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | width90 | 6.000 | 0.080 | 3 | 31 | 15 | 718201 |
| production | all | days 3 to 10 | spread | 1.876 | 0.025 | 3 | 31 | 15 | 718201 |
| production | all | days 11 onward | pinball19 | 12.520 | 0.056 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | bias | 2.654 | 0.102 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | coverage80 | 12.844 | 0.227 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | coverage90 | 16.638 | 0.254 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | width80 | 4.433 | 0.094 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | width90 | 5.838 | 0.127 | 3 | 31 | 15 | 331946 |
| production | all | days 11 onward | spread | 1.799 | 0.037 | 3 | 31 | 15 | 331946 |
| production | PV | days 0 to 2 | pinball19 | 7.318 | 0.034 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | bias | 1.002 | 0.067 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | coverage80 | 12.337 | 0.361 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | coverage90 | 16.063 | 0.504 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | width80 | 3.165 | 0.020 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | width90 | 4.156 | 0.025 | 3 | 6 | 15 | 55505 |
| production | PV | days 0 to 2 | spread | 1.304 | 0.009 | 3 | 6 | 15 | 55505 |
| production | PV | days 3 to 10 | pinball19 | 9.125 | 0.018 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | bias | 0.916 | 0.075 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | coverage80 | 18.642 | 0.171 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | coverage90 | 24.211 | 0.166 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | width80 | 4.825 | 0.067 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | width90 | 6.259 | 0.091 | 3 | 6 | 15 | 130890 |
| production | PV | days 3 to 10 | spread | 1.961 | 0.027 | 3 | 6 | 15 | 130890 |
| production | PV | days 11 onward | pinball19 | 8.836 | 0.044 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | bias | 0.350 | 0.129 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | coverage80 | 21.821 | 0.583 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | coverage90 | 27.762 | 0.386 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | width80 | 4.923 | 0.084 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | width90 | 6.433 | 0.117 | 3 | 6 | 15 | 66608 |
| production | PV | days 11 onward | spread | 1.994 | 0.033 | 3 | 6 | 15 | 66608 |
| production | Wind | days 0 to 2 | pinball19 | 12.994 | 0.111 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | bias | 4.675 | 0.241 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | coverage80 | 17.655 | 0.325 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | coverage90 | 22.418 | 0.304 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | width80 | 9.487 | 0.111 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | width90 | 12.587 | 0.130 | 3 | 3 | 15 | 27972 |
| production | Wind | days 0 to 2 | spread | 3.962 | 0.043 | 3 | 3 | 15 | 27972 |
| production | Wind | days 3 to 10 | pinball19 | 17.152 | 0.069 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | bias | 4.024 | 0.241 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | coverage80 | 13.985 | 0.320 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | coverage90 | 17.868 | 0.305 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | width80 | 9.936 | 0.038 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | width90 | 13.231 | 0.035 | 3 | 3 | 15 | 74352 |
| production | Wind | days 3 to 10 | spread | 4.161 | 0.011 | 3 | 3 | 15 | 74352 |
| production | Wind | days 11 onward | pinball19 | 18.604 | 0.138 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | bias | 2.933 | 0.402 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | coverage80 | 9.351 | 0.128 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | coverage90 | 12.505 | 0.327 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | width80 | 6.654 | 0.141 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | width90 | 8.901 | 0.114 | 3 | 3 | 15 | 33597 |
| production | Wind | days 11 onward | spread | 2.735 | 0.035 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | pinball19 | 7.385 | 0.013 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | bias | 1.388 | 0.017 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage80 | 74.085 | 0.096 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | coverage90 | 76.913 | 0.084 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width80 | 25.303 | 0.020 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | width90 | 27.270 | 0.022 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 0 to 2 | spread | 8.438 | 0.008 | 3 | 31 | 15 | 276283 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | pinball19 | 8.590 | 0.012 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | bias | 1.751 | 0.012 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage80 | 72.401 | 0.097 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | coverage90 | 76.309 | 0.101 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width80 | 28.709 | 0.019 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | width90 | 31.605 | 0.015 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 3 to 10 | spread | 9.814 | 0.008 | 3 | 31 | 15 | 718201 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | pinball19 | 9.278 | 0.013 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | bias | 2.333 | 0.007 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage80 | 73.046 | 0.093 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | coverage90 | 77.019 | 0.096 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width80 | 31.297 | 0.014 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | width90 | 34.388 | 0.023 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | all | days 11 onward | spread | 10.648 | 0.013 | 3 | 31 | 15 | 331946 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | pinball19 | 4.303 | 0.005 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | bias | 0.088 | 0.009 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage80 | 84.316 | 0.081 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | coverage90 | 88.017 | 0.097 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width80 | 16.814 | 0.015 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | width90 | 19.117 | 0.020 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 0 to 2 | spread | 5.991 | 0.006 | 3 | 6 | 15 | 55505 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | pinball19 | 5.926 | 0.005 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | bias | -0.064 | 0.003 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage80 | 81.671 | 0.045 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | coverage90 | 86.961 | 0.029 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width80 | 22.232 | 0.011 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | width90 | 25.505 | 0.016 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 3 to 10 | spread | 8.138 | 0.003 | 3 | 6 | 15 | 130890 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | pinball19 | 5.778 | 0.004 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | bias | -0.954 | 0.005 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage80 | 83.676 | 0.035 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | coverage90 | 88.702 | 0.069 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width80 | 22.104 | 0.021 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | width90 | 25.395 | 0.013 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | PV | days 11 onward | spread | 8.119 | 0.011 | 3 | 6 | 15 | 66608 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | pinball19 | 6.229 | 0.003 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | bias | 1.149 | 0.008 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage80 | 75.327 | 0.104 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | coverage90 | 84.208 | 0.050 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width80 | 28.914 | 0.013 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | width90 | 35.532 | 0.007 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 0 to 2 | spread | 11.113 | 0.001 | 3 | 3 | 15 | 27972 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | pinball19 | 10.390 | 0.003 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | bias | 0.621 | 0.002 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage80 | 72.798 | 0.071 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | coverage90 | 80.960 | 0.065 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width80 | 45.683 | 0.032 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | width90 | 54.702 | 0.032 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 3 to 10 | spread | 17.405 | 0.009 | 3 | 3 | 15 | 74352 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | pinball19 | 11.956 | 0.002 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | bias | 0.853 | 0.014 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage80 | 71.669 | 0.089 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | coverage90 | 78.837 | 0.057 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width80 | 51.348 | 0.091 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | width90 | 60.080 | 0.051 | 3 | 3 | 15 | 33597 |
| r2_e1_c1_a3w_d3 | Wind | days 11 onward | spread | 19.098 | 0.012 | 3 | 3 | 15 | 33597 |

## Fold B (retrospective), sensitivity: delivered descriptive diagnostics by November source era (coverage in % of cases; loss, width, bias and spread in % of frozen capacity)

| Recipe | Series type | Source era | Metric | Value (%) | Seed spread (percentage points) | Fitting seeds | Series | Supported months | Forecast cases per seed |
|---|---|---|---|---|---|---|---|---|---|
| production | all | after November 2024 boundary | pinball19 | 11.556 | 0.035 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | bias | 3.282 | 0.050 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | coverage80 | 12.841 | 0.083 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | coverage90 | 16.455 | 0.112 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | width80 | 4.298 | 0.045 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | width90 | 5.659 | 0.052 | 3 | 31 | 8 | 663954 |
| production | all | after November 2024 boundary | spread | 1.767 | 0.015 | 3 | 31 | 8 | 663954 |
| production | all | before November 2024 boundary | pinball19 | 10.862 | 0.009 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | bias | -0.365 | 0.126 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | coverage80 | 10.840 | 0.149 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | coverage90 | 14.288 | 0.237 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | width80 | 4.376 | 0.098 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | width90 | 5.757 | 0.131 | 3 | 29 | 8 | 662476 |
| production | all | before November 2024 boundary | spread | 1.792 | 0.041 | 3 | 29 | 8 | 662476 |
| production | PV | after November 2024 boundary | pinball19 | 7.377 | 0.007 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | bias | -1.094 | 0.026 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | coverage80 | 22.661 | 0.417 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | coverage90 | 28.419 | 0.402 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | width80 | 4.140 | 0.055 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | width90 | 5.355 | 0.068 | 3 | 6 | 8 | 123295 |
| production | PV | after November 2024 boundary | spread | 1.683 | 0.019 | 3 | 6 | 8 | 123295 |
| production | PV | before November 2024 boundary | pinball19 | 9.866 | 0.047 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | bias | 2.582 | 0.144 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | coverage80 | 13.748 | 0.430 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | coverage90 | 18.540 | 0.475 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | width80 | 4.816 | 0.066 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | width90 | 6.308 | 0.096 | 3 | 6 | 8 | 129708 |
| production | PV | before November 2024 boundary | spread | 1.961 | 0.028 | 3 | 6 | 8 | 129708 |
| production | Wind | after November 2024 boundary | pinball19 | 17.392 | 0.018 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | bias | 2.525 | 0.032 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | coverage80 | 14.390 | 0.060 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | coverage90 | 18.431 | 0.062 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | width80 | 9.278 | 0.119 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | width90 | 12.428 | 0.167 | 3 | 3 | 8 | 66457 |
| production | Wind | after November 2024 boundary | spread | 3.890 | 0.048 | 3 | 3 | 8 | 66457 |
| production | Wind | before November 2024 boundary | pinball19 | 15.949 | 0.184 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | bias | 5.193 | 0.528 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | coverage80 | 12.834 | 0.313 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | coverage90 | 16.567 | 0.399 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | width80 | 8.797 | 0.221 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | width90 | 11.645 | 0.226 | 3 | 3 | 8 | 69464 |
| production | Wind | before November 2024 boundary | spread | 3.650 | 0.067 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | pinball19 | 8.264 | 0.011 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | bias | 2.340 | 0.015 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | coverage80 | 74.845 | 0.090 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | coverage90 | 78.813 | 0.107 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | width80 | 29.177 | 0.011 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | width90 | 31.978 | 0.011 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | after November 2024 boundary | spread | 9.917 | 0.010 | 3 | 31 | 8 | 663954 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | pinball19 | 7.465 | 0.003 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | bias | -0.373 | 0.003 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | coverage80 | 72.721 | 0.030 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | coverage90 | 76.299 | 0.022 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | width80 | 27.041 | 0.013 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | width90 | 29.852 | 0.006 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | all | before November 2024 boundary | spread | 9.264 | 0.002 | 3 | 29 | 8 | 662476 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | pinball19 | 4.798 | 0.002 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | bias | -1.468 | 0.012 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | coverage80 | 84.212 | 0.027 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | coverage90 | 88.982 | 0.109 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | width80 | 18.575 | 0.006 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | width90 | 21.280 | 0.009 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | after November 2024 boundary | spread | 6.804 | 0.002 | 3 | 6 | 8 | 123295 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | pinball19 | 6.227 | 0.006 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | bias | 0.888 | 0.011 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | coverage80 | 81.413 | 0.065 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | coverage90 | 86.383 | 0.010 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | width80 | 23.322 | 0.032 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | width90 | 26.728 | 0.036 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | PV | before November 2024 boundary | spread | 8.477 | 0.010 | 3 | 6 | 8 | 129708 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | pinball19 | 10.450 | 0.003 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | bias | 0.497 | 0.006 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | coverage80 | 74.210 | 0.008 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | coverage90 | 82.197 | 0.009 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | width80 | 46.247 | 0.027 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | width90 | 54.682 | 0.020 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | after November 2024 boundary | spread | 17.341 | 0.005 | 3 | 3 | 8 | 66457 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | pinball19 | 9.414 | 0.002 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | bias | 1.064 | 0.003 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | coverage80 | 71.919 | 0.091 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | coverage90 | 80.057 | 0.096 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | width80 | 41.130 | 0.049 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | width90 | 49.603 | 0.047 | 3 | 3 | 8 | 69464 |
| r2_e1_c1_a3w_d3 | Wind | before November 2024 boundary | spread | 15.752 | 0.011 | 3 | 3 | 8 | 69464 |

## Reproducing the figures

**The commands below rescore all four completed fold/settings from their six saved fits.** Run
from the repository root with the recorded environment. Set `STUDY_DATA_ROOT` to the private
`data/studies/per_study` directory and choose a new `STUDY_REPRO_ROOT`; each output directory must
be absent. The recipe and seed loops name the exact completed fit-directory layout. These are
portable reproduction commands, not a claim that new runs have been executed. They omit the
untriggered best-point refit. The [supplementary tables](calibration-diagnostics-1156-tables.md)
copy the completed selector output. Reproducing that selection also requires a separately labelled
copy of the corresponding public table specification, with its input paths and required report
hashes updated to match the reproduced reports. Preserve the archived specification unchanged.

```bash
set -euo pipefail
STUDY_DATA_ROOT="/path/to/data/studies/per_study"
STUDY_REPRO_ROOT="/path/to/new/calibration-reproduction"
STUDY_PARENT_ROOT="$STUDY_DATA_ROOT/quick_wins_fixed_pipeline_1156"
export POLARS_MAX_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MLFLOW_DISABLE_AGENT_HINT=1

for STUDY_RUN in fold_a_main fold_a_sensitivity fold_b_main fold_b_sensitivity; do
  STUDY_FOLD="${STUDY_RUN%_*}"
  STUDY_SETTING="${STUDY_RUN##*_}"
  STUDY_FITS=()
  for STUDY_RECIPE in production r2_e1_c1_a3w_d3; do
    for STUDY_SEED in 0 1 2; do
      STUDY_FITS+=("$STUDY_PARENT_ROOT/fits/combined/$STUDY_FOLD/$STUDY_RECIPE/$STUDY_SETTING/cuda_seed${STUDY_SEED}_none")
    done
  done
  .venv/bin/python studies/forecast_calibration/calibration_diagnostics.py \
    --frames "$STUDY_PARENT_ROOT/frames/combined/$STUDY_FOLD" \
    --fit "${STUDY_FITS[@]}" \
    --output "$STUDY_REPRO_ROOT/reports/$STUDY_RUN"
done
```

**Figure text must be reviewed before rendering.** The coverage script verifies a completed
report's output hashes and exports the text with `--text-only`. The first-reader and prose reviews
must clear that exact text before a separate render command uses `--reviewed-text`. Use distinct,
new output directories for text export and rendering, and inspect the rendered labels and layout.
The following commands reproduce the Fold A main coverage figure from the rescored report above.

```bash
.venv/bin/python studies/forecast_calibration/calibration_coverage_charts.py \
  --report "$STUDY_REPRO_ROOT/reports/fold_a_main" \
  --output "$STUDY_REPRO_ROOT/coverage-fold-a-text" \
  --text-only
```

After the exported `figure-text.md` has passed the required text reviews:

```bash
.venv/bin/python studies/forecast_calibration/calibration_coverage_charts.py \
  --report "$STUDY_REPRO_ROOT/reports/fold_a_main" \
  --output "$STUDY_REPRO_ROOT/coverage-fold-a-render" \
  --reviewed-text "$STUDY_REPRO_ROOT/coverage-fold-a-text/figure-text.md"
```

**The reproduction records describe the completed diagnostics.** The scorer's immutable report,
aggregate tables and manifest preserve their evidence. Further Part 2 implementation remains on hold.
