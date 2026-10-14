# Pipeline-comparison results tables

These tables support the [wind, radiation and lean-input study](fixed-pipeline-lean-features.md).
This appendix gathers existing aggregate publication selections without recalculating statistics.
The main page retains the deciding comparisons; the tables here retain their printed precision,
fold and training-setting labels. The [main report's public inventory](fixed-pipeline-lean-features.md#data-and-code-availability)
links the published execution specifications and reproduction records.

Losses are divided by each meter's frozen effective capacity before aggregation. Estimates give
equal weight to meters and fitting seeds; `all`, `PV` and `Wind` denote all series, solar farms and
wind farms. Common loss is twice-mean pinball at 19 shared midpoint probabilities, expressed as
percent of capacity. Loss differences are capacity percentage points; negative differences favour
the treatment named first. Nominal 80% coverage is expressed as percent of forecast cases.

The 95% paired bootstrap ranges resample whole validation months and one shared individual fitting
seed, representing month and individual-fit variation; central estimates average three seeds.
Separately labelled equal-month estimates average fitted seeds within each month and give months
equal weight. Their t-intervals describe variation across those month means, a distinct estimand
and treatment of fitting-seed variation. They must not be read as the primary bootstrap bounds.
The fair-CRPS tables retain the original sample-construction caveat and the distinction between
the main-setting G3 gate and a sensitivity check. See the parent page's
[scoring and uncertainty methods](fixed-pipeline-lean-features.md#scoring-and-uncertainty).

## Old-pipeline reproduction

### Old-pipeline reproduction: common pinball score (% capacity)

| Recipe | Estimate | Lower 95% | Upper 95% | Seeds | Months |
|---|---|---|---|---|---|
| old_production | 11.59 | 10.93 | 12.22 | 3 | 12 |
| old_quick_wins | 7.87 | 7.43 | 8.28 | 3 | 12 |
| old_cleaned | 11.40 | 10.78 | 11.99 | 3 | 12 |

### Old-pipeline reproduction: nominal 80% coverage (% cases)

| Recipe | Estimate | Lower 95% | Upper 95% | Seeds | Months |
|---|---|---|---|---|---|
| old_production | 11.00 | 10.19 | 11.77 | 3 | 12 |
| old_quick_wins | 71.27 | 70.17 | 72.40 | 3 | 12 |
| old_cleaned | 10.48 | 9.93 | 11.07 | 3 | 12 |

## Radiation comparisons

### Radiation correction: July 2025–June 2026, main setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.122 | -0.175 | -0.075 | 11 |
| PV | -0.337 | -0.473 | -0.236 | 12 |
| Wind | 0.043 | -0.048 | 0.122 | 4 |

### Radiation correction: July 2025–June 2026, sensitivity setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.112 | -0.189 | -0.002 | 10 |
| PV | -0.397 | -0.511 | -0.292 | 12 |
| Wind | 0.038 | -0.063 | 0.113 | 4 |

### Radiation correction: April 2024–June 2025, retrospective, main setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.088 | -0.146 | -0.022 | 15 |
| PV | -0.346 | -0.498 | -0.237 | 15 |
| Wind | 0.119 | -0.066 | 0.361 | 4 |

### Radiation correction: April 2024–June 2025, retrospective, sensitivity setting

| Series type | Loss difference | 95% lower | 95% upper | Months improved |
|---|---|---|---|---|
| all | -0.106 | -0.156 | -0.060 | 12 |
| PV | -0.329 | -0.432 | -0.237 | 15 |
| Wind | 0.039 | -0.060 | 0.172 | 6 |

## Fold A main

### Absolute common 19-level pinball loss (% capacity): July 2025–June 2026, main setting

| Forecast recipe and pipeline | Series type | Twice-mean pinball (% capacity) |
|---|---|---|
| old_production | all | 11.594 |
| old_production | PV | 8.319 |
| old_production | Wind | 17.159 |
| combined_production | all | 11.479 |
| combined_production | PV | 8.002 |
| combined_production | Wind | 17.208 |
| old_winner | all | 7.871 |
| old_winner | PV | 5.658 |
| old_winner | Wind | 11.403 |
| combined_winner | all | 7.756 |
| combined_winner | PV | 5.371 |
| combined_winner | Wind | 11.397 |

### Pipeline-isolation comparisons, common 19-level pinball loss: July 2025–June 2026, main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Radiation correction minus old pipeline | all | -0.122 | -0.175 | -0.075 | 11 |
| Radiation correction minus old pipeline | PV | -0.337 | -0.473 | -0.236 | 12 |
| Radiation correction minus old pipeline | Wind | 0.043 | -0.048 | 0.122 | 4 |
| Wind correction minus old pipeline | all | 0.005 | -0.033 | 0.064 | 4 |
| Wind correction minus old pipeline | PV | 0.002 | -0.035 | 0.031 | 6 |
| Wind correction minus old pipeline | Wind | -0.019 | -0.117 | 0.080 | 6 |
| Wind correction with radiation corrected | all | 0.007 | -0.028 | 0.040 | 7 |
| Wind correction with radiation corrected | PV | 0.020 | -0.018 | 0.057 | 3 |
| Wind correction with radiation corrected | Wind | 0.006 | -0.140 | 0.108 | 7 |
| Radiation correction with wind corrected | all | -0.120 | -0.183 | -0.042 | 11 |
| Radiation correction with wind corrected | PV | -0.319 | -0.428 | -0.227 | 12 |
| Radiation correction with wind corrected | Wind | 0.068 | -0.057 | 0.178 | 2 |

### Wind–radiation interaction (combined − radiation − wind + old), common 19-level pinball loss: July 2025–June 2026, main setting

| Comparison | Series type | Interaction (capacity points) | 95% lower | 95% upper | Months with negative interaction |
|---|---|---|---|---|---|
| Interaction between wind and radiation corrections | all | 0.002 | -0.083 | 0.061 | 7 |
| Interaction between wind and radiation corrections | PV | 0.018 | -0.037 | 0.074 | 5 |
| Interaction between wind and radiation corrections | Wind | 0.025 | -0.112 | 0.159 | 6 |

### Cleaned-control pipeline comparison, common 19-level pinball loss: July 2025–June 2026, main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections on cleaned feature baseline | all | -0.147 | -0.201 | -0.079 | 11 |
| Both corrections on cleaned feature baseline | PV | -0.337 | -0.447 | -0.251 | 11 |
| Both corrections on cleaned feature baseline | Wind | 0.045 | -0.054 | 0.177 | 5 |

### G3 material-fault gate: fair CRPS, combined production minus old production; July 2025–June 2026, main setting

| Comparison | Series type | Fair CRPS difference (capacity points) | Equal-month mean difference (capacity points) | Month t-interval lower | Month t-interval upper | Months with lower loss | Months scored |
|---|---|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.118 | -0.124 | -0.174 | -0.074 | 11 | 12 |

## Fold B main (retrospective)

### Absolute common 19-level pinball loss (% capacity): April 2024–June 2025 (retrospective), main setting

| Forecast recipe and pipeline | Series type | Twice-mean pinball (% capacity) |
|---|---|---|
| old_production | all | 12.234 |
| old_production | PV | 9.047 |
| old_production | Wind | 16.463 |
| combined_production | all | 12.179 |
| combined_production | PV | 8.715 |
| combined_production | Wind | 16.622 |
| old_winner | all | 8.825 |
| old_winner | PV | 5.979 |
| old_winner | Wind | 9.937 |
| combined_winner | all | 8.745 |
| combined_winner | PV | 5.799 |
| combined_winner | Wind | 9.942 |

### Pipeline-isolation comparisons, common 19-level pinball loss: April 2024–June 2025 (retrospective), main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Radiation correction minus old pipeline | all | -0.088 | -0.146 | -0.022 | 15 |
| Radiation correction minus old pipeline | PV | -0.346 | -0.498 | -0.237 | 15 |
| Radiation correction minus old pipeline | Wind | 0.119 | -0.066 | 0.361 | 4 |
| Wind correction minus old pipeline | all | 0.027 | -0.011 | 0.074 | 2 |
| Wind correction minus old pipeline | PV | -0.006 | -0.072 | 0.039 | 10 |
| Wind correction minus old pipeline | Wind | 0.194 | 0.002 | 0.470 | 4 |
| Wind correction with radiation corrected | all | 0.034 | -0.022 | 0.111 | 2 |
| Wind correction with radiation corrected | PV | 0.014 | -0.030 | 0.049 | 5 |
| Wind correction with radiation corrected | Wind | 0.040 | -0.217 | 0.395 | 6 |
| Radiation correction with wind corrected | all | -0.082 | -0.129 | -0.036 | 14 |
| Radiation correction with wind corrected | PV | -0.326 | -0.481 | -0.219 | 15 |
| Radiation correction with wind corrected | Wind | -0.035 | -0.238 | 0.189 | 11 |

### Wind–radiation interaction (combined − radiation − wind + old), common 19-level pinball loss: April 2024–June 2025 (retrospective), main setting

| Comparison | Series type | Interaction (capacity points) | 95% lower | 95% upper | Months with negative interaction |
|---|---|---|---|---|---|
| Interaction between wind and radiation corrections | all | 0.006 | -0.093 | 0.090 | 5 |
| Interaction between wind and radiation corrections | PV | 0.020 | -0.020 | 0.070 | 5 |
| Interaction between wind and radiation corrections | Wind | -0.155 | -0.585 | 0.201 | 10 |

### Cleaned-control pipeline comparison, common 19-level pinball loss: April 2024–June 2025 (retrospective), main setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Both corrections on cleaned feature baseline | all | -0.061 | -0.105 | -0.002 | 12 |
| Both corrections on cleaned feature baseline | PV | -0.368 | -0.469 | -0.274 | 15 |
| Both corrections on cleaned feature baseline | Wind | 0.125 | -0.083 | 0.387 | 4 |

### G3 material-fault gate: fair CRPS, combined production minus old production; April 2024–June 2025 (retrospective), main setting

| Comparison | Series type | Fair CRPS difference (capacity points) | Equal-month mean difference (capacity points) | Month t-interval lower | Month t-interval upper | Months with lower loss | Months scored |
|---|---|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.057 | -0.053 | -0.083 | -0.024 | 13 | 15 |

## Fold A sensitivity

### Absolute common 19-level pinball loss (% capacity): July 2025–June 2026, sensitivity setting

| Forecast recipe and pipeline | Series type | Twice-mean pinball (% capacity) |
|---|---|---|
| old_production | all | 11.374 |
| old_production | PV | 8.215 |
| old_production | Wind | 16.811 |
| combined_production | all | 11.271 |
| combined_production | PV | 7.818 |
| combined_production | Wind | 16.866 |
| old_winner | all | 7.909 |
| old_winner | PV | 5.553 |
| old_winner | Wind | 11.368 |
| combined_winner | all | 7.775 |
| combined_winner | PV | 5.186 |
| combined_winner | Wind | 11.369 |

### Pipeline-isolation comparisons, common 19-level pinball loss: July 2025–June 2026, sensitivity setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Radiation correction minus old pipeline | all | -0.112 | -0.189 | -0.002 | 10 |
| Radiation correction minus old pipeline | PV | -0.397 | -0.511 | -0.292 | 12 |
| Radiation correction minus old pipeline | Wind | 0.038 | -0.063 | 0.113 | 4 |
| Wind correction minus old pipeline | all | 0.013 | -0.032 | 0.075 | 4 |
| Wind correction minus old pipeline | PV | 0.005 | -0.019 | 0.031 | 6 |
| Wind correction minus old pipeline | Wind | 0.026 | -0.064 | 0.098 | 3 |
| Wind correction with radiation corrected | all | 0.010 | -0.020 | 0.050 | 4 |
| Wind correction with radiation corrected | PV | 0.001 | -0.020 | 0.028 | 6 |
| Wind correction with radiation corrected | Wind | 0.017 | -0.050 | 0.087 | 6 |
| Radiation correction with wind corrected | all | -0.116 | -0.181 | -0.051 | 9 |
| Radiation correction with wind corrected | PV | -0.401 | -0.504 | -0.312 | 12 |
| Radiation correction with wind corrected | Wind | 0.028 | -0.053 | 0.104 | 6 |

### Wind–radiation interaction (combined − radiation − wind + old), common 19-level pinball loss: July 2025–June 2026, sensitivity setting

| Comparison | Series type | Interaction (capacity points) | 95% lower | 95% upper | Months with negative interaction |
|---|---|---|---|---|---|
| Interaction between wind and radiation corrections | all | -0.003 | -0.080 | 0.075 | 8 |
| Interaction between wind and radiation corrections | PV | -0.004 | -0.039 | 0.032 | 6 |
| Interaction between wind and radiation corrections | Wind | -0.010 | -0.087 | 0.075 | 7 |

The [deciding comparisons](fixed-pipeline-lean-features.md#results) remain on the main page.

### Fair CRPS sensitivity check (not the registered G3 gate): July 2025–June 2026

| Comparison | Series type | Fair CRPS difference (capacity points) | Equal-month difference (capacity points) | Equal-month t-interval lower | Equal-month t-interval upper | Months with lower loss | Validation months | Fitting seeds | Minimum supported months per meter | Maximum supported months per meter | Meters missing months (maximum across seeds) | Bootstrap draws omitting meters | Most meters omitted in a draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.105 | -0.123 | -0.212 | -0.034 | 10 | 12 | 3 | 12 | 12 | 0 | 0 | 0 |

## Fold B sensitivity (retrospective)

### Absolute common 19-level pinball loss (% capacity): April 2024–June 2025 (retrospective), sensitivity setting

| Forecast recipe and pipeline | Series type | Twice-mean pinball (% capacity) |
|---|---|---|
| old_production | all | 12.169 |
| old_production | PV | 8.974 |
| old_production | Wind | 16.483 |
| combined_production | all | 12.079 |
| combined_production | PV | 8.651 |
| combined_production | Wind | 16.655 |
| old_winner | all | 8.605 |
| old_winner | PV | 5.727 |
| old_winner | Wind | 9.912 |
| combined_winner | all | 8.510 |
| combined_winner | PV | 5.530 |
| combined_winner | Wind | 9.921 |

### Pipeline-isolation comparisons, common 19-level pinball loss: April 2024–June 2025 (retrospective), sensitivity setting

| Comparison | Series type | Loss difference (capacity points) | 95% lower | 95% upper | Months with lower loss |
|---|---|---|---|---|---|
| Radiation correction minus old pipeline | all | -0.106 | -0.156 | -0.060 | 12 |
| Radiation correction minus old pipeline | PV | -0.329 | -0.432 | -0.237 | 15 |
| Radiation correction minus old pipeline | Wind | 0.039 | -0.060 | 0.172 | 6 |
| Wind correction minus old pipeline | all | 0.013 | -0.011 | 0.044 | 4 |
| Wind correction minus old pipeline | PV | -0.008 | -0.032 | 0.012 | 11 |
| Wind correction minus old pipeline | Wind | 0.121 | 0.048 | 0.204 | 2 |
| Wind correction with radiation corrected | all | 0.016 | -0.016 | 0.046 | 6 |
| Wind correction with radiation corrected | PV | 0.007 | -0.022 | 0.036 | 5 |
| Wind correction with radiation corrected | Wind | 0.133 | -0.018 | 0.310 | 4 |
| Radiation correction with wind corrected | all | -0.102 | -0.151 | -0.059 | 14 |
| Radiation correction with wind corrected | PV | -0.315 | -0.402 | -0.233 | 15 |
| Radiation correction with wind corrected | Wind | 0.052 | -0.036 | 0.166 | 5 |

### Wind–radiation interaction (combined − radiation − wind + old), common 19-level pinball loss: April 2024–June 2025 (retrospective), sensitivity setting

| Comparison | Series type | Interaction (capacity points) | 95% lower | 95% upper | Months with negative interaction |
|---|---|---|---|---|---|
| Interaction between wind and radiation corrections | all | 0.004 | -0.026 | 0.036 | 8 |
| Interaction between wind and radiation corrections | PV | 0.015 | -0.023 | 0.062 | 3 |
| Interaction between wind and radiation corrections | Wind | 0.012 | -0.147 | 0.152 | 8 |

### Fair CRPS sensitivity check (not the registered G3 gate): April 2024–June 2025 (retrospective)

| Comparison | Series type | Fair CRPS difference (capacity points) | Equal-month difference (capacity points) | Equal-month t-interval lower | Equal-month t-interval upper | Months with lower loss | Validation months | Fitting seeds | Minimum supported months per meter | Maximum supported months per meter | Meters missing months (maximum across seeds) | Bootstrap draws omitting meters | Most meters omitted in a draw |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Both corrections minus old pipeline | all | -0.093 | -0.082 | -0.124 | -0.040 | 12 | 15 | 3 | 5 | 15 | 2 | 3 | 1 |
