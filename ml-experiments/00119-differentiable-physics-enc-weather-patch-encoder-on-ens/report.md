## Direction F1 (291 runs, 464,363 decision-site rows)

### Planned contrasts, 99% intervals, points of capacity

| Contrast | Pair | Annual | Day ahead | Winter | MAE of q50 | Sites better | Leave-one-out C1 range |
|---|---|---|---|---|---|---|---|
| C1 | ENC-c minus GAM-K | -0.062 [-0.090, -0.033] | -0.062 [-0.090, -0.037] | -0.041 [-0.076, -0.020] | -0.102 | 5 of 5 | -0.067 to -0.057 |
| C2 | ENC-c minus ENC-c0 | -0.002 [-0.010, +0.007] | -0.004 [-0.011, +0.004] | +0.005 [-0.005, +0.012] | -0.002 | 2 of 5 |  |
| C3 | ENC-c minus ENC-cpt | -0.019 [-0.036, -0.006] | -0.025 [-0.041, -0.010] | -0.018 [-0.041, +0.001] | -0.040 | 5 of 5 |  |
| C3b | ENC-c minus ENC-cps | -0.002 [-0.010, +0.007] | -0.005 [-0.012, +0.003] | +0.005 [-0.005, +0.012] | -0.002 | 3 of 5 |  |
| C4 | ENC-r minus GAM-K | -0.037 [-0.053, -0.020] | -0.046 [-0.068, -0.026] | -0.031 [-0.051, -0.018] | -0.071 | 5 of 5 |  |

### Every scope, C1 (ENC-c minus GAM-K)

| Scope | Estimate [99% interval] |
|---|---|
| 0.5-168 h | -0.062 [-0.090, -0.033] |
| D+1 (15.5-39 h) | -0.062 [-0.090, -0.037] |
| 0-24 h | -0.057 [-0.089, -0.028] |
| 24-48 h | -0.066 [-0.094, -0.039] |
| 48-144 h | -0.064 [-0.094, -0.033] |
| 144-168 h | -0.055 [-0.087, -0.025] |
| winter (Nov-Mar) | -0.041 [-0.076, -0.020] |
| winter evening peak (16-20 UTC) | -0.046 [-0.099, +0.004] |
| summer (Apr-Oct) | -0.075 [-0.112, -0.036] |
| working days | -0.059 [-0.090, -0.027] |
| weekends and bank holidays | -0.069 [-0.099, -0.035] |

### Each arm minus GAM-K (annual), and absolute scores

| Arm | Annual [99% interval] | Winter | Absolute score, annual | Winter | Day ahead |
|---|---|---|---|---|---|
| GAM-K | 0 | 0 | 2.800 | 2.928 | 2.534 |
| ENC-c | -0.062 [-0.090, -0.033] | -0.041 | 2.739 | 2.887 | 2.472 |
| ENC-c0 | -0.060 [-0.087, -0.034] | -0.046 | 2.740 | 2.882 | 2.476 |
| ENC-cps | -0.060 [-0.084, -0.037] | -0.047 | 2.740 | 2.882 | 2.477 |
| ENC-cpt | -0.043 [-0.063, -0.020] | -0.023 | 2.758 | 2.905 | 2.497 |
| ENC-r | -0.037 [-0.053, -0.020] | -0.031 | 2.763 | 2.897 | 2.488 |
| ENC-r0 | -0.037 [-0.053, -0.021] | -0.032 | 2.764 | 2.896 | 2.491 |

GAM-0 (experiment 111, no Kalman filter) on the same rows: 2.951 against GAM-K 2.800.

### Hard days (7-day block bootstrap, 99%), descriptive

| Subset | Days | C1 | C3 |
|---|---|---|---|
| coldest | 13 | -0.115 [-0.197, -0.039] | -0.077 [-0.199, -0.015] |
| showery | 18 | -0.059 [-0.126, +0.000] | -0.016 [-0.052, +0.012] |
| largest_error | 69 | -0.131 [-0.203, -0.017] | -0.036 [-0.077, +0.012] |

### Tail coverage (share at or below the quantile), annual, decision sites

| Arm | q01 | q05 | q95 | q99 |
|---|---|---|---|---|
| K0 | 0.018 | 0.058 | 0.948 | 0.983 |
| ENC-c | 0.013 | 0.052 | 0.947 | 0.984 |
| ENC-cpt | 0.013 | 0.052 | 0.948 | 0.984 |

### Median minus mean forecast, points, by site (GAM-K's, the same for every arm)

D10 -0.037, D11 -0.043, D14 +0.015, D16 +0.002, D5 -0.117

### Five seeds, annual absolute score

- ENC-c: 2.739, 2.740, 2.744, 2.741, 2.743
- ENC-r: 2.764, 2.768, 2.760, 2.764, 2.765

Quantile score and MAE of the median agree in sign: {'C1': True, 'C2': True, 'C3': True, 'C3b': True, 'C4': True}

## Direction F2 (365 runs, 599,158 decision-site rows)

### Planned contrasts, 99% intervals, points of capacity

| Contrast | Pair | Annual | Day ahead | Winter | MAE of q50 | Sites better | Leave-one-out C1 range |
|---|---|---|---|---|---|---|---|
| C1 | ENC-c minus GAM-K | -0.035 [-0.060, -0.017] | -0.038 [-0.062, -0.018] | -0.027 [-0.042, -0.014] | -0.061 | 5 of 5 | -0.038 to -0.031 |
| C2 | ENC-c minus ENC-c0 | +0.003 [-0.003, +0.010] | -0.001 [-0.007, +0.005] | +0.007 [+0.001, +0.016] | +0.006 | 1 of 5 |  |
| C3 | ENC-c minus ENC-cpt | -0.013 [-0.028, +0.002] | -0.023 [-0.036, -0.009] | -0.014 [-0.034, +0.000] | -0.026 | 5 of 5 |  |
| C3b | ENC-c minus ENC-cps | +0.001 [-0.002, +0.005] | -0.000 [-0.004, +0.004] | +0.003 [-0.002, +0.011] | +0.002 | 1 of 5 |  |
| C4 | ENC-r minus GAM-K | -0.035 [-0.054, -0.019] | -0.037 [-0.057, -0.019] | -0.028 [-0.038, -0.011] | -0.067 | 5 of 5 |  |

### Every scope, C1 (ENC-c minus GAM-K)

| Scope | Estimate [99% interval] |
|---|---|
| 0.5-168 h | -0.035 [-0.060, -0.017] |
| D+1 (15.5-39 h) | -0.038 [-0.062, -0.018] |
| 0-24 h | -0.039 [-0.063, -0.020] |
| 24-48 h | -0.037 [-0.060, -0.019] |
| 48-144 h | -0.038 [-0.067, -0.017] |
| 144-168 h | -0.017 [-0.046, +0.012] |
| winter (Nov-Mar) | -0.027 [-0.042, -0.014] |
| winter evening peak (16-20 UTC) | -0.049 [-0.067, -0.032] |
| summer (Apr-Oct) | -0.040 [-0.079, -0.013] |
| working days | -0.032 [-0.056, -0.013] |
| weekends and bank holidays | -0.041 [-0.069, -0.017] |

### Each arm minus GAM-K (annual), and absolute scores

| Arm | Annual [99% interval] | Winter | Absolute score, annual | Winter | Day ahead |
|---|---|---|---|---|---|
| GAM-K | 0 | 0 | 2.733 | 2.787 | 2.428 |
| ENC-c | -0.035 [-0.060, -0.017] | -0.027 | 2.698 | 2.759 | 2.390 |
| ENC-c0 | -0.038 [-0.060, -0.019] | -0.034 | 2.696 | 2.752 | 2.391 |
| ENC-cps | -0.036 [-0.062, -0.017] | -0.031 | 2.697 | 2.756 | 2.390 |
| ENC-cpt | -0.022 [-0.039, -0.006] | -0.013 | 2.711 | 2.774 | 2.412 |
| ENC-r | -0.035 [-0.054, -0.019] | -0.028 | 2.699 | 2.759 | 2.391 |
| ENC-r0 | -0.036 [-0.051, -0.023] | -0.031 | 2.697 | 2.756 | 2.390 |

GAM-0 (experiment 111, no Kalman filter) on the same rows: 2.984 against GAM-K 2.733.

### Hard days (7-day block bootstrap, 99%), descriptive

| Subset | Days | C1 | C3 |
|---|---|---|---|
| coldest | 22 | -0.057 [-0.133, +0.008] | -0.043 [-0.124, +0.028] |
| showery | 27 | -0.028 [-0.092, +0.025] | -0.019 [-0.065, +0.013] |
| largest_error | 84 | -0.061 [-0.133, +0.025] | -0.011 [-0.061, +0.043] |

### Tail coverage (share at or below the quantile), annual, decision sites

| Arm | q01 | q05 | q95 | q99 |
|---|---|---|---|---|
| K0 | 0.015 | 0.054 | 0.946 | 0.987 |
| ENC-c | 0.011 | 0.048 | 0.946 | 0.990 |
| ENC-cpt | 0.011 | 0.049 | 0.947 | 0.990 |

### Median minus mean forecast, points, by site (GAM-K's, the same for every arm)

D10 -0.042, D11 -0.045, D14 +0.009, D16 +0.001, D5 -0.090

### Five seeds, annual absolute score

- ENC-c: 2.696, 2.702, 2.699, 2.700, 2.701
- ENC-r: 2.700, 2.702, 2.700, 2.698, 2.700

Quantile score and MAE of the median agree in sign: {'C1': True, 'C2': True, 'C3': True, 'C3b': True, 'C4': True}

## Settings chosen on each training year

- F1.ENC-c: c 40.0, epochs 11, lambda 0.5
- F1.ENC-r: c 0.4, epochs 1, lambda 0.5
- F2.ENC-c: c 40.0, epochs 4, lambda 0.5
- F2.ENC-r: c 40.0, epochs 2, lambda 0.5
- Parameters: {'c': 973, 'r': 766}

## Shared cells between the six patches (of 37)

- D5: D5 37, D10 23, D11 18, D14 30, D16 30, D9 16
- D10: D5 23, D10 37, D11 30, D14 30, D16 18, D9 30
- D11: D5 18, D10 30, D11 37, D14 24, D16 16, D9 30
- D14: D5 30, D10 30, D11 24, D14 37, D16 24, D9 23
- D16: D5 30, D10 18, D11 16, D14 24, D16 37, D9 12
- D9: D5 16, D10 30, D11 30, D14 23, D16 12, D9 37

Placebo ENC-cps partner (farthest decision site): {'D10': 'D16', 'D11': 'D16', 'D14': 'D11', 'D16': 'D10', 'D5': 'D11', 'D9': 'D16'}

## Gate gate_run1.json

| Injection | Recovered share | Sites with half recovered | Pooled interval |
|---|---|---|---|
| ENC-c dense 0.025 | 1.61 | 5 of 5 | [-0.073, -0.014] |
| ENC-c dense 0.05 | 1.38 | 5 of 5 | [-0.109, -0.033] |
| ENC-c dense 0.1 | 1.15 | 5 of 5 | [-0.160, -0.065] |
| ENC-c sparse 0.05 | 1.05 | 2 of 5 | [-0.066, +0.000] |
| ENC-c0 dense 0.05 | 1.40 | 5 of 5 | [-0.103, -0.043] |
| ENC-cps dense 0.05 | 1.36 | 5 of 5 | [-0.098, -0.037] |
| ENC-cpt dense 0.05 | 0.91 | 3 of 5 | [-0.089, -0.013] |
| ENC-r dense 0.025 | 4.19 | 1 of 5 | [-0.051, +0.024] |
| ENC-r dense 0.05 | 1.83 | 1 of 5 | [-0.062, +0.020] |
| ENC-r dense 0.1 | 1.28 | 3 of 5 | [-0.083, +0.006] |
| ENC-r sparse 0.05 | 1.00 | 1 of 5 | [-0.051, +0.027] |

clean: ENC-c +0.0248, ENC-c0 +0.0419, ENC-cps +0.0261, ENC-cpt +0.0256, ENC-r +0.0151

permuted: ENC-c +0.5616, ENC-c0 +0.5298, ENC-cps +0.5441, ENC-cpt +0.8310, ENC-r +0.2990

Seed spread 0.0081

## Gate gate_run2.json

| Injection | Recovered share | Sites with half recovered | Pooled interval |
|---|---|---|---|
| ENC-c dense 0.025 | 1.28 | 4 of 5 | [-0.062, -0.004] |
| ENC-c dense 0.05 | 1.00 | 4 of 5 | [-0.087, -0.009] |
| ENC-c dense 0.1 | 0.81 | 4 of 5 | [-0.119, -0.030] |
| ENC-c sparse 0.05 | 0.90 | 2 of 5 | [-0.050, +0.000] |
| ENC-c0 dense 0.05 | 0.59 | 2 of 5 | [-0.062, +0.007] |
| ENC-cps dense 0.05 | 0.89 | 4 of 5 | [-0.079, -0.007] |
| ENC-cpt dense 0.05 | 0.46 | 2 of 5 | [-0.056, +0.015] |
| ENC-r dense 0.025 | 0.34 | 1 of 5 | [-0.054, +0.044] |
| ENC-r dense 0.05 | 0.16 | 1 of 5 | [-0.060, +0.055] |
| ENC-r dense 0.1 | 0.13 | 1 of 5 | [-0.079, +0.076] |
| ENC-r sparse 0.05 | nan | 0 of 5 | [+nan, +nan] |

clean: ENC-c +0.0248, ENC-c0 +0.0419, ENC-cps +0.0261, ENC-cpt +0.0256, ENC-r +0.0151

null: ENC-c +0.0092, ENC-c0 +0.0100, ENC-cps +0.0096, ENC-cpt +0.0127, ENC-r -0.0024

Seed spread 0.0081
