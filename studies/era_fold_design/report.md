# Era-wise fold design: how much it changes the published study results

Status: everything in the brief and the coordinator's later requests is done. Second-setting fits exist for Part B (all arms), for Part C under D0, D0trim, I, IIr (and, for wind only, the superseded II), and for `solar_all` (cuda). Design I2 (a second covering offset set) was fitted at the primary setting only.

## Findings in brief

**Coverage (Part A).** Today's folds (D0) hold out calendar months that no training row carries. Wind study: 8,326 of 50,734 scored hours (16.4%), all three farms, June and July. Solar `long` panel: 2,129 of 76,727 hours (2.8%), one farm, May to July. Past-solar `all` panel: 14,853 of 40,243 hours (36.9%), April to June. `wind_icon_dream` panel: 12,570 of 50,041 hours (25.1%), May to July. Rotating the post-upgrade era's fold numbers by 2 leaves none uncovered in all four row sets. `cut_eras` with zero offsets equals `with_eras` fold for fold. Whole-record folds (D2) cover every month but score 85% of solar post-upgrade hours by a model that trained on no post-upgrade row.

**Published studies (Part B, rotation 2 = D1).** D0 reproduces the published per-row losses bit for bit (CPU). No planned contrast of the wind or solar `long` pages changes sign; changes from D0 are at most 0.033 points (solar and wind alike) at the primary setting, and at most 0.05 points at the second setting (wind ICON-EU and UKV minus ERA5, +0.05 [+0.019, +0.084] and +0.052 [+0.018, +0.088], significant but small). Only wind ICON-EU minus UKV loses significance, under D2 and not under D1. A second covering rotation (3) gives the same picture: every change from rotation 2 to rotation 3 is at most 0.026 points and no paired interval excludes zero, so the fold-assignment spread is small next to the intervals.

**Past-solar `all` panel and `wind_icon_dream`, D1 (rotation 2), primary setting, CPU.** `solar_all`: the three planned contrasts move by +0.022, -0.017 and -0.057 points (none of the three paired intervals excludes zero); the significance of the two that are significant under D0 is unchanged. DMI HARMONIE-AROME minus ICON-D2 is +1.016 [+0.730, +1.264] under D0 and +0.959 [+0.690, +1.188] under D1. `wind_icon_dream`: ICON-DREAM-EU minus ERA5 +0.001 [-0.119, +0.131] under D0 and +0.010 [-0.096, +0.125] under D1; minus ICON-EU +0.340 [+0.266, +0.405] and +0.312 [+0.245, +0.371]. Neither panel's planned contrasts change in sign or significance. On cuda the same fits give a change of -0.076 [-0.132, -0.024] for DMI minus ICON-D2 (CPU: -0.057 [-0.127, +0.022]), so a change of this size is within the CPU-to-GPU difference in fitting.

**ENS-horizons page (Part C, #892).** D0 reproduces the saved losses. The headline wind contrast, ENS day-0 mean minus ERA5 (+0.170 [+0.021, +0.322], 50,268 rows), is not significant under any design that handles IFS 49r1: +0.136 [-0.012, +0.291] (drop the 19 straddle days only), +0.001 [-0.119, +0.132] and -0.037 [-0.167, +0.102] (three-era cuts with two different covering offset sets, I and I2), and -0.026 [-0.151, +0.114] on rows from 2024-12-01 with covering folds (IIr). The wind day-14 climatology claim (+1.168 [+0.416, +1.948]) is not significant under I (+0.618 [-0.494, +1.602]) or I2 (+0.698 [-0.499, +1.701]) but is under IIr (+1.207 [+0.514, +1.915]). Solar day-7 ensemble mean minus climatology is -0.157 [-0.581, +0.238] under D0 and I, and -0.868 [-1.603, -0.188] under IIr: it depends on the row set. Most other planned and solar claims survive every design.

**Design II as first fitted had uncovered folds** (fold numbers of the two eras not rotated on the December-2024-onwards rows), as plan review 2 found. Its results are kept and labelled "II (uncovered)"; IIr is the same rows with `rotate_folds(..., {0: 0, 1: 2})` and `raise_on_uncovered_months` before the fit, and is the design to read. IIr and II differ by 0.0 to 0.1 points on most contrasts; the largest differences are the two climatology contrasts (solar day 7: -1.778 for II, -0.868 for IIr; solar day 14: -0.742 for II, +0.186 for IIr).

Conventions. Interval marks: `*` means the 95% interval excludes zero. All contrasts are in percentage points of capacity (capped mean absolute error as a fraction of each row's own capacity); positive means the first arm has the larger error. Each interval resamples whole calendar months and one of three fitting seeds 2,000 times (`studies.bootstrap`). A "change" interval bootstraps, on shared rows, the design's per-row paired difference minus the comparison design's. A contrast is "near the 5% line" when an interval bound is within 20% of the interval's width from zero.

Designs. **D0** today's folds. **D1** post-upgrade era fold numbers rotated by 2 (the first rotation `search_fold_offsets` returns). **D1 rot 3** the same with rotation 3. **D2** whole-record folds by site, no era cut, era feature kept. Part C: **D0trim** D0 without the IFS 49r1 straddle days (2024-11-12 to 2024-11-30). **I** three eras (IFS 49r1 from 2024-12, UKV upgrade from 2026-02), `cut_eras`, era_code feature, offsets A (wind {1: 2, 2: 0}, solar {1: 3, 2: 0}), on D0trim rows. **I2** as I with offsets B (wind {1: 3, 2: 0}, solar {1: 3, 2: 1}; the second covering set `search_fold_offsets` returns). **II (uncovered)** rows from 2024-12-01, today's UKV-era folds, superseded. **IIr** those rows with rotation 2.

Devices. The reproduction checks, Part B and its rotation-3 fits ran on CPU. Part C ran on CPU up to the first batch and on cuda afterwards; the table under "Devices used, per batch" lists each batch. Every contrast pairs two arms fitted on the same device within one design. Change columns that compare a cuda design with a CPU design carry the CPU-to-GPU difference in fitting, about 0.02 to 0.03 points (see the `solar_all` example above).

## Summary tables

### Part B, primary hyperparameter setting: planned contrasts

| Study | Contrast | D0 | D1 | D2 | D1 − D0 | D2 − D0 |
|---|---|---|---|---|---|---|
| wind | icon_eu_wind − era5_wind | -0.314 [-0.453, -0.174]* | -0.298 [-0.431, -0.169]* | -0.329 [-0.461, -0.209]* | +0.016 [-0.028, +0.054] | -0.015 [-0.069, +0.044] |
| wind | ukv_wind − era5_wind | -0.440 [-0.628, -0.241]* | -0.407 [-0.578, -0.229]* | -0.369 [-0.550, -0.178]* | +0.033 [-0.019, +0.080] | +0.071 [+0.015, +0.122]* |
| wind | icon_eu_wind − ukv_wind | +0.126 [+0.009, +0.232]* | +0.109 [+0.006, +0.208]* | +0.040 [-0.060, +0.133] | -0.016 [-0.064, +0.034] | -0.086 [-0.128, -0.045]* |
| wind | icon_d2_wind − icon_eu_wind | -0.264 [-0.334, -0.185]* | -0.275 [-0.342, -0.200]* | -0.279 [-0.345, -0.210]* | -0.011 [-0.055, +0.036] | -0.015 [-0.084, +0.038] |
| wind | icon_d2_wind − ukv_wind (exploratory) | -0.139 [-0.253, -0.026]* | -0.166 [-0.279, -0.055]* | -0.239 [-0.352, -0.134]* | -0.027 [-0.067, +0.010] | -0.100 [-0.167, -0.046]* |
| solar | cams_global − icon_d2_global | -2.678 [-2.899, -2.424]* | -2.708 [-2.932, -2.456]* | -2.682 [-2.898, -2.437]* | -0.030 [-0.059, -0.003]* | -0.004 [-0.051, +0.052] |
| solar | icon_eu_global − icon_d2_global | +0.623 [+0.483, +0.756]* | +0.590 [+0.463, +0.715]* | +0.601 [+0.469, +0.727]* | -0.033 [-0.068, -0.004]* | -0.022 [-0.052, +0.006] |
| solar | icon_eu_global − ukv_global | -0.476 [-0.654, -0.289]* | -0.491 [-0.672, -0.295]* | -0.652 [-0.845, -0.439]* | -0.015 [-0.044, +0.015] | -0.176 [-0.289, -0.068]* |
| solar | icon_global_global − icon_eu_global | +0.097 [+0.046, +0.147]* | +0.093 [+0.048, +0.140]* | +0.116 [+0.065, +0.169]* | -0.003 [-0.029, +0.019] | +0.019 [-0.004, +0.046] |
| solar | sarah3_global − cams_global | +0.402 [+0.294, +0.498]* | +0.400 [+0.296, +0.498]* | +0.395 [+0.294, +0.491]* | -0.002 [-0.018, +0.014] | -0.007 [-0.041, +0.021] |
| solar | icon_dream_global − era5_global | -0.315 [-0.522, -0.116]* | -0.288 [-0.498, -0.077]* | -0.331 [-0.543, -0.124]* | +0.028 [-0.010, +0.064] | -0.015 [-0.055, +0.021] |

### Part B, second hyperparameter setting: planned contrasts

| Study | Contrast | D0 | D1 | D2 | D1 − D0 | D2 − D0 |
|---|---|---|---|---|---|---|
| wind | icon_eu_wind − era5_wind | -0.399 [-0.524, -0.276]* | -0.348 [-0.476, -0.217]* | -0.351 [-0.486, -0.221]* | +0.051 [+0.019, +0.084]* | +0.049 [+0.001, +0.095]* |
| wind | ukv_wind − era5_wind | -0.481 [-0.661, -0.288]* | -0.429 [-0.599, -0.249]* | -0.349 [-0.534, -0.156]* | +0.052 [+0.018, +0.088]* | +0.132 [+0.078, +0.185]* |
| wind | icon_eu_wind − ukv_wind | +0.082 [-0.022, +0.178] | +0.080 [-0.018, +0.170] | -0.001 [-0.107, +0.091] | -0.002 [-0.033, +0.029] | -0.083 [-0.127, -0.040]* |
| wind | icon_d2_wind − icon_eu_wind | -0.243 [-0.318, -0.156]* | -0.257 [-0.329, -0.172]* | -0.262 [-0.323, -0.200]* | -0.014 [-0.048, +0.021] | -0.019 [-0.077, +0.030] |
| wind | icon_d2_wind − ukv_wind (exploratory) | -0.162 [-0.264, -0.056]* | -0.177 [-0.279, -0.073]* | -0.264 [-0.375, -0.154]* | -0.015 [-0.050, +0.017] | -0.102 [-0.155, -0.052]* |
| solar | cams_global − icon_d2_global | -2.649 [-2.868, -2.395]* | -2.659 [-2.878, -2.412]* | -2.653 [-2.869, -2.403]* | -0.010 [-0.030, +0.009] | -0.004 [-0.043, +0.045] |
| solar | icon_eu_global − icon_d2_global | +0.606 [+0.470, +0.737]* | +0.583 [+0.454, +0.714]* | +0.598 [+0.462, +0.732]* | -0.023 [-0.047, -0.001]* | -0.008 [-0.033, +0.019] |
| solar | icon_eu_global − ukv_global | -0.433 [-0.602, -0.255]* | -0.458 [-0.631, -0.276]* | -0.590 [-0.771, -0.389]* | -0.025 [-0.049, -0.000]* | -0.157 [-0.256, -0.063]* |
| solar | icon_global_global − icon_eu_global | +0.094 [+0.051, +0.134]* | +0.090 [+0.051, +0.130]* | +0.104 [+0.060, +0.148]* | -0.004 [-0.024, +0.019] | +0.010 [-0.011, +0.032] |
| solar | sarah3_global − cams_global | +0.422 [+0.315, +0.523]* | +0.417 [+0.311, +0.518]* | +0.417 [+0.313, +0.516]* | -0.005 [-0.019, +0.010] | -0.005 [-0.030, +0.019] |
| solar | icon_dream_global − era5_global | -0.327 [-0.522, -0.135]* | -0.319 [-0.523, -0.119]* | -0.341 [-0.543, -0.138]* | +0.008 [-0.020, +0.035] | -0.014 [-0.042, +0.013] |

### Part C, primary setting: planned contrasts and headline references

| Domain | Contrast | D0 (published) | D0trim | I (era cut, offsets A) | I2 (era cut, offsets B) | II (uncovered folds, superseded) | IIr (II with covering rotation 2) | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|---|
| wind | ens_mean_day1 − ens_mean_day0 (planned) | +0.754 [+0.564, +0.941]* | +0.786 [+0.617, +0.958]* | +0.859 [+0.711, +1.010]* | +0.895 [+0.749, +1.049]* | +0.860 [+0.718, +1.008]* | +0.904 [+0.760, +1.055]* | +0.073 [-0.006, +0.141] | +0.108 [+0.020, +0.200]* | +0.092 [-0.005, +0.195] |
| wind | ens_mean_day7 − ens_mean_day0 (planned) | +9.298 [+8.204, +10.433]* | +9.317 [+8.241, +10.484]* | +9.478 [+8.428, +10.592]* | +9.643 [+8.589, +10.782]* | +9.295 [+8.259, +10.497]* | +9.458 [+8.390, +10.667]* | +0.161 [-0.123, +0.450] | +0.326 [+0.012, +0.684]* | +0.263 [-0.084, +0.611] |
| wind | ens_members_day1 − ens_mean_day1 (planned) | +0.749 [+0.519, +0.994]* | +0.687 [+0.477, +0.909]* | +0.747 [+0.530, +0.987]* | +0.771 [+0.540, +1.024]* | +1.003 [+0.680, +1.377]* | +0.896 [+0.603, +1.211]* | +0.060 [-0.008, +0.130] | +0.084 [-0.014, +0.168] | +0.144 [+0.014, +0.293]* |
| wind | ens_mean_day1 − ens_control_day1 (planned) | -0.399 [-0.529, -0.261]* | -0.381 [-0.508, -0.246]* | -0.343 [-0.458, -0.224]* | -0.342 [-0.445, -0.239]* | -0.309 [-0.422, -0.192]* | -0.271 [-0.388, -0.153]* | +0.038 [-0.039, +0.117] | +0.038 [-0.036, +0.124] | +0.079 [+0.005, +0.169]* |
| wind | ens_mean_day7 − climatology (planned) | -1.593 [-2.463, -0.804]* | -1.805 [-2.920, -0.886]* | -1.610 [-2.822, -0.628]* | -1.555 [-2.791, -0.578]* | -1.071 [-1.913, -0.345]* | -1.113 [-1.987, -0.313]* | +0.194 [-0.180, +0.610] | +0.250 [-0.134, +0.665] | +0.571 [+0.061, +1.116]* |
| wind | ens_mean_day0 − era5 (exploratory) | +0.170 [+0.021, +0.322]* | +0.136 [-0.012, +0.291] | +0.001 [-0.119, +0.132] | -0.037 [-0.167, +0.102] | -0.071 [-0.218, +0.084] | -0.026 [-0.151, +0.114] | -0.135 [-0.257, -0.031]* | -0.173 [-0.304, -0.049]* | -0.135 [-0.241, -0.043]* |
| wind | ens_mean_day1 − era5 (exploratory) | +0.925 [+0.730, +1.140]* | +0.923 [+0.733, +1.137]* | +0.861 [+0.679, +1.077]* | +0.858 [+0.665, +1.078]* | +0.789 [+0.575, +1.010]* | +0.878 [+0.671, +1.115]* | -0.062 [-0.147, +0.013] | -0.065 [-0.144, +0.019] | -0.044 [-0.116, +0.031] |
| wind | ens_mean_day0 − live_gb_rich_xgb (exploratory) | +1.449 [+1.244, +1.672]* | +1.458 [+1.246, +1.684]* | +1.294 [+1.142, +1.444]* | +1.280 [+1.134, +1.428]* | +1.270 [+1.123, +1.422]* | +1.251 [+1.102, +1.420]* | -0.164 [-0.281, -0.054]* | -0.179 [-0.318, -0.043]* | -0.140 [-0.306, -0.000]* |
| wind | ens_mean_day1 − live_gb_rich_xgb (exploratory) | +2.204 [+1.926, +2.492]* | +2.244 [+1.979, +2.507]* | +2.154 [+1.915, +2.394]* | +2.174 [+1.928, +2.421]* | +2.130 [+1.896, +2.370]* | +2.155 [+1.908, +2.435]* | -0.091 [-0.182, -0.004]* | -0.070 [-0.156, +0.016] | -0.048 [-0.162, +0.060] |
| wind | ens_control_day0 − era5 (exploratory) | +0.428 [+0.244, +0.620]* | +0.397 [+0.199, +0.597]* | +0.198 [+0.081, +0.325]* | +0.148 [+0.036, +0.274]* | +0.090 [-0.032, +0.212] | +0.134 [+0.017, +0.268]* | -0.199 [-0.346, -0.071]* | -0.250 [-0.428, -0.093]* | -0.186 [-0.313, -0.074]* |
| wind | live_gb_rich_xgb − era5 (exploratory) | -1.279 [-1.457, -1.086]* | -1.322 [-1.485, -1.136]* | -1.293 [-1.465, -1.107]* | -1.316 [-1.498, -1.126]* | -1.341 [-1.509, -1.140]* | -1.277 [-1.439, -1.087]* | +0.029 [-0.049, +0.111] | +0.005 [-0.067, +0.084] | +0.004 [-0.070, +0.087] |
| wind | ens_members_day1 − ens_control_day1 (exploratory) | +0.350 [+0.104, +0.607]* | +0.306 [+0.066, +0.548]* | +0.404 [+0.166, +0.655]* | +0.429 [+0.179, +0.695]* | +0.694 [+0.354, +1.064]* | +0.625 [+0.317, +0.972]* | +0.098 [+0.006, +0.192]* | +0.122 [+0.023, +0.227]* | +0.223 [+0.065, +0.416]* |
| solar | ens_mean_day1 − ens_mean_day0 (planned) | +0.613 [+0.445, +0.789]* | +0.619 [+0.451, +0.800]* | +0.638 [+0.465, +0.818]* | +0.629 [+0.458, +0.815]* | +0.600 [+0.358, +0.841]* | +0.581 [+0.385, +0.778]* | +0.020 [-0.045, +0.086] | +0.010 [-0.046, +0.069] | +0.005 [-0.073, +0.088] |
| solar | ens_mean_day7 − ens_mean_day0 (planned) | +6.144 [+5.485, +6.851]* | +6.167 [+5.478, +6.892]* | +6.204 [+5.542, +6.897]* | +6.172 [+5.511, +6.856]* | +5.651 [+4.587, +6.746]* | +5.600 [+4.894, +6.357]* | +0.037 [-0.231, +0.326] | +0.004 [-0.287, +0.292] | -0.368 [-0.833, +0.124] |
| solar | ens_members_day1 − ens_mean_day1 (planned) | +0.476 [+0.335, +0.622]* | +0.497 [+0.357, +0.644]* | +0.546 [+0.410, +0.678]* | +0.530 [+0.399, +0.659]* | +0.521 [+0.233, +0.800]* | +0.504 [+0.315, +0.652]* | +0.050 [-0.024, +0.123] | +0.033 [-0.031, +0.093] | +0.104 [-0.014, +0.218] |
| solar | ens_mean_day1 − ens_control_day1 (planned) | -0.360 [-0.465, -0.262]* | -0.351 [-0.461, -0.244]* | -0.361 [-0.472, -0.244]* | -0.369 [-0.482, -0.256]* | -0.361 [-0.531, -0.177]* | -0.319 [-0.441, -0.173]* | -0.010 [-0.058, +0.041] | -0.018 [-0.068, +0.032] | -0.004 [-0.088, +0.083] |
| solar | ens_mean_day7 − climatology (planned) | -0.157 [-0.581, +0.238] | -0.124 [-0.565, +0.264] | -0.181 [-0.624, +0.264] | -0.208 [-0.662, +0.219] | -1.778 [-2.719, -0.890]* | -0.868 [-1.603, -0.188]* | -0.058 [-0.326, +0.247] | -0.084 [-0.362, +0.215] | -0.739 [-1.550, +0.021] |
| solar | ens_mean_day0 − era5 (exploratory) | -0.901 [-1.130, -0.684]* | -0.910 [-1.127, -0.695]* | -0.924 [-1.141, -0.718]* | -0.936 [-1.159, -0.716]* | -0.810 [-1.155, -0.489]* | -0.754 [-0.976, -0.545]* | -0.014 [-0.083, +0.056] | -0.025 [-0.099, +0.044] | +0.040 [-0.092, +0.165] |
| solar | ens_mean_day1 − era5 (exploratory) | -0.288 [-0.532, -0.051]* | -0.292 [-0.534, -0.048]* | -0.286 [-0.523, -0.036]* | -0.307 [-0.553, -0.043]* | -0.210 [-0.547, +0.094] | -0.174 [-0.420, +0.081] | +0.006 [-0.073, +0.072] | -0.015 [-0.103, +0.064] | +0.045 [-0.097, +0.185] |
| solar | ens_mean_day0 − live_gb_rich_xgb (exploratory) | +0.672 [+0.496, +0.876]* | +0.692 [+0.517, +0.897]* | +0.645 [+0.474, +0.828]* | +0.659 [+0.487, +0.852]* | +0.532 [+0.237, +0.812]* | +0.467 [+0.240, +0.669]* | -0.048 [-0.132, +0.039] | -0.033 [-0.123, +0.050] | -0.093 [-0.289, +0.075] |
| solar | ens_mean_day1 − live_gb_rich_xgb (exploratory) | +1.285 [+1.057, +1.551]* | +1.311 [+1.075, +1.579]* | +1.283 [+1.049, +1.545]* | +1.288 [+1.044, +1.564]* | +1.132 [+0.843, +1.383]* | +1.048 [+0.741, +1.323]* | -0.028 [-0.125, +0.054] | -0.023 [-0.117, +0.058] | -0.089 [-0.308, +0.100] |
| solar | ens_control_day0 − era5 (exploratory) | -0.644 [-0.872, -0.422]* | -0.654 [-0.876, -0.430]* | -0.651 [-0.870, -0.440]* | -0.662 [-0.882, -0.440]* | -0.660 [-0.944, -0.392]* | -0.539 [-0.749, -0.338]* | +0.002 [-0.059, +0.058] | -0.009 [-0.062, +0.042] | +0.047 [-0.073, +0.152] |
| solar | live_gb_rich_xgb − era5 (exploratory) | -1.574 [-1.877, -1.303]* | -1.603 [-1.901, -1.336]* | -1.569 [-1.846, -1.310]* | -1.595 [-1.866, -1.339]* | -1.342 [-1.576, -1.101]* | -1.221 [-1.420, -0.997]* | +0.034 [-0.048, +0.111] | +0.008 [-0.064, +0.080] | +0.134 [-0.030, +0.318] |
| solar | ens_members_day1 − ens_control_day1 (exploratory) | +0.116 [-0.008, +0.238] | +0.146 [+0.020, +0.276]* | +0.186 [+0.051, +0.318]* | +0.161 [+0.040, +0.285]* | +0.159 [-0.052, +0.376] | +0.185 [+0.055, +0.319]* | +0.040 [-0.035, +0.108] | +0.015 [-0.052, +0.075] | +0.101 [-0.010, +0.209] |

### Part C, second setting: planned contrasts and headline references

| Domain | Contrast | D0 (published) | D0trim | I (era cut, offsets A) | I2 (era cut, offsets B) | II (uncovered folds, superseded) | IIr (II with covering rotation 2) | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|---|
| wind | ens_mean_day1 − ens_mean_day0 (planned) | +0.756 [+0.610, +0.897]* | +0.796 [+0.655, +0.935]* | +0.824 [+0.699, +0.951]* | n/a | +0.859 [+0.732, +0.988]* | +0.906 [+0.791, +1.024]* | +0.029 [-0.028, +0.084] | n/a | +0.087 [+0.009, +0.168]* |
| wind | ens_mean_day7 − ens_mean_day0 (planned) | +9.001 [+8.033, +10.047]* | +9.008 [+8.018, +10.053]* | +9.088 [+8.098, +10.146]* | n/a | +9.090 [+8.003, +10.324]* | +9.165 [+8.135, +10.305]* | +0.080 [-0.185, +0.331] | n/a | +0.232 [+0.033, +0.441]* |
| wind | ens_members_day1 − ens_mean_day1 (planned) | +0.632 [+0.438, +0.817]* | +0.581 [+0.405, +0.756]* | +0.705 [+0.518, +0.892]* | n/a | +0.870 [+0.602, +1.148]* | +0.778 [+0.553, +1.013]* | +0.124 [+0.053, +0.189]* | n/a | +0.135 [+0.039, +0.250]* |
| wind | ens_mean_day1 − ens_control_day1 (planned) | -0.396 [-0.515, -0.275]* | -0.387 [-0.512, -0.257]* | -0.350 [-0.460, -0.246]* | n/a | -0.310 [-0.424, -0.199]* | -0.286 [-0.412, -0.170]* | +0.037 [-0.022, +0.106] | n/a | +0.067 [+0.004, +0.137]* |
| wind | ens_mean_day7 − climatology (planned) | -1.937 [-2.696, -1.249]* | -2.185 [-3.302, -1.287]* | -2.055 [-3.277, -1.052]* | n/a | -1.346 [-2.219, -0.565]* | -1.461 [-2.323, -0.679]* | +0.131 [-0.218, +0.519] | n/a | +0.539 [+0.131, +1.031]* |
| wind | ens_mean_day0 − era5 (exploratory) | +0.083 [-0.064, +0.239] | +0.089 [-0.062, +0.246] | -0.012 [-0.128, +0.110] | n/a | -0.104 [-0.244, +0.044] | -0.063 [-0.186, +0.076] | -0.101 [-0.209, -0.009]* | n/a | -0.072 [-0.165, +0.012] |
| wind | ens_mean_day1 − era5 (exploratory) | +0.840 [+0.666, +1.034]* | +0.884 [+0.708, +1.082]* | +0.812 [+0.654, +0.996]* | n/a | +0.755 [+0.565, +0.956]* | +0.843 [+0.651, +1.044]* | -0.072 [-0.154, -0.002]* | n/a | +0.015 [-0.072, +0.102] |
| wind | ens_mean_day0 − live_gb_rich_xgb (exploratory) | +1.401 [+1.205, +1.614]* | +1.407 [+1.214, +1.617]* | +1.262 [+1.113, +1.415]* | n/a | +1.250 [+1.087, +1.417]* | +1.192 [+1.026, +1.363]* | -0.145 [-0.252, -0.054]* | n/a | -0.143 [-0.277, -0.033]* |
| wind | ens_mean_day1 − live_gb_rich_xgb (exploratory) | +2.158 [+1.914, +2.411]* | +2.202 [+1.967, +2.440]* | +2.086 [+1.853, +2.311]* | n/a | +2.110 [+1.877, +2.351]* | +2.098 [+1.871, +2.339]* | -0.116 [-0.205, -0.034]* | n/a | -0.056 [-0.176, +0.050] |
| wind | ens_control_day0 − era5 (exploratory) | +0.333 [+0.141, +0.528]* | +0.345 [+0.139, +0.556]* | +0.187 [+0.072, +0.313]* | n/a | +0.071 [-0.032, +0.181] | +0.105 [-0.001, +0.234] | -0.157 [-0.304, -0.034]* | n/a | -0.113 [-0.252, -0.001]* |
| wind | live_gb_rich_xgb − era5 (exploratory) | -1.318 [-1.480, -1.145]* | -1.318 [-1.483, -1.141]* | -1.274 [-1.441, -1.098]* | n/a | -1.355 [-1.537, -1.129]* | -1.256 [-1.417, -1.061]* | +0.044 [-0.008, +0.098] | n/a | +0.071 [+0.007, +0.145]* |
| wind | ens_members_day1 − ens_control_day1 (exploratory) | +0.236 [+0.018, +0.447]* | +0.193 [-0.015, +0.394] | +0.355 [+0.164, +0.544]* | n/a | +0.559 [+0.298, +0.829]* | +0.492 [+0.263, +0.739]* | +0.161 [+0.065, +0.260]* | n/a | +0.202 [+0.083, +0.342]* |
| solar | ens_mean_day1 − ens_mean_day0 (planned) | +0.609 [+0.472, +0.758]* | +0.610 [+0.464, +0.762]* | +0.649 [+0.499, +0.820]* | n/a | n/a | +0.601 [+0.399, +0.803]* | +0.039 [-0.009, +0.089] | n/a | +0.014 [-0.067, +0.094] |
| solar | ens_mean_day7 − ens_mean_day0 (planned) | +5.839 [+5.205, +6.517]* | +5.852 [+5.190, +6.552]* | +5.965 [+5.314, +6.648]* | n/a | n/a | +5.596 [+4.840, +6.405]* | +0.113 [-0.108, +0.391] | n/a | -0.090 [-0.499, +0.361] |
| solar | ens_members_day1 − ens_mean_day1 (planned) | +0.391 [+0.257, +0.532]* | +0.391 [+0.251, +0.532]* | +0.465 [+0.317, +0.605]* | n/a | n/a | +0.481 [+0.282, +0.648]* | +0.074 [+0.004, +0.148]* | n/a | +0.159 [+0.027, +0.283]* |
| solar | ens_mean_day1 − ens_control_day1 (planned) | -0.326 [-0.423, -0.228]* | -0.323 [-0.421, -0.222]* | -0.336 [-0.438, -0.230]* | n/a | n/a | -0.332 [-0.437, -0.201]* | -0.013 [-0.047, +0.023] | n/a | -0.052 [-0.130, +0.024] |
| solar | ens_mean_day7 − climatology (planned) | -0.466 [-0.853, -0.094]* | -0.438 [-0.834, -0.072]* | -0.442 [-0.872, -0.015]* | n/a | n/a | -0.951 [-1.717, -0.201]* | -0.004 [-0.250, +0.293] | n/a | -0.540 [-1.367, +0.225] |
| solar | ens_mean_day0 − era5 (exploratory) | -0.882 [-1.112, -0.665]* | -0.880 [-1.105, -0.659]* | -0.899 [-1.110, -0.694]* | n/a | n/a | -0.784 [-0.983, -0.592]* | -0.019 [-0.079, +0.035] | n/a | +0.003 [-0.110, +0.119] |
| solar | ens_mean_day1 − era5 (exploratory) | -0.273 [-0.502, -0.034]* | -0.270 [-0.504, -0.032]* | -0.250 [-0.470, -0.011]* | n/a | n/a | -0.183 [-0.419, +0.062] | +0.020 [-0.042, +0.081] | n/a | +0.018 [-0.124, +0.134] |
| solar | ens_mean_day0 − live_gb_rich_xgb (exploratory) | +0.663 [+0.484, +0.872]* | +0.680 [+0.497, +0.895]* | +0.676 [+0.502, +0.880]* | n/a | n/a | +0.441 [+0.250, +0.604]* | -0.005 [-0.075, +0.062] | n/a | -0.117 [-0.259, +0.012] |
| solar | ens_mean_day1 − live_gb_rich_xgb (exploratory) | +1.272 [+1.041, +1.535]* | +1.290 [+1.051, +1.561]* | +1.325 [+1.090, +1.599]* | n/a | n/a | +1.042 [+0.751, +1.315]* | +0.035 [-0.042, +0.101] | n/a | -0.102 [-0.267, +0.040] |
| solar | ens_control_day0 − era5 (exploratory) | -0.646 [-0.861, -0.434]* | -0.642 [-0.860, -0.426]* | -0.646 [-0.862, -0.437]* | n/a | n/a | -0.565 [-0.758, -0.380]* | -0.004 [-0.054, +0.042] | n/a | +0.027 [-0.070, +0.129] |
| solar | live_gb_rich_xgb − era5 (exploratory) | -1.545 [-1.849, -1.268]* | -1.560 [-1.862, -1.289]* | -1.575 [-1.870, -1.314]* | n/a | n/a | -1.225 [-1.428, -1.010]* | -0.015 [-0.082, +0.051] | n/a | +0.120 [-0.033, +0.310] |
| solar | ens_members_day1 − ens_control_day1 (exploratory) | +0.065 [-0.067, +0.201] | +0.068 [-0.067, +0.206] | +0.129 [-0.013, +0.267] | n/a | n/a | +0.149 [-0.013, +0.302] | +0.062 [-0.008, +0.132] | n/a | +0.107 [+0.003, +0.206]* |

## Part A: coverage arithmetic (no fits)

A cell is a (site, fold, calendar month) that `uncovered_months` flags: no training row carries that calendar month and the month occurs in more than one year. 'Scored hours' are site-hours (one row each, not row x seed).

### Wind study (`wind_products.py` main frame, three wind farms W1-W3)

- `cut_eras` with zero offsets equals `with_eras` fold for fold on every (site, time): **True**.
- `search_fold_offsets(first_months=[UKV_UPGRADE_MONTH])` finds these covering offsets (era 1's rotation): [2, 3, 4]; uncovered hours by rotation of era 1: {'0': 8326, '1': 16411, '2': 0, '3': 0, '4': 0}. Design D1 uses rotation 2 (fewest non-zero rotations, then smallest sum).

| Design | Scored hours | Uncovered hours | Share | Calendar months uncovered | Post-upgrade hours scored by a model with no post-upgrade training row |
|---|---|---|---|---|---|
| D0 | 50,734 | 8,326 | 16.41% | [6, 7] | 0 of 15,140 |
| D1_rot0 | 50,734 | 8,326 | 16.41% | [6, 7] | 0 of 15,140 |
| D1_rot1 | 50,734 | 16,411 | 32.35% | [2, 3, 4, 5] | 0 of 15,140 |
| D1_rot2 | 50,734 | 0 | 0.00% | none | 0 of 15,140 |
| D1_rot3 | 50,734 | 0 | 0.00% | none | 0 of 15,140 |
| D1_rot4 | 50,734 | 0 | 0.00% | none | 0 of 15,140 |
| D2 | 50,734 | 0 | 0.00% | none | 0 of 15,140 |

Per site (D0 and D2):

| Site | Hours | D0 uncovered hours (share) | D0 months | D2 uncovered hours |
|---|---|---|---|---|
| W1 | 16,886 | 2,774 (16.4%) | [6, 7] | 0 |
| W2 | 17,543 | 2,897 (16.5%) | [6, 7] | 0 |
| W3 | 16,305 | 2,655 (16.3%) | [6, 7] | 0 |

### Solar study (`weather_products.py` `long` panel frame, six solar farms A-F, eight products)

- `cut_eras` with zero offsets equals `with_eras` fold for fold on every (site, time): **True**.
- `search_fold_offsets(first_months=[UKV_UPGRADE_MONTH])` finds these covering offsets (era 1's rotation): [2, 3, 4]; uncovered hours by rotation of era 1: {'0': 2129, '1': 1782, '2': 0, '3': 0, '4': 0}. Design D1 uses rotation 2 (fewest non-zero rotations, then smallest sum).

| Design | Scored hours | Uncovered hours | Share | Calendar months uncovered | Post-upgrade hours scored by a model with no post-upgrade training row |
|---|---|---|---|---|---|
| D0 | 76,727 | 2,129 | 2.77% | [5, 6, 7] | 0 of 15,480 |
| D1_rot0 | 76,727 | 2,129 | 2.77% | [5, 6, 7] | 0 of 15,480 |
| D1_rot1 | 76,727 | 1,782 | 2.32% | [2, 3, 4] | 0 of 15,480 |
| D1_rot2 | 76,727 | 0 | 0.00% | none | 0 of 15,480 |
| D1_rot3 | 76,727 | 0 | 0.00% | none | 0 of 15,480 |
| D1_rot4 | 76,727 | 0 | 0.00% | none | 0 of 15,480 |
| D2 | 76,727 | 0 | 0.00% | none | 13,218 of 15,480 |

Per site (D0 and D2):

| Site | Hours | D0 uncovered hours (share) | D0 months | D2 uncovered hours |
|---|---|---|---|---|
| A | 14,301 | 0 (0.0%) | none | 0 |
| B | 15,534 | 0 (0.0%) | none | 0 |
| C | 15,825 | 0 (0.0%) | none | 0 |
| D | 10,039 | 0 (0.0%) | none | 0 |
| E | 6,007 | 2,129 (35.4%) | [5, 6, 7] | 0 |
| F | 15,021 | 0 (0.0%) | none | 0 |

## Part B: reproduction and details

Reproduction of the published per-row losses under D0 (same arms, same setting, joined on arm, site, time, seed):

| Study / setting | Rows (new, published) | Unmatched | Share bit-identical | Max abs difference (fraction of capacity) |
|---|---|---|---|---|
| wind/pooled | 608,808, 608,808 | 0 | 1.000000 | 0 |
| wind/sensitivity | 608,808, 608,808 | 0 | 1.000000 | 0 |
| solar/pooled | 1,841,448, 1,841,448 | 0 | 1.000000 | 0 |
| solar/sensitivity | 1,841,448, 1,841,448 | 0 | 1.000000 | 0 |

Absolute mean absolute error (percent of capacity) of every fitted arm, with 95% interval:

| Study | Setting | Arm | D0 | D1 | D2 |
|---|---|---|---|---|---|
| wind | pooled | era5_wind | 7.272 [6.65, 7.97] | 7.249 [6.62, 7.98] | 7.343 [6.71, 8.06] |
| wind | pooled | icon_d2_wind | 6.694 [5.95, 7.55] | 6.676 [5.91, 7.54] | 6.735 [6.02, 7.53] |
| wind | pooled | icon_eu_wind | 6.958 [6.24, 7.74] | 6.951 [6.23, 7.75] | 7.015 [6.31, 7.80] |
| wind | pooled | ukv_wind | 6.832 [6.08, 7.68] | 6.842 [6.08, 7.69] | 6.974 [6.22, 7.81] |
| wind | sensitivity | era5_wind | 7.310 [6.66, 8.07] | 7.265 [6.61, 8.03] | 7.317 [6.68, 8.03] |
| wind | sensitivity | icon_d2_wind | 6.668 [5.90, 7.57] | 6.659 [5.87, 7.57] | 6.704 [5.97, 7.54] |
| wind | sensitivity | icon_eu_wind | 6.911 [6.19, 7.72] | 6.917 [6.18, 7.75] | 6.966 [6.24, 7.76] |
| wind | sensitivity | ukv_wind | 6.829 [6.05, 7.70] | 6.836 [6.06, 7.69] | 6.967 [6.19, 7.83] |
| solar | pooled | cams_global | 5.085 [4.84, 5.31] | 5.054 [4.82, 5.28] | 5.111 [4.87, 5.33] |
| solar | pooled | era5_global | 9.080 [8.58, 9.52] | 9.032 [8.53, 9.47] | 9.107 [8.60, 9.55] |
| solar | pooled | icon_d2_global | 7.763 [7.39, 8.09] | 7.763 [7.40, 8.09] | 7.793 [7.43, 8.11] |
| solar | pooled | icon_dream_global | 8.765 [8.32, 9.15] | 8.745 [8.30, 9.13] | 8.776 [8.35, 9.15] |
| solar | pooled | icon_eu_global | 8.386 [7.96, 8.77] | 8.353 [7.94, 8.73] | 8.394 [7.97, 8.77] |
| solar | pooled | icon_global_global | 8.483 [8.07, 8.85] | 8.446 [8.04, 8.81] | 8.511 [8.10, 8.87] |
| solar | pooled | sarah3_global | 5.487 [5.19, 5.76] | 5.454 [5.17, 5.73] | 5.506 [5.23, 5.77] |
| solar | pooled | ukv_global | 8.862 [8.40, 9.26] | 8.843 [8.40, 9.24] | 9.047 [8.60, 9.44] |
| solar | sensitivity | cams_global | 5.105 [4.88, 5.33] | 5.078 [4.84, 5.30] | 5.119 [4.89, 5.34] |
| solar | sensitivity | era5_global | 9.041 [8.55, 9.48] | 8.996 [8.50, 9.44] | 9.058 [8.57, 9.49] |
| solar | sensitivity | icon_d2_global | 7.754 [7.38, 8.08] | 7.737 [7.37, 8.06] | 7.772 [7.41, 8.09] |
| solar | sensitivity | icon_dream_global | 8.714 [8.28, 9.10] | 8.676 [8.26, 9.05] | 8.717 [8.29, 9.10] |
| solar | sensitivity | icon_eu_global | 8.360 [7.95, 8.73] | 8.320 [7.92, 8.69] | 8.370 [7.96, 8.73] |
| solar | sensitivity | icon_global_global | 8.454 [8.05, 8.82] | 8.410 [8.01, 8.77] | 8.474 [8.07, 8.83] |
| solar | sensitivity | sarah3_global | 5.527 [5.24, 5.81] | 5.496 [5.21, 5.77] | 5.536 [5.25, 5.80] |
| solar | sensitivity | ukv_global | 8.793 [8.35, 9.18] | 8.778 [8.34, 9.17] | 8.960 [8.52, 9.33] |

## Part C: absolute error of the fitted arms

| Domain | Setting | Arm | D0 | D0trim | I | I2 | II (uncovered) | IIr |
|---|---|---|---|---|---|---|---|---|
| wind | pooled | ens_mean_day0 | 7.405 [6.73, 8.20] | 7.365 [6.65, 8.17] | 7.158 [6.47, 7.94] | 7.107 [6.43, 7.88] | 7.383 [6.62, 8.29] | 7.332 [6.55, 8.25] |
| wind | pooled | ens_mean_day1 | 8.159 [7.47, 8.95] | 8.152 [7.43, 8.97] | 8.017 [7.31, 8.82] | 8.002 [7.30, 8.79] | 8.243 [7.47, 9.15] | 8.236 [7.46, 9.16] |
| wind | pooled | ens_mean_day2 | 9.400 [8.63, 10.21] | 9.332 [8.55, 10.20] | 9.316 [8.51, 10.20] | 9.242 [8.45, 10.12] | 9.584 [8.67, 10.67] | 9.604 [8.74, 10.65] |
| wind | pooled | ens_mean_day3 | 10.997 [10.05, 12.01] | 10.971 [9.99, 12.05] | 10.947 [9.96, 12.03] | 10.943 [9.94, 12.03] | 11.125 [10.05, 12.40] | 11.210 [10.15, 12.44] |
| wind | pooled | ens_mean_day5 | 14.331 [13.15, 15.45] | 14.219 [13.03, 15.42] | 14.165 [12.99, 15.32] | 14.215 [13.03, 15.37] | 14.024 [12.85, 15.30] | 14.217 [13.00, 15.55] |
| wind | pooled | ens_mean_day7 | 16.702 [15.24, 18.18] | 16.682 [15.19, 18.22] | 16.636 [15.15, 18.17] | 16.751 [15.27, 18.30] | 16.678 [15.11, 18.38] | 16.789 [15.23, 18.51] |
| wind | pooled | ens_mean_day10 | 18.383 [16.59, 20.19] | 18.341 [16.55, 20.17] | 17.968 [16.17, 19.79] | 17.960 [16.21, 19.75] | 17.951 [15.96, 20.07] | 17.927 [16.01, 19.98] |
| wind | pooled | ens_mean_day14 | 19.462 [17.57, 21.35] | 19.365 [17.52, 21.30] | 18.864 [17.14, 20.71] | 19.003 [17.27, 20.81] | 19.152 [17.26, 21.10] | 19.109 [17.17, 21.17] |
| wind | pooled | ens_control_day0 | 7.662 [6.97, 8.45] | 7.626 [6.90, 8.44] | 7.355 [6.66, 8.13] | 7.292 [6.62, 8.05] | 7.544 [6.79, 8.42] | 7.491 [6.73, 8.40] |
| wind | pooled | ens_control_day1 | 8.558 [7.82, 9.32] | 8.532 [7.77, 9.34] | 8.360 [7.63, 9.17] | 8.344 [7.63, 9.13] | 8.552 [7.76, 9.45] | 8.507 [7.74, 9.40] |
| wind | pooled | ens_control_day2 | 9.999 [9.06, 10.93] | 9.885 [8.99, 10.84] | 9.779 [8.88, 10.75] | 9.696 [8.81, 10.66] | 10.018 [9.02, 11.20] | 10.002 [9.02, 11.15] |
| wind | pooled | ens_control_day3 | 12.093 [10.94, 13.26] | 12.000 [10.86, 13.19] | 11.881 [10.75, 13.05] | 11.931 [10.80, 13.12] | 12.025 [10.80, 13.44] | 12.151 [10.92, 13.55] |
| wind | pooled | ens_control_day5 | 15.807 [14.39, 17.17] | 15.642 [14.24, 17.00] | 15.438 [14.09, 16.75] | 15.577 [14.26, 16.87] | 15.389 [14.01, 16.90] | 15.634 [14.25, 17.19] |
| wind | pooled | ens_control_day7 | 17.548 [16.06, 19.01] | 17.580 [16.08, 19.07] | 17.592 [16.12, 19.05] | 17.615 [16.21, 19.06] | 18.277 [16.58, 19.88] | 17.590 [16.06, 19.21] |
| wind | pooled | ens_control_day10 | 18.965 [17.28, 20.59] | 18.939 [17.24, 20.60] | 18.462 [16.80, 20.14] | 18.441 [16.81, 20.13] | 19.100 [17.16, 21.06] | 18.491 [16.60, 20.45] |
| wind | pooled | ens_control_day14 | 19.211 [17.51, 20.84] | 19.204 [17.51, 20.91] | 18.724 [16.99, 20.50] | 18.955 [17.21, 20.74] | 19.331 [17.41, 21.21] | 18.812 [16.86, 20.76] |
| wind | pooled | era5 | 7.234 [6.64, 7.93] | 7.229 [6.59, 7.99] | 7.156 [6.54, 7.87] | 7.144 [6.55, 7.84] | 7.454 [6.79, 8.23] | 7.357 [6.67, 8.17] |
| wind | pooled | live_gb_rich_xgb | 5.955 [5.24, 6.78] | 5.907 [5.16, 6.77] | 5.863 [5.15, 6.70] | 5.828 [5.13, 6.65] | 6.113 [5.36, 7.07] | 6.081 [5.28, 7.04] |
| wind | pooled | climatology | 18.295 [16.66, 19.94] | 18.487 [16.63, 20.42] | 18.246 [16.30, 20.29] | 18.306 [16.39, 20.34] | 17.748 [15.94, 19.74] | 17.902 [16.09, 19.86] |
| wind | pooled | calendar_only | 19.747 [17.89, 21.58] | 19.630 [17.80, 21.45] | 19.061 [17.18, 20.99] | 19.298 [17.43, 21.24] | 20.063 [17.67, 22.61] | 19.315 [17.18, 21.46] |
| wind | pooled | ens_members_day1 | 8.908 [8.08, 9.80] | 8.838 [7.99, 9.76] | 8.764 [7.92, 9.70] | 8.773 [7.94, 9.71] | 9.246 [8.25, 10.42] | 9.131 [8.17, 10.23] |
| wind | pooled | smart_persistence_day0 | 15.034 [13.85, 16.20] | 15.004 [13.81, 16.26] | 14.981 [13.73, 16.26] | 15.033 [13.79, 16.29] | 14.916 [13.70, 16.24] | 14.952 [13.73, 16.28] |
| wind | pooled | smart_persistence_day1 | 17.545 [16.05, 19.01] | 17.690 [16.05, 19.35] | 17.564 [15.80, 19.37] | 17.589 [15.84, 19.35] | 17.256 [15.56, 19.09] | 17.364 [15.71, 19.15] |
| wind | pooled | smart_persistence_day10 | 18.267 [16.62, 19.92] | 18.462 [16.60, 20.41] | 18.218 [16.28, 20.25] | 18.280 [16.37, 20.29] | 17.745 [15.93, 19.74] | 17.902 [16.09, 19.86] |
| wind | pooled | smart_persistence_day14 | 18.269 [16.63, 19.91] | 18.449 [16.62, 20.37] | 18.209 [16.28, 20.24] | 18.275 [16.38, 20.28] | 17.739 [15.93, 19.73] | 17.891 [16.08, 19.84] |
| wind | pooled | smart_persistence_day2 | 18.143 [16.54, 19.76] | 18.344 [16.54, 20.24] | 18.143 [16.22, 20.17] | 18.181 [16.31, 20.17] | 17.708 [15.91, 19.69] | 17.843 [16.03, 19.79] |
| wind | pooled | smart_persistence_day3 | 18.254 [16.62, 19.91] | 18.442 [16.61, 20.36] | 18.230 [16.29, 20.28] | 18.263 [16.36, 20.28] | 17.746 [15.93, 19.73] | 17.895 [16.08, 19.85] |
| wind | sensitivity | ens_mean_day0 | 7.357 [6.68, 8.16] | 7.294 [6.58, 8.11] | 7.103 [6.42, 7.90] | n/a | 7.312 [6.55, 8.23] | 7.276 [6.52, 8.19] |
| wind | sensitivity | ens_mean_day1 | 8.113 [7.44, 8.89] | 8.089 [7.39, 8.92] | 7.928 [7.24, 8.72] | n/a | 8.172 [7.41, 9.09] | 8.182 [7.41, 9.10] |
| wind | sensitivity | ens_mean_day7 | 16.358 [15.04, 17.70] | 16.302 [14.91, 17.71] | 16.192 [14.83, 17.62] | n/a | 16.402 [14.83, 18.10] | 16.441 [14.96, 18.08] |
| wind | sensitivity | ens_mean_day10 | 17.985 [16.30, 19.66] | 17.987 [16.24, 19.76] | 17.629 [15.89, 19.43] | n/a | 17.801 [15.89, 19.88] | 17.665 [15.79, 19.69] |
| wind | sensitivity | ens_mean_day14 | 19.062 [17.17, 20.95] | 19.003 [17.14, 20.92] | 18.491 [16.80, 20.31] | n/a | 19.078 [17.26, 20.96] | 18.677 [16.75, 20.68] |
| wind | sensitivity | ens_control_day0 | 7.607 [6.91, 8.40] | 7.550 [6.83, 8.37] | 7.303 [6.63, 8.09] | n/a | 7.488 [6.75, 8.36] | 7.445 [6.68, 8.35] |
| wind | sensitivity | ens_control_day1 | 8.509 [7.79, 9.29] | 8.477 [7.73, 9.30] | 8.278 [7.55, 9.07] | n/a | 8.482 [7.70, 9.39] | 8.469 [7.68, 9.40] |
| wind | sensitivity | ens_control_day10 | 18.507 [16.78, 20.20] | 18.527 [16.77, 20.28] | 18.167 [16.52, 19.89] | n/a | 18.688 [16.77, 20.59] | 18.096 [16.18, 20.07] |
| wind | sensitivity | ens_control_day14 | 18.819 [17.06, 20.50] | 18.894 [17.12, 20.65] | 18.252 [16.56, 19.96] | n/a | 18.855 [17.02, 20.65] | 18.266 [16.38, 20.16] |
| wind | sensitivity | era5 | 7.274 [6.65, 8.02] | 7.205 [6.58, 7.96] | 7.115 [6.49, 7.85] | n/a | 7.417 [6.74, 8.21] | 7.340 [6.63, 8.17] |
| wind | sensitivity | live_gb_rich_xgb | 5.956 [5.22, 6.80] | 5.887 [5.14, 6.75] | 5.841 [5.13, 6.67] | n/a | 6.062 [5.30, 7.04] | 6.084 [5.28, 7.06] |
| wind | sensitivity | climatology | 18.295 [16.66, 19.94] | 18.487 [16.63, 20.42] | 18.246 [16.30, 20.29] | n/a | 17.748 [15.94, 19.74] | 17.902 [16.09, 19.86] |
| wind | sensitivity | calendar_only | 19.132 [17.34, 20.92] | 19.085 [17.31, 20.91] | 18.465 [16.72, 20.24] | n/a | 18.993 [16.95, 21.08] | 18.411 [16.42, 20.47] |
| wind | sensitivity | ens_members_day1 | 8.745 [7.97, 9.59] | 8.670 [7.87, 9.53] | 8.632 [7.85, 9.49] | n/a | 9.041 [8.14, 10.10] | 8.960 [8.05, 10.01] |
| wind | sensitivity | smart_persistence_day0 | 15.034 [13.85, 16.20] | 15.004 [13.81, 16.26] | 14.981 [13.73, 16.26] | n/a | 14.916 [13.70, 16.24] | 14.952 [13.73, 16.28] |
| wind | sensitivity | smart_persistence_day1 | 17.545 [16.05, 19.01] | 17.690 [16.05, 19.35] | 17.564 [15.80, 19.37] | n/a | 17.256 [15.56, 19.09] | 17.364 [15.71, 19.15] |
| wind | sensitivity | smart_persistence_day10 | 18.267 [16.62, 19.92] | 18.462 [16.60, 20.41] | 18.218 [16.28, 20.25] | n/a | 17.745 [15.93, 19.74] | 17.902 [16.09, 19.86] |
| wind | sensitivity | smart_persistence_day14 | 18.269 [16.63, 19.91] | 18.449 [16.62, 20.37] | 18.209 [16.28, 20.24] | n/a | 17.739 [15.93, 19.73] | 17.891 [16.08, 19.84] |
| wind | sensitivity | smart_persistence_day2 | 18.143 [16.54, 19.76] | 18.344 [16.54, 20.24] | 18.143 [16.22, 20.17] | n/a | 17.708 [15.91, 19.69] | 17.843 [16.03, 19.79] |
| wind | sensitivity | smart_persistence_day3 | 18.254 [16.62, 19.91] | 18.442 [16.61, 20.36] | 18.230 [16.29, 20.28] | n/a | 17.746 [15.93, 19.73] | 17.895 [16.08, 19.85] |
| solar | pooled | ens_mean_day0 | 8.155 [7.68, 8.60] | 8.220 [7.76, 8.64] | 8.146 [7.69, 8.57] | 8.153 [7.69, 8.58] | 8.268 [7.55, 8.84] | 8.066 [7.39, 8.58] |
| solar | pooled | ens_mean_day1 | 8.768 [8.29, 9.25] | 8.839 [8.38, 9.30] | 8.784 [8.32, 9.24] | 8.782 [8.31, 9.24] | 8.868 [8.14, 9.40] | 8.647 [7.93, 9.19] |
| solar | pooled | ens_mean_day2 | 9.761 [9.19, 10.32] | 9.878 [9.30, 10.42] | 9.824 [9.29, 10.36] | 9.803 [9.27, 10.33] | 9.872 [9.06, 10.48] | 9.620 [8.78, 10.23] |
| solar | pooled | ens_mean_day3 | 10.873 [10.19, 11.54] | 10.993 [10.33, 11.63] | 10.928 [10.27, 11.57] | 10.934 [10.26, 11.59] | 10.885 [10.00, 11.59] | 10.568 [9.63, 11.28] |
| solar | pooled | ens_mean_day5 | 13.048 [12.25, 13.87] | 13.154 [12.37, 13.94] | 13.169 [12.37, 13.98] | 13.133 [12.34, 13.94] | 12.697 [11.72, 13.57] | 12.509 [11.52, 13.31] |
| solar | pooled | ens_mean_day7 | 14.299 [13.38, 15.27] | 14.388 [13.46, 15.34] | 14.350 [13.43, 15.27] | 14.325 [13.40, 15.25] | 13.919 [12.71, 15.13] | 13.666 [12.59, 14.62] |
| solar | pooled | ens_mean_day10 | 14.946 [13.97, 15.91] | 15.043 [14.05, 16.02] | 14.855 [13.91, 15.74] | 14.814 [13.86, 15.68] | 14.792 [13.44, 16.13] | 14.252 [13.14, 15.27] |
| solar | pooled | ens_mean_day14 | 15.230 [14.22, 16.19] | 15.336 [14.30, 16.30] | 15.119 [14.19, 15.97] | 15.094 [14.18, 15.92] | 14.955 [13.65, 16.19] | 14.720 [13.51, 15.69] |
| solar | pooled | ens_control_day0 | 8.412 [7.91, 8.89] | 8.477 [7.99, 8.93] | 8.419 [7.93, 8.88] | 8.426 [7.94, 8.88] | 8.418 [7.71, 8.96] | 8.282 [7.59, 8.80] |
| solar | pooled | ens_control_day1 | 9.128 [8.62, 9.63] | 9.190 [8.67, 9.68] | 9.145 [8.62, 9.65] | 9.151 [8.62, 9.66] | 9.230 [8.47, 9.81] | 8.966 [8.21, 9.54] |
| solar | pooled | ens_control_day2 | 10.286 [9.69, 10.85] | 10.394 [9.82, 10.94] | 10.418 [9.82, 10.97] | 10.390 [9.79, 10.94] | 10.390 [9.49, 11.11] | 10.194 [9.28, 10.90] |
| solar | pooled | ens_control_day3 | 11.532 [10.81, 12.23] | 11.637 [10.94, 12.30] | 11.646 [10.91, 12.34] | 11.588 [10.87, 12.28] | 11.547 [10.65, 12.32] | 11.205 [10.31, 11.92] |
| solar | pooled | ens_control_day5 | 13.940 [13.07, 14.78] | 14.036 [13.14, 14.86] | 13.931 [13.05, 14.76] | 13.912 [13.03, 14.73] | 13.342 [12.27, 14.24] | 13.250 [12.20, 14.11] |
| solar | pooled | ens_control_day7 | 14.792 [13.88, 15.68] | 14.903 [13.98, 15.78] | 14.772 [13.85, 15.63] | 14.715 [13.79, 15.55] | 14.686 [13.48, 15.85] | 14.266 [13.18, 15.19] |
| solar | pooled | ens_control_day10 | 15.440 [14.43, 16.41] | 15.531 [14.49, 16.51] | 15.177 [14.23, 16.08] | 15.181 [14.25, 16.07] | 15.082 [13.80, 16.32] | 14.807 [13.55, 15.84] |
| solar | pooled | ens_control_day14 | 15.377 [14.35, 16.35] | 15.465 [14.44, 16.45] | 15.244 [14.31, 16.10] | 15.236 [14.31, 16.07] | 15.147 [13.83, 16.42] | 14.832 [13.59, 15.82] |
| solar | pooled | era5 | 9.056 [8.51, 9.60] | 9.131 [8.59, 9.64] | 9.070 [8.52, 9.57] | 9.089 [8.54, 9.60] | 9.078 [8.34, 9.68] | 8.820 [8.09, 9.38] |
| solar | pooled | live_gb_rich_xgb | 7.482 [7.09, 7.83] | 7.528 [7.15, 7.87] | 7.501 [7.13, 7.84] | 7.494 [7.12, 7.83] | 7.736 [7.16, 8.21] | 7.599 [7.01, 8.09] |
| solar | pooled | climatology | 14.456 [13.52, 15.44] | 14.511 [13.57, 15.48] | 14.531 [13.59, 15.50] | 14.533 [13.59, 15.50] | 15.697 [14.01, 17.27] | 14.534 [13.15, 15.72] |
| solar | pooled | calendar_only | 15.440 [14.39, 16.42] | 15.545 [14.52, 16.52] | 15.258 [14.29, 16.09] | 15.200 [14.27, 16.01] | 14.970 [13.68, 16.20] | 14.755 [13.58, 15.65] |
| solar | pooled | ens_members_day1 | 9.244 [8.70, 9.75] | 9.336 [8.83, 9.83] | 9.330 [8.80, 9.82] | 9.312 [8.78, 9.80] | 9.389 [8.64, 9.99] | 9.151 [8.38, 9.72] |
| solar | sensitivity | ens_mean_day0 | 8.151 [7.70, 8.59] | 8.221 [7.78, 8.64] | 8.124 [7.67, 8.54] | n/a | n/a | 7.987 [7.32, 8.50] |
| solar | sensitivity | ens_mean_day1 | 8.761 [8.31, 9.22] | 8.831 [8.38, 9.28] | 8.773 [8.31, 9.23] | n/a | n/a | 8.588 [7.89, 9.13] |
| solar | sensitivity | ens_mean_day5 | 12.804 [12.03, 13.61] | 12.888 [12.11, 13.68] | 12.965 [12.16, 13.81] | n/a | n/a | 12.390 [11.38, 13.24] |
| solar | sensitivity | ens_mean_day7 | 13.990 [13.11, 14.93] | 14.074 [13.17, 15.02] | 14.089 [13.20, 15.02] | n/a | n/a | 13.583 [12.49, 14.60] |
| solar | sensitivity | ens_mean_day10 | 14.689 [13.72, 15.68] | 14.758 [13.79, 15.74] | 14.657 [13.73, 15.60] | n/a | n/a | 14.156 [13.02, 15.21] |
| solar | sensitivity | ens_mean_day14 | 15.067 [14.06, 16.04] | 15.153 [14.16, 16.12] | 14.970 [14.04, 15.86] | n/a | n/a | 14.614 [13.43, 15.61] |
| solar | sensitivity | ens_control_day0 | 8.388 [7.90, 8.85] | 8.459 [8.00, 8.90] | 8.377 [7.90, 8.82] | n/a | n/a | 8.205 [7.53, 8.71] |
| solar | sensitivity | ens_control_day1 | 9.087 [8.60, 9.57] | 9.154 [8.67, 9.63] | 9.109 [8.61, 9.61] | n/a | n/a | 8.920 [8.19, 9.49] |
| solar | sensitivity | ens_control_day10 | 15.182 [14.18, 16.16] | 15.256 [14.23, 16.23] | 15.016 [14.09, 15.91] | n/a | n/a | 14.689 [13.46, 15.73] |
| solar | sensitivity | ens_control_day14 | 15.216 [14.20, 16.19] | 15.283 [14.30, 16.26] | 15.113 [14.18, 15.98] | n/a | n/a | 14.764 [13.50, 15.79] |
| solar | sensitivity | era5 | 9.033 [8.50, 9.56] | 9.101 [8.58, 9.59] | 9.023 [8.49, 9.53] | n/a | n/a | 8.771 [8.07, 9.32] |
| solar | sensitivity | live_gb_rich_xgb | 7.488 [7.09, 7.84] | 7.541 [7.17, 7.87] | 7.448 [7.09, 7.78] | n/a | n/a | 7.546 [6.98, 8.00] |
| solar | sensitivity | climatology | 14.456 [13.52, 15.44] | 14.511 [13.57, 15.48] | 14.531 [13.59, 15.50] | n/a | n/a | 14.534 [13.15, 15.72] |
| solar | sensitivity | calendar_only | 15.171 [14.16, 16.13] | 15.254 [14.25, 16.21] | 15.040 [14.11, 15.89] | n/a | n/a | 14.656 [13.45, 15.63] |
| solar | sensitivity | ens_members_day1 | 9.152 [8.64, 9.66] | 9.222 [8.72, 9.72] | 9.238 [8.73, 9.75] | n/a | n/a | 9.069 [8.32, 9.65] |

## Part C: every contrast, primary setting

### wind: planned

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day1 − ens_mean_day0 | +0.754 [+0.564, +0.941]* | +0.786 [+0.617, +0.958]* | +0.859 [+0.711, +1.010]* | +0.895 [+0.749, +1.049]* | +0.860 [+0.718, +1.008]* | +0.904 [+0.760, +1.055]* | +0.073 [-0.006, +0.141] | +0.108 [+0.020, +0.200]* | +0.092 [-0.005, +0.195] |
| ens_mean_day7 − ens_mean_day0 | +9.298 [+8.204, +10.433]* | +9.317 [+8.241, +10.484]* | +9.478 [+8.428, +10.592]* | +9.643 [+8.589, +10.782]* | +9.295 [+8.259, +10.497]* | +9.458 [+8.390, +10.667]* | +0.161 [-0.123, +0.450] | +0.326 [+0.012, +0.684]* | +0.263 [-0.084, +0.611] |
| ens_members_day1 − ens_mean_day1 | +0.749 [+0.519, +0.994]* | +0.687 [+0.477, +0.909]* | +0.747 [+0.530, +0.987]* | +0.771 [+0.540, +1.024]* | +1.003 [+0.680, +1.377]* | +0.896 [+0.603, +1.211]* | +0.060 [-0.008, +0.130] | +0.084 [-0.014, +0.168] | +0.144 [+0.014, +0.293]* |
| ens_mean_day1 − ens_control_day1 | -0.399 [-0.529, -0.261]* | -0.381 [-0.508, -0.246]* | -0.343 [-0.458, -0.224]* | -0.342 [-0.445, -0.239]* | -0.309 [-0.422, -0.192]* | -0.271 [-0.388, -0.153]* | +0.038 [-0.039, +0.117] | +0.038 [-0.036, +0.124] | +0.079 [+0.005, +0.169]* |
| ens_mean_day7 − climatology | -1.593 [-2.463, -0.804]* | -1.805 [-2.920, -0.886]* | -1.610 [-2.822, -0.628]* | -1.555 [-2.791, -0.578]* | -1.071 [-1.913, -0.345]* | -1.113 [-1.987, -0.313]* | +0.194 [-0.180, +0.610] | +0.250 [-0.134, +0.665] | +0.571 [+0.061, +1.116]* |

### wind: reference

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − era5 | +0.170 [+0.021, +0.322]* | +0.136 [-0.012, +0.291] | +0.001 [-0.119, +0.132] | -0.037 [-0.167, +0.102] | -0.071 [-0.218, +0.084] | -0.026 [-0.151, +0.114] | -0.135 [-0.257, -0.031]* | -0.173 [-0.304, -0.049]* | -0.135 [-0.241, -0.043]* |
| ens_mean_day1 − era5 | +0.925 [+0.730, +1.140]* | +0.923 [+0.733, +1.137]* | +0.861 [+0.679, +1.077]* | +0.858 [+0.665, +1.078]* | +0.789 [+0.575, +1.010]* | +0.878 [+0.671, +1.115]* | -0.062 [-0.147, +0.013] | -0.065 [-0.144, +0.019] | -0.044 [-0.116, +0.031] |
| ens_mean_day0 − live_gb_rich_xgb | +1.449 [+1.244, +1.672]* | +1.458 [+1.246, +1.684]* | +1.294 [+1.142, +1.444]* | +1.280 [+1.134, +1.428]* | +1.270 [+1.123, +1.422]* | +1.251 [+1.102, +1.420]* | -0.164 [-0.281, -0.054]* | -0.179 [-0.318, -0.043]* | -0.140 [-0.306, -0.000]* |
| ens_mean_day1 − live_gb_rich_xgb | +2.204 [+1.926, +2.492]* | +2.244 [+1.979, +2.507]* | +2.154 [+1.915, +2.394]* | +2.174 [+1.928, +2.421]* | +2.130 [+1.896, +2.370]* | +2.155 [+1.908, +2.435]* | -0.091 [-0.182, -0.004]* | -0.070 [-0.156, +0.016] | -0.048 [-0.162, +0.060] |
| ens_control_day0 − era5 | +0.428 [+0.244, +0.620]* | +0.397 [+0.199, +0.597]* | +0.198 [+0.081, +0.325]* | +0.148 [+0.036, +0.274]* | +0.090 [-0.032, +0.212] | +0.134 [+0.017, +0.268]* | -0.199 [-0.346, -0.071]* | -0.250 [-0.428, -0.093]* | -0.186 [-0.313, -0.074]* |
| live_gb_rich_xgb − era5 | -1.279 [-1.457, -1.086]* | -1.322 [-1.485, -1.136]* | -1.293 [-1.465, -1.107]* | -1.316 [-1.498, -1.126]* | -1.341 [-1.509, -1.140]* | -1.277 [-1.439, -1.087]* | +0.029 [-0.049, +0.111] | +0.005 [-0.067, +0.084] | +0.004 [-0.070, +0.087] |

### wind: members

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_members_day1 − ens_control_day1 | +0.350 [+0.104, +0.607]* | +0.306 [+0.066, +0.548]* | +0.404 [+0.166, +0.655]* | +0.429 [+0.179, +0.695]* | +0.694 [+0.354, +1.064]* | +0.625 [+0.317, +0.972]* | +0.098 [+0.006, +0.192]* | +0.122 [+0.023, +0.227]* | +0.223 [+0.065, +0.416]* |

### wind: mean vs control

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − ens_control_day0 | -0.257 [-0.341, -0.178]* | -0.261 [-0.349, -0.174]* | -0.197 [-0.261, -0.129]* | -0.184 [-0.250, -0.119]* | -0.161 [-0.237, -0.084]* | -0.160 [-0.222, -0.100]* | +0.064 [+0.017, +0.115]* | +0.077 [+0.024, +0.138]* | +0.051 [-0.003, +0.109] |
| ens_mean_day2 − ens_control_day2 | -0.599 [-0.839, -0.358]* | -0.552 [-0.772, -0.337]* | -0.463 [-0.657, -0.279]* | -0.454 [-0.654, -0.268]* | -0.434 [-0.639, -0.256]* | -0.397 [-0.609, -0.208]* | +0.089 [-0.011, +0.189] | +0.098 [-0.003, +0.206] | +0.071 [-0.050, +0.207] |
| ens_mean_day3 − ens_control_day3 | -1.096 [-1.401, -0.781]* | -1.029 [-1.340, -0.724]* | -0.934 [-1.225, -0.652]* | -0.988 [-1.242, -0.752]* | -0.900 [-1.190, -0.642]* | -0.941 [-1.258, -0.653]* | +0.095 [-0.018, +0.218] | +0.041 [-0.105, +0.183] | +0.067 [-0.099, +0.271] |
| ens_mean_day5 − ens_control_day5 | -1.476 [-1.855, -1.124]* | -1.422 [-1.772, -1.086]* | -1.272 [-1.643, -0.966]* | -1.362 [-1.783, -0.994]* | -1.365 [-1.983, -0.853]* | -1.417 [-1.869, -1.098]* | +0.150 [-0.067, +0.371] | +0.060 [-0.218, +0.360] | -0.063 [-0.351, +0.234] |
| ens_mean_day7 − ens_control_day7 | -0.846 [-1.424, -0.285]* | -0.897 [-1.512, -0.291]* | -0.956 [-1.561, -0.422]* | -0.864 [-1.453, -0.311]* | -1.599 [-2.642, -0.642]* | -0.800 [-1.407, -0.190]* | -0.059 [-0.401, +0.274] | +0.033 [-0.400, +0.431] | -0.029 [-0.384, +0.271] |
| ens_mean_day10 − ens_control_day10 | -0.582 [-1.152, +0.038] | -0.598 [-1.168, -0.010]* | -0.494 [-1.065, +0.223] | -0.480 [-1.016, +0.165] | -1.150 [-1.969, -0.316]* | -0.565 [-1.096, +0.032] | +0.104 [-0.280, +0.529] | +0.118 [-0.211, +0.478] | -0.171 [-0.605, +0.244] |
| ens_mean_day14 − ens_control_day14 | +0.252 [-0.174, +0.718] | +0.161 [-0.300, +0.699] | +0.141 [-0.307, +0.626] | +0.049 [-0.429, +0.557] | -0.179 [-0.786, +0.485] | +0.297 [-0.251, +0.922] | -0.021 [-0.358, +0.317] | -0.113 [-0.637, +0.343] | +0.068 [-0.441, +0.592] |

### wind: mean vs climatology

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − climatology | -10.890 [-12.026, -9.753]* | -11.122 [-12.516, -9.808]* | -11.089 [-12.612, -9.660]* | -11.198 [-12.761, -9.760]* | -10.365 [-11.741, -9.136]* | -10.570 [-11.899, -9.337]* | +0.033 [-0.214, +0.312] | -0.077 [-0.362, +0.241] | +0.308 [-0.014, +0.709] |
| ens_mean_day1 − climatology | -10.136 [-11.262, -9.000]* | -10.335 [-11.760, -9.076]* | -10.229 [-11.763, -8.817]* | -10.304 [-11.818, -8.902]* | -9.505 [-10.863, -8.310]* | -9.666 [-11.014, -8.464]* | +0.106 [-0.127, +0.363] | +0.031 [-0.216, +0.306] | +0.400 [+0.070, +0.800]* |
| ens_mean_day2 − climatology | -8.895 [-9.989, -7.849]* | -9.155 [-10.562, -7.905]* | -8.931 [-10.391, -7.589]* | -9.063 [-10.554, -7.695]* | -8.164 [-9.350, -7.059]* | -8.298 [-9.528, -7.165]* | +0.224 [+0.010, +0.469]* | +0.091 [-0.148, +0.353] | +0.476 [+0.163, +0.847]* |
| ens_mean_day3 − climatology | -7.298 [-8.365, -6.288]* | -7.516 [-8.921, -6.318]* | -7.300 [-8.805, -5.999]* | -7.362 [-8.866, -6.070]* | -6.623 [-7.720, -5.597]* | -6.692 [-7.865, -5.555]* | +0.216 [-0.007, +0.466] | +0.153 [-0.089, +0.412] | +0.530 [+0.219, +0.885]* |
| ens_mean_day5 − climatology | -3.964 [-4.888, -3.110]* | -4.268 [-5.479, -3.236]* | -4.081 [-5.486, -2.891]* | -4.091 [-5.484, -2.903]* | -3.724 [-4.907, -2.693]* | -3.685 [-4.855, -2.641]* | +0.187 [-0.137, +0.516] | +0.177 [-0.141, +0.518] | +0.327 [-0.084, +0.775] |
| ens_mean_day10 − climatology | +0.088 [-0.617, +0.958] | -0.146 [-1.071, +0.834] | -0.278 [-1.343, +0.671] | -0.345 [-1.398, +0.516] | +0.202 [-0.330, +0.798] | +0.025 [-0.448, +0.575] | -0.132 [-0.539, +0.257] | -0.199 [-0.678, +0.279] | +0.115 [-0.345, +0.562] |
| ens_mean_day14 − climatology | +1.168 [+0.416, +1.948]* | +0.878 [-0.137, +1.887] | +0.618 [-0.494, +1.602] | +0.698 [-0.499, +1.701] | +1.404 [+0.817, +2.005]* | +1.207 [+0.514, +1.915]* | -0.260 [-0.911, +0.373] | -0.181 [-0.901, +0.507] | +0.399 [-0.193, +1.096] |

### wind: mean vs best baseline (D0's: smart_persistence_day0)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − smart_persistence_day0 | -7.629 [-8.455, -6.846]* | -7.639 [-8.488, -6.826]* | -7.823 [-8.725, -6.946]* | -7.926 [-8.827, -7.043]* | -7.533 [-8.421, -6.738]* | -7.620 [-8.593, -6.774]* | -0.184 [-0.381, -0.006]* | -0.287 [-0.509, -0.081]* | -0.039 [-0.191, +0.119] |

### wind: mean vs calendar-only (post hoc)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − calendar_only | -12.343 [-13.603, -11.017]* | -12.265 [-13.488, -10.931]* | -11.903 [-13.241, -10.534]* | -12.191 [-13.599, -10.777]* | -12.680 [-14.784, -10.835]* | -11.984 [-13.573, -10.455]* | +0.362 [-0.449, +1.183] | +0.075 [-0.967, +1.062] | -0.075 [-0.796, +0.628] |
| ens_mean_day1 − calendar_only | -11.588 [-12.870, -10.230]* | -11.479 [-12.751, -10.142]* | -11.044 [-12.387, -9.711]* | -11.296 [-12.711, -9.916]* | -11.820 [-13.902, -10.004]* | -11.080 [-12.717, -9.583]* | +0.435 [-0.355, +1.256] | +0.183 [-0.838, +1.171] | +0.017 [-0.670, +0.735] |
| ens_mean_day2 − calendar_only | -10.348 [-11.608, -9.033]* | -10.298 [-11.568, -9.009]* | -9.745 [-11.070, -8.426]* | -10.055 [-11.410, -8.718]* | -10.479 [-12.488, -8.733]* | -9.711 [-11.237, -8.281]* | +0.553 [-0.259, +1.408] | +0.242 [-0.770, +1.229] | +0.093 [-0.542, +0.764] |
| ens_mean_day3 − calendar_only | -8.750 [-10.009, -7.499]* | -8.659 [-9.918, -7.467]* | -8.114 [-9.423, -6.829]* | -8.354 [-9.688, -7.073]* | -8.938 [-10.864, -7.230]* | -8.106 [-9.596, -6.735]* | +0.545 [-0.237, +1.368] | +0.304 [-0.720, +1.272] | +0.147 [-0.467, +0.812] |
| ens_mean_day5 − calendar_only | -5.417 [-6.542, -4.322]* | -5.411 [-6.548, -4.309]* | -4.896 [-6.248, -3.703]* | -5.083 [-6.307, -3.962]* | -6.039 [-8.004, -4.464]* | -5.099 [-6.619, -3.736]* | +0.515 [-0.259, +1.307] | +0.328 [-0.465, +1.190] | -0.056 [-0.816, +0.678] |
| ens_mean_day7 − calendar_only | -3.045 [-4.171, -2.019]* | -2.948 [-4.062, -1.942]* | -2.425 [-3.647, -1.372]* | -2.547 [-3.769, -1.498]* | -3.385 [-5.246, -1.792]* | -2.526 [-3.928, -1.318]* | +0.523 [-0.181, +1.232] | +0.401 [-0.390, +1.195] | +0.188 [-0.497, +0.884] |
| ens_mean_day10 − calendar_only | -1.364 [-1.922, -0.800]* | -1.289 [-1.857, -0.704]* | -1.093 [-2.003, -0.212]* | -1.337 [-2.289, -0.441]* | -2.113 [-3.504, -0.844]* | -1.389 [-2.150, -0.585]* | +0.196 [-0.357, +0.818] | -0.048 [-0.666, +0.618] | -0.268 [-0.793, +0.237] |
| ens_mean_day14 − calendar_only | -0.285 [-0.724, +0.191] | -0.265 [-0.706, +0.226] | -0.196 [-0.778, +0.345] | -0.295 [-0.885, +0.247] | -0.911 [-2.235, +0.092] | -0.206 [-0.983, +0.464] | +0.069 [-0.397, +0.463] | -0.029 [-0.594, +0.460] | +0.016 [-0.650, +0.599] |

### wind: chosen upsampling vs linear

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − up_linear_day0 | +0.013 [-0.018, +0.043] | +0.002 [-0.046, +0.038] | +0.014 [-0.016, +0.045] | +0.013 [-0.013, +0.037] | +0.036 [+0.005, +0.070]* | +0.026 [-0.005, +0.053] | +0.011 [-0.031, +0.069] | +0.010 [-0.027, +0.056] | +0.005 [-0.039, +0.044] |
| ens_mean_day1 − up_linear_day1 | -0.010 [-0.044, +0.017] | +0.006 [-0.030, +0.037] | -0.003 [-0.032, +0.036] | -0.004 [-0.036, +0.023] | +0.008 [-0.021, +0.038] | -0.005 [-0.039, +0.027] | -0.009 [-0.041, +0.024] | -0.010 [-0.036, +0.017] | +0.009 [-0.023, +0.043] |
| ens_mean_day2 − up_linear_day2 | +0.016 [-0.025, +0.057] | +0.016 [-0.022, +0.065] | +0.011 [-0.019, +0.043] | +0.009 [-0.027, +0.052] | +0.001 [-0.032, +0.035] | +0.013 [-0.023, +0.045] | -0.005 [-0.055, +0.037] | -0.006 [-0.073, +0.056] | -0.008 [-0.058, +0.040] |
| ens_mean_day3 − up_linear_day3 | -0.010 [-0.041, +0.019] | +0.013 [-0.018, +0.046] | +0.018 [-0.019, +0.064] | +0.009 [-0.027, +0.044] | +0.006 [-0.029, +0.044] | -0.004 [-0.043, +0.033] | +0.005 [-0.046, +0.052] | -0.004 [-0.054, +0.042] | +0.007 [-0.047, +0.050] |
| ens_mean_day5 − up_linear_day5 | -0.007 [-0.065, +0.050] | +0.002 [-0.071, +0.057] | -0.022 [-0.071, +0.034] | +0.004 [-0.049, +0.049] | -0.016 [-0.073, +0.047] | -0.035 [-0.109, +0.019] | -0.024 [-0.095, +0.063] | +0.002 [-0.090, +0.069] | -0.034 [-0.112, +0.058] |
| ens_mean_day7 − up_linear_day7 | -0.011 [-0.088, +0.069] | -0.011 [-0.102, +0.071] | -0.014 [-0.098, +0.086] | -0.011 [-0.105, +0.098] | +0.015 [-0.088, +0.117] | -0.066 [-0.163, +0.025] | -0.003 [-0.133, +0.157] | +0.000 [-0.106, +0.104] | -0.044 [-0.166, +0.059] |
| ens_mean_day10 − up_linear_day10 | -0.072 [-0.180, +0.024] | +0.009 [-0.123, +0.128] | -0.029 [-0.130, +0.073] | -0.024 [-0.105, +0.065] | -0.003 [-0.120, +0.095] | +0.023 [-0.082, +0.144] | -0.037 [-0.155, +0.102] | -0.032 [-0.177, +0.131] | +0.080 [-0.046, +0.239] |
| ens_mean_day14 − up_linear_day14 | -0.083 [-0.198, +0.014] | -0.060 [-0.164, +0.028] | -0.001 [-0.126, +0.111] | -0.037 [-0.181, +0.097] | -0.055 [-0.235, +0.094] | -0.012 [-0.133, +0.116] | +0.059 [-0.067, +0.189] | +0.023 [-0.146, +0.214] | +0.055 [-0.109, +0.209] |

### wind: mean vs best baseline (D0's: smart_persistence_day1)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day1 − smart_persistence_day1 | -9.386 [-10.344, -8.417]* | -9.538 [-10.677, -8.470]* | -9.547 [-10.800, -8.351]* | -9.587 [-10.851, -8.419]* | -9.013 [-10.196, -7.957]* | -9.128 [-10.309, -8.033]* | -0.009 [-0.240, +0.222] | -0.049 [-0.277, +0.183] | +0.208 [-0.029, +0.475] |

### wind: mean vs best baseline (D0's: smart_persistence_day2)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day2 − smart_persistence_day2 | -8.744 [-9.791, -7.731]* | -9.011 [-10.354, -7.796]* | -8.828 [-10.249, -7.508]* | -8.938 [-10.385, -7.620]* | -8.124 [-9.296, -7.027]* | -8.239 [-9.452, -7.113]* | +0.184 [-0.048, +0.433] | +0.073 [-0.154, +0.328] | +0.384 [+0.100, +0.710]* |

### wind: horizon growth

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day2 − ens_mean_day0 | +1.995 [+1.695, +2.339]* | +1.967 [+1.674, +2.320]* | +2.158 [+1.897, +2.480]* | +2.135 [+1.866, +2.464]* | +2.201 [+1.884, +2.661]* | +2.273 [+1.984, +2.638]* | +0.191 [+0.090, +0.300]* | +0.168 [+0.054, +0.287]* | +0.169 [+0.020, +0.328]* |
| ens_mean_day3 − ens_mean_day0 | +3.592 [+3.092, +4.216]* | +3.606 [+3.102, +4.254]* | +3.789 [+3.303, +4.406]* | +3.836 [+3.318, +4.509]* | +3.742 [+3.233, +4.429]* | +3.878 [+3.316, +4.654]* | +0.183 [+0.053, +0.318]* | +0.230 [+0.067, +0.399]* | +0.222 [+0.063, +0.391]* |
| ens_mean_day5 − ens_mean_day0 | +6.926 [+6.129, +7.712]* | +6.854 [+6.064, +7.650]* | +7.008 [+6.163, +7.839]* | +7.107 [+6.265, +7.939]* | +6.641 [+5.883, +7.483]* | +6.885 [+6.038, +7.841]* | +0.153 [-0.087, +0.395] | +0.253 [-0.018, +0.612] | +0.019 [-0.188, +0.236] |
| ens_mean_day10 − ens_mean_day0 | +10.978 [+9.725, +12.270]* | +10.976 [+9.684, +12.289]* | +10.810 [+9.552, +12.116]* | +10.853 [+9.613, +12.070]* | +10.567 [+9.216, +12.029]* | +10.595 [+9.325, +11.966]* | -0.166 [-0.498, +0.195] | -0.123 [-0.555, +0.366] | -0.193 [-0.581, +0.169] |
| ens_mean_day14 − ens_mean_day0 | +12.058 [+10.724, +13.354]* | +12.000 [+10.682, +13.315]* | +11.707 [+10.525, +12.918]* | +11.896 [+10.673, +13.139]* | +11.769 [+10.477, +13.069]* | +11.777 [+10.531, +13.096]* | -0.293 [-0.900, +0.234] | -0.104 [-0.804, +0.534] | +0.092 [-0.436, +0.685] |

### wind: mean vs best baseline (D0's: smart_persistence_day3)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day3 − smart_persistence_day3 | -7.257 [-8.322, -6.258]* | -7.471 [-8.856, -6.291]* | -7.283 [-8.784, -5.977]* | -7.320 [-8.788, -6.036]* | -6.621 [-7.724, -5.597]* | -6.685 [-7.857, -5.550]* | +0.188 [-0.034, +0.441] | +0.151 [-0.089, +0.407] | +0.492 [+0.183, +0.834]* |

### wind: mean vs best baseline (D0's: smart_persistence_day10)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day10 − smart_persistence_day10 | +0.116 [-0.583, +0.977] | -0.121 [-1.038, +0.869] | -0.250 [-1.290, +0.695] | -0.320 [-1.335, +0.527] | +0.206 [-0.323, +0.801] | +0.025 [-0.448, +0.572] | -0.129 [-0.515, +0.259] | -0.199 [-0.657, +0.262] | +0.088 [-0.363, +0.523] |

### wind: mean vs best baseline (D0's: smart_persistence_day14)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day14 − smart_persistence_day14 | +1.193 [+0.446, +1.975]* | +0.916 [-0.076, +1.914] | +0.655 [-0.435, +1.625] | +0.728 [-0.448, +1.710] | +1.413 [+0.827, +2.015]* | +1.218 [+0.532, +1.925]* | -0.261 [-0.911, +0.369] | -0.188 [-0.905, +0.504] | +0.387 [-0.203, +1.084] |

### wind: calendar-only vs climatology

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| calendar_only − climatology | +1.452 [+0.694, +2.260]* | +1.143 [+0.167, +2.158]* | +0.814 [-0.384, +2.011] | +0.992 [-0.296, +2.295] | +2.315 [+1.207, +3.602]* | +1.413 [+0.701, +2.152]* | -0.329 [-1.188, +0.488] | -0.151 [-1.081, +0.806] | +0.383 [-0.221, +1.054] |

### solar: planned

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day1 − ens_mean_day0 | +0.613 [+0.445, +0.789]* | +0.619 [+0.451, +0.800]* | +0.638 [+0.465, +0.818]* | +0.629 [+0.458, +0.815]* | +0.600 [+0.358, +0.841]* | +0.581 [+0.385, +0.778]* | +0.020 [-0.045, +0.086] | +0.010 [-0.046, +0.069] | +0.005 [-0.073, +0.088] |
| ens_mean_day7 − ens_mean_day0 | +6.144 [+5.485, +6.851]* | +6.167 [+5.478, +6.892]* | +6.204 [+5.542, +6.897]* | +6.172 [+5.511, +6.856]* | +5.651 [+4.587, +6.746]* | +5.600 [+4.894, +6.357]* | +0.037 [-0.231, +0.326] | +0.004 [-0.287, +0.292] | -0.368 [-0.833, +0.124] |
| ens_members_day1 − ens_mean_day1 | +0.476 [+0.335, +0.622]* | +0.497 [+0.357, +0.644]* | +0.546 [+0.410, +0.678]* | +0.530 [+0.399, +0.659]* | +0.521 [+0.233, +0.800]* | +0.504 [+0.315, +0.652]* | +0.050 [-0.024, +0.123] | +0.033 [-0.031, +0.093] | +0.104 [-0.014, +0.218] |
| ens_mean_day1 − ens_control_day1 | -0.360 [-0.465, -0.262]* | -0.351 [-0.461, -0.244]* | -0.361 [-0.472, -0.244]* | -0.369 [-0.482, -0.256]* | -0.361 [-0.531, -0.177]* | -0.319 [-0.441, -0.173]* | -0.010 [-0.058, +0.041] | -0.018 [-0.068, +0.032] | -0.004 [-0.088, +0.083] |
| ens_mean_day7 − climatology | -0.157 [-0.581, +0.238] | -0.124 [-0.565, +0.264] | -0.181 [-0.624, +0.264] | -0.208 [-0.662, +0.219] | -1.778 [-2.719, -0.890]* | -0.868 [-1.603, -0.188]* | -0.058 [-0.326, +0.247] | -0.084 [-0.362, +0.215] | -0.739 [-1.550, +0.021] |

### solar: reference

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − era5 | -0.901 [-1.130, -0.684]* | -0.910 [-1.127, -0.695]* | -0.924 [-1.141, -0.718]* | -0.936 [-1.159, -0.716]* | -0.810 [-1.155, -0.489]* | -0.754 [-0.976, -0.545]* | -0.014 [-0.083, +0.056] | -0.025 [-0.099, +0.044] | +0.040 [-0.092, +0.165] |
| ens_mean_day1 − era5 | -0.288 [-0.532, -0.051]* | -0.292 [-0.534, -0.048]* | -0.286 [-0.523, -0.036]* | -0.307 [-0.553, -0.043]* | -0.210 [-0.547, +0.094] | -0.174 [-0.420, +0.081] | +0.006 [-0.073, +0.072] | -0.015 [-0.103, +0.064] | +0.045 [-0.097, +0.185] |
| ens_mean_day0 − live_gb_rich_xgb | +0.672 [+0.496, +0.876]* | +0.692 [+0.517, +0.897]* | +0.645 [+0.474, +0.828]* | +0.659 [+0.487, +0.852]* | +0.532 [+0.237, +0.812]* | +0.467 [+0.240, +0.669]* | -0.048 [-0.132, +0.039] | -0.033 [-0.123, +0.050] | -0.093 [-0.289, +0.075] |
| ens_mean_day1 − live_gb_rich_xgb | +1.285 [+1.057, +1.551]* | +1.311 [+1.075, +1.579]* | +1.283 [+1.049, +1.545]* | +1.288 [+1.044, +1.564]* | +1.132 [+0.843, +1.383]* | +1.048 [+0.741, +1.323]* | -0.028 [-0.125, +0.054] | -0.023 [-0.117, +0.058] | -0.089 [-0.308, +0.100] |
| ens_control_day0 − era5 | -0.644 [-0.872, -0.422]* | -0.654 [-0.876, -0.430]* | -0.651 [-0.870, -0.440]* | -0.662 [-0.882, -0.440]* | -0.660 [-0.944, -0.392]* | -0.539 [-0.749, -0.338]* | +0.002 [-0.059, +0.058] | -0.009 [-0.062, +0.042] | +0.047 [-0.073, +0.152] |
| live_gb_rich_xgb − era5 | -1.574 [-1.877, -1.303]* | -1.603 [-1.901, -1.336]* | -1.569 [-1.846, -1.310]* | -1.595 [-1.866, -1.339]* | -1.342 [-1.576, -1.101]* | -1.221 [-1.420, -0.997]* | +0.034 [-0.048, +0.111] | +0.008 [-0.064, +0.080] | +0.134 [-0.030, +0.318] |

### solar: members

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_members_day1 − ens_control_day1 | +0.116 [-0.008, +0.238] | +0.146 [+0.020, +0.276]* | +0.186 [+0.051, +0.318]* | +0.161 [+0.040, +0.285]* | +0.159 [-0.052, +0.376] | +0.185 [+0.055, +0.319]* | +0.040 [-0.035, +0.108] | +0.015 [-0.052, +0.075] | +0.101 [-0.010, +0.209] |

### solar: mean vs control

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − ens_control_day0 | -0.257 [-0.358, -0.159]* | -0.257 [-0.357, -0.159]* | -0.273 [-0.363, -0.187]* | -0.273 [-0.359, -0.193]* | -0.150 [-0.310, +0.026] | -0.216 [-0.298, -0.137]* | -0.016 [-0.052, +0.021] | -0.016 [-0.063, +0.031] | -0.007 [-0.081, +0.068] |
| ens_mean_day2 − ens_control_day2 | -0.524 [-0.706, -0.343]* | -0.516 [-0.705, -0.326]* | -0.594 [-0.795, -0.390]* | -0.587 [-0.782, -0.392]* | -0.517 [-0.804, -0.264]* | -0.574 [-0.794, -0.357]* | -0.078 [-0.153, -0.002]* | -0.071 [-0.140, +0.006] | -0.068 [-0.220, +0.113] |
| ens_mean_day3 − ens_control_day3 | -0.659 [-0.858, -0.480]* | -0.644 [-0.860, -0.454]* | -0.717 [-0.917, -0.536]* | -0.654 [-0.854, -0.469]* | -0.661 [-0.958, -0.427]* | -0.637 [-0.856, -0.438]* | -0.073 [-0.178, +0.024] | -0.009 [-0.111, +0.104] | +0.010 [-0.134, +0.158] |
| ens_mean_day5 − ens_control_day5 | -0.892 [-1.323, -0.542]* | -0.882 [-1.310, -0.533]* | -0.761 [-1.189, -0.411]* | -0.779 [-1.181, -0.435]* | -0.644 [-1.034, -0.289]* | -0.740 [-1.203, -0.330]* | +0.121 [-0.038, +0.274] | +0.103 [-0.069, +0.268] | +0.213 [-0.048, +0.444] |
| ens_mean_day7 − ens_control_day7 | -0.494 [-0.790, -0.167]* | -0.516 [-0.817, -0.183]* | -0.422 [-0.706, -0.130]* | -0.390 [-0.650, -0.126]* | -0.767 [-1.090, -0.417]* | -0.600 [-0.928, -0.266]* | +0.094 [-0.058, +0.244] | +0.126 [-0.040, +0.282] | -0.031 [-0.274, +0.256] |
| ens_mean_day10 − ens_control_day10 | -0.494 [-0.798, -0.170]* | -0.488 [-0.795, -0.157]* | -0.322 [-0.635, -0.009]* | -0.367 [-0.670, -0.040]* | -0.291 [-0.631, +0.108] | -0.555 [-0.860, -0.194]* | +0.166 [+0.004, +0.334]* | +0.121 [-0.074, +0.320] | +0.073 [-0.231, +0.388] |
| ens_mean_day14 − ens_control_day14 | -0.147 [-0.309, +0.007] | -0.128 [-0.296, +0.041] | -0.125 [-0.302, +0.067] | -0.142 [-0.336, +0.051] | -0.192 [-0.429, +0.012] | -0.112 [-0.279, +0.062] | +0.003 [-0.138, +0.162] | -0.013 [-0.148, +0.142] | -0.027 [-0.188, +0.139] |

### solar: mean vs climatology

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − climatology | -6.301 [-7.001, -5.644]* | -6.291 [-7.002, -5.606]* | -6.385 [-7.082, -5.725]* | -6.380 [-7.072, -5.721]* | -7.429 [-8.969, -5.906]* | -6.468 [-7.342, -5.499]* | -0.094 [-0.217, +0.042] | -0.089 [-0.205, +0.042] | -0.371 [-1.088, +0.256] |
| ens_mean_day1 − climatology | -5.688 [-6.382, -5.041]* | -5.672 [-6.383, -5.015]* | -5.747 [-6.421, -5.095]* | -5.751 [-6.414, -5.107]* | -6.828 [-8.330, -5.390]* | -5.887 [-6.688, -4.966]* | -0.075 [-0.212, +0.071] | -0.079 [-0.207, +0.057] | -0.367 [-1.096, +0.259] |
| ens_mean_day2 − climatology | -4.695 [-5.327, -4.103]* | -4.633 [-5.286, -4.023]* | -4.707 [-5.329, -4.117]* | -4.730 [-5.350, -4.145]* | -5.824 [-7.212, -4.439]* | -4.914 [-5.722, -4.028]* | -0.074 [-0.221, +0.095] | -0.097 [-0.245, +0.065] | -0.376 [-1.123, +0.292] |
| ens_mean_day3 − climatology | -3.583 [-4.136, -3.070]* | -3.519 [-4.081, -2.996]* | -3.603 [-4.121, -3.099]* | -3.599 [-4.113, -3.094]* | -4.811 [-6.099, -3.559]* | -3.966 [-4.698, -3.166]* | -0.084 [-0.226, +0.067] | -0.080 [-0.231, +0.081] | -0.423 [-1.197, +0.222] |
| ens_mean_day5 − climatology | -1.408 [-1.897, -0.966]* | -1.358 [-1.843, -0.922]* | -1.362 [-1.854, -0.917]* | -1.399 [-1.892, -0.950]* | -2.999 [-4.248, -1.917]* | -2.024 [-2.806, -1.305]* | -0.004 [-0.170, +0.196] | -0.042 [-0.210, +0.157] | -0.589 [-1.348, +0.074] |
| ens_mean_day10 − climatology | +0.490 [-0.029, +0.938] | +0.532 [+0.010, +0.979]* | +0.324 [-0.178, +0.757] | +0.281 [-0.203, +0.705] | -0.905 [-1.649, -0.240]* | -0.282 [-1.070, +0.420] | -0.208 [-0.515, +0.146] | -0.251 [-0.576, +0.125] | -0.842 [-1.712, -0.064]* |
| ens_mean_day14 − climatology | +0.774 [+0.180, +1.334]* | +0.825 [+0.223, +1.395]* | +0.587 [+0.077, +1.065]* | +0.561 [+0.059, +1.014]* | -0.742 [-1.643, +0.057] | +0.186 [-0.667, +0.931] | -0.237 [-0.574, +0.124] | -0.264 [-0.661, +0.148] | -0.831 [-1.748, -0.007]* |

### solar: mean vs calendar-only (post hoc)

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − calendar_only | -7.285 [-8.118, -6.465]* | -7.325 [-8.164, -6.470]* | -7.112 [-7.752, -6.436]* | -7.046 [-7.656, -6.394]* | -6.702 [-7.885, -5.632]* | -6.689 [-7.335, -5.971]* | +0.213 [-0.127, +0.555] | +0.278 [-0.142, +0.722] | +0.551 [-0.089, +1.169] |
| ens_mean_day1 − calendar_only | -6.672 [-7.461, -5.885]* | -6.706 [-7.526, -5.895]* | -6.473 [-7.078, -5.848]* | -6.418 [-6.977, -5.806]* | -6.101 [-7.183, -5.108]* | -6.108 [-6.747, -5.372]* | +0.232 [-0.104, +0.552] | +0.288 [-0.126, +0.711] | +0.556 [-0.068, +1.144] |
| ens_mean_day2 − calendar_only | -5.678 [-6.471, -4.902]* | -5.667 [-6.470, -4.851]* | -5.433 [-6.012, -4.821]* | -5.397 [-5.950, -4.819]* | -5.097 [-6.067, -4.165]* | -5.134 [-5.665, -4.524]* | +0.233 [-0.106, +0.555] | +0.270 [-0.134, +0.694] | +0.546 [-0.150, +1.178] |
| ens_mean_day3 − calendar_only | -4.567 [-5.272, -3.857]* | -4.552 [-5.294, -3.824]* | -4.329 [-4.833, -3.774]* | -4.266 [-4.743, -3.746]* | -4.084 [-4.936, -3.289]* | -4.187 [-4.683, -3.637]* | +0.223 [-0.109, +0.525] | +0.286 [-0.110, +0.692] | +0.499 [-0.131, +1.091] |
| ens_mean_day5 − calendar_only | -2.391 [-3.006, -1.742]* | -2.391 [-3.009, -1.729]* | -2.088 [-2.583, -1.564]* | -2.066 [-2.493, -1.600]* | -2.272 [-3.009, -1.694]* | -2.245 [-2.762, -1.733]* | +0.303 [+0.059, +0.545]* | +0.325 [+0.016, +0.633]* | +0.334 [-0.223, +0.788] |
| ens_mean_day7 − calendar_only | -1.141 [-1.619, -0.626]* | -1.157 [-1.648, -0.624]* | -0.908 [-1.316, -0.448]* | -0.875 [-1.287, -0.431]* | -1.051 [-1.467, -0.691]* | -1.089 [-1.548, -0.593]* | +0.249 [+0.018, +0.473]* | +0.282 [-0.021, +0.571] | +0.183 [-0.243, +0.557] |
| ens_mean_day10 − calendar_only | -0.493 [-0.759, -0.207]* | -0.502 [-0.766, -0.213]* | -0.402 [-0.663, -0.112]* | -0.386 [-0.670, -0.095]* | -0.178 [-0.571, +0.227] | -0.503 [-0.873, -0.122]* | +0.100 [-0.053, +0.255] | +0.116 [-0.087, +0.325] | +0.080 [-0.289, +0.409] |
| ens_mean_day14 − calendar_only | -0.209 [-0.442, +0.020] | -0.209 [-0.444, +0.029] | -0.139 [-0.437, +0.145] | -0.106 [-0.403, +0.176] | -0.015 [-0.237, +0.194] | -0.035 [-0.250, +0.176] | +0.070 [-0.055, +0.198] | +0.103 [-0.023, +0.217] | +0.091 [-0.055, +0.247] |

### solar: chosen upsampling vs linear

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day0 − up_linear_day0 | -0.419 [-0.528, -0.322]* | -0.418 [-0.524, -0.326]* | -0.438 [-0.560, -0.338]* | -0.467 [-0.596, -0.359]* | -0.498 [-0.780, -0.237]* | -0.430 [-0.579, -0.286]* | -0.020 [-0.059, +0.016] | -0.048 [-0.101, +0.001] | +0.015 [-0.080, +0.115] |
| ens_mean_day1 − up_linear_day1 | -0.305 [-0.397, -0.224]* | -0.298 [-0.389, -0.217]* | -0.337 [-0.440, -0.244]* | -0.349 [-0.466, -0.243]* | -0.445 [-0.743, -0.194]* | -0.334 [-0.467, -0.202]* | -0.039 [-0.090, +0.007] | -0.051 [-0.100, -0.007]* | +0.006 [-0.076, +0.101] |
| ens_mean_day2 − up_linear_day2 | -0.295 [-0.416, -0.199]* | -0.281 [-0.401, -0.183]* | -0.287 [-0.408, -0.182]* | -0.312 [-0.448, -0.198]* | -0.413 [-0.653, -0.209]* | -0.303 [-0.456, -0.166]* | -0.006 [-0.049, +0.041] | -0.031 [-0.080, +0.018] | +0.007 [-0.056, +0.080] |
| ens_mean_day3 − up_linear_day3 | -0.184 [-0.313, -0.070]* | -0.198 [-0.318, -0.092]* | -0.193 [-0.318, -0.085]* | -0.187 [-0.315, -0.070]* | -0.409 [-0.694, -0.193]* | -0.202 [-0.371, -0.047]* | +0.005 [-0.047, +0.058] | +0.011 [-0.045, +0.067] | +0.013 [-0.053, +0.074] |
| ens_mean_day5 − up_linear_day5 | -0.117 [-0.233, -0.003]* | -0.113 [-0.205, -0.018]* | -0.120 [-0.252, +0.005] | -0.156 [-0.284, -0.041]* | -0.403 [-0.738, -0.149]* | -0.133 [-0.305, +0.020] | -0.007 [-0.093, +0.071] | -0.043 [-0.120, +0.022] | +0.007 [-0.114, +0.116] |
| ens_mean_day7 − up_linear_day7 | -0.164 [-0.308, -0.022]* | -0.152 [-0.295, -0.010]* | -0.114 [-0.292, +0.037] | -0.096 [-0.267, +0.054] | -0.370 [-0.771, -0.083]* | -0.105 [-0.308, +0.061] | +0.038 [-0.072, +0.139] | +0.055 [-0.060, +0.163] | +0.073 [-0.050, +0.195] |
| ens_mean_day10 − up_linear_day10 | -0.033 [-0.104, +0.034] | -0.045 [-0.119, +0.030] | -0.067 [-0.166, +0.034] | -0.068 [-0.166, +0.040] | -0.207 [-0.309, -0.106]* | -0.139 [-0.245, -0.028]* | -0.022 [-0.104, +0.079] | -0.023 [-0.104, +0.074] | -0.089 [-0.215, +0.042] |
| ens_mean_day14 − up_linear_day14 | -0.034 [-0.112, +0.036] | -0.025 [-0.106, +0.047] | +0.050 [-0.020, +0.126] | +0.023 [-0.049, +0.095] | +0.001 [-0.136, +0.123] | +0.037 [-0.068, +0.135] | +0.075 [+0.007, +0.140]* | +0.048 [-0.015, +0.103] | +0.067 [-0.029, +0.158] |

### solar: horizon growth

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day2 − ens_mean_day0 | +1.607 [+1.427, +1.781]* | +1.658 [+1.479, +1.834]* | +1.678 [+1.494, +1.863]* | +1.650 [+1.475, +1.830]* | +1.604 [+1.317, +1.890]* | +1.554 [+1.320, +1.769]* | +0.020 [-0.059, +0.093] | -0.008 [-0.087, +0.066] | -0.005 [-0.146, +0.132] |
| ens_mean_day3 − ens_mean_day0 | +2.718 [+2.374, +3.077]* | +2.772 [+2.428, +3.112]* | +2.783 [+2.431, +3.142]* | +2.780 [+2.425, +3.130]* | +2.617 [+2.131, +3.115]* | +2.502 [+2.129, +2.881]* | +0.010 [-0.083, +0.101] | +0.008 [-0.087, +0.103] | -0.052 [-0.206, +0.110] |
| ens_mean_day5 − ens_mean_day0 | +4.893 [+4.408, +5.404]* | +4.933 [+4.436, +5.450]* | +5.024 [+4.487, +5.555]* | +4.980 [+4.455, +5.508]* | +4.429 [+3.741, +5.110]* | +4.443 [+3.892, +4.971]* | +0.090 [-0.079, +0.248] | +0.047 [-0.127, +0.214] | -0.217 [-0.523, +0.097] |
| ens_mean_day10 − ens_mean_day0 | +6.791 [+6.046, +7.547]* | +6.823 [+6.049, +7.609]* | +6.709 [+6.019, +7.406]* | +6.661 [+5.999, +7.327]* | +6.524 [+5.335, +7.777]* | +6.186 [+5.414, +6.968]* | -0.113 [-0.404, +0.200] | -0.162 [-0.485, +0.175] | -0.470 [-0.901, -0.014]* |
| ens_mean_day14 − ens_mean_day0 | +7.075 [+6.257, +7.900]* | +7.116 [+6.252, +7.960]* | +6.973 [+6.297, +7.629]* | +6.941 [+6.305, +7.531]* | +6.687 [+5.598, +7.806]* | +6.654 [+5.887, +7.365]* | -0.143 [-0.486, +0.212] | -0.175 [-0.586, +0.215] | -0.460 [-1.032, +0.139] |

### solar: calendar-only vs climatology

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| calendar_only − climatology | +0.984 [+0.327, +1.570]* | +1.033 [+0.352, +1.630]* | +0.726 [+0.126, +1.232]* | +0.667 [+0.063, +1.190]* | -0.727 [-1.636, +0.063] | +0.221 [-0.660, +0.959] | -0.307 [-0.642, +0.060] | -0.367 [-0.779, +0.062] | -0.922 [-1.827, -0.033]* |

## Part C: second-setting contrasts (planned, references, and those near the 5% line)

### wind

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day1 − ens_mean_day0 | +0.756 [+0.610, +0.897]* | +0.796 [+0.655, +0.935]* | +0.824 [+0.699, +0.951]* | n/a | +0.859 [+0.732, +0.988]* | +0.906 [+0.791, +1.024]* | +0.029 [-0.028, +0.084] | n/a | +0.087 [+0.009, +0.168]* |
| ens_mean_day7 − ens_mean_day0 | +9.001 [+8.033, +10.047]* | +9.008 [+8.018, +10.053]* | +9.088 [+8.098, +10.146]* | n/a | +9.090 [+8.003, +10.324]* | +9.165 [+8.135, +10.305]* | +0.080 [-0.185, +0.331] | n/a | +0.232 [+0.033, +0.441]* |
| ens_members_day1 − ens_mean_day1 | +0.632 [+0.438, +0.817]* | +0.581 [+0.405, +0.756]* | +0.705 [+0.518, +0.892]* | n/a | +0.870 [+0.602, +1.148]* | +0.778 [+0.553, +1.013]* | +0.124 [+0.053, +0.189]* | n/a | +0.135 [+0.039, +0.250]* |
| ens_mean_day1 − ens_control_day1 | -0.396 [-0.515, -0.275]* | -0.387 [-0.512, -0.257]* | -0.350 [-0.460, -0.246]* | n/a | -0.310 [-0.424, -0.199]* | -0.286 [-0.412, -0.170]* | +0.037 [-0.022, +0.106] | n/a | +0.067 [+0.004, +0.137]* |
| ens_mean_day7 − climatology | -1.937 [-2.696, -1.249]* | -2.185 [-3.302, -1.287]* | -2.055 [-3.277, -1.052]* | n/a | -1.346 [-2.219, -0.565]* | -1.461 [-2.323, -0.679]* | +0.131 [-0.218, +0.519] | n/a | +0.539 [+0.131, +1.031]* |
| ens_mean_day0 − era5 | +0.083 [-0.064, +0.239] | +0.089 [-0.062, +0.246] | -0.012 [-0.128, +0.110] | n/a | -0.104 [-0.244, +0.044] | -0.063 [-0.186, +0.076] | -0.101 [-0.209, -0.009]* | n/a | -0.072 [-0.165, +0.012] |
| ens_mean_day1 − era5 | +0.840 [+0.666, +1.034]* | +0.884 [+0.708, +1.082]* | +0.812 [+0.654, +0.996]* | n/a | +0.755 [+0.565, +0.956]* | +0.843 [+0.651, +1.044]* | -0.072 [-0.154, -0.002]* | n/a | +0.015 [-0.072, +0.102] |
| ens_mean_day0 − live_gb_rich_xgb | +1.401 [+1.205, +1.614]* | +1.407 [+1.214, +1.617]* | +1.262 [+1.113, +1.415]* | n/a | +1.250 [+1.087, +1.417]* | +1.192 [+1.026, +1.363]* | -0.145 [-0.252, -0.054]* | n/a | -0.143 [-0.277, -0.033]* |
| ens_mean_day1 − live_gb_rich_xgb | +2.158 [+1.914, +2.411]* | +2.202 [+1.967, +2.440]* | +2.086 [+1.853, +2.311]* | n/a | +2.110 [+1.877, +2.351]* | +2.098 [+1.871, +2.339]* | -0.116 [-0.205, -0.034]* | n/a | -0.056 [-0.176, +0.050] |
| ens_control_day0 − era5 | +0.333 [+0.141, +0.528]* | +0.345 [+0.139, +0.556]* | +0.187 [+0.072, +0.313]* | n/a | +0.071 [-0.032, +0.181] | +0.105 [-0.001, +0.234] | -0.157 [-0.304, -0.034]* | n/a | -0.113 [-0.252, -0.001]* |
| live_gb_rich_xgb − era5 | -1.318 [-1.480, -1.145]* | -1.318 [-1.483, -1.141]* | -1.274 [-1.441, -1.098]* | n/a | -1.355 [-1.537, -1.129]* | -1.256 [-1.417, -1.061]* | +0.044 [-0.008, +0.098] | n/a | +0.071 [+0.007, +0.145]* |
| ens_members_day1 − ens_control_day1 | +0.236 [+0.018, +0.447]* | +0.193 [-0.015, +0.394] | +0.355 [+0.164, +0.544]* | n/a | +0.559 [+0.298, +0.829]* | +0.492 [+0.263, +0.739]* | +0.161 [+0.065, +0.260]* | n/a | +0.202 [+0.083, +0.342]* |
| ens_mean_day0 − up_linear_day0 | +0.012 [-0.006, +0.030] | +0.012 [-0.005, +0.029] | +0.013 [-0.005, +0.032] | n/a | +0.026 [-0.008, +0.053] | +0.015 [-0.011, +0.040] | +0.001 [-0.015, +0.018] | n/a | -0.001 [-0.021, +0.021] |
| ens_mean_day10 − ens_control_day10 | -0.522 [-0.913, -0.083]* | -0.540 [-0.948, -0.090]* | -0.538 [-1.013, +0.036] | n/a | -0.887 [-1.459, -0.296]* | -0.431 [-0.864, +0.074] | +0.002 [-0.308, +0.346] | n/a | -0.078 [-0.366, +0.224] |
| ens_mean_day10 − calendar_only | -1.146 [-1.658, -0.611]* | -1.097 [-1.592, -0.580]* | -0.836 [-1.397, -0.243]* | n/a | -1.193 [-1.955, -0.429]* | -0.746 [-1.294, -0.154]* | +0.261 [-0.097, +0.680] | n/a | +0.149 [-0.175, +0.497] |
| ens_mean_day10 − up_linear_day10 | -0.028 [-0.105, +0.040] | -0.010 [-0.083, +0.060] | -0.026 [-0.061, +0.010] | n/a | +0.020 [-0.054, +0.100] | -0.035 [-0.086, +0.010] | -0.016 [-0.089, +0.058] | n/a | +0.004 [-0.076, +0.085] |
| ens_mean_day14 − ens_control_day14 | +0.243 [-0.194, +0.748] | +0.109 [-0.310, +0.575] | +0.239 [-0.135, +0.646] | n/a | +0.223 [-0.342, +0.802] | +0.412 [-0.039, +0.905] | +0.131 [-0.070, +0.358] | n/a | +0.189 [-0.169, +0.561] |
| ens_mean_day14 − climatology | +0.767 [+0.079, +1.487]* | +0.516 [-0.399, +1.424] | +0.245 [-0.776, +1.088] | n/a | +1.330 [+0.688, +2.034]* | +0.775 [+0.177, +1.412]* | -0.271 [-0.914, +0.342] | n/a | +0.373 [-0.095, +0.949] |
| ens_mean_day14 − smart_persistence_day14 | +0.793 [+0.111, +1.516]* | +0.554 [-0.336, +1.457] | +0.282 [-0.708, +1.114] | n/a | +1.339 [+0.695, +2.046]* | +0.786 [+0.187, +1.424]* | -0.272 [-0.914, +0.342] | n/a | +0.360 [-0.104, +0.934] |
| ens_mean_day14 − calendar_only | -0.070 [-0.402, +0.301] | -0.082 [-0.415, +0.282] | +0.026 [-0.278, +0.347] | n/a | +0.085 [-0.590, +0.632] | +0.266 [-0.150, +0.661] | +0.108 [-0.099, +0.318] | n/a | +0.292 [-0.042, +0.648] |
| ens_mean_day14 − up_linear_day14 | -0.020 [-0.101, +0.062] | -0.011 [-0.083, +0.056] | -0.006 [-0.088, +0.060] | n/a | +0.060 [-0.013, +0.130] | -0.057 [-0.146, +0.027] | +0.005 [-0.096, +0.091] | n/a | -0.051 [-0.135, +0.034] |
| calendar_only − climatology | +0.837 [+0.166, +1.542]* | +0.598 [-0.268, +1.480] | +0.218 [-0.732, +1.047] | n/a | +1.245 [+0.587, +1.886]* | +0.509 [-0.011, +1.028] | -0.379 [-1.032, +0.250] | n/a | +0.081 [-0.373, +0.572] |

### solar

| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |
|---|---|---|---|---|---|---|---|---|---|
| ens_mean_day1 − ens_mean_day0 | +0.609 [+0.472, +0.758]* | +0.610 [+0.464, +0.762]* | +0.649 [+0.499, +0.820]* | n/a | n/a | +0.601 [+0.399, +0.803]* | +0.039 [-0.009, +0.089] | n/a | +0.014 [-0.067, +0.094] |
| ens_mean_day7 − ens_mean_day0 | +5.839 [+5.205, +6.517]* | +5.852 [+5.190, +6.552]* | +5.965 [+5.314, +6.648]* | n/a | n/a | +5.596 [+4.840, +6.405]* | +0.113 [-0.108, +0.391] | n/a | -0.090 [-0.499, +0.361] |
| ens_members_day1 − ens_mean_day1 | +0.391 [+0.257, +0.532]* | +0.391 [+0.251, +0.532]* | +0.465 [+0.317, +0.605]* | n/a | n/a | +0.481 [+0.282, +0.648]* | +0.074 [+0.004, +0.148]* | n/a | +0.159 [+0.027, +0.283]* |
| ens_mean_day1 − ens_control_day1 | -0.326 [-0.423, -0.228]* | -0.323 [-0.421, -0.222]* | -0.336 [-0.438, -0.230]* | n/a | n/a | -0.332 [-0.437, -0.201]* | -0.013 [-0.047, +0.023] | n/a | -0.052 [-0.130, +0.024] |
| ens_mean_day7 − climatology | -0.466 [-0.853, -0.094]* | -0.438 [-0.834, -0.072]* | -0.442 [-0.872, -0.015]* | n/a | n/a | -0.951 [-1.717, -0.201]* | -0.004 [-0.250, +0.293] | n/a | -0.540 [-1.367, +0.225] |
| ens_mean_day0 − era5 | -0.882 [-1.112, -0.665]* | -0.880 [-1.105, -0.659]* | -0.899 [-1.110, -0.694]* | n/a | n/a | -0.784 [-0.983, -0.592]* | -0.019 [-0.079, +0.035] | n/a | +0.003 [-0.110, +0.119] |
| ens_mean_day1 − era5 | -0.273 [-0.502, -0.034]* | -0.270 [-0.504, -0.032]* | -0.250 [-0.470, -0.011]* | n/a | n/a | -0.183 [-0.419, +0.062] | +0.020 [-0.042, +0.081] | n/a | +0.018 [-0.124, +0.134] |
| ens_mean_day0 − live_gb_rich_xgb | +0.663 [+0.484, +0.872]* | +0.680 [+0.497, +0.895]* | +0.676 [+0.502, +0.880]* | n/a | n/a | +0.441 [+0.250, +0.604]* | -0.005 [-0.075, +0.062] | n/a | -0.117 [-0.259, +0.012] |
| ens_mean_day1 − live_gb_rich_xgb | +1.272 [+1.041, +1.535]* | +1.290 [+1.051, +1.561]* | +1.325 [+1.090, +1.599]* | n/a | n/a | +1.042 [+0.751, +1.315]* | +0.035 [-0.042, +0.101] | n/a | -0.102 [-0.267, +0.040] |
| ens_control_day0 − era5 | -0.646 [-0.861, -0.434]* | -0.642 [-0.860, -0.426]* | -0.646 [-0.862, -0.437]* | n/a | n/a | -0.565 [-0.758, -0.380]* | -0.004 [-0.054, +0.042] | n/a | +0.027 [-0.070, +0.129] |
| live_gb_rich_xgb − era5 | -1.545 [-1.849, -1.268]* | -1.560 [-1.862, -1.289]* | -1.575 [-1.870, -1.314]* | n/a | n/a | -1.225 [-1.428, -1.010]* | -0.015 [-0.082, +0.051] | n/a | +0.120 [-0.033, +0.310] |
| ens_members_day1 − ens_control_day1 | +0.065 [-0.067, +0.201] | +0.068 [-0.067, +0.206] | +0.129 [-0.013, +0.267] | n/a | n/a | +0.149 [-0.013, +0.302] | +0.062 [-0.008, +0.132] | n/a | +0.107 [+0.003, +0.206]* |
| ens_mean_day0 − ens_control_day0 | -0.236 [-0.330, -0.146]* | -0.238 [-0.332, -0.148]* | -0.253 [-0.339, -0.172]* | n/a | n/a | -0.219 [-0.302, -0.139]* | -0.015 [-0.056, +0.020] | n/a | -0.023 [-0.078, +0.036] |
| ens_mean_day5 − up_linear_day5 | -0.128 [-0.249, -0.011]* | -0.133 [-0.256, -0.018]* | -0.102 [-0.196, -0.004]* | n/a | n/a | -0.094 [-0.218, +0.028] | +0.031 [-0.019, +0.089] | n/a | +0.046 [-0.031, +0.125] |
| ens_mean_day7 − up_linear_day7 | -0.159 [-0.265, -0.050]* | -0.152 [-0.257, -0.040]* | -0.123 [-0.254, -0.008]* | n/a | n/a | -0.132 [-0.261, -0.003]* | +0.029 [-0.062, +0.122] | n/a | +0.039 [-0.080, +0.165] |
| ens_mean_day10 − ens_control_day10 | -0.493 [-0.768, -0.190]* | -0.498 [-0.775, -0.199]* | -0.358 [-0.649, -0.058]* | n/a | n/a | -0.533 [-0.837, -0.173]* | +0.139 [+0.012, +0.293]* | n/a | +0.053 [-0.206, +0.324] |
| ens_mean_day10 − climatology | +0.233 [-0.251, +0.653] | +0.247 [-0.244, +0.673] | +0.126 [-0.339, +0.562] | n/a | n/a | -0.378 [-1.182, +0.326] | -0.121 [-0.397, +0.211] | n/a | -0.706 [-1.600, +0.118] |
| ens_mean_day10 − up_linear_day10 | -0.071 [-0.145, +0.008] | -0.077 [-0.151, -0.001]* | -0.093 [-0.191, +0.015] | n/a | n/a | -0.129 [-0.227, -0.018]* | -0.016 [-0.079, +0.055] | n/a | -0.041 [-0.152, +0.066] |
| ens_mean_day14 − ens_control_day14 | -0.149 [-0.267, -0.027]* | -0.130 [-0.245, -0.010]* | -0.143 [-0.284, +0.017] | n/a | n/a | -0.150 [-0.308, +0.026] | -0.013 [-0.101, +0.085] | n/a | -0.030 [-0.156, +0.102] |
| ens_mean_day14 − climatology | +0.611 [+0.070, +1.118]* | +0.642 [+0.100, +1.160]* | +0.439 [+0.009, +0.857]* | n/a | n/a | +0.080 [-0.749, +0.772] | -0.203 [-0.490, +0.125] | n/a | -0.747 [-1.619, +0.027] |
| ens_mean_day14 − calendar_only | -0.105 [-0.289, +0.074] | -0.101 [-0.288, +0.077] | -0.069 [-0.289, +0.140] | n/a | n/a | -0.043 [-0.206, +0.126] | +0.032 [-0.053, +0.119] | n/a | -0.003 [-0.136, +0.138] |
| ens_mean_day14 − up_linear_day14 | -0.012 [-0.081, +0.059] | -0.020 [-0.087, +0.051] | +0.020 [-0.039, +0.090] | n/a | n/a | +0.048 [-0.023, +0.110] | +0.040 [-0.008, +0.086] | n/a | +0.052 [-0.025, +0.133] |
| calendar_only − climatology | +0.716 [+0.158, +1.226]* | +0.743 [+0.164, +1.263]* | +0.508 [+0.030, +0.918]* | n/a | n/a | +0.122 [-0.692, +0.763] | -0.234 [-0.545, +0.079] | n/a | -0.743 [-1.591, +0.060] |

## Devices used, per batch

| Batch | Device | Notes |
|---|---|---|
| Part A | no fit | |
| Part B, D0, D1, D2, both settings | CPU | 8 arms solar, 4 wind; D0 bit-identical to published |
| Part B, D1 rotation 3 | CPU | primary setting only |
| Part C primary, wind D0, D0trim, I, II; solar D0, I, II, D0trim | CPU | 4 workers |
| Part C second setting, wind D0, D0trim, I, II and the wind D0 and D0trim extra near-line arms | CPU | |
| Part C second setting, solar D0 | cuda | started before a device rule; its arms are all second-setting |
| Part C second setting, solar D0trim, I | cuda | |
| Part C primary, wind IIr, I2; solar IIr, I2 | cuda | contrasts within each design use one device; IIr and I2 changes are measured against D0 and D0trim (CPU) |
| Part C second setting, wind IIr, solar IIr | cuda | |
| `solar_all` D0, D1, primary and second setting; `wind_icon_dream` D0, D1 | cuda | priority batch |
| `solar_all` and `wind_icon_dream` D0, D1 primary, reproduction of published losses | CPU | D0 bit-identical (603,645 and 450,369 rows) |

Caution about device mixing: the "II − D0", "IIr − D0", "I2 − D0trim" change intervals compare a cuda fit with a CPU fit. The CPU-cuda difference on the same design is visible in the `solar_all` D0 rows (DMI minus ICON-D2 +1.016 on CPU, +1.039 on cuda) and is about 0.02 to 0.03 points, so a change of under about 0.05 points in those columns is not distinguishable from device noise.

### D1 with rotation 3 versus rotation 2 (Part B, primary setting, CPU)

| Study | Contrast | D0 | D1 (rot 2) | D1 (rot 3) | rot 3 − D0 | rot 3 − rot 2 |
|---|---|---|---|---|---|---|
| wind | icon_eu_wind − era5_wind | -0.314 [-0.453, -0.174]* | -0.298 [-0.431, -0.169]* | -0.300 [-0.446, -0.165]* | +0.014 [-0.039, +0.057] | -0.002 [-0.042, +0.033] |
| wind | ukv_wind − era5_wind | -0.440 [-0.628, -0.241]* | -0.407 [-0.578, -0.229]* | -0.416 [-0.598, -0.219]* | +0.024 [-0.009, +0.055] | -0.009 [-0.049, +0.031] |
| wind | icon_eu_wind − ukv_wind | +0.126 [+0.009, +0.232]* | +0.109 [+0.006, +0.208]* | +0.116 [+0.003, +0.226]* | -0.010 [-0.053, +0.040] | +0.006 [-0.025, +0.041] |
| wind | icon_d2_wind − icon_eu_wind | -0.264 [-0.334, -0.185]* | -0.275 [-0.342, -0.200]* | -0.281 [-0.339, -0.214]* | -0.017 [-0.068, +0.036] | -0.006 [-0.042, +0.028] |
| wind | icon_d2_wind − ukv_wind (exploratory) | -0.139 [-0.253, -0.026]* | -0.166 [-0.279, -0.055]* | -0.165 [-0.272, -0.064]* | -0.027 [-0.064, +0.008] | +0.000 [-0.024, +0.027] |
| solar | cams_global − icon_d2_global | -2.678 [-2.899, -2.424]* | -2.708 [-2.932, -2.456]* | -2.721 [-2.941, -2.472]* | -0.043 [-0.084, -0.007]* | -0.013 [-0.045, +0.016] |
| solar | icon_eu_global − icon_d2_global | +0.623 [+0.483, +0.756]* | +0.590 [+0.463, +0.715]* | +0.604 [+0.468, +0.734]* | -0.019 [-0.054, +0.013] | +0.014 [-0.017, +0.049] |
| solar | icon_eu_global − ukv_global | -0.476 [-0.654, -0.289]* | -0.491 [-0.672, -0.295]* | -0.464 [-0.646, -0.271]* | +0.011 [-0.017, +0.039] | +0.026 [-0.000, +0.053] |
| solar | icon_global_global − icon_eu_global | +0.097 [+0.046, +0.147]* | +0.093 [+0.048, +0.140]* | +0.101 [+0.052, +0.147]* | +0.005 [-0.023, +0.030] | +0.008 [-0.013, +0.029] |
| solar | sarah3_global − cams_global | +0.402 [+0.294, +0.498]* | +0.400 [+0.296, +0.498]* | +0.414 [+0.306, +0.516]* | +0.012 [-0.010, +0.036] | +0.014 [-0.005, +0.035] |
| solar | icon_dream_global − era5_global | -0.315 [-0.522, -0.116]* | -0.288 [-0.498, -0.077]* | -0.309 [-0.513, -0.106]* | +0.007 [-0.033, +0.047] | -0.021 [-0.058, +0.018] |

| Study | Arm | D0 | D1 (rot 2) | D1 (rot 3) |
|---|---|---|---|---|
| wind | era5_wind | 7.272 [6.65, 7.97] | 7.249 [6.62, 7.98] | 7.250 [6.61, 7.97] |
| wind | icon_d2_wind | 6.694 [5.95, 7.55] | 6.676 [5.91, 7.54] | 6.669 [5.91, 7.52] |
| wind | icon_eu_wind | 6.958 [6.24, 7.74] | 6.951 [6.23, 7.75] | 6.950 [6.22, 7.75] |
| wind | ukv_wind | 6.832 [6.08, 7.68] | 6.842 [6.08, 7.69] | 6.834 [6.07, 7.69] |
| solar | cams_global | 5.085 [4.84, 5.31] | 5.054 [4.82, 5.28] | 5.061 [4.82, 5.29] |
| solar | era5_global | 9.080 [8.58, 9.52] | 9.032 [8.53, 9.47] | 9.075 [8.58, 9.52] |
| solar | icon_d2_global | 7.763 [7.39, 8.09] | 7.763 [7.40, 8.09] | 7.782 [7.42, 8.11] |
| solar | icon_dream_global | 8.765 [8.32, 9.15] | 8.745 [8.30, 9.13] | 8.766 [8.32, 9.16] |
| solar | icon_eu_global | 8.386 [7.96, 8.77] | 8.353 [7.94, 8.73] | 8.386 [7.96, 8.78] |
| solar | icon_global_global | 8.483 [8.07, 8.85] | 8.446 [8.04, 8.81] | 8.487 [8.07, 8.87] |
| solar | sarah3_global | 5.487 [5.19, 5.76] | 5.454 [5.17, 5.73] | 5.475 [5.18, 5.76] |
| solar | ukv_global | 8.862 [8.40, 9.26] | 8.843 [8.40, 9.24] | 8.850 [8.41, 9.25] |

## Past-solar `all` panel and `wind_icon_dream`: D1 (rotation 2) against today's folds

Planned contrasts of each page, primary setting. The CPU block reproduces the published per-row losses under D0 bit for bit. The cuda blocks fit both designs on the GPU.

### solar_all, pooled setting, device cpu: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)

| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |
|---|---|---|---|
| ifs_hres_global − icon_eu_global | -0.092 [-0.373, +0.172] | -0.070 [-0.371, +0.204] | +0.022 [-0.062, +0.101] |
| knmi_harmonie_global − icon_eu_global | +0.555 [+0.319, +0.751]* | +0.538 [+0.320, +0.713]* | -0.017 [-0.097, +0.059] |
| dmi_harmonie_global − icon_d2_global | +1.016 [+0.730, +1.264]* | +0.959 [+0.690, +1.188]* | -0.057 [-0.127, +0.022] |

Absolute mean absolute error (percent of capacity) with 95% interval:

| Arm | D0 | D1 |
|---|---|---|
| dmi_harmonie_global | 8.830 [8.07, 9.54] | 8.628 [7.92, 9.29] |
| icon_d2_global | 7.813 [7.21, 8.38] | 7.669 [7.10, 8.22] |
| icon_eu_global | 8.382 [7.74, 8.96] | 8.240 [7.63, 8.82] |
| ifs_hres_global | 8.290 [7.60, 8.91] | 8.170 [7.47, 8.76] |
| knmi_harmonie_global | 8.937 [8.18, 9.66] | 8.778 [8.07, 9.43] |

Coverage: D0 leaves 14,853 of 40,243 hours (36.9%) in uncovered cells, calendar months [4, 5, 6]; D1 leaves 0. Rows per arm: 120,729 (hours x 3 seeds), asserted for every arm and design.

Reproduction of the published per-row losses under D0 (CPU): 603,645 rows, unmatched 0, bit-identical share 1.000000, max absolute difference 0.

### solar_all, pooled setting, device cuda: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)

| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |
|---|---|---|---|
| ifs_hres_global − icon_eu_global | -0.107 [-0.382, +0.155] | -0.087 [-0.391, +0.193] | +0.020 [-0.064, +0.096] |
| knmi_harmonie_global − icon_eu_global | +0.565 [+0.331, +0.763]* | +0.545 [+0.333, +0.720]* | -0.020 [-0.106, +0.066] |
| dmi_harmonie_global − icon_d2_global | +1.039 [+0.755, +1.281]* | +0.964 [+0.697, +1.194]* | -0.076 [-0.132, -0.024]* |

Absolute mean absolute error (percent of capacity) with 95% interval:

| Arm | D0 | D1 |
|---|---|---|
| dmi_harmonie_global | 8.846 [8.11, 9.55] | 8.624 [7.91, 9.28] |
| icon_d2_global | 7.807 [7.20, 8.37] | 7.661 [7.08, 8.21] |
| icon_eu_global | 8.399 [7.76, 8.98] | 8.245 [7.63, 8.82] |
| ifs_hres_global | 8.293 [7.60, 8.90] | 8.158 [7.46, 8.74] |
| knmi_harmonie_global | 8.965 [8.21, 9.68] | 8.790 [8.07, 9.45] |

Coverage: D0 leaves 14,853 of 40,243 hours (36.9%) in uncovered cells, calendar months [4, 5, 6]; D1 leaves 0. Rows per arm: 120,729 (hours x 3 seeds), asserted for every arm and design.

### solar_all, sensitivity setting, device cuda: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)

| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |
|---|---|---|---|
| ifs_hres_global − icon_eu_global | -0.004 [-0.289, +0.263] | -0.049 [-0.337, +0.213] | -0.046 [-0.125, +0.032] |
| knmi_harmonie_global − icon_eu_global | +0.513 [+0.241, +0.746]* | +0.436 [+0.208, +0.634]* | -0.077 [-0.210, +0.037] |
| dmi_harmonie_global − icon_d2_global | +0.964 [+0.682, +1.204]* | +0.868 [+0.603, +1.096]* | -0.096 [-0.180, -0.019]* |

Absolute mean absolute error (percent of capacity) with 95% interval:

| Arm | D0 | D1 |
|---|---|---|
| dmi_harmonie_global | 8.809 [8.07, 9.51] | 8.554 [7.83, 9.21] |
| icon_d2_global | 7.845 [7.24, 8.43] | 7.686 [7.10, 8.26] |
| icon_eu_global | 8.396 [7.77, 8.96] | 8.234 [7.62, 8.80] |
| ifs_hres_global | 8.392 [7.69, 9.01] | 8.185 [7.51, 8.77] |
| knmi_harmonie_global | 8.909 [8.16, 9.60] | 8.670 [7.98, 9.31] |

Coverage: D0 leaves 14,853 of 40,243 hours (36.9%) in uncovered cells, calendar months [4, 5, 6]; D1 leaves 0. Rows per arm: 120,729 (hours x 3 seeds), asserted for every arm and design.

### wind_dream, pooled setting, device cpu: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)

| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |
|---|---|---|---|
| icon_dream_eu_wind − era5_wind | +0.001 [-0.119, +0.131] | +0.010 [-0.096, +0.125] | +0.009 [-0.032, +0.057] |
| icon_dream_eu_wind − icon_eu_wind | +0.340 [+0.266, +0.405]* | +0.312 [+0.245, +0.371]* | -0.028 [-0.061, +0.007] |

Absolute mean absolute error (percent of capacity) with 95% interval:

| Arm | D0 | D1 |
|---|---|---|
| era5_wind | 7.285 [6.66, 7.95] | 7.270 [6.62, 7.98] |
| icon_dream_eu_wind | 7.286 [6.58, 8.06] | 7.280 [6.55, 8.06] |
| icon_eu_wind | 6.946 [6.22, 7.72] | 6.968 [6.23, 7.77] |

Coverage: D0 leaves 12,570 of 50,041 hours (25.1%) in uncovered cells, calendar months [5, 6, 7]; D1 leaves 0. Rows per arm: 150,123 (hours x 3 seeds), asserted for every arm and design.

Reproduction of the published per-row losses under D0 (CPU): 450,369 rows, unmatched 0, bit-identical share 1.000000, max absolute difference 0.

### wind_dream, pooled setting, device cuda: planned contrasts (percentage points of capacity; `*` = 95% interval excludes zero)

| Contrast | D0 | D1 (post-upgrade era rotated by 2) | D1 − D0 |
|---|---|---|---|
| icon_dream_eu_wind − era5_wind | -0.002 [-0.114, +0.121] | +0.006 [-0.098, +0.123] | +0.009 [-0.037, +0.047] |
| icon_dream_eu_wind − icon_eu_wind | +0.334 [+0.263, +0.399]* | +0.318 [+0.247, +0.381]* | -0.016 [-0.043, +0.010] |

Absolute mean absolute error (percent of capacity) with 95% interval:

| Arm | D0 | D1 |
|---|---|---|
| era5_wind | 7.311 [6.68, 7.99] | 7.264 [6.62, 7.97] |
| icon_dream_eu_wind | 7.309 [6.59, 8.08] | 7.271 [6.55, 8.05] |
| icon_eu_wind | 6.975 [6.25, 7.78] | 6.953 [6.22, 7.76] |

Coverage: D0 leaves 12,570 of 50,041 hours (25.1%) in uncovered cells, calendar months [5, 6, 7]; D1 leaves 0. Rows per arm: 150,123 (hours x 3 seeds), asserted for every arm and design.

`solar_all` at the second hyperparameter setting is on cuda only. `wind_icon_dream` was fitted at the primary setting only (its page's second-setting arms were not refit).

## Pinned code

The worktree branch changed `studies/beam_diffuse_split/ens_forecast_horizons.py` during this work (a later commit moved results into `era_covered/`, kept wind rows from 2024-12-01 and pinned offsets). Every Part C fit here imports the copy of that file from commit `34e6b5f4` (main) kept in `code/ens_forecast_horizons.py`, so all designs use the code that produced the D0 numbers. Nothing else in the imported study modules changed between that commit and the branch head.

## Deviations from the study scripts, and uncertainties

- **The wind study's published losses are not under `past_weather_v2/wind/`.** That directory holds only the ERA5-by-year files. The published per-row losses that the wind page's figures come from are `data/studies/beam_diffuse_split/beam_diffuse_wind_products/losses.parquet`, which is what Part B compares against. The solar comparison uses `past_weather_v2/solar_long/losses.parquet` as briefed.
- **The solar study is the `long` panel** (eight products, the page's main row set). Its six planned contrasts are `DECIDING_CONTRASTS` plus the two `NEW_PLANNED_CONTRASTS["long"]`. Only the eight arms in those contrasts were fitted. The other panels (`all`, `published`, `record`) were not run.
- **Part B fits use `run_experiment.run_all` unchanged** at `max_workers=4` (the published runs used 8; the fit is per (arm, site) and the worker count does not enter the model). D0 came out bit-identical to the published losses, so the setting of workers did not matter.
- **The saved ENS input frames (`wind_inputs.parquet`, `solar_inputs.parquet`) do not hold the arm columns**, only the upsampling-chart inputs. I rebuilt the fit frame in memory with the script's own functions (`build_inputs`, then `main_frame` at the combination the saved losses chose: `speed_components` for wind, `clear_sky` for solar) from the saved `ens_members.parquet`, and wrote nothing under `data/`. Under D0 this reproduces the saved primary-setting losses: wind 13,270,752 rows all bit-identical; solar 15,800,616 rows, 99.995% bit-identical, the rest within 1.5e-7 of capacity (thread-order floating-point noise in some arms).
- **The upsampling combination is not re-chosen under each design.** The choosing rule (`choose_method`) compares fitted arms' errors, so a different fold design could choose differently. The saved choice is kept in every design.
- **The best-baseline name per horizon is D0's** in every design (for wind smart persistence at days 0 to 3, 10 and 14 and climatology at days 5 and 7; for solar climatology at every day), so a contrast's meaning does not change with the design. Climatology, and the wind smart persistence's shrinkage weight, are fitted out of fold, so they do depend on the design and were recomputed under each.
- **Design I drops 19 days of rows** (2024-11-12 to 2024-11-30) because those rows carry a month label before the first whole month after IFS 49r1 but are post-change data. D0trim shows what dropping them alone does: the eight largest changes from D0 are all wind contrasts against climatology or smart persistence, from -0.25 to -0.29 points (largest: calendar-only minus climatology, -0.291 [-0.953, +0.097]), and every one of those paired intervals includes zero. Wind day-7 ensemble mean minus climatology moves by -0.192 [-0.683, +0.121]. The wind pre-49r1 era of design I has only about 3 months (12 August to 11 November 2024, 5,976 hours), so its five folds hold at most three non-empty blocks. Six of the 25 pairs of rotations cover every calendar month for wind and two of 25 for solar. I used the first the search returns (wind {1: 2, 2: 0}; solar {1: 3, 2: 0}). Other covering designs were not tried, so design I's numbers carry a fold-assignment uncertainty this report does not measure.
- **Part C refits only these arms**: every ensemble-mean and control arm at every day, the upsampling combinations at every day, the ERA5 and live GB reference arms, the calendar-only arms, the member-by-member arm at day 1 only, and climatology and the persistence baselines. Not refitted: the member-by-member arm at days other than 1 and its median, the mean-applied arm, the native and 6-hourly arms, and the no-subsample arm. So the page's claim that the member-by-member arm loses at almost every horizon is tested at day 1 only.
- **Where the statistics differ between the two "changes".** I minus D0trim and II minus D0 are paired on the rows both designs share. D0trim minus D0 is a row-set change only. An interval on a change says how far the design moves the contrast, not whether the new design is more correct.
- **A second setting was run only where stated.** Part B: every arm at both settings. Part C: see "Not done".
- **Post-hoc arms.** The calendar-only arms and the month-of-year arms are the page's own post hoc arms; they inherit that label.
- **The interval width does not include the fold-assignment uncertainty** for D1 and I (a design with other covering offsets would give slightly different point estimates).

## Checks

- Part B: every arm holds exactly sites x hours x 3 seeds rows at both settings and under all three designs (asserted, `common.take`), so every design scores the same 50,734 wind hours and 76,727 solar hours; the same assertion applies to the published-loss comparison. Wind rows per arm 152,202, solar 230,181.
- Every contrast and absolute-error call in Parts B and C filters on the setting explicitly and asserts each arm's row count before pairing.
- Part C row counts by design: wind 50,268 (D0), 49,065 (D0trim, I), 43,089 (II); solar 50,643, 49,958, 49,958, 37,166. Each is asserted per arm at every call.
- `eh.check_shared_rows` passed for every primary-setting Part C fit.
- Part A: `cut_eras` with zero offsets equals `with_eras` on (site, time, fold) for both studies.

## Not done

- **Design I2 has no second-setting fit**, and design II (uncovered) has none for solar. Design IIr and I have them.
- **`wind_icon_dream` was fitted at the primary setting only**, and its D0 cuda fit and D1 cuda fit are the only second device; the second setting of that page was not run.
- Part C's member-by-member arm was fitted at day 1 only; the mean-applied, median, native, 6-hourly and no-subsample arms were not refitted.
- The choosing rule for the upsampling combination was not re-run under each design.
- Other covering offsets for design I beyond the two searched sets (six for wind, two for solar) were not fitted.

## Files and commands

Worktree used for all imports: `/home/jack/dev/nged-substation-forecast/.claude/worktrees/era-fold-design` (branch `era-fold-design`, main plus nothing). Scratch directory: `/home/jack/dev/nged-substation-forecast/.claude/worktrees/scratch/era-fold/`. Nothing was written under `data/` or committed.

Scripts (all in `scripts/`; run from the worktree as `uv run python ../scratch/era-fold/scripts/<name>`):

- `common.py`: frame builders, the three designs, `take` (setting filter plus row-count assertion).
- `partA.py`: coverage arithmetic, writes `out/partA.json`.
- `partB_fit.py {wind|solar} {D0|D1|D2} [sensitivity]`: fits, writes `out/losses_<study>_<design>[_sens].parquet` and `out/folds_...`.
- `partB_analyse.py`: reproduction check and intervals, writes `out/partB_*.parquet`.
- `partC_fit.py {wind|solar} {D0|D0trim|I|II} 1 [--sens|--extra-only]`: fits, writes `out/losses_ens_<domain>_<design>[_sens].parquet`.
- `partC_analyse.py {pooled|both}`, `nearline.py`, `make_report.py`: analysis and this report's tables (`out/report_tables.md`).
- `runC.sh`, `runS.sh`: the batch drivers used.

Per-row losses (arm, site label, time, month, fold, seed, setting, absolute and signed capped error) are in `out/losses_*.parquet`; interval tables are in `out/partB_contrasts.parquet`, `out/partB_absolute.parquet`, `out/partC_contrasts_both.parquet` and `out/partC_absolute_both.parquet`.

The past-wind page's comparison for ENS minus ERA5 (read-only, `past_weather_v2/ens_hres_past_wind/report.md`): ENS day-0 mean minus ERA5 wind is -0.0693 [-0.1879, +0.0584] pp on 43,555 hours from 2024-12-01 (primary setting; -0.0774 [-0.2258, +0.0783] at the second). The horizons page's D0 figure is +0.170 [+0.021, +0.322] on 50,268 hours from 2024-08-12. The nearest like-for-like refits here are design II (-0.071 [-0.218, +0.084] on 43,089 hours from 2024-12-01) and design I (+0.001 [-0.119, +0.132] on 49,065 hours). The two pages' feature sets differ (the past-wind page cuts three eras with its own offsets and gives ENS a 100 m speed, direction and 10 m speed as here; its row set is 466 hours larger than design II's; I did not establish why, the horizons page drops any hour lacking any horizon's ENS input or any baseline's input, which is a likely but unchecked cause), so agreement of II with the past-wind figure is expected, and the +0.170 is the outlier: it comes from fitting across IFS 49r1 without an era cut.

Sizes of a design's effect on the ENS page's claims are in the tables above; the detailed claim-by-claim reading is in "Effect on each claim" below.

## Effect on each claim of the ENS-horizons page (primary setting unless stated; I and I2 are era cuts, IIr is rows from 2024-12-01 with covering folds)

- **Wind, ENS day-0 mean minus ERA5, +0.170 [+0.021, +0.322]:** +0.136 [-0.012, +0.291] (D0trim), +0.001 [-0.119, +0.132] (I), -0.037 [-0.167, +0.102] (I2), -0.026 [-0.151, +0.114] (IIr). Second setting: +0.083 [-0.064, +0.239] under D0, -0.012 [-0.128, +0.110] under I. The claim that ENS day 0 loses to ERA5 for wind does not survive an era cut or the covering restriction to December 2024 onwards. The past-wind page's own figure on its row set is -0.069 [-0.188, +0.058] (43,555 hours), which agrees with I, I2 and IIr.
- **Wind day-0 control member minus ERA5, +0.428 [+0.244, +0.620]:** +0.198 (I), +0.148 [+0.036, +0.274] (I2), +0.134 [+0.017, +0.268] (IIr). The direction survives and the size falls by half to two thirds.
- **Solar, ENS day-0 mean beats ERA5 by 0.901 [0.684, 1.130], day 1 by 0.288 [0.051, 0.532]:** day 0 stable under every design (-0.754 to -0.936); day 1 is -0.286 (I) and -0.307 (I2) but -0.174 [-0.420, +0.081] under IIr, where it is no longer significant on 37,166 hours.
- **The ensemble mean beats the control member at day 1 (planned):** wind -0.399 (D0), -0.343 (I), -0.342 (I2), -0.271 [-0.388, -0.153] (IIr); solar -0.360, -0.361, -0.369, -0.319. Survives.
- **Horizon growth (planned, day 1 and day 7 minus day 0):** wind +0.754 / +9.298 (D0), +0.859 / +9.478 (I), +0.895 / +9.643 (I2), +0.904 / +9.458 (IIr); solar +0.613 / +6.144 (D0), +0.638 / +6.204 (I), +0.629 / +6.172 (I2), +0.581 / +5.600 (IIr). Survives; the size moves inside the paired intervals.
- **The member-by-member arm loses to the ensemble mean at day 1 (planned):** wind +0.749 (D0), +0.747 (I), +0.771 (I2), +0.896 [+0.603, +1.211] (IIr); solar +0.476, +0.546, +0.530, +0.504. Survives; day 1 only.
- **Ensemble mean minus climatology at day 7 (planned):** wind -1.593 [-2.463, -0.804] (D0), -1.610 (I), -1.555 (I2), -1.113 [-1.987, -0.313] (IIr): survives. Solar -0.157 [-0.581, +0.238] (D0), -0.181 (I), -0.208 (I2), -0.868 [-1.603, -0.188] (IIr): the solar result depends on the row set, because climatology fitted on the shorter record is worse.
- **Wind day 14, climatology beats the ensemble mean by 1.17 points [0.42, 1.95]:** +0.618 [-0.494, +1.602] (I) and +0.698 [-0.499, +1.701] (I2), not significant; +1.207 [+0.514, +1.915] (IIr), significant. At the second setting +0.767 [+0.079, +1.487] (D0), +0.245 [-0.776, +1.088] (I), +0.775 [+0.177, +1.412] (IIr). The claim depends on the fold design and the row set.
- **Solar day 14, climatology beats the ensemble mean by 0.77 [0.18, 1.33]:** +0.587 [+0.077, +1.065] (I), +0.561 [+0.059, +1.014] (I2), +0.186 [-0.667, +0.931] (IIr): survives under the era cut, not under IIr.
- **Post hoc calendar-only check at day 14 (ensemble mean minus calendar-only):** wind -0.285 (D0), -0.196 (I), -0.295 (I2), -0.206 (IIr); solar -0.209, -0.139, -0.106, -0.035. No design gives a significant advantage to the ensemble mean; "adds no statistically significant skill" is unchanged.
- **Solar clear-sky radiation beats straight-line interpolation at day 1 by 0.30 [0.22, 0.40]:** -0.305 (D0), -0.337 (I), -0.445 [-0.743, -0.194] (II, uncovered folds); survives. Wind: no interpolation moves the error by 0.1 point under any design. The combination was not re-chosen.
- **The ensemble mean loses to ERA5 for wind at day 1 and beats it for solar, and loses to UKV with ICON-EU at both days:** wind +0.925 (D0), +0.861 (I), +0.858 (I2), +0.878 (IIr); solar -0.288, -0.286, -0.307, -0.174 (IIr, not significant); UKV with ICON-EU wind +2.204 / +2.154 / +2.174 / +2.155 and solar +1.285 / +1.283 / +1.288 / +1.048. All survive; solar day 1 against ERA5 is the one that does not under IIr.
