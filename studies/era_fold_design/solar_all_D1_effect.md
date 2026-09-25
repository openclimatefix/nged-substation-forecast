# solar_all: effect of the D1 fold design on the three planned contrasts

Device cuda for both designs (both arms of every contrast on the same device). D0 is today's saved fold design re-cut on the same rows; the CPU check against the published losses is in the section below. D1 rotates the post-upgrade era's fold numbers by 2 (covers every calendar month; D0 leaves 36.9% of hours uncovered).

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
Reading: none of the three planned contrasts changes sign or significance under D1. The largest change from D0 is DMI HARMONIE-AROME minus ICON-D2, -0.076 points [-0.132, -0.024], on a contrast of +1.039 under D0. Every arm's absolute error falls by 0.14 to 0.22 points under D1.

## CPU check (reproduces the published per-row losses under D0, bit for bit)

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
## Second setting (cuda)

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
