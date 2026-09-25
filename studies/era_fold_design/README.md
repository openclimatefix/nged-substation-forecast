# Era-fold design measurement (issues #868 and #892)

This folder holds the measurement behind `plans/era-fold-design.md` (PR #906): how many scored hours have a calendar month held out of every training row under today's era folds, and how planned contrasts move when the folds cover every month. `report.md` is the full report, produced by `scripts/`, with a device table per fit batch. `solar_all_D1_effect.md` is the past-solar `all` panel's table. The scripts were run from a scratch directory and read the published per-row losses under `data/studies/`; `saved_cov.py` and `two_shares.py` recompute the shares below from the folds saved with each published loss file.

## Definitions

- **Uncovered share (a).** The fraction of scored farm-hours that sit in a (site, fold, calendar month) cell whose calendar month occurs in two or more years and has no training row in any other fold. A fold design that rotates the post-upgrade era's fold numbers can remove these.
- **One-year-only share (b).** The same fraction for a calendar month that occurs in one year only. No fold design can cover it, because no other year has that month.
- Designs: D0 is today's folds (`assign_folds(by=("site", "era"))`). D1 rotates the post-upgrade era's fold numbers by 2 (`rotate_folds`), the first rotation `search_fold_offsets` returns.

## Shares by block, from the saved folds

| Page block | Scored farm-hours | (a) uncovered | (b) one-year-only |
|---|---|---|---|
| Solar, `long` panel | 76,727 | 2,129 (2.8%), one farm, May to July | 0 |
| Solar, `all` panel (extra Open-Meteo block) | 40,243 | 14,853 (36.9%), April to June | 0 |
| Solar, ENS block (`ens_past_solar`) | 54,447 | 1,194 (2.2%) | 0 |
| Solar, station block | 0 | 0 | 0 |
| Wind, main block | 50,734 | 8,326 (16.4%), June and July, all three farms | 0 |
| Wind, ICON-DREAM-EU block (`wind_icon_dream`) | 50,041 | 12,570 (25.12%), May to July | 0 |
| Wind, ECMWF block (`ens_hres_past_wind`) | 43,555 | 0 (0.00%) | 4,082 (9.37%), months 10 and 11 |
| Wind, station block (`station_wind_arms`) | 34,156 | 0 (0.00%) | 14,411 (42.19%), months 1 to 7 |

The wind main block's 8,326 hours come from recomputing today's folds on the past-wind study's frame (`report.md`, Part A); the saved `beam_diffuse_wind_products` file is an older frame with 52,996 hours and 8,603 (16.23%) uncovered. The solar station block's row is the past-wind station arms, which the past-solar page does not fit.

## Past-solar `all` panel: D1 minus D0, three planned contrasts

Points of capacity, 95% intervals resampling whole calendar months and a seed. D0 reproduces the published per-row losses bit for bit on the CPU (603,645 rows).

| Contrast | Setting | Device | D0 | D1 | D1 minus D0 |
|---|---|---|---|---|---|
| HRES minus ICON-EU | primary | CPU | -0.092 [-0.373, +0.172] | -0.070 [-0.371, +0.204] | +0.022 [-0.062, +0.101] |
| KNMI HARMONIE minus ICON-EU | primary | CPU | +0.555 [+0.319, +0.751] | +0.538 [+0.320, +0.713] | -0.017 [-0.097, +0.059] |
| DMI HARMONIE minus ICON-D2 | primary | CPU | +1.016 [+0.730, +1.264] | +0.959 [+0.690, +1.188] | -0.057 [-0.127, +0.022] |
| HRES minus ICON-EU | primary | cuda | -0.107 [-0.382, +0.155] | -0.087 [-0.391, +0.193] | +0.020 [-0.064, +0.096] |
| KNMI HARMONIE minus ICON-EU | primary | cuda | +0.565 [+0.331, +0.763] | +0.545 [+0.333, +0.720] | -0.020 [-0.106, +0.066] |
| DMI HARMONIE minus ICON-D2 | primary | cuda | +1.039 [+0.755, +1.281] | +0.964 [+0.697, +1.194] | -0.076 [-0.132, -0.024] |
| HRES minus ICON-EU | second | cuda | -0.004 [-0.289, +0.263] | -0.049 [-0.337, +0.213] | -0.046 [-0.125, +0.032] |
| KNMI HARMONIE minus ICON-EU | second | cuda | +0.513 [+0.241, +0.746] | +0.436 [+0.208, +0.634] | -0.077 [-0.210, +0.037] |
| DMI HARMONIE minus ICON-D2 | second | cuda | +0.964 [+0.682, +1.204] | +0.868 [+0.603, +1.096] | -0.096 [-0.180, -0.019] |

No contrast changes sign or significance at either setting. The change of about 0.05 points on DMI minus ICON-D2 between CPU (-0.057) and cuda (-0.076) is the CPU-to-GPU difference in fitting.

Absolute mean absolute error (percent of capacity), primary setting, CPU: DMI 8.830 to 8.628, ICON-D2 7.813 to 7.669, ICON-EU 8.382 to 8.240, HRES 8.290 to 8.170, KNMI 8.937 to 8.778 (D0 to D1). Every arm falls by 0.12 to 0.20 points, and the ranking is unchanged: ICON-D2, HRES, ICON-EU, DMI, KNMI. On cuda every arm falls by 0.14 to 0.22.

## Wind `wind_icon_dream` panel: D1 minus D0 (primary setting, CPU)

ICON-DREAM-EU minus ERA5: +0.001 [-0.119, +0.131] under D0, +0.010 [-0.096, +0.125] under D1 (change +0.009). ICON-DREAM-EU minus ICON-EU: +0.340 [+0.266, +0.405] under D0, +0.312 [+0.245, +0.371] under D1 (change -0.028). Neither contrast changes sign or significance. D0 reproduces the published per-row losses bit for bit (450,369 rows).
