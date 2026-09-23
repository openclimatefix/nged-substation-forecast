# Plan: which weather product best describes past wind at the metered wind farms (#826)

**The problem.** The solar half of #809 ranked six weather products as descriptions of past
sunshine. Nothing yet says which product describes past wind at the three metered wind farms, and
the solar ranking need not carry over.

**The solution.** Download 100 m and 10 m wind from ERA5, UKV, ICON-D2, ICON-EU and ICON global at
each wind generator's coordinates. Score every product with one identical arm, through the tested
per-site out-of-fold loop, on one common row set starting 2024-08-12. Publish the result as a
sibling page to the solar one. CAMS publishes no wind, so it has no arm.

## Verdict, size and departures

**Verdict:** worth doing. #809 asks for this comparison by name.

**Size: complex.** The five triggers:

- **What gets stored:** yes, because the page's numbers will be cited.
- **The production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes.
- **Callers I could not name without searching:** no.

That buys a plan review, two scientific-validity reviews and a diff review, all on Opus 5.5. The one
plan review covered simplicity, correctness and the scientific design together, from the data, and
its findings reshaped the plan below. A second plan review was not run; the two scientific-validity
reviews stand in for it.

**Departures from the issue body:**

- **One height, not a per-product hub height.** Every product serves 100 m wind. It is native for
  ERA5 and UKV, and for ICON it is the native 120 m speed scaled by about 0.98, which a tree cannot
  tell apart from the 120 m speed. Every arm is shown the same four columns: 100 m speed, 100 m
  direction as sine and cosine, and 10 m speed. Showing each product its native heights would give
  ICON more columns than ERA5 and UKV, and a wider arm keeps a small advantage whatever the
  information.
- **The window is short.** UKV's hub-height wind exists only from Open-Meteo's own downloader,
  2024-08-12, so the comparison covers about two years.
- **The terrain hypothesis cannot be tested.** The issue expects resolution to matter most for wind
  because terrain drives hub-height wind. The three sites are in flat Lincolnshire, so the page
  says this study cannot speak to terrain.

## What changes, file by file

### `packages/studies/src/studies/anonymise.py` (done)

`site_labels_for` takes `labels` and `seed`. `WIND_SITE_LABELS` and `WIND_LABEL_PERMUTATION_SEED`
are added. The wind permutation is pinned in a test on placeholder identifiers.

### `studies/beam_diffuse_split/fetch_open_meteo_point.py`

`fetch_point_frame` gains a `base_url` keyword, defaulting to `HISTORICAL_FORECAST_URL`, so ERA5 can
be fetched from the archive API with the same code. Existing callers do not change.

### New `studies/beam_diffuse_split/fetch_wind_point.py`

For each of the five products, the script fetches `wind_speed_100m`, `wind_direction_100m` and
`wind_speed_10m` at the wind generators' coordinates, from 2024-08-12 to `era5_grid.LAST_DATE`. Each
product's download goes to `data/studies/weather/<PRODUCT>/wind_<product>.parquet`, keyed by the
anonymous wind label. The fetcher's all-null guard carries over. Speeds stay in Open-Meteo's
default, km/h; the page divides by 3.6 wherever it quotes a speed.

### New `studies/beam_diffuse_split/wind_products.py`

This is the study script. It builds on `weather_products.py`, importing `METRIC`, `_contrast_line`,
`_mae`, the `UPGRADE_*` constants and the era handling from it, rather than copying them.

- **The roster.** The wind series come from the metadata with at least one year of readings, and
  their capacity from `effective_capacity`. They are labelled with `WIND_SITE_LABELS`.
- **The power hour is centred on the label.**
  - The half-hour stamps are shifted back 30 minutes before `hourly_from_half_hourly`, so the hour
    labelled T holds the half-hours ending at T and T + 30 min, spanning T − 30 to T + 30.
  - The reason is that Open-Meteo's wind is instantaneous at the label. An offset scan in the plan
    review found every product scoring best with the centred hour. The period-ending hour the solar
    study uses would handicap UKV by about 0.2 points.
  - The page states this and cites the scan.
- **Row rules, applied from the power column alone.**
  - Drop every hour holding an exact-zero half-hour. One generator's zeros are disconnections: 51%
    fall where ERA5 reads above 6 m/s, and 9 to 10% of its half-hours at 8 to 10 m/s read exactly
    zero. The rule also removes some genuine calm hours at another generator, so behaviour near
    cut-in is under-sampled, and the page says so.
  - Drop the upgrade tail from 21 to 31 January 2026.
  - Keep a row only if all five products cover it.
  - Set `constrained` to false and `cap_mw` to null for every row. NGED has confirmed that the only
    generator under active network management in the trial area is solar.
- **Folds.** `assign_folds(by=("site", "era"))` cuts the folds, and every arm is given `era_code`.
- **Arms.** Each of the five products gets one arm, `<p>_wind`: the shared hour of day, day of
  year and `era_code`, plus the product's four wind columns.
- **Four deciding contrasts, named before the run:**
  - `icon_eu − era5`
  - `ukv − era5`
  - `icon_eu − ukv`
  - `icon_d2 − icon_eu`

  Each is reported pooled and per site. `icon_global − icon_eu` is exploratory.
- **What is not built:** no leave-one-site-out arm, no lead tables, and no second window.
- **Served leads, stated rather than analysed.**
  - ERA5's wind is an hourly analysis field, and UKV serves T+0.
  - ICON's lead for an instantaneous field is unmeasured.
  - For wind, the lead confound can therefore only work against ICON.
- **Output** goes to `data/studies/beam_diffuse_split/beam_diffuse_wind_products/`.

### Docs

- **A new sibling page, `docs/studies/weather-products-for-past-wind.md`.** It covers the products
  and their leads for wind, the method with its two wind-specific differences (the centred hour and
  the zero rule), the ranking, per-site agreement, the recommendations, and what three sites in one
  flat region cannot show.
  - The only recommendations are for training history and historical features.
  - Capacity estimation and disaggregation get none, because every figure is after per-site
    recalibration.
- **The solar page's "untested for wind" line** links to the new page. The new page is added to the
  nav and to the studies index.

## Scientific-validity reviews

Two fresh Opus reviewers audit the finished study in turn. A re-run follows whenever a reviewer asks
for one.

## Risks

1. **Two of three series are apparent power, in MVA.** Their floor of about 2 to 3% of capacity
   at low wind is reactive power. The per-site model learns it, and the page states it.
2. **One generator shows unrecorded partial availability.** At 100 m wind of 13 m/s or more, its
   output spreads roughly evenly across a wide range. That is noise shared by every arm, and it makes
   that site the noisiest.
3. **Two of the three generators share an ERA5 grid point.** ERA5 therefore supplies only two
   independent weather points.
