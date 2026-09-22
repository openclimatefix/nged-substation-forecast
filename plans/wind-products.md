# Plan: which weather product best describes past wind at the metered wind farms (#826)

**The problem.** The solar half of #809 ranked six weather products as descriptions of past
sunshine. Nothing yet says which product describes past wind. The roster holds three wind
generators, each with seven years of half-hourly readings. Hub-height wind depends on terrain and
boundary-layer structure, which a 31 km reanalysis smooths away, so the solar ranking need not carry
over to wind.

**The solution.** Download hub-height wind from ERA5, UKV, ICON-D2, ICON-EU and ICON global at each
wind generator's coordinates. Build one hourly dataset per product, score each product with the
tested per-site out-of-fold loop on one common row set, and add a wind section to the weather
products page. The design follows the solar study's, so its safeguards carry over:

- a common row set chosen by rules no product decides;
- folds cut inside each UKV era, with the era shown to every model;
- deciding contrasts named before the run;
- the per-row capacity normalisation;
- the served-lead reading.

CAMS publishes no wind, so it has no arm.

## Verdict, size and departures

**Verdict: worth doing.** #809 asks for it by name.

**Size: complex.** The five triggers:

- **What gets stored:** yes. There are new downloads and a published section.
- **The production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The open choices are the heights, the window, and the
  outage rule.
- **Callers I could not name without searching:** no.

That buys two plan reviews, two scientific-validity reviews and one diff review, all on Opus 5.5.

**Departures from the issue body:**

- **The issue's heights (80 m, 100 m or 120 m) become a per-product choice of native heights.**
  - ERA5 and UKV publish 100 m natively. The three ICON products publish 80 m and 120 m, and
    Open-Meteo derives their 100 m value by scaling the 120 m value.
  - Every product is shown its own native heights (see Arms below), and a like-for-like arm is also
    shown the 100 m value each product serves.
- **UKV's hub-height wind exists only from 2024-08-05.**
  - Open-Meteo's UKV archive before its own downloader started (2024-08-12) carries radiation but
    no hub-height wind. So a comparison including UKV covers about two years.
  - A second comparison drops UKV and covers ERA5 and the three ICON products from 2022-11-24,
    about four years.

## What changes, file by file

### `packages/studies/src/studies/anonymise.py`

- `site_labels_for` gains keyword arguments `labels` and `seed`, defaulting to today's solar labels
  and seed.
- New `WIND_SITE_LABELS = ("W1", "W2", "W3")` and `WIND_LABEL_PERMUTATION_SEED`.
- Tests:
  - the solar mapping is unchanged;
  - the wind mapping is pinned on placeholder identifiers;
  - a wrong-sized wind roster raises.

### `studies/beam_diffuse_split/fetch_open_meteo_point.py`

- `fetch_point_frame` gains a `base_url` keyword, defaulting to `HISTORICAL_FORECAST_URL`, and a
  `extra_parameters` string for `wind_speed_unit=ms`.
- ERA5's point values come from the archive API (`archive-api.open-meteo.com`), which takes the
  same query shape.
- No behaviour change for existing callers.

### New `studies/beam_diffuse_split/fetch_wind_point.py`

- **Download.** For each product, fetch `wind_speed_10m` and `wind_direction_10m` plus the
  product's native hub heights and their directions, at each wind generator's coordinates:
  - ERA5: 100 m;
  - UKV: 100 m;
  - ICON: 80 m and 120 m.

  It also fetches `wind_speed_100m` for every product, for the like-for-like arm.
- **Where the files go.** One parquet per product under `data/studies/weather/<PRODUCT>/wind_<product>.parquet`.
- **Coordinates are never written.** Rows are keyed by the anonymous wind labels.
- **The fetch fails loudly** if a product returns only nulls for a site, which is the same guard
  as the solar fetcher.

### New `studies/beam_diffuse_split/wind_products.py` (the study script)

- **Roster.** Take wind series with at least one year of readings from the metadata, with their
  capacity taken from `effective_capacity`, and label them with `site_labels_for(labels=WIND_SITE_LABELS, seed=...)`.
- **Hourly power.** Read the power table and apply `studies.power.hourly_from_half_hourly` (the
  period-ending convention). The target is `power_mw`.
  - Two of the three series are metered in MVA, which is apparent power. A wind farm's reactive
    power keeps apparent power above zero at low wind. The page states this, and the model learns
    it per site.
- **Rows.** Keep an hour only if every product covers it. Drop hours holding a zero or missing
  half-hour where the metered series flat-lines, meaning an exactly constant value for 24 hours or
  more. For wind, "exactly zero for 24 hours" can be a genuine calm, so the outage rule is
  flat-lining at any constant value, not zero alone. That rule reads the power column only, so no
  product decides it.
- **Folds.** `assign_folds(by=("site", "era"))`, with the era boundary at UKV's 2026-02 upgrade
  month. The rows from 21 to 31 January 2026 are dropped.
- **Arms per product.** Shared features are hour of day, day of year and `era_code`.
  - `<p>_native`: the product's native hub-height speeds plus 10 m, with each speed's direction as
    sine and cosine.
  - `<p>_100m`: the served 100 m speed and direction.
- **Deciding contrasts, named before the run, on the five-product window.** These are native arms
  throughout.
  - `ukv − era5`: does the 2 km national model beat the reanalysis?
  - `icon_eu − ukv`: which model covering all of Great Britain is better?
  - `icon_d2 − icon_eu`: does the regional model add anything for wind?
  - `icon_global − icon_eu`.

  On the four-product window, the same contrasts minus those involving UKV.
- **Exploratory.**
  - native against 100 m within each product;
  - the ICON contrasts split by served lead, with ICON-EU's lead measured and ICON global's inferred.
- **Leave one site out.** It uses the months-withheld pattern, on the capacity-normalised target.
  With three sites that means training on two.
- **Output.** Results go to `data/studies/beam_diffuse_split/beam_diffuse_wind_products/`.

### Docs

- A new section, "Past wind at the three wind farms", on
  `docs/studies/weather-products-for-the-past.md`. The page title stays, and the intro gains one
  sentence. Or it is a sibling page, if the section outgrows the page; the reviewer decides.
- The recommendation per consumer, for wind.

## Scientific-validity reviews

Two fresh Opus reviewers review the study in turn, and a re-run is done on request.

## Risks and open questions

1. **Three sites is a small sample**, and all three sit in the same corner of Lincolnshire. The
   intervals come from month blocks shared across sites. Site-level agreement is reported for every
   deciding contrast.
2. **Wind hub heights are unknown.** Native heights shown together let the tree interpolate. Hub
   heights of 80 to 120 m are typical.
3. **Curtailment and availability.** Wind farms curtail, and turbines go down for maintenance.
   Neither is recorded except in the active-network-management data for one generator, which is
   solar. Unrecorded partial availability is noise shared by every arm, so it reduces power to tell
   products apart but does not bias the comparison between them.
4. **Should the studies directory be restructured** into shared weather infrastructure and
   per-study directories, now that three studies share its fetchers and fit loop?
   *Recommendation:* file that as its own issue, and keep this change inside the directory.
