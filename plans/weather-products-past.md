# Plan: which weather product best describes past sunshine at the metered solar farms (#809, solar)

**The problem.** Several offline consumers read an estimate of weather that has already happened:

- capacity estimation;
- training targets and pre-training;
- historical features;
- disaggregation.

None of them has a measured basis for choosing a product. Earlier scripts found an ordering: CAMS,
then ICON-D2, then UKV, then ERA5. Those scripts mixed row sets and normalisations, which #823 has
since fixed. They also compared products served at different forecast leads. UKV's archive holds the
T+0 analysis, while ICON-D2's holds a 1-to-3-hour forecast. The ordering therefore confounds product
quality with lead. ICON-EU and ICON global, the two ICON products that cover the whole of Great
Britain, have never been scored at all.

**The solution.** A study script, `weather_products.py`, puts the products through the tested fit
loop from #823 on one common row set. It runs two comparisons.

- **As served.** This compares CAMS, ERA5, UKV, ICON-D2, ICON-EU and ICON global over the longest
  window all six cover, which runs from 2022-11-23 to 2026-09-10.
- **With the lead held equal.** This compares the four weather models at the same one-day lead,
  from Open-Meteo's Previous Runs API, over the window all four cover at that lead. That window
  runs from 2024-08-06.

ERA5 and CAMS appear in the second comparison as fixed references, because neither has a
forecast lead.

Each comparison reports two separate questions:

- how well each product's global irradiance predicts power;
- whether each product's published beam/diffuse split adds anything on top.

A docs page, `docs/studies/weather-products-for-the-past.md`, states what each offline consumer
should read and why. Wind is #826.

## Verdict, size and departures

**Verdict: worth doing, with four departures from the issue body.**

**Size: complex.** The five triggers:

- **What gets stored:** yes. There are new study downloads and datasets, and a new docs page whose
  numbers will be cited.
- **The production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The open choices are the common window, how to hold
  the lead equal, and the CAMS quality filter.
- **Callers I could not name without searching:** no. The registry names its own readers.

Size complex buys:

- both plan reviews;
- both diff reviews;
- at least two scientific-validity reviews of the results, on Opus 5.5 as the maintainer asked,
  with a re-run whenever a reviewer asks for one.

**Departures from the issue body:**

1. **The lead can be held equal only at one day, and only from August 2024.** The Previous Runs API
   offsets in whole days. It carries `_previous_day1` for the ICON family from January 2024 and for
   UKV from August 2024. UKV and ICON-D2 never populate day 2 or later. No API serves a fixed lead
   of a few hours for the study window. The Single Runs API starts in April 2026. So the equal-lead
   comparison is a forecast comparison at a 24-to-47-hour lead, over about two years. It answers
   "is ICON-D2's advantage a lead artefact?", but it is not an analysis comparison. The as-served
   comparison stays the primary answer to the issue's question, labelled with each product's
   served lead.
2. **SARAH-3 waits for #806**, which needs a person to order the data.
3. **Wind is split out to #826.**
4. **`era_comparison.py` is replaced, not extended.** Its four-product, three-era table becomes the
   era breakdown of the as-served comparison. Its figures are quoted in
   `docs/roadmap/data-sources.md`, in #809 and in the study README. Those citations move to the new
   page. `multi_nwp.py` stays, because it belongs to #810.

## What changes, file by file

### The source registry: `studies/beam_diffuse_split/sources.py`

- **`OpenMeteoModel` gains `lead_days: int`, which defaults to 0.** At 0, the fetcher uses the
  historical-forecast endpoint as today. Above 0, it uses `PREVIOUS_RUNS_URL`, requests each
  variable with the suffix `_previous_day{lead_days}`, and strips the suffix on the way in. The
  written frame therefore has the same columns as any other point source.
- **Four new entries.** They are `ukv-day1`, `icon-d2-day1`, `icon-eu-day1` and
  `icon-global-day1`. Each gets the `models=` value and `native_radiation` of its base model and
  `lead_days=1`. Its `archive_starts` is the first whole day with real day-1 values, probed and not
  copied: 2024-08-07 for UKV and 2024-01-20 for the ICON family, to be confirmed by the fetch.
- **`SourceType`, `SOURCE_CHOICES` and `PER_SITE_SOURCES`** gain the four names.

### The fetcher: `studies/beam_diffuse_split/fetch_open_meteo_point.py`

- `fetch_point_frame` takes the base URL and the variable suffix from the model entry.
- The two served-column checks run unchanged. For `ukv-day1` the backward-mean reconstruction check
  runs on the day-1 `_instant` columns. That is a real test of whether the Previous Runs API applies
  the same conversion.

### The new study script: `studies/beam_diffuse_split/weather_products.py`

- **`_joined(sources)`** inner-joins the per-source datasets that `build_dataset.py` writes, on
  `(site, time)`. It takes power, geometry and ERA5 temperature from the base build, and `ghi_<arm>`
  and `bhi_<arm>` from each product. Every arm is therefore shown the same temperature and the same
  rows, and differs only in its irradiance columns.
- **The base build is CAMS with `--min-cams-reliability 0`, not the default 0.9.**
  - **Why:** the default keeps only hours CAMS itself rates as reliable, which is a selection made
    by one of the contestants. It would favour CAMS on the common row set.
  - **Sensitivity check:** the reliability-filtered row set is run as well, to measure how much the
    filter moves the ranking.
- **Folds come from `assign_folds(by=("site", "era"))`, with the era boundary at the UKV upgrade
  month, 2026-02.** Every fold then holds pre-upgrade and post-upgrade rows, and a model scoring
  UKV after the upgrade has trained on post-upgrade UKV.
  - This replaces `era_comparison.py`'s scheme of separate fits per era, which trained each era's
    models on a third of the data.
  - The era breakdown becomes a scope of one pooled run: the same losses, restricted to each era's
    rows when bootstrapped.
- **Arms.**
  - **Accuracy:** `<product>_global`. This is `SHARED_FEATURES` plus that product's global
    irradiance, one arm per product.
  - **Components:** `<product>_split` for every product that publishes a direct component. This
    is `SHARED_FEATURES` plus global, beam and diffuse irradiance.
- **Contrasts.**
  - **Accuracy:** every product's global arm against ERA5's, plus the adjacent pairs in the ranking.
  - **Components:** each product's split arm against its own global arm.

  All contrasts use `bootstrap_difference` on `absolute_error_capped_fraction_of_capacity`, pooled
  and per era.
- **Two comparisons in one script, chosen by `--comparison as-served|day1`.**
  - As served: CAMS, ERA5, UKV, ICON-D2, ICON-EU and ICON global.
  - Day 1: the four `-day1` sources, plus ERA5 and CAMS as references.
- **Output.** `data/studies/weather_products/<comparison>/` receives `losses.parquet`,
  `intervals.parquet` and `report.md`. The report has an arm table, an accuracy table and a
  components table, each with row counts, months and the served lead of each product.
- **The Fractions Skill Score is out of scope.** This study compares mean errors, not timing.

### Removed

- **`studies/beam_diffuse_split/era_comparison.py`.** Its README row and its citations move to the
  new page.

### Docs

- **`docs/studies/weather-products-for-the-past.md`** is new, and is added to the `mkdocs.yml`
  nav. It covers:
  - the question;
  - the products with their served lead, spatial domain, history and latency;
  - the as-served and day-1 results;
  - the components result;
  - the four traps #809 lists, and how each is avoided here;
  - a recommendation per offline consumer;
  - limitations: six generators in one region, solar only, and SARAH-3 absent.
- **`docs/roadmap/data-sources.md`:** the four-product figures and the UKV − ERA5 contrast point
  to the new page.
- **`studies/beam_diffuse_split/README.md`:** a row for `weather_products.py`, removal of the
  `era_comparison.py` row, and a redirect of the era paragraph.

## Design-philosophy check

This is R&D code, so it fails fast. No production path is touched.

## Tests

The study script is study-tier code: it is not unit-tested, and its evidence is its own report.
Two changes are tested machinery or checkable:

- **`fetch_point_frame`'s suffix handling.** The fetcher is study code. The day-1 served-column
  check on real downloads is its test: `ukv-day1` has to pass the backward-mean reconstruction, and
  every day-1 source has to pass the spread check.
- **`assign_folds(by=("site", "era"))`** is already tested in `packages/studies`.

## Scientific-validity reviews

After the first results exist, two fresh Opus reviewers each audit the study in turn:

- the design: row set, selection, leakage, lead, normalisation, eras, and the interval method;
- the claims on the docs page against the report.

A re-run is made whenever a reviewer asks for one.

## Verification

- The implement-issue set, plus `uv run mkdocs build --strict`, with the rendered page read.
- A re-run of `weather_products.py` for both comparisons from a clean data directory before the
  page is finalised.

## Risks and open questions

1. **Is a 1-day-lead comparison worth including, given it is a forecast comparison?**
   *Recommendation: yes.* It is the only lead-equal evidence available. If the ordering survives
   at equal lead, the as-served ordering is not a lead artefact. If it flips, the page must say
   so.
2. **Should ECMWF IFS join?** `ecmwf_ifs025` has day-1 values from March 2024.
   *Recommendation: not in this PR.* ECMWF's open data comes in 3-hourly steps, so what the hourly
   column holds at Open-Meteo is unmeasured, and the registry refuses `unmeasured`. It can be
   measured in a follow-up issue.
3. **The CAMS quality filter.** *Recommendation:* use the all-hours base as the primary and the
   filtered set as sensitivity, as described above.
4. **Is one pooled run with folds cut inside eras equivalent to `era_comparison.py`'s separate fits
   per era?** It is not. Each model now trains on both eras. That is more data and closer to how a
   production model would be trained, but the era figures will differ from the published ones. The
   page states the change of method.
