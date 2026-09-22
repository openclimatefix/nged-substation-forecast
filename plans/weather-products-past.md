# Plan: which weather product best describes past sunshine at the metered solar farms (#809, solar)

**The problem.** Several offline consumers read an estimate of weather that has already happened:
capacity estimation, training targets, historical features and disaggregation. None of them has a
measured basis for choosing a product. `era_comparison.py` ranked four products: CAMS first, then
ICON-D2, then UKV, then ERA5. Its 2022–2026 common row set left out ICON-EU and ICON global, the
two ICON products that cover the whole of Great Britain. It also fitted each era separately, on a
third of the data. The ranking also mixes forecast leads. UKV's archive holds the T+0 analysis,
while the ICON archives hold forecasts 1 to 3 hours (ICON-D2, ICON-EU) or up to 6 hours (ICON
global) ahead.

**The solution.** Extend `era_comparison.py`, renamed `weather_products.py`, to six products on
one common row set. The products are CAMS, ERA5, UKV, ICON-D2, ICON-EU and ICON global. It makes one
pooled run, with folds cut inside each era. It bounds the lead confound in two ways. The first is
the confound's direction: a lead advantage can only favour the shorter-lead product. The second is
a breakdown of the same losses by each ICON product's served lead. For the separate question of
whether a product's published beam/diffuse split adds anything, run the existing beam/diffuse
instrument per product and read its C − B contrast. A docs page states what each offline consumer
should read. Wind is #826. The comparison at a held-equal 1-day lead belongs to #810, which already
names the Previous Runs API as its instrument.

## Verdict, size and departures

**Verdict:** worth doing, with four departures from the issue body.

**Size: complex.** The five triggers:

- **What gets stored:** yes. The page's numbers will be cited.
- **The production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes. The lead control and the base row set are both
  choices.
- **Callers I could not name without searching:** no.

That buys both plan reviews and both diff reviews. It also buys at least two scientific-validity
reviews on Opus 5.5, with a re-run whenever a reviewer asks for one.

**Departures from the issue body:**

1. **No comparison at a held-equal lead here.** The only fixed-lead archive is Open-Meteo's
   Previous Runs API. It offsets in whole days, and starts in 2024 (ICON family) or August 2024
   (UKV). A comparison at a 1-day lead is a forecast comparison. That is what #810 owns, and #810
   names this instrument itself.

   For #809 the lead is bounded instead, as below. An offline consumer reads the archive as
   served, so the as-served ranking is the answer. The served lead is part of what a consumer gets
   from each product.
2. **SARAH-3 waits for #806.**
3. **Wind is #826.**
4. **Separating components from accuracy does not need new arms.** A split-against-global contrast
   would carry the re-encoding gain that the beam/diffuse study documented. There, adding a Erbs
   split that carries no new information still lowers error by 0.03 to 0.06 points. The existing
   instrument's C − B contrast is the right one: it compares the product's published split against
   a separation model's. That contrast is within one product, so it needs no common row set. The
   components question is therefore answered by `run_experiment.py --source <product>` for each
   ICON product and UKV. The beam/diffuse page already covers CAMS and ERA5.

## What changes, file by file

### `studies/beam_diffuse_split/era_comparison.py` becomes `weather_products.py` (via `git mv`)

- **`PRODUCTS`** gains `icon_eu → icon-eu` and `icon_global → icon-global`. CAMS is read from the
  all-hours build, `cams_allhours`, which is built with `--min-cams-reliability 0`. The default 0.9
  filter keeps only hours CAMS itself rates as reliable. That is a selection by one of the
  contestants, applied to every arm. ERA5 stays the base frame, as today, so power, geometry,
  temperature and the row set's base all come from the reanalysis.
- **One pooled run instead of three separate fits per era.** The dataset gets an `era` column
  (`pre` before 2026-02, `post` from then on). Folds come from `assign_folds(by=("site", "era"))`,
  so every fold holds both eras, and a model scoring a post-upgrade UKV row has trained on
  post-upgrade UKV. Each arm is fitted once. The eras (`all`, `pre`, `pre_matched`, `post`) become
  scopes of the bootstrap over those losses.
- **The lead breakdown.** Each row gets the served lead of each ICON product, inferred from the
  valid hour and the run cadence. The inferred lead is the valid hour minus the latest run at or
  before it, for runs every 3 hours (ICON-D2, ICON-EU) and every 6 hours (ICON global). The report
  adds a table of ICON-D2 − UKV and ICON-EU − UKV for each lead bucket. If a lead artefact drives
  the ordering, the gap shrinks towards lead 0.

  The mapping is an inference, and the page says so. `verify_icon_d2_lineage.py` measured that the
  freshest run matches, which supports it for ICON-D2. It is unmeasured for ICON-EU and ICON global.
- **The direction argument, stated in the report and the page.** Where a longer-lead product beats
  a shorter-lead one, the lead cannot explain the gap, because equalising leads could only widen
  it. Any pair where the shorter-lead product wins is flagged to #810 as unresolved.
- **Seasonal matching.** `pre_matched` is the pre-upgrade months with the same calendar months as
  `post`.
- **Output** goes to `STUDY_DATA_DIR / "beam_diffuse_weather_products"`: `losses.parquet`,
  `intervals.parquet` and `report.md`.

### Per-product components runs, with no code change

- For each of `ukv`, `icon-d2`, `icon-eu` and `icon-global`, run `run_experiment.py --source <x>`
  and `report_results.py --source <x>` on the rebuilt datasets.
- Record C − B and C − B-LEARNED, with their intervals and the positive control, on the page.

### Docs

- **New `docs/studies/weather-products-for-the-past.md`** in the `mkdocs.yml` nav. Its sections:
  - the question;
  - a products table giving each product's served lead, spatial domain, first date and latency;
  - the as-served results: arm table, contrasts, eras, lead breakdown;
  - the components table: C − B per product, linking to the beam/diffuse page;
  - a methods paragraph with one sentence per trap: row set, lead, folds, normalisation;
  - a recommendation per offline consumer;
  - limitations.
- **`docs/roadmap/data-sources.md`**: the four-product figures and the UKV − ERA5 contrast point to
  the new page.
- **`studies/beam_diffuse_split/README.md`**: rename the row, and redirect the era paragraph.

## Tests

Study-tier code, with no unit tests. Two things stand in for them. `assign_folds(by=...)` and the
bootstrap are tested in `packages/studies`. The script's evidence is its report, audited by the
scientific-validity reviews.

## Scientific-validity reviews

After the first results exist, two fresh Opus reviewers audit the study in turn. They cover the
design: row set, selection, leakage, the lead bound, normalisation, eras, and the interval method.
They also check the claims on the page against the reports. A re-run is made whenever a reviewer
asks for one.

## Verification

- The implement-issue set, plus `uv run mkdocs build --strict` with the rendered page read.
- A final re-run of `weather_products.py`, and of the four per-product runs, from the committed
  code before the page's numbers are frozen.

## Risks and open questions

1. **UKV narrowly beating a longer-lead ICON product.** If the as-served run produces this, the pair
   is unresolved, and the page points it to #810 rather than resolving it here.
2. **ECMWF IFS.** It is not a product here: its open data is 3-hourly, so what its hourly column
   holds at Open-Meteo is unmeasured. A candidate for #810.

## Revisions from the correctness and science review

The review found five real defects and three gaps; every one is taken.

1. **Leads follow the hour-ending convention.** A served hourly value is a backward mean over the
   hour ending at its label, so a run initialised at H can supply that hour only when the label is
   at least H + 1. A product's served lead at label T is T minus the latest run at or before T − 1,
   which takes the values 1, 2 or 3 for a 3-hourly model and 1 to 6 for ICON global. No ICON
   product has a lead-0 bucket.
2. **ERA5 is not lead-free.** Its hourly radiation comes from forecasts initialised at 06 and 18
   UTC at steps of 1 to 12 hours. The products table gives each product's served lead: UKV T+0,
   ICON-D2 and ICON-EU 1 to 3 hours, ICON global 1 to 6 hours, ERA5 1 to 12 hours, CAMS a
   satellite retrieval with no forecast step. Under the direction rule, UKV − ERA5 and ICON-D2 −
   ERA5 are unresolved rather than settled, and the page says so. The direction rule is stated as
   an assumption — error rises with lead — supported by the measured lead gradient, not as a
   theorem.
3. **ICON-EU's served lead is measured before its lead buckets are read.**
   `verify_icon_d2_lineage.py` becomes `verify_icon_lineage.py`, taking `--model icon-d2|icon-eu`,
   downloading every run up to each valid hour rather than four fixed runs, and reporting the
   root-mean-square difference per (valid hour, run) against Open-Meteo. ICON global is published
   by DWD only on its icosahedral grid, so its lineage stays unmeasured and its lead buckets are
   not interpreted.
4. **The lead table differences against ERA5, on hours 07 to 19 UTC.** ERA5's lead does not follow
   a 3-hour cycle, so differencing against it isolates the ICON product's lead, and trimming the
   day's ends removes UKV's cos-zenith rescaling artefact at low sun.
5. **Every arm is given an `era` feature**, identical across arms, so a pooled model can learn the
   UKV upgrade's change of mapping. The separate per-era fit is kept for the `post` scope as a
   sensitivity check.
6. **The components question runs on the common row set, with folds cut within eras.** Each
   product gets a `<product>_split` arm (arm C: its own global, beam and diffuse) and a
   `<product>_erbs` arm (arm B: its own global plus the Erbs split computed from that global), and
   the contrast is split − erbs, which cancels the re-encoding gain. The per-product
   `run_experiment.py` runs are dropped. A split contrast is read within one product; the table
   says so, although on the common row set the entries do at least share rows.
7. **The false-zero filter is made independent of the products.** Each build drops an hour with a
   zero half-hour only where its own global irradiance reads bright, and the inner join turns that
   into "drop the row if any product reads bright". The common set instead drops every hour with a
   zero half-hour, whatever any product says, recomputed from the power table with
   `studies.power.hourly_from_half_hourly`. That removes the same rows from every arm, at the cost
   of some genuine low-light zeros.
8. **Known-bad rows are dropped from every arm:** ICON-EU's corrupt block (2023-06-21 01:00 to
   06:00 UTC), and 21 to 31 January 2026, when UKV had already changed but the month is labelled
   pre-upgrade.
9. **The deciding contrasts are named before the run.** The recommendation rests on four pooled
   contrasts; every other contrast is exploratory:
   - `cams − icon_d2`: does a satellite retrieval beat the best weather model?
   - `icon_eu − icon_d2`: what does the Great-Britain-wide ICON cost against the regional one?
   - `icon_eu − ukv`: which Great-Britain-wide weather model is better?
   - `icon_global − icon_eu`: what does the global ICON cost against the European one?
10. **The `post` scope has only eight months, so a percentile bootstrap from eight clusters
    under-covers.** Every `post` interval carries that caveat and the per-fold sign count.
11. **A cheap scope splits UKV at 2024-08-12**, before which Open-Meteo's UKV is a backfill from an
    unnamed source.
12. **Evidence for the consumer recommendations beyond accuracy.** A leave-one-site-out arm per
    product — trained on five sites' capacity-normalised power and scored on the sixth, with no
    per-site fitting — measures how well each product transfers to a generator with no metered
    history, which is the situation disaggregation and capacity estimation are in. The products
    table gives domain coverage against NGED's licence area, history length, and publication
    latency, each with a source. The page states that all six generators sit in one 25 by 23 km
    box inside ICON-D2's domain, so a ranking here is regional evidence.
13. **The page names the capacity snapshot it rests on**, because #825 may move the absolute
    figures.

## Rejected from the simplicity review

- **Dropping the CAMS all-hours primary in favour of the filtered set.** The filter is a selection
  by one contestant, so the unfiltered set is the fair primary. The filtered run is kept as a cheap
  sensitivity check.
