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

## Rejected from the simplicity review

- **Dropping the CAMS all-hours primary in favour of the filtered set.** The filter is a selection
  by one contestant, so the unfiltered set is the fair primary. The filtered run is kept as a cheap
  sensitivity check.
