# Plan: charts for the two weather-product study pages (#830)

**Problem.** The two weather-product study pages give their results only as tables and prose:
[Which weather product best describes past
sunshine?](https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-solar/)
and [Which weather product best describes past
wind?](https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/).
Many technical readers skim a page's charts before reading any text. On these two pages that reader
finds no chart at all, so nothing shows how large each gap is next to its uncertainty.

**Solution.** Draw 13 anonymised charts in the Open Climate Fix (OCF) brand colours: 8 on the solar
page and 5 on the wind page. Each page opens with a headline chart giving every product's error
against ERA5's, with 95% intervals. The remaining charts sit in the sections whose claims they
support. **Each chart, with its title, subtitle, axis labels, and legend, tells its part of the
story without the prose around it.** Almost every number is read from the report each study
already wrote, so a chart cannot disagree with its page. No model is refitted. A small tested
module in `packages/studies` holds the chart form and the colour rule, which every chart shares.

## Verdict, size and departures

**Verdict: worth doing as described.** The charts restate results already on the pages. Five charts
need some numbers the reports lack, and each such number is labelled below.

**Size: complex.** The five trigger answers:

- **What gets stored:** no. The charts add SVG files under `docs/studies/assets/` and change no
  table, contract, or asset.
- **Production serving path:** no.
- **A degradation rule:** no.
- **More than one defensible design:** yes. Each chart's form, the headline's reference product, and
  the colour rule are all choices a reviewer could reasonably make differently.
- **Code whose callers cannot be named without searching:** no. The only callers are two new
  scripts.

**Reviews:** both plan reviews (simplicity, then correctness), and both diff reviews. The second plan
review and the first diff review are also briefed to judge whether any chart misleads. Study code
is not reviewed by humans, and a misleading chart is the scientific failure a chart can commit.

**Departures from the issue:** one addition. The correctness review found that the wind page
misstates the size of the steps in ICON global's served wind, which the step chart would contradict,
so this PR corrects that sentence. The issue asks for "several" charts through each body, and this
plan names them.

## Every chart stands alone

**A reader who sees only the chart must come away with the finding, its unit, its scope, and what
the interval means.** Each chart therefore carries:

- **A title stating the finding**, such as "CAMS describes past sunshine best, by a wide margin".
  The title matches the section heading or bolded lead the chart sits under.
- **A subtitle naming the quantity, the unit, and the scope**, such as "Mean absolute error minus
  ERA5's, percentage points of capacity. Dots: estimate. Lines: 95% interval from resampling whole
  months. 6 solar farms, December 2022 to September 2026."
- **A marked reference.** A vertical rule sits at zero, labelled with what zero means ("same as
  ERA5"), and a side label says which direction is better ("← better than ERA5").
- **A legend for every encoding beyond position**: product family, served lead, generator label.
- **Exploratory results marked on the chart.** A chart holding a contrast that was not named before
  the run says "exploratory" in its subtitle.

The markdown alt text repeats the title, and the page's tables stay as the chart's table view.

## The colour rule

**Colour marks what kind of product a row is (satellite retrieval, reanalysis, or weather model),
not which product it is.** The product's name is always on the axis beside its marks, so identity
never depends on colour. The colours come from the "main data colours" in OCF's 2025 brand
guidelines, which are the colours those guidelines allow in published material. The guidelines'
additional data colours, among them the dark green `#009C75` and the amber `#FC9700`, are marked
for internal use only, so no study chart uses them.

| Palette, checked with the `dataviz` skill's `validate_palette.js` against the OCF background `#FFFBF5` | Result |
|---|---|
| One main data colour per solar product | **Fails.** Data Purple `#B701FF` and Data Blue `#306BFF` sit 2.0 ΔE apart under deuteranopia, below the floor of 8. |
| Brand Orange `#FF4901` (satellite), Data Sky `#10C5F7` (reanalysis), Data Blue `#306BFF` (weather model) | **Passes every check,** with a worst colour-vision pair of 19.0 ΔE. Data Sky has only 2.0:1 contrast with the background, which the validator accepts when every mark is labelled, as every row here is. |

**The satellite retrieval is in Brand Orange because it is the product that stands apart on the
solar page.** The weather models, which fill most rows on both pages, share Data Blue.

**Two conditions of one product use the colour and its light shade, plus a second point shape.**
The guidelines give each main data colour a lighter shade "to create mono-coloured comparative
graphs": Blue Light `#9CB6E1`, Sky Blue Light `#A3D6E0`, and Brand Orange Light `#FF8F73`. The
validator fails a light shade as a stand-alone series colour, for low chroma or for sitting under 15
ΔE from its parent for normal vision. So a light shade never carries a distinction alone: it always
comes with a hollow point where the parent colour's point is filled.

**Typography follows the brand guidelines through `plotting.ocf_theme`.** A separate PR adds the
brand's colours and typography to the theme: Matter XH for text, Matter Semi Mono for labels and
data, and the brand's type-size scale. Each font falls back to DM Sans or Roboto Mono, which OCF's
slide template uses in place of the commercial Matter fonts. The study charts inherit all of this
from the theme and set no font themselves. That PR lands first, and this one is rebased on it.

**The one style the charts set themselves is the slide template's figure caption.** OCF's slide
template (slides 17 to 19) sets "Figure 1:" and the chart title above the chart, under a thin rule in
the guidelines' warm grey `#D9D0CA`. `interval_chart` draws the title and subtitle the same way. The
theme's own grid colour stays as it is, which #833 tracks.

**Out of scope, but reported:** the beam/diffuse page's `make_figures.py` draws UKV in Data Purple
beside CAMS in Data Blue, the pair that fails above, and colours a fourth setup in the internal-only
dark green. Issue #832 tracks the fix.

## The charts

**Every chart is a dot at the estimate and a horizontal line spanning its 95% interval,** unless the
entry says otherwise. The unit is percentage points of capacity, and every subtitle says that
capacity means each generator's 99th-percentile output. "From the report" means the rows are
parsed from that study's `report.md`, selected by section heading, scope, and contrast. "New" means
the chart script computes the rows itself.

**Every row says whether its contrast was named before the run.** Named rows are the exception, so
they are the ones marked: a named row's label is bold and ends "(named before the run)". A chart
holding any row added after the first run says "post hoc" in its subtitle. A row's colour is the
family of the contrast's first product.

**A chart of differences from one reference cannot show whether two other rows differ.** Each
headline chart therefore has a second panel beside it holding the page's named contrasts with their
paired intervals, and its subtitle says: "Intervals are against ERA5. Overlapping intervals do not
mean two products are indistinguishable; the right-hand panel compares them directly."

### Solar page (`weather-products-for-past-solar.md`)

1. **Headline, under the opening paragraphs.**
    - **Left panel:** one row per product, ordered by mean absolute error, best at the top. Each
      product's error minus ERA5's, with the paired interval; ERA5 sits on the zero rule. Each row
      label carries the product's own error ("CAMS · 5.05%"). From the report: section "Every
      product against ERA5, by scope" (exploratory), scope `all`, and the error table.
    - **Right panel:** the four named contrasts, scope `all`. From the report: "Deciding contrasts,
      named before the run".
2. **"CAMS describes past sunshine best": CAMS − ICON-D2, broken down (exploratory).** A reference
   row for the whole record (−2.651) above three facets: generator (A to F), season, and calendar
   year, 2023 to 2026. The report's 2022 row is one month in one fold, and is left out, as it is on
   the page. From the report: "CAMS against ICON-D2, broken down".
3. **"ICON-D2's advantage over ICON-EU fades within hours of each run" (post hoc).** Two panels
   sharing a y axis.
    - **Left:** ICON-D2 − ICON-EU at each hour from 09 to 16 UTC, the hours the page's table shows,
      with point shape marking the served lead. Hours 07, 08, and 17 to 19 are left out: they exist
      in only 27 to 38 of the 46 months, and at low sun a difference in points of capacity shrinks
      whatever the lead. **New:** the per-hour intervals, 8 bootstraps.
    - **Right:** the whole-record contrast and the three matched-lead rows the page's bolded lead
      cites. From the report.
4. **"ICON-EU against ICON global and UKV".** Rows, with each contrast written as ICON-EU's rival
   minus ICON-EU, so a positive value means the rival is worse:
    - ICON global, on all hours (named), on the hours whose leads are equal, and on the hours ICON
      global is served further ahead. These rows carry the section's first claim.
    - UKV as served (named), UKV's two snapshots given as a pair (post hoc), UKV rebuilt as their
      mean (post hoc), and rebuilt UKV with neighbouring hours (post hoc), whose label names its
      reference: "against ICON-EU with neighbouring hours". The pair row is the one construction
      that beats ICON-EU with an interval excluding zero, and it stays on the chart.

   The report prints the UKV rows as ICON-EU minus UKV, so the script flips them with
   `flip_contrast` (below). From the report.
5. **"UKV against ERA5 is unresolved" (exploratory).** Rows: the whole record, since Open-Meteo's
   own UKV downloader started, after the upgrade in the pooled model, and after the upgrade fitted
   on those months alone. The subtitle says the post-upgrade rows rest on 8 monthly clusters, so
   their intervals under-cover. From the report, where the two post-upgrade rows share a scope and a
   contrast and are told apart by their section.
6. **"Every product except ERA5 gains 0.03 to 0.11 points from its own direct beam"
   (exploratory).** Each product's own split minus the Erbs split. The x axis spans the same range
   as the headline chart's left panel, which is set explicitly in both. From the report.
7. **"The ranking holds for a generator trained on its neighbours".** The four named contrasts, each
   drawn twice: from the per-generator models, and from models trained on the other five generators
   with the scored months withheld everywhere. Told apart by shape, and by the light shade of the
   row's colour. From the report.
8. **"Implied capacity by calendar month".** Six small multiples, one line per product in its
   family colour, sharing one y axis. The plotted value for a calendar month is the exponential,
   minus one, of the mean over that month's site-months of the site's calendar-month mean log
   implied capacity minus the site's overall mean. That is the seasonal term `_implied_capacity`
   already computes, and its December value is the one the report prints. The subtitle says there
   is no interval, and the title makes no claim about which product is right, because the page
   cannot say. **New:** the 12 monthly values; the report prints December only.

### Wind page (`weather-products-for-past-wind.md`)

1. **Headline, under the opening paragraphs.** The same two panels as the solar headline: five
   products against ERA5, labelled with their own errors, and the four named contrasts. The left
   panel's ICON-EU and UKV rows are named contrasts, marked as such; its ICON-D2 and ICON global
   rows are exploratory. From the report.
2. **"Both leaders' advantage over ERA5 is mostly a summer result" (exploratory).** One row per
   product (UKV, ICON-D2, ICON-EU), with two points each, April to September and October to March,
   told apart by shape and by the light shade. ICON global is left out: its half-year split is
   confounded by the steps in its served wind, and the page does not discuss it. From the report.
3. **"ICON-D2 against UKV: level across settings, ahead at equal lead" (post hoc).** ICON-D2 − UKV,
   in four labelled groups: the whole window, before and after the upgrade; April to September and
   October to March; the served 100 m wind, the second hyperparameter setting, and UKV at 80 m; and
   ICON-D2 at 0, 1, and 2 hours into its run. The subtitle carries the post-upgrade caveat. From the
   report.
4. **"ICON global is the weakest product, and half of its gap is a pair of steps in its served
   wind" (post hoc).** Two panels.
    - **Left:** ICON global's wind speed divided by ICON-EU's, as a ratio of fortnightly means, at 10
      m and at 80 m, for the generator with the steps, against the same ratio pooled over the other
      two generators. A vertical rule marks each date in `STEP_DATES`, and a horizontal segment
      marks each period's mean. The panel carries no generator label: the steps are a fingerprint
      that public ICON archives could match to a grid cell. **New:** the ratios, read from
      `wind_icon_global.parquet` and `wind_icon_eu.parquet`. The chart shows wind speeds only, never
      output.
    - **Right:** ICON global − ICON-EU for "the generator with the steps" and "the other two
      generators", with and without the model being told which side of the steps each hour falls
      on, told apart by shape. **New:** the pooled row for the other two generators, 2 bootstraps;
      the generator with the steps comes from the report.
5. **"UKV's advantage over ERA5 excludes zero at two of the three generators" (exploratory).** Three
   facets (UKV, ICON-D2, ICON-EU), one row per generator (W1 to W3). The UKV and ICON-EU rows come
   from the report. **New:** the three ICON-D2 rows, 3 bootstraps.

**The page's account of the steps is wrong, and this PR corrects it.** The page says ICON global's
wind relative to ICON-EU's "falls by roughly 12% in early June 2025 and rises by about as much in
early June 2026, at 10 m and at 80 m alike". Measured over the three periods at the generator with
the steps, the ratio of means falls from 1.128 to 0.962 and rises to 1.111 at 10 m, about 15%, but
falls from 1.028 to 0.948 and rises to 1.020 at 80 m, about 8%. The sentence becomes: "falls by
about 15% at 10 m and 8% at 80 m in early June 2025, and rises by about as much in early June
2026". The chart shows both heights.

**Left out:** a week of predicted against measured power. The chart would have to show one
generator's output, even normalised, and the beam/diffuse page already shows that the pipeline
produces a sane forecast.

## What changes, file by file

- **`packages/studies/src/studies/charts.py` (new).** The shared, tested chart machinery. Three
  studies now draw dot-and-interval charts: `make_chart.py` and `make_figures.py` each write the
  form inline, and this change adds two more scripts that need it.
    - `ProductFamily = Literal["satellite", "reanalysis", "weather model"]`.
    - `FAMILY_COLOURS: Final[dict[ProductFamily, str]]` and `FAMILY_COLOURS_LIGHT`: the three
      main data colours above and their light shades, read from `plotting.ocf_theme`.
    - `interval_chart(...)`: the dots, the interval lines, the labelled zero rule, the direction
      label, the named-row marking, the figure caption, an explicit x domain, and a colour scale
      whose domain is every family while the legend lists only the families present. It draws from
      a Polars frame with `difference`, `lower_95`, and `upper_95` columns.
    - `report_contrasts(*, report_path)`: parses every table whose header is the contrast header
      `weather_products._contrast_line` writes, and nothing else. It returns one row per table row,
      with the `section` taken from the preceding `####` heading, and `scope`, `treatment`,
      `reference`, `difference`, `lower_95`, and `upper_95`. It reads both the minus sign `−` and a
      hyphen as negative.
    - `report_errors(*, report_path)`: reads the first table, each product's mean absolute error.
    - `select_contrasts(*, contrasts, wanted)`: selects by (section, scope, treatment, reference),
      returns the rows in the order asked for, and raises naming every key that matches no row or
      more than one.
    - `flip_contrast(*, row)`: turns treatment − reference into reference − treatment, negating
      the estimate and swapping and negating the bounds.
- **`packages/studies/pyproject.toml`:** add `altair`, `plotting`, and `vl-convert-python`, which
  `chart.save` needs to write SVG.
- **`packages/studies/tests/test_charts.py` (new).** Tests below.
- **`studies/beam_diffuse_split/weather_products.py`:** split `_implied_capacity` so that
  `_log_capacity_by_month(*, frame)` returns the per-product frame and the report table reads it.
  Nothing is renamed: the chart scripts import private names, as `wind_products.py` already does.
- **`studies/beam_diffuse_split/weather_product_charts.py` (new).**
    - Reads `report.md`.
    - Computes the 8 per-hour intervals from `losses.parquet`, after reproducing the report's "both
      at lead 1 h" row through the same code path and asserting it matches to the printed digit.
    - Rebuilds the frame chart 8 needs with the same chain `main()` runs,
      `with_export_cap(_with_eras(_add_time_features(_common_rows(_joined()))))`, asserts it holds
      79,384 rows, and asserts each product's December value matches the report.
    - Writes each SVG to `docs/studies/assets/`.
- **`studies/beam_diffuse_split/wind_product_charts.py` (new).** The same for the wind page. Its new
  bootstraps read only `setting == "pooled"` rows through `_renamed(..., suffix="_wind")` and
  `_scoped`, because both settings share arm names and an unfiltered join multiplies rows. Before
  computing them it reproduces two printed rows (UKV − ERA5 from April to September, and ICON-EU −
  ERA5 at W1) and asserts both match.
- **The two pages:** one image per chart, under the heading or bolded lead it supports, and the
  corrected sentence on the steps.
- **`studies/beam_diffuse_split/README.md`:** the two new scripts, and the `svgo` step.
- **`make_chart.py` and `make_figures.py` stay as they are.** Moving them onto `studies.charts`
  would redraw the published beam/diffuse figures, whose numbers #825 already expects to move.

**The 13 SVGs go through `npx svgo@4 --multipass --precision=1 --final-newline`** before they are
committed, as `CLAUDE.md` requires.

## Design-philosophy check

This is R&D code, so it fails fast. A chart script raises if a contrast row, an arm, or a site label
it expects is missing or ambiguous, rather than drawing a chart with a row silently absent or wrong.
No production path, asset, or asset check is touched. Anonymisation holds because no chart plots
output: generators appear only as A to F and W1 to W3, and the one per-generator time series (wind
chart 4) plots wind-speed ratios with no generator label.

## Tests

`test_charts.py`. Each test is written against the bug it exists to catch:

- **`report_contrasts` reads verbatim report lines**, asserting exact values: a minus sign `−` and a
  hyphen, a `+` sign, asymmetric bounds, `**yes**`, `4 of 4`, a row count with a thousands comma,
  and the scope `ICON global lead 1 to 3 h, equal to ICON-EU's, 07–19 UTC`.
- **It skips a non-contrast table** sitting between two contrast tables, and assigns each row the
  section of the heading above it.
- **`select_contrasts` raises on a missing key and on an ambiguous key**, naming each, when two
  sections hold the same scope and contrast; and it returns rows in the order asked for.
- **`flip_contrast` negates the estimate and swaps the bounds**, on asymmetric bounds.
- **`report_errors` reads the error table.**
- **The plotted numbers are the input numbers**, in the given row order.
- **Colour follows the family, not the rank.** The colour scale's domain and range are
  `FAMILY_COLOURS` in full when the input holds one family, and the legend lists only that family.
- **The zero rule sits at zero**, and the direction label points the way `better_direction` says.

The chart scripts, like the study scripts, are not unit-tested. Their check is the reproduction of
printed rows above, and the verification below.

## Verification

- The green-before-push set, as in `ci.yml`: `ruff check`, `ruff format`,
  `uv run --all-packages ty check`, `uv run pytest`, pydoclint, `pymarkdown`, and
  `scripts/lint/check_docs_links.py`.
- `uv run mkdocs build --strict`, then read both rendered pages under `site/`.
- **Look at every chart as a PNG render** (`dataviz` step 7), checking for label collisions and
  clipped intervals, and for whether the chart tells its story with the page text hidden.
- **The implied-capacity split changes nothing:** the rebuilt frame reproduces the report's
  implied-capacity table exactly.
- **Every new number is anchored to a printed one**, by the reproduction asserts above.
- `grep` every committed SVG for every generator identifier and name in the private roster.

## Findings from the simplicity review

- **Accepted: read the intervals from the report instead of recomputing them.** Almost every interval
  is already printed there, so parsing the report makes each chart agree with its page by
  construction. The `chart_intervals.csv` comparison is dropped. The plan's claim that chart 2's
  per-breakdown intervals were new was wrong: the report prints them.
- **Accepted: no public renames of study-script functions**, and a narrower check of the
  `_implied_capacity` split.
- **Accepted: replace the ICON-D2 − UKV lead chart with one showing ICON-D2 − UKV under every
  setting (wind 3)**, and fold the per-generator step contrast into the step chart as a second
  panel.
- **Accepted: a chart for "UKV against ERA5 is unresolved" (solar 5)**, added rather than swapped
  in.
- **Rejected: drop the `studies.charts` module, its tests, and the new dependencies.** Three studies
  now draw this chart form. `packages/studies` exists to hold the machinery studies share, tested,
  and the `vl-convert-python` wheel is what writing any SVG needs. The review's point that a spec
  test checks wiring is taken: the tests cover the report parser, the missing-row guard, and the
  colour rule, which are where a chart silently goes wrong.
- **Rejected: merge the wind headline with the half-year chart.** The headline answers the page's
  question in one glance; the half-year split is the first qualification, and gets its own chart
  directly below.
- **Rejected: plot the report's per-hour point estimates without intervals.** Every contrast chart
  carries its uncertainty. The 13 bootstraps take seconds.
- **Rejected: one OCF hue for every row.** The family colour tells a skim-reader at once that the
  satellite retrieval stands apart, at the cost of a three-entry dict.
- **Rejected: cut or defer the implied-capacity chart.** It is the one place the page shows all 12
  months, and the December dip it shows is the evidence behind the capacity-estimation advice.

## Findings from the correctness review

- **Accepted, all of the must-fix findings:**
    - The report parser reads contrast tables only, keys rows by section as well as scope and
      contrast, and raises on an ambiguous key.
    - Each headline chart gains a panel of the named contrasts, and a subtitle warning that
      overlapping intervals against ERA5 do not mean two products are indistinguishable.
    - The wind page's per-generator ICON-D2 rows are new numbers, and are labelled so.
    - The new wind bootstraps read only the pooled setting. Each script reproduces a printed row
      through the same code path before computing a new one.
    - Chart 8 uses the full chain `main()` runs, and its quantity is defined exactly.
    - Solar chart 4 flips the UKV rows to one sign convention, keeps the UKV pair row, names the
      context row's reference, and adds the two lead-split rows for ICON global.
    - Wind chart 4 plots a ratio of fortnightly means at both heights, and the page's "12% at 10 m
      and 80 m alike" is corrected to the measured 15% and 8% (checked independently while
      triaging).
- **Accepted, the should-fix findings:** named rows marked on every chart; the post-upgrade caveat
  on the chart; wind chart 2 drawn one row per product, with ICON global left out; wind chart 3
  retitled and grouped; solar chart 3 restricted to 09 to 16 UTC with the matched-lead rows beside
  it; solar chart 6 on the headline's x range and titled with the page's bolded lead; solar chart 7
  drawn as contrasts; solar chart 2 given its reference row; the tests rewritten against real report
  lines; wind chart 4's left panel without a generator label.
- **Accepted, the minor notes:** a contrast row takes its first product's family colour; the legend
  lists only the families present; every subtitle defines capacity.

## Risks and open questions

- **Which product should the headline measure against?** Recommendation: ERA5 on both pages. ERA5 is
  the default reanalysis, it is on both pages, and it is the reference of two of the wind page's four
  named contrasts. The alternative is each page's best product, which puts the leader on zero and
  hides its own interval.
- **Should the headline show each product's own error with an interval, instead of its difference
  from ERA5?** Recommendation: no. The products are scored on the same hours, so their errors share
  most of their month-to-month variation. Intervals on the errors would overlap even where the paired
  difference clearly excludes zero, and a reader would take the overlap to mean no difference. The
  error goes in the row label instead, without an interval.
- **Static SVG or interactive chart?** The `dataviz` skill defaults to a hover layer. Every study page
  uses static SVG through MkDocs. Recommendation: static SVG, with the page's tables as the table
  view.
- **Parsing a markdown report ties the charts to the report's text format.** The parser raises on
  any row it cannot read, so a format change fails loudly. The cleaner long-term design is for each
  study script to write its intervals to a table that the report and the charts both read. That
  needs a re-run of both studies, which belongs to the directory restructuring in #829.

## Departures during implementation

- **`interval_chart` is two functions, `interval_panel` and `figure`.** Every chart has one to four
  panels, so the panel (dots, lines, zero rule, direction label, named-row marking, colour scale)
  and the caption (the "Figure N:" title, the subtitle, the warm-grey rule) are drawn separately
  and composed. A public `ticks` sets round tick values, because an explicit domain with
  `nice=False` otherwise gives ticks such as −4.5, −3.5.
- **`flip_contrast` takes a frame of rows (`contrasts`), and `report_errors` takes a `column`.** The
  solar report's error column is "Global only" and the wind report's is "All sites".
- **The condition key is a small chart of its own.** A Vega-Lite shape legend draws every symbol
  alike, so it cannot show that the second condition is hollow.
- **The 2px surface ring on dots is dropped.** At this dot size it cut every short interval line in
  two, which read as a dashed line.
- **Solar chart 3's two panels share the x scale, not a y axis.** The two panels hold different
  rows (hours, and contrasts).
- **Solar chart 8 has one small multiple per product.** The plotted quantity is pooled over
  generators, so a panel per generator is not possible. Its title is "CAMS's implied capacity
  swings the most with the seasons", which makes no claim about which product is right.
- **The script reproduces the whole implied-capacity table**, line for line, not only December.
- **The wind charts are numbered in page order.** The per-generator chart (plan's wind 5) supports
  the same paragraph as the half-year chart, so it is Figure 3; ICON-D2 against UKV is Figure 4,
  and the steps chart Figure 5.
- **Wind chart 4's left panel draws the other two generators in mid-grey** (`ocf.ENSEMBLE_LINE`) as
  a de-emphasised comparison, on a stroke scale kept apart from the family colour scale. The script
  asserts the three period ratios at each height that the corrected sentence rests on.
- **Row labels round each product's error half up** (8.975 to 8.98), as the page does; Python's
  `format` would give 8.97.
- **`packages/studies/README.md` lists the new `charts` module**, and its ownership sentence now says
  the package owns the chart form but not which charts a page draws.
