# Plan: charts for the two weather-product study pages (#830)

**Problem.** The two weather-product study pages give their results only as tables and prose:
[Which weather product best describes past
sunshine?](https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-the-past/)
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

**Verdict: worth doing as described.** The charts restate results already on the pages. Only three
charts need numbers the reports lack, and each is labelled below.

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

**Departures from the issue:** none. The issue asks for "several" charts through each body, and this
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
entry says otherwise. The unit is percentage points of capacity, which is the unit of every contrast
on both pages. "From the report" means the rows are parsed from that study's `report.md`.

### Solar page (`weather-products-for-the-past.md`)

1. **Headline, under the opening paragraphs.** One row per product, ordered by mean absolute error,
   best at the top: each product's error minus ERA5's, with the paired interval. ERA5 sits on the
   zero rule. Each row label carries the product's own error ("CAMS · 5.05%"), because the absolute
   level matters as much as the contrast. From the report: "Every product against ERA5, by scope",
   scope `all`, and the first table.
2. **"CAMS describes past sunshine best": CAMS − ICON-D2, broken down.** Three facets: generator (A
   to F), season, and calendar year, 2023 to 2026. The report's 2022 row is one month with a
   spuriously narrow interval, and is left out, as it is on the page. From the report: "CAMS against
   ICON-D2, broken down".
3. **"ICON-D2 is the best weather model as served": ICON-D2 − ICON-EU by hour of day, 07 to 19 UTC.**
   Hour runs along the x axis, the contrast up the y axis, and point shape marks the served lead (1,
   2, or 3 hours). **New:** the per-hour intervals, 13 bootstraps with `bootstrap_difference`. The
   report prints the per-hour point estimates, which the script checks its own estimates against
   before drawing.
4. **"ICON-EU against ICON global and UKV": each construction against ICON-EU.** Rows: ICON global,
   UKV as served, UKV rebuilt from its two snapshots, and rebuilt UKV with neighbouring hours
   (compared context-against-context, as on the page). From the report.
5. **Same section: UKV − ERA5 across scopes.** Rows: the whole record, since Open-Meteo's own UKV
   downloader started, after the upgrade in the pooled model, and after the upgrade fitted alone.
   The intervals disagree in sign, and the title says so: "UKV against ERA5 is unresolved". From the
   report.
6. **"A product's own direct beam adds little": each product's own split minus the Erbs split.**
   Six rows, on an axis wide enough to show that every effect is smaller than the gaps in chart 1.
   From the report.
7. **"The ranking holds for a generator trained on its neighbours".** One row per product, with two
   points: the per-generator model and the model trained on the other five generators, told apart by
   shape. Both are mean absolute errors, so this chart has no zero rule and no interval. From the
   report's first table.
8. **"Implied capacity from month to month": each product's implied capacity by calendar month, as a
   percentage above or below its own annual mean.** Six small multiples, one line per panel in the
   family colour, sharing one y axis. The quantity is pooled over the six generators, so no
   generator's output appears. **New:** the 12 monthly values, which the report does not print. The
   report prints only December.

### Wind page (`weather-products-for-past-wind.md`)

1. **Headline, under the opening paragraphs.** The same form as the solar headline: five products,
   each against ERA5, each labelled with its own error. From the report.
2. **"Both leaders' advantage over ERA5 is mostly a summer result": each product against ERA5, by
   half-year.** Two facets: April to September, and October to March. **New:** ICON global's two
   half-year rows, 2 bootstraps. The rest come from the report.
3. **"ICON-D2 and UKV cannot be separated robustly": ICON-D2 − UKV under every setting the page
   quotes.** Rows, grouped: the whole window, before and after the upgrade; April to September and
   October to March; the served 100 m wind, the second hyperparameter setting, and UKV at 80 m;
   ICON-D2 at 0, 1, and 2 hours into its run. One chart carries both the robustness paragraph and the
   equal-lead paragraph. From the report.
4. **"ICON global is the weakest product": the steps in ICON global's served wind.** Two panels.
    - **Left:** the monthly mean ratio of ICON global's 80 m speed to ICON-EU's, one line per
      generator (W1 to W3), with a vertical rule at each date in `STEP_DATES`. The generator with the
      steps is drawn in the weather-model Data Blue, the other two in the theme's grey
      (`ENSEMBLE_LINE`). **New:** the ratios, read from `wind_icon_global.parquet` and
      `wind_icon_eu.parquet`. The chart shows wind speeds only, never output.
    - **Right:** ICON global − ICON-EU per generator, with and without the model being told which
      side of the steps each hour falls on, told apart by shape. From the report.
5. **"UKV's advantage over ERA5 excludes zero at two of the three generators": each leading product
   against ERA5, per generator.** Three facets (UKV, ICON-D2, ICON-EU), one row per generator. From
   the report.

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
    - `interval_chart(*, intervals, category, title, subtitle, x_title, reference_label,
      better_direction, colour=None, shape=None, facet=None)`: the dots, the interval lines, the
      labelled zero rule, the direction label, and the fixed colour domain, drawn from a Polars
      frame with `difference`, `lower_95`, and `upper_95` columns.
    - `report_contrasts(*, report_path)`: parses every contrast row of a study report, in the format
      `studies.bootstrap` results are printed in (`| scope | treatment − reference | difference |
      [lower, upper] | …`). It returns a Polars frame with one row per scope and contrast.
    - `select_contrasts(*, contrasts, wanted)`: returns the wanted rows in the order asked for, and
      raises naming every missing row, so a renamed arm cannot leave a chart looking complete with a
      row absent.
- **`packages/studies/pyproject.toml`:** add `altair`, `plotting`, and `vl-convert-python`, which
  `chart.save` needs to write SVG. The chart scripts then run with plain `uv run`.
- **`packages/studies/tests/test_charts.py` (new).** Tests below.
- **`studies/beam_diffuse_split/weather_products.py`:** split `_implied_capacity` so that
  `_log_capacity_by_month(*, frame)` returns the per-product frame and the report table reads it.
  Nothing is renamed: the chart scripts import private names, as `wind_products.py` already does.
- **`studies/beam_diffuse_split/weather_product_charts.py` (new):** reads `report.md`, computes the
  13 per-hour intervals from `losses.parquet`, rebuilds the common rows with `_joined` and
  `_common_rows` (no fitting) for chart 8, and writes each SVG to `docs/studies/assets/`.
- **`studies/beam_diffuse_split/wind_product_charts.py` (new):** the same, for the wind page.
- **The two pages:** one image per chart, placed under the heading or bolded lead it supports.
- **`studies/beam_diffuse_split/README.md`:** the two new scripts, and the `svgo` step.
- **`make_chart.py` and `make_figures.py` stay as they are.** Moving them onto `studies.charts`
  would redraw the published beam/diffuse figures, whose numbers #825 already expects to move.

**The 13 SVGs go through `npx svgo@4 --multipass --precision=1 --final-newline`** before they are
committed, as `CLAUDE.md` requires.

## Design-philosophy check

This is R&D code, so it fails fast. A chart script raises if a contrast row, an arm, or a site label
it expects is missing, rather than drawing a chart with a row silently absent. No production path,
asset, or asset check is touched. Anonymisation holds because no chart plots output: generators
appear only as A to F and W1 to W3, and the one per-generator time series (wind chart 4) plots
wind-speed ratios.

## Tests

`test_charts.py`. Each assertion fails on `main` today, because the module does not exist:

- **`report_contrasts` reads what the report prints.** A three-row report fragment, including a
  bolded "**yes**" column, a negative lower bound, and a scope containing a space, parses to the
  exact differences and bounds.
- **`select_contrasts` raises on a missing row**, naming it, and returns rows in the order asked for.
- **The plotted numbers are the input numbers.** `interval_chart` given three rows yields a spec
  whose inline data holds exactly those values, in the given row order.
- **Colour follows the family, not the rank.** The colour scale's domain and range are
  `FAMILY_COLOURS` in full, even when the input holds only one family. Without that, a chart holding
  only weather models would repaint blue as orange.
- **The zero rule sits at zero**, and the direction label points the way `better_direction` says.

The chart scripts, like the study scripts, are not unit-tested. Their check is the verification
below.

## Verification

- The green-before-push set, as in `ci.yml`: `ruff check`, `ruff format`,
  `uv run --all-packages ty check`, `uv run pytest`, pydoclint, `pymarkdown`, and
  `scripts/lint/check_docs_links.py`.
- `uv run mkdocs build --strict`, then read both rendered pages under `site/`.
- **Look at every chart as a PNG render** (`dataviz` step 7), checking for label collisions and
  clipped intervals, and for whether the chart tells its story with the page text hidden.
- **The implied-capacity split changes nothing.** Rebuild the common rows, run the report half of the
  split function, and diff its table against the implied-capacity block of `report.md`.
- **The new intervals are consistent with the report.** The per-hour point estimates match the
  report's hour-by-hour table to the printed digit.
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
