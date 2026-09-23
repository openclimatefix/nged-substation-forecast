# Plan: charts for the two weather-product study pages (#830)

**Problem.** The two weather-product study pages give their results only as tables and prose. The
pages are [Which weather product best describes past
sunshine?](https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-the-past/)
and [Which weather product best describes past
wind?](https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/).
A skim-reader cannot see how large each gap is next to its uncertainty, and neither page opens with
a picture of its headline result.

**Solution.** Draw 10 anonymised charts in the Open Climate Fix (OCF) brand colours, 5 per page, from
the per-row losses the two studies already saved. No model is refitted. Every interval comes from
the same whole-month paired bootstrap the pages already quote. Each page opens with a headline
chart: every product's error measured against ERA5's, with a 95% interval. The remaining 8 charts
sit in the sections whose claims they support. A small tested module in `packages/studies` holds
the chart form and the colour rule, which every chart shares. Two chart scripts, one per study,
draw the charts.

## Verdict, size and departures

**Verdict: worth doing as described.** The charts only restate results already on the pages, plus a
few new intervals, labelled below, for breakdowns the pages give only as ranges.

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
review and the first diff review are also briefed to judge whether any chart misleads: studies code
is not reviewed by humans, and a misleading chart is the scientific failure a chart can commit.

**Departures from the issue:** none. The issue asks for "several" charts through each body; this plan
settles on four body charts per page, after each headline chart, and names them.

## The colour rule

**Colour marks what kind of product a row is (satellite retrieval, reanalysis, or weather model),
not which product it is.** The product's name is always on the axis, so identity never depends on
colour. The obvious rule, one OCF hue per product, fails the `dataviz` skill's colour-vision
validator (`validate_palette.js`, run against the OCF background `#FFFBF5`):

| Palette | Result |
|---|---|
| Six OCF hues, one per solar product (blue, orange-red, purple, dark green, sky blue, mustard) | **Fails.** Purple and blue sit 2.0 ΔE apart under deuteranopia, below the floor of 8. Mustard and orange-red sit 15.0 apart for normal vision, at the floor. Sky blue and mustard fall below 3:1 contrast. |
| Blue (satellite), orange-red (reanalysis), dark green (weather model) | **Passes every check.** The worst colour-vision pair is 12.6 ΔE apart. |

The three family colours are `plotting.ocf_theme.BLUE`, `ORANGE_RED`, and `DARK_GREEN`. Where a
chart must also tell two conditions apart, such as two halves of the year or two leads, it uses
facets or point shape rather than a new hue, so each hue keeps one meaning on both pages.

**Out of scope, but reported:** the beam/diffuse page's `make_figures.py` draws UKV in purple beside
CAMS in blue, and its docstring says the four-colour set was never measured. The validator now
measures that pair at 2.0 ΔE under deuteranopia. That figure belongs in its own issue.

## The charts

**Every chart is a dot at the estimate and a line spanning its 95% interval,** unless the entry below
says otherwise. Each chart has a vertical rule at zero, and the x axis is in percentage points of
capacity, which is the unit of every contrast on both pages. The tables stay on the pages, as the
chart's table view.

### Solar page (`weather-products-for-the-past.md`)

1. **Headline, directly under the opening paragraphs, before "The six products".** One row per
   product, ordered by mean absolute error, best at the top. The x axis gives each product's mean
   absolute error minus ERA5's, with the paired interval; ERA5 sits at zero, with no interval. Each
   row is labelled with the product's own mean absolute error ("CAMS 5.05%"), because the absolute
   level matters as much as the contrast. Pooled over all rows (scope `all`), with
   the primary hyperparameters and each product's `_global` arm, exactly as the page's table.
2. **"CAMS describes past sunshine best": CAMS − ICON-D2, broken down.** Three facets, one each for
   generator (A to F), season, and calendar year (2023 to 2026). The per-breakdown intervals are new;
   the page quotes only the range of point estimates. Supports "holds at every generator, in every
   season, and in each calendar year".
3. **"ICON-D2 is the best weather model as served": ICON-D2 − ICON-EU by hour of day, 07 to 19 UTC.**
   Point shape marks the served lead (1, 2, or 3 hours), and the x axis is the hour, so this is the
   one chart with a vertical estimate axis. The per-hour intervals are new. Supports "the advantage
   fades within hours of each run", and shows the 12 and 13 UTC pair the page discusses.
4. **"ICON-EU against ICON global and UKV": each UKV construction against ICON-EU.** Rows, top to
   bottom: UKV as served, UKV rebuilt from two snapshots, rebuilt UKV with neighbouring hours, and
   ICON global. The reference is ICON-EU as served. Supports "the whole gap disappears once UKV's hour
   is rebuilt". The context arms are compared context-against-context, as on the page.
5. **"Implied capacity from month to month": each product's implied capacity by calendar month, as a
   percentage above or below its own annual mean.** Six small multiples, one line per panel, in the
   family colour, sharing one y axis. The quantity is pooled over the six generators: no
   generator's output appears. Supports "CAMS swings the most with the seasons", and shows the
   December dip. Needs `_implied_capacity` split, so that one function returns the per-(site, month)
   frame and the report and the chart both read it.

### Wind page (`weather-products-for-past-wind.md`)

1. **Headline, directly under the opening paragraphs, before "The five products".** The same form as
   the solar headline: five products, each measured against ERA5, each labelled with its own mean
   absolute error.
2. **"UKV and ICON-D2 describe past wind best": each product against ERA5, split by half-year.** Two
   facets, April to September and October to March. Supports "mostly a summer result".
3. **Same section: ICON-D2 − UKV by hours since ICON-D2's run started (0, 1, 2).** Supports "at
   equal lead ICON-D2 beats UKV, and its advantage fades within hours of each run".
4. **"ICON global is the weakest product": the steps in ICON global's served wind.** A line of the
   monthly mean ratio of ICON global's 80 m speed to ICON-EU's, one line per generator (W1 to W3),
   with a vertical rule at each of the two dates in `STEP_DATES`. The generator with the steps is
   drawn in the weather-model green and the other two in the theme's grey (`ENSEMBLE_LINE`). The
   chart shows wind speeds only, never output. This is the one chart that reads the downloaded wind
   files (`wind_icon_global.parquet`, `wind_icon_eu.parquet`) rather than the losses.
5. **Same section: ICON global − ICON-EU per generator, with and without the step indicator.** Two
   points per generator, told apart by shape. Supports "cuts ICON global's deficit at that generator
   from 1.39 to 0.44 points".

**Considered and left out:** a predicted-against-measured week of power. Such a chart would have to
show one generator's output, even normalised, and the beam/diffuse page already shows that the
pipeline produces a sane forecast. The own-beam effect (solar) and the leave-one-site-out arm are
left as prose too: each is a small set of numbers that a table or a sentence already carries.

## What changes, file by file

- **`packages/studies/src/studies/charts.py` (new).**
    - `ProductFamily = Literal["satellite", "reanalysis", "weather model"]`.
    - `FAMILY_COLOURS: Final[dict[ProductFamily, str]]`: the three OCF hues above.
    - `interval_chart(*, intervals, category, x_title, colour, shape=None, facet=None)`: builds the
      dot-and-interval layer, the zero rule, and the fixed colour domain from a Polars frame with
      `difference`, `lower_95`, and `upper_95` columns. It returns an Altair chart.
    - `write_svg(*, chart, path)`: writes the chart with `vl_convert`.
- **`packages/studies/pyproject.toml`:** add `altair`, `plotting`, and `vl-convert-python`. The test
  renders an SVG, so the chart scripts no longer need `--with vl-convert-python`.
- **`packages/studies/tests/test_charts.py` (new).** Tests below.
- **`studies/beam_diffuse_split/weather_products.py`:** make `_scope` and `_mae` public as `scope`
  and `mae`, and split `_implied_capacity` into `log_capacity_by_month(*, frame)`, which returns the
  per-product frame, and the report function that reads it. The report must come out byte-identical;
  checked by regenerating `report.md` from the saved frame and diffing it (see Verification).
- **`studies/beam_diffuse_split/wind_products.py`:** make `_scoped` public as `scoped`.
- **`studies/beam_diffuse_split/weather_product_charts.py` (new):** reads `losses.parquet` (and, for
  chart 5, rebuilds the common rows with `_joined` and `_common_rows`, which fits no model), computes
  every interval with `studies.bootstrap.bootstrap_difference`, writes each SVG to
  `docs/studies/assets/`, and writes every plotted interval to `chart_intervals.csv` in the results
  directory.
- **`studies/beam_diffuse_split/wind_product_charts.py` (new):** the same, for the wind page.
- **The two pages:** one image and one caption sentence per chart. Each caption says what one dot
  and one line mean, and names the scope.
- **`studies/beam_diffuse_split/README.md`:** the two new scripts, and the `svgo` step.

**The 10 SVGs go through `npx svgo@4 --multipass --precision=1 --final-newline`,** as `CLAUDE.md`
requires, before they are committed.

## Design-philosophy check

This is R&D code, so it fails fast: a chart script raises if an arm, a scope, or a site label it
expects is missing, rather than drawing a chart with a row silently absent. No production path,
asset, or asset check is touched. Anonymisation (`CLAUDE.md`, "Never publish a metered generator's
time series…") holds because no chart plots output. Generators appear only as A to F and W1 to W3,
and the one per-generator time series (wind chart 4) plots wind-speed ratios.

## Tests

`test_charts.py`. Each assertion fails on `main` today, because the module does not exist:

- **The plotted numbers are the input numbers.** `interval_chart` given three rows yields a spec
  whose inline data holds exactly those `difference`, `lower_95`, and `upper_95` values, in the
  row order given.
- **Colour follows the family, not the rank.** The colour scale's domain and range are
  `FAMILY_COLOURS` in full, even when the input holds only one family. Without that, a chart
  holding only weather models would repaint green as blue.
- **The zero rule is drawn at zero.**
- **A frame missing `lower_95` raises.**
- **`write_svg` writes a file that parses as SVG** and carries the OCF background colour.

The chart scripts, like the study scripts, are not unit-tested. Their check is that every plotted
interval matches the report, which the verification step below compares.

## Verification

- The green-before-push set: `ruff check`, `ruff format`, `uv run --all-packages ty check`,
  `uv run pytest`, pydoclint, `pymarkdown`, and `scripts/lint/check_docs_links.py`, as in `ci.yml`.
- `uv run mkdocs build --strict`, then read both rendered pages under `site/`, and look at every
  chart as a PNG render (`dataviz` step 7): no label collisions, no clipped whiskers.
- **Every number a chart shares with a page matches it.** Diff `chart_intervals.csv` against
  `report.md` for each shared contrast: same scope, same seeds, same bootstrap seed, so the figures
  must agree to the last printed digit.
- **The report refactor changes nothing.** Rerun the solar report assembly from the saved losses and
  common rows, and confirm `report.md` is byte-identical. No fitting is needed.
- `grep` every committed SVG for any generator identifier from the power table, and for any
  unanonymised site key.

## Risks and open questions

- **Which product should the headline measure against?** Recommendation: ERA5 on both pages. ERA5 is
  the default reanalysis, it is on both pages, and it is the reference of two of the wind page's four
  named contrasts. The alternative is the best product on each page (CAMS, ICON-D2), which makes the
  leader sit at zero and hides its own interval.
- **Should the headline show each product's own error with an interval, rather than its difference
  from ERA5?** Recommendation: no. The six products are scored on the same hours, so their levels
  share most of their month-to-month variation. Intervals on the levels would overlap heavily
  even where the paired difference clearly excludes zero, and a reader would read the overlap as "no
  difference". The level goes in the row label instead, without an interval.
- **Static SVG or interactive chart?** The `dataviz` skill defaults to a hover layer. Both study pages
  and the beam/diffuse page use static SVG through MkDocs. Recommendation: static SVG, with the page
  tables serving as the table view.
- **The new per-breakdown intervals (solar 2 and 3) are exploratory figures with no named contrast
  behind them.** Each caption says so.
