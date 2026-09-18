# Dashboard

Marimo web apps for visualising power forecasts, telemetry, and evaluation metrics. The apps live at
the package root; their shared, unit-testable logic (the data-source toggle and the forecast chart
builder) lives in the importable `dashboard` package under `src/`.

## Why the logic sits under `src/` rather than in the notebooks

**A marimo cell cannot be unit tested, so everything worth a unit test is pushed out of the notebook
and into `src/dashboard/`.** A cell is a `def _(...)` function whose parameters are the names other
cells export, and marimo rebuilds the notebook from its cells rather than running the module, so no
test can call a cell the way a caller calls a function. What is left in a notebook is the
arrangement — which controls exist, which Delta queries run, and how the pieces stack on the page.
Two checks cover that arrangement, and both parse the notebook rather than running it:
`scripts/lint/check_marimo_notebooks.py` proves each cell's names are bound and nothing more, and
`packages/dashboard/tests/test_view_forecasts.py` proves that every cell holding a Delta read
descends from the cell referencing the Reload button. Of the two modules under `src/`,
`forecast_chart` is ordinary library code with its tests in `packages/dashboard/tests/`;
`data_source` has no tests today.

**This package owns the arrangement and nothing else.** `contracts` owns what each table means and
where the table lives, `weather_utils` owns the analysis-proxy query the dashboard shares with the
feature pipeline, and `plotting` owns the OCF Altair theme and its colour constants. A change to
what a chart *means* therefore usually belongs in one of those packages; a change to what the reader
*sees* belongs here.

## Rules the two apps have to obey

**The marimo authoring rules apply to the two apps at the package root, and each rule reverses an
ordinary Python habit.** A leading underscore makes a name cell-local rather than private, every
import belongs in the `with app.setup:` block, and `ruff check --fix` must never be run over a
notebook, because an autofix can write a new import outside `app.setup`, where no cell can see the
import, and still report success. The `marimo-notebooks` skill holds the full set, and the [Testing
page](https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#marimo-notebooks-bind-every-name-their-cells-reference)
explains what the pre-commit check does and does not catch. Both apps sit directly in this
directory, so both need an entry in the root `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`
block.

## The apps

- **`view_forecasts.py`** — inspect a single forecast run: pick a time series, a fold (`live` or a
  CV fold), and a forecast init time, then see every forecast ensemble member (thin grey lines)
  against the observed power (thick blue line), from 24 hours before the init time to 14 days after
  it. The x-axis is labelled at Europe/London midnight with the day of week and date. Optional
  coloured lines overlay observed power shifted forward by 7 and by 14 days — the raw material of
  the models' power-lag features. A second panel below the power chart, on the same time axis, plots
  the NWP ensemble that fed the forecast at the H3 cell containing the series, for whichever weather
  variable is picked. A stitched proxy-analysis line on that second panel stands in for the weather
  that actually happened. **Reload data** re-reads the forecast, power, and NWP tables.
- **`map_and_timeseries.py`** — a map of every time series in the trial area; click a dot to see its
  observed power. The power query is capped to recent observations, because the chart inlines its
  rows and Altair refuses more than 5,000 rows by default.

Run an app with:

```bash
uv run marimo edit packages/dashboard/view_forecasts.py
```

## Switching between local and S3 data sources

Each app has a **Data source** toggle (`local` / `s3`) that switches which data tables the app reads
without restarting marimo, so you can compare a fully local pipeline against production data in one
session.

- **local** reads only the root `.env` — the same local-pipeline config the rest of the app uses
  (set up in the [Getting started
  guide](https://openclimatefix.github.io/nged-substation-forecast/getting-started/)).
- **s3** layers a git-ignored `packages/dashboard/.env.s3` on top of the root `.env`, overriding the
  data-path roots (`DATA_PATH_INTERNAL`, `DATA_PATH_DELIVERY`) and the `DATA_STORE_*` credentials to
  point at the real S3 buckets.

To enable S3 mode, copy the committed example and fill in the real values:

```bash
cp packages/dashboard/.env.s3.example packages/dashboard/.env.s3
```

Only the data tables follow the toggle. The local artifact path (the production model) is never
overridden by `.env.s3`, so the production model stays laptop-local in both modes. If `.env.s3` is
missing, the `s3` selection falls back to local paths and the UI flags the fallback. The read-only
IAM user whose credentials belong in `.env.s3` is created in [Step 2 of the AWS
guide](https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/#step-2-grant-data-access-with-iam).

## Contents of `src/dashboard/`

- `data_source` — `settings_for_source()` builds a `contracts.settings.Settings` from the selected
  source, and `source_status_message()` builds the line shown under the radio, including the warning
  that `s3` was selected with no `.env.s3` to read credentials from.
- `forecast_chart` — the two Altair chart builders `view_forecasts.py` calls, plus the constants
  fixing the plotted window, the display time zone, the NWP variables offered, and the power lags
  offered. Every constant's docstring says what that constant fixes, and most say why the value is
  what it is; the module docstring says how the two charts fit together.

## Invariants worth knowing before editing a chart

**Every plotted time is naive Europe/London wall time.** Demand shape follows the local clock, so
midnight ticks and day-of-week labels have to be local rather than UTC. Stripping the zone then
makes a chart render identically in any viewer's browser, because Vega would otherwise re-localise a
tz-aware timestamp to whatever zone the viewer sits in.

**The power chart and the NWP panel are separate Altair specs whose x-axes align only by
construction.** Both charts pin the same x encoding, both pin the y-axis region to the same pixel
width, and both put their legend above the plot rather than beside the plot, so neither chart's plot
area ends up narrower than the other's. Changing any one of those three settings on one chart alone
visibly misaligns the pair.

**A chart's data is rounded and served out of line, because the ensemble is large.** One forecast
run for one series is 51 members × 14 days × 48 half-hours ≈ 34,000 rows, which is past both
Altair's 5,000-row default guard and marimo's maximum output size. `build_view_forecast_chart` calls
`alt.data_transformers.disable_max_rows()` to lift the first guard. For the second, the builders
round values to 3 decimal places as `Float64` before serialising, and the apps hand the result to
`mo.ui.altair_chart`, which serves the rows as a virtual file instead of inlining the rows in the
cell output.
