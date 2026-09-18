# Notebooks

Marimo notebooks for exploratory data analysis and ad-hoc experimentation.

## Why this package exists

**A notebook here answers one question about the data, and is allowed to be rough.** What the NWP
archive holds for one H3 cell, where a weather variable goes missing, and whether a baseline export
looks the way its author expected: each question is worth a chart, and worth no more engineering
than a chart. Nothing in `src/`, and no other package under `packages/`, imports this package, so a
notebook that stops working stops working alone.

**Roughness is acceptable here because a notebook is never the route into production.** [Design
principle 3 — one execution path from research to
production](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#3-one-execution-path-from-research-to-production)
holds that an idea becomes an experiment — a candidate that can enter the leaderboard, and therefore
be promoted to production — only once the idea is implemented in the pipeline's own code, behind the
same data contracts and tests as everything else. Exploring in a notebook first is expected. Lifting
a notebook's code into the pipeline afterwards is not, because the pipeline gets its own
implementation, reviewed and tested.

**`dashboard` owns the marimo apps that somebody other than the author opens.** The apps at
`packages/dashboard/` are meant to be re-opened by whoever is watching the forecast, so each app
keeps its unit-testable logic in an importable package under `packages/dashboard/src/`. That package
carries a test suite of its own, and both apps are expected to keep working. A notebook here keeps
its logic inline, hard-codes the run and the H3 cell the author happened to be looking at, and may
stop working the day the data moves.

## Rules every notebook here has to obey

**Marimo reverses two ordinary Python habits, and both failures are silent.** A leading underscore
makes a name cell-local rather than private, so a helper that more than one cell calls has to carry
a public name. And every import belongs in the `with app.setup:` block, because an import marimo
threads through a cell signature arrives as a function parameter, which ruff treats as always
defined — a genuinely missing import then fails only at runtime. **Never run `ruff check --fix` over
a notebook:** an autofix that needs a new import writes that import into the file's top-level import
block, where no cell can see the import, and reports success. The full set of authoring rules is the
`marimo-notebooks` skill.

**`scripts/lint/check_marimo_notebooks.py` is what catches a notebook broken that way.** The script
reads the names each cell binds and the names each cell references, then reports any referenced name
that no cell binds. The script runs as a pre-commit hook over changed notebooks and, through
`tests/test_marimo_notebooks.py`, over every notebook in this directory on every `uv run pytest`.
What the check covers, and the three properties worth knowing before trusting the check, are on the
[Testing
page](https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#marimo-notebooks-bind-every-name-their-cells-reference).

**Every `.py` file sitting directly in this directory has to be a marimo notebook**, because the
ruff pre-commit hooks and the `[tool.ruff.lint.per-file-ignores]` block in `pyproject.toml` both
assume so. A new notebook needs its own entry in that block: ruff matches `*` across `/`, so a
directory-wide pattern would also silence real library and test code, and the entries therefore name
each notebook individually.

## Running a notebook

```bash
uv run marimo edit packages/notebooks/plot_nwp_map.py
```

Every notebook here except `plot_gb_map.py` reads the project's real data through
`contracts.settings.Settings`, so the root `.env` has to be set up first — see the [Getting started
guide](https://openclimatefix.github.io/nged-substation-forecast/getting-started/). These notebooks
carry no local/S3 toggle of the kind the `dashboard` apps have: each notebook reads whatever the
root `.env` points at.

## Contents

- `plot_gb_map.py` — draws the H3 cells the NWP pipeline aggregates weather onto. Loads the Great
  Britain boundary bundled in `geo`, computes the resolution-5 cells covering that boundary against
  a 0.25° regular latitude/longitude grid, and maps the hexagons. The loader buffers the boundary
  outwards by 0.25° first, so coastal substations and nearby islands fall inside the cells drawn.
  Reads no Delta table, so `plot_gb_map.py` is the one notebook here that runs without an `.env`.
- `plot_nwp_map.py` — two views of one ECMWF ENS run from the `nwp` Delta table: every ensemble
  member's chosen variable over time at one H3 cell, and that same variable across every H3 cell at
  one valid time, shaded by a continuous colour map.
- `plot_missing_NWP_data.py` — shows where the NWP archive is missing values. For one run and one H3
  cell, the notebook draws a grey line per ensemble member and a red tick wherever a variable is
  null or NaN. The notebook is also the worked example the Testing page points at for testing a
  notebook: the chart builder is an `@app.function`, and the `test_*` function beside the chart
  builder is collected by a plain `uv run pytest`, because `python_files` in `pyproject.toml` names
  this file.
- `view_baseline_export.py` — inspects the parquets
  `scripts/forecasting/export_baseline_forecasts.py` writes: observed power against the forecast,
  the observed-minus-ensemble-mean residual that switching-event detection consumes, and the raw
  ensemble members over a chosen window. The notebook's module docstring says what each chart shows.
