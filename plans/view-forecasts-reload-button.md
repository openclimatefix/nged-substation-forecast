# Plan: a "Reload" button in `view_forecasts.py` (#520)

**The problem.** `view_forecasts.py` reads the `power_forecasts`, `power_time_series` and `nwp`
Delta tables once, when each cell first runs. The live forecast asset appends a new run every 30
minutes, but a marimo session opened before that run landed keeps showing the old picture: the
**Forecast date** and **Forecast run** pickers are built from a list of init times captured at
first load, so a new run is invisible until the whole notebook is restarted. There is no control
that re-reads the tables.

**The planned solution.** Add one `mo.ui.refresh` element, `reload`, defined and displayed beside
the existing **Data source** radio, and reference it from the forecast-partition cell — the single
root of the notebook's forecast/power/NWP dependency chain. Marimo re-runs every transitive
descendant of a cell that re-runs, so one reference re-executes all four Delta reads and rebuilds
the fold, experiment, date and run pickers from the freshest data. The time-series picker is
deliberately left off that chain, because a marimo UI element resets to its default whenever its
defining cell re-runs, and resetting the series you are watching would make the button hostile to
use.

## Verdict and departures

**Worth implementing, as described.** The issue body is one sentence ("To get the UI to notice new
data") and there are no comments, so there is nothing stale to overrule. Everything below is
design this plan is adding, not a departure from the issue.

The change is small, and it is what makes the app usable for the job
`docs/live_service/operations.md` gives it — watching live forecasts arrive.

## The mechanism, and why it reaches what it needs to

Marimo's reactivity works on names: a cell re-runs when any name it references is redefined, and
all of that cell's transitive descendants re-run with it. The notebook's dependency graph (read
out of marimo's own compiled cells, so this is the real graph, not a reading of the source) is:

```text
settings ─┬─> metadata_df, series_picker              (line 60, pl.read_parquet)
          ├─> forecast_partitions                      (line 90, DeltaTable(...).partitions())
          └─> available_init_times                     (line 141, pl.scan_delta)

forecast_partitions ──> fold_picker ──> experiment_picker ──> available_init_times
                                                          └─> forecasts, actuals   (line 265, ×2 pl.scan_delta)
available_init_times ──> date_picker ──> run_picker ──────┘
forecasts ────────────> nwp, nwp_analysis               (line 358, ×2 pl.scan_delta)
```

Referencing `reload` from the cell at line 90 therefore re-runs, in order: the partition listing,
`fold_picker`, `experiment_picker`, `available_init_times`, `date_picker`, `run_picker`, the
controls cell, the forecast/actuals load, the power chart, the NWP load and the NWP chart. That is
**every Delta read in the notebook**, and every picker whose options come from the data. The one
data read it does not reach is the metadata parquet at line 60 — which is the point (see below).

Nothing caches underneath: each re-run constructs a fresh `DeltaTable`/`pl.scan_delta`, which reads
the current Delta log, so a re-run genuinely sees new commits.

## What changes, file by file

### `packages/dashboard/view_forecasts.py`

- **Cell at line 40** (defines `source`) — also define
  `reload = mo.ui.refresh(label="Reload data")` and display both:
  `mo.hstack([source, reload], justify="start", gap=2)`. Update the generated `return (source,)` to
  `return reload, source`.

  This cell is the right home for two reasons beyond "it is about where the data comes from".
  First, it references only `mo`, so it never re-runs and `reload` is never reset by its own
  mechanism. Second, it never `mo.stop`s — so the button is still on screen in the state where it
  matters most, when the forecast table was missing or unreadable at load time and the cell at line
  90 has stopped the whole notebook with a callout. A button placed among the pickers would
  disappear in exactly that case.

  `mo.ui.refresh` with neither `options` nor `default_interval` renders as a plain button with no
  auto-refresh dropdown. Its `value` is `""` before the first click and `"<interval> (<count>)"`
  after each one, with `<count>` incrementing — so every click is a genuine value change and the
  descendants re-run every time, not just the first.

- **Cell at line 90** (defines `forecast_partitions`) — add a bare `reload` statement at the top of
  the cell, under a comment naming what it does and what it deliberately misses, and update the
  generated signature from `def _(settings)` to `def _(reload, settings)`. A bare-name expression
  is the idiom for taking a marimo dependency without using the value; `B018` is already declined
  for this file in `pyproject.toml`'s `per-file-ignores`, so it is lint-clean.

- Nothing else in the notebook changes.

### What happens to the pickers, and why that is the wanted behaviour

Marimo gives every UI element a fresh random token on construction, "so that re-running a cell that
creates a UI element will trigger a re-render and reset it to its initial value"
(`marimo/_plugins/ui/_core/ui_element.py`). So each reset below is unavoidable given the mechanism,
and each one lands where it should:

| Element | Defining cell re-runs on reload? | Result |
|---|---|---|
| `series_picker` | no | selection kept |
| `fold_picker` | yes | back to `live` |
| `experiment_picker` | yes | back to the latest experiment |
| `date_picker` | yes | back to the newest date with forecasts |
| `run_picker` | yes | back to the newest run on that date |
| display checkboxes, `nwp_variable_picker` | no | kept |

Jumping the date and run pickers to the newest run *is* the feature: a reload that left them
pointing at the previous run would look like it had done nothing. Jumping the series picker back to
id 24 would be pure loss, which is why the metadata cell stays off the chain. The cost of that
choice is that a time series newly added to the metadata parquet does not appear in the dropdown
until the notebook is re-run in full (or the **Data source** radio is toggled, which re-runs
`settings` and everything under it). New substations arrive when NGED adds one to the feed, not
every half hour, so this is the right side of the trade.

There is no half-updated state to worry about: the pickers are rebuilt from the new data rather
than re-validated against it, so a fold, experiment, date or run that vanished between loads cannot
leave a picker holding a value that is no longer available.

## Design-philosophy check

The dashboard is a read-only inspection app, not the production forecast path, so
`docs/design-philosophy/inherent-stability.md`'s degrade-never-raise rules apply only in spirit —
and the change adds no path that can raise. The two failure modes a reload can hit are already
handled: an unreadable forecast table is caught by the existing `except Exception` at line 99 and
reported in a callout, and an empty result is handled by the existing `mo.stop` callouts. Reload
re-enters those same paths, and because the button lives outside every `mo.stop`, a user who
reloads into a failure can reload back out of it once the data lands. No asset checks are involved.

No principle in `docs/design-philosophy/design-principles.md` is traded away; the change adds one
UI element and one dependency edge.

## Tests

New file `packages/dashboard/tests/test_view_forecasts.py`, holding a small helper that parses the
notebook into marimo cells and builds the name-dependency graph — the same private
`marimo._ast` API `scripts/check_marimo_notebooks.py` documents its use of, since parsing a
notebook without running it is not public API. The helper cannot be imported from that script,
because `scripts/` is not an importable package (which is why `tests/test_marimo_notebooks.py`
drives it as a subprocess).

1. **`test_reload_reaches_every_delta_read`** — every cell whose code calls `pl.scan_delta(` or
   `DeltaTable(` is a transitive descendant of the cell defining `reload`. **Fails on `main`
   today**: no cell defines `reload`, so there is no ancestor to find. This is the test that
   matters — a button wired to the chart cell but not the Delta reads would look like it worked in
   a demo and would never show new data.
2. **`test_reload_does_not_reset_the_time_series_picker`** — the cell defining `series_picker` is
   *not* a descendant of the cell defining `reload`. **Fails on `main` today** for the same reason,
   and afterwards pins the deliberate boundary: wiring `reload` into the metadata cell (the obvious
   "make reload reload everything" edit) would break it.

`tests/test_marimo_notebooks.py` already runs the name-binding checker over every notebook, so a
signature or import mistake in the edited notebook fails there with no new test needed.

## Docs to update

- **`packages/dashboard/README.md`** — one sentence in the `view_forecasts.py` bullet, or a short
  paragraph beside the data-source section: what **Reload** re-reads, that the run selectors jump
  to the newest run, and that the time-series choice is kept.
- **`docs/live_service/operations.md`**, "Inspecting a live forecast" (around line 393) — one
  sentence telling the operator to press **Reload** to pick up runs written since the app was
  opened, rather than restarting marimo.
- **`docs/ml_experimentation/dagster-workflow.md`** describes inspecting *backtest* forecasts,
  where the data does not change under the user; no update needed.

This issue does not complete a roadmap item, so there is no ship-time triage.

## Verification commands

```bash
uv run ruff check .                       # never --fix over a notebook
uv run ruff format .
uv run --all-packages ty check
uv run pytest
uv run python scripts/check_marimo_notebooks.py packages/dashboard/view_forecasts.py
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Plus a manual pass, because no static test can prove the button actually re-reads: open
`uv run marimo edit packages/dashboard/view_forecasts.py`, note the newest forecast run offered,
append a newer run to the local `power_forecasts` table (or point the app at `s3` while the live
job runs), click **Reload**, and confirm the run picker now offers the new run while the
time-series selection is unchanged.

## Risks and open questions

1. **Should reload also re-read the metadata parquet, at the cost of resetting the time-series
   picker?** *Recommendation: no*, as planned. Keeping the selection is worth more than picking up
   a new substation without a restart. If both are wanted, it costs a `mo.state` pair plus an
   `on_change` on the dropdown to restore the previous selection after the cell re-runs — real
   machinery for a rare event, and better as its own issue if it ever bites.
2. **Automatic periodic refresh as well as the button?** *Recommendation: no, out of scope.*
   `mo.ui.refresh(options=[...], default_interval=...)` would give it for one extra argument, so
   this is cheap to add later. It is declined now because of the reset behaviour above: a timer
   firing while someone is studying a run would drag the date and run pickers to the newest run
   underneath them, and each tick re-reads four Delta tables — over S3, that is real traffic for a
   window nobody is looking at.
3. **Button placement.** Beside the **Data source** radio at the top, for the two reasons in the
   file-by-file section (stable cell, and it survives every `mo.stop`). The alternative — in the
   controls row with the pickers it refreshes — is closer to what it affects but disappears exactly
   when the forecast table is unreadable, which is when a retry button is most useful.
4. **The tests read private marimo API** (`marimo._ast`), so a marimo upgrade could break them.
   That risk is already taken and documented by `scripts/check_marimo_notebooks.py`; the failure
   mode is a loud test error on a lock-bump PR, not a silent pass.
