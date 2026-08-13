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

`reload` must be *defined* in a different cell from the one that references it. Marimo subtracts a
name's defining cells from the set it re-runs when that name changes
(`marimo/_runtime/runtime.py`, "never rerun the cell that created the name"), so collapsing the two
into one cell would leave the button inert while looking correct.

## What changes, file by file

### `packages/dashboard/view_forecasts.py`

- **Cell at line 40** (defines `source`) — also define
  `reload = mo.ui.refresh(label="Reload data")` and display both:
  `mo.hstack([source, reload], justify="start", gap=2)`. Update the generated `return (source,)` to
  `return reload, source`.

  This cell is the right home because it references only `mo`, so it never re-runs and `reload` is
  never reset by its own mechanism, and because it never `mo.stop`s — so the button is still on
  screen when the forecast table was missing or unreadable at first load and the cell at line 90
  has stopped the whole notebook with a callout. That is the one state where a reload is the only
  way forward without restarting marimo, and a button placed among the pickers would have
  disappeared with them.

  `mo.ui.refresh` with neither `options` nor `default_interval` renders as a plain button with no
  auto-refresh dropdown. Its `value` is `""` before the first click and `"<interval> (<count>)"`
  after each one, with `<count>` incrementing — so every click is a genuine value change and the
  descendants re-run every time, not just the first.

- **Cell at line 90** (defines `forecast_partitions`) — add a bare `reload` statement at the top of
  the cell, under a comment naming what it does, what it deliberately misses, and that the
  statement is load-bearing rather than dead code. Update the generated signature from
  `def _(settings)` to `def _(reload, settings)`. A bare-name expression is the idiom for taking a
  marimo dependency without using the value; `B018` is already declined for this file in
  `pyproject.toml`'s `per-file-ignores`, so it is lint-clean.

- Nothing else in the notebook changes.

### What happens to the pickers, and why that is the wanted behaviour

Marimo gives every UI element a fresh random token on construction, "so that re-running a cell that
creates a UI element will trigger a re-render and reset it to its initial value"
(`marimo/_plugins/ui/_core/ui_element.py`). So each reset below is unavoidable given the mechanism:

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

The fold and experiment resets are the real cost of anchoring this high, and open question 3 below
records the cheaper-looking alternative and why it is not taken.

## Design-philosophy check

The dashboard is a read-only inspection app, not the production forecast path. The change adds no
path that can raise: an unreadable forecast table is caught by the existing `except Exception` at
line 99 and reported in a callout, an empty result by the existing `mo.stop` callouts, and reload
re-enters those same paths. No asset checks are involved, and no principle in
`docs/design-philosophy/design-principles.md` is traded away.

## Tests

One new file, `packages/dashboard/tests/test_view_forecasts.py`, holding a
`test_reload_reaches_every_delta_read`: every cell whose code calls `pl.scan_delta(` or
`DeltaTable(` is a transitive descendant of the cell defining `reload`. **Fails on `main` today**,
because no cell defines `reload`.

It needs a dozen lines to turn the notebook into cells with their `defs`, `refs` and `code`
(`marimo._ast`'s `get_notebook_status` → `load_notebook_ir` → `cell_manager.cell_data()`) and a
short breadth-first walk over the resulting graph. It does not need the line numbers or the
compile guards that make `scripts/check_marimo_notebooks.py` longer, because
`tests/test_marimo_notebooks.py` already fails loudly if this notebook stops parsing.

This is the one part of the change worth a test, for a reason particular to the mechanism: the
dependency edge is a bare `reload` statement with no assignment, which reads like dead code and is
the obvious thing for a future tidy-up to delete. Deleting it leaves a button that still renders,
still clicks, and silently stops re-reading anything. A comment says not to; a test notices.

The test's blind spot, worth stating in the file: it matches on the cell's own source, so a Delta
read reached through a helper in `packages/dashboard/src/dashboard/` would not be seen. Nothing in
the notebook reads that way today.

## Docs to update

- **`packages/dashboard/README.md`** — one sentence in the `view_forecasts.py` bullet: **Reload**
  re-reads the forecast, power and NWP tables and jumps the run selectors to the newest run,
  keeping the chosen time series.
- **`docs/live_service/operations.md`**, "Inspecting a live forecast" (around line 393) — one
  sentence telling the operator to press **Reload** to pick up runs written since the app was
  opened, rather than restarting marimo.

`docs/ml_experimentation/dagster-workflow.md` describes inspecting backtest forecasts, where the
data does not change under the user, and needs no change. This issue does not complete a roadmap
item, so there is no ship-time triage.

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
3. **Anchor the reload at the partition listing (line 90) or one cell lower, at
   `available_init_times` (line 141)?** *Recommendation: line 90, as planned, but this is a genuine
   trade and worth a second opinion.* Anchoring at line 141 costs exactly the same lines and keeps
   the **Fold** and **Experiment** selections, because their cells stop being descendants — better
   for someone comparing an older CV experiment, who is currently thrown back to `live` on every
   click. What it gives up is that a fold or experiment that did not exist at first load can never
   appear, so pressing **Reload** after a CV job finishes would show nothing new: the button would
   fail at the one thing the issue asks for, in the R&D workflow where a person is most likely to
   be watching for a result. For the live workflow the two anchors behave identically, since
   **Fold** already defaults to `live`.
4. **The test reads private marimo API** (`marimo._ast`), so a marimo upgrade could break it. That
   risk is already taken and documented by `scripts/check_marimo_notebooks.py`; the failure mode is
   a loud test error on a lock-bump PR, not a silent pass.

## First review (simplicity): findings taken and rejected

**Taken**

- Drop the second proposed test (`test_reload_does_not_reset_the_time_series_picker`): it guards a
  preference, not a defect, and its mutant — wiring `reload` into the metadata cell — is a UX
  regression visible in one click rather than a silent failure.
- State that marimo never re-runs a name's defining cell, so the two-cell shape is forced rather
  than chosen, and record the test's helper-call blind spot.
- Trim the prose: one sentence for the README rather than a paragraph, cut the "no update needed"
  bullet down to a clause, shorten the design-philosophy section.

**Rejected**

- *Drop the test file entirely, and rely on the manual check plus "the defect is visible on the
  first click".* Rejected: the thing most likely to break is not the initial wiring but a later
  deletion of a bare statement that looks like dead code, and nobody clicks the button in that
  edit. Cut to one test rather than two, and to a dozen lines of parsing rather than a copy of
  `scripts/check_marimo_notebooks.py`.
- *Anchor at `available_init_times` (line 141) instead, to preserve the fold and experiment
  selections.* Not taken as the plan's default — it structurally prevents a new fold or experiment
  from ever appearing, which is the R&D half of "notice new data" — but recorded as open question 3
  for Jack, since it is a close call.
- *Delete the design-philosophy check section, since it concludes nothing is traded away.* The
  `plan-issue` skill requires the section; shortened to three lines instead.
