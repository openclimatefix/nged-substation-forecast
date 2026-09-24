# Weather forecasts for power, compared at matched lead times

Study for [issue #810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). The
plan that was reviewed and approved is `plans/nwp-forecast-comparison.md` on the branch this study
was built on (deleted once the study merges, per the workspace's own convention — see the plan's
review log for the two adversarial plan reviews it went through). No fit has run yet: this pass
fixed V1 and V1b, built `build_forecast_inputs.py` and ran it for the Previous Runs and ENS parts,
and built the rest of `nwp_forecast_comparison.py` up to (and not past) the point a fit would run.

## Status

**Pre-fit only; the pipeline is built but has not been run to a fit.**
`verify_previous_runs_leads.py` (V1, V1b, V3) and `check_input_steps.py` (V1c) have been run
against the data already on disk; V1's gate now passes. `build_forecast_inputs.py` has been run
for the Previous Runs and ENS parts (GEFS is gated off — see below). `nwp_forecast_comparison.py`
builds rows, folds and the job list; `--dry-run` has been run against a local checkout of #885's
not-yet-merged `studies.cross_validation.cut_eras` (see "Folds and #885" below) and reaches the
fold-assignment step. No XGBoost fit has run. `nwp_forecast_charts.py` is a skeleton: every chart
function raises `NotImplementedError`.

## Verification results

Every table below was written by a committed script; none of these numbers was hand-typed.

### V1 — the Previous Runs run-selection rule (a gate)

**Rewritten this pass, and now passes at both N = 1 and N = 2.** The previous pass compared each
site's served series against a plain average over the Dynamical.org extract's 9 grid cells, and the
resulting spatial-sampling noise (about 3 km/h) swamped the k-to-k signal at N = 1. This version
instead scores, for each site and each candidate offset `k`, every one of the extract's 9 grid
cells against that site's own series, and keeps the cell with the lowest mean absolute error — no
roster coordinate is read, and every offset gets exactly the same freedom to pick its best-fitting
cell.

**Wind speed at 100 m, mean of each site's best-cell mean absolute error:** 0.1018 km/h at k = 0 for
N = 1 (rising monotonically to 3.57 by k = 10), and 0.0997 km/h at k = 0 for N = 2 (rising to a
local peak of 2.80 by k = 7-12). **The gate passes at both N = 1 and N = 2.**

Radiation, checked only at hours divisible by 6 (the only hours where selecting the run by the
hour's label and by the hour's start disagree), also supports the label convention with the same
per-site best-cell scoring: mean absolute difference 4.96 W/m² (N = 1) and 5.62 W/m² (N = 2) by
label, against 21.94 and 16.99 by start.

Only an index rank among the extract's 9 cells is ever printed (never a coordinate); see
`v1_wind.md` for the chosen ranks per k.

Full tables: `v1_wind.md`, `v1_radiation.md` in the output directory.

### V1b — each product's run cycle

**Rewritten this pass with the reviewer's statistic**: per UTC hour, the mean absolute second
difference of the `previous_dayN` series, divided by its own mean over all 24 hours. A product's
cycle is read as the `n` in `{3, 6}` whose switch hours (`h mod n == 0`) have the highest mean ratio
relative to the other hours, reported as "no clear signature" below a ratio of 1.3.

**The result differs from the plan's stated expectation, and is reported as measured rather than
tuned to match it.** 6-hourly: ICON-EU (day 1 and day 2), ICON global (day 1 and day 2), ARPEGE
Europe (day 1 and day 2), and GFS day 2 only (GFS day 1 scores 1.26, below the 1.3 threshold — "no
clear signature"). **No clear signature** (score below 1.3): UKV, ICON-D2, DMI HARMONIE-AROME, IFS
0.25°, GFS day 1, AROME France and KNMI HARMONIE-AROME. The plan's review had read UKV, ICON-D2 and
DMI HARMONIE-AROME as 3-hourly and GFS day 1 as 6-hourly; this pass's statistic does not confirm
either. IFS 0.25° and KNMI's "no clear signature" does match the plan. Full 24-hour table per
product, field and day: `v1b_run_switches.md`.

### V1c — steps in the planned inputs

Unchanged this pass. `check_input_steps.py` reads UKV's and IFS 0.25°'s day-1 monthly mean
radiation and 100 m wind speed, per anonymised site, as a ratio to ICON-EU's day-1 monthly mean at
the same site and month, and flags a month where that ratio moves by 15% or more from the month
before. GEFS and ENS are not covered by this screen (see the script's docstring). See
`v1c_steps.md` for the flagged months. The November-to-December 2024 jump recurs at nearly every
site with almost the same ratio, which looks like a seasonal solar-geometry artefact of the
UKV/ICON-EU ratio near the winter solstice rather than a product version step, but this pass did
not chase it further.

### V3 — timestamp and height conventions

Unchanged this pass. `v3_conventions.md` reads each product's timestamp convention (instantaneous
snapshot or hour-ending mean) from the "best fit" line already recorded in that product's own
`previous_runs/lineage.json` at fetch time, and states the height convention already established
elsewhere in this project's study work (ICON's served "100 m" wind is its native 120 m wind
rescaled by about 0.98). It does not re-run a wind power-hour offset scan.

### V2 — GEFS radiation window conversion

Not run: the GEFS month cache does not yet cover 2024-11 to the month before today (27 months
cached, short of the span needed), so `build_forecast_inputs.py`'s GEFS gate does not open — see
"GEFS" below. `gefs_step_means` (`packages/studies/src/studies/resample.py`) is written and
tested; V2 itself is deferred to a session run after the download completes.

## What is built

- `packages/studies/src/studies/bootstrap.py`: `bracket_verdict(lower_side, upper_side)`.
- `packages/studies/src/studies/resample.py`: `gefs_step_means`.
- Tests for both, in `packages/studies/tests/`.
- `verify_previous_runs_leads.py` (V1, V1b, V3) and `check_input_steps.py` (V1c), rewritten V1 and
  V1b this pass.
- `studies/beam_diffuse_split/ens_forecast_horizons.py`: `_base_frame` renamed to `base_frame`
  (public), no behaviour change, so `build_forecast_inputs.py` can reuse it, its `build_inputs` and
  its `main_frame` for the shared rows and the ENS upsampling, rather than re-deriving either.
- `build_forecast_inputs.py`: builds one arm-input parquet per technology — the shared power rows,
  every Previous Runs product's weather columns at its planned day offsets, and ECMWF ENS's mean
  and control-member columns at days 0-3 (through `ens_forecast_horizons.build_inputs` and
  `.main_frame`, at the `linear` upsampling technique — this pass does not re-run `choose_method`,
  which needs a fit). Run for both technologies against the scratch output directory; GEFS columns
  are not written (see "GEFS" below).
- `nwp_forecast_comparison.py`: `rows()` (the shared row set, planned-arm columns required
  complete), `assign_folds_with_eras()` and `coverage_table()` (against #885's not-yet-merged
  `studies.cross_validation.cut_eras` — see "Folds and #885"), `arm_columns()` (fixed-length
  feature tuples per arm), `jobs()` (one job per (arm, hyperparameter setting), skipping any
  product whose columns are not in the input frame), `add_blend_guard_columns()`
  (`studies.blending.climatology_permutation` on the blend's non-ENS products), `run_jobs()` (not
  called by `--dry-run` or `--report-only`), `baseline_losses()`, `bracket_verdicts()`,
  `blend_verdicts()`, `check_ens_monotonicity()`, `exact_lead_wind_rereads()`, `leaderboard()`,
  `assert_equal_rows()` (called before every interval), `fingerprint()` (Float32 cast before
  hashing) and `write_report()`. `--dry-run`, `--report-only` and `--fit-missing` flags; fitting
  itself (`--report-only`'s absence) refuses to run in this pass and exits with an error naming why.
- `nwp_forecast_charts.py`: skeleton — the five chart functions from the plan's "Charts" section,
  each raising `NotImplementedError` until a real report exists to chart.

## GEFS

`data/studies/weather/GEFS/_month_cache/` holds 27 cached months but does not yet reach the month
before today, so `build_forecast_inputs.py`'s GEFS gate does not open and no `gefs_*` columns are
written. The member-mean build itself (nearest-cell selection per site through
`studies.grid_sampling.nearest_grid_indices`, the ensemble mean, and `gefs_step_means` before any
band slicing) is not implemented in this pass — there is nothing to build against yet beyond a
`GEFS_window_*` test extract, and `--gefs-window-dir` logs that this build step is still missing
rather than attempting it. `rows()` logs and skips `gefs_mean_day1` from the planned-arm shared-row
filter when its columns are absent, so the row set does not depend on GEFS being present.

## Folds and #885

The fold helper (`cut_eras`, `rotate_folds`, `calendar_month_coverage`, `uncovered_months`,
`raise_on_uncovered_months`) is added to `studies.cross_validation` by PR #885
(`origin/ens-hres-past-wind`), not yet on `main`. `assign_folds_with_eras()` and `coverage_table()`
import it inside the function body, with a `TODO(#885)` naming the PR, so the rest of this module
stays importable while #885 is outstanding — confirmed by `uv run ty check`, which reports the
unresolved import only inside those two functions.

`ERA_FOLD_OFFSETS` itself is #885's, tuned on its own wind rows; it is not yet importable either.
`NWP_ERA_FOLD_OFFSETS` in `nwp_forecast_comparison.py` is a **documented placeholder**
(`{0: 0, 1: 0, 2: 2}`), not a re-derivation of #885's search
(`studies.cross_validation.search_fold_offsets`), and is marked `TODO(#885)` to be replaced once the
PR's own constant is importable.

**`--dry-run` was tested against a local, uncommitted copy of #885's `cross_validation.py`** (copied
over the top of this branch's file, the dry run run, then the file restored — never committed to
this branch). Against that copy:

- Without #885 on this branch: `assign_folds_with_eras` raises `ImportError` on `cut_eras`, as
  expected and as the module docstring states.
- With #885's `cross_validation.py` in place: rows build (34,087 solar rows after the shared-row
  filter, GEFS skipped), and folds are cut successfully. `coverage_table()` then raises, because two
  (site, fold, calendar month) cells are uncovered under this module's placeholder
  `NWP_ERA_FOLD_OFFSETS` (site E, fold 3, May; site E, fold 4, July) — exactly the rotation problem
  the PR's own tuned offsets exist to solve. The dry run reaches the fold step and the coverage
  check, as the brief asked; it does not reach the job-list print, because a real run needs the
  PR's tuned offsets rather than this placeholder.

## What is not built or run this pass

- **No XGBoost fit.** `run_jobs()` is written but never called by `main()`; `--fit-missing` and a
  plain run both refuse with a logged error.
- **GEFS's member-mean build** (see "GEFS" above).
- **`choose_method`** (ENS's upsampling technique is chosen by a fit-based rule on the ENS-horizons
  page); this pass reads the `linear` technique instead, documented in
  `build_forecast_inputs.py`'s module docstring.
- **The charts** (`nwp_forecast_charts.py` is a skeleton).
- **The docs page, roadmap update and background-survey correction** the plan's "What changes,
  file by file" section lists — out of scope for a no-fit pass.

## Reproducing this pass

```bash
uv run python studies/nwp_forecast_comparison/verify_previous_runs_leads.py --output-dir <dir>
uv run python studies/nwp_forecast_comparison/check_input_steps.py --output-dir <dir>
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --output-dir <dir>
uv run python studies/nwp_forecast_comparison/nwp_forecast_comparison.py --dry-run \
    --input-dir <dir> --output-dir <dir>
```

`verify_previous_runs_leads.py` and `check_input_steps.py` read `data/studies/weather/` and write
only under `--output-dir`. `build_forecast_inputs.py` reads the past-weather studies' rows, the ENS
member extract, and Previous Runs, and writes only under `--output-dir`; it never writes under
`data/studies/` in this pass. `nwp_forecast_comparison.py --dry-run` reads `--input-dir` (that
output) and stops before any fit; it needs #885's `cross_validation.py` on the branch it runs
against to get past `assign_folds_with_eras` (see "Folds and #885").
