# Weather forecasts for power, compared at matched lead times

Study for [issue #810](https://github.com/openclimatefix/nged-substation-forecast/issues/810). The
plan that was reviewed and approved is `plans/nwp-forecast-comparison.md` on the branch this study
was built on (deleted once the study merges, per the workspace's own convention — see the plan's
review log for the two adversarial plan reviews it went through). No fit has run yet: this pass
built the verification scripts and the input-building code, and ran every verification check that
needs no fit.

## Status

**Pre-fit verification only.** `verify_previous_runs_leads.py` (V1, V1b, V3) and
`check_input_steps.py` (V1c) have been run, against the data already on disk. No XGBoost fit has
run, `build_forecast_inputs.py` has not been run, and `nwp_forecast_comparison.py` and
`nwp_forecast_charts.py` are not yet written — see "What is not built yet" below.

## Verification results

Every table below was written by a committed script; none of these numbers was hand-typed.

### V1 — the Previous Runs run-selection rule (a gate)

`verify_previous_runs_leads.py` compares Open-Meteo's served GFS `_previous_dayN` wind speed and
radiation against the raw Dynamical.org GFS archive in
`data/studies/weather/GFS_window_2025-07-01_2025-07-02/`, for candidate runs `24N + k` hours before
each hour, `k` from 0 to 12. The plan's gate is that the mean absolute difference is lowest at
`k = 0` for both `N = 1` and `N = 2`.

**Wind speed at 100 m: N = 2 passed cleanly (lowest mean absolute difference at k = 0, rising
monotonically after it); N = 1 did not (lowest at k = 3, but by about 1% of the k = 0 value, on a
curve that is flat across k = 0 to k = 4).** The gate as stated in the plan (both N) did not pass.

Wind was compared as a plain average over the Dynamical.org extract's nine grid cells (there is no
per-site match in this extract, and building one needs the private roster's coordinates, which this
pass did not attempt — see "Deviations" below). The candidate-vs-served correlation at k = 0, N = 1
is 0.86, and the two series' means and standard deviations are close, which is consistent with a
real weather signal carrying genuine spatial-sampling noise (grid-cell average against a per-site
point series) rather than a wrong run being read: a wrong run would show materially lower
correlation. At N = 1 the lead is short enough (24 to 36 hours) that this spatial noise is the same
order as the k-to-k signal the gate is trying to detect; at N = 2 (48 to 60 hours) genuine forecast
error has grown enough to dominate it. The likely fix is scoring V1 at each site's own nearest grid
cell rather than the box average, which needs careful, print-nothing handling of the private roster
that this pass did not have time to do safely. **Treat V1's rule as supported at N = 2 and
unconfirmed at N = 1** until that is redone.

Radiation, checked only at hours divisible by 6 (the only hours where selecting the run by the
hour's label and by the hour's start disagree), supports the label convention clearly: mean
absolute difference 19.1 W/m² (N = 1) and 16.8 W/m² (N = 2) by label, against 30.3 and 29.5 by
start.

Full tables: `v1_wind.md`, `v1_radiation.md` in the output directory.

### V1b — each product's run-switch pattern

`_previous_day1`'s mean absolute second difference, by UTC hour of day, for 100 m wind speed and
2 m temperature. **The threshold this script uses (mean plus one standard deviation) is too loose:
it reads a 1-hourly cycle for nearly every product**, including GFS, ICON-EU and ICON global, which
the plan's own review measured as switching 6-hourly. Only IFS 0.25° (3-hourly here) and the
overall shape of the flagged hours are informative; the inferred-cycle column in
`v1b_run_switches.md` should not be read as confirming or contradicting the plan's cycle claims,
and a tighter or differently-shaped detector is needed before V1b's result can be relied on.

### V1c — steps in the planned inputs

`check_input_steps.py` reads UKV's and IFS 0.25°'s day-1 monthly mean radiation and 100 m wind
speed, per anonymised site, as a ratio to ICON-EU's day-1 monthly mean at the same site and month,
and flags a month where that ratio moves by 15% or more from the month before. GEFS and ENS are not
covered by this screen (see the script's docstring). See `v1c_steps.md` for the flagged months. The
November-to-December 2024 jump recurs at nearly every site with almost the same ratio, which looks
like a seasonal solar-geometry artefact of the UKV/ICON-EU ratio near the winter solstice rather
than a product version step, but this pass did not chase it further.

### V3 — timestamp and height conventions

`v3_conventions.md` reads each product's timestamp convention (instantaneous snapshot or
hour-ending mean) from the "best fit" line already recorded in that product's own
`previous_runs/lineage.json` at fetch time, and states the height convention already established
elsewhere in this project's study work (ICON's served "100 m" wind is its native 120 m wind
rescaled by about 0.98). It does not re-run a wind power-hour offset scan (see "Deviations" below).

### V2 — GEFS radiation window conversion

Not run: the GEFS extract this study needs is still downloading, and the plan gates any GEFS check
on it. `gefs_step_means` (`packages/studies/src/studies/resample.py`), the function V2 exercises,
is written and tested (see "What is built" below); V2 itself is deferred to a session run after the
download completes.

## What is built

- `packages/studies/src/studies/bootstrap.py`: `bracket_verdict(lower_side, upper_side)`, turning a
  bracket's two `BootstrapInterval`s into `"beats"`, `"loses"` or `"unresolved"`.
- `packages/studies/src/studies/resample.py`: `gefs_step_means`, converting GEFS's alternating 3-
  and 6-hour window-mean radiation to plain 3-hour step means.
- Tests for both, in `packages/studies/tests/`.
- `verify_previous_runs_leads.py` (V1, V1b, V3) and `check_input_steps.py` (V1c), both runnable with
  `--output-dir`.

## What is not built yet

`build_forecast_inputs.py`, `nwp_forecast_comparison.py` and `nwp_forecast_charts.py`, the scripts
that would build the per-technology weather-arm parquet, run the folds and the out-of-fold fits,
and chart the results, are **not written in this pass**. Two reasons:

1. **V1's gate, as measured, did not pass at N = 1.** Building the rest of the pipeline on top of an
   unconfirmed run-selection rule risks compounding the same spatial-sampling noise into every
   bracket the study computes. The next session should first re-run V1 with a per-site nearest-cell
   match (reading the private roster's coordinates at run time and never printing them, as
   `fetch_wind_point.py` already does elsewhere in this project), confirm the gate at both N = 1 and
   N = 2, and only then build the rest.
2. **Time.** The remaining pipeline — rows and folds shared across nine planned and several
   exploratory arms, the fold call site against #885's not-yet-merged
   `studies.cross_validation.cut_eras`, the out-of-fold loop, the bracket and blend intervals, the
   report, and the chart skeleton — is the larger part of the plan's implementation and was not
   reached in this pass.

Both are recorded here rather than in `plans/nwp-forecast-comparison.md`, which this pass did not
edit.

## Reproducing the verification tables

```bash
uv run python studies/nwp_forecast_comparison/verify_previous_runs_leads.py --output-dir <dir>
uv run python studies/nwp_forecast_comparison/check_input_steps.py --output-dir <dir>
```

Both scripts read `data/studies/weather/` and write only under `--output-dir`; neither writes under
`data/studies/`.
