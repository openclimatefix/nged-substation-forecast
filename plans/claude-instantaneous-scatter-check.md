# Plan: a scattered-null detector for the instantaneous NWP variables

**The problem.** An instantaneous NWP variable (temperature, dew point, the winds, the pressures,
geopotential height) is never legitimately null, and the `Nwp` contract enforces that by making the
columns non-nullable. But the H3 aggregation renormalises each cell over the grid points that
supplied a value, so a *scattered* upstream null in one of these is absorbed by its cell's other
points and never reaches the contract at all. Today nothing counts it. The known-issues page already
concedes the gap in prose — "scattered corruption in a variable that should never carry any is now
mostly invisible at ingest. What it costs is a detector" — and this is that detector. The gap is
one-sided: a *blocky* failure still fails ingest, because every point of a cell goes at once.

**The plan.** Call the existing `assess_upstream_grid_point_nulls` a second time with the
instantaneous variable set, and publish the result as a third `AssetCheckSpec` on `ecmwf_ens` with
its own `passed` — false on any null at all, because for these variables zero is the correct
threshold. Alongside it, give `UpstreamNullRate` the per-variable breakdown its loop already
computes and currently discards, so both checks can say *which* variable is corrupt and by how much.

## Verdict

Worth doing. It closes a detection gap the docs already name, and it reuses the counter unchanged
apart from its return shape — `variables` is already a parameter precisely so this stays small.

**No GitHub issue exists for this yet.** It was agreed in conversation as the follow-on to #505.
Filing one is a one-line job if you want the board to carry it; say the word.

## What changes, file by file

### `packages/contracts/src/contracts/weather_schemas.py`

Add `Nwp.instantaneous_var_names()`, a classmethod deriving the set rather than listing it:
`all_weather_var_names() - categorical_var_names - deaccumulated_var_names`. Derived, so it cannot
drift when a variable is added. `categorical_precipitation_type_surface` is excluded by that
subtraction and should be — it is nullable, it has its own historical invariant, and a category
cannot be averaged, so its nulls are a third thing again.

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/upstream_nulls.py`

Reshape `UpstreamNullRate` around the per-variable numbers:

- New frozen `VariableNullCount` holding `n_null_nwp_grid_points` and `n_affected_nwp_slices`.
- `UpstreamNullRate` holds `per_variable: Mapping[str, VariableNullCount]` plus
  `n_nwp_grid_points_per_variable: int` — a single shared denominator, **not** a per-variable one.
  Every variable in one downloaded run has dims `(lead_time, ensemble_member, latitude, longitude)`
  and therefore identical `size`; verified on the committed real ECMWF slice. Storing it per
  variable writes the same number N times and implies it can differ.
- The four existing scalars plus `affected_nwp_variables` become derived properties, so the
  breakdown and the totals cannot disagree. `null_nwp_grid_point_fraction` keeps its zero-denominator
  guard and `is_healthy` is unchanged.

`assess_upstream_grid_point_nulls` keeps its signature; only its return value gains structure.

### `src/nged_substation_forecast/defs/assets.py`

- A third `AssetCheckSpec`, `nwp_instantaneous_variables_have_no_nulls`, `blocking=False`, with a
  standing `description` in the same voice as `_NWP_QUALITY_CHECK_DESCRIPTION` — it has to say that
  a null here is anomalous rather than tolerated, and that a *blocky* failure never reaches this
  check because it fails ingest instead.
- A second `assess_upstream_grid_point_nulls` call inside the existing `except BaseException` guard,
  and a second `AssetCheckResult` whose `passed` is `upstream_instantaneous.is_healthy`. The
  degraded path must produce a result for this check too.
- Metadata keys reuse the population-naming scheme (`n_null_nwp_grid_points`,
  `n_total_nwp_grid_points`, `null_nwp_grid_point_fraction`, `n_affected_nwp_slices`,
  `affected_nwp_variables`) — same names on a different check, which is unambiguous because the
  check names the population.
- Add the per-variable breakdown to both checks as a `MetadataValue.table` with a fixed
  `TableSchema` alongside `_NWP_NULL_SLICES_SCHEMA`, so an empty table still renders. A table, not
  N× flattened keys: the key set would otherwise change with the variable set.
- `_degraded_nwp_check_result`'s docstring says "an asset declaring two `AssetCheckSpec`s"; that
  count becomes three.
- Materialisation metadata is unchanged. The instantaneous rate goes on the check only — the
  timeline already plots the de-accumulated fraction, and a second near-always-zero series adds
  noise, not signal.

## Design-philosophy check

Production path, so it degrades rather than raises. The new call sits inside the same guard, which
already runs before `write_nwp`; `passed=False` is a WARN, `blocking=False`, and nothing in the new
code can raise on absent input (the fraction's zero guard is the only arithmetic trap and it is
already there). Rules 6 and 7 of Inherent Stability.

The new check is the provider channel again, but for the variables where a null means "something is
wrong" rather than "the known corruption happened" — which is why it gets its own `passed` instead
of widening the existing check's. Pooling the two rates into one number would average over opposite
null semantics and measure nothing.

## Tests

Each of these fails on `main` today:

- **`test_upstream_nulls.py`** — the per-variable breakdown attributes counts to the right variable
  when two variables are corrupt by different amounts (no such field today); the shared denominator
  equals one variable's `size`; the derived pooled scalars equal the sum of the breakdown.
- **`test_assets.py`** — a run whose instantaneous variables carry scattered grid-point nulls while
  its stored cells are clean: the new check is `passed=False` and the de-accumulated check is
  `passed=True`, which is the whole point of separating them. Today there is no such check to
  assert on.
- **`test_assets.py`** — the degraded-assessment test gains the new check to its expected result
  set, so a raise still yields three results and lands the run.
- **`test_weather_schemas.py`** — `instantaneous_var_names()` excludes the de-accumulated three and
  the categorical one, and its union with them is `all_weather_var_names()`.

## Docs to update

- **`docs/architecture/ecmwf-ens-known-issues.md`** — the "A wholly-missing variable, and
  instantaneous nulls (fatal)" section currently ends by saying the detector is what we gave up;
  rewrite that passage to describe the detector we now have. The "Two populations, counted
  separately" section gains the second check.
- **`docs/live_service/operations.md`** — a runbook paragraph next to the existing "Reading the NWP
  check" block: what `passed=False` on the new check means and what to do about it (it is a
  Dynamical.org conversation, not a rerun).
- **`docs/api/dynamical_data/index.md`** — `VariableNullCount` in the rendered API.

## Verification

The green-before-push set, plus `uv run mkdocs build --strict` with the rendered HTML read, plus
**`uv run pytest --run-network -m network`** — this touches the same convention-sensitive path
as #573, which shipped without that run.

## Risks and open questions

1. **Is a zero threshold noisy?** Unknown, and not cheaply knowable: the raw grid is not stored, so
   measuring the historical instantaneous scatter rate means re-downloading archived runs. The
   archive figures we do have cover the de-accumulated variables only (12 of 862 runs carry any
   null). **Recommendation: ship the zero threshold.** The contract already says any null in these
   is anomalous, the check is WARN and non-blocking so noise costs a yellow marker and nothing else,
   and if it does fire often that is itself the finding.
2. **Should this reuse `nwp_has_no_unexpected_nulls` instead of adding a spec?** No — the two need
   different `passed` semantics, and a check whose `passed` mixes "tolerated corruption" with
   "anomalous null" cannot be acted on. Recorded because it is the obvious simplification and the
   answer should be on the record.
