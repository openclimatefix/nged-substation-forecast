# Plan: a scattered-null detector for the instantaneous NWP variables

**The problem.** An instantaneous NWP variable (temperature, dew point, the winds, the pressures,
geopotential height) is never legitimately null, and the `Nwp` contract enforces that by making the
columns non-nullable. But the H3 aggregation renormalises each cell over the grid points that
supplied a value, so a *scattered* upstream null in one of these is absorbed by its cell's other
points and mostly never reaches the contract — the exception is the 10 cells of the V1 grid fed by
a single point, where there are no other points to absorb it. Today nothing counts the absorbed
ones. The known-issues page already concedes the gap in prose — "scattered corruption in a variable
that should never carry any is now mostly invisible at ingest. What it costs is a detector" — and
this is that detector. The gap is one-sided: a *blocky* failure still fails ingest, because every
point of a cell goes at once, so `Nwp.validate` rejects the frame before any check runs.

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

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/download.py`

Add `ECMWF_ENS_INSTANTANEOUS_VARS`, derived from the download list so it cannot drift:

```text
frozenset(_ECMWF_ENS_VARS_TO_DOWNLOAD) - Nwp.deaccumulated_var_names - Nwp.categorical_var_names
```

**The set has to come from the download list, not from the `Nwp` contract**, because the counter
indexes the raw `xr.Dataset` and the two namespaces diverge on wind: the download carries
`wind_u_10m` / `wind_v_10m` / `wind_u_100m` / `wind_v_100m`, and `convert_to_polars` derives
`wind_speed_*` / `wind_direction_*` from them and drops the components. A set built from `Nwp`
would name four variables `ds` does not have, and `ds[name]` would raise `KeyError` on every run —
see "What the review changed" below. `categorical_precipitation_type_surface` is excluded by that
subtraction and should be: it is nullable, aggregated as an area-weighted mode rather than a mean,
and all-null by design before 2024-11-13, so a zero-threshold check counting it would fire on every
early backfill partition.

Nothing changes in `packages/contracts/`.

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/upstream_nulls.py`

Give `UpstreamNullRate` the per-variable breakdown, shaped like its sibling
`contracts.weather_schemas.NwpQualityReport` — one frozen dataclass holding a single
`pl.DataFrame`, with every scalar derived from it:

- `per_variable: pl.DataFrame`, one row per counted variable, columns `variable`, `n_null`,
  `n_affected_slices`, `n_total`. Empty when no variables were counted.
- `n_null_nwp_grid_points`, `n_total_nwp_grid_points`, `n_affected_nwp_slices`,
  `affected_nwp_variables` and `null_nwp_grid_point_fraction` all become derived properties, so the
  breakdown and the totals cannot disagree. The zero-denominator guard and `is_healthy` are
  unchanged.

Keeping a per-variable `n_total` column rather than one shared denominator is deliberate: the
function accepts an arbitrary `xr.Dataset`, so equal sizes are a property of the real feed, not of
the signature, and a shared value would be silently wrong if that ever stopped holding.

`assess_upstream_grid_point_nulls` keeps its signature; only its return value gains structure.

### `src/nged_substation_forecast/defs/assets.py`

- A third `AssetCheckSpec`, `nwp_instantaneous_variables_have_no_nulls`, `blocking=False`, with a
  standing `description` in the same voice as `_NWP_QUALITY_CHECK_DESCRIPTION`. It must say three
  things: a null here is anomalous rather than tolerated; a *blocky* failure never reaches this
  check because it fails ingest instead; and the count **excludes lead-0**, which is a real blind
  spot for these variables (unlike for the de-accumulated ones, where lead-0 nulls are by design).
- A second `assess_upstream_grid_point_nulls` call inside the existing `except BaseException` guard
  at `assets.py:340`, and a second `AssetCheckResult` whose `passed` is
  `upstream_instantaneous.is_healthy`. The degraded path must emit a third
  `_degraded_nwp_check_result`, since Dagster fails the step for a declared spec with no result.
- Metadata keys reuse the population-naming scheme (`n_null_nwp_grid_points`,
  `n_total_nwp_grid_points`, `null_nwp_grid_point_fraction`, `n_affected_nwp_slices`,
  `affected_nwp_variables`) — the same names on a different check, which is unambiguous because the
  check names the population.
- Add the per-variable breakdown to both checks as a `MetadataValue.table` with a fixed
  `TableSchema` alongside `_NWP_NULL_SLICES_SCHEMA`, rendered through the same route as
  `_nwp_null_slices_metadata`, so an empty table still renders. A table, not N× flattened keys: the
  key set would otherwise change with the variable set.
- Materialisation metadata is unchanged. The instantaneous rate goes on the check only — the
  timeline already plots the de-accumulated fraction, and a second near-always-zero series adds
  noise, not signal.
- Two comments say there are two checks and become wrong: `assets.py:334` ("Two non-fatal per-run
  checks") and `assets.py:359` ("both checks share this assessment, so both degrade together" —
  which also stops being true, since the new check has its own assessment). `_degraded_nwp_check_result`'s
  docstring says "an asset declaring two `AssetCheckSpec`s"; that count becomes three.

## Design-philosophy check

Production path, so it degrades rather than raises. The new call sits inside the `try` at
`assets.py:340`, whose `except BaseException` re-raises only `KeyboardInterrupt | SystemExit |
DagsterExecutionInterruptedError`, and `write_nwp` is below it — so a bug here yields three WARN
results and still lands the run, and cannot duplicate rows. WARN, `blocking=False`, and nothing in
the new code can raise on absent input. Rules 6 and 7 of Inherent Stability.

The new check is the provider channel again, but for the variables where a null means "something is
wrong" rather than "the known corruption happened" — which is why it gets its own `passed` instead
of widening the existing check's. Pooling the two rates into one number would average over opposite
null semantics and measure nothing.

## Tests

Each of these fails on `main` today:

- **`test_upstream_nulls.py`** — the per-variable breakdown attributes counts to the right variable
  when two variables are corrupt by different amounts, and the derived pooled scalars still equal
  the independently-computed constants the existing tests pin.
- **`test_upstream_nulls.py`** — the counted instantaneous set is a subset of the committed real
  slice's `data_vars`. This is the test that would have caught the wind-namespace defect, on real
  bytes.
- **`tests/test_assets.py`** — extend `_make_downloaded_ds` to carry all thirteen download
  variables (built from `_ECMWF_ENS_VARS_TO_DOWNLOAD`, with the drift assertion
  `packages/dynamical_data/tests/conftest.py:38-40` already uses), then assert that the new check is
  `passed=False` while `nwp_has_no_unexpected_nulls` is `passed=True`. The fixture already nulls
  `temperature_2m` beyond lead-0, so this is a new assertion on an existing scenario — but the
  fixture must be widened, or a counter reading all nine instantaneous names raises `KeyError` in
  the tests too.
- **`tests/test_assets.py`** — the degraded-assessment test gains the third check to its expected
  result set, so a raise still yields three results and lands the run.

## Docs to update

- **`docs/architecture/ecmwf-ens-known-issues.md`** — line 19 says "two non-fatal reporters sit
  behind the asset's two WARN checks"; the "A wholly-missing variable, and instantaneous nulls
  (fatal)" section ends by naming the detector we gave up, and should describe the one we now have,
  including its lead-0 blind spot; the "Two populations, counted separately" section gains the
  second check.
- **`docs/live_service/operations.md`** — line 290 ("**Both NWP checks share one description**…")
  is the paragraph an operator reads when the degraded path fires, and it is now wrong about how
  many checks go yellow together. Add a runbook paragraph next to "Reading the NWP check": what
  `passed=False` on the new check means and what to do (a Dynamical.org conversation, not a rerun).
- **`tests/test_assets.py:774` and `:813`** — docstrings asserting "Both declared checks".
- **No change to `docs/api/dynamical_data/index.md`** — line 7 is a whole-module mkdocstrings
  directive, so any new public class renders automatically.

## Verification

The green-before-push set, plus `uv run mkdocs build --strict` with the rendered HTML read, plus
**`uv run pytest --run-network -m network`** — this touches the same convention-sensitive path
as #573, which shipped without that run, and the wind-namespace defect is exactly the class of bug
that run exists to catch.

## Risks and open questions

1. **Is a zero threshold noisy?** Two pieces of evidence in the repo say probably not. The
   known-issues page records that an instantaneous variable's nulls "have only ever arrived as
   whole-step dropouts"; and the 862-run archive is itself a partial detector, because scattered
   corruption is spatially clustered and a scatter episode inside our small GB box would likely have
   taken out one of the 10 single-point cells and failed that partition. **Recommendation: ship the
   zero threshold**, on that evidence rather than on a claim of ignorance, and say in the docs that
   it bounds only corruption inside the downloaded box. The check is WARN and non-blocking, so if
   it does fire often, that is itself the finding.
2. **Should this reuse `nwp_has_no_unexpected_nulls` instead of adding a spec?** No — the remedies
   genuinely differ. The runbook's advice for the de-accumulated grid rate is "read the trend, do
   not act on one run"; for an instantaneous null it is "email Dynamical.org now". A `passed` that
   mixes them is unactionable.

## What the review changed

- **The instantaneous set moved from `contracts` to `download.py`.** The plan had
  `Nwp.instantaneous_var_names()` derived from the contract's fields. The reviewer found that the
  contract carries `wind_speed_*`/`wind_direction_*` while the downloaded dataset carries
  `wind_u_*`/`wind_v_*`, so `ds[name]` would have raised `KeyError` **on every run** — and because
  that raise lands in the shared guard, it would have degraded all three checks and dropped the
  materialisation metadata daily, destroying two signals that work today. Verified against
  `download.py:19-33` and the committed real slice: four of the nine names are absent from `ds`.
  This is the single most valuable finding in the review.
- **Two tests added to catch that class of bug**: the real-slice subset test, and building
  `_make_downloaded_ds` from the download list with the existing drift assertion.
- **The lead-0 blind spot is now stated.** The counter filters `lead_time > 0` unconditionally,
  which is right for the de-accumulated variables and is a genuine gap for the instantaneous ones.
  Not worth a parameter, but the description must not claim coverage it lacks.
- **The breakdown is a `pl.DataFrame`, not a `VariableNullCount` dataclass and a shared
  denominator.** The sibling `NwpQualityReport` already holds exactly this shape, and it removes
  the shared-denominator precondition the reviewer objected to.
- **Six "two checks" strings** in docs, comments and test docstrings that the plan had missed, and
  one listed doc edit (`docs/api/`) struck as unnecessary.
- **Two claims corrected**: "never reaches the contract at all" ignored the 10 single-point cells;
  the zero-threshold question is better-evidenced than "not cheaply knowable" allowed.
- **One proposed test dropped** as vacuous: "the derived pooled scalars equal the sum of the
  breakdown" is true by construction once both derive from the same frame.

**Rejected.** The reviewer argued the per-variable breakdown is a separable second change that
should be split into its own PR, since the detector goal is met by `affected_nwp_variables` alone.
That is a fair reading of scope, but bundling it here is Jack's explicit decision, taken so the
instantaneous check and the breakdown settle the "one call or two" question together; and doing it
now means the metadata keys move once rather than twice.
