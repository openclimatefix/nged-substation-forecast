# Plan: XGBoost weekday codes follow the declared `Enum` order (#1059)

**The problem.** `_prepare_features` in `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`
encodes every `String`, `Categorical`, and `Enum` column with `cast(pl.Categorical).to_physical()`.
Casting to `Categorical` numbers values in order of first appearance in the frame, so the same
weekday gets different codes in different frames. `local_day_of_week` is in the default
`selected_features`, so training and prediction can disagree about what a code means, and nothing
raises.

**The solution.** For an `Enum` column, call `to_physical()` directly, so each code is the value's
position in the declared list. For a `String` or `Categorical` column, which has no declared list,
raise a `ValueError` that names the column. No feature uses either dtype today.

## Verdict, size and departures

- **Verdict:** worth doing, as described, with one departure below.
- **Size: complex.** Trigger answers:
    - *What gets stored:* no Patito model, Delta table or asset changes. Saved XGBoost models trained
      with `local_day_of_week` learned the broken mapping; the project's "young project" rule says a
      retrain is acceptable, so no migration path.
    - *Production serving path:* **fires.** `XGBoostForecaster.predict` calls `_prepare_features`.
    - *Degradation rule:* does not fire. The new `ValueError` covers a dtype that is our own bug (a
      contract violation), not absent or stale input from the outside world.
    - *More than one defensible design:* **fires.** Saving a category list with the model versus
      using the declared `Enum` order versus raising on `String`.
    - *Callers not nameable without searching:* does not fire. Two callers, both in `forecaster.py`
      (`train` and `predict`).
- **Reviews bought:** both plan reviews and both diff reviews.
- **Departure from the issue:** the issue leaves open whether to support `String` by saving a list
  with the model. The plan raises instead. Supporting `String` means a new field in `meta.json`,
  load-time handling, and unseen-category rules, for a case no feature reaches.

## What changes, file by file

- `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`, `_prepare_features`:
    - `isinstance(dtype, pl.Enum)`: `pl.col(col).to_physical().cast(pl.Float32)`. Nulls stay null and
      become NaN, as today.
    - `dtype in (pl.String, pl.Categorical)`: raise `ValueError` naming the column and dtype,
      saying only `Enum` columns are supported because their codes are fixed by the declared list.
    - Update the docstring to say so.
- `packages/xgboost_forecaster/tests/test_forecaster.py`: new tests below.

## Design-philosophy check

- `_prepare_features` runs in production (`predict`) and in R&D (`train`). The `ValueError` is
  reachable only by selecting a `String` feature, which `AllFeatures` already types as `Enum` where
  it exists, so it is a contract bug and fail-fast is right. It is not triggered by missing or stale
  inputs, which arrive as nulls and still encode to NaN.
- No asset check is added or edited.
- The change moves the production path closer to "load a model, call `predict`": the encoding no
  longer depends on the rows in the frame.

## Tests

All in `test_forecaster.py`, calling `_prepare_features` directly.

- **Same weekday, different frames, same code.** Two frames of the seven weekdays, one starting on
  Monday and one on Thursday, both typed `pl.Enum([...Monday..Sunday])`. Assert Thursday encodes to
  `3.0` in both. On `main` it is `3.0` and `0.0`.
- **Codes match declared positions.** Assert the encoding of a Wednesday-first, Monday-last frame
  equals `[2.0, ..., 0.0]` positions. On `main` it is `0.0` for the first row.
- **Absent weekdays do not shift later codes.** A frame holding only Friday and Sunday encodes to
  `[4.0, 6.0]`. On `main` it is `[0.0, 1.0]`.
- **Null stays NaN.** An `Enum` frame with a null encodes the null to NaN. Passes on `main`; kept as
  a guard on the new `to_physical()` branch, not as a test of this change.
- **`String` column raises.** `pytest.raises(ValueError, match="<column name>")` on a `String`
  feature column. On `main` it encodes silently.
- **End to end.** Train on a frame that starts on Monday with `local_day_of_week` selected, predict
  on the same rows reordered to start on Thursday, and assert predictions match row for row after
  sorting by `valid_time`. Fails on `main` because the codes differ between frames.

## Docs to update

- `docs/ml_experimentation/model-configuration.md` line 100 says `local_day_of_week` is "a
  categorical". Reword to say XGBoost receives its position in the declared Monday-to-Sunday list
  (0 to 6).
- No roadmap item completes, so no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/xgboost_forecaster
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

Also run pydoclint and the docs-link checker, as CI does.

## Risks and open questions

- **Past results are affected.** Every XGBoost run with `local_day_of_week` selected used the broken
  encoding. Recommendation: do not re-score in this PR. After merge, re-score one fold to size the
  effect, as the issue suggests, as its own piece of work.
- **Raise versus support for `String`.** Recommendation: raise, as above.
- **Saved models.** Existing saved models keep the old learned mapping and now see the correct
  codes. Recommendation: retrain, with no migration path.
