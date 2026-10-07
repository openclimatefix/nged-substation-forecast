# Plan: XGBoost weekday codes follow the declared `Enum` order (#1059)

**The problem.** `_prepare_features` in `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`
encodes every `String`, `Categorical`, and `Enum` column with `cast(pl.Categorical).to_physical()`.
Casting to `Categorical` numbers values in order of first appearance in the frame, so the same
weekday gets different codes in different frames. `local_day_of_week` is in the default
`selected_features`, so training and prediction can disagree about what a code means, and nothing
raises.

**The solution.** For an `Enum` column, call `to_physical()` directly, so each code is the value's
position in the declared list. Every other dtype falls through to the existing `cast(pl.Float32)`,
which already raises on a `String` or `Categorical` column. No feature uses either dtype today.

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
  with the model, or to raise. The plan adds no code for `String`: the existing `Float32` cast
  already fails loudly, naming the column. Supporting `String` would need a new `meta.json` field and
  unseen-category rules for a case no feature reaches.

## What changes, file by file

- `packages/xgboost_forecaster/src/xgboost_forecaster/forecaster.py`, `_prepare_features`: the
  condition becomes `isinstance(dtype, pl.Enum)` and the expression
  `pl.col(col).to_physical().cast(pl.Float32)`. Nulls stay null and become NaN, as today. The
  docstring says `Enum` codes are declared positions and other non-numeric dtypes fail the cast.
- `packages/xgboost_forecaster/tests/test_forecaster.py`: the new test below.

## Design-philosophy check

- `_prepare_features` runs in production (`predict`) and in R&D (`train`). The only new failure is
  unchanged from today's cast: it needs a non-`Enum` string feature, which is a contract bug.
  Missing or stale inputs arrive as nulls and still encode to NaN.
- No asset check is added or edited.
- The change moves the production path closer to "load a model, call `predict`": the encoding no
  longer depends on the rows in the frame.

## Tests

One test in `test_forecaster.py`, calling `_prepare_features` directly on a frame typed
`pl.Enum([Monday..Sunday])` with rows `[Thursday, Friday, Sunday, Monday, null]`. It asserts
`[3.0, 4.0, 6.0, 0.0, NaN]`. On `main` the codes are `[0, 1, 2, 3, NaN]`, so the test fails there. One
frame covers row-order dependence, declared positions, skipped weekdays, and null handling.

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
- **Raise versus support for `String`.** Recommendation: neither; rely on the existing cast.
- **Saved models.** Existing saved models keep the old learned mapping and now see the correct
  codes. Recommendation: retrain, with no migration path.

## Simplicity review: findings and triage

- **Accepted:** drop the hand-written `ValueError` (Polars already raises, naming the column).
- **Accepted:** merge the four encoding tests into one frame.
- **Accepted:** drop the end-to-end train/predict test, since both call the same `_prepare_features`.
- **Accepted:** drop the `String` raise test with the raise.
- **Rejected:** downsize to medium with one diff review. The serving-path trigger fires, and the
  sizing rule makes any firing trigger complex however small the diff.
