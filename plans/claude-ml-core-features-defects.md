# Fix four defects in `ml_core.features`

**The problem.** A clean-room review of `packages/ml_core/src/ml_core/features/` found four
defects. Two are latent correctness bugs: `time_series_type` — a requestable feature — is null on
every bulk-mode row that has no matching power observation, which in a backtest is every row past
the last observation; and `AllFeatures.validate()`, the primary-key uniqueness check that two
commits reason about as the fan-out backstop, is called nowhere and would raise if it were, because
the frame legitimately carries nulls in two columns the schema declares non-nullable. Two are
performance defects that contradict what `docs/architecture/performance.md` promises: the module's
one sanctioned `collect()` runs on the post-upsample frame, making it linear in the NWP window
rather than O(1); and a `rolling().agg()` + join-back drags an unconditional full sort of the
largest frame in the system, which runs even for the production config, which requests no rolling
feature at all.

**The solution.** Four independent fixes in one PR, in a fixed order because two of them interact.
Move the metadata join off the power frame and onto the assembled frame, so
`time_series_type` is populated in both modes. Then make `power` and `time_series_type` nullable on
`AllFeatures` — the declaration is what is wrong, not the data — and start calling
`AllFeatures.validate()` in tests, which is the only place a full `collect()` is affordable. Probe
the raw NWP frame rather than the upsampled one for the control member. And replace
`rolling().agg()` + join-back with `rolling_mean_by(...).over(...)`, which needs no join and no
sort, then delete the sort.

## Verdict and departures

Worth doing, and worth doing as one PR: each fix is small, independently testable, and they touch
overlapping lines in the same two files, so splitting them buys nothing but merge conflicts.

Two departures from the review that produced these findings:

- The review proposed calling `AllFeatures.validate()` "somewhere it is affordable" without saying
  where. There is no affordable place in the pipeline — validation needs a `collect()` on the
  largest frame in the system, which is exactly what `performance.md` exists to prevent. This plan
  enforces it at **test** time instead.
- The review also found three dead branches and a dangling `FLAW-001` label. Those are in scope
  here (~10 lines) but they are cleanup, not defects, and are listed separately below so they can
  be dropped without touching the rest.

## What changes, file by file

Order matters: **fix 1 must land before fix 2**, because `AllFeatures.validate()` cannot pass while
the metadata join is producing nulls.

### Fix 1 — `time_series_type` nulls in bulk mode

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:197`

Today `metadata_lf` is joined onto `power_lf` *before* the NWP join. Bulk mode then left-joins power
onto NWP (`_nwp.py:47-49`), so an NWP row with no matching power observation loses
`time_series_type` even though its `time_series_id` is known and the metadata is sitting in the
input.

Move the join: pass `power_lf` to `_join_nwp_bulk_mode` / `_join_nwp_single_run`, and join
`metadata_lf` onto the assembled frame immediately after, before `_apply_post_join_features`.
`time_series_id` is present on both sides in both modes, so this is a move, not a redesign. Rename
the `power_with_metadata` parameter on both join helpers accordingly.

Checked: nothing between the two points needs metadata. `STATIC_FEATURE_REGISTRY`
(`_parsed_features.py:20`) holds only `windchill`, which is weather-derived; no feature in the
module reads `time_series_type`.

Single-run mode is power-centric, so its output is unchanged. Bulk mode gains non-null values on
every row past the last observation.

### Fix 2 — nullability, then make `validate()` real

`packages/contracts/src/contracts/ml_schemas.py:86` and the `time_series_type` field.

Declare `power: float | None` and `time_series_type: str | None`. `power` is nullable by design —
live inference deliberately feeds an all-null spine past the last observation
(`_production_helpers.py:99-101`), and `XGBoostForecaster.train` drops those rows explicitly
(`forecaster.py:130`). `time_series_type` is nullable for a series absent from the metadata parquet.

Then add end-to-end tests, one per mode, that run the pipeline and call `AllFeatures.validate()` on
the collected result. This is where the primary-key uniqueness check becomes real: at test scale a
`collect()` costs nothing, and commits `75bafdf1` and `7ba598f5` both reason as if a fan-out
regression would be caught by something.

Do **not** call `validate()` inside `_engineer_features`.

### Fix 3 — the `collect()` probe

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:183`

Probe `nwp_lf` instead of `processed_nwp`. `ensemble_member` is one of
`_upsample_nwp_to_half_hourly`'s group-by keys, so the upsample can neither create nor destroy
control-member rows; the two checks are equivalent by construction. `SLICE` cannot push through the
sort and the window functions in the upsample, so today the guard executes the entire upsample of
the control member before answering.

### Fix 4 — `rolling_mean_by`, and the sort it lets us delete

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:319-339` and
`packages/ml_core/src/ml_core/features/_nwp.py:137-139`

Replace the `lf.rolling(...).agg(...)` + four-key join-back with:

```python
pl.col(base_col)
  .rolling_mean_by("valid_time", window_size=f"{window_hours}h", closed="right")
  .over(["time_series_id", "nwp_init_time", "ensemble_member"], order_by="valid_time")
```

Then delete the `.sort([*group_cols, "valid_time"])` at `_nwp.py:137-139`. That sort exists only to
satisfy `LazyFrame.rolling`'s sortedness precondition — the `interpolate()` and `forward_fill()`
calls immediately below it both carry their own `order_by`, so nothing else in the module depends
on it.

Beyond deleting a sort and a join, the window form closes a latent fan-out hazard: `rolling().agg()`
emits one row per input row, so if the frame ever held two rows sharing
`(time_series_id, nwp_init_time, ensemble_member, valid_time)` — which a second `nwp_model_id`
would produce, since `nwp_model_id` is an upsample group key but not a rolling group key — the
join-back would fan out quadratically.

Rewrite the docstring at `:325-330` while here. It claims single-run mode pads each group with
out-of-window rows whose weather is null, and that is not what happens: those rows carry a null
`ensemble_member`, so they form their own group rather than padding a real one. The conclusion (the
aggregation must be null-skipping, never row-count-dependent) stands; the stated mechanism does not.

### Cleanup, droppable without affecting the above

- `_lags.py:36-37` — the `has_ensemble` ternary. Dead since `75bafdf1` repointed the lag source at
  the raw observed-power frame, which by contract has no `ensemble_member`. Its own docstring three
  lines above already states this.
- `tabular_feature_engineer.py:313` — the `"power_fcst_init_time" in ...names()` guard. Both join
  helpers add the column unconditionally on all four of their sub-branches.
- `tabular_feature_engineer.py:298-302` — the `processed_nwp is None` raise.
  `requires_weather_data()` has already raised at `:166-167` for any weather lag with `nwp=None`.
- `_lags.py:128` — `FLAW-001` is a dangling label from a defunct review-numbering scheme.

## Design-philosophy check

All of this is R&D and training-path code, not the live serving path, so the fail-fast side of
`docs/design-philosophy/inherent-stability.md` applies: no degradation paths are added, and no
warning path is touched. No asset checks change.

Fix 2 makes a schema *more* permissive, which cuts against "strict about malformed inputs". The
trade is deliberate: null `power` is not malformed, it is the documented shape of an inference
spine, and declaring it non-nullable is what has kept the check switched off. Fix 1 and fix 4 both
reduce the number of concepts a reader has to hold, which is design principle 4's direction.

No principle in `design-principles.md` is traded away.

## Tests

| Fix | New or changed test | The assertion that fails on `main` today |
|---|---|---|
| 1 | Bulk mode with NWP extending past the last power observation | `time_series_type` is non-null on every output row — today it is null on all rows past the observation |
| 2 | One end-to-end test per mode calling `AllFeatures.validate()` | Bulk mode raises `DataFrameValidationError` on `power` and `time_series_type` missing values |
| 3 | None — see below | — |
| 4 | Rolling mean over a deliberately shuffled input | Current form raises `ComputeError: input data is not sorted`; window form returns correct values |
| 4 | Existing `test_apply_rolling_mean_feature*` and `test_cross_mode_equivalence` | Must stay green — they pin the values and the null-skipping invariant |

**Fix 3 gets no test, deliberately.** Its behaviour is already covered by
`test_engineer_features_raises_when_no_control_member_for_weather_lag`; what changes is plan shape,
and the only test that could pin it would assert where `SLICE` sits in `LazyFrame.explain()` output
— an implementation detail we would regret pinning. The measurement goes in the PR body instead.

## Docs to update

- `docs/architecture/performance.md:58` — names `_build_historical_weather`, which exists nowhere in
  the repo (deleted in `2805d950`), and describes the probe as checking "before building the lazy
  plan", which fix 3 is what makes true. Rewrite to name the real call site and the real behaviour.
- `docs/architecture/performance.md:66` — lists `group_by` + `explode` + `sort` + `interpolate` as
  the upsample's cost. Drop the `sort` once fix 4 removes it.

Nothing here completes a roadmap item, so no ship-time triage.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run --all-packages ty check
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pytest
```

Fix 4 additionally wants the cross-mode suite named explicitly, since it is the invariant most at
risk:

```bash
uv run pytest packages/ml_core/tests/test_cross_mode_equivalence.py -v
```

## Risks and open questions

**Removing the sort changes output row order.** XGBoost histogram construction sums in row order, so
a retrained model may not be bit-identical to one trained before this change. Per CLAUDE.md that
costs a retrain rather than a migration, so I judge it acceptable — but it should be known in
advance rather than discovered as an unexplained leaderboard diff. *Recommendation:* proceed, and
say so in the PR body.

**Fix 2 widens two schema fields, and `AllFeatures` is a published contract.** Nothing outside this
repo consumes it and no trained model encodes it, so this is free today. *Recommendation:* proceed.

**Open question for Jack:** should `time_series_type` being null be an error rather than a widened
type? A series in the power table but absent from the metadata parquet is arguably malformed input,
which the inherent-stability rules would reject at the contract boundary rather than tolerate.
*Recommendation:* widen it now, since fix 1 removes the only common cause of the null, and file the
stricter check separately if it is wanted — making it an error in the same PR would turn a latent
null into a new fail-fast path with no evidence about how often it fires.
