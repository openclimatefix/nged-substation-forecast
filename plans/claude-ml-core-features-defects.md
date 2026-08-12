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

**The solution.** Four independent fixes in one PR. Move the metadata join off the power frame and
onto the assembled frame, so `time_series_type` is populated in both modes. Delete the
`AllFeatures.validate()` override, which cannot pass on any real output and which nothing calls,
and widen the two fields whose declarations state something false. Probe the raw NWP frame rather
than the upsampled one for the control member. And replace `rolling().agg()` + join-back with
`rolling_mean_by(...).over(...)`, which needs neither the join nor the sort, then delete the sort —
which is where nearly all of the measured win sits.

## Verdict and departures

Worth doing, and worth doing as one PR: each fix is small, independently testable, and they touch
overlapping lines in the same two files, so splitting them buys nothing but merge conflicts.

Departures from the clean-room review that produced these findings:

- The review proposed calling `AllFeatures.validate()` "somewhere it is affordable". There is no
  such place, and the check cannot pass anywhere — see fix 2. It gets deleted instead.
- The review also found three dead branches and a dangling `FLAW-001` label. Those are in scope
  here (~10 lines) but they are cleanup, not defects, and are listed separately below so they can
  be dropped without touching the rest.

**Two of the four defects are unreachable by any config in this repo.** `conf/model/xgboost.yaml:18-42`
and `scripts/run_baseline_experiment.py:50` are the only feature sets here, and neither requests a
weather lag or a rolling mean. So fix 3's guard never executes today, and fix 4's *rolling rewrite*
is likewise unreached — but fix 4's *sort deletion* is on the production path and is where the
measured win is. Both fixes are one-liners and still worth taking; the PR body must not imply
production pays for either today.

## What changes, file by file

The four fixes are independent and can land in any order.

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

**What this defect actually is:** train/serve skew on a requestable column, not a live bug. No
consumer of `AllFeatures` reads `time_series_type` today — `metrics` joins it from metadata itself
(`metrics.py:420-423`) and `_build_part` (`forecaster.py:165-181`) does not select it. But it is a
`SafeInputBaseColumn` (`ml_schemas.py:37-44`), so a model that requested it would get non-null
values at train time and nulls at bulk-predict time. Frame it that way in the PR.

### Fix 2 — delete the `AllFeatures.validate()` override

`packages/contracts/src/contracts/ml_schemas.py:130-191`

The override cannot pass on any real pipeline output, and no amount of fixing the *data* changes
that, because two of its four failure modes are structural:

- **`local_day_of_week` (`ml_schemas.py:124`) is the one time feature declared without
  `allow_missing=True`.** Any config that does not request it fails with `Missing column` —
  including `conf/model/xgboost.yaml`, which requests its `_sin`/`_cos` siblings only.
- **Every dynamic feature is "superfluous".** `power_lag_24h`,
  `temperature_2m_rolling_mean_6h` and the rest are deliberately not declared as Patito fields —
  the `AllFeatures` docstring says so at `ml_schemas.py:61-64`. So the override could only ever be
  called with `allow_superfluous_columns=True`, which switches off the column-set check entirely
  and leaves a dtype check plus the primary-key check.

Delete the override (62 lines). Nothing calls it outside
`packages/contracts/tests/test_ml_schemas.py`, so the fan-out backstop that commits `75bafdf1` and
`7ba598f5` both reason about has never existed. Replace it with a primary-key uniqueness assertion
in the existing cross-mode test, where a `collect()` is free.

**Separately, widen the two fields whose declarations state something false**: `power: float | None`
(`ml_schemas.py:86`) and `time_series_type: str | None`. Live inference deliberately feeds an
all-null power spine past the last observation (`_production_helpers.py:99-101`) and
`XGBoostForecaster.train` drops those rows explicitly (`forecaster.py:130`). This is no longer
needed to make `validate()` pass — it is needed because `contracts` is the single source of truth
for data shapes and currently misdescribes the data. Do **not** widen `PowerTimeSeries.power`
(`power_schemas.py:41`): that model's `validate()` genuinely runs on ingested data
(`nged_data/storage.py:211`), where a null power *is* malformed.

**Follow-up for Jack, not this PR:** the primary-key check belongs on `PowerForecast`, whose
`validate()` is called for real on every predict (`forecaster.py:203`) and which today has no
uniqueness check at all — so a fan-out reaches the `power_forecasts` Delta table undetected. It runs
on an already-collected frame, so it adds no `collect()`. That is a new production raise path and a
different change from these four fixes; it should be its own issue.

### Fix 3 — the `collect()` probe

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:183`

Probe `nwp_lf` instead of `processed_nwp`. `ensemble_member` is one of
`_upsample_nwp_to_half_hourly`'s group-by keys, so the upsample can neither create nor destroy
control-member rows; the two checks are equivalent by construction. `SLICE` cannot push through the
sort and the window functions in the upsample, so today the guard executes the entire upsample of
the control member before answering. Measured on a 2.63M-row raw NWP fixture: **41 ms probing the
upsampled frame, 3 ms probing the raw one**, and the gap grows with the window. Unreached by every
config in the repo, so this is latent cost, not cost being paid.

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

**The sort is the point.** The rolling rewrite is unreached by every config here; the sort runs
unconditionally, on the production path, and is a full materialising barrier in the streaming engine
that `train` and `predict` both use. Measured on the same fixture upsampled to 15.5M rows with 4
weather columns: **2.47–2.55 s with the sort, 1.71–1.81 s without — about 30% of the upsample.**
Real NWP carries 13 weather variables and far more rows. Lead the PR body with this number, not with
the rolling rewrite.

The window form also closes a latent fan-out hazard the join-back has: `rolling().agg()` emits one
row per input row, so two rows sharing `(time_series_id, nwp_init_time, ensemble_member,
valid_time)` would fan out quadratically. One clause, not a paragraph — a second `nwp_model_id`
would already fan out the bulk join before the rolling ever ran.

**One genuine behaviour change, unobservable in practice.** Where `ensemble_member` is null and
weather is non-null, the two forms differ: a null join key never matches in the `how="left"`
join-back, so the current form yields null while the window form computes a real value. In the real
pipeline a null `ensemble_member` means the single-run join missed, so the weather is null too and
both forms give null. Say this in the PR so it is not discovered later as an unexplained diff.

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
  **Replace it with `assert processed_nwp is not None`, matching the assert on the next line** —
  deleting it outright fails `ty check` with `invalid-argument-type` at `:300`, because the
  narrowing is load-bearing even though the branch is not. Net −4 lines, not −5.
- `_lags.py:128` — `FLAW-001` is a dangling label from a defunct review-numbering scheme.

## Design-philosophy check

All of this is R&D and training-path code, not the live serving path, so the fail-fast side of
`docs/design-philosophy/inherent-stability.md` applies: no degradation paths are added, and no
warning path is touched. No asset checks change.

Fix 2 makes a schema *more* permissive, which cuts against "strict about malformed inputs". The
trade is deliberate: null `power` is not malformed, it is the documented shape of an inference
spine, and the declaration is what is wrong. Fix 2 also *removes* a check — but a check that has
never run and cannot run is not protection, and leaving it in place is what let two commits reason
as though the module had a fan-out backstop. Fix 1 and fix 4 both reduce the number of concepts a
reader has to hold, which is design principle 4's direction.

No principle in `design-principles.md` is traded away.

## Tests

| Fix | New or changed test | The assertion that fails on `main` today |
|---|---|---|
| 1 | Bulk mode with NWP extending past the last power observation | `time_series_type` is non-null on every output row — today it is null on all rows past the observation |
| 2 | One primary-key uniqueness assertion added to the existing cross-mode bulk test | None: it passes today. It is a regression guard replacing a deleted one, and the plan says so rather than pretending otherwise |
| 3 | None — see below | — |
| 4 | Parametrise the existing `test_apply_rolling_mean_feature` with `shuffle=True` | Current form raises `ComputeError: input data is not sorted` on shuffled input; window form returns the values already pinned there |
| 4 | Existing `test_apply_rolling_mean_feature_partitions_by_group` and `test_cross_mode_equivalence` | Must stay green — they pin the values and the null-skipping invariant |

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

**Fix 2 deletes a check and widens two schema fields.** `AllFeatures` is a contract, but nothing
outside this repo consumes it and no trained model encodes it, so this is free today.
*Recommendation:* proceed, and file the `PowerForecast` primary-key check as its own issue in the
same breath, so deleting the dead check and adding a live one are visibly two halves of one
argument.

**Open question for Jack:** should `time_series_type` being null be an error rather than a widened
type? A series in the power table but absent from the metadata parquet is arguably malformed input,
which the inherent-stability rules would reject at the contract boundary rather than tolerate.
*Recommendation:* widen it now, since fix 1 removes the only common cause of the null, and file the
stricter check separately if it is wanted — making it an error in the same PR would turn a latent
null into a new fail-fast path with no evidence about how often it fires.

**Second open question:** `time_series_type` sits in `_select_output_columns`' unconditional
`base_cols` (`tabular_feature_engineer.py:249`), so every frame carries it whether or not anything
asked for it. Removing it from `base_cols` keeps all 627 tests green and saves an Enum column —
roughly 460 MB on the 116M-row predict chunk `performance.md` sizes at a 9 GB peak. That is a real
win but it is an optimisation, not a defect fix, and it is orthogonal to fix 1 (a *requested*
`time_series_type` still needs the join moved). *Recommendation:* keep it out of this PR; file it.

## What the first adversarial review changed

Recorded so the reasoning survives into the PR.

**Adopted.** Fix 2 was rewritten wholesale: the reviewer proved `AllFeatures.validate()` fails on
four things, not the two this plan claimed, and that two of them (`local_day_of_week` without
`allow_missing`, dynamic features being structurally superfluous) cannot be fixed by fixing the
data. I confirmed both against `ml_schemas.py:61-64` and `:124` — the original fix 2 would not have
worked. Also adopted: the `assert` in the cleanup group (deleting the branch fails `ty check`);
parametrising the existing rolling test instead of adding one; leading fix 4 with the sort
measurement rather than the rolling rewrite; recording that two of the four defects are unreached by
any config here; and the note that the two rolling forms genuinely differ on a null
`ensemble_member`.

**Rejected.** The reviewer implied the two nullability widenings become unnecessary once
`validate()` is deleted. They are kept: `contracts` is the single source of truth for data shapes,
and a field declared non-nullable that the pipeline routinely nulls is a false statement regardless
of whether anything validates against it.

**Deferred, not rejected.** Dropping `time_series_type` from `base_cols`, and the `PowerForecast`
primary-key check — both good, both larger than a defect fix, both above as open questions for Jack.
