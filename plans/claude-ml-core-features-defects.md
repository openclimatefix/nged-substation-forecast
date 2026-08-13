# Fix four defects in `ml_core.features`, and move one check to where it runs

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

**The solution.** Six independent fixes in one PR. Move the metadata join off the power frame and
onto the assembled frame, so `time_series_type` is populated in both modes. Delete the
`AllFeatures.validate()` override, which nothing calls and which could only ever run behind a
`collect()` this module exists to avoid, and widen the two fields whose declarations state something
false. Probe the raw NWP frame rather
than the upsampled one for the control member. And replace `rolling().agg()` + join-back with
`rolling_mean_by(...).over(...)`, which needs neither the join nor the sort, then delete the sort —
which is where nearly all of the measured win sits. Then put the primary-key uniqueness check on
`PowerForecast`, where `validate()` really runs, and stop emitting `time_series_type` on every
frame that never asked for it.

## Verdict and departures

Worth doing, and worth doing as one PR: each fix is small, independently testable, and they touch
overlapping lines in the same two files, so splitting them buys nothing but merge conflicts.

Fixes 5 and 6 were written up as open questions for Jack in the first draft of this plan. He has
since decided both: fold them in.

Departures from the clean-room review that produced these findings:

- The review proposed calling `AllFeatures.validate()` "somewhere it is affordable". There is no
  such place — validating `AllFeatures` means collecting the largest frame in the system, and
  `docs/architecture/code-style.md:178` puts `.validate()` at persistence edges only. It gets
  deleted instead; see fix 2.
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

The six fixes are independent and can land in any order.

### Fix 1 — `time_series_type` nulls in bulk mode

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:197`

Today `metadata_lf` is joined onto `power_lf` *before* the NWP join. Bulk mode then left-joins power
onto NWP (`_nwp.py:45-47`), so an NWP row with no matching power observation loses
`time_series_type` even though its `time_series_id` is known and the metadata is sitting in the
input. Measured: 20 of 22 rows null, with NWP running 8 hours past the last power observation.

Move the join: pass `power_lf` to `_join_nwp_bulk_mode` / `_join_nwp_single_run`, and join
`metadata_lf` onto the assembled frame immediately after, before `_apply_post_join_features`.
`time_series_id` is present on both sides in both modes, so this is a move, not a redesign. Rename
the `power_with_metadata` parameter on both join helpers, **and update the docstrings that describe
the frame they receive** — both helpers' docstrings in `_nwp.py` and `_apply_post_join_features`'s
`raw_data` argument doc at `tabular_feature_engineer.py:273-274` ("The power-with-metadata frame
already joined to NWP").

No new fan-out: `TimeSeriesMetadata.time_series_id` is `unique=True` (`power_schemas.py:175`). No
column collision: after `_attach_nearest_nwp_cell` the NWP frame shares no column name with
`TimeSeriesMetadata`, and `h3_res_5` is consumed by the cell join and never reaches the pipeline.

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

Delete the override (64 lines, including the `Sequence` and `Self` imports at `ml_schemas.py:7,9`
that go dead with it). Nothing calls it outside `packages/contracts/tests/test_ml_schemas.py`.

The reason is **where validation is affordable, not whether this one can pass.** It can: with the
two widenings below and `allow_superfluous_columns=True`, it validates real bulk-mode output.
But:

- **Validating `AllFeatures` means a `collect()` on the largest frame in the system.** The module
  is deliberately lazy end-to-end, and `docs/architecture/code-style.md:178` puts `.validate()` at
  persistence edges only. `AllFeatures` is not a persistence edge — it is the in-memory hand-off to
  the model.
- **Half the override is unusable for this model by construction.** Every dynamic feature —
  `power_lag_24h`, `temperature_2m_rolling_mean_6h` — is deliberately not declared as a Patito
  field (`ml_schemas.py:61-64`), so any call must pass `allow_superfluous_columns=True`. What
  survives is a dtype check and the primary-key check.

Keep the primary-key check, as an assertion in the existing cross-mode bulk test where a
`collect()` is free. **That assertion is a real guard, not a formality**, because fix 4 removes the
only thing that surfaces a fan-out today: the existing `len(bulk) == 3*2*12` row-count assertion
catches a duplicate primary key only because `rolling().agg()`'s join-back amplifies it into a
visible explosion. After fix 4 a duplicate is absorbed silently. See "Risks".

**Separately, widen the two fields whose declarations state something false**: `power: float | None`
(`ml_schemas.py:86`) and `time_series_type: str | None`. Live inference deliberately feeds an
all-null power spine past the last observation (`_production_helpers.py:99-100`) and
`XGBoostForecaster.train` drops those rows explicitly (`forecaster.py:130`). `contracts` is the
single source of truth for data shapes and currently misdescribes the data. Do **not** widen
`PowerTimeSeries.power` (`power_schemas.py:42`): that model's `validate()` genuinely runs on
ingested data (`nged_data/storage.py:211`), where a null power *is* malformed.

Say explicitly what happens to `packages/contracts/tests/test_ml_schemas.py`: all three `validate()`
call sites stay green, but `test_all_features_validation` becomes a plain-Patito check rather than
an exercise of the override.

The primary-key check itself does not disappear — it moves to `PowerForecast` in fix 5.

### Fix 3 — the `collect()` probe

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:183`

Probe `nwp_lf` instead of `processed_nwp`, **and change the guard's first clause from
`processed_nwp is not None` to `nwp_lf is not None`.** The two are equivalent by construction
(`processed_nwp is None` iff `nwp_lf is None`, `:169-178`), and without it `ty check` fails with
`unresolved-attribute: Attribute 'filter' is not defined on 'None' in union 'LazyFrame | None'` at
`:183` — `nwp_lf` is `pl.LazyFrame | None` at `:163` and the existing narrowing does not reach it.
Verified: with the clause swapped, `ty check` passes.

`ensemble_member` is one of
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
valid_time)` would fan out quadratically (verified: 4 rows become 8 under the current form, stay 4
under the window form). One clause, not a paragraph — a second `nwp_model_id` would already fan out
the bulk join before the rolling ever ran.

**But that hazard is also a detector, and this removes it.** A duplicate primary key today makes the
frame visibly explode; after fix 4 it is absorbed silently. That is why fix 2 keeps the primary-key
assertion, and it is the strongest argument for the `PowerForecast` follow-up.

**One genuine behaviour change, unobservable in practice.** Where `ensemble_member` is null and
weather is non-null, the two forms differ: a null join key never matches in the `how="left"`
join-back, so the current form yields null while the window form computes a real value. In the real
pipeline a null `ensemble_member` means the single-run join missed, so the weather is null too and
both forms give null. Say this in the PR so it is not discovered later as an unexplained diff.

Rewrite the docstring at `:325-330` while here. It claims single-run mode pads each group with
out-of-window rows whose weather is null, and that is not what happens: those rows carry a null
`ensemble_member`, so they form their own group rather than padding a real one. The conclusion (the
aggregation must be null-skipping, never row-count-dependent) stands; the stated mechanism does not.

### Fix 5 — put the primary-key uniqueness check where `validate()` actually runs

`packages/contracts/src/contracts/power_schemas.py:296`

Fix 2 deletes a uniqueness check that has never executed. Fix 4 removes the row-count explosion that
made a fan-out visible without it. Between them, nothing detects a duplicate primary key on the
forecast path at all — so add the check to `PowerForecast`, whose `validate()` **is** called on
every predict (`forecaster.py:203`) and on every CV fold (`cv_assets.py:851`), and which today has
no uniqueness check.

`PowerForecast`'s primary key is `(time_series_id, power_fcst_init_time, valid_time,
ensemble_member)`. `ensemble_member` is non-nullable and always present here
(`power_schemas.py:321-323`), so the override is simpler than the one being deleted: no conditional
key assembly, just `is_duplicated().any()` on four columns of an already-collected frame.

**This adds a production raise path, which is the point to weigh.** The inherent-stability rules
reserve raising for states that are our own bug rather than the outside
world misbehaving, and a duplicated forecast primary key is exactly that: it means the join fanned
out, which no absent or stale input can cause. `PowerForecast.validate()` already raises in
production on the `valid_time > power_fcst_init_time` constraint (`power_schemas.py:308`), so this
adds a second reason to fail, not a first.

Measure `is_duplicated()` on a predict-sized frame during implementation and put the number in the
PR body; if it is not cheap, that changes the decision and I will say so rather than ship it
quietly.

### Fix 6 — stop emitting `time_series_type` on frames that never asked for it

`packages/ml_core/src/ml_core/features/tabular_feature_engineer.py:249`

`time_series_type` sits in `_select_output_columns`' unconditional `base_cols`, so every frame
carries an Enum column whether or not anything requested it. Drop it from `base_cols`. It stays
requestable — it is a `SafeInputBaseColumn` (`ml_schemas.py:37-44`) and `ParsedFeatures` already
routes it through `base_features` (`test_features.py:580-583`), so a config that asks for it still
gets it, now with fix 1's non-null values.

Roughly 460 MB on the 116M-row predict chunk `performance.md` sizes at a 9 GB peak. With fix 1
having moved the metadata join downstream, an unrequested `time_series_type` also lets projection
pushdown drop the metadata join entirely.

`AllFeatures.time_series_type` therefore becomes `allow_missing=True` as well as `str | None`: the
column is now optional in the output, and `contracts` should say so.

Nothing outside the module reads it off `AllFeatures`. `metrics` joins it from metadata itself
(`metrics.py:420-423`), `_build_part` (`forecaster.py:165-181`) does not select it, and the two
dashboards read it from the metadata parquet.

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

Fixes 1–4 and 6 are R&D and training-path code, not the live serving path, so the fail-fast side of
`docs/design-philosophy/inherent-stability.md` applies: no degradation paths are added, and no
warning path is touched. No asset checks change.

Fix 5 is the exception: it adds a raise to a `validate()` that runs in live inference. That is
inside the rules rather than against them — raising is reserved for our own bugs, and a duplicated
forecast primary key can only come from a join fanning out, never from an absent or stale input. The
same `validate()` already raises there on the hindcast-row constraint.

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
| 2 | `AllFeatures.validate()` on a one-row frame with null `power`, in `test_ml_schemas.py` | Raises `1 missing values` today; passes after the widening. This is the guard for the half of fix 2 that changes a contract |
| 2 | One primary-key uniqueness assertion added to the existing cross-mode bulk test | None: it passes today. It replaces the fan-out detector fix 4 removes, so it guards against a regression that only becomes possible in this PR |
| 3 | None — see below | — |
| 4 | Parametrise the existing `test_apply_rolling_mean_feature` with a reversed input | Current form raises `ComputeError: input data is not sorted`; window form returns the values already pinned there. Use `df.reverse()`, **not** `sample(shuffle=True, seed=…)` — a random shuffle of four rows can land sorted, and one did |
| 4 | Existing `test_apply_rolling_mean_feature_partitions_by_group` and `test_cross_mode_equivalence` | Must stay green — they pin the values and the null-skipping invariant |
| 5 | `PowerForecast.validate()` on a frame with one duplicated `(time_series_id, power_fcst_init_time, valid_time, ensemble_member)` | Passes today; raises after. Plus a near-miss row differing only in `ensemble_member`, which must still validate |
| 6 | The fix-1 test requests `time_series_type` explicitly; add one asserting it is absent when unrequested | Today it is present unrequested, so the absence assertion fails on `main` |

**Fix 3 gets no test, deliberately.** Its behaviour is already covered by
`test_engineer_features_raises_when_no_control_member_for_weather_lag`; what changes is plan shape,
and the only test that could pin it would assert where `SLICE` sits in `LazyFrame.explain()` output
— an implementation detail we would regret pinning. The measurement goes in the PR body instead.

## Docs to update

- `docs/architecture/performance.md:58` — names `_build_historical_weather`, which exists nowhere in
  the repo (deleted in `2805d950`). Rewrite to name the real call site and the real behaviour, and
  **do not reuse the phrase "before building the lazy plan"**: even after fix 3 the probe collects
  partway through plan construction. What changes is that it no longer executes the upsample.
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
outside this repo consumes it and no trained model encodes it, so this is free today. Fix 5 puts the
uniqueness check back where it runs, so deleting the dead check and adding a live one are two halves
of one argument in one PR.

**Fix 5 costs a `is_duplicated()` on every predict.** No extra `collect()`, but not free either.
*Recommendation:* measure it on a predict-sized frame during implementation, put the number in the
PR body, and if it is expensive enough to matter, say so rather than shipping it silently.

**Open question for Jack, still open and deliberately not fixed here:** should `time_series_type`
being null be an error rather than a widened type? A series in the power table but absent from the
metadata parquet is arguably malformed input, which the inherent-stability rules would reject at the
contract boundary rather than tolerate. *Recommendation:* widen it now, since fix 1 removes the only
common cause of the null, and file the stricter check separately if it is wanted — making it an
error in the same PR would turn a latent null into a new fail-fast path with no evidence about how
often it fires.

**Fix 1 moves a join from the small side to the large one.** Metadata × power becomes metadata ×
NWP-joined, on the production path. The metadata columns are already replicated across every NWP row
today, so it should be close to a wash — but this plan measures fix 3 and fix 4 to the millisecond
and should not stay silent about the one fix that touches the largest join.
*Recommendation:* measure it during implementation and put the number in the PR body.

## What the two adversarial reviews changed

Recorded so the reasoning survives into the PR.

### Review 1 — simplicity

**Adopted.** Fix 2 was rewritten from "make `validate()` real" to "delete the override": it has no
callers, and calling it needs a `collect()` this module exists to avoid. Also adopted: the `assert`
in the cleanup group (deleting the branch fails `ty check`); parametrising the existing rolling test
instead of adding one; leading fix 4 with the sort measurement rather than the rolling rewrite;
recording that two of the four defects are unreached by any config here; and the note that the two
rolling forms genuinely differ on a null `ensemble_member`.

**Rejected.** The reviewer implied the two nullability widenings become unnecessary once
`validate()` is deleted. They are kept: `contracts` is the single source of truth for data shapes,
and a field declared non-nullable that the pipeline routinely nulls is a false statement regardless
of whether anything validates against it.

**Deferred, then adopted.** Dropping `time_series_type` from `base_cols` and the `PowerForecast`
primary-key check both came from this reviewer. They were put to Jack as open questions rather than
folded in silently, because one is an optimisation and the other adds a production raise path. He
decided both in; they are fixes 6 and 5.

### Review 2 — correctness and testability

The second reviewer found that **three of the claims this plan had just taken from the first
reviewer were false.** All three are corrected above; all three were confirmed against the code
before correcting.

- **`AllFeatures.validate()` can pass on real output** — with the two widenings and
  `allow_superfluous_columns=True`, it validates real bulk-mode output. The first review's "cannot
  pass" rested on a probe whose feature set omitted `local_day_of_week`. The deletion still stands,
  but on affordability, not impossibility.
- **`conf/model/xgboost.yaml:29` does request `local_day_of_week`** — as does
  `scripts/run_baseline_experiment.py:58`. The `Missing column` failure fires for no config here.
- **Neither `75bafdf1` nor `7ba598f5` mentions `validate()`, a backstop or uniqueness.** That
  provenance came from the clean-room report and this plan repeated it twice. What `7ba598f5`
  actually argues is that the rolling join-back is 1:1 on the full primary key. Claim dropped.

**Also adopted:** fix 3 needs its guard clause swapped or `ty check` fails (reproduced here); the
widening ships with a test that fails first; `df.reverse()` rather than a seeded shuffle; the
detector-removal consequence of fix 4; fix 1's stale docstrings and its move of a join to the large
side; the dead `Sequence`/`Self` imports; `test_ml_schemas.py`'s disposition; and two line
references that were off by two.

**Nothing rejected.** Every finding checked out.
