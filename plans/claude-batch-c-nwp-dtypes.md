# `Nwp` contract dtype change

## What's wrong

Four `Nwp` fields declare a dtype that does not match what actually ends up on disk, and one of
them is outright unwritable for part of its documented range.

`ensemble_member` (`UInt8`), `h3_index` (`UInt64`) and `nwp_model_id` (`Enum`) are declared as
types Delta Lake cannot store. `ensemble_member`/`h3_index` land on disk as `int8`/`int64` anyway,
because delta-rs has no unsigned integer type and silently narrows on write; `nwp_model_id` lands
as `string`, because `write_nwp` explicitly casts it (Delta cannot store dictionary-encoded/`Enum`
columns at all). `Nwp.scan_delta` then has to cast every one of those columns back to its declared
(wrong) dtype before handing the frame to a caller, and that cast sits between the scan and any
`.filter()` a caller applies. Confirmed directly against this repo's `write_nwp` + `Nwp.scan_delta`
(evidence below): that cast defeats Parquet predicate pushdown, so a real production caller today —
`load_engineering_inputs`'s `ensemble_member`/`h3_index` filters, used by every training and
prediction run — reads every partition in full instead of skipping row groups, silently, with no
error and no warning.

`categorical_precipitation_type_surface` is worse: it is declared `UInt8`, but its own field
description names `255` ("Missing") as a legitimate value, and `UInt8` values ≥128 do not survive
delta-rs's write path — confirmed directly: `write_deltalake` raises `Cast error: Can't cast value
200 to type Int8` for any such value today. Any upstream ECMWF run that actually emits a "missing"
ptype code cannot be ingested.

## What this plan does

Change all four fields' declared dtype to match what's already physically storable, add back the
validation that a signed-int-only Delta model would otherwise drop silently, and fix the write path
so the one field with a genuine physical widening (`categorical_precipitation_type_surface`,
`UInt8`→`Int16`) lands without a table rewrite. A same-file inconsistency in `Nwp._check_unique`
(it checks fewer columns than the class's own documented primary key) gets folded in. Three
docs/docstrings currently claim a pushdown speed-up that isn't happening get corrected, with the
exact numbers deferred to a follow-up measurement that needs the maintainer's go-ahead before it
runs (it reads real production NWP data).

**No data rewrite is needed.** Verified empirically against `deltalake` 1.6.2 (this repo's pinned
version) on a local scratch table: writing the widened dtype for one partition with
`schema_mode="overwrite"` updates only the table's *logical* schema; existing partitions' physical
Parquet bytes are untouched, and both `pl.read_delta` and `pl.scan_delta` — including a scan
*filtered down to only the old, physically-narrower partition* — read the whole table back
correctly and losslessly at the new logical dtype, with no error. See "Question 1" below for the
full reproduction.

## Verdict, size and departures

**Verdict**: worth doing. The pushdown regression is live today in a real, hot production code path
(`load_engineering_inputs`, used by every CV fold and every live forecast), and the
`categorical_precipitation_type_surface` write bug means the pipeline cannot currently ingest a run
that emits ECMWF's own documented "missing" ptype code.

**Size: complex**, per the calling instructions — it changes a Patito contract and what is stored on
disk. Contract sign-off already obtained (2026-08-14, "Go ahead with the dtype change" — see
`packages/contracts/README.md`'s "get the change agreed before making it" rule). This plan does not
itself launch the adversarial plan reviews or diff reviews `plan-issue`/`implement-issue` call for;
that is for whoever picks the approved plan up next.

**Departures from a naive "declare the right dtype" fix**:

- `nwp_model_id`'s vocabulary validation is **kept**, not dropped, via a Patito `constraints=`
  field (see "nwp_model_id vocabulary" below) — the notes this plan was seeded from flagged this as
  the one trade-off a one-line fix would silently lose.
- `write_nwp`'s `nwp_model_id` cast-and-model-strip workaround (its longest comment block) is
  **deleted**, not merely retargeted — once every field's declared dtype is already
  delta-rs-storable, nothing needs stripping or casting before `.to_arrow()`. See "File-by-file"
  below.
- The docs/docstrings naming a specific pushdown speed-up (≈50×, ~5×, ~2%) are **not** simply
  restored to their old wording. The mechanism is verified locally (below), but the specific numbers
  were measured before this regression existed and need re-measuring against real data — which is a
  large-read operation this plan explicitly does not run. See "Docs to update".

## The three questions, answered

### Question 1 — does the write path widen `int8`→`int16` in place, or reject the mismatch?

**Neither, by default — and there's a third option that's better than both.** Reproduced directly
against `delta_store.nwp.write_nwp`'s exact call shape (`mode="overwrite"`, a `predicate` scoping
one partition, `partition_by=["nwp_model_id", "init_time"]`) on a local scratch table:

- **Default (`schema_mode=None`)**: writing `Int16` data into a partition whose table schema still
  says `Int8` fails: `DeltaError: ... Cast error: Can't cast value 200 to type Int8`. delta-rs
  conforms incoming data *down* to the existing table schema, not the other way round.
- **`schema_mode="merge"`**: same failure. `merge` only adds new *columns*; it does not widen an
  existing column's type.
- **`schema_mode="overwrite"`**: succeeds, and updates only the table's *logical* schema (the
  `_delta_log` metadata) to the incoming frame's (wider) type. Confirmed this is metadata-only, not
  a rewrite: after writing one new partition this way, the *other*, previously-written partition's
  Parquet files on disk are byte-for-byte unchanged (`int8`/`byte` physical Arrow schema, checked
  with `pyarrow.parquet.ParquetFile(...).schema_arrow`).
- **Reading the resulting mixed-physical-type table**: `pl.read_delta`, `pl.scan_delta(...).collect()`,
  and — critically — `pl.scan_delta(...).filter(init_time == <the old, physically-int8
  partition>).collect()` all return correct `Int16`-typed data with no error. delta-rs's Parquet
  reader promotes each file's physical type to the table's current logical type on read; it does not
  require every file to already agree.

So the fix is: `write_nwp` passes `schema_mode="overwrite"` on every write. Since its input is
always a `pt.DataFrame[Nwp]` — already cast to the *current* contract's dtypes by `Nwp.validate()`
before it ever reaches `write_nwp` — the only way `schema_mode="overwrite"` can ever change the
table's logical schema is a genuine, deliberate contract dtype change like this one; it can never
silently drop a column, because the input always carries the model's full column set. The very next
`ecmwf_ens` asset run after this change ships performs the one-time schema widening automatically,
as a side effect of writing that day's new partition — no manual migration step, no rewrite, no
downtime.

### Question 2 — if a rewrite is needed, what is its shape and cost?

**Not needed at all**, per Question 1. This supersedes the original notes' framing (which
anticipated needing to weigh a targeted-partition rewrite against a whole-table one); the answer
turned out to be "neither," once the third `schema_mode` option was tested directly.

One consequence worth the maintainer's attention rather than assuming: because `write_nwp` already
**rejects** any `categorical_precipitation_type_surface` value ≥128 today (same reproduction as
above, with `UInt8` in place of `Int16` — `Cast error: Can't cast value 128/255 to type Int8`), it
follows that **no partition currently on disk can contain such a value** — any `ecmwf_ens` run that
tried would have failed to write, not silently corrupted. That's good news for correctness (every
already-stored `int8` value is trivially representable as `int16`, so the schema widening is
lossless by construction), but it also means this bug may have been silently failing/retrying real
ingests whenever Dynamical.org's `ptype` field actually emitted a documented "missing" code. **Open
question for the maintainer**: worth checking Dagster's run history for `ecmwf_ens` for a retry or
failure pattern that stopped once a particular run happened not to contain ptype 255 — that would
confirm whether this has been silently costing real ingest attempts.

### Question 3 — should `Nwp._check_unique` gain `nwp_model_id`?

**Yes, folded in.** `Nwp`'s class docstring already documents the primary key as `(nwp_model_id,
init_time, valid_time, ensemble_member, h3_index)`, but `_check_unique`
(`packages/contracts/src/contracts/weather_schemas.py:411-419`) only checks the last four. Checking
*fewer* columns is stricter, not more lenient: today it would reject two legitimate rows that agree
on `(init_time, valid_time, ensemble_member, h3_index)` but differ in `nwp_model_id` — exactly the
row shape a second ingested NWP model produces. Not reachable today (only one `NwpModelId` value
exists), but low-cost to fix now, already touching this file, and closes the gap before issue #114
(per-model H3 resolution) or any other multi-model work makes it live. No trade-off to record; this
is a straight bug-vs-docstring fix.

## nwp_model_id vocabulary — kept, not dropped

`NwpModelId` (a `StrEnum`, currently one member: `ECMWF_ENS_0_25_degree`) stays as the single source
of truth for the vocabulary. The field becomes:

```python
nwp_model_id: str = pt.Field(
    dtype=pl.String,
    constraints=pl.col("nwp_model_id").is_in([model.name for model in NwpModelId]),
    description="Which NWP model produced this row (e.g. 'ECMWF_ENS_0_25_degree').",
)
```

This follows the one existing precedent for a Patito `constraints=` field in this codebase
(`PowerForecast.valid_time`'s `valid_time > power_fcst_init_time` check), rather than inventing a
new pattern. Tradeoff to record explicitly, per the `polars-patito-gotchas` skill's own guidance:
a `constraints=` violation reports the generic "N rows do not match custom constraints", not which
value was invalid or what the valid set is. Given the codebase's existing precedent accepts the same
tradeoff for the `valid_time` constraint, and this field's vocabulary is small and rarely wrong (one
value today), this plan keeps the plain `constraints=` form rather than adding a custom
`_check_nwp_model_id_vocabulary` classmethod with a friendlier message — but that's a defensible
alternative if a reviewer wants a clearer failure message.

Why the vocabulary check is *not* dropped, unlike `PowerForecast.power_fcst_model_name` /
`experiment_name` / `fold_id` (all `String`, no vocabulary constraint): those are genuinely
open-ended, config-driven labels (`experiment_name` is free text; `fold_id` is whatever
`conf/cv/default.yaml` defines). `nwp_model_id` is the opposite — a small, closed set of NWP models
we actually know how to ingest, functionally a real enum that happens to need `String` as its
on-disk physical type. Dropping the check would let a typo'd or unrecognised model id land silently
in the `nwp` table with no ingest-time signal.

**Why `nwp_model_id` needed fixing at all, beyond the pushdown argument already covered by the
other three fields**: reproduced separately that filtering on `nwp_model_id` *itself* — a Delta
*partition* column — does not prune partitions today, for the identical reason (the `Enum` cast
sits between scan and filter). No production caller filters on it yet, but it is the table's other
partition key alongside `init_time`, and a future multi-model caller filtering on it would silently
read every model's data instead of pruning to one.

## File-by-file

### `packages/contracts/src/contracts/weather_schemas.py`

- `nwp_model_id` field: `dtype=pl.String` with the `constraints=` shown above (was
  `dtype=NWP_MODEL_ID_DTYPE`).
- `ensemble_member` field: `dtype=pl.Int8` (was `pl.UInt8`). No `ge`/`le` needed — the value range
  (0–50) is far inside `Int8`'s signed range and the dtype itself is the only bound today.
- `h3_index` field: `dtype=pl.Int64` (was `pl.UInt64`).
- `categorical_precipitation_type_surface` field: `dtype=pl.Int16` (was `pl.UInt8`), **add**
  `ge=0, le=255` — today's `UInt8` dtype enforces that range implicitly with no separate `ge`/`le`;
  widening the dtype means the range must now be stated explicitly or it silently disappears.
- Remove the `NWP_MODEL_ID_DTYPE` module constant — dead once the field no longer uses it and
  `convert_to_polars.py` no longer needs it either (see below).
- `_check_unique`: add `"nwp_model_id"` to the selected columns and to the error message, matching
  the class docstring's documented key.

### `packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py`

- Line 125 (`_process_chunk_for_1_lead_time_and_1_ens_member`): `.cast(pl.UInt8)` on
  `categorical_vars` → `.cast(pl.Int16)`.
- Line 71: `nwp_model_id=pl.lit(NwpModelId.ECMWF_ENS_0_25_degree.name).cast(NWP_MODEL_ID_DTYPE)` →
  `.cast(pl.String)` (or drop the cast — `pl.lit` of a `str` already infers `String` — but keeping
  an explicit cast preserves the line's self-documenting intent). Drop `NWP_MODEL_ID_DTYPE` from the
  import.

### `packages/delta_store/src/delta_store/nwp.py`

- `write_deltalake(...)` call in `write_nwp`: add `schema_mode="overwrite"` (see Question 1's
  reasoning for why this is safe to leave on permanently, not just for one transitional write).
- **Delete** the `nwp_model_id`-cast-and-strip workaround entirely:

  ```python
  prepared = pl.DataFrame._from_pydf(rounded._df).cast({"nwp_model_id": pl.String}).to_arrow()
  ```

  becomes

  ```python
  prepared = rounded.to_arrow()
  ```

  — and delete the comment block explaining the now-nonexistent problem (`Nwp`'s own declared
  dtypes are all delta-rs-storable after this change, so there is nothing left to strip a Patito
  model for: `.to_arrow()` isn't a `.cast()` call, so the "swallowed dict-cast on a model-bearing
  frame" gotcha never applied to it in the first place once no cast is needed). This is a genuine
  simplification, not just a rename — it removes one of the five workarounds documented in the
  `polars-patito-gotchas` skill's "friction budget", the skill explicitly invites revisiting an
  approach rather than accumulating a sixth.
- `NWP_SORT_COLS` docstring: the "~2% storage cost" claim is justified by row-group skipping that
  is not currently happening (see "Docs to update").

### Tests

See "Tests" section below for the full list; touches `packages/delta_store/tests/test_nwp.py`,
`packages/contracts/tests/test_weather_schemas_validation.py`, and
`packages/dynamical_data/tests/test_convert_to_polars.py`.

### Not touched, and why

- `AllFeatures.ensemble_member` (`packages/contracts/src/contracts/ml_schemas.py:76`, still
  `UInt8`) — deliberately left alone. `AllFeatures` is never persisted to Delta (no
  `delta_store` module handles it; it's computed fresh per call), so the pushdown argument driving
  this whole change does not apply to it. The cast from `Nwp`'s new `Int8` into `AllFeatures`'s
  `UInt8` during feature engineering is a normal, safe, always-in-range Patito model cast — no
  different in kind from the identity cast that exists today, just no longer an identity. Changing
  it would be unrelated churn outside this change's scope.
- `AllFeatures.categorical_precipitation_type_surface` (`ml_schemas.py:114`) — already `Float32`
  via the shared `_FEATURE_DTYPE`, unaffected by `Nwp`'s dtype either before or after this change.
- `H3GridWeights.h3_index` and `TimeSeriesMetadata.h3_res_5` (both `UInt64`, in
  `contracts.geo_schemas` / `contracts.power_schemas`) — separate contracts, not `Nwp`, out of this
  change's stated scope. Both back much smaller tables where the pushdown argument has far less
  payoff; not proposing to touch them here.
- `Nwp.scan_delta` itself — no code change. The `.cast()` call stays; for all four fields it becomes
  an identity cast once the declared dtype matches what's on disk, which is the entire point.

## Design-philosophy check

This is a contract/storage-format change, not a production degradation path — no
`AssetCheckSeverity` or fail-open/fail-closed question applies. The one thing worth checking against
`docs/design-philosophy/inherent-stability.md`'s "liberal about missing inputs, strict about
malformed ones": the `nwp_model_id` vocabulary constraint makes `Nwp.validate` *reject* an
unrecognised model id rather than accept it — correct under that rule, since an unrecognised model
id is malformed (a typo or a genuinely new, not-yet-onboarded model), not merely absent data.

No `docs/design-philosophy/engineering-hypotheses.md` hypothesis is directly targeted by this
change; it's an internal correctness/performance fix, not new capability.

## Tests

New tests, each with the assertion that would fail on `main` today:

1. **The regression test the task calls for** (`packages/delta_store/tests/test_nwp.py`): write a
   small table via `write_nwp`, then assert `"ensemble_member"` appears in the `SELECTION` line of
   `Nwp.scan_delta(table).filter(pl.col("ensemble_member") == 0).explain()`. **Fails on `main`
   today** — reproduced directly above: today's `explain()` output has no `SELECTION` line at all
   for this filter (it appears as a `FILTER` node *above* the cast, and the plan reads all
   partitions). Recommend the identical assertion for `h3_index` and `nwp_model_id` too, in the same
   test or as siblings — both independently reproduced as suffering the same pushdown loss today,
   even though the task named only `ensemble_member` explicitly.
2. **`categorical_precipitation_type_surface` ≥128 round-trips** (`test_nwp.py`): a frame built via
   `_make_nwp` with a ptype value of 255 (the documented "Missing" sentinel), written via
   `write_nwp`. **Fails on `main` today** — reproduced directly: `write_deltalake` raises `Cast
   error: Can't cast value 255 to type Int8`.
3. **Mixed old/new physical partitions read correctly** (`test_nwp.py`): write one partition with
   the old, narrower physical dtype (constructing it directly, to simulate data written before this
   change), then a second partition through the (changed) `write_nwp`, then assert
   `Nwp.scan_delta(table).collect()` returns correct values for both partitions with no error. This
   is new coverage, not a `main`-failing regression test in the strict sense (there's no old code
   path to compare against), but it's the test that stops "no rewrite is needed" from silently
   rotting the way the original pushdown claim did.
4. **`categorical_precipitation_type_surface` range enforcement**
   (`packages/contracts/tests/test_weather_schemas_validation.py`): extend the existing
   out-of-range parametrization (`test_out_of_range_continuous_weather_var_is_fatal`, or a sibling
   test) with a `categorical_precipitation_type_surface` value of `300` (and/or `-1`), asserting
   `DataFrameValidationError`. **New coverage** — today's `UInt8` dtype makes an out-of-range value
   physically inexpressible, so a dedicated range test was never needed. It's needed now that the
   range comes from an explicit, alterable `ge`/`le` rather than the dtype itself.
5. **`nwp_model_id` vocabulary rejection** (`test_weather_schemas_validation.py`): a frame with an
   unrecognised `nwp_model_id` string raises `DataFrameValidationError`. **New coverage** — no such
   check exists before this change (the `Enum` dtype itself would have made this un-constructible
   without a cast error rather than a validation error, so this is genuinely new behaviour).
6. **`_check_unique` catches a same-key-different-model duplicate**
   (`packages/contracts/tests/test_weather_schemas_validation.py` or alongside `_check_unique`):
   call `Nwp._check_unique` directly (not through the full `Nwp.validate()` pipeline) with two rows
   sharing `(init_time, valid_time, ensemble_member, h3_index)` but differing `nwp_model_id`, and
   assert it does **not** raise. Direct-call is necessary, not just convenient: once the vocabulary
   constraint above is in place, only one `nwp_model_id` value is legal, so a two-model frame cannot
   pass the full `Nwp.validate()` pipeline at all — this test necessarily exercises `_check_unique`
   in isolation. **Fails on `main` today** in the sense that `main`'s `_check_unique` would
   incorrectly raise on this input if it could be constructed at all (it can be constructed as a
   bare DataFrame without going through the vocabulary-gated `validate()`).

Existing tests needing updates (found by reading every caller; see "File-by-file" above for the
non-test callers):

- `packages/delta_store/tests/test_nwp.py:130` —
  `assert collected.schema["nwp_model_id"] == pl.Enum(["ECMWF_ENS_0_25_degree"])` →
  `pl.String`.
- `packages/dynamical_data/tests/test_convert_to_polars.py:518` —
  `assert df["categorical_precipitation_type_surface"].dtype == pl.UInt8` → `pl.Int16`.

Checked and **not** needing changes (Patito's non-strict `.cast()` absorbs the widening/narrowing
transparently for literal test values already inside the new range):
`packages/ml_core/tests/test_features.py` (several `UInt8`-typed fixture columns for
`ensemble_member`/`categorical_precipitation_type_surface`, none asserting the dtype directly);
`packages/contracts/tests/test_nwp_run_completeness.py`; `packages/weather_utils/tests/
test_analysis_proxy.py`. Confirm with a full `uv run pytest` rather than trusting this
read — that's what the verification set is for.

## Docs to update

Three locations claim a specific pushdown speed-up that is not currently happening (confirmed by
reproduction, not by re-reading the prose): the mechanism these docs describe will be **true again**
once this change ships (verified locally, see Question 1's reproduction method applied to
`ensemble_member`), but the **specific numbers** (≈50×, ~5× faster/~5× less memory, ~2% storage
cost) were measured before this regression existed and must not be silently restored as if they
still hold. This plan does not re-measure them — that reads real production NWP data, which is
explicitly out of bounds without the maintainer's go-ahead.

1. **`packages/delta_store/src/delta_store/nwp.py`**, `NWP_SORT_COLS` docstring — rewrite the "for a
   ~2% storage cost" clause to state the mechanism (member-early sort lets row-group min/max stats
   skip most row groups for a single-member predicate, *once that predicate reaches the Parquet scan
   unchanged* — which it now does) without repeating the unverified factor.
2. **`docs/architecture/performance.md`**, the pruning table (line 73, `ensemble_member`) and its
   surrounding paragraph (line 78, the "~5× faster and ~5× less peak memory" claim) — same
   treatment. While here: line 74 (`h3_index ∈ {cells}` → "Restricts to the cells") is *also*
   currently false for the identical reason (reproduced directly — `h3_index` gets the same
   `.strict_cast(UInt64)` today and shows no `SELECTION` line), even though the task's brief named
   only line 73. Fix both in the same edit.
3. **`src/nged_substation_forecast/defs/_engineering_inputs.py:57-60`**, the `ensemble_member`
   bullet in `load_engineering_inputs`'s docstring — same treatment.

**Follow-up measurement to hand back to the maintainer, not to run now**: re-run the historical
benchmark shape (a 29-day window × the 9 V1 H3 cells × the control member, collected through
`load_engineering_inputs`/`Nwp.scan_delta` — the exact production path, not a raw `pl.scan_delta`)
after this change ships, timing it and recording peak memory (`collect(engine="streaming")`, per
the existing convention in this codepath) against the same shape run today. That read touches real
production NWP data on S3; it should not run without explicit sign-off, even though it's a small,
bounded slice rather than a whole-table scan.

## Verification commands

Standard green-before-push set:

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
```

Additionally for this change:

```bash
# convert_to_polars.py is convention-sensitive to real ECMWF ENS data per its own docstring
uv run pytest --run-network -m network
```

Manually re-run this plan's local scratch reproductions (or fold them into the new tests above) as
part of implementation, to confirm the fix actually closes what it claims to:
`Nwp.scan_delta(...).filter(pl.col("ensemble_member") == 0).explain()` should now show `SELECTION`;
`write_nwp` should accept a ptype of 255.

## Risks and open questions

- **Has this write bug already caused silent `ecmwf_ens` failures or retries?** Per Question 2:
  every currently-stored partition is *guaranteed* free of ptype ≥128 (the write would have failed,
  not corrupted), but that leaves open whether some historical run *tried* to land such a value and
  failed/retried. Worth the maintainer checking Dagster's run history for `ecmwf_ens` for a pattern
  matching this. Not something this plan can check from a local worktree.
- **`nwp_model_id` vocabulary error message quality**: the plain `constraints=` field gives a
  generic "N rows do not match custom constraints" rather than naming the offending value or the
  valid set. Recommend accepting this (matches existing precedent for `PowerForecast.valid_time`)
  unless a reviewer wants the friendlier custom-classmethod form instead — flagged explicitly here
  rather than decided unilaterally, per the task's instruction not to let this trade-off pass
  silently.
- **`schema_mode="overwrite"` left on permanently** (Question 1): safe given `write_nwp`'s input is
  always an already-validated `pt.DataFrame[Nwp]` with the full column set, so the only schema
  change it can ever produce is a deliberate future contract dtype change like this one — but this
  does remove a "the write blows up loudly if code and disk disagree" safety net. Worth the
  maintainer's explicit agreement, since it's a permanent write-path behaviour change, not just a
  one-off migration step.
- **Docs numbers left unmeasured**: the three doc locations above will describe the *mechanism*
  correctly but omit the specific speed-up factors until the flagged follow-up measurement runs.
  That measurement needs sign-off before it runs against production data.
