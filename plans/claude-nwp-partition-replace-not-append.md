# Make an NWP partition write replace rather than append (#580)

**The problem.** `delta_store.nwp.write_nwp` is hard-coded `mode="append"`
(`packages/delta_store/src/delta_store/nwp.py:102`). The `ecmwf_ens` asset writes one whole
`(nwp_model_id, init_time)` Delta partition per Dagster daily partition, so re-materialising a
partition that has already landed appends a *second* copy of the run alongside the first.
`Nwp._check_unique` (`packages/contracts/src/contracts/weather_schemas.py:411`) already rejects a
duplicated `(init_time, valid_time, ensemble_member, h3_index)`, but only within the frame being
validated — it cannot see rows at rest — so the duplicate keys land silently and every later
`Nwp.scan_delta` read fans out. Since PR #577 that fan-out costs the whole live-forecast slot:
`PowerForecast.validate` raises on the duplicated primary key. Today the only defence is an
instruction to a human in the runbook ("do not re-materialise a partition that has already
landed").

**The planned solution.** `write_nwp` derives the `(nwp_model_id, init_time)` it has been handed
from the frame itself and writes with `mode="overwrite"` and that partition as the predicate,
instead of appending. Re-materialising a Dagster partition then replaces its rows rather than
duplicating them, and a run that Dynamical.org republishes after we ingested a short copy is
corrected by re-running the partition. No new parameter, no caller change, no operator toggle, and
no change to any Patito contract. The one thing `write_nwp`'s docstring said needed proving before
this could be trusted — a `replaceWhere` predicate on a `Timestamp` partition column — is proven
below.

## Verdict: worth implementing, and it also closes #476

### Departures from the issue and its comment

- **The issue comment's proposed fix is rejected.** It quotes PR #577's conclusion that "rejecting
  duplicated NWP at `Nwp.validate` on ingest — where the fault actually is — remains the better
  fix". `Nwp.validate` sees only the frame in hand, and already rejects duplicates within it. To
  reject a duplicate of what is *at rest* it would have to read the Delta table: object-store IO
  inside a Patito contract, on every ingest, over a 93 GB table. The fault is in the write, not in
  the contract, so the fix is in the write. **No contract change is proposed, and none is needed.**
- **#476 is the same change, arrived at from the other side**, and asks two further questions that
  this plan dissolves rather than answers. It proposes an opt-in `replace_partition` parameter plus
  "its own explicit op/run-config toggle rather than being reachable from the normal
  partition-materialise path, to avoid an accidental overwrite", and its comment asks whether
  `write_nwp` should instead *refuse* to append onto an existing partition. If the write always
  replaces the partition it is given, there is no unsafe default left to guard, no toggle to get
  wrong, and nothing to refuse. Recommendation: ship this, and close #476 as covered — see "Risks
  and open questions".
- **`write_power_forecasts`'s `replace_partition: tuple[str, str] | None` API is deliberately not
  mirrored.** That parameter exists because a multi-chunk CV materialisation must overwrite on its
  first chunk and append on the rest, and must clear the partition even when the first chunk is
  empty. NWP has neither shape: one caller, one whole run per call. An optional parameter whose
  default is the behaviour this issue exists to remove would be the worst of both.

### Already established at plan time

**The `Timestamp` partition predicate works.** Written against a temp Delta table partitioned by
`["nwp_model_id", "init_time"]` with `init_time` as `Datetime("us", "UTC")`, exactly as `write_nwp`
partitions:

- Overwriting one of three partitions with the predicate
  `nwp_model_id = '…' AND init_time = '2026-08-11T00:00:00+00:00'` replaced only that partition;
  the other two came back frame-equal to before. The literals `'2026-08-11 00:00:00'` and
  `'2026-08-11 00:00:00+00:00'` matched identically, so the format is not delicate.
- On-disk partition directories are percent-encoded
  (`init_time=2026-08-11%2000%3A00%3A00.000000`) and the predicate matched anyway — it is evaluated
  against the typed schema, not the directory name.
- `mode="overwrite"` with a predicate works on a **non-existent** table (it creates it), and for a
  **brand-new** partition of an existing table (3 rows → 6). So neither the first write nor the
  normal daily write is a special case.
- When the data written does not satisfy the predicate, delta-rs raises
  `DeltaError: Invalid data found: 3 rows failed validation check` and leaves the table unchanged.
  Deriving the predicate from the frame makes that unreachable, but it is a real backstop.

**There is no duplicated data at rest to clean up** in the local 93 GB `data/NWP` table. Read from
the transaction log (`DeltaTable.get_add_actions`), all 865 `(nwp_model_id, init_time)` partitions
hold exactly 7,243,785 rows — standard deviation zero, and exactly the complete-V1-run figure
`docs/architecture/ecmwf-ens-known-issues.md` quotes (51 members × 85 steps × 1671 cells). Cleanup
is therefore out of scope for this issue. The S3 table cannot be read from this machine: the
workspace `.env` sets no `data_path_internal` and no `data_store_*` credentials. See "Risks and open
questions".

## What changes, file by file

### `packages/delta_store/src/delta_store/nwp.py`

- **`write_nwp`** — unchanged signature. Take the target partition from the frame's first row
  (`nwp_model_id`, `init_time`) and call `write_deltalake` with `mode="overwrite"` and
  `predicate=f"nwp_model_id = '{model_id}' AND init_time = '{init_time.isoformat()}'"`, keeping
  `partition_by`, `writer_properties` and `storage_options` exactly as they are. **No guard against
  a frame spanning more than one partition**: no caller can produce one — the sole production caller
  is fed by `convert_nwp_xarray_dataset_to_polars_dataframe`, which stamps both partition columns
  with `pl.lit` over the whole frame
  (`packages/dynamical_data/src/dynamical_data/ecmwf_ens/convert_to_polars.py:71-72`) — and
  delta-rs already rejects any row that fails the predicate and leaves the table untouched (verified
  twice, below; it validates *every* row, not a sample — 1 bad row in 5001, placed last, was
  caught). An **empty** frame is a different case and is not covered by that backstop: taking the
  partition from row 0 raises `IndexError` before anything is written, which is the outcome we want
  (had an empty frame reached delta-rs it would have cleared the whole partition), and an NWP frame
  with no rows is our own bug rather than an absent input — `convert_nwp_xarray_dataset_to_polars_dataframe`
  raises on `pl.concat([])` first, so reaching `write_nwp` empty takes an empty `h3_grid_weights`
  parquet. No guard for it, and a docstring line saying the frame must be non-empty.
- **Docstrings** — the module docstring needs no change. `write_nwp`'s summary line (`:64`, "Append
  ``Nwp`` rows to the ``nwp`` Delta table…") and its "Append-only: …" paragraph
  (`:71–76`) are replaced by what it now does: each `(nwp_model_id, init_time)` partition is written
  whole, so re-materialising a partition replaces its rows rather than duplicating them. Say that
  the partition comes from the frame's first row and that delta-rs rejects a frame not wholly inside
  it. Record the `Timestamp`-predicate verification here, in the same form
  `write_power_forecasts` records its own ("confirmed empirically: …"), because this docstring is
  what asked for the verification.

### `src/nged_substation_forecast/defs/assets.py`

**No code change.** The `ecmwf_ens` docstring (`:329–339`) says the asset "appends it to the Delta
table through `delta_store.nwp.write_nwp`"; that clause becomes "writes it … replacing that
partition". This is the only edit in this file — the rest of it belongs to #506 and #488.

### Tests

`packages/delta_store/tests/test_nwp.py`:

- `test_rewriting_a_partition_replaces_only_that_partition` — one new test, covering the whole
  change at this level. Write runs at three `init_time`s, then rewrite the middle one with different
  continuous values; assert the table still holds `3n` rows rather than `4n`, that the outer two
  partitions are frame-equal to what was written, and that the middle holds only the new values.
  That single test states both halves — replace rather than append, and replace *only* the named
  partition, which is the `Timestamp`-partition predicate pinned in-repo rather than in a scratch
  spike. **Fails on `main`**: `4n` rows, with the middle partition holding both copies.
- `test_successive_appends_create_separate_partitions` (`:99`) is renamed to
  `…successive_runs…` — "appends" is no longer what happens — and keeps every assertion, because
  what it actually covers is the `Nwp.scan_delta` round-trip (dtypes and the `nwp_model_id` `Enum`),
  which the new test has no reason to repeat.

`tests/test_assets.py`:

- `test_ecmwf_ens_re_materialising_a_partition_does_not_duplicate_rows` — materialise `ecmwf_ens`
  twice for the same partition key against the same table, reusing `_stub_ecmwf_download` and the
  `env` fixture; assert the row count is identical after the second run and that the primary key has
  no duplicates. **Fails on `main`**: the row count doubles. This is the end-to-end statement of the
  issue's own title, and the one test that would survive `write_nwp` being rewritten.

### Docs

- `docs/live_service/operations.md:309–316` — "**Do not re-materialise a partition that has already
  landed**" is now false. Rewrite it to say a re-materialisation replaces the partition, so a run
  that Dynamical republishes is corrected by re-running the Dagster partition. The #476 pointer goes
  with it. Two things the rewrite must add, both established by review rather than assumed:
  **re-materialising a partition while another materialisation of the same partition is in flight
  fails one of them** with delta-rs' `CommitFailedError` (verified: 4 concurrent same-partition
  writers, 3 raised) — one lost run instead of a corrupted table, and the opposite of what the same
  race does today, which is to commit both copies; and **the superseded parquet stays on disk**
  (nothing in this repo calls `vacuum`), so re-running a V1 partition leaves ~7.24M dead rows behind.
- `docs/live_service/operations.md:318–323` — "A partition whose run *failed* is safe to
  re-materialise" stays true and gets simpler: the "check the table before re-running it" advice for
  a run killed between the Delta commit and Dagster recording success is no longer needed, because
  re-running replaces whatever landed.
- `docs/live_service/operations.md:278` — "Combined with the append-only write above, a
  badly-degraded run cannot be corrected in place either." Now it can; rewrite the sentence rather
  than deleting the paragraph, which is otherwise still right about the check being the only signal.
- `docs/live_service/operations.md:298` — "The action is to chase Dynamical.org, not to touch
  the table." That is exactly the situation this change exists to unblock: the action becomes chase
  Dynamical *and then* re-materialise the partition once they republish.
- `docs/design-philosophy/inherent-stability.md:147` — the failure-modes row whose trigger is "most
  plausibly from NWP rows duplicated at rest, since `ecmwf_ens` appends without dedup". That trigger
  is closed; the row stays (a duplicated forecast row is still a hard failure) with the trigger
  restated as a code bug.
- `docs/design-philosophy/inherent-stability.md:193–195` — rule 7's justifying clause, "`ecmwf_ens`
  appends its NWP run with no dedup, so a bug that raised after the append would leave the rows
  committed on a failed run and duplicate them when the partition was re-materialised". The rule
  itself is untouched and the first half stays true; the duplication half does not.
- `docs/ml_experimentation/dagster-workflow.md:35` — the `ecmwf_ens` backfill step says the asset
  "appends it to `nwp_data.delta`". Reuse the wording that page already uses at `:47` for
  `eligible_time_series` ("an idempotent partition overwrite, so re-materialising replaces rather
  than appends"), which also tells a backfiller that re-running a date range is now safe.
- `docs/architecture/ecmwf-ens-known-issues.md:239` — "A run that fails ingest writes nothing
  (validation runs before the Delta append) …". Still true, but the word "append" goes stale, and
  this is one of the two pages #476 cites as telling an operator to leave a landed partition alone,
  so a reader arriving from there should find the new behaviour named.
- `CLAUDE.md:248` — "appends it to Delta Lake via `delta_store.nwp.write_nwp`".
- `packages/delta_store/tests/test_nwp.py:6` (module docstring, "appends landing as separate …
  Hive partitions") and the name `tests/test_assets.py:410::test_ecmwf_ens_materialises_and_appends_nwp`.

Checked and deliberately **not** changed: `docs/design-philosophy/common-incident-classes.md:125–131`.
Its account of the propagation ("the trigger is a bug in our own ingest rather than a routine
outage") never names the append and stays true — a duplicated key can still only arrive from a code
bug. Rewriting it would be rewriting correct prose.

Nothing here touches `docs/roadmap/live-service.md` or `.claude/skills/`, which belong to #583 in
this wave. `docs/roadmap/live-service.md:489` mentions duplicate rows, but in `power_forecasts`, and
is unaffected.

## Design-philosophy check

`ecmwf_ens` is production code (`PRODUCTION_LAYER_TAGS`), so
[Inherent Stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
governs. The two ways the upstream run can genuinely be absent or incomplete are untouched:
`NwpRunNotYetAvailable` and `NwpVariableWhollyMissing` are still caught and retried before the write
is reached, and a short run still lands with `nwp_run_is_complete` WARNing. What the change mainly
does is make the *recovery* from a short run possible at all — a degradation path getting shorter.

It does add one new way for `ecmwf_ens` to fail, which the runbook edit above has to name: two
materialisations of the *same* partition running at once now contend, and delta-rs raises
`CommitFailedError` on the loser (verified: 4 concurrent writers, 3 raised). Under `mode="append"`
they both commit — which is precisely the silent duplication this issue exists to remove — so
failing is the right behaviour, and it costs one run rather than a corrupted table. It is our own
concurrency bug rather than the outside world misbehaving, so rule 1 permits it; it surfaces as a
missed NWP run, which the failure-modes table (`inherent-stability.md:135–152`) already covers, so
that table needs no new row. Disjoint partitions do not contend at all (8 concurrent writers, 8
partitions, all committed), so the daily schedule and a non-overlapping backfill are unaffected.

No asset check is added or edited, so the WARN/`blocking=False` question does not arise, and rule
7's ordering constraint (assess before the non-idempotent write) is untouched — the change makes the
consequence rule 7 warns about smaller, since the write is no longer non-idempotent.

No principle in `design-principles.md` is traded away. The change moves NWP onto the same footing
that page already describes for CV folds (`:416`): re-materialising a partition overwrites it rather
than appending. Against
[engineering-hypotheses](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/engineering-hypotheses/),
it serves **T1.1** and **T1.4**: it deletes a runbook instruction a human has to remember and
replaces it with behaviour the code enforces, which is precisely what "the operator recovers …
unaided, using only the runbooks" needs fewer of.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

`mkdocs build --strict` matters here because the operations-runbook edit removes a cross-reference
to issue #476 and rewrites text around anchors other pages link to; read the rendered HTML for the
two edited `docs/live_service/operations.md` sections rather than trusting the linter.

## Risks and open questions

1. **This overrules a requirement written into #476, and that needs a yes before any code is
   written.** #476's body says "day-to-day `ecmwf_ens` materialisations should stay strictly
   append-only by default … to avoid an accidental overwrite", and asks for the replace to sit
   behind its own operator toggle. This plan makes every write a replace and has no toggle. The
   concrete cost of that is the next risk. *Recommendation: overrule it* — an append-only default is
   not a safe default here, it is the defect, and a toggle only moves the hazard to whether the
   operator sets it.
2. **A re-materialisation can now overwrite good rows with worse ones.** If a partition is re-run
   while Dynamical.org is mid-publication, a complete run on disk is replaced by a short one.
   *Recommendation: accept.* The same action under today's code duplicates the run instead, which is
   both worse and silent; refusing the write would leave #476's correction flow nowhere to go; and
   the short result is named by `nwp_run_is_complete` on the very materialisation that caused it.
3. **#476 is the same change and should be closed when this ships.** *Recommendation: `Closes #580`
   and `Closes #476` in the PR body.* Your call, since it is a second issue's fate.
4. **The S3 table has not been checked for duplicates at rest**, only the local 93 GB one, which is
   clean. *Recommendation: run the same `get_add_actions` row-count-per-partition check against the
   S3 table from a machine with credentials before this deploys.* If it does show duplicates, that
   is its own issue, not this one — an in-place rewrite must read from a version pinned before the
   rewrite starts and must not vacuum until the result is verified.
5. **Should `write_nwp` keep any way to append?** *Recommendation: no.* Nothing calls it that way,
   and a young project can add the parameter back the day a caller needs it.

**Checked so the next reader need not check it again**, all reproduced independently in the second
review against the pinned `deltalake` 1.6.2:

- Concurrent writes to *disjoint* partitions do not contend (8 threads, 8 partitions, all
  committed, table at version 8). Concurrent writes to the *same* partition do — see the
  design-philosophy section.
- The replace is one commit carrying both the remove and the add, so there is no window in which
  the old partition is gone and the new one not yet written. A kill mid-write leaves an orphan
  parquet no reader sees, exactly as an append does. Rule 7's ordering is untouched.
- The S3 path behaves identically to local (run against moto with this repo's `storage_options`
  shape): create, new partition, replace, and the mismatched-frame rejection all match.
- Every NWP reader goes through the Delta log (`Nwp.scan_delta` at `weather_schemas.py:483–489`,
  `packages/dashboard/view_forecasts.py:384,406`); nothing globs parquet under `nwp_data_path`, and
  nothing in the repo calls `vacuum` or `optimize`, so a long-running read cannot be pulled out from
  under.
- The predicate survives a non-midnight and a microsecond-precision `init_time`.

## What the reviews changed

**First review (simplicity).** Accepted: dropped the planned `ValueError` guards for an empty frame
and for a frame spanning two partitions, together with their test — unreachable from any caller, and
delta-rs already rejects a mismatched row atomically; collapsed four new tests into one new plus one
rename; dropped the `common-incident-classes.md` edit as prose that stays true; added the three doc
sites the plan had missed (`dagster-workflow.md:35`, the "chase Dynamical, not the table" sentence
in the operations runbook, and the two "append" names in the tests); promoted the #476 override from
a risk note to the blocking question above. Modified rather than accepted: the review proposed
folding `test_successive_appends_create_separate_partitions` into the new test — kept and renamed
instead, because its real subject is the `Nwp.scan_delta` round-trip (dtypes, the `nwp_model_id`
`Enum`), which the new test has no reason to repeat.

**Second review (correctness).** It re-ran every empirical claim above and reproduced all of them,
and it *built and ran both proposed tests* — confirming that `test_ecmwf_ens_re_materialising_…` is
achievable with the existing `env` fixture and `_stub_ecmwf_download` (4 rows → 8 on `main`, 4 → 4
with the change) and that the `delta_store` test is `3n` versus `4n` as written. Accepted from it:
the concurrent same-partition `CommitFailedError`, which the plan had wrongly claimed did not exist,
now named in the design-philosophy section and required of the runbook edit; the empty-frame
rationale, which the first review had recorded as "delta-rs rejects it" when in fact delta-rs would
have *cleared the partition* and it is the `IndexError` that prevents that; the fact that delta-rs
validates every row rather than sampling; the dead parquet left behind by a replace; and three more
edit sites (`nwp.py:64`, `ecmwf-ens-known-issues.md:239`, and a duplicated pair of bullets and a
wrong line number in this file). Nothing was rejected — every finding checked out.
