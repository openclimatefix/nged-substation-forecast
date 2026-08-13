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

- **`write_nwp`** — unchanged signature. Before writing, derive the target partition from the input
  frame: the unique `nwp_model_id` and the unique `init_time`. Raise `ValueError` if the frame is
  empty, or if either column holds more than one distinct value, naming what was found — the
  function already documents its argument as "validated NWP rows for a single
  `(nwp_model_id, init_time)` partition", and this makes that a checked precondition rather than a
  convention. Then call `write_deltalake` with `mode="overwrite"` and
  `predicate=f"nwp_model_id = '{model_id}' AND init_time = '{init_time.isoformat()}'"`, keeping
  `partition_by`, `writer_properties` and `storage_options` exactly as they are.
- **Docstrings** — the module docstring needs no change. `write_nwp`'s "Append-only: …" paragraph
  (`:71–76`) is replaced by what it now does: each `(nwp_model_id, init_time)` partition is written
  whole, so re-materialising a partition replaces its rows rather than duplicating them. Record the
  `Timestamp`-predicate verification in the same place and in the same form
  `write_power_forecasts` records its own ("confirmed empirically: …"), because that docstring is
  what asked for the verification.

### `src/nged_substation_forecast/defs/assets.py`

**No code change.** The `ecmwf_ens` docstring (`:329–339`) says the asset "appends it to the Delta
table through `delta_store.nwp.write_nwp`"; that clause becomes "writes it … replacing that
partition". This is the only edit in this file — the rest of it belongs to #506 and #488.

### Tests

`packages/delta_store/tests/test_nwp.py`:

- `test_rewriting_a_partition_replaces_rather_than_appends` — write the same run twice into one
  table; assert the table holds `n` rows rather than `2n`, and that
  `(init_time, valid_time, ensemble_member, h3_index)` has no duplicates. **Fails on `main`**: `2n`
  rows and every key duplicated.
- `test_replacing_one_partition_leaves_the_other_partitions_intact` — write runs at three
  `init_time`s, then rewrite the middle one with different continuous values; assert the outer two
  are frame-equal to what was written and the middle holds only the new values. This is the
  `Timestamp`-partition predicate pinned in-repo rather than in a scratch spike. **Fails on `main`**:
  the middle partition holds both copies.
- `test_write_nwp_rejects_a_frame_spanning_more_than_one_partition` — a frame with two `init_time`s
  raises `ValueError` naming them. **Fails on `main`**: both partitions are written happily.
- `test_successive_appends_create_separate_partitions` (`:99`) keeps its assertions — two different
  runs still land as two partitions — but is renamed to `…successive_runs…`, since "appends" is no
  longer what happens.

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
  with it.
- `docs/live_service/operations.md:318–323` — "A partition whose run *failed* is safe to
  re-materialise" stays true and gets simpler: the "check the table before re-running it" advice for
  a run killed between the Delta commit and Dagster recording success is no longer needed, because
  re-running replaces whatever landed.
- `docs/live_service/operations.md:278` — "Combined with the append-only write above, a
  badly-degraded run cannot be corrected in place either." Now it can; rewrite the sentence rather
  than deleting the paragraph, which is otherwise still right about the check being the only signal.
- `docs/design-philosophy/inherent-stability.md:147` — the failure-modes row whose trigger is "most
  plausibly from NWP rows duplicated at rest, since `ecmwf_ens` appends without dedup". That trigger
  is closed; the row stays (a duplicated forecast row is still a hard failure) with the trigger
  restated as a code bug.
- `docs/design-philosophy/inherent-stability.md:194` — rule 7's justifying clause, "`ecmwf_ens`
  appends its NWP run with no dedup, so a bug that raised after the append would leave the rows
  committed on a failed run and duplicate them when the partition was re-materialised". The rule
  itself is untouched and the first half stays true; the duplication half does not.
- `docs/design-philosophy/common-incident-classes.md:125–131` — the "one case is now settled"
  paragraph, whose settled case is duplicated NWP at rest fanning the feature join out. Rewrite to
  record that the ingest-side trigger is now closed, keeping the propagation shape as the code-bug
  case it remains.
- `CLAUDE.md:248` — "appends it to Delta Lake via `delta_store.nwp.write_nwp`".

Nothing here touches `docs/roadmap/live-service.md` or `.claude/skills/`, which belong to #583 in
this wave. `docs/roadmap/live-service.md:489` mentions duplicate rows, but in `power_forecasts`, and
is unaffected.

## Design-philosophy check

`ecmwf_ens` is production code (`PRODUCTION_LAYER_TAGS`), so
[Inherent Stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
governs. The change adds one raise, on a path that runs *before* the write: a frame that is empty or
spans more than one partition. Both mean the converter produced something it cannot produce — our
own bug, not the outside world misbehaving — so rule 1 permits the raise and rule 2 ("strict about
malformed") wants it. The two ways the upstream run can genuinely be absent or incomplete are
unaffected: `NwpRunNotYetAvailable` and `NwpVariableWhollyMissing` are still caught and retried
before the write is reached, and a short run still lands with `nwp_run_is_complete` WARNing.

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

1. **Replacing means a re-materialisation can overwrite good rows with worse ones.** If an operator
   re-runs a partition while Dynamical.org is mid-publication, a complete run on disk is replaced by
   a short one. *Recommendation: accept.* The same action under today's code produces duplicates,
   which is strictly worse and silent; refusing instead would leave #476's correction flow with
   nowhere to go; and the short result is named by `nwp_run_is_complete` on the very
   materialisation that caused it.
2. **#476 is the same change and should be closed when this ships.** *Recommendation: close it as
   covered by this PR, noting that its "explicit operator toggle" and "refuse to append" questions
   are dissolved rather than deferred.* Your call, since it is a second issue's fate.
3. **The S3 table has not been checked for duplicates at rest**, only the local 93 GB one, which is
   clean. *Recommendation: run the same `get_add_actions` row-count-per-partition check against the
   S3 table from a machine with credentials before this deploys.* If it does show duplicates, that
   is its own issue, not this one — an in-place rewrite must read from a version pinned before the
   rewrite starts and must not vacuum until the result is verified.
4. **Should `write_nwp` keep any way to append?** *Recommendation: no.* Nothing calls it that way,
   and a young project can add the parameter back the day a caller needs it.
5. **Empty frame: raise or no-op?** *Recommendation: raise.* A run with no rows cannot be written as
   a partition replacement at all, since there is no `init_time` to derive; and it is unreachable
   except as our own bug, because an upstream run that published nothing raises
   `NwpRunNotYetAvailable` well before the write.
