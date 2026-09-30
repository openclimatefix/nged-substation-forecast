# Plan: cut MOGREPS-UK over to the shared recorder code, and speed up the network fetch (#965)

**Problem.** The live MOGREPS-UK recorder runs from the `mogreps-recorder` branch of the `nwp-archivist` repository. The `mogreps-g-recorder` branch was built on top of it and already runs MOGREPS-Global in production. It adds the shared backfill-urgency file, so the two Met Office recorders can hand each other the backfill slice, and it has had one Opus review. Until MOGREPS-UK runs the same code, the urgency file has one writer and no reader, so yielding never happens. Separately, MOGREPS-UK's backfill is slow: about one run per 13 minutes, and the oldest run still to fetch (2026-09-08) leaves the bucket in about 8 days.

**Solution.** Fast-forward `mogreps-recorder` to `mogreps-g-recorder` and point the MOGREPS-UK unit at it with `--backfill-urgency-file`, in a planned window between two cycles, keeping the old commit as the rollback. Then measure whether more fetch worker processes raise throughput, and only write new fetch code if they do not.

## Verdict, size and departures

Worth doing as described. The work is in `nwp-archivist`, a separate local repository with no remote, so this branch carries only this plan and any docs; the recorder changes are commits on the `nwp-archivist` branches.

Size: **complex**, by one trigger, with one review (the coordinator's per-task cap).

- **Stores anything new:** no. The Icechunk layout, `layout_version` and status arrays are unchanged.
- **Touches the production serving path:** yes. The cutover changes the code and unit of a live recorder.
- **Touches a degradation rule:** no. The urgency file is opt-in and already fails open.
- **More than one defensible design:** yes, for the fetch: more processes, a thread prefetch layer, or both.
- **Callers not nameable without searching:** no. Every change is in `pool.py`, `cli.py` and two unit files.

One Opus review of this plan (rollout, rollback, measurement design) rather than two. The cutover ships code that MOGREPS-Global has already run in production for three days, so a second review of the diff would re-review reviewed code. A second review is added if the fetch step needs new code in `pool.py` or `mogreps.py`.

Departure from the issue body: the issue says to "profile" the fetch. A cycle uses 2.6 minutes of CPU in 10 minutes of wall clock, so the fetch waits on the network and profiling the Python would find nothing. The plan measures throughput against worker count instead.

## What changes

1. **Cutover, in the `nwp-archivist-mogreps` worktree.**
   - Record the current `HEAD` (`3cb261c`) as the rollback point and copy the installed unit to `nwp-archive-mogreps.service.rollback`.
   - Stop `nwp-archive-mogreps.timer`, wait for `nwp-archive-mogreps.service` to be inactive (never kill a running cycle), then `git merge --ff-only mogreps-g-recorder` and `uv sync`.
   - Reinstall the repository's `deploy/nwp-archive-mogreps.service` with the `dev/nwp-archivist` to `dev/nwp-archivist-mogreps` substitution, which adds `--backfill-urgency-file`. `daemon-reload`, start the timer.
   - Check the next two cycles: the MOGREPS-UK cycle logs a normal pass, and `/mnt/data/nwp-archive-backfill-urgency-mogreps.json` gains a `mogreps-uk` key beside `mogreps-g`.
   - Rollback: stop the timer, `git reset --hard 3cb261c` in that worktree (after `git status` shows nothing uncommitted), restore the `.rollback` unit, `daemon-reload`, start the timer. No stored data is touched either way, because the layout is identical.
2. **Fetch measurement.** With the cutover done, change `--fetch-processes` on the MOGREPS-UK unit from 4 to 8, then 12 if 8 helped, one step per several backfill slices. Record fields per minute (from the `received=` counts in the journal), the NIC throughput (`/sys/class/net/enp5s0f0np0/statistics/rx_bytes`), main-process memory, and any `run_error` fault. The NIC is 1 Gb/s, and MOGREPS-Global shares it. Keep the highest count that still raises fields per minute without a fault, and stay under `MemoryMax=20G` (each worker holds about 0.5 GB).
3. **Prefetch thread pool, only if step 2 plateaus below the link.** The h5py global lock serialises chunk-index reads inside one process, so a layer that issues byte-range GETs ahead of h5py from a thread pool is the remaining option. It changes `mogreps.py`'s read path and would get its own tests and the second review. If step 2 reaches the link's limit or the source's rate limit, this step is not built.

## Design-philosophy check

The recorders are production code, so nothing may raise because an input is absent. The cutover adds no new failure path: `_apply_backfill_urgency` already catches every exception and reports a `run_error` fault, and a missing, stale or corrupt urgency file means "proceed normally". A worker crash or pool timeout already leaves the run `waiting` with nothing committed.

## Tests

No new tests for steps 1 and 2. Step 1 moves the live branch to code with 183 passing tests, and the rollout checks above are the acceptance test. Step 2 changes one CLI value. Step 3, if built, needs a test that a prefetch failure leaves the run `waiting` and commits nothing, which would fail on today's code because the layer does not exist.

## Docs to update

None in this repository unless step 3 is built. `--fetch-processes` and the urgency flag are documented in the `nwp-archivist` unit-file comments, which the cutover updates. The architecture page for #966 records the outcome.

## Verification

`uv run pytest`, `ruff`, `ty` and `uv run pre-commit run --all-files` in `nwp-archivist-mogreps` after the fast-forward, before the unit is reinstalled. Then the two-cycle log check above.

## Risks and open questions

- **The MOGREPS-UK unit is not in the repository's copy of its own unit.** The installed unit lacks the urgency flag; the repository's copy has it. Reinstalling from the repository unit is the fix, and the diff between the two must be read before reinstalling, so no other setting changes silently.
- **More workers raise the Met Office bucket request rate.** The bucket is anonymous S3, so a rate limit would show as `503 SlowDown` responses, which the recorder treats as "not yet". Watch for a rise in `is not usable` lines while stepping up.
- **Yielding delays MOGREPS-UK's backfill when MOGREPS-Global is more urgent.** That is the intended behaviour, but it lowers MOGREPS-UK's rate for a while. The oldest MOGREPS-Global run has 6.5 days left and MOGREPS-UK's about 8, so expect UK to yield first.
