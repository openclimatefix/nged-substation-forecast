# Plan: cut MOGREPS-UK over to the shared recorder code, and speed up the network fetch (#965)

**Problem.** The live MOGREPS-UK recorder runs from the `mogreps-recorder` branch of the `nwp-archivist` repository. The `mogreps-g-recorder` branch was built on top of it and already runs MOGREPS-Global in production. It adds the shared backfill-urgency file, so the two Met Office recorders can hand each other the backfill slice, and it has had one Opus review. Until MOGREPS-UK runs the same code, the urgency file has one writer and no reader, so yielding never happens. Separately, MOGREPS-UK's backfill is slow: about one run per 13 minutes, and the oldest run still to fetch (2026-09-08) leaves the bucket in about 8 days.

**Solution.** Fast-forward `mogreps-recorder` to `mogreps-g-recorder` and point the MOGREPS-UK unit at it with `--backfill-urgency-file`, in a planned window between two cycles, keeping the old commit as the rollback. Then measure whether more fetch worker processes raise throughput, and only write new fetch code if they do not.

## Verdict, size and departures

Worth doing as described. The work is in the `nwp-archivist` repository, so this branch carries only this plan and any docs; the recorder changes are commits on the `nwp-archivist` branches.

Size: **complex**, by one trigger, with one review (the coordinator's per-task cap).

- **Stores anything new:** no. The Icechunk layout, `layout_version` and status arrays are unchanged.
- **Touches the production serving path:** yes. The cutover changes the code and unit of a live recorder.
- **Touches a degradation rule:** no. The urgency file is opt-in and already fails open.
- **More than one defensible design:** yes, for the fetch: more processes, a thread prefetch layer, or both.
- **Callers not nameable without searching:** no. Every change is in `pool.py`, `cli.py` and two unit files.

One Opus review of this plan (rollout, rollback, measurement design) rather than two. The cutover ships code that MOGREPS-Global has already run in production for three days, so a second review of the diff would re-review reviewed code. A second review is added if the fetch step needs new code in `pool.py` or `mogreps.py`.

Departure from the issue body: the issue says to "profile" the fetch. A cycle uses 2.6 minutes of CPU in 10 minutes of wall clock, so the fetch waits on the network or on request latency and profiling the Python would find nothing. The plan measures throughput against worker count instead.

## What changes

0. **Fix found in review (commit `2f6665d` on `nwp-archivist` branch `urgency-oldest-run`, built on `mogreps-g-recorder`).** `_apply_backfill_urgency` took `backfill_runs[0]` as the oldest run, but the backfill order puts 00, 06, 12 and 18 UTC runs first. MOGREPS-Global only has those hours, so it was unaffected. MOGREPS-UK runs hourly, so an incomplete recent 6-hourly run would have reported about 29 days left while a run with about 8.8 days left waited, and MOGREPS-UK would have yielded to MOGREPS-Global every cycle. The urgency now comes from the oldest unarchived run before the `max_backfill_runs` cut. A test drives `_apply_backfill_urgency` with a recent 6-hourly run first and asserts the written days-left belongs to the older hourly run. Checks ran in a separate worktree so the live timers' code was not touched.
1. **Cutover in two phases, in the `nwp-archivist-mogreps` worktree, each between two cycles.**
   - Phase A, code only, old unit: record the current `HEAD` (`3cb261c`) as the rollback point and copy the installed unit to `nwp-archive-mogreps.service.rollback`. Stop `nwp-archive-mogreps.timer`, wait for `nwp-archive-mogreps.service` to be inactive (never kill a running cycle), `git status` clean, then `git merge --ff-only urgency-oldest-run` and `uv sync` (a no-op: `pyproject.toml` and `uv.lock` are unchanged, and the venv is an editable install). Start the timer and check `list-timers` shows a NEXT time. Without `--backfill-urgency-file` the urgency code is skipped, so Phase A changes no behaviour except the code itself. The full test suite and `pre-commit` run in the separate worktree beforehand, because pre-commit fixers would dirty the live worktree.
   - Phase B, after the step 2 measurement: reinstall the unit from the repository copy with a global substitution (`sed 's#dev/nwp-archivist\b#dev/nwp-archivist-mogreps#g'`, which must change both `WorkingDirectory` and `ExecStart`), then diff it against the `.rollback` copy. The only intended differences are the urgency flag and comments. `daemon-reload`, restart between cycles. Then check two cycles: a normal pass, and a `mogreps-uk` key in `/mnt/data/nwp-archive-backfill-urgency-mogreps.json` beside `mogreps-g`. The MOGREPS-Global worktree is fast-forwarded to the same commit in a window between its cycles.
   - Rollback: stop the timer, `git status`, `git reset --hard 3cb261c` in that worktree, restore the `.rollback` unit, `daemon-reload`, start the timer. No stored data is touched, because the layout is identical. A leftover `mogreps-uk` urgency entry goes stale after 1 hour and is then ignored.
2. **Fetch measurement, between Phase A and Phase B, so that no cycle yields.** Change `--fetch-processes` from 4 to 8 on the MOGREPS-UK unit and compare several backfill slices at each setting. Fields per minute is the per-cycle change in `received=` for a backfill run, because `received=` accumulates across cycles. The network card counter `rx_bytes` also counts MOGREPS-Global and DWD traffic, so read it only as an upper bound. Record `MemoryPeak`, since a cycle with 4 workers peaks at 5.9 to 7.8 GB, so each process holds about 1.3 to 1.6 GB rather than 0.5 GB. Stop at 8 workers, and only raise `MemoryMax` deliberately before trying more. Watch for a rise in `is not usable` lines (bucket throttling) and for any `run_error` fault.
3. **Prefetch thread pool, only if step 2 shows fetching is not latency-bound.** If fields per minute rises roughly in proportion to workers, each request waits mostly on latency, and the answer is more workers or the prefetch layer. If it does not rise, the source or the link is the limit, and no layer would help. The prefetch layer changes `mogreps.py`'s read path, and would get its own tests and the second review.

## Design-philosophy check

The recorders are production code, so nothing may raise because an input is absent. The cutover adds no new failure path: `_apply_backfill_urgency` already catches every exception and reports a `run_error` fault, and a missing, stale or corrupt urgency file means "proceed normally". A worker crash or pool timeout already leaves the run `waiting` with nothing committed.

## Tests

No new tests for steps 1 and 2. Step 1 moves the live branch to code with 183 passing tests, and the rollout checks above are the acceptance test. Step 2 changes one CLI value. Step 3, if built, needs a test that a prefetch failure leaves the run `waiting` and commits nothing, which would fail on today's code because the layer does not exist.

## Docs to update

None in this repository unless step 3 is built. `--fetch-processes` and the urgency flag are documented in the `nwp-archivist` unit-file comments, which the cutover updates. The architecture page for #966 records the outcome.

## Verification

`uv run pytest`, `ruff`, `ty` and `uv run pre-commit run --all-files` in `nwp-archivist-mogreps` after the fast-forward, before the unit is reinstalled. Then the two-cycle log check above.

## Risks and open questions

- **The installed MOGREPS-UK unit differs from the repository copy** in the urgency flag and comments, and the repository copy names `dev/nwp-archivist` (the DWD worktree) in both `WorkingDirectory` and `ExecStart`. A substitution without the global flag would start the DWD code, so the diff against the `.rollback` copy is mandatory.
- **More workers raise the request rate against the Met Office bucket.** A throttle would appear as `is not usable` lines, which the recorder treats as "not yet".
- **Yielding delays MOGREPS-UK's backfill when MOGREPS-Global is more urgent.** That is the intended behaviour. MOGREPS-Global's oldest run has about 6.4 days left and MOGREPS-UK's about 8.8, so MOGREPS-UK yields first once the flag is on, and the backfill rate measured in step 2 is not the rate afterwards.
- **Two recorders writing the urgency file at the same moment can drop one entry for a cycle.** The recorder fails open, so the cost is one cycle of both proceeding.
