# Backing up the workstation

How to back up the workstation's data, MLflow store, literature library, Dagster history, and
credential files to the 18 TB USB disk mounted at `/mnt/wd_18tb`, and how to restore from that
backup. A systemd timer runs the backup once a day.

## The daily backup

**A systemd user timer runs the backup script every day at 03:20 UTC, and skips the day's run while
the USB disk is unplugged.** 03:20 falls between the Dagster schedules: the live forecasts at 00:00
and 06:00, the power ingest at 55 minutes past each hour, and the ECMWF download at 10:30. The unit
files are `scripts/maintenance/systemd/nged-backup.service` and `nged-backup.timer`. Install them
once:

```bash
cp scripts/maintenance/systemd/nged-backup.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now nged-backup.timer
```

The timer needs lingering turned on (`loginctl enable-linger`) so that it runs while nobody is
logged in. The timer also runs the backup 15 minutes after each start-up, which covers a day the
machine was off at 03:20 and gives the USB disk time to mount first.

**Read each run's output in the journal.** `systemctl --user list-timers nged-backup.timer` shows
when the backup last ran and when it runs next, and `journalctl --user -u nged-backup.service` shows
each run's warnings. A day the disk was unplugged appears as `skipped, unmet condition check
ConditionPathIsMountPoint=/mnt/wd_18tb`, not as a failure.

**Nothing outside the journal reports a failed or skipped backup.** No alert fires, so check
`systemctl --user list-timers nged-backup.timer` now and then: a `LAST` time more than a day old, or
a failed run in the journal, means the backup has stopped.

## Run the backup by hand

**Run the script from the main checkout, while no Dagster run is in progress:**

```bash
cd ~/dev/nged-substation-forecast
uv run python scripts/maintenance/backup_workstation.py
```

Check the Dagster UI (`http://localhost:3000`, Runs tab) first for runs in the `Started` or `Queued`
state. A Delta table copied while a run is writing to it can be captured inconsistently.

The script writes one snapshot directory per run, named after the time in UTC, such as
`/mnt/wd_18tb/nged-substation-forecast-backups/2026-10-05T183000Z/`. Each snapshot holds:

- `data_internal/` — the whole of `data/`: the NWP and NGED power tables, `power_forecasts`,
  `forecast_metrics`, the production model, and the study outputs under `data/studies/`
- `mlruns/` — the MLflow artifacts, which hold the trained models
- `literature/` — the git-ignored library of papers and reference documents
- `dagster_history/` and `dagster_home/` — Dagster's run history and its `$DAGSTER_HOME` directory
- `mlflow.db` — the MLflow database
- `secrets/` — `.env`, `.google_account_for_weathernext3.json`, and `packages/dashboard/.env.s3`,
  which the docs cannot recreate; the directory is readable by its owner only
- `sources.json` — the workstation path each directory and file above was copied from

**Every SQLite database is copied through SQLite's backup routine rather than as a plain file.**
Dagster keeps its run history in SQLite databases whose recent commits sit in a `-wal` file beside
each database, and a plain copy can capture a database and its `-wal` file at different moments.

**The first snapshot copies about 200 GB; each later snapshot stores only the files that changed.**
The script hard-links every unchanged file to the previous snapshot, so a snapshot takes disk space
only for new files. The SQLite databases are copied afresh every run, which adds about 450 MB a day,
or about 165 GB a year. Every snapshot is nevertheless a complete copy, and deleting an old snapshot
never damages a newer one. Delete old snapshots by hand when the disk fills, oldest first.

**The script stops if `data/` or the MLflow database is missing**, for example because `/mnt/data`
is not mounted, so a snapshot can never look complete while missing the forecasts and models. The
other directories are skipped with a warning when absent.

**The script refuses to run when the backup disk is not mounted.** An unmounted `/mnt/wd_18tb` is an
empty directory on the system disk, so the script checks that the destination is on a different disk
from every source. The script also refuses to run from a git worktree, because `mlruns/`,
`literature/`, `dagster_history/`, the credential files, and the MLflow database path are all found
relative to the checkout the script runs from.

## Read the output

**A warning naming an MLflow experiment means that experiment's trained models are not being backed
up.** MLflow stores each experiment's artifact location as an absolute path, and an experiment
created from a git worktree stores its models inside that worktree rather than in `mlruns/`. Move
the models into `mlruns/`, or accept that they are lost when the worktree is removed.

**The script ends by listing every file or directory created, changed, or deleted while the backup
ran.** An empty list means the snapshot matches the workstation exactly. A long list usually means a
Dagster run was writing during the backup: delete that snapshot and run the script again.

A directory ending in `.partial` is a snapshot whose run was interrupted. The script never uses a
`.partial` snapshot as the base for the next snapshot, and warns about each one it finds. Delete it
by hand.

## Restore

**Restore one directory at a time, with `--delete`, after stopping Dagster and the MLflow UI.**
`--delete` makes the restored directory an exact copy of the snapshot. Without `--delete`, a Delta
commit written after the snapshot would survive the restore, and Delta would go on reading the table
at that newer version. Set `SNAPSHOT` to the snapshot to restore from, and restore only the
directories that need it. Every command below runs from the main checkout. Each command writes
`${SNAPSHOT:?}` and `${DAGSTER_HOME:?}` rather than `$SNAPSHOT` and `$DAGSTER_HOME`, so that the
shell stops with an error when a variable is unset instead of running `rsync --delete` against `/`.
For example, to restore the `power_forecasts` table:

```bash
cd ~/dev/nged-substation-forecast
SNAPSHOT=/mnt/wd_18tb/nged-substation-forecast-backups/2026-10-05T183000Z
rsync -aH --delete "${SNAPSHOT:?}/data_internal/power_forecasts/" data/power_forecasts/
```

Never restore the whole of `data_internal/` with `--delete` unless every table needs restoring,
because `--delete` also removes every table and study output written since the snapshot.

**Delete the database's sidecar files before restoring `mlflow.db`.** SQLite applies a leftover
`mlflow.db-wal` or `mlflow.db-journal` file to whatever database file sits beside it, and doing so
corrupts the restored copy:

```bash
rm -f mlflow.db-wal mlflow.db-shm mlflow.db-journal
cp "${SNAPSHOT:?}/mlflow.db" mlflow.db
rsync -aH --delete "${SNAPSHOT:?}/mlruns/" mlruns/
```

Restore Dagster's history the same way, with `dg dev` stopped, and copy the credential files back
from `secrets/`:

```bash
rsync -aH --delete "${SNAPSHOT:?}/dagster_history/" dagster_history/
rsync -aH --delete "${SNAPSHOT:?}/dagster_home/" "${DAGSTER_HOME:?}/"
cp -p "${SNAPSHOT:?}/secrets/.env" "${SNAPSHOT:?}/secrets/.google_account_for_weathernext3.json" ./
cp -p "${SNAPSHOT:?}/secrets/packages/dashboard/.env.s3" packages/dashboard/
```

**Restore `mlruns/` and `mlflow.db` together.** The database names each run's artifact directory, so
a database from one snapshot and artifacts from another can point at models that are missing.

**On a fresh machine, clone the repository to the same path as before.** MLflow stores artifact
locations as absolute paths, such as `/home/jack/dev/nged-substation-forecast/mlruns/1`, so a clone
at a different path finds no models. Run `uv sync`, set up Dagster as in [Getting
started](../getting-started.md), recreate the `data/` symlink if `data/` lived on another disk, and
then restore each directory and the credential files as above. `sources.json` in the snapshot lists
where each directory came from.
