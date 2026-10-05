# Backing up the workstation

How to back up the workstation's data, MLflow store, and literature library to the 18 TB disk
mounted at `/mnt/wd_18tb`, and how to restore from that backup. The backup is run by hand, about
once a week.

## Run the backup

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
- `mlflow.db` — the MLflow database, copied through SQLite's backup routine
- `sources.json` — the workstation path each directory above was copied from

**The first snapshot copies about 200 GB; each later snapshot stores only the files that changed.**
The script hard-links every unchanged file to the previous snapshot, so a snapshot takes disk space
only for new files. Every snapshot is nevertheless a complete copy, and deleting an old snapshot
never damages a newer one. Delete old snapshots by hand when the disk fills, oldest first.

**The script refuses to run when the backup disk is not mounted.** An unmounted `/mnt/wd_18tb` is an
empty directory on the system disk, so the script checks that the destination is on a different
disk from every source. The script also refuses to run from a git worktree, because `mlruns/`,
`literature/`, `.env`, and the MLflow database path are all found relative to the checkout the
script runs from.

## Read the output

**A warning naming an MLflow experiment means that experiment's trained models are not being backed
up.** MLflow stores each experiment's artifact location as an absolute path, and an experiment
created from a git worktree stores its models inside that worktree rather than in `mlruns/`. Move
the models into `mlruns/`, or accept that they are lost when the worktree is removed.

**The script ends by listing every file that changed while the backup ran.** An empty list means the
snapshot matches the workstation exactly. A long list usually means a Dagster run was writing during
the backup: delete that snapshot and run the script again.

A directory ending in `.partial` is a snapshot whose run was interrupted. The script never uses a
`.partial` snapshot as the base for the next snapshot, and warns about each one it finds. Delete it
by hand.

## Restore

**Restore one directory at a time, with `--delete`, after stopping Dagster and the MLflow UI.**
`--delete` makes the restored directory an exact copy of the snapshot. Without `--delete`, a Delta
commit written after the snapshot would survive the restore, and Delta would go on reading the table
at that newer version. Set `SNAPSHOT` to the snapshot to restore from, and restore only the
directories that need it. For example, to restore the `power_forecasts` table:

```bash
SNAPSHOT=/mnt/wd_18tb/nged-substation-forecast-backups/2026-10-05T183000Z
rsync -aH --delete "$SNAPSHOT/data_internal/power_forecasts/" data/power_forecasts/
```

Never restore the whole of `data_internal/` with `--delete` unless every table needs restoring,
because `--delete` also removes every table and study output written since the snapshot.

**Delete the database's sidecar files before restoring `mlflow.db`.** SQLite applies a leftover
`mlflow.db-wal` or `mlflow.db-journal` file to whatever database file sits beside it, and doing so
corrupts the restored copy:

```bash
rm -f mlflow.db-wal mlflow.db-shm mlflow.db-journal
cp "$SNAPSHOT/mlflow.db" mlflow.db
rsync -aH --delete "$SNAPSHOT/mlruns/" mlruns/
```

**Restore `mlruns/` and `mlflow.db` together.** The database names each run's artifact directory,
so a database from one snapshot and artifacts from another can point at models that are missing.

**On a fresh machine, clone the repository to the same path as before.** MLflow stores artifact
locations as absolute paths, such as `/home/jack/dev/nged-substation-forecast/mlruns/1`, so a clone
at a different path finds no models. Run `uv sync`, create `.env` as in [Getting
started](../getting-started.md), recreate the `data/` symlink if `data/` lived on another disk, and
then restore each directory as above. `sources.json` in the snapshot lists where each directory
came from.
