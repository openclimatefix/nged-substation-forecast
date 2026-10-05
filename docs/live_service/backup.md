# Backing up the workstation

How to copy the machine-learning experiment outputs on the workstation to an external USB hard disk.
The procedure is manual, and is meant to run about once a week.

**Back up the experiment outputs first, because they cannot be re-downloaded.** The NWP and NGED
power tables can be downloaded again in an evening, so they are an optional second command. The
experiment outputs took compute time to produce:

- the `power_forecasts` Delta table, which holds the forecasts from every experiment
- the `forecast_metrics` Delta table
- the MLflow store, which is the `mlflow.db` SQLite database plus the `mlruns/` artifacts directory
  holding the trained models
- the promoted production model directory

## Before you start

**Run the backup only while no Dagster run is in progress and the MLflow UI is stopped.** A copy
taken while a run is writing can capture a Delta table whose transaction log and data files
disagree, or an `mlflow.db` file that is half-way through a transaction. Neither fault shows up
until a restore. Check the Dagster UI (`http://localhost:3000`, Runs tab) for runs in the `Started`
or `Queued` state, and stop `uv run mlflow ui` if it is running.

**Use an ext4 disk.** The commands below use `rsync -a`, which keeps permissions, modification
times, and symlinks. An exFAT or NTFS disk cannot hold those, and `rsync -a` reports an error on
every file. If the disk is exFAT, replace `-aH` with `-rt` in every command below.

## Find the mount point

Plug in the disk and list the block devices:

```bash
lsblk -o NAME,LABEL,SIZE,FSTYPE,MOUNTPOINT
```

The disk is the row with a `MOUNTPOINT` such as `/media/jack/<label>`. If `MOUNTPOINT` is empty,
mount it from the desktop's file manager, or run `udisksctl mount -b /dev/sdX1`, where `sdX1` is the
partition's `NAME` from the listing. Then set the destination used by every command below:

```bash
BACKUP=/media/jack/<label>/nged-backup
mkdir -p "$BACKUP"
```

## Find the source paths

The data paths come from `Settings`
([`packages/contracts/src/contracts/settings.py`](../api/contracts/index.md)), so they follow
`DATA_PATH_INTERNAL`, `DATA_PATH_DELIVERY`, `LOCAL_ARTIFACTS_PATH`, and `MLFLOW_TRACKING_URI` in
`.env`. Run this from the repository root. On the workstation, `data/` is a symlink to `/mnt/data`,
and the paths below resolve through it.

```bash
eval "$(uv run python - <<'PY'
from contracts.settings import get_settings
s = get_settings()
print(f"POWER_FORECASTS={s.power_forecasts_data_path}")
print(f"FORECAST_METRICS={s.forecast_metrics_data_path}")
print(f"PRODUCTION_MODEL={s.production_model_path}")
print(f"MLFLOW_DB={s.mlflow_tracking_uri.removeprefix('sqlite:///')}")
print(f"NWP={s.nwp_data_path}")
print(f"NGED={s.nged_data_path}")
PY
)"
echo "$POWER_FORECASTS" "$FORECAST_METRICS" "$PRODUCTION_MODEL" "$MLFLOW_DB"
```

Each printed path must be a local directory or file. A path starting with `s3://` is in an S3
bucket, and `rsync` cannot copy it. A relative `MLFLOW_DB` such as `mlflow.db` is relative to the
repository root. MLflow stores its artifacts in `mlruns/` in the directory where the run started,
which is the repository root.

## Back up the experiment outputs

```bash
rsync -aH --info=progress2 "$POWER_FORECASTS/" "$BACKUP/power_forecasts/"
rsync -aH --info=progress2 "$FORECAST_METRICS/" "$BACKUP/forecast_metrics/"
rsync -aH --info=progress2 "$PRODUCTION_MODEL/" "$BACKUP/production_model/"
rsync -aH --info=progress2 mlruns/ "$BACKUP/mlruns/"
sqlite3 "$MLFLOW_DB" ".backup '$BACKUP/mlflow.db'"
```

What each part does:

- `-a` keeps permissions, modification times, and symlinks, which is what lets a second run skip
  every file that has not changed.
- `-H` keeps hard links as hard links, so linked files are not stored twice.
- `--info=progress2` prints one progress line for the whole transfer instead of one per file.
- A trailing `/` on the source copies the directory's contents into the destination directory.
  Without it, `rsync` creates a nested `power_forecasts/power_forecasts/`.
- `sqlite3 ... ".backup"` copies the MLflow database through SQLite's own backup routine, which
  produces a consistent file. A plain `rsync` of an SQLite file can miss changes sitting in its
  write-ahead log, which is a sidecar file next to the database.

The commands never pass `--delete`, so a file deleted from the workstation stays on the disk. That
keeps one mistaken deletion from reaching the backup. The cost is that the backup slowly gains the
old files that Delta compaction replaces. Delta ignores files its transaction log does not name,
so the extra files are harmless.

## Optionally back up the NWP and NGED power tables

These tables are large, so run this only if the disk has the space. Check with `du -sh "$NWP"
"$NGED"` and `df -h "$BACKUP"`.

```bash
rsync -aH --info=progress2 "$NWP/" "$BACKUP/NWP/"
rsync -aH --info=progress2 "$NGED/" "$BACKUP/NGED/"
```

## Verify the copy

A dry run with `--itemize-changes` lists each file that still differs. Right after a backup, it must
print nothing:

```bash
for pair in "$POWER_FORECASTS:power_forecasts" "$FORECAST_METRICS:forecast_metrics" \
            "$PRODUCTION_MODEL:production_model"; do
  rsync -aHn --itemize-changes "${pair%%:*}/" "$BACKUP/${pair##*:}/"
done
rsync -aHn --itemize-changes mlruns/ "$BACKUP/mlruns/"
```

Add `--checksum` to any of these to compare file contents instead of sizes and modification times.
That reads every byte on both disks, so use it occasionally rather than every week.

Then check that the copies open. Delta must read the backed-up `power_forecasts` table, and SQLite
must accept the backed-up database:

```bash
uv run python -c "import polars as pl, sys; print(pl.scan_delta(sys.argv[1]).select(pl.len()).collect())" \
  "$BACKUP/power_forecasts"
sqlite3 "$BACKUP/mlflow.db" "PRAGMA integrity_check;"
```

The first command prints the row count. The second prints `ok`.

## Restore

Stop Dagster and the MLflow UI first, so that nothing writes while the files are being replaced.
Restore into the paths printed by the "Find the source paths" step, by swapping source and
destination:

```bash
rsync -aH --info=progress2 "$BACKUP/power_forecasts/" "$POWER_FORECASTS/"
rsync -aH --info=progress2 "$BACKUP/forecast_metrics/" "$FORECAST_METRICS/"
rsync -aH --info=progress2 "$BACKUP/production_model/" "$PRODUCTION_MODEL/"
rsync -aH --info=progress2 "$BACKUP/mlruns/" mlruns/
cp "$BACKUP/mlflow.db" "$MLFLOW_DB"
```

To restore onto a fresh machine, run `uv sync` and create `.env` first, as in [Getting
started](../getting-started.md). Then run the "Find the source paths" step so the variables point at
the new machine's directories, and run the commands above.
