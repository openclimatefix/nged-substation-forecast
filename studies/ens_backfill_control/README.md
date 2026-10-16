# ECMWF ENS control-member backfill fetch

**These scripts fetch the ECMWF ensemble control member for every complete date of Dynamical.org's
staged archive (2021-03-21 to 2024-03-31) and check the result.** The destination is
`/mnt/wd_18tb/data/nwp-archive/ECMWF-ENS-BACKFILL-CONTROL/`. The fetch code is shared with the
[pilot](../ens_backfill_pilot/README.md) through `studies.ens_backfill_fetch`.

- `fetch_control.py` fetches 1,020 messages (12 variables at 85 steps) per date, about 485 MB over
  the network and 14 MB on disk. It fetches the second validation year (2023-07-01 onwards) first,
  skips dates already written, and takes `--shard` and `--shard-count` to split dates between
  processes.
- `check_fetch.py` validates every file and writes `report.md`, `README.md` and `LINEAGE.json`
  into the destination folder.

```bash
uv run python studies/ens_backfill_control/fetch_control.py --shard 0 --shard-count 2 --workers 12
uv run python studies/ens_backfill_control/check_fetch.py
```

Measured on this machine: 8 connections gave 4.7 MB/s, 16 gave 10.5 MB/s, and two processes of 16
gave 23.5 MB/s together.
