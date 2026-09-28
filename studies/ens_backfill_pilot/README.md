# ECMWF ENS backfill pilot

**This pilot tests whether ECMWF's control-member forecasts for 2021-03-21 to 2024-03-31 can be
rebuilt from the GRIB files that Dynamical.org stages on Source Cooperative.** Dynamical.org's own
archive starts on 2024-04-01. The staged files hold ECMWF's GRIB edition 1 messages for the earlier
dates, one 2 MB message per (ensemble member, step, parameter). The pilot fetches the control
member only, for 23 dates, and checks that the values decode exactly and are physically plausible.

## Scripts

- `fetch_pilot.py` fetches, for each date, the 12 variables the `Nwp` contract needs (`2t`, `2d`,
  `10u`, `10v`, `100u`, `100v`, `sp`, `msl`, `tp`, `strd`, `ssrd`, and `z` at 500 hPa) at all 85
  forecast steps. That is 1,020 messages per date. Each message costs one range request for its
  first 475,357 bytes, which hold the message header and grid rows 0 to 164. The script keeps rows
  115 to 164 (61.25 to 49 degrees north) and the 69 columns from 12 degrees west to 5 degrees east.
  It writes one checkpoint file per date and skips dates already written.
- `check_pilot.py` compares a sample of messages with ecCodes and writes `report.md`. Every number
  in the report is printed by the script.
- `pilot_common.py` holds the constants and the HTTP code that the two scripts share.

The decoding, sidecar parsing, and de-accumulation live in `packages/studies/src/studies/`
(`grib1_simple.py`, `ens_grib_source.py`, `deaccumulation.py`), where they have unit tests.

## Commands

```bash
uv run python studies/ens_backfill_pilot/fetch_pilot.py --dry-run
uv run python studies/ens_backfill_pilot/fetch_pilot.py
uv run --with eccodes python studies/ens_backfill_pilot/check_pilot.py
```

`--dates 2023-06-26,2023-06-28` replaces the drawn dates, `--workers` sets the number of concurrent
connections (8 by default, at most 16), and `--members all` fetches all 51 members. The drawn dates
are saved in `data/studies/ens_backfill_pilot/pilot_dates.json` so that a resumed run uses the same
dates. Delete that file to draw again.

## Output

Each checkpoint file `data/studies/ens_backfill_pilot/control/<date>.npz` holds arrays indexed by
`(variable, member, step)`, then row, then column:

- `values`: decoded `float32` values in ECMWF's units (K, Pa, m s-1, m2 s-2 for `z`, and totals
  accumulated since the forecast start for `tp` in m, and `strd` and `ssrd` in J m-2).
- `raw_x`: the packed 16-bit integers, from which `values` can be rebuilt exactly with
  `reference_value`, `binary_scale` and `decimal_scale`.
- `sha256`, `range_bytes`, `message_offset`, `message_length`, `file_name`: which bytes were
  fetched, and the hash of the fetched range.
- `latitudes`, `longitudes` (-180 to 180), `longitudes_degrees_east` (0 to 360), `variables`,
  `members`, `steps_hours`, and `date`.

## Source

The files are at `s3://us-west-2.opendata.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/`,
read through the proxy `https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/`. The
proxy answers 403 to Python's default user agent, ignores multi-range requests by returning the
whole file, and resets connections at 32 or more concurrent 2 MB requests.
