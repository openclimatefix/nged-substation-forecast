"""Build the demand experiment's ENS rows for a rotating-weekday set of runs, from the archive.

The row set is `fold_a_every5d_v1`.

The fold A frames hold Tuesday runs only, so a demand forecast scored on them is scored on a fixed
weekday at lead 0 to 1 day. This script reads the daily 00:00 UTC ENS runs from the archive and
builds the same columns as `demand_inputs.ens_features`, for:

- validation: every fifth day from 2025-07-01 to 2026-06-30 (73 runs), all 51 members;
- training: every daily run from 2024-04-01 to 2025-06-30, control member only, keeping valid
  times up to 2025-06-30 23:30.

Each run is cut to the 22 series' H3 cells, resampled to the half-hourly grid by production's
`_upsample_nwp_to_half_hourly` (short-wave energy-conserving, long-wave PCHIP, other variables
linear, the precipitation type forward-filled), and cut to valid times from `nwp_init_time + 9 h`
(`power_fcst_init_time`) to 351 h later. Power and the four power lags come from the observed
half-hourly power; a lag is null where it is no longer than the lead. `--check` compares the
2025-07-01 run with the fold A frame's columns.

Output: `data/studies/per_study/stage1_pilot/demand/ens_every5d/val/init=<date>.parquet` and
`train.parquet`.

    uv run python studies/stage1_pilot/demand_build_ens.py [--check]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import demand_inputs as demand
import polars as pl
import stage1_inputs as inputs
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.sources import PER_STUDY_DIR

ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
OUT_DIR: Final[Path] = demand.demand_subfolder(name="ens_every5d")
ARCHIVE_DIR: Final[Path] = (
    PER_STUDY_DIR.parent.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
"""The ENS archive's Delta table directory, one partition folder per initialisation time."""
FIRST_VALIDATION_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_VALIDATION_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
VALIDATION_STRIDE_DAYS: Final[int] = 5
FIRST_TRAINING_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
PUBLICATION_DELAY_HOURS: Final[int] = 9
"""`ml_core.features.NWP_PUBLICATION_DELAY_HOURS`: `power_fcst_init_time` is the run plus this."""
HORIZON_HOURS: Final[float] = 351.0
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "h3_index",
    "ensemble_member",
    "valid_time",
    *(c for c in demand.ENS_COLUMNS.values() if not c.startswith("wind_speed")),
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)


def validation_runs() -> list[datetime]:
    """Return the validation runs: every fifth day from 2025-07-01 to 2026-06-30."""
    runs = []
    run = FIRST_VALIDATION_RUN
    while run <= LAST_VALIDATION_RUN:
        runs.append(run)
        run += timedelta(days=VALIDATION_STRIDE_DAYS)
    return runs


def training_runs() -> list[datetime]:
    """Return every daily run from 2024-04-01 to 2025-06-30."""
    count = (LAST_TRAINING_VALID - FIRST_TRAINING_RUN).days + 1
    return [FIRST_TRAINING_RUN + timedelta(days=i) for i in range(count)]


def partition_path(*, run: datetime) -> Path:
    """Return the archive folder of a run, whose name percent-encodes the init time."""
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    return ARCHIVE_DIR / f"init_time={stamp}"


def run_rows(
    *, run: datetime, series: pl.DataFrame, observed: pl.DataFrame, control_only: bool
) -> pl.DataFrame:
    """Return one run's half-hourly weather, power, and lags for every series.

    Args:
        run: The run's initialisation time (00:00 UTC).
        series: `demand.series_table()`.
        observed: `demand_build_inputs.observed_power(series=series)`.
        control_only: Keep only ensemble member 0.

    Returns:
        Rows keyed like `stage1_inputs.KEY_COLUMNS` with `power`, `WEATHER`, `CALENDAR`, `LAGS`,
        from `power_fcst_init_time` to `HORIZON_HOURS` after it.
    """
    cells = series["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(partition_path(run=run) / "*.parquet")
        .select(SOURCE_COLUMNS)
        .filter(pl.col("h3_index").is_in(cells))
    )
    if control_only:
        native = native.filter(pl.col("ensemble_member") == 0)
    native = (
        native.join(
            series.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
    init = run + timedelta(hours=PUBLICATION_DELAY_HOURS)
    rows = (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind_speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
        )
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=int(HORIZON_HOURS) + 9)
            )
        )
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        .filter(pl.col("lead_hours") <= HORIZON_HOURS + 9.0)
    )
    rows = rows.join(observed, on=["time_series_id", "valid_time"], how="left").with_columns(
        *[
            pl.when(pl.col("lead_hours") >= hours).then(None).otherwise(pl.col(name)).alias(name)
            for name, hours in demand.LAG_HOURS.items()
        ],
        *demand.calendar_columns(),
    )
    return rows.select(
        *inputs.KEY_COLUMNS,
        "power",
        *[
            pl.col(source).alias(name)
            for name, source in demand.ENS_COLUMNS.items()
            if name not in ("wind_speed_10m", "wind_speed_100m")
        ],
        "wind_speed_10m",
        "wind_speed_100m",
        *demand.CALENDAR,
        *demand.LAGS,
        "lead_hours",
    )


def check_against_frame(*, series: pl.DataFrame, observed: pl.DataFrame) -> list[str]:
    """Compare the 2025-07-01 run with the fold A frame's rows for the same run.

    Args:
        series: `demand.series_table()`.
        observed: Observed power with lags.

    Returns:
        Report lines.
    """
    mine = run_rows(run=FIRST_VALIDATION_RUN, series=series, observed=observed, control_only=False)
    frame = pl.read_parquet(inputs.FRAMES_DIR / "val" / "init=2025-07-01.parquet").filter(
        pl.col("time_series_id").is_in(series["time_series_id"].to_list())
    )
    theirs = demand.ens_features(frame=frame)
    keys = ["time_series_id", "ensemble_member", "valid_time"]
    joined = mine.join(theirs, on=keys, how="full", suffix="_frame", coalesce=True)
    both = joined.filter(
        pl.col("power_fcst_init_time").is_not_null()
        & pl.col("power_fcst_init_time_frame").is_not_null()
    )
    lines = [
        "## New extraction against the fold A frame, run of 2025-07-01",
        "",
        f"- Rows: new {mine.height:,}, frame {theirs.height:,}; in both {both.height:,}.",
        "",
        "| column | largest absolute difference | mean absolute difference |",
        "| --- | --- | --- |",
    ]
    for column in (*demand.WEATHER, *demand.LAGS[:2]):
        diff = (both[column].cast(pl.Float64) - both[f"{column}_frame"].cast(pl.Float64)).abs()
        lines.append(f"| {column} | {diff.max():.4g} | {diff.mean():.4g} |")
    return [*lines, ""]


def balance_table(*, directory: Path, title: str) -> list[str]:
    """Return the share of cases by local weekday of the valid time and lead band.

    Args:
        directory: A folder of `init=<date>.parquet` files with `valid_time` and `lead_hours`.
        title: The table's heading.

    Returns:
        Report lines: for each lead band, the percentage of its cases on each weekday.
    """
    frames = [
        pl.read_parquet(path, columns=["valid_time", "power_fcst_init_time"])
        .filter(pl.col("valid_time") < datetime(2026, 7, 1, tzinfo=UTC))
        .select(
            weekday=pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.weekday(),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
        for path in sorted(directory.glob("init=*.parquet"))
    ]
    cases = pl.concat(frames)
    band = pl.lit("none")
    for name, low, high in reversed(inputs.LEAD_BANDS):
        band = (
            pl.when((pl.col("lead_hours") >= low) & (pl.col("lead_hours") < high))
            .then(pl.lit(name))
            .otherwise(band)
        )
    shares = (
        cases.with_columns(band=band)
        .group_by("band", "weekday")
        .agg(n=pl.len())
        .with_columns(share=100.0 * pl.col("n") / pl.col("n").sum().over("band"))
    )
    lines = [
        f"## {title}",
        "",
        "Percentage of each lead band's cases on each local weekday of the valid time.",
        "",
        "| lead band | Mon | Tue | Wed | Thu | Fri | Sat | Sun |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, _, _ in inputs.LEAD_BANDS:
        part = shares.filter(pl.col("band") == name)
        cells = [f"{part.filter(pl.col('weekday') == d)['share'].sum():.0f}" for d in range(1, 8)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def main() -> None:
    """Build the validation and training rows, skipping files that exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--balance", action="store_true")
    args = parser.parse_args()
    if args.balance:
        report = [
            f"# Row set {ROW_SET_ID}: weekday balance",
            "",
            *balance_table(directory=OUT_DIR / "val", title=f"{ROW_SET_ID} (73 runs)"),
            *balance_table(
                directory=demand.INPUTS_DIR / "ens_features" / "val",
                title="fold_a_tuesdays_v1 (53 runs)",
            ),
        ]
        (OUT_DIR / "balance.md").write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return
    series = demand.series_table()
    observed = pl.read_parquet(demand.INPUTS_DIR / "observed.parquet")
    if args.check:
        report = check_against_frame(series=series, observed=observed)
        print("\n".join(report))
        return
    (OUT_DIR / "val").mkdir(parents=True, exist_ok=True)
    for run in validation_runs():
        target = OUT_DIR / "val" / f"init={run:%Y-%m-%d}.parquet"
        if target.exists():
            continue
        started = time.monotonic()
        rows = run_rows(run=run, series=series, observed=observed, control_only=False)
        rows.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(
            f"{target.name}: {rows.height:,} rows in {time.monotonic() - started:.0f} s", flush=True
        )
    train_path = OUT_DIR / "train.parquet"
    if not train_path.exists():
        pieces = []
        for run in training_runs():
            started = time.monotonic()
            pieces.append(
                run_rows(run=run, series=series, observed=observed, control_only=True).filter(
                    pl.col("valid_time") <= LAST_TRAINING_VALID
                )
            )
            if len(pieces) % 20 == 0:
                print(
                    f"training run {run:%Y-%m-%d} ({time.monotonic() - started:.1f} s)", flush=True
                )
        pl.concat(pieces).write_parquet(train_path.with_suffix(".tmp"))
        train_path.with_suffix(".tmp").rename(train_path)


if __name__ == "__main__":
    main()
