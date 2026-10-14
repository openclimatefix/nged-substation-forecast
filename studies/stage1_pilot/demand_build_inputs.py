"""Build the demand experiment's inputs once, and run the checks that must pass before any fit.

Writes under `data/studies/per_study/stage1_pilot/demand/inputs/`:

- `series.parquet`: the 22 series with label, group, and capacity (coordinates included; private).
- `era5_features.parquet`: ERA5 weather at each series' nearest cell on the half-hourly grid.
- `observed.parquet`: cleaned power on the half-hourly grid from 2019-09, with the four lags.
- `ens_features/train.parquet`, `ens_features/val/init=<date>.parquet`: the fold A frame rows with
  the same columns computed from ENS weather.
- `checks.md`: the checks, printed from the data. The script stops if one fails.

    uv run python studies/stage1_pilot/demand_build_inputs.py
"""

import time
from pathlib import Path
from typing import Final

import demand_inputs as demand
import numpy as np
import polars as pl
import stage1_inputs as inputs
from studies.power import scan_power
from studies.solar import cos_zenith_hour_mean

INPUT_DIR: Final[Path] = demand.INPUTS_DIR
CORRELATION_FLOOR: Final[float] = 0.9
"""ENS member 0 at under 36 h against ERA5: weather columns must correlate at least this much."""


def observed_power(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power in MW on the half-hourly grid with the lagged values.

    Args:
        series: `series_table()`.

    Returns:
        One row per observed `(time_series_id, valid_time)` from the ERA5 era to the end of
        the data, with `power` and a `power_lag_<h>h` column, null where the lagged half-hour has
        no observation.
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(series["time_series_id"].to_list()),
            pl.col("time") >= inputs.STAGE1_TRAIN_START,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .filter(pl.col("power").is_finite())
    )
    frame = power
    for name, hours in demand.LAG_HOURS.items():
        lagged = power.select(
            "time_series_id",
            (pl.col("valid_time") + pl.duration(hours=int(hours))).alias("valid_time"),
            pl.col("power").alias(name),
        )
        frame = frame.join(lagged, on=["time_series_id", "valid_time"], how="left")
    return frame.sort("time_series_id", "valid_time")


def check_diurnal(*, series: pl.DataFrame, hourly: pl.DataFrame) -> list[str]:
    """Check ERA5's hourly `ssrd` against the solar zenith: is the label shifted?

    An hourly accumulation labelled `T` is the total over the hour ending at `T`. If the label is
    right, `ssrd` correlates best with the mean top-of-atmosphere cosine over the hour ending at
    `T`, rather than the hour before or after.

    Args:
        series: `series_table()`.
        hourly: `demand.era5_hourly(series=series)`.

    Returns:
        Report lines.

    Raises:
        ValueError: If the unshifted label does not fit best.
    """
    row = series.row(0, named=True)
    one = hourly.filter(pl.col("time_series_id") == row["time_series_id"]).sort("time")
    day = one.filter(pl.col("ssrd") > 0)
    correlations = {}
    for shift in (-1, 0, 1):
        cosine = cos_zenith_hour_mean(
            stamps=one["time"].dt.offset_by(f"{shift}h"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        correlations[shift] = float(
            np.corrcoef(one["ssrd"].clip(lower_bound=0.0).to_numpy(), cosine)[0, 1]
        )
    best = max(correlations, key=lambda k: correlations[k])
    june = one.filter(pl.col("time").dt.month() == 6).with_columns(hour=pl.col("time").dt.hour())
    profile = (
        june.group_by("hour")
        .agg(mean=pl.col("ssrd").clip(lower_bound=0.0).mean() / 3600.0)
        .sort("hour")
    )
    peak_hour = int(profile.sort("mean", descending=True)["hour"][0])
    lines = [
        "## ERA5 `ssrd` against the solar zenith",
        "",
        (
            "- Correlation of hourly `ssrd` (clipped at zero) with the top-of-atmosphere cosine of "
            "the zenith, mean over the hour ending at the label shifted by -1, 0, +1 h (one "
            "series): " + ", ".join(f"{k:+d} h: {v:.4f}" for k, v in correlations.items()) + "."
        ),
        f"- Hour of the June peak of the mean `ssrd` (UTC label, hour-ending): {peak_hour}.",
        (
            f"- Lowest `ssrd` in the box files: {one['ssrd'].min():.2f} J m-2 (float noise below "
            f"zero, clipped); hours with `ssrd` above zero: {day.height:,}."
        ),
    ]
    if best != 0:
        msg = f"ERA5 ssrd label looks shifted: correlations by shift {correlations}"
        raise ValueError(msg)
    lines.append("- The unshifted label fits best, so the label is not shifted.")
    return [*lines, ""]


def check_ens(*, ens: pl.DataFrame, era5: pl.DataFrame) -> list[str]:
    """Compare ENS member 0 at short lead with ERA5 at the same valid time.

    Args:
        ens: Rows of several validation runs.
        era5: `demand.era5_features(series=series)`.

    Returns:
        Report lines.

    Raises:
        ValueError: If a continuous weather column correlates below the floor.
    """
    lines = [
        "## ENS rows against ERA5 at the same valid time",
        "",
        (
            "Control-member rows with lead under 36 h; the columns must correlate at least "
            f"{CORRELATION_FLOOR}."
        ),
        "",
        "| column | rows | correlation | mean ENS | mean ERA5 |",
        "| --- | --- | --- | --- | --- |",
    ]
    joined = ens.filter(
        (pl.col("ensemble_member") == 0)
        & ((pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_hours() < 36)
    ).join(era5, on=["time_series_id", "valid_time"], suffix="_era5", how="inner")
    for column in demand.WEATHER:
        pair = joined.select(column, f"{column}_era5").drop_nulls().drop_nans()
        corr = float(np.corrcoef(pair[column], pair[f"{column}_era5"])[0, 1])
        lines.append(
            f"| {column} | {pair.height:,} | {corr:.3f} | {pair[column].mean():.4g} | "
            f"{pair[f'{column}_era5'].mean():.4g} |"
        )
        if column not in ("precipitation", "precipitation_type") and corr < CORRELATION_FLOOR:
            msg = f"{column} correlates {corr:.3f} with ERA5"
            raise ValueError(msg)
    return [*lines, ""]


def check_lags(*, ens: pl.DataFrame, observed: pl.DataFrame) -> list[str]:
    """Check the frames' lag columns against the observed power at the lagged time.

    Args:
        ens: Rows of several validation runs.
        observed: `observed_power(series=series)`.

    Returns:
        Report lines.

    Raises:
        ValueError: If a lag column holds a value that differs from the observed power, or is
            present where the lag is no longer than the lead.
    """
    lines = ["## Power lags in the fold A frames", ""]
    control = ens.filter(pl.col("ensemble_member") == 0).with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    joined = control.join(
        observed.select("time_series_id", "valid_time", *[f"{c}" for c in demand.LAGS]),
        on=["time_series_id", "valid_time"],
        how="left",
        suffix="_obs",
    )
    for column, hours in demand.LAG_HOURS.items():
        present = joined.filter(
            pl.col(column).is_not_null() & pl.col(f"{column}_obs").is_not_null()
        )
        gap = (present[column] - present[f"{column}_obs"]).abs().max()
        leaking = joined.filter(pl.col(column).is_not_null() & (pl.col("lead_hours") >= hours))
        lines.append(
            f"- `{column}`: largest difference from the observed power {gap:.4g} MW over "
            f"{present.height:,} rows; rows with the lag present at a lead of at least "
            f"{hours:.0f} h: {leaking.height}."
        )
        if (gap is not None and gap > 1e-3) or leaking.height:  # ty: ignore[unsupported-operator]
            msg = f"{column} disagrees with the observed power or leaks"
            raise ValueError(msg)
    return [*lines, ""]


def main() -> None:
    """Build and check the inputs, skipping files that exist."""
    (INPUT_DIR / "ens_features" / "val").mkdir(parents=True, exist_ok=True)
    series = demand.series_table()
    series.write_parquet(INPUT_DIR / "series.parquet")
    ids = series["time_series_id"].to_list()
    report = ["# Demand input checks", ""]
    report += check_diurnal(series=series, hourly=demand.era5_hourly(series=series))
    print("\n".join(report), flush=True)

    era5_path = INPUT_DIR / "era5_features.parquet"
    if not era5_path.exists():
        started = time.monotonic()
        features = demand.era5_features(series=series)
        features.write_parquet(era5_path.with_suffix(".tmp"))
        era5_path.with_suffix(".tmp").rename(era5_path)
        print(f"era5 features: {features.height:,} rows in {time.monotonic() - started:.0f} s")
    observed_path = INPUT_DIR / "observed.parquet"
    if not observed_path.exists():
        observed_power(series=series).write_parquet(observed_path.with_suffix(".tmp"))
        observed_path.with_suffix(".tmp").rename(observed_path)
    era5 = pl.read_parquet(era5_path)
    observed = pl.read_parquet(observed_path)

    train_path = INPUT_DIR / "ens_features" / "train.parquet"
    if not train_path.exists():
        frame = pl.read_parquet(inputs.FRAMES_DIR / "train.parquet").filter(
            pl.col("time_series_id").is_in(ids)
        )
        demand.ens_features(frame=frame).write_parquet(train_path.with_suffix(".tmp"))
        train_path.with_suffix(".tmp").rename(train_path)
    for path in sorted((inputs.FRAMES_DIR / "val").glob("init=*.parquet")):
        target = INPUT_DIR / "ens_features" / "val" / path.name
        if target.exists():
            continue
        frame = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(ids))
        demand.ens_features(frame=frame).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{path.name}: {frame.height:,} rows", flush=True)

    val_files = sorted((INPUT_DIR / "ens_features" / "val").glob("init=*.parquet"))
    sample = pl.concat([pl.read_parquet(p) for p in val_files[::6]])
    report += check_ens(ens=sample, era5=era5)
    report += check_lags(ens=sample, observed=observed)
    spans = observed.group_by("time_series_id").agg(first=pl.col("valid_time").min(), rows=pl.len())
    report += [
        "## Observed power",
        "",
        (
            f"- Series: {spans.height}. Rows per series: {spans['rows'].min():,} to "
            f"{spans['rows'].max():,}; the first observation falls between "
            f"{spans['first'].min():%Y-%m} and {spans['first'].max():%Y-%m}."
        ),
        "",
    ]
    (INPUT_DIR / "checks.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
