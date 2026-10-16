"""Short check of MOGREPS-UK against the CAMS anomaly, outside the validation year.

MOGREPS-UK (3 members, hourly runs, `shortwave_total`, instantaneous, hourly steps to 126 h) has
about 7 weeks of archive, so nothing can be calibrated or fitted. For each daily ENS origin (the 00
UTC run plus 9 hours) from 2026-08-27, the 06 UTC MOGREPS-UK run of the same day is used (3 hours
old at the origin, as for UKV `t54`). Every anomaly is the source's `u` minus the pooled
month-by-half-hour climatology that experiment 00046 built from CAMS (MOGREPS-UK) or from ENS (ENS)
over the training window; nothing is calibrated. The equal-weight blend is the mean of the two
anomalies. Reference: the CAMS anomaly, which exists only up to 2026-09-11 00:00 UTC; after that
date the script reports only the agreement between the MOGREPS-UK and ENS anomalies, which is not a
skill score. Intervals resample whole origin days (10,000 resamples).

    uv run python ml-experiments/00053-*/e00053_mogreps.py
"""

import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import Final

import e00053_common as common
import icechunk
import numpy as np
import polars as pl
import zarr
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

FIRST_ORIGIN_DAY: Final[datetime] = datetime(2026, 8, 27, tzinfo=UTC)
MOGREPS_RUN_HOUR: Final[int] = 6
SHORT_WAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
N_RESAMPLES: Final[int] = 10_000


def _array(group: zarr.Group, name: str) -> zarr.Array:
    """Return an array the store layout guarantees exists.

    Args:
        group: The opened store root.
        name: The array name.

    Returns:
        The array.

    Raises:
        TypeError: If the member is not an array.
    """
    member = group[name]
    if not isinstance(member, zarr.Array):
        message = f"{name} is not an array"
        raise TypeError(message)
    return member


def mogreps_u(*, days: list[datetime], series: pl.DataFrame) -> pl.DataFrame:
    """Return the member-mean MOGREPS-UK `u` for the 06 UTC run of each day at each site.

    Args:
        days: Origin days (00:00 UTC).
        series: `common.series_table()`.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `u_mog`.
    """
    repository = icechunk.Repository.open(
        icechunk.local_filesystem_storage(str(common.MOGREPS_DIR))
    )
    group = zarr.open_group(repository.readonly_session(branch="main").store, mode="r")
    latitude = np.asarray(_array(group, "clat")[:], dtype=np.float64)
    longitude = np.asarray(_array(group, "clon")[:], dtype=np.float64)
    cells = []
    for row in series.iter_rows(named=True):
        dx = (longitude - row["longitude"]) * np.cos(np.radians(row["latitude"]))
        cells.append(int(np.hypot(dx, latitude - row["latitude"]).argmin()))
    order = np.argsort(cells)
    sorted_cells = np.array(cells)[order]
    init = np.asarray(_array(group, "init_time")[:])
    status = np.asarray(_array(group, "status")[:])
    step_minutes = np.asarray(_array(group, "step")[:])
    hourly = np.flatnonzero((step_minutes % 60 == 0) & (step_minutes > 0))
    array = _array(group, "shortwave_total")
    pieces = []
    for day in days:
        run = day + timedelta(hours=MOGREPS_RUN_HOUR)
        slots = np.flatnonzero(init == int(run.timestamp()))
        if len(slots) == 0 or status[slots[0]] != 1:
            continue
        block = np.asarray(array.oindex[int(slots[0]), :, hourly, sorted_cells], dtype=np.float64)
        origin = day + common.PUBLICATION_DELAY
        node_times = (
            np.datetime64(run.replace(tzinfo=None), "us")
            + (step_minutes[hourly] // 60).astype("timedelta64[h]")
        ).astype("datetime64[us]")
        stamps = pl.datetime_range(
            origin + timedelta(minutes=30),
            node_times[-1].astype(datetime).replace(tzinfo=UTC),
            interval="30m",
            time_zone="UTC",
            eager=True,
        )
        for position, row in enumerate(series.iter_rows(named=True)):
            column = int(np.flatnonzero(order == position)[0])
            members = []
            for member in range(block.shape[0]):
                ghi = common.half_hour_ghi_from_instants(
                    node_times=node_times,
                    node_ghi=block[member][:, column],
                    stamps=stamps,
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                )
                members.append(ghi)
            u = common.u_from_half_hour_ghi(
                stamps=stamps,
                ghi=np.stack(members),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            pieces.append(
                pl.DataFrame(
                    {"valid_time": stamps, "u_mog": u.mean(axis=0).astype(np.float32)}
                ).with_columns(
                    site=pl.lit(row["site"]),
                    power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
                )
            )
    return pl.concat(pieces)


def ens_u(*, days: list[datetime], series: pl.DataFrame) -> pl.DataFrame:
    """Return the member-mean ENS `u` of the 00 UTC run of each day at each site.

    Args:
        days: Run days (00:00 UTC).
        series: `common.series_table()`.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `u_ens`.
    """
    pieces = []
    for day in days:
        stamp = urllib.parse.quote(f"{day:%Y-%m-%d} 00:00:00.000000", safe="")
        folder = common.ENS_ARCHIVE_DIR / f"init_time={stamp}"
        if not folder.exists():
            continue
        native = (
            pl.scan_parquet(folder / "*.parquet")
            .select("h3_index", "ensemble_member", "valid_time", SHORT_WAVE)
            .filter(pl.col("h3_index").is_in(series["h3_res_5"].unique().to_list()))
            .join(
                series.lazy().select("time_series_id", "h3_res_5"),
                left_on="h3_index",
                right_on="h3_res_5",
            )
            .drop("h3_index")
            .with_columns(nwp_init_time=pl.lit(day).cast(pl.Datetime("us", "UTC")))
        )
        coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
        half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
        origin = day + common.PUBLICATION_DELAY
        for row in series.iter_rows(named=True):
            own = half_hourly.filter(
                (pl.col("time_series_id") == row["time_series_id"])
                & (pl.col("valid_time") > origin)
            ).sort("valid_time", "ensemble_member")
            if own.is_empty():
                continue
            times = own["valid_time"].unique(maintain_order=True)
            members = own["ensemble_member"].unique().sort()
            ghi = (
                own[SHORT_WAVE].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
            )
            u = common.u_from_half_hour_ghi(
                stamps=times, ghi=ghi, latitude=row["latitude"], longitude=row["longitude"]
            )
            pieces.append(
                pl.DataFrame(
                    {"valid_time": times, "u_ens": u.mean(axis=0).astype(np.float32)}
                ).with_columns(
                    site=pl.lit(row["site"]),
                    power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
                )
            )
    return pl.concat(pieces)


def correlation(*, x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation of two arrays.

    Args:
        x: First array.
        y: Second array.

    Returns:
        The correlation, or NaN with fewer than 3 points.
    """
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def block_bootstrap(*, rows: pl.DataFrame, columns: list[str], reference: str) -> pl.DataFrame:
    """Return correlation, uncalibrated slope, and MSE per column with day-resampled intervals.

    Args:
        rows: Rows with `origin_day`, the reference column, and the score columns.
        columns: The anomaly columns to score.
        reference: The reference anomaly column.

    Returns:
        One row per column with point values and 95% intervals.
    """
    days = rows["origin_day"].unique().sort().to_list()
    rng = np.random.default_rng(1016)
    by_day = {day: rows.filter(pl.col("origin_day") == day) for day in days}
    out = []
    for column in columns:
        draws = []
        for _ in range(N_RESAMPLES // 10):
            pick = rng.integers(0, len(days), len(days))
            part = pl.concat([by_day[days[i]] for i in pick])
            x, y = part[column].to_numpy(), part[reference].to_numpy()
            draws.append(
                (
                    correlation(x=x, y=y),
                    float((x * y).sum() / (x * x).sum()),
                    float(((y - x) ** 2).mean()),
                )
            )
        draws_array = np.array(draws)
        low, high = np.nanpercentile(draws_array, [2.5, 97.5], axis=0)
        x, y = rows[column].to_numpy(), rows[reference].to_numpy()
        out.append(
            {
                "arm": column,
                "n": rows.height,
                "days": len(days),
                "correlation": correlation(x=x, y=y),
                "correlation_low": float(low[0]),
                "correlation_high": float(high[0]),
                "slope": float((x * y).sum() / (x * x).sum()),
                "mse": float(((y - x) ** 2).mean()),
                "mse_low": float(low[2]),
                "mse_high": float(high[2]),
            }
        )
    return pl.DataFrame(out)


def main() -> None:
    """Run the check and print the tables."""
    series = common.series_table().filter(pl.col("site").is_in(common.VERDICT_SITES))
    last_day = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    days = []
    day = FIRST_ORIGIN_DAY
    while day < last_day:
        days.append(day)
        day += timedelta(days=1)
    mog = mogreps_u(days=days, series=series)
    ens = ens_u(days=days, series=series)
    print(f"MOGREPS-UK rows {mog.height:,}, ENS rows {ens.height:,}", flush=True)
    clim = pl.read_parquet(common.E46_SOLAR / "climatology.parquet").select(
        "site", "month", "hh", "clim_cams", "clim_ens", "daylight"
    )
    cams = pl.read_parquet(common.E46_SOLAR / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    rows = (
        mog.join(ens, on=list(common.KEYS), how="inner")
        .pipe(lambda f: common.with_cell(frame=f, time_column="valid_time"))
        .join(clim, on=["site", "month", "hh"], how="left")
        .join(cams, on=["site", "valid_time"], how="left")
        .filter(pl.col("daylight").fill_null(False))
        .with_columns(
            a_mog=pl.col("u_mog") - pl.col("clim_cams"),
            a_ens=pl.col("u_ens") - pl.col("clim_ens"),
            a_cams=pl.col("u_cams") - pl.col("clim_cams"),
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ),
            origin_day=pl.col("power_fcst_init_time").dt.date(),
        )
        .with_columns(a_blend=(pl.col("a_mog") + pl.col("a_ens")) / 2.0)
    )
    rows = rows.with_columns(
        band=pl.Series(common.band_index(lead_hours=rows["lead_hours"].to_numpy()))
    )
    print("CAMS u present up to", rows.filter(pl.col("u_cams").is_not_null())["valid_time"].max())
    print(
        "Rows by origin day with CAMS:",
        rows.filter(pl.col("a_cams").is_not_null())["origin_day"].n_unique(),
        "days",
    )
    scored = rows.filter(pl.col("a_cams").is_not_null())
    tables = []
    for name, bands in (
        ("0-6 h", [0]),
        ("6-12 h", [1]),
        ("12-24 h", [2]),
        ("24-48 h", [3]),
        ("48-54 h", [4]),
        ("54-120 h", [5]),
        ("0-24 h", [0, 1, 2]),
        ("0-120 h", [0, 1, 2, 3, 4, 5]),
    ):
        part = scored.filter(pl.col("band").is_in(bands))
        if part.height < 100:
            continue
        tables.append(
            block_bootstrap(
                rows=part, columns=["a_ens", "a_mog", "a_blend"], reference="a_cams"
            ).with_columns(band=pl.lit(name))
        )
    table = pl.concat(tables)
    common.write_atomically(frame=table, target=common.RESULTS_DIR / "mogreps_scores.parquet")
    agreement = []
    for name, bands in (
        ("0-24 h", [0, 1, 2]),
        ("24-54 h", [3, 4]),
        ("54-120 h", [5]),
        ("0-120 h", [0, 1, 2, 3, 4, 5]),
    ):
        part = rows.filter(pl.col("band").is_in(bands))
        agreement.append(
            {
                "band": name,
                "n": part.height,
                "days": part["origin_day"].n_unique(),
                "correlation_mog_vs_ens": correlation(
                    x=part["a_mog"].to_numpy(), y=part["a_ens"].to_numpy()
                ),
            }
        )
    agreement_table = pl.DataFrame(agreement)
    common.write_atomically(
        frame=agreement_table, target=common.RESULTS_DIR / "mogreps_agreement.parquet"
    )
    with pl.Config(tbl_rows=100, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        print(table)
        print(agreement_table)
        print(
            "Mean u by source (daylight rows):",
            rows.select(pl.col("u_mog").mean(), pl.col("u_ens").mean(), pl.col("u_cams").mean()),
        )


if __name__ == "__main__":
    main()
