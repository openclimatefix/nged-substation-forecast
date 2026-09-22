"""Measure how late NGED's half-hourly power stamps are, on each side of the feed's correction.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**A stamped half-hour can be checked against the sun, because the sun's position is known exactly
and a solar farm's output follows it.** Three measurements do that here, and each fails
differently, so their agreement is worth more than any one of them alone.

The **clear-day centroid** takes the power-weighted mean time of each clear day's output and
compares it with solar noon. A symmetric day peaks at solar noon, so a centroid half an hour later
says the stamps are half an hour late. An array facing away from due south also moves the centroid,
which is why the measurement is reported per site: a stamping fault is common to every meter, an
azimuth error is not.

The **generating window** takes the midpoint between the first and last half-hour carrying output.
The window's edges are set by the horizon rather than by the array's orientation, so this survives
an azimuth error that would move the centroid. Its weakness is the threshold defining "generating",
so the threshold is swept.

The **lagged correlation** shifts the power series against the satellite's global irradiance and
reports the shift that maximises the correlation. Irradiance and power are different quantities
from different measurement systems, so a shared clock fault is the only thing that would align them
at a non-zero lag.

**This measures the repair, not the fault.** `PowerTimeSeries.correct_late_timestamps` moves the
late stamps at ingestion, so the stored table is already repaired and every measurement below should
read close to zero on *both* sides of the correction instant. The era split is what makes that
checkable: an offset that reappears on the `before` side means the repair has stopped matching the
feed — either NGED has republished the early readings with corrected stamps, in which case the
ingest is now shifting rows that need no shift, or the fault did not stop where NGED reported.

Run it with `uv run --no-project --with polars --with numpy --with pandas --with pvlib
--with deltalake python scripts/experiments/beam_diffuse_split/stamp_alignment.py`.
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_dataset import (
    CAMS_PATH,
    REPO_DATA_DIR,
    _pv_sites,
)
from contracts.power_schemas import POWER_TIMESTAMPS_CORRECTED_BEFORE

_LOG: Final[logging.Logger] = logging.getLogger("stamp_alignment")

POWER_DELTA: Final[str] = str(REPO_DATA_DIR / "NGED" / "power_time_series.delta")

MINUTES_PER_HOUR: Final[int] = 60

CLEAR_DAY_RATIO: Final[float] = 0.90
"""How close to the clear-sky total a day must run before it counts as clear."""

CLEAR_DAY_WORST_RATIO: Final[float] = 0.80
"""How close to clear-sky the dimmest bright half-hour must be, which rejects a day carrying one
cloud band across an otherwise clear sky."""

MIN_BRIGHT_HALF_HOURS: Final[int] = 8
"""How many half-hours above the clear-sky floor a day needs before its centroid is stable."""

CLEAR_SKY_FLOOR_W_M2: Final[float] = 50.0
"""The clear-sky global irradiance above which a half-hour counts towards the day's clearness."""

GENERATING_THRESHOLDS: Final[tuple[float, ...]] = (0.001, 0.01, 0.05, 0.10)
"""Fractions of the day's peak output that each define "generating". Sweeping the threshold
shows the window's edge is not what sets the answer."""

LAG_GRID_MINUTES: Final[tuple[int, ...]] = (-60, -30, 0, 30, 60)
"""The stamp shifts the lagged correlation is evaluated at."""

MIN_ROWS_FOR_A_CORRELATION: Final[int] = 200
"""How many joined bright hours an era needs before its correlation is worth reporting."""

STAMPS_PER_HOUR: Final[int] = 2
"""How many half-hourly readings a complete hour holds. An hour missing one is dropped, so a gap
cannot pass as a dim hour and drag the correlation."""


def _solar_noons(*, latitude: float, longitude: float, days: list) -> dict[str, float]:
    """Return each day's solar noon in minutes after midnight UTC.

    Solar noon is the minute of least apparent zenith on a one-minute grid, which needs no
    equation-of-time algebra of its own.

    Args:
        latitude: The site's latitude in degrees.
        longitude: The site's longitude in degrees.
        days: The dates to evaluate.

    Returns:
        Solar noon per ISO date string, in minutes after midnight UTC.
    """
    noons: dict[str, float] = {}
    for day in days:
        grid = pl.datetime_range(
            datetime(day.year, day.month, day.day, 10, 0, tzinfo=UTC),
            datetime(day.year, day.month, day.day, 14, 0, tzinfo=UTC),
            interval="1m",
            eager=True,
            time_zone="UTC",
        )
        position = pvlib.solarposition.get_solarposition(
            grid.to_pandas(), latitude=latitude, longitude=longitude
        )
        best = int(np.argmin(position["apparent_zenith"].to_numpy()))
        noons[day.isoformat()] = float(grid[best].hour * MINUTES_PER_HOUR + grid[best].minute)
    return noons


def _power_for(*, time_series_id: int) -> pl.DataFrame:
    """Read one meter's half-hourly power, labelled with its era and minute of day."""
    return (
        pl.scan_delta(POWER_DELTA)
        .filter(pl.col("time_series_id") == time_series_id)
        .select("time", "power")
        .collect()
        .with_columns(
            day=pl.col("time").dt.date(),
            # `dt.hour()` is Int8, and Int8 cannot hold hour * 60. Widen before multiplying,
            # or every stamp past 02:00 wraps and the measurement reads plausible nonsense.
            minute_of_day=pl.col("time").dt.hour().cast(pl.Int32) * MINUTES_PER_HOUR
            + pl.col("time").dt.minute().cast(pl.Int32),
            era=pl.when(pl.col("time") < pl.lit(POWER_TIMESTAMPS_CORRECTED_BEFORE))
            .then(pl.lit("before"))
            .otherwise(pl.lit("after")),
        )
    )


def _clear_days(*, cams: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return the clear days at one site, judged by the satellite's own clear-sky column."""
    return (
        cams.filter(
            (pl.col("site") == site) & (pl.col("clear_sky_ghi_w_m2") > CLEAR_SKY_FLOOR_W_M2)
        )
        .with_columns(day=pl.col("time").dt.date())
        .group_by("day")
        .agg(
            bright=pl.len(),
            ratio=pl.col("ghi_w_m2").sum() / pl.col("clear_sky_ghi_w_m2").sum(),
            worst=(pl.col("ghi_w_m2") / pl.col("clear_sky_ghi_w_m2")).min(),
        )
        .filter(
            (pl.col("ratio") > CLEAR_DAY_RATIO)
            & (pl.col("worst") > CLEAR_DAY_WORST_RATIO)
            & (pl.col("bright") >= MIN_BRIGHT_HALF_HOURS)
        )
        .sort("day")
    )


def _with_offset(*, frame: pl.DataFrame, noons: dict[str, float], column: str) -> pl.DataFrame:
    """Turn a minute-of-day column into minutes after that day's solar noon."""
    return (
        frame.with_columns(noon=pl.col("day").cast(pl.Utf8).replace_strict(noons, default=None))
        .drop_nulls("noon")
        .with_columns(offset=pl.col(column) - pl.col("noon"))
    )


def _centroids(
    *, power: pl.DataFrame, clear: pl.DataFrame, noons: dict[str, float]
) -> pl.DataFrame:
    """Return each clear day's power-weighted centroid, in minutes after solar noon."""
    centroid = (
        power.join(clear.select("day"), on="day", how="semi")
        .filter(pl.col("power") > 0)
        .group_by("day", "era")
        .agg(centroid=(pl.col("minute_of_day") * pl.col("power")).sum() / pl.col("power").sum())
    )
    return _with_offset(frame=centroid, noons=noons, column="centroid")


def _windows(
    *, power: pl.DataFrame, clear: pl.DataFrame, noons: dict[str, float], threshold: float
) -> pl.DataFrame:
    """Return each clear day's generating-window midpoint, in minutes after solar noon."""
    midpoint = (
        power.join(clear.select("day"), on="day", how="semi")
        .with_columns(peak=pl.col("power").max().over("day"))
        .filter(pl.col("power") > pl.col("peak") * threshold)
        .group_by("day", "era")
        .agg(midpoint=(pl.col("minute_of_day").min() + pl.col("minute_of_day").max()) / 2)
    )
    return _with_offset(frame=midpoint, noons=noons, column="midpoint")


def _lags(*, power: pl.DataFrame, cams: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return the correlation with global irradiance at each candidate stamp shift."""
    irradiance = cams.filter(pl.col("site") == site).select("time", "ghi_w_m2")
    records: list[dict[str, object]] = []
    for era in ("before", "after"):
        rows = power.filter(pl.col("era") == era).select("time", "power")
        if rows.is_empty():
            continue
        for lag in LAG_GRID_MINUTES:
            # Average onto the irradiance's own hourly grid, period-ending as both series are.
            # Joining half-hourly power to hourly irradiance on an exact stamp would instead
            # compare which half of the readings happened to land on the hour.
            hourly = (
                rows.with_columns(time=pl.col("time").dt.offset_by(f"{lag}m"))
                .sort("time")
                .group_by_dynamic("time", every="1h", label="right", closed="right")
                .agg(power=pl.col("power").mean(), stamps=pl.len())
                .filter(pl.col("stamps") == STAMPS_PER_HOUR)
            )
            joined = hourly.join(irradiance, on="time", how="inner").filter(pl.col("ghi_w_m2") > 0)
            if joined.height < MIN_ROWS_FOR_A_CORRELATION:
                continue
            records.append(
                {
                    "era": era,
                    "lag": lag,
                    "correlation": float(
                        np.corrcoef(joined["power"].to_numpy(), joined["ghi_w_m2"].to_numpy())[0, 1]
                    ),
                }
            )
    return pl.DataFrame(records)


def main() -> int:
    """Measure the stamp offset on both sides of the correction, and print the three readings."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sites = _pv_sites().sort("site")
    cams = pl.read_parquet(CAMS_PATH)

    print(
        f"Correction instant, as NGED reported it: "
        f"{POWER_TIMESTAMPS_CORRECTED_BEFORE:%Y-%m-%d %H:%M} UTC"
    )
    print(
        "Every offset below is minutes after solar noon. A feed stamped as its contract states"
        " reads zero; a feed half an hour late reads +30.\n"
    )

    centroid_frames: list[pl.DataFrame] = []
    window_frames: list[pl.DataFrame] = []
    lag_frames: list[pl.DataFrame] = []

    for row in sites.iter_rows(named=True):
        site = row["site"]
        clear = _clear_days(cams=cams, site=site)
        power = _power_for(time_series_id=row["time_series_id"])
        noons = _solar_noons(
            latitude=row["latitude"], longitude=row["longitude"], days=clear["day"].to_list()
        )
        centroid_frames.append(
            _centroids(power=power, clear=clear, noons=noons).with_columns(site=pl.lit(site))
        )
        window_frames.extend(
            _windows(power=power, clear=clear, noons=noons, threshold=threshold).with_columns(
                site=pl.lit(site), threshold=pl.lit(threshold)
            )
            for threshold in GENERATING_THRESHOLDS
        )
        lag_frames.append(_lags(power=power, cams=cams, site=site).with_columns(site=pl.lit(site)))
        _LOG.info("site %s: %d clear days", site, clear.height)

    centroids = pl.concat(centroid_frames)
    windows = pl.concat(window_frames)
    lags = pl.concat(lag_frames)

    print("\n## Clear-day power-weighted centroid\n")
    print("| Site | Before the correction | Clear days | After the correction | Clear days |")
    print("|---|---|---|---|---|")
    for site in sorted(centroids["site"].unique().to_list()):
        cells: list[str] = []
        for era in ("before", "after"):
            rows = centroids.filter((pl.col("site") == site) & (pl.col("era") == era))
            cells.append(
                f"{rows['offset'].median():+.1f} | {rows.height}" if rows.height else "— | 0"
            )
        print(f"| {site} | {cells[0]} | {cells[1]} |")

    for era in ("before", "after"):
        rows = centroids.filter(pl.col("era") == era)
        if rows.is_empty():
            continue
        per_site = rows.group_by("site").agg(median=pl.col("offset").median())["median"]
        print(
            f"\n{era.capitalize()} the correction: {rows.height} clear site-days,"
            f" pooled median {rows['offset'].median():+.1f} minutes,"
            f" per-site medians from {per_site.min():+.1f} to {per_site.max():+.1f}."
        )

    print("\n## Generating-window midpoint\n")
    print("| Threshold, as a fraction of the day's peak | Before | After |")
    print("|---|---|---|")
    for threshold in GENERATING_THRESHOLDS:
        cells = []
        for era in ("before", "after"):
            rows = windows.filter((pl.col("threshold") == threshold) & (pl.col("era") == era))
            cells.append(f"{rows['offset'].median():+.1f}" if rows.height else "—")
        print(f"| {threshold:.3f} | {cells[0]} | {cells[1]} |")

    print("\n## Stamp shift that maximises the correlation with global irradiance\n")
    print("| Site | Before | After |")
    print("|---|---|---|")
    for site in sorted(lags["site"].unique().to_list()):
        cells = []
        for era in ("before", "after"):
            rows = lags.filter((pl.col("site") == site) & (pl.col("era") == era))
            if rows.is_empty():
                cells.append("—")
                continue
            best = rows.sort("correlation", descending=True).row(0, named=True)
            cells.append(f"{best['lag']:+d} min (r = {best['correlation']:.4f})")
        print(f"| {site} | {cells[0]} | {cells[1]} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
