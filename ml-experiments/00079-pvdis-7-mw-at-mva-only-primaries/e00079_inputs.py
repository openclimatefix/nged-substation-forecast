"""Load experiment 00039's frame and build the per-Primary season-year windows and farm tables."""

from dataclasses import dataclass

import e00079_common as common
import e00079_fit as fit
import numpy as np
import polars as pl


@dataclass(frozen=True)
class YearWindow:
    """One Primary's April to September daylight rows in one season-year.

    Attributes:
        site: The Primary label.
        year: The season-year.
        time: Period-ending time per row.
        power: Observed power, MW (the magnitude for an MVA-only Primary).
        cells: Calendar-cell code per row.
        nuisance: Heating degree-days and three cloud-by-hour-band columns, `(n, 4)`.
        half_hour: Local half-hour of the day, 0 to 47.
        days: Integer code of the local date.
        shape: The fleet unit shape at the Primary.
    """

    site: str
    year: int
    time: np.ndarray
    power: np.ndarray
    cells: np.ndarray
    nuisance: np.ndarray
    half_hour: np.ndarray
    days: np.ndarray
    shape: np.ndarray

    def midday_scale(self, *, mask: np.ndarray) -> float:
        """Return the median absolute power at local 11:00 to 13:30 over the selected rows."""
        lo, hi = common.MIDDAY_HALF_HOURS
        midday = mask & (self.half_hour >= lo) & (self.half_hour < hi)
        return float(np.median(np.abs(self.power[midday])))


def load_prepared() -> pl.DataFrame:
    """Return the prepared daylight rows of the frame, after the row-count check."""
    frame = pl.read_parquet(common.FRAME_PATH)
    assert frame.height == common.FRAME_ROWS, f"frame has {frame.height} rows"
    return fit.prepare(frame=frame)


def year_windows(*, prepared: pl.DataFrame, primary: str) -> list[YearWindow]:
    """Build every summer season-year window of one Primary with enough rows."""
    shapes = pl.read_parquet(common.SHAPES_PATH).filter(pl.col("site") == primary).drop("site")
    rows = prepared.filter(
        (pl.col("site") == primary)
        & pl.col("power").is_not_null()
        & pl.col("hdd").is_not_null()
        & pl.col("month").is_in(common.SUMMER_MONTHS)
    ).join(shapes, on="time", how="inner")
    windows = []
    for year in sorted(rows["season_year"].unique().to_list()):
        block = rows.filter(pl.col("season_year") == year).sort("time")
        if block.height < common.MIN_WINDOW_ROWS:
            continue
        half_hour = block["half_hour"].to_numpy()
        nuisance = np.column_stack(
            [block["hdd"].to_numpy().astype(float)]
            + [
                block["cloud"].to_numpy() * ((half_hour >= lo) & (half_hour < hi))
                for lo, hi in common.HOUR_BANDS
            ]
        )
        windows.append(
            YearWindow(
                site=primary,
                year=year,
                time=block["time"].to_numpy(),
                power=block["power"].to_numpy().astype(float),
                cells=(block["month"] * 1000 + block["half_hour"] * 10 + block["off"]).to_numpy(),
                nuisance=nuisance,
                half_hour=half_hour,
                days=block["date"].cast(pl.Int32).to_numpy(),
                shape=fit.fleet_shape(
                    g_e=block["gE"].to_numpy(),
                    g_s=block["gS"].to_numpy(),
                    g_w=block["gW"].to_numpy(),
                ),
            )
        )
    return windows


def farm_series(*, prepared: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return each farm's sorted times and normalised output (99th percentile 1, floored at 0)."""
    out = {}
    for farm in common.FARMS:
        rows = prepared.filter((pl.col("site") == farm) & pl.col("power").is_not_null()).sort(
            "time"
        )
        p99 = float(np.nanpercentile(rows["power"].to_numpy(), 99))
        out[farm] = (
            rows["time"].to_numpy(),
            np.clip(rows["power"].to_numpy().astype(float), 0.0, None) / p99,
        )
    return out


def aligned_injection(
    *, window: YearWindow, farm: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the window rows that have a farm reading, and the farm's normalised output there.

    Returns:
        A boolean mask over the window's rows, and the normalised output on the masked rows.
    """
    times, values = farm
    position = np.clip(np.searchsorted(times, window.time), 0, len(times) - 1)
    has = times[position] == window.time
    return has, values[position][has]
