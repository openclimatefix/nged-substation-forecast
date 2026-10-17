"""Load experiment 00039's frame and build the per-Primary, per-season-year windows and farm tables.

A `YearWindow` holds the April to September daylight rows of one Primary in one season-year, with
power, the nuisance columns, the cell codes, the Primary's own plane-of-array fractions, and the
fractions at each farm location, on the rows where all of them exist.
"""

from dataclasses import dataclass

import e00073_common as common
import e00073_fit as fit
import numpy as np
import polars as pl


@dataclass(frozen=True)
class YearWindow:
    """One Primary's summer daylight rows in one season-year.

    Attributes:
        site: The Primary label.
        year: The season-year.
        time: Period-ending time per row.
        power: Observed power, MW.
        cells: Calendar-cell code per row.
        nuisance: Heating degree-days and three cloud-by-hour-band columns, `(n, 4)`.
        clear: True on rows of clear days (day-mean cloud under `CLEAR_CLOUD_MAX`).
        days: Integer code of the local date.
        g_primary: Plane-of-array fractions at the Primary, keyed `"E25"`, `"S25"`, and so on.
        g_farm: The same at each farm location, keyed `(farm, "E25")`.
    """

    site: str
    year: int
    time: np.ndarray
    power: np.ndarray
    cells: np.ndarray
    nuisance: np.ndarray
    clear: np.ndarray
    days: np.ndarray
    g_primary: dict[str, np.ndarray]
    g_farm: dict[tuple[str, str], np.ndarray]

    def design(self, *, mask: np.ndarray) -> fit.Design:
        """Return the design of the rows selected by `mask`."""
        return fit.Design.build(cells=self.cells[mask], nuisance=self.nuisance[mask])

    def primary_basis(self, *, ratio: float, tilt: float, mask: np.ndarray) -> np.ndarray:
        """Return the east, south, and west unit shapes at the Primary, `(n, 3)`."""
        return np.column_stack(
            [fit.clip_shape(g=self.g_primary[f"{o}{int(tilt)}"][mask], ratio=ratio) for o in "ESW"]
        )

    def farm_basis(self, *, farm: str, ratio: float, tilt: float, mask: np.ndarray) -> np.ndarray:
        """Return the east, south, and west unit shapes at a farm location, `(n, 3)`."""
        return np.column_stack(
            [
                fit.clip_shape(g=self.g_farm[(farm, f"{o}{int(tilt)}")][mask], ratio=ratio)
                for o in "ESW"
            ]
        )


def load_prepared() -> pl.DataFrame:
    """Return the prepared daylight rows of the frame, after the row-count check."""
    frame = pl.read_parquet(common.FRAME_PATH)
    assert frame.height == common.FRAME_ROWS, f"frame has {frame.height} rows"
    return fit.prepare(frame=frame)


def shape_columns() -> list[str]:
    """Return the names of the plane-of-array columns in `shapes.parquet`."""
    return [f"g{o}{int(t)}" for t in common.TILTS for o in "ESW"]


def year_windows(*, prepared: pl.DataFrame, primary: str) -> list[YearWindow]:
    """Build every summer season-year window of one Primary with enough rows."""
    shapes = pl.read_parquet(common.SHAPES_PATH)
    columns = shape_columns()
    rows = prepared.filter(
        (pl.col("site") == primary)
        & pl.col("power").is_not_null()
        & pl.col("hdd").is_not_null()
        & pl.col("month").is_in(common.SUMMER_MONTHS)
    ).join(shapes.filter(pl.col("site") == primary).drop("site"), on="time", how="inner")
    for farm in common.FARMS:
        farm_shapes = shapes.filter(pl.col("site") == farm).drop("site")
        farm_shapes = farm_shapes.rename({c: f"{farm}_{c}" for c in columns})
        rows = rows.join(farm_shapes, on="time", how="inner")
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
                clear=(block["day_cloud"] < common.CLEAR_CLOUD_MAX).to_numpy(),
                days=block["date"].cast(pl.Int32).to_numpy(),
                g_primary={c[1:]: block[c].to_numpy() for c in columns},
                g_farm={
                    (farm, c[1:]): block[f"{farm}_{c}"].to_numpy()
                    for farm in common.FARMS
                    for c in columns
                },
            )
        )
    return windows


def farm_tables(*, prepared: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return each farm's normalised power (`farm_norm`, 99th percentile 1) without dropouts.

    A half-hour at or below zero power while CAMS shows more than 100 W/m² is a meter dropout and
    is removed; the remaining negative readings are set to zero. The table also holds the farm's
    own day-mean cloud response, the season-year, and the month.
    """
    out = {}
    for farm in common.FARMS:
        rows = prepared.filter((pl.col("site") == farm) & pl.col("power").is_not_null())
        p99 = float(np.nanpercentile(rows["power"].to_numpy(), 99))
        rows = rows.filter(
            ~((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2))
        )
        out[farm] = rows.select(
            "time",
            "season_year",
            "month",
            "day_cloud",
            "date",
            farm_norm=pl.col("power").clip(lower_bound=0.0) / p99,
        )
    return out
