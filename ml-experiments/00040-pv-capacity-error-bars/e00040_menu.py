"""Fit the east, south, and west orientation menu to each directional Primary-year (descriptive).

The menu replaces the single south-facing PV shape by three non-negative capacities, one per
orientation, fitted to the April to September rows with the M2h cloud terms as free nuisance
terms. The three orientations differ only in the time of day of their output, and the cell effects
absorb the mean diurnal shape, so the fit is not expected to be identified. The result decides
nothing.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python ml-experiments/00040-pv-capacity-error-bars/e00040_menu.py
"""

import e00040_common as common
import e00040_inputs as inputs
import e00040_model as model
import numpy as np
import polars as pl
from scipy.optimize import nnls
from studies.pv_physics import half_hour_sun_and_sky
from studies.pv_separation import solar_basis

DC_AC_RATIO = 1.15


def orientation_basis() -> pl.DataFrame:
    """Return the east, south, and west unit PV shapes at each directional Primary."""
    series = pl.read_parquet(common.SERIES_PATH).filter(
        pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES)
    )
    cams = pl.read_parquet(common.CAMS_DEMAND_POINTS_PATH)
    tables = []
    for row in series.iter_rows(named=True):
        reliable = cams.filter(
            (pl.col("site") == row["site"]) & (pl.col("reliability") >= common.MIN_RELIABILITY)
        ).sort("time")
        sky = half_hour_sun_and_sky(
            hourly=reliable.select("time", pl.col("ghi_w_m2").cast(pl.Float64)),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        basis = solar_basis(sky=sky, dc_ac_ratio=DC_AC_RATIO)
        tables.append(
            pl.DataFrame(
                {
                    "time": sky.half_hour_end_time,
                    "uE": basis[:, 0],
                    "uS": basis[:, 1],
                    "uW": basis[:, 2],
                }
            ).with_columns(
                pl.col("time").cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
                site=pl.lit(row["site"]),
            )
        )
    return pl.concat(tables)


def main() -> None:
    """Write `orientation_menu.parquet`."""
    prepared = inputs.load_prepared()
    primaries = inputs.primary_rows(prepared=prepared).join(
        orientation_basis(), on=["site", "time"], how="left"
    )
    out = []
    for primary in common.DIRECTIONAL_PRIMARIES:
        for year, rows in inputs.season_year_rows(rows=primaries.filter(pl.col("site") == primary)):
            summer = rows.filter(pl.col("month").is_in(common.SUMMER_MONTHS)).drop_nulls(
                ["uE", "uS", "uW"]
            )
            window = model.window_from(rows=summer)
            bands = [
                (window.half_hour >= lo) & (window.half_hour < hi) for lo, hi in common.HOUR_BANDS
            ]
            nuisance = [window.hdd, *(window.cloud * band for band in bands)]
            columns = [-summer[name].to_numpy() for name in ("uE", "uS", "uW")] + nuisance
            x = np.column_stack([model.demean(values=v, cells=window.cells) for v in columns])
            y = model.demean(values=window.power, cells=window.cells)
            # The nuisance columns enter as a +/- pair, so they are free in sign.
            design = np.column_stack([x[:, :3], x[:, 3:], -x[:, 3:]])
            coefficients, _ = nnls(design, y)
            out.append(
                {
                    "site": primary,
                    "year": year,
                    "east": coefficients[0],
                    "south": coefficients[1],
                    "west": coefficients[2],
                }
            )
    pl.DataFrame(out).write_parquet(common.MENU_PATH)
    print(pl.DataFrame(out))


if __name__ == "__main__":
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    main()
