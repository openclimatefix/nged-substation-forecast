"""Fit the east, south, and west shares of each Primary, and run the evening-west check.

The shares are fitted on the training window only, one set per Primary, and the script writes the
coefficients of every solar shape.

The fit is experiment 00073's pooled orientation fit (arm O3, no prior, clear days only, one set of
shares per Primary shared by every season-year), with two changes: it reads only half-hours up to
`LAST_TRAINING_VALID` (2025-06-30 23:30 UTC), and it uses the unit shapes of `e00077_orient.py`,
which are the CAMS shapes at the Primary's own location with a tilt of 25 degrees and a DC to AC
ratio of 1.15. Within each season-year the power is modelled as

    power = calendar cell + heating degree-days + cloud response in three hour bands
            - capacity * (shares . [uE, uS, uW]) + error

and the shares are the non-negative least-squares solution pooled over season-years.

`work/shares.parquet` holds each Primary's shares and its mean south-only capacity.
`work/coefficients.parquet` holds the weights `(cE, cS, cW)` of each solar shape in each series:

- `blend`: the fitted shares.
- `s25`: `(0, 1, 0)`.
- `shuffled`: the shares of the next series in `common.SITES`, a fixed cyclic permutation.

`work/evening.parquet` holds, for each Primary, the slope of its after-dark demand (calendar cell
and heating degree-days removed) on the day's mean daytime cloud and on the day's heating
degree-days, with 95% intervals from 200 resamples of 7-day blocks. No verdict reads it.

    POLARS_MAX_THREADS=6 uv run --with holidays python ml-experiments/00077-*/e00077_shares.py
"""

from dataclasses import dataclass
from typing import Final

import e00077_common as common
import e00077_orient as orient
import holidays
import numpy as np
import polars as pl
from scipy.optimize import nnls

SHARES_PATH: Final = common.BASE_WORK_DIR / "shares.parquet"
COEFFICIENTS_PATH: Final = common.BASE_WORK_DIR / "coefficients.parquet"
EVENING_PATH: Final = common.BASE_WORK_DIR / "evening.parquet"
FRAME_ROWS: Final[int] = 2_891_184
MAX_ZENITH_DEG: Final[float] = 75.0
AFTER_DARK_ZENITH_DEG: Final[float] = 90.0
EVENING_START_HALF_HOUR: Final[int] = 36
HDD_BASE_C: Final[float] = 15.5
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
HOUR_BANDS: Final[tuple[tuple[int, int], ...]] = ((0, 20), (20, 28), (28, 48))
CLEAR_CLOUD_MAX: Final[float] = 0.15
MIN_CLEAR_DAYS: Final[int] = 20
MIN_WINDOW_ROWS: Final[int] = 1000
BLOCK_DAYS: Final[int] = 7
EVENING_DRAWS: Final[int] = 200
EVENING_SEED: Final[int] = 77


def bank_holidays() -> pl.Series:
    """Return the English bank holidays of 2018 to 2027 as a date series."""
    return pl.Series(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=range(2018, 2028)))
    ).cast(pl.Date)


def local_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the local half-hour, day type, month, date, season-year, and heating degree-days."""
    return frame.with_columns(
        local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
    ).with_columns(
        half_hour=(
            pl.col("local").dt.hour().cast(pl.Int64) * 2
            + pl.col("local").dt.minute().cast(pl.Int64) // 30
        ),
        off=(
            (pl.col("local").dt.weekday() >= 6)
            | pl.col("local").dt.date().is_in(bank_holidays().implode())
        ).cast(pl.Int64),
        month=pl.col("local").dt.month().cast(pl.Int64),
        date=pl.col("local").dt.date(),
        season_year=(pl.col("local").dt.year() - (pl.col("local").dt.month() < 9).cast(pl.Int32)),
        hdd=(HDD_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
    )


def training_frame() -> pl.DataFrame:
    """Return the verdict series' frame rows up to the last training valid time.

    Raises:
        AssertionError: If the frame has an unexpected number of rows or a row is too late.
    """
    frame = pl.read_parquet(common.FRAME_00039_PATH)
    assert frame.height == FRAME_ROWS, f"frame has {frame.height} rows"
    rows = frame.filter(
        (pl.col("kind") == "substation")
        & pl.col("site").is_in(common.SITES)
        & (pl.col("time") <= common.LAST_TRAINING_VALID)
    )
    latest = common.latest_time(times=rows["time"])
    assert latest <= common.LAST_TRAINING_VALID, "a frame row is after the training window"
    print(f"latest frame time used: {latest}")
    return local_columns(frame=rows)


def demean(*, values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Subtract each cell's mean from its rows (column-wise on a 2-D array)."""
    _, inverse = np.unique(cells, return_inverse=True)
    counts = np.bincount(inverse)
    if values.ndim == 1:
        return values - (np.bincount(inverse, weights=values) / counts)[inverse]
    out = np.empty_like(values, dtype=float)
    for k in range(values.shape[1]):
        out[:, k] = demean(values=values[:, k], cells=cells)
    return out


@dataclass(frozen=True)
class Design:
    """Rows' calendar cells and the orthonormal basis of the demeaned nuisance columns."""

    cells: np.ndarray
    basis: np.ndarray

    @staticmethod
    def build(*, cells: np.ndarray, nuisance: np.ndarray) -> Design:
        """Demean the nuisance columns within cells and orthonormalise them."""
        x = demean(values=nuisance, cells=cells)
        u, s, _ = np.linalg.svd(x, full_matrices=False)
        keep = s > 1e-9 * max(s.max(), 1e-12)
        return Design(cells=cells, basis=u[:, keep])

    def residual(self, *, values: np.ndarray) -> np.ndarray:
        """Return `values` with the cell means and the nuisance terms removed."""
        v = demean(values=values, cells=self.cells)
        return v - self.basis @ (self.basis.T @ v)


def fit_pooled(*, designs: list[Design], ys: list[np.ndarray], shapes: list[np.ndarray]) -> tuple:
    """Return the pooled shares `m` (capacities relative to each year's south-only scale).

    Each year's scale `s_y` comes from a south-only fit; the stack fits `m >= 0` to
    `y_y = ... - s_y * (u_y @ m)` with no prior.

    Returns:
        The shares as capacities in units of `s_y` and the list of `s_y`, MW.
    """
    scale, blocks, targets = [], [], []
    for design, y, u in zip(designs, ys, shapes, strict=True):
        my = design.residual(values=y)
        mu = design.residual(values=-u)
        south, _ = nnls(mu[:, [1]], my)
        s = max(float(south[0]), 0.1)
        scale.append(s)
        blocks.append(s * mu)
        targets.append(my)
    m, _ = nnls(np.vstack(blocks), np.concatenate(targets))
    return m, scale


def daylight_rows(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return the daylight summer rows of the Primaries with the shapes and the cloud columns."""
    cams = pl.read_parquet(orient.ORIENT_DIR / "cams.parquet")
    return (
        rows.filter(pl.col("zenith") < MAX_ZENITH_DEG)
        .with_columns(
            cloud=1.0
            - (pl.col("ghi_u") / pl.col("ghi_u_clear")).clip(upper_bound=1.0).fill_nan(0.0)
        )
        .with_columns(day_cloud=pl.col("cloud").mean().over("site", "date"))
        .filter(
            pl.col("power").is_not_null()
            & pl.col("hdd").is_not_null()
            & pl.col("month").is_in(SUMMER_MONTHS)
        )
        .join(cams, on=["site", "time"], how="inner")
        .sort("site", "time")
    )


def windows(*, site_rows: pl.DataFrame) -> list[dict[str, np.ndarray]]:
    """Return one dictionary of arrays per season-year with enough rows and clear days."""
    out = []
    for year in sorted(site_rows["season_year"].unique().to_list()):
        block = site_rows.filter(pl.col("season_year") == year)
        if block.height < MIN_WINDOW_ROWS:
            continue
        half_hour = block["half_hour"].to_numpy()
        nuisance = np.column_stack(
            [block["hdd"].to_numpy().astype(float)]
            + [
                block["cloud"].to_numpy() * ((half_hour >= lo) & (half_hour < hi))
                for lo, hi in HOUR_BANDS
            ]
        )
        clear = (block["day_cloud"] < CLEAR_CLOUD_MAX).to_numpy()
        if len(np.unique(block["date"].cast(pl.Int32).to_numpy()[clear])) < MIN_CLEAR_DAYS:
            continue
        cells = (block["month"] * 1000 + block["half_hour"] * 10 + block["off"]).to_numpy()
        out.append(
            {
                "year": year,
                "power": block["power"].to_numpy().astype(float)[clear],
                "cells": cells[clear],
                "nuisance": nuisance[clear],
                "shapes": block.select(*orient.COLUMNS).to_numpy().astype(float)[clear],
            }
        )
    return out


def fit_site(*, site_rows: pl.DataFrame) -> dict[str, object]:
    """Return one Primary's shares, mean south-only capacity, and the years that entered."""
    parts = windows(site_rows=site_rows)
    designs = [Design.build(cells=w["cells"], nuisance=w["nuisance"]) for w in parts]
    m, scales = fit_pooled(
        designs=designs, ys=[w["power"] for w in parts], shapes=[w["shapes"] for w in parts]
    )
    shares = m / m.sum() if m.sum() > 0 else np.array([0.0, 1.0, 0.0])
    return {
        "site": site_rows["site"][0],
        "east": float(shares[0]),
        "south": float(shares[1]),
        "west": float(shares[2]),
        "mean_south_capacity_mw": float(np.mean(scales)),
        "years": ",".join(str(w["year"]) for w in parts),
        "latest_row": site_rows["time"].max(),
    }


def coefficient_table(*, shares: pl.DataFrame) -> pl.DataFrame:
    """Return the weights of the east, south, and west shapes of every solar shape and series."""
    order = list(common.SITES)
    rows = []
    for site in order:
        own = shares.filter(pl.col("site") == site).row(0, named=True)
        mine = np.array([own["east"], own["south"], own["west"]])
        other = shares.filter(pl.col("site") == order[(order.index(site) + 1) % len(order)]).row(
            0, named=True
        )
        theirs = np.array([other["east"], other["south"], other["west"]])
        for arm, c in (
            ("blend", mine),
            ("s25", np.array([0.0, 1.0, 0.0])),
            ("shuffled", theirs),
        ):
            rows.append({"arm": arm, "site": site, "cE": c[0], "cS": c[1], "cW": c[2]})
    return pl.DataFrame(rows)


def evening_check(*, rows: pl.DataFrame, daylight: pl.DataFrame) -> pl.DataFrame:
    """Return each Primary's after-dark demand response to daytime cloud and heating degree-days."""
    day = (
        daylight.select("site", "date", "day_cloud")
        .unique(subset=["site", "date"])
        .with_columns(pl.col("date"))
    )
    dark = (
        rows.filter(
            (pl.col("zenith") > AFTER_DARK_ZENITH_DEG)
            & (pl.col("half_hour") >= EVENING_START_HALF_HOUR)
            & pl.col("power").is_not_null()
            & pl.col("hdd").is_not_null()
            & pl.col("month").is_in(SUMMER_MONTHS)
        )
        .join(day, on=["site", "date"], how="inner")
        .sort("site", "time")
    )
    daily_hdd = dark.group_by("site", "date").agg(day_hdd=pl.col("hdd").mean())
    dark = dark.join(daily_hdd, on=["site", "date"])
    out = []
    rng = np.random.default_rng(EVENING_SEED)
    for site in common.SITES:
        own = dark.filter(pl.col("site") == site)
        cells = (
            own["season_year"] * 100000 + own["month"] * 1000 + own["half_hour"] * 10 + own["off"]
        ).to_numpy()
        x = np.column_stack([own["day_cloud"].to_numpy(), own["day_hdd"].to_numpy()]).astype(float)
        y = own["power"].to_numpy().astype(float)
        days = own["date"].cast(pl.Int32).to_numpy()

        def slopes(
            index: np.ndarray, *, cells: np.ndarray = cells, x: np.ndarray = x, y: np.ndarray = y
        ) -> np.ndarray:
            xr = demean(values=x[index], cells=cells[index])
            yr = demean(values=y[index], cells=cells[index])
            return np.linalg.lstsq(xr, yr, rcond=None)[0]

        point = slopes(np.arange(len(y)))
        blocks = (days - days.min()) // BLOCK_DAYS
        ids = np.unique(blocks)
        draws = []
        for _ in range(EVENING_DRAWS):
            picks = rng.choice(ids, size=len(ids))
            index = np.concatenate([np.nonzero(blocks == b)[0] for b in picks])
            draws.append(slopes(index))
        low, high = np.quantile(np.array(draws), [0.025, 0.975], axis=0)
        out.append(
            {
                "site": site,
                "n_rows": len(y),
                "cloud_slope_mw": float(point[0]),
                "cloud_low": float(low[0]),
                "cloud_high": float(high[0]),
                "hdd_slope_mw": float(point[1]),
                "hdd_low": float(low[1]),
                "hdd_high": float(high[1]),
            }
        )
    return pl.DataFrame(out)


def main() -> None:
    """Fit the shares, write the coefficient table, and run the evening-west check."""
    rows = training_frame()
    daylight = daylight_rows(rows=rows)
    shares = pl.DataFrame(
        [fit_site(site_rows=daylight.filter(pl.col("site") == s)) for s in common.SITES]
    )
    shares = shares.with_columns(west_capacity_mw=pl.col("west") * pl.col("mean_south_capacity_mw"))
    assert shares["latest_row"].max() <= common.LAST_TRAINING_VALID  # ty: ignore[unsupported-operator]
    common.write_atomically(frame=shares, target=SHARES_PATH)
    coefficients = coefficient_table(shares=shares)
    common.write_atomically(frame=coefficients, target=COEFFICIENTS_PATH)
    evening = evening_check(rows=rows, daylight=daylight).join(
        shares.select("site", "west", "west_capacity_mw", "mean_south_capacity_mw"), on="site"
    )
    common.write_atomically(frame=evening, target=EVENING_PATH)
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(shares)
        print(coefficients)
        print(evening)


if __name__ == "__main__":
    main()
