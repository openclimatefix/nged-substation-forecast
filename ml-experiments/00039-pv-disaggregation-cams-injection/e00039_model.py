"""The separation models of experiment 00039: M1 (half-hourly), M1d (daily), M1L (richer demand).

Each fits, within one season-year window of daylight half-hours,

    power = cell + b * HDD [+ d * L] - c * u + error

where `u` is the unit PV shape (AC output per MW of a generic fleet driven by CAMS irradiance, whose
99th percentile is 1.0), `HDD` is heating degree-days below 15.5 °C, `L` is a flat cloud response,
and `cell` is a fixed effect for (calendar month, local half-hour, day type). Least squares within
cells is done by subtracting each cell's mean from every variable. `c` is in MW and equals the 99th
percentile of the fitted PV component `c * u`.
"""

from dataclasses import dataclass

import e00039_common as common
import holidays
import numpy as np
import polars as pl


@dataclass(frozen=True)
class Window:
    """The daylight half-hours of one series in one season-year, as aligned arrays.

    Attributes:
        power: Observed power, MW, positive for demand.
        u: Unit PV shape at the irradiance point under test.
        hdd: Heating degree-days, °C.
        cloud: The flat cloud response `1 - min(GHI / clear-sky GHI, 1)`.
        cells: Integer code of (month, local half-hour, day type).
        days: Integer code of the local date, the cluster for standard errors.
        month: Calendar month, 1 to 12.
        half_hour: Local half-hour of the day, 0 to 47.
        off: 1 on weekends and English bank holidays, else 0.
    """

    power: np.ndarray
    u: np.ndarray
    hdd: np.ndarray
    cloud: np.ndarray
    cells: np.ndarray
    days: np.ndarray
    month: np.ndarray
    half_hour: np.ndarray
    off: np.ndarray

    def take(self, *, mask: np.ndarray) -> Window:
        """Return the rows selected by a boolean mask."""
        return Window(*(getattr(self, name)[mask] for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class Fit:
    """One fit's result.

    Attributes:
        c: Fitted PV capacity, MW (the 99th percentile of the PV component).
        se: Day-clustered standard error of `c`, MW.
        b: Coefficient on HDD, MW per °C.
        pv: The fitted PV component, `c * u`, per row.
        demand: The cell plus HDD term, per row (zero for the daily fit).
        residual: What is left, per row.
        constrained: True when the unconstrained `c` was negative and `c` was set to zero.
    """

    c: float
    se: float
    b: float
    pv: np.ndarray
    demand: np.ndarray
    residual: np.ndarray
    constrained: bool


def bank_holidays() -> pl.Series:
    """Return the English bank holidays of 2018 to 2027 as a date series."""
    return pl.Series(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=range(2018, 2028)))
    ).cast(pl.Date)


def prepare(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Keep daylight half-hours and add the calendar columns every window needs.

    Args:
        frame: `frame.parquet`.

    Returns:
        The daylight rows with `local`, `half_hour`, `off`, `month`, `date`, `season_year`, `hdd`,
        and `cloud` added.
    """
    return (
        frame.filter(pl.col("zenith") < common.MAX_ZENITH_DEG)
        .with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
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
            season_year=(
                pl.col("local").dt.year() - (pl.col("local").dt.month() < 9).cast(pl.Int32)
            ),
            hdd=(common.HDD_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
            cloud=1.0
            - (pl.col("ghi_u") / pl.col("ghi_u_clear")).clip(upper_bound=1.0).fill_nan(0.0),
        )
    )


def window_from(*, rows: pl.DataFrame, u_column: str = "u") -> Window:
    """Build a `Window` from prepared rows, taking `u` from `u_column`.

    Args:
        rows: Prepared daylight rows of one series in one season-year, all with non-null power
            and `hdd`.
        u_column: The column holding the unit PV shape to use.

    Returns:
        The aligned arrays.
    """
    cells = (rows["month"] * 1000 + rows["half_hour"] * 10 + rows["off"]).to_numpy()
    return Window(
        power=rows["power"].to_numpy().astype(float),
        u=rows[u_column].to_numpy().astype(float),
        hdd=rows["hdd"].to_numpy().astype(float),
        cloud=rows["cloud"].to_numpy().astype(float),
        cells=cells,
        days=rows["date"].cast(pl.Int32).to_numpy(),
        month=rows["month"].to_numpy(),
        half_hour=rows["half_hour"].to_numpy(),
        off=rows["off"].to_numpy(),
    )


def demean(*, values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Subtract each cell's mean from its rows."""
    _, inverse = np.unique(cells, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    return values - (sums / counts)[inverse]


def _ols(
    *, y: np.ndarray, regressors: list[np.ndarray], cells: np.ndarray, days: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least squares within cells; return coefficients, clustered errors, and the fitted parts.

    Args:
        y: The series to fit.
        regressors: The regressors, each one value per row.
        cells: Integer code of the fixed-effect cell of each row.
        days: Integer code of the cluster (local date) of each row.

    Returns:
        The coefficients, their day-clustered standard errors, the demeaned fitted values per
        regressor (columns), and the demeaned residual.
    """
    y_demeaned = demean(values=y, cells=cells)
    x = np.column_stack([demean(values=r, cells=cells) for r in regressors])
    beta, *_ = np.linalg.lstsq(x, y_demeaned, rcond=None)
    residual = y_demeaned - x @ beta
    xtx_inverse = np.linalg.pinv(x.T @ x)
    _, inverse = np.unique(days, return_inverse=True)
    scores = np.column_stack(
        [np.bincount(inverse, weights=x[:, k] * residual) for k in range(x.shape[1])]
    )
    covariance = xtx_inverse @ (scores.T @ scores) @ xtx_inverse
    return beta, np.sqrt(np.diag(covariance)), x * beta, residual


def fit_half_hourly(
    *,
    y: np.ndarray,
    window: Window,
    extra: list[np.ndarray] | None = None,
    constrain: bool = True,
    use_hdd: bool = True,
) -> Fit:
    """Fit M1 (or M1L when `extra` holds the cloud response) to a series on a window's rows.

    The series `y` need not be `window.power`: injection fits `window.power + injection` with the
    window's own design.

    Args:
        y: The series to fit, per row.
        window: The rows and the design.
        extra: Additional demand regressors, such as `[window.cloud]` for M1L.
        constrain: When true and the unconstrained `c` is negative, set `c` to zero and refit
            the other coefficients.
        use_hdd: When false, leave out the temperature term (the farms have no temperature).

    Returns:
        The fit.
    """
    nuisance = ([window.hdd] if use_hdd else []) + list(extra or [])
    beta, se, parts, residual = _ols(
        y=y, regressors=[window.u, *nuisance], cells=window.cells, days=window.days
    )
    constrained = False
    if constrain and beta[0] > 0.0:
        constrained = True
        if nuisance:
            beta_n, _, parts_n, residual = _ols(
                y=y, regressors=nuisance, cells=window.cells, days=window.days
            )
            beta = np.concatenate([[0.0], beta_n])
            parts = np.column_stack([np.zeros(len(y)), parts_n])
        else:
            beta, parts, residual = (
                np.zeros(1),
                np.zeros((len(y), 1)),
                demean(values=y, cells=window.cells),
            )
    pv = -beta[0] * window.u
    nuisance_part = parts[:, 1:].sum(axis=1) if parts.shape[1] > 1 else np.zeros(len(y))
    # y = cell mean of y + c * cell mean of u + nuisance - c * u + residual, so the demand part
    # carries the cell mean of the PV shape and the PV part is exactly c * u.
    cell_part = (y - demean(values=y, cells=window.cells)) + (-beta[0]) * (
        window.u - demean(values=window.u, cells=window.cells)
    )
    return Fit(
        c=float(-beta[0]),
        se=float(se[0]),
        b=float(beta[1]) if nuisance else 0.0,
        pv=pv,
        demand=cell_part + nuisance_part,
        residual=residual,
        constrained=constrained,
    )


def daily_sums(*, window: Window, y: np.ndarray) -> tuple[np.ndarray, Window]:
    """Aggregate a window to one row per local date that has enough daylight half-hours.

    The daily cell is (month, day type, number of daylight half-hours), so a short winter day and a
    long summer day have separate means.

    Args:
        window: The half-hourly rows.
        y: The series to sum, per row.

    Returns:
        The daily sums of `y`, and a `Window` whose `power`, `u`, and `hdd` are daily sums (the
        other arrays are the day's first row's, with `cells` rebuilt and `days` unique).
    """
    unique_days, inverse, counts = np.unique(window.days, return_inverse=True, return_counts=True)
    keep = counts >= common.MIN_DAILY_ROWS

    def total(values: np.ndarray) -> np.ndarray:
        return np.bincount(inverse, weights=values)[keep]

    first = np.unique(inverse, return_index=True)[1][keep]
    month = window.month[first]
    off = window.off[first]
    daily = Window(
        power=total(window.power),
        u=total(window.u),
        hdd=total(window.hdd),
        cloud=total(window.cloud),
        cells=month * 100000 + counts[keep] * 10 + off,
        days=unique_days[keep],
        month=month,
        half_hour=np.zeros(keep.sum(), dtype=window.half_hour.dtype),
        off=off,
    )
    return total(y), daily


def fit_daily(*, y: np.ndarray, window: Window, constrain: bool = True) -> Fit:
    """Fit M1d: the half-hourly series summed per day, then `c * sum(u)` against the day's sums.

    The returned PV series is `c * u` at the half-hours of `window`, so it compares with a
    half-hourly truth; `demand` and `residual` are zero-filled because the model has no
    half-hourly demand.

    Args:
        y: The series to fit, per half-hourly row of `window`.
        window: The half-hourly rows.
        constrain: As in `fit_half_hourly`.

    Returns:
        The fit.
    """
    y_daily, daily = daily_sums(window=window, y=y)
    result = fit_half_hourly(y=y_daily, window=daily, constrain=constrain)
    zeros = np.zeros(len(y))
    return Fit(
        c=result.c,
        se=result.se,
        b=result.b,
        pv=result.c * window.u,
        demand=zeros,
        residual=zeros,
        constrained=result.constrained,
    )


def placebo_u(*, window: Window, rng: np.random.Generator) -> np.ndarray:
    """Replace each day's `u` by that of another day of the same month, matched by half-hour.

    Args:
        window: The rows whose `u` is permuted.
        rng: The random generator.

    Returns:
        The permuted `u`, with NaN where the donor day has no row at that half-hour.
    """
    unique_days, day_index = np.unique(window.days, return_inverse=True)
    table = np.full((len(unique_days), 48), np.nan)
    table[day_index, window.half_hour] = window.u
    day_month = np.zeros(len(unique_days), dtype=int)
    day_month[day_index] = window.month
    donor = np.arange(len(unique_days))
    for month in np.unique(day_month):
        members = np.nonzero(day_month == month)[0]
        donor[members] = rng.permutation(members)
    return table[donor[day_index], window.half_hour]
