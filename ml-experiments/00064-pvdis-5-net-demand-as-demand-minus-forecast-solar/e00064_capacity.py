"""Fit one embedded-solar capacity per verdict series, using data before 2025-07-01 only.

The capacity is the 2024/25 season-year constant of experiment 00045 (one capacity per September
to August year, non-negative least squares on the usable daylight half-hours with the demand cells
and heating degree-days projected out), refitted on the weeks before the first week that holds a
validation half-hour. The fit code is copied from experiment 00045's `e00045_model.py`; an
experiment may not import another experiment. Choices where the plan is silent:

- `origin_week` is the week of Monday 2025-06-30. That week is left out entirely, so no validation
  actual enters the capacity, at the price of one training day.
- The fitted constant is divided by experiment 00045's recovery ratio of the same series and
  variant (`season_year_constant`), read from its stored final table, because experiment 00045's
  final constants are corrected the same way. That ratio was computed from the full record, not
  truncated at the origin; it is a property of the injection test on the tuning farms.
- A series whose refit has no 2024/25 constant (its weeks are all unobserved; D14) reads experiment
  00040's 2023/24 `C`, the value experiment 00046 used. The set of such series is
  `FALLBACK_TO_EXPERIMENT_00040`.

The script stops if any capacity is zero or not finite, prints the table beside experiment 00046's
`C` and experiment 00045's final constants for the 2024/25 season-year, and writes
`work/capacity.parquet` (`site`, `capacity_mw`, `source`).

    uv run python ml-experiments/00064-*/e00064_capacity.py
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import e00064_common as common
import holidays
import numpy as np
import polars as pl
import scipy.linalg
import scipy.optimize
import scipy.sparse

E45_FIRST_WEEK_START: Final[datetime] = datetime(2019, 9, 2, tzinfo=UTC)
"""Monday of week 0 in experiment 00045's weekly index."""
E45_N_WEEKS: Final[int] = 361
HDD_BASE_C: Final[float] = 15.5
"""Heating degree-days are the degrees that the temperature falls below this base."""
RIDGE: Final[float] = 1e-10
"""Added to the diagonal of the normal matrix so that it can be factorised."""
SEASON_YEAR: Final[int] = 2024
"""The season-year (September 2024 to August 2025) whose constant is used."""
FALLBACK_TO_EXPERIMENT_00040: Final[frozenset[str]] = frozenset({"D14"})
"""Series whose weeks of the season-year are all unobserved; they read experiment 00040's `C`."""
FIT_NAME: Final[str] = "season_year_constant"
"""Experiment 00045's name for this variant in its final table."""
TRAINING_END_MONDAY: Final[datetime] = datetime(2025, 6, 30, tzinfo=UTC)


@dataclass(frozen=True)
class Rows:
    """The usable half-hours of one Primary, as aligned arrays.

    Attributes:
        power: Observed power, MW.
        u: Unit PV shape.
        hdd: Heating degree-days, degrees C.
        cells: Integer code of the demand cell of each row.
        week: Week index of each row.
    """

    power: np.ndarray
    u: np.ndarray
    hdd: np.ndarray
    cells: np.ndarray
    week: np.ndarray

    def before(self, *, origin_week: int) -> Rows:
        """Return the rows of weeks strictly before `origin_week`."""
        keep = self.week < origin_week
        return Rows(
            power=self.power[keep],
            u=self.u[keep],
            hdd=self.hdd[keep],
            cells=self.cells[keep],
            week=self.week[keep],
        )


@dataclass(frozen=True)
class Problem:
    """The projected least-squares problem of one design at one origin.

    Attributes:
        span: First and last week with data (inclusive).
        gram: The normal matrix of the weekly capacities, after the cells and HDD are removed.
        rhs: The matching right-hand side.
        season_years: Season-year of each week in the span.
    """

    span: tuple[int, int]
    gram: np.ndarray
    rhs: np.ndarray
    season_years: np.ndarray


def week_start(*, week: int) -> datetime:
    """Return the UTC Monday that starts `week`."""
    return E45_FIRST_WEEK_START + timedelta(weeks=week)


def season_years_of_weeks() -> np.ndarray:
    """Return the season-year (September to August, named by its start year) of every week."""
    return np.array(
        [(start := week_start(week=w)).year - (start.month < 9) for w in range(E45_N_WEEKS)]
    )


def usable_rows_table(*, half_hourly: pl.DataFrame, weeks: pl.DataFrame, site: str) -> pl.DataFrame:
    """Return one Primary's usable half-hours with the demand cell and heating degree-days.

    Args:
        half_hourly: Experiment 00045's `inputs_half_hourly.parquet`.
        weeks: Experiment 00045's `inputs_primary_weeks.parquet`.
        site: Frame site code, such as `"D16"`.

    Returns:
        The usable half-hours of observed weeks, sorted by `time`, with `time`, `power` (MW), `u`,
        `week`, `cell` (Int64 demand-cell code), and `hdd` (degrees C).
    """
    unobserved = weeks.filter((pl.col("site") == site) & pl.col("unobserved"))["week"]
    holiday_dates = pl.Series(
        sorted(holidays.country_holidays("GB", subdiv="ENG", years=range(2018, 2028)))
    ).cast(pl.Date)
    local = pl.col("local")
    return (
        half_hourly.filter(
            (pl.col("site") == site)
            & pl.col("usable")
            & ~pl.col("week").is_in(unobserved.implode())
        )
        .with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
            cell=(
                (local.dt.year() - (local.dt.month() < 9).cast(pl.Int32)).cast(pl.Int64) * 1_000_000
                + local.dt.month().cast(pl.Int64) * 10_000
                + (local.dt.hour().cast(pl.Int64) * 2 + local.dt.minute().cast(pl.Int64) // 30) * 10
                + ((local.dt.weekday() >= 6) | local.dt.date().is_in(holiday_dates.implode())).cast(
                    pl.Int64
                )
            ),
            hdd=(HDD_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
        )
        .sort("time")
        .select("time", "power", "u", "week", "cell", "hdd")
    )


def rows_from_table(*, table: pl.DataFrame) -> Rows:
    """Return the arrays of a `usable_rows_table`."""
    return Rows(
        power=table["power"].to_numpy().astype(float),
        u=table["u"].to_numpy().astype(float),
        hdd=table["hdd"].to_numpy().astype(float),
        cells=table["cell"].to_numpy(),
        week=table["week"].to_numpy().astype(np.int64),
    )


def project_cells_and_hdd(*, rows: Rows) -> tuple[np.ndarray, np.ndarray]:
    """Remove the demand cells and HDD from the weekly regression.

    The regressor of week `w` is minus `u` on that week's rows, so a positive capacity lowers power.

    Args:
        rows: The half-hours to use.

    Returns:
        The weekly normal matrix and right-hand side, over all `E45_N_WEEKS` weeks.
    """
    n = len(rows.power)
    _, cell_index = np.unique(rows.cells, return_inverse=True)
    n_cells = int(cell_index.max()) + 1
    counts = np.bincount(cell_index).astype(float)
    members = scipy.sparse.csr_matrix((np.ones(n), (np.arange(n), cell_index)), shape=(n, n_cells))
    weekly = scipy.sparse.csr_matrix((-rows.u, (np.arange(n), rows.week)), shape=(n, E45_N_WEEKS))
    hdd = scipy.sparse.csr_matrix(rows.hdd.reshape(-1, 1))
    design = scipy.sparse.hstack([weekly, hdd], format="csr")
    to_cells = (design.T @ members).toarray()
    y_cells = members.T @ rows.power
    gram_full = (design.T @ design).toarray() - (to_cells / counts) @ to_cells.T
    rhs_full = design.T @ rows.power - (to_cells / counts) @ y_cells
    g_hh = gram_full[-1, -1]
    if g_hh > 1e-12:
        g_ch = gram_full[:-1, -1]
        return (
            gram_full[:-1, :-1] - np.outer(g_ch, g_ch) / g_hh,
            rhs_full[:-1] - g_ch * rhs_full[-1] / g_hh,
        )
    return gram_full[:-1, :-1], rhs_full[:-1]


def build_problem(*, rows: Rows) -> Problem:
    """Project out the cells and HDD and restrict to the span from the first to the last week."""
    gram, rhs = project_cells_and_hdd(rows=rows)
    observed_all = np.bincount(rows.week, minlength=E45_N_WEEKS) > 0
    first, last = (int(i) for i in np.nonzero(observed_all)[0][[0, -1]])
    span = slice(first, last + 1)
    return Problem(
        span=(first, last),
        gram=gram[span, span],
        rhs=rhs[span],
        season_years=season_years_of_weeks()[span],
    )


def nnls(*, gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Minimise `0.5 x' G x - rhs' x` subject to `x >= 0` through the Cholesky factor of `G`."""
    factor = scipy.linalg.cholesky(gram, lower=False)
    target = scipy.linalg.solve_triangular(factor, rhs, trans="T", lower=False)
    return scipy.optimize.nnls(factor, target)[0]


def fit_season_year_constant(*, rows: Rows, origin_week: int) -> np.ndarray:
    """Fit one capacity per season-year with `c >= 0` on the weeks before `origin_week`.

    Args:
        rows: The Primary's usable half-hours.
        origin_week: The first week not to use.

    Returns:
        The capacity of every week in MW, length `E45_N_WEEKS`; NaN outside the span from the first
        to the last week that holds rows before the origin.
    """
    problem = build_problem(rows=rows.before(origin_week=origin_week))
    years, inverse = np.unique(problem.season_years, return_inverse=True)
    pooling = scipy.sparse.csr_matrix(
        (np.ones(len(inverse)), (np.arange(len(inverse)), inverse)),
        shape=(len(inverse), len(years)),
    )
    pooled_gram = (pooling.T @ problem.gram @ pooling) + RIDGE * np.eye(len(years))
    pooled = nnls(gram=np.asarray(pooled_gram), rhs=np.asarray(pooling.T @ problem.rhs).ravel())
    path = np.full(E45_N_WEEKS, np.nan)
    path[problem.span[0] : problem.span[1] + 1] = pooled[inverse]
    return path


def origin_week() -> int:
    """Return the week index of the Monday 2025-06-30, the first week the refit leaves out."""
    return (TRAINING_END_MONDAY - E45_FIRST_WEEK_START).days // 7


def capacity_for(
    *, site: str, half_hourly: pl.DataFrame, weeks: pl.DataFrame, ratio: float
) -> tuple[float, float]:
    """Return one series' refitted 2024/25 constant before and after the recovery ratio.

    Args:
        site: Frame site code.
        half_hourly: Experiment 00045's half-hourly inputs.
        weeks: Experiment 00045's weekly inputs.
        ratio: The recovery ratio to divide by.

    Returns:
        The raw refitted constant and the constant divided by `ratio`, in MW; NaN for both when no
        week of the season-year lies in the fit's span.
    """
    origin = origin_week()
    table = usable_rows_table(half_hourly=half_hourly, weeks=weeks, site=site)
    path = fit_season_year_constant(rows=rows_from_table(table=table), origin_week=origin)
    in_season = season_years_of_weeks() == SEASON_YEAR
    values = path[in_season & (np.arange(E45_N_WEEKS) < origin)]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite[-1]), float(finite[-1] / ratio)


def main() -> None:
    """Fit and print the capacity table, stop on a zero or non-finite value, and write it."""
    half_hourly = pl.read_parquet(common.E45_DIR / "inputs_half_hourly.parquet")
    weeks = pl.read_parquet(common.E45_DIR / "inputs_primary_weeks.parquet")
    final = pl.read_parquet(common.E45_DIR / "capacity_weekly.parquet").filter(
        pl.col("variant") == FIT_NAME
    )
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    e40 = pl.read_parquet(common.E40_CAPACITY_PATH).filter(
        pl.col("year") == common.E40_CAPACITY_SEASON_YEAR
    )
    e40_c = dict(zip(e40["site"], e40["C"], strict=True))
    rows = []
    for site in common.SITES:
        own = final.filter(pl.col("time_series_id") == ids[site])
        ratio = float(own["recovery_ratio"][0])
        raw, corrected = capacity_for(site=site, half_hourly=half_hourly, weeks=weeks, ratio=ratio)
        if site in FALLBACK_TO_EXPERIMENT_00040 or not np.isfinite(corrected) or corrected <= 0:
            if site not in FALLBACK_TO_EXPERIMENT_00040:
                msg = f"{site}: the refit has no positive 2024/25 constant ({corrected}); stop"
                raise ValueError(msg)
            capacity, source = float(e40_c[site]), "experiment 00040 2023/24 C"
        else:
            capacity, source = corrected, "refit before 2025-07-01 / recovery ratio"
        season_start = datetime(SEASON_YEAR, 9, 1, tzinfo=UTC)
        in_season = own.filter(
            (pl.col("week_start") >= season_start.date())
            & (pl.col("week_start") < datetime(SEASON_YEAR + 1, 9, 1, tzinfo=UTC).date())
        )["capacity_mw"]
        rows.append(
            {
                "site": site,
                "capacity_mw": capacity,
                "source": source,
                "refit_raw_mw": raw,
                "recovery_ratio": ratio,
                "e46_C_mw": float(e40_c[site]),
                "e45_final_2024_25_mw": float(in_season.to_numpy().mean())
                if in_season.len()
                else None,
            }
        )
    table = pl.DataFrame(rows)
    with pl.Config(tbl_cols=20, tbl_width_chars=220):
        print("Capacity (MW): used, refit pieces, experiment 46's C, experiment 45's final")
        print(table)
    used = table["capacity_mw"]
    if not used.is_finite().all() or (used <= 0).any():
        msg = "a verdict series has a zero or non-finite capacity"
        raise ValueError(msg)
    common.write_atomically(
        frame=table.select("site", "capacity_mw", "source"), target=common.CAPACITY_FILE
    )
    print(f"written {common.CAPACITY_FILE}")


if __name__ == "__main__":
    main()
