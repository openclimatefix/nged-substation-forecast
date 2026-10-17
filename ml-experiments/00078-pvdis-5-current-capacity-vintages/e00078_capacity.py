"""Build the real-time capacity vintages of experiment 00078, one per series and forecast run.

For each validation forecast run, the season-year constant of experiment 00045 is refitted on the
usable half-hours of the weeks before the Monday of the run's week minus one week, using that
experiment's fit (copied from experiment 00064's `e00064_capacity.py`; an experiment may not import
another experiment). The run's capacity is chosen in this order:

1. the vintage's constant for the run's season-year, if that season-year has at least
   `MIN_OBSERVED_WEEKS` observed weeks before the origin and the constant is finite and positive;
2. otherwise the vintage's constant for the previous season-year, under the same conditions;
3. otherwise experiment 00064's held capacity (`capacity.parquet`; for D14 experiment 00040's
   2023/24 constant, because D14's weeks from 2023-09-04 to 2025-08-25 are unobserved).

Every constant is divided by experiment 00045's stored recovery ratio of the series, as experiment
00064 does. The weekly sensitivity is the mean of the 4 latest weeks that are not `unobserved` of
experiment 00045's `frozen` weekly vintage of the series with the latest `origin_monday` on or
before the Monday of the run's week minus 7 days, and no more than 8 weeks before that Monday;
with no such weeks, the held capacity applies.

The script prints, per series, the capacity of each run beside the held capacity and experiment
00045's final 2025/26 constant, stops on a capacity that is not finite and positive, and writes
`work/capacity_vintages.parquet`. It reads no power forecast and scores no arm.

    uv run python ml-experiments/00078-*/e00078_capacity.py
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Final

import e00078_common as common
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
MIN_OBSERVED_WEEKS: Final[int] = 8
"""A season-year needs this many observed weeks before the origin for its constant to be used."""
GAP_WEEKS: Final[int] = 1
"""Weeks between the last week fitted and the Monday of the run's week."""
WEEKLY_AVERAGE: Final[int] = 4
"""Weekly sensitivity: the number of latest observed weeks of a vintage that are averaged."""
MAX_WEEK_AGE: Final[int] = 8
"""Weekly sensitivity: a week older than this many weeks before the run's Monday is not averaged."""
WEEKLY_VARIANT: Final[str] = "frozen"
FIT_NAME: Final[str] = "season_year_constant"
"""Experiment 00045's name for this variant in its final table."""


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


def monday_index(*, day: date) -> int:
    """Return the week index of the Monday on or before `day`."""
    return (day - E45_FIRST_WEEK_START.date()).days // 7


def run_dates() -> list[date]:
    """Return the initialisation date of every validation run, from the run files' names."""
    return sorted(
        date.fromisoformat(path.stem.removeprefix("init="))
        for path in common.VALIDATION_DIR.glob("init=*.parquet")
    )


def season_year_of(*, day: date) -> int:
    """Return the start year of the September-to-August season-year that holds `day`."""
    return day.year - (day.month < 9)


def choose_capacity(
    *,
    path: np.ndarray,
    observed_weeks: np.ndarray,
    origin: int,
    season_year: int,
    held_mw: float,
) -> tuple[float, int]:
    """Return a run's seasonal-vintage capacity in MW and the step (1, 2, or 3) that supplied it.

    Args:
        path: The vintage's season-year constants over every week, MW (NaN outside the span).
        observed_weeks: Week indices that are observed (not `unobserved`) and before the origin.
        origin: The first week not used by the fit.
        season_year: The run's season-year.
        held_mw: Experiment 00064's held capacity.

    Returns:
        The capacity and the step.
    """
    years = season_years_of_weeks()
    for step, year in ((1, season_year), (2, season_year - 1)):
        in_year = years == year
        count = int(np.isin(observed_weeks, np.nonzero(in_year)[0]).sum())
        values = path[in_year & (np.arange(E45_N_WEEKS) < origin)]
        finite = values[np.isfinite(values)]
        if count >= MIN_OBSERVED_WEEKS and finite.size and finite[-1] > 0:
            return float(finite[-1]), step
    return held_mw, 3


def weekly_capacity(
    *, vintages: pl.DataFrame, series_id: int, monday: date, held_mw: float
) -> tuple[float, str]:
    """Return a run's weekly-vintage capacity in MW and whether it came from `weekly` or `held`."""
    eligible = vintages.filter(
        (pl.col("origin_monday") <= monday - timedelta(days=7))
        & (pl.col("time_series_id") == series_id)
    )
    if eligible.is_empty():
        return held_mw, "held"
    latest = eligible["origin_monday"].max()
    oldest_allowed = monday - timedelta(weeks=MAX_WEEK_AGE)
    weeks = (
        eligible.filter(
            (pl.col("origin_monday") == latest)
            & ~pl.col("unobserved")
            & pl.col("capacity_mw").is_finite()
            & (pl.col("week_start") >= oldest_allowed)
        )
        .sort("week_start")
        .tail(WEEKLY_AVERAGE)
    )
    mean = float(weeks["capacity_mw"].to_numpy().mean()) if weeks.height else float("nan")
    if not mean > 0:
        return held_mw, "held"
    return mean, "weekly"


def main() -> None:
    """Build the vintages, print them beside the held capacity, and write them."""
    half_hourly = pl.read_parquet(common.E45_DIR / "inputs_half_hourly.parquet")
    weeks = pl.read_parquet(common.E45_DIR / "inputs_primary_weeks.parquet")
    final = pl.read_parquet(common.E45_DIR / "capacity_weekly.parquet").filter(
        pl.col("variant") == FIT_NAME
    )
    vintages = pl.read_parquet(common.E45_DIR / "capacity_weekly_vintages.parquet").filter(
        pl.col("variant") == WEEKLY_VARIANT
    )
    series = common.series_table().filter(pl.col("site").is_in(list(common.SITES)))
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    held = dict(zip(*common.capacities().select("site", "capacity_mw"), strict=True))
    dates = run_dates()
    rows = []
    for site in common.SITES:
        own_final = final.filter(pl.col("time_series_id") == ids[site])
        ratio = float(own_final["recovery_ratio"][0])
        table = usable_rows_table(half_hourly=half_hourly, weeks=weeks, site=site)
        usable = rows_from_table(table=table)
        observed = np.sort(
            weeks.filter((pl.col("site") == site) & ~pl.col("unobserved"))["week"].to_numpy()
        )
        paths: dict[int, np.ndarray] = {}
        for day in dates:
            monday = day - timedelta(days=day.weekday())
            origin = monday_index(day=day) - GAP_WEEKS
            if origin not in paths:
                paths[origin] = fit_season_year_constant(rows=usable, origin_week=origin) / ratio
            capacity, step = choose_capacity(
                path=paths[origin],
                observed_weeks=observed[observed < origin],
                origin=origin,
                season_year=season_year_of(day=day),
                held_mw=held[site],
            )
            weekly, weekly_source = weekly_capacity(
                vintages=vintages, series_id=ids[site], monday=monday, held_mw=held[site]
            )
            rows.append(
                {
                    "site": site,
                    "run_date": day,
                    "origin_week": origin,
                    "season_year_mw": capacity,
                    "season_year_step": step,
                    "weekly_mw": weekly,
                    "weekly_source": weekly_source,
                    "held_mw": held[site],
                }
            )
    table = pl.DataFrame(rows)
    smallest = table.select(pl.min_horizontal("season_year_mw", "weekly_mw")).to_series()
    if not smallest.is_finite().all() or (smallest <= 0).any():
        msg = "a vintage capacity is not finite and positive"
        raise ValueError(msg)
    final_2025 = (
        final.filter(pl.col("week_start") >= date(2025, 9, 1))
        .group_by("time_series_id")
        .agg(e45_final_2025_26_mw=pl.col("capacity_mw").mean())
    )
    summary = (
        table.with_columns(month=pl.col("run_date").dt.strftime("%Y-%m"))
        .group_by("site", "month")
        .agg(
            runs=pl.len(),
            held=pl.col("held_mw").first(),
            season_year=pl.col("season_year_mw").mean(),
            steps=pl.col("season_year_step").unique().sort().implode(),
            weekly=pl.col("weekly_mw").mean(),
            weekly_sources=pl.col("weekly_source").unique().sort().implode(),
        )
        .sort("site", "month")
    )
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print("Capacity (MW) per series and month: held, season-year vintage, weekly vintage")
        print(summary)
        print("Experiment 00045's final constant, mean over weeks from 2025-09-01 (MW)")
        print(
            final_2025.join(
                pl.DataFrame({"site": list(ids), "time_series_id": list(ids.values())}),
                on="time_series_id",
            ).select("site", "e45_final_2025_26_mw")
        )
    common.write_atomically(frame=table, target=common.VINTAGES_FILE)
    print(f"written {common.VINTAGES_FILE}")


if __name__ == "__main__":
    main()
