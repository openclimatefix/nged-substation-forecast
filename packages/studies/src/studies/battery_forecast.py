"""Probabilistic forecasts of a battery's half-hourly output: issue times, baselines, and scores.

The machinery behind the embedded-battery forecasting study. A forecast is a set of quantiles of the
output for each half-hour, issued at one of three times, each defined by what is known at that
moment:

- `DA-early`: 06:00 UTC on the day before the target day, the production slot before the N2EX
  day-ahead auction result.
- `DA-late`: 18:00 UTC on the day before the target day, the production slot after the auction
  result.
- `ID-1h`: 60 minutes before the target half-hour, which is gate closure for that half-hour.

Every function takes and returns half-hours labelled by their start, in UTC, as the price files do.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Final, Literal

import numpy as np
import polars as pl

IssueType = Literal["DA-early", "DA-late", "ID-1h"]
"""When a forecast is issued, relative to the half-hour it describes."""

DayType = Literal["working", "non_working"]
"""A working day, or a weekend day or bank holiday."""

HALF_HOURS_PER_DAY: Final[int] = 48
CLIMATOLOGY_WINDOW_DAYS: Final[int] = 56
"""How many complete days before the issue time the probabilistic climatology looks back over."""

CLIMATOLOGY_TIME_OF_DAY_SPREAD: Final[int] = 1
"""How many half-hours either side of the target's half-hour of day the climatology also samples."""

ID_PERSISTENCE_LAG_HALF_HOURS: Final[int] = 3
"""At ID-1h, persistence is the half-hour that ended at the issue time: the target's start less 90
minutes."""

BANK_HOLIDAYS_ENGLAND_AND_WALES: Final[frozenset[date]] = frozenset(
    {
        date(2025, 12, 25),
        date(2025, 12, 26),
        date(2026, 1, 1),
        date(2026, 4, 3),
        date(2026, 4, 6),
        date(2026, 5, 4),
        date(2026, 5, 25),
        date(2026, 8, 31),
    }
)
"""England and Wales bank holidays between 1 September 2025 and 31 August 2026."""

SYMMETRIC_BANDS: Final[tuple[tuple[float, float], ...]] = (
    (0.35, 0.65),
    (0.20, 0.80),
    (0.10, 0.90),
    (0.05, 0.95),
    (0.02, 0.98),
    (0.01, 0.99),
)
"""The six symmetric central bands of the 13 delivery levels, narrowest first."""

BOUND_QUANTILES: Final[tuple[float, float]] = (0.001, 0.999)
"""The percentiles of training output between which every forecast quantile is clipped."""


def day_type(*, day: date, non_working_dates: frozenset[date]) -> DayType:
    """Return whether a date is a working day.

    Args:
        day: The date.
        non_working_dates: The bank holidays.

    Returns:
        `"non_working"` for a Saturday, a Sunday, or a date in `non_working_dates`.
    """
    if day.weekday() >= 5 or day in non_working_dates:
        return "non_working"
    return "working"


def issue_time_for(*, target_start: datetime, issue: IssueType) -> datetime:
    """Return when a forecast of one half-hour is issued.

    The target day is the UTC day, which is the `delivery_date` of the N2EX day-ahead file: that
    file's README places its hourly grid on UTC.

    Args:
        target_start: The start of the target half-hour, UTC.
        issue: The issue type.

    Returns:
        06:00 on the day before the target's UTC day for `DA-early`, 18:00 on that day for
        `DA-late`, and 60 minutes before `target_start` for `ID-1h`.
    """
    day_start = target_start.replace(hour=0, minute=0, second=0, microsecond=0)
    match issue:
        case "DA-early":
            return day_start - timedelta(days=1) + timedelta(hours=6)
        case "DA-late":
            return day_start - timedelta(days=1) + timedelta(hours=18)
        case "ID-1h":
            return target_start - timedelta(hours=1)


def with_issue_time(*, frame: pl.DataFrame, issue: IssueType) -> pl.DataFrame:
    """Add the `issue_time` of each half-hour to a frame.

    Args:
        frame: A frame with `time`, the start of each target half-hour (UTC, any time unit).
        issue: The issue type.

    Returns:
        The frame with a new `issue_time` column of the same dtype as `time`.
    """
    day_start = pl.col("time").dt.truncate("1d")
    expressions: dict[IssueType, pl.Expr] = {
        "DA-early": day_start - pl.duration(days=1) + pl.duration(hours=6),
        "DA-late": day_start - pl.duration(days=1) + pl.duration(hours=18),
        "ID-1h": pl.col("time") - pl.duration(hours=1),
    }
    return frame.with_columns(issue_time=expressions[issue])


def asof_at_issue_time(
    *,
    targets: pl.DataFrame,
    vintages: pl.DataFrame,
    value_columns: Sequence[str],
) -> pl.DataFrame:
    """Attach, to each target half-hour, the latest vintage published by its issue time.

    A forecast file that keeps every publication (the NESO wind forecast is one) holds several rows
    for one valid time. A forecaster issuing at `issue_time` knows only the rows published at or
    before that moment, and uses the newest of them.

    Args:
        targets: Rows with `time` (the valid time) and `issue_time`.
        vintages: Rows with `time` (the valid time), `publish_time`, and `value_columns`.
        value_columns: The columns to attach.

    Returns:
        `targets` in its original order, with `value_columns` and `publish_time` added. A target
        with no vintage published by its issue time has nulls in every added column. A vintage
        published exactly at the issue time counts as known.
    """
    ordered = targets.with_row_index("_row").sort("issue_time")
    joined = ordered.join_asof(
        vintages.select("time", "publish_time", *value_columns).sort("publish_time"),
        left_on="issue_time",
        right_on="publish_time",
        by="time",
        strategy="backward",
        check_sortedness=False,
    )
    return joined.sort("_row").drop("_row")


def persistence_source_times(
    *, target_times: Sequence[datetime], issue: IssueType
) -> list[datetime]:
    """Return the start of the half-hour whose output a persistence forecast repeats.

    At `DA-early` and `DA-late`, the source is the same half-hour of day on the most recent earlier
    day whose half-hour had ended by the issue time: usually one day before for half-hours that
    ended by the issue time, and two days before for the rest. At `ID-1h`, the source is the
    half-hour that ended at the issue time.

    Args:
        target_times: The starts of the target half-hours, UTC.
        issue: The issue type.

    Returns:
        The start of each source half-hour, in the order of `target_times`.
    """
    half_hour = timedelta(minutes=30)
    sources: list[datetime] = []
    for target in target_times:
        if issue == "ID-1h":
            sources.append(target - ID_PERSISTENCE_LAG_HALF_HOURS * half_hour)
            continue
        issued = issue_time_for(target_start=target, issue=issue)
        days_back = 1
        while target - timedelta(days=days_back) + half_hour > issued:
            days_back += 1
        sources.append(target - timedelta(days=days_back))
    return sources


def _daily_matrix(*, history: pl.DataFrame) -> tuple[date, np.ndarray]:
    """Return the output as a days-by-half-hours matrix, NaN where a half-hour is missing."""
    daily = history.select(
        day=pl.col("time").dt.date(),
        tod=pl.col("time").dt.hour().cast(pl.Int64) * 2 + pl.col("time").dt.minute() // 30,
        output_mw=pl.col("output_mw").cast(pl.Float64),
    )
    first_day = daily["day"].min()
    last_day = daily["day"].max()
    assert isinstance(first_day, date)
    assert isinstance(last_day, date)
    matrix = np.full(((last_day - first_day).days + 1, HALF_HOURS_PER_DAY), np.nan)
    rows = np.asarray([(d - first_day).days for d in daily["day"].to_list()], dtype=np.int64)
    matrix[rows, daily["tod"].to_numpy()] = daily["output_mw"].to_numpy()
    return first_day, matrix


def climatology_quantiles(
    *,
    history: pl.DataFrame,
    targets: pl.DataFrame,
    levels: Sequence[float],
    non_working_dates: frozenset[date],
    window_days: int = CLIMATOLOGY_WINDOW_DAYS,
) -> pl.DataFrame:
    """Return the probabilistic climatology: empirical quantiles of recent same-type output.

    For a target half-hour, the sample is every output value in `window_days` complete days before
    the issue time, at the target's half-hour of day and the half-hours either side of it, on days
    of the same type as the target day. A day is complete when it ended at or before the issue
    time, so the window never includes the target day, and at `DA-late` never includes a half-hour
    ending after 18:00 on the day before. A day missing some half-hours contributes the rest.

    Args:
        history: Rows with `time` (start of the half-hour, UTC) and `output_mw`.
        targets: Rows with `time` and `issue_time`.
        levels: The quantile levels, ascending.
        non_working_dates: The bank holidays, for the day type.
        window_days: How many complete days to look back over. Fewer exist at the start of the
            history, and the climatology then uses all of them.

    Returns:
        `targets` in its original order, with a Float64 column `q<level>` per level (for example
        `q0.5`) holding the quantile, and `climatology_n`, the sample size. A target with an empty
        sample has nulls.
    """
    first_day, matrix = _daily_matrix(history=history)
    day_types = np.asarray(
        [
            day_type(day=first_day + timedelta(days=i), non_working_dates=non_working_dates)
            == "working"
            for i in range(matrix.shape[0])
        ]
    )
    cache: dict[tuple[date, bool, int], np.ndarray] = {}
    quantile_rows: list[list[float | None]] = []
    sizes: list[int] = []
    for target_time, issue_time in zip(
        targets["time"].to_list(), targets["issue_time"].to_list(), strict=True
    ):
        target_day = target_time.date()
        last_complete = issue_time.date() - timedelta(days=1)
        is_working = day_type(day=target_day, non_working_dates=non_working_dates) == "working"
        tod = target_time.hour * 2 + target_time.minute // 30
        key = (last_complete, is_working, tod)
        if key not in cache:
            stop = min((last_complete - first_day).days + 1, matrix.shape[0])
            start = max(stop - window_days, 0)
            columns = [
                c
                for c in range(
                    tod - CLIMATOLOGY_TIME_OF_DAY_SPREAD, tod + CLIMATOLOGY_TIME_OF_DAY_SPREAD + 1
                )
                if 0 <= c < HALF_HOURS_PER_DAY
            ]
            window = matrix[start:stop][day_types[start:stop] == is_working][:, columns]
            cache[key] = window[~np.isnan(window)]
        sample = cache[key]
        sizes.append(sample.size)
        if sample.size == 0:
            quantile_rows.append([None] * len(levels))
        else:
            quantile_rows.append([float(v) for v in np.quantile(sample, levels)])
    columns_out = {
        f"q{level}": pl.Series([row[i] for row in quantile_rows], dtype=pl.Float64)
        for i, level in enumerate(levels)
    }
    return targets.with_columns(**columns_out, climatology_n=pl.Series(sizes, dtype=pl.Int64))


def residual_quantile_table(
    *, residuals: np.ndarray, groups: np.ndarray, levels: Sequence[float]
) -> dict[object, np.ndarray]:
    """Return the empirical quantiles of residuals within each group.

    Args:
        residuals: Observed minus centre, shape (n_rows,), on the training folds.
        groups: The group of each row (a half-hour of day, or a schedule state), shape (n_rows,).
        levels: The quantile levels.

    Returns:
        A dictionary from each group in `groups` to its residual quantiles, shape (n_levels,).
        Rows with a NaN residual are left out.
    """
    table: dict[object, np.ndarray] = {}
    for group in np.unique(groups):
        group_residuals = residuals[(groups == group) & ~np.isnan(residuals)]
        if group_residuals.size > 0:
            table[group.item()] = np.quantile(group_residuals, levels)
    return table


def conformal_quantiles(
    *, centre: np.ndarray, groups: np.ndarray, table: Mapping[Any, np.ndarray]
) -> np.ndarray:
    """Return a centre plus the residual quantiles of its group.

    Args:
        centre: The point forecast (the persistence value or the scaled schedule), shape (n_rows,).
        groups: The group of each row, as used to build `table`.
        table: The output of `residual_quantile_table`.

    Returns:
        Quantiles, shape (n_rows, n_levels). A row whose group is missing from `table`, or whose
        centre is NaN, is all NaN.
    """
    n_levels = len(next(iter(table.values())))
    quantiles = np.full((centre.size, n_levels), np.nan)
    for group, residual_quantiles in table.items():
        selected = groups == group
        quantiles[selected] = centre[selected, None] + residual_quantiles[None, :]
    return quantiles


def output_bounds(*, training_output: np.ndarray) -> tuple[float, float]:
    """Return the 0.1st and 99.9th percentiles of the training output.

    Args:
        training_output: The battery's output on the training folds, NaNs allowed.

    Returns:
        The lower and upper bound between which `repair_quantiles` clips.
    """
    lower, upper = np.nanquantile(training_output, BOUND_QUANTILES)
    return float(lower), float(upper)


def repair_quantiles(*, quantiles: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Sort each row's quantiles to repair crossing, then clip them to the output's bounds.

    Args:
        quantiles: Shape (n_rows, n_levels).
        lower: The smallest value a quantile may take.
        upper: The largest value a quantile may take.

    Returns:
        The repaired quantiles, shape (n_rows, n_levels). A row with a NaN stays NaN.
    """
    return np.clip(np.sort(quantiles, axis=1), lower, upper)


def level_weights(*, levels: Sequence[float]) -> np.ndarray:
    """Return the weight of each quantile level in the CRPS sum.

    Each level carries half the distance to each neighbouring level, with the first and last
    extended to 0 and 1. Nine equally spaced levels from 0.1 to 0.9 each weigh 0.1.

    Args:
        levels: Ascending quantile levels strictly between 0 and 1.

    Returns:
        The weights, shape (n_levels,).
    """
    padded = np.concatenate([[0.0], np.asarray(levels, dtype=np.float64), [1.0]])
    return (padded[2:] - padded[:-2]) / 2.0


def pinball_losses(
    *, truth: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Return the pinball loss at each level.

    Args:
        truth: Observed output, shape (n_rows,).
        quantiles: Forecast quantiles, shape (n_rows, n_levels).
        levels: The quantile levels, shape (n_levels,).

    Returns:
        The loss for each row and level, shape (n_rows, n_levels).
    """
    tau = np.asarray(levels, dtype=np.float64)[None, :]
    difference = truth[:, None] - quantiles
    return np.where(difference >= 0, difference * tau, -difference * (1.0 - tau))


def weighted_crps(
    *, truth: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> np.ndarray:
    """Return the per-row CRPS approximated from the quantiles, each level weighted by its gap.

    Args:
        truth: Observed output, shape (n_rows,).
        quantiles: Forecast quantiles, shape (n_rows, n_levels). Sort them first with
            `repair_quantiles`.
        levels: The quantile levels.

    Returns:
        Twice the weighted sum of the pinball losses, shape (n_rows,), in the units of `truth`.
    """
    losses = pinball_losses(truth=truth, quantiles=quantiles, levels=levels)
    return 2.0 * (losses * level_weights(levels=levels)[None, :]).sum(axis=1)


def crps_skill_score(*, crps: np.ndarray, reference_crps: np.ndarray) -> float:
    """Return one minus the ratio of mean CRPS values.

    Args:
        crps: The forecast's per-row CRPS.
        reference_crps: The reference's per-row CRPS on the same rows.

    Returns:
        A positive value when the forecast beats the reference.
    """
    return float(1.0 - np.mean(crps) / np.mean(reference_crps))


def band_coverage_and_width(
    *,
    truth: np.ndarray,
    quantiles: np.ndarray,
    levels: Sequence[float],
    bands: Sequence[tuple[float, float]] = SYMMETRIC_BANDS,
) -> pl.DataFrame:
    """Return the coverage and mean width of each central band.

    Args:
        truth: Observed output, shape (n_rows,).
        quantiles: Forecast quantiles, shape (n_rows, n_levels).
        levels: The quantile levels.
        bands: Pairs of lower and upper levels, each of which must be in `levels`.

    Returns:
        One row per band with `lower_level`, `upper_level`, `nominal_coverage`, `coverage`
        (the share of rows with the lower quantile at or below the truth and the truth at or below
        the upper quantile, so a value on a band's edge is inside), and `mean_width`.
    """
    index = {round(level, 6): i for i, level in enumerate(levels)}
    rows = []
    for lower_level, upper_level in bands:
        lower = quantiles[:, index[round(lower_level, 6)]]
        upper = quantiles[:, index[round(upper_level, 6)]]
        rows.append(
            {
                "lower_level": lower_level,
                "upper_level": upper_level,
                "nominal_coverage": upper_level - lower_level,
                "coverage": float(np.mean((truth >= lower) & (truth <= upper))),
                "mean_width": float(np.mean(upper - lower)),
            }
        )
    return pl.DataFrame(rows)


def reliability_table(
    *, truth: np.ndarray, quantiles: np.ndarray, levels: Sequence[float]
) -> pl.DataFrame:
    """Return the share of half-hours whose output was at or below each forecast quantile.

    Args:
        truth: Observed output, shape (n_rows,).
        quantiles: Forecast quantiles, shape (n_rows, n_levels).
        levels: The quantile levels.

    Returns:
        One row per level with `level` and `observed_share`. The two are equal when calibrated.
    """
    shares = np.mean(truth[:, None] <= quantiles, axis=0)
    return pl.DataFrame({"level": list(levels), "observed_share": shares.tolist()})


def day_shuffle_map(
    *, days: Sequence[date], non_working_dates: frozenset[date], seed: int
) -> pl.DataFrame:
    """Draw, for each day, a different day of the same calendar month and day type.

    The `shuffled` price arm and the `neighbour_fpn_shuffled` arm replace a day's inputs with the
    inputs of its drawn day. The draw keeps the month's typical daily shape and removes the target
    day's own information.

    Args:
        days: Every day that can be drawn or replaced.
        non_working_dates: The bank holidays, for the day type.
        seed: The random seed.

    Returns:
        Columns `date` and `source_date`, one row per day in `days`, with `source_date` never equal
        to `date`.

    Raises:
        ValueError: If a calendar month holds only one day of some day type.
    """
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int, DayType], list[date]] = {}
    for day in sorted(set(days)):
        groups.setdefault(
            (day.year, day.month, day_type(day=day, non_working_dates=non_working_dates)), []
        ).append(day)
    drawn: dict[date, date] = {}
    for (year, month, kind), members in groups.items():
        if len(members) < 2:
            msg = f"{year}-{month:02d} has only one {kind} day, so no other day can be drawn."
            raise ValueError(msg)
        for day in members:
            others = [m for m in members if m != day]
            drawn[day] = others[int(rng.integers(len(others)))]
    ordered = sorted(drawn)
    return pl.DataFrame({"date": ordered, "source_date": [drawn[d] for d in ordered]})


def neighbour_ids(
    *,
    target: str,
    lead_party: Mapping[str, str],
    rule: Literal["different_party", "same_party"] = "different_party",
) -> list[str]:
    """Return the batteries whose Physical Notifications count as the target's neighbours.

    Args:
        target: The target battery's BMU identifier, which is a key of `lead_party`.
        lead_party: The lead party of every battery in the testbed.
        rule: `different_party` keeps every other battery with a different lead party from the
            target. `same_party` keeps the other batteries of the target's own lead party.

    Returns:
        The BMU identifiers, sorted. The target is never in the list.
    """
    own_party = lead_party[target]
    if rule == "different_party":
        chosen = [b for b, party in lead_party.items() if party != own_party]
    else:
        chosen = [b for b, party in lead_party.items() if party == own_party and b != target]
    return sorted(chosen)


def neighbour_statistics(
    *, fpn: pl.DataFrame, neighbours: Sequence[str], p99_mw: Mapping[str, float]
) -> pl.DataFrame:
    """Return the three neighbour slots for every half-hour.

    The statistic for half-hour `t` reads the neighbours' Physical Notifications at `t` and at the
    half-hour before `t`, and no later half-hour. A neighbour with no notification for a half-hour
    drops out of that half-hour's statistics.

    Args:
        fpn: Rows with `time` (start of the half-hour), `bmu_id`, and `fpn_mw` (positive is export).
        neighbours: The BMU identifiers that count.
        p99_mw: Each battery's 99th percentile absolute output, which scales its notification.

    Returns:
        Columns `time`, `neighbour_mean_fraction` (the mean of each notification as a fraction of
        its own battery's p99), `neighbour_mean_fraction_previous` (the same mean at the half-hour
        before), and `neighbour_discharging_share` (the share of neighbours with a notification
        above zero), sorted by time. No neighbours gives an empty frame.
    """
    if not neighbours:
        return pl.DataFrame(
            schema={
                "time": fpn.schema["time"],
                "neighbour_mean_fraction": pl.Float64,
                "neighbour_mean_fraction_previous": pl.Float64,
                "neighbour_discharging_share": pl.Float64,
            }
        )
    scaled = (
        fpn.filter(pl.col("bmu_id").is_in(list(neighbours)))
        .filter(pl.col("fpn_mw").is_not_null())
        .with_columns(
            fraction=pl.col("fpn_mw")
            / pl.col("bmu_id").replace_strict(dict(p99_mw), return_dtype=pl.Float64)
        )
    )
    now = (
        scaled.group_by("time")
        .agg(
            neighbour_mean_fraction=pl.col("fraction").mean(),
            neighbour_discharging_share=(pl.col("fpn_mw") > 0).mean(),
        )
        .sort("time")
    )
    previous = now.select(
        time=pl.col("time") + pl.duration(minutes=30),
        neighbour_mean_fraction_previous=pl.col("neighbour_mean_fraction"),
    )
    return now.join(previous, on="time", how="left").select(
        "time",
        "neighbour_mean_fraction",
        "neighbour_mean_fraction_previous",
        "neighbour_discharging_share",
    )
