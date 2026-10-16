"""The oracle step: which logged closures moved load, and was the load conserved across the pair?

The oracle reads the logs and the residual, so it labels events for validation. No detector reads
its output while running.
"""

from datetime import datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from e00037_common import MVA_SERIES, TUNING_MONTHS
from e00037_grid import GRID_LENGTH, GRID_START, stamp_months
from e00037_logs import LONG_INTERVAL_HOURS

MIN_HOURS: Final[float] = 2.0
FIRST_YEAR: Final[int] = 2022
Z_TWO_SIDED: Final[float] = 3.0
Z_ONE_SIDED: Final[float] = 4.0
SIZE_BAND_EDGES: Final[tuple[float, float]] = (0.10, 0.25)
SHARE_FLOOR_MW: Final[float] = 0.5
NULL_DRAWS: Final[int] = 400
DURATION_CAP: Final[int] = 400
DURATION_BUCKETS: Final[tuple[int, ...]] = (4, 6, 8, 12, 16, 24, 32, 48, 72, 96, 144, 192, 288, 400)
PLACEBO_SEEDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)
DURATION_EDGES_HOURS: Final[tuple[float, ...]] = (2.0, 6.0, 24.0, 72.0, 336.0)


def step_statistic(*, values: np.ndarray, first: int, last: int, excluded: np.ndarray) -> float:
    """Return the in-interval mean minus the reference mean, or NaN if too few stamps are valid.

    The reference is the same stamps a day before and after for intervals up to 48 stamps, and 144
    stamps either side for longer intervals. Stamps in `excluded` (other logged intervals) are
    left out of the reference.
    """
    length = last - first
    inside = values[first:last]
    if np.isfinite(inside).sum() < max(1, 0.5 * length):
        return float("nan")
    if length <= 48:
        parts = [(first - 48, last - 48), (first + 48, last + 48)]
    else:
        parts = [(first - 144, first), (last, last + 144)]
    reference = np.concatenate(
        [
            values[max(a, 0) : min(b, GRID_LENGTH)][~excluded[max(a, 0) : min(b, GRID_LENGTH)]]
            for a, b in parts
        ]
    )
    if np.isfinite(reference).sum() < 0.5 * min(length, 144):
        return float("nan")
    return float(np.nanmean(inside) - np.nanmean(reference))


def logged_cover(*, intervals: pl.DataFrame, series: int) -> np.ndarray:
    """Count, at every stamp, the logged intervals (padded by a day) that touch `series`."""
    cover = np.zeros(GRID_LENGTH, dtype=np.int32)
    for row in intervals.iter_rows(named=True):
        if str(series) in (row["side_a"], row["side_b"]):
            cover[_padded(first=row["first_stamp"], last=row["last_stamp_exclusive"])] += 1
    return cover


def _padded(*, first: int, last: int) -> slice:
    return slice(max(first - 48, 0), min(last + 48, GRID_LENGTH))


def null_scale(
    *,
    values: np.ndarray,
    length: int,
    mask: np.ndarray,
    month_of_stamp: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Return 1.4826 times the MAD of the step at random event-free starts in tuning months."""
    bucket = min(DURATION_BUCKETS, key=lambda b: abs(b - min(length, DURATION_CAP)))
    draws: list[float] = []
    for _ in range(10 * NULL_DRAWS):
        if len(draws) >= NULL_DRAWS:
            break
        start = int(rng.integers(200, GRID_LENGTH - bucket - 200))
        if (
            month_of_stamp[start] not in TUNING_MONTHS
            or mask[start - 48 : start + bucket + 48].any()
        ):
            continue
        draws.append(step_statistic(values=values, first=start, last=start + bucket, excluded=mask))
    finite = np.array([d for d in draws if np.isfinite(d)])
    if len(finite) < 30:
        return float("nan")
    return float(1.4826 * np.median(np.abs(finite - np.median(finite))))


def band_of_hours(*, hours: float) -> str:
    """Name the duration band of an interval of `hours`."""
    labels = ("2-6 h", "6-24 h", "1-3 d", "3-14 d", ">14 d")
    for edge, label in zip(DURATION_EDGES_HOURS[1:], labels, strict=False):
        if hours < edge:
            return label
    return labels[-1]


def band_of_share(*, share: float) -> str:
    """Name the size band of a transfer that is `share` of the receiving side's load."""
    if share < SIZE_BAND_EDGES[0]:
        return "<10%"
    return "10-25%" if share < SIZE_BAND_EDGES[1] else ">=25%"


def eligible_intervals(*, intervals: pl.DataFrame) -> pl.DataFrame:
    """Keep closed intervals since 2022 lasting 2 hours to 60 days."""
    return intervals.filter(
        pl.col("hours") >= MIN_HOURS,
        pl.col("hours") <= LONG_INTERVAL_HOURS,
        pl.col("start_utc").dt.year() >= FIRST_YEAR,
    )


def _side_series(*, label: str | None, arrays: dict[int, np.ndarray]) -> int | None:
    if label is None or label == "unknown":
        return None
    return int(label) if int(label) in arrays else None


def oracle_table(
    *,
    intervals: pl.DataFrame,
    residuals: dict[int, np.ndarray],
    observed: dict[int, np.ndarray],
    expected: dict[int, np.ndarray],
    placebo_seed: int | None,
) -> pl.DataFrame:
    """Score every eligible interval: step, z, relative step, transfer, size, and confirmation.

    With `placebo_seed`, every interval is shifted by plus or minus 7 to 28 days (same time of day)
    on both sides together, and shifted spans that touch a real logged interval are dropped.
    """
    months = np.array(stamp_months())
    # Several switches closed in one job on one sheet give duplicate rows; keep the longest.
    intervals = intervals.sort("hours", descending=True).unique(
        subset=["sheet", "first_stamp"], keep="first", maintain_order=True
    )
    rng = np.random.default_rng(0 if placebo_seed is None else 1000 + placebo_seed)
    shift_rng = np.random.default_rng(placebo_seed)
    covers = {sid: logged_cover(intervals=intervals, series=sid) for sid in residuals}
    scales: dict[tuple[int, int], float] = {}
    rows: list[dict] = []
    for row in eligible_intervals(intervals=intervals).iter_rows(named=True):
        first, last = row["first_stamp"], row["last_stamp_exclusive"]
        if placebo_seed is not None:
            offset = int(shift_rng.choice([-1, 1]) * shift_rng.integers(7, 29) * 48)
            first, last = first + offset, last + offset
            if first < 300 or last > GRID_LENGTH - 300:
                continue
        sides = {
            tag: _side_series(label=row[f"side_{tag}"], arrays=residuals) for tag in ("a", "b")
        }
        record = {
            "sheet": row["sheet"],
            "switch": row["switch"],
            "job": row["job"],
            "operation": row["operation"],
            "start_utc": row["start_utc"],
            "hours": row["hours"],
            "first_stamp": first,
            "last_stamp_exclusive": last,
            "placebo_seed": placebo_seed or 0,
            "series_a": sides["a"],
            "series_b": sides["b"],
        }
        length = last - first
        for tag, sid in sides.items():
            record |= {f"step_{tag}": None, f"z_{tag}": None, f"rel_{tag}": None}
            record |= {f"rel_first_{tag}": None, f"rel_second_{tag}": None}
            if sid is None:
                continue
            cover = covers[sid]
            own = cover.copy()
            if placebo_seed is None:
                own[_padded(first=first, last=last)] -= 1
            elif cover[first:last].any():
                continue
            mask = own > 0
            step = step_statistic(values=residuals[sid], first=first, last=last, excluded=mask)
            key = (sid, min(DURATION_BUCKETS, key=lambda b: abs(b - min(length, DURATION_CAP))))
            if key not in scales:
                scales[key] = null_scale(
                    values=residuals[sid],
                    length=length,
                    mask=cover > 0,
                    month_of_stamp=months,
                    rng=rng,
                )
            record[f"step_{tag}"] = step
            record[f"z_{tag}"] = step / scales[key] if scales[key] > 0 else float("nan")
            record |= _relative_step(
                tag=tag,
                residual=residuals[sid],
                expected=expected[sid],
                first=first,
                last=last,
                mask=mask,
            )
            record[f"expected_{tag}"] = float(np.nanmean(expected[sid][first:last]))
            before = observed[sid][max(first - 336, 0) : first]
            record[f"load_before_{tag}"] = (
                float(max(np.nanmean(np.abs(before)), SHARE_FLOOR_MW))
                if np.isfinite(before).any()
                else float("nan")
            )
        rows.append(record)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return _finish(frame=frame, months=months)


def _relative_step(
    *,
    tag: str,
    residual: np.ndarray,
    expected: np.ndarray,
    first: int,
    last: int,
    mask: np.ndarray,
) -> dict[str, float]:
    """Return the step divided by expected power over the event, and over each half of it."""
    out: dict[str, float] = {}
    middle = (first + last) // 2
    for name, (a, b) in {
        "rel": (first, last),
        "rel_first": (first, middle),
        "rel_second": (middle, last),
    }.items():
        if b - a < 2:
            out[f"{name}_{tag}"] = float("nan")
            continue
        step = step_statistic(values=residual, first=a, last=b, excluded=mask)
        level = float(np.nanmean(np.abs(expected[a:b])))
        out[f"{name}_{tag}"] = step / level if level > 0 else float("nan")
    return out


def _finish(*, frame: pl.DataFrame, months: np.ndarray) -> pl.DataFrame:
    """Add transfer, confirmation, size, bands, split, and a de-duplication flag."""
    for column in ("step_a", "step_b", "z_a", "z_b", "load_before_a", "load_before_b"):
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    frame = frame.with_columns(pl.col(pl.Float64).fill_nan(None))
    both = pl.col("step_a").is_not_null() & pl.col("step_b").is_not_null()
    frame = frame.with_columns(
        kind=pl.when(both).then(pl.lit("two_sided")).otherwise(pl.lit("one_sided")),
        transfer_mw=pl.when(both)
        .then((pl.col("step_b") - pl.col("step_a")).abs() / 2)
        .otherwise(pl.coalesce(pl.col("step_a").abs(), pl.col("step_b").abs())),
    )
    two_sided_ok = (
        both
        & (pl.col("step_a") * pl.col("step_b") < 0)
        & (pl.col("z_a").abs() >= Z_TWO_SIDED)
        & (pl.col("z_b").abs() >= Z_TWO_SIDED)
    )
    one_z = pl.coalesce(pl.col("z_a").abs(), pl.col("z_b").abs())
    frame = frame.with_columns(
        confirmed=pl.when(both).then(two_sided_ok).otherwise(one_z >= Z_ONE_SIDED).fill_null(False),
        receiver_load_mw=pl.when(pl.col("step_a") > pl.col("step_b").fill_null(-1e9))
        .then(pl.col("load_before_a"))
        .otherwise(pl.col("load_before_b")),
        has_mva=pl.col("series_a").is_in(list(MVA_SERIES))
        | pl.col("series_b").is_in(list(MVA_SERIES)),
    ).with_columns(
        share=pl.col("transfer_mw") / pl.col("receiver_load_mw"),
        duration_band=pl.col("hours").map_elements(
            lambda h: band_of_hours(hours=h), return_dtype=pl.String
        ),
        month=pl.col("first_stamp").map_elements(
            lambda i: int(months[min(max(i, 0), GRID_LENGTH - 1)]), return_dtype=pl.Int64
        ),
    )
    frame = frame.with_columns(
        size_band=pl.col("share").map_elements(
            lambda s: band_of_share(share=s) if s is not None else None, return_dtype=pl.String
        ),
        split=pl.when(pl.col("month").is_in(list(TUNING_MONTHS)))
        .then(pl.lit("tuning"))
        .otherwise(pl.lit("validation")),
    )
    # Several switches closed in one job on one sheet give duplicate rows; keep the longest.
    return frame.sort("hours", descending=True).with_columns(
        primary=pl.struct("sheet", "first_stamp").is_first_distinct()
    )


def stamp_to_datetime(*, stamp: int) -> datetime:
    """Return the UTC time of the period-ending stamp with this index."""
    return GRID_START + timedelta(minutes=30 * stamp)
