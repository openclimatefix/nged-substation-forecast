"""Split experiment 00033's Primary best guess's scores by labelled switching half-hours.

Reads the best guess's per-case scores (`n1_cs`, N1 + 14-day correction + spread factor) and
experiment 41's labelled events, flags each case, and writes every share, bound, and interval to
`results.parquet` in the experiment's private data folder. Nothing is fitted.

The flags, for the case's series:

- `a_before`: the valid time is in an event, and every such event starts after the forecast origin.
- `a_after`: the valid time is in an event that started at or before the origin.
- `b1`: one of the three lags N1 reads (newest usable observation, same time on the latest earlier
  day, same time on the latest earlier week) is in an event.
- `b2`: an event overlaps the 14-day window the correction averages.
- `b`: `b1` or `b2`. Category (b) counts only cases that are not in (a).

    uv run python ml-experiments/00044-sw2-2-switching-share-of-best-guess-score/e00044_score.py
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00044-sw2-2-switching-share-of-best-guess-score"
METRICS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "metrics"
    / "Primary"
    / "n1_cs.parquet"
)
LABELS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
HOLDOUT: Final[str] = "2026-07-01"
"""No experiment scores a valid time on or after this date."""
REAL_KINDS: Final[tuple[str, ...]] = (
    "real_operation",
    "real_short",
    "real_edge_close",
    "real_edge_open",
    "alarm",
)
"""Label kinds that describe real events; sentinel and placebo cards are not events."""
VISIBLE: Final[tuple[str, ...]] = ("conserved", "one_sided")
MIN_SHARE_OF_RECEIVER: Final[float] = 0.25
MAX_MISSING_SHARE: Final[float] = 0.5
HINDSIGHT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "hindsight"
STEP_US: Final[int] = 30 * 60 * 1_000_000
STEPS_PER_DAY: Final[int] = 48
NEWEST_AGE: Final[int] = 7
"""Half-hours between the origin and the newest usable observation (3 hours of telemetry delay and
the half-hour a value spans), as experiment 00033 uses."""
WINDOW: Final[int] = 14 * STEPS_PER_DAY
"""Half-hours in the correction's window."""
EVENT_SETS: Final[tuple[str, ...]] = ("trustworthy", "all_visible", "trustworthy_complete")
LOSSES: Final[tuple[str, ...]] = ("qs", "tw_p99")
CATEGORIES: Final[tuple[str, ...]] = ("a_before", "a_after", "a_any", "b1", "b2", "b", "flagged")
"""`flagged` is (a) or (b); `b1`, `b2`, and `b` exclude cases in (a)."""
BOUNDS: Final[tuple[str, ...]] = ("metered", "nra", "ceiling")
COUNTERFACTUALS: Final[tuple[str, ...]] = ("series_mean", "series_lead_hour_mean")
N_RESAMPLES: Final[int] = 2000
RESAMPLE_SEED: Final[int] = 20261016
MAX_LEAD_BUCKET: Final[int] = 60
HOURS: Final[int] = 24


@dataclass(frozen=True)
class Cases:
    """The scored cases, one row per series, run, and valid time, seeds averaged.

    Attributes:
        frame: The cases with `time_series_id`, `power_fcst_init_time`, `valid_time`,
            `lead_hours`, `qs`, `tw_p99`, `month`, and `cell`.
        series: The series ids in `cell` order.
    """

    frame: pl.DataFrame
    series: list[int]


def load_cases() -> Cases:
    """Return the best guess's Primary cases before the hold-out, seeds averaged."""
    frame = (
        pl.read_parquet(METRICS_PATH)
        .filter(
            pl.col("shared")
            & (pl.col("valid_time") < pl.lit(HOLDOUT).str.to_datetime("%Y-%m-%d", time_zone="UTC"))
        )
        .group_by("time_series_id", "power_fcst_init_time", "valid_time")
        .agg(pl.col("lead_hours").first(), pl.col(*LOSSES).mean(), n_seeds=pl.len())
        .sort("time_series_id", "power_fcst_init_time", "valid_time")
    )
    if frame["n_seeds"].n_unique() != 1:
        msg = "series differ in how many seeds they hold"
        raise ValueError(msg)
    series = sorted(frame["time_series_id"].unique().to_list())
    frame = frame.with_columns(
        month=pl.col("power_fcst_init_time").dt.year() * 12
        + pl.col("power_fcst_init_time").dt.month(),
        series_index=pl.col("time_series_id").replace_strict(
            {sid: index for index, sid in enumerate(series)}, return_dtype=pl.Int64
        ),
        lead_bucket=pl.col("lead_hours").floor().clip(0, MAX_LEAD_BUCKET).cast(pl.Int64),
        hour=pl.col("valid_time").dt.hour().cast(pl.Int64),
    ).with_columns(
        cell=(pl.col("series_index") * (MAX_LEAD_BUCKET + 1) + pl.col("lead_bucket")) * HOURS
        + pl.col("hour")
    )
    return Cases(frame=frame, series=series)


def _usual_load(*, series_id: int, onset: datetime) -> float:
    """Return a series' mean absolute power in MW over the 28 days before `onset`."""
    frame = pl.read_parquet(
        HINDSIGHT_DIR / f"{series_id}.parquet", columns=["valid_time", "obs_mw"]
    )
    window = frame.filter(
        (pl.col("valid_time") < onset) & (pl.col("valid_time") >= onset - pl.duration(days=28))
    )
    return window.select(pl.col("obs_mw").abs().mean()).item() if window.height else float("nan")


def load_events(*, series: list[int]) -> pl.DataFrame:
    """Return one row per event and affected series, with the three event-set flags.

    The stored `moved_share_of_receiver` is NaN for every event whose donor or receiver ids the
    labeller wrote as strings, so the share is recomputed here as the median moved MW over the sum
    of the named metered receivers' mean absolute power in the 28 days before the onset. An event
    with no metered receiver has no share and is never in the trustworthy band.

    Args:
        series: The Primary series ids; events on other series are dropped.

    Returns:
        Columns `event_id`, `time_series_id`, `onset`, `end`, `share`, `stored_share`,
        `trustworthy`, `trustworthy_complete`, and `all_visible` (always true).
    """
    labels = pl.read_parquet(LABELS_PATH).filter(
        pl.col("kind").is_in(REAL_KINDS) & pl.col("visibility").is_in(VISIBLE)
    )
    rows = []
    for event_id, row in enumerate(labels.iter_rows(named=True)):
        donor = {int(row["donor"])} if row["donor"] is not None else set()
        receivers = {int(r["series"]) for r in json.loads(row["receivers"] or "[]")}
        metered = sorted(receivers & set(series))
        moved = row["moved_mw_p50"]
        loads = [_usual_load(series_id=sid, onset=row["onset_utc"]) for sid in metered]
        share = moved / sum(loads) if metered and moved is not None and sum(loads) > 0 else None
        duration = row["end_utc"] - row["onset_utc"]
        trustworthy = (
            share is not None
            and share >= MIN_SHARE_OF_RECEIVER
            and duration.total_seconds() >= 24 * 3600
        )
        complete = trustworthy and row["max_side_missing_share"] <= MAX_MISSING_SHARE
        rows.extend(
            {
                "event_id": event_id,
                "time_series_id": sid,
                "onset": row["onset_utc"],
                "end": row["end_utc"],
                "share": share,
                "stored_share": row["moved_share_of_receiver"],
                "trustworthy": trustworthy,
                "trustworthy_complete": complete,
                "all_visible": True,
            }
            for sid in sorted((donor | receivers) & set(series))
        )
    return pl.DataFrame(
        rows,
        schema_overrides={"onset": pl.Datetime("us", "UTC"), "end": pl.Datetime("us", "UTC")},
    )


def _microseconds(*, column: pl.Series) -> np.ndarray:
    """Return a datetime series as int64 microseconds since the epoch."""
    return column.dt.epoch("us").to_numpy()


def flag_cases(*, cases: Cases, events: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the boolean flag arrays of `CATEGORIES` for the cases, in the cases' row order."""
    frame = cases.frame
    n_rows = frame.height
    valid = _microseconds(column=frame["valid_time"])
    origin = _microseconds(column=frame["power_fcst_init_time"])
    sid = frame["time_series_id"].to_numpy()
    newest = origin - NEWEST_AGE * STEP_US
    gap = (valid - newest) // STEP_US
    day = valid - STEPS_PER_DAY * STEP_US * -(-gap // STEPS_PER_DAY)
    week = valid - 7 * STEPS_PER_DAY * STEP_US * -(-gap // (7 * STEPS_PER_DAY))
    window_start = newest - (WINDOW - 1) * STEP_US
    flags = {name: np.zeros(n_rows, dtype=bool) for name in ("a_before", "a_after", "b1", "b2")}
    for series_id in cases.series:
        rows = np.flatnonzero(sid == series_id)
        mine = events.filter(pl.col("time_series_id") == series_id)
        if mine.is_empty():
            continue
        onset = _microseconds(column=mine["onset"])[None, :]
        end = _microseconds(column=mine["end"])[None, :]

        def inside(
            stamp: np.ndarray, onset: np.ndarray = onset, end: np.ndarray = end
        ) -> np.ndarray:
            return (onset <= stamp[:, None]) & (stamp[:, None] <= end)

        target = inside(valid[rows])
        started = target & (onset <= origin[rows][:, None])
        flags["a_after"][rows] = started.any(axis=1)
        flags["a_before"][rows] = target.any(axis=1) & ~started.any(axis=1)
        flags["b1"][rows] = (
            inside(newest[rows]).any(axis=1)
            | inside(day[rows]).any(axis=1)
            | inside(week[rows]).any(axis=1)
        )
        flags["b2"][rows] = (
            (onset <= newest[rows][:, None]) & (end >= window_start[rows][:, None])
        ).any(axis=1)
    a_any = flags["a_before"] | flags["a_after"]
    flags["a_any"] = a_any
    flags["b1"] &= ~a_any
    flags["b2"] &= ~a_any
    flags["b"] = flags["b1"] | flags["b2"]
    flags["flagged"] = a_any | flags["b"]
    return flags


def _month_aggregates(
    *, cases: Cases, flags: dict[str, np.ndarray], n_cells: int
) -> dict[str, np.ndarray]:
    """Return per-month sums, shape (months, ...), from which every statistic is computed."""
    frame = cases.frame
    months = np.unique(frame["month"].to_numpy())
    month_index = np.searchsorted(months, frame["month"].to_numpy())
    cell = frame["cell"].to_numpy()
    series_index = frame["series_index"].to_numpy()
    n_series = len(cases.series)
    unflagged = ~flags["flagged"]
    masks = {
        "metered": flags["a_after"] | flags["b"],
        "nra": flags["b"],
        "ceiling": flags["flagged"],
    }
    out: dict[str, np.ndarray] = {}
    n_months = len(months)
    out["n_all"] = np.bincount(month_index, minlength=n_months).astype(float)[:, None]
    out["n_nra_total"] = np.bincount(
        month_index, weights=~flags["a_any"], minlength=n_months
    ).astype(float)[:, None]
    for name in CATEGORIES:
        out[f"n_{name}"] = np.bincount(month_index, weights=flags[name], minlength=n_months)[
            :, None
        ]
    for loss in LOSSES:
        values = frame[loss].to_numpy().astype(float)
        out[f"sum_{loss}_all"] = np.bincount(month_index, weights=values, minlength=n_months)[
            :, None
        ]
        out[f"sum_{loss}_nra_total"] = np.bincount(
            month_index, weights=values * ~flags["a_any"], minlength=n_months
        )[:, None]
        for name in CATEGORIES:
            out[f"sum_{loss}_{name}"] = np.bincount(
                month_index, weights=values * flags[name], minlength=n_months
            )[:, None]
        for ids, size, label in ((series_index, n_series, "s"), (cell, n_cells, "c")):
            flat = month_index * size + ids
            out[f"n_unf_{label}"] = np.bincount(
                flat, weights=unflagged, minlength=n_months * size
            ).reshape(n_months, size)
            out[f"sum_{loss}_unf_{label}"] = np.bincount(
                flat, weights=values * unflagged, minlength=n_months * size
            ).reshape(n_months, size)
            for bound, mask in masks.items():
                out[f"n_{bound}_{label}"] = np.bincount(
                    flat, weights=mask, minlength=n_months * size
                ).reshape(n_months, size)
                out[f"sum_{loss}_{bound}_{label}"] = np.bincount(
                    flat, weights=values * mask, minlength=n_months * size
                ).reshape(n_months, size)
    return out


def _statistics(
    *, totals: dict[str, np.ndarray], n_series: int, series_of_cell: np.ndarray
) -> dict[str, float]:
    """Return every reported statistic from summed aggregates."""
    stats: dict[str, float] = {}
    n_all = totals["n_all"].sum()
    for name in CATEGORIES:
        stats[f"half_hour_share:{name}"] = 100 * totals[f"n_{name}"].sum() / n_all
    for loss in LOSSES:
        total = totals[f"sum_{loss}_all"].sum()
        unflagged_mean = totals[f"sum_{loss}_unf_s"].sum() / totals["n_unf_s"].sum()
        for name in CATEGORIES:
            share = 100 * totals[f"sum_{loss}_{name}"].sum() / total
            stats[f"score_share:{loss}:{name}"] = share
            stats[f"excess_ratio:{loss}:{name}"] = (
                share / stats[f"half_hour_share:{name}"]
                if stats[f"half_hour_share:{name}"]
                else np.nan
            )
            n_name = totals[f"n_{name}"].sum()
            stats[f"mean_loss:{loss}:{name}"] = (
                totals[f"sum_{loss}_{name}"].sum() / n_name if n_name else np.nan
            )
        stats[f"mean_loss:{loss}:unflagged"] = unflagged_mean
        for bound in BOUNDS:
            denominator = totals[f"sum_{loss}_nra_total"].sum() if bound == "nra" else total
            for counterfactual, label in zip(COUNTERFACTUALS, ("s", "c"), strict=True):
                unf_n = totals[f"n_unf_{label}"].sum(axis=0)
                unf_sum = totals[f"sum_{loss}_unf_{label}"].sum(axis=0)
                series_mean = totals[f"sum_{loss}_unf_s"].sum(axis=0) / np.maximum(
                    totals["n_unf_s"].sum(axis=0), 1
                )
                fallback = series_mean if label == "s" else series_mean[series_of_cell]
                mean = np.where(unf_n > 0, unf_sum / np.maximum(unf_n, 1), fallback)
                counted_n = totals[f"n_{bound}_{label}"].sum(axis=0)
                counted_sum = totals[f"sum_{loss}_{bound}_{label}"].sum(axis=0)
                stats[f"bound:{loss}:{bound}:{counterfactual}"] = (
                    100 * (counted_sum - counted_n * mean).sum() / denominator
                )
    return stats


def score(*, cases: Cases, flags: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return each statistic with its point estimate and 95% month-resampling interval."""
    n_series = len(cases.series)
    n_cells = n_series * (MAX_LEAD_BUCKET + 1) * HOURS
    series_of_cell = np.arange(n_cells) // ((MAX_LEAD_BUCKET + 1) * HOURS)
    aggregates = _month_aggregates(cases=cases, flags=flags, n_cells=n_cells)
    n_months = aggregates["n_all"].shape[0]
    point = _statistics(totals=aggregates, n_series=n_series, series_of_cell=series_of_cell)
    generator = np.random.default_rng(RESAMPLE_SEED)
    draws = []
    for _ in range(N_RESAMPLES):
        picked = generator.integers(0, n_months, size=n_months)
        resampled = {name: value[picked] for name, value in aggregates.items()}
        draws.append(
            _statistics(totals=resampled, n_series=n_series, series_of_cell=series_of_cell)
        )
    rows = []
    for name, value in point.items():
        sample = np.array([draw[name] for draw in draws], dtype=float)
        rows.append(
            {
                "statistic": name,
                "point": value,
                "lower_95": float(np.nanpercentile(sample, 2.5)),
                "upper_95": float(np.nanpercentile(sample, 97.5)),
            }
        )
    return pl.DataFrame(rows)


def per_series(*, cases: Cases, flags: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return each series' half-hour share and score share of the flagged cases (points only)."""
    frame = cases.frame.with_columns(
        a_any=pl.Series(flags["a_any"]),
        b=pl.Series(flags["b"]),
        flagged=pl.Series(flags["flagged"]),
    )
    return (
        frame.group_by("time_series_id")
        .agg(
            n=pl.len(),
            n_a=pl.col("a_any").sum(),
            n_b=pl.col("b").sum(),
            half_hour_share=100 * pl.col("flagged").mean(),
            qs_share=100 * (pl.col("qs") * pl.col("flagged")).sum() / pl.col("qs").sum(),
            tw_p99_share=100
            * (pl.col("tw_p99") * pl.col("flagged")).sum()
            / pl.col("tw_p99").sum(),
        )
        .sort("time_series_id")
    )


def main() -> None:
    """Score every event set and write the results."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    events = load_events(series=cases.series)
    first_valid, last_valid = cases.frame["valid_time"].min(), cases.frame["valid_time"].max()
    print(
        f"cases {cases.frame.height}, series {len(cases.series)}, "
        f"valid {first_valid} to {last_valid}"
    )
    results = []
    series_tables = []
    for event_set in EVENT_SETS:
        chosen = events.filter(pl.col(event_set))
        in_year = chosen.filter((pl.col("end") >= first_valid) & (pl.col("onset") <= last_valid))
        print(
            f"\n== {event_set}: {chosen['event_id'].n_unique()} events "
            f"({chosen.height} event-series), "
            f"{in_year['event_id'].n_unique()} overlap the scored year"
        )
        flags = flag_cases(cases=cases, events=chosen)
        table = score(cases=cases, flags=flags).with_columns(event_set=pl.lit(event_set))
        results.append(table)
        series_tables.append(
            per_series(cases=cases, flags=flags).with_columns(event_set=pl.lit(event_set))
        )
        print(table.filter(pl.col("statistic").str.starts_with("half_hour_share")))
        print(table.filter(pl.col("statistic").str.starts_with("score_share")))
        print(table.filter(pl.col("statistic").str.starts_with("bound")))
        in_year.write_parquet(WORK_DIR / f"events_in_year_{event_set}.parquet")
    pl.concat(results).write_parquet(WORK_DIR / "results.parquet")
    pl.concat(series_tables).write_parquet(WORK_DIR / "per_series.parquet")


if __name__ == "__main__":
    main()
