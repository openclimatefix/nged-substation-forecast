"""Split experiment 00033's best-guess and `manual_heuristic` Primary scores by meter unit.

Nothing is fitted. The script reads the two arms' per-case quantile scores, the observed power over
capacity, and each series' meter unit, cuts the cases by sunny midday and by near-zero flow, and
writes every skill, absolute score, gap, and interval to the experiment's private data folder.

    uv run python ml-experiments/00056-mva-only-primaries-skill-against-mw-primaries/e00056_score.py
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00056-mva-only-primaries-skill-against-mw-primaries"
)
METRICS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "metrics"
    / "Primary"
)
OBSERVED_PATH: Final[Path] = (
    study_dir_for(study="stage1_pilot") / "demand" / "inputs" / "observed.parquet"
)
SERIES_PATH: Final[Path] = (
    study_dir_for(study="stage1_pilot") / "demand" / "inputs" / "series.parquet"
)
"""Experiment 00003's series table, which holds `effective_capacity_mw`."""
METADATA_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
BEST_GUESS: Final[str] = "n1_cs"
BENCHMARK: Final[str] = "manual_heuristic"
HOLDOUT: Final[str] = "2026-07-01"
"""No experiment scores a valid time on or after this date."""
SUNNY_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
SUNNY_HOURS: Final[tuple[int, ...]] = (10, 11, 12, 13)
"""Valid-time hours (UTC) from 10:00 to 13:30."""
NEAR_ZERO_OF_CAPACITY: Final[float] = 0.10
LOW_SHARE_OF_P99: Final[float] = 0.10
THRESHOLD_POINTS: Final[float] = 5.0
N_RESAMPLES: Final[int] = 10_000
RESAMPLE_SEED: Final[int] = 20261016
GROUPS: Final[tuple[str, ...]] = ("MW", "MVA")
CUTS: Final[dict[str, tuple[bool | None, bool | None]]] = {
    "all": (None, None),
    "sunny_midday": (True, None),
    "other_hours": (False, None),
    "near_zero_flow": (None, True),
    "away_from_zero": (None, False),
    "sunny_midday_near_zero": (True, True),
    "sunny_midday_away_from_zero": (True, False),
}
"""For each cut, the required (sunny midday, near zero flow) values; None means either."""


def load_aggregates() -> pl.DataFrame:
    """Return the cases summed per series, month, sunny-midday flag, and near-zero flag.

    Returns:
        Columns `time_series_id`, `units`, `month`, `sunny`, `near_zero`, `n`, `sum_best`, and
        `sum_benchmark` (sums of the quantile score over the group's cases, seeds averaged).
    """
    arms = [
        (
            pl.read_parquet(METRICS_DIR / f"{arm}.parquet")
            .filter(
                pl.col("shared")
                & (
                    pl.col("valid_time")
                    < pl.lit(HOLDOUT).str.to_datetime("%Y-%m-%d", time_zone="UTC")
                )
            )
            .group_by("time_series_id", "power_fcst_init_time", "valid_time")
            .agg(pl.col("qs").mean().alias(arm))
        )
        for arm in (BEST_GUESS, BENCHMARK)
    ]
    cases = arms[0].join(
        arms[1],
        on=["time_series_id", "power_fcst_init_time", "valid_time"],
        how="inner",
        validate="1:1",
    )
    if cases.height != arms[0].height or cases.height != arms[1].height:
        msg = "the two arms do not cover the same cases"
        raise ValueError(msg)
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "units"]).with_columns(
        pl.col("units").cast(pl.String)
    )
    capacity = pl.read_parquet(SERIES_PATH, columns=["time_series_id", "effective_capacity_mw"])
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    joined = (
        cases.join(metadata, on="time_series_id", how="left", validate="m:1")
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .join(observed, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            month=pl.col("valid_time").dt.year() * 12 + pl.col("valid_time").dt.month(),
            sunny=pl.col("valid_time").dt.month().is_in(SUNNY_MONTHS)
            & pl.col("valid_time").dt.hour().is_in(SUNNY_HOURS),
            near_zero=(pl.col("power") / pl.col("effective_capacity_mw")).abs()
            <= NEAR_ZERO_OF_CAPACITY,
        )
    )
    if (
        joined["power"].null_count()
        or joined["power"].is_nan().any()
        or joined["units"].null_count()
        or joined["effective_capacity_mw"].null_count()
    ):
        msg = "some cases have no observed power, meter unit, or capacity"
        raise ValueError(msg)
    return joined.group_by("time_series_id", "units", "month", "sunny", "near_zero").agg(
        n=pl.len(),
        sum_best=pl.col(BEST_GUESS).sum(),
        sum_benchmark=pl.col(BENCHMARK).sum(),
    )


def _cut_mask(*, table: pl.DataFrame, cut: str) -> pl.Expr:
    """Return the expression selecting the rows of `table` in `cut`."""
    sunny, near_zero = CUTS[cut]
    mask = pl.lit(True)
    if sunny is not None:
        mask &= pl.col("sunny") == sunny
    if near_zero is not None:
        mask &= pl.col("near_zero") == near_zero
    return mask


def _skill(*, best: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    """Return 100 x (1 - best / benchmark), elementwise over the last axis."""
    return 100 * (1 - best / benchmark)


def score(*, aggregates: pl.DataFrame) -> pl.DataFrame:
    """Return, per cut, the absolute scores, skills, and MW-minus-MVA gap with 95% intervals.

    Two intervals are given: one resampling whole months (the series fixed), one resampling the
    series within each unit group (the months fixed).
    """
    generator = np.random.default_rng(RESAMPLE_SEED)
    months = np.sort(aggregates["month"].unique().to_numpy())
    month_draws = generator.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    series_by_group = {
        group: np.sort(aggregates.filter(pl.col("units") == group)["time_series_id"].unique())
        for group in GROUPS
    }
    rows = []
    for cut in CUTS:
        selected = aggregates.filter(_cut_mask(table=aggregates, cut=cut))
        by_month: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        by_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for group in GROUPS:
            mine = selected.filter(pl.col("units") == group)
            month_sums = mine.group_by("month").agg(pl.col("n", "sum_best", "sum_benchmark").sum())
            aligned = (
                pl.DataFrame({"month": months})
                .join(month_sums, on="month", how="left")
                .fill_null(0)
            )
            by_month[group] = (
                aligned["n"].to_numpy().astype(float),
                aligned["sum_best"].to_numpy().astype(float),
                aligned["sum_benchmark"].to_numpy().astype(float),
            )
            series_sums = mine.group_by("time_series_id").agg(
                pl.col("n", "sum_best", "sum_benchmark").sum()
            )
            aligned = (
                pl.DataFrame({"time_series_id": series_by_group[group]})
                .join(series_sums, on="time_series_id", how="left")
                .fill_null(0)
            )
            by_series[group] = (
                aligned["n"].to_numpy().astype(float),
                aligned["sum_best"].to_numpy().astype(float),
                aligned["sum_benchmark"].to_numpy().astype(float),
            )
        for label, sums, draws in (
            ("months", by_month, month_draws),
            ("series", by_series, None),
        ):
            stats: dict[str, tuple[float, np.ndarray]] = {}
            for group in GROUPS:
                n, best, benchmark = sums[group]
                if draws is None:
                    picks = generator.integers(0, len(n), size=(N_RESAMPLES, len(n)))
                else:
                    picks = draws
                with np.errstate(invalid="ignore", divide="ignore"):
                    point_skill = float(_skill(best=best.sum(), benchmark=benchmark.sum()))
                    draw_skill = _skill(best=best[picks].sum(1), benchmark=benchmark[picks].sum(1))
                    draw_best = best[picks].sum(1) / n[picks].sum(1)
                    draw_benchmark = benchmark[picks].sum(1) / n[picks].sum(1)
                stats[f"skill_{group}"] = (point_skill, draw_skill)
                stats[f"best_score_{group}"] = (best.sum() / n.sum(), draw_best)
                stats[f"benchmark_score_{group}"] = (benchmark.sum() / n.sum(), draw_benchmark)
                stats[f"n_{group}"] = (n.sum(), np.full(N_RESAMPLES, n.sum()))
            stats["skill_gap_MW_minus_MVA"] = (
                stats["skill_MW"][0] - stats["skill_MVA"][0],
                stats["skill_MW"][1] - stats["skill_MVA"][1],
            )
            for name, (point, draw) in stats.items():
                rows.append(
                    {
                        "cut": cut,
                        "resampled": label,
                        "statistic": name,
                        "point": float(point),
                        "lower_95": float(np.nanpercentile(draw, 2.5)),
                        "upper_95": float(np.nanpercentile(draw, 97.5)),
                    }
                )
    return pl.DataFrame(rows)


def per_series(*, aggregates: pl.DataFrame) -> pl.DataFrame:
    """Return each series' skill and absolute scores for every cut (points only)."""
    tables = [
        (
            aggregates.filter(_cut_mask(table=aggregates, cut=cut))
            .group_by("time_series_id", "units")
            .agg(pl.col("n", "sum_best", "sum_benchmark").sum())
            .with_columns(
                cut=pl.lit(cut),
                skill=100 * (1 - pl.col("sum_best") / pl.col("sum_benchmark")),
                best_score=pl.col("sum_best") / pl.col("n"),
                benchmark_score=pl.col("sum_benchmark") / pl.col("n"),
            )
        )
        for cut in CUTS
    ]
    return pl.concat(tables).sort("cut", "units", "time_series_id")


def low_magnitude_counts() -> pl.DataFrame:
    """Return, per series, the sunny-midday half-hours whose magnitude is under 10% of its P99.

    The P99 is the series' own 99th percentile of the magnitude of the observed power
    across all its history. Counts are given for the scored year (valid time from 2025-07-01) and
    for all history.
    """
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "units"]).with_columns(
        pl.col("units").cast(pl.String)
    )
    series = pl.read_parquet(METRICS_DIR / f"{BEST_GUESS}.parquet", columns=["time_series_id"])
    primaries = series["time_series_id"].unique().to_list()
    observed = (
        pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
        .filter(pl.col("time_series_id").is_in(primaries))
        .drop_nulls("power")
        .filter(pl.col("valid_time") < pl.lit(HOLDOUT).str.to_datetime("%Y-%m-%d", time_zone="UTC"))
        .with_columns(magnitude=pl.col("power").abs())
        .with_columns(
            p99=pl.col("magnitude").quantile(0.99).over("time_series_id"),
            sunny=pl.col("valid_time").dt.month().is_in(SUNNY_MONTHS)
            & pl.col("valid_time").dt.hour().is_in(SUNNY_HOURS),
            scored_year=pl.col("valid_time").dt.date() >= pl.date(2025, 7, 1),
        )
        .filter(pl.col("sunny"))
        .with_columns(low=pl.col("magnitude") < LOW_SHARE_OF_P99 * pl.col("p99"))
    )
    return (
        observed.group_by("time_series_id")
        .agg(
            n_sunny_all=pl.len(),
            n_low_all=pl.col("low").sum(),
            n_sunny_year=pl.col("scored_year").sum(),
            n_low_year=(pl.col("low") & pl.col("scored_year")).sum(),
        )
        .join(metadata, on="time_series_id", how="left", validate="1:1")
        .with_columns(
            share_low_all=100 * pl.col("n_low_all") / pl.col("n_sunny_all"),
            share_low_year=100 * pl.col("n_low_year") / pl.col("n_sunny_year"),
        )
        .sort("units", "time_series_id")
    )


def main() -> None:
    """Score every cut and write the tables."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    aggregates = load_aggregates()
    print(f"{aggregates['n'].sum()} cases, {aggregates['time_series_id'].n_unique()} series")
    results = score(aggregates=aggregates)
    results.write_parquet(WORK_DIR / "results.parquet")
    per_series(aggregates=aggregates).write_parquet(WORK_DIR / "per_series.parquet")
    counts = low_magnitude_counts()
    counts.write_parquet(WORK_DIR / "low_magnitude_counts.parquet")
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60, tbl_width_chars=200):
        print(results.filter(pl.col("statistic").str.starts_with("skill")))
        print(results.filter(pl.col("statistic").str.contains("score")))
        print(counts)
        print(
            per_series(aggregates=aggregates).filter(pl.col("cut").is_in(["all", "sunny_midday"]))
        )


if __name__ == "__main__":
    main()
