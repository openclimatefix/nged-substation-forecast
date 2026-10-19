"""Contrasts, shares of the gap, and tables for experiment 00096, from the per-case cells.

The unit is a case: one (shape, event, series, origin). A case's error is the lead-weighted mean
absolute error of the median forecast over leads up to 10 days, in points of capacity. Cases are
weighted by their event's mean injected MW, so a large transfer counts for more. A statistic is
computed for each shape and averaged over the shapes with equal weight. Intervals resample whole
calendar months of the transfer's onset, holding the shapes' months together.

    uv run python ml-experiments/00096-*/e00096_stats.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00096_common as common
import numpy as np
import polars as pl

HEADLINE_SHAPES: Final[tuple[str, ...]] = ("S2", "S3", "S4", "S6")
ALL_SHAPES: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4", "S6")
SCORED_MONTH_NUMBERS: Final[frozenset[int]] = frozenset({2, 4, 6, 8, 10, 12})
N_CONTRASTS: Final[int] = 4
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_CONTRASTS)
"""The coverage in percent of a two-sided interval Bonferroni-corrected across four contrasts."""
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 9600
METRICS: Final[tuple[str, ...]] = (
    "mae",
    "qs",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cov90",
    "cov98",
    "below01",
    "below05",
    "ex95",
    "ex99",
)
ONSET_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-3 d", 24.0, 72.0),
    ("over 3 d", 72.0, 1e9),
)
STEP_BINS: Final[tuple[tuple[str, float, float], ...]] = (
    ("under 2", -1e9, 2.0),
    ("2 to 4", 2.0, 4.0),
    ("4 to 6", 4.0, 6.0),
    ("6 or more", 6.0, 1e9),
)
ARMS: Final[dict[str, str]] = {
    "plain": "plain",
    "ceiling": "ceiling",
    "real_inj": "nra_real_inj",
    "real_caus": "nra_real_caus",
    "real_cmax": "nra_real_cmax",
    "synth_inj": "nra_synth_inj",
    "synth_caus": "nra_synth_caus",
    "synth_cmax": "nra_synth_cmax",
    "old_inj": "nra_oracle_inj",
    "old_caus": "nra_oracle_caus",
}
"""The arms of the plan. Arm 1 is `real_*` (real-only training); arms 2, 3, and 4 are `synth_inj`,
`synth_caus`, and `synth_cmax` (synthetic-trained, with the injected, RO-NC, and largest-step MW).
`plain` and `ceiling` are experiment 00088's references, and `old_*` are its nine-level
`nra_oracle` scores, used only to check that this experiment's `real_*` scores reproduce them."""
BASELINE: Final[dict[str, str]] = {
    "synth_inj": "real_inj",
    "synth_caus": "real_caus",
    "synth_cmax": "real_cmax",
}
"""The real-trained twin of each synthetic-trained arm, with the same forecast-time MW."""


@dataclass(frozen=True)
class Interval:
    """A point value and its resampling interval."""

    point: float
    low: float
    high: float

    def text(self, *, digits: int = 3) -> str:
        """Return `point (low to high)`."""
        return f"{self.point:+.{digits}f} ({self.low:+.{digits}f} to {self.high:+.{digits}f})"


def case_table(*, cells: list[pl.DataFrame], events: pl.DataFrame) -> pl.DataFrame:
    """Return one row per case with every variant's metrics, plus the event's attributes.

    Args:
        cells: Frames of cells (one row per case, variant, and band): this experiment's, and
            experiment 00088's for the variants it forecast. A missing metric is left null.
        events: The concatenated `events_*.parquet` files.

    Returns:
        A frame keyed by (shape, event_id, series, origin) with `<metric>_<variant>` columns for
        the whole lead range up to 10 days (lead-weighted over the bands), the family, the hours
        since onset (family A) or since end (family B) at the origin, the event's weight `w`, its
        donor step in noise units, its onset month, and whether that month is a scored month.
    """
    keys = ["shape", "event_id", "series", "site", "origin", "family", "hours"]
    weighted = pl.concat(
        [
            part.group_by([*keys, "variant"]).agg(
                n=pl.col("n").sum(),
                **{
                    m: (pl.col(m) * pl.col("n")).sum() / pl.col("n").sum()
                    for m in METRICS
                    if m in part.columns
                },
                caus_mw=pl.col("caus_mw_newest").first(),
                inj_mw=pl.col("inj_mw_newest").first(),
            )
            for part in cells
        ],
        how="diagonal",
    )
    wide = weighted.pivot(on="variant", index=[*keys, "n"], values=[*METRICS])
    info = weighted.group_by(keys).agg(
        caus_mw=pl.col("caus_mw").first(), inj_mw=pl.col("inj_mw").first()
    )
    return (
        wide.join(info, on=keys, how="left", validate="1:1")
        .join(
            events.select(
                "shape",
                "event_id",
                "moved_mw",
                "donor_step_z",
                "onset_month",
                "duration_band",
                "size_band",
                "n_receivers",
                "hours",
            ).rename({"hours": "event_hours"}),
            on=["shape", "event_id"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            w=pl.col("moved_mw").clip(lower_bound=0.0),
            month=(pl.col("onset_month") % 100),
            hours_at_origin=pl.col("hours") + 3.5,
        )
        .with_columns(scored=pl.col("month").is_in(list(SCORED_MONTH_NUMBERS)))
    )


def _by_month(
    *, frame: pl.DataFrame, shapes: tuple[str, ...], numerator: str, weight: str = "w"
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return, for each shape and month, the sums of the weighted numerator and the weight.

    Args:
        frame: Cases.
        shapes: The shapes to keep.
        numerator: A column of per-case values to weight.
        weight: The weight column.

    Returns:
        The months (sorted), and arrays of shape (shapes, months) for the two sums.
    """
    kept = frame.filter(pl.col("shape").is_in(shapes))
    months = sorted(kept["onset_month"].unique().to_list())
    top = np.zeros((len(shapes), len(months)))
    bottom = np.zeros_like(top)
    for i, shape in enumerate(shapes):
        part = kept.filter(pl.col("shape") == shape)
        sums = part.group_by("onset_month").agg(
            top=(pl.col(numerator) * pl.col(weight)).sum(), bottom=pl.col(weight).sum()
        )
        for month, t, b in sums.iter_rows():
            top[i, months.index(month)] = t
            bottom[i, months.index(month)] = b
    return months, top, bottom


def _draws(*, n_months: int) -> np.ndarray:
    """Return the multiplicity of each month in each resample: shape (N_RESAMPLES, months)."""
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, n_months, size=(N_RESAMPLES, n_months))
    return np.stack([np.bincount(row, minlength=n_months) for row in picks])


def _interval(*, point: float, draws: np.ndarray, level: float) -> Interval:
    """Return the percentile interval of resampled statistics around a point value."""
    finite = draws[np.isfinite(draws)]
    if finite.size < draws.size:
        print(f"warning: {draws.size - finite.size} of {draws.size} resamples were not finite")
    if finite.size == 0:
        return Interval(point, float("nan"), float("nan"))
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return Interval(point, float(low), float(high))


def mean_value(
    *, frame: pl.DataFrame, shapes: tuple[str, ...], column: str, level: float = 95.0
) -> Interval:
    """Return the weighted mean of a per-case column, shapes averaged with equal weight."""
    months, top, bottom = _by_month(frame=frame, shapes=shapes, numerator=column)
    if not months:
        return Interval(float("nan"), float("nan"), float("nan"))
    point = float(np.mean(top.sum(axis=1) / bottom.sum(axis=1)))
    counts = _draws(n_months=len(months))
    with np.errstate(invalid="ignore", divide="ignore"):
        resampled = np.mean(
            (counts @ top.T) / (counts @ bottom.T),
            axis=1,
        )
    return _interval(point=point, draws=resampled, level=level)


def contrast(
    *,
    frame: pl.DataFrame,
    shapes: tuple[str, ...],
    arm_a: str,
    arm_b: str,
    metric: str = "mae",
    level: float = LEVEL,
) -> Interval:
    """Return the weighted mean of `arm_a` minus `arm_b` (negative when `arm_a` is better).

    Args:
        frame: Cases, from `case_table`.
        shapes: The shapes pooled with equal weight.
        arm_a: A key of `ARMS`.
        arm_b: A key of `ARMS`.
        metric: One of `METRICS`.
        level: The interval's coverage in percent.

    Returns:
        The weighted mean difference and its interval.
    """
    column = f"d_{metric}_{arm_a}_{arm_b}"
    diff = frame.with_columns(
        (pl.col(f"{metric}_{ARMS[arm_a]}") - pl.col(f"{metric}_{ARMS[arm_b]}")).alias(column)
    ).drop_nulls(column)
    return mean_value(frame=diff, shapes=shapes, column=column, level=level)


def share_of_gap(
    *,
    frame: pl.DataFrame,
    shapes: tuple[str, ...],
    arm: str,
    metric: str = "mae",
    level: float = 95.0,
) -> tuple[Interval, Interval]:
    """Return the share of the arm-1-to-ceiling gap that `arm` closes, and the gap itself.

    The share is the weighted mean of `plain - arm` over the weighted mean of `plain - ceiling`, per
    shape, averaged over the shapes. It is meaningful only where the gap's interval excludes zero.

    Args:
        frame: Cases.
        shapes: The shapes pooled with equal weight.
        arm: A key of `ARMS`.
        metric: One of `METRICS`.
        level: The interval's coverage in percent.

    Returns:
        The share and the gap (plain minus the ceiling, in points of capacity).
    """
    one, ceiling, other = (f"{metric}_{ARMS[a]}" for a in ("plain", "ceiling", arm))
    work = frame.with_columns(
        (pl.col(one) - pl.col(other)).alias("_closed"),
        (pl.col(one) - pl.col(ceiling)).alias("_gap"),
    ).drop_nulls(["_closed", "_gap"])
    months, closed, bottom = _by_month(frame=work, shapes=shapes, numerator="_closed")
    _, gap, _ = _by_month(frame=work, shapes=shapes, numerator="_gap")
    if not months:
        nan = Interval(float("nan"), float("nan"), float("nan"))
        return nan, nan
    point_gap = float(np.mean(gap.sum(axis=1) / bottom.sum(axis=1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        point = float(np.mean(closed.sum(axis=1) / gap.sum(axis=1)))
        counts = _draws(n_months=len(months))
        share = np.mean((counts @ closed.T) / (counts @ gap.T), axis=1)
        gaps = np.mean((counts @ gap.T) / (counts @ bottom.T), axis=1)
    return (
        _interval(point=point, draws=share, level=level),
        _interval(point=point_gap, draws=gaps, level=level),
    )


def month_signs(
    *, frame: pl.DataFrame, shapes: tuple[str, ...], arm_a: str, arm_b: str
) -> dict[int, float]:
    """Return the weighted mean of `arm_a` minus `arm_b` in each onset month, shapes averaged."""
    out: dict[int, float] = {}
    column = f"_d_{arm_a}_{arm_b}"
    diff = frame.with_columns(
        (pl.col(f"mae_{ARMS[arm_a]}") - pl.col(f"mae_{ARMS[arm_b]}")).alias(column)
    ).drop_nulls(column)
    months, top, bottom = _by_month(frame=diff, shapes=shapes, numerator=column)
    for j, month in enumerate(months):
        with np.errstate(invalid="ignore", divide="ignore"):
            out[month] = float(np.nanmean(top[:, j] / bottom[:, j]))
    return out


def absolute_scores(*, frame: pl.DataFrame, shapes: tuple[str, ...]) -> pl.DataFrame:
    """Return every arm's weighted mean of each metric with a 95% interval."""
    rows = []
    for arm, variant in ARMS.items():
        for metric in METRICS:
            diff = frame.with_columns(pl.col(f"{metric}_{variant}").alias("_v")).drop_nulls("_v")
            interval = mean_value(frame=diff, shapes=shapes, column="_v")
            rows.append(
                {
                    "arm": arm,
                    "variant": variant,
                    "metric": metric,
                    "value": interval.point,
                    "low": interval.low,
                    "high": interval.high,
                }
            )
    return pl.DataFrame(rows)


def cases_dir_files(*, pattern: str, directory: Path = common.CASES_DIR) -> list[Path]:
    """Return the sorted per-batch files that match a pattern."""
    return sorted(directory.glob(pattern))
