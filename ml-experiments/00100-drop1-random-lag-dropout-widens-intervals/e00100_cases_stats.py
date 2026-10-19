"""Contrasts on experiment 00088's held-out injected cases, for experiment 00100.

Reads the cells of `e00100_cases.py` (one row per variant, case, lead band, and target kind),
averages each case's metrics over its lead bands weighted by the number of valid times, pairs the
variants on the case, and resamples whole onset months. The four shapes S2, S3, S4, and S6 are
pooled with equal weight: each resample averages the shapes' mean differences.

    uv run python ml-experiments/00100-*/e00100_cases_stats.py
"""

from dataclasses import dataclass
from typing import Final

import e00100_common as common
import numpy as np
import polars as pl

from studies import bootstrap

PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4.0)
N_RESAMPLES_PLANNED: Final[int] = 20_000
N_RESAMPLES_95: Final[int] = 4_000
METRICS: Final[tuple[str, ...]] = (
    "qs",
    "cov90",
    "cov98",
    "w90",
    "w98",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "below01",
    "below05",
    "ex95",
    "ex99",
    "mae",
)
KEY: Final[tuple[str, ...]] = ("shape", "seed", "event_id", "series", "origin")
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("drop10_inj", "plain"),
    ("mask_inj", "plain"),
    ("drop10_inj", "mask_inj"),
    ("drop30_inj", "drop10_inj"),
    ("drop30_inj", "plain"),
    ("plain_masked_inj", "plain"),
    ("drop10_inj", "plain_masked_inj"),
    ("ceiling", "plain"),
)
PLANNED: Final[dict[str, tuple[str, str, str]]] = {
    "P2": ("drop10_inj", "plain", "cov90"),
    "G1": ("drop10_inj", "plain", "qs"),
}


@dataclass(frozen=True)
class Scope:
    """Which cases a contrast covers.

    Attributes:
        family: `A` (the newest usable lag is inside the transfer) or `B` (after its end).
        target: `all` valid times, or only the valid times `inside` the transfer.
        hours: A name for a range of hours since onset (family A) or `all`.
    """

    family: str
    target: str


def load_cases() -> pl.DataFrame:
    """Return one row per variant and case with metrics averaged over lead bands, plus the month."""
    cells = pl.concat(
        [pl.read_parquet(path) for path in sorted(common.CASES_DIR.glob("cells_*.parquet"))],
        how="diagonal",
    )
    events = pl.concat(
        [pl.read_parquet(path) for path in sorted(common.CASES_DIR.glob("events_*.parquet"))]
    ).select("shape", "seed", "event_id", "onset_month")
    weighted = cells.group_by(*KEY, "family", "variant", "target").agg(
        pl.col("n").sum().alias("n"),
        *[(pl.col(m) * pl.col("n")).sum().alias(m) for m in METRICS],
        pl.col("hours").first(),
    )
    weighted = weighted.with_columns([pl.col(m) / pl.col("n") for m in METRICS])
    return weighted.join(events, on=["shape", "seed", "event_id"], how="left", validate="m:1")


def paired(
    *, cases: pl.DataFrame, treatment: str, reference: str, scope: Scope, metric: str
) -> pl.DataFrame:
    """Return the treatment and reference value per case in a scope."""
    base = cases.filter((pl.col("family") == scope.family) & (pl.col("target") == scope.target))
    left = base.filter(pl.col("variant") == treatment).select(
        *KEY, "shape", "onset_month", t=pl.col(metric)
    )
    right = base.filter(pl.col("variant") == reference).select(*KEY, r=pl.col(metric))
    return left.join(right, on=list(KEY), how="inner", validate="1:1")


def shape_month_bounds(
    *, frame: pl.DataFrame, level: float, n_resamples: int
) -> tuple[float, float, float]:
    """Return the equal-shape-weight mean difference and its interval, resampling onset months."""
    shapes = sorted(frame["shape"].unique().to_list())
    months = sorted(frame["onset_month"].unique().to_list())
    sums = np.zeros((len(shapes), len(months)))
    counts = np.zeros((len(shapes), len(months)))
    for i, shape in enumerate(shapes):
        part = frame.filter(pl.col("shape") == shape)
        for j, month in enumerate(months):
            values = part.filter(pl.col("onset_month") == month)
            sums[i, j] = float(np.sum(values["t"].to_numpy() - values["r"].to_numpy()))
            counts[i, j] = values.height
    point = float(np.mean(sums.sum(axis=1) / counts.sum(axis=1)))
    generator = np.random.default_rng(bootstrap.BOOTSTRAP_SEED)
    resampled = np.empty(n_resamples)
    for draw in range(n_resamples):
        drawn = generator.integers(0, len(months), size=len(months))
        resampled[draw] = np.mean(sums[:, drawn].sum(axis=1) / counts[:, drawn].sum(axis=1))
    tail = (100.0 - level) / 2.0
    return (
        point,
        float(np.percentile(resampled, tail)),
        float(np.percentile(resampled, 100.0 - tail)),
    )


def main() -> None:
    """Write the injected-case contrasts and absolute means."""
    cases = load_cases()
    rows: list[dict[str, object]] = []
    for family in ("A", "B"):
        for target in ("all", "inside"):
            scope = Scope(family=family, target=target)
            for treatment, reference in PAIRS:
                for metric in METRICS:
                    frame = paired(
                        cases=cases,
                        treatment=treatment,
                        reference=reference,
                        scope=scope,
                        metric=metric,
                    )
                    if frame.is_empty():
                        continue
                    planned = [
                        name
                        for name, (t, r, m) in PLANNED.items()
                        if (t, r, m) == (treatment, reference, metric)
                        and family == "A"
                        and target == "all"
                    ]
                    levels = [("95", 95.0, N_RESAMPLES_95)]
                    if planned:
                        levels.insert(0, ("planned", PLANNED_LEVEL, N_RESAMPLES_PLANNED))
                    for label, level, n_resamples in levels:
                        point, lower, upper = shape_month_bounds(
                            frame=frame, level=level, n_resamples=n_resamples
                        )
                        rows.append(
                            {
                                "id": planned[0] if planned and label == "planned" else "D",
                                "family": family,
                                "target": target,
                                "treatment": treatment,
                                "reference": reference,
                                "metric": metric,
                                "level_label": label,
                                "difference": point,
                                "lower": lower,
                                "upper": upper,
                                "treatment_mean": float(np.mean(frame["t"].to_numpy())),
                                "reference_mean": float(np.mean(frame["r"].to_numpy())),
                                "ratio": float(
                                    np.sum(frame["t"].to_numpy()) / np.sum(frame["r"].to_numpy())
                                ),
                                "cases": frame.height,
                                "months": frame["onset_month"].n_unique(),
                            }
                        )
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    common.atomic_write(
        frame=pl.DataFrame(rows, infer_schema_length=None),
        target=common.RESULTS_DIR / "injected_contrasts.parquet",
    )
    print(f"wrote {len(rows):,} contrast rows from {cases.height:,} case rows", flush=True)


if __name__ == "__main__":
    main()
