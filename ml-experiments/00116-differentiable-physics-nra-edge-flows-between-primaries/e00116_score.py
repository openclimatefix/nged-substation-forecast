"""Scoring helpers for experiment 116: spreads, quantiles, quantile scores, and intervals.

The quantile score is twice the mean pinball loss over the 11 quantiles of experiment 87, in
points of effective capacity. The spread of an arm is fitted on the arm's own pipeline run over the
year before the scored year (`e00116_run.py replica`), so its residuals are as stale as the scored
forecasts, and it is centred on each group's median.
"""

from pathlib import Path
from typing import Final

import e00116_common as common
import numpy as np
import polars as pl
from studies.dp_edge_flow import LeadHoursSpread, seed_month_bootstrap
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS

BURN_IN_HALF_HOURS: Final[int] = 14 * 48
"""The first days of a replica's origins are left out of its spread, while the filter converges."""
N_RESAMPLES: Final[int] = 4000
"""Resamples of the ratio bootstraps."""
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3.0)
"""The coverage of one of the three planned contrasts, 98.33%."""
SPREAD_ARM: Final[dict[str, str]] = {"a0": "a0", "a4": "a4", "a5": "a5", "or": "a4"}
"""Whose spread each arm uses; the oracle has no replica, so it borrows A4's."""


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points.

    Args:
        actual: Measured values, shape (n,).
        quantiles: Predicted quantiles, shape (n, 11).

    Returns:
        The quantile score of each row.
    """
    levels = np.array(QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def read_replica(*, year_name: str, arm: str) -> pl.DataFrame:
    """Read every chunk of an arm's replica forecasts for a scored year."""
    folder = common.WORK_DIR / "replica"
    return pl.concat(
        [pl.read_parquet(path) for path in sorted(folder.glob(f"{year_name}_{arm}_*"))]
    )


def fit_spreads(
    *, cache: FeatureCache, year_name: str, arms: tuple[str, ...] = ("a0", "a4", "a5")
) -> dict[tuple[str, str], LeadHoursSpread]:
    """Fit the lead-hours spread of each arm and node from its replica forecasts.

    Args:
        cache: The feature cache, for the calendar band and quarter of each target.
        year_name: The scored year (`replicate` or `scored`).
        arms: The arms to fit.

    Returns:
        The spread by (arm, node), fitted outside the hand-filtered events.
    """
    spreads: dict[tuple[str, str], LeadHoursSpread] = {}
    for arm in arms:
        frame = read_replica(year_name=year_name, arm=arm).filter(~pl.col("switching"))
        for node in sorted(frame["node"].unique().to_list()):
            part = frame.filter(pl.col("node") == node)
            target = part["target"].to_numpy()
            spreads[(arm, node)] = LeadHoursSpread.fit(
                residuals=(part["actual"] - part["gam"] - part["correction"]).to_numpy(),
                band=cache.calendar.band[target],
                quarter=cache.calendar.quarter[target],
                lead_hours=part["lead_hours"].to_numpy(),
            )
    return spreads


def with_quantiles(
    *, cache: FeatureCache, rows: pl.DataFrame, spreads: dict[tuple[str, str], LeadHoursSpread]
) -> pl.DataFrame:
    """Add the mean, the 11 quantiles, and the quantile score to transfer rows.

    The mean is the frozen fit plus the level, the anomaly, and the flow term. The scored target
    is `actual`, the injected power; `clean` is the uninjected power, for the normal-running score.

    Args:
        cache: The feature cache.
        rows: Long rows from `e00116_run.py transfers`.
        spreads: The spreads from `fit_spreads`.

    Returns:
        The rows with `mean`, `q01` to `q99`, `score`, `score_nra` (the quantile score against the
        uninjected power of the forecast without its flow term), and `month`.
    """
    out = []
    for (arm, node), part in rows.group_by(["arm", "node"], maintain_order=True):
        target = part["target"].to_numpy()
        mean = (part["gam"] + part["level"] + part["anomaly"] + part["flow"]).to_numpy()
        mean_nra = (part["gam"] + part["level"] + part["anomaly"]).to_numpy()
        spread = spreads[(SPREAD_ARM[arm], node)]
        common_args = {
            "band": cache.calendar.band[target],
            "quarter": cache.calendar.quarter[target],
            "lead_hours": part["lead_hours"].to_numpy(),
        }
        quantiles = np.sort(spread.quantiles(mean=mean, **common_args), axis=1)
        quantiles_nra = np.sort(spread.quantiles(mean=mean_nra, **common_args), axis=1)
        month = cache.end_times[target].astype("datetime64[M]").astype(int)
        out.append(
            part.with_columns(
                pl.Series("mean", mean),
                *(pl.Series(c, quantiles[:, k]) for k, c in enumerate(QUANTILE_COLUMNS)),
                pl.Series(
                    "score", quantile_score(actual=part["actual"].to_numpy(), quantiles=quantiles)
                ),
                pl.Series(
                    "score_nra",
                    quantile_score(actual=part["clean"].to_numpy(), quantiles=quantiles_nra),
                ),
                pl.Series("month", month),
                pl.Series("abs_error", np.abs(part["actual"].to_numpy() - mean)),
                pl.Series("abs_error_nra", np.abs(part["clean"].to_numpy() - mean_nra)),
            )
        )
    return pl.concat(out)


def attach_seeds(*, rows: pl.DataFrame, transfers: pl.DataFrame) -> pl.DataFrame:
    """Add each transfer's seed, size, duration, and edge to its rows."""
    return rows.join(
        transfers.select("transfer_id", "seed", "size", "duration", "node_a", "node_b"),
        on="transfer_id",
        how="left",
    )


def paired_contrast(
    *,
    rows: pl.DataFrame,
    treatment: str,
    reference: str,
    metric: str,
    level: float = BONFERRONI_LEVEL,
) -> dict[str, float]:
    """Bound the paired difference of two arms on the same rows, resampling months within seeds.

    Args:
        rows: Rows with `arm`, `transfer_id`, `node`, `cutoff`, `target`, `seed`, `month`, and the
            metric.
        treatment: The arm compared.
        reference: The arm it is compared against.
        metric: The column to difference.
        level: The coverage in percent.

    Returns:
        The estimate, the interval, the number of rows, and the number of transfers.
    """
    keys = ["transfer_id", "node", "cutoff", "target"]
    paired = (
        rows.filter(pl.col("arm") == reference)
        .select(*keys, "seed", "month", reference=pl.col(metric))
        .join(
            rows.filter(pl.col("arm") == treatment).select(*keys, treatment=pl.col(metric)),
            on=keys,
            how="inner",
        )
    )
    difference = (paired["treatment"] - paired["reference"]).to_numpy()
    estimate, lower, upper = seed_month_bootstrap(
        values=difference,
        seeds=paired["seed"].to_numpy(),
        months=paired["month"].to_numpy(),
        level=level,
    )
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "n_rows": paired.height,
        "n_transfers": paired["transfer_id"].n_unique(),
    }


def write_json(*, path: Path, payload: dict) -> None:
    """Write a result file atomically."""
    import json
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=float))
    temporary.replace(path)
