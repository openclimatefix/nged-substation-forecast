"""Inputs of experiment 277: stored forecasts of the existing models, as frames in MW.

Nothing here builds or refits a forecast. The Primaries' forecasts are experiment 251's stored
arrays (which hold experiment 87's BG2, climatology and manual heuristic, and experiment 220's GAM
stack, arm 1). The bulk supply points' forecasts are experiment 204's top-down and bottom-up
frames plus experiment 48's climatology and manual heuristic.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from nged_data.storage import scan_cleaned_power
from studies.node_curtailment import NodeTable, scale_to_megawatts
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 277
SLUG: Final[str] = "lsc-curtailment-metric-on-existing-forecasts"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
OUT_DIR: Final[Path] = DATA_DIR / "work"

E251_SCORED: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2"
    / "work"
    / "arms"
    / "scored"
)
E87_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00087-bg2-best-guess-assembled-with-drop-one-ablations"
    / "predictions"
)
E48_PREDICTIONS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00048-weekly-retraining-confirmation-and-extension-demand-generators"
    / "predictions"
)
E204_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00204-dp-bsp-full-bottom-up-with-signed-mva-children"
    / "work"
    / "dp_preds"
)
E161_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00161-dp-bsp-top-down-against-bottom-up" / "work" / "dp_preds"
)
E163_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00163-switching-real-event-comparison-gam-stack-against-bg2"
)
CAPACITY_PATH: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156")
    / "frames"
    / "combined"
    / "fold_a"
    / "capacity.parquet"
)
CLEANED_POWER_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"

SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
"""First scored valid time; scaled limits and tail thresholds use only power strictly before it."""
SCORED_END: Final[dt.datetime] = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)

PRIMARIES: Final[dict[str, int]] = {"D5": 17, "D9": 7, "D10": 10, "D11": 8, "D14": 16, "D16": 18}
"""Anonymised Primary substation label to time series id."""
BSPS: Final[dict[str, int]] = {"D13": 2, "D17": 1}
"""Anonymised bulk supply point label to time series id."""
BSP_CHILDREN: Final[dict[str, tuple[str, ...]]] = {
    "D13": ("D2", "D4", "D9", "D10", "D11", "D19", "D21"),
    "D17": ("D3", "D5", "D8", "D12", "D14", "D15", "D16", "D18", "D22"),
}

QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time"]

PRIMARY_ARMS: Final[tuple[str, ...]] = (
    "climatology",
    "manual_heuristic",
    "bg2",
    "arm1",
    "xgboost",
)
"""Stored arm names of experiment 251; `arm1` is the GAM stack, `xgboost` the weak reference."""
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
    "bg2": "XGBoost BG2",
    "arm1": "GAM stack (arm 1)",
    "xgboost": "Weak XGBoost (reference)",
    "td": "Top-down GAM",
    "bu": "Bottom-up GAM",
    "K0": "GAM-K stack, unmasked (K0)",
}
FRACTIONS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)
HEADLINE_FRACTION: Final[float] = 0.5
PRICE_GBP_PER_MWH: Final[float] = 2000.0
N_RESAMPLES: Final[int] = 20_000
POWER_FOR_80_PERCENT: Final[float] = 2.8
"""Standard errors in the smallest difference an interval excludes zero for at 80% power and 5%."""


def write_json_atomic(*, path: Path, content: object) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def capacity_mw() -> dict[int, float]:
    """Return each series' effective capacity in MW."""
    frame = pl.read_parquet(CAPACITY_PATH)
    return {
        int(s): float(c)
        for s, c in zip(frame["time_series_id"], frame["effective_capacity_mw"], strict=True)
    }


def read_flows(*, series: list[int]) -> pl.DataFrame:
    """Return cleaned, unflagged power in MW before the holdout, as `scaled_limits` reads it."""
    unflagged = scan_cleaned_power(str(CLEANED_POWER_PATH))
    plain = pl.LazyFrame._from_pyldf(unflagged._ldf)  # strip the Patito model, zero-copy
    return (
        plain.filter(pl.col("time_series_id").is_in(series), pl.col("time") < SCORED_END)
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            pl.col("time").alias("valid_time"),
            flow_mw=pl.col("power").cast(pl.Float64),
        )
        .collect()
    )


def node_table(*, sites: dict[str, int]) -> NodeTable:
    """Return a node-only table: every site a node, positive power the flow that can breach.

    The sign is assumed, as in experiment 251: a positive demand reading is the flow that could
    exceed a limit. The table has no generators, so planned node curtailment is assumed delivered.
    """
    nodes = pl.DataFrame(
        {
            "node_id": list(sites),
            "time_series_id": list(sites.values()),
            "node_type": ["BSP"] * len(sites),
            "parent_node_id": [None] * len(sites),
            "export_sign": [1] * len(sites),
            "sign_source": ["assumed: a positive demand reading is the flow that can breach"]
            * len(sites),
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )
    limits = pl.DataFrame(
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        }
    )
    generators = pl.DataFrame(
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        }
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators)


def thresholds(*, sites: dict[str, int], capacity: dict[int, float]) -> dict[int, dict[str, float]]:
    """Return each series' 95th and 99th percentile of power before the scored year, in points."""
    flows = read_flows(series=list(sites.values())).filter(pl.col("valid_time") < SCORED_START)
    out: dict[int, dict[str, float]] = {}
    for (series,), group in flows.group_by("time_series_id"):
        points = group["flow_mw"].to_numpy() / capacity[int(series)] * 100.0  # type: ignore[arg-type]
        out[int(series)] = {  # type: ignore[arg-type]
            "95": float(np.quantile(points, 0.95)),
            "99": float(np.quantile(points, 0.99)),
        }
    return out


def read_primary_rows() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return experiment 251's row table and its stored quantile arrays (in points)."""
    rows = pl.read_parquet(
        E251_SCORED / "rows.parquet",
        columns=[
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "seed",
            "actual",
            "mean",
            "site",
        ],
    )
    arms = {name: np.load(E251_SCORED / f"{name}.npy") for name in PRIMARY_ARMS}
    for name, array in arms.items():
        if array.shape != (rows.height, len(QUANTILES)):
            msg = f"{name} has shape {array.shape}, expected ({rows.height}, {len(QUANTILES)})"
            raise ValueError(msg)
    return rows, arms


def stored_means(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each Primary arm's mean forecast (points) where one is stored.

    Only arm 1 has one: experiment 251's own `mean` column, which is arm D's mean before the neural
    head's correction. Experiments 48 and 87 stored quantiles only, so the other arms have no mean.
    """
    return {"arm1": rows["mean"].to_numpy().astype(np.float64)}


def quantile_frame(
    *, rows: pl.DataFrame, quantiles: np.ndarray, capacity: dict[int, float]
) -> pl.DataFrame:
    """Return the node-forecast frame in MW: keys, seed, `actual`, and the eleven quantiles."""
    frame = rows.select(
        pl.col("time_series_id").cast(pl.Int64),
        "power_fcst_init_time",
        "valid_time",
        pl.col("seed").cast(pl.Int64),
        pl.col("actual").cast(pl.Float64),
    ).with_columns(
        [pl.Series(name, quantiles[:, i].astype(np.float64)) for i, name in enumerate(QUANTILES)]
    )
    scaled = frame.with_columns([(pl.col(c) / 100.0) for c in ("actual", *QUANTILES)])
    return scale_to_megawatts(frame=scaled, capacity_mw=capacity)


def quantile_score(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the eleven quantiles, per row."""
    error = actual[:, None] - q
    pin = np.maximum(LEVELS[None, :] * error, (LEVELS[None, :] - 1.0) * error)
    return 2.0 * pin.mean(axis=1)


def per_row_scores(
    *, actual: np.ndarray, q: np.ndarray, mean: np.ndarray | None, edges: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return per-row `qs`, `mae` (median), `mae_mean`, `tw95` and `tw99`, in points."""
    q = q.astype(np.float64)
    actual = actual.astype(np.float64)
    out = {"qs": quantile_score(actual=actual, q=q), "mae": np.abs(actual - q[:, 5])}
    out["mae_mean"] = np.abs(actual - mean) if mean is not None else np.full(len(actual), np.nan)
    for label in ("95", "99"):
        edge = edges[label]
        out[f"tw{label}"] = quantile_score(
            actual=np.maximum(actual, edge), q=np.maximum(q, edge[:, None])
        )
    return out
