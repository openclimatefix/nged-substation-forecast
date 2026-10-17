"""Inputs of experiment 84: the pilot node table, the metered flows, and experiment 48's arms.

The node table joins generators to substations, so it is private: `write_private_tables` writes it,
with a lineage note, under `data/ml-experiments/`, which is never committed. Nothing here prints a
generator's identifier, capacity, or output on its own.
"""

import datetime as dt
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import polars as pl
from contracts.settings import PROJECT_ROOT
from nged_data.storage import scan_cleaned_power
from studies.node_curtailment import NodeTable, node_table_pilot, scale_to_megawatts
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for
from studies.tail_calibration import read_arms

EXPERIMENT_ID: Final[int] = 84
SLUG: Final[str] = "substation-limit-curtailment-cost-worked-example"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
PRIVATE_DIR: Final[Path] = DATA_DIR / "private"
"""Holds the tables that join a generator to a node, and their lineage note. Never committed."""

SOURCE_EXPERIMENT_ID: Final[int] = 48
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
"""The first instant of the scored year; scaled limits use only flows strictly before it."""
SCORED_END: Final[dt.datetime] = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
"""The holdout starts here, so no flow on or after it is read."""

METADATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "NGED" / "metadata.parquet"
SUBSTATIONS_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
)
CLEANED_POWER_PATH: Final[Path] = PROJECT_ROOT / "data" / "NGED" / "cleaned_power_time_series.delta"
CAPACITY_PATH: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156")
    / "frames"
    / "combined"
    / "fold_a"
    / "capacity.parquet"
)

BOSTON: Final[str] = "BOSTON 33kV S STN"
SKEGNESS: Final[str] = "SKEGNESS 33kV S STN"
BICKER_FEN: Final[str] = "BICKER FEN 132kV S STN"
WALPOLE: Final[str] = "WALPOLE 132kV S STN"

NODE_LABELS: Final[Mapping[str, str]] = {
    BOSTON: "Boston",
    SKEGNESS: "Skegness",
    BICKER_FEN: "Bicker Fen",
    WALPOLE: "Walpole",
}
"""Short names for charts and the page; the four nodes are substations, which may be named."""

EXPORT_LIMITS_MW: Final[Mapping[str, tuple[float, ...]]] = {
    BOSTON: (45.0, 45.1),
    SKEGNESS: (29.8, 31.3),
    BICKER_FEN: (263.5,),
    WALPOLE: (73.4,),
}
"""NGED's published per-transformer export pre-event limits (Boston, Skegness) and Technical Limits.

Supported by NGED Open Data. The firm limit of a bulk supply point is the smaller of its pair. The
two grid supply points carry one Technical Limit each (Walpole lists 3 of its 5 transformers), and
neither is used for scoring because their signs are unconfirmed.
"""

SIGNS: Final[Mapping[str, tuple[int | None, str]]] = {
    BOSTON: (-1, "our data: a positive reading is import (matches the Primaries beneath it)"),
    SKEGNESS: (-1, "our data: a positive reading is import (matches the Primaries beneath it)"),
    BICKER_FEN: (1, "UNCONFIRMED: a positive reading looks like export; awaiting NGED"),
    WALPOLE: (None, "unknown; awaiting NGED"),
}
"""Each node's `export_sign` and where it came from."""

SCORED_NODES: Final[tuple[str, ...]] = (BOSTON, SKEGNESS)
"""The two bulk supply points that are scored."""


def read_capacity_mw() -> dict[int, float]:
    """Return each series' effective capacity in MW, as experiment 48 normalised by."""
    capacity = pl.read_parquet(CAPACITY_PATH)
    return {
        int(s): float(c)
        for s, c in zip(capacity["time_series_id"], capacity["effective_capacity_mw"], strict=True)
    }


def read_arm_frames(*, names: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Read experiment 48's named prediction sets, dropping the label columns."""
    raw = read_arms(experiment_id=SOURCE_EXPERIMENT_ID, names=names, row_set_id=ROW_SET_ID)
    return {
        name: frame.drop("site", "group", "lead_hours", strict=False) for name, frame in raw.items()
    }


def build_pilot_table(*, capacity_mw: Mapping[int, float], forecast_ids: set[int]) -> NodeTable:
    """Build the pilot node and generator tables from NGED's private files."""
    return node_table_pilot(
        metadata=pl.read_parquet(METADATA_PATH),
        substations=pl.read_csv(SUBSTATIONS_PATH),
        transformer_export_limits_mw=EXPORT_LIMITS_MW,
        signs=SIGNS,
        capacity_mw=capacity_mw,
        forecast_ids=forecast_ids,
    )


def scoring_table(*, table: NodeTable) -> NodeTable:
    """Return the table of the nodes that are scored: the two bulk supply points.

    The grid supply points are left out of scoring because their signs are unconfirmed and no
    generator is attached to them without the nesting pass. Their node rows are dropped and the
    bulk supply points' parent links are cleared.
    """
    keep = list(SCORED_NODES)
    return NodeTable(
        nodes=table.nodes.filter(pl.col("node_id").is_in(keep)).with_columns(
            parent_node_id=pl.lit(None, dtype=pl.String)
        ),
        limits=table.limits.filter(pl.col("node_id").is_in(keep)),
        generators=table.generators.filter(pl.col("node_id").is_in(keep)),
    )


def read_flows(*, table: NodeTable) -> pl.DataFrame:
    """Return the metered flows of the table's nodes from cleaned, unflagged power.

    Only our own data is read, never NGED's public-portal flows, and nothing from the holdout.
    """
    series = table.nodes["time_series_id"].to_list()
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


def to_megawatts(
    *, frames: Mapping[str, pl.DataFrame], capacity_mw: Mapping[int, float]
) -> dict[str, pl.DataFrame]:
    """Convert each frame from fractions of capacity to MW."""
    return {n: scale_to_megawatts(frame=f, capacity_mw=capacity_mw) for n, f in frames.items()}


def write_private_tables(*, table: NodeTable, notes: str) -> None:
    """Write the generator-to-node tables and the lineage note to the private store."""
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    table.nodes.write_parquet(PRIVATE_DIR / "nodes.parquet")
    table.limits.write_parquet(PRIVATE_DIR / "limits.parquet")
    table.generators.write_parquet(PRIVATE_DIR / "generators.parquet")
    (PRIVATE_DIR / "LINEAGE.md").write_text(notes)
