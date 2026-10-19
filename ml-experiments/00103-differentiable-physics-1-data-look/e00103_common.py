"""Shared paths, constants, and loaders for experiment 103 (differentiable physics, part 1).

Every script in this folder reads the half-hourly panel that `e00103_build.py` writes, so the
joins to ERA5 temperature, CAMS irradiance, and the switching masks happen once.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.experiment_store import reserve_experiment  # noqa: F401  (id reserved in the plan)
from studies.sources import ML_EXPERIMENTS_DATA_DIR, STUDIES_DATA_DIR

EXPERIMENT_SLUG: Final[str] = "00103-differentiable-physics-1-data-look"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_SLUG
"""The experiment's private folder: the panel, the handed-on files, and the printed reports."""
FIGURE_DIR: Final[Path] = Path(__file__).parent / "draft"
"""Where the figures are written while the docs freeze is in force."""

DEMAND_INPUTS_DIR: Final[Path] = STUDIES_DATA_DIR / "per_study/stage1_pilot/demand/inputs"
SERIES_PATH: Final[Path] = DEMAND_INPUTS_DIR / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_INPUTS_DIR / "era5_features.parquet"
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet"
)
LABELLED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate/real/labelled_set.parquet"
)
NGED_DIR: Final[Path] = STUDIES_DATA_DIR.parent / "NGED"
CLEANED_POWER_PATH: Final[Path] = NGED_DIR / "cleaned_power_time_series.delta"
RAW_POWER_PATH: Final[Path] = NGED_DIR / "power_time_series.delta"
METADATA_PATH: Final[Path] = NGED_DIR / "metadata.parquet"
PANEL_PATH: Final[Path] = WORK_DIR / "panel.parquet"
SERIES_TABLE_PATH: Final[Path] = WORK_DIR / "series_table.parquet"

PERIOD_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
PERIOD_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Exclusive end: nothing on or after 2026-07-01 enters a statistic."""
PANEL_END: Final[datetime] = datetime(2026, 9, 11, tzinfo=UTC)
"""Exclusive end of the panel: CAMS runs to 2026-09-10, used only by the descriptive timestamp and
early-calibration checks. Every scored or counted statistic stops at `PERIOD_END`."""
STAMP_FIX_UTC: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)
"""NGED's late stamps before this instant are corrected at ingest; the timestamp figure splits here."""
LOCAL_TIME_ZONE: Final[str] = "Europe/London"

MW_PRIMARIES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16", "D18", "D15")
"""Eight MW Primaries; D15 (which carries an embedded generator) is last."""
MW_LONG_HISTORY: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
MVA_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
PRIMARIES: Final[tuple[str, ...]] = (*MW_PRIMARIES, *MVA_PRIMARIES)
GRID_POINTS: Final[tuple[str, ...]] = ("D1", "D6", "D13", "D17")
GENERATOR_SERIES: Final[dict[str, str]] = {"D7": "G-a", "D20": "G-b"}
"""The two 'battery or biofuel' series and their anonymised plot labels."""

HOURS_BANDS: Final[dict[str, tuple[int, int]]] = {
    "night": (0, 5),
    "evening": (17, 21),
    "midday": (11, 15),
}
"""Local-hour ranges [start, end) of the night, evening, and midday bands."""


def plot_label(*, site: str) -> str:
    """The label a figure may show: D-labels for substations, G-a and G-b for the generators."""
    return GENERATOR_SERIES.get(site, site)


def load_series() -> pl.DataFrame:
    """The 22 series: `site`, `time_series_id`, `group`, `units`, `effective_capacity_mw`."""
    metadata = pl.read_parquet(METADATA_PATH).select("time_series_id", "units")
    return (
        pl.read_parquet(SERIES_PATH)
        .join(metadata, on="time_series_id", how="left")
        .select("site", "time_series_id", "group", "units", "latitude", "longitude",
                "effective_capacity_mw")
        .sort("site")
    )


def hand_filtered_events() -> pl.DataFrame:
    """The 66 hand-filtered real transfers, with every series each event touches.

    Filtered exactly as experiment 101 does: `~is_sentinel`, `kind == "real_operation"`, and
    `visibility` in `conserved` or `one_sided`. Returns `onset_utc`, `end_utc`, and the list
    `series_ids` of the donor, both sides, and every receiver.
    """
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    rows = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        touched = sorted(
            {
                int(s)
                for s in (row["donor"], row["side_a"], row["side_b"], *receivers)
                if s is not None
            }
        )
        rows.append(
            {
                "card_id": row["card_id"],
                "onset_utc": row["onset_utc"],
                "end_utc": row["end_utc"],
                "series_ids": touched,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "card_id": pl.String,
            "onset_utc": pl.Datetime("us", "UTC"),
            "end_utc": pl.Datetime("us", "UTC"),
            "series_ids": pl.List(pl.Int64),
        },
    )
