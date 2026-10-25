"""Stage 4 data of experiment 00347: the case frames and the plants' ensemble summaries.

Loads the hindcast and scored case frames in experiment 00339's order and the summaries that
`e00347_s4_inputs.py` stored, farms W1 to W3 stacked in order. Used by `e00347_s4_run.py` (head
training) and `e00347_s4_score.py` (scoring). Run with `PYTHONPATH` set to the folders of
experiments 00347, 00230, 00303, 00310, 00326, and 00336.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00303_common as c303  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # ty: ignore[unresolved-import]
import e00303_head as head303  # ty: ignore[unresolved-import]
import e00303_score as score303  # ty: ignore[unresolved-import]
import e00347_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

S4_DIR: Final[Path] = common.DATA_DIR / "s4"
PARTS_DIR: Final[Path] = S4_DIR / "parts"
"""One file per fitted head: `<name>.npy`, written atomically."""
FARMS: Final[tuple[str, ...]] = common.FARMS
MAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
REPLICATE_CUTOFF: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
REPLICATE_START: Final[datetime] = datetime(2025, 1, 1, tzinfo=UTC)
E339_FINAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00339-wind-research-default-1-adapter-plus-mixture-head" / "final"
)
E324_FINAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00324-wind-farm-solar-cross-breeds-long-run" / "final"
)
E326_FINAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00326-wind-research-default-1-change-arms" / "work" / "final"
)
E336_FINAL: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00336-wind-research-default-1-below-cut-in-zeroing"
    / "work"
    / "final"
)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = ("median", "mean", "sd", "p10", "p90", "saturation")


def hindcast_frame() -> pl.DataFrame:
    """Return the hindcast cases with an observed power and `masked`, in experiment 339's order."""
    sites = c303.site_table()
    hind_all = ens_final.hindcast_cases(sites=sites)
    usable = hind_all["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    return head303.with_summaries(cases=hind_all.filter(pl.Series(usable)), set_name="hindcast")


def scored_frame() -> pl.DataFrame:
    """Return the scored cases (53 Tuesday runs), in the order of the stored arms."""
    return score303.ordered_cases()


def summaries(*, arm_plant: str, set_name: str) -> np.ndarray:
    """Return the plant's ensemble summaries (cases, 6), farms stacked in order W1 to W3."""
    return np.concatenate(
        [np.load(S4_DIR / f"summary_{arm_plant}_{set_name}_{farm}.npy") for farm in FARMS]
    )


def mixtures(*, arm_plant: str, set_name: str) -> np.ndarray:
    """Return the plant's member-mixture quantiles (cases, 11), farms stacked in order."""
    return np.concatenate(
        [np.load(S4_DIR / f"mix_{arm_plant}_{set_name}_{farm}.npy") for farm in FARMS]
    )


def era5_quantiles(*, arm_plant: str) -> np.ndarray:
    """Return the ERA5-driven plant's scored quantiles (cases, 11), farms stacked in order."""
    return np.concatenate(
        [np.load(S4_DIR / f"era5_{arm_plant}_scored_{farm}.npy") for farm in FARMS]
    )
