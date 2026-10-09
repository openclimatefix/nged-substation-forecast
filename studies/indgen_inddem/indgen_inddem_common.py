"""Paths, constants, and the boundary-to-zone table shared by the INDDEM and INDGEN study's scripts.

The study looks at Elexon's indicated demand (INDDEM) and indicated generation (INDGEN) series,
which sum the final Physical Notifications (PNs) of every Balancing Mechanism Unit (BMU), beside
Elexon's settled energy taken from the transmission system by each Grid Supply Point (GSP) group
(AGV). All of the data is public, so the scripts name the BMUs and the GSP groups.
"""

from datetime import date
from pathlib import Path
from typing import Final, Literal

import numpy as np
from studies.sources import MARKET_DOWNLOADS_DIR, study_dir_for

STUDY_START: Final[date] = date(2025, 9, 1)
"""First settlement date of the study window."""
STUDY_END: Final[date] = date(2026, 9, 30)
"""Last settlement date of the study window."""
AGV_END: Final[date] = date(2026, 9, 20)
"""The last settlement date with an `SF` settlement run when the data was downloaded."""

STUDY_DIR: Final[Path] = study_dir_for(study="indgen_inddem")
"""Where the study's tables, `report.md`, and figures are written."""
ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
"""Where the page's figures are written, as SVG."""
INDDEM_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "elexon_inddem" / "elexon_inddem.parquet"
INDGEN_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "elexon_indgen" / "elexon_indgen.parquet"
AGV_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "elexon_agv" / "elexon_agv.parquet"
PV_LIVE_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "pv_live" / "pv_live.parquet"
INDO_PATH: Final[Path] = (
    MARKET_DOWNLOADS_DIR / "elexon_demand_outturn" / "elexon_demand_outturn.parquet"
)
PN_SAMPLE_PATH: Final[Path] = MARKET_DOWNLOADS_DIR / "elexon_pn_sample" / "elexon_pn_sample.parquet"
BMU_REFERENCE_PATH: Final[Path] = (
    MARKET_DOWNLOADS_DIR / "elexon_bmu_reference" / ("bmu_reference.parquet")
)

DatasetType = Literal["inddem", "indgen"]
ViewType = Literal["latest", "first_of_day"]

BOUNDARIES: Final[tuple[str, ...]] = ("N", *(f"B{number}" for number in range(1, 18)))
"""The national total and the 17 boundaries, in the order the matrix below uses."""
ZONES: Final[tuple[str, ...]] = tuple(f"Z{number}" for number in range(1, 18))

BOUNDARY_ZONE_NUMBERS: Final[dict[str, tuple[int, ...]]] = {
    "N": tuple(range(1, 18)),
    "B1": (1,),
    "B2": (1, 2),
    "B3": (3,),
    "B4": (1, 2, 3, 4),
    "B5": (1, 2, 3, 4, 5),
    "B6": (1, 2, 3, 4, 5, 6),
    "B7": (1, 2, 3, 4, 5, 6, 7),
    "B8": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "B9": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    "B10": (16, 17),
    "B11": (1, 2, 3, 4, 5, 6, 7, 8),
    "B12": (13, 16, 17),
    "B13": (17,),
    "B14": (14,),
    "B15": (15,),
    "B16": (1, 2, 3, 4, 5, 6, 7, 8, 10),
    "B17": (11,),
}
"""The study zones that make up each boundary, transcribed from Elexon CVA Change Circular 235
(2012), Appendix 1. `N` is the national total, which holds every zone."""

BOUNDARY_NAMES: Final[dict[str, str]] = {
    "B1": "B1",
    "B2": "B2",
    "B3": "B3 (Sloy)",
    "B4": "B4 (SHETL-SPT)",
    "B5": "B5",
    "B6": "B6 (SPT-NGET)",
    "B7": "B7 (Upper North-North)",
    "B8": "B8 (North to Midlands)",
    "B9": "B9 (Midlands to South)",
    "B10": "B10 (South Coast)",
    "B11": "B11 (North East and Yorkshire)",
    "B12": "B12 (South and South West)",
    "B13": "B13 (South West)",
    "B14": "B14 (London)",
    "B15": "B15 (Thames Estuary)",
    "B16": "B16 (North East, Trent and Yorkshire)",
    "B17": "B17 (West Midlands)",
}
"""The names the circular gives each boundary."""

GSP_GROUPS: Final[tuple[str, ...]] = tuple(f"_{letter}" for letter in "ABCDEFGHJKLMNP")
NGED_GROUPS: Final[dict[str, str]] = {
    "_B": "East Midlands",
    "_E": "West Midlands",
    "_K": "South Wales",
    "_L": "South West",
}
"""NGED's four licence areas, as the GSP group letter and the name Elexon gives the group."""

SIGN_TOLERANCE_MW: Final[float] = 1.0
"""INDDEM and INDGEN values are whole megawatts, so a derived zone may stray this far past zero."""


def membership_matrix() -> np.ndarray:
    """Return the 18 by 17 matrix whose entry is 1 where the boundary holds the zone.

    Returns:
        The matrix, with a row for each of `BOUNDARIES` and a column for each of `ZONES`.

    Raises:
        ValueError: If the matrix does not have full column rank, so the zones could not be
            recovered from the boundaries.
    """
    matrix = np.zeros((len(BOUNDARIES), len(ZONES)))
    for row, boundary in enumerate(BOUNDARIES):
        for number in BOUNDARY_ZONE_NUMBERS[boundary]:
            matrix[row, number - 1] = 1.0
    if np.linalg.matrix_rank(matrix) != len(ZONES):
        raise ValueError("the boundary-to-zone matrix has less than full column rank")
    return matrix
