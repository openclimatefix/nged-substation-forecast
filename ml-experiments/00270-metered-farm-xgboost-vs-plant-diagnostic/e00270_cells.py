"""Nearest store cell to a farm, with the rule of experiment 224's members script."""

import h3
import numpy as np
import polars as pl
from studies.ens_members import MIXTURE_LEVELS  # noqa: F401


def nearest_cell(*, latitude: float, longitude: float, uri: str) -> int:
    """Return the store's H3 cell whose centre is nearest the farm."""
    cells = (
        pl.scan_delta(uri)
        .filter(pl.col("init_time") == pl.lit("2024-04-01").str.to_datetime(time_zone="UTC"))
        .select("h3_index")
        .unique()
        .collect()["h3_index"]
        .to_list()
    )
    centres = np.array([h3.cell_to_latlng(h3.int_to_str(c)) for c in cells])
    distance = (centres[:, 0] - latitude) ** 2 + (
        (centres[:, 1] - longitude) * np.cos(np.radians(latitude))
    ) ** 2
    return int(cells[int(np.argmin(distance))])
