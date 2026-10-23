"""The 13 metered generators under D13 and D17: the child table and their readings.

Generators are labelled G1 to G13 in time-series-id order. The labels, the kinds, and the history
lengths are the only things any page may print about them. The mapping from label to time series
stays in the private work folder.
"""

from typing import Final

import e00223_common as common
import numpy as np
import polars as pl
from nged_data.storage import scan_cleaned_power
from studies.dp_features import FeatureCache
from studies.sources import REPO_DATA_DIR

GENERATORS: Final[dict[str, dict]] = {
    # label: time_series_id, bsp, kind, unit, voltage class
    "G1": {"id": 4, "bsp": "D13", "kind": "wind", "unit": "MW", "voltage": "EHV"},
    "G2": {"id": 5, "bsp": "D13", "kind": "wind", "unit": "MVA", "voltage": "EHV"},
    "G3": {"id": 6, "bsp": "D17", "kind": "wind", "unit": "MVA", "voltage": "EHV"},
    "G4": {"id": 19, "bsp": "D17", "kind": "biomass", "unit": "MW", "voltage": "EHV"},
    "G5": {"id": 20, "bsp": "D17", "kind": "battery", "unit": "MW", "voltage": "EHV"},
    "G6": {"id": 21, "bsp": "D17", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G7": {"id": 22, "bsp": "D17", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G8": {"id": 23, "bsp": "D13", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G9": {"id": 29, "bsp": "D17", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G10": {"id": 30, "bsp": "D13", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G11": {"id": 31, "bsp": "D13", "kind": "solar", "unit": "MW", "voltage": "EHV"},
    "G12": {"id": 32, "bsp": "D17", "kind": "solar", "unit": "MW", "voltage": "HV"},
    "G13": {"id": 33, "bsp": "D17", "kind": "other", "unit": "MW", "voltage": "HV"},
}
"""Time series id, bulk supply point, kind, reading unit, and voltage class of each generator."""


def load_generators(*, cache: FeatureCache) -> dict[str, np.ndarray]:
    """Return each generator's output in MW on the cache grid, `NaN` where there is no reading.

    A reading in MVA is multiplied by the power factor to give MW. Generation is positive for
    every kind except the battery, whose sign is as stored (positive when it discharges).
    """
    ids = [spec["id"] for spec in GENERATORS.values()]
    raw = (
        scan_cleaned_power(str(REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"))
        .filter(pl.col("time_series_id").is_in(ids))
        .select("time_series_id", "time", "power")
        .collect()
    )
    grid = pl.DataFrame({"time": cache.end_times}).with_row_index("row")
    out: dict[str, np.ndarray] = {}
    for label, spec in GENERATORS.items():
        series = (
            raw.filter(pl.col("time_series_id") == spec["id"])
            .with_columns(pl.col("time").dt.replace_time_zone(None))
            .select("time", "power")
            .unique(subset="time")
        )
        joined = grid.join(series, on="time", how="left").sort("row")
        values = joined["power"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if spec["unit"] == "MVA":
            values = values * common.POWER_FACTOR
        out[label] = values
    return out


def child_table(*, cache: FeatureCache, series: dict[str, np.ndarray]) -> list[dict]:
    """Return the child table: label, bulk supply point, kind, unit, voltage class, and history."""
    stamps = cache.end_times
    rows = []
    for label, spec in GENERATORS.items():
        ok = np.isfinite(series[label])
        first = stamps[ok][0] if ok.any() else None
        last = stamps[ok][-1] if ok.any() else None
        rows.append(
            {
                "label": label,
                "bsp": spec["bsp"],
                "kind": spec["kind"],
                "unit": spec["unit"],
                "voltage": spec["voltage"],
                "readings": int(ok.sum()),
                "years": float(ok.sum() / (48 * 365.25)),
                "first": str(first),
                "last": str(last),
                "mean_mw": float(np.nanmean(series[label])) if ok.any() else None,
                "max_mw": float(np.nanmax(series[label])) if ok.any() else None,
                "min_mw": float(np.nanmin(series[label])) if ok.any() else None,
            }
        )
    return rows
