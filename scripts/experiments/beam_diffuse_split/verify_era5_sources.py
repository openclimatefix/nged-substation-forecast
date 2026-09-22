"""Check that Open-Meteo's ERA5 mirror carries the same fields as the Copernicus download.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**The experiment is void if the mirror's beam field turns out to be a separation model's estimate
rather than ERA5's own `fdir`.** Arm C would then be a copy of arm B, and a null result would be
guaranteed by construction rather than measured. This script is what settles it: it joins the two
downloads on (time, latitude, longitude) and reports how far apart they are, hour by hour, over
every hour both cover.

A mirror that agrees to Open-Meteo's 1 W m⁻² rounding is serving the same field. One that ran a
separation model would disagree by tens of W m⁻², which is the scale at which Erbs misses ERA5's
beam on this data.

Run it with `uv run --no-project --with polars --with xarray --with netcdf4 --with numpy python
scripts/experiments/beam_diffuse_split/verify_era5_sources.py`.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_dataset import REPO_DATA_DIR, _read_era5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("verify_era5_sources")

OUTPUT_PATH: Final[Path] = REPO_DATA_DIR / "ERA5" / "era5_source_agreement.json"

COMPARED_FIELDS: Final[tuple[str, ...]] = ("ghi_w_m2", "bhi_w_m2", "temp_c")

ROUNDING_TOLERANCE_W_M2: Final[float] = 1.0
"""Open-Meteo rounds irradiance to whole W m⁻².

That rounding is the floor on any disagreement between the two sources, so a difference at this
scale says nothing and only a much larger one would.
"""


def _scalar(value: object) -> float:
    """Narrow a Polars aggregate to a plain float.

    `Series.max()` and friends are typed as a union covering every dtype a Series could hold, so a
    checker cannot know these columns are floats. `build_dataset` drops nulls before either source
    reaches here, so there is nothing to guard against at runtime.
    """
    return float(value)  # ty: ignore[invalid-argument-type]


def _agreement(*, joined: pl.DataFrame, field: str) -> dict[str, float]:
    """Summarise how far apart the two sources are for one field.

    Args:
        joined: The two sources joined on (time, latitude, longitude).
        field: The column to compare.

    Returns:
        The worst and typical disagreement, and how often it exceeds Open-Meteo's rounding.
    """
    difference = (joined[f"{field}_cds"] - joined[f"{field}_open_meteo"]).abs()
    return {
        "max_absolute_difference": _scalar(difference.max()),
        "mean_absolute_difference": _scalar(difference.mean()),
        "99th_percentile_absolute_difference": _scalar(difference.quantile(0.99)),
        "fraction_beyond_rounding": _scalar(difference.gt(ROUNDING_TOLERANCE_W_M2).mean()),
        "cds_standard_deviation": _scalar(joined[f"{field}_cds"].std()),
    }


def main() -> int:
    """Join the two downloads over their shared hours and report the agreement."""
    cds = _read_era5(source="cds")
    open_meteo = _read_era5(source="open-meteo")
    _LOG.info(
        "Copernicus: %d rows to %s; Open-Meteo: %d rows to %s",
        cds.height,
        cds["time"].max(),
        open_meteo.height,
        open_meteo["time"].max(),
    )

    keys = ["time", "latitude", "longitude"]
    joined = cds.rename({field: f"{field}_cds" for field in COMPARED_FIELDS}).join(
        open_meteo.rename({field: f"{field}_open_meteo" for field in COMPARED_FIELDS}),
        on=keys,
        how="inner",
    )
    report = {
        "shared_hours": joined.height,
        "copernicus_hours": cds.height,
        "open_meteo_hours": open_meteo.height,
        "fields": {field: _agreement(joined=joined, field=field) for field in COMPARED_FIELDS},
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    _LOG.info("%s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
