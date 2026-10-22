# ruff: noqa: E501
"""Each valid time's 24-hour-averaged temperature and its decile among a substation's training rows.

The temperature is the feature cache's analysis-frame 24-hour trailing mean at the valid time, one
series per substation, so it does not depend on which run forecast the row. The decile cut points
come from valid times before the tuning period (2024-07-01), the model's training side.
"""

from datetime import UTC, datetime
from typing import Final

import e00216_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_features import FeatureCache, naive_utc

WINDOW_POSITION: Final[int] = SMOOTHING_HOURS.index(24)
TRAINING_END: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)


def temperature_frame(*, cache: FeatureCache, site: str) -> pl.DataFrame:
    """Return `valid_time`, `temp24_c`, and `decile` (1 to 10) for every grid time of a substation."""
    temperature = cache.sites[site].smoothed_temperature_c[:, WINDOW_POSITION]
    valid = pl.Series(cache.end_times).dt.replace_time_zone("UTC")
    before = cache.end_times < np.datetime64(naive_utc(moment=TRAINING_END), "us")
    reference = temperature[before & np.isfinite(temperature)]
    cuts = np.percentile(reference, np.arange(10, 100, 10))
    decile = np.searchsorted(cuts, temperature, side="right") + 1
    return pl.DataFrame(
        {"valid_time": valid, "temp24_c": temperature, "decile": decile.astype(np.int8)}
    ).filter(pl.col("temp24_c").is_finite())


def all_frames(*, cache: FeatureCache | None = None) -> pl.DataFrame:
    """Return the temperature frames of the six substations, with a `site` column."""
    cache = cache if cache is not None else common.load_cache()
    return pl.concat(
        [
            temperature_frame(cache=cache, site=s).with_columns(site=pl.lit(s))
            for s in common.SCORED_SITES
        ]
    )
