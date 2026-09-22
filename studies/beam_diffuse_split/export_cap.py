"""Put NGED's active-network-management export cap on the experiment's hourly grid.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

`anm_setpoints.py` turns NGED's raw setpoint export into a half-hourly export cap, one file per
generator. This module joins that cap onto the modelling dataset and marks the hours in which the
network operator had moved the cap off the generator's connection limit.

**The two uses pull in opposite directions, which is why both a flag and a cap are joined on.** An
hour whose cap moved is an hour no irradiance product could have predicted, so it is dropped from
the training folds: a model that trains on it learns to imitate network instructions from the sky.
At scoring time the same hour is kept and the prediction is held down to the cap, because a
forecaster that knew the cap would never have predicted above it.

**The clamp uses the cap that was actually in force, which a day-ahead forecast does not have.** It
is therefore an upper bound on what the cap could buy, not a forecast, and the contrasts in the
write-up are computed with the constrained hours dropped rather than clamped.

**The setpoint signal exists before the scheme is enforcing anything, and reads zero while it
waits.** On the one generator that has an export, the feedback value sits at zero for 85% of the
generator's first six months and never once reaches the connection limit. Across all 431 bright
hours in that span the cap forbids export outright, and the generator exported through 423 of them
anyway, at a median of 44% of its capacity. A cap that permits 44% output is not a cap. Taking
those readings at face value would mask most of the generator's first six months out of training
and clamp its predictions to zero on days it ran normally, which is the worst error available
here.

**So the cap is honoured only from the first moment it reaches the connection limit.** A live
scheme on an unconstrained generator rests at the connection limit most of the time, so the first
reading at that limit is the earliest point the feedback value can be shown to mean what it says.
It is a marker taken from the cap alone, with no reference to metered output, so it cannot be
tuned to flatter a result.

**One of the six generators has a setpoint export, and NGED has confirmed that generator is the
only one in the trial area connected under active network management.** The five sites with no
export are therefore uncapped rather than unrecorded, so a missing cap is a real absence of
curtailment and not a gap in what we were sent. The only rows left without an honoured cap are the
ones before the scheme went live. `constrained` is False wherever no honoured record exists, and
`cap_mw` is null there, which is what stops the clamp reaching a row it has no cap for.
"""

import logging
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from build_dataset import _pv_sites
from sources import ANM_DATA_DIR

_LOG: Final[logging.Logger] = logging.getLogger("export_cap")

CAP_FILE_PREFIX: Final[str] = "export_cap_"
"""The filenames `anm_setpoints.py` writes, each ending in the generator's `time_series_id`."""

CAP_TOLERANCE_MW: Final[float] = 0.001
"""How far below the connection limit a cap must sit before its hour counts as constrained.

The cap is exported to three decimal places, so anything smaller is a rounding artefact rather than
an instruction to turn down.
"""


def _live_only(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Drop the readings that precede the scheme going live.

    Args:
        half_hourly: One generator's export cap, as `anm_setpoints.py` wrote it.

    Returns:
        The same frame from the first half-hour whose cap reaches the connection limit onwards.
    """
    limit = float(np.max(half_hourly["cap_mw"].to_numpy()))
    live = half_hourly.filter(pl.col("cap_mw") >= limit - CAP_TOLERANCE_MW)
    if live.is_empty():
        _LOG.warning("no cap ever reaches the connection limit: discarding the whole record")
        return half_hourly.clear()
    went_live = live["time"][0]
    dropped = half_hourly.filter(pl.col("time") < went_live).height
    _LOG.info("cap honoured from %s, discarding %d earlier half-hours", went_live, dropped)
    return half_hourly.filter(pl.col("time") >= went_live)


def _hourly_cap(*, path: Path) -> pl.DataFrame:
    """Aggregate one generator's half-hourly export cap onto the hourly, period-ending grid.

    The connection limit is taken as the largest cap the generator's own record ever reaches, which
    is what makes `constrained` mean "moved off the limit" rather than "below some fixed number".

    Args:
        path: An `export_cap_<time_series_id>.parquet` written by `anm_setpoints.py`.

    Returns:
        One row per hour ending at `time`, with the mean cap over the hour and whether the operator
        had moved the cap off the connection limit at any point inside it.
    """
    half_hourly = _live_only(half_hourly=pl.read_parquet(path))
    limit = float(np.max(half_hourly["cap_mw"].to_numpy()))
    return (
        half_hourly.group_by_dynamic("time", every="1h", label="right", closed="right")
        .agg(cap_mw=pl.col("cap_mw").mean(), lowest_cap_mw=pl.col("lowest_cap_mw").min())
        .with_columns(constrained=pl.col("lowest_cap_mw") < limit - CAP_TOLERANCE_MW)
        .drop("lowest_cap_mw")
        .sort("time")
    )


def with_export_cap(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the export cap and the constrained flag to every row of the experiment dataset.

    The anonymous site labels come from `build_dataset._pv_sites`, so the mapping from NGED's
    `time_series_id` to a label stays in the one place that owns it and no export filename has to
    be matched to a label by hand.

    Args:
        dataset: Any frame carrying `site` and `time`.

    Returns:
        `dataset` with `cap_mw` — null where no setpoint record covers the row — and `constrained`,
        which is False both where the cap never moved and where no record exists.
    """
    labels: dict[int, str] = dict(_pv_sites().select("time_series_id", "site").iter_rows())
    frames: list[pl.DataFrame] = []
    for path in sorted(ANM_DATA_DIR.glob(f"{CAP_FILE_PREFIX}*.parquet")):
        time_series_id = int(path.stem.removeprefix(CAP_FILE_PREFIX))
        site = labels.get(time_series_id)
        if site is None:
            _LOG.warning("%s belongs to no site in this experiment, skipping", path.name)
            continue
        frames.append(_hourly_cap(path=path).with_columns(site=pl.lit(site)))

    if not frames:
        _LOG.warning("no export caps under %s: every row will read as unconstrained", ANM_DATA_DIR)
        return dataset.with_columns(
            cap_mw=pl.lit(None, dtype=pl.Float64), constrained=pl.lit(value=False)
        )

    joined = dataset.join(pl.concat(frames), on=["site", "time"], how="left").with_columns(
        constrained=pl.col("constrained").fill_null(value=False)
    )
    _LOG.info(
        "export cap covers %d of %d rows, of which %d are constrained",
        int(joined["cap_mw"].is_not_null().sum()),
        joined.height,
        int(joined["constrained"].sum()),
    )
    return joined


def clamp_to_cap(*, prediction: np.ndarray, cap_mw: pl.Series) -> np.ndarray:
    """Hold a prediction at or below the export cap that was in force.

    Args:
        prediction: Either one prediction per row, or one row per prediction and one column per
            quantile level.
        cap_mw: The cap for each row, null where no setpoint record covers it.

    Returns:
        `prediction`, with every element that exceeded its row's cap replaced by that cap. Rows
        with no cap are returned unchanged.
    """
    ceiling = cap_mw.fill_null(np.inf).to_numpy()
    if prediction.ndim > 1:
        ceiling = ceiling[:, np.newaxis]
    return np.minimum(prediction, ceiling)
