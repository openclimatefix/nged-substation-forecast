"""Loading the observed power and gridded NWP that feature engineering runs on.

Shared by the research assets in ``cv_assets`` and the production asset in ``production_assets``,
so it belongs to neither. The caller supplies the ``TimeSeriesMetadata`` that goes alongside,
because the two layers get it from different places.
"""

from datetime import datetime, timedelta
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import Nwp

MAX_NWP_LEAD: Final[timedelta] = timedelta(days=16)
"""Upper bound on an NWP run's forecast horizon, used to prune the ``init_time``-partitioned scan.

A run initialised at ``init_time = T`` only produces ``valid_time``s in ``[T, T + horizon]``, so a
run can cover a ``valid_time`` window ``[start, end]`` only if ``init_time`` lies in
``[start - horizon, end]``. ECMWF ENS forecasts to 15 days; 16 gives a safe margin. See
``load_engineering_inputs``.
"""


def load_engineering_inputs(
    settings: Settings,
    time_series_ids: list[int],
    metadata: pt.DataFrame[TimeSeriesMetadata],
    window_start: datetime,
    window_end: datetime,
    ensemble_members: list[int] | None = None,
    init_time_start: datetime | None = None,
    init_time_end: datetime | None = None,
) -> tuple[pt.LazyFrame[PowerTimeSeries], pt.LazyFrame[Nwp]]:
    """Load observed power and NWP for a window and time-series population.

    Called by ``trained_cv_model`` (training window + eligible population), ``cv_power_forecasts``
    (validation window + trained population) and ``live_forecasts`` (the live window + the promoted
    model's population). Both returned frames are filtered to the inclusive
    ``[window_start, window_end]`` window and to ``time_series_ids``.

    Every filter below is applied directly to the ``Nwp.scan_delta`` scan, so only the surviving
    rows are ever decoded — the difference between a few GB and an OOM on the multi-tens-of-GB NWP
    Delta. See [Bounding feature-engineering
    memory](https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#bounding-feature-engineering-memory-prune-the-inputs-not-the-output)
    for why each predicate below prunes, and by how much. The returned NWP frame is filtered to:

    - ``init_time`` in ``[init_time_start, init_time_end]`` (default
      ``[window_start - MAX_NWP_LEAD, window_end]``): the runs that can cover the window.
    - ``ensemble_member`` in ``ensemble_members``, if given; every member otherwise.
    - ``h3_index`` restricted to the cells the requested series sit in (one cell can cover several
      series; the feature engineer's spatial join later replicates each cell's weather across
      them).

    Args:
        settings: Application settings (data paths, credentials).
        time_series_ids: IDs to include; power is filtered to this population.
        metadata: The metadata for those series, whose ``h3_res_5`` decides which NWP cells are
            scanned. R&D passes the roster; production passes the promoted model's frozen copy.
        window_start: Inclusive start of the time window for power observations and NWP
            ``valid_time``.
        window_end: Inclusive end of the time window for power observations and NWP
            ``valid_time``.
        ensemble_members: If provided, NWP is filtered to these ``ensemble_member`` indices. If
            ``None`` (the default), every ensemble member is carried through. Training restricts
            to the control member (``[0]``), which stops every training row fanning out across all
            ~51 members against one power target; prediction passes ``None`` because the
            probabilistic leaderboard metrics need the full ensemble.
        init_time_start: Optional explicit lower ``init_time`` partition bound. Defaults to
            ``window_start - MAX_NWP_LEAD``, the earliest run that can cover the window.
        init_time_end: Optional explicit upper ``init_time`` partition bound. Defaults to
            ``window_end``. Together with ``init_time_start`` this lets ``cv_power_forecasts`` pass
            a narrower sub-range and process the validation window in ``init_time`` chunks, so one
            chunk's full-ensemble forecast frame stays in RAM while the rest streams from the
            pruned scan.

    Returns:
        ``(power_time_series, nwp)`` — a lazy power frame and a lazy NWP frame, both filtered to
        ``time_series_ids`` and the requested window.
    """
    if init_time_start is None:
        init_time_start = window_start - MAX_NWP_LEAD
    if init_time_end is None:
        init_time_end = window_end
    storage_options = settings.storage_options
    power_lf = pl.scan_delta(
        settings.power_time_series_data_path, storage_options=typeddict_to_dict(storage_options)
    ).filter(
        pl.col("time_series_id").is_in(time_series_ids),
        pl.col("time") >= window_start,
        pl.col("time") <= window_end,
    )
    power_ts = pt.LazyFrame.from_existing(power_lf).set_model(PowerTimeSeries)

    # The H3 cells the requested series sit in (many series may share one cell).
    cells = metadata["h3_res_5"].unique().to_list()

    nwp_scan = Nwp.scan_delta(settings.nwp_data_path, storage_options=storage_options).filter(
        # init_time is the partition key — this prunes whole partitions, not just row groups.
        pl.col("init_time") >= init_time_start,
        pl.col("init_time") <= init_time_end,
        pl.col("valid_time") >= window_start,
        pl.col("valid_time") <= window_end,
        pl.col("h3_index").is_in(cells),
    )
    if ensemble_members is not None:
        nwp_scan = nwp_scan.filter(pl.col("ensemble_member").is_in(ensemble_members))
    # ``.filter`` is *typed* as a plain ``pl.LazyFrame`` even though the model survives at runtime,
    # so re-wrap to satisfy the return annotation. (Zero-copy.)
    nwp_lf = pt.LazyFrame.from_existing(nwp_scan).set_model(Nwp)

    return power_ts, nwp_lf
