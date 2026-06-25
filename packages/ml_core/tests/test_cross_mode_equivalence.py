"""Cross-mode feature equivalence — the enforceable no-skew guarantee (§5.7).

The "zero training-serving skew" requirement is met by both CV/backtest and production calling
the *same* ``engineer_features()``, differing only in operating mode:

- **Bulk / backtest** (``power_fcst_init_time=None``): NWP-centric, vectorised over the whole
  window, one forecast per NWP run, ``power_fcst_init_time = nwp_init_time + delay`` per row.
- **Single-run / production** (explicit ``power_fcst_init_time``): one NWP run, stamped t0.

This test takes a fixture spanning several **daily** NWP runs (real ECMWF ENS is issued once
per day at 00 UTC), runs bulk mode, then *replays* each NWP run in single-run mode with
``power_fcst_init_time = nwp_init_time + delay`` and asserts the rows match exactly on the
primary key and on every requested feature column. If a future change diverges the two modes,
this fails.

Scope note: this test exercises the **weather, time, and power-lag** features. Weather/time
features depend on the bulk-vs-single-run NWP join; power lags are included because both modes
now source them from the same dense observed-power series (Phase 1.5 / Option B), so they are
identical too. The fixture's power series extends back before each NWP window (the pre-window
history) so that an in-window power lag resolves to a genuine observed value rather than being
nullified or reaching off the edge of the data.
"""

from datetime import datetime, timedelta

import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory
from ml_core.features import engineer_features
from polars.testing import assert_frame_equal

_DELAY_HOURS = 6
_NWP_RUNS = [
    datetime(2024, 1, 1, 0, 0),
    datetime(2024, 1, 2, 0, 0),
    datetime(2024, 1, 3, 0, 0),
]
_MEMBERS = [0, 1]
_FEATURES = {
    "temperature_2m",
    "wind_speed_10m",
    "windchill",
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "power_lag_3h",
}
_COMPARE_COLS = [
    "time_series_id",
    "ensemble_member",
    "nwp_init_time",
    "power_fcst_init_time",
    "valid_time",
    "power",
    "nwp_lead_time_hours",
    "temperature_2m",
    "wind_speed_10m",
    "windchill",
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "power_lag_3h",
]
_SORT_COLS = ["time_series_id", "power_fcst_init_time", "valid_time", "ensemble_member"]


def _run_valid_times(run_init: datetime) -> list[datetime]:
    """A non-overlapping 06:00–12:00 half-hourly window for each daily run.

    Non-overlapping windows mean each (time_series_id, valid_time) appears in exactly one run,
    which keeps the bulk and single-run row sets directly comparable.
    """
    start = run_init + timedelta(hours=_DELAY_HOURS)
    return [start + timedelta(minutes=30 * i) for i in range(13)]  # 06:00 .. 12:00 inclusive


def _power_observation_times(run_init: datetime) -> list[datetime]:
    """Half-hourly power observations from each run's init through the end of its NWP window.

    Crucially this includes the pre-window history (init .. init + delay) so that a power lag on
    an in-window row reaches back to a genuine observed value instead of being nullified.
    """
    return [run_init + timedelta(minutes=30 * i) for i in range(25)]  # 00:00 .. 12:00 inclusive


def _build_fixtures() -> tuple[
    pt.LazyFrame[PowerTimeSeries], pt.DataFrame[TimeSeriesMetadata], pt.LazyFrame[NwpInMemory]
]:
    nwp_rows = []
    for run in _NWP_RUNS:
        for member in _MEMBERS:
            for vt in _run_valid_times(run):
                nwp_rows.append(
                    {
                        "time_series_id": "ts1",
                        "valid_time": vt,
                        "ensemble_member": member,
                        "init_time": run,
                        "temperature_2m": 10.0 + vt.hour + member * 0.5,
                        "wind_speed_10m": 3.0 + (vt.minute / 30.0) + member,
                    }
                )
    nwp_df = pl.DataFrame(nwp_rows)

    # Power observations span each run's full window plus the pre-NWP-window history, so a
    # power lag on an in-window row resolves to a genuine observed value.
    power_times = sorted({vt for run in _NWP_RUNS for vt in _power_observation_times(run)})
    power_df = pl.DataFrame(
        {
            "time_series_id": ["ts1"] * len(power_times),
            "time": power_times,
            "power": [
                float(100 + vt.day * 10 + vt.hour * 2 + vt.minute // 30) for vt in power_times
            ],
        }
    )

    metadata_df = pl.DataFrame({"time_series_id": ["ts1"], "time_series_type": ["substation"]})

    return (
        pt.LazyFrame.from_existing(power_df.lazy()).set_model(PowerTimeSeries),
        pt.DataFrame(metadata_df).set_model(TimeSeriesMetadata),
        pt.LazyFrame.from_existing(nwp_df.lazy()).set_model(NwpInMemory),
    )


def test_bulk_and_single_run_features_are_identical() -> None:
    power_ts, metadata, nwp = _build_fixtures()

    bulk = engineer_features(
        selected_features=_FEATURES,
        power_time_series=power_ts,
        time_series_metadata=metadata,
        nwp=nwp,
        power_fcst_init_time=None,
        nwp_publication_delay_hours=_DELAY_HOURS,
    ).collect()

    # Replay each NWP run in single-run mode at the same t0 bulk derives for it.
    single_run_parts = []
    for run in _NWP_RUNS:
        replay = engineer_features(
            selected_features=_FEATURES,
            power_time_series=power_ts,
            time_series_metadata=metadata,
            nwp=nwp,
            power_fcst_init_time=run + timedelta(hours=_DELAY_HOURS),
            nwp_init_time=run,
        ).collect()
        # Keep only rows the NWP run actually covers (single-run mode is power-centric and
        # emits null-weather rows for valid_times outside this run's window).
        single_run_parts.append(replay.filter(pl.col("nwp_lead_time_hours").is_not_null()))
    single_run = pl.concat(single_run_parts)

    assert len(bulk) == len(_NWP_RUNS) * len(_MEMBERS) * 13
    # Guard: the power lag must actually resolve to non-null observed values for some rows,
    # otherwise "identical" would be a vacuous all-null match on both sides.
    assert bulk["power_lag_3h"].is_not_null().any()
    assert_frame_equal(
        bulk.select(_COMPARE_COLS).sort(_SORT_COLS),
        single_run.select(_COMPARE_COLS).sort(_SORT_COLS),
        check_dtypes=False,
    )
