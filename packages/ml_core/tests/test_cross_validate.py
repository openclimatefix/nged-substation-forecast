"""Tests for cross_validate()."""

from datetime import date, datetime, timezone

import patito as pt
import polars as pl
import pytest
from contracts.hydra_schemas import CvFoldConfig
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from ml_core.cross_validate import cross_validate


# ---------------------------------------------------------------------------
# Minimal stub forecaster for testing
# ---------------------------------------------------------------------------


class _StubConfig(BaseForecasterConfig):
    """No extra params needed for the stub."""


class _StubForecaster(BaseForecaster):
    """Returns a constant power_fcst of 10.0 for every row."""

    model_params: _StubConfig

    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        pass  # stateless stub

    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        df = data.collect()
        result = df.select(["valid_time", "time_series_id", "power_fcst_init_time"]).with_columns(
            ensemble_member=pl.lit(0, dtype=pl.Int8),
            nwp_init_time=pl.lit(None, dtype=pl.Datetime("us", "UTC")),
            power_fcst=pl.lit(10.0, dtype=pl.Float32),
            power_fcst_model_name=pl.lit("stub").cast(pl.Categorical),
            power_fcst_model_version=pl.lit(1, dtype=pl.Int16),
            ml_flow_experiment_id=pl.lit(None, dtype=pl.Int32),
            fold_id=pl.lit("live").cast(pl.Categorical),
        )
        return PowerForecast.validate(result)

    def save(self, path):  # type: ignore[override]
        pass

    @classmethod
    def load(cls, path):  # type: ignore[override]
        return cls(
            _StubConfig(
                selected_features=set(),
                power_fcst_model_name="stub",
                power_fcst_model_version=1,
            )
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TS_ID = 1


def _make_power_lf(start: datetime, end: datetime) -> pt.LazyFrame[PowerTimeSeries]:
    """Half-hourly power data for a single time series."""
    times = pl.datetime_range(
        start, end, interval="30m", time_unit="us", eager=True
    ).dt.replace_time_zone("UTC")
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([_TS_ID] * len(times), dtype=pl.Int32),
            "time": times,
            "power": pl.Series([5.0] * len(times), dtype=pl.Float32),
        }
    )
    return pt.LazyFrame.from_existing(df.lazy()).set_model(PowerTimeSeries)


def _make_metadata() -> pt.DataFrame[TimeSeriesMetadata]:
    df = pl.DataFrame(
        {
            "time_series_id": pl.Series([_TS_ID], dtype=pl.Int32),
            "time_series_name": ["Test Station"],
            "time_series_type": ["Raw Flow"],
            "units": ["MW"],
            "licence_area": ["EMids"],
            "substation_number": pl.Series([1], dtype=pl.Int32),
            "substation_type": ["Primary"],
            "latitude": pl.Series([52.0], dtype=pl.Float32),
            "longitude": pl.Series([-1.0], dtype=pl.Float32),
            "h3_res_5": pl.Series([123456789], dtype=pl.UInt64),
        }
    )
    return pt.DataFrame(df).set_model(TimeSeriesMetadata)


def _make_config() -> _StubConfig:
    return _StubConfig(
        selected_features={"power_lag_24h"},
        power_fcst_model_name="stub",
        power_fcst_model_version=1,
    )


def _single_fold() -> CvFoldConfig:
    # val_start.year must be in FoldId ("2022" is the first valid CV year)
    return CvFoldConfig(
        fold_id=1,
        train_start=date(2020, 1, 1),
        train_end=date(2021, 12, 31),
        val_start=date(2022, 1, 1),
        val_end=date(2022, 12, 31),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cross_validate_returns_power_forecast():
    power_lf = _make_power_lf(
        datetime(2019, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 12, 31, 23, 30, tzinfo=timezone.utc),
    )
    result = cross_validate(
        forecaster_class=_StubForecaster,
        forecaster_config=_make_config(),
        power_lf=power_lf,
        nwp_lf=None,
        metadata_df=_make_metadata(),
        folds=[_single_fold()],
    )
    assert isinstance(result, pl.DataFrame)
    assert "fold_id" in result.columns
    # fold_id is the validation year string (val_start.year for this fold = 2022)
    assert result["fold_id"].unique().to_list() == ["2022"]
    assert (result["power_fcst"] == 10.0).all()


def test_cross_validate_creates_fresh_forecaster_per_fold():
    """Two folds → two independent forecaster instances (state does not bleed)."""
    power_lf = _make_power_lf(
        datetime(2019, 1, 1, tzinfo=timezone.utc),
        datetime(2023, 12, 31, 23, 30, tzinfo=timezone.utc),
    )
    folds = [
        CvFoldConfig(
            fold_id=1,
            train_start=date(2019, 1, 1),
            train_end=date(2021, 12, 31),
            val_start=date(2022, 1, 1),
            val_end=date(2022, 12, 31),
        ),
        CvFoldConfig(
            fold_id=2,
            train_start=date(2019, 1, 1),
            train_end=date(2022, 12, 31),
            val_start=date(2023, 1, 1),
            val_end=date(2023, 12, 31),
        ),
    ]
    result = cross_validate(
        forecaster_class=_StubForecaster,
        forecaster_config=_make_config(),
        power_lf=power_lf,
        nwp_lf=None,
        metadata_df=_make_metadata(),
        folds=folds,
    )
    assert sorted(result["fold_id"].unique().to_list()) == ["2022", "2023"]


def test_cross_validate_excludes_ineligible_time_series():
    """A time series that starts too late is excluded from the fold."""
    # Power data only starts 3 months before val_start — not enough for min_training_months=6.
    power_lf = _make_power_lf(
        datetime(2021, 10, 1, tzinfo=timezone.utc),
        datetime(2022, 12, 31, 23, 30, tzinfo=timezone.utc),
    )
    with pytest.raises(ValueError, match="No eligible time series"):
        cross_validate(
            forecaster_class=_StubForecaster,
            forecaster_config=_make_config(),
            power_lf=power_lf,
            nwp_lf=None,
            metadata_df=_make_metadata(),
            folds=[_single_fold()],
            min_training_months=6,
        )


def test_cross_validate_result_validates_as_power_forecast():
    """The return value validates as PowerForecast."""
    power_lf = _make_power_lf(
        datetime(2019, 1, 1, tzinfo=timezone.utc),
        datetime(2022, 12, 31, 23, 30, tzinfo=timezone.utc),
    )
    result = cross_validate(
        forecaster_class=_StubForecaster,
        forecaster_config=_make_config(),
        power_lf=power_lf,
        nwp_lf=None,
        metadata_df=_make_metadata(),
        folds=[_single_fold()],
    )
    # validate() would raise if the schema doesn't match
    PowerForecast.validate(result, allow_superfluous_columns=True)
