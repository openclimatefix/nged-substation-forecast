import patito as pt
import polars as pl
from collections.abc import Mapping
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from ml_core.model import BaseForecaster
from ml_core.experimental import LocalForecasters
from datetime import datetime, timezone


class MockForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.trained = False

    def train(
        self,
        config: ModelConfig,
        flows_30m: pl.LazyFrame,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
    ):
        self.trained = True
        return self

    def predict(
        self,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        inference_params: InferenceParams,
        flows_30m: pl.LazyFrame,
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        # Return a dummy prediction
        sub_id = time_series_metadata["time_series_id"][0]
        df = pl.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 2, tzinfo=timezone.utc)],
                "time_series_id": [sub_id],
                "ensemble_member": [0],
                "power_fcst_model_name": ["mock"],
                "power_fcst_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "nwp_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "power_fcst_init_year_month": ["2026-01"],
                "power_fcst": [10.0 * float(sub_id)],
            }
        ).with_columns(
            [
                pl.col("time_series_id").cast(pl.Int64),
                pl.col("ensemble_member").cast(pl.UInt8),
                pl.col("power_fcst_model_name").cast(pl.Categorical),
                pl.col("power_fcst").cast(pl.Float32),
            ]
        )
        return pt.DataFrame[PowerForecast](df)

    def log_model(self, model_name: str) -> None:
        pass


def test_local_forecasters():
    # Setup dummy data
    sub_meta = pt.DataFrame[TimeSeriesMetadata](
        {
            "substation_number": [1, 2],
            "substation_name": ["Sub1", "Sub2"],
            "latitude": [51.0, 52.0],
            "longitude": [-1.0, -2.0],
            "h3_res_5": [123, 456],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
            "time_series_id": [1, 2],
        }
    )

    sub_flows = pt.DataFrame[PowerTimeSeries](
        {
            "time_series_id": [1, 2],
            "start_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
            "period_end_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)] * 2,
            "power": [10.0, 20.0],
        }
    ).lazy()

    config = ModelConfig(
        power_fcst_model_name="mock",
        hyperparameters={"learning_rate": 0.1, "n_estimators": 10, "max_depth": 3},
        features=ModelFeaturesConfig(nwps=[]),
    )

    # Initialize LocalForecasters
    local_forecasters = LocalForecasters(forecaster_cls=MockForecaster, some_arg="value")

    # Train
    local_forecasters.train(
        config=config,
        flows_30m=sub_flows,
        time_series_metadata=sub_meta,
    )

    assert len(local_forecasters.models) == 2
    assert isinstance(local_forecasters.models[1], MockForecaster)
    assert isinstance(local_forecasters.models[2], MockForecaster)
    assert local_forecasters.models[1].trained
    assert local_forecasters.models[2].trained
    assert local_forecasters.models[1].kwargs == {"some_arg": "value"}

    # Predict
    inference_params = InferenceParams(
        forecast_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        power_fcst_model_name="mock",
    )

    preds = local_forecasters.predict(
        time_series_metadata=sub_meta,
        inference_params=inference_params,
        flows_30m=sub_flows,
    )

    assert len(preds) == 2
    assert preds.filter(pl.col("time_series_id") == 1)["power_fcst"][0] == 10.0
    assert preds.filter(pl.col("time_series_id") == 2)["power_fcst"][0] == 20.0
