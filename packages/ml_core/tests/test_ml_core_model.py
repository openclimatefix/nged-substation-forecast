import patito as pt
import polars as pl
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    ProcessedNwp,
    SubstationFlows,
    SubstationMetadata,
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
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
    ):
        self.trained = True
        return self

    def predict(
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        # Return a dummy prediction
        sub_num = substation_metadata["substation_number"][0]
        df = pl.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 2, tzinfo=timezone.utc)],
                "substation_number": [sub_num],
                "ensemble_member": [0],
                "power_fcst_model_name": ["mock"],
                "power_fcst_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "nwp_init_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "power_fcst_init_year_month": ["2026-01"],
                "MW_or_MVA": [10.0 * sub_num],
            }
        ).with_columns(
            [
                pl.col("substation_number").cast(pl.Int32),
                pl.col("ensemble_member").cast(pl.UInt8),
                pl.col("power_fcst_model_name").cast(pl.Categorical),
                pl.col("MW_or_MVA").cast(pl.Float32),
            ]
        )
        return pt.DataFrame[PowerForecast](df)

    def log_model(self, model_name: str) -> None:
        pass


def test_local_forecasters():
    # Setup dummy data
    sub_meta = pt.DataFrame[SubstationMetadata](
        {
            "substation_number": [1, 2],
            "substation_name": ["Sub1", "Sub2"],
            "latitude": [51.0, 52.0],
            "longitude": [-1.0, -2.0],
            "h3_res_5": [123, 456],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
        }
    )

    sub_flows = pt.DataFrame[SubstationFlows](
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
            "substation_number": [1, 2],
            "MW": [10.0, 20.0],
            "MVA": [10.0, 20.0],
            "MVAr": [0.0, 0.0],
            "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
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
        substation_power_flows=sub_flows,
        substation_metadata=sub_meta,
    )

    assert len(local_forecasters.models) == 2
    assert isinstance(local_forecasters.models[1], MockForecaster)
    assert isinstance(local_forecasters.models[2], MockForecaster)
    assert local_forecasters.models[1].trained
    assert local_forecasters.models[2].trained
    assert local_forecasters.models[1].kwargs == {"some_arg": "value"}

    # Predict
    inference_params = InferenceParams(
        nwp_init_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        power_fcst_model_name="mock",
    )

    preds = local_forecasters.predict(
        substation_metadata=sub_meta,
        inference_params=inference_params,
        substation_power_flows=sub_flows,
    )

    assert len(preds) == 2
    assert preds.filter(pl.col("substation_number") == 1)["MW_or_MVA"][0] == 10.0
    assert preds.filter(pl.col("substation_number") == 2)["MW_or_MVA"][0] == 20.0
