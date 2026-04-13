import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import PowerTimeSeries

@dg.asset
def train_xgboost(
    context: dg.AssetExecutionContext,
    # The ML asset is completely pure. It just asks for DataFrames.
    cleaned_power_time_series: pt.DataFrame[PowerTimeSeries],
    processed_nwp_data: pl.DataFrame 
):
    """
    Trains the XGBoost model.
    
    WHY PURE? The ML engineer doesn't need to know about Delta Lake, Parquet, 
    or Dagster partitions. They just write a function that takes DataFrames 
    and returns a model.
    """
    # 1. Initialize model configuration
    config = load_model_config("xgboost")
    
    # 2. Train the model using the pure DataFrames
    model = XGBoostForecaster(config)
    model.fit(
        power_data=cleaned_power_time_series,
        weather_data=processed_nwp_data
    )
    
    # 3. Return the trained model object
    return model
