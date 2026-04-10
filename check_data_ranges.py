import polars as pl
from contracts.settings import Settings
from src.nged_substation_forecast.defs.data_cleaning_assets import get_cleaned_actuals_lazy

settings = Settings()
# Mock settings
actuals_lf = get_cleaned_actuals_lazy(settings, context=None)
actuals = actuals_lf.collect()

print("Actuals range:", actuals["period_end_time"].min(), actuals["period_end_time"].max())
metadata = pl.read_parquet(settings.nged_data_path / "parquet" / "time_series_metadata.parquet")
# ...
