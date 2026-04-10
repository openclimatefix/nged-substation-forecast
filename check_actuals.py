from contracts.settings import Settings
from src.nged_substation_forecast.defs.data_cleaning_assets import get_cleaned_actuals_lazy

settings = Settings()
# Use a mock context as we don't have a real one here.
# Assuming get_cleaned_actuals_lazy can handle None context.

actuals = get_cleaned_actuals_lazy(settings, context=None)

# Collect and check for nulls
df_actuals = actuals.collect()

print("Null counts in cleaned actuals:")
print(df_actuals.null_count())
print(f"Total rows: {len(df_actuals)}")

col = "power"
if col in df_actuals.columns:
    print(f"Nulls in {col}: {df_actuals[col].is_null().sum()}")
else:
    print(f"{col} not in cleaned actuals")
