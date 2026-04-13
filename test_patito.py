import polars as pl
import patito as pt
from contracts.data_schemas import PowerTimeSeries

# Create a dummy delta table
df = pl.DataFrame(
    {"period_end_time": ["2023-01-01T00:00:00Z"], "substation_id": ["A"], "active_power_mw": [1.0]}
)
df.write_delta("dummy_delta")


# Test the function
def scan_delta_table(delta_path: str) -> pt.LazyFrame[PowerTimeSeries]:
    lf = pl.scan_delta(delta_path)
    return pt.LazyFrame.from_existing(lf).set_model(PowerTimeSeries)


pt_lf = scan_delta_table("dummy_delta")
print(type(pt_lf))
print(pt_lf.collect())
