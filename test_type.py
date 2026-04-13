import polars as pl
import patito as pt
from contracts.data_schemas import PowerTimeSeries


def scan_delta_table(delta_path: str) -> pt.LazyFrame[PowerTimeSeries]:
    return pt.LazyFrame.from_existing(pl.scan_delta(delta_path)).set_model(PowerTimeSeries)
