import patito as pt
import polars as pl

from contracts.data_schemas import ScalingParams
from ml_core.scaling import uint8_to_physical_unit


def test_uint8_to_physical_unit():
    params_df = pt.DataFrame[ScalingParams](
        {
            "col_name": ["temp", "wind"],
            "buffered_min": [-10.0, 0.0],
            "buffered_range": [50.0, 100.0],
        }
    )

    exprs = uint8_to_physical_unit(params_df)

    assert len(exprs) == 2

    # Test the expressions on a dataframe
    df = pl.DataFrame(
        {
            "temp": [0, 255, 127],
            "wind": [0, 255, 51],
        },
        schema={"temp": pl.UInt8, "wind": pl.UInt8},
    )

    result = df.select(exprs)

    # Expected temp:
    # 0 -> -10.0
    # 255 -> -10.0 + 50.0 = 40.0
    # 127 -> -10.0 + (127/255)*50.0 = 14.90196

    # Expected wind:
    # 0 -> 0.0
    # 255 -> 100.0
    # 51 -> (51/255)*100.0 = 20.0

    assert result["temp"][0] == -10.0
    assert result["temp"][1] == 40.0
    assert abs(result["temp"][2] - 14.90196) < 1e-4

    assert result["wind"][0] == 0.0
    assert result["wind"][1] == 100.0
    assert abs(result["wind"][2] - 20.0) < 1e-4
