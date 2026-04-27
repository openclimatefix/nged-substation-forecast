import patito as pt
import polars as pl
import pytest
from contracts.geo_schemas import H3GridWeights
from patito.exceptions import DataFrameValidationError


def test_h3_grid_weights_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "h3_index": [123456789],
                "nwp_lat": [52.0],
                "nwp_lng": [-1.0],
                "len": [10],
                "total": [100],
                "proportion": [0.1],
            }
        )
        .set_model(H3GridWeights)
        .cast()
    )
    df.validate()


def test_h3_grid_weights_invalid_lat():
    # Invalid latitude (too high)
    df = pt.DataFrame(
        {
            "h3_index": [123456789],
            "nwp_lat": [91.0],
            "nwp_lng": [-1.0],
            "len": [10],
            "total": [100],
            "proportion": [0.1],
        }
    ).set_model(H3GridWeights)

    with pytest.raises(DataFrameValidationError):
        df.cast().validate()
