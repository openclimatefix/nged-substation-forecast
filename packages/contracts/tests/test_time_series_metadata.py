import patito as pt
import pytest
from contracts.power_schemas import TimeSeriesMetadata


def test_time_series_metadata_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "time_series_name": ["ALFORD 33 11kV S STN"],
                "time_series_type": ["BESS"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [1],
                "substation_type": ["BSP"],
                "latitude": [52.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            }
        )
        .set_model(TimeSeriesMetadata)
        .cast()
    )

    # Should pass
    df.validate()


@pytest.mark.parametrize(
    "data, expected_error",
    [
        # Invalid latitude (too high)
        (
            {
                "time_series_id": [123],
                "time_series_name": ["Name"],
                "time_series_type": ["BESS"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [1],
                "substation_type": ["BSP"],
                "latitude": [62.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            },
            "latitude",
        ),
        # Invalid substation_type
        (
            {
                "time_series_id": [123],
                "time_series_name": ["Name"],
                "time_series_type": ["BESS"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [1],
                "substation_type": ["InvalidType"],
                "latitude": [52.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            },
            "substation_type",
        ),
        # Invalid time_series_type
        (
            {
                "time_series_id": [123],
                "time_series_name": ["Name"],
                "time_series_type": ["InvalidType"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [1],
                "substation_type": ["BSP"],
                "latitude": [52.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            },
            "time_series_type",
        ),
        # Invalid units
        (
            {
                "time_series_id": [123],
                "time_series_name": ["Name"],
                "time_series_type": ["BESS"],
                "units": ["InvalidUnits"],
                "licence_area": ["EMids"],
                "substation_number": [1],
                "substation_type": ["BSP"],
                "latitude": [52.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            },
            "units",
        ),
        # Invalid substation_number (too low)
        (
            {
                "time_series_id": [123],
                "time_series_name": ["Name"],
                "time_series_type": ["BESS"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [0],
                "substation_type": ["BSP"],
                "latitude": [52.0],
                "longitude": [-1.0],
                "h3_res_5": [123456789],
            },
            "substation_number",
        ),
    ],
)
def test_time_series_metadata_invalid_data(data, expected_error):
    # We need to cast to ensure the types are checked
    df = pt.DataFrame(data).set_model(TimeSeriesMetadata)

    # We expect validation to fail
    with pytest.raises(Exception, match=expected_error):
        df.cast().validate()
