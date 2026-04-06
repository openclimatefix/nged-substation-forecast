import pytest
from dagster import materialize
from geo.assets import uk_boundary, gb_h3_grid_weights
from geo.io_managers import CompositeIOManager
from upath import UPath


@pytest.mark.slow
def test_materialize_gb_h3_grid_weights():
    io_manager = CompositeIOManager(base_path=UPath("/tmp/dagster_test"))
    result = materialize([uk_boundary, gb_h3_grid_weights], resources={"io_manager": io_manager})
    assert result.success
