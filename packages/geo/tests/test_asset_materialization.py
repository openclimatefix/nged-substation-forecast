from dagster import materialize, FilesystemIOManager
from geo.assets import uk_boundary, gb_h3_grid_weights


def test_materialize_gb_h3_grid_weights():
    io_manager = FilesystemIOManager(base_dir="/tmp/dagster_test")
    result = materialize([uk_boundary, gb_h3_grid_weights], resources={"io_manager": io_manager})
    assert result.success
