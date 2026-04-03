from dagster import materialize, FilesystemIOManager
from geo.assets import uk_boundary, gb_h3_grid_weights
from contracts.settings import Settings
from pathlib import Path


def test_materialize_gb_h3_grid_weights():
    io_manager = FilesystemIOManager(base_dir="/tmp/dagster_test")
    settings = Settings(
        nged_ckan_token="dummy",
        nged_s3_bucket_url="http://dummy.com",
        nged_s3_bucket_access_key="dummy",
        nged_s3_bucket_secret="dummy",
        nwp_data_path=Path("/tmp/dagster_test/nwp"),
    )
    result = materialize(
        [uk_boundary, gb_h3_grid_weights],
        resources={"io_manager": io_manager, "settings": settings},
    )
    assert result.success
