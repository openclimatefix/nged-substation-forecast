import pytest
import mlflow
import tempfile
import shutil


@pytest.fixture(scope="session", autouse=True)
def mlflow_test_setup():
    tmp_dir = tempfile.mkdtemp()
    tracking_uri = f"file://{tmp_dir}"
    mlflow.set_tracking_uri(tracking_uri)
    yield
    shutil.rmtree(tmp_dir)
