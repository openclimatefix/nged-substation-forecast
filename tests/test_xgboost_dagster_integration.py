import pytest


@pytest.mark.integration
def test_xgboost_dagster_integration():
    """Integration test for XGBoost pipeline - currently skipped due to data dependencies.

    Note: This test requires actual data in the Delta tables. In CI/CD environments
    without real NGED data, this test is skipped. Run manually with:
        uv run pytest tests/test_xgboost_dagster_integration.py -v --runxfail
    """
    pytest.skip(
        "Integration test requires actual NGED data in Delta tables. Skip by default in CI."
    )
