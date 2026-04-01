import pytest
from pydantic import ValidationError
from src.nged_substation_forecast.defs.xgb_assets import XGBoostConfig


def test_apply_config_overrides_invalid_date():
    with pytest.raises(ValidationError):
        XGBoostConfig(train_start="invalid-date")
