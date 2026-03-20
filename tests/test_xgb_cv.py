from src.nged_substation_forecast.defs.xgb_jobs import generate_expanding_windows, CVConfig
from src.nged_substation_forecast.defs.xgb_assets import XGBoostTrainingParams


def test_generate_expanding_windows():
    config = CVConfig(start_date="2025-01-01", end_date="2025-04-01", fold_size_months=1)
    outputs = list(generate_expanding_windows(config))

    assert len(outputs) == 3

    # Fold 1
    fold1 = outputs[0].value
    assert isinstance(fold1, XGBoostTrainingParams)
    assert fold1.train_start_date == "2025-01-01"
    assert fold1.train_end_date == "2025-02-01"
    assert fold1.test_end_date == "2025-03-01"

    # Fold 2
    fold2 = outputs[1].value
    assert fold2.train_start_date == "2025-01-01"
    assert fold2.train_end_date == "2025-03-01"
    assert fold2.test_end_date == "2025-04-01"

    # Fold 3
    fold3 = outputs[2].value
    assert fold3.train_start_date == "2025-01-01"
    assert fold3.train_end_date == "2025-04-01"
    assert fold3.test_end_date == "2025-04-01"


def test_generate_expanding_windows_short():
    config = CVConfig(start_date="2025-01-01", end_date="2025-01-15", fold_size_months=1)
    outputs = list(generate_expanding_windows(config))

    assert len(outputs) == 1
    assert outputs[0].value.train_start_date == "2025-01-01"
    assert outputs[0].value.train_end_date == "2025-01-15"
    assert outputs[0].value.test_end_date is None
