from datetime import date
from src.nged_substation_forecast.defs.xgb_jobs import generate_expanding_windows, CVConfig
from contracts.hydra_schemas import TrainingConfig


def test_generate_expanding_windows():
    config = CVConfig(start_date="2025-01-01", end_date="2025-04-01", fold_size_months=1)
    outputs = list(generate_expanding_windows(config))

    assert len(outputs) == 3

    # Fold 1
    fold1 = outputs[0].value
    assert isinstance(fold1, TrainingConfig)
    assert fold1.data_split.train_start == date(2025, 1, 1)
    assert fold1.data_split.train_end == date(2025, 2, 1)
    assert fold1.data_split.test_end == date(2025, 3, 1)

    # Fold 2
    fold2 = outputs[1].value
    assert fold2.data_split.train_start == date(2025, 1, 1)
    assert fold2.data_split.train_end == date(2025, 3, 1)
    assert fold2.data_split.test_end == date(2025, 4, 1)

    # Fold 3
    fold3 = outputs[2].value
    assert fold3.data_split.train_start == date(2025, 1, 1)
    assert fold3.data_split.train_end == date(2025, 4, 1)
    assert fold3.data_split.test_end == date(2025, 4, 1)


def test_generate_expanding_windows_short():
    config = CVConfig(start_date="2025-01-01", end_date="2025-01-15", fold_size_months=1)
    outputs = list(generate_expanding_windows(config))

    assert len(outputs) == 1
    assert outputs[0].value.data_split.train_start == date(2025, 1, 1)
    assert outputs[0].value.data_split.train_end == date(2025, 1, 15)
