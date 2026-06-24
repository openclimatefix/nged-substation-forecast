from datetime import date

import pytest
from pydantic import ValidationError
from contracts.hydra_schemas import (
    CvConfig,
    CvFoldConfig,
)


def test_cv_fold_config_valid():
    fold = CvFoldConfig(
        fold_id="2022",
        train_start=date(2020, 1, 1),
        train_end=date(2021, 12, 31),
        val_start=date(2022, 1, 1),
        val_end=date(2022, 12, 31),
    )
    assert fold.fold_id == "2022"
    assert fold.val_end == date(2022, 12, 31)


def test_cv_fold_config_invalid_fold_id():
    with pytest.raises(ValidationError):
        CvFoldConfig.model_validate(
            {
                "fold_id": "9999",  # not in FoldId Literal
                "train_start": date(2020, 1, 1),
                "train_end": date(2021, 12, 31),
                "val_start": date(2022, 1, 1),
                "val_end": date(2022, 12, 31),
            }
        )


def test_cv_config_valid():
    config = CvConfig(
        folds=[
            CvFoldConfig(
                fold_id="2022",
                train_start=date(2020, 1, 1),
                train_end=date(2021, 12, 31),
                val_start=date(2022, 1, 1),
                val_end=date(2022, 12, 31),
            ),
            CvFoldConfig(
                fold_id="2023",
                train_start=date(2020, 1, 1),
                train_end=date(2022, 12, 31),
                val_start=date(2023, 1, 1),
                val_end=date(2023, 12, 31),
            ),
        ]
    )
    assert len(config.folds) == 2
    assert config.min_training_months == 6


def test_cv_config_custom_min_training_months():
    config = CvConfig(
        folds=[
            CvFoldConfig(
                fold_id="2022",
                train_start=date(2020, 1, 1),
                train_end=date(2021, 12, 31),
                val_start=date(2022, 1, 1),
                val_end=date(2022, 12, 31),
            )
        ],
        min_training_months=3,
    )
    assert config.min_training_months == 3
