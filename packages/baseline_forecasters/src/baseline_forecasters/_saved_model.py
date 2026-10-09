"""The ``meta.json`` that every baseline forecaster writes into its saved-model directory."""

import json
import shutil
from pathlib import Path
from typing import TypedDict

from contracts.config_schemas import class_target
from ml_core.base_forecaster import BaseForecaster

_META_FILENAME = "meta.json"


class SavedModelMeta(TypedDict):
    """The three keys of ``meta.json``, which ``read_meta`` returns."""

    model_params: dict
    trained_time_series_ids: list[int]
    model_class: str


def clear_directory_and_write_meta(*, path: Path, forecaster: BaseForecaster) -> None:
    """Replace ``path`` with an empty directory holding only ``meta.json``.

    The directory is cleared first so that a re-materialised fold never keeps a file that an
    earlier, larger model wrote.

    Args:
        path: The directory to clear and recreate.
        forecaster: The forecaster whose config, trained population, and class are recorded.
    """
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    meta: SavedModelMeta = {
        "model_params": forecaster.model_params.model_dump(mode="json"),
        "trained_time_series_ids": forecaster.trained_time_series_ids,
        "model_class": class_target(forecaster),
    }
    (path / _META_FILENAME).write_text(json.dumps(meta))


def read_meta(path: Path) -> SavedModelMeta:
    """Read back the ``meta.json`` that ``clear_directory_and_write_meta`` wrote into ``path``.

    Args:
        path: The saved-model directory.

    Returns:
        The config, the trained population, and the class of the forecaster that was saved.
    """
    meta: SavedModelMeta = json.loads((path / _META_FILENAME).read_text())
    return meta
