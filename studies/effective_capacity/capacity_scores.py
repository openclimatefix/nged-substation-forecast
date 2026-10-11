"""Turn saved raw estimates into issue-time scales, for the scoring and report scripts.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
"""

from functools import cache
from pathlib import Path

import numpy as np
from capacity_paths import (
    ORIGINAL,
    estimate_path,
    load_estimate,
    record_for,
    setting_name,
)
from studies.capacity_estimators import (
    CALIBRATED_ESTIMATORS,
    ISSUE_LAG_DAYS,
    CapacityRecord,
    IssueScale,
    RawEstimate,
    issue_time_estimates,
    reference_levels,
)


@cache
def cached_record(*, root: Path, record_id: str) -> CapacityRecord:
    """Return a record, built once per process.

    Args:
        root: The output root.
        record_id: The record's identifier.

    Returns:
        The record.
    """
    return record_for(root=root, record_id=record_id)


def record_id_of(*, file_id: str) -> str:
    """Return the identifier of the record an estimates file was computed on."""
    if file_id in (f"{ORIGINAL}__tuning", "hindsight_forward"):
        return ORIGINAL
    return file_id


def original_file_id(*, estimator: str, tuning: bool) -> str:
    """Return the file identifier of the original record's estimates.

    Args:
        estimator: The estimator. DP1 runs once over the whole record, so it has no tuning file.
        tuning: Whether the tuning phase's smaller file is wanted.

    Returns:
        `original__tuning` or `original`.
    """
    return f"{ORIGINAL}__tuning" if tuning and estimator != "DP1" else ORIGINAL


def load_raw(*, root: Path, estimator: str, params: dict[str, float], file_id: str) -> RawEstimate:
    """Read an estimator's raw estimates.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its settings.
        file_id: The file identifier.

    Returns:
        The raw estimate.
    """
    path = estimate_path(
        root=root, estimator=estimator, setting=setting_name(params=params), record_id=file_id
    )
    return load_estimate(path=path)


def reference_for(
    *, root: Path, estimator: str, params: dict[str, float], tuning: bool
) -> np.ndarray | None:
    """Return the reference-year constants of a calibrated estimator, from the original record.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its settings.
        tuning: Whether to read the tuning phase's original file.

    Returns:
        One constant per plant, or `None` for an estimator already in megawatts.
    """
    if estimator not in CALIBRATED_ESTIMATORS:
        return None
    raw = load_raw(
        root=root,
        estimator=estimator,
        params=params,
        file_id=original_file_id(estimator=estimator, tuning=tuning),
    )
    return reference_levels(raw=raw, record=cached_record(root=root, record_id=ORIGINAL))


def scale_for(
    *,
    root: Path,
    estimator: str,
    params: dict[str, float],
    file_id: str,
    tuning: bool = False,
    lag_days: int = ISSUE_LAG_DAYS,
) -> IssueScale:
    """Return the issue-time scale of an estimator on one record.

    Args:
        root: The output root.
        estimator: The estimator.
        params: Its settings.
        file_id: The estimates file to read (`original`, `draw03`, `control`, ...).
        tuning: Whether the reference constants come from the tuning phase's original file.
        lag_days: The read lag (2 for the primary, 3 for the satellite-latency sensitivity).

    Returns:
        The scale, its relative standard deviation, and the fallback reasons.
    """
    raw = load_raw(root=root, estimator=estimator, params=params, file_id=file_id)
    record = cached_record(root=root, record_id=record_id_of(file_id=file_id))
    return issue_time_estimates(
        record=record,
        raw=raw,
        reference=reference_for(root=root, estimator=estimator, params=params, tuning=tuning),
        lag_days=lag_days,
    )
