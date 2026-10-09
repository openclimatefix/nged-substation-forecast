"""Read the saved out-of-fold forecasts back, for the reports and the charts.

Every function reads the files `forecast_fit.run_job` wrote, so a chart or a table needs no refit.
"""

from collections.abc import Sequence

import polars as pl
from forecast_fit import FITS_DIR, SettingType, arm_file

LOSS_COLUMNS: tuple[str, ...] = (
    "time",
    "month",
    "fold",
    "seed",
    "truth_mw",
    "p99_mw",
    "crps_pct",
    "pinball_pct",
    "median_abs_error_pct",
)
"""The columns `load_losses` returns beside `site` and `arm`."""


def saved_batteries(*, setting: SettingType, issue: str) -> list[str]:
    """Return the batteries that have at least one saved arm at an issue time and setting."""
    directory = FITS_DIR / setting / issue
    if not directory.exists():
        return []
    return sorted({p.name.split("__")[0] for p in directory.glob("*.parquet")})


def load_losses(
    *,
    setting: SettingType,
    issue: str,
    arms: Sequence[str],
    batteries: Sequence[str],
    columns: Sequence[str] = LOSS_COLUMNS,
) -> pl.DataFrame:
    """Return the saved per-row losses of some arms for some batteries.

    Args:
        setting: The hyperparameter setting.
        issue: The issue type.
        arms: The arm names.
        batteries: The battery identifiers.
        columns: The columns to read from each file.

    Returns:
        One row per battery, arm, seed, and scored half-hour, with `site` (the battery) and `arm`
        added to `columns`. A battery lacking a file for an arm contributes no rows for it.
    """
    frames = []
    for battery in batteries:
        for arm in arms:
            path = arm_file(setting=setting, issue=issue, battery_id=battery, arm=arm)
            if path.exists():
                frames.append(
                    pl.read_parquet(path, columns=list(columns)).with_columns(
                        site=pl.lit(battery), arm=pl.lit(arm)
                    )
                )
    return pl.concat(frames)
