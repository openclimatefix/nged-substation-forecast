"""The six detector arms, the daily fit, and the confound shapes of experiment 00040.

An arm fits a series within one season-year window by least squares within (month, local half-hour,
day type) cells, and reports its PV capacity `c`. Every arm is linear in the series, so the
capacity fitted to a sum is the sum of the capacities fitted to the parts. That is why a confound
shape and a unit of injected PV are each fitted once, alone, and then added as vectors.
"""

import e00040_common as common
import e00040_model as model
import numpy as np


def _capacity(*, y: np.ndarray, regressors: list[np.ndarray], cells: np.ndarray) -> float:
    """Return minus the first coefficient of `y` on `regressors`, within cells."""
    _, inverse = np.unique(cells, return_inverse=True)
    counts = np.bincount(inverse)

    def demeaned(values: np.ndarray) -> np.ndarray:
        return values - (np.bincount(inverse, weights=values) / counts)[inverse]

    x = np.column_stack([demeaned(r) for r in regressors])
    beta, *_ = np.linalg.lstsq(x, demeaned(y), rcond=None)
    return float(-beta[0])


def _band_masks(*, half_hour: np.ndarray) -> list[np.ndarray]:
    return [(half_hour >= lo) & (half_hour < hi) for lo, hi in common.HOUR_BANDS]


def arm_capacities(*, y: np.ndarray, window: model.Window) -> np.ndarray:
    """Return the capacity of every fit in `common.ALL_FITS` for the series `y` on `window`.

    The summer-only arms drop the October to March rows before the cells are built.

    Args:
        y: The series to fit, one value per row of `window`.
        window: The rows and the design.

    Returns:
        The capacities, MW, in the order of `common.ALL_FITS` (six arms, then the daily fit M1d).
    """
    summer = np.isin(window.month, common.SUMMER_MONTHS)
    season = summer.astype(float)
    cloud = window.cloud
    bands = _band_masks(half_hour=window.half_hour)
    base = [window.u, window.hdd]
    full = {
        "M1": base,
        "M1L": [*base, cloud],
        "M2": [*base, cloud * season, cloud * (1.0 - season)],
        "M2h": [
            *base,
            *(cloud * s * band for s in (season, 1.0 - season) for band in bands),
        ],
    }
    out = [_capacity(y=y, regressors=full[arm], cells=window.cells) for arm in common.ARMS[:4]]
    sw = window.take(mask=summer)
    ys = y[summer]
    sbands = _band_masks(half_hour=sw.half_hour)
    out.append(_capacity(y=ys, regressors=[sw.u, sw.hdd], cells=sw.cells))
    out.append(
        _capacity(
            y=ys, regressors=[sw.u, sw.hdd, *(sw.cloud * band for band in sbands)], cells=sw.cells
        )
    )
    y_daily, daily = model.daily_sums(window=window, y=y)
    out.append(_capacity(y=y_daily, regressors=[daily.u, daily.hdd], cells=daily.cells))
    return np.array(out)


def confound_shapes(*, window: model.Window) -> dict[str, np.ndarray]:
    """Return the eight confound shapes, each divided by its mean over the window's rows.

    The shapes are demand loads in MW per unit of mean load, so a positive shape raises the demand.
    The load `sunload` is the same as the PV shape `u`, with the opposite sign of PV.

    Args:
        window: Daylight rows with `cloud`, `u_clear`, `day_cloud`, and `u` filled.

    Returns:
        The shapes by name, in `common.SHAPES`.
    """
    cloud = window.cloud
    summer = np.isin(window.month, common.SUMMER_MONTHS)
    lowsun = cloud * (1.0 - window.u_clear)
    raw = {
        "flat": cloud,
        "lowsun": lowsun,
        "winter": cloud * ~summer,
        "dayL": window.day_cloud,
        "evening": cloud * (window.half_hour >= common.EVENING_HALF_HOUR),
        "threshold": (cloud > common.THRESHOLD_CLOUD).astype(float),
        "mixture": 0.5 * cloud / cloud.mean() + 0.5 * lowsun / lowsun.mean(),
        "sunload": window.u,
    }
    return {name: values / values.mean() for name, values in raw.items()}
