# ruff: noqa: E501
"""The flexible sunshine response: a tensor-product linear spline in clear-sky index and sun elevation.

The response is `B(csi, elevation) @ gamma`, in points of effective capacity, added to demand in
experiment 184's T-all fit. `csi` is the CAMS (training) or calibrated member (forecast)
irradiance over the Haurwitz clear-sky irradiance. The basis is a product of linear hat functions on
`CSI_KNOTS` (6 knots) and `ELEVATION_KNOTS_DEG` (5 free knots, the response is zero with the sun at
the horizon), so it has 30 columns. The penalty is a second difference along both axes plus a
ridge.
"""

import e00323_common as common
import numpy as np
from studies.baselines import haurwitz_w_m2
from studies.dp_gam import SITE_YEAR_ROWS

N_CSI: int = len(common.CSI_KNOTS)
N_ELEVATION: int = len(common.ELEVATION_KNOTS_DEG) - 1
N_COLUMNS: int = N_CSI * N_ELEVATION
MIN_CLEAR_SKY_W_M2: float = 20.0


def clear_sky_index(*, ghi_w_m2: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Return the clear-sky index, 0 where the clear-sky irradiance is under 20 W/m2."""
    clear = haurwitz_w_m2(apparent_zenith_deg=90.0 - elevation_deg)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        index = np.where(
            clear > MIN_CLEAR_SKY_W_M2, ghi_w_m2 / np.where(clear > 0, clear, 1.0), 0.0
        )
    return np.clip(np.nan_to_num(index, nan=0.0), common.CSI_KNOTS[0], common.CSI_KNOTS[-1])


def hats(*, value: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Return the (n, K) linear-interpolation weights of `value` on `knots`, held beyond them."""
    clipped = np.clip(value, knots[0], knots[-1])
    upper = np.clip(np.searchsorted(knots, clipped, side="right"), 1, len(knots) - 1)
    lower = upper - 1
    fraction = (clipped - knots[lower]) / (knots[upper] - knots[lower])
    weights = np.zeros((len(value), len(knots)))
    rows = np.arange(len(value))
    weights[rows, lower] = 1.0 - fraction
    weights[rows, upper] = fraction
    return weights


def basis(*, ghi_w_m2: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Return the (n, 30) basis; rows with the sun at or below the horizon are all zero.

    Columns are elevation-major: column `e * 6 + c` is the hat of elevation knot `e + 1` times the
    hat of clear-sky-index knot `c`.
    """
    csi = hats(
        value=clear_sky_index(ghi_w_m2=ghi_w_m2, elevation_deg=elevation_deg),
        knots=common.CSI_KNOTS,
    )
    elevation = hats(value=elevation_deg, knots=common.ELEVATION_KNOTS_DEG)[:, 1:]
    elevation = elevation * (elevation_deg > 0.0)[:, None]
    return (elevation[:, :, None] * csi[:, None, :]).reshape(len(csi), -1)


def penalty(*, n_rows: int) -> np.ndarray:
    """Return the (30, 30) penalty: second differences along both axes plus a ridge."""
    scale = n_rows / SITE_YEAR_ROWS
    second_csi = np.diff(np.eye(N_CSI), n=2, axis=0)
    second_elevation = np.diff(np.eye(N_ELEVATION), n=2, axis=0)
    along_csi = np.kron(np.eye(N_ELEVATION), second_csi.T @ second_csi)
    along_elevation = np.kron(second_elevation.T @ second_elevation, np.eye(N_CSI))
    return scale * (
        common.LAMBDA_DIFFERENCE * (along_csi + along_elevation)
        + common.LAMBDA_RIDGE * np.eye(N_COLUMNS)
    )
