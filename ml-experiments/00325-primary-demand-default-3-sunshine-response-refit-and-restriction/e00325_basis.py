"""The sunshine response's basis for the unrestricted (F1) and restricted (F1s) variants.

F1 is experiment 323's basis (a tensor-product linear spline in clear-sky index and sun
elevation, 30 columns). F1s multiplies each row of that basis by a smooth weight that is zero
below a clear-sky index of 0.6 and below a sun elevation of 15 degrees, so the response is zero
there and free above, with the ridge and second-difference penalties of experiment 323.
"""

import e00323_basis as base323  # ty: ignore[unresolved-import]
import e00325_common as common
import numpy as np

N_COLUMNS: int = base323.N_COLUMNS
penalty = base323.penalty


def restriction_weight(*, ghi_w_m2: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
    """Return the weight in [0, 1] that F1s multiplies the response by."""
    index = base323.clear_sky_index(ghi_w_m2=ghi_w_m2, elevation_deg=elevation_deg)
    w_index = np.clip((index - common.RESTRICT_INDEX_START) / common.RESTRICT_INDEX_RAMP, 0.0, 1.0)
    w_elevation = np.clip(
        (elevation_deg - common.RESTRICT_ELEVATION_START_DEG) / common.RESTRICT_ELEVATION_RAMP_DEG,
        0.0,
        1.0,
    )
    return w_index * w_elevation


def basis(*, ghi_w_m2: np.ndarray, elevation_deg: np.ndarray, variant: str) -> np.ndarray:
    """Return the (n, 30) basis of a response variant (`f1` or `f1s`)."""
    columns = base323.basis(ghi_w_m2=ghi_w_m2, elevation_deg=elevation_deg)
    if variant == "f1":
        return columns
    if variant == "f1s":
        return columns * restriction_weight(ghi_w_m2=ghi_w_m2, elevation_deg=elevation_deg)[:, None]
    msg = f"unknown response variant {variant!r}"
    raise ValueError(msg)
