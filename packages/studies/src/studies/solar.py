"""Solar position and the extraterrestrial flux, for a series of stamps at one coordinate.

**These are primitives rather than a column set.** The studies want different columns from the same
geometry — one wants the azimuth to fit a panel's orientation, another wants the hour-mean cosine to
establish what an hourly irradiance column holds — so each caller composes what it needs. Promoting
a merged column set instead would add every caller's columns to every caller's output, and the
hour-mean cosine alone costs 60 solar-position evaluations per stamp.
"""

from typing import Final

import numpy as np
import polars as pl
import pvlib

SOLAR_CONSTANT_W_M2: Final[float] = 1361.0
"""The solar irradiance at the top of the atmosphere at one astronomical unit."""

COS_ZENITH_SUBSAMPLES: Final[int] = 60
"""How many samples the mean cosine of the solar zenith angle over an hour is taken from.

One sample a minute. The quantity is smooth in time except at sunrise and sunset, where the clip at
zero puts a corner in it, so the error a minute's spacing leaves is far below the 1 W m⁻² rounding
of a served irradiance column.
"""


def zenith(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the apparent solar zenith angle in degrees at each stamp.

    Args:
        stamps: The instants to evaluate at, as a UTC datetime series.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One angle per stamp, in degrees from the vertical.
    """
    position = pvlib.solarposition.get_solarposition(
        time=stamps.to_numpy(), latitude=latitude, longitude=longitude
    )
    return position["apparent_zenith"].to_numpy().astype(np.float64)


def azimuth(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the solar azimuth in degrees clockwise from north at each stamp.

    Args:
        stamps: The instants to evaluate at, as a UTC datetime series.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One angle per stamp, in degrees clockwise from north.
    """
    position = pvlib.solarposition.get_solarposition(
        time=stamps.to_numpy(), latitude=latitude, longitude=longitude
    )
    return position["azimuth"].to_numpy().astype(np.float64)


def cos_zenith(*, zenith_deg: np.ndarray) -> np.ndarray:
    """Return the cosine of the solar zenith angle, clipped at zero below the horizon.

    Args:
        zenith_deg: Zenith angles in degrees.

    Returns:
        The cosine, never negative.
    """
    return np.clip(np.cos(np.radians(zenith_deg)), 0.0, None)


def cos_zenith_hour_mean(*, stamps: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the mean cosine of the solar zenith angle over the hour *ending* at each stamp.

    Args:
        stamps: The instants each hour ends at, as a UTC datetime series.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One mean per stamp.
    """
    minutes = np.arange(COS_ZENITH_SUBSAMPLES) + 0.5 - COS_ZENITH_SUBSAMPLES
    samples = [
        cos_zenith(
            zenith_deg=zenith(
                stamps=stamps.dt.offset_by(f"{int(offset)}s"),
                latitude=latitude,
                longitude=longitude,
            )
        )
        for offset in np.round(minutes * 60.0)
    ]
    return np.mean(samples, axis=0)


def extraterrestrial_horizontal(*, stamps: pl.Series, zenith_deg: np.ndarray) -> np.ndarray:
    """Return the flux onto a horizontal plane at the top of the atmosphere.

    The denominator of the clearness index, so a value of zero means the sun is down and the
    clearness index is undefined rather than zero.

    Args:
        stamps: The instants the fluxes belong to, used for the Earth-Sun distance.
        zenith_deg: The solar zenith angle at each stamp, in degrees.

    Returns:
        One flux per stamp, in W m⁻², never negative.
    """
    normal = np.asarray(
        pvlib.irradiance.get_extra_radiation(
            datetime_or_doy=stamps.dt.ordinal_day().to_numpy(),
            solar_constant=SOLAR_CONSTANT_W_M2,
        )
    )
    return normal * cos_zenith(zenith_deg=zenith_deg)
