"""A forward model of a photovoltaic (PV) plant, driven by hourly global horizontal irradiance.

The model turns one hourly series of global horizontal irradiance (GHI) into the half-hourly AC
power of a plant with a given tilt, azimuth, DC capacity, and AC capacity. The chain follows
[Differentiable physics](https://openclimatefix.github.io/nged-substation-forecast/techniques/differentiable-physics/):
the hour's GHI is shared between its two half-hours by the cosine of the solar zenith, the Erbs
correlation splits each half-hour's GHI into direct and diffuse parts, an isotropic-sky model puts
the three parts on the tilted plane, and the inverters clip the DC power at the AC capacity.

Azimuths follow pvlib: degrees clockwise from north, so south is 180, east is 90, and west is 270.
Everything here is plain NumPy so that a fit can use SciPy's optimisers.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import pvlib

STANDARD_IRRADIANCE_W_M2: Final[float] = 1000.0
"""The plane-of-array irradiance at which a plant's DC capacity is quoted."""
GROUND_ALBEDO: Final[float] = 0.2
"""The ground's broadband reflectance, applied to the light that reaches the tilted plane."""
MAX_TRACKER_ROTATION_DEG: Final[float] = 60.0
"""The largest rotation of a single-axis tracker's surface from the horizontal."""
MIN_ELEVATION_DEG: Final[float] = 5.0
"""A half-hour with the sun lower than this is treated as night."""


@dataclass(frozen=True)
class SunAndSky:
    """The sun's position and the sky's irradiance at each half-hour of one place.

    Every array has one entry per half-hour, in the order of `half_hour_end_time`.
    """

    half_hour_end_time: np.ndarray
    ghi_w_m2: np.ndarray
    dni_w_m2: np.ndarray
    dhi_w_m2: np.ndarray
    zenith_deg: np.ndarray
    sun_azimuth_deg: np.ndarray


@dataclass(frozen=True)
class PlantParameters:
    """The physical parameters of one plant, or of one homogeneous part of a fleet.

    Attributes:
        tilt_deg: The panels' tilt from the horizontal. Unused when `tracker` is true.
        azimuth_deg: The panels' azimuth, clockwise from north. Unused when `tracker` is true.
        dc_capacity_mw: The DC capacity: output at 1000 W/m² of plane-of-array irradiance.
        ac_capacity_mw: The most the inverters can export.
        tracker: True for a single-axis tracker on a north-south axis, without backtracking.
    """

    tilt_deg: float
    azimuth_deg: float
    dc_capacity_mw: float
    ac_capacity_mw: float
    tracker: bool = False

    @property
    def dc_ac_ratio(self) -> float:
        """The DC capacity divided by the AC capacity."""
        return self.dc_capacity_mw / self.ac_capacity_mw


def solar_position(
    *, stamps: pl.Series, latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the apparent zenith and the azimuth of the sun at each stamp.

    Args:
        stamps: The instants to evaluate at, as a UTC datetime series.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The zenith in degrees from the vertical, and the azimuth in degrees clockwise from north.
    """
    position = pvlib.solarposition.get_solarposition(
        time=stamps.to_numpy(), latitude=latitude, longitude=longitude
    )
    return (
        position["apparent_zenith"].to_numpy().astype(np.float64),
        position["azimuth"].to_numpy().astype(np.float64),
    )


def half_hour_sun_and_sky(*, hourly: pl.DataFrame, latitude: float, longitude: float) -> SunAndSky:
    """Spread hourly GHI over half-hours and split it into direct and diffuse parts.

    An hourly value labelled `T` is the mean over the hour that ends at `T`. Its two half-hours end
    at `T - 30 min` and at `T`. Each half-hour receives the hour's GHI times its own cosine of the
    zenith over the mean of the two cosines, so the two half-hours average back to the hour's GHI
    and the sun's rise and fall inside the hour is kept.

    Args:
        hourly: Columns `time` (UTC, the end of the hour) and `ghi_w_m2`, one row per hour.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The two half-hours of every hour in `hourly`, sorted by time.
    """
    ordered = hourly.sort("time")
    hour_ends = ordered["time"]
    first_halves = hour_ends.dt.offset_by("-30m")
    stamps = pl.concat([first_halves, hour_ends]).sort()
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    cos_zenith = np.clip(np.cos(np.radians(zenith)), 0.0, None)
    # `stamps` interleaves the two half-hours of each hour: positions 2k and 2k + 1.
    hour_mean_cos = np.repeat(cos_zenith.reshape(-1, 2).mean(axis=1), 2)
    hour_ghi = np.repeat(ordered["ghi_w_m2"].to_numpy().astype(np.float64), 2)
    share = np.divide(
        cos_zenith, hour_mean_cos, out=np.zeros_like(cos_zenith), where=hour_mean_cos > 0
    )
    ghi = np.clip(hour_ghi * share, 0.0, None)
    split = pvlib.irradiance.erbs(
        ghi=ghi,
        zenith=zenith,
        datetime_or_doy=pl.Series(midpoints).dt.ordinal_day().to_numpy(),
    )
    dni = np.nan_to_num(np.asarray(split["dni"]).astype(np.float64))
    dhi = np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64))
    return SunAndSky(
        half_hour_end_time=stamps.to_numpy(),
        ghi_w_m2=ghi,
        dni_w_m2=dni,
        dhi_w_m2=dhi,
        zenith_deg=zenith,
        sun_azimuth_deg=sun_azimuth,
    )


def plane_of_array_w_m2(
    *, sky: SunAndSky, tilt_deg: float, azimuth_deg: float, tracker: bool
) -> np.ndarray:
    """Return the irradiance on the plant's plane at each half-hour.

    The sum of the direct beam projected onto the plane, the isotropic sky's diffuse light, and the
    light the ground reflects. A tracker rotates about a horizontal north-south axis to face the
    sun, up to `MAX_TRACKER_ROTATION_DEG`, and ignores backtracking.

    Args:
        sky: The sun and sky.
        tilt_deg: The panels' tilt. Ignored for a tracker.
        azimuth_deg: The panels' azimuth, clockwise from north. Ignored for a tracker.
        tracker: True for a single-axis tracker.

    Returns:
        The plane-of-array irradiance in watts per square metre, never negative.
    """
    zenith = np.radians(sky.zenith_deg)
    sun_azimuth = np.radians(sky.sun_azimuth_deg)
    if tracker:
        east = np.sin(zenith) * np.sin(sun_azimuth)
        up = np.cos(zenith)
        rotation = np.clip(
            np.arctan2(east, np.maximum(up, 1e-9)),
            -np.radians(MAX_TRACKER_ROTATION_DEG),
            np.radians(MAX_TRACKER_ROTATION_DEG),
        )
        # The surface normal lies in the east-up plane at `rotation` from the vertical.
        cos_incidence = east * np.sin(rotation) + up * np.cos(rotation)
        cos_tilt = np.cos(rotation)
    else:
        tilt = np.radians(tilt_deg)
        panel_azimuth = np.radians(azimuth_deg)
        cos_incidence = np.cos(zenith) * np.cos(tilt) + np.sin(zenith) * np.sin(tilt) * np.cos(
            sun_azimuth - panel_azimuth
        )
        cos_tilt = np.cos(tilt)
    beam = sky.dni_w_m2 * np.clip(cos_incidence, 0.0, None)
    sky_diffuse = sky.dhi_w_m2 * (1.0 + cos_tilt) / 2.0
    ground = sky.ghi_w_m2 * GROUND_ALBEDO * (1.0 - cos_tilt) / 2.0
    return np.clip(beam + sky_diffuse + ground, 0.0, None)


def ac_power_mw(
    *, poa_w_m2: np.ndarray, dc_capacity_mw: float, ac_capacity_mw: float
) -> np.ndarray:
    """Return the plant's AC power: DC power from the plane's irradiance, clipped at AC capacity.

    Args:
        poa_w_m2: The plane-of-array irradiance.
        dc_capacity_mw: The DC capacity.
        ac_capacity_mw: The inverters' AC limit.

    Returns:
        The power in megawatts.
    """
    return np.minimum(dc_capacity_mw * poa_w_m2 / STANDARD_IRRADIANCE_W_M2, ac_capacity_mw)


def plant_power_mw(*, sky: SunAndSky, parameters: PlantParameters) -> np.ndarray:
    """Return the plant's AC power at each half-hour.

    Args:
        sky: The sun and sky.
        parameters: The plant.

    Returns:
        The power in megawatts.
    """
    poa = plane_of_array_w_m2(
        sky=sky,
        tilt_deg=parameters.tilt_deg,
        azimuth_deg=parameters.azimuth_deg,
        tracker=parameters.tracker,
    )
    return ac_power_mw(
        poa_w_m2=poa,
        dc_capacity_mw=parameters.dc_capacity_mw,
        ac_capacity_mw=parameters.ac_capacity_mw,
    )


def daylight(*, sky: SunAndSky) -> np.ndarray:
    """Return a mask of the half-hours with the sun above `MIN_ELEVATION_DEG`.

    Args:
        sky: The sun and sky.

    Returns:
        True where the sun is up.
    """
    return (90.0 - sky.zenith_deg) > MIN_ELEVATION_DEG
