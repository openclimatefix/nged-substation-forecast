"""A small fitted PV model, used as a second instrument beside XGBoost.

One-off throwaway module for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**XGBoost has to learn the transposition from data; this model is given it.** The physical route
from irradiance to power runs through the plane of array, and the plane-of-array irradiance is where
the beam and the diffuse components stop being interchangeable: the beam arrives from one direction
and is projected onto the panel by the cosine of its incidence angle, while the diffuse arrives from
the whole sky and is not. A gradient-boosted tree can only approximate that projection out of the
features it is shown, so a null result from the tree is ambiguous between "the split carries no
information" and "the tree could not use it". The physical model removes the second reading, because
the split enters it in the one place physics says it belongs.

Everything here runs on numpy arrays rather than a dataframe, because the fit evaluates the model a
few thousand times per fold.

The free parameters are per site, and the fit is repeated on every training fold, so no parameter
is ever fitted on the rows it is scored against.
"""

from typing import Final, NamedTuple

import numpy as np

GROUND_ALBEDO: Final[float] = 0.2
"""The fraction of global irradiance the ground reflects back towards the panel.

Grass and crops sit near this value, and the ground-reflected term is a few percent of the
plane-of-array total at these tilts, so fitting it would buy less than the extra parameter costs.
"""

MIN_COS_ZENITH: Final[float] = 0.05
"""The floor applied before dividing the beam's horizontal flux by it to get the direct normal one.

At a solar elevation under about 3 degrees the division is numerically unstable and the physics is
wrong anyway, because the beam is passing through so much atmosphere that the horizontal flux is
rounding noise. Flooring rather than dropping keeps every arm on identical rows.
"""

REFERENCE_IRRADIANCE_W_M2: Final[float] = 1000.0
"""Plane-of-array irradiance at standard test conditions, which the capacity term is defined at."""

CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
"""How much hotter than the air a module runs at `NOMINAL_IRRADIANCE_W_M2`.

Taken from the nominal-operating-cell-temperature convention rather than fitted: the temperature
term changes power by a few percent, and a free rise would trade off against the free efficiency
coefficient without either being identified.
"""

NOMINAL_IRRADIANCE_W_M2: Final[float] = 800.0
"""The irradiance `CELL_TEMPERATURE_RISE_K` is quoted at."""

REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
"""The cell temperature the capacity term is defined at."""


class Geometry(NamedTuple):
    """The sun's position and the irradiance fields one arm shows the model.

    Attributes:
        cos_zenith: Cosine of the solar zenith angle, floored at `MIN_COS_ZENITH`.
        sin_zenith: Sine of the solar zenith angle.
        solar_azimuth_rad: Solar azimuth, clockwise from north, in radians.
        global_horizontal: Global horizontal irradiance in W m⁻².
        beam_horizontal: The arm's beam-on-horizontal irradiance in W m⁻², or `None` for the arm
            that is shown no split at all.
        diffuse_horizontal: The arm's diffuse horizontal irradiance in W m⁻², or `None`.
        air_temperature_c: Air temperature at 2 m in degrees Celsius.
    """

    cos_zenith: np.ndarray
    sin_zenith: np.ndarray
    solar_azimuth_rad: np.ndarray
    global_horizontal: np.ndarray
    beam_horizontal: np.ndarray | None
    diffuse_horizontal: np.ndarray | None
    air_temperature_c: np.ndarray


def plane_of_array(*, geometry: Geometry, tilt_rad: float, azimuth_rad: float) -> np.ndarray:
    """Return the irradiance falling on a panel at one tilt and azimuth.

    The beam is projected by the cosine of its incidence angle, the sky diffuse is taken as
    isotropic, and the ground reflection is the isotropic complement. An arm with no split gets the
    global horizontal irradiance unchanged, because with one number there is no transposition to do
    and any tilt factor is a constant the capacity term absorbs.

    Args:
        geometry: The sun's position and the arm's irradiance fields.
        tilt_rad: Panel tilt from horizontal, in radians.
        azimuth_rad: Panel azimuth, clockwise from north, in radians.

    Returns:
        Plane-of-array irradiance in W m⁻², one value per row.
    """
    if geometry.beam_horizontal is None or geometry.diffuse_horizontal is None:
        return geometry.global_horizontal

    cos_tilt = np.cos(tilt_rad)
    cos_incidence = geometry.cos_zenith * cos_tilt + geometry.sin_zenith * np.sin(
        tilt_rad
    ) * np.cos(geometry.solar_azimuth_rad - azimuth_rad)
    direct_normal = geometry.beam_horizontal / geometry.cos_zenith
    beam = direct_normal * np.clip(cos_incidence, 0.0, None)
    sky = geometry.diffuse_horizontal * (1.0 + cos_tilt) / 2.0
    ground = geometry.global_horizontal * GROUND_ALBEDO * (1.0 - cos_tilt) / 2.0
    return beam + sky + ground


def power_mw(
    *,
    geometry: Geometry,
    tilt_rad: float,
    azimuth_rad: float,
    capacity_mw: float,
    temperature_coefficient: float,
    clip_mw: float,
) -> np.ndarray:
    """Return the modelled power output in MW.

    Args:
        geometry: The sun's position and the arm's irradiance fields.
        tilt_rad: Panel tilt from horizontal, in radians.
        azimuth_rad: Panel azimuth, clockwise from north, in radians.
        capacity_mw: Output at `REFERENCE_IRRADIANCE_W_M2` and `REFERENCE_CELL_TEMPERATURE_C`.
        temperature_coefficient: Fractional change in output per degree above reference.
        clip_mw: The inverter's own ceiling.

    Returns:
        Modelled power in MW, one value per row.
    """
    irradiance = plane_of_array(geometry=geometry, tilt_rad=tilt_rad, azimuth_rad=azimuth_rad)
    cell_temperature = (
        geometry.air_temperature_c + irradiance * CELL_TEMPERATURE_RISE_K / NOMINAL_IRRADIANCE_W_M2
    )
    efficiency = 1.0 + temperature_coefficient * (cell_temperature - REFERENCE_CELL_TEMPERATURE_C)
    modelled = capacity_mw * irradiance / REFERENCE_IRRADIANCE_W_M2 * efficiency
    return np.clip(modelled, 0.0, clip_mw)
