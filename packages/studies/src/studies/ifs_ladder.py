"""The ladder of IFS variable groups, and the arms built from it.

Written for the IFS solar-variables study planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1137>. Scripts in
`studies/ifs_solar_variables/` import it.

**The ladder is seven groups of variables ("rungs"), each adding one physical idea to every rung
below it.** `RUNG_ADDITIONS` is the single copy of the list. `rung_features`,
`drop_one_group_features`, and the control functions all read it, so the ladder, the drop-one-group
arms, and the negative controls cannot disagree about which variable belongs to which group.

**Every negative control has the columns of the arm it pads, with the new variables replaced by
permuted copies of themselves**, so a control and the arm it is read against differ in information
and never in the number of columns.
"""

from typing import Final, Literal

from studies.blending import PERMUTED_SUFFIX
from studies.era5_ladder import SHARED_FEATURES as ERA5_SHARED_FEATURES
from studies.era5_ladder import RungType as Era5RungType
from studies.era5_ladder import rung_variables as era5_rung_variables
from studies.ifs_lead_days import PARTNER_COLUMNS

RungType = Literal["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
"""The IFS rungs."""

RUNGS: Final[tuple[RungType, ...]] = ("f0", "f1", "f2", "f3", "f4", "f5", "f6")
"""The rungs from the smallest set to the largest, in the order each adds to the last."""

RUNG_ADDITIONS: Final[dict[RungType, tuple[str, ...]]] = {
    "f0": ("shortwave_radiation", "temperature_2m"),
    "f1": ("cloud_cover",),
    "f2": ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"),
    "f3": ("direct_radiation",),
    "f4": ("dew_point_2m", "total_column_integrated_water_vapour", "boundary_layer_height"),
    "f5": ("cape", "convective_inhibition", "visibility"),
    "f6": (
        "surface_temperature",
        "snow_depth",
        "snowfall",
        "precipitation",
        "surface_pressure",
        "wind_speed_10m",
        "wind_gusts_10m",
    ),
}
"""The IFS variables each rung adds, by the names Open-Meteo serves them under."""

MISSING_BY_DESIGN: Final[tuple[str, ...]] = ("convective_inhibition",)
"""The variables Open-Meteo leaves missing where they are undefined; XGBoost reads that natively."""

SHARED_FEATURES: Final[tuple[str, ...]] = (*ERA5_SHARED_FEATURES, "era_code")
"""The solar geometry, calendar, and IFS-era columns that every arm is shown.

The era column tells the XGBoost model which side of the IFS cycle change a row falls on.
"""

PRODUCTION_VARIABLES: Final[tuple[str, ...]] = (
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
)
"""The IFS variables the production ensemble feed carries beyond radiation and temperature."""

ERA5_PREFIX: Final[str] = "era5_"
"""What the built frame prepends to an ERA5 variable's name, so that `cape` does not clash."""


def rung_variables(*, rung: RungType) -> tuple[str, ...]:
    """Return every IFS variable a rung is shown, which is the variables of every rung up to it."""
    last = RUNGS.index(rung)
    return tuple(name for step in RUNGS[: last + 1] for name in RUNG_ADDITIONS[step])


def rung_features(*, rung: RungType) -> tuple[str, ...]:
    """Return every feature column a rung is shown: the shared columns, then its variables."""
    return (*SHARED_FEATURES, *rung_variables(rung=rung))


def production_features() -> tuple[str, ...]:
    """Return the columns of the production reference arm: the minimal set and four variables."""
    return (*rung_features(rung="f0"), *PRODUCTION_VARIABLES)


def with_partner_features(*, rung: RungType) -> tuple[str, ...]:
    """Return a rung's columns and the second forecast's radiation and temperature."""
    return (*rung_features(rung=rung), *PARTNER_COLUMNS)


def drop_one_group_features(*, dropped: RungType) -> tuple[str, ...]:
    """Return the full set's columns with one rung's own variables removed.

    Args:
        dropped: The rung to remove. It cannot be `f0`, the base every arm keeps.

    Returns:
        The columns of `f6` less the dropped rung's variables.

    Raises:
        ValueError: If `dropped` is `f0`.
    """
    if dropped == "f0":
        msg = "f0 is the base every arm keeps, so it cannot be dropped"
        raise ValueError(msg)
    removed = set(RUNG_ADDITIONS[dropped])
    return tuple(name for name in rung_features(rung=RUNGS[-1]) if name not in removed)


def later_groups_columns() -> tuple[str, ...]:
    """Return the variables that `f3` to `f6` add, the columns the `f2` control permutes."""
    return tuple(name for rung in RUNGS[RUNGS.index("f3") :] for name in RUNG_ADDITIONS[rung])


def permuted(*, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the names of the permuted copies of columns."""
    return tuple(f"{name}{PERMUTED_SUFFIX}" for name in columns)


def later_groups_control_features() -> tuple[str, ...]:
    """Return `f2` and permuted copies of every later variable: as many columns as `f6`."""
    return (*rung_features(rung="f2"), *permuted(columns=later_groups_columns()))


def cloud_cover_control_features() -> tuple[str, ...]:
    """Return `f0` and a permuted copy of total cloud cover: as many columns as `f1`."""
    return (*rung_features(rung="f0"), *permuted(columns=RUNG_ADDITIONS["f1"]))


def cloud_layers_control_features() -> tuple[str, ...]:
    """Return `f1` and permuted copies of the three cloud layers: as many columns as `f2`."""
    return (*rung_features(rung="f1"), *permuted(columns=RUNG_ADDITIONS["f2"]))


def partner_control_features(*, rung: RungType) -> tuple[str, ...]:
    """Return a rung and permuted copies of the partner's two variables."""
    return (*rung_features(rung=rung), *permuted(columns=PARTNER_COLUMNS))


def positive_control_features() -> tuple[str, ...]:
    """Return `f2` and CAMS global irradiance at the valid hour, a column that must help."""
    return (*rung_features(rung="f2"), "cams_ghi_w_m2")


def era5_comparison_features(*, rung: Era5RungType) -> tuple[str, ...]:
    """Return the columns of an ERA5 arm refitted on the IFS rows.

    The arm gets the IFS study's shared columns and the rung's ERA5 variables, without ERA5's
    derived clearness index, so the two studies' arms differ in the variables alone.

    Args:
        rung: The ERA5 rung.

    Returns:
        The shared columns and the prefixed ERA5 variables.
    """
    return (
        *SHARED_FEATURES,
        *(f"{ERA5_PREFIX}{name}" for name in era5_rung_variables(rung=rung)),
    )


OpenDataType = Literal["carried", "not carried", "unchecked"]
"""Whether ECMWF's free open-data feed carries a variable."""

OPEN_DATA_AVAILABILITY: Final[dict[str, OpenDataType]] = {
    "shortwave_radiation": "carried",
    "temperature_2m": "carried",
    "cloud_cover": "carried",
    "cloud_cover_low": "not carried",
    "cloud_cover_mid": "not carried",
    "cloud_cover_high": "not carried",
    "direct_radiation": "not carried",
    "dew_point_2m": "carried",
    "total_column_integrated_water_vapour": "carried",
    "boundary_layer_height": "not carried",
    "cape": "carried",
    "convective_inhibition": "unchecked",
    "visibility": "unchecked",
    "surface_temperature": "carried",
    "snow_depth": "unchecked",
    "snowfall": "unchecked",
    "precipitation": "carried",
    "surface_pressure": "carried",
    "wind_speed_10m": "carried",
    "wind_gusts_10m": "unchecked",
}
"""Whether the open-data feed carries each variable, as the plan states it.

The plan records that the feed carries total column water vapour, CAPE, skin temperature, and total
cloud cover, and carries no cloud layers, direct radiation, or boundary-layer height. The production
ensemble feed's variables are carried by definition. The plan does not state the rest, so they are
`unchecked`. The plan's statement dates from 2026-10-09 and asks for a recheck on a recent run
before a page assigns classes.
"""

OPEN_DATA_CHECK_NOTE: Final[str] = (
    "Open-data availability is the plan's statement of 2026-10-09; variables the plan does not "
    "state are marked unchecked, and the availability must be rechecked on a recent run before "
    "the page assigns classes."
)
"""The sentence the report prints beside the availability table."""

GROUP_NAMES: Final[dict[RungType, str]] = {
    "f1": "total cloud",
    "f2": "cloud layers",
    "f3": "direct radiation",
    "f4": "humidity, column water, boundary-layer height",
    "f5": "convection and visibility",
    "f6": "surface and weather state",
}
"""What each rung adds, in words, for the report and the charts."""

PRODUCTION_FEED_VARIABLES: Final[frozenset[str]] = frozenset(
    {"shortwave_radiation", "temperature_2m", *PRODUCTION_VARIABLES}
)
"""The IFS variables the production ensemble feed already carries."""
