"""The product profiles and the field table of the CEDA UKV archive.

`studies/weather_downloads/fetch_ukv_ceda.py` downloads the archive and
`studies/nwp_forecast_comparison/build_ukv_ceda_inputs.py` builds the study's inputs from it, so
both take the two profiles, the run cycle, and the field table from here.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

PRODUCT_NAME: Final[str] = "UKV-CEDA"
SLOT_EPOCH: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
"""The first slot of the `init_time` axis. A run's slot is its offset from here in whole cycles."""


CYCLE_HOURS: Final[int] = 6
RUN_HOURS: Final[tuple[int, ...]] = (0, 6, 12, 18)
"""The runs the default profile archives. CEDA holds eight runs a day, and `T120` files only for
03 and 15 UTC, which the `ukv-ceda-t120` profile archives."""


MAX_STEP_HOURS: Final[int] = 54
PLAIN_LAST_STEP: Final[int] = 36


HOURLY_LAST_STEP: Final[int] = 48
T54_STEPS: Final[tuple[int, ...]] = (*range(37, HOURLY_LAST_STEP + 1), 51, 54)
"""Leads in a `T54` file: hourly to 48 hours, then 3-hourly."""


T120_STEPS: Final[tuple[int, ...]] = tuple(range(57, 121, 3))
"""Leads in a `T120` file: 3-hourly, 57 to 120 hours."""


@dataclass(frozen=True)
class Profile:
    """One archived product: which runs it keeps, how its slots are spaced, and how far it reaches.

    Attributes:
        product_name: The product's name, which is also its default store directory name.
        slot_epoch: The initialisation time of slot 0 of the `init_time` axis.
        cycle_hours: The hours between slots.
        run_hours: The UTC hours of day at which the archived runs start.
        max_step_hours: The longest lead archived, in hours.
        has_t120: Whether each run also has the three `T120` files.
    """

    product_name: str
    slot_epoch: datetime
    cycle_hours: int
    run_hours: tuple[int, ...]
    max_step_hours: int
    has_t120: bool

    @property
    def n_steps(self) -> int:
        """The length of the `step` axis."""
        return self.max_step_hours + 1

    @property
    def file_tags(self) -> tuple[str, ...]:
        """The files a run needs: plain files, then `T54` files, then `T120` files."""
        return file_tags(has_t120=self.has_t120)


DEFAULT_PROFILE: Final[Profile] = Profile(
    product_name=PRODUCT_NAME,
    slot_epoch=SLOT_EPOCH,
    cycle_hours=CYCLE_HOURS,
    run_hours=RUN_HOURS,
    max_step_hours=MAX_STEP_HOURS,
    has_t120=False,
)


T120_PROFILE: Final[Profile] = Profile(
    product_name="UKV-CEDA-T120",
    slot_epoch=datetime(2019, 9, 1, 3, tzinfo=UTC),
    cycle_hours=12,
    run_hours=(3, 15),
    max_step_hours=120,
    has_t120=True,
)


_active_profile: Profile = DEFAULT_PROFILE


def active_profile() -> Profile:
    """The profile that slots, steps, file tags, and documents are read from."""
    return _active_profile


def set_profile(profile: Profile) -> None:
    """Make `profile` the one every function in this module reads. `main` calls this once."""
    global _active_profile  # noqa: PLW0603
    _active_profile = profile


STATUS_COMPLETE: Final[int] = 1
STATUS_PARTIAL: Final[int] = 2


STATUS_MISSING: Final[int] = 3


@dataclass(frozen=True)
class FieldSpec:
    """One GRIB field kept in the store, identified by its GRIB2 parameter and level keys.

    Attributes:
        variable: The Zarr array name.
        wholesale: The `Wholesale` group (1 to 4) whose files carry the field.
        discipline: GRIB2 discipline.
        category: GRIB2 parameter category.
        number: GRIB2 parameter number.
        first_surface_type: GRIB2 `typeOfFirstFixedSurface`.
        first_surface_value: GRIB2 `scaledValueOfFirstFixedSurface` (a height in metres, or a
            pressure in pascals for `first_surface_type` 100).
        second_surface_type: GRIB2 `typeOfSecondFixedSurface`, 255 when absent.
        second_surface_value: GRIB2 `scaledValueOfSecondFixedSurface`, `NO_SURFACE` when absent.
        template: GRIB2 `productDefinitionTemplateNumber`: 0 is an instantaneous value, 8 a value
            over the interval since the previous step, and 5 a probability.
        units: The units as served.
        description: A one-line description for the README.
        invalid_below: Values below this are a "no value" flag and are stored as NaN, or `None`.
        process: GRIB2 `typeOfStatisticalProcessing` that an interval-valued field must carry
            (1 accumulation, 2 maximum), or `None` for an instantaneous field.
        lower_limit: The probability threshold, as `scaledValueOfLowerLimit`, that a probability
            field must carry, or `None`.
    """

    variable: str
    wholesale: int
    discipline: int
    category: int
    number: int
    first_surface_type: int
    first_surface_value: int
    second_surface_type: int
    second_surface_value: int
    template: int
    units: str
    description: str
    invalid_below: float | None = None
    process: int | None = None
    lower_limit: int | None = None

    @property
    def key(self) -> tuple[int, ...]:
        """The tuple of GRIB keys that identifies this field within a file."""
        return (
            self.discipline,
            self.category,
            self.number,
            self.first_surface_type,
            self.first_surface_value,
            self.second_surface_type,
            self.second_surface_value,
            self.template,
        )

    @property
    def interval_valued(self) -> bool:
        """Whether the field is over the interval since the previous step, so it has no lead 0."""
        return self.template == 8

    @property
    def tags(self) -> tuple[str, ...]:
        """The file tags of the active profile that hold the field."""
        return self.tags_for(has_t120=active_profile().has_t120)

    def tags_for(self, *, has_t120: bool) -> tuple[str, ...]:
        """The file tags that hold the field: the plain file, its `T54` file, and its `T120` file.

        Wholesale4 has only the plain file. A `T120` file counts only where `has_t120` is true.
        """
        plain = f"Wholesale{self.wholesale}"
        if self.wholesale == 4:
            return (plain,)
        if has_t120:
            return (plain, f"{plain}T54", f"{plain}T120")
        return (plain, f"{plain}T54")

    def expected_steps(self, *, tag: str) -> tuple[int, ...]:
        """The leads, in hours, that the file `tag` serves for this field."""
        if tag.endswith("T120"):
            return T120_STEPS
        if tag.endswith("T54"):
            return T54_STEPS
        first = 1 if self.interval_valued else 0
        return tuple(range(first, PLAIN_LAST_STEP + 1))


NO_SURFACE: Final[int] = 2147483647


def _field(
    variable: str,
    wholesale: int,
    parameter: tuple[int, int, int],
    surface: tuple[int, int],
    units: str,
    description: str,
    *,
    second: tuple[int, int] = (255, NO_SURFACE),
    template: int = 0,
    invalid_below: float | None = None,
    process: int | None = None,
    lower_limit: int | None = None,
) -> FieldSpec:
    """Build a `FieldSpec` from grouped keys, so that the table below stays one line per field."""
    return FieldSpec(
        variable=variable,
        wholesale=wholesale,
        discipline=parameter[0],
        category=parameter[1],
        number=parameter[2],
        first_surface_type=surface[0],
        first_surface_value=surface[1],
        second_surface_type=second[0],
        second_surface_value=second[1],
        template=template,
        units=units,
        description=description,
        invalid_below=invalid_below,
        process=process,
        lower_limit=lower_limit,
    )


_HEIGHT: Final[int] = 103
_GROUND: Final[int] = 1


_PRESSURE: Final[int] = 100
FIELDS: Final[tuple[FieldSpec, ...]] = (
    _field("temperature_1p5m", 1, (0, 0, 0), (_HEIGHT, 1), "K", "Screen-level temperature"),
    _field("temperature_0m", 1, (0, 0, 0), (_HEIGHT, 0), "K", "Temperature at height 0 m"),
    _field("dew_point_1p5m", 1, (0, 0, 6), (_HEIGHT, 1), "K", "Screen-level dew point"),
    _field("relative_humidity_1p5m", 1, (0, 1, 1), (_HEIGHT, 1), "%", "Screen-level humidity"),
    _field("visibility_1p5m", 1, (0, 19, 0), (_HEIGHT, 1), "m", "Screen-level visibility"),
    _field(
        "visibility_below_1km_probability",
        1,
        (0, 19, 0),
        (_HEIGHT, 1),
        "fraction",
        "Probability that visibility is below 1000 m",
        template=5,
        lower_limit=1000,
    ),
    _field("precipitation_rate", 1, (0, 1, 7), (_GROUND, 0), "kg m-2 s-1", "Precipitation rate"),
    _field(
        "precipitation_amount",
        1,
        (0, 1, 8),
        (_GROUND, 0),
        "kg m-2",
        "Precipitation accumulated since the previous served step",
        template=8,
        process=1,
    ),
    _field(
        "param_0_1_230",
        1,
        (0, 1, 230),
        (_GROUND, 0),
        "unknown",
        "Unidentified precipitation-category parameter, zero in the early leads of a dry run",
    ),
    _field("wind_speed_10m", 1, (0, 2, 1), (_HEIGHT, 10), "m s-1", "10 m wind speed"),
    _field("wind_direction_10m", 1, (0, 2, 0), (_HEIGHT, 10), "degrees", "10 m wind direction"),
    _field("pressure_msl", 1, (0, 3, 1), (101, 0), "Pa", "Pressure reduced to mean sea level"),
    _field("cloud_total", 2, (0, 6, 1), (10, 0), "%", "Total cloud cover, whole atmosphere"),
    _field(
        "cloud_low",
        2,
        (0, 6, 3),
        (_HEIGHT, 0),
        "%",
        "Low cloud cover, 0 to 1524 m",
        second=(_HEIGHT, 1524),
    ),
    _field(
        "cloud_very_low",
        2,
        (0, 6, 3),
        (_HEIGHT, 0),
        "%",
        "Very low cloud cover, 0 to 305 m",
        second=(_HEIGHT, 305),
    ),
    _field(
        "cloud_medium",
        2,
        (0, 6, 4),
        (_HEIGHT, 1524),
        "%",
        "Medium cloud cover, 1524 to 4572 m",
        second=(_HEIGHT, 4572),
    ),
    _field(
        "cloud_high",
        2,
        (0, 6, 5),
        (_HEIGHT, 4572),
        "%",
        "High cloud cover, 4572 to 30000 m",
        second=(_HEIGHT, 30000),
    ),
    _field("cloud_base_height", 2, (0, 6, 11), (2, 0), "m", "Height of the lowest cloud base"),
    _field(
        "cloud_param_0_6_26",
        2,
        (0, 6, 26),
        (_GROUND, 0),
        "unknown",
        "Unidentified cloud-category parameter, 2 to 1800 m where valid",
    ),
    _field(
        "convective_cloud_top_height",
        2,
        (0, 6, 27),
        (_GROUND, 0),
        "m",
        "Height of convective cloud top, NaN where no convective cloud",
        invalid_below=-32000.0,
    ),
    _field("snow_depth", 2, (0, 1, 11), (_GROUND, 0), "m", "Snow depth"),
    _field(
        "shortwave_down",
        2,
        (0, 4, 7),
        (_GROUND, 0),
        "W m-2",
        "Downward shortwave flux at the surface, instantaneous",
    ),
    _field(
        "longwave_down",
        2,
        (0, 5, 3),
        (_GROUND, 0),
        "W m-2",
        "Downward longwave flux at the surface, instantaneous",
    ),
    *(
        _field(
            f"{name}_{level // 100}hpa",
            3,
            parameter,
            (_PRESSURE, level),
            units,
            f"{label} at {level // 100} hPa, NaN where the level is below the ground",
        )
        for level in (100000, 92500)
        for name, parameter, units, label in (
            ("wind_speed", (0, 2, 1), "m s-1", "Wind speed"),
            ("wind_direction", (0, 2, 0), "degrees", "Wind direction"),
            ("geopotential_height", (0, 3, 5), "gpm", "Geopotential height"),
        )
    ),
    _field("gust_10m", 4, (0, 2, 22), (_HEIGHT, 10), "m s-1", "10 m wind gust, instantaneous"),
    _field(
        "gust_10m_max",
        4,
        (0, 2, 22),
        (_HEIGHT, 10),
        "m s-1",
        "Maximum 10 m wind gust since the previous step",
        template=8,
        process=2,
    ),
)


def _tag_rank(tag: str) -> tuple[int, str]:
    """Sort key that puts plain files first, then `T54` files, then `T120` files."""
    tier = 2 if tag.endswith("T120") else 1 if tag.endswith("T54") else 0
    return tier, tag


def file_tags(*, has_t120: bool) -> tuple[str, ...]:
    """Every file tag that holds a kept field, in `_tag_rank` order."""
    tags = {tag for spec in FIELDS for tag in spec.tags_for(has_t120=has_t120)}
    return tuple(sorted(tags, key=_tag_rank))
