"""The ladder of ERA5 variable groups, and the small tested pieces the ladder study is built from.

Written for the study of which ERA5 variables help predict solar power, planned in
<https://github.com/openclimatefix/nged-substation-forecast/pull/1097>. Scripts in
`studies/era5_solar_variables/` import it.

**The ladder is eleven groups of variables ("rungs"), each adding one physical idea to every rung
below it.** `RUNG_ADDITIONS` is the single copy of the list, `rung_variables` and `rung_features`
read it, and `drop_one_group_features` and `negative_control_columns` derive the two control arms
from it, so the ladder, the drop-one-group arms, and the negative control cannot disagree about
which variable belongs to which group.

**Every ERA5 variable falls into exactly one of two hour conventions.** The accumulations are totals
over the hour ending at the label, and are converted to hourly rates here. Every other variable is a
snapshot at the label, and is averaged over the labels one hour earlier and at the label by
`studies.hourly_means.hourly_from_snapshots`, so the value describes the same hour as the power.
"""

from collections.abc import Sequence
from typing import Final, Literal

import polars as pl

RungType = Literal["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]
"""The ERA5 rungs. The aerosol rung (G10) is not an ERA5 rung and is handled on its own."""

RUNGS: Final[tuple[RungType, ...]] = ("g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9")
"""The rungs from the smallest set to the largest, in the order each adds to the last."""

RUNG_ADDITIONS: Final[dict[RungType, tuple[str, ...]]] = {
    "g0": ("ssrd", "t2m"),
    "g1": ("tcc",),
    "g2": ("lcc", "mcc", "hcc"),
    "g3": ("ssrdc",),
    "g4": ("tclw", "tciw", "tcslw", "cbh"),
    "g5": ("fdir", "cdir"),
    "g6": ("u10", "v10", "strd"),
    "g7": ("d2m", "tcwv", "blh"),
    "g8": ("sd", "sf", "asn", "fal"),
    "g9": ("tp", "tcrw", "tcsw", "cape", "cin", "skt", "tco3", "uvb", "i10fg", "sp", "deg0l"),
}
"""The ERA5 variables each rung adds, by ERA5 short name.

The names are the columns the built dataset carries. Radiation and precipitation accumulations hold
hourly rates (W m⁻² and mm), and every other variable holds the unit ERA5 documents for it, except
that the held copy's `t2m` is in degrees Celsius.
"""

DERIVED_BY_RUNG: Final[dict[RungType, tuple[str, ...]]] = {
    "g0": ("clearness_index",),
    "g3": ("clear_sky_index",),
}
"""The features computed from a rung's own variables, which travel with the rung.

The clearness index is `ssrd` over the top-of-atmosphere flux, and the clear-sky index is `ssrd`
over `ssrdc`. A rung that is dropped takes its derived feature with it.
"""

SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "hour_of_day",
    "day_of_year",
)
"""The solar geometry and calendar features every arm is shown, beside its rung's variables."""

ACCUMULATED_VARIABLES: Final[tuple[str, ...]] = (
    "ssrd",
    "ssrdc",
    "fdir",
    "cdir",
    "strd",
    "tp",
    "sf",
    "uvb",
)
"""The ladder variables that are totals over the hour ending at the label."""

INSTANTANEOUS_VARIABLES: Final[tuple[str, ...]] = (
    "t2m",
    "tcc",
    "lcc",
    "mcc",
    "hcc",
    "tclw",
    "tciw",
    "tcslw",
    "cbh",
    "u10",
    "v10",
    "d2m",
    "tcwv",
    "blh",
    "sd",
    "asn",
    "fal",
    "tcrw",
    "tcsw",
    "cape",
    "cin",
    "skt",
    "tco3",
    "i10fg",
    "sp",
    "deg0l",
)
"""The ladder variables that are snapshots at the label."""

MISSING_UNDER_CLEAR_SKY_VARIABLES: Final[tuple[str, ...]] = ("cbh", "cin")
"""The variables whose missing values may be information, so the build keeps them as missing.

ERA5 sets the cloud base height missing where there is no cloud, and can do the same for convective
inhibition. XGBoost treats a missing value as a value of its own. Every other ladder variable is
complete, and the build raises if one is not.
"""

AEROSOL_COLUMNS: Final[tuple[str, ...]] = ("aod550", "duaod550")
"""The CAMS EAC4 aerosol optical depths at 550 nm (total and dust) that the aerosol rung adds."""

MIN_EXTRATERRESTRIAL_W_M2: Final[float] = 50.0
"""A row is a daylight row only if the top-of-atmosphere horizontal flux exceeds this.

Below it the clearness index is unstable, because a small error in the denominator becomes a large
error in the ratio. The threshold depends on the clock and the place only, never on a weather value.
"""

SECONDS_PER_HOUR: Final[float] = 3600.0
"""Seconds in the hour an accumulation covers, which turns joules per square metre into watts."""

MILLIMETRES_PER_METRE: Final[float] = 1000.0
"""Turns an accumulation of metres of water into millimetres."""

CLEAR_SKY_INDEX_THRESHOLDS: Final[tuple[float, float]] = (0.4, 0.8)
"""The CAMS clear-sky index below which an hour is overcast, and from which it is clear.

Fixed before any result existed. Between the two, an hour is broken cloud.
"""

TOTAL_CLOUD_COVER_THRESHOLDS: Final[tuple[float, float]] = (0.2, 0.8)
"""The ERA5 total cloud cover below which an hour is clear, and from which it is overcast.

Fixed before any result existed. Between the two, an hour is broken cloud.
"""

SkyRegimeType = Literal["clear", "broken", "overcast"]
"""The three weather regimes the regime figures split by."""

SKY_REGIMES: Final[tuple[SkyRegimeType, ...]] = ("clear", "broken", "overcast")
"""The regimes from the brightest to the dullest."""

SeasonType = Literal["winter", "spring", "summer", "autumn"]
"""The four meteorological seasons."""

SEASONS: Final[tuple[SeasonType, ...]] = ("winter", "spring", "summer", "autumn")
"""The seasons in calendar order, starting from the December that opens the winter."""

AEROSOL_SAMPLE_OFFSETS_MINUTES: Final[tuple[int, int]] = (-60, 0)
"""The instants, relative to an hour's label, at which the aerosol series is read and averaged."""

AEROSOL_STEP_HOURS: Final[int] = 3
"""The spacing of the EAC4 analysis times (00, 03, ..., 21 UTC)."""


def rung_variables(*, rung: RungType) -> tuple[str, ...]:
    """Return every variable a rung is shown, which is the variables of every rung up to it.

    Args:
        rung: The rung.

    Returns:
        The ERA5 variables, in the order the rungs add them.
    """
    last = RUNGS.index(rung)
    return tuple(name for step in RUNGS[: last + 1] for name in RUNG_ADDITIONS[step])


def rung_derived_features(*, rung: RungType) -> tuple[str, ...]:
    """Return the derived features of a rung and every rung below it.

    Args:
        rung: The rung.

    Returns:
        The derived feature names, in the order the rungs add them.
    """
    last = RUNGS.index(rung)
    return tuple(name for step in RUNGS[: last + 1] for name in DERIVED_BY_RUNG.get(step, ()))


def rung_features(*, rung: RungType) -> tuple[str, ...]:
    """Return every feature column a rung is shown.

    Args:
        rung: The rung.

    Returns:
        The shared geometry and calendar features, the rung's variables, and its derived features.
    """
    return (*SHARED_FEATURES, *rung_variables(rung=rung), *rung_derived_features(rung=rung))


def _rung_own_columns(*, rung: RungType) -> tuple[str, ...]:
    """Return the variables and derived features that a rung adds on top of the rung below."""
    return (*RUNG_ADDITIONS[rung], *DERIVED_BY_RUNG.get(rung, ()))


def drop_one_group_features(*, dropped: RungType) -> tuple[str, ...]:
    """Return the features of the full set with one rung's own variables removed.

    Args:
        dropped: The rung to remove. It cannot be `g0`, because the minimal set is the base that
            every arm keeps.

    Returns:
        The features of `g9` less the dropped rung's variables and derived features.

    Raises:
        ValueError: If `dropped` is `g0`.
    """
    if dropped == "g0":
        msg = "g0 is the base every arm keeps, so it cannot be dropped"
        raise ValueError(msg)
    removed = set(_rung_own_columns(rung=dropped))
    return tuple(name for name in rung_features(rung=RUNGS[-1]) if name not in removed)


def negative_control_columns() -> tuple[str, ...]:
    """Return the columns the negative control permutes: everything `g3` to `g9` adds.

    Returns:
        The variables and derived features of the rungs above `g2`, in ladder order.
    """
    return tuple(
        name for rung in RUNGS[RUNGS.index("g3") :] for name in _rung_own_columns(rung=rung)
    )


def negative_control_features(*, suffix: str) -> tuple[str, ...]:
    """Return the negative control's features: `g2`, then a permuted copy of every later column.

    The control has as many columns as `g9`.

    Args:
        suffix: What the permutation appends to a column's name, as
            `studies.blending.PERMUTED_SUFFIX`.

    Returns:
        The `g2` features followed by the permuted column names.
    """
    return (
        *rung_features(rung="g2"),
        *(f"{name}{suffix}" for name in negative_control_columns()),
    )


def accumulation_to_hourly_rate(*, variable: str) -> pl.Expr:
    """Return the expression turning an hour's accumulation into the hour's mean rate.

    ERA5 stores radiation in joules per square metre over the hour, so dividing by the seconds in
    the hour gives watts per square metre. Precipitation and snowfall are stored in metres of water
    over the hour, which becomes millimetres per hour.

    Args:
        variable: The ERA5 short name of an accumulation, such as `ssrd` or `tp`.

    Returns:
        An expression over the column named `variable`.

    Raises:
        ValueError: If `variable` is not an accumulation.
    """
    if variable not in ACCUMULATED_VARIABLES:
        msg = f"{variable!r} is not an accumulation: {ACCUMULATED_VARIABLES}"
        raise ValueError(msg)
    if variable in ("tp", "sf"):
        return pl.col(variable) * MILLIMETRES_PER_METRE
    return pl.col(variable) / SECONDS_PER_HOUR


def ratio_index(
    *, numerator: pl.Expr, denominator: pl.Expr, minimum_denominator: float = 0.0
) -> pl.Expr:
    """Return a ratio that is missing, not infinite, where the denominator is too small to trust.

    The clearness index (global irradiance over the top-of-atmosphere flux) and the clear-sky index
    (global irradiance over the clear-sky irradiance) are both this ratio.

    Args:
        numerator: The expression on top.
        denominator: The expression underneath.
        minimum_denominator: The ratio is missing wherever the denominator is at or below this.

    Returns:
        The ratio, or a null where the denominator is at or below `minimum_denominator`.
    """
    return pl.when(denominator > minimum_denominator).then(numerator / denominator)


def sky_regime(*, index: pl.Expr, thresholds: tuple[float, float]) -> pl.Expr:
    """Return the regime of an hour from a brightness index that is higher for clearer sky.

    Args:
        index: An expression that rises with the clarity of the sky, such as the clear-sky index.
        thresholds: The index below which an hour is overcast, and from which it is clear.

    Returns:
        An expression of `"clear"`, `"broken"`, or `"overcast"`, null where the index is null.
    """
    overcast_below, clear_from = thresholds
    return (
        pl.when(index.is_null())
        .then(None)
        .when(index >= clear_from)
        .then(pl.lit("clear"))
        .when(index < overcast_below)
        .then(pl.lit("overcast"))
        .otherwise(pl.lit("broken"))
    )


def sky_regime_from_cloud_cover(*, cloud_cover: pl.Expr) -> pl.Expr:
    """Return the regime of an hour from ERA5 total cloud cover, which is lower for clearer sky.

    Args:
        cloud_cover: Total cloud cover as a fraction from 0 to 1.

    Returns:
        An expression of `"clear"`, `"broken"`, or `"overcast"`, null where the cover is null.
    """
    clear_below, overcast_from = TOTAL_CLOUD_COVER_THRESHOLDS
    return (
        pl.when(cloud_cover.is_null())
        .then(None)
        .when(cloud_cover < clear_below)
        .then(pl.lit("clear"))
        .when(cloud_cover >= overcast_from)
        .then(pl.lit("overcast"))
        .otherwise(pl.lit("broken"))
    )


def season_of_month(*, month_number: pl.Expr) -> pl.Expr:
    """Return the meteorological season of a calendar month number.

    Args:
        month_number: The month as 1 to 12.

    Returns:
        `"winter"` for December to February, `"spring"` for March to May, `"summer"` for June to
        August, and `"autumn"` for September to November.
    """
    return (
        pl.when(month_number.is_in([12, 1, 2]))
        .then(pl.lit("winter"))
        .when(month_number.is_in([3, 4, 5]))
        .then(pl.lit("spring"))
        .when(month_number.is_in([6, 7, 8]))
        .then(pl.lit("summer"))
        .otherwise(pl.lit("autumn"))
    )


def raise_unless_same_rows(*, losses: pl.DataFrame, arms: Sequence[str]) -> None:
    """Raise unless every named arm holds exactly the same (site, time, seed) rows.

    A paired difference joins its two arms' rows and silently drops any row held by only one, so
    two arms scored on different rows would produce a difference over the overlap and say nothing.

    Args:
        losses: Per-row losses carrying `arm`, `site`, `time`, and `seed`.
        arms: The arms of one contrast, or of every arm that the page compares.

    Raises:
        ValueError: If an arm has no rows, or if an arm's rows differ from the first arm's.
    """
    keys = ["site", "time", "seed"]
    first = losses.filter(pl.col("arm") == arms[0]).select(keys)
    if first.is_empty():
        msg = f"arm {arms[0]!r} has no rows"
        raise ValueError(msg)
    for arm in arms[1:]:
        other = losses.filter(pl.col("arm") == arm).select(keys)
        only_first = first.join(other, on=keys, how="anti").height
        only_other = other.join(first, on=keys, how="anti").height
        if only_first or only_other:
            msg = (
                f"arms {arms[0]!r} and {arm!r} hold different rows: {only_first} (site, time, "
                f"seed) rows only in the first and {only_other} only in the second"
            )
            raise ValueError(msg)


def aerosol_hour_ending_mean(
    *, aerosol: pl.DataFrame, labels: pl.DataFrame, value_columns: Sequence[str]
) -> pl.DataFrame:
    """Return the mean over each labelled hour of a 3-hourly series, read by linear interpolation.

    The hour labelled `H` is the hour ending at `H`. The series is interpolated in time to the
    instants `H - 1 h` and `H`, and the two values are averaged. For a series that rises in a
    straight line the result is the series' value half an hour before the label, so a series
    shifted by one analysis step moves every result by one step's rise.

    Args:
        aerosol: One row per (`site`, `time`), with `time` on the analysis times (00, 03, ..., 21
            UTC), and one column per name in `value_columns`.
        labels: The (`site`, `time`) hours to return a value for.
        value_columns: The columns of `aerosol` to interpolate.

    Returns:
        One row per row of `labels`, with `value_columns`. A value is null where either analysis
        time on either side of an instant is missing.
    """
    step = f"{AEROSOL_STEP_HOURS}h"
    instants = pl.concat(
        labels.select("site", "time", instant=pl.col("time").dt.offset_by(f"{offset}m"))
        for offset in AEROSOL_SAMPLE_OFFSETS_MINUTES
    ).with_columns(lower=pl.col("instant").dt.truncate(step))
    prepared = instants.with_columns(
        upper=pl.col("lower").dt.offset_by(step),
        weight=(pl.col("instant") - pl.col("lower")).dt.total_seconds()
        / (AEROSOL_STEP_HOURS * SECONDS_PER_HOUR),
    )
    lower_values = aerosol.rename({"time": "lower", **{c: f"{c}__lower" for c in value_columns}})
    upper_values = aerosol.rename({"time": "upper", **{c: f"{c}__upper" for c in value_columns}})
    interpolated = (
        prepared.join(lower_values, on=["site", "lower"], how="left")
        .join(upper_values, on=["site", "upper"], how="left")
        .with_columns(
            **{
                c: pl.col(f"{c}__lower") * (1.0 - pl.col("weight"))
                + pl.col(f"{c}__upper") * pl.col("weight")
                for c in value_columns
            }
        )
    )
    return (
        interpolated.group_by("site", "time", maintain_order=True)
        .agg(
            pl.when(pl.col(c).null_count() > 0).then(None).otherwise(pl.col(c).mean()).alias(c)
            for c in value_columns
        )
        .sort("site", "time")
    )
