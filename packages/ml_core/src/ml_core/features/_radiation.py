"""Interval-aware radiation reconstruction on the half-hour grid.

Short-wave uses daylight midpoint indices to shape each native window before conserving its
energy. Long-wave fits a piecewise cubic Hermite interpolating polynomial (PCHIP) through native
means placed at their window midpoints, then evaluates the spline at half-hour midpoints. Missing
native windows stay missing. Policies and worked plots:
<https://openclimatefix.github.io/nged-substation-forecast/architecture/nwp-variable-conventions/#radiation-resampling>.
"""

import logging
from datetime import timedelta
from functools import partial
from itertools import pairwise
from math import pi
from typing import Final

import polars as pl
from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS, Nwp, NwpModelId

logger = logging.getLogger(__name__)

SHORT_WAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
LONG_WAVE: Final[str] = "downward_long_wave_radiation_flux_surface"
DAYLIGHT_FLOOR_W_M2: Final[float] = 20.0
"""Minimum native clear-sky mean for an index to influence neighbouring windows.

This conditioning threshold excludes weak dawn/dusk denominators. Excluded windows retain their
own energy; neither radiation values nor usable clear-sky indices are clipped.
"""


def _check_radiation_schedule(frame: pl.DataFrame, *, strict: bool) -> pl.DataFrame:
    """Reject unsupported native support in research; filter faulty rows in serving.

    Validation runs on native batches during collection, before expansion. Missing scheduled rows
    are allowed; an off-schedule timestamp or unknown weather source cannot define a window width.

    Args:
        frame: Native rows with UTC datetime columns `nwp_init_time` and `valid_time`. Each row
            represents a period-ending native radiation mean. Optional `nwp_model_id` identifies
            the source; when absent, ECMWF ENS is assumed. Other columns are passed through.
        strict: Raise `ValueError` for unsupported source/lead times when true; otherwise log
            `unsupported_radiation_schedule` and remove those rows. Scheduled lead zero is allowed
            here, although it has no backward window and is removed during expansion.

    Returns:
        The input columns and dtypes, with all rows retained if valid. In non-strict mode only
        supported ECMWF ENS rows remain. This helper does not insert missing rows or null values;
        the caller's later join onto the forecast grid leaves excluded timestamps null.
    """
    supported = (pl.col("valid_time") - pl.col("nwp_init_time")).is_in(
        [timedelta(hours=lead) for lead in ECMWF_ENS_LEAD_TIME_HOURS]
    )
    if "nwp_model_id" in frame:
        supported &= pl.col("nwp_model_id") == NwpModelId.ECMWF_ENS_0_25_degree.name
    invalid = frame.filter(~supported.fill_null(False))
    if not invalid.is_empty():
        identity = [
            name for name in ("nwp_model_id", "nwp_init_time", "valid_time") if name in frame
        ]
        examples = invalid.select(identity).unique().head(5).to_dicts()
        message = (
            f"unsupported_radiation_schedule: {invalid.height} native rows in this batch have "
            f"unsupported source/lead times; examples: {examples}. "
            "Radiation requires ECMWF ENS scheduled native windows."
        )
        if strict:
            raise ValueError(message)
        logger.warning(message)
        # Exclude unknown sources as well as off-schedule rows before applying ECMWF widths.
        frame = frame.filter(supported.fill_null(False))
    return frame


def _check_solar_coordinates(frame: pl.DataFrame) -> pl.DataFrame:
    """Reject incomplete solar inputs before null geometry becomes darkness.

    Reusable-helper validation: production's validated site relation supplies finite coordinates,
    but direct callers can omit a site or provide null/nonfinite values. Those inputs are programmer
    errors rather than fully-dark weather windows, so the constant-energy fallback must not apply.

    Args:
        frame: Joined solar target rows containing `time_series_id`, numeric `latitude` and
            `longitude` in degrees, and any other columns to pass through. The caller supplies
            one row per site/target timestamp. Coordinates must be finite, non-null, and within
            [-90, 90] latitude and [-180, 180] longitude; violations raise `ValueError` with
            `invalid_solar_coordinates`.

    Returns:
        The same frame, with every row, column, dtype, and value unchanged when validation passes.
    """
    valid = (
        pl.col("latitude").is_finite()
        & pl.col("latitude").is_between(-90, 90)
        & pl.col("longitude").is_finite()
        & pl.col("longitude").is_between(-180, 180)
    )
    invalid = frame.filter(~valid.fill_null(False))
    if not invalid.is_empty():
        sites = invalid["time_series_id"].unique().head(5).to_list()
        raise ValueError(
            "invalid_solar_coordinates: short-wave requires finite latitude/longitude for "
            f"every time series; invalid examples: {sites}."
        )
    return frame


def _half_hour_clear_sky(*, targets: pl.LazyFrame, coordinates: pl.LazyFrame) -> pl.LazyFrame:
    """Evaluate Haurwitz means using two 15-minute midpoint samples per target.

    US National Oceanic and Atmospheric Administration (NOAA) fractional-year solar geometry
    includes longitude and equation of time. Geometry is independent of run/member, so compute it
    once per series and target timestamp.

    Args:
        targets: Rows containing `time_series_id` and UTC datetime `valid_time`, the ending label
            of a half-hour interval. Repeated site/time pairs across runs or ensemble members are
            deduplicated; all other input columns are ignored.
        coordinates: One row per `time_series_id`, with numeric `latitude` and `longitude` in
            degrees. Supply exactly these three columns. Every target site must have valid
            coordinates; duplicate IDs fail the many-to-one join and missing/invalid coordinates
            fail `_check_solar_coordinates` during collection.

    Returns:
        One row per distinct target site/time pair, with exactly `time_series_id`, `valid_time`
        (their input dtypes), `_clear_sky` (Float64, W/m²), and `_daylight_block` (Int64).
        `_clear_sky` averages samples 7.5 and 22.5 minutes before `valid_time` and is zero in
        darkness. `_daylight_block` counts transitions from zero to positive clear-sky flux in
        timestamp order per site, starting at one if the first target is sunlit, otherwise zero.
        No run/member keys or coordinate columns are returned; output row order is unspecified.
    """
    samples = (
        targets.select("time_series_id", "valid_time")
        .unique()
        .join(coordinates, on="time_series_id", how="left", validate="m:1")
        .map_batches(
            _check_solar_coordinates,
            predicate_pushdown=False,
            projection_pushdown=False,
            slice_pushdown=False,
            streamable=True,
        )
        .with_columns(_offset=pl.lit([450, 1350]))
        .explode("_offset")
        .with_columns(_stamp=pl.col("valid_time") - pl.duration(seconds=pl.col("_offset")))
        .with_columns(
            _hour=pl.col("_stamp").dt.hour()
            + pl.col("_stamp").dt.minute() / 60
            + pl.col("_stamp").dt.second() / 3600,
            _latitude=pl.col("latitude").cast(pl.Float64).radians(),
        )
        .with_columns(
            _gamma=2
            * pi
            / pl.when(pl.col("_stamp").dt.is_leap_year()).then(366).otherwise(365)
            * (pl.col("_stamp").dt.ordinal_day() - 1 + (pl.col("_hour") - 12) / 24)
        )
    )
    gamma = pl.col("_gamma")
    samples = samples.with_columns(
        _declination=0.006918
        - 0.399912 * gamma.cos()
        + 0.070257 * gamma.sin()
        - 0.006758 * (2 * gamma).cos()
        + 0.000907 * (2 * gamma).sin()
        - 0.002697 * (3 * gamma).cos()
        + 0.00148 * (3 * gamma).sin(),
        _equation=229.18
        * (
            0.000075
            + 0.001868 * gamma.cos()
            - 0.032077 * gamma.sin()
            - 0.014615 * (2 * gamma).cos()
            - 0.040849 * (2 * gamma).sin()
        ),
    ).with_columns(
        _cosine=(
            pl.col("_latitude").sin() * pl.col("_declination").sin()
            + pl.col("_latitude").cos()
            * pl.col("_declination").cos()
            * ((pl.col("_hour") * 60 + pl.col("_equation") + 4 * pl.col("longitude")) / 4 - 180)
            .radians()
            .cos()
        ).clip(lower_bound=0)
    )
    sky = samples.group_by("time_series_id", "valid_time").agg(
        _clear_sky=pl.when(pl.col("_cosine") > 0)
        .then(1098 * pl.col("_cosine") * (-0.059 / pl.col("_cosine")).exp())
        .otherwise(0.0)
        .mean()
    )
    # Half-hour support, rather than native windows, detects short northern summer nights.
    return (
        sky.with_columns(
            _day_start=(pl.col("_clear_sky") > 0)
            & (
                pl.col("_clear_sky").shift().over("time_series_id", order_by="valid_time") <= 0
            ).fill_null(True)
        )
        .with_columns(
            _daylight_block=pl.col("_day_start")
            .cast(pl.Int64)
            .cum_sum()
            .over("time_series_id", order_by="valid_time")
        )
        .drop("_day_start")
    )


def _native_targets(*, native: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:
    """Expand supported source windows, using scheduled widths even when rows are missing.

    Args:
        native: One row per `group_cols`/`valid_time`, with Datetime[us, UTC] `nwp_init_time` and
            `valid_time` and the available radiation columns named by `SHORT_WAVE`/`LONG_WAVE`.
            Radiation values are nullable period-ending means in W/m². Source rows must have
            passed schedule validation; other columns, if supplied, are also retained.
        group_cols: Columns identifying independent trajectories, including `nwp_init_time` and
            normally `time_series_id`, `ensemble_member`, and optional `nwp_model_id`.

    Returns:
        Six rows per supported three-hour window or twelve per six-hour window. Every input
        column is retained, except source `valid_time` becomes `_window_end` and a new
        `valid_time` labels each half-hour ending from `_window_start` + 30 minutes through
        `_window_end`. Radiation means and other source values repeat across their window;
        null means remain null. Both window bounds use the input datetime dtype.
        Added columns are `_lead` (Duration[us], valid time minus initialisation),
        `_width_minutes` (Int64, scheduled window length), `_window_start`, and `_anchor_second`
        (Int64, Unix seconds at the native window midpoint). If short-wave is present, `_break`
        (Boolean) marks the first window, a missing preceding window, or a null current/previous
        short-wave mean; `_segment` (Int64) cumulatively counts these breaks per trajectory.
        Lead-zero and unsupported leads produce no rows. Missing source windows are not filled,
        and complete windows are expanded without cropping to the caller's forecast horizon.
    """
    leads = ECMWF_ENS_LEAD_TIME_HOURS
    schedule = (
        pl.DataFrame(
            {
                "_lead": [timedelta(hours=lead) for lead in leads[1:]],
                "_width_minutes": [(end - start) * 60 for start, end in pairwise(leads)],
            }
        )
        .cast({"_lead": pl.Duration("us")})
        .lazy()
    )
    windows = (
        native.with_columns(
            _lead=(pl.col("valid_time") - pl.col("nwp_init_time")).cast(pl.Duration("us"))
        )
        .join(schedule, on="_lead", how="inner")
        .rename({"valid_time": "_window_end"})
    )
    windows = windows.with_columns(
        _window_start=pl.col("_window_end") - pl.duration(minutes=pl.col("_width_minutes")),
        _anchor_second=pl.col("_window_end").dt.epoch("s") - pl.col("_width_minutes") * 30,
    )
    if SHORT_WAVE in native.collect_schema():
        windows = windows.with_columns(
            _break=pl.col(SHORT_WAVE).is_null()
            | pl.col(SHORT_WAVE).shift().over(group_cols, order_by="_window_end").is_null()
            | (
                pl.col("_window_start")
                != pl.col("_window_end").shift().over(group_cols, order_by="_window_end")
            ).fill_null(True)
        ).with_columns(
            _segment=pl.col("_break")
            .cast(pl.Int64)
            .cum_sum()
            .over(group_cols, order_by="_window_end")
        )
    return windows.with_columns(
        valid_time=pl.datetime_ranges(
            start=pl.col("_window_start") + timedelta(minutes=30),
            end=pl.col("_window_end"),
            interval="30m",
        )
    ).explode("valid_time", empty_as_null=False)


def _shape_short_wave(
    *, targets: pl.LazyFrame, group_cols: list[str], coordinates: pl.LazyFrame
) -> pl.LazyFrame:
    """Interpolate indices in daylight blocks, then conserve complete native-window energy.

    Args:
        targets: Complete expanded windows from `_native_targets`, including `SHORT_WAVE`,
            `time_series_id`, `valid_time`, `_window_end`, `_anchor_second`, `_segment`, and
            every `group_cols` column. Each window's native short-wave mean in W/m² is repeated
            on its half-hour rows. Additional columns, including long-wave results, pass through.
        group_cols: The trajectory identity columns used to expand `targets`.
        coordinates: Exactly `time_series_id`, `latitude`, and `longitude`, with one finite,
            non-null coordinate pair in degrees per target site, as for `_half_hour_clear_sky`.

    Returns:
        The same target rows and input columns, with `SHORT_WAVE` replaced by Float64 half-hour
        means in W/m² whose average within each complete native window equals its source mean.
        Null source means stay null; missing windows are not inserted or crossed by interpolation.
        Added Float64 columns are `_clear_sky` and `_native_clear_sky` (half-hour and native-window
        clear-sky means, W/m²), `_ratio` (usable native mean/clear-sky index), `_earlier_ratio` and
        `_later_ratio` (indices bracketing the target midpoint within its segment/daylight block),
        `_shape` and `_shape_mean` (interpolated solar weights and their native-window mean, W/m²).
        Added Int64 columns are `_daylight_block`, `_target_second` (Unix seconds at the half-hour
        midpoint), and nullable `_earlier_second`/`_later_second` (the index anchor times).
        `_ratio` is null when the source mean is null, the native clear-sky mean is below
        `DAYLIGHT_FLOOR_W_M2`, or its window has no sunlit target in this daylight block. Neighbour
        indices are null when no usable anchor exists on that side within the segment/block; they
        can include the window itself according to the target midpoint. Missing neighbours use the
        available index, then local clear-sky weights if no index is usable. Fully dark windows
        keep a constant source mean.
    """
    window_keys = [*group_cols, "_window_end"]
    sky = _half_hour_clear_sky(targets=targets, coordinates=coordinates)
    targets = targets.join(sky, on=["time_series_id", "valid_time"], how="left").with_columns(
        _native_clear_sky=pl.col("_clear_sky").mean().over(window_keys),
        _target_second=pl.col("valid_time").dt.epoch("s") - 900,
    )
    interpolation_keys = [*group_cols, "_segment", "_daylight_block"]
    knots = targets.group_by(*interpolation_keys, "_window_end").agg(
        _anchor_second=pl.col("_anchor_second").first(),
        _ratio=(
            pl.when(pl.col("_native_clear_sky") >= DAYLIGHT_FLOOR_W_M2)
            .then(pl.col(SHORT_WAVE) / pl.col("_native_clear_sky"))
            .filter(pl.col("_clear_sky") > 0)
            .first()
        ),
    )
    # Find neighbouring usable indices on the native-window table. A window can contribute to
    # two daylight blocks; its midpoint remains the anchor in each block. Source segments keep
    # forward/backward fills from crossing missing windows.
    for shift, label, fill in ((1, "earlier", "forward"), (-1, "later", "backward")):
        for column, suffix in (
            (pl.col("_ratio"), "ratio"),
            (pl.when(pl.col("_ratio").is_not_null()).then(pl.col("_anchor_second")), "second"),
        ):
            shifted = column.shift(shift)
            neighbour = shifted.forward_fill() if fill == "forward" else shifted.backward_fill()
            knots = knots.with_columns(
                **{
                    f"_{label}_{suffix}": neighbour.over(
                        interpolation_keys, order_by="_anchor_second"
                    )
                }
            )
    shaped = targets.join(
        knots.drop("_anchor_second"),
        on=[*interpolation_keys, "_window_end"],
        how="left",
    )
    before_anchor = pl.col("_target_second") < pl.col("_anchor_second")
    own_second = pl.when(pl.col("_ratio").is_not_null()).then(pl.col("_anchor_second"))
    shaped = shaped.with_columns(
        _earlier_second=pl.when(before_anchor)
        .then(pl.col("_earlier_second"))
        .otherwise(pl.coalesce(own_second, pl.col("_earlier_second"))),
        _earlier_ratio=pl.when(before_anchor)
        .then(pl.col("_earlier_ratio"))
        .otherwise(pl.coalesce("_ratio", "_earlier_ratio")),
        _later_second=pl.when(before_anchor)
        .then(pl.coalesce(own_second, pl.col("_later_second")))
        .otherwise(pl.col("_later_second")),
        _later_ratio=pl.when(before_anchor)
        .then(pl.coalesce("_ratio", "_later_ratio"))
        .otherwise(pl.col("_later_ratio")),
    )
    span = pl.col("_later_second") - pl.col("_earlier_second")
    ratio = (
        pl.when(span > 0)
        .then(
            pl.col("_earlier_ratio")
            + (pl.col("_later_ratio") - pl.col("_earlier_ratio"))
            * (pl.col("_target_second") - pl.col("_earlier_second"))
            / span
        )
        .otherwise(pl.coalesce("_earlier_ratio", "_later_ratio"))
    )
    shaped = shaped.with_columns(_shape=ratio.fill_null(1.0) * pl.col("_clear_sky"))
    shaped = shaped.with_columns(_shape_mean=pl.col("_shape").mean().over(window_keys))
    # Local solar weights are the fallback if no usable index exists or the interpolated sum is
    # zero. A positive fully-dark mean has no solar shape; retain its energy as a constant.
    return shaped.with_columns(
        **{
            SHORT_WAVE: pl.col(SHORT_WAVE)
            * pl.when(pl.col("_shape_mean") > 0)
            .then(pl.col("_shape") / pl.col("_shape_mean"))
            .when(pl.col("_native_clear_sky") > 0)
            .then(pl.col("_clear_sky") / pl.col("_native_clear_sky"))
            .otherwise(1.0)
        }
    )


def _pchip_edge_slope(
    *, width: pl.Expr, next_width: pl.Expr, slope: pl.Expr, next_slope: pl.Expr
) -> pl.Expr:
    """Limit the one-sided endpoint derivative used by PCHIP.

    Args:
        width: Expression for the positive gap in seconds between the endpoint and next anchor.
        next_width: Expression for the next positive anchor gap in seconds; null when absent.
        slope: Expression for the secant over `width`, in W/m² per second.
        next_slope: Expression for the secant over `next_width`, in W/m² per second.

    Returns:
        An expression for the endpoint derivative in W/m² per second. With no next gap it uses
        `slope`; otherwise it limits the one-sided estimate to preserve the local slope's sign
        and bounds it to three times that slope when the two secants change sign. Null inputs
        can produce null output; the caller handles isolated anchors.
    """
    estimate = ((2 * width + next_width) * slope - width * next_slope) / (width + next_width)
    return (
        pl.when(next_width.is_null())
        .then(slope)
        .when(estimate * slope <= 0)
        .then(0.0)
        .when((slope * next_slope <= 0) & (estimate.abs() > 3 * slope.abs()))
        .then(3 * slope)
        .otherwise(estimate)
    )


def _shape_long_wave(*, targets: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:
    """Evaluate midpoint PCHIP lazily, with constant edges and no interpolation across gaps.

    Native interval means act as midpoint estimates, without enforcing interval energy. PCHIP's
    harmonic interior slopes and limited endpoint slopes preserve the shape of the supplied means.

    Args:
        targets: Complete expanded windows from `_native_targets`, including `LONG_WAVE`,
            `valid_time`, `_window_start`, `_window_end`, `_anchor_second`, and every `group_cols`
            column. Each window's native long-wave mean in W/m² repeats on its half-hour rows.
            Other input columns, including short-wave means, pass through.
        group_cols: The trajectory identity columns used to expand `targets`.

    Returns:
        The same target rows and input columns, with `LONG_WAVE` replaced by Float64 midpoint
        estimates in W/m² evaluated 15 minutes before each ending `valid_time`. Each contiguous
        sequence holds its nearest anchor's value outside the anchor range; null source windows
        stay null and absent source windows produce no rows. Neither kind of gap is interpolated.
        Added columns are `_earlier_value`/`_later_value` (nullable Float64 neighbouring native
        means, W/m²), `_earlier_width`/`_later_width` (nullable Int64 anchor gaps in seconds, null
        for noncontiguous or null neighbours), `_earlier_secant`/`_later_secant` (Float64 neighbour
        secants, W/m² per second), `_derivative` (Float64 derivative at this anchor), and
        `_earlier_derivative`/`_later_derivative` (nullable Float64 neighbouring derivatives in
        the same units). Neighbour values, secants and derivatives can refer across gaps; only
        non-null widths permit their use in the spline. Singleton anchors use derivative zero.
    """
    window_keys = [*group_cols, "_window_end"]
    windows = targets.select(*window_keys, "_window_start", "_anchor_second", LONG_WAVE).unique()
    for shift, label in ((1, "earlier"), (-1, "later")):
        neighbour = pl.col(LONG_WAVE).shift(shift).over(group_cols, order_by="_window_end")
        anchor = pl.col("_anchor_second").shift(shift).over(group_cols, order_by="_window_end")
        contiguous = (
            pl.col("_window_start")
            == pl.col("_window_end").shift(1).over(group_cols, order_by="_window_end")
            if shift == 1
            else pl.col("_window_end")
            == pl.col("_window_start").shift(-1).over(group_cols, order_by="_window_end")
        )
        windows = windows.with_columns(
            **{
                f"_{label}_value": neighbour.cast(pl.Float64),
                f"_{label}_width": pl.when(contiguous & neighbour.is_not_null()).then(
                    (anchor - pl.col("_anchor_second")).abs()
                ),
                f"_{label}_secant": (neighbour.cast(pl.Float64) - pl.col(LONG_WAVE))
                / (anchor - pl.col("_anchor_second")),
            }
        )
    left_h, right_h = pl.col("_earlier_width"), pl.col("_later_width")
    left_d, right_d = pl.col("_earlier_secant"), pl.col("_later_secant")
    left_weight, right_weight = 2 * right_h + left_h, right_h + 2 * left_h
    derivative = (
        pl.when(left_h.is_not_null() & right_h.is_not_null())
        .then(
            pl.when(left_d * right_d > 0)
            .then((left_weight + right_weight) / (left_weight / left_d + right_weight / right_d))
            .otherwise(0.0)
        )
        .when(left_h.is_null())
        .then(
            _pchip_edge_slope(
                width=right_h,
                next_width=right_h.shift(-1).over(group_cols, order_by="_window_end"),
                slope=right_d,
                next_slope=right_d.shift(-1).over(group_cols, order_by="_window_end"),
            )
        )
        .otherwise(
            _pchip_edge_slope(
                width=left_h,
                next_width=left_h.shift(1).over(group_cols, order_by="_window_end"),
                slope=left_d,
                next_slope=left_d.shift(1).over(group_cols, order_by="_window_end"),
            )
        )
        .fill_null(0.0)
    )
    windows = windows.with_columns(_derivative=derivative).with_columns(
        _earlier_derivative=pl.col("_derivative").shift(1).over(group_cols, order_by="_window_end"),
        _later_derivative=pl.col("_derivative").shift(-1).over(group_cols, order_by="_window_end"),
    )
    shaped = targets.join(
        windows.drop("_window_start", "_anchor_second", LONG_WAVE), on=window_keys, how="left"
    )
    offset = pl.col("valid_time").dt.epoch("s") - 900 - pl.col("_anchor_second")
    expressions = []
    for direction, label in ((-1, "earlier"), (1, "later")):
        width = pl.col(f"_{label}_width")
        fraction = offset.abs() / width
        curve = (
            (2 * fraction**3 - 3 * fraction**2 + 1) * pl.col(LONG_WAVE)
            + (fraction**3 - 2 * fraction**2 + fraction) * direction * width * pl.col("_derivative")
            + (-2 * fraction**3 + 3 * fraction**2) * pl.col(f"_{label}_value")
            + (fraction**3 - fraction**2) * direction * width * pl.col(f"_{label}_derivative")
        )
        expressions.append(pl.when(width.is_not_null()).then(curve).otherwise(pl.col(LONG_WAVE)))
    return shaped.with_columns(
        **{LONG_WAVE: pl.when(offset < 0).then(expressions[0]).otherwise(expressions[1])}
    )


def _resample_radiation(
    *,
    native: pl.LazyFrame,
    group_cols: list[str],
    solar_coordinates: pl.LazyFrame | None,
    strict_schedule: bool = True,
) -> pl.LazyFrame:
    """Return half-hour radiation features over supported backward native windows.

    Args:
        native: One row per trajectory/`valid_time`, with Datetime[us, UTC] `nwp_init_time` and
            `valid_time`, every `group_cols` column, and the available `Nwp.radiation_var_names`
            columns as nullable floating-point period-ending means in W/m². Optional `nwp_model_id`
            identifies the source; its absence assumes ECMWF ENS. Other weather columns are ignored.
        group_cols: Identity columns separating independent trajectories; must include
            `nwp_init_time` and, for short-wave, `time_series_id`. Production also supplies
            `ensemble_member`; include `nwp_model_id` whenever present so it survives selection and
            source/run/member sequences stay separate.
        solar_coordinates: Exactly `time_series_id`, numeric `latitude` and `longitude` in degrees,
            with one finite, non-null coordinate pair per target site. Required for short-wave;
            may be None for long-wave alone. Duplicates or missing/invalid coordinates fail lazily.
        strict_schedule: Raise on unsupported source/lead times during collection when true;
            otherwise log `unsupported_radiation_schedule` and exclude faulty native rows.

    Returns:
        Exactly `group_cols`, half-hour-ending `valid_time`, and the radiation columns present in
        `native` (in sorted name order). Identity columns and radiation retain their source dtypes;
        `valid_time` retains the source datetime dtype. Short-wave values are backward half-hour
        means conserving each native-window mean; long-wave values are PCHIP estimates at half-hour
        midpoints, without an energy constraint. Null source windows remain null, absent windows
        and lead zero produce no rows, and unsupported rows are excluded in non-strict mode.
        Every retained native window expands fully, including targets outside the caller's desired
        horizon; the caller crops by joining onto its forecast grid. No helper columns are returned
        and row order is unspecified.
    """
    schema = native.collect_schema()
    radiation = sorted(Nwp.radiation_var_names.intersection(schema.names()))
    checked = native.select(*group_cols, "valid_time", *radiation).map_batches(
        partial(_check_radiation_schedule, strict=strict_schedule),
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
        streamable=True,
    )
    targets = _native_targets(native=checked, group_cols=group_cols)
    if LONG_WAVE in radiation:
        targets = _shape_long_wave(targets=targets, group_cols=group_cols)
    if SHORT_WAVE in radiation:
        if solar_coordinates is None:
            raise ValueError("Short-wave reconstruction requires solar_coordinates.")
        targets = _shape_short_wave(
            targets=targets, group_cols=group_cols, coordinates=solar_coordinates
        )
    return targets.select(*group_cols, "valid_time", *radiation).cast(
        {column: schema[column] for column in radiation}
    )
