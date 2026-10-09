"""Build the tables behind the INDDEM and INDGEN page, and write every number it quotes.

Run after the downloads, which are listed in this folder's README:
`uv run python studies/indgen_inddem/build_tables.py`. The script writes parquet tables and
`report.md` to the study's data folder, and prints the report.

Each INDDEM or INDGEN issue covers a run of future half-hours, so a target half-hour has many
values, one from each issue published before it starts. The script keeps two views of every
target half-hour: the latest issue published at or before the half-hour starts (`latest`), and the
first issue of the target's UTC day, which is the latest issue published at or before 00:00 UTC
(`first_of_day`). Both views are plain "as of a cut-off" lookups, so neither uses a value published
after its cut-off.
"""

import itertools
import warnings
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from indgen_inddem_common import (
    AGV_END,
    AGV_PATH,
    BMU_REFERENCE_PATH,
    BOUNDARIES,
    GSP_GROUPS,
    INDDEM_MATCH_TOLERANCE_MW,
    INDDEM_PATH,
    INDGEN_PATH,
    INDO_PATH,
    LONDON,
    NGED_GROUPS,
    PN_SAMPLE_PATH,
    PV_LIVE_PATH,
    SIGN_TOLERANCE_MW,
    STUDY_DIR,
    STUDY_END,
    STUDY_START,
    ZONES,
    DatasetType,
    ViewType,
    membership_matrix,
)
from scipy.optimize import nnls

VALUE_COLUMNS: Final[dict[DatasetType, str]] = {"inddem": "demand_mw", "indgen": "generation_mw"}
SOURCE_PATHS: Final[dict[DatasetType, str]] = {
    "inddem": str(INDDEM_PATH),
    "indgen": str(INDGEN_PATH),
}
VIEWS: Final[tuple[ViewType, ...]] = ("latest", "first_of_day")
HALF_HOUR_MINUTES: Final[int] = 30
LAST_LOCAL_SLOTS_FROM: Final[int] = 44
"""The first UK local half-hour slot (22:00) of the late-evening slots the report lists."""
FIRST_LOCAL_SLOTS_TO: Final[int] = 4
"""The slots 0 to 3 (00:00 to 02:00 UK local) that the report lists with the late-evening slots."""
LONG_ISSUE_HOURS: Final[float] = 30.0
"""An issue that reaches further than this is the long issue published from about 12:00 UK local."""
LAST_SLOTS_FROM: Final[int] = 44
"""The first UTC half-hour of the day (22:00 UTC) in the late-day slots the report lists."""
SATURDAY: Final[int] = 6
"""The ISO weekday number of Saturday, so a weekday number at or above it is a weekend day."""
STUDY_HALF_HOURS: Final[int] = 18960
"""The half-hours of the study window, 395 days of 48 half-hours, with the two clock-change days
making up for each other."""
SUM_TO_ONE_WEIGHT: Final[float] = 20.0
"""How many times the largest summed PN the fit weights its rows that make each group's fractions
sum to 1."""
WINDOW_START_UTC: Final[datetime] = datetime(
    STUDY_START.year, STUDY_START.month, STUDY_START.day, tzinfo=UTC
)
WINDOW_AFTER_UTC: Final[datetime] = datetime(
    STUDY_END.year, STUDY_END.month, STUDY_END.day, tzinfo=UTC
) + timedelta(days=1)
"""Midnight after the last day of the study window."""
UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")

report_lines: list[str] = []


def say(text: object = "") -> None:
    """Add a line to the report, turning a table into its printed form."""
    report_lines.append(str(text))


def issues_table(*, dataset: DatasetType) -> pl.DataFrame:
    """Read every issue of one dataset as `boundary, time, publish_time, value_mw`."""
    return (
        pl.scan_parquet(SOURCE_PATHS[dataset])
        .select(
            "boundary",
            "time",
            "publish_time",
            value_mw=pl.col(VALUE_COLUMNS[dataset]).cast(pl.Float64),
        )
        .collect()
    )


def views_table(*, dataset: DatasetType, issues: pl.DataFrame) -> pl.DataFrame:
    """Look up both views of every (boundary, target half-hour) in the study window."""
    targets = (
        issues.filter(pl.col("time").is_between(WINDOW_START_UTC, WINDOW_AFTER_UTC, closed="left"))
        .select("boundary", "time")
        .unique()
    )
    by_publish = issues.sort("publish_time")
    frames = []
    for view in VIEWS:
        cutoff = pl.col("time") if view == "latest" else pl.col("time").dt.truncate("1d")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            looked_up = (
                targets.with_columns(cutoff=cutoff)
                .sort("cutoff")
                .join_asof(
                    by_publish,
                    left_on="cutoff",
                    right_on="publish_time",
                    by=["boundary", "time"],
                    strategy="backward",
                )
            )
        frames.append(
            looked_up.select(
                dataset=pl.lit(dataset),
                view=pl.lit(view),
                boundary="boundary",
                time="time",
                publish_time="publish_time",
                value_mw="value_mw",
            )
        )
    return pl.concat(frames).sort("view", "boundary", "time")


def zones_from_boundaries(*, views: pl.DataFrame) -> pl.DataFrame:
    """Recover the 17 study zones from the national total and the 17 boundaries.

    The zones solve the 18 by 17 system in `membership_matrix` by least squares. The system has
    one more equation than unknowns, so `boundary_identity_residual` in the report checks the
    transcription.
    """
    matrix = membership_matrix()
    inverse = np.linalg.pinv(matrix)
    frames = []
    for (dataset, view), group in views.group_by("dataset", "view", maintain_order=True):
        wide = group.pivot(on="boundary", index="time", values="value_mw").drop_nulls().sort("time")
        values = wide.select(list(BOUNDARIES)).to_numpy()
        zones = values @ inverse.T
        frames.append(
            pl.DataFrame({"time": wide["time"], **{z: zones[:, i] for i, z in enumerate(ZONES)}})
            .unpivot(index="time", variable_name="zone", value_name="value_mw")
            .with_columns(dataset=pl.lit(dataset), view=pl.lit(view))
            .select("dataset", "view", "zone", "time", "value_mw")
        )
    return pl.concat(frames).sort("dataset", "view", "zone", "time")


def boundary_identity_residual(*, views: pl.DataFrame) -> pl.DataFrame:
    """Return B16 - B11 - (B9 - B17 - B8) for every half-hour, which must be zero.

    Both sides equal zone Z10 in the circular's table, so a transcription error in any of the
    five boundaries shows up as a residual.
    """
    wide = views.pivot(on="boundary", index=["dataset", "view", "time"], values="value_mw")
    return wide.select(
        "dataset",
        "view",
        "time",
        residual_mw=pl.col("B16") - pl.col("B11") - (pl.col("B9") - pl.col("B17") - pl.col("B8")),
    )


def issue_reach(*, issues: pl.DataFrame, dataset: DatasetType) -> pl.DataFrame:
    """Return, for each issue of boundary `N`, its half-hours and how far ahead it reaches."""
    return (
        issues.filter(pl.col("boundary") == "N")
        .group_by("publish_time")
        .agg(half_hours=pl.len(), last_time=pl.col("time").max())
        .select(
            dataset=pl.lit(dataset),
            publish_time="publish_time",
            half_hours="half_hours",
            end_time=pl.col("last_time") + pl.duration(minutes=HALF_HOUR_MINUTES),
            reach_hours=(
                (pl.col("last_time") - pl.col("publish_time")).dt.total_minutes()
                + HALF_HOUR_MINUTES
            )
            / 60,
        )
        .sort("publish_time")
    )


def zone_sign_report(*, zones: pl.DataFrame) -> pl.DataFrame:
    """Count, for each dataset, view, and zone, the half-hours with the wrong sign."""
    wrong = (
        pl.when(pl.col("dataset") == "inddem")
        .then(pl.col("value_mw") > SIGN_TOLERANCE_MW)
        .otherwise(pl.col("value_mw") < -SIGN_TOLERANCE_MW)
    )
    return (
        zones.group_by("dataset", "view", "zone")
        .agg(half_hours=pl.len(), wrong_sign=wrong.sum())
        .with_columns(share=pl.col("wrong_sign") / pl.col("half_hours"))
        .sort("dataset", "view", "zone")
    )


def agv_import_mw() -> pl.DataFrame:
    """Return AGV as net import from the transmission system, in MW, for each GSP group."""
    return (
        pl.read_parquet(AGV_PATH)
        .select(
            "time",
            "gsp_group",
            import_mw=pl.when(pl.col("import_export") == "I")
            .then(2.0 * pl.col("take_mwh"))
            .otherwise(-2.0 * pl.col("take_mwh")),
        )
        .sort("gsp_group", "time")
    )


def agv_against_indo(*, agv: pl.DataFrame) -> pl.DataFrame:
    """Join the national AGV total to the initial national demand outturn (INDO)."""
    total = agv.group_by("time").agg(agv_mw=pl.col("import_mw").sum(), groups=pl.len())
    indo = pl.read_parquet(INDO_PATH).select("time", "indo_mw").unique("time")
    return (
        total.join(indo, on="time", how="inner")
        .with_columns(ratio=pl.col("agv_mw") / pl.col("indo_mw"))
        .sort("time")
    )


def period_average_mw(*, segments: pl.DataFrame) -> pl.DataFrame:
    """Time-average each BMU's Physical Notification over its settlement period, in MW.

    The notified level changes linearly between the two ends of a segment, and the segments of one
    BMU tile the settlement period, so the average is the sum of the trapezoids over 30 minutes.
    """
    return (
        segments.with_columns(
            minutes=(pl.col("time_to") - pl.col("time_from")).dt.total_seconds() / 60,
            period_start=pl.col("time_from").min().over("settlement_date", "settlement_period"),
        )
        .group_by("period_start", "national_grid_bmu_id")
        .agg(
            average_mw=(
                (pl.col("level_from_mw") + pl.col("level_to_mw")) / 2 * pl.col("minutes")
            ).sum()
            / HALF_HOUR_MINUTES,
            covered_minutes=pl.col("minutes").sum(),
        )
        .rename({"period_start": "time"})
    )


def bmu_groups() -> pl.DataFrame:
    """Return Elexon's register of each BMU's GSP group and interconnector.

    The table has one row for each National Grid identifier. The register holds some BMUs twice
    with identical rows, so the table drops exact duplicates.
    """
    return (
        pl.read_parquet(BMU_REFERENCE_PATH)
        .select(
            "national_grid_bmu_id",
            gsp_group="gsp_group_id",
            interconnector="interconnector_id",
        )
        .unique()
    )


def group_sums_from_pn() -> pl.DataFrame:
    """Sum the sampled BMUs' average PN by column, split into import and export.

    The BMUs of one interconnector are netted into one unit before the split, because an
    interconnector's trading BMUs import and export in the same half-hour and INDDEM and INDGEN
    count the interconnector once. Each interconnector is its own column, `IC <name>`. Every other
    BMU is its own unit and goes in the column of its GSP group. A BMU that the register gives no
    GSP group, or does not list, goes in the column `none`.
    """
    averages = period_average_mw(segments=pl.read_parquet(PN_SAMPLE_PATH))
    units = (
        averages.join(bmu_groups(), on="national_grid_bmu_id", how="left", validate="m:1")
        .with_columns(
            unit=pl.coalesce("interconnector", "national_grid_bmu_id"),
            column=pl.coalesce(
                pl.when(pl.col("interconnector").is_not_null())
                .then(pl.lit("IC ") + pl.col("interconnector"))
                .otherwise(None),
                "gsp_group",
                pl.lit("none"),
            ),
        )
        .group_by("time", "unit", "column")
        .agg(net_mw=pl.col("average_mw").sum())
    )
    return (
        units.group_by("time", gsp_group=pl.col("column"))
        .agg(
            import_mw=pl.col("net_mw").filter(pl.col("net_mw") < 0).sum(),
            export_mw=pl.col("net_mw").filter(pl.col("net_mw") > 0).sum(),
            units=pl.len(),
        )
        .sort("time", "gsp_group")
    )


def report_bmu_group_coverage() -> None:
    """Report how many sampled BMUs have a GSP group in Elexon's register."""
    sampled = pl.read_parquet(PN_SAMPLE_PATH).select("national_grid_bmu_id").unique()
    registered = bmu_groups()
    joined = sampled.join(registered, on="national_grid_bmu_id", how="left", validate="m:1")
    in_register = sampled.join(registered, on="national_grid_bmu_id", how="semi").height
    say()
    say("## Sampled BMUs and Elexon's register")
    say(
        f"- {sampled.height} distinct BMUs in the sample; {in_register} are in the register; "
        f"{joined['gsp_group'].is_not_null().sum()} have a GSP group."
    )


def zone_group_fractions(
    *, zones: pl.DataFrame, sums: pl.DataFrame, keep_times: pl.Series
) -> pl.DataFrame:
    """Fit the share of each GSP group's import PNs that sits in each zone.

    Every zone's INDDEM is modelled as the sum over GSP groups of the group's summed import PNs
    times a non-negative fraction, and each group's fractions over the 17 zones sum to 1. The
    study fits all zones jointly by non-negative least squares, with the sum-to-one rows weighted
    heavily. If a GSP group lies wholly inside one zone, its fraction there is close to 1.

    Args:
        zones: Every zone's value for every half-hour, from `zones_from_boundaries`.
        sums: The sampled half-hours' PN sums by GSP group, from `group_sums_from_pn`.
        keep_times: The half-hours to fit on.

    Returns:
        One row for every zone and GSP group (or `none`), with the fitted fraction, and the RMS
        residual of the fit in MW.
    """
    wide = (
        sums.filter(pl.col("time").is_in(keep_times.implode()))
        .pivot(on="gsp_group", index="time", values="import_mw")
        .fill_null(0.0)
    )
    target = zones.filter((pl.col("dataset") == "inddem") & (pl.col("view") == "latest")).pivot(
        on="zone", index="time", values="value_mw"
    )
    frame = wide.join(target, on="time", how="inner", validate="1:1").sort("time")
    groups = [column for column in wide.columns if column != "time"]
    design = -frame.select(groups).to_numpy()
    observed = -frame.select(list(ZONES)).to_numpy()
    times, group_count, zone_count = design.shape[0], len(groups), len(ZONES)
    scale = float(np.abs(design).max()) * SUM_TO_ONE_WEIGHT
    matrix = np.zeros((times * zone_count + group_count, group_count * zone_count))
    target_vector = np.zeros(times * zone_count + group_count)
    for zone_index in range(zone_count):
        rows = slice(zone_index * times, (zone_index + 1) * times)
        matrix[rows, zone_index * group_count : (zone_index + 1) * group_count] = design
        target_vector[rows] = observed[:, zone_index]
    for group_index in range(group_count):
        for zone_index in range(zone_count):
            matrix[times * zone_count + group_index, zone_index * group_count + group_index] = scale
        target_vector[times * zone_count + group_index] = scale
    coefficients, _ = nnls(matrix, target_vector)
    fitted = matrix[: times * zone_count] @ coefficients
    rms = float(np.sqrt(np.mean((fitted - target_vector[: times * zone_count]) ** 2)))
    shaped = coefficients.reshape(zone_count, group_count)
    return pl.DataFrame(
        [
            {
                "zone": zone,
                "gsp_group": group,
                "fraction": float(shaped[zone_index, group_index]),
                "rms_residual_mw": rms,
                "half_hours": times,
            }
            for zone_index, zone in enumerate(ZONES)
            for group_index, group in enumerate(groups)
        ]
    )


def leave_one_day_out(
    *, zones: pl.DataFrame, sums: pl.DataFrame, keep_times: pl.Series
) -> pl.DataFrame:
    """Refit the fractions leaving out each sample day in turn, and report each column's top zone.

    Returns:
        One row for each column, with the zone of the largest fraction in the full fit, and the
        number of the full fit and the leave-one-day-out fits that put the largest fraction in that
        zone.
    """
    days = keep_times.dt.convert_time_zone(LONDON).dt.date()
    sample_days = sorted(set(days))
    fits = {"all": zone_group_fractions(zones=zones, sums=sums, keep_times=keep_times)}
    for day in sample_days:
        kept = keep_times.filter(days != day)
        if len(kept):
            fits[f"without {day}"] = zone_group_fractions(zones=zones, sums=sums, keep_times=kept)
    tops = pl.concat(
        [
            fit.sort("fraction", descending=True)
            .group_by("gsp_group", maintain_order=True)
            .first()
            .select("gsp_group", "zone", "fraction")
            .with_columns(fit=pl.lit(name))
            for name, fit in fits.items()
        ]
    )
    full = tops.filter(pl.col("fit") == "all").select("gsp_group", full_zone="zone")
    return (
        tops.join(full, on="gsp_group")
        .group_by("gsp_group")
        .agg(
            top_zone_all=pl.col("full_zone").first(),
            fits=pl.len(),
            fits_agreeing=(pl.col("zone") == pl.col("full_zone")).sum(),
        )
        .sort("gsp_group")
    )


def first_versus_latest(*, views: pl.DataFrame) -> pl.DataFrame:
    """Join the national total's two views, so the page can show how far they differ."""
    national = views.filter(pl.col("boundary") == "N")
    latest = national.filter(pl.col("view") == "latest").select(
        "dataset", "time", latest_mw="value_mw"
    )
    first = national.filter(pl.col("view") == "first_of_day").select(
        "dataset", "time", first_of_day_mw="value_mw"
    )
    return (
        latest.join(first, on=["dataset", "time"])
        .with_columns(
            difference_mw=pl.col("first_of_day_mw") - pl.col("latest_mw"),
            utc_half_hour=pl.col("time").dt.hour().cast(pl.Int32) * 2
            + pl.col("time").dt.minute().cast(pl.Int32) // HALF_HOUR_MINUTES,
            local_half_hour=pl.col("time").dt.convert_time_zone(LONDON).dt.hour().cast(pl.Int32) * 2
            + pl.col("time").dt.convert_time_zone(LONDON).dt.minute().cast(pl.Int32)
            // HALF_HOUR_MINUTES,
            clocks=pl.when(
                pl.col("time").dt.convert_time_zone(LONDON).dt.dst_offset().dt.total_hours() == 0
            )
            .then(pl.lit("GMT"))
            .otherwise(pl.lit("BST")),
        )
        .sort("dataset", "time")
    )


def remove_component(*, series: pl.Series, national: pl.Series) -> np.ndarray:
    """Return `series` minus its least-squares multiple of `national`, as an array.

    Both series are anomalies with mean zero, so the fit needs no intercept.
    """
    values = series.to_numpy()
    reference = national.to_numpy()
    slope = float(np.dot(values, reference) / np.dot(reference, reference))
    return values - slope * reference


def anomaly_correlations(*, agv: pl.DataFrame, zones: pl.DataFrame) -> pl.DataFrame:
    """Correlate each GSP group's AGV with each zone's INDDEM, raw and after removing the cycle.

    The anomaly of a series is the series minus its own mean at the same UK local half-hour of the
    day, the same day type (weekday or weekend), and the same month. Every series shares the daily,
    weekly, and seasonal cycles, so correlations of the raw series sit near 0.76 between any two GSP
    groups, and the anomaly correlation shows what is left. A GSP group's AGV anomaly and a zone's
    INDDEM anomaly also share the national anomaly, which is mostly weather. The study therefore
    also removes from each series the multiple of the national anomaly (the sum of the 14 groups,
    or of the 17 zones) that a least-squares fit gives, and correlates what remains.
    """
    wide_agv = agv.pivot(on="gsp_group", index="time", values="import_mw")
    wide_zones = (
        zones.filter((pl.col("dataset") == "inddem") & (pl.col("view") == "latest"))
        .pivot(on="zone", index="time", values="value_mw")
        .with_columns(-pl.col(zone) for zone in ZONES)
    )
    joined = wide_agv.join(wide_zones, on="time").sort("time")
    local = pl.col("time").dt.convert_time_zone(LONDON)
    keyed = joined.with_columns(
        half_hour=local.dt.hour().cast(pl.Int32) * 2
        + local.dt.minute().cast(pl.Int32) // HALF_HOUR_MINUTES,
        weekend=local.dt.weekday() >= SATURDAY,
        month=local.dt.month(),
    )
    series = [column for column in joined.columns if column != "time"]
    anomalies = keyed.with_columns(
        pl.col(column) - pl.col(column).mean().over("half_hour", "weekend", "month")
        for column in series
    )
    national_agv = anomalies.select(pl.sum_horizontal(list(GSP_GROUPS))).to_series()
    national_inddem = anomalies.select(pl.sum_horizontal(list(ZONES))).to_series()
    without_national = {
        **{
            group: remove_component(series=anomalies[group], national=national_agv)
            for group in GSP_GROUPS
        },
        **{
            zone: remove_component(series=anomalies[zone], national=national_inddem)
            for zone in ZONES
        },
    }
    rows = [
        {
            "gsp_group": group,
            "zone": zone,
            "correlation_raw": joined.select(pl.corr(group, zone)).item(),
            "correlation_anomaly": anomalies.select(pl.corr(group, zone)).item(),
            "correlation_without_national": float(
                np.corrcoef(without_national[group], without_national[zone])[0, 1]
            ),
        }
        for group in GSP_GROUPS
        for zone in ZONES
    ]
    return pl.DataFrame(rows)


def agv_pair_correlations(*, agv: pl.DataFrame) -> dict[str, float]:
    """Return the median raw and anomaly correlation between two GSP groups' AGV."""
    wide = agv.pivot(on="gsp_group", index="time", values="import_mw").sort("time")
    local = pl.col("time").dt.convert_time_zone(LONDON)
    keyed = wide.with_columns(
        half_hour=local.dt.hour().cast(pl.Int32) * 2
        + local.dt.minute().cast(pl.Int32) // HALF_HOUR_MINUTES,
        weekend=local.dt.weekday() >= SATURDAY,
        month=local.dt.month(),
    )
    anomalies = keyed.with_columns(
        pl.col(group) - pl.col(group).mean().over("half_hour", "weekend", "month")
        for group in GSP_GROUPS
    )
    daily_profile = keyed.with_columns(
        pl.col(group) - pl.col(group).mean().over("half_hour") for group in GSP_GROUPS
    )
    pairs = list(itertools.combinations(GSP_GROUPS, 2))
    return {
        "raw": float(np.median([wide.select(pl.corr(a, b)).item() for a, b in pairs])),
        "daily_profile": float(
            np.median([daily_profile.select(pl.corr(a, b)).item() for a, b in pairs])
        ),
        "anomaly": float(np.median([anomalies.select(pl.corr(a, b)).item() for a, b in pairs])),
    }


def build_views_and_zones() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build and report the two views, the zones, the reach of each issue, and the zone checks.

    Returns:
        The views, the zones, and the reach of each issue.
    """
    all_views = []
    reaches = []
    for dataset in ("inddem", "indgen"):
        issues = issues_table(dataset=dataset)
        all_views.append(views_table(dataset=dataset, issues=issues))
        reaches.append(issue_reach(issues=issues, dataset=dataset))
        say(f"- {dataset}: {issues.height} rows, {issues['publish_time'].n_unique()} issues.")
    views = pl.concat(all_views)
    reach = pl.concat(reaches)
    views.write_parquet(STUDY_DIR / "views.parquet")
    reach.write_parquet(STUDY_DIR / "issue_reach.parquet")
    say()
    expected_rows = len(BOUNDARIES) * STUDY_HALF_HOURS * len(VIEWS) * 2
    say(
        f"Views: {views.height} rows, expected {expected_rows} (18 boundaries x {STUDY_HALF_HOURS} "
        f"half-hours x {len(VIEWS)} views x 2 datasets). Rows with no issue: "
        f"{views['value_mw'].null_count()}."
    )
    zones = zones_from_boundaries(views=views)
    zones.write_parquet(STUDY_DIR / "zones.parquet")
    residual = boundary_identity_residual(views=views).drop_nulls()
    say()
    say("## Transcription check: B16 - B11 - (B9 - B17 - B8) = 0")
    say(
        residual.group_by("dataset", "view")
        .agg(
            mean_mw=pl.col("residual_mw").mean(),
            max_abs_mw=pl.col("residual_mw").abs().max(),
            share_nonzero=(pl.col("residual_mw").abs() > 0).mean(),
        )
        .sort("dataset", "view")
    )
    say()
    say("## Zone sign check")
    sign = zone_sign_report(zones=zones)
    sign.write_parquet(STUDY_DIR / "zone_signs.parquet")
    say(sign.filter(pl.col("view") == "latest"))
    report_levels(views=views, zones=zones)
    return views, zones, reach


def report_levels(*, views: pl.DataFrame, zones: pl.DataFrame) -> None:
    """Report the national and zone levels the page quotes, and how many issues are missing."""
    national = views.filter((pl.col("view") == "latest") & (pl.col("boundary") == "N"))
    say()
    say("## National means by month, MW (INDDEM sign reversed)")
    monthly = (
        national.group_by("dataset", month=pl.col("time").dt.truncate("1mo"))
        .agg(mean_mw=pl.col("value_mw").mean())
        .sort("dataset", "month")
    )
    for row in monthly.iter_rows(named=True):
        sign = -1.0 if row["dataset"] == "inddem" else 1.0
        say(f"- {row['dataset']} {row['month']:%Y-%m}: {sign * row['mean_mw']:.0f}")
    overall = national.group_by("dataset").agg(mean_mw=pl.col("value_mw").mean()).sort("dataset")
    for row in overall.iter_rows(named=True):
        say(f"- {row['dataset']} year mean: {abs(row['mean_mw']):.0f}")
    say()
    say("## Zone means, MW, latest view (INDDEM sign reversed)")
    zone_means = (
        zones.filter(pl.col("view") == "latest")
        .group_by("dataset", "zone")
        .agg(mean_mw=pl.col("value_mw").mean())
        .pivot(on="dataset", index="zone", values="mean_mw")
        .with_columns(inddem=-pl.col("inddem"))
        .sort(pl.col("zone").str.slice(1).cast(pl.Int32))
    )
    for row in zone_means.iter_rows(named=True):
        flag = "INDGEN above INDDEM" if row["indgen"] > row["inddem"] else "INDDEM above INDGEN"
        say(f"- {row['zone']}: INDDEM {row['inddem']:.0f}, INDGEN {row['indgen']:.0f} ({flag})")


def report_issues(*, views: pl.DataFrame, reach: pl.DataFrame) -> None:
    """Report how far apart the two views are and how far ahead each issue reaches."""
    fvl = first_versus_latest(views=views)
    fvl.write_parquet(STUDY_DIR / "first_versus_latest.parquet")
    say()
    say("## First issue of the day against latest issue, national total")
    say(
        fvl.group_by("dataset")
        .agg(
            mean_difference_mw=pl.col("difference_mw").mean(),
            mean_abs_difference_mw=pl.col("difference_mw").abs().mean(),
            p90_abs_difference_mw=pl.col("difference_mw").abs().quantile(0.9),
            mean_abs_latest_mw=pl.col("latest_mw").abs().mean(),
        )
        .sort("dataset")
    )
    say()
    say("Mean first-minus-latest difference by UTC target half-hour, last four slots (MW):")
    for row in (
        fvl.filter(pl.col("utc_half_hour") >= LAST_SLOTS_FROM)
        .group_by("dataset", "utc_half_hour")
        .agg(mean_mw=pl.col("difference_mw").mean())
        .sort("dataset", "utc_half_hour")
        .iter_rows(named=True)
    ):
        say(f"- {row['dataset']} slot {row['utc_half_hour']}: {row['mean_mw']:.0f}")
    say()
    say("Mean first-minus-latest difference by UK local target half-hour and clocks, MW:")
    seasonal = (
        fvl.filter(
            pl.col("local_half_hour").is_in(
                [*range(LAST_LOCAL_SLOTS_FROM, 48), *range(FIRST_LOCAL_SLOTS_TO)]
            )
        )
        .group_by("dataset", "clocks", "local_half_hour")
        .agg(mean_mw=pl.col("difference_mw").mean())
        .sort("dataset", "clocks", "local_half_hour")
    )
    for row in seasonal.iter_rows(named=True):
        say(
            f"- {row['dataset']} {row['clocks']} local slot {row['local_half_hour']}: "
            f"{row['mean_mw']:.0f}"
        )
    outside = (
        fvl.filter(
            ~pl.col("local_half_hour").is_in(
                [*range(LAST_LOCAL_SLOTS_FROM, 48), *range(FIRST_LOCAL_SLOTS_TO)]
            )
        )
        .group_by("dataset")
        .agg(
            largest_abs_mean_mw=pl.col("difference_mw").mean().abs().max(),
            mean_abs_mw=pl.col("difference_mw").abs().mean(),
        )
        .sort("dataset")
    )
    say(f"Outside local slots {LAST_LOCAL_SLOTS_FROM} to 47 and 0 to {FIRST_LOCAL_SLOTS_TO - 1}:")
    say(outside)
    say()
    say("## Issue reach, hours ahead")
    say(
        reach.group_by("dataset")
        .agg(
            issues=pl.len(),
            half_hours_min=pl.col("half_hours").min(),
            half_hours_max=pl.col("half_hours").max(),
            reach_hours_min=pl.col("reach_hours").min(),
            reach_hours_max=pl.col("reach_hours").max(),
        )
        .sort("dataset")
    )
    regular = reach.filter((pl.col("dataset") == "inddem") & (pl.col("reach_hours") > 1))
    say(
        f"Reach of the {regular.height} regular INDDEM issues: {regular['reach_hours'].min():.1f} "
        f"to {regular['reach_hours'].max():.1f} hours."
    )
    local_hour = pl.col("publish_time").dt.convert_time_zone(LONDON).dt.hour()
    say("Median reach by UK local hour of publication (hours ahead):")
    say(regular.group_by(hour=local_hour).agg(median=pl.col("reach_hours").median()).sort("hour"))
    slots = (
        regular.select(
            day=pl.col("publish_time").dt.date(), slot=pl.col("publish_time").dt.truncate("30m")
        )
        .unique()
        .group_by("day")
        .agg(slots=pl.len())
    )
    say(f"Slots with an issue in a UTC day: {slots['slots'].value_counts().sort('slots')}")
    local_slots = (
        regular.select(
            slot=(
                pl.col("publish_time").dt.convert_time_zone(LONDON).dt.hour().cast(pl.Int32) * 2
                + pl.col("publish_time").dt.convert_time_zone(LONDON).dt.minute().cast(pl.Int32)
                // HALF_HOUR_MINUTES
            )
        )
        .unique()
        .sort("slot")["slot"]
        .to_list()
    )
    missing_local = [slot for slot in range(48) if slot not in local_slots]
    say(f"UK local half-hour slots (0 to 47) that never hold a regular issue: {missing_local}")
    ends = (
        regular.select(
            end_local=pl.col("end_time").dt.convert_time_zone(LONDON).dt.strftime("%H:%M"),
            long_issue=pl.col("reach_hours") > LONG_ISSUE_HOURS,
        )
        .group_by("long_issue", "end_local")
        .agg(issues=pl.len())
        .sort("long_issue", "end_local")
    )
    say("End of the last half-hour of an issue, UK local time (long issue means over 30 hours):")
    say(ends)
    odd = reach.filter(pl.col("reach_hours") < 1)
    say(f"Issues that reach less than one hour ahead: {odd.height}.")
    for row in odd.iter_rows(named=True):
        say(
            f"- {row['dataset']} issue published {row['publish_time']:%Y-%m-%d %H:%M} UTC holds "
            f"{row['half_hours']} half-hours that end {row['reach_hours']:.1f} hours "
            "after publication."
        )


def report_nged_levels(
    *, agv: pl.DataFrame, zones: pl.DataFrame, correlations: pl.DataFrame
) -> None:
    """Report, for each NGED group, its best-correlated zone, the mean levels, and the solar share.

    The means use only the half-hours that AGV, the zone, and PV_Live all cover, so the ratios and
    the shares describe one window.
    """
    zone_inddem = (
        zones.filter((pl.col("dataset") == "inddem") & (pl.col("view") == "latest"))
        .select("zone", "time", zone_mw=-pl.col("value_mw"))
        .sort("zone", "time")
    )
    pv = pl.read_parquet(PV_LIVE_PATH).select("time", "gsp_group", solar_mw="generation_mw")
    rows = []
    for group, name in NGED_GROUPS.items():
        best = correlations.filter(pl.col("gsp_group") == group).sort(
            "correlation_without_national", descending=True
        )
        zone = best["zone"][0]
        joined = (
            agv.filter(pl.col("gsp_group") == group)
            .join(zone_inddem.filter(pl.col("zone") == zone), on="time", validate="1:1")
            .join(pv.filter(pl.col("gsp_group") == group), on=["time", "gsp_group"], validate="1:1")
        )
        rows.append(
            {
                "gsp_group": group,
                "name": name,
                "best_zone": zone,
                "correlation": best["correlation_without_national"][0],
                "correlation_anomaly": best["correlation_anomaly"][0],
                "half_hours": joined.height,
                "mean_agv_mw": joined["import_mw"].mean(),
                "mean_zone_inddem_mw": joined["zone_mw"].mean(),
                "zone_to_agv_ratio": joined.select(
                    pl.col("zone_mw").mean() / pl.col("import_mw").mean()
                ).item(),
                "mean_solar_mw": joined["solar_mw"].mean(),
                "solar_over_agv": joined.select(
                    pl.col("solar_mw").sum() / pl.col("import_mw").sum()
                ).item(),
                "solar_share_of_gross": joined.select(
                    pl.col("solar_mw").sum() / (pl.col("import_mw") + pl.col("solar_mw")).sum()
                ).item(),
                "negative_agv_half_hours": int((joined["import_mw"] < 0).sum()),
            }
        )
    table = pl.DataFrame(rows)
    table.write_parquet(STUDY_DIR / "nged_levels.parquet")
    say()
    say("## NGED groups: best-correlated zone, mean levels (MW), and solar share")
    for row in table.iter_rows(named=True):
        say(
            f"- {row['gsp_group']} {row['name']}: best zone {row['best_zone']} "
            f"(correlation without the national anomaly {row['correlation']:.2f}; anomaly "
            f"correlation {row['correlation_anomaly']:.2f}); mean AGV {row['mean_agv_mw']:.0f}, "
            "mean zone "
            f"INDDEM {row['mean_zone_inddem_mw']:.0f}, ratio {row['zone_to_agv_ratio']:.2f}; "
            f"mean solar {row['mean_solar_mw']:.0f}, solar over AGV {row['solar_over_agv']:.3f}, "
            f"solar over AGV plus solar "
            f"{row['solar_share_of_gross']:.3f}; AGV negative in "
            f"{row['negative_agv_half_hours']} of {row['half_hours']} half-hours."
        )


def report_national_against_indo(*, views: pl.DataFrame) -> None:
    """Report how national INDDEM compares with the national demand outturn (INDO)."""
    national = views.filter(
        (pl.col("view") == "latest") & (pl.col("boundary") == "N") & (pl.col("dataset") == "inddem")
    ).select("time", inddem_mw=-pl.col("value_mw"))
    indo = pl.read_parquet(INDO_PATH).select("time", "indo_mw").unique("time")
    joined = national.join(indo, on="time", validate="1:1").with_columns(
        ratio=pl.col("inddem_mw") / pl.col("indo_mw")
    )
    say()
    say("## National INDDEM against INDO")
    say(
        joined.select(
            half_hours=pl.len(),
            median_ratio=pl.col("ratio").median(),
            correlation=pl.corr("inddem_mw", "indo_mw"),
        )
    )


def supplier_import_share() -> float:
    """Return the share of the sampled import PNs that supplier base BMUs (`2__`) hold."""
    averages = period_average_mw(segments=pl.read_parquet(PN_SAMPLE_PATH))
    register = (
        pl.read_parquet(BMU_REFERENCE_PATH)
        .select("national_grid_bmu_id", "elexon_bmu_id")
        .unique("national_grid_bmu_id")
    )
    imports = averages.filter(pl.col("average_mw") < 0).join(
        register, on="national_grid_bmu_id", how="left"
    )
    return imports.select(
        pl.col("average_mw").filter(pl.col("elexon_bmu_id").str.starts_with("2__")).sum()
        / pl.col("average_mw").sum()
    ).item()


def build_agv(*, views: pl.DataFrame, zones: pl.DataFrame) -> pl.DataFrame:
    """Build and report the AGV tables, the unit check, and the correlations with the zones."""
    agv = agv_import_mw()
    agv.write_parquet(STUDY_DIR / "agv_groups.parquet")
    correlations = anomaly_correlations(agv=agv, zones=zones)
    correlations.write_parquet(STUDY_DIR / "anomaly_correlations.parquet")
    pair = agv_pair_correlations(agv=agv)
    say()
    say("## Correlation between GSP groups' AGV, median of the 91 pairs")
    say(
        f"raw {pair['raw']:.3f}, minus the mean daily profile {pair['daily_profile']:.3f}, "
        f"anomaly {pair['anomaly']:.3f}"
    )
    for column, label in (
        ("correlation_anomaly", "anomaly correlation"),
        ("correlation_without_national", "correlation with the national anomaly removed"),
    ):
        say(f"Highest {label} of each GSP group with a zone's INDDEM (top three zones):")
        for group in GSP_GROUPS:
            top = correlations.filter(pl.col("gsp_group") == group).sort(column, descending=True)
            best = ", ".join(
                f"{row['zone']} {row[column]:.2f}" for row in top.head(3).iter_rows(named=True)
            )
            say(f"- {group}{' ' + NGED_GROUPS[group] if group in NGED_GROUPS else ''}: {best}")
    say(
        f"Over all {correlations.height} pairs of a GSP group and a zone, the median correlation "
        "is "
        f"{correlations['correlation_raw'].median():.2f} for the raw series and "
        f"{correlations['correlation_anomaly'].median():.2f} for the anomalies, and "
        f"{correlations['correlation_without_national'].median():.2f} with the national anomaly "
        "removed."
    )
    for column in ("correlation_anomaly", "correlation_without_national"):
        top = correlations.sort(column, descending=True).row(0, named=True)
        say(f"The highest {column} is {top[column]:.2f}, {top['gsp_group']} with {top['zone']}.")
    report_nged_levels(agv=agv, zones=zones, correlations=correlations)
    report_national_against_indo(views=views)
    check = agv_against_indo(agv=agv)
    check.write_parquet(STUDY_DIR / "agv_against_indo.parquet")
    say()
    say("## AGV against INDO")
    say(
        check.select(
            n=pl.len(),
            median_ratio=pl.col("ratio").median(),
            p01=pl.col("ratio").quantile(0.01),
            p99=pl.col("ratio").quantile(0.99),
            correlation=pl.corr("agv_mw", "indo_mw"),
        )
    )
    say(f"AGV windows: SF rows to {AGV_END}; groups per time: {check['groups'].unique().to_list()}")
    by_hour = (
        check.group_by(hour=pl.col("time").dt.hour())
        .agg(median_ratio=pl.col("ratio").median())
        .sort("hour")
    )
    say(
        f"Median ratio by UTC hour of day: {by_hour['median_ratio'].min():.3f} to "
        f"{by_hour['median_ratio'].max():.3f}."
    )
    say(
        f"GSP groups in AGV: {sorted(agv['gsp_group'].unique().to_list())}; "
        f"expected {list(GSP_GROUPS)}"
    )
    say(f"NGED groups: {NGED_GROUPS}")
    estimate = agv.height
    flagged = pl.read_parquet(AGV_PATH).select((pl.col("estimate_indicator") == "T").mean()).item()
    say(f"AGV rows with Estimate Indicator T: {flagged:.3f} of {estimate} rows.")
    pv = pl.read_parquet(PV_LIVE_PATH)
    say(f"PV_Live rows {pv.height}, groups {sorted(pv['gsp_group'].unique().to_list())}")
    summer = pv.filter(
        (pl.col("gsp_group") == "_L") & pl.col("time").dt.month().is_in([6, 7])
    ).with_columns(hour=pl.col("time").dt.hour().cast(pl.Float64) + pl.col("time").dt.minute() / 60)
    centroid = summer.select(
        (pl.col("hour") * pl.col("generation_mw")).sum() / pl.col("generation_mw").sum()
    ).item()
    say(
        f"PV_Live South West, June and July: generation-weighted mean start of the half-hour "
        f"{centroid:.2f} h UTC, so the centre of the half-hour is {centroid + 0.25:.2f} h UTC."
    )
    return agv


def build_pn_fit(*, views: pl.DataFrame, zones: pl.DataFrame) -> None:
    """Sum the sampled PNs by GSP group, compare them with INDDEM, and fit the zone weights."""
    sums = group_sums_from_pn()
    sums.write_parquet(STUDY_DIR / "pn_group_sums.parquet")
    say()
    say(f"## Sampled PNs: {sums.height} rows of group sums")
    national = views.filter((pl.col("view") == "latest") & (pl.col("boundary") == "N")).pivot(
        on="dataset", index="time", values="value_mw"
    )
    totals = sums.group_by("time").agg(
        pn_import_mw=pl.col("import_mw").sum(), pn_export_mw=pl.col("export_mw").sum()
    )
    joined = totals.join(national, on="time")
    joined.write_parquet(STUDY_DIR / "pn_against_inddem.parquet")
    say(
        joined.select(
            half_hours=pl.len(),
            inddem_minus_pn_median_mw=(pl.col("inddem") - pl.col("pn_import_mw")).median(),
            inddem_minus_pn_max_abs_mw=(pl.col("inddem") - pl.col("pn_import_mw")).abs().max(),
            indgen_minus_pn_median_mw=(pl.col("indgen") - pl.col("pn_export_mw")).median(),
            indgen_minus_pn_max_abs_mw=(pl.col("indgen") - pl.col("pn_export_mw")).abs().max(),
        )
    )
    reproduced = joined.filter(
        (pl.col("inddem") - pl.col("pn_import_mw")).abs() <= INDDEM_MATCH_TOLERANCE_MW
    )["time"]
    say(
        f"Half-hours where the PN import sum reproduces INDDEM's national total to within "
        f"{INDDEM_MATCH_TOLERANCE_MW:g} MW: {len(reproduced)} of {joined.height}."
    )
    per_day = (
        joined.with_columns(
            day=pl.col("time").dt.convert_time_zone(LONDON).dt.date(),
            reproduced=(pl.col("inddem") - pl.col("pn_import_mw")).abs()
            <= INDDEM_MATCH_TOLERANCE_MW,
            local_hour=pl.col("time").dt.convert_time_zone(LONDON).dt.hour(),
        )
        .group_by("day")
        .agg(
            half_hours=pl.len(),
            reproduced=pl.col("reproduced").sum(),
            first_local_hour=pl.col("local_hour").filter(pl.col("reproduced")).min(),
            last_local_hour=pl.col("local_hour").filter(pl.col("reproduced")).max(),
        )
        .sort("day")
    )
    say("Reproduced half-hours by UK local day:")
    say(per_day)
    gaps = joined.select(
        inddem_gap=pl.col("inddem") - pl.col("pn_import_mw"),
        indgen_gap=pl.col("indgen") - pl.col("pn_export_mw"),
    )
    both = ((pl.col("inddem_gap") > 5) & (pl.col("indgen_gap") < -5)).mean()
    say(
        "Correlation of the INDDEM gap with the INDGEN gap: "
        f"{gaps.select(pl.corr('inddem_gap', 'indgen_gap')).item():.2f}; share of half-hours with "
        "an INDDEM gap above 5 MW and an INDGEN gap below -5 MW: "
        f"{gaps.select(both).item():.2f}"
    )
    fractions = zone_group_fractions(zones=zones, sums=sums, keep_times=reproduced)
    fractions.write_parquet(STUDY_DIR / "zone_group_fractions.parquet")
    report_bmu_group_coverage()
    say()
    say(
        f"## Fitted fraction of each GSP group's import PNs in each zone ({len(reproduced)} "
        f"half-hours, RMS residual {fractions['rms_residual_mw'][0]:.0f} MW)"
    )
    say(
        fractions.pivot(on="gsp_group", index="zone", values="fraction")
        .with_columns(pl.exclude("zone").round(2))
        .sort(pl.col("zone").str.slice(1).cast(pl.Int32))
    )
    stability = leave_one_day_out(zones=zones, sums=sums, keep_times=reproduced)
    say()
    say(
        "Leave-one-day-out check: zone of the largest fraction of each column, full fit versus fits"
    )
    say("without each sample day:")
    say(stability)
    say(f"Supplier base BMUs (2__) hold {supplier_import_share():.3f} of the sampled import PNs.")
    sums_by_group = sums.group_by("gsp_group").agg(
        mean_import_mw=pl.col("import_mw").mean(), mean_export_mw=pl.col("export_mw").mean()
    )
    say()
    say("Mean summed PNs by GSP group, MW:")
    say(sums_by_group.sort("gsp_group"))


def main() -> None:
    """Build every table and write the report."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(240)
    say("# INDDEM, INDGEN, and GSP-group take: numbers the page quotes")
    say()
    views, zones, reach = build_views_and_zones()
    report_issues(views=views, reach=reach)
    build_agv(views=views, zones=zones)
    build_pn_fit(views=views, zones=zones)
    (STUDY_DIR / "report.md").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
