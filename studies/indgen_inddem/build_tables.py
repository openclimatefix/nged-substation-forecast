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
    INDDEM_PATH,
    INDGEN_PATH,
    INDO_PATH,
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
WINDOW_START_UTC: Final[datetime] = datetime(
    STUDY_START.year, STUDY_START.month, STUDY_START.day, tzinfo=UTC
)
WINDOW_AFTER_UTC: Final[datetime] = datetime(
    STUDY_END.year, STUDY_END.month, STUDY_END.day, tzinfo=UTC
) + timedelta(days=1)
"""Midnight after the last day of the study window."""
UTC_TIME: Final[pl.Datetime] = pl.Datetime(time_unit="us", time_zone="UTC")

report_lines: list[str] = []


def say(text: str = "") -> None:
    """Add a line to the report."""
    report_lines.append(text)


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
            warnings.simplefilter("ignore")
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


def group_sums_from_pn() -> pl.DataFrame:
    """Sum the sampled BMUs' average PN by GSP group, split into import and export."""
    averages = period_average_mw(segments=pl.read_parquet(PN_SAMPLE_PATH))
    groups = pl.read_parquet(BMU_REFERENCE_PATH).select(
        "national_grid_bmu_id", gsp_group="gsp_group_id"
    )
    return (
        averages.join(groups, on="national_grid_bmu_id", how="left")
        .with_columns(gsp_group=pl.col("gsp_group").fill_null("none"))
        .group_by("time", "gsp_group")
        .agg(
            import_mw=pl.col("average_mw").filter(pl.col("average_mw") < 0).sum(),
            export_mw=pl.col("average_mw").filter(pl.col("average_mw") > 0).sum(),
            bmus=pl.len(),
        )
        .sort("time", "gsp_group")
    )


def zone_group_weights(
    *, zones: pl.DataFrame, sums: pl.DataFrame, dataset: DatasetType
) -> pl.DataFrame:
    """Fit each zone's INDDEM or INDGEN on the GSP groups' summed PNs, with non-negative weights.

    Each BMU belongs to one GSP group in Elexon's BMU register, so if a GSP group lies wholly
    inside one zone, the weight of that group in that zone's fit is close to 1 and every other
    weight is close to 0. A group that straddles two zones shows two fractional weights.

    Args:
        zones: Every zone's value for every half-hour, from `zones_from_boundaries`.
        sums: The sampled half-hours' PN sums by GSP group, from `group_sums_from_pn`.
        dataset: Whether to fit INDDEM on the import sums, or INDGEN on the export sums.

    Returns:
        One row for every zone and GSP group (or `none`, the BMUs the register gives no group),
        with the fitted weight, and the zone's fit in MW.
    """
    side = "import_mw" if dataset == "inddem" else "export_mw"
    wide = sums.pivot(on="gsp_group", index="time", values=side).fill_null(0.0).sort("time")
    groups = [column for column in wide.columns if column != "time"]
    design = wide.select(groups).to_numpy()
    target = (
        zones.filter((pl.col("dataset") == dataset) & (pl.col("view") == "latest"))
        .pivot(on="zone", index="time", values="value_mw")
        .join(wide.select("time"), on="time", how="semi")
        .sort("time")
    )
    rows = []
    for zone in ZONES:
        values = target[zone].to_numpy()
        coefficients, norm = nnls(
            design * (-1.0 if dataset == "inddem" else 1.0),
            values * (-1.0 if dataset == "inddem" else 1.0),
        )
        rms = norm / np.sqrt(len(values))
        rows.extend(
            {
                "dataset": dataset,
                "zone": zone,
                "gsp_group": group,
                "weight": float(weight),
                "rms_residual_mw": float(rms),
                "zone_rms_mw": float(np.sqrt(np.mean(values**2))),
            }
            for group, weight in zip(groups, coefficients, strict=True)
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Build every table and write the report."""
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    say("# INDDEM, INDGEN, and GSP-group take: numbers the page quotes")
    say()
    all_views = []
    reaches = []
    for dataset in ("inddem", "indgen"):
        issues = issues_table(dataset=dataset)
        all_views.append(views_table(dataset=dataset, issues=issues))
        reaches.append(issue_reach(issues=issues, dataset=dataset))
        say(f"- {dataset}: {issues.height} rows, {issues['publish_time'].n_unique()} issues.")
    views = pl.concat(all_views)
    views.write_parquet(STUDY_DIR / "views.parquet")
    pl.concat(reaches).write_parquet(STUDY_DIR / "issue_reach.parquet")
    say()
    say(f"Views: {views.height} rows. Rows with no issue: {views['value_mw'].null_count()}.")

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
        .__str__()
    )
    say()
    say("## Zone sign check")
    sign = zone_sign_report(zones=zones)
    sign.write_parquet(STUDY_DIR / "zone_signs.parquet")
    say(sign.filter(pl.col("view") == "latest").__str__())

    agv = agv_import_mw()
    agv.write_parquet(STUDY_DIR / "agv_groups.parquet")
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
        ).__str__()
    )
    say(f"AGV windows: SF rows to {AGV_END}; groups per time: {check['groups'].unique().to_list()}")
    say()
    say(
        f"GSP groups in AGV: {sorted(agv['gsp_group'].unique().to_list())}; "
        f"expected {list(GSP_GROUPS)}"
    )
    say(f"NGED groups: {NGED_GROUPS}")
    pv = pl.read_parquet(PV_LIVE_PATH)
    say(f"PV_Live rows {pv.height}, groups {sorted(pv['gsp_group'].unique().to_list())}")
    if PN_SAMPLE_PATH.exists():
        sums = group_sums_from_pn()
        sums.write_parquet(STUDY_DIR / "pn_group_sums.parquet")
        say(f"PN group sums: {sums.height} rows")
        weights = pl.concat(
            [zone_group_weights(zones=zones, sums=sums, dataset=d) for d in ("inddem", "indgen")]
        )
        weights.write_parquet(STUDY_DIR / "zone_group_weights.parquet")

    (STUDY_DIR / "report.md").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
