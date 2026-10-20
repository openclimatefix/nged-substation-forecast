"""Draw the figures of experiment 111's write-up from `work/results.json` and the stored files.

Every figure carries its own scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder, PNGs to a scratch folder for the render check.

    uv run python ml-experiments/00111-*/e00111_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00111_arms as arms
import e00111_common as common
import e00111_dp_run as dp_run
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
ARM_LABELS: Final[dict[str, str]] = {
    "gamk": "GAM-K (DP plus Kalman filter)",
    "gam0": "GAM-0 (DP, no correction)",
    "gam_margin": "GAM-margin (XGBoost with DP start)",
    "xnl": "X-nl (lagless XGBoost)",
    "xnl_k": "X-nl+K (lagless XGBoost plus filter)",
    "xlag_d": "Xlag-d (lagged XGBoost)",
    "bg2": "BG2 (production recipe)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "gamk": ocf.BRAND_ORANGE,
    "gam0": ocf.DATA_SKY,
    "gam_margin": ocf.DATA_PURPLE,
    "xnl": ocf.DATA_BLUE,
    "xnl_k": ocf.DATA_BLUE_LIGHT,
    "xlag_d": ocf.DATA_GREEN,
    "bg2": DARK,
    "climatology": "#C8C8C8",
    "manual_heuristic": GREY,
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def results() -> dict:
    """Return the stored results."""
    return json.loads((common.WORK_DIR / "results.json").read_text())


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts with 98.75% intervals, and the exploratory ones beside them."""
    rows = []
    for item in data["contrasts"]:
        if item["sites"] != "decision":
            continue
        q = item["quantile_score"]
        rows.append(
            {
                "contrast": item["name"],
                "estimate": q["estimate"],
                "low": q["lower"],
                "high": q["upper"],
                "kind": "planned (98.75% interval)"
                if item["planned"]
                else "exploratory (95% interval)",
                "label": f"{q['sites_better']} of {q['n_sites']} sites better",
            }
        )
    table = pl.DataFrame(rows).with_columns(label_x=pl.max_horizontal("estimate", "high"))
    order = table["contrast"].to_list()
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["planned (98.75% interval)", "exploratory (95% interval)"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=6, fontSize=10)
        .encode(y=alt.Y("contrast:N", sort=order), x="label_x:Q", text="label:N")
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=WIDTH - 250,
        height=26 * len(order),
        title=_title(
            text="Planned and exploratory contrasts of the convex model and XGBoost",
            subtitle=(
                "Dot: difference in mean quantile score over the five decision substations (D5, "
                "D10, "
                "D11, D14, D16), leads 0.5 to 168 hours, rows outside the switching events, scored "
                "year 1 July 2025 to 30 June 2026. Line: interval from resampling whole calendar "
                "months. Dashed line: no difference. All weather is ECMWF ensemble forecasts."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Mean quantile score against lead day for the headline arms."""
    rows = [r for r in data["headline_scores_decision"] if r["site"] is not None]
    table = (
        pl.DataFrame(rows)
        .group_by("arm", "day")
        .agg(qs=(pl.col("qs") * pl.col("n")).sum() / pl.col("n").sum())
        .with_columns(label=pl.col("arm").replace(ARM_LABELS))
    )
    domain = [a for a in ARM_LABELS if a in set(table["arm"].to_list())]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("day:O", title="Lead (days after the 09:00 UTC forecast origin)"),
            y=alt.Y(
                "qs:Q",
                title="Mean quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in domain], range=[ARM_COLOURS[a] for a in domain]
                ),
                legend=alt.Legend(
                    title=None, orient="top", columns=2, labelLimit=330, symbolType="stroke"
                ),
            ),
        )
        .properties(width=WIDTH - 60, height=260)
    )
    return chart.properties(
        title=_title(
            text="Every arm scores worse as the lead grows",
            subtitle=(
                "Mean quantile score over the five decision substations by day of lead, on the "
                "rows "
                "every arm holds, scored year, rows outside the switching events."
            ),
        )
    )


def figure_by_site(*, data: dict) -> alt.TopLevelMixin:
    """Mean quantile score per decision substation and arm."""
    rows = list(data["headline_scores_decision"])
    table = (
        pl.DataFrame(rows)
        .group_by("arm", "site")
        .agg(qs=(pl.col("qs") * pl.col("n")).sum() / pl.col("n").sum())
        .with_columns(label=pl.col("arm").replace(ARM_LABELS))
    )
    cache = common.load_cache()
    label_of = {cache.sites[s].table.time_series_id: s for s in common.DECISION_SITES}
    table = table.with_columns(site_label=pl.col("site").replace_strict(label_of, default=None))
    domain = [a for a in ARM_LABELS if a in set(table["arm"].to_list())]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y("site_label:N", sort=list(common.DECISION_SITES), title=None),
            x=alt.X("qs:Q", title="Mean quantile score (points of capacity; smaller is better)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in domain], range=[ARM_COLOURS[a] for a in domain]
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(width=WIDTH - 60, height=190)
    )
    return chart.properties(
        title=_title(
            text="The ranking of the arms differs from substation to substation",
            subtitle=(
                "Mean quantile score by substation and arm, leads 0.5 to 168 hours, on the rows "
                "every "
                "arm holds, scored year, rows outside the switching events."
            ),
        )
    )


def figure_calibration() -> alt.TopLevelMixin:
    """The fitted temperature shift and irradiance ratio by lead band, one line per season."""
    lead_labels = ["0-24 h", "24-48 h", "48-144 h", "144-168 h"]
    season_labels = ["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"]
    rows = []
    for site in common.DECISION_SITES:
        calibration = arms.load_calibration(site=site)
        for season, season_label in enumerate(season_labels):
            for lead, lead_label in enumerate(lead_labels):
                rows.append(
                    {
                        "site": site,
                        "season": season_label,
                        "lead": lead_label,
                        "shift": float(np.mean(calibration.temperature_shift_c[season, lead])),
                        "ratio": float(np.mean(calibration.irradiance_ratio[season, lead, 1:])),
                    }
                )
    table = (
        pl.DataFrame(rows)
        .group_by("season", "lead")
        .agg(pl.col("shift").mean(), pl.col("ratio").mean())
    )

    def panel(column: str, title: str) -> alt.Chart:
        return (
            alt.Chart(table)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, strokeWidth=2)
            .encode(
                x=alt.X("lead:N", sort=lead_labels, title="Lead band"),
                y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "season:N",
                    scale=alt.Scale(
                        domain=season_labels,
                        range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                    ),
                    legend=alt.Legend(title="Season", orient="top", direction="horizontal"),
                ),
            )
            .properties(width=WIDTH - 60, height=140)
        )

    return alt.vconcat(
        panel("shift", "Added to ensemble temperature (K)"),
        panel("ratio", "Multiplies ensemble irradiance"),
    ).properties(
        title=_title(
            text="The ensemble runs close to the weather the convex model was trained on",
            subtitle=(
                "Calibration fitted on the odd months of 2024-07 to 2025-06: the mean over the "
                "five "
                "decision substations of the temperature shift (top) and the irradiance ratio in "
                "daylight hour bands (bottom), by season and lead band. 0 and 1 mean no change."
            ),
        )
    )


def figure_gate() -> alt.TopLevelMixin:
    """The known-answer gate's readings as pass or fail dots."""
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    rows = [
        {
            "check": f"{c['test']}: {c['check']}"[:78],
            "value": c["value"],
            "limit": c["limit"],
            "passed": "passed" if c["passed"] else "failed",
        }
        for c in gate["checks"]
        if not c["test"].startswith("2")
    ]
    table = pl.DataFrame(rows).with_columns(pl.col("limit").cast(pl.String))
    order = table["check"].to_list()
    base = alt.Chart(table).encode(
        y=alt.Y("check:N", sort=order, title=None, axis=alt.Axis(labelLimit=380))
    )
    dots = base.mark_point(filled=True, size=80).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q", title="Reading (units differ by row; the limit is printed beside each dot)"
        ),
        color=alt.Color(
            "passed:N",
            scale=alt.Scale(domain=["passed", "failed"], range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE]),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    text = base.mark_text(align="left", dx=8, fontSize=10).encode(x="value:Q", text="limit:N")  # ty: ignore[unresolved-attribute]
    return (dots + text).properties(
        width=WIDTH - 340,
        height=24 * len(order),
        title=_title(
            text="The known-answer gate passed every reading",
            subtitle=(
                "Each dot is one blocking reading of the gate that ran on the tuning months before "
                "any replicate or scored run. The irradiance-timing readings at each substation "
                "are "
                "in the write-up's text."
            ),
        ),
    )


def figure_tails(*, data: dict) -> alt.TopLevelMixin:
    """Coverage minus nominal at the four tail levels, per arm."""
    table = pl.DataFrame(data["tails_decision"]).with_columns(
        error=100.0 * (pl.col("coverage") - pl.col("nominal")),
        label=pl.col("arm").replace(ARM_LABELS),
    )
    table = table.filter(pl.col("arm").is_in(list(ARM_LABELS)))
    order = [ARM_LABELS[a] for a in ARM_LABELS if ARM_LABELS[a] in set(table["label"].to_list())]
    levels = ["q01", "q05", "q95", "q99"]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=45)
        .encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("error:Q", title="Coverage minus nominal (percentage points; 0 is calibrated)"),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(
                    domain=levels,
                    range=[ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title="Quantile level", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 280, height=22 * len(order))
    )
    zero = alt.Chart(table).mark_rule(color=GREY, strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    return (
        (zero + chart)
        .facet(row=alt.Row("lead_band:N", title=None))
        .properties(
            title=_title(
                text="Tail calibration of every arm",
                subtitle=(
                    "Share of observations at or below each tail quantile, minus the nominal "
                    "share, over "
                    "the five decision substations, scored year, by lead band. A dot to the right "
                    "of the "
                    "dashed line means the quantile sits too high."
                ),
            )
        )
    )


def figure_correction() -> alt.TopLevelMixin:
    """The filter's level and transient by lead day for GAM-K on ensemble and on ERA5 residuals."""
    pieces = []
    for arm in ("gamk", "gamk_era5"):
        for site in common.DECISION_SITES:
            frame = pl.read_parquet(dp_run.out_path(arm=arm, period="scored", site=site)).filter(
                ~pl.col("switching_mask")
            )
            pieces.append(
                frame.select(
                    arm=pl.lit(arm),
                    day=((pl.col("lead_hours") - 1e-9) // 24).cast(pl.Int32) + 1,
                    level=pl.col("level").abs(),
                    anomaly=pl.col("anomaly").abs(),
                )
            )
    table = (
        pl.concat(pieces)
        .group_by("arm", "day")
        .agg(pl.col("level").mean(), pl.col("anomaly").mean())
        .unpivot(index=["arm", "day"], variable_name="part", value_name="size")
    )
    table = table.with_columns(
        series=pl.concat_str(
            pl.col("arm").replace(
                {"gamk": "ensemble residuals", "gamk_era5": "ERA5 and CAMS residuals"}
            ),
            pl.col("part").replace({"level": "persistent level", "anomaly": "decaying transient"}),
            separator=", ",
        )
    )
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("day:O", title="Lead (days after the 09:00 UTC forecast origin)"),
            y=alt.Y("size:Q", title="Mean absolute correction (points of capacity)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=220,
            title=_title(
                text="The filter's correction is carried by its persistent level",
                subtitle=(
                    "Mean size of the persistent level and of the decaying transient that the "
                    "Kalman "
                    "filter adds to the convex model's forecast, by lead day, over the five "
                    "decision "
                    "substations, scored year."
                ),
            ),
        )
    )


WEEK_ARMS: Final[tuple[str, ...]] = ("gamk", "gam_margin", "xnl", "bg2")
SEASONS: Final[dict[str, tuple[int, ...]]] = {"winter": (11, 12, 1, 2), "summer": (6, 7, 8)}


def _loaded_for_weeks() -> dict[str, pl.DataFrame]:
    """Load the arms of the weeks figure on the rows they share, at the decision substations."""
    import e00111_report as report
    import e00111_score as score

    loaded = report.load_all(periods={"scored"}, sites=common.DECISION_SITES)
    present = {arm: loaded[arm] for arm in WEEK_ARMS if arm in loaded}
    return score.common_rows(arms_by_name=present)


def figure_weeks(*, season: str, shared: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Actual power and the arms' median forecasts for the week of largest swing in a season.

    The week of each substation is the 7-day block of the scored year, inside the season's months,
    in which the observed power ranges most between its lowest and highest half-hour.
    """
    cache = common.load_cache()
    label_of = {cache.sites[s].table.time_series_id: s for s in common.DECISION_SITES}
    rows = []
    for site_id, site in label_of.items():
        reference = shared["gamk"].filter(pl.col("time_series_id") == site_id)
        daily = (
            reference.filter(pl.col("valid_time").dt.month().is_in(list(SEASONS[season])))
            .with_columns(week=pl.col("valid_time").dt.truncate("1w"))
            .group_by("week")
            .agg(swing=pl.col("actual").max() - pl.col("actual").min(), n=pl.len())
            .filter(pl.col("n") > 300)
            .sort("swing", descending=True)
        )
        week = daily["week"][0]
        for arm, frame in shared.items():
            part = frame.filter(
                (pl.col("time_series_id") == site_id)
                & (pl.col("valid_time") >= week)
                & (pl.col("valid_time") < week + pl.duration(days=7))
            )
            for panel, (low, high) in (
                ("1 day ahead", (15.0, 39.0)),
                ("6 days ahead", (111.0, 135.0)),
            ):
                chosen = part.filter((pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high))
                rows += [
                    {
                        "site": site,
                        "panel": panel,
                        "time": r["valid_time"],
                        "series": ARM_LABELS[arm],
                        "power": r["q50"],
                    }
                    for r in chosen.select("valid_time", "q50").iter_rows(named=True)
                ]
                if arm == "gamk":
                    rows += [
                        {
                            "site": site,
                            "panel": panel,
                            "time": r["valid_time"],
                            "series": "Observed power",
                            "power": r["actual"],
                        }
                        for r in chosen.select("valid_time", "actual").iter_rows(named=True)
                    ]
    table = pl.DataFrame(rows)
    domain = ["Observed power", *[ARM_LABELS[a] for a in WEEK_ARMS if a in shared]]
    colours = [DARK, *[ARM_COLOURS[a] for a in WEEK_ARMS if a in shared]]
    colours[1:] = [c if c != DARK else GREY for c in colours[1:]]
    chart = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.3)
        .encode(
            x=alt.X("time:T", title=None, axis=alt.Axis(format="%d %b", tickCount=7)),
            y=alt.Y("power:Q", title=None, scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=330),
            ),
        )
        .properties(width=250, height=70)
    )
    return (
        chart.facet(
            row=alt.Row("site:N", sort=list(common.DECISION_SITES), title=None),
            column=alt.Column("panel:N", sort=["1 day ahead", "6 days ahead"], title=None),
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text=f"A {season} week at each decision substation: power and four forecasts",
                subtitle=(
                    "Power as a percentage of the substation's effective capacity. Median forecast "
                    "made one day ahead (left) and six days ahead (right) of each half-hour, "
                    "against the observed power. Each substation's week is the 7-day block "
                    f"of the scored year in the {season} months with the largest swing in "
                    "observed power."
                ),
            )
        )
    )


def main() -> None:
    """Draw every figure."""
    data = results()
    shared = _loaded_for_weeks()
    _save(figure_weeks(season="winter", shared=shared), name="figure_01a_winter_week")
    _save(figure_weeks(season="summer", shared=shared), name="figure_01b_summer_week")
    _save(figure_headline(data=data), name="figure_02_headline")
    _save(figure_by_lead(data=data), name="figure_03_by_lead")
    _save(figure_by_site(data=data), name="figure_04_by_site")
    _save(figure_correction(), name="figure_05_correction")
    _save(figure_calibration(), name="figure_06_calibration")
    _save(figure_tails(data=data), name="figure_07_tails")
    _save(figure_gate(), name="figure_08_gate")


if __name__ == "__main__":
    main()
