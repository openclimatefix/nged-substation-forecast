"""Draw the figures of the UKV-from-CEDA against UKV-from-Open-Meteo study from the saved tables.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. It reads
`intervals.parquet` and `report.md` (`ukv_ceda_vs_openmeteo_fit.py`), `direct_differences.parquet`
and `direct_report.md` (`ukv_ceda_vs_openmeteo_compare.py`), and the saved losses for the
prediction figures. **Every number a figure draws is read from a table, and the script stops before
drawing unless each difference it draws appears in the report that table was printed into.**
Nothing is bootstrapped or refitted here.

- **Figure 1 (headline).** The planned contrasts, CEDA minus Open-Meteo, with 95% intervals and the
  margin as a band: wind power (P1 and the transfer penalty P3) and solar power (P2 and P3). Each
  row's label carries both archives' absolute errors, and the second hyperparameter setting is a
  hollow triangle.
- **Figures 2 and 3.** Out-of-fold power against measured for every generator (wind in Figure 2,
  solar in Figure 3), in three weeks chosen by a stated rule.
- **Figure 4.** The mean absolute difference between the two archives by CEDA lead, for each
  variable.
- **Figure 5.** The mean absolute difference between the two archives by calendar month, with the
  PS47 upgrade marked.
- **Figure 6.** The controls and the partial swaps of the transfer scoring (exploratory).
- **Figure 7.** The median ratio of CEDA's irradiance to Open-Meteo's at lead 0, by era and hour,
  which shows that Open-Meteo builds its hourly irradiance differently after PS47.

Generators appear only as A to F and W1 to W3.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_charts.py`, after the fit and
compare scripts. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import PLOT_WIDTH_PX, figure, planning, wrapped
from ukv_ceda_vs_era5_charts import (
    check_against_report,
    contrast_panel,
    predictions_chart,
)
from ukv_ceda_vs_openmeteo_build import OUTPUT_DIR
from ukv_ceda_vs_openmeteo_compare import DIFFERENCES_NAME, RATIOS_NAME
from ukv_ceda_vs_openmeteo_fit import (
    INTERVALS_NAME,
    PRIMARY_SETTING,
    REPORT_NAME,
    SECOND_SETTING,
    IntervalRecord,
    verdicts,
)

ASSETS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets" / "ukv_ceda_vs_openmeteo"
)
"""Where the published figures go."""

FAMILY: Final[str] = "weather model"
"""The colour family of every row: each row compares two archives of one weather model."""

CAPACITY_UNIT: Final[str] = "points of capacity"
WEEKS_NAME: Final[str] = "chart_weeks.md"
READING_WORDS: Final[dict[str, str]] = {
    "interchangeable": "interchangeable",
    "differ": "different",
    "unresolved": "unresolved",
    "no_penalty": "no transfer penalty",
    "penalty": "a transfer penalty",
}
"""How a reading appears in the headline title."""


def headline_title(*, records: Sequence[dict[str, Any]]) -> str:
    """State the headline finding from the verdicts, scoped to what was tested.

    Args:
        records: The interval records.

    Returns:
        A title naming each planned contrast's reading, with the solar contrasts scoped to the era
        before PS47 and the transfer scored on Open-Meteo's lead-0 analysis.
    """
    reading = {
        (verdict.domain, verdict.label): READING_WORDS[verdict.reading]
        for verdict in verdicts(records=cast("list[IntervalRecord]", list(records)))
    }
    return (
        f"UKV from CEDA and from Open-Meteo gave {reading[('wind', 'P1')]} wind power and "
        f"{reading[('solar', 'P2')]} solar power before PS47, and moving a CEDA-trained model onto "
        f"Open-Meteo's values showed {reading[('wind', 'P3')]} for wind and "
        f"{reading[('solar', 'P3')]} for solar"
    )


ERA_0_LABEL: Final[str] = "Before PS47"
ERA_1_LABEL: Final[str] = "After PS47"
SCOPES: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "wind": (
        ("all", "All months and hours"),
        ("era 0", ERA_0_LABEL),
        ("era 1", ERA_1_LABEL),
        ("lead 0 only", "Hours at CEDA lead 0 only"),
    ),
    "solar": (
        ("era 0", ERA_0_LABEL),
        ("all", "Both eras"),
        ("era 1", ERA_1_LABEL),
        ("lead 0 only", "Both eras, CEDA lead 0 only"),
    ),
}
"""The scopes the headline figure draws for each domain, with their row labels.

The first row of each domain is its planned scope (`PLANNED_SCOPE`). Solar rows that include era 1
are exploratory and carry the irradiance-construction note.
"""


def _one(*, records: Sequence[dict[str, Any]], where: dict[str, Any]) -> dict[str, Any]:
    found = [r for r in records if all(r.get(key) == value for key, value in where.items())]
    if len(found) != 1:
        msg = f"{len(found)} records match {where}; expected one"
        raise ValueError(msg)
    return found[0]


def contrast_rows(
    *, records: Sequence[dict[str, Any]], selectors: Sequence[tuple[str, dict[str, Any]]]
) -> pl.DataFrame:
    """Build the rows of a panel, with the second setting as a hollow marker.

    Args:
        records: The interval records.
        selectors: Each row's label and the fields that pick its record, apart from the setting.

    Returns:
        Rows for `interval_panel`, with both arms' absolute errors in the label.
    """
    rows = []
    for label, where in selectors:
        primary = _one(records=records, where={"setting": PRIMARY_SETTING, **where})
        second = _one(records=records, where={"setting": SECOND_SETTING, **where})
        text = (
            f"{primary['treatment_mae_pp']:.2f} against {primary['reference_mae_pp']:.2f} "
            "% of capacity"
        )
        shown = f"{label}, {primary['note']}" if primary["note"] else label
        rows.append(
            {
                "label": f"{shown} ({text})",
                "family": FAMILY,
                "difference": primary["difference_pp"],
                "lower_95": primary["lower_95_pp"],
                "upper_95": primary["upper_95_pp"],
                "planned": bool(primary["planned"]),
                "dashed": not primary["enough_months"],
                "second_difference": second["difference_pp"],
                "margin": primary["margin_pp"],
            }
        )
    return pl.DataFrame(rows, schema_overrides={"second_difference": pl.Float64})


def headline_figure(
    *, records: Sequence[dict[str, Any]]
) -> tuple[alt.VConcatChart, list[pl.DataFrame]]:
    """Draw Figure 1: the planned contrasts, by scope.

    Args:
        records: The interval records.

    Returns:
        The figure and its rows.
    """
    panels = {
        "Wind power, P1: CEDA minus Open-Meteo": ("wind", "P1", "ceda_wind_10m"),
        "Wind power, P3: transfer penalty": ("wind", "P3", "ceda_wind_10m_scored_on_om"),
        "Solar power, P2: CEDA minus Open-Meteo": ("solar", "P2", "ceda_ghi_temp"),
        "Solar power, P3: transfer penalty": ("solar", "P3", "ceda_ghi_temp_scored_on_om"),
    }
    rows = {
        title: contrast_rows(
            records=records,
            selectors=[
                (
                    label,
                    {"domain": domain, "label": planned, "treatment": treatment, "scope": scope},
                )
                for scope, label in SCOPES[domain]
            ],
        )
        for title, (domain, planned, treatment) in panels.items()
    }
    kind = planning(rows=list(rows.values()))
    drawn = [
        contrast_panel(
            rows=frame,
            title=title,
            x_title=f"Difference in power error ({CAPACITY_UNIT}; negative favours CEDA)",
            planned_figure=kind,
            zero_label="same error",
            better_label="CEDA lower",
        )
        for title, frame in rows.items()
    ]
    chart = figure(
        panels=drawn,
        number=1,
        title=headline_title(records=records),
        subtitle=[
            "Dot: the primary hyperparameter setting. Hollow triangle: the second setting.",
            "Grey band: the margin. Line: 95% interval from resampling whole calendar months.",
            (
                "Wind reads the whole overlap. Solar is planned on the era before the PS47 upgrade "
                "only, because Open-Meteo builds its hourly irradiance differently afterwards."
            ),
        ],
        figure_planning=kind,
    )
    return chart, list(rows.values())


def lead_figure(*, records: Sequence[dict[str, Any]]) -> alt.VConcatChart:
    """Draw Figure 4: the mean absolute difference between the archives by CEDA lead.

    Args:
        records: The model-free records.

    Returns:
        The figure.
    """
    panels = []
    for variable in dict.fromkeys(r["variable"] for r in records):
        data = pl.DataFrame(
            [r for r in records if r["variable"] == variable and r["split"] == "lead"]
        )
        unit = data["unit"][0]
        base = alt.Chart(data).encode(
            x=alt.X(
                "label:N",
                title="CEDA lead (hours since the run started)",
                sort=data["label"].to_list(),
                axis=alt.Axis(labelAngle=0, labelExpr="replace(datum.value, 'lead ', '')"),
            )
        )
        panels.append(
            alt.layer(
                base.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y(
                        "mean_absolute_difference_lower_95:Q",
                        title=f"Mean absolute difference ({unit})",
                    ),
                    y2="mean_absolute_difference_upper_95:Q",
                ),
                base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y="mean_absolute_difference:Q"
                ),
            ).properties(
                width=PLOT_WIDTH_PX,
                height=110,
                title=alt.TitleParams(variable[:1].upper() + variable[1:], anchor="start"),
            )
        )
    return figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=4,
        title="The two archives agree most closely at lead 0 and drift apart with CEDA's lead",
        subtitle=[
            "Mean absolute difference between CEDA and Open-Meteo, at the nine generator sites.",
            "Line: 95% interval from resampling whole months. A smaller difference means closer.",
        ],
        figure_planning="exploratory",
    )


def month_figure(*, records: Sequence[dict[str, Any]]) -> alt.VConcatChart:
    """Draw Figure 5: the mean absolute difference by calendar month, with the upgrade marked.

    Args:
        records: The model-free records.

    Returns:
        The figure.
    """
    panels = []
    boundary = pl.DataFrame({"label": ["2026-02"]})
    for variable in dict.fromkeys(r["variable"] for r in records):
        data = pl.DataFrame(
            [r for r in records if r["variable"] == variable and r["split"] == "month"]
        )
        unit = data["unit"][0]
        base = alt.Chart(data).encode(
            x=alt.X(
                "label:N",
                title="Month",
                sort=data["label"].to_list(),
                axis=alt.Axis(labelAngle=-60),
            )
        )
        panels.append(
            alt.layer(
                base.mark_line(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y(
                        "mean_absolute_difference:Q", title=f"Mean absolute difference ({unit})"
                    )
                ),
                alt.Chart(boundary)
                .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
                .encode(x="label:N"),  # ty: ignore[unresolved-attribute]
            ).properties(
                width=PLOT_WIDTH_PX,
                height=110,
                title=alt.TitleParams(variable[:1].upper() + variable[1:], anchor="start"),
            )
        )
    return figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=5,
        title="Month by month, any step in the difference between the archives shows",
        subtitle=[
            (
                "Mean absolute difference at all hours. The orange rule marks the first month "
                "after the PS47 upgrade of 2026-01-21 (2026-01 itself is dropped)."
            ),
        ],
        figure_planning="exploratory",
    )


def ratio_figure(*, ratios: pl.DataFrame) -> alt.VConcatChart:
    """Draw Figure 7: CEDA's irradiance over Open-Meteo's at lead 0, by era and hour.

    Args:
        ratios: The compare script's irradiance ratios by era and hour.

    Returns:
        The figure.
    """
    data = ratios.with_columns(
        era=pl.when(pl.col("era_code") == 0)
        .then(pl.lit(ERA_0_LABEL))
        .otherwise(pl.lit(ERA_1_LABEL))
    )
    panels = []
    for column, title in (
        ("rebuilt_ratio", "CEDA's snapshot rebuilt as Open-Meteo built its value before PS47"),
        ("raw_ratio", "CEDA's snapshot, unscaled"),
    ):
        encoding = {
            "x": alt.X(
                "hour_of_day:O",
                title="UTC hour of day (lead-0 hours with sunshine)",
                axis=alt.Axis(labelAngle=0),
            ),
            "y": alt.Y(
                f"{column}:Q",
                title="Median CEDA over Open-Meteo (1 means equal)",
                scale=alt.Scale(domain=[0.4, 1.6], zero=False, nice=False),
            ),
            "color": alt.Color(
                "era:N",
                title="",
                scale=alt.Scale(
                    domain=[ERA_0_LABEL, ERA_1_LABEL],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
            ),
        }
        line = alt.Chart(data).mark_line().encode(**encoding)  # ty: ignore[unresolved-attribute]
        points = (
            alt.Chart(data).mark_point(filled=True, size=70).encode(**encoding)  # ty: ignore[unresolved-attribute]
        )
        rule = alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color=ocf.BLACK_1).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(rule, line, points).properties(
                width=PLOT_WIDTH_PX, height=150, title=alt.TitleParams(title, anchor="start")
            )
        )
    return figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=7,
        title=(
            "Before PS47 CEDA's rebuilt snapshot matches Open-Meteo's irradiance, and after it "
            "does not"
        ),
        subtitle=[
            "Median ratio at the hours where CEDA's lead is 0, at the nine generator sites.",
            "Black rule: the two archives equal.",
        ],
        figure_planning="exploratory",
    )


def controls_figure(
    *, records: Sequence[dict[str, Any]]
) -> tuple[alt.VConcatChart, list[pl.DataFrame]]:
    """Draw Figure 6: the controls and the partial swaps of the transfer scoring.

    Args:
        records: The interval records.

    Returns:
        The figure and its rows.
    """
    exploratory = [r for r in records if not r["planned"]]
    rows = {}
    for domain in ("wind", "solar"):
        chosen = [r for r in exploratory if r["domain"] == domain]
        rows[f"{domain.capitalize()}: controls and partial swaps"] = pl.DataFrame(
            [
                {
                    "label": f"{r['label']} ({r['treatment_mae_pp']:.2f} against "
                    f"{r['reference_mae_pp']:.2f} % of capacity)",
                    "family": FAMILY,
                    "difference": r["difference_pp"],
                    "lower_95": r["lower_95_pp"],
                    "upper_95": r["upper_95_pp"],
                    "planned": False,
                    "dashed": not r["enough_months"],
                    "second_difference": None,
                    "margin": r["margin_pp"],
                }
                for r in chosen
            ],
            schema_overrides={"second_difference": pl.Float64},
        )
    kind = planning(rows=list(rows.values()))
    drawn = [
        contrast_panel(
            rows=frame,
            title=title,
            x_title=f"Difference in power error ({CAPACITY_UNIT})",
            planned_figure=kind,
            zero_label="same error",
            better_label="first arm lower",
        )
        for title, frame in rows.items()
    ]
    chart = figure(
        panels=drawn,
        number=6,
        title="The controls show what the pipeline produces from shuffled weather",
        subtitle=["Every row is at the primary setting."],
        figure_planning=kind,
    )
    return chart, list(rows.values())


def main() -> int:
    """Check each figure against its report, and write the SVG files and the weeks report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    records = pl.read_parquet(directory / INTERVALS_NAME).to_dicts()
    direct = pl.read_parquet(directory / DIFFERENCES_NAME).to_dicts()
    report = (directory / REPORT_NAME).read_text()
    assets: Path = arguments.assets_dir
    assets.mkdir(parents=True, exist_ok=True)

    headline, headline_rows = headline_figure(records=records)
    controls, controls_rows = controls_figure(records=records)
    for rows in (*headline_rows, *controls_rows):
        check_against_report(rows=rows, report=report, name="the set B report")
    headline.save(assets / "fig01_headline.svg")
    controls.save(assets / "fig06_controls.svg")
    ratio_figure(ratios=pl.read_parquet(directory / RATIOS_NAME)).save(
        assets / "fig07_irradiance_ratio.svg"
    )
    lead_figure(records=direct).save(assets / "fig04_leads.svg")
    month_figure(records=direct).save(assets / "fig05_months.svg")

    lines = ["### Weeks drawn in Figures 2 and 3, chosen by rule", ""]
    for number, (domain, arms) in enumerate(
        (
            ("wind", {"ceda_wind_10m": "CEDA", "om_wind_10m": "Open-Meteo"}),
            ("solar", {"ceda_ghi_temp": "CEDA", "om_ghi_temp": "Open-Meteo"}),
        ),
        start=2,
    ):
        chart, weeks = predictions_chart(
            losses=pl.read_parquet(directory / f"losses_{domain}.parquet"),
            arms=arms,
            title=wrapped(
                text=f"Figure {number}: Out-of-fold {domain} farm power as a share of capacity, "
                "in three weeks chosen by rule",
                width=72,
            )[0],
        )
        chart.save(assets / f"fig0{number}_{domain}_weeks.svg")
        lines += [f"- {domain}, {name}: {start:%B %Y}." for name, start in weeks.items()]
    weeks_path = directory / WEEKS_NAME
    if not weeks_path.exists():
        weeks_path.write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + f"\nWrote figures to {assets}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
