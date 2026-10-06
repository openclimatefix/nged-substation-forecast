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

Generators appear only as A to F and W1 to W3.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_charts.py`, after the fit and
compare scripts. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

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
from ukv_ceda_vs_openmeteo_compare import DIFFERENCES_NAME
from ukv_ceda_vs_openmeteo_fit import (
    INTERVALS_NAME,
    PRIMARY_SETTING,
    REPORT_NAME,
    SECOND_SETTING,
)

ASSETS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets" / "ukv_ceda_vs_openmeteo"
)
"""Where the published figures go."""

FAMILY: Final[str] = "weather model"
"""The colour family of every row: each row compares two archives of one weather model."""

CAPACITY_UNIT: Final[str] = "points of capacity"
WEEKS_NAME: Final[str] = "chart_weeks.md"
HEADLINE_TITLE: Final[str] = (
    "Power error of XGBoost models given UKV from CEDA and UKV from Open-Meteo, and the cost of "
    "moving a CEDA-trained model onto Open-Meteo's values"
)
"""A descriptive title. The finding replaces it once the run has a result."""
SCOPES: Final[tuple[tuple[str, str], ...]] = (
    ("all", "All hours"),
    ("era 0", "Before the 2026-01-21 upgrade"),
    ("era 1", "After the upgrade"),
    ("lead 0 only", "Hours at CEDA lead 0 only"),
)
"""The scopes the headline figure draws, with their row labels."""


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
        rows.append(
            {
                "label": f"{label} ({text})",
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
                for scope, label in SCOPES
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
        title=HEADLINE_TITLE,
        subtitle=[
            "Dot: the primary hyperparameter setting. Hollow triangle: the second setting.",
            "Grey band: the margin. Line: 95% interval from resampling whole calendar months.",
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
            x=alt.X("label:N", title="CEDA lead", sort=data["label"].to_list())
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
                width=PLOT_WIDTH_PX, height=110, title=alt.TitleParams(variable, anchor="start")
            )
        )
    return figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=4,
        title="The two archives agree most closely at lead 0 and drift apart with CEDA's lead",
        subtitle=[
            "Mean absolute difference between CEDA and Open-Meteo, at the nine generator sites.",
            "Line: 95% interval from resampling whole calendar months. All rows are exploratory.",
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
                width=PLOT_WIDTH_PX, height=110, title=alt.TitleParams(variable, anchor="start")
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
            "All rows are exploratory.",
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
        subtitle=["All rows are exploratory, at the primary setting."],
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
