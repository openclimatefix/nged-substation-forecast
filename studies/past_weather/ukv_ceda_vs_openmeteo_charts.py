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
- **Figure 5.** The median ratio of CEDA's irradiance to Open-Meteo's at lead 0, by era and hour,
  which shows that Open-Meteo builds its hourly irradiance differently after PS47.
- **Figure 6.** The mean absolute difference between the two archives by calendar month, with the
  PS47 upgrade marked.
- **Figure 7.** The daily ratio of Open-Meteo's 10 m wind speed to CEDA's around the two spans in
  which Open-Meteo builds its speed differently.
- **Figure 8.** The planned contrasts by CEDA lead.
- **Figure 9.** Every arm's absolute error on the planned scope.
- **Figure 10.** Each wind arm's mean signed error, which shows the transfer penalty as a level
  bias.
- **Figure 11.** The controls and the partial swaps of the transfer scoring (exploratory).

Generators appear only as A to F and W1 to W3.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_openmeteo_charts.py`, after the fit and
compare scripts. Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`
before committing it.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import PLOT_WIDTH_PX, figure, planning
from ukv_ceda_vs_era5_charts import (
    check_against_report,
    contrast_panel,
    predictions_chart,
)
from ukv_ceda_vs_openmeteo_build import OPEN_METEO_WIND_STEP_DAYS, OUTPUT_DIR
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
    "differ": "differ",
    "interchangeable": "match within the margin",
    "unresolved": "are unresolved",
    "no_penalty": "shows no penalty",
    "penalty": "shows a penalty",
}
"""How a reading appears in the headline title."""

TRANSFER_WORDS: Final[dict[str, str]] = {
    "no_penalty": "shows no penalty",
    "penalty": "shows a penalty",
    "unresolved": "is unresolved",
}


def headline_title(*, records: Sequence[dict[str, Any]]) -> str:
    """State the headline finding from the verdicts, scoped to what was tested.

    Args:
        records: The interval records.

    Returns:
        A title naming each planned contrast's reading, with the solar contrasts scoped to the era
        before PS47 and the transfer scored on Open-Meteo's lead-0 analysis.
    """
    reading = {
        (verdict.domain, verdict.label): verdict.reading
        for verdict in verdicts(records=cast("list[IntervalRecord]", list(records)))
    }
    return (
        "Power errors from CEDA's and Open-Meteo's UKV "
        f"{READING_WORDS[reading[('wind', 'P1')]]} for wind and "
        f"{READING_WORDS[reading[('solar', 'P2')]]} for solar before PS47, and moving a "
        "CEDA-trained model onto Open-Meteo's values "
        f"{TRANSFER_WORDS[reading[('wind', 'P3')]]} for wind and "
        f"{TRANSFER_WORDS[reading[('solar', 'P3')]]} for solar"
    )


ERA_0_LABEL: Final[str] = "Before PS47"
ERA_1_LABEL: Final[str] = "After PS47"
SCOPES: Final[dict[str, tuple[tuple[str, str, str | None], ...]]] = {
    "wind": (
        ("all", "All months and hours", None),
        ("era 0", "Before PS47", None),
        ("era 1", "After PS47", None),
        ("lead 0 only", "CEDA lead 0 only", None),
    ),
    "solar": (
        ("era 0", "Before PS47, trained on both eras", None),
        ("all", "Before PS47, trained on era 0 alone", "solar_era0"),
        ("era 1", "After PS47", None),
        ("era 0, lead 0 only", "Before PS47, CEDA lead 0 only", None),
    ),
}
"""The scopes the headline figure draws for each domain: scope, row label, and the row set where
it is not the domain's own.

The first row of each domain is its planned scope (`PLANNED_SCOPE`). The solar contrasts are
planned on era 0 in two fits, and a verdict stands only if both agree.
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
        text = f"{primary['treatment_mae_pp']:.2f} against {primary['reference_mae_pp']:.2f} %"
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
                    {
                        "domain": override or domain,
                        "label": planned,
                        "treatment": treatment,
                        "scope": scope,
                    },
                )
                for scope, label, override in SCOPES[domain]
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
            zero_label="no penalty" if "P3" in title else "same error",
            better_label="CEDA model better" if "P3" in title else "CEDA lower",
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
                "Row labels give the two models' mean absolute errors in % of capacity, CEDA "
                "against Open-Meteo (P3: the CEDA-trained model on Open-Meteo's values against "
                "the Open-Meteo-trained model)."
            ),
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
        number=6,
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
        number=5,
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


STEP_DAILY_NAME: Final[str] = "wind_step_daily.parquet"

CONTROL_LABELS: Final[dict[str, str]] = {
    "control": "Shuffled weather, CEDA against Open-Meteo",
    "GPU against CPU": "Open-Meteo model refit on the GPU against the CPU",
}
"""Plain names for the two rows whose report labels are terse."""

PLANNED_LABELS: Final[tuple[str, ...]] = ("P1", "P2", "P3")
"""The labels of the planned contrasts; every other label is a control or a partial swap."""

LEADS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)

ARM_LABELS: Final[dict[str, str]] = {
    "ceda_wind_10m": "CEDA model, CEDA's wind",
    "om_wind_10m": "Open-Meteo model, Open-Meteo's wind",
    "ceda_wind_10m_scored_on_om": "CEDA model, Open-Meteo's wind",
    "ceda_wind_10m_scored_on_om_speed_rescaled": "CEDA model, Open-Meteo's wind rescaled",
    "ceda_ghi_temp": "CEDA model, CEDA's inputs",
    "om_ghi_temp": "Open-Meteo model, Open-Meteo's inputs",
    "ceda_ghi_temp_scored_on_om": "CEDA model, Open-Meteo's inputs",
}
"""The arms the absolute-error and signed-error figures draw, with plain labels."""

ABSOLUTE_HEADINGS: Final[dict[str, str]] = {
    "## wind: every arm's absolute error, planned scope (all)": "Wind farms, all months",
    "## solar: every arm's absolute error, planned scope (era 0)": (
        "Solar farms, era 0, trained on both eras"
    ),
    "## solar_era0: every arm's absolute error, planned scope (all)": (
        "Solar farms, era 0, trained on era 0 alone"
    ),
}
SIGNED_HEADING: Final[str] = "## wind: mean signed error on the planned scope (all)"


def table_after(*, report: str, heading: str) -> list[list[str]]:
    """Read the rows of the first Markdown table under a heading of the report.

    Args:
        report: The text of `report.md`.
        heading: The heading line, such as `## wind: contrasts`.

    Returns:
        The data rows, each as its cells, without the header and the rule.
    """
    lines = report.splitlines()
    start = lines.index(heading)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


def absolute_error_rows(*, report: str) -> pl.DataFrame:
    """Read each drawn arm's absolute error on the planned scope from the report.

    Args:
        report: The text of `report.md`.

    Returns:
        `panel`, `arm`, `label`, `setting`, `value`, `lower`, `upper`.
    """
    rows: list[dict[str, Any]] = []
    for heading, panel in ABSOLUTE_HEADINGS.items():
        for cells in table_after(report=report, heading=heading):
            if cells[0] not in ARM_LABELS:
                continue
            lower, upper = (float(x) for x in cells[3].strip("[]").split(","))
            rows.append(
                {
                    "panel": panel,
                    "arm": cells[0],
                    "label": ARM_LABELS[cells[0]],
                    "setting": cells[1],
                    "value": float(cells[2]),
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pl.DataFrame(rows)


def absolute_figure(*, errors: pl.DataFrame) -> alt.VConcatChart:
    """Draw Figure 9: every drawn arm's mean absolute error with its 95% interval.

    Args:
        errors: `absolute_error_rows`'s result.

    Returns:
        The figure.
    """
    panels = []
    for panel in dict.fromkeys(errors["panel"].to_list()):
        data = errors.filter(pl.col("panel") == panel)
        order = list(dict.fromkeys(data["label"].to_list()))
        base = alt.Chart(data).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "setting:N",
                title="Setting",
                scale=alt.Scale(
                    domain=["primary", "second"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
            ),
        )
        rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lower:Q",
                scale=alt.Scale(zero=False),
                title="Mean absolute error (% of capacity; smaller is better)",
                axis=alt.Axis(tickCount=5, format=".1f"),
            ),
            x2="upper:Q",
            yOffset="setting:N",
        )
        dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", yOffset="setting:N"
        )
        panels.append(
            alt.layer(rule, dot).properties(
                width=380,
                height=34 * len(order),
                title=alt.TitleParams(panel, anchor="start"),
            )
        )
    return figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=9,
        title=("Every XGBoost model's error is within about half a point of the others'"),
        subtitle=[
            "Mean absolute error of the XGBoost models on the planned scope, with 95% intervals.",
            "Rows are scored on the same hours within each panel. Figure 11 has the controls.",
        ],
        figure_planning="exploratory",
    )


def signed_error_figure(*, report: str) -> alt.VConcatChart:
    """Draw Figure 10: each wind arm's mean signed error, in all hours and at CEDA lead 0.

    Args:
        report: The text of `report.md`.

    Returns:
        The figure.
    """
    rows = []
    for cells in table_after(report=report, heading=SIGNED_HEADING):
        if cells[0] in ARM_LABELS and cells[1] == "primary":
            rows += [
                {"label": ARM_LABELS[cells[0]], "scope": "All hours", "value": float(cells[2])},
                {
                    "label": ARM_LABELS[cells[0]],
                    "scope": "CEDA lead 0 only",
                    "value": float(cells[3]),
                },
            ]
    data = pl.DataFrame(rows)
    order = list(dict.fromkeys(data["label"].to_list()))
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.Color(
            "scope:N",
            title="",
            scale=alt.Scale(
                domain=["All hours", "CEDA lead 0 only"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
        ),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q",
            title="Mean signed error (% of capacity; negative is an under-prediction)",
        ),
        yOffset="scope:N",
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    panel = alt.layer(bars, zero).properties(width=380, height=40 * len(order))
    return figure(
        panels=[panel],  # ty: ignore[invalid-argument-type]
        number=10,
        title=(
            "A CEDA-trained wind model under-predicts by about 2 points more when given "
            "Open-Meteo's wind"
        ),
        subtitle=[
            "Mean of prediction minus measured power as a share of capacity, primary setting.",
            "Black rule: no bias. The wind farms' months and hours are those of the planned scope.",
        ],
        figure_planning="exploratory",
    )


def step_figure(*, daily: pl.DataFrame) -> alt.VConcatChart:
    """Draw Figure 7: the daily ratio of Open-Meteo's 10 m speed to CEDA's around the two spans.

    Args:
        daily: `wind_step_daily.parquet`: `day`, `ratio`, `n` and `in_span`.

    Returns:
        The figure. The ratio is pooled over the nine generator sites.
    """
    spans = pl.DataFrame(
        {
            "first": [first for first, _ in OPEN_METEO_WIND_STEP_DAYS],
            "last": [last + timedelta(days=1) for _, last in OPEN_METEO_WIND_STEP_DAYS],
        }
    )
    shade = (
        alt.Chart(spans)
        .mark_rect(color=ocf.BRAND_ORANGE, opacity=0.18)
        .encode(x="first:T", x2="last:T")  # ty: ignore[unresolved-attribute]
    )
    line = (
        alt.Chart(daily)
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:T", title="UTC day", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y(
                "ratio:Q",
                title="Open-Meteo's 10 m speed over CEDA's (1 means equal)",
                scale=alt.Scale(domain=[0.85, 1.25], zero=False),
            ),
        )
    )
    one = alt.Chart(pl.DataFrame({"y": [1.0]})).mark_rule(color=ocf.BLACK_1).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    panel = alt.layer(shade, one, line).properties(width=PLOT_WIDTH_PX, height=170)
    return figure(
        panels=[panel],  # ty: ignore[invalid-argument-type]
        number=7,
        title="Open-Meteo's 10 m wind speed steps up against CEDA's in two spans",
        subtitle=[
            "Daily median ratio at CEDA lead 0, pooled over the nine generator sites.",
            "Orange bands: the two spans that every wind arm drops. Black rule: equal speeds.",
        ],
        figure_planning="exploratory",
    )


def by_lead_figure(
    *, records: Sequence[dict[str, Any]]
) -> tuple[alt.VConcatChart, list[pl.DataFrame]]:
    """Draw Figure 8: each planned contrast by CEDA lead, at both settings.

    Args:
        records: The interval records.

    Returns:
        The figure and the rows it draws, which are checked against the report.
    """
    specs = {
        "Wind power, P1: CEDA minus Open-Meteo": ("wind", "P1", "ceda_wind_10m", LEAD_FORMAT),
        "Wind power, P3: transfer penalty": (
            "wind",
            "P3",
            "ceda_wind_10m_scored_on_om",
            LEAD_FORMAT,
        ),
        "Solar power, P2, era 0: CEDA minus Open-Meteo": (
            "solar",
            "P2",
            "ceda_ghi_temp",
            ERA_0_LEAD_FORMAT,
        ),
        "Solar power, P3, era 0: transfer penalty": (
            "solar",
            "P3",
            "ceda_ghi_temp_scored_on_om",
            ERA_0_LEAD_FORMAT,
        ),
    }
    panels = []
    drawn: list[pl.DataFrame] = []
    for title, (domain, label, treatment, scope_format) in specs.items():
        rows = pl.DataFrame(
            [
                {
                    "lead": lead,
                    "setting": record["setting"],
                    "difference": record["difference_pp"],
                    "lower_95": record["lower_95_pp"],
                    "upper_95": record["upper_95_pp"],
                    "margin": record["margin_pp"],
                }
                for lead in LEADS
                for setting in ("primary", "second")
                for record in records
                if record["domain"] == domain
                and record["label"] == label
                and record["treatment"] == treatment
                and record["scope"] == scope_format.format(lead=lead)
                and record["setting"] == setting
            ]
        )
        drawn.append(rows)
        margin = float(rows["margin"][0])
        band = (
            alt.Chart(pl.DataFrame({"low": [-margin], "high": [margin]}))
            .mark_rect(color=ocf.GREY_3, opacity=0.5)
            .encode(y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
        )
        zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=ocf.BLACK_1).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        encoding = {
            "x": alt.X(
                "lead:O",
                title="CEDA lead (hours since the run started)",
                axis=alt.Axis(labelAngle=0),
            ),
            "color": alt.Color(
                "setting:N",
                title="Setting",
                scale=alt.Scale(
                    domain=["primary", "second"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
            ),
            "xOffset": "setting:N",
        }
        rule = (
            alt.Chart(rows)
            .mark_rule()
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("lower_95:Q", title="Points of capacity"), y2="upper_95:Q", **encoding
            )
        )
        dot = (
            alt.Chart(rows)
            .mark_point(filled=True, size=55)
            .encode(  # ty: ignore[unresolved-attribute]
                y="difference:Q", **encoding
            )
        )
        panels.append(
            alt.layer(band, zero, rule, dot).properties(
                width=PLOT_WIDTH_PX, height=130, title=alt.TitleParams(title, anchor="start")
            )
        )
    chart = figure(
        panels=panels,  # ty: ignore[invalid-argument-type]
        number=8,
        title=("The gaps between the archives grow with CEDA's lead, and are small at lead 0"),
        subtitle=[
            "Dot: estimate. Line: 95% interval from resampling whole months. Grey: the margin.",
            "Positive means CEDA's error is larger. Every row is an exploratory subset.",
        ],
        figure_planning="exploratory",
    )
    return chart, drawn


LEAD_FORMAT: Final[str] = "lead {lead} only"
ERA_0_LEAD_FORMAT: Final[str] = "era 0, lead {lead} only"


def controls_figure(
    *, records: Sequence[dict[str, Any]]
) -> tuple[alt.VConcatChart, list[pl.DataFrame]]:
    """Draw Figure 6: the controls and the partial swaps of the transfer scoring.

    Args:
        records: The interval records.

    Returns:
        The figure and its rows.
    """
    exploratory = [
        r
        for r in records
        if r["label"] not in PLANNED_LABELS
        and "shuffled arm" not in r["label"]
        and r["setting"] == PRIMARY_SETTING
    ]
    rows = {}
    for domain in ("wind", "solar"):
        chosen = [r for r in exploratory if r["domain"] == domain]
        rows[f"{domain.capitalize()}: controls and partial swaps"] = pl.DataFrame(
            [
                {
                    "label": f"{CONTROL_LABELS.get(r['label'], r['label'])} "
                    f"({r['treatment_mae_pp']:.2f} against "
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
        number=11,
        title=(
            "Most of the wind transfer penalty is the speed level: rescaling Open-Meteo's speed "
            "removes most of it"
        ),
        subtitle=[
            "Every row is at the primary setting. Each row's second arm is the reference.",
            (
                "The shuffled-weather controls and the GPU-against-CPU refit show the noise "
                "around a difference."
            ),
        ],
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
    controls.save(assets / "fig11_controls.svg")
    by_lead, by_lead_rows = by_lead_figure(records=records)
    for rows in by_lead_rows:
        check_against_report(rows=rows, report=report, name="the set B report")
    by_lead.save(assets / "fig08_contrasts_by_lead.svg")
    absolute_figure(errors=absolute_error_rows(report=report)).save(
        assets / "fig09_absolute_errors.svg"
    )
    signed_error_figure(report=report).save(assets / "fig10_signed_errors.svg")
    step_figure(daily=pl.read_parquet(directory / STEP_DAILY_NAME)).save(
        assets / "fig07_wind_step.svg"
    )
    ratio_figure(ratios=pl.read_parquet(directory / RATIOS_NAME)).save(
        assets / "fig05_irradiance_ratio.svg"
    )
    lead_figure(records=direct).save(assets / "fig04_leads.svg")
    month_figure(records=direct).save(assets / "fig06_months.svg")

    lines = ["### Weeks drawn in Figures 2 and 3, chosen by rule", ""]
    for number, (domain, arms) in enumerate(
        (
            ("wind", {"ceda_wind_10m": "CEDA", "om_wind_10m": "Open-Meteo"}),
            ("solar", {"ceda_ghi_temp": "CEDA", "om_ghi_temp": "Open-Meteo"}),
        ),
        start=2,
    ):
        chart, weeks = predictions_chart(
            # The shared chart function filters on the earlier setting name, "pooled".
            losses=pl.read_parquet(directory / f"losses_{domain}.parquet").with_columns(
                setting=pl.col("setting").replace({"primary": "pooled"})
            ),
            arms=arms,
            title=f"Figure {number}: Out-of-fold {domain} power in three weeks chosen by rule",
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
