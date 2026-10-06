"""Draw the figures of the UKV-CEDA against ERA5 study from the saved interval tables.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. It reads
`station_intervals.parquet` (set A) and `intervals.parquet` (set B), which
`ukv_ceda_station_scores.py` and `ukv_ceda_vs_era5_fit.py` wrote, and the saved losses for the
prediction figure. **Every number a figure draws is read from a table, and the script stops before
drawing unless each difference it draws appears in the report that table was printed into.**
Nothing is bootstrapped or refitted here.

- **Figure 1 (headline).** The four planned contrasts, P1 to P4, each UKV-CEDA minus ERA5 with its
  95% interval, grouped by variable and window. Each row's label carries both products' absolute
  error. The margin is drawn as a band, and the second hyperparameter setting of P3 and P4 as a
  hollow triangle.
- **Figure 2.** Set A by calendar year, half-year, station, and UKV-CEDA lead (exploratory, except
  the planned P2-lead rows).
- **Figure 3.** Set B by calendar year and half-year (exploratory splits of P3 and P4).
- **Figure 4.** The controls and checks of set B (exploratory).
- **Figures 5 and 6.** Out-of-fold power against measured for every generator, in three weeks chosen
  by a stated rule: the week of highest mean output, the week of the largest hour-to-hour spread,
  and the week of lowest mean output.

Generators appear only as A to F and W1 to W3, and stations as S1 upwards.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_era5_charts.py`, after the fit script.
Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import figure, interval_panel, planning
from ukv_ceda_station_scores import INTERVALS_NAME as STATION_INTERVALS_NAME
from ukv_ceda_station_scores import PRIMARY_SCORE
from ukv_ceda_station_scores import REPORT_NAME as STATION_REPORT_NAME
from ukv_ceda_vs_era5_build import OUTPUT_DIR
from ukv_ceda_vs_era5_fit import INTERVALS_NAME, PRIMARY_SETTING, REPORT_NAME, SECOND_SETTING

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the published figures go."""

FAMILY: Final[str] = "weather model"
"""The colour family of every row: each row is UKV-CEDA, a weather model, against ERA5."""

WINDOW_SCOPES: Final[tuple[str, ...]] = ("all", "early window", "late window")
WINDOW_LABELS: Final[dict[str, str]] = {
    "all": "All months",
    "early window": "Early window, 2019-09 to 2020-12",
    "late window": "Late window, 2021-01 onwards",
}

WEEK_DAYS: Final[int] = 7


def _mae_text(*, era5: float, ukv: float, unit: str) -> str:
    return f"ERA5 {era5:.2f}, UKV-CEDA {ukv:.2f} {unit}"


Selector = tuple[str, dict[str, Any]]
"""A row's label and the fields that pick its record out of a table."""


def _one(*, records: Sequence[dict[str, Any]], where: dict[str, Any]) -> dict[str, Any]:
    found = [r for r in records if all(r.get(key) == value for key, value in where.items())]
    if len(found) != 1:
        msg = f"{len(found)} records match {where}; expected one"
        raise ValueError(msg)
    return found[0]


def station_rows(
    *, records: Sequence[dict[str, Any]], selectors: Sequence[Selector]
) -> pl.DataFrame:
    """Build the rows of a set A panel, on the primary score.

    Args:
        records: Set A's interval records.
        selectors: Each row's label and the fields that pick its record, such as the variable, the
            label and the scope.

    Returns:
        Rows for `interval_panel`, with the absolute errors in the label.
    """
    rows = []
    for label, where in selectors:
        record = _one(records=records, where={"score": PRIMARY_SCORE, **where})
        unit = "m/s" if record["variable"] == "wind" else "K"
        text = _mae_text(era5=record["era5_mae"], ukv=record["ukv_mae"], unit=unit)
        rows.append(
            {
                "label": f"{label} ({text})",
                "family": FAMILY,
                "difference": record["difference"],
                "lower_95": record["lower_95"],
                "upper_95": record["upper_95"],
                "planned": bool(record["planned"]),
                "dashed": not record["enough_months"],
                "second_difference": None,
                "margin": record["margin"],
            }
        )
    return pl.DataFrame(rows, schema_overrides={"second_difference": pl.Float64})


def set_b_rows(
    *, records: Sequence[dict[str, Any]], selectors: Sequence[Selector], with_second: bool
) -> pl.DataFrame:
    """Build the rows of a set B panel, with the second setting as a hollow marker.

    Args:
        records: Set B's interval records.
        selectors: Each row's label and the fields that pick its record, apart from the setting.
        with_second: Whether each row also has a record at the second setting.

    Returns:
        Rows for `interval_panel`, with the absolute errors in the label.
    """
    rows = []
    for label, where in selectors:
        primary = _one(records=records, where={"setting": PRIMARY_SETTING, **where})
        second = (
            _one(records=records, where={"setting": SECOND_SETTING, **where})
            if with_second
            else None
        )
        text = _mae_text(
            era5=primary["reference_mae_pp"], ukv=primary["treatment_mae_pp"], unit="% of capacity"
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
                "second_difference": None if second is None else second["difference_pp"],
                "margin": primary["margin_pp"],
            }
        )
    return pl.DataFrame(rows, schema_overrides={"second_difference": pl.Float64})


def check_against_report(*, rows: pl.DataFrame, report: str, name: str) -> None:
    """Raise unless every difference a figure draws appears in the report, at its precision.

    Args:
        rows: The rows a panel draws.
        report: The text of the report the interval table was printed into.
        name: The report's name, for the message.

    Raises:
        ValueError: Naming the differences that the report does not print.
    """
    missing = [
        f"{value:+.3f}"
        for value in rows["difference"].drop_nulls().to_list()
        if f"{value:+.3f}" not in report
    ]
    if missing:
        msg = f"{name} does not print these differences, so the chart would disagree: {missing}"
        raise ValueError(msg)


def margin_band(*, margin: float, x_domain: tuple[float, float]) -> alt.Chart:
    """Draw the margin as a band either side of zero.

    Args:
        margin: The margin, in the panel's unit.
        x_domain: The panel's x range.

    Returns:
        A chart that spans the panel's full height.
    """
    return (
        alt.Chart(pl.DataFrame({"low": [-margin], "high": [margin]}))
        .mark_rect(color=ocf.GREY_3, opacity=0.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", scale=alt.Scale(domain=list(x_domain)), axis=None),
            x2="high:Q",
        )
    )


def x_domain_for(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Choose a symmetric x range that holds every interval, point, and margin.

    Args:
        rows: The panel's rows.

    Returns:
        The range, rounded up to two significant figures.
    """
    values = [
        abs(v)
        for column in ("lower_95", "upper_95", "difference", "second_difference", "margin")
        for v in rows[column].drop_nulls().to_list()
        if v == v  # noqa: PLR0124 - drops NaN
    ]
    largest = max(values) * 1.15
    magnitude = 10 ** (len(str(int(1 / largest))) - 1) if largest < 1 else 1
    bound = round(largest * magnitude * 10 + 0.5) / (10 * magnitude)
    return (-bound, bound)


def contrast_panel(*, rows: pl.DataFrame, title: str, x_title: str, planned_figure: Any) -> Any:
    """Draw one panel of dots and intervals, with the margin band behind it.

    Args:
        rows: A panel's rows.
        title: The panel's title.
        x_title: The x axis title, naming the quantity and unit.
        planned_figure: What `planning` returns for the whole figure.

    Returns:
        The layered panel.
    """
    x_domain = x_domain_for(rows=rows)
    panel = interval_panel(
        rows=rows.drop("margin"),
        x_domain=x_domain,
        x_title=x_title,
        zero_label="same as ERA5",
        better_label="better than ERA5",
        better_direction="negative",
        panel_title=title,
        family_key=False,
        condition_key=False,
        figure_planning=planned_figure,
        colour_by_family=True,
        value_labels=True,
    )
    margin = float(rows["margin"][0])
    if margin != margin:  # noqa: PLR0124 - a control has no margin, so it draws no band
        return panel
    return alt.layer(margin_band(margin=margin, x_domain=x_domain), panel)


def _windows(*, base: dict[str, Any], window_label: str | None = None) -> list[Selector]:
    """List the selectors of a variable's whole row set and its two windows.

    Args:
        base: The fields every record shares.
        window_label: The `label` of the window records where it differs from the whole row set's.

    Returns:
        One selector per scope of `WINDOW_SCOPES`.
    """
    return [
        (
            WINDOW_LABELS[scope],
            {**base, "scope": scope}
            if scope == "all" or window_label is None
            else {**base, "label": window_label, "scope": scope},
        )
        for scope in WINDOW_SCOPES
    ]


def _figure(
    *,
    panels: dict[str, pl.DataFrame],
    units: dict[str, str],
    number: int,
    title: str,
    subtitle: Sequence[str],
) -> alt.VConcatChart:
    kind = planning(rows=list(panels.values()))
    drawn = [
        contrast_panel(
            rows=rows,
            title=name,
            x_title=f"UKV-CEDA minus ERA5 error ({units[name]})",
            planned_figure=kind,
        )
        for name, rows in panels.items()
    ]
    return figure(
        panels=drawn,
        number=number,
        title=title,
        subtitle=subtitle,
        figure_planning=kind,
    )


KEY_SUBTITLE: Final[str] = (
    "Each dot is UKV-CEDA's mean absolute error minus ERA5's, so a negative value favours "
    "UKV-CEDA, and each line is a 95% interval from resampling whole calendar months. The grey "
    "band is the margin a difference must pass to read as clear."
)


def interval_figures(
    *, set_a: Sequence[dict[str, Any]], set_b: Sequence[dict[str, Any]]
) -> dict[str, tuple[alt.VConcatChart, list[pl.DataFrame]]]:
    """Draw Figures 1 to 4 and return the rows each draws.

    Args:
        set_a: Set A's interval records.
        set_b: Set B's interval records.

    Returns:
        Each figure by its file stem, with its panels' rows for the check against the reports.
    """
    years = sorted({r["scope"] for r in set_b if r["kind"] == "year" and r["label"] == "P3"})
    half_years = ("October to March", "April to September")
    headline = {
        "P1: wind speed against stations": station_rows(
            records=set_a,
            selectors=_windows(
                base={"variable": "wind", "label": "P1"}, window_label="exploratory window"
            ),
        ),
        "P2: air temperature against stations": station_rows(
            records=set_a, selectors=_windows(base={"variable": "temperature", "label": "P2"})
        ),
        "P3: wind farm power, XGBoost models": set_b_rows(
            records=set_b, selectors=_windows(base={"label": "P3"}), with_second=True
        ),
        "P4: solar farm power, XGBoost models": set_b_rows(
            records=set_b, selectors=_windows(base={"label": "P4"}), with_second=True
        ),
    }
    units = {
        name: "m/s"
        if name.startswith("P1")
        else "K"
        if name.startswith("P2")
        else "points of capacity"
        for name in headline
    }
    set_a_splits = {
        f"{variable.capitalize()}, stations": station_rows(
            records=set_a,
            selectors=[
                *[
                    (scope.capitalize(), {"variable": variable, "scope": scope})
                    for scope in (*(f"year {y}" for y in range(2019, 2026)), *half_years)
                ],
                *[
                    (f"Station {s}", {"variable": variable, "scope": f"station {s}"})
                    for s in sorted(
                        {r["scope"][-2:] for r in set_a if r["scope"].startswith("station")}
                    )
                ],
            ],
        )
        for variable in ("wind", "temperature")
    }
    set_a_splits["Temperature by UKV-CEDA lead (P2-lead, planned)"] = station_rows(
        records=set_a,
        selectors=[
            (scope.capitalize(), {"variable": "temperature", "label": "P2-lead", "scope": scope})
            for scope in ("leads 0 to 2", "leads 3 to 5")
        ],
    )
    set_b_splits = {
        f"{name}, XGBoost models": set_b_rows(
            records=set_b,
            selectors=[
                (scope.capitalize(), {"label": label, "scope": scope})
                for scope in (*years, *half_years)
            ],
            with_second=True,
        )
        for name, label in (("P3 wind farm power", "P3"), ("P4 solar farm power", "P4"))
    }
    controls = {
        f"{domain.capitalize()} controls and checks": set_b_rows(
            records=set_b,
            selectors=[
                (label.capitalize(), {"domain": domain, "label": label, "scope": "all"})
                for label in labels
            ],
            with_second=False,
        )
        for domain, labels in (
            (
                "wind",
                (
                    "control",
                    "era5 against its shuffled arm",
                    "UKV-CEDA against its shuffled arm",
                    "hour-ending pair",
                    "ERA5 power-hour offset",
                    "UKV-CEDA power-hour offset",
                    "GPU against CPU",
                ),
            ),
            (
                "solar",
                (
                    "control",
                    "era5 against its shuffled arm",
                    "UKV-CEDA against its shuffled arm",
                ),
            ),
        )
    }
    split_units = {
        name: "m/s"
        if name.startswith("Wind, st")
        else "K"
        if name.endswith("stations") or name.startswith("Temperature")
        else "points of capacity"
        for name in {**set_a_splits, **set_b_splits}
    }
    control_units = dict.fromkeys(controls, "points of capacity")
    return {
        "ukv_vs_era5_headline": (
            _figure(
                panels=headline,
                units=units,
                number=1,
                title="UKV-CEDA against ERA5 for wind speed and air temperature, 2019 to 2026",
                subtitle=[
                    KEY_SUBTITLE,
                    (
                        "A hollow triangle is the second hyperparameter setting. A dashed line "
                        "has too few months for an interval."
                    ),
                ],
            ),
            list(headline.values()),
        ),
        "ukv_vs_era5_stations": (
            _figure(
                panels=set_a_splits,
                units=split_units,
                number=2,
                title="UKV-CEDA against ERA5 at four stations, by year, season, station and lead",
                subtitle=[KEY_SUBTITLE],
            ),
            list(set_a_splits.values()),
        ),
        "ukv_vs_era5_power_splits": (
            _figure(
                panels=set_b_splits,
                units=split_units,
                number=3,
                title="UKV-CEDA against ERA5 for farm power, by year and half-year",
                subtitle=[KEY_SUBTITLE],
            ),
            list(set_b_splits.values()),
        ),
        "ukv_vs_era5_controls": (
            _figure(
                panels=controls,
                units=control_units,
                number=4,
                title="Shuffled weather columns and power-hour checks",
                subtitle=[
                    KEY_SUBTITLE,
                    (
                        "The control shuffles each product's weather within a generator, month, "
                        "and hour of day, so the two shuffled arms should differ by about zero."
                    ),
                ],
            ),
            list(controls.values()),
        ),
    }


def pick_weeks(*, hourly: pl.DataFrame) -> dict[str, datetime]:
    """Pick three weeks by a stated rule, never by eye.

    Args:
        hourly: `time` (UTC) and `fraction`, the pooled power as a fraction of capacity.

    Returns:
        The first day of the week of highest mean output, of the largest standard deviation of
        output, and of lowest mean output, among the whole 7-day weeks.
    """
    first = hourly["time"].min()
    weeks = (
        hourly.with_columns(
            week=((pl.col("time") - first).dt.total_days() // WEEK_DAYS).cast(pl.Int64)
        )
        .group_by("week")
        .agg(
            mean=pl.col("fraction").mean(),
            spread=pl.col("fraction").std(),
            n=pl.len(),
            start=pl.col("time").min().dt.truncate("1d"),
        )
        .filter(pl.col("n") >= WEEK_DAYS * 24 // 2)
        .sort("week")
    )
    return {
        "highest mean output": weeks.sort("mean")["start"][-1],
        "largest spread": weeks.sort("spread")["start"][-1],
        "lowest mean output": weeks.sort("mean")["start"][0],
    }


def predictions_chart(*, losses: pl.DataFrame, arms: dict[str, str], title: str) -> alt.Chart:
    """Draw out-of-fold power against measured, per generator, in three chosen weeks.

    Args:
        losses: Saved losses at the primary setting with `actual_mw`, `prediction_mw`,
            `effective_capacity_mw`, `site`, `time`, `arm` and `seed`.
        arms: Each arm to draw, mapped to its label.
        title: The chart's title.

    Returns:
        The chart.
    """
    drawn = losses.filter(pl.col("arm").is_in(list(arms)), pl.col("setting") == PRIMARY_SETTING)
    measured = (
        drawn.filter(pl.col("arm") == next(iter(arms)))
        .group_by("site", "time")
        .agg(
            value=(pl.col("actual_mw") / pl.col("effective_capacity_mw")).first(),
        )
        .with_columns(series=pl.lit("Measured"))
    )
    predicted = (
        drawn.group_by("arm", "site", "time")
        .agg(value=(pl.col("prediction_mw") / pl.col("effective_capacity_mw")).mean())
        .with_columns(series=pl.col("arm").replace(arms))
        .drop("arm")
    )
    long = pl.concat([measured, predicted.select(measured.columns)])
    pooled = measured.group_by("time").agg(fraction=pl.col("value").mean())
    weeks = pick_weeks(hourly=pooled)
    parts = [
        long.filter(pl.col("time") >= start, pl.col("time") < start + timedelta(days=WEEK_DAYS))
        .with_columns(week=pl.lit(f"{name}: from {start:%Y-%m-%d}"))
        .with_columns(hours=(pl.col("time") - start).dt.total_hours())
        for name, start in weeks.items()
    ]
    data = pl.concat(parts)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:Q", title="Hours from the start of the week"),
            y=alt.Y("value:Q", title="Power, share of capacity"),
            color=alt.Color(
                "series:N",
                title="",
                scale=alt.Scale(
                    domain=["Measured", *arms.values()],
                    range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE][: len(arms) + 1],
                ),
            ),
        )
        .properties(width=170, height=70, title=title)
        .facet(row=alt.Row("site:N", title=""), column=alt.Column("week:N", title=""))
    )


def main() -> int:
    """Check each figure against its report, and write the SVG files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    set_a = pl.read_parquet(directory / STATION_INTERVALS_NAME).to_dicts()
    set_b = pl.read_parquet(directory / INTERVALS_NAME).to_dicts()
    reports = (directory / STATION_REPORT_NAME).read_text() + (directory / REPORT_NAME).read_text()
    arguments.assets_dir.mkdir(parents=True, exist_ok=True)
    drawn = interval_figures(set_a=set_a, set_b=set_b)
    for rows in (rows for _, panels in drawn.values() for rows in panels):
        check_against_report(rows=rows, report=reports, name="the reports")
    for stem, (chart, _) in drawn.items():
        chart.save(arguments.assets_dir / f"{stem}.svg")
    for domain, arms in (
        (
            "wind",
            {"era5_wind": "ERA5", "ukv_ceda_wind": "UKV-CEDA"},
        ),
        (
            "solar",
            {"solar_era5_temp": "ERA5 temperature", "solar_ukv_ceda_temp": "UKV-CEDA temperature"},
        ),
    ):
        losses = pl.read_parquet(directory / f"losses_{domain}.parquet")
        chart = predictions_chart(
            losses=losses,
            arms=arms,
            title=f"Out-of-fold {domain} farm power in three weeks chosen by rule",
        )
        chart.save(arguments.assets_dir / f"ukv_vs_era5_{domain}_weeks.svg")
    sys.stdout.write(f"Wrote figures to {arguments.assets_dir}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
