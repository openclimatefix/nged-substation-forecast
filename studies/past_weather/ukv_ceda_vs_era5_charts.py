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
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import figure, interval_panel, planning, wrapped
from ukv_ceda_station_scores import INTERVALS_NAME as STATION_INTERVALS_NAME
from ukv_ceda_station_scores import PRIMARY_SCORE
from ukv_ceda_station_scores import REPORT_NAME as STATION_REPORT_NAME
from ukv_ceda_vs_era5_build import OUTPUT_DIR
from ukv_ceda_vs_era5_fit import INTERVALS_NAME, PRIMARY_SETTING, REPORT_NAME, SECOND_SETTING

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the published figures go."""

CAPACITY_UNIT: Final[str] = "points of capacity"
ROW_STEP_PX: Final[int] = 52
"""A row is taller than the default, because a row label wraps onto three lines."""
N_LEADS: Final[int] = 6

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
    *,
    records: Sequence[dict[str, Any]],
    selectors: Sequence[Selector],
    with_second: bool,
    with_errors: bool = True,
) -> pl.DataFrame:
    """Build the rows of a set B panel, with the second setting as a hollow marker.

    Args:
        records: Set B's interval records.
        selectors: Each row's label and the fields that pick its record, apart from the setting.
        with_second: Whether each row also has a record at the second setting.
        with_errors: Whether the label carries ERA5's and UKV-CEDA's absolute errors, which only a
            contrast of the two products has.

    Returns:
        Rows for `interval_panel`, with the absolute errors in the label where `with_errors`.
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
                "label": f"{label} ({text})" if with_errors else label,
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


def contrast_panel(
    *,
    rows: pl.DataFrame,
    title: str,
    x_title: str,
    planned_figure: Any,
    zero_label: str = "same as ERA5",
    better_label: str = "better than ERA5",
) -> Any:
    """Draw one panel of dots and intervals, with the margin band behind it.

    Args:
        rows: A panel's rows.
        title: The panel's title.
        x_title: The x axis title, naming the quantity and unit.
        planned_figure: What `planning` returns for the whole figure.
        zero_label: What a difference of zero means.
        better_label: What the better direction means.

    Returns:
        The layered panel.
    """
    x_domain = x_domain_for(rows=rows)
    panel = interval_panel(
        rows=rows.drop("margin"),
        x_domain=x_domain,
        x_title=x_title,
        zero_label=zero_label,
        better_label=better_label,
        better_direction="negative",
        row_step_px=ROW_STEP_PX,
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
    x_title: str | None = None,
    zero_label: str = "same as ERA5",
    better_label: str = "better than ERA5",
) -> alt.VConcatChart:
    kind = planning(rows=list(panels.values()))
    drawn = [
        contrast_panel(
            rows=rows,
            title=name,
            x_title=x_title or f"UKV-CEDA minus ERA5 error ({units[name]})",
            planned_figure=kind,
            zero_label=zero_label,
            better_label=better_label,
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


def _cap(text: str) -> str:
    """Capitalise the first letter only, leaving the rest as written."""
    return text[:1].upper() + text[1:]


def _units(*, names: Sequence[str], unit: str) -> dict[str, str]:
    return dict.fromkeys(names, unit)


def _lead_selectors(*, base: dict[str, Any]) -> list[Selector]:
    return [(f"Lead {n} h", {**base, "scope": f"lead {n} h"}) for n in range(N_LEADS)]


FigureRows = tuple[alt.VConcatChart, list[pl.DataFrame]]
"""A figure and the rows its panels draw, for the check against the reports."""


def headline_figure(
    *, set_a: Sequence[dict[str, Any]], set_b: Sequence[dict[str, Any]]
) -> FigureRows:
    """Draw Figure 1: the four planned contrasts by window.

    Args:
        set_a: Set A's interval records.
        set_b: Set B's interval records.

    Returns:
        The figure and its rows.
    """
    panels = {
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
        name: "m/s" if name.startswith("P1") else "K" if name.startswith("P2") else CAPACITY_UNIT
        for name in panels
    }
    chart = _figure(
        panels=panels,
        units=units,
        number=1,
        title=(
            "At four stations UKV-CEDA is closer than ERA5 for wind and temperature, but pooled "
            "over three wind farms ERA5 gives the lower power error, and the choice does not move "
            "solar power"
        ),
        subtitle=[
            KEY_SUBTITLE,
            (
                "A hollow triangle is the second hyperparameter setting. A dashed line has too few "
                "months for an interval."
            ),
        ],
    )
    return chart, list(panels.values())


def station_lead_figure(*, set_a: Sequence[dict[str, Any]]) -> FigureRows:
    """Draw Figure 5: set A by single UKV-CEDA lead.

    Args:
        set_a: Set A's interval records.

    Returns:
        The figure and its rows.
    """
    panels = {
        "Temperature by UKV-CEDA lead (post hoc)": station_rows(
            records=set_a,
            selectors=_lead_selectors(base={"variable": "temperature", "label": "post hoc lead"}),
        ),
        "Wind speed by UKV-CEDA lead (post hoc)": station_rows(
            records=set_a,
            selectors=_lead_selectors(base={"variable": "wind", "label": "post hoc lead"}),
        ),
    }
    units = {name: "K" if name.startswith("Temp") else "m/s" for name in panels}
    chart = _figure(
        panels=panels,
        units=units,
        number=5,
        title="UKV-CEDA's advantage at the four stations shrinks as the lead grows",
        subtitle=[
            KEY_SUBTITLE,
            (
                "A lead is the UTC hour modulo 6, so a lead is also an hour of day. All rows are "
                "post hoc."
            ),
        ],
    )
    return chart, list(panels.values())


def wind_lead_figure(*, set_b: Sequence[dict[str, Any]]) -> FigureRows:
    """Draw Figure 6: P3 by lead and by window, and the matched 10 m pair.

    Args:
        set_b: Set B's interval records.

    Returns:
        The figure and its rows.
    """
    published = next(r["scope"] for r in set_b if r["kind"] == "published window")
    era_scopes = {
        r["scope"][:5]: r["scope"] for r in set_b if r["kind"] == "era" and r["label"] == "P3"
    }
    windows: list[Selector] = [
        (WINDOW_LABELS[scope], {"label": "P3", "scope": scope}) for scope in WINDOW_SCOPES
    ]
    windows += [
        (
            "From 2024-08-12, the published wind page's window",
            {"label": "P3", "scope": published},
        ),
        ("Era 1, 2020-01 to 2026-01", {"label": "P3", "scope": era_scopes["era 1"]}),
        ("Era 2, 2026-02 onward", {"label": "P3", "scope": era_scopes["era 2"]}),
    ]
    panels = {
        "P3 by UKV-CEDA lead (post hoc)": set_b_rows(
            records=set_b, selectors=_lead_selectors(base={"label": "P3"}), with_second=True
        ),
        "P3 by window and era": set_b_rows(records=set_b, selectors=windows, with_second=True),
        "Matched 10 m pair by lead (post hoc)": set_b_rows(
            records=set_b,
            selectors=[
                ("All hours", {"label": "post hoc matched 10 m", "scope": "all"}),
                *_lead_selectors(base={"label": "post hoc matched 10 m"}),
            ],
            with_second=True,
        ),
    }
    chart = _figure(
        panels=panels,
        units=_units(names=list(panels), unit=CAPACITY_UNIT),
        number=6,
        title=(
            "ERA5's wind-power advantage grows with UKV-CEDA's lead, and holds with 10 m wind alone"
        ),
        subtitle=[
            KEY_SUBTITLE,
            (
                "Lead is also hour of day. Planned: P3 on all hours and P3's early and late "
                "windows. "
                "Every other row is post hoc."
            ),
        ],
    )
    return chart, list(panels.values())


def splits_figures(
    *, set_a: Sequence[dict[str, Any]], set_b: Sequence[dict[str, Any]]
) -> dict[str, FigureRows]:
    """Draw Figures 7 and 8: the station splits and the power splits.

    Args:
        set_a: Set A's interval records.
        set_b: Set B's interval records.

    Returns:
        Each figure by its file stem.
    """
    years = sorted({r["scope"] for r in set_b if r["kind"] == "year" and r["label"] == "P3"})
    half_years = ("October to March", "April to September")
    stations = sorted({r["scope"][-2:] for r in set_a if r["scope"].startswith("station")})
    set_a_panels = {
        f"{_cap(variable)}, by year, season, station, and without one station": station_rows(
            records=set_a,
            selectors=[
                *[
                    (_cap(scope), {"variable": variable, "scope": scope})
                    for scope in (*(f"year {y}" for y in range(2019, 2026)), *half_years)
                ],
                *[
                    (f"Station {s}", {"variable": variable, "scope": f"station {s}"})
                    for s in stations
                ],
                *[
                    (
                        f"Without station {s} (post hoc)",
                        {
                            "variable": variable,
                            "label": "post hoc leave one out",
                            "scope": f"without station {s}",
                        },
                    )
                    for s in stations
                ],
            ],
        )
        for variable in ("wind", "temperature")
    }
    set_a_panels["Temperature by UKV-CEDA lead group (P2-lead, planned)"] = station_rows(
        records=set_a,
        selectors=[
            (_cap(scope), {"variable": "temperature", "label": "P2-lead", "scope": scope})
            for scope in ("leads 0 to 2", "leads 3 to 5")
        ],
    )
    set_a_units = {name: "m/s" if name.startswith("Wind") else "K" for name in set_a_panels}
    set_b_panels = {
        f"{name}, XGBoost models": set_b_rows(
            records=set_b,
            selectors=[
                (_cap(scope), {"label": label, "scope": scope}) for scope in (*years, *half_years)
            ],
            with_second=True,
        )
        for name, label in (("P3 wind farm power", "P3"), ("P4 solar farm power", "P4"))
    }
    farms = sorted(
        {r["scope"] for r in set_b if r["kind"] == "site" and r["label"] == "P3 by generator"}
    )
    set_b_panels["P3 wind farm power by wind farm, XGBoost models"] = set_b_rows(
        records=set_b,
        selectors=[
            (
                f"Wind farm {farm.removeprefix('generator ')}",
                {"label": "P3 by generator", "scope": farm},
            )
            for farm in farms
        ],
        with_second=True,
    )
    return {
        "fig07_stations": (
            _figure(
                panels=set_a_panels,
                units=set_a_units,
                number=7,
                title=(
                    "UKV-CEDA is closer than ERA5 at three of four stations, and no one station "
                    "decides the sign"
                ),
                subtitle=[KEY_SUBTITLE],
            ),
            list(set_a_panels.values()),
        ),
        "fig08_power_splits": (
            _figure(
                panels=set_b_panels,
                units=_units(names=list(set_b_panels), unit=CAPACITY_UNIT),
                number=8,
                title=(
                    "ERA5's wind-power advantage is concentrated in October to March and at one of "
                    "three farms"
                ),
                subtitle=[KEY_SUBTITLE],
            ),
            list(set_b_panels.values()),
        ),
    }


def controls_figure(*, set_b: Sequence[dict[str, Any]]) -> FigureRows:
    """Draw Figure 9: the negative controls, the noise floor, and the power-hour checks.

    The arm-against-its-own-shuffled-arm rows are not drawn, because their size (12 points of
    capacity for wind) would flatten every other row. Figure 4 shows each arm's own error.

    Args:
        set_b: Set B's interval records.

    Returns:
        The figure and its rows.
    """
    wind_labels = {
        "Shuffled UKV-CEDA minus shuffled ERA5 (negative control)": ("wind", "control"),
        "ERA5 model refitted on the CPU minus fitted on the GPU": ("wind", "GPU against CPU"),
        "UKV-CEDA minus ERA5, power hour centred on the label": (
            "wind_hour_starting",
            "scan, centred pair",
        ),
        "UKV-CEDA minus ERA5, power hour ending at the label": (
            "wind_hour_starting",
            "scan, hour-ending pair",
        ),
        "UKV-CEDA minus ERA5, power hour starting at the label": (
            "wind_hour_starting",
            "scan, hour-starting pair",
        ),
        "ERA5, hour ending minus centred": ("wind_hour_starting", "scan, ERA5 hour-ending offset"),
        "ERA5, hour starting minus centred": (
            "wind_hour_starting",
            "scan, ERA5 hour-starting offset",
        ),
        "UKV-CEDA, hour ending minus centred": (
            "wind_hour_starting",
            "scan, UKV-CEDA hour-ending offset",
        ),
        "UKV-CEDA, hour starting minus centred": (
            "wind_hour_starting",
            "scan, UKV-CEDA hour-starting offset",
        ),
    }
    panels = {
        "Wind controls and checks": set_b_rows(
            records=set_b,
            selectors=[
                (text, {"domain": domain, "label": label, "scope": "all", "kind": "all"})
                for text, (domain, label) in wind_labels.items()
            ],
            with_second=False,
            with_errors=False,
        ),
        "Solar negative control": set_b_rows(
            records=set_b,
            selectors=[
                (
                    "Shuffled UKV-CEDA minus shuffled ERA5 temperature (negative control)",
                    {"domain": "solar", "label": "control", "scope": "all", "kind": "all"},
                )
            ],
            with_second=False,
            with_errors=False,
        ),
    }
    chart = _figure(
        panels=panels,
        units=_units(names=list(panels), unit=CAPACITY_UNIT),
        number=9,
        x_title="First arm minus second arm (points of capacity)",
        zero_label="no difference",
        better_label="first arm better",
        title=(
            "Shuffled UKV-CEDA and shuffled ERA5 differ by about zero, with a wind interval "
            "half-width close to the 0.16-point margin"
        ),
        subtitle=[
            (
                "Each dot is the first arm's mean absolute error minus the second's, named in the "
                "row label, and each line is a 95% interval from resampling whole calendar "
                "months. All rows are exploratory and have no margin."
            ),
            (
                "The control shuffles each product's weather within a generator, month, and hour "
                "of day, so the two shuffled arms should differ by about zero."
            ),
        ],
    )
    return chart, list(panels.values())


def interval_figures(
    *, set_a: Sequence[dict[str, Any]], set_b: Sequence[dict[str, Any]]
) -> dict[str, FigureRows]:
    """Draw every interval figure and return the rows each draws.

    Args:
        set_a: Set A's interval records.
        set_b: Set B's interval records.

    Returns:
        Each figure by its file stem, with its panels' rows for the check against the reports.
    """
    return {
        "fig01_headline": headline_figure(set_a=set_a, set_b=set_b),
        "fig05_station_leads": station_lead_figure(set_a=set_a),
        "fig06_wind_leads": wind_lead_figure(set_b=set_b),
        **splits_figures(set_a=set_a, set_b=set_b),
        "fig09_controls": controls_figure(set_b=set_b),
    }


ABSOLUTE_ROW: Final[re.Pattern[str]] = re.compile(
    r"^\| (\S+) \| (pooled|sensitivity) \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \| [\d,]+ \|$"
)
ABSOLUTE_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^#### (\w+): every arm's mean absolute error"
)
SHOWN_DOMAINS: Final[dict[str, str]] = {
    "wind": "Wind farms: as-available arms (ERA5 100 m and 10 m; UKV-CEDA 10 m and 925 hPa)",
    "wind_matched": "Wind farms: matched 10 m arms (post hoc)",
    "solar": "Solar farms: temperature arms",
}
ARM_LABELS: Final[dict[str, str]] = {
    "era5_wind": "ERA5 wind",
    "ukv_ceda_wind": "UKV-CEDA wind",
    "era5_wind_shuffled": "ERA5 wind, shuffled (control)",
    "ukv_ceda_wind_shuffled": "UKV-CEDA wind, shuffled (control)",
    "era5_wind_10m": "ERA5 10 m wind",
    "ukv_ceda_wind_10m": "UKV-CEDA 10 m wind",
    "solar_era5_temp": "ERA5 temperature",
    "solar_ukv_ceda_temp": "UKV-CEDA temperature",
    "solar_era5_temp_shuffled": "ERA5 temperature, shuffled (control)",
    "solar_ukv_ceda_temp_shuffled": "UKV-CEDA temperature, shuffled (control)",
}


def absolute_errors(*, report: str) -> pl.DataFrame:
    """Read every arm's absolute error and interval from the set B report.

    Args:
        report: The text of `report.md`.

    Returns:
        `domain`, `arm`, `setting`, `value`, `lower`, `upper`, one row per table row of a shown
        domain and a labelled arm.
    """
    rows: list[dict[str, Any]] = []
    domain = ""
    for line in report.splitlines():
        heading = ABSOLUTE_HEADING.match(line)
        if heading:
            domain = heading.group(1)
            continue
        match = ABSOLUTE_ROW.match(line)
        if match and domain in SHOWN_DOMAINS and match.group(1) in ARM_LABELS:
            arm, setting, value, lower, upper = match.groups()
            rows.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "setting": setting,
                    "value": float(value),
                    "lower": float(lower),
                    "upper": float(upper),
                }
            )
    return pl.DataFrame(rows)


def absolute_figure(*, errors: pl.DataFrame) -> alt.VConcatChart:
    """Draw Figure 4: every arm's mean absolute error, as a dot with its 95% interval.

    Args:
        errors: `absolute_errors`'s result.

    Returns:
        The figure.
    """
    panels = []
    for domain, title in SHOWN_DOMAINS.items():
        data = errors.filter(pl.col("domain") == domain).with_columns(
            label=pl.col("arm").replace(ARM_LABELS),
            product=pl.when(pl.col("arm").str.contains("ukv_ceda"))
            .then(pl.lit("UKV-CEDA"))
            .otherwise(pl.lit("ERA5")),
        )
        order = list(dict.fromkeys(data.sort("arm")["label"].to_list()))
        base = alt.Chart(data).encode(
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "product:N",
                title="",
                scale=alt.Scale(
                    domain=["ERA5", "UKV-CEDA"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
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
        )
        dot = base.mark_point(filled=True, size=60).encode(x="value:Q")  # ty: ignore[unresolved-attribute]
        panels.append(
            alt.layer(
                rule.transform_filter(alt.datum.setting == "pooled"),
                dot.transform_filter(alt.datum.setting == "pooled"),
            ).properties(
                width=380, height=26 * len(order), title=alt.TitleParams(title, anchor="start")
            )
        )
    return (
        alt.vconcat(*panels, spacing=40)
        .properties(
            title=alt.TitleParams(
                wrapped(
                    text="Figure 4: An XGBoost model given weather that has been shuffled errs "
                    "about 12 points more than one given the real weather",
                    width=72,
                ),
                subtitle=[
                    (
                        "Dot: mean absolute error at the primary setting. Line: 95% interval from "
                        "resampling whole calendar months and a fitting seed. Smaller is better. "
                        "Rows are scored on the same hours within each panel."
                    ),
                ],
                anchor="start",
                offset=24,
            )
        )
        .configure_view(stroke=None)
        .configure_legend(orient="bottom", direction="horizontal")
    )


def steps_figure(*, steps: pl.DataFrame) -> alt.VConcatChart:
    """Draw Figure 10: the monthly means of UKV-CEDA and ERA5 against the stations.

    Args:
        steps: `monthly_steps.parquet`.

    Returns:
        The figure, with the licence line in its subtitle.
    """
    long = steps.unpivot(
        ["ukv_minus_era5", "ukv_minus_station", "era5_minus_station"],
        index=["variable", "month"],
        variable_name="series",
        value_name="mean",
    ).with_columns(
        date=pl.col("month").str.to_date("%Y-%m"),
        series=pl.col("series").replace(SERIES_LABELS),
    )
    panels = [
        alt.Chart(long.filter(pl.col("variable") == variable))
        .mark_line(strokeWidth=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", title="Month", axis=alt.Axis(format="%Y", tickCount="year")),
            y=alt.Y("mean:Q", title=f"Monthly mean difference ({unit})"),
            color=alt.Color(
                "series:N",
                title="",
                scale=alt.Scale(
                    domain=list(SERIES_LABELS.values()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
            ),
        )
        .properties(width=520, height=110, title=title)
        for variable, unit, title in (
            ("wind", "m/s", "Wind speed"),
            ("temperature", "K", "Air temperature"),
        )
    ]
    return (
        alt.vconcat(*panels, spacing=20)
        .properties(
            title=alt.TitleParams(
                wrapped(
                    text="Figure 10: No step in UKV-CEDA minus ERA5 marks a change of UKV, but "
                    "the station series step together around August 2021",
                    width=72,
                ),
                subtitle=[
                    (
                        "Monthly means at the stations with values in every month. Post hoc and "
                        "exploratory: no threshold was set before the series was seen."
                    ),
                    (
                        "Contains Met Office UKV data from CEDA (CC BY-NC-SA 4.0), Met Office "
                        "(2016): NWP-UKV, Centre for Environmental Data Analysis."
                    ),
                ],
                anchor="start",
            )
        )
        .configure_view(stroke=None)
        .configure_legend(orient="bottom", direction="horizontal")
    )


SERIES_LABELS: Final[dict[str, str]] = {
    "ukv_minus_era5": "UKV-CEDA minus ERA5",
    "ukv_minus_station": "UKV-CEDA minus station",
    "era5_minus_station": "ERA5 minus station",
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


def predictions_chart(
    *, losses: pl.DataFrame, arms: dict[str, str], title: str
) -> tuple[alt.FacetChart, dict[str, datetime]]:
    """Draw out-of-fold power against measured, per generator, in three chosen weeks.

    Args:
        losses: Saved losses at the primary setting with `actual_mw`, `prediction_mw`,
            `effective_capacity_mw`, `site`, `time`, `arm` and `seed`.
        arms: Each arm to draw, mapped to its label.
        title: The chart's title.

    Returns:
        The chart, and the first day of each chosen week, which only the text may carry as a month
        and year.
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
        .with_columns(week=pl.lit(name.capitalize()))
        .with_columns(hours=(pl.col("time") - start).dt.total_hours())
        for name, start in weeks.items()
    ]
    grid = (
        pl.DataFrame({"hours": range(WEEK_DAYS * 24)})
        .join(pl.DataFrame({"week": [p["week"][0] for p in parts]}), how="cross")
        .join(long.select("site", "series").unique(), how="cross")
    )
    data = grid.join(
        pl.concat(parts).select("week", "site", "series", "hours", "value"),
        on=["week", "site", "series", "hours"],
        how="left",
    )
    return (
        (
            alt.Chart(data)
            .mark_line(strokeWidth=1, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("hours:Q", title="Hours from the start of the week"),
                y=alt.Y("value:Q", title=None),
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(
                        domain=["Measured", *arms.values()],
                        range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE][: len(arms) + 1],
                    ),
                ),
            )
            .properties(width=150, height=60)
            .facet(row=alt.Row("site:N", title=""), column=alt.Column("week:N", title=""))
            .properties(title=alt.TitleParams(title, anchor="start", offset=16))
        ),
        weeks,
    )


WEEKS_NAME: Final[str] = "chart_weeks.md"
"""The report that names the month and year of each week the prediction figures draw."""


def main() -> int:
    """Check each figure against its report, and write the SVG files and the weeks report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    set_a = pl.read_parquet(directory / STATION_INTERVALS_NAME).to_dicts()
    set_b = pl.read_parquet(directory / INTERVALS_NAME).to_dicts()
    report = (directory / REPORT_NAME).read_text()
    reports = (directory / STATION_REPORT_NAME).read_text() + report
    assets: Path = arguments.assets_dir
    assets.mkdir(parents=True, exist_ok=True)
    drawn = interval_figures(set_a=set_a, set_b=set_b)
    for rows in (rows for _, panels in drawn.values() for rows in panels):
        check_against_report(rows=rows, report=reports, name="the reports")
    for stem, (chart, _) in drawn.items():
        chart.save(assets / f"{stem}.svg")
    absolute_figure(errors=absolute_errors(report=report)).save(
        assets / "fig04_absolute_errors.svg"
    )
    steps_figure(steps=pl.read_parquet(directory / "monthly_steps.parquet")).save(
        assets / "fig10_monthly_steps.svg"
    )
    lines = ["### Weeks drawn in Figures 2 and 3, chosen by rule", ""]
    for number, (domain, arms) in enumerate(
        (
            ("wind", {"era5_wind": "ERA5", "ukv_ceda_wind": "UKV-CEDA"}),
            (
                "solar",
                {
                    "solar_era5_temp": "ERA5 temperature",
                    "solar_ukv_ceda_temp": "UKV-CEDA temperature",
                },
            ),
        ),
        start=2,
    ):
        chart, weeks = predictions_chart(
            losses=pl.read_parquet(directory / f"losses_{domain}.parquet"),
            arms=arms,
            title=(
                f"Figure {number}: Out-of-fold {domain} farm power as a share of capacity, "
                "in three weeks chosen by rule"
            ),
        )
        chart.save(assets / f"fig0{number}_{domain}_weeks.svg")
        lines += [f"- {domain}, {name}: {start:%B %Y}." for name, start in weeks.items()]
    weeks_path = directory / WEEKS_NAME
    if not weeks_path.exists():
        weeks_path.write_text("\n".join(lines) + "\n")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.write(f"Wrote figures to {assets}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
