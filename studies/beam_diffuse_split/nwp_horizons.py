"""Draw the bar chart of each NWP's forecast horizon on the data sources page.

One-off script for the chart on
<https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/#weather-data>.
Every horizon in ``HORIZONS`` comes from `docs/roadmap/data-sources.md` itself, from
`docs/background/weather-products-survey.md`, or from the producer's own documentation, and each
row's `source` field says which. Where a model's horizon differs by run (UKV reaches 120 hours on
its 03 and 15 UTC runs and 54 on the rest, for example), the value here is the longest routine run.
Reanalyses and satellite products carry no forecast horizon and are left out. GFS, GEFS, ARPEGE,
AROME, and the shared DMI/KNMI HARMONIE-AROME feed are charted even though `data-sources.md`
discusses them only as products the project does not read or evaluate — their horizons come from
`weather-products-survey.md`'s product table.

Run it with `uv run python studies/beam_diffuse_split/nwp_horizons.py`, then optimise the SVG with
`npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

HORIZONS: Final[tuple[tuple[str, float, str], ...]] = (
    (
        "GEFS",
        35,
        (
            'weather-products-survey.md: NOAA GEFS 35-day, "Runs, horizon, step": "00 UTC; '
            "840 h\" (35 days); NOAA's SCN 20-75 GEFS v12 upgrade notice "
            '(nwp-model-upgrades.csv) gives the full picture: "16-day range (35 days at 00 '
            'UTC)" — 16 days on the other three daily cycles'
        ),
    ),
    (
        "GFS",
        16,
        'weather-products-survey.md: NOAA GFS, "Runs, horizon, step": "4 a day; 16 days"',
    ),
    (
        "ECMWF ENS",
        15,
        (
            'data-sources.md: the AIFS-ENS row calls out "the same 51 members and 15-day horizon '
            'as the physics ensemble", i.e. ECMWF ENS'
        ),
    ),
    ("AIFS-ENS", 15, 'data-sources.md: "15-day horizon"'),
    (
        "AIFS Single",
        15,
        (
            "ECMWF's open data documentation "
            "(https://confluence.ecmwf.int/display/FCST/Implementation+of+AIFS+Single+v1): "
            "AIFS Single runs the same 00/06/12/18 UTC cycles and 0-to-360-hour, 6-hourly step "
            "grid as AIFS-ENS"
        ),
    ),
    (
        "ECMWF IFS HRES (9 km)",
        15,
        (
            "weather-products-survey.md: the 00 and 12 UTC runs reach 15 days (06 and 18 UTC "
            "reach 144 hours)"
        ),
    ),
    (
        "WeatherNext 3",
        15,
        (
            'data-sources.md: "running 15 days from the 00/06/12/18 UTC cycles" (48 hours from '
            "the hourly interim runs)"
        ),
    ),
    (
        "ICON global",
        7.5,
        (
            "DWD's ICON documentation, via Open-Meteo (https://open-meteo.com/en/docs/dwd-api): "
            "the 00 and 12 UTC runs reach 7.5 days (06 and 18 UTC reach 120 hours)"
        ),
    ),
    (
        "Met Office global (10 km)",
        7,
        'data-sources.md: "the global model reaches 168 hours"',
    ),
    ("MOGREPS-UK", 5.25, 'data-sources.md: "MOGREPS-UK reaches 126 hours"'),
    (
        "UKV",
        5,
        'data-sources.md: "UKV reaches 120 hours on its 03 and 15 UTC runs" (54 hours on the rest)',
    ),
    ("ICON-EU", 5, 'data-sources.md: "4 runs a day out to 5 days"'),
    (
        "ARPEGE",
        4,
        (
            'weather-products-survey.md: Meteo-France ARPEGE Europe and World rows, "Runs, '
            'horizon, step": "4 a day; 4 days"'
        ),
    ),
    (
        "HARMONIE-AROME (DMI/KNMI)",
        2.5,
        (
            "data-sources.md: DMI and KNMI serve one shared UWC-West HARMONIE-AROME run; "
            'weather-products-survey.md gives both "Runs, horizon, step" as 2.5 days ("8 a day; '
            '2.5 days" for DMI, "Hourly; 2.5 days" for KNMI)'
        ),
    ),
    (
        "ICON-D2",
        2,
        (
            "DWD's ICON documentation, via Open-Meteo (https://open-meteo.com/en/docs/dwd-api): "
            "8 runs a day, each reaching 2 days"
        ),
    ),
    (
        "AROME France",
        2,
        (
            'weather-products-survey.md: Meteo-France AROME France row, "Runs, horizon, '
            'step": "8 a day; 2 days"'
        ),
    ),
)
"""Each charted model's longest routine forecast horizon in days, and where it comes from."""

NGED_HORIZON_DAYS: Final[int] = 14
"""NGED's forecast horizon, per `docs/roadmap/data-sources.md`'s Met Office discussion: "No Met
Office model covers NGED's 14-day horizon"."""

X_MAX_DAYS: Final[int] = 35
"""The x axis's upper bound: the longest horizon any charted model reaches (GEFS, on its 00 UTC
cycle)."""

WIDTH_PX: Final[int] = 620
HEIGHT_PX: Final[int] = 410

OUTPUT_PATH: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "roadmap" / "assets" / "nwp_horizons.svg"
)
"""Where the data-sources page reads the chart from."""


def horizon_chart() -> alt.LayerChart:
    """Draws the horizontal bar chart of each NWP's forecast horizon.

    Returns:
        The bars, sorted longest-to-shortest, layered with a reference rule at
        `NGED_HORIZON_DAYS` and a direct label of each bar's horizon.
    """
    rows = [{"model": model, "horizon_days": horizon} for model, horizon, _ in HORIZONS]
    model_order = [model for model, _, _ in sorted(HORIZONS, key=lambda row: row[1], reverse=True)]

    base = alt.Chart(alt.Data(values=rows))
    bars = base.mark_bar(color=ocf.BRAND_ORANGE, height=14).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "horizon_days:Q",
            title="Forecast horizon (days)",
            scale=alt.Scale(domain=[0, X_MAX_DAYS]),
            axis=alt.Axis(values=list(range(0, X_MAX_DAYS + 1, 5))),
        ),
        y=alt.Y("model:N", title=None, sort=model_order),
    )
    labels = base.mark_text(
        font=ocf.FONT_LABEL, fontSize=10, color=ocf.TEXT, align="left", dx=4
    ).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("horizon_days:Q"),
        y=alt.Y("model:N", sort=model_order),
        text=alt.Text("horizon_days:Q", format=".3~g"),
    )
    nged_rule = (
        alt.Chart(alt.Data(values=[{"horizon_days": NGED_HORIZON_DAYS}]))
        .mark_rule(color=ocf.TEXT, strokeWidth=1.5, strokeDash=[4, 3])
        .encode(x=alt.X("horizon_days:Q"))  # ty: ignore[unresolved-attribute]
    )
    nged_label_row = [{"horizon_days": NGED_HORIZON_DAYS, "label": "NGED's forecast horizon"}]
    nged_label = (
        alt.Chart(alt.Data(values=nged_label_row))
        .mark_text(
            font=ocf.FONT_LABEL,
            fontSize=10,
            color=ocf.TEXT,
            align="right",
            angle=270,
            dx=-4,
            dy=-4,
            baseline="bottom",
        )
        .encode(x=alt.X("horizon_days:Q"), text=alt.Text("label:N"))  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=[bars, labels, nged_rule, nged_label]).properties(
        width=WIDTH_PX,
        height=HEIGHT_PX,
        title=alt.Title(
            text=("Only ECMWF's models, WeatherNext 3, GFS, and GEFS reach NGED's 14-day horizon"),
            subtitle=[
                "Each bar is the model's longest routine run. Where a horizon differs by run,",
                "the shorter runs are noted in the source list below the chart.",
            ],
            anchor="start",
        ),
    )


def main() -> int:
    """Writes the chart to `OUTPUT_PATH`.

    Returns:
        The process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    horizon_chart().save(OUTPUT_PATH)
    _LOG.info("wrote %s", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
