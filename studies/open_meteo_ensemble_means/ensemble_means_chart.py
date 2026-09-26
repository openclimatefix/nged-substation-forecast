"""Draw the ensemble-mean page's headline chart: each product's mean absolute error, solar and wind.

Reads `mae_by_arm.parquet` from `ensemble_means_mae.py`'s output folder and draws Figure 1, two
panels one above the other. The top panel is solar power and the bottom panel is wind power with
every product given its 10 m wind speed. Each row is one product, sorted best first, and the
reference (CAMS for solar, ERA5 for wind) and the no-weather baseline sit among the rows, so a
reader sees each product's absolute error beside both. A circle is the primary hyperparameter
setting and a vertical bar is the second setting. Nothing is drawn with a generator's name or an
interval, because the study is descriptive.

Run it with `uv run python studies/open_meteo_ensemble_means/ensemble_means_chart.py`, then
optimise the SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing it.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import altair as alt

# Importing the theme module registers and enables the OCF Altair theme as a side effect.
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, LABEL_WIDTH_PX, wrapped

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
from sources import STUDIES_DATA_DIR

_LOG: Final[logging.Logger] = logging.getLogger("ensemble_means_chart")

MAE_PATH: Final[Path] = STUDIES_DATA_DIR / "open_meteo_ensemble_means" / "mae_by_arm.parquet"
DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets" / "ensemble_means_mae.svg"
)

LABELS: Final[dict[str, str]] = {
    "mogreps_uk_mean": "MOGREPS-UK mean",
    "icon_d2_eps_mean": "ICON-D2-EPS mean",
    "icon_eu_eps_mean": "ICON-EU-EPS mean",
    "ecmwf_ens_025_mean": "ECMWF ENS mean (Open-Meteo)",
    "ecmwf_ens_local_mean": "ECMWF ENS mean (local)",
    "cams": "CAMS (reference)",
    "era5": "ERA5 (reference)",
    "no_weather": "No weather (baseline)",
}
"""Each arm's row label."""

ROLES: Final[dict[str, str]] = {
    "mogreps_uk_mean": "Open-Meteo ensemble mean",
    "icon_d2_eps_mean": "Open-Meteo ensemble mean",
    "icon_eu_eps_mean": "Open-Meteo ensemble mean",
    "ecmwf_ens_025_mean": "Open-Meteo ensemble mean",
    "ecmwf_ens_local_mean": "Local ECMWF ENS mean",
    "cams": "Reference",
    "era5": "Reference",
    "no_weather": "No-weather baseline",
}
ROLE_COLOURS: Final[dict[str, str]] = {
    "Open-Meteo ensemble mean": ocf.DATA_BLUE,
    "Local ECMWF ENS mean": ocf.DATA_SKY,
    "Reference": ocf.DATA_GREEN,
    "No-weather baseline": ocf.ENSEMBLE_LINE,
}
PANEL_HEIGHT_PX: Final[int] = 230
X_MAX: Final[float] = 15.0


def _panel(*, rows: pl.DataFrame, title: str) -> alt.LayerChart:
    """Draw one panel of dots and value labels, best product first.

    Args:
        rows: One design's rows from `mae_by_arm.parquet`, with `arm`, `setting` and `mae_pct`.
        title: The panel's title.

    Returns:
        The panel.
    """
    data = rows.with_columns(
        product=pl.col("arm").str.split(":").list.last(),
    ).with_columns(
        label=pl.col("product").replace_strict(LABELS),
        role=pl.col("product").replace_strict(ROLES),
    )
    order = data.filter(pl.col("setting") == "primary").sort("mae_pct")["label"].to_list()
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=LABEL_WIDTH_PX))
    x = alt.X(
        "mae_pct:Q",
        title="Mean absolute error (% of capacity; smaller is better)",
        scale=alt.Scale(domain=[0, X_MAX]),
    )
    colour = alt.Color(
        "role:N",
        scale=alt.Scale(domain=list(ROLE_COLOURS), range=list(ROLE_COLOURS.values())),
        legend=alt.Legend(title=None),
    )
    second = (
        alt.Chart(data.filter(pl.col("setting") == "sensitivity"))
        .mark_tick(color=ocf.BLACK_1, thickness=2, size=22, aria=False)
        .encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(data.filter(pl.col("setting") == "primary"))
        .mark_point(filled=True, size=90, opacity=1.0, aria=False)
        .encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    labels = (
        alt.Chart(data.filter(pl.col("setting") == "primary"))
        .mark_text(align="left", dx=10, aria=False, color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y=y, text=alt.Text("mae_pct:Q", format=".1f")
        )
    )
    return alt.layer(second, points, labels).properties(  # ty: ignore[invalid-return-type]
        title=alt.TitleParams(title, anchor="start"),
        width=CONTENT_WIDTH_PX - LABEL_WIDTH_PX - 10,
        height=PANEL_HEIGHT_PX,
    )


def figure(*, absolute: pl.DataFrame) -> alt.VConcatChart:
    """Stack the solar and wind panels under Figure 1's caption.

    Args:
        absolute: `mae_by_arm.parquet`.

    Returns:
        The figure.
    """
    caption = alt.TitleParams(
        wrapped(
            text=(
                "Figure 1: ICON-D2-EPS's ensemble mean had the lowest solar error and tied for "
                "the lowest wind error at 10 m"
            ),
            width=72,
        ),
        subtitle=[
            line
            for text in (
                (
                    "Each dot is the mean absolute error of one XGBoost model per generator, "
                    "scored on weeks it did not train on, over 25 June to 20 September 2026."
                ),
                (
                    "Six solar and three wind generators in Lincolnshire. Circle: primary "
                    "setting, with its error beside it. Vertical bar: second setting."
                ),
                (
                    "Every product's 10 m wind speed is used for wind. Ensemble means are "
                    "stitched series, not forecasts at a stated lead."
                ),
            )
            for line in wrapped(text=text)
        ],
        anchor="start",
        offset=14,
        subtitleColor=ocf.BLACK_1,
        subtitlePadding=6,
    )
    solar = absolute.filter(pl.col("design") == "solar")
    wind = absolute.filter(pl.col("design") == "wind_10m")
    return (
        alt.vconcat(
            _panel(rows=solar, title="Solar power"),
            _panel(rows=wind, title="Wind power"),
            spacing=24,
        )
        .properties(title=caption)
        .resolve_scale(color="shared")
        .configure_view(stroke=None)
        .configure_legend(orient="bottom", direction="horizontal")
    )


def main() -> int:
    """Draw the figure and save it as an SVG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    figure(absolute=pl.read_parquet(MAE_PATH)).save(arguments.output)
    _LOG.info("wrote %s", arguments.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
