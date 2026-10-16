"""Draw experiment 00057's figures from the saved results, as SVG, into the docs folder.

1. Each arm minus the ENS-only arm (or the named reference), at 24 to 54 hours and at lead day 0.
2. The ENS-plus-UKV arm minus the ENS-only arm, per solar farm.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00057-product-trained-xgboost-solar-farms/e00057_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00057_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00057-{common.SLUG}"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "6 solar farms, 52 Tuesday runs, 3 seeds, July 2025 to June 2026; UKV from the CEDA archive."
)
CONTRASTS: Final[dict[tuple[str, str], str]] = {
    ("b", "a"): "ENS + UKV minus ENS only",
    ("b", "b_3h"): "ENS + UKV minus ENS + 3-hourly UKV",
    ("c", "a"): "UKV only minus ENS only",
    ("b_placebo", "a"): "ENS + shuffled UKV minus ENS only",
    ("b", "ref_best_guess"): "ENS + UKV minus ERA5-trained best guess",
    ("a", "ref_best_guess"): "ENS only minus ERA5-trained best guess",
}
BANDS: Final[dict[str, str]] = {
    "days 1-2 (24-54 h)": "24 to 54 hours",
    "lead day 0 (0-24 h)": "Lead day 0 (exploratory)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _layers(*, data: pl.DataFrame, y: alt.Y, height: int) -> alt.TopLevelMixin:
    colour = alt.Color(
        "scope_label:N",
        scale=alt.Scale(domain=list(BANDS.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(data).encode(y=y, yOffset=alt.YOffset("scope_label:N"), color=colour)
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Difference in quantile score (points of capacity; negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-common.THRESHOLD]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, threshold, bars, points).properties(width=PLOT_WIDTH, height=height)


def figure_1(*, results: dict) -> alt.TopLevelMixin:
    """Each contrast at 24 to 54 hours and at lead day 0."""
    rows = [
        e
        for e in results["exploratory"] + [results["c1"]]
        if (e["treatment"], e["reference"]) in CONTRASTS
        and not (
            (e["treatment"], e["reference"]) == ("b", "a")
            and e["band"] == "days 1-2 (24-54 h)"
            and "verdict" not in e
        )
        and e["band"] in BANDS
        and e.get("series") is None
        and "split" not in e
    ]
    data = pl.DataFrame(rows).with_columns(
        row=pl.struct("treatment", "reference").map_elements(
            lambda s: CONTRASTS[(s["treatment"], s["reference"])], return_dtype=pl.String
        ),
        scope_label=pl.col("band").replace_strict(BANDS),
    )
    chart = _layers(
        data=data,
        y=alt.Y("row:N", sort=list(CONTRASTS.values()), title=None, axis=alt.Axis(labelLimit=300)),
        height=240,
    )
    return chart.properties(
        title=_title(
            text="Contrasts between arms trained on ENS, UKV, and both",
            lines=[
                "Quantile score of the first arm minus the second (smaller is better). Solid grey "
                "line: no difference. Dashed line: the 0.10-point threshold. Dot: estimate. Line: "
                "95% interval from re-drawing whole months. " + SCOPE
            ],
        )
    )


def figure_2(*, results: dict) -> alt.TopLevelMixin:
    """The ENS-plus-UKV arm minus the ENS-only arm, per solar farm."""
    rows = [e for e in results["exploratory"] if "site" in e and e["band"] in BANDS]
    data = pl.DataFrame(rows).with_columns(scope_label=pl.col("band").replace_strict(BANDS))
    chart = _layers(data=data, y=alt.Y("site:N", title="Solar farm"), height=140)
    return chart.properties(
        title=_title(
            text="Per solar farm: ENS plus UKV minus ENS only",
            lines=[
                "Quantile score of the ENS-plus-UKV arm minus the ENS-only arm (smaller is "
                "better). Solid grey line: no difference. Dashed line: the 0.10-point threshold. "
                "Dot: estimate. Line: 95% interval from re-drawing whole months. " + SCOPE
            ],
        )
    )


def main() -> None:
    """Write both figures."""
    results = json.loads((common.WORK_DIR / "results" / "results.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(results=results)),
        ("figure_2", figure_2(results=results)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
