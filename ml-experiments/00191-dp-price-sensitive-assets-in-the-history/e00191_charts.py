# ruff: noqa: E501
"""Draw the figures of experiment 191 from the score file and the stored fits."""

import json
import pickle
from pathlib import Path
from typing import Final

import altair as alt
import e00191_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments/00191-dp-price-sensitive-assets-in-the-history")
WIDTH: Final[int] = CONTENT_WIDTH_PX - 80
ARM_LABELS: Final[dict[str, str]] = {
    "h": "H: true price at forecast time (cannot be forecast)",
    "m": "M: expected price given hour, day type, quarter",
    "p": "P: last settled day's price",
    "t": "T: price-driven half-hours removed from training",
}
CAPACITY_MW: Final[dict[str, float]] = {"D8": 10.379, "D14": 11.706}


def forest() -> None:
    """Figure 1: each arm's mean absolute error minus the reference arm's, at D8 and D14."""
    report = json.loads((common.WORK_DIR / "score_gated.json").read_text())
    rows = []
    for site in common.ASSET_SITES:
        row = report[site]["all"]
        base = row["a0"]["ae"]
        for arm, label in ARM_LABELS.items():
            c = row[f"{arm}_minus_a0_ae"]
            rows.append(
                {
                    "site": f"{site} (reference error {base:.1f} points)",
                    "label": label,
                    "diff": c["pct"],
                    "lo": c["low"] / base * 100,
                    "hi": c["high"] / base * 100,
                }
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
        )
    )
    layer = (
        base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Mean absolute error minus the reference's (% of the reference; below 0 is better)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=90).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0], "t": ["same as reference"]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (
        layer.properties(width=WIDTH - 330, height=150)
        .facet(
            row=alt.Row(
                "site:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=11)
            )
        )
        .properties(
            title=alt.TitleParams(
                "At D8 and D14, no price-aware arm lowers the error clearly",
                subtitle=[
                    "Dot: estimate. Line: 95% interval from resampling the 12 months of the scored year.",
                    "Orange dashed line: same error as the reference, which has no price input.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            )
        )
    )
    chart.save(DOCS_DIR / "figure_1_error_against_reference.svg")


def response() -> None:
    """Figure 2: the fitted price response in MW at each monthly fit, D8 and D14."""
    rows = []
    for site in common.ASSET_SITES:
        fits = pickle.loads((common.FITS_DIR / "gated" / f"{site}.pkl").read_bytes())
        for cut_fits in fits:
            theta = cut_fits.joint.theta_price
            rows.append(
                {
                    "site": site,
                    "cut": cut_fits.cut.replace(tzinfo=None),
                    "mw": float(theta.sum()) * CAPACITY_MW[site] / 100.0,
                }
            )
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("cut:T", title="Month of the fit (each fit uses data up to the 1st)"),
            y=alt.Y("mw:Q", title="Fitted price response (MW)"),
            color=alt.Color(
                "site:N",
                title=None,
                scale=alt.Scale(domain=["D8", "D14"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                "The model finds 1.2 to 1.9 MW of price response at D8 and almost none at D14",
                subtitle=[
                    "Output that the model takes off net load when all four price indicators are on,",
                    "from each monthly fit of the price-aware model.",
                ],
                fontSize=12,
                subtitleFontSize=10,
                anchor="start",
                offset=14,
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_fitted_price_response.svg")


def main() -> None:  # noqa: D103
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    forest()
    response()


if __name__ == "__main__":
    main()
