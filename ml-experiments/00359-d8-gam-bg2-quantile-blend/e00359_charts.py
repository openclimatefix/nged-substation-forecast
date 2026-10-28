# ruff: noqa: E501
"""Draw the figures of experiment 359 from `work/blend_scores.json`.

Run with the folders of experiments 252, 264, and 354 on `PYTHONPATH`:

    PYTHONPATH=ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history:ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8 \
        uv run --no-sync python ml-experiments/00359-*/e00359_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00354_charts as c354  # ty: ignore[unresolved-import]
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00359-d8-gam-bg2-quantile-blend"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG / "work"
c354.DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
c354.PNG_DIR = WORK / "png"
ARM_LABELS: Final[dict[str, str]] = {
    "gam": "GAM stack alone",
    "b25": "Blend, 0.25 on the GAM stack",
    "b50": "Blend, 0.5 on the GAM stack",
    "b75": "Blend, 0.75 on the GAM stack",
}


def _rows(*, scores: dict, site: str, flag: str, base: str, group: str) -> list[dict]:
    return [
        {"label": label, "estimate": c[0], "low": c[1], "high": c[2], "group": group}
        for arm, label in ARM_LABELS.items()
        if arm != base
        for c in [scores[site]["contrast"][f"c_{arm}__{base}"][flag]["ci95"]]
    ]


def main() -> None:
    """Write the three figures."""
    scores = json.loads((WORK / "blend_scores.json").read_text())
    rows = _rows(scores=scores, site="D8", flag="h2", base="bg2", group="Second half")
    rows += [
        r | {"group": "Whole year"}
        for r in _rows(scores=scores, site="D8", flag="all", base="bg2", group="Whole year")
    ]
    rows = [
        r
        | {
            "label": f"{r['label']}, {'second half' if r['group'].startswith('Second') else 'whole year'}"
        }
        for r in rows
    ]
    c354.forest(
        rows=rows,
        x_title="Quantile score minus BG2's (points of capacity; smaller is better)",
        title="Figure 1: at D8 no blend clearly beats BG2 on the second half of the year",
        subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Zero is BG2 (7.593 over the year).",
        name="figure_1_blend_minus_bg2",
        height=300,
    )
    d8 = scores["D8"]["monthly"]
    monthly = [
        {"month": m, "arm": label, "score": v[key]}
        for m, v in d8.items()
        for label, key in (
            ("GAM stack alone", "gam"),
            ("Blend, 0.25", "b25"),
            ("Blend, 0.5", "b50"),
            ("BG2", "bg2"),
        )
    ]
    chart = (
        alt.Chart(pl.DataFrame(monthly))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Month of valid time", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y(
                "score:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False)
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["GAM stack alone", "Blend, 0.25", "Blend, 0.5", "BG2"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=c354.WIDTH,
            height=240,
            title=c354._title(
                "Figure 2: the 0.5 blend scores below BG2 in 8 of 12 months",
                "Mean quantile score of each valid-time month at D8.",
            ),
        )
    )
    c354._save(chart, name="figure_2_monthly")
    placebo_rows = [
        {
            "label": f"{site}: {ARM_LABELS[arm]} minus BG2",
            "estimate": c[0],
            "low": c[1],
            "high": c[2],
            "group": site,
        }
        for site in ("D2", "D3", "D4")
        for arm in ("gam", "b50")
        for c in [scores[site]["contrast"][f"c_{arm}__bg2"]["all"]["ci95"]]
    ]
    c354.forest(
        rows=placebo_rows,
        x_title="Quantile score minus BG2's (points of capacity; smaller is better)",
        title="Figure 3: at three Primaries with no gas engine the 0.5 blend scores below BG2",
        subtitle="Dot: estimate over the whole scored year. Line: 95% interval from resampling whole months.",
        name="figure_3_placebos",
        height=300,
    )


if __name__ == "__main__":
    main()
