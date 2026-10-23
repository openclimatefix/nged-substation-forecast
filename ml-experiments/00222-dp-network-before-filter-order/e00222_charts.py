# ruff: noqa: E501
"""Draw experiment 222's figures from the stored score files.

uv run --no-sync python ml-experiments/00222-*/e00222_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00222_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
ZERO: Final[alt.Chart] = (
    alt.Chart(pl.DataFrame({"x": [0.0]}))
    .mark_rule(color="#808080", strokeDash=[4, 3])
    .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def figure_contrasts(*, main: dict, blocked: dict) -> alt.TopLevelMixin:
    """Headline: each contrast at D+1, with its 99% month-resampled interval."""
    rows = []
    for variant, score in (
        ("pre-registered network", main),
        ("post-hoc: out-of-time stopping", blocked),
    ):
        for label, name in (
            ("C1 PRE minus POST", "C1 PRE-POST"),
            ("C2 PRE minus R0", "C2 PRE-R0"),
            ("C3 PRE-T minus PRE", "C3 PRE-T minus PRE"),
        ):
            cell = score["contrast"][name]["by_group"]["d_plus_1"]
            rows.append(
                {
                    "variant": variant,
                    "contrast": label,
                    "difference": cell["mean"],
                    "lower": cell["99%"][0],
                    "upper": cell["99%"][1],
                }
            )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("contrast:N", sort=None, title=None),
        color=alt.Color(
            "variant:N",
            legend=alt.Legend(orient="top", title=None, labelLimit=320),
            scale=alt.Scale(
                domain=["pre-registered network", "post-hoc: out-of-time stopping"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
        ),
        yOffset="variant:N",
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Quantile score difference at day D+1, points (negative is better)"
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=90).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    return (ZERO + bars + points).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "A network before the filter does not beat one after it",
            subtitle=[
                "Dot: estimate. Line: 99% interval from resampling the 12 months of the scored year.",
                "Five decision substations, leads 15.5 to 39 hours. Dashed line: no difference.",
            ],
        ),
    )


def figure_c1_by_decile(*, score: dict) -> alt.TopLevelMixin:
    """PRE minus POST by decile of the driver temperature."""
    rows = [
        {
            "decile": k + 1,
            "difference": cell["mean"],
            "lower": cell["95%"][0],
            "upper": cell["95%"][1],
        }
        for k, cell in enumerate(score["c1_by_decile"])
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "decile:O",
            title="Decile of the ensemble-mean temperature (1 = coldest, 10 = warmest)",
            axis=alt.Axis(labelAngle=0),
        )
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lower:Q", title="PRE minus POST, points (negative is better)"), y2="upper:Q"
    )
    points = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(y="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + bars + points).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            "The network before the filter is worse than the one after it in every temperature decile",
            subtitle=[
                "Dot: estimate. Line: 95% interval from resampling months. Day-ahead leads, the five",
                "decision substations, pre-registered network. Dashed line: no difference.",
            ],
        ),
    )


def figure_level(*, main: dict, blocked: dict) -> alt.TopLevelMixin:
    """The filter's mean absolute level by temperature decile, for R0 and both PRE networks."""
    rows = []
    for label, values in (
        (
            "R0 (arm D, and POST: the network after the filter leaves the level unchanged)",
            main["level_by_decile"]["R0"],
        ),
        ("PRE, pre-registered network", main["level_by_decile"]["PRE"]),
        ("PRE, out-of-time stopping (post hoc)", blocked["level_by_decile"]["PRE"]),
    ):
        rows += [{"series": label, "decile": k + 1, "level": v} for k, v in enumerate(values)]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=2.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "decile:O",
                title="Decile of the ensemble-mean temperature (1 = coldest, 10 = warmest)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "level:Q", title="Mean absolute filter level, points", scale=alt.Scale(zero=True)
            ),
            color=alt.Color(
                "series:N",
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=520),
                scale=alt.Scale(
                    domain=[
                        "R0 (arm D, and POST: the network after the filter leaves the level unchanged)",
                        "PRE, pre-registered network",
                        "PRE, out-of-time stopping (post hoc)",
                    ],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY],
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                "Correcting the convex model first does not shrink the filter's level on extreme days",
                subtitle=[
                    "Level of the Kalman filter at the forecast origin, day-ahead leads, five decision",
                    "substations, scored year. A smaller value means the filter holds less of the error.",
                ],
            ),
        )
    )


def main() -> None:
    """Write every figure."""
    main_score = json.loads((common.WORK_DIR / "score.json").read_text())
    blocked = json.loads((common.WORK_DIR / "score_blocked.json").read_text())
    _save(figure_contrasts(main=main_score, blocked=blocked), name="figure_1_contrasts")
    _save(figure_c1_by_decile(score=main_score), name="figure_2_c1_by_decile")
    _save(figure_level(main=main_score, blocked=blocked), name="figure_3_level_by_decile")


if __name__ == "__main__":
    main()
