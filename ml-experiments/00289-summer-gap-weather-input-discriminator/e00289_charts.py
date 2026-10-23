# ruff: noqa: E501
"""Draw the figures of experiment 289 from `work/report.json`.

uv run --no-sync python ml-experiments/00289-*/e00289_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00289_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]


def _chart(data: pl.DataFrame) -> Any:
    return alt.Chart(data)


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def dots(
    *,
    rows: list[dict],
    order: list[str],
    title: str,
    subtitle: str,
    xtitle: str,
    name: str,
    legend: str,
    zero: bool = True,
) -> None:
    """Dot (estimate) and line (interval) per label, coloured by `group`."""
    data = pl.DataFrame(rows)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    colour = alt.Color(
        "group:N",
        title=legend,
        scale=alt.Scale(domain=groups, range=COLOURS[: len(groups)]),
        legend=alt.Legend(orient="top"),
    )
    y = alt.Y("label:N", sort=order, title=None)
    offset = alt.YOffset("group:N", sort=groups)
    base = _chart(data)
    interval = base.mark_rule(strokeWidth=2).encode(
        y=y, yOffset=offset, x=alt.X("lo:Q", title=xtitle), x2="hi:Q", color=colour
    )
    point = base.mark_circle(size=40, opacity=1).encode(
        y=y, yOffset=offset, x="est:Q", color=colour
    )
    layers = [interval, point]
    if zero:
        layers.insert(
            0,
            _chart(pl.DataFrame({"z": [0.0]}))
            .mark_rule(color="#444444", strokeDash=[4, 3])
            .encode(x="z:Q"),
        )
    chart = alt.layer(*layers).properties(
        width=WIDTH,
        height=14 * len(order) * len(groups) * 0.8 + 40,
        title=alt.TitleParams(
            text=textwrap.wrap(title, 85),
            subtitle=textwrap.wrap(subtitle, 110),
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )
    _save(chart.configure_view(stroke=None), name=name)


def row(*, label: str, group: str, c: dict) -> dict:
    """Return one chart row from an estimate with its interval."""
    return {"label": label, "group": group, "est": c["est"], "lo": c["lo"], "hi": c["hi"]}


def main() -> None:
    """Draw every figure."""
    r = json.loads((common.WORK_DIR / "report.json").read_text())
    amj, jas = r["decision"], r["negative_control_jas"]
    arms = [
        ("score_bg2", "BG2"),
        ("score_d_ens", "Arm D, ensemble sunshine"),
        ("score_d_id", "Arm D, ensemble sunshine, no calibration"),
        ("score_d_cams", "Arm D, realised sunshine"),
    ]
    dots(
        rows=[
            row(label=n, group=g, c=c[k])
            for g, c in (("April to June", amj), ("July to September", jas))
            for k, n in arms
        ],
        order=[n for _, n in arms],
        title="Realised sunshine lowers arm D's error in April to June and in July to September alike",
        subtitle="Quantile score (points of effective capacity, lower is better), daylight half-hours at 0 to 36 hours ahead, D5, D9, D10, and D16 averaged; 97.5% intervals over ISO weeks.",
        xtitle="Quantile score (points)",
        name="figure_1_scores",
        legend="Period",
        zero=False,
    )
    dots(
        rows=[
            row(label=n, group=g, c=c[k])
            for g, c in (("April to June", amj), ("July to September", jas))
            for k, n in (
                ("gap_qs", "Gap to BG2, ensemble sunshine"),
                ("gap_qs_cams", "Gap to BG2, realised sunshine"),
            )
        ],
        order=["Gap to BG2, ensemble sunshine", "Gap to BG2, realised sunshine"],
        title="Realised sunshine closes 71% of the April to June gap, and turns July to September into a win over BG2",
        subtitle="Arm D minus BG2 in quantile score; positive means arm D is worse. Same rows and intervals as figure 1.",
        xtitle="Arm D minus BG2 (points)",
        name="figure_2_gap_closure",
        legend="Period",
    )
    seasonal = r["post_hoc_seasonal"]["qs"]
    dots(
        rows=[
            row(label=n, group="April-to-June gap beyond July to September's", c=seasonal[k])
            for k, n in (
                ("excess_before", "Excess with ensemble sunshine"),
                ("excess_after_cams", "Excess with realised sunshine"),
                ("removed", "Removed by realised sunshine"),
            )
        ],
        order=[
            "Excess with ensemble sunshine",
            "Excess with realised sunshine",
            "Removed by realised sunshine",
        ],
        title="Realised sunshine removes about 14% of the gap that is particular to April to June (post hoc)",
        subtitle="(April to June gap) minus (July to September gap) in quantile score, arm D minus BG2, 0 to 36 hours ahead; 95% intervals.",
        xtitle="Points of quantile score",
        name="figure_3_seasonal_excess",
        legend="Contrast",
    )
    sites = ["D5", "D9", "D10", "D11", "D14", "D16"]
    dots(
        rows=[
            row(label=s, group=g, c=r["by_site"][s][k])
            for s in sites
            for g, k in (
                ("Gap to BG2, ensemble sunshine", "gap_qs"),
                ("Gain from realised sunshine", "C1_qs_ens_minus_cams"),
            )
        ],
        order=sites,
        title="Realised sunshine helps most where arm D's gap is smallest, and least at D5, where the gap is largest",
        subtitle="Quantile score in points, April to June, daylight, 0 to 36 hours ahead, by anonymised Primary substation; 95% intervals.",
        xtitle="Quantile score (points)",
        name="figure_4_by_site",
        legend="Quantity",
    )
    bands = list(r["by_lead_band"])
    dots(
        rows=[
            row(label=b, group=g, c=r["by_lead_band"][b][k])
            for b in bands
            for g, k in (
                ("Gap to BG2, ensemble sunshine", "gap_qs"),
                ("Gain from realised sunshine", "C1_qs_ens_minus_cams"),
            )
        ],
        order=bands,
        title="The gain from realised sunshine grows with lead, so at long leads it is a ceiling no forecast can reach",
        subtitle="Quantile score in points, April to June, daylight half-hours, D5, D9, D10, and D16 averaged; 95% intervals.",
        xtitle="Quantile score (points)",
        name="figure_5_by_lead",
        legend="Quantity",
    )


if __name__ == "__main__":
    main()
