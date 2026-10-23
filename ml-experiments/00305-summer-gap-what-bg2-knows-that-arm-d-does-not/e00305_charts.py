# ruff: noqa: E501
"""Draw the figures of experiment 305 from `work/report.json`.

uv run --no-sync python ml-experiments/00305-*/e00305_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00305_fit as fit  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / "00305-summer-gap-what-bg2-knows-that-arm-d-does-not"
)  # fmt: skip
WIDTH: Final[int] = 640
LABELS: Final[dict[str, str]] = {
    "no_solar_x_temp_or_lags": "No solar-by-temperature or solar-by-lags interaction",
    "no_solar_x_lags": "No solar-by-lags interaction",
    "no_solar_x_temp": "No solar-by-temperature interaction",
    "additive_post_hoc": "No interaction of any kind (post hoc)",
}
COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN]


def _chart(data: pl.DataFrame) -> Any:
    return alt.Chart(data)


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(fit.DATA_DIR / "work" / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def figure_share(*, report: dict) -> None:
    """Draw the share of BG2's advantage that each interaction ban removes, with intervals."""
    rows = [
        {"group": group, "variant": LABELS[v], "est": x["est"], "lo": x["lo"], "hi": x["hi"]}
        for group, variants in report["S"].items()
        for v, x in variants.items()
        if v != "median_abs_error"
    ]
    data = pl.DataFrame(rows)
    order = list(report["S"])
    colour = alt.Color(
        "variant:N",
        sort=list(LABELS.values()),
        scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
        legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=500, symbolType="stroke"),
    )
    y = alt.Y("group:N", sort=order, title=None)
    offset = alt.YOffset("variant:N", sort=list(LABELS.values()))
    base = _chart(data)
    rule = (
        _chart(pl.DataFrame({"x": [0.5]}))
        .mark_rule(strokeDash=[4, 3], color="#444")
        .encode(x="x:Q")
    )
    chart = alt.layer(
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X(
                "lo:Q",
                title="Share of BG2's advantage over arm D that the ban removes (0 = none, 1 = all)",
                scale=alt.Scale(domain=[-0.05, 1.0]),
            ),
            x2="hi:Q",
            y=y,
            yOffset=offset,
            color=colour,
        ),
        base.mark_point(filled=True, size=60).encode(x="est:Q", y=y, yOffset=offset, color=colour),
        rule,
    ).properties(
        width=WIDTH - 130,
        height=360,
        title=_title(
            title="Banning interactions between sunshine and temperature or recent power removes a small share of BG2's advantage",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole weeks. Dashed line: the 0.5 the plan required. D5 and D9 are the targets, D10 and D16 are placebos. Rows of the first five groups: the quarter of April-to-June daylight half-hours where arm D trails BG2 most. Last group: every daylight half-hour. Larger is a bigger share removed.",
        ),
    )
    _save(chart.configure_view(stroke=None), name="figure_1_share_of_advantage")


def figure_score(*, report: dict) -> None:
    """Draw the quantile score of each arm on the selected rows at D5 and D9."""
    s = report["S"]["D5+D9"]
    names = {
        "Arm D (convex stack)": s["no_solar_x_temp"]["qs_d"],
        "BG2 as stored (blend of five models, 51 members)": s["no_solar_x_temp"]["qs_bg2_stored"],
        "BG2 twin refit (control member)": s["no_solar_x_temp"]["qs_full"],
        "Twin, no solar-by-temperature or solar-by-lags": s["no_solar_x_temp_or_lags"][
            "qs_constrained"
        ],
        "Twin, no interaction of any kind (post hoc)": s["additive_post_hoc"]["qs_constrained"],
    }
    data = pl.DataFrame({"arm": list(names), "qs": list(names.values())})
    chart = (
        _chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("arm:N", sort=list(names), title=None, axis=alt.Axis(labelLimit=330)),
            x=alt.X(
                "qs:Q",
                title="Quantile score on the selected half-hours (points of effective capacity; smaller is better)",
            ),
        )
        .properties(
            width=WIDTH - 340,
            height=170,
            title=_title(
                title="A twin with no interactions at all still scores about 4.1 where arm D scores 8.7",
                subtitle="Quantile score (twice the mean pinball loss over 11 quantiles). D5 and D9 together, April to June 2026, the quarter of daylight half-hours where arm D trails stored BG2 most.",
            ),
        )
    )
    _save(chart.configure_view(stroke=None), name="figure_2_quantile_score_by_arm")


def figure_attribution(*, report: dict) -> None:
    """Draw mean absolute Shapley value and permutation loss by input family."""
    a = report["attribution"]
    rows = [
        {
            "family": f,
            "measure": "Mean absolute Shapley value of the median forecast (points)",
            "value": v,
        }
        for f, v in a["mean_abs_shap_median_output"].items()
    ] + [
        {
            "family": f,
            "measure": "Quantile score added by shuffling the family (points per half-hour)",
            "value": v,
        }
        for f, v in a["permutation_qs_increase_per_row"].items()
    ]
    data = pl.DataFrame(rows)
    order = list(a["mean_abs_shap_median_output"])
    chart = (
        _chart(data)
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(
            y=alt.Y("family:N", sort=order, title=None),
            x=alt.X("value:Q", title=None),
            row=alt.Row(
                "measure:N",
                title=None,
                header=alt.Header(
                    labelOrient="top",
                    labelAngle=0,
                    labelAlign="left",
                    labelFontSize=11,
                    labelPadding=6,
                ),
            ),
        )
        .properties(
            width=WIDTH - 90,
            height=100,
            title=_title(
                title="Sunshine inputs carry most of the twin's forecast; recent power and the calendar come next",
                subtitle="Twin refit, D5 and D9, April to June 2026, the selected half-hours. Larger bars mean the twin relies on the family more. Solar: shortwave radiation, solar zenith, solar azimuth.",
            ),
        )
    )
    _save(chart.configure_view(stroke=None), name="figure_3_input_families")


def main() -> None:
    """Draw the three figures."""
    report = json.loads((fit.DATA_DIR / "work" / "report.json").read_text())
    figure_share(report=report)
    figure_score(report=report)
    figure_attribution(report=report)


if __name__ == "__main__":
    main()
