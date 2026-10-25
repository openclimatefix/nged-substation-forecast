# ruff: noqa: E501
"""Draw experiment 00348's figures as SVG from `score.json` and `rows.parquet`.

Optimise afterwards with `npx svgo@4 --multipass --precision=1 --final-newline f.svg`. Farms are
W1 to W3 and no chart shows a calendar date.

    PYTHONPATH=ml-experiments/00348-wind-plant-mdn-upper-tail uv run --no-sync python \
        ml-experiments/00348-wind-plant-mdn-upper-tail/e00348_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00348_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

OUT: Final[Path] = Path("docs/ml-experiments/00348-wind-plant-mdn-upper-tail")
WIDTH: Final[int] = 560
WRAP: Final[int] = 100
SCOPE: Final[str] = (
    "Three metered wind farms W1 to W3; 53 Tuesday runs, kept half-hours, true days 1 to 10 (lead 24 to 240 hours)."
)
LABELS: Final[dict[str, str]] = {
    "y2": "y2 (experiment 347)",
    "z1": "z1 weighted pinball loss",
    "z2": "z2 point mass + cap",
    "z3": "z3 monotone tail increments",
    "z4": "z4 = z1 + z2",
    "z5": "z5 blend (band + XGBoost)",
    "default": "default",
}
GROUPS: Final[dict[str, str]] = {
    "y2": "reference head",
    "default": "default",
    **dict.fromkeys(("z1", "z2", "z3", "z4"), "plant + head, tail change"),
    "z5": "plant band + XGBoost blend",
}
COLOURS: Final[dict[str, str]] = {
    "reference head": ocf.DATA_PURPLE,
    "default": ocf.DATA_BLUE,
    "plant + head, tail change": ocf.BRAND_ORANGE,
    "plant band + XGBoost blend": ocf.DATA_GREEN,
}


def title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a wrapped title."""
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def colour() -> alt.Color:
    """Return the group colour encoding."""
    return alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=400),
    )


def save(*, chart: alt.Chart | alt.VConcatChart | alt.LayerChart, name: str) -> None:
    """Write a chart as SVG."""
    chart.save(OUT / f"{name}.svg")


def figure_1(*, score: dict) -> None:
    """Headline: each arm minus the default at true days 1 to 10, with intervals."""
    rows = []
    for arm in ("y2", "z1", "z2", "z3", "z4", "z5"):
        c = score["contrasts"][f"{arm} - default | true_1_10"]
        level = next(k for k in c if k.startswith("0.99"))
        rows.append(
            {
                "arm": LABELS[arm],
                "group": GROUPS[arm],
                "mean": c["mean"],
                "lo95": c["0.9500"][0],
                "hi95": c["0.9500"][1],
                "lo": c[level][0],
                "hi": c[level][1],
            }
        )
    frame = pl.DataFrame(rows)
    order = [LABELS[a] for a in ("y2", "z1", "z2", "z3", "z4", "z5")]
    y = alt.Y("arm:N", sort=order, title=None)
    wide = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=1.5, opacity=0.5)
        .encode(
            y=y,
            x=alt.X(
                "lo:Q",
                title="Quantile score minus the default's (points of capacity; more negative is better)",
            ),
            x2="hi:Q",
            color=colour(),
        )
    )
    narrow = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=4)
        .encode(y=y, x="lo95:Q", x2="hi95:Q", color=colour())
    )
    dots = alt.Chart(frame).mark_point(filled=True, size=70).encode(y=y, x="mean:Q", color=colour())  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    chart = alt.layer(wide, narrow, dots, zero).properties(
        width=WIDTH,
        height=190,
        title=title(
            text="Every arm against Wind farms research default 1",
            lines=[
                SCOPE,
                "Dot: estimate. Thick line: 95% interval; thin line: 99.8% interval (Bonferroni across five arms), both from resampling whole calendar months. Dashed line: same as the default.",
            ],
        ),
    )
    save(chart=chart, name="figure_1_headline")  # ty: ignore[invalid-argument-type]


def figure_2(*, score: dict) -> None:
    """Tail: share above q99 and 0.99 pinball loss per arm."""
    rows = []
    for name in ("default", "y2", "z1", "z2", "z3", "z4", "z5"):
        tail = score["tails"]["true_1_10"][name]
        rows.append({"arm": LABELS[name], "group": GROUPS[name], "above": tail[3], "p99": tail[7]})
    frame = pl.DataFrame(rows)
    order = [LABELS[a] for a in ("default", "y2", "z1", "z2", "z3", "z4", "z5")]
    top = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X("above:Q", title="Share of observations above q99 (percent; the target is 1)"),
            color=colour(),
        )
        .properties(width=WIDTH, height=150)
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    bottom = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=order, title=None),
            x=alt.X("p99:Q", title="0.99 pinball loss (points of capacity; smaller is better)"),
            color=colour(legend=None)
            if False
            else alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
                legend=None,
            ),
        )
        .properties(width=WIDTH, height=150)
    )
    chart = alt.vconcat(top + rule, bottom).properties(
        title=title(
            text="Upper-tail calibration of every arm",
            lines=[SCOPE, "Dashed line: the nominal 1% of observations above the 0.99 quantile."],
        )
    )
    save(chart=chart, name="figure_2_upper_tail")


def figure_3(*, rows: pl.DataFrame, arm: str) -> None:
    """Forecast q95 and q99 against the observed upper quantiles, per farm and forecast-median bin."""
    kept = rows.filter(~pl.col("masked") & pl.col("lead_hours").is_between(24, 240))
    out = []
    for farm in common.FARMS:
        sub = kept.filter(pl.col("site") == farm)
        x = sub[f"q50__{arm}"].to_numpy()
        edges = np.quantile(x, np.linspace(0, 1, 9))
        which = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 7)
        for b in range(8):
            sel = which == b
            if sel.sum() < 30:
                continue
            obs = sub["actual"].to_numpy()[sel]
            for level, tag in ((0.95, "q95"), (0.99, "q99")):
                out.append(
                    {
                        "farm": farm,
                        "x": 100 * float(x[sel].mean()),
                        "kind": f"{arm} forecast {tag}",
                        "value": 100 * float(sub[f"{tag}__{arm}"].to_numpy()[sel].mean()),
                    }
                )
                out.append(
                    {
                        "farm": farm,
                        "x": 100 * float(x[sel].mean()),
                        "kind": f"observed {tag}",
                        "value": 100 * float(np.quantile(obs, level)),
                    }
                )
    frame = pl.DataFrame(out)
    domain = [f"{arm} forecast q95", "observed q95", f"{arm} forecast q99", "observed q99"]
    pal = [ocf.DATA_BLUE, "#306BFF55", ocf.BRAND_ORANGE, "#FF490155"]
    base = alt.Chart(frame).encode(
        x=alt.X("x:Q", title="Forecast median (% of capacity)"),
        y=alt.Y("value:Q", title="Upper quantile (% of capacity)"),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=domain, range=pal),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    lines = base.transform_filter(alt.datum.kind == domain[0]).mark_line(  # ty: ignore[unsupported-operator]
        point=True
    ) + base.transform_filter(alt.datum.kind == domain[2]).mark_line(point=True)
    dots = base.transform_filter(alt.datum.kind == domain[1]).mark_point(  # ty: ignore[unsupported-operator]
        shape="cross", size=70
    ) + base.transform_filter(alt.datum.kind == domain[3]).mark_point(shape="cross", size=70)
    chart = (
        alt.layer(lines, dots)
        .properties(width=WIDTH, height=130)
        .facet(row=alt.Row("farm:N", title=None))
        .properties(
            title=title(
                text=f"Arm {arm}: forecast upper quantiles against the observed ones",
                lines=[
                    SCOPE,
                    "Lines: mean forecast q95 and q99 in each of 8 bins of the forecast median. Crosses: the observed 95th and 99th percentile of power in the same bin. Lines on the crosses mean calibrated.",
                ],
            )
        )
    )
    save(chart=chart, name=f"figure_3_power_curve_{arm}")  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Draw every figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    score = json.loads((common.DATA_DIR / "score.json").read_text())
    rows = pl.read_parquet(common.DATA_DIR / "rows.parquet")
    figure_1(score=score)
    figure_2(score=score)
    best = score["standard"]["best"]
    figure_3(rows=rows, arm=best if best in ("y2", "z1", "z2", "z3", "z4") else "z4")
    figure_3(rows=rows, arm="z5")
    figure_3(rows=rows, arm="default")


if __name__ == "__main__":
    main()
