# ruff: noqa: E501,D103,FURB171
"""Draw experiment 221's figures from the stored results.

uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00221_common as common
import plotting.ocf_theme as ocf
import polars as pl
from e00221_score import BANDS, PRIMARY_BAND, losses

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
LABELS: Final[dict[str, str]] = {
    "A1": "A1: quantiles + lead",
    "A2": "A2: A1 + ensemble wind",
    "A3": "A3: blind (lead, calendar)",
    "A4": "A4: ensemble-mean base",
    "A5": "A5: table per lead band",
    "A1_lofo": "A1, farm held out",
    "A2_lofo": "A2, farm held out",
    "A3_lofo": "A3, farm held out",
}
KINDS: Final[dict[str, str]] = {
    "A1": "pooled head",
    "A2": "pooled head",
    "A3": "pooled head",
    "A4": "pooled head",
    "A5": "table",
    "A1_lofo": "head, farm held out",
    "A2_lofo": "head, farm held out",
    "A3_lofo": "head, farm held out",
}
COLOURS: Final[dict[str, str]] = {
    "pooled head": ocf.BRAND_ORANGE,
    "table": ocf.DATA_BLUE,
    "head, farm held out": ocf.DATA_PURPLE,
}


def interval_chart(
    *, data: pl.DataFrame, order: list[str], title: str | list[str], x_title: str
) -> alt.Chart:
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "kind:N",
            legend=alt.Legend(title=None, orient="top"),
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
        ),
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", title=x_title), x2="upper:Q"
    )
    points = base.mark_point(filled=True, size=90).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + bars + points).properties(
        width=WIDTH,
        height=30 * len(order) + 20,
        title=alt.TitleParams(title, anchor="start", fontSize=13),
    )


def main() -> None:
    results = json.loads((common.WORK_DIR / "results" / "results.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    band = BANDS[PRIMARY_BAND]
    rows = [
        {
            "label": LABELS[c["treatment"]],
            "kind": KINDS[c["treatment"]],
            "difference": c["difference"],
            "lower": c["lower"],
            "upper": c["upper"],
        }
        for c in [*results["contrasts"], results["C1"]]
        if c["reference"] == "A0"
        and c["metric"] == "qs"
        and c["site"] is None
        and c["band"] == band
        and c["treatment"] in LABELS
        and c.get("label", "C1") not in ("A4_mean_only_vs_A0",)
    ]
    interval_chart(
        data=pl.DataFrame(rows),
        order=list(LABELS.values()),
        title=[
            "No head resolves a gain over the stored best guess",
            "at lead days 1 and 2 (9 scored months, 3 farms)",
        ],
        x_title="Quantile score minus the best guess's, points (negative is better); 95%",
    ).save(DOCS_DIR / "figure_1_heads_minus_best_guess.svg")

    coverage = pl.DataFrame(
        [
            {
                "label": "A0 best guess" if r["arm"] == "A0" else LABELS[r["arm"]],
                "covered": r["covered"],
            }
            for r in results["absolute"]
            if r["scope"] == band and r["arm"] in ("A0", *LABELS)
        ]
    )
    base = alt.Chart(coverage).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300))
    )
    bars = base.mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "covered:Q",
            scale=alt.Scale(domain=[0.6, 1.0]),
            title="Share of cases inside the 5% to 95% band",
        )
    )
    target = (
        alt.Chart(pl.DataFrame({"x": [0.90]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    (bars + target).properties(
        width=WIDTH,
        height=25 * coverage.height + 20,
        title=alt.TitleParams(
            [
                "Every head covers less than the 0.90 target (dashed)",
                "on unseen months, lead days 1 and 2",
            ],
            anchor="start",
            fontSize=13,
        ),
    ).save(DOCS_DIR / "figure_2_coverage.svg")

    insample = pl.concat(
        [
            pl.read_parquet(p)
            for p in sorted((common.WORK_DIR / "posthoc").glob("insample_*.parquet"))
        ]
    )
    gaps = {}
    for arm in ("A0", "A1", "A2"):
        part = insample.filter(pl.col("arm") == arm).drop("arm", "month")
        scored = losses(frame=part.sort(*common.KEY_COLUMNS, "seed"))
        gaps[arm] = float(scored.filter(pl.col("band") == PRIMARY_BAND)["qs"].to_numpy().mean())
    out_of_sample = {
        "A1": next(
            c["difference"]
            for c in results["contrasts"]
            if c["label"] == "C4" and c["band"] == band
        ),
        "A2": results["C1"]["difference"],
    }
    fit = pl.DataFrame(
        [
            {"arm": name, "where": place, "difference": value}
            for name in ("A1", "A2")
            for place, value in (
                ("on the rows it trained on", gaps[name] - gaps["A0"]),
                ("on the next month", out_of_sample[name]),
            )
        ]
    )
    chart = (
        alt.Chart(fit)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", title=None, axis=alt.Axis(labelExpr="datum.label + ' head'")),
            x=alt.X(
                "difference:Q",
                title="Quantile score minus the best guess's, points (negative is better)",
            ),
            yOffset="where:N",
            color=alt.Color(
                "where:N",
                legend=alt.Legend(title=None, orient="top"),
                scale=alt.Scale(
                    domain=["on the rows it trained on", "on the next month"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=120,
            title=alt.TitleParams(
                [
                    "The heads gain on the rows they train on and do not on the next",
                    "month (lead days 1 and 2)",
                ],
                anchor="start",
                fontSize=13,
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_3_in_sample_against_next_month.svg")

    farms = pl.DataFrame(
        [
            {
                "label": f"{LABELS[c['treatment']].split(':')[0]}, {c['site']}",
                "kind": "pooled head" if c["treatment"] != "A2_lofo" else "head, farm held out",
                "difference": c["difference"],
                "lower": c["lower"],
                "upper": c["upper"],
            }
            for c in results["contrasts"]
            if c["label"].endswith("_vs_A0_by_site")
            and c["band"] == band
            and c["treatment"] in ("A1", "A2", "A2_lofo")
        ]
    )
    interval_chart(
        data=farms,
        order=farms["label"].to_list(),
        title=["The heads help W2 and hurt W3 at lead days 1 and 2"],
        x_title="Quantile score minus the best guess's, points (negative is better); 95%",
    ).save(DOCS_DIR / "figure_4_by_farm.svg")
    print("wrote 4 figures")


if __name__ == "__main__":
    main()
