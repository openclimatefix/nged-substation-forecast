"""Draw experiment 00093's figures from `results/report.json`, as SVG.

1. The age penalty of the old training year, raw and level-normalised, with intervals (headline).
2. Each arm's absolute quantile score with its 95% interval.
3. The age penalty by series, raw and level-normalised.
4. Tail calibration: how often the actual value fell below the 0.01, 0.05, 0.95, and 0.99 quantiles.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00093-*/e00093_charts.py
"""

import json
from typing import Final

import altair as alt
import e00093_common as common
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 540
SCOPE: Final[str] = (
    "13 Primary substations, 365 daily forecast runs, July 2025 to June 2026. "
    "Frozen models, each trained on one year."
)
ARM_LABELS: Final[dict[str, str]] = {
    "old1_raw": "Trained 2021-07 to 2022-06, raw",
    "old1_all": "Trained 2021-07 to 2022-06, level removed",
    "old1_hh": "Trained 2021-07 to 2022-06, half-hourly level removed (post hoc)",
    "old2_raw": "Trained 2022-07 to 2023-06, raw",
    "old2_all": "Trained 2022-07 to 2023-06, level removed",
    "recent_raw": "Trained 2024-07 to 2025-06, raw",
    "recent_all": "Trained 2024-07 to 2025-06, level removed",
    "recent_hh": "Trained 2024-07 to 2025-06, half-hourly level removed (post hoc)",
}
TAIL_LABELS: Final[dict[str, str]] = {
    "old1_raw": "2021-07 year, raw",
    "old1_all": "2021-07 year, level removed",
    "recent_raw": "2024-07 year, raw",
    "recent_all": "2024-07 year, level removed",
}
TAIL_ARMS: Final[tuple[str, ...]] = ("old1_raw", "old1_all", "recent_raw", "recent_all")
RAW_COLOUR: Final[str] = ocf.BRAND_ORANGE
NORMALISED_COLOUR: Final[str] = ocf.DATA_BLUE


def _report() -> dict:
    """Return the saved report."""
    return json.loads((common.RESULTS_DIR / "report.json").read_text())


def figure_1() -> alt.TopLevelMixin:
    """The age penalty of the 2021-07 year, raw and with the level removed."""
    report = _report()
    planned = report["planned"]
    post = report["post_hoc"]
    rows = [
        ("All 13 series", "Raw", planned["C1"]),
        ("All 13 series", "Level removed", planned["C2"]),
        (
            "Without the series with the largest penalty",
            "Raw",
            post["C1 without the largest-C1 series"],
        ),
        (
            "Without the series with the largest penalty",
            "Level removed",
            post["C2 without the largest-C1 series"],
        ),
        ("Year starting 2022-07 (all 13 series)", "Raw", post["R1 (2022-07 start), raw"]),
        (
            "Year starting 2022-07 (all 13 series)",
            "Level removed",
            post["R2 (2022-07 start), all-hours level"],
        ),
    ]
    data = pl.DataFrame(
        [
            {"group": g, "kind": k, "estimate": r["estimate"], "low": r["lo"], "high": r["hi"]}
            for g, k, r in rows
        ]
    )
    order = list(dict.fromkeys(g for g, _, _ in rows))
    base = alt.Chart(data).encode(
        y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=330)),
        yOffset=alt.YOffset("kind:N", sort=["Raw", "Level removed"]),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["Raw", "Level removed"], range=[RAW_COLOUR, NORMALISED_COLOUR]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title="Old-year score minus recent-year score (points of capacity)")
    )
    bars = base.mark_rule().encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            "Removing the level takes off under half of the old year's penalty",
            subtitle=[
                SCOPE,
                "Dot: estimate. Line: interval from re-drawing whole months (98.3% for the top",
                "two rows, 95% elsewhere). Zero: same score as the model trained on 2024-07 to",
                "2025-06. Above zero: the old year is worse.",
            ],
            anchor="start",
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Absolute quantile score of every arm."""
    absolute = _report()["absolute"]
    data = pl.DataFrame(
        [
            {"label": ARM_LABELS[arm], "qs": v["score11"], "low": v["lo"], "high": v["hi"]}
            for arm, v in absolute.items()
        ]
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=380)
        )
    )
    points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "qs:Q",
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=WIDTH,
        height=210,
        title=alt.TitleParams(
            "Absolute score of each model",
            subtitle=[SCOPE, "Dot: mean. Line: 95% interval from re-drawing whole months."],
            anchor="start",
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """The age penalty by series, raw and level removed."""
    by_series = _report()["by_series"]
    rows = []
    for site, v in by_series.items():
        rows.append({"site": site, "kind": "Raw", "penalty": v["c1"]})
        rows.append({"site": site, "kind": "Level removed", "penalty": v["c2"]})
    data = pl.DataFrame(rows)
    order = sorted(by_series, key=lambda s: -by_series[s]["c1"])
    base = alt.Chart(data).encode(
        y=alt.Y("site:N", sort=order, title="Primary substation (anonymised label)"),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["Raw", "Level removed"], range=[RAW_COLOUR, NORMALISED_COLOUR]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("penalty:Q", title="Old-year score minus recent-year score (points of capacity)")
    )
    return alt.layer(zero, points).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            "One substation carries most of the penalty; elsewhere removing the level barely helps",
            subtitle=[
                SCOPE,
                "Each dot is one substation's mean difference. Above zero: the old year is worse.",
            ],
            anchor="start",
        ),
    )


def figure_4() -> alt.TopLevelMixin:
    """Share of actual values at or below each tail quantile, per arm."""
    tails = _report()["tails"]
    rows = []
    for arm in TAIL_ARMS:
        for column, v in tails[arm].items():
            rows.append(
                {
                    "arm": TAIL_LABELS[arm],
                    "level": column,
                    "coverage": v["coverage"],
                    "target": float(column[1:]) / 100.0,
                }
            )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X(
            "level:N",
            sort=["q01", "q05", "q95", "q99"],
            title="Quantile level",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=[
                    ARM_LABELS[a] for a in ("old1_raw", "old1_all", "recent_raw", "recent_all")
                ],
                range=[RAW_COLOUR, NORMALISED_COLOUR, ocf.DATA_PURPLE, ocf.DATA_GREEN],
            ),
            legend=alt.Legend(orient="top", title=None, columns=1),
        ),
        xOffset=alt.XOffset("arm:N"),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("coverage:Q", title="Share of actual values at or below the quantile")
    )
    target = (
        alt.Chart(
            pl.DataFrame({"level": ["q01", "q05", "q95", "q99"], "t": [0.01, 0.05, 0.95, 0.99]})
        )
        .mark_tick(color="#222222", thickness=2, size=70)
        .encode(x=alt.X("level:N", sort=["q01", "q05", "q95", "q99"]), y="t:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(bars, target).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "Old-year models miss the lower quantiles too often",
            subtitle=[
                SCOPE,
                "Bar: share of actual values at or below the quantile. Tick: the ideal share.",
            ],
            anchor="start",
        ),
    )


def main() -> None:
    """Write the figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
        ("figure_4", figure_4()),
    ):
        chart.save(common.CHART_DIR / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
