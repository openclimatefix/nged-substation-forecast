# ruff: noqa: E501
"""Draw the figures of experiment 195's write-up from the stored per-event tables.

uv run --no-sync python ml-experiments/00195-*/e00195_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00195_report import ARM_ORDER, BASELINE, bootstrap_difference, bootstrap_recall, load

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00195-switching-neural-plus-detector-features"
)
WIDTH: Final[int] = 560
GROUP: Final[dict[str, str]] = {
    "RO-NC": "Existing rule",
    "Pair alarm": "Existing rule",
    "U RO-NC + pair (rule)": "Existing rule",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
DOMAIN: Final[list[str]] = ["Existing rule", "Learned score", "Learned score plus a rule"]


def _group(arm: str) -> str:
    if arm in GROUP:
        return GROUP[arm]
    return "Learned score plus a rule" if arm.startswith(("U ", "I ")) else "Learned score"


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _dots(
    *, frame: pl.DataFrame, title: list[str], axis_title: str, rule: float | None, rule_label: str
) -> Any:
    base: Any = alt.Chart(frame)
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=DOMAIN, range=COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("arm:N", sort=list(ARM_ORDER), title=None)
    layers = [
        base.mark_rule(strokeWidth=2).encode(
            y=y, x=alt.X("lo:Q", title=axis_title), x2="hi:Q", color=colour
        ),
        base.mark_point(filled=True, size=70, opacity=1).encode(y=y, x="point:Q", color=colour),
    ]
    if rule is not None:
        line: Any = alt.Chart(pl.DataFrame({"x": [rule], "label": [rule_label]}))
        layers.append(line.mark_rule(strokeDash=[4, 3], color="#555555").encode(x="x:Q"))
        layers.append(
            line.mark_text(
                align="left", dx=4, dy=-4, baseline="bottom", color="#555555", fontSize=10
            ).encode(x="x:Q", text="label:N", y=alt.value(0))
        )
    return alt.layer(*layers).properties(
        width=WIDTH, height=300, title=alt.TitleParams(title, anchor="start", fontSize=12)
    )


def main() -> None:
    """Draw the three figures."""
    real, injected, _ = load()
    long = real.filter(pl.col("hours") > 24)
    rows, diffs = [], []
    for arm in ARM_ORDER:
        p, lo, hi = bootstrap_recall(table=long, arm=arm, weighted=True)
        rows.append({"arm": arm, "group": _group(arm), "point": p, "lo": lo, "hi": hi})
        if arm != BASELINE:
            d, dlo, dhi = bootstrap_difference(table=long, arm=arm, base=BASELINE, weighted=True)
            diffs.append({"arm": arm, "group": _group(arm), "point": d, "lo": dlo, "hi": dhi})
    ronc = rows[0]["point"]
    _save(
        _dots(
            frame=pl.DataFrame(rows),
            title=[
                "Share of moved MW found, 35 real transfers over 1 day, at 80 unexplained odd-month alarms.",
                "Dot: estimate. Line: 95% interval from resampling whole Primary pairs. Higher is better.",
            ],
            axis_title="MW-weighted recall of the 35 transfers (higher is better)",
            rule=ronc,
            rule_label="RO-NC",
        ),
        name="figure_1_mw_weighted_recall",
    )
    _save(
        _dots(
            frame=pl.DataFrame(diffs),
            title=[
                "Each arm minus RO-NC, in MW-weighted recall of the 35 real transfers.",
                "Dot: estimate. Line: 95% interval from resampling whole Primary pairs. Right of zero is better than RO-NC.",
            ],
            axis_title="MW-weighted recall minus RO-NC's (higher is better)",
            rule=0.0,
            rule_label="same as RO-NC",
        ),
        name="figure_2_difference_from_ronc",
    )
    inj = (
        injected.filter(pl.col("kind") == "eval_held")
        .with_columns(
            receiver=pl.when(pl.col("n_receivers") == 0)
            .then(pl.lit("No metered receiver"))
            .otherwise(pl.lit("Metered receiver"))
        )
        .group_by("arm", "receiver")
        .agg(recall=pl.col("found").mean())
        .with_columns(group=pl.col("arm").map_elements(_group, return_dtype=pl.String))
    )
    chart: Any = (
        alt.Chart(inj)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=list(ARM_ORDER), title=None),
            x=alt.X(
                "recall:Q",
                title="Share of injected transfers found, start and end (higher is better)",
                axis=alt.Axis(values=[0.0, 0.1, 0.2, 0.3]),
            ),
            yOffset="receiver:N",
            color=alt.Color(
                "receiver:N",
                scale=alt.Scale(
                    domain=["Metered receiver", "No metered receiver"],
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY_LIGHT],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=420,
            title=alt.TitleParams(
                [
                    "Injected transfers in odd months, on Primary pairs the injector never saw (998 events).",
                    "Bars: share found at the same alarm thresholds as the real record. Higher is better.",
                ],
                anchor="start",
                fontSize=12,
            ),
        )
    )
    _save(chart, name="figure_3_injected_recall")


if __name__ == "__main__":
    main()
