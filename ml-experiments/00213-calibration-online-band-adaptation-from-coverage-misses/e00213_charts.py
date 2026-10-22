"""Draw the figures of experiment 213's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00213-*/e00213_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00213_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT: Final[Path] = common.REPORT_DIR / "report.json"
WIDTH: Final[int] = 680
GREY: Final[str] = "#808080"
ARM_COLOURS: Final[dict[str, str]] = {
    "A: experiment 169 arm A": GREY,
    "CW: constant widening": ocf.DATA_GREEN,
    "ACI, gamma 0.01 (chosen)": ocf.DATA_BLUE,
    "ACI, gamma 0.03": ocf.DATA_SKY,
    "ACI, gamma 0.1": ocf.BRAND_ORANGE,
}
ARM_KEYS: Final[dict[str, str]] = {
    "a": "A: experiment 169 arm A",
    "cw": "CW: constant widening",
    "aci_g0.01": "ACI, gamma 0.01 (chosen)",
    "aci_g0.03": "ACI, gamma 0.03",
    "aci_g0.1": "ACI, gamma 0.1",
}
COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"
PERIOD_LABELS: Final[dict[str, str]] = {
    "tuning": "tuning months",
    "replicate": "replicate months",
    "scored": "scored year",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def figure_coverage(*, data: dict) -> alt.TopLevelMixin:
    """Cold-night q95 coverage by arm and period, with the pass line and the nominal level."""
    rows = []
    for period, label in PERIOD_LABELS.items():
        if period == "tuning":
            for entry in data["tuning_sweep"].values():
                for arm, values in entry["coverage"].items():
                    rows.append(
                        {"arm": ARM_KEYS[arm], "period": label, "coverage": values["cold_q95"]}
                    )
            continue
        pieces = {
            **data[period]["gamma_sweep"]["0.01"]["coverage"],
            **data[period]["gamma_sweep"]["0.03"]["coverage"],
            **data[period]["gamma_sweep"]["0.1"]["coverage"],
            **data[period]["cw_decision"]["coverage"],
        }
        for arm, values in pieces.items():
            rows.append({"arm": ARM_KEYS[arm], "period": label, "coverage": values["cold_q95"]})
    frame = pl.DataFrame(rows).unique(subset=["arm", "period"])
    points = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=120)
        .encode(
            y=alt.Y("arm:N", sort=list(ARM_COLOURS), title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.84, 0.96]),
                title="share of cold-night readings at or under the q95",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
    )
    labels = points.mark_text(dy=-14, fontSize=11).encode(text=alt.Text("coverage:Q", format=".3f"))
    pass_line = alt.Chart().mark_rule(strokeDash=[4, 3], color="#444444").encode(x=alt.datum(0.90))  # ty: ignore[unresolved-attribute]
    nominal = alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0.95))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(points, labels, pass_line, nominal)
        .properties(width=WIDTH / 3 - 110, height=170)
        .facet(
            column=alt.Column("period:N", sort=list(PERIOD_LABELS.values()), title=None),
            data=frame,
        )
        .properties(
            title=_title(
                text="Online adaptation lifts cold-night q95 coverage in the scored year only, "
                "and not above a constant widening",
                subtitle=(
                    "Pooled over D5, D10, D11, D14, and D16. "
                    "Dashed line: pass line 0.90. Solid line: nominal 0.95."
                ),
            )
        )
    )


def figure_spells(*, data: dict) -> alt.TopLevelMixin:
    """Cold-night q95 coverage by place in the cold spell, scored year."""
    chosen = f"aci_g{data['gamma']:g}"
    order = ["first night", "days 1-3 after", "later nights"]
    rows = [
        {
            "arm": ARM_KEYS[row["arm"]],
            "stage": row["spell_label"],
            "coverage": row["cov_q95"],
            "rows": row["rows"],
        }
        for row in data["scored"]["tables"]["spells"]
        if row["arm"] in ("a", "cw", chosen)
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("stage:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_COLOURS)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "coverage:Q",
            scale=alt.Scale(domain=[0.75, 0.96]),
            title="share of cold-night readings at or under the q95",
        )
    )
    text = base.mark_text(align="left", dx=3, fontSize=11).encode(  # ty: ignore[unresolved-attribute]
        x="coverage:Q", text=alt.Text("coverage:Q", format=".3f")
    )
    return alt.layer(bars, text).properties(
        width=WIDTH - 150,
        height=170,
        title=_title(
            text="Online adaptation helps only on the later nights of a cold spell",
            subtitle="Scored year, five decision substations; the stages hold 2, 5, and 8 dates.",
        ),
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads(REPORT.read_text())
    _save(figure_coverage(data=data), name="figure_1_cold_night_coverage")
    _save(figure_spells(data=data), name="figure_2_cold_spell_stage")


if __name__ == "__main__":
    main()
