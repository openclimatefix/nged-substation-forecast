# ruff: noqa: E501
# ruff: noqa: D103
"""Draw experiment 244's figures from `work/report.json` and `work/gate_scale.json`.

uv run --no-sync python ml-experiments/00244-*/e00244_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00244_data as data_module
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{data_module.EXPERIMENT_ID:05d}-{data_module.SLUG}"
)
WIDTH: Final[int] = 640


def save(chart: Any, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / name)
    print("wrote", name)


def decile_figure(report: dict) -> None:
    panels = []
    for scope, label in (("cold_nights", "Cold nights"), ("all_rows", "All rows")):
        data = pl.DataFrame(report["deciles"][scope])
        y = alt.Y(
            "E0_cov_q95:Q",
            title="Share at or under q95",
            scale=alt.Scale(domain=[0.7, 1.0]),
        )
        line = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, color=ocf.BRAND_ORANGE)
            .encode(
                x=alt.X(
                    "decile:O", title="Decile of the spread of the 20 corrections (1 = smallest)"
                ),
                y=y,
            )
        )
        nominal = (
            alt.Chart(pl.DataFrame({"y": [0.95]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0.7, 1.0])))
        )
        panels.append((nominal + line).properties(width=WIDTH, height=150, title=label))
    chart = alt.vconcat(*panels).properties(
        title=[
            "E0's q95 coverage shows no trend across deciles of the spread",
            "Scored year, five decision substations, leads 15.5 to 39 h. Dashed line: nominal 0.95.",
        ]
    )
    save(chart, "figure_1_coverage_by_spread_decile.svg")


def gate_figure(gate: dict) -> None:
    data = pl.DataFrame(
        [
            {"shift": f"x{m}", "multiplier": m, "spread": gate[f"x{m}"]["mean_spread"]}
            for m in (0, 1, 5, 25, 100)
        ]
    )
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "shift:N",
                sort=[f"x{m}" for m in (0, 1, 5, 25, 100)],
                title="Shift of the forecast-level input, as a multiple of the registered 4 C shift (x1)",
            ),
            y=alt.Y("spread:Q", scale=alt.Scale(type="symlog"), title="Mean spread (points)"),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=[
                "The 20 networks agree on a realistic shifted input and disagree on an extreme one",
                "Cold-night rows of the scored year, network without error inputs, 894 rows.",
                "Spread: standard deviation of the 20 median corrections. x0 is the unshifted input.",
            ],
        )
    )
    save(chart, "figure_2_gate_shift_size.svg")


def seed_figure(report: dict) -> None:
    rows = []
    for kind, label in (("ref", "Without error inputs"), ("err", "With error inputs")):
        rows += [{"network": label, "gain": g} for g in report["prediction_3"][kind]["gains"]]
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, opacity=0.8)
        .encode(
            x=alt.X(
                "gain:Q", title="Gain over arm D, points of effective capacity (larger is better)"
            ),
            y=alt.Y("network:N", title=None),
            color=alt.Color(
                "network:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Without error inputs", "With error inputs"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=130,
            title=[
                "Twenty seeds of one network gain between 0.08 and 0.13 points over arm D",
                "Scored year, five decision substations. One dot is one seed.",
            ],
        )
    )
    save(chart, "figure_3_seed_gains.svg")


def lift_figure(report: dict) -> None:
    labels = {
        "scored/E3_ref_seed-E0": "E3: real spread",
        "scored/P_within-E0": "P: shuffled within month and hour band",
        "scored/P_across-E0": "P: shuffled across substations",
    }
    rows = []
    for key, label in labels.items():
        m, lo, hi = report["contrasts"][key]["cold_q95_coverage"]
        rows.append({"arm": label, "lift": m, "lower": lo, "upper": hi})
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=300))
    )
    bars = base.mark_rule(strokeWidth=3, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Cold-night q95 coverage minus E0's (larger is closer to 0.95); 95% interval",
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE).encode(x="lift:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (zero + bars + points).properties(
        width=WIDTH,
        height=150,
        title=[
            "Scaling the bands by the spread lifts cold-night coverage no more than a shuffled spread",
            "Scored year, 782 cold-night rows on 10 days. Interval: resampling whole days.",
        ],
    )
    save(chart, "figure_4_coverage_lift_and_placebo.svg")


def main() -> None:
    report = json.loads((data_module.WORK_DIR / "report.json").read_text())
    gate = json.loads((data_module.WORK_DIR / "gate_scale.json").read_text())
    decile_figure(report)
    gate_figure(gate)
    seed_figure(report)
    lift_figure(report)


if __name__ == "__main__":
    main()
