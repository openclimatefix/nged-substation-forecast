# ruff: noqa: B007,D103
"""Draw experiment 200's figures from the stored gate and stage 2 files.

uv run --no-sync python ml-experiments/00200-*/e00200_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00200_data as data_module
import e00200_score as score_module
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{data_module.EXPERIMENT_ID:05d}-{data_module.SLUG}"
)
WIDTH: Final[int] = 560
DECISION: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16"]


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def contrasts() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    head = pl.read_parquet(data_module.WORK_DIR / "stage2_head_rows.parquet")
    blind = pl.read_parquet(data_module.WORK_DIR / "stage2_blind_rows.parquet")
    keep = pl.col("site").is_in(DECISION)
    h, b, d = (
        x.filter(keep & (pl.col("arm") == arm))
        for x, arm in ((head, "head"), (blind, "head"), (head, "arm_d"))
    )
    months = d["month"].to_numpy()
    rows = []
    for name, left, right in (
        ("Head minus arm D", h, d),
        ("Blind head minus arm D", b, d),
        ("Head minus blind head", h, b),
    ):
        diff = left["loss"].to_numpy() - right["loss"].to_numpy()
        m, lo, hi = score_module.interval(diff=diff, months=months, level=0.99, rng=rng)
        rows.append({"contrast": name, "difference": m, "lower": lo, "upper": hi})
    return pl.DataFrame(rows)


def figure_contrasts() -> alt.TopLevelMixin:
    data = contrasts()
    base = alt.Chart(data).encode(
        y=alt.Y("contrast:N", sort=None, title=None),
        color=alt.Color(
            "contrast:N",
            legend=None,
            scale=alt.Scale(
                domain=["Head minus arm D", "Blind head minus arm D", "Head minus blind head"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
            ),
        ),
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q", title="Quantile score difference, points (negative is better); 99% interval"
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=90).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    threshold = (
        alt.Chart(pl.DataFrame({"x": [-0.02]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[1, 3])
        .encode(x="x:Q")
    )
    return (zero + threshold + bars + points).properties(
        width=WIDTH,
        height=130,
        title="The head beats arm D, but most of the gain comes without reading any error",
    )


def figure_gate() -> alt.TopLevelMixin:
    gate = json.loads((data_module.WORK_DIR / "gate.json").read_text())
    rows = []
    for kind, label in (("step", "Level step"), ("saturday", "Saturday-evening shift")):
        for seed, value in enumerate(gate[kind]["head_recovery_by_seed"]):
            rows.append({"test": label, "who": "Head, one seed", "recovery": value})
        rows.append({"test": label, "who": "Recipe R", "recovery": gate[kind]["recipe_r_recovery"]})
    data = pl.DataFrame(rows)
    gates = pl.DataFrame({"test": ["Level step", "Saturday-evening shift"], "gate": [0.30, 0.15]})
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X(
                "recovery:Q",
                title="Share of the injected 3 points recovered",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("test:N", title=None),
            color=alt.Color(
                "who:N",
                title=None,
                scale=alt.Scale(
                    domain=["Head, one seed", "Recipe R"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
            ),
        )
    )
    ticks = (
        alt.Chart(gates)  # ty: ignore[unresolved-attribute]
        .mark_tick(color="#808080", thickness=2, size=40)
        .encode(x="gate:Q", y="test:N")
    )
    return (ticks + points).properties(
        width=WIDTH,
        height=110,
        title="Known-answer gate: the head recovers about a third of the step, recipe R two thirds",
    )


def figure_sites() -> alt.TopLevelMixin:
    head = pl.read_parquet(data_module.WORK_DIR / "stage2_head_rows.parquet")
    blind = pl.read_parquet(data_module.WORK_DIR / "stage2_blind_rows.parquet")
    d = head.filter(pl.col("arm") == "arm_d")
    rows = []
    for name, frame in (("Head", head), ("Blind head", blind)):
        h = frame.filter(pl.col("arm") == "head")
        for site in sorted(set(d["site"])):
            a = d.filter(pl.col("site") == site)["loss"].mean()
            b = h.filter(pl.col("site") == site)["loss"].mean()
            rows.append({"site": site, "arm": name, "difference": b - a})  # ty: ignore[unsupported-operator]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X(
                "difference:Q", title="Quantile score minus arm D's, points (negative is better)"
            ),
            y=alt.Y("site:N", title=None),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(
                    domain=["Head", "Blind head"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title="Four of six substations gain in both arms; D11 and D9 get worse with the head",
        )
    )


def main() -> None:
    alt.theme.enable("ocf") if hasattr(alt, "theme") else None
    _save(figure_contrasts(), name="figure_1_contrasts")
    _save(figure_gate(), name="figure_2_gate")
    _save(figure_sites(), name="figure_3_sites")


if __name__ == "__main__":
    main()
