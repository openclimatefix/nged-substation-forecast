# ruff: noqa: D103
"""Draw experiment 209's figures from the stored per-row losses.

uv run --no-sync python ml-experiments/00209-*/e00209_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00209_table as table
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{table.EXPERIMENT_ID:05d}-{table.SLUG}"
)
LABELS: Final[dict[str, str]] = {
    "T0": "Table (T0)",
    "T1": "Table + temperature extremes (T1)",
    "T2": "Table without forecast level by era (T2)",
    "T3": "Table, calendar fitted per era (T3)",
    "T4": "Table + extra level interactions (T4)",
    "blind": "Network without error inputs",
    "head": "Network with error inputs",
}
WIDTH: Final[int] = 560


def intervals(*, rows: pl.DataFrame, scope: tuple[str, ...]) -> pl.DataFrame:
    """Each arm's score minus arm D's, with a 99% month-resampled interval."""
    rng = np.random.default_rng(0)
    sub = rows.filter(pl.col("site").is_in(scope))
    months = sub["month"].to_numpy()
    out = []
    for arm, label in LABELS.items():
        diff = sub[arm].to_numpy() - sub["arm_d"].to_numpy()
        m, lo, hi = table.interval(diff=diff, months=months, rng=rng)
        kind = "network" if arm in ("blind", "head") else "table"
        out.append({"arm": label, "kind": kind, "difference": m, "lower": lo, "upper": hi})
    return pl.DataFrame(out)


def main() -> None:
    rows = pl.read_parquet(table.WORK_DIR / "rows.parquet")
    data = intervals(rows=rows, scope=table.DECISION)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(LABELS.values()), title=None),
        color=alt.Color(
            "kind:N",
            legend=None,
            scale=alt.Scale(domain=["table", "network"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        ),
    )
    bars = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Quantile score minus arm D's, points (negative is better); 99% interval",
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=90).encode(x="difference:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    chart = (zero + bars + points).properties(
        width=WIDTH,
        height=240,
        title="No table gets more than 0.04 points; the network without error inputs gets 0.10",
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "figure_1_tables_and_networks.svg")
    print("wrote figure_1")


if __name__ == "__main__":
    main()
