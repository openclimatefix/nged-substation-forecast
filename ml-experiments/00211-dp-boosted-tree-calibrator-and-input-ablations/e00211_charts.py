# ruff: noqa: D103
"""Draw experiment 211's figure from the stored per-row losses.

uv run --no-sync python ml-experiments/00211-*/e00211_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00211_data as data_module
import e00211_run as run
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{data_module.EXPERIMENT_ID:05d}-{data_module.SLUG}"
)
ARMS: Final[dict[str, tuple[str, str]]] = {
    "table_T0": ("Lookup table (experiment 209)", "table"),
    "stored_blind": ("Network without error inputs (experiment 200)", "network"),
    "reference": ("Network without error inputs, retrained here", "network"),
    "xgboost": ("Boosted-tree model", "tree"),
    "no_lead": ("Retrained network without lead", "ablation"),
    "no_half_hour": ("Retrained network without half-hour", "ablation"),
    "no_level": ("Retrained network without forecast level", "ablation"),
    "no_era": ("Retrained network without era flag", "ablation"),
}
KINDS: Final[list[str]] = ["table", "network", "tree", "ablation"]
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, "#2E8B57", "#808080"]


def main() -> None:
    rows = pl.read_parquet(data_module.WORK_DIR / "rows.parquet").filter(
        pl.col("site").is_in(run.DECISION)
    )
    months = rows["month"].to_numpy()
    rng = np.random.default_rng(0)
    out = []
    for arm, (label, kind) in ARMS.items():
        diff = rows[arm].to_numpy() - rows["arm_d"].to_numpy()
        m, lo, hi = run.interval(diff=diff, months=months, rng=rng)
        out.append({"arm": label, "kind": kind, "difference": m, "lower": lo, "upper": hi})
    data = pl.DataFrame(out)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=[v[0] for v in ARMS.values()], title=None),
        color=alt.Color("kind:N", legend=None, scale=alt.Scale(domain=KINDS, range=COLOURS)),
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
        width=480,
        height=280,
        title="The boosted-tree model gains as much as the network; forecast level and era matter",
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / "figure_1_tree_and_ablations.svg")
    print("wrote figure_1")


if __name__ == "__main__":
    main()
