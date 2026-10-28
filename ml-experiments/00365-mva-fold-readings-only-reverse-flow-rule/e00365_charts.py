"""Draw the figure of experiment 365 from `work/rule.parquet` and `work/census.parquet`.

uv run --no-sync python ml-experiments/00365-*/e00365_charts.py
"""

from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-readings-only-reverse-flow-rule"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00365-{SLUG}" / "work"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00365-{SLUG}"
)
THRESHOLD: Final[float] = 5.0
CASES: Final[dict[float, str]] = {
    0.0: "no injection",
    1.0: "injected, level 1.0",
    2.0: "injected, level 2.0",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def main() -> None:
    """Draw the statistic by substation and case, with the threshold."""
    folded = (
        pl.read_parquet(WORK / "rule.parquet")
        .filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0")
        .select(
            "site",
            case=pl.col("level").replace_strict(CASES),
            value=pl.col("p1_training_sunny"),
        )
    )
    census = pl.read_parquet(WORK / "census.parquet").select(
        "site", case=pl.lit("MVA-only Primary, as metered"), value=pl.col("p1_training_sunny")
    )
    data = pl.concat([folded, census]).to_pandas()
    sites = ["D5", "D10", "D11", "D14", "D16", "D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22"]
    points = (
        _chart(data)
        .mark_circle(size=90)
        .encode(
            x=alt.X("site:N", sort=sites, title="Primary substation", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "value:Q", title="1st percentile of training sunny-midday readings (% of capacity)"
            ),
            color=alt.Color(
                "case:N",
                scale=alt.Scale(
                    domain=[*CASES.values(), "MVA-only Primary, as metered"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_AMBER, ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
    )
    rule = (
        _chart(pl.DataFrame({"y": [THRESHOLD]}).to_pandas())
        .mark_rule(strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    (points + rule).properties(width=560, height=260).save(
        DOCS_DIR / "figure_1_rule_statistic_by_substation.svg"
    )


if __name__ == "__main__":
    main()
