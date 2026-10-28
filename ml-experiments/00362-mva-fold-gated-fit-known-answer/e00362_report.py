"""Print every number of experiment 362's page and draw its figure.

uv run --no-sync python ml-experiments/00362-*/e00362_report.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-gated-fit-known-answer"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00362-{SLUG}" / "work" / "gated"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00362-{SLUG}"
)
ARMS: Final[dict[str, str]] = {
    "s": "S: signed fit (ceiling)",
    "l0": "L0: ungated",
    "g": "G: gate at ratio 0.75 (registered)",
    "g1": "G: gate at ratio 1.0 (post hoc)",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def load() -> pl.DataFrame:
    """Return every job's result as one table."""
    rows = [json.loads(path.read_text()) for path in sorted(WORK.glob("*.json"))]
    return pl.DataFrame(rows, infer_schema_length=None)


def tables(*, frame: pl.DataFrame) -> None:
    """Print the numbers the page quotes."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    for cut in ("scored", "replicate"):
        part = frame.filter(pl.col("cut") == cut, pl.col("reactive") == "Q0").sort("level", "site")
        print(
            f"\n== {cut} cut, zero reactive power: sunny direction accuracy and capacity ratio =="
        )
        print(
            part.select(
                "level",
                "site",
                "n_sunny",
                *[
                    pl.col(f"acc_{arm}_sunny").round(3)
                    for arm in ("s", "n", "l0", "g", "g1", "g1_25")
                ],
                *[
                    (pl.col(f"capacity_{arm}") / pl.col("capacity_s")).round(2).alias(f"cap_{arm}")
                    for arm in ("l0", "g", "g1", "g1_25")
                ],
                "n_starts_rejected",
                "n_starts_rejected_g1",
                "all_rejected_g1",
            )
        )
        for ratio in ("g", "g1"):
            print(
                cut,
                ratio,
                "sites within 0.02 of S at level 1.0:",
                int(
                    (
                        (
                            part.filter(pl.col("level") == 1.0)[f"acc_{ratio}_sunny"]
                            - part.filter(pl.col("level") == 1.0)["acc_s_sunny"]
                        ).abs()
                        <= 0.02
                    ).sum()
                ),
                "level 2.0:",
                int(
                    (
                        (
                            part.filter(pl.col("level") == 2.0)[f"acc_{ratio}_sunny"]
                            - part.filter(pl.col("level") == 2.0)["acc_s_sunny"]
                        ).abs()
                        <= 0.02
                    ).sum()
                ),
                "D5 natural gap to S:",
                round(
                    float(
                        part.filter(pl.col("level") == 0.0, pl.col("site") == "D5")["acc_s_sunny"][
                            0
                        ]
                        - part.filter(pl.col("level") == 0.0, pl.col("site") == "D5")[
                            f"acc_{ratio}_sunny"
                        ][0]
                    ),
                    3,
                ),
            )


def figure(*, frame: pl.DataFrame) -> None:
    """Draw direction accuracy by arm: D5 with no injection, and the five sites at level 2.0."""
    scored = frame.filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0")
    cells = scored.filter(
        ((pl.col("level") == 0.0) & (pl.col("site") == "D5")) | (pl.col("level") == 2.0)
    ).with_columns(
        cell=pl.when(pl.col("level") == 0.0)
        .then(pl.lit("D5, no injection"))
        .otherwise(pl.concat_str(pl.col("site"), pl.lit(", level 2.0")))
    )
    long = pl.concat(
        [
            cells.select("cell", arm=pl.lit(label), accuracy=pl.col(f"acc_{arm}_sunny"))
            for arm, label in ARMS.items()
        ]
    ).to_pandas()
    chart = (
        _chart(long)
        .mark_bar()
        .encode(
            x=alt.X("cell:N", title=None, axis=alt.Axis(labelAngle=-30)),
            xOffset="arm:N",
            y=alt.Y(
                "accuracy:Q",
                title="Share of sunny half-hours with the flow direction right",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(ARMS.values()),
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_AMBER, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
        )
        .properties(width=520, height=240)
    )
    chart.save(DOCS_DIR / "figure_1_gated_direction_accuracy.svg")


if __name__ == "__main__":
    data = load()
    tables(frame=data)
    figure(frame=data)
