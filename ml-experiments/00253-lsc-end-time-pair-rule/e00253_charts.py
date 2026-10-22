"""Draw experiment 253's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00253-*/e00253_charts.py
"""

from __future__ import annotations

import contextlib
import io
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00253_contrasts import analyse, sides

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / "00253-lsc-end-time-pair-rule"
)
WIDTH: Final[int] = 480
WRAP: Final[int] = 80
VARIANTS: Final[dict[str, str]] = {
    "clean": "Clean draw",
    "C": "Unrelated step added",
    "U10": "Step sizes 10% unequal",
}
RULES: Final[list[str]] = ["R0", "P1", "P2", "P3"]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def collect() -> pl.DataFrame:
    """Every share, contrast, and disagreement row of every variant."""
    data = sides()
    frames = []
    for variant, name in VARIANTS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            rows = analyse(variant=variant, data=data)["rows"]
        frames.append(
            pl.DataFrame(rows, infer_schema_length=None).with_columns(variant=pl.lit(name))
        )
    return pl.concat(frames, how="diagonal")


def figure_contrasts(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """The four planned contrasts of the share found, by variant, with 98.75% intervals."""
    order = ["P2 minus R0", "P1 minus R0", "P3 minus P2", "P3 minus R0"]
    data = rows.filter(pl.col("kind") == "contrast", pl.col("setting").is_in(order))
    base = alt.Chart(data).encode(y=alt.Y("setting:N", title=None, sort=order))
    return (
        alt.layer(
            base.mark_rule().encode(x="low:Q", x2="high:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_point(filled=True, color=ocf.DATA_BLUE, size=60).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("point:Q", title="Change in share found (98.75% interval)")
            ),
        )
        .properties(width=WIDTH // 2, height=110)
        .facet(column=alt.Column("variant:N", title=None, sort=list(VARIANTS.values())))
        .properties(
            title=_title(
                text="The pair rule helps when an unrelated step follows the end, "
                "and loses a little otherwise",
                line="Change in the share of moved MW whose end is found, side by side, against "
                "the 24-hour lookahead rule R0.",
            )
        )
    )


def figure_disagreement(*, rows: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of pair transfers whose two sides end more than 6 hours apart."""
    data = rows.filter(pl.col("kind") == "disagreement", pl.col("setting").is_in(RULES))
    return (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("setting:N", title=None, sort=RULES),
            y=alt.Y("point:Q", title="Share ending over 6 hours apart"),
            column=alt.Column("variant:N", title=None, sort=list(VARIANTS.values())),
        )
        .properties(
            width=WIDTH // 3 - 20,
            height=150,
            title=_title(
                text="The joint score P2 halves the disagreement after an unrelated step",
                line="Share of transfers, where both sides alarmed, whose two ends differ by "
                "more than 6 hours.",
            ),
        )
    )


def main() -> None:
    """Write the two figures."""
    rows = collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_contrasts(rows=rows).save(OUT_DIR / "contrasts.svg")
    figure_disagreement(rows=rows).save(OUT_DIR / "disagreement.svg")


if __name__ == "__main__":
    main()
