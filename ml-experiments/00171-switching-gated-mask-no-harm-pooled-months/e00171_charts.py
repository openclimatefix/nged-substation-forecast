"""Draw experiment 171's figures as SVG from the stored tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00171-*/e00171_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00171_report as rep
import plotting.ocf_theme as ocf
import polars as pl

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00171-switching-gated-mask-no-harm-pooled-months"
)
WIDTH: Final[int] = 520


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _intervals(
    *, data: pl.DataFrame, title: alt.TitleParams, domain: list[float], height: int
) -> alt.TopLevelMixin:
    limits = pl.DataFrame({"v": [-0.03, 0.03]})
    return alt.layer(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2, color=ocf.DATA_GREEN)
        .encode(
            x=alt.X(
                "low:Q",
                title="Gated mask minus no mask (points; lower is better)",
                scale=alt.Scale(domain=domain),
            ),
            x2="high:Q",
            y=alt.Y("name:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
        ),
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=ocf.DATA_GREEN)
        .encode(x="mean:Q", y=alt.Y("name:N", sort=None)),
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(limits).mark_rule(strokeDash=[2, 2], color=ocf.BRAND_ORANGE).encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=height, title=title)


def _row(*, name: str, frame: pl.DataFrame) -> dict:
    t = rep.contrast(frame=frame, arm="MASK_GATED")
    return {"name": name, "mean": t[0], "low": t[1], "high": t[2]}


def main() -> None:
    """Draw the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wide = rep.build()
    scored = wide.filter(pl.col("valid_time") >= rep.SCORED_YEAR_START)
    earlier = wide.filter(pl.col("valid_time") < rep.SCORED_YEAR_START)
    groups = [
        ("Pooled, 27 months (G1)", wide),
        ("November to March, 10 months (G2)", wide.filter(pl.col("winter"))),
        ("April to October, 17 months", wide.filter(~pl.col("winter"))),
        ("July 2025 to June 2026 (experiment 166's year)", scored),
        ("April 2024 to June 2025", earlier),
    ]
    data = pl.DataFrame([_row(name=n, frame=f) for n, f in groups])
    title = _title(
        text="The pooled interval is still wider than the limit of 0.03",
        line=(
            "Dot: estimate. Line: 95% interval from resampling whole calendar months. Grey rule: "
            "no difference. Orange rules: the pre-registered limit of +/-0.03. Real rows outside "
            "logged transfers, six Primary substations."
        ),
    )
    _intervals(data=data, title=title, domain=[-0.12, 0.15], height=170).save(
        OUT_DIR / "figure_1_headline.svg"
    )
    sites = pl.DataFrame(
        [_row(name=s, frame=wide.filter(pl.col("site") == s)) for s in rep.SITES]
        + [_row(name=f"All except {s}", frame=wide.filter(pl.col("site") != s)) for s in ("D10",)]
    )
    title = _title(
        text="One substation, D10, carries most of the extra error",
        line=(
            "Each row is the contrast on one substation's rows, except the last row, which leaves "
            "D10 out. Dot: estimate. Line: 95% interval from resampling whole calendar months."
        ),
    )
    _intervals(data=sites, title=title, domain=[-0.3, 0.45], height=190).save(
        OUT_DIR / "figure_2_by_substation.svg"
    )
    monthly = (
        wide.with_columns(d=pl.col("MASK_GATED") - pl.col("K0"))
        .group_by("month")
        .agg(d=pl.col("d").mean())
        .sort("month")
    )
    bars = (
        alt.Chart(monthly)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("month:N", title="Month of the valid time", axis=alt.Axis(labelAngle=-60)),
            y=alt.Y("d:Q", title="Gated mask minus no mask (points; lower is better)"),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The extra error comes from a few months",
                line=(
                    "Mean quantile score of the gated mask minus the unmasked stack, per calendar "
                    "month, on real rows outside logged transfers, six Primary substations."
                ),
            ),
        )
    )
    bars.save(OUT_DIR / "figure_3_by_month.svg")


if __name__ == "__main__":
    main()
