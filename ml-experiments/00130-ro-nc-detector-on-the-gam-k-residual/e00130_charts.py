"""Draw experiment 130's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00130-*/e00130_charts.py
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00130_contrasts import STEP_EDGES, contrast, pooled, populations
from e00130_diagnose import table

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00130-ro-nc-detector-on-the-gam-k-residual"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
ARM_LABELS: Final[dict[str, str]] = {
    "A15_XL": "A: lagless XGBoost residual",
    "B_XL": "B: frozen convex-model residual (headline)",
    "C_XL": "C: filter innovations of the convex model",
    "B_LOGIT_SAME": "D: arm B with one model, no refit offset",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of RO-NC on each residual, with 95% intervals."""
    frames = pooled(data=data, population="base", parity="even")
    rows = []
    for arm, name in ARM_LABELS.items():
        point, low, high = contrast(frames=frames, arm=arm)
        rows.append({"arm": name, "point": point, "low": low, "high": high})
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color(
            "arm:N", legend=None, scale=alt.Scale(domain=list(ARM_LABELS.values()), range=COLOURS)
        ),
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=110)
    return chart.properties(
        title=_title(
            text="Arm B finds a little more of the moved power; arm C finds much less",
            lines=[
                (
                    "Share of the megawatts (MW) moved by injected transfers that RO-NC finds, on "
                    "each "
                    "residual (larger is better). Transfers over 1 day starting in even months, 15 "
                    "Primary substations, three injection shapes pooled. Dot: estimate. Line: 95% "
                    "interval from resampling whole calendar months."
                )
            ],
        )
    )


def figure_contrasts(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """The planned contrast and the descriptive contrasts, with intervals."""
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    specs = [
        ("C1 (planned): B minus A", base, "B_XL", "A15_XL"),
        ("C2: C minus A", base, "C_XL", "A15_XL"),
        ("D minus B: one model, no refit offset", base, "B_LOGIT_SAME", "B_XL"),
        ("B minus A on transfers of 15.6 to 60 days", long, "B_XL", "A15_XL"),
    ]
    rows = []
    for label, frames, arm, ref in specs:
        point, low, high = contrast(frames=frames, arm=arm, base=ref)
        rows.append({"contrast": label, "point": point, "low": low, "high": high})
    frame = pl.DataFrame(rows)
    order = [r[0] for r in specs]
    y = alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=320))
    marks = (
        alt.Chart(frame)
        .mark_rule(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, x=alt.X("low:Q", title="Difference in share of moved MW found"), x2="high:Q"
        )
        + alt.Chart(frame)
        .mark_point(filled=True, size=70, color=ocf.DATA_BLUE)
        .encode(y=y, x="point:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0, 0.03], "label": ["no change", "threshold +0.03"]}))
        .mark_rule(strokeDash=[4, 3], color="gray")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (marks + zero).properties(width=WIDTH, height=130)
    return chart.properties(
        title=_title(
            text="The headline gain of arm B is small and its interval touches zero",
            lines=[
                (
                    "Paired difference in the share of moved MW found by RO-NC (larger is better), "
                    "transfers over 1 day unless stated, even months. Dot: estimate. Line: 95% "
                    "interval from resampling whole calendar months. Grey rules: no change, and "
                    "the "
                    "+0.03 that the plan set for a supported verdict."
                )
            ],
        )
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of transfers found, by donor step size, for arms A, B, and C."""
    rows = []
    for metered, mlabel in ((True, "A metered receiver exists"), (False, "No metered receiver")):
        for low, high in itertools.pairwise(STEP_EDGES):
            n, found = 0, dict.fromkeys(ARM_LABELS, 0)
            for frame in pooled(data=data, population="base", parity="even"):
                part = frame.filter(
                    (pl.col("metered_receivers") > 0) == metered,
                    pl.col("donor_step_z").abs().is_between(low, high, "left"),
                )
                n += part.height
                for arm in ("A15_XL", "B_XL", "C_XL"):
                    found[arm] += int(part[arm].sum())
            if n < 30:
                continue
            label = f"{low:g} to {high:g}" if high != float("inf") else f"{low:g} or more"
            rows.extend(
                {
                    "panel": mlabel,
                    "step": label,
                    "order": low,
                    "arm": ARM_LABELS[arm],
                    "share": found[arm] / n,
                }
                for arm in ("A15_XL", "B_XL", "C_XL")
            )
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "step:N",
                sort=["0 to 2", "2 to 4", "4 to 6", "6 to 10", "10 or more"],
                title="Donor step (noise units)",
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("arm:N", sort=list(ARM_LABELS.values())),
            y=alt.Y("share:Q", title="Share of transfers found", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm:N",
                sort=list(ARM_LABELS.values()),
                scale=alt.Scale(domain=list(ARM_LABELS.values())[:3], range=COLOURS[:3]),
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
            ),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("panel:N", title=None))
    )
    return chart.properties(
        title=_title(
            text="Arm B finds more of the transfers whose step is 2 to 4 noise units",
            lines=[
                (
                    "Share of injected transfers that RO-NC finds, by the size of the donor's step "
                    "in that arm's noise units (larger is better), transfers over 1 day starting "
                    "in "
                    "even months, three injection shapes summed. Bins with under 30 transfers are "
                    "left out."
                )
            ],
        )
    )


def figure_diagnostic() -> alt.TopLevelMixin:
    """Per series: noise of B and C relative to A, and the share of a step that the filter keeps."""
    rows = []
    for row in table():
        rows.append(
            {
                "series": f"Primary {int(row['series'])}",
                "kind": "B noise / A noise",
                "value": row["scale_b"] / row["scale_a"],
            }
        )
        rows.append(
            {
                "series": f"Primary {int(row['series'])}",
                "kind": "C noise / A noise",
                "value": row["scale_c"] / row["scale_a"],
            }
        )
        rows.append(
            {
                "series": f"Primary {int(row['series'])}",
                "kind": "Share of a 3-day step kept by C",
                "value": row["retained_share"],
            }
        )
    kinds = ["B noise / A noise", "C noise / A noise", "Share of a 3-day step kept by C"]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_circle(size=60)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", title="Ratio (a step statistic in MW)"),
            y=alt.Y("series:N", title=None),
            color=alt.Color(
                "kind:N",
                sort=kinds,
                scale=alt.Scale(
                    domain=kinds, range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN]
                ),
                legend=alt.Legend(orient="top", title=None, columns=1, labelLimit=400),
            ),
        )
        .properties(width=WIDTH, height=260)
    )
    return chart.properties(
        title=_title(
            text="The filter shrinks the noise to 0.2 of arm A's but keeps only 0.2 of a step",
            lines=[
                (
                    "For each Primary: the odd-month noise of the 24-hour step statistic of arm B "
                    "and arm C divided by arm A's, and the share of a constant 3-day step added to "
                    "the residual that arm C's statistic still shows (median over 30 starts in "
                    "even "
                    "months)."
                )
            ],
        )
    )


def main() -> None:
    """Draw every figure."""
    data = populations()
    _save(figure_recall(data=data), name="figure_1_recall_by_arm.svg")
    _save(figure_contrasts(data=data), name="figure_2_contrasts.svg")
    _save(figure_by_step(data=data), name="figure_3_share_found_by_step.svg")
    _save(figure_diagnostic(), name="figure_4_noise_and_retained_step.svg")


if __name__ == "__main__":
    main()
