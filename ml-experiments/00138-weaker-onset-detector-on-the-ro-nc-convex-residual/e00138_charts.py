"""Draw experiment 138's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00138-*/e00138_charts.py
"""

from __future__ import annotations

import itertools
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00138_arms import ARMS
from e00138_contrasts import CONTROL, RUN_DIR, STEP_EDGES, WEAK_ARMS, contrast, pooled, populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00138-weaker-onset-detector-on-the-ro-nc-convex-residual"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
ARM_LABELS: Final[dict[str, str]] = {
    "B": "B: RO-NC as in experiment 130 (control)",
    "W1": "W1: lower candidate bar",
    "W2": "W2: lower bar and persistence feature",
    "W3": "W3: slow 72-hour window added",
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


def _interval_chart(
    *, rows: list[dict], order: list[str], axis_title: str, domain: list[float]
) -> alt.LayerChart:
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color(
            "label:N", legend=None, scale=alt.Scale(domain=order, range=COLOURS[: len(order)])
        ),
    )
    return (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=axis_title, scale=alt.Scale(domain=domain)), x2="high:Q"
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=110)


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of every arm with 95% intervals."""
    frames = pooled(data=data, population="base", parity="even")
    rows = []
    for arm, label in ARM_LABELS.items():
        point, low, high = contrast(frames=frames, arm=arm, level=0.95)
        rows.append({"label": label, "point": point, "low": low, "high": high})
    return _interval_chart(
        rows=rows,
        order=list(ARM_LABELS.values()),
        axis_title="Share of moved MW found",
        domain=[0.5, 1.0],
    ).properties(
        title=_title(
            text="All four detectors find about 0.79 of the moved power",
            lines=[
                (
                    "Share of the megawatts (MW) moved by injected transfers that each detector "
                    "finds, at 80 false alarms on odd months (larger is better). Transfers over "
                    "1 day starting in even months, 15 Primary substations, three injection shapes "
                    "pooled. Dot: estimate. Line: 95% interval from resampling whole calendar "
                    "months."
                )
            ],
        )
    )


def figure_contrasts(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """The three planned contrasts with their 98.33% intervals."""
    frames = pooled(data=data, population="base", parity="even")
    rows = []
    for arm in WEAK_ARMS:
        point, low, high = contrast(frames=frames, arm=arm, base=CONTROL)
        rows.append(
            {
                "label": ARM_LABELS[arm].split(":")[0] + " minus B",
                "point": point,
                "low": low,
                "high": high,
            }
        )
    chart = _interval_chart(
        rows=rows,
        order=[str(r["label"]) for r in rows],
        axis_title="Difference in share of moved MW found",
        domain=[-0.06, 0.06],
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    gain = (
        alt.Chart(pl.DataFrame({"x": [0.03]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[2, 2])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (chart + zero + gain).properties(
        title=_title(
            text="No weaker-onset arm finds more of the moved power than arm B",
            lines=[
                (
                    "Planned contrasts, each arm minus arm B. Dot: estimate. Line: 98.33% interval "
                    "(Bonferroni for three contrasts). Orange dashed line: the +0.03 bar a "
                    "gain must clear."
                )
            ],
        )
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of transfers found by donor step, for transfers with no metered receiver."""
    rows = []
    for low, high in itertools.pairwise(STEP_EDGES[:4]):
        n, found = 0, dict.fromkeys(ARMS, 0)
        for frame in pooled(data=data, population="base", parity="even"):
            part = frame.filter(
                pl.col("metered_receivers") == 0,
                pl.col("donor_step_z").abs().is_between(low, high, "left"),
            )
            n += part.height
            for arm in ARMS:
                found[arm] += int(part[arm].sum())
        rows.extend(
            {
                "band": f"{low:g} to {high:g} noise units (n={n})",
                "arm": ARM_LABELS[arm],
                "share": found[arm] / n,
            }
            for arm in ARMS
        )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("band:N", title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
            xOffset="arm:N",
            y=alt.Y("share:Q", title="Share of transfers found"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=COLOURS),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Lowering the bar finds fewer weak transfers, not more",
                lines=[
                    (
                        "Share of injected transfers found, by the size of the donor's first step, "
                        "for transfers with no metered receiving substation (even months, three "
                        "shapes summed)."
                    )
                ],
            ),
        )
    )


def figure_real() -> alt.TopLevelMixin:
    """Labelled real transfers over 1 day found by each arm."""
    real = pl.read_parquet(RUN_DIR / "real_results.parquet").filter(pl.col("hours") > 24)
    rows = [
        {"label": ARM_LABELS[a], "found": int(real.filter(pl.col("setting") == a)["found"].sum())}
        for a in ARMS
    ]
    n = real.filter(pl.col("setting") == CONTROL).height
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)
            ),
            x=alt.X(
                "found:Q", title=f"Real transfers found (of {n})", scale=alt.Scale(domain=[0, n])
            ),
            color=alt.Color(
                "label:N",
                legend=None,
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=COLOURS),
            ),
        )
        .properties(
            width=WIDTH,
            height=110,
            title=_title(
                text="On real transfers, W1 to W3 find 3 to 5 more than arm B",
                lines=[
                    (
                        "Labelled real transfers lasting over 1 day found by each detector. "
                        "Descriptive: the scores were fitted on injected transfers only."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Draw the four figures."""
    data = populations()
    _save(figure_recall(data=data), name="figure_1_recall_by_arm.svg")
    _save(figure_contrasts(data=data), name="figure_2_contrasts.svg")
    _save(figure_by_step(data=data), name="figure_3_share_found_by_step.svg")
    _save(figure_real(), name="figure_4_real_transfers_found.svg")


if __name__ == "__main__":
    main()
