# ruff: noqa: E501, PTH207, B007
"""Draw the figures of experiment 328 from the stored scores.

PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-* uv run --no-sync python \
        ml-experiments/00328-*/e00328_charts.py
"""

import glob
import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00328_common as c
import e00328_report as r
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{c.EXPERIMENT_ID:05d}-{c.SLUG}"
)
WIDTH: Final[int] = 640
NAMES: Final[dict[str, str]] = {
    "A": "A: all history", "B": "B: last 24 months",
    "C2": "C2: regime effects, regime-only bands", "D2": "D2: C2 plus filter reset",
    "C2t": "C2t: regime effects, trailing bands (post hoc)", "D2t": "D2t: C2t plus filter reset (post hoc)",
    "C2o": "C2o: true change date (oracle)",
}  # fmt: skip


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 85),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: Any, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def detection_curve() -> None:
    """Recall of injected 10% steps within 60 days against false changes per Primary-year."""
    chosen = json.loads((c.WORK_DIR / "chosen_detectors.json").read_text())
    base = pl.concat(
        [pl.read_parquet(p) for p in glob.glob(str(c.WORK_DIR / "base" / "*_alarms.parquet"))]
    )
    inj = pl.concat(
        [pl.read_parquet(p) for p in glob.glob(str(c.WORK_DIR / "inject" / "*.parquet"))]
    )
    spans = {}
    import numpy as np

    for path in glob.glob(str(c.WORK_DIR / "base" / "*.npz")):
        t = np.load(path)["times"]
        spans[path.split("/")[-1].removesuffix(".npz")] = (
            float((t[-1] - t[0]) / np.timedelta64(1, "D")) / 365.25
        )
    sites = [s for s in spans if s != "D8"]
    years = sum(spans[s] for s in sites)
    rows = []
    for (det, param), g in base.filter(pl.col("site").is_in(sites)).group_by("detector", "param"):
        rec = inj.filter(
            (pl.col("detector") == det)
            & (pl.col("param") == param)
            & (pl.col("shape") == "step")
            & (pl.col("size") == 0.10)
        )["detected"].mean()
        rows.append({"detector": det, "param": param, "false": g.height / years, "recall": rec})
    for det, param in (
        ("rolling_gam", c.CUSUM_H_GRID[-1]),
        ("histogram", c.HISTOGRAM_BETA_GRID[-1]),
    ):
        pass
    frame = pl.DataFrame(rows).with_columns(
        detector=pl.col("detector").replace(
            {"rolling_gam": "Rolling lagless GAM", "histogram": "Weekly histogram"}
        )
    )
    line = alt.Chart(frame).encode(
        x=alt.X("false:Q", title="False changes per Primary-year (real records, 14 Primaries)"),
        y=alt.Y(
            "recall:Q",
            title="Injected 10% steps found within 60 days",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color(
            "detector:N", scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]), title=None
        ),
    )
    chart = (line.mark_line(point=True) + alt.Chart(pl.DataFrame({"x": [1.0]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q")).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH, height=300,
        title=_title(title="Neither detector reaches 80% recall of 10% steps within one false change per Primary-year",
                     subtitle=f"Each point is one threshold; chosen thresholds {chosen}. All 28 onsets, 14 Primaries, 60-day window."),
    )  # fmt: skip
    _save(chart, name="detection_curve")


def headline() -> None:
    """B minus each arm, quantile score, injected 25% steps, with 98.75% intervals."""
    inj = r.load(pattern="inj_*.parquet").with_columns(
        cluster=pl.col("site") + "|" + pl.col("world")
    )
    inj = inj.with_columns(size=pl.col("world").str.extract(r"step_([0-9.]+)_").cast(pl.Float64))
    rows = []
    for size in c.STEP_SIZES:
        sub = inj.filter(pl.col("size") == size)
        for label, groups in (
            ("first 90 days", r.GROUPS_90),
            ("after day 90", ("d90_180", "d180p")),
        ):
            for arm in ("C2", "D2", "C2t", "D2t"):
                m, lo, hi = r.clustered_difference(
                    frame=sub, cluster="cluster", arm_a="B", arm_b=arm, groups=groups
                )
                rows.append(
                    {
                        "size": f"{size:.0%} step",
                        "rows": label,
                        "arm": NAMES[arm],
                        "diff": m,
                        "lo": lo,
                        "hi": hi,
                    }
                )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("arm:N", title=None, sort=[NAMES[a] for a in ("C2", "D2", "C2t", "D2t")]),
        color=alt.Color(
            "arm:N",
            legend=None,
            scale=alt.Scale(
                domain=[NAMES[a] for a in ("C2", "D2", "C2t", "D2t")],
                range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
            ),
        ),
    )
    chart = (
        base.mark_errorbar(thickness=2).encode(x=alt.X("lo:Q", title="B minus arm, quantile score (positive: the regime arm is better)"), x2="hi:Q")  # ty: ignore[unresolved-attribute]
        + base.mark_point(filled=True, size=70).encode(x="diff:Q")  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH // 2 - 60, height=170).facet(row=alt.Row("size:N", title=None), column=alt.Column("rows:N", title=None)).properties(
        title=_title(title="Regime arms do not beat the recent window in the first 90 days after an injected step",
                     subtitle="Worlds of 14 Primaries x 8 even-month onsets; 98.75% intervals resample whole worlds. Positive means the regime arm scores better than arm B.")
    )  # fmt: skip
    _save(chart, name="headline_contrasts")


def d8_groups() -> None:
    """D8 quantile score by days since the break for selected arms."""
    real = r.load(pattern="real_D8.parquet")
    order = ["pre", "d0_30", "d30_90", "d90_180", "d180p"]
    labels = {
        "pre": "before",
        "d0_30": "days 0-30",
        "d30_90": "days 30-90",
        "d90_180": "days 90-180",
        "d180p": "after day 180",
    }
    rows = [
        {"group": labels[g], "arm": NAMES[a], "score": r.score(frame=real, arm=a, groups=(g,))}
        for g in order
        for a in ("A", "B", "C2t", "C2o")
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_line(point=True, strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("group:N", sort=[labels[g] for g in order], title="Days since D8's break (2023-08-15)"),
            y=alt.Y("score:Q", title="Quantile score (points of capacity)"),
            color=alt.Color("arm:N", title=None, scale=alt.Scale(domain=[NAMES[a] for a in ("A", "B", "C2t", "C2o")], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN])),
        )
        .properties(width=WIDTH, height=300, title=_title(title="At D8 every arm scores 4 times worse in the first 90 days after the break than before it", subtitle="Quantile score by days since the break; the true change date is used only by the oracle arm."))
    )  # fmt: skip
    _save(chart, name="d8_by_group")


if __name__ == "__main__":
    detection_curve()
    headline()
    d8_groups()
