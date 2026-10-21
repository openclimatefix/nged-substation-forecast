# ruff: noqa: E501
"""Draw experiment 156's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00156-*/e00156_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00156_consumer_report as rep
import e00156_detect_report as det
import plotting.ocf_theme as ocf
import polars as pl
from e00156_common import WORK_DIR

HERE: Final[Path] = Path(__file__).resolve().parent
OUT_DIR: Final[Path] = (
    HERE.parents[1]
    / "docs"
    / "ml-experiments"
    / "00156-switching-faster-onset-alarm-from-opposite-steps"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "K0": "No correction",
    "GAM_ONLY": "Weather model only (ignore recent power)",
    "MASK_RONC": "Mask from RO-NC alarm",
    "MASK_UNION": "Mask from either alarm (pre-registered union)",
    "MASK_NEW": "Mask from pair alarm only",
    "MASK_UNION2_MATCHED": "Post hoc: pair + matched RO-NC (uses truth)",
    "CLEAN": "Forecast fed the true series (ceiling)",
}
COLOURS: Final[dict[str, str]] = {
    "No correction": ocf.BRAND_ORANGE,
    "Weather model only (ignore recent power)": ocf.DATA_PURPLE,
    "Mask from RO-NC alarm": ocf.DATA_SKY,
    "Mask from either alarm (pre-registered union)": ocf.DATA_BLUE,
    "Mask from pair alarm only": ocf.DATA_GREEN,
    "Post hoc: pair + matched RO-NC (uses truth)": "#8FF0C8",
    "Forecast fed the true series (ceiling)": "#444444",
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / f"{name}.svg")


def _dots(
    *,
    data: pl.DataFrame,
    x: str,
    title: str,
    y: str,
    colour: str,
    height: int,
    rule: float | None = None,
) -> alt.TopLevelMixin:
    base = alt.Chart(data)
    enc_y = alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(labelLimit=330))
    layers = [
        base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=title),
            x2="high:Q",
            y=enc_y,
            color=alt.Color(
                f"{colour}:N",
                legend=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            ),
        ),
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=f"{x}:Q",
            y=enc_y,
            color=alt.Color(
                f"{colour}:N",
                legend=None,
                scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            ),
        ),
    ]
    if rule is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"v": [rule]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#888")
            .encode(x="v:Q")
        )
    return alt.layer(*layers).properties(width=WIDTH, height=height)


def main() -> None:
    """Draw the figures."""
    frame = rep.load()
    table = rep.wide(frame)
    extra = pl.concat(
        [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores_posthoc").glob("*.parquet"))]
    )
    keys = ["site", "world", "period", "row"]
    table = table.join(
        extra.pivot(on="arm", index=keys, values="qs").select(*keys, "MASK_UNION2_MATCHED"), on=keys
    )
    months = table["month"].to_numpy()
    rows = []
    for arm, label in LABELS.items():
        p, lo, hi, _, _ = rep.interval(months=months, num=table[arm].to_numpy(), den=None)
        rows.append({"arm": label, "score": p, "low": lo, "high": hi})
    chart = _dots(
        data=pl.DataFrame(rows),
        x="score",
        y="arm",
        colour="arm",
        height=190,
        title="Mean quantile score, points of capacity (smaller is better)",
    ).properties(
        title=_title(
            text="A mask driven by the pair alarm scores 2.96, against 3.34 for RO-NC and 3.55 with no correction",
            line="Dot: mean over all forecasts issued after a simulated load transfer began, 277,056 forecast rows, scored against the true series. Line: 95% interval from resampling whole months of target time.",
        )
    )
    _save(chart=chart, name="figure_1_headline")

    sets = {
        "Before the RO-NC alarm is known": table.filter(pl.col("known_ronc") > 0),
        "After the RO-NC alarm is known": table.filter(pl.col("known_ronc") <= 0),
        "Targets inside the transfer": table.filter(pl.col("inside")),
        "Targets in the 72 h after the transfer": table.filter(~pl.col("inside")),
    }
    rows = []
    for set_name, part in sets.items():
        gap = (part["K0"] - part["CLEAN"]).to_numpy()
        for arm in ("MASK_RONC", "MASK_UNION", "MASK_NEW"):
            p, lo, hi, _, _ = rep.interval(
                months=part["month"].to_numpy(), num=(part["K0"] - part[arm]).to_numpy(), den=gap
            )
            rows.append(
                {
                    "set": set_name,
                    "arm": LABELS[arm],
                    "share": 100 * p,
                    "low": 100 * lo,
                    "high": 100 * hi,
                }
            )
    data = pl.DataFrame(rows)
    panels = [
        _dots(
            data=data.filter(pl.col("set") == s),
            x="share",
            y="arm",
            colour="arm",
            height=70,
            title="Share of the gap closed, % (larger is better)",
            rule=0,
        ).properties(title=s)
        for s in sets
    ]
    chart = alt.vconcat(*panels).properties(  # ty: ignore[invalid-argument-type]
        title=_title(
            text="The pair alarm helps before RO-NC has spoken and loses to it after",
            line="Share of the gap between no correction and the true series that each mask closes. Dot: estimate. Line: 95% interval from resampling whole months. Dashed rule: same as no correction.",
        )
    )
    _save(chart=chart, name="figure_2_row_sets")

    frame_events = det.load_events()
    even = frame_events.filter(pl.col("month") % 2 == 0)
    pts = []
    for fam, label in (
        ("RONC", "RO-NC"),
        ("NEW4", "2 h"),
        ("NEW8", "4 h"),
        ("NEW12", "6 h"),
    ):
        part = even.filter(pl.col("family") == fam)
        w = part["moved_mw"]
        pts.append(
            {
                "alarm": label,
                "lag": float(part.filter(pl.col("found"))["lag_hours"].median()),  # ty: ignore[invalid-argument-type]
                "recall": float((part["found"].cast(pl.Float64) * w).sum() / w.sum()),  # ty: ignore[unsupported-operator]
            }
        )
    pts_df = pl.DataFrame(pts)
    base = alt.Chart(pts_df)
    chart = alt.layer(
        base.mark_point(filled=True, size=120, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lag:Q",
                title="Median hours from the step to the alarm being known (smaller is better)",
            ),
            y=alt.Y("recall:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0.6, 1])),
        ),
        base.mark_text(align="left", dx=8, dy=-12).encode(x="lag:Q", y="recall:Q", text="alarm:N"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="Pair alarms (labels give the window) are known 3 to 9 hours after a step instead of 48, and find less of the moved MW",
            line="Injected transfers that start in even months, 611 transfers. Moved MW is the megawatts shifted, summed over transfers; each alarm's false-alarm rate on real data is matched to RO-NC's.",
        ),
    )
    _save(chart=chart, name="figure_3_speed_and_recall")


if __name__ == "__main__":
    main()
