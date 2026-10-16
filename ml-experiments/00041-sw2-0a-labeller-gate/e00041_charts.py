"""Draw experiment 00041's figures as SVG: a raw-data example first, then recall by band.

The series are Primary substations (named); no NGED generator is drawn. The SVGs are afterwards
optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00041-sw2-0a-labeller-gate/e00041_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00041_cards import (
    apply_injection,
    make_injection,
    make_placebo,
    real_operations,
    revert_injection,
)
from e00041_common import BIG_SET, TEST_SET, WORK_DIR
from e00041_diagnose import wilson
from e00041_panel import INDEX, excess, load_panel
from e00041_report import band
from e00041_score_sonnet import load

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / "00041-sw2-0a-labeller-gate"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
VIS = ("conserved", "one_sided")


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Excess power around one injected transfer and around one placebo window."""
    panel = load_panel()
    ops = real_operations()
    rng = np.random.default_rng(5)
    frames = []
    while True:
        made = make_injection(panel=panel, ops=ops, rng=rng, jitter=False)
        if made is None:
            continue
        spec, truth, inj = made
        if (
            (truth.moved_share_of_receiver or 0.0) >= 0.25
            and 24 <= (truth.end_stamp or 0) - (truth.onset_stamp or 0) + 1 <= 140
            and not truth.unmetered_receiver
            and len(truth.receivers or []) == 1
        ):
            break
    rows = [INDEX[spec.side_a]] + ([INDEX[spec.side_b]] if spec.side_b in INDEX else [])
    saved = panel.excluded[rows].copy()
    panel.excluded[rows, spec.first - 48 : spec.last + 49] = True
    apply_injection(panel=panel, inj=inj)
    for _tag, sid in (("A", spec.side_a), ("B", spec.side_b)):
        if sid not in INDEX:
            continue
        lo, hi = spec.first - 96, spec.last + 97
        e = excess(panel=panel, row=INDEX[sid], lo=lo, hi=hi)
        frames.append(
            pl.DataFrame(
                {
                    "hours": (np.arange(lo, hi) - spec.first) / 2,
                    "excess": e,
                    "series": f"Series {sid}",
                    "panel": "Injected transfer (known answer)",
                }
            )
        )
    revert_injection(panel=panel, inj=inj)
    panel.excluded[rows] = saved
    # placebo
    while True:
        m = make_placebo(panel=panel, ops=ops, rng=rng)
        if m is not None and 24 <= m[0].last - m[0].first + 1 <= 140:
            break
    pspec = m[0]
    prow = [INDEX[pspec.side_a]] + ([INDEX[pspec.side_b]] if pspec.side_b in INDEX else [])
    psaved = panel.excluded[prow].copy()
    panel.excluded[prow, pspec.first - 48 : pspec.last + 49] = True
    for sid in (pspec.side_a, pspec.side_b):
        if sid not in INDEX:
            continue
        lo, hi = pspec.first - 96, pspec.last + 97
        e = excess(panel=panel, row=INDEX[sid], lo=lo, hi=hi)
        frames.append(
            pl.DataFrame(
                {
                    "hours": (np.arange(lo, hi) - pspec.first) / 2,
                    "excess": e,
                    "series": f"Series {sid}",
                    "panel": "Placebo window (nothing happened)",
                }
            )
        )
    panel.excluded[prow] = psaved
    data = pl.concat(frames)
    pl.DataFrame(
        {
            "panel": ["Injected transfer (known answer)", "Placebo window (nothing happened)"],
            "start": [0.0, 0.0],
            "end": [(spec.last - spec.first + 1) / 2, (pspec.last - pspec.first + 1) / 2],
        }
    )

    def panel_chart(name: str, span_end: float, show_x: bool) -> alt.LayerChart | alt.FacetChart:
        sub = data.filter(pl.col("panel") == name)
        shade = (
            alt.Chart(pl.DataFrame({"start": [0.0], "end": [span_end]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color="#888888")
            .encode(x="start:Q", x2="end:Q")
        )
        zero = (
            alt.Chart(pl.DataFrame({"v": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#999999", strokeDash=[4, 3])
            .encode(y="v:Q")
        )
        line = (
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.3, clip=True)
            .encode(
                x=alt.X("hours:Q", title="Hours from the logged close" if show_x else None),
                y=alt.Y("excess:Q", title="Excess (MW)"),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=sorted(sub["series"].unique().to_list()),
                        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                    ),
                ),
            )
        )
        return alt.layer(shade, zero, line).properties(
            width=WIDTH - 130,
            height=150,
            title=alt.TitleParams(text=name, anchor="start", fontSize=12),
        )

    return (
        alt.vconcat(
            panel_chart(
                "Injected transfer (known answer)", float((spec.last - spec.first + 1) / 2), False
            ),
            panel_chart(
                "Placebo window (nothing happened)", float((pspec.last - pspec.first + 1) / 2), True
            ),
        )
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Moved load shows as two opposite steps; a placebo has no such pair",
                lines=[
                    (
                        "Shaded: the logged closure. Lines: half-hourly excess of the two "
                        "substations on the sheet, in MW, "
                        "where excess is observed power minus the median of the same weekday and "
                        "clock time over the 4 weeks before and after. "
                        "Top: load injected from one substation into the other. "
                        "Bottom: a real operation's sheet pair moved by whole weeks into a "
                        "quiet period."
                    )
                ],
            )
        )
    )


def figure_2() -> alt.TopLevelMixin:
    """Recall by band for the rules-only labeller and Sonnet, with Wilson intervals."""
    truths = {
        **json.loads((WORK_DIR / TEST_SET / "truths.json").read_text()),
        **json.loads((WORK_DIR / BIG_SET / "truths.json").read_text()),
    }
    labs = {
        "Rules only": {
            **json.loads((WORK_DIR / TEST_SET / "rules_labels.json").read_text()),
            **json.loads((WORK_DIR / BIG_SET / "rules_labels.json").read_text()),
        },
        "Sonnet": {
            **load(which=TEST_SET, folder="labels_sonnet1"),
            **load(which=BIG_SET, folder="labels_sonnet1"),
        },
    }
    rows = []
    for name, labels in labs.items():
        groups: dict[tuple[str, str], list[bool]] = {}
        for c, tr in truths.items():
            if tr["kind"] == "injected":
                groups.setdefault(band(tr), []).append(labels[c]["visibility"] in VIS)
        for (share, dur), v in groups.items():
            lo, hi = wilson(sum(v), len(v))
            rows.append(
                {
                    "labeller": name,
                    "band": f"{share} of receiver load, {dur}",
                    "share": share,
                    "recall": sum(v) / len(v),
                    "lo": lo,
                    "hi": hi,
                    "n": len(v),
                }
            )
    data = pl.DataFrame(rows)
    order = [
        f"{s} of receiver load, {d}"
        for s in ("under 10%", "10% to 25%", "25% or more")
        for d in ("2 to 6 h", "6 to 24 h", "over 1 d")
    ]
    y = alt.Y("band:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
    colour = alt.Color(
        "labeller:N",
        title=None,
        scale=alt.Scale(domain=["Rules only", "Sonnet"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
    )
    off = alt.YOffset(
        "labeller:N", scale=alt.Scale(domain=["Rules only", "Sonnet"], paddingOuter=0.0)
    )
    bars = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            y=y,
            yOffset=off,
            x=alt.X(
                "lo:Q",
                title="Share of injected transfers labelled visible (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            x2="hi:Q",
            color=colour,
        )
    )
    dots = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, opacity=1)
        .encode(
            y=y,
            yOffset=off,
            x="recall:Q",
            color=colour,
            shape=alt.Shape("labeller:N", legend=None),
            tooltip=["labeller", "band", "recall", "n"],
        )
    )
    gate = (
        alt.Chart(pl.DataFrame({"v": [0.8]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(x="v:Q")
    )
    return alt.layer(gate, bars, dots, data=data).properties(
        width=WIDTH - 280,
        height=300,
        title=_title(
            text="Both labellers see large, long transfers and miss small ones",
            lines=[
                (
                    "Dot: share of 210 injected transfers (150 test cards, 60 further cards "
                    "in the 25% or more, 6 h or more band) "
                    "labelled visible. Line: 95% Wilson interval. Dashed: the gate's bar of "
                    "0.80. Bands under 0.80 are don't-care bands in later scoring."
                )
            ],
        ),
    )


def main() -> None:
    """Write both figures as SVG."""
    OUT.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1_raw_example", figure_1()),
        ("figure_2_recall_by_band", figure_2()),
    ):
        chart.save(OUT / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
