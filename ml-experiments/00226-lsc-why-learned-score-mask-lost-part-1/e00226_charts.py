# ruff: noqa: E501, PERF401
"""Draw experiment 226's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python e00226_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00226_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00226-lsc-why-learned-score-mask-lost-part-1"
)
WIDTH: Final[int] = 520
NAMES: Final[dict[str, str]] = {
    "PAIR_NEW": "Pair-alarm mask",
    "LEARNED_R": "Learned-score mask",
    "U1": "U1: pair + learned (union)",
    "U2": "U2: learned, balance-gated",
    "U3": "U3: learned, 148 end rule",
    "U4": "U4 (post hoc): pair onset, learned end",
    "L_BOTH": "Learned alarms a pair alarm also raises",
    "L_ONLY": "Learned-only alarms",
    "L_ONLY_NOBAL": "Learned-only, balanced-opposite removed",
}


SHORT: Final[dict[str, str]] = {
    "LEARNED_R": "learned score",
    "U1": "U1",
    "U2": "U2",
    "U3": "U3",
    "U4": "U4 (post hoc)",
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _intervals(
    *,
    data: pl.DataFrame,
    title: alt.TitleParams,
    x_title: str,
    domain: list[float],
    limits: list[float],
    height: int,
) -> alt.TopLevelMixin:
    return alt.layer(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("low:Q", title=x_title, scale=alt.Scale(domain=domain)),
            x2="high:Q",
            y=alt.Y("row:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "arm:N",
                legend=None,
                scale=alt.Scale(
                    domain=list(NAMES.values()),
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_SKY,
                        ocf.DATA_AMBER,
                        ocf.DATA_DEEP_TEAL,
                        ocf.DATA_MAGENTA,
                        ocf.DATA_BURNT_ORANGE,
                    ],
                ),
            ),
        ),
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(x="mean:Q", y=alt.Y("row:N", sort=None), color=alt.Color("arm:N", legend=None)),
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(pl.DataFrame({"v": limits}))
        .mark_rule(strokeDash=[2, 2], color="#444")
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=height, title=title)


def main() -> None:
    """Draw the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((WORK_DIR / "report.json").read_text())
    extra = json.loads((WORK_DIR / "extra.json").read_text())
    rows = []
    for scope, label in (
        ("shared", "7 load shifts"),
        ("stack_only", "11 load shifts"),
    ):
        for arm in ("LEARNED_R", "U1", "U2", "U3", "U4"):
            m, lo, hi = report[f"G1|{scope}|{arm}|PAIR_NEW"]
            rows.append(
                {
                    "row": f"{label}: {SHORT[arm]}",
                    "arm": NAMES[arm],
                    "mean": m,
                    "low": lo,
                    "high": hi,
                }
            )
    _intervals(
        data=pl.DataFrame(rows),
        title=_title(
            text="No combined mask beats the pair-alarm mask after real load shifts",
            line="Dot: estimate. Line: 95% interval from resampling whole load shifts. Score of the 72 hours after each logged load shift ends, minus the pair-alarm mask's score, in points of effective capacity (lower is better). Grey rule: no difference.",
        ),
        x_title="Mask minus pair-alarm mask (points; lower is better)",
        domain=[-4, 10],
        limits=[],
        height=480,
    ).save(OUT_DIR / "figure_1_gain_after_load_shifts.svg")
    rows = []
    for arm in (
        "PAIR_NEW",
        "LEARNED_R",
        "U1",
        "U2",
        "U3",
        "U4",
        "L_BOTH",
        "L_ONLY",
        "L_ONLY_NOBAL",
    ):
        m, lo, hi = report[f"G2|{arm}"]["pooled"]
        rows.append({"row": NAMES[arm], "arm": NAMES[arm], "mean": m, "low": lo, "high": hi})
    _intervals(
        data=pl.DataFrame(rows),
        title=_title(
            text="Learned alarms that a pair alarm also raises do no harm; the other alarms do",
            line="Dot: estimate. Line: 95% interval from resampling whole calendar months. Real rows outside logged load shifts, six Primary substations. Grey rule: no difference from no mask. Dotted rules: the pre-registered limit of +/-0.05.",
        ),
        x_title="Mask minus no mask (points of effective capacity; lower is better)",
        domain=[-0.1, 0.25],
        limits=[-0.05, 0.05],
        height=330,
    ).save(OUT_DIR / "figure_2_harm_ordinary_days.svg")
    dec = pl.DataFrame(extra["decomposition"]).with_columns(
        label=pl.col("site") + " " + pl.col("card").str.slice(0, 4)
    )
    alt.Chart(dec).mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "gap_R_minus_pair:Q",
            title="Contribution to the gap, learned-score mask minus pair-alarm mask (points)",
        ),
        y=alt.Y("label:N", title=None, sort="-x"),
    ).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="Two load shifts make up the whole 0.99 point gap",
            line="Each bar is one logged load shift at one Primary substation (substation and the first four characters of its card id), weighted by its share of the 3,675 scored rows. The bars add to +0.989.",
        ),
    ).save(OUT_DIR / "figure_3_gap_decomposition.svg")
    _mask_figure()


def _mask_figure() -> None:
    """Masked regions of three alarm sources against the logged interval, for the shared load shifts."""
    import e00226_alarms as al
    import e00226_decompose as dc
    import e00226_events as ev
    import e00226_fed as fed

    tables = dc.source_tables()
    folds = al.card_folds()
    pt = pl.read_parquet(WORK_DIR / "per_transfer.parquet").filter(
        pl.col("run_period").is_in(["scored"])
    )
    shared = pt.filter(pl.col("BG2").is_not_null()).select("site", "card")
    shared = shared.with_columns(card10=pl.col("card"))
    names = {"PAIR_NEW": "Pair alarm", "LEARNED_R": "Learned score", "U1": "U1 (union)"}
    bars, truth, marks = [], [], []
    logged = ev.logged_transfers()
    for site, card10 in shared.select("site", "card").iter_rows():
        series = fed.SERIES_OF_SITE[site]
        r = logged.filter(pl.col("card").str.starts_with(card10)).row(0, named=True)
        lo, hi = r["onset"], r["end"]
        label = f"{site} {card10[:4]} ({(hi - lo) / 2:.0f} h)"
        truth.append({"label": label, "start": 0.0, "end": (hi - lo) / 2})
        lim_lo, lim_hi = lo - 144, hi + 288
        for source, name in names.items():
            table = tables[source][folds.get(r["card"], 0)].filter(pl.col("series") == series)
            for a in table.filter((pl.col("end") > lim_lo) & (pl.col("onset") < lim_hi)).iter_rows(
                named=True
            ):
                bars.append(
                    {
                        "label": label,
                        "source": name,
                        "start": max(a["onset"], lim_lo) - lo,
                        "end": min(a["end"], lim_hi) - lo,
                    }
                )
                marks.append({"label": label, "source": name, "known": a["onset_known"] - lo})
    scale = 0.5
    bd = pl.DataFrame(bars).with_columns(pl.col("start") * scale, pl.col("end") * scale)
    td = pl.DataFrame(truth)
    md = pl.DataFrame(marks).with_columns(pl.col("known") * scale)
    colour = alt.Color(
        "source:N",
        sort=list(names.values()),
        scale=alt.Scale(
            domain=list(names.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    panels = []
    for label in td["label"].to_list():
        panels.append(
            alt.layer(
                alt.Chart(td.filter(pl.col("label") == label))
                .mark_rect(color="#d9d9d9", clip=True)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "start:Q",
                        title="Hours since the logged onset",
                        scale=alt.Scale(domain=[-72, 456]),
                    ),
                    x2="end:Q",
                ),
                alt.Chart(bd.filter(pl.col("label") == label))
                .mark_bar(height=7, clip=True)
                .encode(  # ty: ignore[unresolved-attribute]
                    x="start:Q",
                    x2="end:Q",
                    y=alt.Y("source:N", title=None, sort=list(names.values())),
                    color=colour,
                ),
                alt.Chart(md.filter(pl.col("label") == label))
                .mark_tick(color="#000", thickness=2, size=9, clip=True)
                .encode(  # ty: ignore[unresolved-attribute]
                    x="known:Q", y=alt.Y("source:N", sort=list(names.values()))
                ),
            ).properties(
                width=420, height=54, title=alt.TitleParams(label, anchor="start", fontSize=11)
            )
        )
    alt.vconcat(*panels).properties(
        title=_title(
            text="The learned score marks late or not at all where the pair alarm marks at once",
            line="Grey: the logged interval. Bars: hours the mask covers (from the alarm's step stamp to its estimated end), from 72 hours before the logged onset to 144 hours after the logged end. Black tick: when the alarm is known.",
        )
    ).resolve_scale(color="shared").save(OUT_DIR / "figure_4_masks_against_logged_interval.svg")


if __name__ == "__main__":
    main()
