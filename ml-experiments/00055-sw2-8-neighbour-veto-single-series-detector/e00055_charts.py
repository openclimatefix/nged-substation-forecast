"""Draw experiment 55's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00055-*/e00055_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00055_common import LABELLED_META_PATH, LABELLED_PATH, WORK_DIR
from e00055_detect import neighbour_check_for_cards
from e00055_report import (
    SEED,
    alarms_of,
    choose_veto,
    paired,
    results,
)

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00055-sw2-8-neighbour-veto-single-series-detector"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
D0_COLOUR: Final[str] = ocf.DATA_BLUE
D2_COLOUR: Final[str] = ocf.BRAND_ORANGE
HEADER: Final[alt.Header] = alt.Header(labelAngle=0, labelOrient="top", labelAlign="left")


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_operating_points() -> alt.TopLevelMixin:
    """Unexplained odd-month alarms against threshold, 24-hour window, by veto strength."""
    curves = pl.read_parquet(WORK_DIR / "curves.parquet").filter(pl.col("window") == 48)
    curves = curves.with_columns(
        arm=pl.when(pl.col("veto_z") == "none")
        .then(pl.lit("No veto (D0)"))
        .otherwise(pl.lit("Veto at ") + pl.col("veto_z") + pl.lit(" noise units"))
    )
    order = [
        "No veto (D0)",
        "Veto at 1.5 noise units",
        "Veto at 2 noise units",
        "Veto at 3 noise units",
    ]
    colours = [D0_COLOUR, ocf.DATA_PURPLE, D2_COLOUR, ocf.DATA_GREEN]
    lines = (
        alt.Chart(curves)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X("threshold:Q", title="Threshold (noise units; alarm when the step exceeds it)"),
            y=alt.Y("odd_unexplained:Q", title="Unexplained alarms, odd months"),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=order, range=colours),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
    )
    target = alt.Chart(pl.DataFrame({"y": [80]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return (lines + target).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="The veto lets the 24-hour threshold fall from 4.00 to 3.70 to 4.00",
            lines=[
                (
                    "Each line counts the 24-hour alarms that no logged closure explains, in odd "
                    "months only. The dashed line is the count of D0 at experiment 43's threshold "
                    "(80). "
                    "A stronger veto (smaller noise units) removes more alarms."
                )
            ],
        ),
    )


def figure_veto_rates() -> alt.TopLevelMixin:
    """Share of alarms vetoed, by class of alarm, 24-hour and 4-hour windows."""
    veto_z = choose_veto(out=WORK_DIR)
    rows = []
    for window in (8, 48):
        table = alarms_of(out=WORK_DIR, veto_z=veto_z).filter(pl.col("setting") == f"D2_{window}")
        odd = pl.col("month") % 2 == 1
        for label, mask in (
            ("Unexplained, odd months", ~pl.col("logged_matched") & odd),
            ("Unexplained, all months", ~pl.col("logged_matched")),
            ("Explained by a logged closure", pl.col("logged_matched")),
        ):
            part = table.filter(mask)
            rows.append(
                {
                    "window": "24-hour window" if window == 48 else "4-hour window",
                    "class": label,
                    "share": part.filter(pl.col("tier") == "vetoed").height / part.height,
                }
            )
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("class:N", title=None, sort=None),
            x=alt.X("share:Q", title="Share of alarms vetoed", axis=alt.Axis(format="%")),
            color=alt.Color(
                "class:N",
                legend=None,
                scale=alt.Scale(
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE],
                ),
            ),
            row=alt.Row("window:N", title=None, header=HEADER),
        )
        .properties(
            width=WIDTH - 140,
            height=70,
            title=_title(
                text="The veto removes unexplained alarms only a little more often",
                lines=[
                    (
                        "Share of the real alarms at the matched threshold that the veto removes. "
                        "A useful veto would remove unexplained alarms at least twice as often as "
                        "alarms that a logged closure explains."
                    )
                ],
            ),
        )
    )


def figure_contrasts() -> alt.TopLevelMixin:
    """C1 to C4 with their intervals."""
    veto_z = choose_veto(out=WORK_DIR)
    data = results(out=WORK_DIR, veto_z=veto_z)
    base = data.filter(pl.col("headline"), pl.col("even"), pl.col("set") != "holiday")
    specs = {
        "C1: 24-hour window, all transfers": ("48", None),
        "C2: 4-hour window, all transfers": ("8", None),
        "C3: 24-hour, with a metered receiver": ("48", pl.col("metered_receivers") > 0),
        "C4: 24-hour, no metered receiver": ("48", pl.col("metered_receivers") == 0),
    }
    rows = []
    for name, (window, mask) in specs.items():
        part = base if mask is None else base.filter(mask)
        mean, low, high = paired(
            new=part.filter(pl.col("setting") == f"D2_{window}"),
            reference=part.filter(pl.col("setting") == f"D0M_{window}"),
            seed=SEED,
        )
        rows.append({"contrast": name, "mean": mean, "low": low, "high": high})
    frame = pl.DataFrame(rows)
    y = alt.Y("contrast:N", title=None, sort=None, axis=alt.Axis(labelLimit=300))
    rule = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    bar = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_rule(color=D2_COLOUR, strokeWidth=3)
        .encode(
            y=y, x=alt.X("low:Q", title="Share found by D2 minus share found by D0M"), x2="high:Q"
        )
    )
    dot = alt.Chart(frame).mark_point(color=D2_COLOUR, filled=True, size=70).encode(y=y, x="mean:Q")  # ty: ignore[unresolved-attribute]
    return (rule + bar + dot).properties(
        width=WIDTH - 300,
        height=130,
        title=_title(
            text="The veto's gain over the same detector without it is at most 2 points",
            lines=[
                (
                    "Injected transfers of 25% to 40% of the receiving load lasting over 1 day, "
                    "starting in even months. Dot: mean paired difference. Line: 98.75% interval "
                    "from resampling whole calendar months. Dashed line: no difference. Larger is "
                    "better for D2."
                )
            ],
        ),
    )


def figure_found_by_set() -> alt.TopLevelMixin:
    """Share found by D0M and D2 on each set of transfers, all months."""
    veto_z = choose_veto(out=WORK_DIR)
    data = results(out=WORK_DIR, veto_z=veto_z).filter(
        pl.col("headline"), pl.col("setting").is_in(["D0M_48", "D2_48"])
    )
    names = {
        "original": "Experiment 43's 160",
        "extra": "New 1,000",
        "holiday": "300 near a holiday",
    }
    frame = (
        data.group_by("set", "setting")
        .agg(share=pl.col("found").mean(), n=pl.len())
        .with_columns(
            set=pl.col("set").replace_strict(names),
            detector=pl.col("setting").replace_strict(
                {"D0M_48": "D0 (no veto)", "D2_48": "D2 (veto)"}
            ),
        )
    )
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("set:N", title=None, sort=list(names.values())),
            yOffset="detector:N",
            x=alt.X("share:Q", title="Share of transfers found", axis=alt.Axis(format="%")),
            color=alt.Color(
                "detector:N",
                title=None,
                scale=alt.Scale(domain=["D0 (no veto)", "D2 (veto)"], range=[D0_COLOUR, D2_COLOUR]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 140,
            height=150,
            title=_title(
                text="Near bank holidays the veto finds slightly fewer transfers",
                lines=[
                    (
                        "24-hour window, injected transfers of 25% to 40% of the receiving load "
                        "lasting over 1 day, all months. Holiday set: onset within 1 day of a "
                        "bank holiday outside Christmas and New Year."
                    )
                ],
            ),
        )
    )


def figure_tiers() -> alt.TopLevelMixin:
    """Logged-explained share of real alarms and labelled-visible rate, by tier."""
    veto_z = choose_veto(out=WORK_DIR)
    table = alarms_of(out=WORK_DIR, veto_z=veto_z).filter(pl.col("setting") == "D2_48")
    first = (
        table.group_by("tier")
        .agg(share=pl.col("logged_matched").mean(), n=pl.len())
        .with_columns(measure=pl.lit("Explained by a logged closure (all real alarms)"))
    )
    meta = json.loads(LABELLED_META_PATH.read_text())
    labelled = pl.read_parquet(LABELLED_PATH).filter(pl.col("kind") == "alarm")
    cards = pl.DataFrame(
        [
            {"card_id": c, "detector": m["detector"], "stamp": m["alarm_stamp"]}
            for c, m in meta.items()
            if m["kind"] == "alarm"
        ]
    ).join(labelled.select("card_id", series="side_a", visibility="visibility"), on="card_id")
    tiers = neighbour_check_for_cards(cards=cards, veto_z=veto_z)
    second = (
        tiers.group_by("tier")
        .agg(
            share=pl.col("visibility").is_in(["conserved", "one_sided"]).mean(),
            n=pl.len(),
        )
        .with_columns(measure=pl.lit("Labelled a visible transfer (148 sampled alarms)"))
    )
    frame = pl.concat([first, second.select(first.columns)])
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("tier:N", title=None, sort=["confirmed", "unconfirmed", "vetoed"]),
            x=alt.X("share:Q", title="Share of the tier's alarms", axis=alt.Axis(format="%")),
            color=alt.Color(
                "tier:N",
                legend=None,
                scale=alt.Scale(
                    domain=["confirmed", "unconfirmed", "vetoed"],
                    range=[D2_COLOUR, D0_COLOUR, ocf.DATA_PURPLE],
                ),
            ),
            row=alt.Row("measure:N", title=None, header=HEADER),
        )
        .properties(
            width=WIDTH - 140,
            height=90,
            title=_title(
                text="Confirmed alarms match logged closures more often, but not visible transfers",
                lines=[
                    (
                        "24-hour window. Confirmed: a neighbour stepped the opposite way and the "
                        "two steps balance. Vetoed: neighbours stepped the same way. Few sampled "
                        "alarms are visible transfers (9 of 148), so the second panel is coarse."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write every figure."""
    _save(figure_operating_points(), name="figure_1_operating_points.svg")
    _save(figure_veto_rates(), name="figure_2_veto_rates.svg")
    _save(figure_contrasts(), name="figure_3_contrasts.svg")
    _save(figure_found_by_set(), name="figure_4_found_by_set.svg")
    _save(figure_tiers(), name="figure_5_tiers.svg")


if __name__ == "__main__":
    main()
