# ruff: noqa: E501, PERF401
"""Draw experiment 166's figures as SVG from `event_rows.parquet`, `balance.parquet`, and `report.json`.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00166-*/e00166_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00166_common import WORK_DIR
from e00166_events import BALANCE_GATE

HERE: Final[Path] = Path(__file__).resolve().parent
OUT_DIR: Final[Path] = (
    HERE.parents[1]
    / "docs"
    / "ml-experiments"
    / "00166-switching-gated-mask-and-soft-mask-while-alarmed"
)
WIDTH: Final[int] = 520
LABELS: Final[dict[str, str]] = {
    "MASK_UNION": "Hard mask, either alarm (experiment 165)",
    "MASK_NEW": "Hard mask, every pair alarm",
    "MASK_GATED": "Hard mask, pair alarm with balance score 0.7 or more",
    "SOFT_F1000": "Soft mask, noise x1000 (largest pre-registered)",
    "SOFT_F1000000": "Soft mask, noise x1,000,000 (post hoc)",
}
COLOURS: Final[dict[str, str]] = {
    "MASK_UNION": ocf.DATA_BLUE,
    "MASK_NEW": ocf.DATA_SKY,
    "MASK_GATED": ocf.DATA_GREEN,
    "SOFT_F1000": ocf.BRAND_ORANGE,
    "SOFT_F1000000": ocf.DATA_PURPLE,
}
SOFT_SWEEP: Final[tuple[str, ...]] = (
    "SOFT_F10",
    "SOFT_F100",
    "SOFT_F1000",
    "SOFT_F10000",
    "SOFT_F100000",
    "SOFT_F1000000",
)


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / f"{name}.svg")


def _scale(arms: tuple[str, ...]) -> alt.Scale:
    return alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms])


def _intervals(
    *,
    data: pl.DataFrame,
    y: str,
    arms: tuple[str, ...],
    x_title: str,
    domain: list[float],
    title: alt.TitleParams,
    height: int,
) -> alt.TopLevelMixin:
    colour = alt.Color(
        "label:N",
        legend=alt.Legend(orient="top", title=None, labelLimit=400, columns=1),
        scale=_scale(arms),
    )
    return alt.layer(
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X("low:Q", title=x_title, scale=alt.Scale(domain=domain)),
            x2="high:Q",
            y=alt.Y(f"{y}:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            yOffset="label:N",
            color=colour,
        ),
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x="mean:Q",
            y=alt.Y(f"{y}:N", sort=None),
            yOffset="label:N",
            color=alt.Color("label:N", legend=None, scale=_scale(arms)),
        ),
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=height, title=title)


def main() -> None:
    """Draw the figures."""
    ev = pl.read_parquet(WORK_DIR / "event_rows.parquet")
    report = json.loads((WORK_DIR / "report.json").read_text())
    arms = tuple(LABELS)
    # figure 1: the headline, change against the unmasked stack in each row group
    rows = []
    for name, key in (
        ("During a logged transfer (INSIDE)", "INSIDE|shared|{arm} minus K0"),
        ("72 hours after a transfer ends (AFTER)", "AFTER|shared|{arm} minus K0"),
    ):
        for arm in arms:
            m, lo, hi = report[key.format(arm=arm)]
            rows.append({"group": name, "label": LABELS[arm], "mean": m, "low": lo, "high": hi})
    _save(
        chart=_intervals(
            data=pl.DataFrame(rows),
            y="group",
            arms=arms,
            x_title="Score minus the unmasked stack, points (negative is better)",
            domain=[-11, 8],
            height=200,
            title=_title(
                text="The gated hard mask keeps the whole gain after a transfer; no soft mask keeps it without the extra error",
                line="Mean quantile score of each mask minus the unmasked stack, points of effective capacity. Dot: mean difference. Line: 95% interval, resampling logged transfers (INSIDE: 3, AFTER: 7, rows shared with best guess 2). Six Primary substations, July 2025 to June 2026.",
            ),
        ),
        name="figure_1_headline",
    )
    # figure 2: the trade-off, AFTER change against OUTSIDE change
    pts = []
    for arm, label in (
        ("MASK_UNION", "Hard mask, either alarm"),
        ("MASK_NEW", "Hard mask, every pair alarm"),
        ("MASK_GATED", "Hard mask, gated pair alarm"),
        *[
            (
                a,
                "Soft mask, "
                + {
                    "SOFT_F10": "x10",
                    "SOFT_F100": "x100",
                    "SOFT_F1000": "x1000",
                    "SOFT_F10000": "x10,000 (post hoc)",
                    "SOFT_F100000": "x100,000 (post hoc)",
                    "SOFT_F1000000": "x1,000,000 (post hoc)",
                }[a],
            )
            for a in SOFT_SWEEP
        ],
    ):
        pts.append(
            {
                "label": label,
                "outside": report[f"outside|all|{arm} minus K0"][0],
                "after": report[f"AFTER|shared|{arm} minus K0"][0],
                "kind": "Hard mask" if arm.startswith("MASK") else "Soft mask",
            }
        )
    d2 = pl.DataFrame(pts)
    offsets = {
        "Soft mask, x10": (7, -10),
        "Soft mask, x100": (7, 5),
        "Soft mask, x1000": (7, 6),
        "Hard mask, gated pair alarm": (7, 14),
        "Hard mask, every pair alarm": (7, 26),
        "Soft mask, x1,000,000 (post hoc)": (7, -9),
    }
    texts = []
    for r in d2.iter_rows(named=True):
        dx, dy = offsets.get(r["label"], (7, 0))
        texts.append(
            alt.Chart(pl.DataFrame([r]))  # ty: ignore[unresolved-attribute]
            .mark_text(align="left" if dx > 0 else "right", dx=dx, dy=dy, fontSize=10)
            .encode(x="outside:Q", y="after:Q", text="label:N")
        )
    chart2 = alt.layer(
        alt.Chart(d2)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "outside:Q",
                title="Extra error on ordinary days, points (smaller is better)",
                scale=alt.Scale(domain=[-0.02, 0.16]),
            ),
            y=alt.Y(
                "after:Q",
                title="Change in the 72 hours after a transfer, points (more negative is better)",
            ),
            color=alt.Color(
                "kind:N",
                legend=alt.Legend(orient="top", title=None),
                scale=alt.Scale(
                    domain=["Hard mask", "Soft mask"], range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE]
                ),
            ),
        ),
        *texts,
    ).properties(
        width=WIDTH,
        height=300,
        title=_title(
            text="A soft mask gains only as much as it harms: the two move together",
            line="Each dot is one arm. Horizontal: mean score minus the unmasked stack outside logged transfers. Vertical: the same on the 72 hours after a logged transfer ends (7 transfers shared with best guess 2). The unmasked stack is at the origin.",
        ),
    )
    _save(chart=chart2, name="figure_2_trade_off")
    # figure 3: outside logged transfers by season
    rows3 = []
    for season in ("winter", "other"):
        for arm in ("MASK_UNION", "MASK_NEW", "MASK_GATED", "SOFT_F1000", "SOFT_F1000000"):
            m, lo, hi = report[f"outside|{season}|{arm} minus K0"]
            rows3.append(
                {
                    "season": "Winter (November to March)"
                    if season == "winter"
                    else "Other months",
                    "label": LABELS[arm],
                    "mean": m,
                    "low": lo,
                    "high": hi,
                }
            )
    _save(
        chart=_intervals(
            data=pl.DataFrame(rows3),
            y="season",
            arms=arms,
            x_title="Score minus the unmasked stack, points (negative is better)",
            domain=[-0.3, 0.4],
            height=180,
            title=_title(
                text="Outside winter the gated mask adds about 0.013 points; the mask from either alarm adds 0.161",
                line="Outside logged transfers, 95% interval from resampling calendar months. Dot: mean difference. Dashed line: no difference.",
            ),
        ),
        name="figure_3_outside_by_season",
    )
    # figure 4: per transfer, AFTER
    after = ev.filter((pl.col("set") == "AFTER") & pl.col("shared"))
    per = (
        ev.filter(pl.col("set") == "AFTER")
        .group_by("card", "site")
        .agg(
            pl.len().alias("rows"),
            *[
                pl.col(a).mean()
                for a in ("K0", "MASK_UNION", "MASK_GATED", "SOFT_F1000", "SOFT_F1000000")
            ],
            pl.col("known_gated").first(),
            pl.col("end").first(),
        )
        .sort("K0", descending=True)
        .with_row_index("n", offset=1)
    )
    long = []
    for r in per.iter_rows(named=True):
        gated = "gated alarm" if 0 <= r["known_gated"] < r["end"] else "no gated alarm"
        name = f"Transfer {r['n']} ({gated}, {r['rows']} rows)"
        for arm in ("K0", "MASK_UNION", "MASK_GATED", "SOFT_F1000", "SOFT_F1000000"):
            long.append(
                {
                    "event": name,
                    "label": "Stack, no mask" if arm == "K0" else LABELS[arm],
                    "score": r[arm],
                }
            )
    names = {
        "Stack, no mask": ocf.BRAND_ORANGE,
        **{
            LABELS[a]: COLOURS[a]
            for a in ("MASK_UNION", "MASK_GATED", "SOFT_F1000", "SOFT_F1000000")
        },
    }
    names["Stack, no mask"] = "#444444"
    chart4 = (
        alt.Chart(pl.DataFrame(long))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "score:Q",
                title="Mean quantile score, points of effective capacity (smaller is better)",
            ),
            y=alt.Y("event:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "label:N",
                legend=alt.Legend(orient="top", columns=1, title=None, labelLimit=400),
                scale=alt.Scale(domain=list(names), range=list(names.values())),
            ),
        )
        .properties(
            width=WIDTH,
            height=26 * per.height + 30,
            title=_title(
                text="The gain sits in transfers that have a gated alarm; the gate keeps those and drops the rest",
                line=f"Each row is one logged transfer at one substation ({after['card'].n_unique()} transfers shared with best guess 2; all 13 pairs shown), scored on the 72 hours after it ends. Transfers are numbered from the largest unmasked score; substations are not named.",
            ),
        )
    )
    _save(chart=chart4, name="figure_4_per_transfer")
    # figure 5: balance scores of the pair alarms
    balance = pl.read_parquet(WORK_DIR / "balance.parquet")
    chart5 = alt.layer(
        alt.Chart(balance.select("balance_score"))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "balance_score:Q",
                bin=alt.Bin(step=0.05),
                title="Balance score (1 = the opposite steps cancel exactly)",
            ),
            y=alt.Y("count():Q", title="Pair alarms"),
        ),
        alt.Chart(pl.DataFrame({"v": [BALANCE_GATE]}))
        .mark_rule(strokeDash=[4, 3], color=ocf.BRAND_ORANGE, strokeWidth=2)
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text=f"{int((balance['balance_score'] >= BALANCE_GATE).sum())} of {balance.height} pair alarms pass the gate at a balance score of 0.7",
            line="All 282 real pair alarms of experiment 156 (window 4 hours, 3.75 noise units), January 2022 to June 2026. Dashed line: the gate.",
        ),
    )
    _save(chart=chart5, name="figure_5_balance_scores")


if __name__ == "__main__":
    main()
