# ruff: noqa: E501
"""Draw experiment 203's figures as SVG from the stored report tables.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python e00203_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00203_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00203-switching-replay-learned-score-in-masked-forecast"
)
WIDTH: Final[int] = 520
NAMES: Final[dict[str, str]] = {
    "LEARNED_R": "Re-ranker",
    "LEARNED_F": "Logistic regression",
    "RONC_195": "RO-NC re-run",
    "PAIR_GATED": "Gated pair alarm",
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
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
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
    scopes = {
        "shared": "7 transfers with a best guess 2 score",
        "stack_only": "11 transfers, scored and replicate months",
        "all_runs": "13 transfers, every cached run",
    }
    rows = []
    for scope, label in scopes.items():
        for arm in ("LEARNED_R", "LEARNED_F", "RONC_195"):
            m, lo, hi = report[f"G1|{scope}|{arm}|PAIR_NEW"]
            rows.append(
                {
                    "row": f"{label}: {NAMES[arm]}",
                    "arm": NAMES[arm],
                    "mean": m,
                    "low": lo,
                    "high": hi,
                }
            )
    _intervals(
        data=pl.DataFrame(rows),
        title=_title(
            text="No learned-score mask is shown to beat the pair-alarm mask after real transfers",
            line="Dot: estimate. Line: 95% interval from resampling whole transfers. Score of the 72 hours after each logged transfer ends, minus the pair-alarm mask's score, in points of effective capacity (lower is better). Grey rule: no difference. Dotted rule: the 0.3 point gain bar.",
        ),
        x_title="Learned-score mask minus pair-alarm mask (points; lower is better)",
        domain=[-4, 8],
        limits=[-0.3],
        height=330,
    ).save(OUT_DIR / "figure_1_gain_after_transfers.svg")
    rows = []
    for arm in ("PAIR_GATED", "LEARNED_R", "LEARNED_F", "RONC_195"):
        for key, label in (("pooled", "27 months"), ("winter", "November to March")):
            m, lo, hi = report[f"G2|{arm}"][key]
            rows.append(
                {
                    "row": f"{label}: {NAMES[arm]}",
                    "arm": NAMES[arm],
                    "mean": m,
                    "low": lo,
                    "high": hi,
                }
            )
    _intervals(
        data=pl.DataFrame(rows),
        title=_title(
            text="The learned-score masks do no better than the pair-alarm mask on ordinary days",
            line="Dot: estimate. Line: 95% interval from resampling whole calendar months. Real rows outside logged transfers, six Primary substations. Grey rule: no difference from no mask. Dotted rules: the pre-registered limit of +/-0.05.",
        ),
        x_title="Mask minus no mask (points of effective capacity; lower is better)",
        domain=[-0.15, 0.6],
        limits=[-0.05, 0.05],
        height=250,
    ).save(OUT_DIR / "figure_2_harm_ordinary_days.svg")
    per = pl.read_parquet(WORK_DIR / "per_transfer.parquet").sort("site", "card")
    per = per.with_columns(
        label=pl.col("site") + " " + pl.col("run_period") + " " + pl.col("card").str.slice(0, 4)
    )
    long = per.unpivot(
        on=["K0", "PAIR_NEW", "LEARNED_R"],
        index=["label", "rows"],
        variable_name="arm",
        value_name="score",
    )
    names = {"K0": "No mask", "PAIR_NEW": "Pair-alarm mask", "LEARNED_R": "Re-ranker mask"}
    long = long.with_columns(pl.col("arm").replace(names))
    alt.Chart(long).mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "score:Q",
            title="Mean score of the 72 hours after the transfer ends (points; lower is better)",
            scale=alt.Scale(type="symlog"),
        ),
        y=alt.Y("label:N", title=None, sort=None),
        color=alt.Color(
            "arm:N",
            sort=list(names.values()),
            scale=alt.Scale(
                domain=list(names.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
            ),
            legend=alt.Legend(orient="top", title=None),
        ),
    ).properties(
        width=WIDTH,
        height=330,
        title=_title(
            text="One long transfer at D5 is masked only by the pair alarm",
            line="Each row is one logged transfer at one Primary substation (substation, period, first four characters of its card id). Score axis is a signed-log scale.",
        ),
    ).save(OUT_DIR / "figure_3_per_transfer.svg")


if __name__ == "__main__":
    main()
