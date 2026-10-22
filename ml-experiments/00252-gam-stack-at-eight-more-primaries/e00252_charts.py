# ruff: noqa: E501
"""Draw the figures of experiment 252 from `work/mva/scores_a0.json` and `scores_a1.json`.

uv run --no-sync python ml-experiments/00252-*/e00252_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00252_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.DATA_DIR / "work" / "mva" / "png"
WIDTH: Final[int] = 640
ARMS: Final[dict[str, tuple[str, str]]] = {
    "a0": ("A0: no solar term", ocf.DATA_BLUE),
    "a1": ("A1: solar capacity fitted (post hoc)", ocf.BRAND_ORANGE),
}
SLICE_LABELS: Final[dict[str, str]] = {
    "all": "All rows",
    "sunny": "Sunny daylight",
    "winter": "Winter (Nov to Mar)",
    "winter_night": "Winter nights",
    "cold_night": "Cold nights",
}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def forest(
    *, rows: list[dict], order: list[str], height: int, title: str, subtitle: str, name: str
) -> None:
    """Draw each arm's score minus BG2's per group, with its 95% interval."""
    domain = [v[0] for v in ARMS.values()]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[v[1] for v in ARMS.values()]),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Quantile score minus BG2 (points of capacity; below zero is better)"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=height,
        title=alt.TitleParams(
            text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110),
            anchor="start", fontSize=14, subtitleFontSize=11,
        ),
    )  # fmt: skip
    _save(chart, name=name)


def main() -> None:
    """Draw the per-site and per-slice figures."""
    scores = {
        arm: json.loads((common.DATA_DIR / "work" / "mva" / f"scores_{arm}.json").read_text())
        for arm in ARMS
    }
    sites = scores["a0"]["sites"]
    by_site = []
    for arm, (label, _) in ARMS.items():
        res = scores[arm]["contrast"]["all"]["d_bg2"]
        for site, (est, low, high) in zip(sites, res["site"], strict=True):
            by_site.append({"group": site, "arm": label, "estimate": est, "low": low, "high": high})
        est, low, high = res["pooled"]
        by_site.append({"group": "Pooled", "arm": label, "estimate": est, "low": low, "high": high})
    forest(
        rows=by_site, order=[*sites, "Pooled"], height=300,
        title="Without a solar term the stack is within 0.05 points of BG2 at only 2 of 8 Primaries",
        subtitle="Quantile score minus BG2 over leads 0.5 to 168 hours, scored year. Dot: estimate. Line: 95% interval from resampling whole months. Pooled is the mean of the eight sites.",
        name="figure_1_by_site",
    )  # fmt: skip
    by_slice = []
    for arm, (label, _) in ARMS.items():
        for key, text in SLICE_LABELS.items():
            est, low, high = scores[arm]["contrast"][key]["d_bg2"]["pooled"]
            by_slice.append(
                {"group": text, "arm": label, "estimate": est, "low": low, "high": high}
            )
    forest(
        rows=by_slice, order=list(SLICE_LABELS.values()), height=220,
        title="The gap to BG2 sits in the sunny daylight rows; winter nights are level or better",
        subtitle="Quantile score minus BG2 pooled over the eight Primaries, by group of rows. Dot: estimate. Line: 95% interval from resampling whole months.",
        name="figure_2_by_slice",
    )  # fmt: skip


if __name__ == "__main__":
    main()
