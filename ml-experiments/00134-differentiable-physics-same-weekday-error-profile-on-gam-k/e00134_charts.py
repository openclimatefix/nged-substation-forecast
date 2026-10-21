"""Draw experiment 134's figures from `work/results.json` (written by `e00134_report.py`).

uv run --no-sync python ml-experiments/00134-*/e00134_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00134_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00134-differentiable-physics-same-weekday-error-profile-on-gam-k"
)
WIDTH: Final[int] = 640
PERIOD_COLOURS: Final[dict[str, str]] = {
    "replicate months": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}
PERIOD_KEYS: Final[dict[str, str]] = {"replicate": "replicate months", "scored": "scored year"}
WEEKDAYS: Final[tuple[str, ...]] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
BAND_ORDER: Final[tuple[str, ...]] = ("0-24 h", "D+1", "24-48 h", "48-144 h", "144-168 h")
SUBTITLE: Final[tuple[str, ...]] = (
    (
        "Dot: estimate. Line: 99% interval from resampling whole months. Rule at zero: same as "
        "the reference."
    ),
)


def rows(*, results: dict, labels: list[str], key: str, group: str = "planned") -> pl.DataFrame:
    """Collect the estimate and interval of one breakdown for several contrasts and both periods."""
    out = []
    for period, name in PERIOD_KEYS.items():
        for label in labels:
            entry = results[period][group][label].get(key)
            if entry is None:
                continue
            q = entry["qs"]
            out.append(
                {
                    "contrast": label,
                    "period": name,
                    "estimate": q["estimate"],
                    "lower": q["lower"],
                    "upper": q["upper"],
                }
            )
    return pl.DataFrame(out)


def forest(
    *, frame: pl.DataFrame, title: str, subtitle: list[str], order: list[str], height: int
) -> alt.Chart:
    """Draw a dot-and-interval chart with one row per contrast and both periods dodged."""
    base = alt.Chart(frame).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Quantile score difference (points of capacity; more negative is better)",
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=60, aria=False).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (
        (zero + bars + dots)
        .properties(
            width=WIDTH - 260, height=height, title=alt.TitleParams(title, subtitle=subtitle)
        )
        .configure_view(stroke=None)
    )


def save(*, chart: alt.Chart, name: str) -> None:
    """Write a chart as SVG beside the write-up."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")


def main() -> None:
    """Draw every figure."""
    results = json.loads((common.WORK_DIR / "results.json").read_text())
    planned = list(results["scored"]["planned"])
    save(
        chart=forest(
            frame=rows(results=results, labels=planned, key="all D+1"),
            title="Same-weekday profile: planned contrasts at day D+1",
            subtitle=[
                *SUBTITLE,
                "Five decision substations. Each contrast is the first arm minus the second.",
            ],
            order=planned,
            height=220,
        ),
        name="figure_01_headline",
    )
    c1 = "C1 RB - R"
    weekday = []
    for period, name in PERIOD_KEYS.items():
        for day in WEEKDAYS:
            entry = results[period]["planned"][c1].get(f"weekday {day}")
            if entry is None:
                continue
            q = entry["qs"]
            weekday.append(
                {
                    "contrast": day,
                    "period": name,
                    "estimate": q["estimate"],
                    "lower": q["lower"],
                    "upper": q["upper"],
                }
            )
    save(
        chart=forest(
            frame=pl.DataFrame(weekday),
            title="Gain of the weekday profile over the all-days profile, by weekday",
            subtitle=[
                *SUBTITLE,
                "RB minus R at day D+1; the target's weekday; holiday-like days left out.",
            ],
            order=list(WEEKDAYS),
            height=320,
        ),
        name="figure_02_by_weekday",
    )
    bands = []
    for period, name in PERIOD_KEYS.items():
        for label in (c1, "C5 RB-day-types - RB"):
            for band, entry in results[period]["planned"][label]["lead_bands"]["bands"].items():
                if entry is None or band not in BAND_ORDER:
                    continue
                q = entry["qs"]
                bands.append(
                    {
                        "contrast": f"{label}, {band}",
                        "period": name,
                        "estimate": q["estimate"],
                        "lower": q["lower"],
                        "upper": q["upper"],
                    }
                )
    order = [f"{label}, {b}" for label in (c1, "C5 RB-day-types - RB") for b in BAND_ORDER]
    save(
        chart=forest(
            frame=pl.DataFrame(bands),
            title="The gain is at day D+1 and fades at longer leads",
            subtitle=[*SUBTITLE, "Lead bands count from the 09:00 UTC forecast origin."],
            order=order,
            height=420,
        ),
        name="figure_03_by_lead",
    )
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())["median_recovery"]
    shown = {
        "R": "R",
        "Rh": "Rh",
        "RB-ref": "RB (reference setting)",
        "RB-ref-placebo": "RB placebo (reference)",
        "RB": "RB (tuned)",
        "RW": "RW (tuned)",
    }
    frame = pl.DataFrame({"arm": list(shown.values()), "share": [gate[k] for k in shown]})
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(
            y=alt.Y("arm:N", sort=list(shown.values()), title=None),
            x=alt.X(
                "share:Q",
                title="Share of the injected +3-point shift recovered (larger is better)",
            ),
        )
        .properties(
            width=WIDTH - 260,
            height=200,
            title=alt.TitleParams(
                "Known-answer gate: only the correct weekday recovers the injected shift",
                subtitle=[
                    "Median over five substations, at the third to eighth injected Saturday."
                ],
            ),
        )
        .configure_view(stroke=None)
    )
    save(chart=chart, name="figure_04_gate")


if __name__ == "__main__":
    main()
