"""Draw the figures of experiment 137 from `work/report.json` and the stored prediction sets.

OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00137-*/e00137_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00137_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.experiment_store import read_predictions

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
CONTRAST_KEYS: Final[dict[str, str]] = {"t_const": "Cconst"}
AXIS_LABELS: Final[str] = (
    "['21:30','00:30','03:30','06:30','09:00'][indexof([0,6,12,18,23], datum.value)]"
)
SLOT_LABELS: Final[list[str]] = [
    f"{((21 * 60 + 30 + 30 * s) // 60) % 24:02d}:{(30 + 30 * s) % 60:02d}" for s in range(24)
]
ARMS_SHOWN: Final[dict[str, tuple[str, str]]] = {
    "base": ("Base forecast", ocf.DATA_BLUE),
    "t_site": ("Night term (per substation)", ocf.BRAND_ORANGE),
    "oracle": ("Oracle term (scored year)", ocf.DATA_GREEN),
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_contrasts(*, report: dict) -> alt.TopLevelMixin:
    """Contrasts against the base on three row subsets, with 99% intervals."""
    labels = {
        "C1": "Term, per substation",
        "C2": "Term, shared by substations",
        "t_const": "Constant per half-hour only",
        "oracle": "Oracle (fitted on scored year)",
        "placebo": "Placebo (cold shuffled across nights)",
    }
    subsets = {
        "headline": "01:00-03:00 UTC, cold winter nights (365 rows)",
        "window_winter": "01:00-03:00 UTC, all winter nights (3,353 rows)",
    }
    rows = []
    for subset, subset_label in subsets.items():
        for key, label in labels.items():
            c = report["contrasts"][f"{CONTRAST_KEYS.get(key, key)}@{subset}"]
            rows.append({"arm": label, "subset": subset_label, "estimate": c["estimate"],
                         "low": c["lower"], "high": c["upper"]})  # fmt: skip
    c4 = report["contrasts"]["C4@all"]
    rows.append({"arm": labels["C1"], "subset": "Every row of the scored year (561,363 rows)",
                 "estimate": c4["estimate"], "low": c4["lower"], "high": c4["upper"]})  # fmt: skip
    table = pl.DataFrame(rows)
    colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.DATA_GREEN, GREY]
    panels = []
    for position, subset in enumerate(table["subset"].unique(maintain_order=True)):
        part = table.filter(pl.col("subset") == subset)
        shown = [v for v in labels.values() if v in set(part["arm"])]
        base = alt.Chart(part).encode(
            y=alt.Y("arm:N", sort=shown, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(
                "arm:N", sort=shown,
                scale=alt.Scale(domain=list(labels.values()), range=colours), legend=None,
            ),
        )  # fmt: skip
        last = position == table["subset"].n_unique() - 1
        x_title = "Quantile score minus base's (points; more negative is better)" if last else None
        marks = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=x_title), x2="high:Q"
        ) + base.mark_circle(size=70).encode(  # ty: ignore[unresolved-attribute]
            x="estimate:Q"
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        panels.append(
            (zero + marks).properties(
                width=WIDTH - 260,
                height=20 * len(shown) + 10,
                title=alt.TitleParams(subset, anchor="start", fontSize=12),
            )
        )
    return (
        alt.vconcat(*panels)
        .properties(
            title=_title(
                text="The night term lowers the day-ahead score most on cold winter nights",
                subtitle="Dot: estimate. Line: 99% interval from resampling calendar months of the"
                " scored year (July 2025 to June 2026). Leads 12 to 36 hours, except the last panel"
                " (every lead). Six Primary substations. Zero (grey rule): same as the base.",
            )
        )
        .resolve_scale(x="shared")
    )


def profile(*, name: str) -> pl.DataFrame:
    """Return the mean actual, mean forecast, and count by night slot on cold winter nights."""
    frame = read_predictions(
        experiment_id=common.EXPERIMENT_ID, name=name, row_set_id=common.ROW_SET_ID
    )
    return (
        frame.filter(
            (pl.col("slot") >= 0)
            & (pl.col("temp_c") < common.COLD_DAY_C)
            & pl.col("valid_time").dt.month().is_in(list(common.WINTER_MONTHS))
            & (pl.col("lead_hours") > 12)
            & (pl.col("lead_hours") <= 36)
        )
        .group_by("slot")
        .agg(actual=pl.col("actual").mean(), forecast=pl.col("mean").mean(), rows=pl.len())
        .sort("slot")
        .with_columns(arm=pl.lit(name))
    )


def figure_profile(*, profiles: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Mean demand and mean forecast by UTC half-hour on cold winter nights."""
    base = profiles["base"]
    actual = base.select(["slot", "actual"]).with_columns(
        series=pl.lit("Actual demand"), value=pl.col("actual")
    )
    lines = pl.concat(
        [actual.select(["slot", "series", "value"])]
        + [
            p.select(
                [
                    "slot",
                    pl.lit(ARMS_SHOWN[n][0]).alias("series"),
                    pl.col("forecast").alias("value"),
                ]
            )
            for n, p in profiles.items()
            if n in {"base", "t_site"}
        ]
    ).with_columns(
        label=pl.col("slot").replace_strict(dict(enumerate(SLOT_LABELS)), return_dtype=pl.String)
    )
    domain = ["Actual demand", ARMS_SHOWN["base"][0], ARMS_SHOWN["t_site"][0]]
    return (
        alt.Chart(lines)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, clip=True)
        .encode(
            x=alt.X(
                "slot:Q",
                title="Half-hour ending (UTC)",
                axis=alt.Axis(
                    values=[0, 6, 12, 18, 23],
                    labelExpr=AXIS_LABELS,
                ),
            ),
            y=alt.Y(
                "value:Q",
                title="Mean demand (points: % of 99th percentile)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series:N",
                sort=domain,
                scale=alt.Scale(domain=domain, range=[GREY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", title=None, columns=1),
            ),
        )
        .properties(
            width=WIDTH - 80,
            height=260,
            title=_title(
                text="On cold winter nights the base forecast runs below demand from 01:00 UTC",
                subtitle="Mean over the six Primary substations, leads 12 to 36 hours, scored year,"
                " nights with a forecast 24-hour mean temperature below 3 C (365 rows in the"
                " 01:00-03:00 window). Points: % of each substation's 99th percentile of demand.",
            ),
        )
    )


def figure_error(*, profiles: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Mean actual minus mean forecast by UTC half-hour for the base, the term, and the oracle."""
    rows = [
        p.select(
            [
                "slot",
                pl.lit(ARMS_SHOWN[n][0]).alias("series"),
                (pl.col("actual") - pl.col("forecast")).alias("error"),
            ]
        )
        for n, p in profiles.items()
        if n in ARMS_SHOWN
    ]
    table = pl.concat(rows)
    domain = [v[0] for v in ARMS_SHOWN.values()]
    colours = [v[1] for v in ARMS_SHOWN.values()]
    lines = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "slot:Q",
                title="Half-hour ending (UTC)",
                axis=alt.Axis(
                    values=[0, 6, 12, 18, 23],
                    labelExpr=AXIS_LABELS,
                ),
            ),
            y=alt.Y("error:Q", title="Actual minus forecast (points; zero is unbiased)"),
            color=alt.Color(
                "series:N",
                sort=domain,
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(orient="top", title=None, columns=1),
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return (zero + lines).properties(
        width=WIDTH - 80,
        height=240,
        title=_title(
            text="The term removes most of the forecast's miss on cold winter nights",
            subtitle="Mean of actual demand minus the mean forecast, six Primary substations,"
            " leads 12 to 36 hours, scored year, forecast 24-hour mean below 3 C. Positive:"
            " the forecast is too low. The oracle is the term fitted on the scored year itself.",
        ),
    )


def main() -> None:
    """Draw every figure and print the profile numbers the write-up quotes."""
    report = json.loads((common.WORK_DIR / "report.json").read_text())
    profiles = {n: profile(name=n) for n in ARMS_SHOWN}
    for name, p in profiles.items():
        print(name, p.select(["slot", "actual", "forecast", "rows"]).to_dicts()[:0])
        window = p.filter(pl.col("slot").is_between(7, 11))
        print(
            name,
            "mean error slots 7-11:",
            float(np.mean((window["actual"] - window["forecast"]).to_numpy())),
            "rows",
            int(window["rows"].sum()),
        )
    _save(figure_contrasts(report=report), name="figure_1_contrasts")
    _save(figure_profile(profiles=profiles), name="figure_2_profile")
    _save(figure_error(profiles=profiles), name="figure_3_error")


if __name__ == "__main__":
    main()
