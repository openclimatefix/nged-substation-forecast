# ruff: noqa: E501
"""Draw experiment 162's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00162-*/e00162_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00162_part1 as p1
import e00162_report1 as r1
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00162_common import WORK_DIR

HERE: Final[Path] = Path(__file__).resolve().parent
OUT_DIR: Final[Path] = (
    HERE.parents[1]
    / "docs"
    / "ml-experiments"
    / "00162-switching-shaped-transfers-donor-demand-and-pv-shapes"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80


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


def figure_profiles() -> None:
    """The donor's demand term and solar output over the day, in two months."""
    terms = np.load(WORK_DIR / "terms" / "terms_8.npz")
    stamps = np.arange(len(terms["demand"]))
    half_hour = stamps % 48
    month = (
        (np.datetime64("2022-01-01") + (stamps * 30).astype("timedelta64[m]"))
        .astype("datetime64[M]")
        .astype(int)
        % 12
    ) + 1
    rows = []
    for m, name in ((1, "January"), (6, "June")):
        for term, key, sign in (("Demand term", "demand", 1.0), ("Solar output", "solar", -1.0)):
            values = sign * terms[key]
            for h in range(48):
                sel = (month == m) & (half_hour == h) & np.isfinite(values)
                rows.append(
                    {"hour": h / 2, "mw": float(values[sel].mean()), "term": term, "month": name}
                )
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2.5, clip=True)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Hour of day (UTC)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("mw:Q", title="Mean term (MW)"),
            color=alt.Color(
                "term:N",
                title=None,
                scale=alt.Scale(
                    domain=["Demand term", "Solar output"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.StrokeDash("month:N", title=None, legend=alt.Legend(orient="top")),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="The demand term and the solar output of one Primary substation have different daily shapes",
                line="Mean of the convex model's two terms by half-hour, for one Primary substation (labelled D11), in January and June, 2022 to 2026. Solid and dashed lines are the two months.",
            ),
        )
    )
    _save(chart=chart, name="figure_1_profiles")


def figure_reductions() -> None:
    """Held-out reduction in squared error for placebos, controls, and the real intervals."""
    controls = r1.with_reductions(pl.read_parquet(r1.OUT / "controls.parquet"))
    real = r1.main_series(r1.with_reductions(pl.read_parquet(r1.OUT / "real.parquet")))
    detected = r1.main_series(r1.with_reductions(pl.read_parquet(r1.OUT / "detected.parquet")))
    rows = []
    labels = {
        "placebo": "Placebo: nothing injected",
        "demand_only": "Injected: demand-shaped",
        "solar_only": "Injected: solar-shaped",
        "both": "Injected: both",
    }
    for scenario, label in labels.items():
        v = controls.filter(pl.col("scenario") == scenario)["red_shape"].to_numpy()
        rows.append(
            {
                "label": label,
                "kind": "control",
                "median": float(np.median(v)),
                "low": float(np.percentile(v, 25)),
                "high": float(np.percentile(v, 75)),
            }
        )
    for label, frame in (
        ("Logged real transfers over 1 day", real),
        ("RO-NC detected intervals over 1 day", detected),
    ):
        v = frame["red_shape"].to_numpy()
        rows.append(
            {
                "label": label,
                "kind": "real",
                "median": float(np.median(v)),
                "low": float(np.percentile(v, 25)),
                "high": float(np.percentile(v, 75)),
            }
        )
    data = pl.DataFrame(rows)
    order = list(data["label"])
    base = alt.Chart(data)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    colour = alt.Color(
        "kind:N",
        legend=None,
        scale=alt.Scale(domain=["control", "real"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
    )
    chart = (
        base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="Held-out reduction in squared error against a constant (larger is better)",
            ),
            x2="high:Q",
            y=y,
            color=colour,
        )
        + base.mark_point(filled=True, size=80, opacity=1).encode(x="median:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x="x:Q")
    ).properties(
        width=WIDTH,
        height=190,
        title=_title(
            text="Real transfers fit a daily shape no better than a placebo does",
            line="Dot: median of the intervals. Line: 25th to 75th percentile. Reduction in squared error on held-out days when a daily shape (demand term and solar term) replaces a constant offset; the dashed line is no gain. Blue: known answers. Orange: recorded intervals.",
        ),
    )
    _save(chart=chart, name="figure_2_held_out_reduction")


def figure_identification() -> None:
    """Share of injected transfers whose demand or solar share is identified, by duration."""
    c = r1.with_reductions(pl.read_parquet(r1.OUT / "controls.parquet"))
    plac = c.filter(pl.col("scenario") == "placebo")
    rows = []
    for scenario, column, positive, label in (
        ("demand_only", "alpha", False, "Demand share, demand-shaped injection"),
        ("solar_only", "beta", True, "Solar share, solar-shaped injection"),
    ):
        for name, lo, hi in r1.DURATION_BANDS:
            band = (pl.col("days") >= lo) & (pl.col("days") < hi)
            null = plac.filter(band)[column]
            sub = c.filter((pl.col("scenario") == scenario) & band)
            cut = float(null.quantile(0.975)) if positive else float(null.quantile(0.025))  # ty: ignore[invalid-argument-type]
            ok = (sub[column] > cut) if positive else (sub[column] < cut)
            rows.append({"band": name, "share": float(ok.mean()), "series": label})  # ty: ignore[invalid-argument-type]
    data = pl.DataFrame(rows)
    order = [b[0] for b in r1.DURATION_BANDS]
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5)
        .encode(
            x=alt.X(
                "band:N",
                sort=order,
                title="Length of the injected transfer",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share identified", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", labelLimit=400, columns=1),
            ),
        )
        .properties(
            width=WIDTH,
            height=210,
            title=_title(
                text="A demand-shaped transfer is identified after about a week; a solar-shaped one is not",
                line="Share of injected transfers (about 25 to 75 per length band) whose fitted share lies beyond the 2.5th or 97.5th percentile of the same coefficient on placebos of that length. Moved: 15% of the donor's demand term, or 35% of its solar output.",
            ),
        )
    )
    _save(chart=chart, name="figure_3_identification")


ARM_LABELS: Final[dict[str, str]] = {
    "SHARE": "Share of net power (experiments 141, 156)",
    "DEMAND": "Demand-shaped",
    "SOLAR": "Solar-shaped",
    "MIX": "Half demand, half solar",
    "CONST": "Constant MW",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.DATA_SKY,
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
    "#444444",
]


def _forest(
    *, rows: list[dict], title: alt.TitleParams, x_title: str, height: int
) -> alt.TopLevelMixin:
    data = pl.DataFrame(rows)
    order = list(dict.fromkeys(data["label"]))
    base = alt.Chart(data)
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=340))
    colour = alt.Color(
        "arm:N",
        legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
        scale=alt.Scale(
            domain=[ARM_LABELS[a] for a in ("SHARE", "DEMAND", "SOLAR", "MIX")],
            range=ARM_COLOURS[:4],
        ),
    )
    return (
        base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=x_title),
            x2="high:Q",
            y=y,
            yOffset=alt.YOffset("arm:N", sort=list(ARM_LABELS.values())),
            color=colour,
        )
        + base.mark_point(filled=True, size=80, opacity=1).encode(  # ty: ignore[unresolved-attribute]
            x="point:Q",
            y=y,
            yOffset=alt.YOffset("arm:N", sort=list(ARM_LABELS.values())),
            color=colour,
        )
        + alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#444444", strokeDash=[4, 3])
        .encode(x="x:Q")
    ).properties(width=WIDTH, height=height, title=title)


def figure_contrasts() -> None:
    """Contrasts of each shaped arm against the constant-MW arm."""
    import json

    summary = json.loads((WORK_DIR / "part2_summary.json").read_text())
    share_rows, score_rows = [], []
    for arm in ("SHARE", "DEMAND", "SOLAR", "MIX"):
        for key, label in (
            ("P4 pair-alarm recall", "Pair alarm: share of moved MW found"),
            ("P5 RO-NC recall", "RO-NC: share of moved MW found"),
            ("P7", "End rule: share of the true interval covered"),
        ):
            point, _, _, lo, hi = summary[f"{arm}_{key}"]
            share_rows.append(
                {"label": label, "arm": ARM_LABELS[arm], "point": point, "low": lo, "high": hi}
            )
        point, _, _, lo, hi = summary[f"{arm}_P8/P9 MASK_UNION"]
        score_rows.append(
            {
                "label": "Mask forecast: mean quantile score",
                "arm": ARM_LABELS[arm],
                "point": point,
                "low": lo,
                "high": hi,
            }
        )
    top = _forest(
        rows=share_rows,
        x_title="Difference from constant MW (share; larger is better)",
        height=260,
        title=_title(
            text="A solar-shaped transfer is much harder for the detectors to find than a constant one",
            line="Each dot: the mean difference from a transfer of the same mean MW moved at a constant rate, on the 611 simulated transfers that start in even months. Line: interval from resampling whole months at a level corrected for nine contrasts. Dashed line: no difference.",
        ),
    )
    bottom = _forest(
        rows=score_rows,
        x_title="Difference from constant MW (points of capacity; smaller is better)",
        height=90,
        title=alt.TitleParams(text=""),
    )
    _save(chart=alt.vconcat(top, bottom).resolve_scale(color="shared"), name="figure_4_contrasts")  # ty: ignore[invalid-argument-type]


def figure_scores() -> None:
    """Mean quantile score of each forecast on each injection arm."""
    rows = []
    for arm in ("SHARE", "CONST", "DEMAND", "SOLAR", "MIX"):
        frame = pl.concat(
            [pl.read_parquet(p) for p in sorted((WORK_DIR / "scores" / arm).glob("*.parquet"))]
        )
        for name, label in (
            ("K0", "No correction"),
            ("GAM_ONLY", "Weather model only"),
            ("MASK_RONC", "Mask from RO-NC"),
            ("MASK_UNION", "Mask from either alarm"),
            ("CLEAN", "True series (ceiling)"),
        ):
            rows.append(
                {
                    "arm": ARM_LABELS[arm],
                    "forecast": label,
                    "score": float(frame.filter(pl.col("arm") == name)["qs"].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1)
        .encode(
            x=alt.X(
                "score:Q",
                title="Mean quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "arm:N",
                title=None,
                sort=[ARM_LABELS[a] for a in ("CONST", "SHARE", "DEMAND", "MIX", "SOLAR")],
                axis=alt.Axis(labelLimit=330),
            ),
            color=alt.Color(
                "forecast:N",
                title=None,
                legend=alt.Legend(orient="top", columns=2),
                scale=alt.Scale(
                    domain=[
                        "No correction",
                        "Weather model only",
                        "Mask from RO-NC",
                        "Mask from either alarm",
                        "True series (ceiling)",
                    ],
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_PURPLE,
                        ocf.DATA_SKY,
                        ocf.DATA_BLUE,
                        "#444444",
                    ],
                ),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="The mask forecast gains least on solar-shaped transfers",
                line="Mean quantile score of each forecast on forecast origins after a simulated transfer began, by how the transfer's power was shaped over the day. Scores are against the true series without the transfer.",
            ),
        )
    )
    _save(chart=chart, name="figure_5_scores")


def main() -> None:
    """Draw the part 1 figures."""
    _ = p1
    figure_profiles()
    figure_reductions()
    figure_identification()
    figure_contrasts()
    figure_scores()


if __name__ == "__main__":
    main()
