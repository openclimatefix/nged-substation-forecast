# ruff: noqa: E501
"""Print every number the write-up quotes and draw its figures as SVG.

Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
No metered generator's time series is drawn. D-labels are substations.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00199-*/e00199_report.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00199_common import DATA_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00199-ps2b-engine-detector-v2-ramp-night-price"
)
ARM_LABELS: Final[dict[str, str]] = {
    "v1": "v1 (experiment 196)",
    "N": "Night evidence only",
    "v2 = N+S+P": "Pre-registered v2",
    "post hoc N+S(either edge)+P": "Post hoc: lenient edge test",
}
ARM_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
SITE_LABELS: Final[dict[str, str]] = {
    "D8_clean": "D8 engine-free history",
    "D8_2025-03_to_2025-12": "D8 2025-03 to 2025-12",
    "D8_2026-01_to_2026-06": "D8 2026-01 to 2026-06",
}


def title(text: str, sub: str) -> alt.TitleParams:
    """A wrapped title and subtitle."""
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(sub, 105),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_false_alarms() -> None:
    """Share of clean night half-hours flagged, by site and arm."""
    data = json.loads((DATA_DIR / "false_alarms.json").read_text())
    rows = []
    for site, r in data.items():
        for arm, label in ARM_LABELS.items():
            v = r[f"{arm} | night"]
            rows.append(
                {
                    "site": SITE_LABELS.get(site, site),
                    "arm": label,
                    "share": 100 * v["flagged_share"],
                    "days": v["false_active_days_per_month"],
                }
            )
            print(
                f"{site:24s} {label:34s} {100 * v['flagged_share']:5.1f}%  {v['false_active_days_per_month']:5.2f} d/mo"
            )
    frame = pl.DataFrame(rows)
    order = [SITE_LABELS.get(s, s) for s in data]
    base = alt.Chart(frame)
    points = base.mark_circle(size=70, opacity=0.95).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("site:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X("share:Q", title="Night half-hours flagged (%; smaller is better)"),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300, symbolLimit=0),
        ),
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [3.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    (points + rule).properties(
        width=520,
        height=300,
        title=title(
            "At night, experiment 196's detector is already quiet at five of six Primaries",
            "Share of night half-hours flagged as an engine running. Dashed line: the 3% bar. "
            "D3 to D16 and D8's engine-free history: even months, which the noise fit did not "
            "use. D8 from 2025: all months.",
        ),
    ).save(OUT_DIR / "figure_1_false_alarms_at_night.svg")


def figure_known_answers() -> None:
    """Recall at 1 unit and precision on K1 and K2 together, night rows."""
    data = json.loads((DATA_DIR / "known_answers.json").read_text())
    rows = []
    for arm, label in ARM_LABELS.items():
        v = data[f"K12 | {arm} | night"]
        for metric, name in (("recall_1_unit", "Recall at 1 unit"), ("precision", "Precision")):
            m = v[metric]
            rows.append(
                {"arm": label, "metric": name, "v": m["value"], "lo": m["lo"], "hi": m["hi"]}
            )
            print(
                f"K12 night {label:34s} {name:17s} {m['value']:.2f} ({m['lo']:.2f} to {m['hi']:.2f})"
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame)
    enc = {
        "y": alt.Y(
            "arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
        ),
        "color": alt.Color(
            "metric:N",
            scale=alt.Scale(
                domain=["Recall at 1 unit", "Precision"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        "yOffset": "metric:N",
    }
    bars = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lo:Q", title="Share (larger is better)", scale=alt.Scale(domain=[0, 1])),
        x2="hi:Q",
        **enc,
    )
    dots = base.mark_circle(size=70).encode(x="v:Q", **enc)  # ty: ignore[unresolved-attribute]
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.83]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    (bars + dots + rule).properties(
        width=480,
        height=220,
        title=title(
            "The pre-registered edge test removes about half of the injected engine",
            "Night half-hours in which G2's recorded output was injected into D8's engine-free "
            "history (K1 and K2 together). Dot: estimate. Line: 95% interval from resampling whole "
            "days. Dashed line: the 0.83 recall bar.",
        ),
    ).save(OUT_DIR / "figure_2_known_answers_at_night.svg")


def figure_edge_shapes() -> None:
    """Share of edges with a partial half-hour, by kind."""
    edges = pl.read_parquet(DATA_DIR / "shape_edges.parquet")
    table = (
        edges.group_by("kind")
        .agg(n=pl.len(), partial=pl.col("intermediate").mean())
        .with_columns(sharp=1 - pl.col("partial"))
        .sort("kind")
    )
    print(table)
    labels = {
        "engine G2": "Engine, recorded output",
        "engine G2 injected into D8 noise": "Engine, injected into D8's residual",
        "real transfer onset": "Real switching transfer, start",
        "real transfer end": "Real switching transfer, end",
        "unlabelled D8 clean-history step": "Unlabelled steps, D8 engine-free history",
    }
    frame = table.with_columns(label=pl.col("kind").replace(labels), pct=100 * pl.col("sharp"))
    alt.Chart(frame).mark_bar(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=300)),
        x=alt.X(
            "pct:Q",
            title="Edges that are one clean half-hour jump (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
    ).properties(
        width=420,
        height=200,
        title=title(
            "An engine's edges are sharp, but so are most unlabelled steps",
            "Share of edges of at least 1 MW where one half-hour carries at least 80% of the "
            "step. Switching transfers move in several half-hours; unlabelled steps in D8's "
            "engine-free history are mostly sharp too.",
        ),
    ).save(OUT_DIR / "figure_3_edge_shapes.svg")


def figure_step5() -> None:
    """Flagged days against matched unflagged days: difference in positive-spread share."""
    data = json.loads((DATA_DIR / "step5.json").read_text())
    rows = []
    for site, r in data.items():
        d = r["any"].get("diff_positive_points")
        if d:
            rows.append(
                {
                    "site": site,
                    "lo": d[0],
                    "mid": d[1],
                    "hi": d[2],
                    "n": r["any"]["flagged_days_matched"],
                }
            )
        print(
            site,
            json.dumps(
                {k: v for k, v in r.items() if k != "hour_profile_of_flagged_utc"}, default=float
            )[:600],
        )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame)
    y = alt.Y("site:N", sort=list(data), title=None, axis=alt.Axis(labelLimit=300))
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("lo:Q", title="Flagged days minus matched unflagged days (points)"), x2="hi:Q"
    )
    dots = base.mark_circle(size=70, color=ocf.DATA_BLUE).encode(y=y, x="mid:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#555555")
        .encode(x="x:Q")
    )
    (bars + dots + zero).properties(
        width=480,
        height=170,
        title=title(
            "Flagged days at D5, D10, and D11 do not follow the spark spread; D8's engine does",
            "Share of days with a positive daily spark spread: flagged days minus unflagged days "
            "matched on calendar month and weekend (positive: the detector flags more on "
            "profitable days). Dot: median. Line: 95% interval. Top row: D8 while the engine "
            "was visible (positive control).",
        ),
    ).save(OUT_DIR / "figure_4_flagged_days_follow_spread.svg")


def figure_hours() -> None:
    """Share of flagged half-hours by UTC hour."""
    data = json.loads((DATA_DIR / "step5.json").read_text())
    rows = []
    for site, r in data.items():
        prof = r["hour_profile_of_flagged_utc"]
        total = sum(prof)
        rows += [{"site": site, "hour": h, "share": 100 * c / total} for h, c in enumerate(prof)]
    alt.Chart(pl.DataFrame(rows)).mark_line(point=True).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("hour:Q", title="Hour of the day (UTC)", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
        y=alt.Y("share:Q", title="Share of the site's flagged half-hours (%)"),
        color=alt.Color(
            "site:N",
            sort=list(data),
            scale=alt.Scale(
                range=[
                    ocf.BRAND_ORANGE,
                    ocf.DATA_PURPLE,
                    ocf.DATA_BLUE,
                    ocf.DATA_SKY,
                    ocf.DATA_GREEN,
                ]
            ),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300, symbolLimit=0),
        ),
    ).properties(
        width=520,
        height=220,
        title=title(
            "D8's engine runs in the evening; the flags at D5 and D10 fall at midday",
            "Share of each site's flagged half-hours by hour of the UTC day, for all rows "
            "(experiment 196's detector). A flat line is a few long spells.",
        ),
    ).save(OUT_DIR / "figure_5_flagged_hours.svg")


def main() -> None:
    """Draw all figures."""
    figure_false_alarms()
    figure_known_answers()
    figure_edge_shapes()
    figure_step5()
    figure_hours()


if __name__ == "__main__":
    main()
