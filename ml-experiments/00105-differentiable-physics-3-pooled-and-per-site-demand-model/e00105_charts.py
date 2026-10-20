"""Draw the figures of experiment 105 as SVG into `docs/ml-experiments/<id>-<slug>/`.

Only Primary substations appear, under their D-labels, so dates may be shown. The substation cards
come from the decision arm P's saved fit (`work/model_p.pt`), with 80% intervals from the
variational posterior. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 uv run --no-sync python ml-experiments/00105-*/e00105_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00105_common as common
import e00105_report as report
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
import torch
from studies.dp_calendar import BAND_NAMES, DAY_TYPES, HOLIDAY_PERIODS

FIGURE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
"""The page folder in this checkout; the shared data folders resolve to the main checkout."""
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
SITE_ORDER: Final[list[str]] = ["D5", "D10", "D11", "D14", "D16", "D9"]
Z_80: Final[float] = 1.2816
ARM_LABELS: Final[dict[str, str]] = {
    "p": "P pooled",
    "p_map": "P-map",
    "p_nonet": "P-nonet",
    "s": "S per-site",
    "p_all": "P-all",
    "s_all": "S-all",
    "p_onb4w": "P-onb4w",
    "s_4w": "S-4w",
    "x_4w": "X-4w",
    "p_short3m": "P-short3m",
    "s_3m": "S-3m",
    "x_3m": "X-3m",
    "x_2y": "X-2y",
    "x_2yamq": "X-2yamq",
    "g1": "G1",
    "g_all": "G-all",
    "g_all_d1": "G-all+D1",
    "climatology": "climatology",
    "manual_heuristic": "manual heuristic",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    wrapped = textwrap.wrap(subtitle, WRAP)
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / f"{name}.svg")
    print("wrote", name)


def _arms() -> dict[str, pl.DataFrame]:
    ids = report.site_of_id()
    out = {}
    for name in ARM_LABELS:
        folder = common.PART_2_DIR if name in report.PART_2_ARMS else common.DATA_DIR
        frame = report.load_arm(name=name, folder=folder, ids=ids)
        if frame is not None:
            out[name] = report.headline(frame)
    return out


def figure_headline() -> alt.TopLevelMixin:
    """The verdict contrasts C1 to C3 and the comparisons against XGBoost."""
    contrasts = json.loads((common.WORK_DIR / "report_contrasts.json").read_text())
    rows = []
    for label, key, kind in (
        ("C1 full history: P minus S", "C1 pooled vs per-site, all rows", "verdict"),
        ("C2 4 weeks: P-onb4w minus S-4w", "C2 five sites", "verdict"),
        ("C3 3 months: P-short3m minus S-3m", "C3 five sites", "verdict"),
        ("4 weeks: P-onb4w minus X-4w", "p_onb4w-x_4w", "exploratory"),
        ("3 months: P-short3m minus X-3m", "p_short3m-x_3m", "exploratory"),
        ("C4 full history: P minus X-2y", "C4 vs x_2y", "exploratory"),
        ("P-all minus S-all", "p_all-s_all", "exploratory"),
        ("P-map minus P", "p_map-p", "exploratory"),
        ("P-nonet minus P", "p_nonet-p", "exploratory"),
    ):
        if key in contrasts:
            result = contrasts[key]
            rows.append(
                {
                    "contrast": label,
                    "estimate": result["estimate"],
                    "low": result["low"],
                    "high": result["high"],
                    "kind": kind,
                    "sites": f"{result['better']} better, {result['worse']} worse",
                }
            )
    table = pl.DataFrame(rows)
    order = table["contrast"].to_list()
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(
                domain=["verdict", "exploratory"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title="Interval", orient="top", direction="horizontal"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    text = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=6, fontSize=10)
        .encode(y=alt.Y("contrast:N", sort=order), x="high:Q", text="sites:N")
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="#888888", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=WIDTH - 150,
        height=26 * len(order),
        title=_title(
            text="Pooling helps a new or short-history substation, and is worse at full history",
            subtitle=(
                "Dot: difference in mean quantile score over the five decision substations (the "
                "six scored for the XGBoost comparisons), 1 July 2025 to 30 June 2026, rows "
                "outside the switching events. Line: 98.33% interval (verdict contrasts) or 95% "
                "interval, from resampling whole calendar months. Gate failed: see the page."
            ),
        ),
    )


def figure_absolute() -> alt.TopLevelMixin:
    """Every arm's mean quantile score at each site."""
    arms = _arms()
    rows = [
        {"arm": ARM_LABELS[name], "site": site, "score": float(score)}
        for name, frame in arms.items()
        for site, score in frame.group_by("site").agg(pl.col("score").mean()).iter_rows()
        if site in SITE_ORDER
    ]
    table = pl.DataFrame(rows)
    order = [ARM_LABELS[name] for name in ARM_LABELS if name in arms]
    base = alt.Chart(table).encode(
        x=alt.X("site:N", sort=SITE_ORDER, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("arm:N", sort=order, title=None),
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[2.0, 8.0], range=["#FFFFFF", ocf.DATA_BLUE], clamp=True),
            legend=alt.Legend(title="Score (points)", orient="top", direction="horizontal"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text=alt.Text("score:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    return (heat + text).properties(
        width=WIDTH - 130,
        height=18 * len(order),
        title=_title(
            text="Quantile score of every arm at every site",
            subtitle=(
                "Twice the mean pinball loss over 11 quantile levels, in points of effective "
                "capacity (smaller is better), 1 July 2025 to 30 June 2026, on rows outside the "
                "switching events. The 4-week and 3-month arms use only that much history."
            ),
        ),
    )


def scored_mean(*, frame: pl.DataFrame) -> float:
    """Return the mean score over the six scored substations."""
    kept = frame.filter(pl.col("site").is_in(common.SCORED_SITES))
    return float(kept["score"].to_numpy().mean())


def figure_onboarding() -> alt.TopLevelMixin:
    """Score against weeks of history for the pooled onboarding arms, with reference marks."""
    ids = report.site_of_id()
    rows, marks = [], []
    for weeks in (0, 2, 4, 8):
        frame = report.headline(
            report.load_arm(name=f"p_onb{weeks}w", folder=common.DATA_DIR, ids=ids)  # ty: ignore[invalid-argument-type]
        )
        rows.append(
            {
                "weeks": weeks,
                "score": scored_mean(frame=frame),
            }
        )
    arms = _arms()
    for name, label in (("p", "P (full history)"), ("s", "S (full history)"), ("x_4w", "X-4w")):
        frame = arms[name].filter(pl.col("site").is_in(common.SCORED_SITES))
        mean_score: float = frame["score"].mean()  # ty: ignore[invalid-assignment]
        marks.append({"label": label, "score": mean_score})
    line = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(x=alt.X("weeks:Q", title="Weeks of the new substation's history"), y="score:Q")
    )
    rules = (
        alt.Chart(pl.DataFrame(marks))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3])
        .encode(
            y=alt.Y("score:Q", title="Quantile score (points)"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=[m["label"] for m in marks],
                    range=[ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
    )
    return (rules + line).properties(
        width=WIDTH - 60,
        height=240,
        title=_title(
            text="Four weeks of history put a new substation within 0.8 points of full history",
            subtitle=(
                "Orange: the pooled model fitted without the substation, then given its last "
                "0, 2, 4, or 8 weeks. Dashed lines: references scored on the same six "
                "substations and the same year."
            ),
        ),
    )


def load_pooled() -> dict[str, np.ndarray]:
    """Return the fitted P model's parameters as arrays."""
    state = torch.load(common.WORK_DIR / "model_p.pt", map_location="cpu")
    return {key: value.detach().numpy() for key, value in state.items()}


def softplus(x: np.ndarray) -> np.ndarray:
    """Return log(1 + exp(x))."""
    return np.log1p(np.exp(x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Return the softmax of `x` along `axis`."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def figure_shapes() -> alt.TopLevelMixin:
    """The shared daily shapes, and each substation's mix of them."""
    state = load_pooled()
    shapes = state["shapes"].reshape(-1, len(DAY_TYPES), 48)
    mix = softmax(state["site_logits"], axis=1)
    rows = [
        {"shape": f"shape {k + 1}", "day type": DAY_TYPES[d], "hour": h / 2.0, "value": float(v)}
        for k in range(shapes.shape[0])
        for d in range(2)
        for h, v in enumerate(shapes[k, d])
    ]
    shape_chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("hour:Q", title="Local hour of day", scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("value:Q", title="Shape (points)"),
            color=alt.Color(
                "shape:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            strokeDash=alt.StrokeDash("day type:N", legend=alt.Legend(title=None, orient="top")),
        )
        .properties(width=WIDTH - 60, height=170)
    )
    sites = [s for s in common.POOL if s in SITE_ORDER or s in common.MVA_POOL or s == "D18"]
    mix_rows = [
        {
            "site": sites[i],
            "shape": f"shape {k + 1}",
            "weight": float(mix[common.POOL.index(sites[i]), k]),
        }
        for i in range(len(sites))
        for k in range(mix.shape[1])
    ]
    mix_chart = (
        alt.Chart(pl.DataFrame(mix_rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("site:N", sort=list(common.POOL), title=None),
            y=alt.Y("weight:Q", title="Share of the substation's mix", stack="normalize"),
            color=alt.Color(
                "shape:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]),
                legend=None,
            ),
        )
        .properties(width=WIDTH - 60, height=140)
    )
    return alt.vconcat(shape_chart, mix_chart).properties(
        title=_title(
            text="Provisional: the shared daily shapes and each substation's mix of them",
            subtitle=(
                "Fitted to past weather on a screening run. Top: the shared shapes (weekday "
                "solid, Saturday dashed) of the pooled model P. Bottom: each substation's mix of "
                "the shapes; substations differ by their mix and by a small own deviation."
            ),
        )
    )


def capacity_sources() -> pl.DataFrame:
    """Return capacity per scored substation and season-year: pooled fit P and the convex model."""
    state = load_pooled()
    trained = state["trained"]
    capacity = np.cumsum(softplus(state["capacity_raw"]) * trained, axis=1)
    sd = np.exp(state["posterior_log_sd.capacity_raw"])
    first_year = 2019
    rows = []
    for site in common.SCORED_SITES:
        i = common.POOL.index(site)
        for offset in range(capacity.shape[1]):
            if trained[i, offset] > 0:
                value = float(capacity[i, offset])
                spread = float(
                    Z_80 * np.sqrt(((sd[i, : offset + 1]) ** 2 * trained[i, : offset + 1]).sum())
                )
                rows.append(
                    {
                        "site": site,
                        "season_year": first_year + offset,
                        "source": "pooled fit P",
                        "points": value,
                        "low": max(value - spread, 0.0),
                        "high": value + spread,
                    }
                )
    convex = pl.read_parquet(common.PART_2_DIR / "work" / "gam_capacities_scored.parquet").filter(
        pl.col("arm") == "g_all"
    )
    rows.extend(
        {
            "site": row["site"],
            "season_year": row["season_year"],
            "source": "convex model G-all",
            "points": row["capacity_points"],
            "low": row["capacity_points"],
            "high": row["capacity_points"],
        }
        for row in convex.iter_rows(named=True)
    )
    return pl.DataFrame(rows).filter(pl.col("site").is_in(common.SCORED_SITES))


def figure_cards_capacity() -> alt.TopLevelMixin:
    """Substation cards, panel 1: capacity per season-year, pooled fit against convex model."""
    table = capacity_sources()
    base = alt.Chart(table).encode(
        x=alt.X("season_year:O", title="Season-year (July to June)"),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(
                domain=["pooled fit P", "convex model G-all"],
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    band = base.mark_errorbar().encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("low:Q", title="PV capacity (points)"), y2="high:Q"
    )
    points = base.mark_point(filled=True).encode(y="points:Q")  # ty: ignore[unresolved-attribute]
    return (
        (band + points)
        .properties(width=170, height=120)
        .facet(facet=alt.Facet("site:N", sort=SITE_ORDER, title=None), columns=3)
        .properties(
            title=_title(
                text="Provisional: inferred PV capacity per substation card",
                subtitle=(
                    "Fitted to past weather on a screening run. Orange with 80% posterior "
                    "intervals: the pooled fit P. Blue: Part 2's convex model. The pooled fit's "
                    "capacity follows its start (the loss has a minimum near the convex "
                    "capacity, identified to about 15%) and is not a measurement."
                ),
            )
        )
    )


def figure_cards_profiles() -> alt.TopLevelMixin:
    """Substation cards, panel 2: the daily profile per day type, and the heating response."""
    state = load_pooled()
    mix = softmax(state["site_logits"], axis=1)
    shapes = state["shapes"]
    deviation = state["deviation"]
    profile_rows, heating_rows = [], []
    inertia = softmax(state["inertia_logits"], axis=0)
    del inertia
    temperature = np.linspace(-2, 20, 45)
    for site in common.SCORED_SITES:
        i = common.POOL.index(site)
        profile = (mix[i] @ shapes + deviation[i]).reshape(len(DAY_TYPES), 48)
        for d in range(3):
            profile_rows.extend(
                {
                    "site": site,
                    "day type": DAY_TYPES[d],
                    "hour": h / 2.0,
                    "value": float(profile[d, h]),
                }
                for h in range(48)
            )
        slope = softplus(state["slope_raw"][i]).mean()
        threshold = float(state["threshold"][i])
        heating_rows.extend(
            {
                "site": site,
                "temperature": float(t),
                "points": float(slope * max(threshold - t, 0) / 10.0),
                "threshold": threshold,
            }
            for t in temperature
        )
    profiles = (
        alt.Chart(pl.DataFrame(profile_rows))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("hour:Q", title="Local hour of day", scale=alt.Scale(domain=[0, 24])),
            y=alt.Y("value:Q", title="Profile minus level (points)"),
            color=alt.Color(
                "day type:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(width=170, height=110)
        .facet(facet=alt.Facet("site:N", sort=SITE_ORDER, title=None), columns=3)
    )
    heating = (
        alt.Chart(pl.DataFrame(heating_rows))  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("temperature:Q", title="Mixed smoothed ERA5 temperature (°C)"),
            y=alt.Y("points:Q", title="Heating demand (points)"),
        )
        .properties(width=170, height=110)
        .facet(facet=alt.Facet("site:N", sort=SITE_ORDER, title=None), columns=3)
    )
    return alt.vconcat(profiles, heating).properties(
        title=_title(
            text="Provisional: daily profile and heating response of each substation",
            subtitle=(
                "Fitted to past weather on a screening run, pooled fit P. Top: the daily profile "
                "(shared shapes mixed, plus the substation's own deviation) for weekdays, "
                "Saturdays, and Sundays. Bottom: the heating response, the slope times the "
                "degrees below the substation's threshold."
            ),
        )
    )


def figure_shared_effects() -> alt.TopLevelMixin:
    """The holiday effects (by hour band) and the thermal-inertia mix, shared by all substations."""
    state = load_pooled()
    holiday = state["holiday"]
    hours = (6, 24, 72, 168)
    inertia = softmax(state["inertia_logits"], axis=0)
    rows = [
        {"period": HOLIDAY_PERIODS[p], "band": BAND_NAMES[b], "effect": float(holiday[p, b])}
        for p in range(len(HOLIDAY_PERIODS))
        for b in range(len(BAND_NAMES))
    ]
    effects = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None),
            y=alt.Y("period:N", sort=list(HOLIDAY_PERIODS), title=None),
            color=alt.Color(
                "effect:Q",
                scale=alt.Scale(scheme="redblue", domainMid=0),
                legend=alt.Legend(title="Points", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=170, height=14 * len(HOLIDAY_PERIODS))
    )
    mixture = (
        alt.Chart(pl.DataFrame({"hours": [f"{h} h" for h in hours], "weight": inertia.tolist()}))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "hours:N",
                sort=[f"{h} h" for h in hours],
                title="Smoothing window of the temperature",
            ),
            y=alt.Y("weight:Q", title="Weight in the mix"),
        )
        .properties(width=170, height=120)
    )
    lighting = state["lighting_coef"]
    del lighting
    return alt.hconcat(effects, mixture).properties(
        title=_title(
            text="Provisional: holiday effects and thermal-inertia mix shared by every substation",
            subtitle=(
                "Fitted to past weather on a screening run, pooled fit P. Left: the effect of "
                "each named holiday period in each hour band (points; the calendar terms are "
                "centred, so the effects are relative). Right: the weights of the four smoothed "
                "temperatures."
            ),
        )
    )


def figure_profile_check() -> alt.TopLevelMixin | None:
    """The capacity profile: loss and summer-midday error against the capacity multiplier."""
    path = common.WORK_DIR / "capacity_profile.json"
    if not path.exists():
        return None
    data = {k: v for k, v in json.loads(path.read_text()).items() if k != "free_fit"}
    rows = []
    best = min(entry["loss_nats"] for entry in data.values())
    for multiplier, entry in data.items():
        rows.append(
            {
                "multiplier": float(multiplier),
                "kind": "loss",
                "value": entry["loss_nats"] - best,
                "site": "all",
            }
        )
        for site, mae in entry["summer_midday_mae"].items():
            rows.append(
                {"multiplier": float(multiplier), "kind": "mae", "value": mae, "site": site}
            )
    table = pl.DataFrame(rows)
    loss = (
        alt.Chart(table.filter(pl.col("kind") == "loss"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE)
        .encode(
            x=alt.X("multiplier:Q", title="Capacity as a multiple of the convex model's"),
            y=alt.Y("value:Q", title="Training loss above the best (nats)"),
        )
        .properties(width=WIDTH - 60, height=130)
    )
    mae = (
        alt.Chart(table.filter(pl.col("kind") == "mae"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("multiplier:Q", title="Capacity as a multiple of the convex model's"),
            y=alt.Y("value:Q", title="Summer-midday mean absolute error (points)"),
            color=alt.Color(
                "site:N",
                sort=SITE_ORDER,
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 60, height=170)
    )
    return alt.vconcat(loss, mae).properties(
        title=_title(
            text="Provisional: how well the data pin the capacity",
            subtitle=(
                "Every scored substation's capacity held at a multiple of the convex model's "
                "capacity, everything else refitted (tuning cut, 2024-07-01). The loss has a "
                "minimum near the convex capacity, so the data pin the capacity to about 15%."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    for name, function in (
        ("figure_01_headline", figure_headline),
        ("figure_02_absolute", figure_absolute),
        ("figure_03_onboarding", figure_onboarding),
        ("figure_04_shapes", figure_shapes),
        ("figure_05_cards_capacity", figure_cards_capacity),
        ("figure_06_cards_profiles", figure_cards_profiles),
        ("figure_07_shared_effects", figure_shared_effects),
        ("figure_08_capacity_profile", figure_profile_check),
    ):
        chart = function()
        if chart is not None:
            _save(chart, name=name)


if __name__ == "__main__":
    main()
