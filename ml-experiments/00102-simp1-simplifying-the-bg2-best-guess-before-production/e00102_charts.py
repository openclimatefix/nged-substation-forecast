"""Draw experiment 00102's figures (SIMP1, simplifying the BG2 best guess) as SVG.

Every title, axis title, and legend entry is in plain words and says which direction is better.
Generators appear as `Solar farm A` to `F` and `Wind farm A` to `C`; time series carry a count of
days (1 to 7) and never a calendar date. Primary substations are labelled `Primary A` to `P`.

Figures:

1. `figure_1_<class>_contrasts`: each simplification minus BG2, with a 95% interval, per class.
2. `figure_2_by_lead`: quantile score by forecast lead band, for BG2 and the simplest candidate.
3. `figure_3_midday_summer`: score of every Primary arm at midday in summer, at night, and in the
   other hours, and the offset contrast in each.
4. `figure_4_<class>_<week>`: example forecasts of every configuration in a summer, a winter, and
   a large-error week.
5. `figure_5_blend_members_<week>`: the five blend members, the blend, and the actual.
6. `figure_6_blend_weights`: the blend weight of each member in each month.

    uv run python ml-experiments/00102-*/e00102_charts.py [--no-svgo]
"""

import argparse
import datetime
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import altair as alt
import e00102_common as common
import e00102_score as score
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import wrapped

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCS_DIR: Final[Path] = (
    REPO_ROOT / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
CHARTS_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
SCORE_TITLE: Final[str] = "Quantile score (points of capacity; smaller is better)"
SCOPE: Final[str] = "292 daily forecast runs, July 2025 to June 2026."
INTERVAL_NOTE: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
CLASS_TITLES: Final[dict[str, str]] = {
    "Primary": "16 Primary substations",
    "BSP/GSP": "2 bulk and 2 grid supply points",
    "solar": "6 solar farms",
    "wind": "3 wind farms",
}
ARM_LABELS: Final[dict[str, str]] = {
    "p_s6_bg2": "BG2 (blend of 5, offset, correction)",
    "p_nl_weekly": "Simplest: weekly model, no lags, no offset",
    "p_nl_monthly": "Monthly model, no lags",
    "p_s4_weekly": "Weekly model with lags",
    "p_s5_blend_no_offset": "BG2 without the solar offset",
    "p_a5_spread_off": "BG2 without the spread factor",
    "p_b_bg2_st_uncorrected": "BG2 without the 14-day correction",
    "ref_manual_heuristic": "NGED manual heuristic",
    "p_s2_stack": "Stack (frozen model, 14-day correction)",
    "b_weekly": "BG2 (weekly model with lags)",
    "b_nl_weekly": "Simplest: weekly model, no lags",
    "b_stack": "Stack (frozen model, 14-day correction)",
    "g_ens_weekly": "BG2 (ENS-trained, monthly refit, with lags)",
    "g_nl_monthly": "Simplest: no lags",
    "g_a_spread_on": "BG2 with the fitted spread factor",
    "g_a_wind_margin": "BG2 with the power-curve margin",
}
for _member, _name in (
    ("w", "weekly"),
    ("m", "monthly"),
    ("an", "analogue"),
    ("pt", "pooled by type"),
    ("st", "frozen with correction"),
):
    ARM_LABELS[f"p_m_{_member}"] = f"Member alone: {_name}"
BAND_COLOUR: Final[str] = ocf.DATA_BLUE_LIGHT
WEEKS: Final[tuple[str, ...]] = ("summer", "winter", "large_error")
WEEK_TITLES: Final[dict[str, str]] = {
    "summer": "a sunny summer week",
    "winter": "a cold winter week",
    "large_error": "a week with large errors",
}


def label_for(*, arm: str) -> str:
    """Return a plain-words label for an arm."""
    return ARM_LABELS.get(arm, arm)


def contrast_figure(*, contrasts: pl.DataFrame, cls: str) -> alt.Chart:
    """Return the forest plot of a class's contrasts with BG2 (all rows)."""
    frame = (
        contrasts.filter(
            (pl.col("cls") == cls)
            & (pl.col("metric") == "qs")
            & (pl.col("scope") == "all")
            & ~pl.col("contrast").str.starts_with("C5")
        )
        .with_columns(
            label=pl.col("contrast"),
            safe=pl.col("upper") < score.THRESHOLDS[cls],
        )
        .to_pandas()
    )
    threshold = score.THRESHOLDS[cls]
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=330))
    )
    rule = (
        alt.Chart(
            pl.DataFrame({"x": [0.0, threshold], "kind": ["same as BG2", "safe limit"]}).to_pandas()
        )
        .mark_rule(strokeDash=[4, 3])
        .encode(
            x="x:Q",
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["same as BG2", "safe limit"], range=[ocf.GREY_5, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    interval = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")
    dot = base.mark_point(filled=True, color=ocf.DATA_BLUE, size=60, aria=False).encode(
        x=alt.X(
            "mean:Q",
            title=wrapped(
                text=f"Score minus BG2's (points of capacity; negative is better). {INTERVAL_NOTE}",
                width=95,
            ),
        )
    )
    return (rule + interval + dot).properties(
        width=PLOT_WIDTH - 130,
        height=22 * frame.shape[0] + 20,
        title=alt.TitleParams(
            text=wrapped(text=f"Each simplification against BG2: {CLASS_TITLES[cls]}", width=80),
            subtitle=wrapped(
                text=f"{SCOPE} Dashed orange line: the largest loss that still counts as safe.",
                width=95,
            ),
        ),
    )


def lead_figure(*, absolute: pl.DataFrame) -> alt.Chart:
    """Return the score by forecast lead band for BG2 and the simplest candidate, per class."""
    arms = {
        "p_s6_bg2": "Primary",
        "p_nl_weekly": "Primary",
        "p_s4_weekly": "Primary",
        "b_weekly": "BSP/GSP",
        "b_nl_weekly": "BSP/GSP",
        "g_ens_weekly": "solar",
        "g_nl_monthly": "solar",
    }
    bands = [f"lead {name}" for name, _, _ in common.LEAD_BANDS]
    panels = []
    for cls in ("Primary", "BSP/GSP", "solar", "wind"):
        names = [a for a, c in arms.items() if c == cls or (cls == "wind" and c == "solar")]
        frame = absolute.filter(
            (pl.col("cls") == cls)
            & (pl.col("metric") == "qs")
            & pl.col("arm").is_in([*names, "ref_manual_heuristic"])
            & pl.col("scope").is_in(bands)
        ).with_columns(
            band=pl.col("scope").str.replace("lead ", ""),
            label=pl.col("arm").map_elements(lambda a: label_for(arm=a), return_dtype=pl.String),
        )
        panels.append(
            alt.Chart(frame.to_pandas())
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "band:N", sort=[b.replace("lead ", "") for b in bands], title="Forecast lead"
                ),
                y=alt.Y("mean:Q", title=SCORE_TITLE),
                color=alt.Color("label:N", legend=alt.Legend(orient="top", title=None, columns=1)),
            )
            .properties(width=PLOT_WIDTH, height=150, title=CLASS_TITLES[cls])
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text="Score by forecast lead",
            subtitle=wrapped(text=f"{SCOPE} Lower is better.", width=95),
        )
    )


def midday_figure(*, absolute: pl.DataFrame, contrasts: pl.DataFrame) -> alt.Chart:
    """Return the Primary scores at midday in summer, at night, and in other hours."""
    scopes = ["midday_summer", "night", "other"]
    panels = []
    for suffix, title in (
        ("|offset5", "5 Primary substations with a solar offset"),
        ("", "All 16 Primary substations"),
    ):
        wanted = [s + suffix for s in scopes]
        frame = absolute.filter(
            (pl.col("cls") == "Primary")
            & (pl.col("metric") == "qs")
            & pl.col("scope").is_in(wanted)
        ).with_columns(
            label=pl.col("arm").map_elements(lambda a: label_for(arm=a), return_dtype=pl.String),
            hours=pl.col("scope").str.replace(r"\|offset5", "").str.replace("_", " "),
        )
        panels.append(
            alt.Chart(frame.to_pandas())
            .mark_point(filled=True, size=50, aria=False)
            .encode(
                x=alt.X("mean:Q", title=SCORE_TITLE, scale=alt.Scale(zero=False)),
                y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=300)),
                color=alt.Color("hours:N", legend=alt.Legend(orient="top", title=None)),
            )
            .properties(
                width=PLOT_WIDTH - 120, height=14 * frame["arm"].n_unique() + 30, title=title
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text="Primary substations: midday in summer, night, and other hours",
            subtitle=wrapped(
                text="Midday in summer is 10:00 to 14:00 UK time, April to September. Night is "
                "23:00 to 03:59. Lower is better. " + SCOPE,
                width=95,
            ),
        )
    )


def anonymous_label(*, site: str, cls: str, primaries: Sequence[str]) -> str:
    """Return the anonymous label of a series."""
    if cls == "Primary":
        return f"Primary {chr(ord('A') + list(primaries).index(site))}"
    if cls == "solar":
        return f"Solar farm {site}"
    if cls == "wind":
        return f"Wind farm {chr(ord('A') + int(site[1:]) - 1)}"
    return f"Supply point {site}"


def read_arm_series(*, arm: str, site: str) -> pl.DataFrame:
    """Return one series' rows of an arm, restricted to the next-day window of each run."""
    path = score.arm_path(name=arm)
    return (
        pl.scan_parquet(path)
        .filter(pl.col("site") == site)
        .filter((pl.col("lead_hours") >= 15) & (pl.col("lead_hours") < 39))
        .collect()
    )


def choose_weeks(
    *, bg2: pl.DataFrame, no_offset: pl.DataFrame | None
) -> dict[str, datetime.datetime]:
    """Choose the first day of the example weeks from BG2's rows of one series.

    A summer week is the week in June to August with the largest mean daytime actual-minus-median
    swing, which for a solar series is the week with the highest output; a winter week is the
    December to February week with the highest mean actual; and the large-error week has the
    highest mean absolute error of the median.
    """
    daily = (
        bg2.with_columns(
            day=pl.col("valid_time").dt.truncate("1d"),
            error=(pl.col("actual") - pl.col("q50")).abs(),
        )
        .group_by("day")
        .agg(actual=pl.col("actual").mean(), error=pl.col("error").mean())
        .sort("day")
    )
    weekly = (
        daily.with_columns(
            week=pl.col("day").dt.truncate("1w"),
        )
        .group_by("week")
        .agg(actual=pl.col("actual").mean(), error=pl.col("error").mean(), days=pl.len())
        .filter(pl.col("days") == 7)
    )
    month = weekly["week"].dt.month()
    summer = weekly.filter(month.is_in([6, 7, 8])).sort("actual", descending=True)
    winter = weekly.filter(month.is_in([12, 1, 2])).sort("actual", descending=True)
    large = weekly.sort("error", descending=True)
    return {
        "summer": summer["week"][0],
        "winter": winter["week"][0],
        "large_error": large["week"][0],
    }


def week_figure(
    *,
    cls: str,
    site: str,
    arms: Sequence[str],
    week_start: datetime.datetime,
    week: str,
    title_site: str,
) -> alt.Chart:
    """Return stacked panels, one per arm, with the actual and each arm's median and bands."""
    frames = []
    for arm in arms:
        frame = read_arm_series(arm=arm, site=site)
        frame = frame.filter(
            (pl.col("valid_time") >= week_start)
            & (pl.col("valid_time") < week_start + datetime.timedelta(days=7))
        )
        frames.append(
            frame.with_columns(
                label=pl.lit(label_for(arm=arm)),
                day=((pl.col("valid_time") - week_start).dt.total_minutes() / 1440.0 + 1.0),
            )
        )
    panels = []
    for frame, arm in zip(frames, arms, strict=True):
        data = frame.select("day", "q05", "q25", "q50", "q75", "q95", "actual").with_columns(
            pl.col("q05", "q25", "q50", "q75", "q95", "actual") * 100.0
        )
        base = alt.Chart(data.to_pandas()).encode(
            x=alt.X(
                "day:Q",
                scale=alt.Scale(domain=[1, 8]),
                axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7, 8], title="Day of the week (1 to 7)"),
            )
        )
        outer = base.mark_area(color=BAND_COLOUR, opacity=0.6, clip=True).encode(
            y=alt.Y("q05:Q", title="% of capacity"), y2="q95:Q"
        )
        inner = base.mark_area(color=ocf.DATA_BLUE, opacity=0.35, clip=True).encode(
            y="q25:Q", y2="q75:Q"
        )
        median = base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5, clip=True).encode(y="q50:Q")
        actual = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.5, clip=True).encode(
            y="actual:Q"
        )
        panels.append(
            (outer + inner + median + actual).properties(
                width=PLOT_WIDTH, height=95, title=label_for(arm=arm)
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text=f"{title_site}: {WEEK_TITLES[week]}",
            subtitle=wrapped(
                text="Orange line: actual. Blue line: median forecast. Dark band: 25% to 75% "
                "quantiles. Light band: 5% to 95% quantiles. Each day shows the 15 to 39 hour "
                "lead window of the run that started that morning.",
                width=95,
            ),
        )
    )


def members_figure(
    *, site: str, title_site: str, week_start: datetime.datetime, week: str
) -> alt.Chart:
    """Return the five members' medians, the blend median, and the actual over one week."""
    frames = []
    arms = ["p_m_w", "p_m_m", "p_m_an", "p_m_pt", "p_m_st", "p_s6_bg2"]
    names = {
        "p_m_w": "Weekly",
        "p_m_m": "Monthly",
        "p_m_an": "Analogue",
        "p_m_pt": "Pooled by type",
        "p_m_st": "Frozen with correction",
        "p_s6_bg2": "Blend (BG2)",
    }
    actual = None
    for arm in arms:
        frame = read_arm_series(arm=arm, site=site).filter(
            (pl.col("valid_time") >= week_start)
            & (pl.col("valid_time") < week_start + datetime.timedelta(days=7))
        )
        day = (pl.col("valid_time") - week_start).dt.total_minutes() / 1440.0 + 1.0
        frames.append(
            frame.select(day.alias("day"), (pl.col("q50") * 100.0).alias("value")).with_columns(
                series=pl.lit(names[arm])
            )
        )
        if actual is None:
            actual = frame.select(
                day.alias("day"), (pl.col("actual") * 100.0).alias("value")
            ).with_columns(series=pl.lit("Actual"))
    data = pl.concat([*frames, actual])
    order = ["Actual", *names.values()]
    colours = [
        ocf.GREY_5,
        ocf.DATA_SKY,
        ocf.DATA_BLUE,
        ocf.DATA_PURPLE,
        ocf.DATA_GREEN,
        ocf.DATA_AMBER,
        ocf.BRAND_ORANGE,
    ]
    return (
        alt.Chart(data.to_pandas())
        .mark_line(clip=True, strokeWidth=1.5)
        .encode(
            x=alt.X(
                "day:Q",
                scale=alt.Scale(domain=[1, 8]),
                axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7, 8], title="Day of the week (1 to 7)"),
            ),
            y=alt.Y("value:Q", title="% of capacity"),
            color=alt.Color(
                "series:N",
                sort=order,
                scale=alt.Scale(domain=order, range=colours),
                legend=alt.Legend(orient="top", title=None, columns=4),
            ),
            strokeWidth=alt.condition(
                alt.datum.series == "Blend (BG2)", alt.value(3), alt.value(1.3)
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=alt.TitleParams(
                text=f"{title_site}: median forecast of each blend member, {WEEK_TITLES[week]}",
                subtitle=wrapped(
                    text="Each day shows the 15 to 39 hour lead window of the run that started "
                    "that morning. Orange line: the blend. Grey line: actual.",
                    width=95,
                ),
            ),
        )
    )


def weights_figure(*, results_dir: Path) -> alt.Chart:
    """Return the blend weight of each member in each origin month."""
    table = pl.read_parquet(results_dir / "blend_weights_W_M_AN_PT_ST.parquet")
    long = table.unpivot(
        index="month",
        on=[c for c in table.columns if c.startswith("w_")],
        variable_name="member",
        value_name="weight",
    ).with_columns(
        member=pl.col("member")
        .str.replace(r"w_\d_", "")
        .replace(
            {
                "W": "Weekly",
                "M": "Monthly",
                "AN": "Analogue",
                "PT": "Pooled by type",
                "ST": "Frozen with correction",
            }
        )
    )
    return (
        alt.Chart(long.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("month:N", title="Month in which the runs start"),
            y=alt.Y("weight:Q", title="Blend weight", stack="zero"),
            color=alt.Color(
                "member:N",
                scale=alt.Scale(
                    domain=[
                        "Weekly",
                        "Monthly",
                        "Analogue",
                        "Pooled by type",
                        "Frozen with correction",
                    ],
                    range=[
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_SKY,
                    ],
                ),
                legend=alt.Legend(orient="top", title=None, columns=3),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=200,
            title=alt.TitleParams(
                text="Blend weights of the five members, month by month",
                subtitle=wrapped(
                    text="The weights are chosen on the cases of earlier months only, so the first "
                    "month has equal weights. Rebuilt BG2 blend, 16 Primary substations.",
                    width=95,
                ),
            ),
        )
    )


def figures(*, results_dir: Path) -> dict[str, Callable[[], alt.TopLevelMixin]]:
    """Return the drawing function of every figure."""
    absolute = pl.read_parquet(results_dir / "absolute.parquet")
    contrasts = pl.read_parquet(results_dir / "contrasts.parquet")
    out: dict[str, Callable[[], alt.TopLevelMixin]] = {}
    for index, cls in enumerate(("Primary", "BSP/GSP", "solar", "wind"), start=1):
        out[f"figure_1{chr(96 + index)}_{cls.lower().replace('/', '_')}_contrasts"] = (
            lambda cls=cls: contrast_figure(contrasts=contrasts, cls=cls)
        )
    out["figure_2_by_lead"] = lambda: lead_figure(absolute=absolute)
    out["figure_3_midday_summer"] = lambda: midday_figure(absolute=absolute, contrasts=contrasts)
    out["figure_6_blend_weights"] = lambda: weights_figure(results_dir=results_dir)
    return out


EXAMPLES: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "primary": (
        "Primary",
        "D10",
        ("p_s6_bg2", "p_s5_blend_no_offset", "p_s4_weekly", "p_nl_weekly", "ref_manual_heuristic"),
    ),
    "supply": ("BSP/GSP", "D17", ("b_weekly", "b_nl_weekly", "b_stack", "ref_manual_heuristic")),
    "solar": (
        "solar",
        "A",
        ("g_ens_weekly", "g_nl_monthly", "g_a_spread_on", "ref_manual_heuristic"),
    ),
    "wind": (
        "wind",
        "W1",
        ("g_ens_weekly", "g_nl_monthly", "g_a_wind_margin", "ref_manual_heuristic"),
    ),
}
MEMBER_SITES: Final[tuple[str, ...]] = ("D10", "D2", "D15")


def example_figures() -> dict[str, Callable[[], alt.TopLevelMixin]]:
    """Return the example-forecast and blend-member figures, which read the arm tables."""
    series = common.series_table()
    primaries = series.filter(pl.col("substation_type") == "Primary")["site"].to_list()
    out: dict[str, Callable[[], alt.TopLevelMixin]] = {}
    for key, (cls, site, arms) in EXAMPLES.items():
        reference = read_arm_series(arm=arms[0], site=site)
        weeks = choose_weeks(bg2=reference, no_offset=None)
        title_site = anonymous_label(site=site, cls=cls, primaries=primaries)
        for week, start in weeks.items():
            out[f"figure_4_{key}_{week}"] = (
                lambda s=start, w=week, c=cls, si=site, a=arms, t=title_site: week_figure(
                    cls=c, site=si, arms=a, week_start=s, week=w, title_site=t
                )
            )
    for site in MEMBER_SITES:
        reference = read_arm_series(arm="p_s6_bg2", site=site)
        weeks = choose_weeks(bg2=reference, no_offset=None)
        title_site = anonymous_label(site=site, cls="Primary", primaries=primaries)
        for week, start in weeks.items():
            letter = title_site[-1].lower()
            out[f"figure_5_members_{letter}_{week}"] = (
                lambda s=start, w=week, si=site, t=title_site: members_figure(
                    site=si, title_site=t, week_start=s, week=w
                )
            )
    return out


def write_all(*, drawers: dict[str, Callable[[], alt.TopLevelMixin]], optimise: bool) -> None:
    """Write every figure as SVG and PNG, then optimise the SVGs if asked."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, draw in drawers.items():
        chart = draw().configure_view(stroke=None).properties(padding=8)
        chart.save(DOCS_DIR / f"{name}.svg")
        chart.save(CHARTS_DIR / f"{name}.png", scale_factor=2)
        if optimise:
            subprocess.run(
                [
                    "npx",
                    "svgo@4",
                    "--multipass",
                    "--precision=1",
                    "--final-newline",
                    str(DOCS_DIR / f"{name}.svg"),
                ],
                check=True,
                capture_output=True,
            )
        print("wrote", name, flush=True)


def main() -> None:
    """Draw every figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-svgo", action="store_true", help="skip the SVG optimiser")
    parser.add_argument("--only", nargs="+", help="draw only figures whose name starts with this")
    args = parser.parse_args()
    drawers = {
        **figures(results_dir=common.WORK_DIR / "results"),
        **example_figures(),
    }
    if args.only:
        drawers = {n: d for n, d in drawers.items() if n.startswith(tuple(args.only))}
    write_all(drawers=drawers, optimise=not args.no_svgo)


if __name__ == "__main__":
    main()
