# ruff: noqa: E501
"""Draw the figures of experiment 311 from the stored arms and `work/report/report_scored.json`.

uv run --no-sync python ml-experiments/00311-*/e00311_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

import e00311_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
GREY: Final[str] = "#6B6B6B"
LABELS: Final[dict[str, str]] = {"default": "Default", "M": "Arm M: band head", "MW": "Arm MW: band head + widening", "bg2": "BG2", "manual_heuristic": "Manual heuristic", "climatology": "Climatology"}  # fmt: skip
COLOURS: Final[dict[str, str]] = {"default": GREY, "M": ocf.BRAND_ORANGE, "MW": ocf.DATA_BLUE, "bg2": ocf.DATA_PURPLE, "manual_heuristic": ocf.DATA_GREEN, "climatology": ocf.DATA_SKY}  # fmt: skip
GROUPS: Final[tuple[str, ...]] = ("all rows", "ordinary rows", "cold winter nights", "sunny daylight", "Economy 7 night (00:30-07:30)", "Christmas", "bank holidays", "leads 12-36 h")  # fmt: skip


def chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def save(figure: alt.TopLevelMixin, *, name: str) -> None:
    """Write the SVG beside the page and a PNG for inspection."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    figure.save(DOCS_DIR / f"{name}.svg")
    figure.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def title(*, text: str, lines: list[str]) -> alt.TitleParams:
    """Return a title with a wrapped subtitle."""
    return alt.TitleParams(text=text, subtitle=lines, anchor="start", fontSize=14, subtitleFontSize=11, subtitleColor="#444")  # fmt: skip


def forest(
    *, rows: list[dict], name: str, head: str, lines: list[str], x_title: str, height_per: int = 26
) -> None:
    """Draw an estimate-and-interval chart, one row per label, arms in colour."""
    data = pl.DataFrame(rows)
    order = list(dict.fromkeys(data["label"].to_list()))
    arms = [a for a in ("M", "MW") if a in data["arm"].to_list()]
    colour = alt.Color("arm_name:N", scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms]), legend=alt.Legend(orient="top", title=None))  # fmt: skip
    data = data.with_columns(arm_name=pl.col("arm").replace(LABELS))
    base = chart(data).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), yOffset=alt.YOffset("arm_name:N"))  # fmt: skip
    rule = (
        chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222", strokeDash=[4, 3]).encode(x="x:Q")
    )
    bars = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lo:Q", title=x_title), x2="hi:Q", color=colour
    )
    dots = base.mark_point(filled=True, size=60).encode(x="est:Q", color=colour)
    figure = alt.layer(rule, bars, dots).properties(width=WIDTH - 230, height=height_per * len(order) * len(arms) // 1, title=title(text=head, lines=lines))  # fmt: skip
    save(figure, name=name)


def main() -> None:
    """Draw every figure."""
    rep = json.loads((common.REPORT_DIR / "report_scored.json").read_text())
    c = rep["contrasts"]
    # figure 1: by group
    rows = [{"label": g, "arm": a, "est": c[a]["groups"][g]["est"], "lo": c[a]["groups"][g]["lo"], "hi": c[a]["groups"][g]["hi"]} for g in GROUPS for a in ("M", "MW")]  # fmt: skip
    forest(rows=rows, name="figure_1_gain_by_group", head="The band head lowers the quantile score except on sunny daylight half-hours", lines=["Quantile score of each arm minus Primary demand research default 1, six Primaries, 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours.", "Dot: estimate. Line: 95% interval from resampling whole months. Left of zero is better."], x_title="Quantile score minus the default (points of capacity; more negative is better)")  # fmt: skip
    # figure 2: by Primary
    rows = [{"label": s.replace("site ", ""), "arm": a, "est": c[a]["groups"][s]["est"], "lo": c[a]["groups"][s]["lo"], "hi": c[a]["groups"][s]["hi"]} for s in [f"site {x}" for x in common.SCORED_SITES] for a in ("M",)]  # fmt: skip
    forest(rows=rows, name="figure_2_gain_by_primary", head="One Primary, D10, carries most of the gain", lines=["Arm M quantile score minus Primary demand research default 1, per Primary. Dot: estimate. Line: 95% interval from resampling whole months. Left of zero is better."], x_title="Quantile score minus the default (points of capacity; more negative is better)", height_per=34)  # fmt: skip
    # figure 3: gain by quantile level
    by_level = pl.DataFrame([{"level": q, "gain": c["M"]["by_level"][q]} for q in common.QUANTILES])
    fig3 = chart(by_level).mark_bar(color=ocf.BRAND_ORANGE).encode(x=alt.X("level:N", sort=list(common.QUANTILES), title="Quantile level", axis=alt.Axis(labelAngle=0)), y=alt.Y("gain:Q", title="Change in quantile score (points; negative is better)")).properties(width=WIDTH - 40, height=190, title=title(text="Most of the gain comes from the 0.05 to 0.25 quantiles, not from the extreme tails", lines=["Each bar is one quantile level's share of the quantile score change, arm M minus Primary demand research default 1, six Primaries, scored year.", "The bars add up to the headline change."]))  # fmt: skip
    save(fig3, name="figure_3_gain_by_level")
    # figure 4: coverage
    cov = []
    for arm in ("default", "M", "MW", "bg2", "manual_heuristic"):
        for k, nominal in (("q01", 0.01), ("q05", 0.05), ("q95", 0.95), ("q99", 0.99)):
            cov.append(
                {
                    "arm": LABELS[arm],
                    "level": k,
                    "coverage": rep["absolute"][arm][f"{k}_cov"],
                    "nominal": nominal,
                }
            )
    covd = pl.DataFrame(cov)
    nominal = chart(pl.DataFrame({"level": ["q01", "q05", "q95", "q99"], "nominal": [0.01, 0.05, 0.95, 0.99]})).mark_tick(color="#222", thickness=2, size=40).encode(x=alt.X("level:N", sort=["q01", "q05", "q95", "q99"]), y="nominal:Q")  # fmt: skip
    pts = chart(covd).mark_point(filled=True, size=70).encode(x=alt.X("level:N", sort=["q01", "q05", "q95", "q99"], title="Quantile level"), y=alt.Y("coverage:Q", title="Share of actuals at or below the quantile"), color=alt.Color("arm:N", scale=alt.Scale(domain=[LABELS[a] for a in ("default", "M", "MW", "bg2", "manual_heuristic")], range=[COLOURS[a] for a in ("default", "M", "MW", "bg2", "manual_heuristic")]), legend=alt.Legend(orient="top", title=None, columns=2)), xOffset="arm:N")  # fmt: skip
    save(alt.layer(nominal, pts).properties(width=WIDTH - 60, height=230, title=title(text="Tail coverage: the band head sits close to the default and to BG2", lines=["Share of half-hours at or below each quantile, six Primaries, scored year. Black tick: the level it should hit."])), name="figure_4_tail_coverage")  # fmt: skip
    # figures 5 and 6: example weeks and the Economy 7 step
    base = common.WORK_DIR / "arms" / "scored"
    rows_df = pl.read_parquet(base / "rows.parquet")
    arms = {a: np.load(base / f"{a}.npy") for a in ("default", "M")}
    frame = rows_df.with_columns(q05_d=pl.Series(arms["default"][:, 1]), q95_d=pl.Series(arms["default"][:, 9]), q05_m=pl.Series(arms["M"][:, 1]), q95_m=pl.Series(arms["M"][:, 9]), median=pl.Series(arms["default"][:, 5]))  # fmt: skip
    day_ahead = frame.filter((pl.col("lead_hours") >= 15.0) & (pl.col("lead_hours") < 39.0))
    for name, site, mask, text, note in (
        (
            "figure_5_cold_night_week",
            "D10",
            (pl.col("cold2")),
            "A cold winter week at D10: the band head gives narrower bands",
            "A week in January 2026",
        ),
        (
            "figure_6_sunny_week",
            "D5",
            (pl.col("sunny") & pl.col("daylight")),
            "A sunny week at D5: the band head gives a narrower lower band at midday",
            "A week in June 2026",
        ),
    ):
        pick = day_ahead.filter(pl.col("site") == site)
        window_month = (1, 2026) if "cold" in name else (6, 2026)
        cand = pick.filter(
            (pl.col("valid_time").dt.month() == window_month[0])
            & (pl.col("valid_time").dt.year() == window_month[1])
            & mask
        )
        start = (
            cand["valid_time"]
            .dt.date()
            .value_counts()
            .sort("count", descending=True)["valid_time"][0]
        )
        start = pl.Series([start]).cast(pl.Date)[0]
        window = pick.filter((pl.col("valid_time").dt.date() >= start) & (pl.col("valid_time").dt.date() < start + pl.duration(days=7))).sort("valid_time")  # fmt: skip
        window = window.with_columns(
            hours=(pl.col("valid_time") - pl.col("valid_time").min()).dt.total_seconds() / 86400.0
            + 1.0
        )
        window = window.with_columns(
            segment=((pl.col("hours").diff().fill_null(0.0) > 0.1).cast(pl.Int32).cum_sum())
        )
        long = pl.concat([window.select("segment", "hours", pl.col("q05_d").alias("lo"), pl.col("q95_d").alias("hi"), pl.lit(LABELS["default"]).alias("arm")), window.select("segment", "hours", pl.col("q05_m").alias("lo"), pl.col("q95_m").alias("hi"), pl.lit(LABELS["M"]).alias("arm"))])  # fmt: skip
        domain = [LABELS["default"], LABELS["M"]]
        band = chart(long).mark_area(opacity=0.35).encode(x=alt.X("hours:Q", title="Day of the week (1 to 7)", axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7], format="d")), y=alt.Y("lo:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)), y2="hi:Q", detail="segment:N", color=alt.Color("arm:N", scale=alt.Scale(domain=domain, range=[GREY, ocf.BRAND_ORANGE]), legend=alt.Legend(orient="top", title=None)))  # fmt: skip
        line = (
            chart(window)
            .mark_line(color="#111", strokeWidth=1.4)
            .encode(x="hours:Q", y="actual:Q", detail="segment:N")
        )
        save(alt.layer(band, line).properties(width=WIDTH - 60, height=230, title=title(text=text, lines=[f"{note}. Shaded: 5% to 95% band at day-ahead leads (15 to 39 hours). Black line: observed power. Substation {site}."])), name=name)  # fmt: skip
    winter = day_ahead.filter(pl.col("site") == "D10").filter(
        pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.month().is_in([12, 1, 2])
    )
    winter = winter.with_columns(hh=(pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.hour() * 2 + pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.minute() // 30))  # fmt: skip
    prof = winter.group_by("hh").agg(pl.col("actual").mean().alias("actual"), pl.col("q05_d").mean(), pl.col("q95_d").mean(), pl.col("q05_m").mean(), pl.col("q95_m").mean()).sort("hh")  # fmt: skip
    long = pl.concat([prof.select("hh", pl.col("q05_d").alias("lo"), pl.col("q95_d").alias("hi"), pl.lit(LABELS["default"]).alias("arm")), prof.select("hh", pl.col("q05_m").alias("lo"), pl.col("q95_m").alias("hi"), pl.lit(LABELS["M"]).alias("arm"))]).with_columns(hour=pl.col("hh") / 2.0)  # fmt: skip
    band = chart(long).mark_area(opacity=0.35).encode(x=alt.X("hour:Q", title="Local hour of day", axis=alt.Axis(values=[0, 6, 12, 18, 24])), y=alt.Y("lo:Q", title="Power (% of capacity)", scale=alt.Scale(zero=False)), y2="hi:Q", color=alt.Color("arm:N", scale=alt.Scale(domain=domain, range=[GREY, ocf.BRAND_ORANGE]), legend=alt.Legend(orient="top", title=None)))  # fmt: skip
    line = (
        chart(prof.with_columns(hour=pl.col("hh") / 2.0))
        .mark_line(color="#111", strokeWidth=1.4)
        .encode(x="hour:Q", y="actual:Q")
    )
    save(alt.layer(band, line).properties(width=WIDTH - 60, height=230, title=title(text="The Economy 7 night-time step stays inside both bands on average", lines=["Mean 5% to 95% band and mean observed power by local half-hour, December to February, day-ahead leads, substation D10. Black line: observed power."])), name="figure_7_economy7_profile")  # fmt: skip


if __name__ == "__main__":
    main()
