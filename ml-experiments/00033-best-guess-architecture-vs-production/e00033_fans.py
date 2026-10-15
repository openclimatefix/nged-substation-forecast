"""Draw experiment 00033's example probabilistic forecasts (fan charts), as SVG.

Each figure shows one 3-day window of one series: the nine quantiles of the best guess beside
production's, with the observed power. Windows are chosen from the saved metrics, not by hand: a
good and a bad window per series by the best guess's mean quantile score, the coldest window for
Primary demand, the sunniest and cloudiest summer windows for solar, and the windiest and calmest
windows for wind. Generators carry anonymised labels, the axis counts days (no calendar date), the
text gives the month and year only, and values are fractions of capacity.

    uv run python ml-experiments/00033-*/e00033_fans.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00033_common as common
import e00033_gen_common as gen
import plotting.ocf_theme as ocf
import polars as pl

METRICS_DIR: Final[Path] = common.WORK_DIR / "metrics"
TABLES_DIR: Final[Path] = common.WORK_DIR / "generator_tables"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 560
WINDOW_HOURS: Final[int] = 72
SUMMER_MONTHS: Final[tuple[int, ...]] = (5, 6, 7, 8)
BEST: Final[dict[str, str]] = {
    "Primary": "n1_cs",
    "BSP": "n1",
    "GSP": "n1",
    "solar": "bg_base_spread",
    "wind": "bg_full",
}
BASELINE: Final[str] = "production"
WINDOW_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time")


def window_scores(*, group: str) -> pl.DataFrame:
    """Return the best guess's mean quantile score per series and run over the first 72 hours."""
    return (
        pl.read_parquet(METRICS_DIR / group / f"{BEST[group]}.parquet")
        .filter((pl.col("seed") == 0) & (pl.col("lead_hours") <= WINDOW_HOURS))
        .group_by(*WINDOW_KEYS)
        .agg(score=pl.col("qs").mean(), cases=pl.len())
        .filter(pl.col("cases") >= WINDOW_HOURS)
    )


def condition_table(*, group: str) -> pl.DataFrame:
    """Return mean temperature, shortwave, and wind speed over each series' 72-hour window."""
    scores = window_scores(group=group)
    if group in ("solar", "wind"):
        analysis = pl.read_parquet(
            gen.ANALYSIS_PATH,
            columns=["time_series_id", "valid_time", "shortwave", "wind_speed_100m"],
        )
        columns = ("shortwave", "wind_speed_100m")
    else:
        series = common.series_table()
        arrays = common.load_series_arrays(series=series)
        names = common.feature_columns(calendar=True)
        analysis = pl.concat(
            [
                pl.DataFrame({"valid_time": common.grid_times()}).with_columns(
                    time_series_id=pl.lit(int(row["time_series_id"])).cast(pl.Int32),
                    temperature_2m=pl.Series(
                        arrays[row["site"]].features[:, names.index("temperature_2m")]
                    ),
                    shortwave=pl.Series(arrays[row["site"]].features[:, names.index("shortwave")]),
                )
                for row in series.iter_rows(named=True)
            ]
        )
        columns = ("temperature_2m", "shortwave")
    keys = (
        pl.read_parquet(METRICS_DIR / group / f"{BEST[group]}.parquet")
        .filter((pl.col("seed") == 0) & (pl.col("lead_hours") <= WINDOW_HOURS))
        .select(*WINDOW_KEYS, "valid_time")
    )
    joined = keys.join(analysis, on=["time_series_id", "valid_time"], how="left")
    means = joined.group_by(*WINDOW_KEYS).agg(*[pl.col(c).mean().alias(c) for c in columns])
    return scores.join(means, on=list(WINDOW_KEYS))


def pick(*, table: pl.DataFrame, column: str, highest: bool, month_filter: bool = False) -> dict:
    """Return the key of the window with the highest or lowest value of a column."""
    part = table
    if month_filter:
        part = part.filter(pl.col("power_fcst_init_time").dt.month().is_in(SUMMER_MONTHS))
    row = part.sort(column, descending=highest).row(0, named=True)
    return {k: row[k] for k in WINDOW_KEYS}


def pick_quantile(*, table: pl.DataFrame, series_id: int, fraction: float) -> dict:
    """Return the series' window at a quantile of the best guess's score."""
    part = table.filter(pl.col("time_series_id") == series_id).sort("score")
    row = part.row(round(fraction * (part.height - 1)), named=True)
    return {k: row[k] for k in WINDOW_KEYS}


def load_window(*, group: str, arm: str, key: dict) -> pl.DataFrame:
    """Return an arm's nine quantiles and the observation for one series and run, first 72 hours."""
    if group in ("solar", "wind"):
        frame = pl.scan_parquet(TABLES_DIR / f"{arm}.parquet").rename({"actual_norm": "actual"})
    else:
        frame = pl.scan_parquet(common.ARMS_DIR / f"{arm}.parquet")
    return (
        frame.filter(
            (pl.col("time_series_id") == key["time_series_id"])
            & (pl.col("power_fcst_init_time") == key["power_fcst_init_time"])
            & (pl.col("seed") == 0)
        )
        .select("valid_time", "actual", *common.LEVEL_COLUMNS)
        .collect()
        .sort("valid_time")
        .with_columns(
            day=(pl.col("valid_time") - key["power_fcst_init_time"]).dt.total_minutes() / 1440.0
            + 0.0
        )
        .filter(pl.col("day") <= WINDOW_HOURS / 24.0)
    )


def panel(
    *, data: pl.DataFrame, colour: str, label: str, y_title: str
) -> alt.LayerChart | alt.FacetChart:
    """Return one fan: 5%-95% and 25%-75% bands, the median, and the observation."""
    scale = 100.0
    scaled = data.with_columns(
        *[(pl.col(c) * scale).alias(c) for c in (*common.LEVEL_COLUMNS, "actual")],
        day_number=pl.col("day") + 1.0,
    )
    base = alt.Chart(scaled).encode(
        x=alt.X(
            "day_number:Q",
            title="Day of the window",
            axis=alt.Axis(values=[1, 2, 3, 4], format="d"),
            scale=alt.Scale(domain=[1, 4]),
        )
    )
    outer = base.mark_area(opacity=0.22, color=colour, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title=y_title), y2="q95:Q"
    )
    inner = base.mark_area(opacity=0.45, color=colour, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y="q25:Q", y2="q75:Q"
    )
    median = base.mark_line(color=colour, strokeWidth=2, clip=True, aria=False).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
    actual = base.mark_line(color="#222222", strokeWidth=1.5, clip=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y="actual:Q"
    )
    return alt.layer(outer, inner, median, actual).properties(
        width=PLOT_WIDTH, height=130, title=alt.TitleParams(label, anchor="start", fontSize=12)
    )


def figure(
    *, group: str, key: dict, label: str, caption: str, month_text: str
) -> alt.TopLevelMixin:
    """Return the two-panel figure for one window."""
    y_title = "Power (% of capacity)"
    panels = []
    for arm, colour, name in (
        (BEST[group], ocf.BRAND_ORANGE, "Best guess"),
        (BASELINE, ocf.DATA_BLUE, "Production"),
    ):
        panels.append(
            panel(
                data=load_window(group=group, arm=arm, key=key),
                colour=colour,
                label=name,
                y_title=y_title,
            )
        )
    title = alt.TitleParams(
        text=f"{label}: {caption}",
        subtitle=[
            (
                f"First 72 hours of one forecast run, {month_text}. Shade: 5% to 95% and 25% to "
                "75% bands; coloured line: median; black line: observed power."
            ),
        ],
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )
    return alt.vconcat(*panels).properties(title=title)


def month_text(*, key: dict) -> str:
    """Return the run's month and year, never the day."""
    return key["power_fcst_init_time"].strftime("%B %Y")


def build_plan() -> list[dict]:
    """Choose every window to draw."""
    plan: list[dict] = []
    series = common.series_table()
    sites = gen.site_table()
    label_of = {int(r["time_series_id"]): r["site"] for r in series.iter_rows(named=True)}
    generator_label = {
        int(r["time_series_id"]): (
            f"Solar farm {r['site']}"
            if r["technology"] == "solar"
            else f"Wind farm {r['site'][1:]}"
        )
        for r in sites.iter_rows(named=True)
    }
    primary = condition_table(group="Primary")
    ranked = primary.group_by("time_series_id").agg(score=pl.col("score").mean()).sort("score")
    chosen = [
        int(ranked["time_series_id"][i])
        for i in (ranked.height // 4, ranked.height // 2, 3 * ranked.height // 4)
    ]
    for series_id in chosen:
        for fraction, caption in (
            (0.1, "a window the best guess forecast well"),
            (0.95, "one of the worst"),
        ):
            key = pick_quantile(table=primary, series_id=series_id, fraction=fraction)
            plan.append(
                {
                    "group": "Primary",
                    "key": key,
                    "label": f"Primary substation {label_of[series_id]}",
                    "caption": caption,
                }
            )
    cold = pick(table=primary, column="temperature_2m", highest=False)
    plan.append(
        {
            "group": "Primary",
            "key": cold,
            "label": f"Primary substation {label_of[cold['time_series_id']]}",
            "caption": "the coldest window of the year",
        }
    )
    sunny = pick(table=primary, column="shortwave", highest=True, month_filter=True)
    cloudy = pick(table=primary, column="shortwave", highest=False, month_filter=True)
    plan.append(
        {
            "group": "Primary",
            "key": sunny,
            "label": f"Primary substation {label_of[sunny['time_series_id']]}",
            "caption": "a sunny summer window",
        }
    )
    plan.append(
        {
            "group": "Primary",
            "key": cloudy,
            "label": f"Primary substation {label_of[cloudy['time_series_id']]}",
            "caption": "a cloudy summer window",
        }
    )
    for group in ("BSP", "GSP"):
        table = condition_table(group=group)
        series_id = int(table["time_series_id"][0])
        key = pick_quantile(table=table, series_id=series_id, fraction=0.9)
        plan.append(
            {
                "group": group,
                "key": key,
                "label": f"{group} substation {label_of[series_id]}",
                "caption": "a poorly forecast window",
            }
        )
    for group, column, high_text, low_text in (
        ("solar", "shortwave", "a sunny summer window", "a cloudy summer window"),
        ("wind", "wind_speed_100m", "a windy window", "a calm window"),
    ):
        table = condition_table(group=group)
        for highest, text in ((True, high_text), (False, low_text)):
            key = pick(table=table, column=column, highest=highest, month_filter=group == "solar")
            plan.append(
                {
                    "group": group,
                    "key": key,
                    "label": generator_label[key["time_series_id"]],
                    "caption": text,
                }
            )
        worst = table.sort("score", descending=True).row(0, named=True)
        key = {k: worst[k] for k in WINDOW_KEYS}
        plan.append(
            {
                "group": group,
                "key": key,
                "label": generator_label[key["time_series_id"]],
                "caption": "the worst-forecast window",
            }
        )
    return plan


def main() -> None:
    """Write every fan chart and a list of what each shows."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    listing = []
    for number, item in enumerate(build_plan(), start=1):
        name = f"fan_{number:02d}_{item['group'].lower()}"
        figure(
            group=item["group"],
            key=item["key"],
            label=item["label"],
            caption=item["caption"],
            month_text=month_text(key=item["key"]),
        ).save(CHART_DIR / f"{name}.svg")
        listing.append(
            {
                "file": f"{name}.svg",
                "group": item["group"],
                "label": item["label"],
                "caption": item["caption"],
                "month": month_text(key=item["key"]),
            }
        )
        print(f"wrote {name}: {item['label']}, {item['caption']}", flush=True)
    (CHART_DIR / "fans.json").write_text(json.dumps(listing, indent=2) + "\n")


if __name__ == "__main__":
    main()
