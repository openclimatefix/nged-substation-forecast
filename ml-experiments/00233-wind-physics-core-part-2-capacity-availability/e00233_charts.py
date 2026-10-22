# ruff: noqa: E501,D103
"""Draw experiment 233's figures from the stored results and print the numbers the page quotes.

uv run --no-sync python ml-experiments/00233-*/e00233_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00233_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
VARIANT_LABELS: Final[dict[str, str]] = {
    "core": "above-rated half-hours, with the core (the plan)",
    "busy": "half-hours at 20% of scale or more (post hoc)",
    "const": "above-rated half-hours, constant expected power",
}
VARIANT_COLOURS: Final[dict[str, str]] = {
    VARIANT_LABELS["core"]: ocf.BRAND_ORANGE,
    VARIANT_LABELS["busy"]: ocf.DATA_BLUE,
    VARIANT_LABELS["const"]: ocf.DATA_PURPLE,
}


def figure_detection(*, results: dict) -> None:
    rows = []
    for farm, entries in results["farms"].items():
        for variant, label in VARIANT_LABELS.items():
            entry = entries[f"{variant}/60"]
            for loss, summary in entry["injections"].items():
                rows.append(
                    {
                        "case": f"{farm}, {round(float(loss) * 100)}% loss",
                        "variant": label,
                        "share": summary["share_detected_within_120"],
                        "lower": summary["share_interval"][0],
                        "upper": summary["share_interval"][1],
                    }
                )
    data = pl.DataFrame(rows)
    order = [f"{farm}, {loss}% loss" for farm in sorted(results["farms"]) for loss in (10, 25, 50)]
    base = alt.Chart(data).encode(
        y=alt.Y("case:N", sort=order, title=None),
        yOffset="variant:N",
        color=alt.Color(
            "variant:N",
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
            scale=alt.Scale(domain=list(VARIANT_COLOURS), range=list(VARIANT_COLOURS.values())),
        ),
    )
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Share of the 30 start dates detected within 120 days (95% interval)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=50).encode(x="share:Q")  # ty: ignore[unresolved-attribute]
    (bars + points).properties(
        width=WIDTH,
        height=330,
        title=alt.TitleParams(
            [
                "The planned statistic finds a capacity loss at too few start dates;",
                "a variant using busier half-hours finds a 25% loss at 87% to 100% of start dates (60-day window)",
            ],
            anchor="start",
            fontSize=13,
        ),
    ).save(DOCS_DIR / "figure_1_detection_share.svg")


def figure_ratio_series(*, results: dict) -> None:
    series = pl.read_parquet(common.WORK_DIR / "ratio_series.parquet").filter(
        pl.col("day") >= pl.date(2023, 7, 1), pl.col("day").dt.weekday() == 1
    )
    names = {
        "real": "real observed power",
        "drift_9": "with a 9% seasonal drift",
        "injected_25": "with a 25% loss from a mid-sample start",
    }
    colours = {
        names["real"]: ocf.DATA_BLUE,
        names["drift_9"]: ocf.BRAND_ORANGE,
        names["injected_25"]: ocf.DATA_PURPLE,
    }
    data = series.with_columns(
        kind=pl.col("kind").replace(names), ratio=pl.col("ratio").fill_nan(None)
    )
    panels = []
    for farm, entries in results["farms"].items():
        lines = (
            alt.Chart(data.filter(pl.col("farm") == farm))  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=8)
            .encode(
                x=alt.X(
                    "day:T",
                    title=None,
                    scale=alt.Scale(domain=["2023-07-01", "2026-07-01"]),
                ),
                y=alt.Y(
                    "ratio:Q",
                    title=f"{farm}: relative ratio",
                    scale=alt.Scale(zero=False, domain=[0.3, 1.3]),
                ),
                color=alt.Color(
                    "kind:N",
                    legend=alt.Legend(
                        title=None, orient="top", direction="vertical", labelLimit=500
                    )
                    if farm == "W1"
                    else None,
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                ),
            )
        )
        rule = (
            alt.Chart(pl.DataFrame({"threshold": [entries["core/60"]["threshold"]]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(color="#808080", strokeDash=[4, 3])
            .encode(y="threshold:Q")
        )
        panels.append(alt.layer(lines, rule).properties(width=WIDTH, height=120))
    alt.vconcat(*panels).save(DOCS_DIR / "figure_2_ratio_series.svg")


def figure_coverage(*, results: dict) -> None:
    table = pl.read_parquet(common.WORK_DIR / "daily.parquet")
    rows = []
    for farm in sorted(results["farms"]):
        part = table.filter(pl.col("site") == farm).with_columns(month=pl.col("day").dt.month())
        for variant, column in (("core", "n_above"), ("busy", "n_busy")):
            monthly = (
                part.group_by("month")
                .agg((pl.col(column) >= 4).mean().alias("share"))
                .sort("month")
            )
            rows += [
                {
                    "farm": farm,
                    "variant": VARIANT_LABELS[variant],
                    "month": m,
                    "share": s,
                }
                for m, s in zip(monthly["month"], monthly["share"], strict=True)
            ]
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "month:O",
                title="Calendar month",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of days with at least 4 such half-hours"),
            color=alt.Color(
                "variant:N",
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
                scale=alt.Scale(
                    domain=[VARIANT_LABELS["core"], VARIANT_LABELS["busy"]],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
            ),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("farm:N", title=None))
    )
    chart.save(DOCS_DIR / "figure_3_days_with_enough_half_hours.svg")


def figure_real_monthly() -> None:
    sites = common.site_table()
    daily = pl.read_parquet(common.WORK_DIR / "daily.parquet").with_columns(
        month=pl.col("day").dt.truncate("1mo")
    )
    monthly = daily.group_by("site", "month").agg(
        ratio_busy=pl.col("obs_busy").sum() / pl.col("exp_busy").sum(),
        n_above=pl.col("n_above").sum(),
        ratio_core=pl.col("obs_above").sum() / pl.col("exp_above").sum(),
    )
    monthly = monthly.with_columns(
        ratio_core=pl.when(pl.col("n_above") >= 48).then(pl.col("ratio_core"))
    ).sort("site", "month")
    monthly.write_parquet(common.WORK_DIR / "monthly_ratio.parquet")
    long = monthly.unpivot(
        index=["site", "month"],
        on=["ratio_core", "ratio_busy"],
        variable_name="kind",
        value_name="ratio",
    ).drop_nulls("ratio")
    long = long.with_columns(
        kind=pl.col("kind").replace(
            {
                "ratio_core": "above-rated half-hours (months with 48 or more)",
                "ratio_busy": "half-hours at 20% of scale or more",
            }
        )
    )
    domain = [
        "above-rated half-hours (months with 48 or more)",
        "half-hours at 20% of scale or more",
    ]
    (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=25)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("ratio:Q", title="Observed / expected", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "kind:N",
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
                scale=alt.Scale(domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            ),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("site:N", title=None))
        .save(DOCS_DIR / "figure_4_real_monthly_ratio.svg")
    )
    power = (
        common.observed_power(sites=sites)
        .join(sites.select("time_series_id", "site", "effective_capacity_mw"), on="time_series_id")
        .with_columns(month=pl.col("valid_time").dt.truncate("1mo"))
        .group_by("site", "month")
        .agg(
            p99_mw=(pl.col("target") * pl.col("effective_capacity_mw")).quantile(0.99),
        )
        .sort("site", "month")
    )
    (
        alt.Chart(power)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.2, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y("p99_mw:Q", title="Monthly 99th percentile", scale=alt.Scale(zero=False)),
        )
        .properties(width=WIDTH, height=120)
        .facet(row=alt.Row("site:N", title=None))
        .resolve_scale(y="independent")
        .save(DOCS_DIR / "figure_5_monthly_p99.svg")
    )


def print_numbers(*, results: dict) -> None:
    for farm, entries in results["farms"].items():
        for key in ("core/60", "busy/60", "const/60", "core/30", "core/90"):
            entry = entries[key]
            print(
                farm,
                key,
                f"threshold {entry['threshold']:.3f} reference {entry['reference']:.3f}",
                f"days with statistic {entry['share_days_defined']:.2f}",
                f"hold-out alarm days {entry['holdout_alarm_days']}/{entry['holdout_days_defined']}",
                f"un-injected alarm days after 2024-07 {entry['uninjected_alarm_days_after_2024_07']}",
            )
            for loss, s in entry["injections"].items():
                print(
                    "   loss",
                    loss,
                    f"median {s['median_delay_days']} {s['median_interval']}",
                    f"within120 {s['share_detected_within_120']:.2f} {[round(x, 2) for x in s['share_interval']]}",
                    f"starts {s['starts']} excluded {s['excluded_already_alarming']}",
                    f"before-fit {s['before_fit_end']['share_detected_within_120']} after-fit {s['after_fit_end']['share_detected_within_120']}",
                )
            print(
                "   drift",
                {n: (d["alarm_days"], d["new_alarm_days"]) for n, d in entry["drift"].items()},
            )


def main() -> None:
    results = json.loads((common.WORK_DIR / "detect.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print_numbers(results=results)
    figure_detection(results=results)
    figure_ratio_series(results=results)
    figure_coverage(results=results)
    figure_real_monthly()


if __name__ == "__main__":
    main()
