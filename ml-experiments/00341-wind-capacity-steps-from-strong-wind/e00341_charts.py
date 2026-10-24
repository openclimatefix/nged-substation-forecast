# ruff: noqa: E501, D103, PERF401
"""Draw experiment 00341's figures from the stored results.

    uv run --no-sync python ml-experiments/00341-*/e00341_charts.py

The time axes count years since the start of the record (the first ERA5 month) so that no figure
puts calendar dates on a metered farm's series.
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00341_common as common
import e00341_inject as inj
import e00341_lib as lib
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
WIDTH: Final[int] = 560
FARM_COLOURS: Final[dict[str, str]] = {
    "W1": ocf.DATA_BLUE,
    "W2": ocf.BRAND_ORANGE,
    "W3": ocf.DATA_PURPLE,
}
ARM_LABELS: Final[dict[str, str]] = {
    "A_p999@2": "A: 99.9th percentile, strong set (2 m/s margin)",
    "A_p999@0": "A: 99.9th percentile, saturated set (post hoc)",
    "B": "B: weighted log ratio",
    "C": "C: B with speed-scale term",
    "D": "D: experiment 233's ratio",
}
ARM_COLOURS: Final[dict[str, str]] = {
    ARM_LABELS["A_p999@2"]: ocf.DATA_SKY,
    ARM_LABELS["A_p999@0"]: ocf.DATA_BLUE,
    ARM_LABELS["B"]: ocf.BRAND_ORANGE,
    ARM_LABELS["C"]: ocf.DATA_PURPLE,
    ARM_LABELS["D"]: ocf.DATA_GREEN,
}


def scale_for(arms: tuple[str, ...]) -> alt.Scale:
    labels = [ARM_LABELS[a] for a in arms]
    return alt.Scale(domain=labels, range=[ARM_COLOURS[x] for x in labels])


def legend(show: bool) -> alt.Legend | None:
    return (
        alt.Legend(title=None, orient="top", direction="vertical", labelLimit=600) if show else None
    )


def figure_strong_share() -> None:
    monthly = pl.read_parquet(common.WORK_DIR / "strong_by_month.parquet")
    data = monthly.with_columns(
        years=pl.col("month_index") * 30.4375 / 365.25,
        share=pl.col("len") / (30.4375 * 48) * 100,
    ).rename({"site": "farm"})
    (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "years:Q",
                title="Years since the start of the record",
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(values=list(range(8))),
            ),
            y=alt.Y("share:Q", title="Share of half-hours in the strong set (%)"),
            color=alt.Color(
                "farm:N",
                legend=alt.Legend(title=None, orient="top"),
                scale=alt.Scale(domain=list(FARM_COLOURS), range=list(FARM_COLOURS.values())),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                [
                    "Strong wind is rare: the share of half-hours with ERA5 speed",
                    "2 m/s above saturation (and below cut-out), by month",
                ],
                anchor="start",
                fontSize=13,
            ),
        )
        .save(DOCS_DIR / "figure_1_strong_set_share.svg")
    )


def real_series() -> dict:
    penalty = json.loads((common.WORK_DIR / "penalty.json").read_text())
    inj.init_worker()
    out: dict = {}
    for site in common.SITES:
        for arm, block in (("A_p999@0", 28), ("A_p999@2", 28), ("D", 14)):
            arms, _ = inj.arms_for(site=site, effects=(), block=block)
            series = arms[arm]
            group = "A" if arm.startswith("A_") else arm
            last, _ = lib.partition(series=series, penalty_c=penalty["c"][group])
            starts = lib.backtrack(last=last, t=len(series.y))
            ends = [*starts[1:], len(series.y)]
            seg = [
                {
                    "x0": float(series.pos[a] * block / 365.25),
                    "x1": float((series.pos[b - 1] + 1) * block / 365.25),
                    "level": float(np.average(series.y[a:b], weights=series.omega[a:b])),
                }
                for a, b in zip(starts, ends, strict=True)
            ]
            out[(site, arm)] = (series, block, seg)
    return out


def figure_real(
    *,
    real: dict,
    arm_set: tuple[str, ...],
    name: str,
    title: list[str],
    ytitle: str,
    domain: list[float],
) -> None:
    panels = []
    outage = pl.read_parquet(common.E210_DIR / "outage_runs.parquet")
    for site in common.SITES:
        layers = []
        for arm in arm_set:
            series, block, seg = real[(site, arm)]
            pts = pl.DataFrame(
                {"x": series.pos * block / 365.25, "y": series.y, "series": ARM_LABELS[arm]}
            )
            layers.append(
                alt.Chart(pts)
                .mark_point(filled=True, size=14, aria=False)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "x:Q",
                        title="Years since the start of the record" if site == "W3" else None,
                        scale=alt.Scale(domain=[0, 7]),
                        axis=alt.Axis(values=list(range(8))),
                    ),
                    y=alt.Y(
                        "y:Q", title=f"{site}: {ytitle}", scale=alt.Scale(domain=domain, zero=False)
                    ),
                    color=alt.Color(
                        "series:N", legend=legend(site == "W1"), scale=scale_for(arm_set)
                    ),
                )
            )
            layers.append(
                alt.Chart(pl.DataFrame(seg))
                .mark_rule(strokeWidth=2, color="#222222")
                .encode(x="x0:Q", x2="x1:Q", y="level:Q")  # ty: ignore[unresolved-attribute]
            )
        runs = outage.filter(pl.col("site") == site).with_columns(
            x0=(pl.col("start") - pl.lit(common.ERA5_START)).dt.total_days() / 365.25,
            x1=(pl.col("end") - pl.lit(common.ERA5_START)).dt.total_days() / 365.25 + 0.01,
            y=pl.lit(domain[0]),
        )
        layers.append(
            alt.Chart(runs)
            .mark_rule(strokeWidth=3, color="#808080")
            .encode(x="x0:Q", x2="x1:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(alt.layer(*layers).properties(width=WIDTH, height=130))
    alt.vconcat(*panels).properties(title=alt.TitleParams(title, anchor="start", fontSize=13)).save(
        DOCS_DIR / name
    )


def figure_detection() -> None:
    summary = json.loads((common.WORK_DIR / "summary.json").read_text())
    steps = summary["steps_d90_b28"]
    rows = []
    for key, v in steps.items():
        site, arm, size, detector = key.split("|")
        if detector != "pelt" or arm not in ARM_LABELS or not v.get("n"):
            continue
        n = common.N_TURBINES[site]
        frac = {"t1": 1 / n, "t2": 2 / n, "half": 0.5 / n, "p10": 0.1, "p25": 0.25, "p50": 0.5}[
            size
        ]
        rows.append(
            {
                "farm": site,
                "series": ARM_LABELS[arm],
                "step": frac * 100,
                "share": v["share"],
                "delay": v["median_delay"] if np.isfinite(v["median_delay"]) else None,
            }
        )
    data = pl.DataFrame(rows).sort("farm", "series", "step")
    panels = []
    for site in common.SITES:
        sub = data.filter(pl.col("farm") == site)
        panels.append(
            alt.Chart(sub)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "step:Q",
                    title="Size of the step in capacity (% of capacity)" if site == "W3" else None,
                    scale=alt.Scale(domain=[0, 52]),
                ),
                y=alt.Y("share:Q", title=f"{site}: share found", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    legend=legend(site == "W1"),
                    scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                ),
            )
            .properties(width=WIDTH, height=130)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Share of 30 start dates at which a step lasting 90 days is found within 60 days,",
                "by step size (28-day blocks, change-point search; 80% is the target)",
            ],
            anchor="start",
            fontSize=13,
        )
    ).save(DOCS_DIR / "figure_4_detection_by_step_size.svg")


def figure_episodes() -> None:
    summary = json.loads((common.WORK_DIR / "summary.json").read_text())
    rows = []
    for key, v in summary["episodes_b7"].items():
        site, arm, size, dur = key.split("|")
        if size != "p25" or arm not in ARM_LABELS or not v.get("n"):
            continue
        rows.append(
            {"farm": site, "series": ARM_LABELS[arm], "duration": int(dur[1:]), "share": v["share"]}
        )
    data = pl.DataFrame(rows).sort("farm", "series", "duration")
    panels = []
    for site in common.SITES:
        panels.append(
            alt.Chart(data.filter(pl.col("farm") == site))  # ty: ignore[unresolved-attribute]
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "duration:Q",
                    title="Length of the 25% episode (days)" if site == "W3" else None,
                    scale=alt.Scale(type="log"),
                    axis=alt.Axis(values=[3, 7, 14, 30, 60]),
                ),
                y=alt.Y("share:Q", title=f"{site}: share found", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "series:N",
                    legend=legend(site == "W1"),
                    scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                ),
            )
            .properties(width=WIDTH, height=130)
        )
    alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Share of 30 start dates at which a 25% episode that then ends is found",
                "(7-day blocks; within the episode's length plus 60 days)",
            ],
            anchor="start",
            fontSize=13,
        )
    ).save(DOCS_DIR / "figure_5_episode_detection.svg")


def figure_false_alarms() -> None:
    penalty = json.loads((common.WORK_DIR / "penalty.json").read_text())
    rows = []
    for group, table in penalty["c_excess_per_farm_year"].items():
        label = {
            "A": ARM_LABELS["A_p999@0"],
            "B": ARM_LABELS["B"],
            "C": ARM_LABELS["C"],
            "D": ARM_LABELS["D"],
        }[group]
        for c, v in table.items():
            rows.append({"series": label, "penalty": float(c), "excess": v})
    data = pl.DataFrame(rows)
    line = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "penalty:Q",
                title="Penalty multiplier c (larger: fewer change points)",
                scale=alt.Scale(type="log", domain=[1, 16]),
                axis=alt.Axis(values=[1, 2, 4, 8, 16]),
            ),
            y=alt.Y("excess:Q", title="Worst farm: false alarms per farm-year"),
            color=alt.Color(
                "series:N",
                legend=legend(True),
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    (line + rule).properties(
        width=WIDTH,
        height=200,
        title=alt.TitleParams(
            [
                "Change points that vanish at the largest penalty, in the calibration years,",
                "per farm-year; dashed line: the limit of one",
            ],
            anchor="start",
            fontSize=13,
        ),
    ).save(DOCS_DIR / "figure_6_false_alarms_by_penalty.svg")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_strong_share()
    real = real_series()
    figure_real(
        real=real, arm_set=("A_p999@0", "A_p999@2"), name="figure_2_strong_p999_series.svg",
        title=["The 99.9th percentile of output in 28-day blocks of strong or saturated wind;", "black: change-point levels; grey: experiment 210's idle runs"],
        ytitle="share of scale", domain=[0.4, 1.3],
    )  # fmt: skip
    figure_real(
        real=real, arm_set=("D",), name="figure_3_ratio_series.svg",
        title=["Observed over expected power in 14-day blocks, half-hours at 20% of scale or more;", "black: change-point levels; grey: experiment 210's idle runs"],
        ytitle="observed / expected", domain=[0.0, 1.4],
    )  # fmt: skip
    figure_detection()
    figure_episodes()
    figure_false_alarms()


if __name__ == "__main__":
    main()
