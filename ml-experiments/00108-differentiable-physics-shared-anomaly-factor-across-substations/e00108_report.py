"""Print experiment 108's numbers and draw its figures, from the tuning year only.

The experiment stopped at the known-answer gate, so no replicate-year or scored-year row is read.
This script reports:

- the gate (`work/gate.json`, written by `e00108_gate.py`);
- the shared state's response to a step of the factor, from the tuned shared filter;
- the 20 largest shared-anomaly days of the tuning year (2023-07-01 to 2024-06-30), ranked from
  the residual streams alone before any event is matched to them.

    uv run python ml-experiments/00108-*/e00108_report.py > docs/ml-experiments/00108-*/report.txt
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00108_common as common
import e00108_factor as factor_lib
import e00108_tune as tune_lib
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_assimilation import correction_at, run_kalman
from studies.dp_features import FeatureCache

FIGURE_DIR: Final[Path] = Path(__file__).parent / "draft"
WIDTH: Final[int] = 560
N_DAYS: Final[int] = 20
BAND_NAMES: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")


def top_days(*, cache: FeatureCache, panel: factor_lib.Panel) -> pl.DataFrame:
    """Rank the tuning year's days by the largest absolute band mean of the all-site median."""
    window = tune_lib.tuning_window(cache=cache)
    sites = common.DECISION_SITES
    stack = np.stack([panel.scaled[s][window] for s in sites])
    counts = np.isfinite(stack).sum(axis=0)
    median = np.nanmedian(stack, axis=0)
    median = np.where(counts >= common.MIN_CONTRIBUTORS, median, np.nan)
    temperature = cache.sites["D5"].temperature_c
    local = pl.DataFrame(
        {
            "date": cache.calendar.local_date[window],
            "band": cache.calendar.band[window].astype(int),
            "value": median,
            "count": counts,
            "temperature": temperature[window],
        }
    )
    daily_temperature = (
        pl.DataFrame({"date": cache.calendar.local_date, "temperature": temperature})
        .group_by("date")
        .agg(pl.col("temperature").mean().alias("day_temperature"))
        .sort("date")
    )
    anomaly = daily_temperature.with_columns(
        (
            pl.col("day_temperature") - pl.col("day_temperature").rolling_mean(30, min_samples=10)
        ).alias("temperature_anomaly")
    )
    per_band = (
        local.group_by(["date", "band"])
        .agg(
            pl.col("value").mean().alias("band_mean"), pl.col("count").mean().alias("contributors")
        )
        .drop_nulls("band_mean")
    )
    best = (
        per_band.with_columns(pl.col("band_mean").abs().alias("size"))
        .sort("size", descending=True)
        .group_by("date")
        .first()
        .sort("size", descending=True)
        .head(N_DAYS)
        .join(anomaly.select("date", "temperature_anomaly"), on="date", how="left")
    )
    return best.with_columns(
        pl.col("date").dt.strftime("%a").alias("weekday"),
        pl.col("band")
        .map_elements(lambda b: BAND_NAMES[b], return_dtype=pl.String)
        .alias("band_name"),
    ).sort("size", descending=True)


def shared_response(*, tuning: tune_lib.Tuning) -> dict[str, list[float]]:
    """Return the shared filter's level after a three-day unit step of the factor in one band.

    The step is 1 scaled unit on every half-hour of the first three days; the response is read
    at the 05:30 UTC cutoff of each following day.
    """
    n = 48 * 14
    stream = np.full(n, np.nan)
    stream[: 48 * 3] = 1.0
    stream[48 * 3 :] = 0.0
    band = (np.arange(n) // 12) % 4
    trace = run_kalman(residual=stream, band=band, hyper=tuning.shared_hyper)
    rows = np.arange(n)
    daily = []
    for day in range(1, 13):
        cutoff = np.full(n, day * 48 - 38)
        target = np.full(n, day * 48 + 24)
        correction = correction_at(trace=trace, cutoff=cutoff, target=target, band=np.full(n, 2))
        daily.append(float(correction.total[0]))
    del rows
    return {"days_from_start": list(range(1, 13)), "response": daily}


def figure_gate(*, gate: dict) -> alt.Chart:
    """Draw the shared part's share of the injected anomaly, by episode and by day."""
    rows = gate["3_shared_release"]["episodes"]
    frame = pl.DataFrame(
        {
            "episode": [r["episode"] for r in rows],
            "peak": [r["peak"] for r in rows],
            "three_days_after_end": [r["three_days_after_end"] for r in rows],
        }
    ).with_columns((pl.col("three_days_after_end") / pl.col("peak")).alias("share"))
    points = (
        alt.Chart(frame)
        .mark_circle(size=70, color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("episode:N", sort=None, title=None),
            x=alt.X(
                "share:Q",
                title="Share of the peak still present 3 days after the episode ends",
                scale=alt.Scale(domain=[0, 1]),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.25]}))
        .mark_rule(color=ocf.DATA_BLUE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (points + rule).properties(
        width=WIDTH,
        height=22 * len(rows),
        title=[
            "The shared part had not released after 3 days in any of the 12 episodes",
            "(dashed line: the pre-registered limit of 25% of the peak)",
        ],
    )


def figure_response(*, response: dict[str, list[float]]) -> alt.Chart:
    """Draw the tuned shared filter's response to a three-day step of the factor."""
    frame = pl.DataFrame({"day": response["days_from_start"], "level": response["response"]})
    line = (
        alt.Chart(frame)
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Target day (the factor is 1 on days 1 to 3, then 0)"),
            y=alt.Y("level:Q", title="Shared state at day-ahead cutoff"),
        )
    )
    return line.properties(
        width=WIDTH,
        height=200,
        title=[
            "The tuned shared filter rises slowly during a three-day step and falls slowly after it"
        ],
    )


def figure_days(*, days: pl.DataFrame) -> alt.Chart:
    """Draw the 20 largest shared-anomaly days of the tuning year."""
    frame = days.select(
        pl.col("date").cast(pl.String).alias("day"),
        pl.col("band_mean").alias("value"),
        pl.col("band_name"),
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("day:N", sort=None, title=None),
            x=alt.X("value:Q", title="Band mean of the all-site median of scaled residuals"),
            color=alt.Color(
                "band_name:N",
                scale=alt.Scale(
                    domain=list(BAND_NAMES),
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                title="Hour band",
            ),
        )
        .properties(
            width=WIDTH,
            height=18 * len(frame),
            title="The 20 largest shared-anomaly days of the tuning year (2023-07 to 2024-06)",
        )
    )


def main() -> None:
    """Print the numbers and draw the figures."""
    cache = common.load_cache()
    all_fits = factor_lib.load_fits(cache=cache)
    panel = factor_lib.build_panel(cache=cache, all_fits=all_fits)
    tuning = tune_lib.load_real_tuning()
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    print("== known-answer gate (tuning-year injections) ==")
    for name, value in gate.items():
        if name == "episodes":
            continue
        detail = {k: v for k, v in value.items() if k != "episodes"}
        print(f"{name}: {'PASS' if value['pass'] else 'FAIL'} {detail}")
    print("\nrelease by episode")
    for row in gate["3_shared_release"]["episodes"]:
        print(
            f"  {row['episode']}: peak {row['peak']:.3f}, 3 days after end "
            f"{row['three_days_after_end']:.3f}, share {row['ratio']:.3f}"
        )
    print("\n== frozen settings ==")
    print("raw Huber loadings:", {k: round(v, 3) for k, v in tuning.raw_beta.items()})
    print("shrunk loadings:", {k: round(v, 3) for k, v in tuning.beta.items()})
    hyper = tuning.shared_hyper
    print(
        f"shared filter: q {hyper.q:.4f} per day, phi {hyper.phi:.3f} per half-hour, "
        f"sigma_z {hyper.sigma_z:.3f}, sigma_eps {hyper.sigma_eps:.3f} (scaled units)"
    )
    for site, local in tuning.local_hyper.items():
        print(
            f"local filter {site}: q {local.q:.3f}, phi {local.phi:.3f}, "
            f"sigma_z {local.sigma_z:.3f}, sigma_eps {local.sigma_eps:.3f}"
        )
    response = shared_response(tuning=tuning)
    print("\n== shared state after a three-day unit step (days 1 to 12) ==")
    print(" ".join(f"{v:.3f}" for v in response["response"]))
    days = top_days(cache=cache, panel=panel)
    print("\n== 20 largest shared-anomaly days, tuning year only ==")
    print("date weekday band value contributors temperature_anomaly_c")
    for row in days.iter_rows(named=True):
        print(
            f"{row['date']} {row['weekday']} {row['band_name']} {row['band_mean']:+.2f} "
            f"{row['contributors']:.1f} {row['temperature_anomaly']:+.1f}"
        )
    FIGURE_DIR.mkdir(exist_ok=True)
    for name, chart in {
        "figure_01_gate_release": figure_gate(gate=gate),
        "figure_02_shared_response": figure_response(response=response),
        "figure_03_largest_days": figure_days(days=days),
    }.items():
        chart.save(FIGURE_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
