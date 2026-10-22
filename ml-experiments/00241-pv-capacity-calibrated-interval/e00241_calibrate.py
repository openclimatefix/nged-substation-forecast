"""Calibrate experiment 129's known-answer recoveries leave-one-farm-out and draw the result.

The script fits nothing. It reads the stored arm 2 recoveries, divides the capacity found by the
median recovery of the other farms at the same Primary, and builds an 80% interval from their 10th
and 90th percentiles. The plan is in
`docs/ml-experiments/00241-pv-capacity-calibrated-interval/result.md`.

    uv run --no-sync python ml-experiments/00241-*/e00241_calibrate.py
"""

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 241
SLUG: Final[str] = "pv-capacity-calibrated-interval"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
KNOWN_ANSWER_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00129-differentiable-physics-pv-capacity-known-answer-test"
    / "work"
    / "kat_arm2.parquet"
)
"""Experiment 129's arm 2 known-answer table."""
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"{EXPERIMENT_ID:05d}-{SLUG}"
)
PNG_DIR: Final[Path] = Path("/tmp/e00241_png")
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
READ_OUT_SEASON_YEAR: Final[int] = 2024
LOW_PERCENTILE: Final[float] = 10.0
HIGH_PERCENTILE: Final[float] = 90.0
TOLERANCE: Final[float] = 0.10
"""A corrected estimate within this share of the injected capacity counts as accurate."""
PASSING_SITES: Final[int] = 4
COVERAGE_RANGE: Final[tuple[float, float]] = (0.70, 0.90)
WIDTH: Final[int] = 600
WRAP: Final[int] = 125
MODES: Final[dict[str, str]] = {
    "same_year": "Calibrated on the other farms in the same season-year (12 recoveries)",
    "all_years": "Calibrated on the other farms in all season-years (up to 72 recoveries)",
}
FARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]


def _stat(series: pl.Series, function: Callable[[np.ndarray], float]) -> float:
    """Return `function` of the series as a float."""
    return float(function(series.to_numpy().astype(float)))


def load() -> pl.DataFrame:
    """Return experiment 129's recoveries for the matched real-farm injections."""
    table = pl.read_parquet(KNOWN_ANSWER_PATH).filter(
        pl.col("source").is_in(FARMS), pl.col("plant") == "matched"
    )
    found = pl.col("capacity_injected") - pl.col("capacity_base")
    check = table.select((found / pl.col("injected_points") - pl.col("recovery")).abs().max())
    if check.item() > 1e-9:
        msg = "the stored recovery differs from the capacity found over the injected capacity"
        raise ValueError(msg)
    return table.select(
        "site",
        pl.col("source").alias("farm"),
        "fraction",
        "season_year",
        "cloud_load",
        "injected_points",
        "recovery",
    )


def held_out(*, table: pl.DataFrame, mode: str) -> pl.DataFrame:
    """Return every held-out injection with its calibration median and 80% interval ends.

    Args:
        table: The recoveries.
        mode: `same_year` or `all_years`, the calibration set.

    Returns:
        One row per injection (with or without cloud-driven load), with `median`, `low`, and `high`
        from the same Primary's injections without load of the four other farms.
    """
    pool = table.filter(~pl.col("cloud_load"))
    rows = []
    for row in table.iter_rows(named=True):
        others = pool.filter(
            pl.col("site") == row["site"],
            pl.col("farm") != row["farm"],
            (pl.col("season_year") == row["season_year"])
            if mode == "same_year"
            else pl.lit(value=True),
        )["recovery"].to_numpy()
        rows.append(
            {
                **row,
                "median": float(np.median(others)),
                "low": float(np.percentile(others, LOW_PERCENTILE)),
                "high": float(np.percentile(others, HIGH_PERCENTILE)),
                "n_calibration": len(others),
            }
        )
    return (
        pl.DataFrame(rows)
        .with_columns(
            corrected=pl.col("recovery") / pl.col("median"),
            covered=(pl.col("recovery") >= pl.col("low")) & (pl.col("recovery") <= pl.col("high")),
        )
        .with_columns(
            error=pl.col("corrected") - 1.0,
            width=(1.0 / pl.col("low") - 1.0 / pl.col("high")) * pl.col("median"),
        )
    )


def summaries(*, results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the per-Primary summaries at the read-out season-year, without load."""
    rows = []
    for mode, result in results.items():
        read_out = result.filter(
            pl.col("season_year") == READ_OUT_SEASON_YEAR, ~pl.col("cloud_load")
        )
        for site in DECISION_SITES:
            chosen = read_out.filter(pl.col("site") == site)
            rows.append(
                {
                    "mode": mode,
                    "site": site,
                    "n": chosen.height,
                    "median_recovery": _stat(chosen["median"], np.median),
                    "median_abs_error_corrected": _stat(chosen["error"].abs(), np.median),
                    "median_abs_error_raw": _stat((1.0 - chosen["recovery"]).abs(), np.median),
                    "share_within_10pct": _stat(chosen["error"].abs() <= TOLERANCE, np.mean),
                    "coverage": _stat(chosen["covered"], np.mean),
                    "interval_width_over_median": _stat(chosen["width"], np.median),
                }
            )
    return pl.DataFrame(rows)


def coverage_table(*, results: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the coverage of the intervals by Primary, for each mode and load setting."""
    rows = []
    for mode, result in results.items():
        read_out = pl.col("season_year") == READ_OUT_SEASON_YEAR
        for label, selected in (
            ("read_out_no_load", read_out & ~pl.col("cloud_load")),
            ("all_years_no_load", ~pl.col("cloud_load")),
            ("read_out_cloud_load", read_out & pl.col("cloud_load")),
            ("all_years_cloud_load", pl.col("cloud_load")),
        ):
            chosen = result.filter(selected)
            for site in (*DECISION_SITES, "Pooled"):
                part = chosen if site == "Pooled" else chosen.filter(pl.col("site") == site)
                part = (
                    part if site != "Pooled" else part.filter(pl.col("site").is_in(DECISION_SITES))
                )
                rows.append(
                    {
                        "mode": mode,
                        "setting": label,
                        "site": site,
                        "n": part.height,
                        "coverage": _stat(part["covered"], np.mean),
                        "median_corrected": _stat(part["corrected"], np.median),
                    }
                )
    return pl.DataFrame(rows)


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def figure_error(*, result: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the corrected estimate over the injected capacity for each held-out farm."""
    read_out = result.filter(pl.col("season_year") == READ_OUT_SEASON_YEAR, ~pl.col("cloud_load"))
    data = read_out.group_by("site", "farm").agg(value=pl.col("corrected").median())
    scale = alt.Scale(domain=[0.7, 1.3])
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=list(DECISION_SITES), title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("farm:N", sort=list(FARMS)),
            y=alt.Y("value:Q", scale=scale, title="Corrected estimate over injected capacity"),
            color=alt.Color(
                "farm:N",
                sort=list(FARMS),
                scale=alt.Scale(domain=list(FARMS), range=FARM_COLOURS),
                legend=alt.Legend(title="Held-out farm", orient="top", direction="horizontal"),
            ),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"y": [1.0, 1.0 - TOLERANCE, 1.0 + TOLERANCE]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y=alt.Y("y:Q", scale=scale))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rules, points).properties(
        width=WIDTH,
        height=260,
        title=chart_title(
            text="Dividing by the other farms' median recovery puts the estimate near the truth",
            subtitle=(
                "Held-out injection of one real solar farm at a Primary substation, read at"
                " season-year 2024. Corrected estimate: the capacity found, divided by the median"
                " recovery of the other four farms at the same Primary and season-year, over the"
                " injected capacity (1 is exact; dot: median over the three injected fractions)."
                " Dashed lines: exact, and 10% either side."
            ),
        ),
    )


def figure_coverage(*, coverage: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the coverage of the 80% interval by Primary under four settings."""
    settings = {
        ("same_year", "read_out_no_load"): "Same-year calibration, season-year 2024",
        ("all_years", "read_out_no_load"): "All-years calibration, season-year 2024",
        ("all_years", "all_years_no_load"): "All-years calibration, all season-years",
        ("all_years", "all_years_cloud_load"): "All-years calibration, cloud-driven load added",
    }
    labels = list(settings.values())
    data = pl.concat(
        [
            coverage.filter(pl.col("mode") == mode, pl.col("setting") == setting).with_columns(
                label=pl.lit(label)
            )
            for (mode, setting), label in settings.items()
        ]
    )
    scale = alt.Scale(domain=[0.0, 1.0])
    sites = [*DECISION_SITES, "Pooled"]
    band = (
        alt.Chart(pl.DataFrame({"low": [COVERAGE_RANGE[0]], "high": [COVERAGE_RANGE[1]]}))
        .mark_rect(color="#DDDDDD", opacity=0.6)
        .encode(y=alt.Y("low:Q", scale=scale), y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("site:N", sort=sites, title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("label:N", sort=labels),
            y=alt.Y("coverage:Q", scale=scale, title="Share of held-out injections covered"),
            color=alt.Color(
                "label:N",
                sort=labels,
                scale=alt.Scale(
                    domain=labels,
                    range=[ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
            ),
        )
    )
    return alt.layer(band, points).properties(
        width=WIDTH,
        height=280,
        title=chart_title(
            text="The 80% interval covers 59% to 79% of injections, and 2% under cloud load",
            subtitle=(
                "Share of held-out injections whose injected capacity lies inside the 80% interval"
                " from the 10th and 90th percentiles of the other four farms' recoveries. Same-year"
                " calibration: the other farms in the same season-year (12 recoveries). All-years"
                " calibration: the other farms in every season-year. Grey band: the 70% to 90% that"
                " the plan accepts. Pooled: all five decision Primaries."
            ),
        ),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    table = load()
    print("rows per Primary and season-year (no load, with load):")
    print(
        table.group_by("site", "season_year", "cloud_load")
        .agg(n=pl.len())
        .pivot(on="cloud_load", index=["site", "season_year"], values="n")
        .sort("site", "season_year")
    )
    results = {mode: held_out(table=table, mode=mode) for mode in MODES}
    with pl.Config(tbl_rows=80, tbl_cols=20, float_precision=3, tbl_width_chars=200):
        summary = summaries(results=results)
        print("Per Primary at season-year 2024, no load")
        print(summary)
        for mode in MODES:
            chosen = summary.filter(pl.col("mode") == mode)
            passing = int((chosen["median_abs_error_corrected"] <= TOLERANCE).sum())
            print(f"{mode}: {passing} of 5 Primaries with median absolute error <= {TOLERANCE}")
        coverage = coverage_table(results=results)
        print("Coverage of the 80% interval")
        print(coverage.sort("mode", "setting", "site"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pl.concat(
        [result.with_columns(mode=pl.lit(mode)) for mode, result in results.items()]
    ).write_parquet(DATA_DIR / "held_out.parquet")
    save_chart(figure_error(result=results["same_year"]), name="figure_01_error")
    save_chart(figure_coverage(coverage=coverage), name="figure_02_coverage")


if __name__ == "__main__":
    main()
