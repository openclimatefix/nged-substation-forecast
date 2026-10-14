"""Fit static CAMS corrections to the stage 1a daytime residuals and score them on the next year.

Residuals come from experiment 00006's out-of-fold `finals` fits (mean over the three seeds).
Year A (fold `select_2024`) fits each substation's correction and year B (fold `confirm_2025`)
scores it with the coefficients frozen. Prints every number the write-up quotes, with substations
shown by their anonymous labels only, and writes the hourly table and the figures.

    uv run python ml-experiments/00007-cams-static-correction-stage1a-residual/e00007_correction.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR, STUDIES_DATA_DIR

EXPERIMENT_00006_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep"
)
EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00007-cams-static-correction-stage1a-residual"
)
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads" / "reanalysis" / "CAMS" / "demand_points"
) / "cams_demand_points.parquet"
DEMAND_SERIES_PATH: Final[Path] = (
    PER_STUDY_DIR / "stage1_pilot" / "demand" / "inputs" / "series.parquet"
)
CHOSEN: Final[str] = "W+clock+covid_flag+xmas|recency"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
YEAR_A: Final[str] = "select_2024"
YEAR_B: Final[str] = "confirm_2025"
MAX_ZENITH_DEGREES: Final[float] = 80.0
N_RESAMPLES: Final[int] = 10_000
LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 3)
"""Bonferroni across the three interval conditions of the decision rule."""
MIN_DENOMINATOR: Final[float] = 0.05
ARMS: Final[dict[str, tuple[str, ...]]] = {
    "none": (),
    "shift": (),
    "era5": ("era5",),
    "cams": ("cams",),
    "era5+cams": ("era5", "cams"),
    "cams-era5": ("diff",),
}
FIGURES_DIR: Final[Path] = EXPERIMENT_DIR / "results"


def hourly_residuals() -> pl.DataFrame:
    """Return daylight hourly residuals of Primary series with ERA5 and CAMS, for years A and B."""
    series = pl.read_parquet(EXPERIMENT_00006_DIR / "inputs" / "series.parquet").select(
        "time_series_id", "site", "substation_type"
    )
    demand = pl.read_parquet(DEMAND_SERIES_PATH).select("time_series_id", demand_site="site")
    labels = series.join(demand, on="time_series_id", validate="1:1").filter(
        pl.col("substation_type") == "Primary"
    )
    cams = pl.read_parquet(CAMS_PATH).select(demand_site="site", hour="time", cams="ghi_w_m2")
    features = pl.read_parquet(EXPERIMENT_00006_DIR / "inputs" / "features.parquet").select(
        "time_series_id", "valid_time", era5="shortwave", zenith="solar_zenith"
    )
    pieces = []
    for year, fold in (("A", YEAR_A), ("B", YEAR_B)):
        for seed in SEEDS:
            folder = EXPERIMENT_00006_DIR / "fits" / "finals" / fold / "series" / f"seed{seed}"
            for path in sorted(folder.glob("*.parquet")):
                if path.stem.endswith("_importance"):
                    continue
                pieces.append(
                    pl.read_parquet(
                        path, columns=["time_series_id", "valid_time", "target", CHOSEN]
                    ).with_columns(year=pl.lit(year), seed=pl.lit(seed))
                )
    rows = pl.concat(pieces)
    return (
        rows.group_by("year", "time_series_id", "valid_time")
        .agg(target=pl.col("target").first(), forecast=pl.col(CHOSEN).mean())
        .join(features, on=["time_series_id", "valid_time"])
        .join(labels, on="time_series_id")
        .with_columns(
            residual=(pl.col("target") - pl.col("forecast")) * 100.0,
            hour=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1),
        )
        .group_by("year", "site", "demand_site", "hour")
        .agg(
            residual=pl.col("residual").mean(),
            era5=pl.col("era5").mean(),
            zenith=pl.col("zenith").mean(),
        )
        .join(cams, on=["demand_site", "hour"], how="inner")
        .filter(pl.col("zenith") < MAX_ZENITH_DEGREES)
        .with_columns(diff=pl.col("cams") - pl.col("era5"))
        .with_columns(month=(pl.col("hour") - pl.duration(minutes=30)).dt.truncate("1mo"))
        .sort("site", "hour")
        .pipe(_in_both_years)
    )


def _in_both_years(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the substations with daylight hours in both years.

    A substation with no year A rows has nothing to fit its correction on.
    """
    both = frame.group_by("site").agg(pl.col("year").n_unique().alias("n")).filter(pl.col("n") == 2)
    return frame.join(both.select("site"), on="site")


def design(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return the design matrix with an intercept column first."""
    return np.column_stack([np.ones(frame.height), *[frame[c].to_numpy() for c in columns]])


def corrected_residuals(*, train: pl.DataFrame, test: pl.DataFrame, arm: str) -> np.ndarray:
    """Fit the arm on `train` and return the test residuals left after its correction."""
    if arm == "none":
        return test["residual"].to_numpy()
    columns = ARMS[arm]
    beta, *_ = np.linalg.lstsq(
        design(frame=train, columns=columns), train["residual"].to_numpy(), rcond=None
    )
    return test["residual"].to_numpy() - design(frame=test, columns=columns) @ beta


def score_frame(*, fit_year: str, score_year: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the score year's rows with each arm's remaining residual, fitted per substation."""
    outputs = []
    for site in frame["site"].unique(maintain_order=True):
        train = frame.filter((pl.col("site") == site) & (pl.col("year") == fit_year))
        test = frame.filter((pl.col("site") == site) & (pl.col("year") == score_year))
        columns = {arm: corrected_residuals(train=train, test=test, arm=arm) for arm in ARMS}
        outputs.append(
            test.select("site", "hour", "month").with_columns(
                **{arm: pl.Series(values) for arm, values in columns.items()}
            )
        )
    return pl.concat(outputs)


def statistics(*, scored: pl.DataFrame) -> dict[str, float]:
    """Return the share removed, the partial R squared, and the MAE gain, pooled."""
    sse = {arm: float((scored[arm] ** 2).sum()) for arm in ARMS}
    mae = {arm: float(np.abs(scored[arm].to_numpy()).mean()) for arm in ARMS}
    return {
        "share_removed": 1.0 - sse["era5+cams"] / sse["none"],
        "share_removed_shift": 1.0 - sse["shift"] / sse["none"],
        "share_removed_era5": 1.0 - sse["era5"] / sse["none"],
        "share_removed_cams": 1.0 - sse["cams"] / sse["none"],
        "partial_r2": 1.0 - sse["era5+cams"] / sse["era5"],
        "mae_gain_vs_shift": mae["shift"] - mae["era5+cams"],
        "mae_none": mae["none"],
        "mae_shift": mae["shift"],
        "mae_era5+cams": mae["era5+cams"],
    }


def month_block_bounds(
    *, scored: pl.DataFrame, names: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Bootstrap whole months (with every substation in them) and bound each statistic."""
    months = scored["month"].unique().sort().to_list()
    by_month = {
        m: scored.filter(pl.col("month") == m).drop("month", "site", "hour") for m in months
    }
    sums = {
        m: {
            **{f"sse_{arm}": float((f[arm] ** 2).sum()) for arm in ARMS},
            **{f"abs_{arm}": float(f[arm].abs().sum()) for arm in ARMS},
            "n": float(f.height),
        }
        for m, f in by_month.items()
    }
    rng = np.random.default_rng(7)
    keys = list(next(iter(sums.values())))
    table = np.array([[sums[m][k] for k in keys] for m in months])
    index = {k: i for i, k in enumerate(keys)}
    draws: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(N_RESAMPLES):
        total = table[rng.integers(0, len(months), len(months))].sum(axis=0)
        sse = {a: total[index[f"sse_{a}"]] for a in ARMS}
        n = total[index["n"]]
        values = {
            "share_removed": 1.0 - sse["era5+cams"] / sse["none"],
            "partial_r2": 1.0 - sse["era5+cams"] / sse["era5"],
            "mae_gain_vs_shift": (total[index["abs_shift"]] - total[index["abs_era5+cams"]]) / n,
        }
        for name in names:
            draws[name].append(values[name])
    tail = (100.0 - LEVEL) / 2.0
    return {
        n: (float(np.percentile(v, tail)), float(np.percentile(v, 100.0 - tail)))
        for n, v in draws.items()
    }


def main() -> None:
    """Print the numbers the write-up quotes and draw the figures."""
    frame = hourly_residuals()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(EXPERIMENT_DIR / "hourly_residuals.parquet")
    print(
        frame.group_by("year").agg(
            pl.len(),
            pl.col("site").n_unique().alias("series"),
            pl.col("hour").min().alias("first"),
            pl.col("hour").max().alias("last"),
        )
    )
    out_of_sample = score_frame(fit_year="A", score_year="B", frame=frame)
    in_year = score_frame(fit_year="B", score_year="B", frame=frame)
    oos = statistics(scored=out_of_sample)
    inyr = statistics(scored=in_year)
    print("OUT OF SAMPLE (fit A, score B):", {k: round(v, 4) for k, v in oos.items()})
    print("IN YEAR (fit B, score B):", {k: round(v, 4) for k, v in inyr.items()})
    denominator = inyr["share_removed"]
    ratio = oos["share_removed"] / denominator if denominator >= MIN_DENOMINATOR else float("nan")
    print(f"stability ratio {ratio:.3f} (in-year share {denominator:.3f})")
    names = ("share_removed", "partial_r2", "mae_gain_vs_shift")
    bounds = month_block_bounds(scored=out_of_sample, names=names)
    print(
        f"{LEVEL:.2f}% month-block intervals:",
        {k: tuple(round(x, 4) for x in v) for k, v in bounds.items()},
    )
    per_site = []
    for site in frame["site"].unique().sort():
        s = out_of_sample.filter(pl.col("site") == site)
        per_site.append({"site": site, **statistics(scored=s)})
    per_site_frame = pl.DataFrame(per_site).sort(pl.col("site").str.slice(1).cast(pl.Int32))
    print(
        per_site_frame.select(
            "site",
            "share_removed",
            "share_removed_shift",
            "partial_r2",
            "mae_none",
            "mae_era5+cams",
        ).with_columns(pl.exclude("site").round(3))
    )
    print(per_site_frame.select("site", "share_removed", "partial_r2", "mae_none").to_dicts())
    print(
        "substations worse after correction:",
        per_site_frame.filter(pl.col("share_removed") < 0)["site"].to_list(),
    )
    print(
        "median uncorrected MAE without P2:",
        per_site_frame.filter(pl.col("site") != "P2")["mae_none"].median(),
    )
    print(
        "max partial R2:",
        per_site_frame.sort("partial_r2")[-1].select("site", "partial_r2").to_dicts(),
    )
    for arm in ARMS:
        print(f"MAE {arm}: {np.abs(out_of_sample[arm].to_numpy()).mean():.3f}")
    print("leave-one-substation-out share removed:")
    for site in frame["site"].unique().sort():
        s = out_of_sample.filter(pl.col("site") != site)
        print(f"  without {site}: {statistics(scored=s)['share_removed']:.3f}")
    print("median per-substation share removed:", per_site_frame["share_removed"].median())
    long = per_site_frame.select(
        "site",
        era5="share_removed_era5",
        cams="share_removed_cams",
        both="share_removed",
        shift="share_removed_shift",
    ).unpivot(index="site", variable_name="arm", value_name="share")
    order = per_site_frame.sort("share_removed", descending=True)["site"].to_list()
    colours = {
        "shift": "#999999",
        "era5": "#444444",
        "cams": ocf.DATA_BLUE,
        "both": ocf.BRAND_ORANGE,
    }
    chart = (
        alt.Chart(long.with_columns(share=pl.col("share") * 100.0))
        .mark_circle(size=70, opacity=0.9)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X(
                "share:Q", title="Share of the next year's daylight squared residual removed (%)"
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=4),
            ),
        )
        .properties(
            width=520,
            height=300,
            title="A fixed correction removes little of the next year's daytime error",
        )
    )
    chart.save(FIGURES_DIR / "figure_1.svg")
    chart.save(FIGURES_DIR / "figure_1.png", scale_factor=2)


if __name__ == "__main__":
    main()
