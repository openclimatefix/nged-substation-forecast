"""Score experiment 00095's arms on the validation year and print every number the page quotes.

Reads the forecasts `e00095_fit.py --stage predict` wrote, plus experiment 00093's control-member
forecasts (read as they are), scores each row with the quantile score (twice the mass-weighted
mean pinball loss over the 11 levels, in points of capacity), and averages to one value per series
and calendar month. An interval resamples whole calendar months (50,000 draws, all series
together), so neighbouring runs that share weather stay together. The five planned contrasts use
the Bonferroni level for a family of five (99%); every other interval is at 95% and is descriptive.

    uv run python ml-experiments/00095-*/e00095_report.py
"""

import json
from typing import Final

import e00095_common as common
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 50_000
BONFERRONI_LEVEL: Final[float] = 99.0
SEED: Final[int] = 95
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
AGE1_ARMS: Final[dict[str, str]] = {
    "old1_nwp": "old1_raw",
    "old2_nwp": "old2_raw",
    "recent_nwp": "recent_raw",
}
"""Each experiment 93 control-member arm experiment 95 reuses, and the folder it is read from."""
NEW_ARMS: Final[tuple[str, ...]] = (
    "old1_era5",
    "old2_era5",
    "mid_era5",
    "recent_era5",
    "mid_nwp",
)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1 (control member, 2021-07 minus 2024-07)": ("old1_nwp", "recent_nwp"),
    "C2 (ERA5, 2021-07 minus 2024-07)": ("old1_era5", "recent_era5"),
    "C4 (9 km control member, 2023-07 minus 2024-07)": ("mid_nwp", "recent_nwp"),
    "C5 (18 km control member, 2021-07 minus 2022-07)": ("old1_nwp", "old2_nwp"),
}
POST_HOC: Final[dict[str, tuple[str, str]]] = {
    "ERA5, 2022-07 minus 2024-07": ("old2_era5", "recent_era5"),
    "ERA5, 2023-07 minus 2024-07": ("mid_era5", "recent_era5"),
    "ERA5, 2021-07 minus 2023-07": ("old1_era5", "mid_era5"),
    "Control member, 2022-07 minus 2024-07": ("old2_nwp", "recent_nwp"),
    "2023-07 year, control member minus ERA5": ("mid_nwp", "mid_era5"),
    "2024-07 year, control member minus ERA5": ("recent_nwp", "recent_era5"),
}
D8: Final[str] = "D8"
"""The series that carries most of experiment 93's penalty."""


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return one arm's forecasts for all series, sorted by key."""
    folder = common.AGE1_PREDS_DIR / AGE1_ARMS[arm] if arm in AGE1_ARMS else common.PREDS_DIR / arm
    paths = sorted(folder.glob("*.parquet"))
    frames = [pl.read_parquet(path) for path in paths]
    sites = [path.name.rsplit("_c", 1)[0] for path in paths]
    frame = pl.concat([f.with_columns(site=pl.lit(s)) for f, s in zip(frames, sites, strict=True)])
    return frame.sort("site", *common.KEYS)


def scores(*, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-row 11-level and 9-level quantile scores, in points of capacity."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
    out = []
    for columns, levels in (
        (slice(None), common.LEVELS),
        (common.INNER_NINE, common.LEVELS[common.INNER_NINE]),
    ):
        q = quantiles[:, columns]
        masses = np.diff(np.concatenate([[0.0], (levels[:-1] + levels[1:]) / 2, [1.0]]))
        diff = actual[:, None] - q
        pin = np.where(diff >= 0, levels * diff, (levels - 1) * diff)
        out.append(200.0 * (pin * masses).sum(axis=1))
    return out[0], out[1]


def month_table(*, frame: pl.DataFrame, values: np.ndarray, name: str) -> pl.DataFrame:
    """Return the mean of `values` for each series and calendar month."""
    months = frame["valid_time"].dt.strftime("%Y-%m")
    return (
        frame.select("site")
        .with_columns(month=months, value=pl.Series(values))
        .group_by("site", "month")
        .agg(pl.col("value").mean().alias(name), pl.len().alias("n"))
    )


def interval(*, cells: np.ndarray, level: float) -> tuple[float, float, float]:
    """Return the mean over series and months, with a month-resampling interval.

    Args:
        cells: Array (months, series) of per-cell means; NaN cells are ignored per draw.
        level: Coverage in percent.

    Returns:
        The point estimate (mean of all cells) and the lower and upper percentiles.
    """
    rng = np.random.default_rng(SEED)
    n = cells.shape[0]
    draws = rng.integers(0, n, size=(N_DRAWS, n))
    means = np.nanmean(cells[draws].reshape(N_DRAWS, -1), axis=1)
    tail = (100.0 - level) / 2.0
    lower, upper = np.percentile(means, [tail, 100.0 - tail])
    return float(np.nanmean(cells)), float(lower), float(upper)


def cell_grid(
    *, values: np.ndarray, frame: pl.DataFrame, months: list[str], sites: list[str]
) -> np.ndarray:
    """Return the per-month, per-series means of `values` as an array (months, series)."""
    table = month_table(frame=frame, values=values, name="v")
    grid = np.full((len(months), len(sites)), np.nan)
    for row in table.iter_rows(named=True):
        grid[months.index(row["month"]), sites.index(row["site"])] = row["v"]
    return grid


def contrast(
    *, treat: str, ref: str, level: float, source: dict[str, np.ndarray]
) -> dict[str, float]:
    """Return the mean difference `treat - ref` with its month-resampling interval."""
    est, lo, hi = interval(cells=source[treat] - source[ref], level=level)
    return {"estimate": est, "lo": lo, "hi": hi, "level": level}


def report_absolute(*, frames: dict[str, pl.DataFrame], grids: dict[str, dict]) -> dict:
    """Print and return each arm's absolute score."""
    print("\nAbsolute scores (11-level, points; 9-level; normal-arrangement rows only)")
    out = {}
    for arm in frames:
        est, lo, hi = interval(cells=grids["s11"][arm], level=95.0)
        nine, nra = float(np.nanmean(grids["s9"][arm])), float(np.nanmean(grids["nra"][arm]))
        print(f"{arm:11s} {est:.3f} ({lo:.3f} to {hi:.3f})  9-level {nine:.3f}  NRA rows {nra:.3f}")
        out[arm] = {"score11": est, "lo": lo, "hi": hi, "score9": nine, "score11_nra": nra}
    return out


def share_interval(*, numerator: np.ndarray, denominator: np.ndarray, level: float) -> tuple:
    """Return mean `numerator` over mean `denominator` with a month-resampling interval."""
    rng = np.random.default_rng(SEED)
    n = numerator.shape[0]
    draws = rng.integers(0, n, size=(N_DRAWS, n))
    top = np.nanmean(numerator[draws].reshape(N_DRAWS, -1), axis=1)
    bottom = np.nanmean(denominator[draws].reshape(N_DRAWS, -1), axis=1)
    tail = (100.0 - level) / 2.0
    lower, upper = np.percentile(top / bottom, [tail, 100.0 - tail])
    return float(np.nanmean(numerator) / np.nanmean(denominator)), float(lower), float(upper)


def contrast_block(
    *, cells: dict[str, np.ndarray], level: float, label: str, post_hoc: bool = True
) -> dict:
    """Print and return every contrast, C3, and the share of C1 that survives on ERA5."""
    out: dict[str, dict] = {}
    print(f"\n{label} (points; positive = first arm worse), {level:.2f}% intervals")
    for name, (treat, ref) in {**CONTRASTS, **(POST_HOC if post_hoc else {})}.items():
        row = contrast(treat=treat, ref=ref, level=level, source=cells)
        print(f"{name}: {row['estimate']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})")
        out[name] = row
    c1 = cells["old1_nwp"] - cells["recent_nwp"]
    c2 = cells["old1_era5"] - cells["recent_era5"]
    est, lo, hi = interval(cells=c1 - c2, level=level)
    share, s_lo, s_hi = share_interval(numerator=c2, denominator=c1, level=level)
    print(
        f"C3 (C1 - C2): {est:.3f} ({lo:.3f} to {hi:.3f}); share of C1 on ERA5 {share:.2f} "
        f"({s_lo:.2f} to {s_hi:.2f})"
    )
    out["C3 (C1 minus C2)"] = {"estimate": est, "lo": lo, "hi": hi, "level": level}
    out["share of C1 surviving on ERA5 (C2 over C1)"] = {"estimate": share, "lo": s_lo, "hi": s_hi}
    return out


def report_contrasts(*, grids: dict[str, dict], sites: list[str]) -> dict:
    """Print and return the planned contrasts, the post hoc contrasts, and the repeats."""
    cells = grids["s11"]
    out: dict[str, dict] = {
        "planned": contrast_block(
            cells=cells, level=BONFERRONI_LEVEL, label="Planned", post_hoc=False
        )
    }
    out["all_95"] = contrast_block(cells=cells, level=95.0, label="All series")
    j = sites.index(D8)
    kept = {arm: np.delete(values, j, axis=1) for arm, values in cells.items()}
    out["without_d8"] = contrast_block(cells=kept, level=95.0, label=f"Without {D8} (post hoc)")
    out["nra"] = contrast_block(
        cells=grids["nra"], level=95.0, label="Normal-arrangement rows only"
    )
    return out


def report_by_series(*, grids: dict[str, dict], sites: list[str]) -> dict:
    """Print and return the old-minus-recent gap per series, on each weather product."""
    cells = grids["s11"]
    c1 = cells["old1_nwp"] - cells["recent_nwp"]
    c2 = cells["old1_era5"] - cells["recent_era5"]
    print("\nBy series: C1 (control member) and C2 (ERA5), points")
    out = {}
    for j, site in enumerate(sites):
        d1, d2 = float(np.nanmean(c1[:, j])), float(np.nanmean(c2[:, j]))
        recent = float(np.nanmean(cells["recent_nwp"][:, j]))
        print(f"{site}: C1 {d1:+.3f}  C2 {d2:+.3f}  recent control member {recent:.3f}")
        out[site] = {"c1": d1, "c2": d2, "recent_nwp": recent}
    return out


def report_d8() -> dict:
    """Inspect the largest-penalty series' observed power for a step change or a meter issue.

    Returns:
        For each calendar month from 2021-07 to 2026-06, the mean power over capacity, the share of
        half-hours observed, the share of repeated consecutive values, and the largest day-to-day
        change of the daily mean; plus the largest jump between consecutive 60-day mean levels.
    """
    series = common.series_table().filter(pl.col("site") == D8)
    arrays = common.load_series_arrays(series=series)[D8]
    frame = pl.DataFrame(
        {"valid_time": arrays.valid_time, "power": arrays.power}, nan_to_null=True
    ).filter(
        (pl.col("valid_time") >= common.WINDOWS["old1"][0])
        & (pl.col("valid_time") <= common.LAST_TRAINING_VALID.replace(year=2026, month=6, day=30))
    )
    frame = frame.with_columns(
        month=pl.col("valid_time").dt.strftime("%Y-%m"),
        repeat=(pl.col("power") == pl.col("power").shift(1)).cast(pl.Float64),
        observed=pl.col("power").is_finite().cast(pl.Float64),
    )
    table = (
        frame.group_by("month")
        .agg(
            pl.col("power").mean().alias("mean"),
            pl.col("observed").mean().alias("observed"),
            pl.col("repeat").mean().alias("repeat"),
        )
        .sort("month")
    )
    daily = (
        frame.group_by(pl.col("valid_time").dt.truncate("1d").alias("day"))
        .agg(pl.col("power").mean().alias("v"))
        .sort("day")
    )
    rolling = daily.with_columns(level=pl.col("v").rolling_mean(window_size=60)).drop_nulls()
    jump = rolling["level"].diff(n=60).drop_nulls()
    step = int(jump.abs().arg_max() or 0)
    biggest = float(jump.abs().to_numpy().max())
    ending = f"{rolling['day'][step + 60]:%Y-%m-%d}"
    print(f"\n{D8} monthly mean power over capacity, observed share, repeated-value share")
    for row in table.iter_rows(named=True):
        print(f"{row['month']}: {row['mean']:.3f}  {row['observed']:.3f}  {row['repeat']:.3f}")
    print(f"largest change of the 60-day mean level over 60 days: {biggest:.3f} ending {ending}")
    return {
        "months": table.to_dicts(),
        "largest_60_day_change": biggest,
        "ending": ending,
    }


def report_tails(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Print and return coverage and pinball loss at the tail levels."""
    print("\nTail calibration: share of actuals at or below the quantile, and pinball (points)")
    out = {}
    for arm, frame in frames.items():
        actual = frame["actual"].to_numpy().astype(np.float64)
        parts = {}
        for level in TAIL_LEVELS:
            column = f"q{round(level * 100):02d}"
            error = actual - frame[column].to_numpy().astype(np.float64)
            pin = np.maximum(level * error, (level - 1.0) * error) * 100.0
            parts[column] = {
                "coverage": float((error <= 0).mean()),
                "pinball": float(pin.mean()),
            }
        out[arm] = parts
        text = "  ".join(
            f"{c} cov {v['coverage']:.3f} pin {v['pinball']:.3f}" for c, v in parts.items()
        )
        print(f"{arm:11s} {text}")
    return out


def main() -> None:
    """Print the report and write `results/report.json`."""
    arms = [*AGE1_ARMS, *NEW_ARMS]
    frames = {arm: load_arm(arm=arm) for arm in arms}
    first = frames[arms[0]]
    for arm in arms:
        if not frames[arm].select("site", *common.KEYS).equals(first.select("site", *common.KEYS)):
            msg = f"{arm} covers different rows"
            raise ValueError(msg)
    keep = pl.Series(np.isfinite(first["actual"].to_numpy()))
    frames = {arm: f.filter(keep) for arm, f in frames.items()}
    base = frames[arms[0]]
    print(f"rows {base.height:,}; series {base['site'].n_unique()}; row set {common.ROW_SET_ID}")
    for arm, frame in frames.items():
        missing = int(frame["q50"].is_nan().sum()) + frame["q50"].null_count()
        print(f"{arm}: {missing:,} rows with a missing forecast")
    months = sorted(base["valid_time"].dt.strftime("%Y-%m").unique().to_list())
    sites = sorted(base["site"].unique().to_list())
    grids: dict[str, dict] = {"s11": {}, "s9": {}, "nra": {}}
    for arm, frame in frames.items():
        s11, s9 = scores(frame=frame)
        nra = ~frame["ara"].to_numpy()
        for key, values in (("s11", s11), ("s9", s9), ("nra", np.where(nra, s11, np.nan))):
            grids[key][arm] = cell_grid(values=values, frame=frame, months=months, sites=sites)
    result = {
        "rows": base.height,
        "series": sites,
        "absolute": report_absolute(frames=frames, grids=grids),
        **report_contrasts(grids=grids, sites=sites),
        "by_series": report_by_series(grids=grids, sites=sites),
        "tails": report_tails(frames=frames),
        "d8": report_d8(),
    }
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (common.RESULTS_DIR / "report.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
