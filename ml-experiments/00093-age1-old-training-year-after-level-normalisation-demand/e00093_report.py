"""Score experiment 00093's arms on the validation year and print every number the page quotes.

Reads the forecasts `e00093_fit.py --stage predict` wrote, scores each row with the quantile score
(twice the mass-weighted mean pinball loss over the 11 levels, in points of capacity), and averages
to one value per series and calendar month. An interval resamples whole calendar months (50,000
draws, all series together), so neighbouring runs that share weather stay together. The three
planned contrasts use the Bonferroni level for a family of three (98.33%); every other interval is
at 95% and is descriptive.

    uv run python ml-experiments/00093-*/e00093_report.py
"""

import json
from typing import Final

import e00093_common as common
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 50_000
BONFERRONI_LEVEL: Final[float] = 100.0 - 5.0 / 3.0
SEED: Final[int] = 93
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("old1_raw", "recent_raw"),
    "C2": ("old1_all", "recent_all"),
}
POST_HOC: Final[dict[str, tuple[str, str]]] = {
    "R1 (2022-07 start), raw": ("old2_raw", "recent_raw"),
    "R2 (2022-07 start), all-hours level": ("old2_all", "recent_all"),
    "H1 half-hourly level, 2021-07 start": ("old1_hh", "recent_hh"),
}


def load_arm(*, arm: str) -> pl.DataFrame:
    """Return one arm's forecasts for all series, sorted by key."""
    frames = [pl.read_parquet(path) for path in sorted((common.PREDS_DIR / arm).glob("*.parquet"))]
    sites = [
        path.name.rsplit("_c", 1)[0] for path in sorted((common.PREDS_DIR / arm).glob("*.parquet"))
    ]
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


def report_contrasts(*, grids: dict[str, dict]) -> dict:
    """Print and return the planned contrasts, the repeats, and the post hoc contrasts."""
    cells = grids["s11"]
    out: dict[str, dict] = {"planned": {}, "post_hoc": {}, "nra": {}}
    print("\nPlanned contrasts (points; positive = old year worse)")
    for name, (treat, ref) in CONTRASTS.items():
        row = contrast(treat=treat, ref=ref, level=BONFERRONI_LEVEL, source=cells)
        print(
            f"{name} {treat} - {ref}: {row['estimate']:.3f} ({row['lo']:.3f} to {row['hi']:.3f}) "
            f"at {BONFERRONI_LEVEL:.2f}%"
        )
        out["planned"][name] = row
    c1 = cells["old1_raw"] - cells["recent_raw"]
    c2 = cells["old1_all"] - cells["recent_all"]
    est, lo, hi = interval(cells=c1 - c2, level=BONFERRONI_LEVEL)
    both = np.isfinite(c1) & np.isfinite(c2)
    share = float(np.mean((c1 - c2)[both]) / np.mean(c1[both]))
    print(f"C3 C1 - C2: {est:.3f} ({lo:.3f} to {hi:.3f}); share of C1 removed {share:.2f}")
    out["planned"]["C3"] = {"estimate": est, "lo": lo, "hi": hi, "share_removed": share}
    print("\nPost hoc and repeat contrasts (95%)")
    for name, (treat, ref) in POST_HOC.items():
        row = contrast(treat=treat, ref=ref, level=95.0, source=cells)
        print(f"{name}: {row['estimate']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})")
        out["post_hoc"][name] = row
    c1b = cells["old2_raw"] - cells["recent_raw"]
    c2b = cells["old2_all"] - cells["recent_all"]
    est, lo, hi = interval(cells=c1b - c2b, level=95.0)
    both_b = np.isfinite(c1b) & np.isfinite(c2b)
    share_b = float(np.mean((c1b - c2b)[both_b]) / np.mean(c1b[both_b]))
    print(
        f"R3 (2022-07 start) C1 - C2: {est:.3f} ({lo:.3f} to {hi:.3f}); share removed {share_b:.2f}"
    )
    out["post_hoc"]["R3 (2022-07 start) C1 - C2"] = {
        "estimate": est,
        "lo": lo,
        "hi": hi,
        "share_removed": share_b,
    }
    worst = int(np.nanargmax(np.nanmean(c1, axis=0)))
    kept = {arm: np.delete(values, worst, axis=1) for arm, values in cells.items()}
    print("\nWithout the series with the largest C1 (post hoc, 95%)")
    for name, (treat, ref) in CONTRASTS.items():
        row = contrast(treat=treat, ref=ref, level=95.0, source=kept)
        print(f"{name}: {row['estimate']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})")
        out["post_hoc"][f"{name} without the largest-C1 series"] = row
    d1 = kept["old1_raw"] - kept["recent_raw"]
    d2 = kept["old1_all"] - kept["recent_all"]
    share_k = float(np.nanmean(d1 - d2) / np.nanmean(d1))
    print(f"share of C1 removed without that series: {share_k:.2f}")
    out["post_hoc"]["share removed without the largest-C1 series"] = {"share_removed": share_k}
    print("\nNormal-arrangement rows only, C1 and C2 (95%)")
    for name, (treat, ref) in CONTRASTS.items():
        row = contrast(treat=treat, ref=ref, level=95.0, source=grids["nra"])
        print(f"{name}: {row['estimate']:.3f} ({row['lo']:.3f} to {row['hi']:.3f})")
        out["nra"][name] = row
    return out


def report_by_series(*, grids: dict[str, dict], sites: list[str]) -> dict:
    """Print and return C1 and C2 for each series."""
    cells = grids["s11"]
    c1 = cells["old1_raw"] - cells["recent_raw"]
    c2 = cells["old1_all"] - cells["recent_all"]
    print("\nBy series: C1 and C2 (points)")
    out = {}
    for j, site in enumerate(sites):
        d1, d2 = float(np.nanmean(c1[:, j])), float(np.nanmean(c2[:, j]))
        recent = float(np.nanmean(cells["recent_raw"][:, j]))
        print(f"{site}: C1 {d1:+.3f}  C2 {d2:+.3f}  recent raw {recent:.3f}")
        out[site] = {"c1": d1, "c2": d2, "recent_raw": recent}
    return out


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
    arms = sorted(path.name for path in common.PREDS_DIR.iterdir())
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
        **report_contrasts(grids=grids),
        "by_series": report_by_series(grids=grids, sites=sites),
        "tails": report_tails(frames=frames),
    }
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (common.RESULTS_DIR / "report.json").write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
