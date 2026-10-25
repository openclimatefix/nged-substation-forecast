"""Stage 1 of experiment 00347: power against ERA5 wind, and the direction curve per farm.

Plots (as tables for `e00347_charts.py`) observed power against ERA5 100 m wind speed with binned
medians and 5% to 95% bands, by calendar year and by direction sector. Then fits an all-direction
monotone curve of power on speed and estimates how observed power departs from it as a function of
ERA5 direction, using only half-hours whose fitted power lies between 40% and 90% of the farm's
top level. The departure is a von Mises kernel ratio of summed observed to summed fitted power, with
month-block bootstrap bands.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_stage1.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json  # noqa: E402
from typing import Final  # noqa: E402

import e00347_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

SPEED_BIN_M_S: Final[float] = 0.5
MAX_SPEED_M_S: Final[float] = 25.0
MIN_BIN_COUNT: Final[int] = 30
WINDOW: Final[tuple[float, float]] = (0.40, 0.90)
SUB_WINDOWS: Final[tuple[tuple[float, float], ...]] = ((0.40, 0.65), (0.65, 0.90))
KAPPA: Final[float] = 10.0
"""Von Mises kernel concentration (a standard deviation of about 18 degrees)."""
GRID_DEG: Final[np.ndarray] = np.arange(360)
BOOTSTRAPS: Final[int] = 300
SPEED_RANGES: Final[tuple[tuple[float, float], ...]] = (
    (0.0, 3.0),
    (3.0, 6.0),
    (6.0, 9.0),
    (9.0, 12.0),
    (12.0, 15.0),
    (15.0, 99.0),
)
DENSITY_Y_BINS: Final[int] = 24
MIN_DENSITY_COUNT: Final[int] = 5


def monotone_curve(*, speed: np.ndarray, target: np.ndarray) -> PchipInterpolator:
    """Return a monotone curve of mean power on speed: 0.5 m/s bin means made non-decreasing."""
    edges = np.arange(0.0, MAX_SPEED_M_S + SPEED_BIN_M_S, SPEED_BIN_M_S)
    index = np.clip(np.digitize(speed, edges) - 1, 0, edges.size - 2)
    count = np.bincount(index, minlength=edges.size - 1)
    total = np.bincount(index, weights=target, minlength=edges.size - 1)
    centres = edges[:-1] + SPEED_BIN_M_S / 2
    usable = count >= MIN_BIN_COUNT
    means = np.maximum.accumulate(total[usable] / count[usable])
    return PchipInterpolator(
        np.concatenate([[0.0], centres[usable], [MAX_SPEED_M_S + 5.0]]),
        np.concatenate([[means[0] * 0.0], means, [means[-1]]]),
        extrapolate=False,
    )


def kernel_matrix() -> np.ndarray:
    """Return the (360, 360) circular von Mises kernel on a 1 degree grid."""
    difference = np.radians(GRID_DEG[:, None] - GRID_DEG[None, :])
    kernel = np.exp(KAPPA * (np.cos(difference) - 1.0))
    return kernel / kernel.sum(axis=1, keepdims=True)


def direction_ratio(
    *,
    speed: np.ndarray,
    direction_rad: np.ndarray,
    target: np.ndarray,
    curve: PchipInterpolator,
    top: float,
    window: tuple[float, float],
    kernel: np.ndarray,
) -> np.ndarray:
    """Return kernel-smoothed summed observed over summed fitted power on the 1 degree grid.

    Only rows whose fitted power lies in `window` (shares of `top`) enter. The ratio is divided by
    its all-direction value, so the curve is 1 on average (a window picked by the fitted power
    otherwise leaves a level offset).
    """
    fitted = np.nan_to_num(curve(speed))
    inside = (fitted >= window[0] * top) & (fitted <= window[1] * top)
    degree = (np.degrees(direction_rad[inside]).round().astype(np.int64)) % 360
    observed = np.bincount(degree, weights=target[inside], minlength=360)
    expected = np.bincount(degree, weights=fitted[inside], minlength=360)
    overall = observed.sum() / max(expected.sum(), 1e-9)
    return (kernel @ observed) / np.maximum(kernel @ expected, 1e-9) / overall


def dip_summary(*, ratio: np.ndarray) -> dict[str, float]:
    """Return the depth, centre (degrees), and full width at half depth of the deepest dip."""
    low = int(np.argmin(ratio))
    depth = float(1.0 - ratio[low])
    level = 1.0 - depth / 2
    below = ratio < level
    width = 0
    for direction in (1, -1):
        step = 0
        while below[(low + direction * (step + 1)) % 360] and step < 359:
            step += 1
        width += step
    return {"depth": depth, "centre_deg": float(GRID_DEG[low]), "width_deg": float(width + 1)}


def local_minima(*, ratio: np.ndarray, count: int = 3) -> list[dict[str, float]]:
    """Return up to `count` circular local minima of `ratio`, deepest first, 40 degrees apart."""
    padded = np.concatenate([ratio[-90:], ratio, ratio[:90]])
    peaks, _ = find_peaks(-padded, distance=40)
    peaks = peaks[(peaks >= 90) & (peaks < 450)] - 90
    order = peaks[np.argsort(ratio[peaks])][:count]
    return [{"centre_deg": float(p), "depth": float(1.0 - ratio[p])} for p in order]


def circular_difference(*, a: float, b: float) -> float:
    """Return the signed angular difference `a - b` in degrees, in -180 to 180."""
    return float((a - b + 180.0) % 360.0 - 180.0)


def main() -> None:  # noqa: PLR0915
    """Compute and save Stage 1 tables and print the numbers a write-up will quote."""
    farms = common.load_farms()
    kernel = kernel_matrix()
    rng = np.random.default_rng(0)
    density_rows, stat_rows, direction_rows, split_rows = [], [], [], []
    report = ["STAGE 1: data analysis (training span: power record start to 2024-03; kept rows)"]
    summary: dict[str, dict] = {}
    for label, farm in farms.items():
        kept_all = ~farm.idle_mask
        train = farm.train
        kept = ~train.idle_mask
        speed, direction, target = (
            train.speed_100m[kept],
            train.direction_rad[kept],
            train.target[kept],
        )
        curve = monotone_curve(speed=speed, target=target)
        grid = np.linspace(0.0, 24.0, 481)
        top = float(np.nanmax(curve(grid)))
        report.append(
            f"{label}: rows {kept.sum()} kept of {train.speed_100m.size}; "
            f"top level of the monotone curve = {top:.4f} of effective capacity"
        )
        # Density, medians and bands over all kept rows (training and hold-out), shares of top.
        all_speed, all_target = farm.speed_100m[kept_all], farm.target[kept_all] / top
        all_years = farm.time[kept_all].astype("datetime64[Y]").astype(int) + 1970
        all_sector = common.sector_index(direction_rad=farm.direction_rad[kept_all])
        speed_bin = np.clip(
            (all_speed / SPEED_BIN_M_S).astype(int), 0, int(MAX_SPEED_M_S / SPEED_BIN_M_S) - 1
        )
        y_bin = np.clip((all_target / 1.2 * DENSITY_Y_BINS).astype(int), 0, DENSITY_Y_BINS - 1)
        y_bin = np.where(all_target < 0, 0, y_bin)
        counts = np.zeros((int(MAX_SPEED_M_S / SPEED_BIN_M_S), DENSITY_Y_BINS), dtype=np.int64)
        np.add.at(counts, (speed_bin, y_bin), 1)
        for i, j in zip(*np.nonzero(counts >= MIN_DENSITY_COUNT), strict=True):
            density_rows.append(
                {
                    "farm": label,
                    "speed": (i + 0.5) * SPEED_BIN_M_S,
                    "power": (j + 0.5) * 1.2 / DENSITY_Y_BINS,
                    "count": int(counts[i, j]),
                }
            )
        groups = {("all", -1): np.ones(all_speed.size, bool)}
        for year in np.unique(all_years):
            groups[("year", int(year))] = all_years == year
        for sector in range(common.SECTOR_COUNT):
            groups[("sector", sector)] = all_sector == sector
        for (kind, key), selection in groups.items():
            for b in np.unique(speed_bin[selection]):
                values = all_target[selection & (speed_bin == b)]
                if values.size < MIN_BIN_COUNT:
                    continue
                q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
                stat_rows.append(
                    {
                        "farm": label,
                        "kind": kind,
                        "key": key,
                        "speed": (b + 0.5) * SPEED_BIN_M_S,
                        "q05": float(q05),
                        "median": float(q50),
                        "q95": float(q95),
                        "n": int(values.size),
                    }
                )
        # Counts of half-hours per speed range and sector (kept training rows).
        sector_train = common.sector_index(direction_rad=direction)
        report.append(
            f"{label}: kept training half-hours by ERA5 speed range (rows) and direction "
            "sector (columns)"
        )
        report.append("  speed range     " + " ".join(f"{n:>6}" for n in common.SECTOR_NAMES))
        for low, high in SPEED_RANGES:
            row = [
                int(((speed >= low) & (speed < high) & (sector_train == s)).sum()) for s in range(8)
            ]
            report.append(
                f"  {low:>4.0f}-{min(high, 99):<3.0f} m/s   " + " ".join(f"{v:>6}" for v in row)
            )
        # Direction curve with month-block bootstrap.
        month = train.month[kept]
        ratio = direction_ratio(
            speed=speed,
            direction_rad=direction,
            target=target,
            curve=curve,
            top=top,
            window=WINDOW,
            kernel=kernel,
        )
        base = dip_summary(ratio=ratio)
        minima = local_minima(ratio=ratio)
        labels = np.unique(month)
        index_by_month = [np.flatnonzero(month == m) for m in labels]
        draws, bands = [], []
        for _ in range(BOOTSTRAPS):
            chosen = rng.integers(0, labels.size, size=labels.size)
            index = np.concatenate([index_by_month[k] for k in chosen])
            c = monotone_curve(speed=speed[index], target=target[index])
            t = float(np.nanmax(c(grid)))
            r = direction_ratio(
                speed=speed[index],
                direction_rad=direction[index],
                target=target[index],
                curve=c,
                top=t,
                window=WINDOW,
                kernel=kernel,
            )
            bands.append(r)
            draws.append(dip_summary(ratio=r))
        bands_arr = np.array(bands)
        lo, hi = np.quantile(bands_arr, [0.025, 0.975], axis=0)
        for deg in GRID_DEG:
            direction_rows.append(  # noqa: PERF401
                {
                    "farm": label,
                    "direction_deg": int(deg),
                    "ratio": float(ratio[deg]),
                    "lo": float(lo[deg]),
                    "hi": float(hi[deg]),
                }
            )
        depth_ci = np.quantile([d["depth"] for d in draws], [0.025, 0.975])
        width_ci = np.quantile([d["width_deg"] for d in draws], [0.025, 0.975])
        centre_dev = np.array(
            [circular_difference(a=d["centre_deg"], b=base["centre_deg"]) for d in draws]
        )
        centre_ci = np.quantile(centre_dev, [0.025, 0.975]) + base["centre_deg"]
        # Speed dependence: depth at the deepest dip's centre in each half of the window.
        for window in SUB_WINDOWS:
            sub_ratio = direction_ratio(
                speed=speed,
                direction_rad=direction,
                target=target,
                curve=curve,
                top=top,
                window=window,
                kernel=kernel,
            )
            boot = []
            for _ in range(100):
                chosen = rng.integers(0, labels.size, size=labels.size)
                index = np.concatenate([index_by_month[k] for k in chosen])
                c = monotone_curve(speed=speed[index], target=target[index])
                t = float(np.nanmax(c(grid)))
                boot.append(
                    direction_ratio(
                        speed=speed[index],
                        direction_rad=direction[index],
                        target=target[index],
                        curve=c,
                        top=t,
                        window=window,
                        kernel=kernel,
                    )
                )
            boot_arr = np.array(boot)
            centre = int(base["centre_deg"])
            split_rows.append(
                {
                    "farm": label,
                    "window": f"{window[0]:.0%} to {window[1]:.0%}",
                    "depth": float(1 - sub_ratio[centre]),
                    "lo": float(np.quantile(1 - boot_arr[:, centre], 0.025)),
                    "hi": float(np.quantile(1 - boot_arr[:, centre], 0.975)),
                }
            )
        lower, upper = split_rows[-2], split_rows[-1]
        steep = np.gradient(
            np.log(np.maximum(np.nan_to_num(curve(grid)), 1e-6)), np.log(np.maximum(grid, 1e-3))
        )
        in_window = (np.nan_to_num(curve(grid)) >= WINDOW[0] * top) & (
            np.nan_to_num(curve(grid)) <= WINDOW[1] * top
        )
        slope = float(np.mean(steep[in_window]))
        window_rows = int(
            (
                (np.nan_to_num(curve(speed)) >= WINDOW[0] * top)
                & (np.nan_to_num(curve(speed)) <= WINDOW[1] * top)
            ).sum()
        )
        report += [
            (
                f"{label}: direction curve from {window_rows} half-hours with fitted power at 40% "
                "to 90% of the top level; "
                f"median ratio over the circle = {float(np.median(ratio)):.3f}"
            ),
            (
                f"{label}: deepest dip depth {base['depth']:.3f} (95% band {depth_ci[0]:.3f} to "
                f"{depth_ci[1]:.3f}), "
                f"centre {base['centre_deg']:.0f} deg ({centre_ci[0]:.0f} to {centre_ci[1]:.0f}), "
                f"width at half depth {base['width_deg']:.0f} deg ({width_ci[0]:.0f} to "
                f"{width_ci[1]:.0f})"
            ),
            (
                f"{label}: direction curve ranges from {ratio.min():.3f} to {ratio.max():.3f}, so "
                "power at the same ERA5 speed "
                f"is {ratio.max() / ratio.min():.2f} times higher in the best direction than in "
                "the worst"
            ),
            f"{label}: three deepest local minima: "
            + "; ".join(f"{m['centre_deg']:.0f} deg depth {m['depth']:.3f}" for m in minima),
            (
                f"{label}: mean log-slope of power on speed in the window = {slope:.2f}; "
                f"so the dip equals a speed error of {base['depth'] / slope:.3f} (relative) if it "
                "were all speed error"
            ),
            (
                f"{label}: dip depth at the deepest centre in the lower half of the window (40% "
                "to 65%) = "
                f"{lower['depth']:.3f} (95% {lower['lo']:.3f} to {lower['hi']:.3f}); "
                f"upper half (65% to 90%) = {upper['depth']:.3f} (95% "
                f"{upper['lo']:.3f} to {upper['hi']:.3f}). "
                "A speed error makes the dip deeper where the curve is steeper (lower half); a "
                "loss in proportion to power keeps it equal."
            ),
        ]
        summary[label] = {
            "top": top,
            "ratio": ratio.tolist(),
            "dip": base,
            "minima": minima,
            "log_slope": slope,
            "depth_ci": depth_ci.tolist(),
            "width_ci": width_ci.tolist(),
            "centre_ci": centre_ci.tolist(),
        }
    report.append(
        "Caution: the dip is a departure of power from the all-direction curve at ERA5's "
        "wind speed. It "
        "absorbs a wake loss, a direction-dependent ERA5 speed error (terrain or a 25 km cell that "
        "misrepresents the wind at the turbines), and any direction-dependent curtailment. "
        "The plant model "
        "calls the sum 'plant loss'."
    )
    work = common.DATA_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(density_rows).write_parquet(work / "stage1_density.parquet")
    pl.DataFrame(stat_rows).write_parquet(work / "stage1_bin_stats.parquet")
    pl.DataFrame(direction_rows).write_parquet(work / "stage1_direction.parquet")
    pl.DataFrame(split_rows).write_parquet(work / "stage1_split.parquet")
    (common.DATA_DIR / "stage1_summary.json").write_text(json.dumps(summary, indent=1))
    common.write_report(name="stage1_report.txt", lines=report)
    print("\n".join(report))


if __name__ == "__main__":
    main()
