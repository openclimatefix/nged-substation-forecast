"""Score experiment 00303's arms and read the planned contrasts.

Reads `work/final/<arm>.npy` (shape (3 seeds, scored cases, 11)), scores each arm on the quantile
score of experiment 33 (points of capacity), and reads the contrasts C1 to C8 of the plan with
month-and-seed resampled intervals at two levels: 95% (as the brief asks) and the Bonferroni level
for 8 planned contrasts (99.375%). Contrasts are `first - second`, so a negative number means the
first arm is better. Farms are not resampled.

Output: `work/score.json` and `work/rows.parquet` (per-row covariates and seed-mean scores).

    uv run --no-sync python e00303_score.py
"""

import json
import os
from collections.abc import Callable
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

FINAL_DIR = common.WORK_DIR / "final"
RESAMPLES: Final[int] = 50_000
PLANNED_CONTRASTS: Final[int] = 8
LEVELS_REPORTED: Final[tuple[float, float]] = (0.95, 1 - 0.05 / PLANNED_CONTRASTS)
ROW_SETS: Final[tuple[str, ...]] = ("all", "kept", "lead_12_36", "days_1_2")
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
ARM_ORDER: Final[tuple[str, ...]] = (
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P4T",
    "P5T",
    "P7T",
    "P3_LOFO",
    "P5_LOFO",
)
METRICS: Final[tuple[str, ...]] = ("qs", "p01", "p99", "median_abs_error", "mean_abs_error")
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "P2", "P0"),
    ("C2", "P3", "P1"),
    ("C3", "P5", "P3"),
    ("C4", "P4", "P5"),
    ("C5", "P6", "P3"),
    ("C6", "P8", "P3"),
    ("C8", "P5", "P1"),
)
EXTRA_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("x1", "P5T", "P1"),
    ("x2", "P4T", "P1"),
    ("x3", "P5T", "P5"),
    ("x4", "P8", "P1"),
    ("x5", "P7", "P5"),
    ("x6", "P3", "P0"),
    ("x7", "P6", "P1"),
    ("x8", "P3_LOFO", "P3"),
    ("x9", "P5_LOFO", "P5"),
    ("x10", "P5_LOFO", "P3_LOFO"),
)


def ordered_cases() -> pl.DataFrame:
    """Return the scored cases, farm by farm in the order `e00303_final.py` stores them."""
    sites = common.site_table()
    cases = common.cases_table(sites=sites)
    pieces = [cases.filter(pl.col("site") == row["site"]) for row in sites.iter_rows(named=True)]
    return pl.concat(pieces).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        all=pl.lit(value=True),
        kept=~pl.col("masked"),
        lead_12_36=(pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36),
        days_1_2=pl.col("lead_day").is_between(1, 2),
    )


def row_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-seed, per-row metrics (points of capacity): shape (seeds, rows)."""
    scale = common.POINTS_PER_FRACTION
    names = (*METRICS, "median_minus_mean", "below_median", "covered")
    out: dict[str, list[np.ndarray]] = {k: [] for k in names}
    for q in quantiles.astype(np.float64):
        mean_forecast = q[:, 1:10] @ common.NINE_MASSES
        out["qs"].append(common.quantile_score(quantiles=q, actual=actual) * scale)
        out["median_abs_error"].append(np.abs(actual - q[:, common.MEDIAN_INDEX]) * scale)
        out["mean_abs_error"].append(np.abs(actual - mean_forecast) * scale)
        out["median_minus_mean"].append((q[:, common.MEDIAN_INDEX] - mean_forecast) * scale)
        out["p01"].append(common.pinball_at(quantiles=q, actual=actual, column="q01") * scale)
        out["p99"].append(common.pinball_at(quantiles=q, actual=actual, column="q99") * scale)
        out["below_median"].append((actual < q[:, common.MEDIAN_INDEX]).astype(np.float64))
        out["covered"].append(common.covered(quantiles=q, actual=actual).astype(np.float64))
    return {k: np.stack(v) for k, v in out.items()}


class Resampler:
    """Month-and-seed resampling shared by every statistic, so contrasts use the same draws."""

    def __init__(self, *, month: np.ndarray, seeds: int) -> None:
        """Draw the resamples once."""
        self.month, self.months = month, np.unique(month)
        self.seeds = seeds
        rng = np.random.default_rng(0)
        self.draw_m = rng.integers(0, self.months.size, size=(RESAMPLES, self.months.size))
        self.draw_s = rng.integers(0, seeds, size=(RESAMPLES, seeds))

    def draws(self, *, values: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the resampled (sums, counts) of `values` (seeds, rows) over `keep`."""
        sums = np.zeros((self.months.size, self.seeds))
        counts = np.zeros(self.months.size)
        for i, m in enumerate(self.months):
            in_month = keep & (self.month == m)
            sums[i] = values[:, in_month].sum(axis=1)
            counts[i] = in_month.sum()
        picked = sums[self.draw_m[:, :, None], self.draw_s[:, None, :]].sum(axis=(1, 2))
        totals = counts[self.draw_m].sum(axis=1) * self.seeds
        return picked, totals

    def interval(
        self, *, values: np.ndarray, keep: np.ndarray, statistic: Callable | None = None
    ) -> dict[str, float | list[float]]:
        """Return the mean and the intervals at both levels."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        point = float(values[:, keep].mean())
        out: dict[str, float | list[float]] = {"mean": point}
        for level in LEVELS_REPORTED:
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.5f}"] = [float(low), float(high)]
        return out


def share_closed(
    *, resampler: Resampler, qs: dict[str, np.ndarray], keep: np.ndarray
) -> dict[str, float | list[float]]:
    """Return C7: the share of the P3-to-P5 gap that P7 closes, with intervals."""
    gap_p5 = resampler.draws(values=qs["P3"] - qs["P5"], keep=keep)
    gap_p7 = resampler.draws(values=qs["P3"] - qs["P7"], keep=keep)
    ratio = gap_p7[0] / np.where(np.abs(gap_p5[0]) < 1e-9, np.nan, gap_p5[0])
    point = float((qs["P3"] - qs["P7"])[:, keep].mean() / (qs["P3"] - qs["P5"])[:, keep].mean())
    out: dict[str, float | list[float]] = {"mean": point}
    for level in LEVELS_REPORTED:
        low, high = np.nanquantile(ratio, [(1 - level) / 2, 1 - (1 - level) / 2])
        out[f"{level:.5f}"] = [float(low), float(high)]
    return out


def main() -> None:
    """Score every stored arm and the planned contrasts, and write the report."""
    cases = ordered_cases()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    flags = {name: cases[name].to_numpy() for name in ROW_SETS}
    resampler = Resampler(month=cases["month"].to_numpy(), seeds=len(common.SEEDS))
    arms: dict[str, dict[str, np.ndarray]] = {}
    for name in ARM_ORDER:
        path = FINAL_DIR / f"{name}.npy"
        if path.exists():
            arms[name] = row_metrics(quantiles=np.load(path), actual=actual)
            print("scored", name, flush=True)
    report: dict = {"scores": {}, "contrasts": {}, "rows_in_set": {}, "by_lead_day": {}}
    for set_name, mask in flags.items():
        report["rows_in_set"][set_name] = int(mask.sum())
    for name, metrics in arms.items():
        report["scores"][name] = {
            row_set: {m: float(v[:, flags[row_set]].mean()) for m, v in metrics.items()}
            for row_set in ROW_SETS
        }
        report["by_lead_day"][name] = {
            str(day): float(metrics["qs"][:, (cases["lead_day"] == day).to_numpy()].mean())
            for day in range(15)
            if (cases["lead_day"] == day).any()
        }
        report["scores"][name]["by_farm_kept"] = {
            farm: float(metrics["qs"][:, flags["kept"] & (cases["site"] == farm).to_numpy()].mean())
            for farm in FARMS
        }
    for label, first, second in (*CONTRASTS, *EXTRA_CONTRASTS):
        if first not in arms or second not in arms:
            continue
        key = f"{label}: {first} - {second}"
        report["contrasts"][key] = {}
        for metric in (*METRICS, "below_median", "covered"):
            difference = arms[first][metric] - arms[second][metric]
            for row_set in ROW_SETS:
                report["contrasts"][key][f"{row_set} | {metric}"] = resampler.interval(
                    values=difference, keep=flags[row_set]
                )
        for farm in FARMS:
            site = (cases["site"] == farm).to_numpy()
            difference = arms[first]["qs"] - arms[second]["qs"]
            report["contrasts"][key][f"kept only {farm} | qs"] = resampler.interval(
                values=difference, keep=flags["kept"] & site
            )
            report["contrasts"][key][f"kept without {farm} | qs"] = resampler.interval(
                values=difference, keep=flags["kept"] & ~site
            )
    if all(a in arms for a in ("P3", "P5", "P7")):
        report["contrasts"]["C7: share of the P3 - P5 gap that P7 closes"] = share_closed(
            resampler=resampler, qs={a: arms[a]["qs"] for a in ("P3", "P5", "P7")}, keep=flags["kept"]
        )
    columns = {
        "site": cases["site"],
        "valid_time": cases["valid_time"],
        "lead_day": cases["lead_day"],
        "lead_hours": cases["lead_hours"],
        "month": cases["month"],
        "masked": cases["masked"],
    }
    for name, metrics in arms.items():
        columns[f"qs__{name}"] = pl.Series(metrics["qs"].mean(axis=0))
    pl.DataFrame(columns).write_parquet(common.WORK_DIR / "rows.parquet")
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for key, value in report["contrasts"].items():
        kept = value.get("kept | qs")
        if kept:
            print(key, round(kept["mean"], 3), [round(x, 3) for x in kept["0.95000"]])


if __name__ == "__main__":
    main()
