"""Score experiment 00303's arms and read the planned contrasts.

Reads `work/preds/*.npy` (written by `e00303_run.py`), applies the experiment 33 spread factor to
the ensemble mixtures, and scores every variant on the quantile score (points of capacity), by row
set and setting. Contrasts are `first - second`, so a positive number means the second variant is
better. Their 95% intervals resample whole months (of the ENS run) and the three seeds; farms are
not resampled.

Output: `work/score.json` and `work/rows.parquet` (per-row covariates and seed-mean scores).

    cd ml-experiments/00303-wind-effective-wind-adapters
    uv run --no-sync python e00303_score.py
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as common  # noqa: E402
import e00303_models as models  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

PRED_DIR = common.WORK_DIR / "preds"
FACT_DIR = common.WORK_DIR / "facts"
SCORED_DIR = common.WORK_DIR / "scored"
FINAL_DIR = common.WORK_DIR / "ens_final"
RESAMPLES = 20_000
ROW_SETS = ("all", "kept", "lead_12_36")
SETTINGS = ("era5", "ens")
FARMS = ("W1", "W2", "W3")


def ordered_cases() -> pl.DataFrame:
    """Return the scored cases, farm by farm in the order `e00303_run.py` stores them.

    Adds the covariates the diagnostics group by: ERA5 100 m wind speed, direction (degrees),
    `month` (of the ENS run), `valid_hour`, `season`, and the row-set flags.
    """
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites).select(
        "time_series_id", "valid_time", "wind_speed_100m", "wind_dir_sin_100m", "wind_dir_cos_100m"
    )
    cases = common.cases_table(sites=sites)
    pieces = [
        cases.filter(pl.col("time_series_id") == int(row["time_series_id"]))
        for row in sites.iter_rows(named=True)
    ]
    facts = {farm: json.loads((FACT_DIR / f"P__{farm}.json").read_text()) for farm in FARMS}
    rated = {farm: facts[farm]["rated_m_s"] for farm in FARMS}
    return (
        pl.concat(pieces)
        .join(era5, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            valid_hour=pl.col("valid_time").dt.hour(),
            valid_month=pl.col("valid_time").dt.month(),
            direction_deg=(
                pl.arctan2(pl.col("wind_dir_sin_100m"), pl.col("wind_dir_cos_100m")).degrees() % 360
            ),
            all=pl.lit(value=True),
            kept=~pl.col("masked"),
            idle=pl.col("masked"),
            lead_12_36=(pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36),
            above_rated=pl.col("wind_speed_100m")
            >= pl.col("site").replace_strict(rated, return_dtype=pl.Float64),
        )
    )


def load_quantiles(*, spec: models.Spec, setting: str, cases: pl.DataFrame) -> np.ndarray | None:
    """Return shape (seeds, rows, 11) quantiles of one variant, or None if not all stored."""
    cache = SCORED_DIR / f"{spec.name}__{setting}.npy"
    if cache.exists():
        return np.load(cache)
    stack = []
    for seed in common.SEEDS:
        stored_seed = seed if seed in spec.seeds else spec.seeds[0]
        parts = []
        for farm in FARMS:
            path = PRED_DIR / f"{spec.name}__{farm}__s{stored_seed}__{setting}.npy"
            if not path.exists():
                return None
            parts.append(np.load(path))
        values = np.concatenate(parts)
        if setting == "ens":
            values = common.apply_spread(cases=cases, mixture=values.astype(np.float64)).astype(
                np.float32
            )
        stack.append(values)
    out = np.stack(stack)
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def row_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-seed, per-row metrics (points of capacity): shape (seeds, rows)."""
    scale = common.POINTS_PER_FRACTION
    out: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "qs",
            "median_abs_error",
            "mean_abs_error",
            "median_minus_mean",
            "p01",
            "p99",
            "below_median",
            "covered",
        )
    }
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


def bootstrap_mean(*, values: np.ndarray, keep: np.ndarray, month: np.ndarray) -> list[float]:
    """Return the mean of `values` (seeds, rows) over `keep` and a 95% month-and-seed interval."""
    months, seeds = np.unique(month[keep]), values.shape[0]
    sums = np.zeros((months.size, seeds))
    counts = np.zeros(months.size)
    for i, m in enumerate(months):
        in_month = keep & (month == m)
        sums[i] = values[:, in_month].sum(axis=1)
        counts[i] = in_month.sum()
    rng = np.random.default_rng(0)
    draw_m = rng.integers(0, months.size, size=(RESAMPLES, months.size))
    draw_s = rng.integers(0, seeds, size=(RESAMPLES, seeds))
    picked = sums[draw_m[:, :, None], draw_s[:, None, :]].sum(axis=(1, 2))
    totals = counts[draw_m].sum(axis=1) * seeds
    means = picked / totals
    point = float(sums.sum() / (counts.sum() * seeds))
    low, high = np.quantile(means, [0.025, 0.975])
    return [point, float(low), float(high)]


PART_A: Final = ("P", "X", "A2", "A3", "A23", "A4", "A5", "A6")
PART_B_ARMS: Final = ("B0", "B1", "B2", "B3", "B4", "B5_B1", "B5_B2", "B5_B3")
PART_A_PAIRS: Final = (
    ("P", "X"),
    ("P", "A2"),
    ("P", "A3"),
    ("P", "A23"),
    ("P", "A4"),
    ("P", "A5"),
    ("P", "A6"),
    ("A23", "A4"),
    ("A4", "A5"),
    ("A6", "A4"),
    ("A4", "X"),
)
LEVEL_NOTE: Final = "A positive contrast means the second arm is better."


def part_b_pairs() -> list[tuple[str, str]]:
    """Return the Part B contrasts for the core and for XGBoost."""
    pairs = []
    for model in ("core", "xgb"):
        names = [f"{model}_{arm}" for arm in PART_B_ARMS]
        pairs += [(names[0], name) for name in names[1:]]
        pairs += [(f"{model}_B4", f"{model}_B5_B{i}") for i in (1, 2, 3)]
        pairs += [(f"{model}_B1", f"{model}_B2"), (f"{model}_B1", f"{model}_B3")]
    pairs += [("core_B0", "xgb_B0"), ("core_B4", "xgb_B4")]
    return pairs


def load_final(*, name: str) -> np.ndarray | None:
    """Return the stored Part B quantiles of `<model>_<arm>`, shape (seeds, rows, 11)."""
    path = FINAL_DIR / f"{name}.npy"
    return np.load(path) if path.exists() else None


def main() -> None:  # noqa: PLR0912
    """Score every stored arm and the planned contrasts, and write the report."""
    specs = models.build_specs()
    cases = ordered_cases()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    month = cases["month"].to_numpy()
    flags = {name: cases[name].to_numpy() for name in ROW_SETS}
    arms: dict[str, dict[str, np.ndarray]] = {}
    for name in PART_A:
        quantiles = load_quantiles(spec=specs[name], setting="era5", cases=cases)
        if quantiles is not None:
            arms[f"era5_{name}"] = row_metrics(quantiles=quantiles, actual=actual)
            print("scored", name, flush=True)
    for model in ("core", "xgb"):
        for arm in PART_B_ARMS:
            quantiles = load_final(name=f"{model}_{arm}")
            if quantiles is not None:
                arms[f"{model}_{arm}"] = row_metrics(quantiles=quantiles, actual=actual)
                print("scored", model, arm, flush=True)
    report: dict = {"scores": {}, "contrasts": {}, "rows_in_set": {}, "by_lead_day": {}}
    for name, mask in flags.items():
        report["rows_in_set"][name] = int(mask.sum())
    for name, metrics in arms.items():
        report["scores"][name] = {
            row_set: {m: float(v[:, flags[row_set]].mean()) for m, v in metrics.items()}
            for row_set in ROW_SETS
        }
        report["by_lead_day"][name] = {
            str(day): float(metrics["qs"][:, (cases["lead_day"] == day).to_numpy()].mean())
            for day in range(15)
        }
    pairs = [(f"era5_{a}", f"era5_{b}") for a, b in PART_A_PAIRS] + part_b_pairs()
    for first, second in pairs:
        if first not in arms or second not in arms:
            continue
        key = f"{first} - {second}"
        report["contrasts"][key] = {}
        for metric in ("qs", "median_abs_error", "mean_abs_error", "p01", "p99"):
            difference = arms[first][metric] - arms[second][metric]
            for row_set in ROW_SETS:
                report["contrasts"][key][f"{row_set} | {metric}"] = bootstrap_mean(
                    values=difference, keep=flags[row_set], month=month
                )
        for farm in FARMS:
            report["contrasts"][key][f"kept only {farm} | qs"] = bootstrap_mean(
                values=arms[first]["qs"] - arms[second]["qs"],
                keep=flags["kept"] & (cases["site"] == farm).to_numpy(),
                month=month,
            )
            report["contrasts"][key][f"kept without {farm} | qs"] = bootstrap_mean(
                values=arms[first]["qs"] - arms[second]["qs"],
                keep=flags["kept"] & (cases["site"] != farm).to_numpy(),
                month=month,
            )
    columns = {
        "site": cases["site"],
        "valid_time": cases["valid_time"],
        "lead_day": cases["lead_day"],
        "lead_hours": cases["lead_hours"],
        "month": cases["month"],
        "speed_100m": cases["wind_speed_100m"],
        "masked": cases["masked"],
    }
    for name, metrics in arms.items():
        columns[f"qs__{name}"] = pl.Series(metrics["qs"].mean(axis=0))
    pl.DataFrame(columns).write_parquet(common.WORK_DIR / "rows.parquet")
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for key in ("era5_P - era5_X", "era5_P - era5_A4", "core_B0 - core_B4", "core_B0 - core_B1"):
        if key in report["contrasts"]:
            print(key, [round(x, 3) for x in report["contrasts"][key]["kept | qs"]])


if __name__ == "__main__":
    main()
