"""Score experiment 00276's variants and read the diagnostics' contrasts.

Reads `work/preds/*.npy` (written by `e00276_run.py`), applies the experiment 33 spread factor to
the ensemble mixtures, and scores every variant on the quantile score (points of capacity), by row
set and setting. Contrasts are `first - second`, so a positive number means the second variant is
better. Their 95% intervals resample whole months (of the ENS run) and the three seeds; farms are
not resampled.

Output: `work/score.json` and `work/rows.parquet` (per-row covariates and seed-mean scores).

    cd ml-experiments/00276-wind-xgboost-vs-physics-gap-diagnostics
    uv run --no-sync python e00276_score.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00276_common as common  # noqa: E402
import e00276_models as models  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

PRED_DIR = common.WORK_DIR / "preds"
FACT_DIR = common.WORK_DIR / "facts"
SCORED_DIR = common.WORK_DIR / "scored"
RESAMPLES = 20_000
ROW_SETS = ("all", "kept", "idle", "above_cut_in", "above_rated")
SETTINGS = ("era5", "ens")
FARMS = ("W1", "W2", "W3")


def ordered_cases() -> pl.DataFrame:
    """Return the scored cases, farm by farm in the order `e00276_run.py` stores them.

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
    cut_in = {farm: facts[farm]["cut_in_m_s"] for farm in FARMS}
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
            above_cut_in=pl.col("wind_speed_100m")
            >= pl.col("site").replace_strict(cut_in, return_dtype=pl.Float64),
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
        for k in ("qs", "median_abs_error", "mean_abs_error", "median_minus_mean", "p01", "p99")
    }
    for q in quantiles.astype(np.float64):
        mean_forecast = q[:, 1:10] @ common.NINE_MASSES
        out["qs"].append(common.quantile_score(quantiles=q, actual=actual) * scale)
        out["median_abs_error"].append(np.abs(actual - q[:, common.MEDIAN_INDEX]) * scale)
        out["mean_abs_error"].append(np.abs(actual - mean_forecast) * scale)
        out["median_minus_mean"].append((q[:, common.MEDIAN_INDEX] - mean_forecast) * scale)
        out["p01"].append(common.pinball_at(quantiles=q, actual=actual, column="q01") * scale)
        out["p99"].append(common.pinball_at(quantiles=q, actual=actual, column="q99") * scale)
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


def main() -> None:  # noqa: PLR0912
    """Score every stored variant and the contrasts, and write the report."""
    specs = models.build_specs()
    cases = ordered_cases()
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    month = cases["month"].to_numpy()
    flags = {name: cases[name].to_numpy() for name in ROW_SETS}
    arms: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for setting in SETTINGS:
        for spec in specs.values():
            if setting == "ens" and not spec.ens:
                continue
            quantiles = load_quantiles(spec=spec, setting=setting, cases=cases)
            if quantiles is not None:
                arms[(spec.name, setting)] = row_metrics(quantiles=quantiles, actual=actual)
                print("scored", spec.name, setting, flush=True)
    report: dict = {"scores": {}, "contrasts": {}, "rows_in_set": {}}
    for name, mask in flags.items():
        report["rows_in_set"][name] = int(mask.sum())
    for (name, setting), metrics in arms.items():
        report["scores"].setdefault(setting, {})[name] = {
            row_set: {m: float(v[:, flags[row_set]].mean()) for m, v in metrics.items()}
            for row_set in ROW_SETS
        }
    pairs = [
        ("P", "X"),
        ("P_nm", "X_nm"),
        ("P_nm", "P"),
        ("X_nm", "X"),
        ("S", "X"),
        ("P", "S"),
        ("P", "X1"),
        ("P", "X2"),
        ("P", "X3"),
        ("P", "X4"),
        ("P", "X5"),
        ("P", "X6"),
        ("X1", "X2"),
        ("X2", "X3"),
        ("X3", "X4"),
        ("X4", "X5"),
        ("X5", "X6"),
        ("X4", "X6"),
        ("X", "X4"),
        ("P_noshear", "P"),
        ("P_dir", "P"),
        ("P_dens", "P"),
        ("P_net", "P"),
        ("P_net", "S"),
        ("P_bin_speed", "P"),
        ("P_pinball", "P"),
        ("P_pinball", "X"),
        ("P_bin_speed", "X"),
        ("P_year", "P"),
        ("P_month", "P"),
        ("P_year", "X"),
        ("P_month", "X"),
        ("P_scaled", "P"),
        ("X_scaled", "X"),
        ("P_scaled", "X_scaled"),
    ]
    pairs += [(f"P_w{m}", f"X_w{m}") for m in (3, 6, 12, 24, 48)]
    pairs += [(f"P_w{m}", "P") for m in (3, 6, 12, 24, 48)]
    pairs += [(f"X_w{m}", "X") for m in (3, 6, 12, 24, 48)]
    for setting in SETTINGS:
        for first, second in pairs:
            if (first, setting) not in arms or (second, setting) not in arms:
                continue
            key = f"{setting} | {first} - {second}"
            report["contrasts"][key] = {}
            for metric in ("qs", "median_abs_error", "mean_abs_error"):
                difference = arms[(first, setting)][metric] - arms[(second, setting)][metric]
                for row_set in ROW_SETS:
                    report["contrasts"][key][f"{row_set} | {metric}"] = bootstrap_mean(
                        values=difference, keep=flags[row_set], month=month
                    )
            # Leave-one-farm-out on the kept rows, quantile score only.
            for farm in FARMS:
                keep = flags["kept"] & (cases["site"] != farm).to_numpy()
                report["contrasts"][key][f"kept without {farm} | qs"] = bootstrap_mean(
                    values=arms[(first, setting)]["qs"] - arms[(second, setting)]["qs"],
                    keep=keep,
                    month=month,
                )
    # Per-row covariates and seed-mean scores for the D3 tables and figures.
    columns = {
        "site": cases["site"],
        "valid_time": cases["valid_time"],
        "lead_day": cases["lead_day"],
        "lead_hours": cases["lead_hours"],
        "month": cases["month"],
        "valid_hour": cases["valid_hour"],
        "valid_month": cases["valid_month"],
        "speed_100m": cases["wind_speed_100m"],
        "direction_deg": cases["direction_deg"],
        "masked": cases["masked"],
        "above_cut_in": cases["above_cut_in"],
        "above_rated": cases["above_rated"],
    }
    for (name, setting), metrics in arms.items():
        columns[f"qs__{name}__{setting}"] = pl.Series(metrics["qs"].mean(axis=0))
    pl.DataFrame(columns).write_parquet(common.WORK_DIR / "rows.parquet")
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for key in ("era5 | P - X", "ens | P - X", "era5 | P_nm - X_nm", "ens | P_nm - X_nm"):
        if key in report["contrasts"]:
            print(
                key,
                {
                    k: [round(x, 3) for x in v]
                    for k, v in report["contrasts"][key].items()
                    if k.endswith("| qs") and "without" not in k
                },
            )


if __name__ == "__main__":
    main()
