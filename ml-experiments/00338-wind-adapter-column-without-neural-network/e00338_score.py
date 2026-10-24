"""Score experiment 00338's arms and read the five adoption conditions.

Reads `work/final/` (written by `e00338_final.py`), scores each arm on the quantile score of
experiment 33 (points of capacity), and reads the contrasts (default plus x) minus the default,
with the default being experiment 285's stored `xgb_B0` mixtures put through the table (the same
quantiles as experiment 303's P1). Intervals resample whole months of the run date and the three
seeds (50,000 resamples); the farms are not resampled. Condition 1 is read at 99.75% (four arms),
condition 2 as a one-sided 95% upper bound on the 0.01 and 0.99 pinball losses, and condition 3 on
the replicate period (hindcast runs initialised 2025-01-01 to 2025-06-16) with a t test on the
monthly paired differences.

Output: `work/score.json` and `work/rows.parquet`.

    uv run --no-sync python ml-experiments/00338-*/e00338_score.py   (PYTHONPATH as for the fit)
"""

import json
import math
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_models as models  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00338_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
DEFAULT: Final[str] = "stored_default"
ARM_ORDER: Final[tuple[str, ...]] = (
    "stored_default",
    "default",
    "x4",
    "x1",
    "x2",
    "x3",
    "x1_no_direction",
    "x1_no_ratio",
    "x1_pl",
    "x2_pl",
)
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x1", "x2", "x3")
CONTRAST_ARMS: Final[tuple[str, ...]] = (
    *ADOPTION_ORDER,
    "x4",
    "x1_no_direction",
    "x1_no_ratio",
    "default",
    "x1_pl",
    "x2_pl",
)
SHARE_ARMS: Final[tuple[str, ...]] = ("x1", "x2", "x3", "x1_no_direction", "x1_no_ratio")
MIN_GAIN: Final[float] = 0.02
PLACEBO: Final[dict[str, str]] = {"x1": "x1_pl", "x2": "x2_pl"}
CONDITION_1_LEVEL: Final[float] = 1 - 0.01 / 4
TAIL_TOLERANCE: Final[float] = 0.05
RATED_SHARE: Final[float] = 0.95
RESAMPLES = score303.RESAMPLES
STUDIES_ARMS_DIR = (
    common.ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work" / "arms"
)
SUBSETS: Final[tuple[str, ...]] = (
    "all",
    "kept",
    "days_1_2",
    "days_3_10",
    "lead_12_36",
    "above_rated",
)


def extra_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-seed, per-row tail metrics: pinball at 0.05 and 0.95, and the tail exceedances."""
    scale = c303.POINTS_PER_FRACTION
    out: dict[str, list[np.ndarray]] = {
        k: [] for k in ("p05", "p95", "below_q01", "above_q99", "below_q05", "above_q95")
    }
    for q in quantiles.astype(np.float64):
        out["p05"].append(c303.pinball_at(quantiles=q, actual=actual, column="q05") * scale)
        out["p95"].append(c303.pinball_at(quantiles=q, actual=actual, column="q95") * scale)
        out["below_q01"].append((actual < q[:, 0]).astype(np.float64))
        out["above_q99"].append((actual > q[:, 10]).astype(np.float64))
        out["below_q05"].append((actual < q[:, 1]).astype(np.float64))
        out["above_q95"].append((actual > q[:, 9]).astype(np.float64))
    return {k: np.stack(v) for k, v in out.items()}


def rated_speeds(*, sites: pl.DataFrame) -> dict[str, float]:
    """Return each farm's rated 100 m speed: the first speed where its ERA5 curve reaches 95%."""
    era5 = c303.era5_rows(sites=sites)
    training = c303.training_table(sites=sites, era5=era5).filter(
        pl.col("valid_time") < c303.BASE_END
    )
    out = {}
    for row in sites.iter_rows(named=True):
        train = training.filter(pl.col("time_series_id") == row["time_series_id"])
        centres, curve = models.binned_curve(
            speed=train["wind_speed_100m"].to_numpy(), target=train["target"].to_numpy()
        )
        out[row["site"]] = float(centres[np.argmax(curve >= RATED_SHARE * curve.max())])
    return out


def with_flags(
    *, cases: pl.DataFrame, ensemble_speed: pl.DataFrame, rated: dict[str, float]
) -> pl.DataFrame:
    """Add the subset flags and the ensemble-mean 100 m speed to a case table."""
    joined = cases.join(
        ensemble_speed, on=list(c303.KEY_COLUMNS), how="left", validate="1:1", maintain_order="left"
    )
    if joined["ens_speed_100m"].null_count():
        msg = "some cases have no ensemble-mean speed in experiment 307's table"
        raise ValueError(msg)
    return joined.with_columns(
        all=pl.lit(value=True),
        kept=~pl.col("masked"),
        days_1_2=pl.col("lead_day").is_between(1, 2),
        days_3_10=pl.col("lead_day").is_between(3, 10),
        lead_12_36=(pl.col("lead_hours") >= 12) & (pl.col("lead_hours") < 36),
        above_rated=(
            pl.col("ens_speed_100m")
            >= pl.col("site").replace_strict(rated, return_dtype=pl.Float64)
        ).fill_null(value=False),
    )


class Intervals(score303.Resampler):
    """The shared resampler plus the 99.75% interval and the one-sided 95% upper bound."""

    def read(self, *, values: np.ndarray, keep: np.ndarray) -> dict[str, float | list[float]]:
        """Return the mean, the 95% and 99.75% intervals, and the one-sided 95% upper bound."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        out: dict[str, float | list[float]] = {"mean": float(values[:, keep].mean())}
        for level in (0.95, CONDITION_1_LEVEL):
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.4f}"] = [float(low), float(high)]
        out["upper_95_one_sided"] = float(np.quantile(means, 0.95))
        return out

    def share(
        self, *, numerator: np.ndarray, denominator: np.ndarray, keep: np.ndarray
    ) -> dict[str, float | list[float]]:
        """Return the ratio of two mean differences with a 95% interval."""
        top, _ = self.draws(values=numerator, keep=keep)
        bottom, _ = self.draws(values=denominator, keep=keep)
        ratio = top / np.where(np.abs(bottom) < 1e-9, np.nan, bottom)
        low, high = np.nanquantile(ratio, [0.025, 0.975])
        return {
            "mean": float(numerator[:, keep].mean() / denominator[:, keep].mean()),
            "0.9500": [float(low), float(high)],
        }


def t_test(*, monthly: np.ndarray) -> dict[str, float]:
    """Return the mean, t statistic, and two-sided p value of monthly paired differences."""
    n = monthly.size
    mean, sd = float(monthly.mean()), float(monthly.std(ddof=1))
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    df = n - 1
    if not math.isfinite(t):
        return {"months": n, "mean": mean, "t": t, "p_two_sided": 0.0}
    grid = np.linspace(-abs(t), abs(t), 20_001)
    pdf = np.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)) * (
        1 + grid**2 / df
    ) ** (-(df + 1) / 2)
    p = float(1 - np.trapezoid(pdf, grid))
    return {"months": n, "mean": mean, "t": t, "p_two_sided": max(p, 0.0)}


def benchmark_frame(*, name: str, cases: pl.DataFrame) -> np.ndarray | None:
    """Return experiment 230's stored climatology or persistence quantiles on `cases`, or None."""
    path = STUDIES_ARMS_DIR / f"{name}_spread.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path).select(*c303.KEY_COLUMNS, "seed", *c303.LEVEL_COLUMNS)
    frame = frame.group_by(*c303.KEY_COLUMNS).agg(pl.col(c303.LEVEL_COLUMNS).mean())
    joined = cases.select(*c303.KEY_COLUMNS).join(
        frame, on=list(c303.KEY_COLUMNS), how="left", maintain_order="left"
    )
    return joined.select(c303.LEVEL_COLUMNS).to_numpy()[None, :, :].astype(np.float64)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score every stored arm, read the conditions, and write the report."""
    sites = c303.site_table()
    rated = rated_speeds(sites=sites)
    scored_speed = pl.read_parquet(
        common.IMPLIED_CASES_PATH.with_name("cases_scored.parquet")
    ).select(*c303.KEY_COLUMNS, "ens_speed_100m")
    hindcast_speed = pl.read_parquet(common.IMPLIED_CASES_PATH).select(
        *c303.KEY_COLUMNS, "ens_speed_100m"
    )
    cases = with_flags(
        cases=score303.ordered_cases().drop("all", "kept", "lead_12_36", "days_1_2"),
        ensemble_speed=scored_speed,
        rated=rated,
    )
    rep = with_flags(
        cases=pl.read_parquet(common.FINAL_DIR / "replicate_cases.parquet"),
        ensemble_speed=hindcast_speed,
        rated=rated,
    )
    report: dict = {"rated_speed_m_s": rated, "scores": {}, "contrasts": {}, "benchmarks": {}}
    for label, frame, suffix in (("scored", cases, ""), ("replicate", rep, "_rep")):
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        flags = {name: frame[name].to_numpy() for name in SUBSETS}
        resampler = Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for arm in (*ARM_ORDER, "stored_default_notable"):
            path = common.FINAL_DIR / f"{arm}{suffix}.npy"
            if path.exists():
                quantiles = np.load(path)
                metrics[arm] = {
                    **score303.row_metrics(quantiles=quantiles, actual=actual),
                    **extra_metrics(quantiles=quantiles, actual=actual),
                }
        report["scores"][label] = {
            arm: {
                subset: {m: float(v[:, flags[subset]].mean()) for m, v in values.items()}
                for subset in SUBSETS
            }
            | {
                "by_farm_kept": {
                    farm: {
                        m: float(v[:, flags["kept"] & (frame["site"] == farm).to_numpy()].mean())
                        for m, v in values.items()
                        if m in ("qs", "median_minus_mean", "below_median")
                    }
                    for farm in common.FARMS
                }
            }
            for arm, values in metrics.items()
        }
        report["contrasts"][label] = {}
        for arm in CONTRAST_ARMS:
            if arm not in metrics or DEFAULT not in metrics:
                continue
            entry: dict = {}
            for metric in ("qs", "p01", "p99", "p05", "p95", "median_abs_error", "mean_abs_error"):
                difference = metrics[arm][metric] - metrics[DEFAULT][metric]
                for subset in SUBSETS:
                    entry[f"{subset} | {metric}"] = resampler.read(
                        values=difference, keep=flags[subset]
                    )
            qs_difference = metrics[arm]["qs"] - metrics[DEFAULT]["qs"]
            for farm in common.FARMS:
                site = (frame["site"] == farm).to_numpy()
                entry[f"kept only {farm} | qs"] = resampler.read(
                    values=qs_difference, keep=flags["kept"] & site
                )
                entry[f"kept without {farm} | qs"] = resampler.read(
                    values=qs_difference, keep=flags["kept"] & ~site
                )
            months = np.unique(frame["month"].to_numpy())
            monthly = np.array(
                [
                    qs_difference[:, flags["kept"] & (frame["month"] == m).to_numpy()].mean()
                    for m in months
                ]
            )
            entry["monthly kept | qs"] = {"months": months.tolist(), "values": monthly.tolist()}
            entry["t test on monthly kept | qs"] = t_test(monthly=monthly)
            report["contrasts"][label][arm] = entry
        report["contrasts"][label]["vs_refit_default"] = {}
        if "default" in metrics:
            for arm in (*ADOPTION_ORDER, "x4", "stored_default"):
                if arm in metrics:
                    report["contrasts"][label]["vs_refit_default"][arm] = resampler.read(
                        values=metrics[arm]["qs"] - metrics["default"]["qs"], keep=flags["kept"]
                    )
        report["contrasts"][label]["placebo_share"] = {}
        for arm, placebo in PLACEBO.items():
            if arm in metrics and placebo in metrics and DEFAULT in metrics:
                report["contrasts"][label]["placebo_share"][arm] = resampler.share(
                    numerator=metrics[placebo]["qs"] - metrics[DEFAULT]["qs"],
                    denominator=metrics[arm]["qs"] - metrics[DEFAULT]["qs"],
                    keep=flags["kept"],
                )
        report["contrasts"][label]["share_of_x4_gain"] = {}
        if "x4" in metrics and DEFAULT in metrics:
            for arm in SHARE_ARMS:
                if arm not in metrics:
                    continue
                report["contrasts"][label]["share_of_x4_gain"][arm] = {
                    subset: resampler.share(
                        numerator=metrics[arm]["qs"] - metrics[DEFAULT]["qs"],
                        denominator=metrics["x4"]["qs"] - metrics[DEFAULT]["qs"],
                        keep=flags[subset],
                    )
                    for subset in ("kept", "days_1_2", "days_3_10")
                }
        if label == "scored":
            for name in ("CLIM", "PERS"):
                reference = benchmark_frame(name=name, cases=frame)
                if reference is None:
                    continue
                present = np.isfinite(reference[0]).all(axis=1)
                row_score = (
                    c303.quantile_score(quantiles=np.nan_to_num(reference[0]), actual=actual)
                    * c303.POINTS_PER_FRACTION
                )
                keep = flags["kept"] & present
                default_score = metrics[DEFAULT]["qs"][:, keep].mean()
                benchmark_mean = row_score[keep].mean()
                report["benchmarks"][name] = {
                    "rows": int(keep.sum()),
                    "benchmark": float(benchmark_mean),
                    "default_on_same_rows": float(default_score),
                    "default_minus_benchmark": resampler.read(
                        values=metrics[DEFAULT]["qs"] - row_score[None, :], keep=keep
                    ),
                }
            if "stored_default_notable" in metrics:
                report["benchmarks"]["XGBoost_reference"] = {
                    "benchmark": float(
                        metrics["stored_default_notable"]["qs"][:, flags["kept"]].mean()
                    ),
                    "default_minus_benchmark": resampler.read(
                        values=metrics[DEFAULT]["qs"] - metrics["stored_default_notable"]["qs"],
                        keep=flags["kept"],
                    ),
                }
        if label == "scored":
            pl.DataFrame(
                {
                    "site": frame["site"],
                    "valid_time": frame["valid_time"],
                    "lead_day": frame["lead_day"],
                    "month": frame["month"],
                    "masked": frame["masked"],
                    **{f"qs__{a}": pl.Series(m["qs"].mean(axis=0)) for a, m in metrics.items()},
                }
            ).write_parquet(common.WORK_DIR / "rows.parquet")
    gate = json.loads((common.FINAL_DIR / "reproduction.json").read_text())
    known = json.loads((common.WORK_DIR / "known_answer.json").read_text())
    report["reproduction_gate"] = gate
    report["known_answer_test"] = known
    report["replicate_masked_share"] = float(rep["masked"].to_numpy().mean())
    report["scored_masked_share"] = float(cases["masked"].to_numpy().mean())
    gate_ok = gate["gate_difference"] <= 0.01
    adoption = {}
    scored_contrasts = report["contrasts"]["scored"]
    replicate_contrasts = report["contrasts"].get("replicate", {})
    for arm in ADOPTION_ORDER:
        if arm not in scored_contrasts:
            continue
        main = scored_contrasts[arm]["kept | qs"]
        low, high = main[f"{CONDITION_1_LEVEL:.4f}"]
        gain = -main["mean"]
        condition_1 = bool(high < 0 and gain >= MIN_GAIN)
        tails = {
            t: scored_contrasts[arm][f"kept | {t}"]["upper_95_one_sided"] for t in ("p01", "p99")
        }
        condition_2 = all(v <= TAIL_TOLERANCE for v in tails.values())
        replicate = replicate_contrasts.get(arm)
        condition_3 = None
        if replicate:
            same_sign = replicate["kept | qs"]["mean"] < 0
            half = -replicate["kept | qs"]["mean"] >= 0.5 * min(gain, MIN_GAIN)
            test = replicate["t test on monthly kept | qs"]
            contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
            condition_3 = bool(same_sign and half and not contradicted)
        adoption[arm] = {
            "gain": gain,
            "interval_99.75": [low, high],
            "condition_1": condition_1,
            "tail_upper_bounds": tails,
            "condition_2": condition_2,
            "condition_3": condition_3,
            "condition_4_gate_and_known_answer": bool(gate_ok and known["passes"]),
        }
    report["adoption"] = adoption
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)
    print(report["benchmarks"].keys())


if __name__ == "__main__":
    main()
