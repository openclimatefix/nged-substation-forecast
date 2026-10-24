"""Score experiment 00342's arms and read the five adoption conditions.

Reads `work/final/` (written by `e00342_final.py`). Contrasts are (default plus x) minus the
default. The targeted metric `tgt` is the mean of the 0.95 and 0.99 pinball losses (points of
capacity). Intervals resample whole months of the run date and the three seeds (50,000
resamples), as experiment 00340 did; farms are not resampled. Condition 1 is read at 1 - 0.01/3;
condition 2 as a one-sided 95% upper bound on the headline quantile score (0.02), the 0.01
pinball loss (0.05), and the maintainer's three row sets (0.1); condition 3 on the replicate
period with a t test on the monthly paired differences.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00342-*/e00342_score.py   (PYTHONPATH: see e00342_common)
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_score as score336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00342_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c336
c340 = common.c340
DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x1", "x2", "x3")
DESCRIPTIVE: Final[tuple[str, ...]] = ("x1lit", "x1cap", "x1capstrong", "x2base", "x2plc", "x2lofo")
REFERENCES: Final[dict[str, str]] = {"ref_drift": "Sdrift", "ref_mixture": "x1"}
"""Reference arms read from other experiments: experiment 340's drift shift, 339's mixture head."""
ARM_ORDER: Final[tuple[str, ...]] = (DEFAULT, *ADOPTION_ORDER, *DESCRIPTIVE, *REFERENCES)
MIN_GAIN: Final[dict[str, float]] = {"x1": 0.0, "x2": 0.03, "x3": 0.02}
CONDITION_1_LEVEL: Final[float] = 1 - 0.01 / 3
HEADLINE_TOLERANCE: Final[float] = 0.02
TAIL_TOLERANCE: Final[float] = 0.05
LOW_WIND_TOLERANCE: Final[float] = 0.1
LOW_WIND_SUBSETS: Final[tuple[str, ...]] = ("era5_masked", "idle210", "all")
SUBSETS: Final[tuple[str, ...]] = (
    "all",
    "kept",
    "era5_masked",
    "idle210",
    "outside_era5",
    "days_1_2",
    "days_3_10",
    "above_rated",
)
TABLE_METRICS: Final[tuple[str, ...]] = (
    "qs",
    "tgt",
    "p01",
    "p05",
    "p95",
    "p99",
    "below_q01",
    "below_q05",
    "above_q95",
    "above_q99",
    "median_abs_error",
    "mean_abs_error",
    "median_minus_mean",
    "covered",
)
REFERENCE_DIRS: Final = {
    "ref_drift": (c340.FINAL_DIR, "Sdrift"),
    "ref_mixture": (
        common.ML_EXPERIMENTS_DATA_DIR
        / "00339-wind-research-default-1-adapter-plus-mixture-head"
        / "final",
        "x1",
    ),
}


class Intervals(score326.Intervals):
    """The 326 resampler, with condition 1 read at the level for three arms."""

    def read(self, *, values: np.ndarray, keep: np.ndarray) -> dict[str, float | list[float]]:
        """Return the mean, the 95% and 99.67% intervals, and the one-sided 95% upper bound."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        out: dict[str, float | list[float]] = {"mean": float(values[:, keep].mean())}
        for level in (0.95, CONDITION_1_LEVEL):
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.4f}"] = [float(low), float(high)]
        out["upper_95_one_sided"] = float(np.quantile(means, 0.95))
        return out


def load_arm(*, arm: str, suffix: str) -> np.ndarray:
    """Return an arm's stored quantiles (seeds, cases, 11); references come from other folders."""
    if arm in REFERENCE_DIRS:
        folder, name = REFERENCE_DIRS[arm]
        return np.load(folder / f"{name}{suffix}.npy")
    return np.load(common.FINAL_DIR / f"{arm}{suffix}.npy")


def all_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return the per-row metrics of the 303 and 326 scorers plus the targeted metric."""
    metrics = {
        **score303.row_metrics(quantiles=quantiles, actual=actual),
        **score326.extra_metrics(quantiles=quantiles, actual=actual),
    }
    metrics["tgt"] = (metrics["p95"] + metrics["p99"]) / 2.0
    return metrics


def main() -> None:  # noqa: PLR0915
    """Score every stored arm, read the conditions, and write the report."""
    sites = c303.site_table()
    rated = score326.rated_speeds(sites=sites)
    cases = score303.ordered_cases().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    usable_hind = pl.read_parquet(common.FINAL_DIR / "replicate_cases.parquet")
    stats = {
        (kind, name): np.concatenate(
            [np.load(c336.stats_path(kind=kind, set_name=name, farm=f)) for f in common.FARMS]
        )
        for kind in ("speed", "era5")
        for name in ("hindcast", "scored")
    }
    hind = ens_final.hindcast_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    in_rep = (
        hind.filter(pl.Series(usable))["power_fcst_init_time"] >= common.REPLICATE_START
    ).to_numpy()
    frames = {
        "scored": (
            cases,
            score336.flags_for(
                frame=cases,
                era5=stats[("era5", "scored")],
                speed=stats[("speed", "scored")],
                rated=rated,
            ),
            "",
        ),
        "replicate": (
            usable_hind.with_columns(lead_day=pl.col("lead_day").clip(0, 10)),
            score336.flags_for(
                frame=usable_hind,
                era5=stats[("era5", "hindcast")][usable][in_rep],
                speed=stats[("speed", "hindcast")][usable][in_rep],
                rated=rated,
            ),
            "_rep",
        ),
    }
    fits = json.loads((common.FINAL_DIR / "fits.json").read_text())
    gate = json.loads((common.FINAL_DIR / "reproduction.json").read_text())
    report: dict = {"gate": gate, "fits": fits, "scores": {}, "contrasts": {}, "benchmarks": {}}
    for label, (frame, flags, suffix) in frames.items():
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        resampler = Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
        metrics = {
            arm: all_metrics(quantiles=load_arm(arm=arm, suffix=suffix), actual=actual)
            for arm in ARM_ORDER
        }
        site = frame["site"].to_numpy()
        report["scores"][label] = {
            arm: {
                subset: {m: float(v[:, flags[subset]].mean()) for m, v in values.items()}
                for subset in SUBSETS
                if flags[subset].any()
            }
            | {
                "by_farm_kept": {
                    farm: {
                        m: float(v[:, flags["kept"] & (site == farm)].mean())
                        for m, v in values.items()
                    }
                    for farm in common.FARMS
                }
            }
            for arm, values in metrics.items()
        }
        report["contrasts"][label] = {}
        months = np.unique(frame["month"].to_numpy())
        for arm in ARM_ORDER[1:]:
            entry: dict = {}
            for metric in TABLE_METRICS:
                difference = metrics[arm][metric] - metrics[DEFAULT][metric]
                for subset in SUBSETS:
                    if flags[subset].any():
                        entry[f"{subset} | {metric}"] = resampler.read(
                            values=difference, keep=flags[subset]
                        )
            tgt_difference = metrics[arm]["tgt"] - metrics[DEFAULT]["tgt"]
            for farm in common.FARMS:
                entry[f"kept only {farm} | tgt"] = resampler.read(
                    values=tgt_difference, keep=flags["kept"] & (site == farm)
                )
                entry[f"kept without {farm} | tgt"] = resampler.read(
                    values=tgt_difference, keep=flags["kept"] & (site != farm)
                )
            monthly = np.array(
                [
                    tgt_difference[:, flags["kept"] & (frame["month"] == m).to_numpy()].mean()
                    for m in months
                ]
            )
            entry["t test on monthly kept | tgt"] = score326.t_test(monthly=monthly)
            report["contrasts"][label][arm] = entry
        if label == "scored":
            # x2 against its recalibration-only control, and the shares of its gain they keep.
            report["contrasts"][label]["x2_minus_x2base"] = {
                "kept | tgt": resampler.read(
                    values=metrics["x2"]["tgt"] - metrics["x2base"]["tgt"], keep=flags["kept"]
                )
            }
            report["contrasts"][label]["x2base_share_of_x2_gain"] = resampler.share(
                numerator=metrics["x2base"]["tgt"] - metrics[DEFAULT]["tgt"],
                denominator=metrics["x2"]["tgt"] - metrics[DEFAULT]["tgt"],
                keep=flags["kept"],
            )
            report["contrasts"][label]["x2plc_share_of_x2_gain"] = resampler.share(
                numerator=metrics["x2plc"]["tgt"] - metrics[DEFAULT]["tgt"],
                denominator=metrics["x2"]["tgt"] - metrics[DEFAULT]["tgt"],
                keep=flags["kept"],
            )
            five = np.load(common.FINAL_DIR / "x2_5seeds.npy")
            kept = flags["kept"]
            per_seed = [all_metrics(quantiles=five[i : i + 1], actual=actual) for i in range(5)]
            report["x2_five_seeds_kept"] = {
                m: [float(p[m][:, kept].mean()) for p in per_seed] for m in ("qs", "p99", "tgt")
            }
            for name in ("CLIM", "PERS"):
                reference = score326.benchmark_frame(name=name, cases=frame)
                if reference is None:
                    continue
                present = np.isfinite(reference[0]).all(axis=1)
                row_score = (
                    c303.quantile_score(quantiles=np.nan_to_num(reference[0]), actual=actual)
                    * c303.POINTS_PER_FRACTION
                )
                keep = flags["kept"] & present
                report["benchmarks"][name] = {
                    "rows": int(keep.sum()),
                    "benchmark": float(row_score[keep].mean()),
                    "default_on_same_rows": float(metrics[DEFAULT]["qs"][:, keep].mean()),
                }
            notable = np.load(common.FINAL_DIR / "default_notable.npy")
            notable_m = all_metrics(quantiles=notable, actual=actual)
            report["benchmarks"]["XGBoost_reference"] = {
                m: float(notable_m[m][:, flags["kept"]].mean())
                for m in ("qs", "p99", "tgt", "above_q99", "above_q95")
            }
    known = gate["known_answer_test"]
    known_ok = bool(
        abs(known["fit_high_mean"] - known["injected_high"]) <= 0.005
        and abs(known["fit_low_mean"]) <= 0.005
    )
    leak = gate["leakage_check_fit_rows"]
    leakage_ok = bool(
        leak["latest_hindcast_valid_time"] < leak["earliest_scored_valid_time"]
        and leak["latest_replicate_fit_valid_time"] < "2025-01-01"
    )
    scored_c, replicate_c = report["contrasts"]["scored"], report["contrasts"]["replicate"]
    placebo_share = scored_c["x2plc_share_of_x2_gain"]["mean"]
    adoption = {}
    for arm in ADOPTION_ORDER:
        main = scored_c[arm]["kept | tgt"]
        low, high = main[f"{CONDITION_1_LEVEL:.4f}"]
        gain = -main["mean"]
        minimum = MIN_GAIN[arm]
        headline = scored_c[arm]["kept | qs"]["upper_95_one_sided"]
        p01 = scored_c[arm]["kept | p01"]["upper_95_one_sided"]
        p99 = scored_c[arm]["kept | p99"]["upper_95_one_sided"]
        low_wind = {s: scored_c[arm][f"{s} | qs"]["upper_95_one_sided"] for s in LOW_WIND_SUBSETS}
        rep = replicate_c[arm]
        test = rep["t test on monthly kept | tgt"]
        rep_gain = -rep["kept | tgt"]["mean"]
        contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
        entry = {
            "gain": gain,
            "interval": [low, high],
            "minimum_gain": minimum,
            "condition_1": bool(high < 0 and gain >= minimum),
            "headline_upper_bound": headline,
            "p01_upper_bound": p01,
            "low_wind_upper_bounds": low_wind,
            "condition_2": bool(
                headline <= HEADLINE_TOLERANCE
                and p01 <= TAIL_TOLERANCE
                and p99 <= TAIL_TOLERANCE
                and all(v <= LOW_WIND_TOLERANCE for v in low_wind.values())
            ),
            "replicate_gain": rep_gain,
            "condition_3": bool(
                rep_gain > 0 and rep_gain >= 0.5 * min(gain, minimum) and not contradicted
            ),
            "condition_4": bool(
                gate["gate_passed"]
                and known_ok
                and leakage_ok
                and (placebo_share < 0.5 and gain > 0 if arm == "x2" else True)
            ),
            "condition_5": True,
        }
        entry["all_five"] = all(entry[f"condition_{i}"] for i in range(1, 6))
        adoption[arm] = entry
    passing = [a for a in ADOPTION_ORDER if adoption[a]["all_five"]]
    report["adopted"] = passing[0] if passing else None
    report["adoption"] = adoption
    report["controls"] = {
        "known_answer_ok": known_ok,
        "leakage_ok": leakage_ok,
        "x2_placebo_share_of_gain": placebo_share,
    }
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)


if __name__ == "__main__":
    main()
