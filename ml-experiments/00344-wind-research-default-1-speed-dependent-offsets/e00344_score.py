"""Score experiment 00344's arms and read the five adoption conditions.

Reads `work/final/` (written by `e00344_final.py`). Contrasts are (default plus x) minus the
default on the quantile score of experiment 33 (points of capacity). Intervals resample whole months
of the run date and the three seeds (50,000 resamples), as experiment 00340 did; farms are not
resampled. Condition 1 is read at 1 - 0.01/3 (three arms); condition 2 as one-sided 95% upper
bounds (0.02 on all half-hours, 0.05 on the 0.01 and 0.99 pinball losses, 0.1 on three low-wind row
sets); condition 3 on the replicate period with a t test on the monthly paired differences.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00344-*/e00344_score.py   (PYTHONPATH: see e00344_common)
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
import e00344_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c336
DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x2", "x1", "x3")
DESCRIPTIVE: Final[tuple[str, ...]] = (
    "x1shuf",
    "x2shuf",
    "x3shuf",
    "x2drift",
    "x2oracle",
    "Sdrift",
)
ARM_ORDER: Final[tuple[str, ...]] = (DEFAULT, *ADOPTION_ORDER, *DESCRIPTIVE)
MIN_GAIN: Final[float] = 0.02
CONDITION_1_LEVEL: Final[float] = 1 - 0.01 / 3
HEADLINE_TOLERANCE: Final[float] = 0.02
TAIL_TOLERANCE: Final[float] = 0.05
LOW_WIND_TOLERANCE: Final[float] = 0.1
LOW_WIND_SUBSETS: Final[tuple[str, ...]] = ("era5_masked", "mean_masked", "idle210")
PLACEBO_SHARE: Final[float] = 0.25
SPEED_GROUPS: Final[tuple[str, ...]] = ("speed_lt5", "speed_5_8", "speed_8_15", "speed_ge15")
SUBSETS: Final[tuple[str, ...]] = (
    "all",
    "kept",
    "era5_masked",
    "mean_masked",
    "idle210",
    "outside_mean",
    "above_rated",
    "days_1_2",
    "days_3_10",
    *SPEED_GROUPS,
)
TABLE_METRICS: Final[tuple[str, ...]] = (
    "qs",
    "p01",
    "p99",
    "median_abs_error",
    "mean_abs_error",
    "median_minus_mean",
    "covered",
)


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


def speed_flags(*, speed: np.ndarray) -> dict[str, np.ndarray]:
    """Return the reported speed groups of the ensemble-mean 100 m speed."""
    low, middle, high = common.REPORT_EDGES
    return {
        "speed_lt5": speed < low,
        "speed_5_8": (speed >= low) & (speed < middle),
        "speed_8_15": (speed >= middle) & (speed < high),
        "speed_ge15": speed >= high,
    }


def main() -> None:  # noqa: PLR0912, PLR0915
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
    rep_speed = stats[("speed", "hindcast")][usable][in_rep]
    scored_flags = score336.flags_for(
        frame=cases, era5=stats[("era5", "scored")], speed=stats[("speed", "scored")], rated=rated
    ) | speed_flags(speed=stats[("speed", "scored")][:, 0])
    rep_flags = score336.flags_for(
        frame=usable_hind,
        era5=stats[("era5", "hindcast")][usable][in_rep],
        speed=rep_speed,
        rated=rated,
    ) | speed_flags(speed=rep_speed[:, 0])
    frames = {
        "scored": (cases, scored_flags, ""),
        "replicate": (
            usable_hind.with_columns(lead_day=pl.col("lead_day").clip(0, 10)),
            rep_flags,
            "_rep",
        ),
    }
    gate = json.loads((common.FINAL_DIR / "reproduction.json").read_text())
    drift_dir = common.c340.FINAL_DIR
    report: dict = {"gate": gate, "scores": {}, "contrasts": {}, "benchmarks": {}}
    report["gate"]["default_max_abs_difference_to_experiment_340"] = float(
        np.abs(np.load(common.FINAL_DIR / "default.npy") - np.load(drift_dir / "default.npy")).max()
    )
    for label, (frame, flags, suffix) in frames.items():
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        resampler = Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARM_ORDER:
            source = drift_dir if arm == "Sdrift" else common.FINAL_DIR
            quantiles = np.load(source / f"{arm}{suffix}.npy")
            metrics[arm] = {
                **score303.row_metrics(quantiles=quantiles, actual=actual),
                **score326.extra_metrics(quantiles=quantiles, actual=actual),
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
                        if m in ("qs", "p01", "p99", "median_minus_mean")
                    }
                    for farm in common.FARMS
                }
            }
            for arm, values in metrics.items()
        }
        report["contrasts"][label] = {}
        for arm in (*ADOPTION_ORDER, *DESCRIPTIVE):
            entry: dict = {}
            for metric in TABLE_METRICS:
                difference = metrics[arm][metric] - metrics[DEFAULT][metric]
                for subset in SUBSETS:
                    if flags[subset].any():
                        entry[f"{subset} | {metric}"] = resampler.read(
                            values=difference, keep=flags[subset]
                        )
            qs_difference = metrics[arm]["qs"] - metrics[DEFAULT]["qs"]
            for farm in common.FARMS:
                entry[f"kept only {farm} | qs"] = resampler.read(
                    values=qs_difference, keep=flags["kept"] & (site == farm)
                )
                entry[f"kept without {farm} | qs"] = resampler.read(
                    values=qs_difference, keep=flags["kept"] & (site != farm)
                )
            months = np.unique(frame["month"].to_numpy())
            monthly = np.array(
                [
                    qs_difference[:, flags["kept"] & (frame["month"] == m).to_numpy()].mean()
                    for m in months
                ]
            )
            entry["t test on monthly kept | qs"] = score326.t_test(monthly=monthly)
            # Share of the kept-row gain that comes from rows at 8 to 15 m/s (prediction P4).
            entry["share of kept gain at 8 to 15 m/s"] = resampler.share(
                numerator=qs_difference * flags["speed_8_15"][None, :],
                denominator=qs_difference,
                keep=flags["kept"],
            )
            report["contrasts"][label][arm] = entry
        for first, second in (("x3", "x2"), ("x1", "x2")):
            diff = metrics[first]["qs"] - metrics[second]["qs"]
            report["contrasts"][label][f"{first}_minus_{second}"] = {
                f"{subset} | qs": resampler.read(values=diff, keep=flags[subset])
                for subset in ("kept", "all")
            }
        if label == "scored":
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
                    "default_minus_benchmark": resampler.read(
                        values=metrics[DEFAULT]["qs"] - row_score[None, :], keep=keep
                    ),
                }
            notable = np.load(common.FINAL_DIR / "default_notable.npy")
            notable_qs = score303.row_metrics(quantiles=notable, actual=actual)["qs"]
            report["benchmarks"]["XGBoost_reference"] = {
                "benchmark": float(notable_qs[:, flags["kept"]].mean()),
                "default_minus_benchmark": resampler.read(
                    values=metrics[DEFAULT]["qs"] - notable_qs, keep=flags["kept"]
                ),
            }
    known = gate["known_answer_test"]
    known_ok = bool(known["x2_max_relative_error"] <= 0.2 and known["x3_max_relative_error"] <= 0.2)
    leak = gate["leakage_check_fit_rows"]
    leakage_ok = bool(leak["latest_hindcast_valid_time"] < leak["earliest_scored_valid_time"])
    scored_c, replicate_c = report["contrasts"]["scored"], report["contrasts"]["replicate"]
    adoption = {}
    for arm in ADOPTION_ORDER:
        main_entry = scored_c[arm]["kept | qs"]
        low, high = main_entry[f"{CONDITION_1_LEVEL:.4f}"]
        gain = -main_entry["mean"]
        tails = {t: scored_c[arm][f"kept | {t}"]["upper_95_one_sided"] for t in ("p01", "p99")}
        headline_harm = scored_c[arm]["all | qs"]["upper_95_one_sided"]
        low_wind = {s: scored_c[arm][f"{s} | qs"]["upper_95_one_sided"] for s in LOW_WIND_SUBSETS}
        rep = replicate_c[arm]
        test = rep["t test on monthly kept | qs"]
        rep_gain = -rep["kept | qs"]["mean"]
        contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
        placebo_gain = -scored_c[f"{arm}shuf"]["kept | qs"]["mean"]
        entry = {
            "gain": gain,
            "interval": [low, high],
            "minimum_gain": MIN_GAIN,
            "condition_1": bool(high < 0 and gain >= MIN_GAIN),
            "headline_upper_bound": headline_harm,
            "tail_upper_bounds": tails,
            "low_wind_upper_bounds": low_wind,
            "condition_2": bool(
                headline_harm <= HEADLINE_TOLERANCE
                and all(v <= TAIL_TOLERANCE for v in tails.values())
                and all(v <= LOW_WIND_TOLERANCE for v in low_wind.values())
            ),
            "replicate_gain": rep_gain,
            "condition_3": bool(
                rep_gain > 0 and rep_gain >= 0.5 * min(gain, MIN_GAIN) and not contradicted
            ),
            "placebo_gain": placebo_gain,
            "condition_4": bool(
                gate["gate_passed"]
                and known_ok
                and leakage_ok
                and placebo_gain < PLACEBO_SHARE * max(gain, 1e-9)
            ),
            "condition_5": True,
        }
        entry["all_five"] = all(entry[f"condition_{i}"] for i in range(1, 6))
        adoption[arm] = entry
    passing = [a for a in ADOPTION_ORDER if adoption[a]["all_five"]]
    report["adopted"] = passing[0] if passing else None
    report["adoption"] = adoption
    x2_scored = scored_c["x2"]
    report["predictions"] = {
        "P1_x2_gain_0.05_to_0.12": -x2_scored["kept | qs"]["mean"],
        "P1_x2_below_cut_in_mean_mask_worsening_under_0.1": x2_scored["mean_masked | qs"]["mean"],
        "P2_x1_gain_replicate": -replicate_c["x1"]["kept | qs"]["mean"],
        "P2_x2_gain_replicate": -replicate_c["x2"]["kept | qs"]["mean"],
        "P3_x3_minus_x2_kept": scored_c["x3_minus_x2"]["kept | qs"]["mean"],
        "P4_share_of_x2_gain_at_8_to_15": x2_scored["share of kept gain at 8 to 15 m/s"]["mean"],
        "P5_p99_change": {a: scored_c[a]["kept | p99"]["mean"] for a in ADOPTION_ORDER},
    }
    report["controls"] = {"known_answer_ok": known_ok, "leakage_ok": leakage_ok}
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)
    print(json.dumps(report["predictions"], indent=1))


if __name__ == "__main__":
    main()
