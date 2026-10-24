"""Score experiment 00343's arms and read the five adoption conditions.

Reads `work/final/`. Contrasts are (arm minus default) on the quantile score of experiment 33
(points of capacity; negative is better). Intervals resample whole months of the run date and the
three seeds (50,000 resamples); farms are not resampled. Condition 1 is read at 1 - 0.01/2 (two
arms; MS is read only if M fails), condition 2 as a one-sided 95% upper bound on the 0.01 and 0.99
pinball losses (tolerance 0.05) and on the rows below the ERA5 cut-in, the experiment 210 idle
rows, and all half-hours (tolerance 0.1), and condition 3 on the replicate period.

Output: `work/score.json`.

    uv run --no-sync python ml-experiments/00343-*/e00343_score.py   (PYTHONPATH as for the fit)
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
import e00340_score as score340  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00343_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c340.c336
DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("M", "MS")
OTHER_ARMS: Final[tuple[str, ...]] = ("M0", "Mpl")
ARM_ORDER: Final[tuple[str, ...]] = (DEFAULT, *ADOPTION_ORDER, *OTHER_ARMS)
REFERENCE_KEPT_SCORE: Final[float] = 11.352
GATE_TOLERANCE: Final[float] = 0.01
CONDITION_1_LEVEL: Final[float] = 1 - 0.01 / 2
TAIL_TOLERANCE: Final[float] = 0.05
LOW_WIND_TOLERANCE: Final[float] = 0.1
LOW_WIND_SUBSETS: Final[tuple[str, ...]] = ("era5_masked", "idle210", "all")
SUBSETS: Final = score340.SUBSETS
TABLE_METRICS: Final = score340.TABLE_METRICS
MEDIAN_TOLERANCE: Final[float] = 0.02


class Intervals(score340.Intervals):
    """The 340 resampler with the level for two arms."""

    def read(self, *, values: np.ndarray, keep: np.ndarray) -> dict[str, float | list[float]]:
        """Return the mean, the 95% and 99.5% intervals, and the one-sided 95% upper bound."""
        picked, totals = self.draws(values=values, keep=keep)
        means = picked / np.maximum(totals, 1)
        out: dict[str, float | list[float]] = {"mean": float(values[:, keep].mean())}
        for level in (0.95, CONDITION_1_LEVEL):
            low, high = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
            out[f"{level:.4f}"] = [float(low), float(high)]
        out["upper_95_one_sided"] = float(np.quantile(means, 0.95))
        return out


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
        hind.filter(pl.Series(usable))["power_fcst_init_time"] >= common.c326.REPLICATE_START
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
    controls = json.loads((common.FINAL_DIR / "controls.json").read_text())
    report: dict = {"scores": {}, "contrasts": {}, "benchmarks": {}}
    for label, (frame, flags, suffix) in frames.items():
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        resampler = Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARM_ORDER:
            path = common.FINAL_DIR / f"{arm}{suffix}.npy"
            if not path.exists():
                continue
            quantiles = np.load(path)
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
                        if m in ("qs", "p01", "p99", "median_abs_error")
                    }
                    for farm in common.FARMS
                }
            }
            for arm, values in metrics.items()
        }
        report["contrasts"][label] = {}
        for arm in (*ADOPTION_ORDER, *OTHER_ARMS):
            if arm not in metrics:
                continue
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
            report["contrasts"][label][arm] = entry
        if label == "scored":
            # The arms against each other: MS minus M, and the table-free pair for P2.
            report["contrasts"][label]["MS_minus_M"] = {
                f"{s} | {m}": resampler.read(
                    values=metrics["MS"][m] - metrics["M"][m], keep=flags[s]
                )
                for s in ("kept", "all")
                for m in ("qs", "p01", "p99", "median_abs_error")
            }
            notable = np.load(common.FINAL_DIR / "default_notable.npy")
            notable_m = score303.row_metrics(quantiles=notable, actual=actual)
            keep = flags["kept"]
            report["p2"] = {
                "M0_minus_M": float(
                    metrics["M0"]["qs"][:, keep].mean() - metrics["M"]["qs"][:, keep].mean()
                ),
                "default_notable_minus_default": float(
                    notable_m["qs"][:, keep].mean() - metrics[DEFAULT]["qs"][:, keep].mean()
                ),
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
                kp = keep & present
                report["benchmarks"][name] = {
                    "rows": int(kp.sum()),
                    "benchmark": float(row_score[kp].mean()),
                    "default_on_same_rows": float(metrics[DEFAULT]["qs"][:, kp].mean()),
                }
    scored_scores = report["scores"]["scored"]
    gate_difference = abs(scored_scores[DEFAULT]["kept"]["qs"] - REFERENCE_KEPT_SCORE)
    known = controls["known_answer_test"]
    leak = score340_leak()
    gate_ok = gate_difference <= GATE_TOLERANCE
    known_ok = bool(known["true"] < 0.03 and known["placebo"] > 0.1)
    scored_c, replicate_c = report["contrasts"]["scored"], report["contrasts"]["replicate"]
    placebo_gain = -scored_c["Mpl"]["kept | qs"]["mean"]
    adoption = {}
    for arm in ADOPTION_ORDER:
        main = scored_c[arm]["kept | qs"]
        low, high = main[f"{CONDITION_1_LEVEL:.4f}"]
        gain = -main["mean"]
        tails = {t: scored_c[arm][f"kept | {t}"]["upper_95_one_sided"] for t in ("p01", "p99")}
        low_wind = {s: scored_c[arm][f"{s} | qs"]["upper_95_one_sided"] for s in LOW_WIND_SUBSETS}
        rep = replicate_c[arm]
        test = rep["t test on monthly kept | qs"]
        rep_gain = -rep["kept | qs"]["mean"]
        contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
        entry = {
            "gain": gain,
            "interval": [low, high],
            "condition_1": bool(high < 0 and gain >= 0),
            "tail_upper_bounds": tails,
            "low_wind_upper_bounds": low_wind,
            "condition_2": bool(
                all(v <= TAIL_TOLERANCE for v in tails.values())
                and all(v <= LOW_WIND_TOLERANCE for v in low_wind.values())
            ),
            "replicate_gain": rep_gain,
            "condition_3": bool(rep_gain > 0 and not contradicted),
            "condition_4": bool(gate_ok and known_ok and leak["ok"] and placebo_gain <= 0),
            "condition_5": True,
        }
        entry["all_five"] = all(entry[f"condition_{i}"] for i in range(1, 6))
        adoption[arm] = entry
    adopted = None
    if adoption["M"]["all_five"]:
        adopted = "M"
    elif adoption["MS"]["all_five"]:
        adopted = "MS"
    report["adopted"] = adopted
    report["adoption"] = adoption
    report["controls"] = {
        "gate_difference_from_11.352": gate_difference,
        "gate_ok": gate_ok,
        "known_answer": known,
        "known_answer_ok": known_ok,
        "leakage": leak,
        "placebo_gain_kept": placebo_gain,
    }
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)
    print(report["p2"], report["controls"])


def score340_leak() -> dict:
    """Return the leakage check: the latest hindcast valid time precedes the earliest scored one."""
    cases = score303.ordered_cases()
    hind = ens_final.hindcast_cases(sites=c303.site_table())
    latest = hind["valid_time"].max()
    earliest = cases["valid_time"].min()
    return {
        "latest_hindcast_valid": str(latest),
        "earliest_scored_valid": str(earliest),
        "ok": bool(latest < earliest),
    }


if __name__ == "__main__":
    main()
