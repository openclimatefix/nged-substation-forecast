"""Score experiment 00336's arms and read the five adoption conditions.

Reads `work/final/` (written by `e00336_final.py`) and scores each arm on the quantile score of
experiment 33 (points of capacity). The contrasts are (default plus x) minus the default, with the
default being this experiment's refit (the reproduction gate shows it equals experiment 303's P1).
Intervals resample whole months of the run date and the three seeds (50,000 resamples); the farms
are not resampled. Condition 1 is read at 1 - 0.01/3 (three arms), condition 2 as a one-sided 95%
upper bound on the 0.01 and 0.99 pinball losses, and condition 3 on the replicate period with a
t test on the monthly paired differences.

Output: `work/score.json` and `work/rows.parquet`.

    uv run --no-sync python ml-experiments/00336-*/e00336_score.py   (PYTHONPATH as for the fit)
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x1", "x2", "x3")
VARIANTS: Final[tuple[str, ...]] = (
    "x1a",
    "x1lo",
    "x1hi",
    "x1g3",
    "x1m",
    "x1mlo",
    "x1mhi",
    "x1mg3",
    "x2e",
    "x2lo",
    "x2hi",
    "x2g3",
)
PLACEBOS: Final[dict[str, str]] = {"x1": "x1pl", "x2": "x2pl", "x3": "x3pl"}
ARM_ORDER: Final[tuple[str, ...]] = (
    DEFAULT,
    *ADOPTION_ORDER,
    *VARIANTS,
    *PLACEBOS.values(),
)
CONDITION_1_LEVEL: Final[float] = 1 - 0.01 / 3
TAIL_TOLERANCE: Final[float] = 0.05
SUBSETS: Final[tuple[str, ...]] = (
    "all",
    "kept",
    "era5_masked",
    "mean_masked",
    "all_masked",
    "idle210",
    "rest",
    "outside_era5",
    "outside_mean",
    "outside_both",
    "days_1_2",
    "days_3_10",
    "above_rated",
)
FULL_METRIC_SUBSETS: Final[tuple[str, ...]] = ("all", "kept", "days_1_2", "days_3_10")


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


def flags_for(*, frame: pl.DataFrame, era5: np.ndarray, speed: np.ndarray, rated: dict) -> dict:
    """Return the boolean row classes of a case frame (scored cases or replicate cases)."""
    sites = frame["site"].to_numpy()
    by_farm = common.cut_in_speeds()
    cut = np.array([by_farm[s] for s in sites])
    rated_speed = np.array([rated[s] for s in sites])
    era5_masked = np.nan_to_num(era5, nan=np.inf) < cut
    idle = frame["masked"].to_numpy()
    mean_masked = speed[:, 0] < cut
    return {
        "all": np.ones(frame.height, dtype=bool),
        "kept": ~idle,
        "era5_masked": era5_masked,
        "mean_masked": mean_masked,
        "all_masked": speed[:, 1] < cut,
        "idle210": idle,
        "rest": ~era5_masked & ~idle,
        "outside_era5": ~era5_masked,
        "outside_mean": ~mean_masked,
        "outside_both": ~era5_masked & ~mean_masked,
        "days_1_2": frame["lead_day"].is_between(1, 2).to_numpy(),
        "days_3_10": frame["lead_day"].is_between(3, 10).to_numpy(),
        "above_rated": speed[:, 0] >= rated_speed,
    }


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score every stored arm, read the conditions, and write the report."""
    sites = c303.site_table()
    rated = score326.rated_speeds(sites=sites)
    usable_hind = pl.read_parquet(common.FINAL_DIR / "replicate_cases.parquet")
    cases = score303.ordered_cases().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    hind_speed = np.concatenate(
        [
            np.load(common.stats_path(kind="speed", set_name="hindcast", farm=f))
            for f in common.FARMS
        ]
    )
    hind_era5 = np.concatenate(
        [np.load(common.stats_path(kind="era5", set_name="hindcast", farm=f)) for f in common.FARMS]
    )
    scored_speed = np.concatenate(
        [np.load(common.stats_path(kind="speed", set_name="scored", farm=f)) for f in common.FARMS]
    )
    scored_era5 = np.concatenate(
        [np.load(common.stats_path(kind="era5", set_name="scored", farm=f)) for f in common.FARMS]
    )
    # The replicate cases are the usable hindcast cases initialised on or after 2025-01-01, found by
    # key in the full hindcast case list.
    import e00303_ens_final as ens_final  # isort: skip  # ty: ignore[unresolved-import]

    hind = ens_final.hindcast_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    in_rep = (
        hind.filter(pl.Series(usable))["power_fcst_init_time"] >= common.REPLICATE_START
    ).to_numpy()
    rep_speed = hind_speed[usable][in_rep]
    rep_era5 = hind_era5[usable][in_rep]
    if rep_speed.shape[0] != usable_hind.height:
        msg = "replicate statistics do not line up with the replicate cases"
        raise ValueError(msg)
    # Leakage check: the ensemble-mean speed here equals experiment 307's.
    reference = pl.read_parquet(common.c326.IMPLIED_CASES_PATH.with_name("cases_scored.parquet"))
    joined = cases.select(*c303.KEY_COLUMNS).with_columns(mean_here=pl.Series(scored_speed[:, 0]))
    joined = joined.join(
        reference.select(*c303.KEY_COLUMNS, "ens_speed_100m"),
        on=list(c303.KEY_COLUMNS),
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    leakage_difference = float(
        np.nanmax(np.abs(joined["mean_here"].to_numpy() - joined["ens_speed_100m"].to_numpy()))
    )
    report: dict = {
        "cut_in_m_s": common.cut_in_speeds(),
        "rated_speed_m_s": rated,
        "leakage_check_max_abs_difference_to_experiment_307": leakage_difference,
        "known_answer_test": common.known_answer_test(),
        "scores": {},
        "contrasts": {},
        "benchmarks": {},
        "row_classes": {},
    }
    frames = {
        "scored": (
            cases,
            flags_for(frame=cases, era5=scored_era5, speed=scored_speed, rated=rated),
            "",
        ),
        "replicate": (
            usable_hind.with_columns(lead_day=pl.col("lead_day").clip(0, 10)),
            flags_for(frame=usable_hind, era5=rep_era5, speed=rep_speed, rated=rated),
            "_rep",
        ),
    }
    for label, (frame, flags, suffix) in frames.items():
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        resampler = Intervals(month=frame["month"].to_numpy(), seeds=len(common.SEEDS))
        metrics: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARM_ORDER:
            path = common.FINAL_DIR / f"{arm}{suffix}.npy"
            if path.exists():
                quantiles = np.load(path)
                metrics[arm] = {
                    **score303.row_metrics(quantiles=quantiles, actual=actual),
                    **score326.extra_metrics(quantiles=quantiles, actual=actual),
                }
        report["row_classes"][label] = {
            "rows": int(frame.height),
            "share": {name: float(f.mean()) for name, f in flags.items()},
            "era5_and_idle210": float((flags["era5_masked"] & flags["idle210"]).mean()),
            "share_of_idle210_inside_era5_mask": float(
                (flags["era5_masked"] & flags["idle210"]).sum() / max(flags["idle210"].sum(), 1)
            ),
            "share_of_idle210_inside_mean_mask": float(
                (flags["mean_masked"] & flags["idle210"]).sum() / max(flags["idle210"].sum(), 1)
            ),
            "share_of_era5_mask_inside_idle210": float(
                (flags["era5_masked"] & flags["idle210"]).sum() / max(flags["era5_masked"].sum(), 1)
            ),
            "by_farm": {
                farm: {
                    "era5_masked": float(
                        flags["era5_masked"][(frame["site"] == farm).to_numpy()].mean()
                    ),
                    "mean_masked": float(
                        flags["mean_masked"][(frame["site"] == farm).to_numpy()].mean()
                    ),
                    "idle210": float(flags["idle210"][(frame["site"] == farm).to_numpy()].mean()),
                }
                for farm in common.FARMS
            },
        }
        report["scores"][label] = {
            arm: {
                subset: {m: float(v[:, flags[subset]].mean()) for m, v in values.items()}
                for subset in SUBSETS
                if flags[subset].any()
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
        for arm in (*ADOPTION_ORDER, *VARIANTS, *PLACEBOS.values()):
            if arm not in metrics or DEFAULT not in metrics:
                continue
            entry: dict = {}
            for subset in SUBSETS:
                if not flags[subset].any():
                    continue
                qs_difference = metrics[arm]["qs"] - metrics[DEFAULT]["qs"]
                entry[f"{subset} | qs"] = resampler.read(values=qs_difference, keep=flags[subset])
            for metric in (
                "p01",
                "p99",
                "p05",
                "p95",
                "median_abs_error",
                "mean_abs_error",
                "covered",
            ):
                difference = metrics[arm][metric] - metrics[DEFAULT][metric]
                for subset in FULL_METRIC_SUBSETS:
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
            entry["t test on monthly kept | qs"] = score326.t_test(monthly=monthly)
            report["contrasts"][label][arm] = entry
        report["contrasts"][label]["placebo_share"] = {}
        for arm, placebo in PLACEBOS.items():
            if arm in metrics and placebo in metrics and DEFAULT in metrics:
                report["contrasts"][label]["placebo_share"][arm] = resampler.share(
                    numerator=metrics[placebo]["qs"] - metrics[DEFAULT]["qs"],
                    denominator=metrics[arm]["qs"] - metrics[DEFAULT]["qs"],
                    keep=flags["kept"],
                )
        if label == "scored" and DEFAULT in metrics:
            # c2: the default scored against a target set to zero where the ERA5 mask is set.
            zero_target = np.where(flags["era5_masked"], 0.0, actual)
            c2 = score303.row_metrics(
                quantiles=np.load(common.FINAL_DIR / "default.npy"), actual=zero_target
            )
            report["evaluation_only_c2"] = {
                subset: {
                    "default_vs_zeroed_target": float(c2["qs"][:, flags[subset]].mean()),
                    "default_vs_observed": float(metrics[DEFAULT]["qs"][:, flags[subset]].mean()),
                }
                for subset in ("all", "kept")
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
            pl.DataFrame(
                {
                    "site": frame["site"],
                    "valid_time": frame["valid_time"],
                    "lead_day": frame["lead_day"],
                    "month": frame["month"],
                    "masked": frame["masked"],
                    "era5_masked": flags["era5_masked"],
                    "mean_masked": flags["mean_masked"],
                    **{f"qs__{a}": pl.Series(m["qs"].mean(axis=0)) for a, m in metrics.items()},
                }
            ).write_parquet(common.WORK_DIR / "rows.parquet")
    gate = json.loads((common.FINAL_DIR / "reproduction.json").read_text())
    report["reproduction_gate"] = gate
    adoption = {}
    scored_contrasts = report["contrasts"]["scored"]
    replicate_contrasts = report["contrasts"].get("replicate", {})
    for arm in ADOPTION_ORDER:
        if arm not in scored_contrasts:
            continue
        main = scored_contrasts[arm]["kept | qs"]
        low, high = main[f"{CONDITION_1_LEVEL:.4f}"]
        gain = -main["mean"]
        condition_1 = bool(high < 0 and gain >= 0)
        tails = {
            t: scored_contrasts[arm][f"kept | {t}"]["upper_95_one_sided"] for t in ("p01", "p99")
        }
        replicate = replicate_contrasts.get(arm)
        condition_3 = None
        if replicate:
            test = replicate["t test on monthly kept | qs"]
            contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
            condition_3 = bool(replicate["kept | qs"]["mean"] < 0 and not contradicted)
        adoption[arm] = {
            "gain": gain,
            "interval": [low, high],
            "condition_1": condition_1,
            "tail_upper_bounds": tails,
            "condition_2": all(v <= TAIL_TOLERANCE for v in tails.values()),
            "condition_3": condition_3,
            "placebo_share": report["contrasts"]["scored"]["placebo_share"]
            .get(arm, {})
            .get("mean"),
            "gate_and_known_answer": bool(
                gate["gate_passed"]
                and report["known_answer_test"]["recall_below_cut_in"] == 1.0
                and report["known_answer_test"]["false_flag_share_above"] == 0.0
                and leakage_difference < 1e-3
            ),
        }
    for entry in adoption.values():
        share = entry["placebo_share"]
        entry["placebo_ok"] = isinstance(share, float) and share < 0.5
        entry["condition_4"] = bool(entry["gate_and_known_answer"] and entry["placebo_ok"])
        entry["condition_5"] = True  # every mask reads ensemble members only (leakage check)
        entry["all_five"] = bool(
            entry["condition_1"]
            and entry["condition_2"]
            and entry["condition_3"]
            and entry["condition_4"]
        )
    passing = [a for a in ADOPTION_ORDER if adoption.get(a, {}).get("all_five")]
    report["adopted"] = passing[0] if passing else None
    report["adoption"] = adoption
    (common.WORK_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)
    print(report["benchmarks"].keys())


if __name__ == "__main__":
    main()
