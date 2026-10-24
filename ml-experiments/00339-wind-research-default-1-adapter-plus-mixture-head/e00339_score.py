"""Score experiment 00339's arms and read the five adoption conditions.

Reads the head parts in `parts/` (written by `e00339_run.py`) and the stored frames of experiments
336 (the default), 303 (the physics core plus adapter, P2 and P3), and 324 (W2, W3, W4). Builds the
rule arms (`x2` = `x1` with all 11 quantiles set to the farm's idle floor where the ensemble-mean
100 m speed is below the farm's cut-in; `default_rule` the same on the default), stores the stitched
arms in `final/`, and writes `score.json`. Intervals resample whole months and the three seeds
(50,000 resamples); farms are not resampled. Condition 1 is read at 1 - 0.01/3.

    uv run --no-sync python ml-experiments/00339-*/e00339_score.py   (PYTHONPATH as for the run)
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_head as head303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_common as c336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_score as score336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00339_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = c336.c303
DEFAULT: Final[str] = "default"
ADOPTION_ORDER: Final[tuple[str, ...]] = ("x2", "x3", "x1")
MIN_GAIN: Final[float] = 0.03
LEVEL: Final[float] = score336.CONDITION_1_LEVEL
TAIL_TOLERANCE: Final[float] = 0.05
ALL_TOLERANCE: Final[float] = 0.05
ORACLE_SHIFT: Final[float] = 0.03
"""Post hoc: +3 points of capacity added to every default quantile, the constant that minimises the
scored kept-row score (an optimistic bound on what a fitted bias correction could reach)."""
HEAD_ARMS: Final[tuple[str, ...]] = ("x1", "x2", "x3")
OTHER_ARMS: Final[tuple[str, ...]] = ("default_rule", "x1_placebo", "x1_lofo")
SUBSETS: Final[tuple[str, ...]] = (
    "all",
    "kept",
    "mean_masked",
    "idle210",
    "rest",
    "days_1_2",
    "days_3_10",
)
FULL_METRIC_SUBSETS: Final[tuple[str, ...]] = ("all", "kept", "days_1_2", "days_3_10")
FAMILIES: Final[dict[str, str]] = {
    "XGBoost (default)": DEFAULT,
    "physics core + adapter (P2)": "P2",
    "physics core + adapter + table (P3)": "P3",
    "weather-only mixture network (W3)": "W3",
}


def parts(*, arm: str, head_seed: int | None = None) -> np.ndarray:
    """Return (base seeds, rows, 11) stacked from the part files of the main tasks."""
    files = [
        common.PARTS_DIR / f"{arm}_b{s}_h{s if head_seed is None else head_seed}.npy"
        for s in common.BASE_SEEDS
    ]
    return np.stack([np.load(f) for f in files])


def kept_score(*, quantiles: np.ndarray, actual: np.ndarray, keep: np.ndarray) -> float:
    """Return the seed-averaged quantile score (points of capacity) over the rows in `keep`."""
    per_seed = [
        c303.quantile_score(quantiles=q[keep].astype(np.float64), actual=actual[keep]).mean()
        for q in quantiles
    ]
    return float(np.mean(per_seed) * c303.POINTS_PER_FRACTION)


def apply_rule(*, quantiles: np.ndarray, mask: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Return `quantiles` with every quantile of the masked cases set to that case's farm floor."""
    out = quantiles.copy()
    out[:, mask, :] = floor[mask][None, :, None]
    return out


def main() -> None:  # noqa: PLR0912, PLR0915
    """Build the arms, score them, and read the adoption conditions."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    rated = score326.rated_speeds(sites=sites)
    cases = score303.ordered_cases().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    hind_all = ens_final.hindcast_cases(sites=sites)
    usable = hind_all["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hindcast = head303.with_summaries(cases=hind_all.filter(pl.Series(usable)), set_name="hindcast")
    in_rep = (hindcast["power_fcst_init_time"] >= common.REPLICATE_START).to_numpy()
    replicate = pl.read_parquet(c336.FINAL_DIR / "replicate_cases.parquet").with_columns(
        lead_day=pl.col("lead_day").clip(0, 10)
    )
    assert replicate.height == int(in_rep.sum()), (replicate.height, int(in_rep.sum()))
    assert (replicate["masked"].to_numpy() == hindcast["masked"].to_numpy()[in_rep]).all()

    def stats(kind: str, set_name: str) -> np.ndarray:
        return np.concatenate(
            [np.load(c336.stats_path(kind=kind, set_name=set_name, farm=f)) for f in common.FARMS]
        )

    scored_speed, scored_era5 = stats("speed", "scored"), stats("era5", "scored")
    hind_speed, hind_era5 = stats("speed", "hindcast")[usable], stats("era5", "hindcast")[usable]
    frames = {
        "scored": (
            cases,
            score336.flags_for(frame=cases, era5=scored_era5, speed=scored_speed, rated=rated),
            "",
            scored_speed,
        ),
        "replicate": (
            replicate,
            score336.flags_for(
                frame=replicate, era5=hind_era5[in_rep], speed=hind_speed[in_rep], rated=rated
            ),
            "_rep",
            hind_speed[in_rep],
        ),
    }
    for _, flags, _, _ in frames.values():
        flags["rest"] = ~flags["mean_masked"] & ~flags["idle210"]
    cut = c336.cut_in_speeds()
    floors = {}
    for window, end in (("scored", common.MAIN_CUTOFF), ("replicate", common.REPLICATE_START)):
        rows = (
            hindcast["masked"].to_numpy()
            & (hindcast["valid_time"] < end).to_numpy()
            & (hindcast["power_fcst_init_time"] < end).to_numpy()
        )
        floors[window] = {
            f: float(
                np.median(
                    hindcast["actual_norm"].to_numpy()[rows & (hindcast["site"] == f).to_numpy()]
                )
            )
            for f in common.FARMS
        }
    report: dict = {
        "cut_in_m_s": cut,
        "idle_floor_fraction_of_capacity": floors,
        "head_input_columns": [*head303.SUMMARY_FEATURES, "the 11 quantiles of P2"],
        "scores": {},
        "contrasts": {},
        "benchmarks": {},
        "row_classes": {},
    }
    stored_w4 = np.load(common.E324_FINAL / "W4.npy")
    x1_main = parts(arm="x1")
    report["x1_vs_experiment_324_W4_max_abs_difference"] = float(np.abs(x1_main - stored_w4).max())
    report["x1_vs_experiment_324_W4_kept_score"] = {
        "here": kept_score(
            quantiles=x1_main,
            actual=cases["actual_norm"].to_numpy().astype(np.float64),
            keep=frames["scored"][1]["kept"],
        ),
        "stored": kept_score(
            quantiles=stored_w4,
            actual=cases["actual_norm"].to_numpy().astype(np.float64),
            keep=frames["scored"][1]["kept"],
        ),
    }
    arms_by_label: dict[str, dict[str, np.ndarray]] = {}
    for label, (frame, flags, suffix, speed) in frames.items():
        sites_arr = frame["site"].to_numpy()
        mask = speed[:, 0] < np.array([cut[s] for s in sites_arr])
        floor = np.array([floors[label][s] for s in sites_arr])
        arms: dict[str, np.ndarray] = {DEFAULT: np.load(c336.FINAL_DIR / f"{DEFAULT}{suffix}.npy")}
        arms["x1"] = parts(arm=f"x1{suffix}")
        arms["x3"] = parts(arm=f"x3{suffix}")
        arms["x2"] = apply_rule(quantiles=arms["x1"], mask=mask, floor=floor)
        arms["default_rule"] = apply_rule(quantiles=arms[DEFAULT], mask=mask, floor=floor)
        if label == "scored":
            arms["x1_placebo"] = parts(arm="x1_placebo")
            arms["x1_lofo"] = parts(arm="x1_lofo")
            for name, source in (("P2", "P2"), ("P3", "P3")):
                arms[name] = np.load(c336.c326.STORED_303_FINAL_DIR / f"{source}.npy")
            arms["W2"] = np.load(common.E324_FINAL / "W2.npy")
            arms["W3"] = np.load(common.E324_FINAL / "W3.npy")
            arms["x1_baseonly"] = parts(arm="x1_baseonly")
            arms["default_shift_oracle"] = np.clip(arms[DEFAULT] + ORACLE_SHIFT, 0.0, 1.0)
            arms["default_notable"] = np.load(c336.FINAL_DIR / "default_notable.npy")
        arms_by_label[label] = arms
        for name, array in arms.items():
            np.save(common.FINAL_DIR / f"{name}{suffix}.npy", array)
        report["row_classes"][label] = {
            "rows": int(frame.height),
            "share": {name: float(f.mean()) for name, f in flags.items()},
        }
        actual = frame["actual_norm"].to_numpy().astype(np.float64)
        resampler = score336.Intervals(
            month=frame["month"].to_numpy(), seeds=len(common.BASE_SEEDS)
        )
        metrics = {
            name: {
                **score303.row_metrics(quantiles=q, actual=actual),
                **score326.extra_metrics(quantiles=q, actual=actual),
            }
            for name, q in arms.items()
        }
        report["scores"][label] = {
            name: {
                subset: {m: float(v[:, flags[subset]].mean()) for m, v in values.items()}
                for subset in SUBSETS
                if flags[subset].any()
            }
            | {
                "by_farm_kept": {
                    f: float(
                        values["qs"][:, flags["kept"] & (frame["site"] == f).to_numpy()].mean()
                    )
                    for f in common.FARMS
                },
                "by_lead_day_kept": {
                    str(d): float(
                        values["qs"][:, flags["kept"] & (frame["lead_day"] == d).to_numpy()].mean()
                    )
                    for d in range(1, 11)
                    if (flags["kept"] & (frame["lead_day"] == d).to_numpy()).any()
                },
            }
            for name, values in metrics.items()
        }
        report["contrasts"][label] = {}
        reference = (
            *HEAD_ARMS,
            "default_rule",
            "x1_placebo",
            "x1_lofo",
            "x1_baseonly",
            "default_shift_oracle",
        )
        for arm in reference:
            if arm not in metrics:
                continue
            entry: dict = {}
            difference = metrics[arm]["qs"] - metrics[DEFAULT]["qs"]
            for subset in SUBSETS:
                if flags[subset].any():
                    entry[f"{subset} | qs"] = resampler.read(values=difference, keep=flags[subset])
            for metric in (
                "p01",
                "p99",
                "p05",
                "p95",
                "median_abs_error",
                "mean_abs_error",
                "covered",
            ):
                d = metrics[arm][metric] - metrics[DEFAULT][metric]
                for subset in FULL_METRIC_SUBSETS:
                    entry[f"{subset} | {metric}"] = resampler.read(values=d, keep=flags[subset])
            for farm in common.FARMS:
                site = (frame["site"] == farm).to_numpy()
                entry[f"kept only {farm} | qs"] = resampler.read(
                    values=difference, keep=flags["kept"] & site
                )
                entry[f"kept without {farm} | qs"] = resampler.read(
                    values=difference, keep=flags["kept"] & ~site
                )
            months = np.unique(frame["month"].to_numpy())
            monthly = np.array(
                [
                    difference[:, flags["kept"] & (frame["month"] == m).to_numpy()].mean()
                    for m in months
                ]
            )
            entry["monthly kept | qs"] = {"months": months.tolist(), "values": monthly.tolist()}
            entry["t test on monthly kept | qs"] = score326.t_test(monthly=monthly)
            report["contrasts"][label][arm] = entry
        if label == "scored":
            # Contrasts among arms.
            extra = {
                "x1_minus_x1_placebo": ("x1", "x1_placebo"),
                "x2_minus_x1": ("x2", "x1"),
                "x3_minus_x1": ("x3", "x1"),
                "x2_minus_default_rule": ("x2", "default_rule"),
                "x1_lofo_minus_default": ("x1_lofo", DEFAULT),
                "x1_minus_x1_baseonly": ("x1", "x1_baseonly"),
                "x1_minus_default_shift_oracle": ("x1", "default_shift_oracle"),
            }
            report["pairs"] = {}
            for name, (a, b) in extra.items():
                d = metrics[a]["qs"] - metrics[b]["qs"]
                report["pairs"][name] = {
                    s: resampler.read(values=d, keep=flags[s]) for s in SUBSETS if flags[s].any()
                }
                report["pairs"][name]["idle210_loss_recovered"] = None
            # Share of x1's idle-row loss that x2 and x3 recover.
            idle = flags["idle210"]
            loss = (metrics["x1"]["qs"] - metrics[DEFAULT]["qs"])[:, idle].mean()
            for arm in ("x2", "x3"):
                after = (metrics[arm]["qs"] - metrics[DEFAULT]["qs"])[:, idle].mean()
                report.setdefault("idle_loss_recovered", {})[arm] = {
                    "x1_loss_on_idle210": float(loss),
                    "arm_loss_on_idle210": float(after),
                    "share_recovered": float((loss - after) / loss),
                }
            # Standard architecture and the best single family.
            fam = {k: float(metrics[v]["qs"][:, flags["kept"]].mean()) for k, v in FAMILIES.items()}
            report["family_kept_scores"] = fam
            best_family = min(fam, key=fam.__getitem__)
            best_head = min(
                HEAD_ARMS, key=lambda a: float(metrics[a]["qs"][:, flags["kept"]].mean())
            )
            gap = metrics[best_head]["qs"] - metrics[FAMILIES[best_family]]["qs"]
            relative = gap / np.maximum(metrics[FAMILIES[best_family]]["qs"], 1e-9)
            report["standard_architecture"] = {
                "arm": best_head,
                "best_family": best_family,
                "gap_points_kept": resampler.read(values=gap, keep=flags["kept"]),
                "gap_points_all": resampler.read(values=gap, keep=flags["all"]),
                "gap_relative_kept": float(
                    (metrics[best_head]["qs"][:, flags["kept"]].mean())
                    / metrics[FAMILIES[best_family]]["qs"][:, flags["kept"]].mean()
                    - 1
                ),
                "relative_row_mean_kept": float(relative[:, flags["kept"]].mean()),
            }
            report["family_contrasts_vs_default"] = {
                k: resampler.read(
                    values=metrics[v]["qs"] - metrics[DEFAULT]["qs"], keep=flags["kept"]
                )
                for k, v in FAMILIES.items()
                if v != DEFAULT
            }
            # Seed check on base seed 0.
            checks = {}
            for head in range(5):
                path = common.PARTS_DIR / f"x1_b0_h{head}.npy"
                if path.exists():
                    q = np.load(path)[None]
                    checks[str(head)] = float(
                        c303.quantile_score(
                            quantiles=q[0][flags["kept"]].astype(np.float64),
                            actual=actual[flags["kept"]],
                        ).mean()
                        * c303.POINTS_PER_FRACTION
                    )
            values = np.array(list(checks.values()))
            report["seed_check_x1_base0_kept"] = {
                "by_head_seed": checks,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                "default_base0_kept": float(metrics[DEFAULT]["qs"][0, flags["kept"]].mean()),
            }
            # Benchmarks.
            for name in ("CLIM", "PERS"):
                ref = score326.benchmark_frame(name=name, cases=frame)
                if ref is None:
                    continue
                present = np.isfinite(ref[0]).all(axis=1)
                row_score = (
                    c303.quantile_score(quantiles=np.nan_to_num(ref[0]), actual=actual)
                    * c303.POINTS_PER_FRACTION
                )
                keep = flags["kept"] & present
                report["benchmarks"][name] = {
                    "rows": int(keep.sum()),
                    "benchmark": float(row_score[keep].mean()),
                    "default_on_same_rows": float(metrics[DEFAULT]["qs"][:, keep].mean()),
                    **{
                        f"{a}_on_same_rows": float(metrics[a]["qs"][:, keep].mean())
                        for a in HEAD_ARMS
                    },
                }
            report["benchmarks"]["XGBoost_reference"] = {
                "benchmark": float(metrics["default_notable"]["qs"][:, flags["kept"]].mean())
            }
            pl.DataFrame(
                {
                    "site": frame["site"],
                    "valid_time": frame["valid_time"],
                    "lead_day": frame["lead_day"],
                    "month": frame["month"],
                    "masked": frame["masked"],
                    "mean_masked": flags["mean_masked"],
                    **{f"qs__{a}": pl.Series(m["qs"].mean(axis=0)) for a, m in metrics.items()},
                }
            ).write_parquet(common.DATA_DIR / "rows.parquet")
    gate = json.loads((c336.FINAL_DIR / "reproduction.json").read_text())
    report["reproduction_gate"] = gate
    scored = report["contrasts"]["scored"]
    rep = report["contrasts"].get("replicate", {})
    adoption = {}
    placebo = report["pairs"]["x1_minus_x1_placebo"]["kept"]["mean"]
    seed = report["seed_check_x1_base0_kept"]
    for arm in ADOPTION_ORDER:
        main = scored[arm]["kept | qs"]
        low, high = main[f"{LEVEL:.4f}"]
        gain = -main["mean"]
        tails = {t: scored[arm][f"kept | {t}"]["upper_95_one_sided"] for t in ("p01", "p99")}
        test = rep.get(arm, {}).get("t test on monthly kept | qs")
        condition_3 = None
        if test:
            contradicted = test["t"] > 0 and test["p_two_sided"] < 0.05
            condition_3 = bool(rep[arm]["kept | qs"]["mean"] < 0 and not contradicted)
        adoption[arm] = {
            "gain": gain,
            "interval": [low, high],
            "condition_1": bool(high < 0 and gain >= MIN_GAIN),
            "tail_upper_bounds": tails,
            "condition_2": all(v <= TAIL_TOLERANCE for v in tails.values()),
            "all_half_hours_difference": scored[arm]["all | qs"]["mean"],
            "all_half_hours_within_tolerance": bool(
                scored[arm]["all | qs"]["mean"] <= ALL_TOLERANCE
            ),
            "condition_3": condition_3,
            "placebo_loses_gain": bool(placebo < -0.5 * gain),
            "seed_sd_below_half_gain": bool(seed["sd"] is not None and seed["sd"] < 0.5 * gain),
            "gate": bool(gate["gate_passed"]),
        }
        entry = adoption[arm]
        entry["condition_4"] = bool(
            entry["gate"] and entry["placebo_loses_gain"] and entry["seed_sd_below_half_gain"]
        )
        entry["all_five_numeric"] = bool(
            entry["condition_1"]
            and entry["condition_2"]
            and entry["condition_3"]
            and entry["condition_4"]
        )
    passing = [a for a in ADOPTION_ORDER if adoption[a]["all_five_numeric"]]
    report["adopted"] = passing[0] if passing else None
    report["adoption"] = adoption
    (common.DATA_DIR / "score.json").write_text(json.dumps(report, indent=1))
    for arm, entry in adoption.items():
        print(arm, entry)
    print("standard architecture", report["standard_architecture"])
    print("idle", report["idle_loss_recovered"])
    print("seed", seed)
    print(
        "w4 check",
        report["x1_vs_experiment_324_W4_max_abs_difference"],
        report["x1_vs_experiment_324_W4_kept_score"],
    )


if __name__ == "__main__":
    main()
