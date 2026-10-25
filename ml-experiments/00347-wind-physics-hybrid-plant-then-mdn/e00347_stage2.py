"""Stage 2 of experiment 00347: the speed-to-power plant with priors, fitted per farm on ERA5.

Fits the curve (no direction term) by maximum a posteriori with a Laplace likelihood (the pinball
loss at the median) on ERA5 weather before 2024-04-01, with experiment 210's idle-band rows given
weight zero. The plant used downstream takes the 100 m speed to a fitted hub height with the shear
exponent of the 10 m and 100 m speeds and an air-density factor (as experiment 230's core does);
a variant on the 100 m speed alone is reported beside it. Reports the prior-only curve beside the
fitted curve, the ERA5 hold-out scores, and the comparison with experiment 230's physics core
refitted on the same span. The known-answer test is in `e00347_stage2_known_answer.py`.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_stage2.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import time  # noqa: E402
from typing import Final  # noqa: E402

import e00230_core as core230  # noqa: E402  # ty: ignore[unresolved-import]
import e00347_common as common  # noqa: E402
import e00347_eval as evaluation  # noqa: E402
import e00347_plant as plant  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

LIMIT_POINTS: Final[float] = 0.05


def fit_stage2(
    *, farm: common.FarmData, prior_scale: float, shear_density: bool
) -> plant.PlantParameters:
    """Fit the no-direction plant on the farm's training rows (idle-band rows weigh zero)."""
    train = farm.train
    return plant.fit_plant(
        weather=train.weather,
        target=train.target,
        weight=(~train.idle_mask).astype(np.float64),
        prior_scale=prior_scale,
        with_direction=False,
        start=plant.prior_parameters(prior_scale=prior_scale, shear_density=shear_density),
    ).parameters


def core230_scores(*, farm: common.FarmData) -> evaluation.Evaluation:
    """Refit experiment 230's physics core on the training span and score it on the hold-out."""
    train, holdout = farm.train, farm.holdout

    def weather(part: common.FarmData) -> core230.WeatherArrays:
        return core230.WeatherArrays(
            speed_10m=part.speed_10m.astype(np.float32),
            speed_100m=part.speed_100m.astype(np.float32),
            temperature_c=part.temperature_c.astype(np.float32),
        )

    core = core230.fit_core(
        weather=weather(train),
        target=train.target.astype(np.float32),
        with_idle_layer=farm.label != "W2",
        seed=0,
    )
    return evaluation.evaluate_expected(
        farm=farm,
        expected_train=core230.predict_expected_power(core=core, weather=weather(train)).astype(
            np.float64
        ),
        expected_holdout=core230.predict_expected_power(core=core, weather=weather(holdout)).astype(
            np.float64
        ),
    )


def main() -> None:  # noqa: PLR0915
    """Fit, score, test, and save Stage 2."""
    farms = common.load_farms()
    priors = common.prior_scales()
    report = ["STAGE 2: speed-to-power plant with priors (fitted on ERA5 before 2024-04-01)"]
    fitted: dict[str, plant.PlantParameters] = {}
    prior_only: dict[str, plant.PlantParameters] = {}
    spec_only: dict[str, plant.PlantParameters] = {}
    evals: dict[str, dict[str, evaluation.Evaluation]] = {}
    for label, farm in farms.items():
        start = time.time()
        fitted[label] = fit_stage2(farm=farm, prior_scale=priors[label], shear_density=True)
        spec_only[label] = fit_stage2(farm=farm, prior_scale=priors[label], shear_density=False)
        prior_only[label] = plant.prior_parameters(prior_scale=priors[label], shear_density=True)
        evals[label] = {
            "fitted": evaluation.evaluate_plant(
                farm=farm, parameters=fitted[label], prior_scale=priors[label]
            ),
            "prior": evaluation.evaluate_plant(
                farm=farm, parameters=prior_only[label], prior_scale=priors[label]
            ),
            "core230": core230_scores(farm=farm),
            "speed100": evaluation.evaluate_plant(
                farm=farm, parameters=spec_only[label], prior_scale=priors[label]
            ),
        }
        train = farm.train
        report.append(
            f"{label}: training span {str(train.time[0])[:7]} to 2024-03, "
            f"{int((~train.idle_mask).sum())} "
            f"fitted half-hours of {train.time.size} (idle-band rows weight 0); "
            f"{time.time() - start:.0f} s for four fits"
        )
    common.write_json(
        name="plant_stage2.json",
        payload={
            label: {**fitted[label].as_dict(), "prior_scale": priors[label]}
            for label in common.FARMS
        },
    )
    common.write_json(
        name="plant_stage2_speed100_only.json",
        payload={
            label: {**spec_only[label].as_dict(), "prior_scale": priors[label]}
            for label in common.FARMS
        },
    )
    report.append(
        "Plant inputs: ERA5 100 m speed taken to a fitted hub height by the shear exponent "
        "of the 10 m and 100 m speeds, "
        "times the cube root of the air-density ratio (2 m temperature), then a Gaussian "
        "blur over speed "
        "(the 25 km cell's sub-grid wind), then the curve. A variant on the 100 m speed "
        "alone (no hub height) is reported as 'speed100'."
    )
    report.append("-- fitted parameters (prior mean, sd) --")
    for label in common.FARMS:
        p = fitted[label]
        q = spec_only[label]
        report.append(
            f"{label}: cut_in {p.cut_in:.2f} (3.0, 0.5); rated {p.rated:.2f} (12.5, 1.5); "
            f"cut_out {p.cut_out:.2f} (25, 2) "
            "[ERA5 100 m speed above 25 m/s on "
            f"{np.mean(farms[label].train.speed_100m > 25) * 100:.3f}% of training "
            "half-hours, "
            f"above 20 m/s on {np.mean(farms[label].train.speed_100m > 20) * 100:.2f}%; the "
            "data barely inform cut_out; the prior is kept]; "
            f"cut_out_width {p.cut_out_width:.2f} (1.0, 0.5); saturation_width "
            f"{p.saturation_width:.3f} (0.08, 0.05); "
            f"blur {p.blur:.2f} m/s (1.0, 0.7); hub_height {p.hub_height:.1f} m (65, log sd "
            "0.2; a fitted effective height, not a physical one); "
            f"scale {p.scale:.3f} (prior mean {priors[label]:.3f}, sd "
            f"{plant.SCALE_PRIOR_RELATIVE_SD * priors[label]:.3f}) of effective capacity"
        )
        report.append(
            f"{label} speed100 variant: cut_in {q.cut_in:.2f}; rated {q.rated:.2f}; cut_out "
            f"{q.cut_out:.2f}; blur {q.blur:.2f}; scale {q.scale:.3f}"
        )
    report.append(
        "-- ERA5 hold-out 2024-04 to 2025-06: quantile score (points; smaller is better), R "
        "squared of the median, MAE of the median (points) --"
    )
    pooled: dict[str, list[np.ndarray]] = {
        name: [] for name in ("fitted", "prior", "core230", "speed100")
    }
    pooled_months, pooled_kept = [], []
    for label in common.FARMS:
        e = evals[label]
        kept = e["fitted"].kept
        everything = np.ones_like(kept)
        for name, ev in e.items():
            report.append(
                f"{label} {name:>8}: score kept rows {ev.mean_score(rows=kept):.3f}, all rows "
                f"{ev.mean_score():.3f}; "
                f"R2 kept {ev.r_squared(rows=kept):.3f}, all {ev.r_squared(rows=everything):.3f}; "
                f"MAE kept {ev.mean_absolute_error(rows=kept):.2f}, all "
                f"{ev.mean_absolute_error(rows=everything):.2f}"
            )
            pooled[name].append(ev.score)
        pooled_months.append(e["fitted"].month)
        pooled_kept.append(kept)
        diff_kept = common.month_block_gap(
            losses_a=e["fitted"].score[kept],
            losses_b=e["core230"].score[kept],
            months=e["fitted"].month[kept],
        )
        diff_all = common.month_block_gap(
            losses_a=e["fitted"].score, losses_b=e["core230"].score, months=e["fitted"].month
        )
        report.append(
            f"{label}: plant minus 230 core: kept {diff_kept[0]:+.3f} (95% "
            f"{diff_kept[1]:+.3f} to {diff_kept[2]:+.3f}); "
            f"all {diff_all[0]:+.3f} ({diff_all[1]:+.3f} to {diff_all[2]:+.3f})"
        )
    months, kept = np.concatenate(pooled_months), np.concatenate(pooled_kept)
    scores = {name: np.concatenate(values) for name, values in pooled.items()}
    gate_ok = True
    for scope, rows in (("kept rows", kept), ("all rows", np.ones_like(kept))):
        for name in scores:
            report.append(  # noqa: PERF401
                f"3 farms pooled, {scope}, {name}: {scores[name][rows].mean():.3f}"
            )
        for a, b in (
            ("fitted", "core230"),
            ("fitted", "prior"),
            ("speed100", "core230"),
            ("fitted", "speed100"),
        ):
            gap = common.month_block_gap(
                losses_a=scores[a][rows], losses_b=scores[b][rows], months=months[rows]
            )
            report.append(
                f"3 farms pooled, {scope}, {a} minus {b}: {gap[0]:+.3f} (95% {gap[1]:+.3f} to "
                f"{gap[2]:+.3f})"
            )
            if (a, b) == ("fitted", "core230") and abs(gap[0]) > LIMIT_POINTS:
                gate_ok = False
    report.append(
        "GATE (plant within 0.05 of experiment 230's core, pooled kept and all rows): "
        f"{'PASS' if gate_ok else 'FAIL'}"
    )
    grid = np.linspace(0.0, 30.0, 301)
    curves = []
    for label in common.FARMS:
        constant = plant.Weather(
            speed_100m=grid,
            speed_10m=grid / 1.3,
            temperature_c=np.full_like(grid, 10.0),
            direction_rad=np.zeros_like(grid),
        )
        for name, parameters in (("prior only", prior_only[label]), ("fitted", fitted[label])):
            power = plant.expected_power(
                parameters=parameters, weather=constant, prior_scale=priors[label]
            )
            curves.extend(
                {"farm": label, "curve": name, "speed": float(s), "power": float(p)}
                for s, p in zip(grid, power, strict=True)
            )
    work = common.DATA_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(curves).write_parquet(work / "stage2_curves.parquet")
    common.write_report(name="stage2_report.txt", lines=report)
    print("\n".join(report))


if __name__ == "__main__":
    main()
