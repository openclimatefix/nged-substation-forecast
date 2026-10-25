"""Stage 3 of experiment 00347: the plant's direction-dependent loss (pass 1 of Stage 3b).

Adds a multiplicative factor on the wind speed, `f(direction) = 1 - sum of 3 von Mises dips`, to
the Stage 2 plant. Depths are held between 0 and 15% by a sigmoid and pulled towards 0 by a
half-normal prior (so `f` is 1 everywhere at the prior); the dip centres start at the deepest local
minima of the Stage 1 direction curve. The fit uses every training half-hour (experiment 210's
idle-band rows have weight 0), compares with the Stage 2 plant on the ERA5 hold-out, and writes the
frozen pass-1 plant.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_stage3.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json  # noqa: E402
from dataclasses import replace  # noqa: E402
from typing import Final  # noqa: E402

import e00347_common as common  # noqa: E402
import e00347_eval as evaluation  # noqa: E402
import e00347_plant as plant  # noqa: E402
import e00347_stage1 as stage1  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

START_DEPTH_FLOOR: Final[float] = 0.01
START_DEPTH_CEILING: Final[float] = 0.08
START_KAPPA: Final[float] = 4.0


def start_parameters(
    *, base: plant.PlantParameters, minima: list[dict[str, float]], log_slope: float
) -> plant.PlantParameters:
    """Return the Stage 2 parameters with 3 dips started at the Stage 1 curve's deepest minima.

    The Stage 1 depth is a loss of power; dividing by the log-slope of the curve gives the
    equivalent loss of wind speed that the dip multiplies.
    """
    centres, depths = [], []
    for minimum in minima[: plant.DIP_COUNT]:
        centres.append(minimum["centre_deg"])
        depths.append(
            float(np.clip(minimum["depth"] / log_slope, START_DEPTH_FLOOR, START_DEPTH_CEILING))
        )
    fill = (
        [(centres[0] + 120.0 * k) % 360.0 for k in range(1, plant.DIP_COUNT + 1)]
        if centres
        else [0.0, 120.0, 240.0]
    )
    while len(centres) < plant.DIP_COUNT:
        centres.append(fill[len(centres)])
        depths.append(START_DEPTH_FLOOR)
    return replace(
        base,
        depths=tuple(depths),
        centres_deg=tuple(centres),
        kappas=(START_KAPPA,) * plant.DIP_COUNT,
    )


def plant_direction_ratio(
    *,
    farm: common.FarmData,
    stage2: plant.PlantParameters,
    stage3: plant.PlantParameters,
    prior_scale: float,
) -> np.ndarray:
    """Return the plant's direction ratio on the 1 degree grid, as Stage 1 measures the data's.

    The ratio is the kernel-smoothed sum of the Stage 3 plant's power over the Stage 2 plant's
    power on training half-hours whose Stage 2 power lies at 40% to 90% of its flat top, divided by
    its all-direction value.
    """
    train = farm.train
    kept = ~train.idle_mask
    weather = train.weather.take(np.flatnonzero(kept))
    without = plant.expected_power(parameters=stage2, weather=weather, prior_scale=prior_scale)
    with_dir = plant.expected_power(parameters=stage3, weather=weather, prior_scale=prior_scale)
    top = stage2.scale
    inside = (without >= stage1.WINDOW[0] * top) & (without <= stage1.WINDOW[1] * top)
    degree = (np.degrees(weather.direction_rad[inside]).round().astype(np.int64)) % 360
    numerator = np.bincount(degree, weights=with_dir[inside], minlength=360)
    denominator = np.bincount(degree, weights=without[inside], minlength=360)
    kernel = stage1.kernel_matrix()
    overall = numerator.sum() / max(denominator.sum(), 1e-9)
    return (kernel @ numerator) / np.maximum(kernel @ denominator, 1e-9) / overall


def main() -> None:
    """Fit the Stage 3 plants, score them on the hold-out, and save the frozen parameters."""
    farms = common.load_farms()
    priors = common.prior_scales()
    stage2 = {
        label: plant.PlantParameters.from_dict(v)
        for label, v in common.read_json(name="plant_stage2.json").items()
    }
    summary = json.loads((common.DATA_DIR / "stage1_summary.json").read_text())
    report = [
        (
            "STAGE 3: direction-dependent loss (K = 3 von Mises dips on wind speed); this fit is "
            "pass 1 of Stage 3b"
        )
    ]
    fitted, evals2, evals3 = {}, {}, {}
    ratio_rows = []
    for label, farm in farms.items():
        train = farm.train
        start = start_parameters(
            base=stage2[label],
            minima=summary[label]["minima"],
            log_slope=summary[label]["log_slope"],
        )
        fitted[label] = plant.fit_plant(
            weather=train.weather,
            target=train.target,
            weight=(~train.idle_mask).astype(np.float64),
            prior_scale=priors[label],
            with_direction=True,
            start=start,
        ).parameters
        evals2[label] = evaluation.evaluate_plant(
            farm=farm, parameters=stage2[label], prior_scale=priors[label]
        )
        evals3[label] = evaluation.evaluate_plant(
            farm=farm, parameters=fitted[label], prior_scale=priors[label]
        )
        ratio = plant_direction_ratio(
            farm=farm, stage2=stage2[label], stage3=fitted[label], prior_scale=priors[label]
        )
        data_ratio = np.array(summary[label]["ratio"])
        ratio_rows.extend(
            {"farm": label, "direction_deg": d, "plant_ratio": float(ratio[d])} for d in range(360)
        )
        p = fitted[label]
        report.append(
            f"{label}: Stage 3 parameters: cut_in {p.cut_in:.2f}; rated {p.rated:.2f}; "
            f"cut_out {p.cut_out:.2f}; blur {p.blur:.2f}; hub_height {p.hub_height:.1f}; "
            f"scale {p.scale:.3f}"
        )
        for k in range(plant.DIP_COUNT):
            width = 2 * np.degrees(np.arccos(1 + np.log(0.5) / p.kappas[k]))
            report.append(
                f"{label}: dip {k + 1}: depth {p.depths[k]:.3f} of wind speed (started at "
                f"{start.depths[k]:.3f}), centre {p.centres_deg[k]:.0f} deg "
                f"(Stage 1 start {start.centres_deg[k]:.0f}), width at half depth {width:.0f} "
                f"deg (kappa {p.kappas[k]:.1f})"
            )
        low = int(np.argmin(ratio))
        explained = 1 - np.sum((ratio - data_ratio) ** 2) / np.sum(
            (data_ratio - data_ratio.mean()) ** 2
        )
        report.append(
            f"{label}: the plant's implied power ratio by direction has its minimum "
            f"{1 - ratio[low]:.3f} at {low} deg "
            f"(Stage 1 data curve: {summary[label]['dip']['depth']:.3f} at "
            f"{summary[label]['dip']['centre_deg']:.0f} deg); "
            "RMS difference between the two curves "
            f"{np.sqrt(np.mean((ratio - data_ratio) ** 2)):.3f}; "
            "share of the data curve's variance about 1 that the plant curve explains "
            f"{explained:.2f}"
        )
    common.write_json(
        name="plant_stage3.json",
        payload={
            label: {**fitted[label].as_dict(), "prior_scale": priors[label]}
            for label in common.FARMS
        },
    )
    report.append(
        "-- ERA5 hold-out 2024-04 to 2025-06, quantile score in points (smaller is better) --"
    )
    gains = {}
    pooled2, pooled3, pooled_kept, pooled_months = [], [], [], []
    for label in common.FARMS:
        e2, e3 = evals2[label], evals3[label]
        kept = e2.kept
        everything = np.ones_like(kept)
        for scope, rows in (("kept rows", kept), ("all rows", everything)):
            gap = common.month_block_gap(
                losses_a=e3.score[rows], losses_b=e2.score[rows], months=e2.month[rows]
            )
            report.append(
                f"{label} {scope}: Stage 2 {e2.mean_score(rows=rows):.3f}, Stage 3 "
                f"{e3.mean_score(rows=rows):.3f}, "
                f"Stage 3 minus Stage 2 {gap[0]:+.3f} (95% {gap[1]:+.3f} to {gap[2]:+.3f}); "
                f"R2 Stage 2 {e2.r_squared(rows=rows):.3f}, Stage 3 {e3.r_squared(rows=rows):.3f}; "
                f"MAE Stage 2 {e2.mean_absolute_error(rows=rows):.2f}, Stage 3 "
                f"{e3.mean_absolute_error(rows=rows):.2f}"
            )
            gains[(label, scope)] = -gap[0]
        (
            pooled2.append(e2.score),
            pooled3.append(e3.score),
            pooled_kept.append(kept),
            pooled_months.append(e2.month),
        )
    s2, s3, kept, months = map(np.concatenate, (pooled2, pooled3, pooled_kept, pooled_months))
    for scope, rows in (("kept rows", kept), ("all rows", np.ones_like(kept))):
        gap = common.month_block_gap(losses_a=s3[rows], losses_b=s2[rows], months=months[rows])
        report.append(
            f"3 farms pooled {scope}: Stage 2 {s2[rows].mean():.3f}, Stage 3 "
            f"{s3[rows].mean():.3f}, Stage 3 minus Stage 2 {gap[0]:+.3f} (95% {gap[1]:+.3f} "
            f"to {gap[2]:+.3f})"
        )
    k = "kept rows"
    pooled_gain = float(s2[kept].mean() - s3[kept].mean())
    report.append(
        "PREDICTION CHECK (Stage 3 lowers the kept-row hold-out score by 0.02 to 0.1 "
        f"points): pooled gain {pooled_gain:.3f} -> "
        f"{'holds' if 0.02 <= pooled_gain <= 0.1 else 'FAILS'}"
    )
    report.append(
        "PREDICTION CHECK (gain larger at W1 and W2 than at W3): kept-row gains W1 "
        f"{gains[('W1', k)]:.3f}, W2 {gains[('W2', k)]:.3f}, W3 {gains[('W3', k)]:.3f} -> "
        f"{'holds' if min(gains[('W1', k)], gains[('W2', k)]) > gains[('W3', k)] else 'FAILS'}"
    )
    work = common.DATA_DIR / "work"
    pl.DataFrame(ratio_rows).write_parquet(work / "stage3_plant_ratio.parquet")
    common.write_report(name="stage3_report.txt", lines=report)
    print("\n".join(report))


if __name__ == "__main__":
    main()
