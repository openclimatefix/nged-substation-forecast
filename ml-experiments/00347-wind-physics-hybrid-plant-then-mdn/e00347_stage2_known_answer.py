"""Known-answer test for the Stage 2 plant fit of experiment 00347.

Simulates power from known plant parameters at the real ERA5 weather plus noise, refits with the
Stage 2 procedure, and checks that the fit recovers the parameters within the Laplace-approximation
posterior. The noise is real: each half-hour takes the residual of a random training half-hour in
the same bin of fitted power (25 bins), after removing the bin's median so that the simulated
conditional median is the known curve. Noise draws are independent between half-hours.

    export PYTHONPATH=ml-experiments/00230-wind-physics-core-part-1:\
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn
    uv run --no-sync python \
        ml-experiments/00347-wind-physics-hybrid-plant-then-mdn/e00347_stage2_known_answer.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from typing import Final  # noqa: E402

import e00347_common as common  # noqa: E402
import e00347_plant as plant  # noqa: E402
import numpy as np  # noqa: E402

REPLICATES: Final[int] = 6
TRUTH: Final[dict[str, float]] = {
    "cut_in": 3.3,
    "rated": 12.0,
    "cut_out": 24.0,
    "cut_out_width": 1.2,
    "saturation_width": 0.10,
    "blur": 1.5,
    "hub_height": 40.0,
}
SCALE_FACTOR: Final[float] = 1.03
"""The true flat top is this multiple of the prior mean."""
NAMES: Final[tuple[str, ...]] = (*TRUTH, "scale")
NOISE_BINS: Final[int] = 25
WORKERS: Final[int] = 9
POSTERIOR_SDS: Final[float] = 2.0


def task(args: tuple[str, int]) -> tuple[str, int, dict[str, float], dict[str, float]]:
    """Simulate once at one farm, refit, and return the estimates and their posterior sds."""
    label, replicate = args
    farm = common.load_farms()[label]
    prior_scale = common.prior_scales()[label]
    train = farm.train
    weight = (~train.idle_mask).astype(np.float64)
    real = plant.PlantParameters.from_dict(common.read_json(name="plant_stage2.json")[label])
    real_fit = plant.expected_power(parameters=real, weather=train.weather, prior_scale=prior_scale)
    residual = train.target - real_fit
    truth = plant.PlantParameters(
        cut_in=TRUTH["cut_in"],
        rated=TRUTH["rated"],
        cut_out=TRUTH["cut_out"],
        cut_out_width=TRUTH["cut_out_width"],
        saturation_width=TRUTH["saturation_width"],
        blur=TRUTH["blur"],
        hub_height=TRUTH["hub_height"],
        scale=prior_scale * SCALE_FACTOR,
    )
    expected = plant.expected_power(
        parameters=truth, weather=train.weather, prior_scale=prior_scale
    )
    rng = np.random.default_rng(1000 + replicate)
    edges = np.quantile(real_fit[weight > 0], np.linspace(0, 1, NOISE_BINS + 1))
    donor_bin = np.clip(np.digitize(real_fit, edges[1:-1]), 0, NOISE_BINS - 1)
    row_bin = np.clip(np.digitize(expected, edges[1:-1]), 0, NOISE_BINS - 1)
    donors = []
    for b in range(NOISE_BINS):
        rows = np.flatnonzero((donor_bin == b) & (weight > 0))
        donors.append(residual[rows] - np.median(residual[rows]))
    noise = np.array([donors[b][rng.integers(0, donors[b].size)] for b in row_bin])
    result = plant.fit_plant(
        weather=train.weather,
        target=expected + noise,
        weight=weight,
        prior_scale=prior_scale,
        with_direction=False,
        start=plant.prior_parameters(prior_scale=prior_scale, shear_density=True),
    )
    sd = plant.posterior_sd(
        parameters=result.parameters,
        weather=train.weather,
        weight=weight,
        prior_scale=prior_scale,
        laplace_scale=result.laplace_scale,
    )
    return label, replicate, {n: float(getattr(result.parameters, n)) for n in NAMES}, sd


def main() -> None:
    """Run the test and write its report."""
    priors = common.prior_scales()
    report = [
        (
            "STAGE 2 KNOWN-ANSWER TEST: simulate at real ERA5 weather from known parameters, "
            "refit, compare"
        ),
        "truth "
        + ", ".join(f"{k} {v}" for k, v in TRUTH.items())
        + ", scale 1.03 x the prior mean; "
        f"{REPLICATES} noise draws per farm; recovered means |mean estimate - truth| <= "
        f"{POSTERIOR_SDS:.0f} Laplace posterior sd",
    ]
    tasks = [(label, k) for label in common.FARMS for k in range(REPLICATES)]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        results = list(pool.map(task, tasks))
    passed_all = True
    for label in common.FARMS:
        mine = [r for r in results if r[0] == label]
        truth = {**TRUTH, "scale": priors[label] * SCALE_FACTOR}
        for name in NAMES:
            estimates = np.array([r[2][name] for r in mine])
            posterior = float(np.mean([r[3][name] for r in mine]))
            mean, spread = float(estimates.mean()), float(estimates.std(ddof=1))
            error = mean - truth[name]
            ok = abs(error) <= POSTERIOR_SDS * posterior
            passed_all &= ok
            report.append(
                f"{label} {name}: truth {truth[name]:.3f}, mean estimate {mean:.3f} (sd over "
                f"draws {spread:.3f}), "
                f"posterior sd {posterior:.3f}, error {error:+.3f} = {error / posterior:+.2f} "
                "posterior sd, "
                f"{'recovered' if ok else 'NOT recovered'}"
            )
    report.append(f"KNOWN-ANSWER TEST: {'PASS' if passed_all else 'FAIL'}")
    common.write_report(name="stage2_known_answer_report.txt", lines=report)
    print("\n".join(report))


if __name__ == "__main__":
    main()
