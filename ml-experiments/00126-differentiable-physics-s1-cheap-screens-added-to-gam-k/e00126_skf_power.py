"""G-skf's power gate: does it recover a known regime path and track the state better than plain?

**The synthetic stream** is drawn from G-skf's own model at one substation's tuned hyperparameters
and its fitted transition matrix: a regime path from the matrix, a level that follows a random walk
(100 times the step variance in the shift regime), a transient that follows an AR(1), and an
observation noise with 100 times the variance in the outlier regime. There are 100 draws of 180 days
of consecutive half-hours.

**Pass** if, over the draws, the filter's regime call (shift or outlier against normal, at a
probability of 0.5) is right on at least 80% of half-hours, and the mean squared error of its
tracked state (level plus transient) is below the plain filter's with a 98.75% bootstrap interval
over draws wholly below zero.

    uv run --no-sync python ml-experiments/00126-*/e00126_skf_power.py [--site D10]
"""

import argparse
import json
from typing import Final

import e00126_common as common
import e00126_skf as skf
import numpy as np
from studies.dp_assimilation import FilterHyperparameters

DRAWS: Final[int] = 100
HALF_HOURS: Final[int] = 180 * 48
LEVEL: Final[float] = 98.75
MIN_RECOVERY: Final[float] = 0.80
SEED_TAG: Final[int] = 5


def simulate(
    *, hyper: FilterHyperparameters, transition: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw regime paths, true states, and readings for `DRAWS` streams."""
    q_step = hyper.q / skf.HALF_HOURS_PER_DAY
    stationary = hyper.sigma_z**2 / (1.0 - hyper.phi**2)
    regime = np.zeros((DRAWS, HALF_HOURS), dtype=int)
    cumulative = np.cumsum(transition, axis=1)
    for t in range(1, HALF_HOURS):
        draw = rng.random(DRAWS)
        regime[:, t] = (draw[:, None] > cumulative[regime[:, t - 1]]).sum(axis=1)
    level = np.zeros((DRAWS, HALF_HOURS))
    anomaly = np.zeros((DRAWS, HALF_HOURS))
    anomaly[:, 0] = rng.normal(0.0, np.sqrt(stationary), DRAWS)
    for t in range(1, HALF_HOURS):
        multiplier = np.where(regime[:, t] == skf.SHIFT, skf.SHIFT_MULTIPLIER, 1.0)
        level[:, t] = level[:, t - 1] + rng.normal(0.0, np.sqrt(q_step * multiplier))
        anomaly[:, t] = hyper.phi * anomaly[:, t - 1] + rng.normal(0.0, hyper.sigma_z, DRAWS)
    noise_scale = np.where(regime == skf.OUTLIER, np.sqrt(skf.OUTLIER_MULTIPLIER), 1.0)
    reading = level + anomaly + rng.normal(0.0, hyper.sigma_eps, (DRAWS, HALF_HOURS)) * noise_scale
    return regime, level + anomaly, reading


def main() -> None:
    """Run the gate and print its numbers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="D10")
    arguments = parser.parse_args()
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    hyper = FilterHyperparameters.from_array(
        values=np.array(decisions["filter_hyper"]["t_all"][arguments.site])
    )
    transitions = json.loads(
        (common.WORK_DIR / "transitions_tuning_replicate_scored.json").read_text()
    )
    transition = np.array(transitions[arguments.site]["tuning"])
    rng = np.random.default_rng([common.EXPERIMENT_ID, SEED_TAG])
    regime, truth, reading = simulate(hyper=hyper, transition=transition, rng=rng)
    positions = np.arange(HALF_HOURS)
    switching = skf.gpb1_band(
        positions=positions, values=reading, hyper=hyper, transition=transition
    )
    plain = skf.gpb1_band(
        positions=positions,
        values=reading,
        hyper=hyper,
        transition=transition,
        multipliers=(1.0, 1.0),
    )
    called = switching.p_bad > 0.5
    actual = regime != skf.NORMAL
    recovery = float((called == actual).mean())
    bad_recall = float(called[actual].mean()) if actual.any() else float("nan")
    gskf_mse = ((switching.level + switching.anomaly - truth) ** 2).mean(axis=1)
    plain_mse = ((plain.level + plain.anomaly - truth) ** 2).mean(axis=1)
    difference = gskf_mse - plain_mse
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    resampled = rng.choice(difference, size=(5000, DRAWS)).mean(axis=1)
    half = (100.0 - LEVEL) / 2.0
    low, high = np.percentile(resampled, [half, 100.0 - half])
    report = {
        "site": arguments.site,
        "share_of_half_hours_bad": float(actual.mean()),
        "regime_call_correct": recovery,
        "bad_regime_recall": bad_recall,
        "mse_gskf": float(gskf_mse.mean()),
        "mse_plain": float(plain_mse.mean()),
        "mse_difference": float(difference.mean()),
        "interval": [float(low), float(high)],
        "passed": bool(recovery >= MIN_RECOVERY and high < 0.0),
    }
    (common.WORK_DIR / "skf_power.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
