"""Known-answer gate of experiment 279: can each arm recover a heating curve it was given?

**The truth is synthetic.** For one substation and the scored cut, the script takes T-all's fit,
zeroes its heating term, and adds a known heating curve of the real temperatures, plus Gaussian
noise as large as T-all's own residual, to make a synthetic target for the 24-month training
window. Each arm is fitted to that target, forecasts the year after the cut, and is scored by the
root mean squared error of its forecast against the noiseless synthetic truth.

- **Curved truth.** Heating grows with the cold excess of a lagged temperature (half the weight on
  the last day's mean, 30% on the day before, 20% on the day before that), is zero above 12 C,
  rises at 3 points per 10 C between 3 and 12 C, and rises 2.5 times as fast below 3 C.
- **Straight truth.** Heating is 3 points per 10 C below 15.5 C on the last day's mean temperature.

**Pass rules, written before any gate result was read.** On the curved truth the network (A3b)
passes when its root mean squared error on cold rows (last day's mean at most 3 C) is at least 25%
below T-all's. On the straight truth A3b passes when its root mean squared error over all rows is
at most 25% above T-all's plus 0.05 points. An arm that fails its gate has its label capped at
inconclusive.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00279-*/e00279_gate.py
"""

import json
from typing import Final

import e00279_common as common
import e00279_design as design
import e00279_fit as fit
import e00279_pipeline as pipeline
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import pinned_solar_points

GATE_SITES: Final[tuple[str, ...]] = ("D10", "D14")
GATE_ARMS: Final[tuple[str, ...]] = ("t_all", "a1", "a2", "a3b")
LAG_WEIGHTS: Final[tuple[float, float, float]] = (0.5, 0.3, 0.2)
CURVE_SLOPE: Final[float] = 3.0
KNEE_COLD_C: Final[float] = 3.0
KNEE_WARM_C: Final[float] = 12.0
COLD_STEEPNESS: Final[float] = 2.5
STRAIGHT_THRESHOLD_C: Final[float] = 15.5
CURVED_GAIN: Final[float] = 0.25
STRAIGHT_TOLERANCE: Final[float] = 0.25
STRAIGHT_FLOOR: Final[float] = 0.05
SEED: Final[int] = 0


def curved_truth(*, d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Return the curved heating (points) from the last three daily mean temperatures."""
    lagged = LAG_WEIGHTS[0] * d0 + LAG_WEIGHTS[1] * d1 + LAG_WEIGHTS[2] * d2
    mild = np.clip(KNEE_WARM_C - lagged, 0.0, KNEE_WARM_C - KNEE_COLD_C)
    cold = np.clip(KNEE_COLD_C - lagged, 0.0, None)
    return CURVE_SLOPE * (mild + COLD_STEEPNESS * cold) / 10.0


def straight_truth(*, d0: np.ndarray) -> np.ndarray:
    """Return the straight heating (points) from the last day's mean temperature."""
    return CURVE_SLOPE * np.clip(STRAIGHT_THRESHOLD_C - d0, 0.0, None) / 10.0


def temperatures(*, cache: FeatureCache, site: str, index: np.ndarray) -> np.ndarray:
    """Return the last three daily mean temperatures at the rows, shape (n, 3)."""
    aux = design.site_aux(cache=cache, site=site)
    features = cache.sites[site]
    return np.column_stack([features.smoothed_temperature_c[index, 1], aux.previous_days[index]])


def run_site(*, site: str, truth: str) -> dict[str, dict[str, float]]:
    """Fit every gate arm on one synthetic truth at one site and return the errors."""
    cache = fit._STATE["cache"]
    vintages = fit._STATE["vintages"]
    weight = float(fit.verified_decisions()["lambda"]["t_all"])
    cut = common.SCORED_CUT
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
    context = design.ArmContext(aux=design.site_aux(cache=cache, site=site))
    reference = pipeline.arm_spec(arm_name="t_all", penalty_weight=weight)
    real = design.fit_arm(cache=cache, base=base, arm=reference, context=context)
    heating = design.heating_piece_columns(layout=base.layout, arm=reference)
    theta = real.theta.copy()
    theta[heating] = 0.0

    def other_terms(index: np.ndarray) -> np.ndarray:
        x0 = design.design_matrix(
            cache=cache, features=cache.sites[site], layout=base.layout, index=index
        )
        x = design.arm_matrix(
            cache=cache,
            layout=base.layout,
            site=site,
            index=index,
            x0=x0,
            arm=reference,
            context=context,
        )
        return x @ theta

    def heating_of(index: np.ndarray) -> np.ndarray:
        t = temperatures(cache=cache, site=site, index=index)
        if truth == "curved":
            return curved_truth(d0=t[:, 0], d1=t[:, 1], d2=t[:, 2])
        return straight_truth(d0=t[:, 0])

    rng = np.random.default_rng(SEED)
    noise_sd = float(np.sqrt(real.unpenalised_loss / real.n_rows))
    train_truth = other_terms(base.index) + heating_of(base.index)
    synthetic = train_truth + rng.normal(0.0, noise_sd, size=len(base.index))
    score_index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    score_truth = other_terms(score_index) + heating_of(score_index)
    cold = temperatures(cache=cache, site=site, index=score_index)[:, 0] <= KNEE_COLD_C
    out: dict[str, dict[str, float]] = {}
    for arm_name in GATE_ARMS:
        arm = pipeline.arm_spec(arm_name=arm_name, penalty_weight=weight)
        arm_fit = design.fit_arm(cache=cache, base=base, arm=arm, context=context, target=synthetic)
        # The arms subtract the pinned solar output from their forecast; the truth does not, so
        # add it back before comparing.
        prediction = design.predict_arm(cache=cache, base=base, fit=arm_fit, index=score_index)
        assert base.pinned is not None
        pinned = pinned_solar_points(
            cache=cache,
            features=cache.sites[site],
            pinned=base.pinned,
            index=score_index,
        )
        error = (prediction + pinned) - score_truth
        out[arm_name] = {
            "rmse_all": float(np.sqrt((error**2).mean())),
            "rmse_cold": float(np.sqrt((error[cold] ** 2).mean())),
            "cold_rows": int(cold.sum()),
        }
        print(f"gate {site} {truth} {arm_name}: {out[arm_name]}", flush=True)
    return out


def main() -> None:
    """Run the gate at two substations on both truths and print the verdicts."""
    fit._init_worker()
    record: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    verdicts: dict[str, bool] = {}
    for site in GATE_SITES:
        record[site] = {}
        for truth in ("curved", "straight"):
            record[site][truth] = run_site(site=site, truth=truth)
    for site in GATE_SITES:
        curved = record[site]["curved"]
        straight = record[site]["straight"]
        verdicts[f"{site}_curved"] = (
            curved["a3b"]["rmse_cold"] <= (1.0 - CURVED_GAIN) * curved["t_all"]["rmse_cold"]
        )
        verdicts[f"{site}_straight"] = (
            straight["a3b"]["rmse_all"]
            <= (1.0 + STRAIGHT_TOLERANCE) * straight["t_all"]["rmse_all"] + STRAIGHT_FLOOR
        )
    print("verdicts:", verdicts)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "gate.json").write_text(
        json.dumps({"errors": record, "verdicts": verdicts}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
