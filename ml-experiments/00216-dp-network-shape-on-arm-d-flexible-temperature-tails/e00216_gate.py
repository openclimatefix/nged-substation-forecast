# ruff: noqa: E501
"""The known-answer gate: can the tail columns express a temperature-tail error added to the target?

At each substation and at the monthly cuts of the tuning months, an amount of 3% of the training
window's mean absolute demand (net power plus the pinned solar output) is added to the training
target on rows whose 24-hour-averaged temperature is above the training window's 95th percentile
(the step), or rising from zero at the 95th to the full amount at the 99th (the ramp), or on rows
below the 5th percentile (cold), or on the middle half of the range (the placebo), or nowhere (the
null). Each arm is fitted on the altered and on the unaltered target. Recovery is the mean over the
rows of the fitted value on the altered target minus its fitted value on the unaltered target, as
a share of the added amount: over the injected training rows, and over the held-out rows of the
next three months that satisfy the same temperature condition.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00216-*/e00216_gate.py [--workers 4]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import Final

import e00216_common as common
import e00216_design as design
import e00216_run as run
import e00216_tall as tall
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_calendar import SMOOTHING_HOURS
from studies.dp_pinned_capacity import read_vintages

ARMS: Final[tuple[str, ...]] = ("r0", "f1_1", "f2_1")
VARIANTS: Final[tuple[str, ...]] = ("step", "ramp", "cold", "placebo", "null")
WINDOW: Final[int] = SMOOTHING_HOURS.index(24)
AMOUNT_SHARE: Final[float] = 0.03
HELD_OUT_DAYS: Final[int] = 91


def _temperature(*, cache, site: str, index: np.ndarray) -> np.ndarray:  # noqa: ANN001
    return cache.sites[site].smoothed_temperature_c[index, WINDOW]


def injection(*, variant: str, temperature: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return the weight (0 to 1) each row carries, from the training window's percentiles."""
    p = {q: np.percentile(reference, q) for q in (5, 25, 75, 95, 99)}
    if variant == "step":
        return (temperature > p[95]).astype(float)
    if variant == "ramp":
        return np.clip((temperature - p[95]) / (p[99] - p[95]), 0.0, 1.0)
    if variant == "cold":
        return (temperature < p[5]).astype(float)
    if variant == "placebo":
        return ((temperature > p[25]) & (temperature < p[75])).astype(float)
    return np.zeros_like(temperature)


def _job(job: tuple[str, datetime]) -> list[dict]:
    torch.set_num_threads(1)
    site, cut = job
    cache = common.load_cache()
    vintages = read_vintages()
    features = cache.sites[site]
    rows = []
    cut64 = np.datetime64(cut.replace(tzinfo=None), "us")
    held = np.arange(
        int(np.searchsorted(cache.end_times, cut64, side="right")),
        int(np.searchsorted(cache.end_times, cut64 + np.timedelta64(HELD_OUT_DAYS, "D"))),
    )
    held = held[
        np.isfinite(features.smoothed_temperature_c[held]).all(axis=1)
        & np.isfinite(features.pv_unit_shape[held])
        & np.isfinite(features.temperature_c[held])
    ]
    for arm in ARMS:
        spec = run.arm_spec(arm=arm)
        clean = tall.fit_tall(
            cache=cache, vintages=vintages, site=site, cut=cut,
            penalty_weight=common.T_ALL_LAMBDA, arm=spec,
        )  # fmt: skip
        for variant in VARIANTS:
            state: dict[str, np.ndarray | float] = {}

            def shift(
                base: design.BaseFit, variant: str = variant, state: dict = state
            ) -> np.ndarray:
                temperature = _temperature(cache=cache, site=site, index=base.index)
                weight = injection(variant=variant, temperature=temperature, reference=temperature)
                amount = AMOUNT_SHARE * float(np.abs(base.target).mean())
                state["amount"], state["weight"] = amount, weight
                state["reference"] = temperature
                return amount * weight

            altered = tall.fit_tall(
                cache=cache, vintages=vintages, site=site, cut=cut,
                penalty_weight=common.T_ALL_LAMBDA, arm=spec, target_shift=shift,
            )  # fmt: skip
            amount = float(state["amount"])
            weight = np.asarray(state["weight"])
            reference = np.asarray(state["reference"])
            base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut)
            train = base.index
            difference = tall.predict_tall(
                cache=cache, fit=altered, index=train
            ) - tall.predict_tall(cache=cache, fit=clean, index=train)
            held_weight = injection(
                variant=variant,
                temperature=_temperature(cache=cache, site=site, index=held),
                reference=reference,
            )
            held_difference = tall.predict_tall(cache=cache, fit=altered, index=held) - (
                tall.predict_tall(cache=cache, fit=clean, index=held)
            )
            tail = injection(variant="step", temperature=reference, reference=reference) > 0
            held_tail = (
                injection(
                    variant="step",
                    temperature=_temperature(cache=cache, site=site, index=held),
                    reference=reference,
                )
                > 0
            )
            rows.append(
                {
                    "site": site, "cut": cut.isoformat(), "arm": arm, "variant": variant,
                    "amount": amount,
                    "train_recovery": _share(difference, weight, amount, null=variant == "null"),
                    "held_recovery": _share(
                        held_difference, held_weight, amount, null=variant == "null"
                    ),
                    "train_tail_change": float(difference[tail].mean() / amount) if tail.any() else None,
                    "held_tail_change": float(held_difference[held_tail].mean() / amount)
                    if held_tail.any()
                    else None,
                }
            )  # fmt: skip
    return rows


def _share(
    difference: np.ndarray, weight: np.ndarray, amount: float, *, null: bool
) -> float | None:
    """Return the weighted mean of the fitted change as a share of the injected amount."""
    total = float(weight.sum())
    if null:
        return float(np.abs(difference).mean() / amount)
    if total == 0.0:
        return None
    return float((difference * weight).sum() / (amount * total))


def main() -> None:
    """Run the gate at every substation and tuning-month cut and write `work/gate.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cuts = [datetime(y, m, 1, tzinfo=UTC) for y, m in sorted(common.TUNING_MONTHS)]
    jobs = [(site, cut) for site in common.SCORED_SITES for cut in cuts]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(_job, jobs):
            results.extend(rows)
    common.write_json_atomic(path=common.WORK_DIR / "gate.json", content={"rows": results})
    print(f"{len(results)} rows written")


if __name__ == "__main__":
    main()
