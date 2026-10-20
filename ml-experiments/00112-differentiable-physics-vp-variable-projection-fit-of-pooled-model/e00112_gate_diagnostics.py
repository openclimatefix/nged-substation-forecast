"""Post hoc diagnostics of the failed known-answer gate (experiment 112).

The gate (`e00112_gate.py`) failed on its pre-registered definition, which stays the record. These
runs ask why, on the same eight-site synthetic fleet, and write `work/gate_diagnostics.json`:

- **continue:** the saved default-start fit continued for 1,500 evaluations, which tells whether the
  failure is under-training.
- **noise-free and sd 1.0:** the fleet's clean targets, or with Gaussian noise of sd 1.0 point,
  fitted with the network off, on, and on with temperature hidden by day. The noise-free network-off
  fit is the correctness control: when the problem is identified, the recovery must be exact.
- **pre-registered noise, network off, and network on with temperature hidden by day.**

Every fit is a post hoc variant and is labelled as such where the write-up quotes it.

    uv run --no-sync python ml-experiments/00112-*/e00112_gate_diagnostics.py
"""

from dataclasses import replace
from typing import Any, Final

import e00112_common as common
import e00112_gate as gate
import e00112_vp as vp
import torch
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel, PopulationScales
from studies.dp_pooled_vp import VpSettings, fit_pooled_vp, freeze_operator_for_vp

EVALUATIONS: Final[int] = 600
CONTINUED_EVALUATIONS: Final[int] = 1500
SMALL_NOISE_SD: Final[float] = 1.0
NOISE_FREE_SD: Final[float] = 0.05
"""The noise scale a noise-free fit assumes, so the prior is not infinitely strong."""


def new_variant(*, batch: PooledBatch, network: bool, dark_only: bool) -> PooledDemandModel:
    """Return a fleet model with the network off, on, or on with temperature hidden by day."""
    torch.manual_seed(0)
    scales = PopulationScales(deviation=common.tuned()["map_deviation"])
    config = DpConfig(
        shared_shapes=2,
        interaction_hidden=common.INTERACTION if network else (),
        population_scales=scales,
        network_temperature_dark_only=dark_only,
    )
    model = PooledDemandModel(
        config=config,
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    freeze_operator_for_vp(model=model)
    return model


def summarise(
    *, model: PooledDemandModel, truth: PooledDemandModel, batch: PooledBatch, result: Any, cache
) -> dict[str, Any]:
    """Return a fit's objective, convergence, and worst-case recovery errors."""
    recovery = gate.recovery(model=model, truth=truth, batch=batch, cache=cache)
    return {
        "objective": result.objective,
        "evaluations": result.evaluations,
        "converged": result.converged,
        "gradient_max_abs": result.gradient_max_abs,
        "seconds": result.seconds,
        "worst": {k: (max(v) if isinstance(v, list) else v) for k, v in recovery.items()},
    }


def main() -> None:
    """Run the diagnostics and write their results."""
    cache = common.load_cache()
    batch, truth, noise_sd = gate.build_synthetic(cache=cache)
    truth.train()
    with torch.no_grad():
        clean = truth(batch)
    out: dict[str, Any] = {"preregistered_noise_sd": noise_sd}
    settings = VpSettings(noise_sd=noise_sd, deviation=common.tuned()["map_deviation"])
    model = vp.load_fit(name="gate_fleet_s0", cut=gate.GATE_CUT, batch=batch)
    result = fit_pooled_vp(
        model=model,
        batch=batch,
        settings=settings,
        max_evaluations=CONTINUED_EVALUATIONS,
        adam_steps=0,
        warm_start_steps=0,
    )
    out["continue_default_start"] = summarise(
        model=model, truth=truth, batch=batch, result=result, cache=cache
    )
    generator = torch.Generator(device="cpu").manual_seed(5)
    small = clean + SMALL_NOISE_SD * torch.randn(
        clean.shape, generator=generator, dtype=clean.dtype
    ).to(clean.device)
    cases = (
        ("noise_free", clean, NOISE_FREE_SD, (("off", False, False), ("on", True, False))),
        (
            "sd_1",
            small,
            SMALL_NOISE_SD,
            (("off", False, False), ("on", True, False), ("dark_only", True, True)),
        ),
        (
            "preregistered",
            batch.y,
            noise_sd,
            (("off", False, False), ("dark_only", True, True)),
        ),
    )
    for case, targets, sd, variants in cases:
        rows = replace(batch, y=targets)
        for label, network, dark_only in variants:
            model = new_variant(batch=rows, network=network, dark_only=dark_only)
            result = fit_pooled_vp(
                model=model,
                batch=rows,
                settings=VpSettings(noise_sd=sd, deviation=common.tuned()["map_deviation"]),
                max_evaluations=EVALUATIONS,
                adam_steps=0,
                warm_start_steps=common.WARM_START_STEPS,
            )
            out[f"{case}_{label}"] = summarise(
                model=model, truth=truth, batch=rows, result=result, cache=cache
            )
            print(case, label, out[f"{case}_{label}"]["worst"], flush=True)
    common.write_json(name="gate_diagnostics.json", payload=out)


if __name__ == "__main__":
    main()
