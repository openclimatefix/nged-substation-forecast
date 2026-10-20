"""Compare the fits on one objective: VP-P, P-map-same, P-map-polish, and experiment 105's P-map.

Every number is the variable-projection objective in nats (`vp_objective`), on the same training
rows and noise standard deviation of its cut. Experiment 105's P-map, fitted by gradient descent on
its own objective with a learned noise scale and priors before the softplus, is evaluated on this
objective with its parameters mapped through the softplus and the MVA operator set to zero reactive
power. Differences are in nats; the write-up quotes them as V1's content, independent of scores.

    uv run --no-sync python ml-experiments/00112-*/e00112_objectives.py
"""

import e00112_common as common
import e00112_gradient as gradient
import e00112_vp as vp
import torch
from studies.dp_pooled import PooledDemandModel
from studies.dp_pooled_vp import freeze_operator_for_vp, vp_objective

ARMS = ("vp_p_s0", "vp_p_s1", "p_map_same", "p_map_polish")


def p_map_105_objective(*, cache, batch, settings) -> float:  # noqa: ANN001
    """Return experiment 105's stored P-map (scored cut) on the variable-projection objective."""
    path = (
        common.DATA_DIR.parent
        / "00105-differentiable-physics-3-pooled-and-per-site-demand-model"
        / "work"
        / "model_p_map.pt"
    )
    config = gradient.config_for(kind="p_map", settings=common.tuned())
    model = PooledDemandModel(
        config=config,
        n_sites=len(batch.site_labels),
        n_season_years=batch.n_season_years,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    state = torch.load(path, map_location=batch.y.device)
    converted = {k: v.to(batch.y.dtype) if v.is_floating_point() else v for k, v in state.items()}
    # The stored state predates the `weeks_of_history` buffer, which onboarding alone uses.
    incompatible = model.load_state_dict(converted, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - {"weeks_of_history"}:
        msg = f"experiment 105's P-map does not fit the model: {incompatible}"
        raise ValueError(msg)
    freeze_operator_for_vp(model=model)
    with torch.no_grad():
        return float(vp_objective(model=model, batch=batch, settings=settings))


def weight_sizes(*, model: PooledDemandModel) -> dict[str, float]:
    """Return the sizes of the linear coefficients that an exact solve can drive large."""
    out = {
        "largest_holiday": float(model.holiday.abs().max()),
        "largest_harmonic": float(model.harmonic_coef.abs().max()),
        "largest_tent": float(model.tent_coef.abs().max()),
        "largest_shape": float(model.shapes.abs().max()),
    }
    if model.interaction is not None:
        out["network_last_layer_norm"] = float(list(model.interaction.parameters())[-2].norm())
    return out


def main() -> None:
    """Write `work/objectives.json`."""
    cache = common.load_cache()
    out: dict[str, dict[str, float]] = {}
    for cut in common.CUTS:
        key = cut.date().isoformat()
        out[key] = {}
        for name in ARMS:
            if vp.has_fit(name=name, cut=cut):
                out[key][name] = vp.read_record(name=name, cut=cut)["objective"]
    batch = common.pooled_batch_for(
        cache=cache,
        sites=common.POOL,
        cut=common.FINAL_CUT,
        device=vp.DEVICE,
        dtype=common.BATCH_DTYPE,
    )
    settings = common.vp_settings(cut=common.FINAL_CUT, round_name="round2")
    key = common.FINAL_CUT.date().isoformat()
    out[key]["p_map_105"] = p_map_105_objective(cache=cache, batch=batch, settings=settings)
    weights: dict[str, dict[str, float]] = {}
    for name, network in (
        ("vp_p_s0", True),
        ("vp_p_s1", True),
        ("p_map_same", True),
        ("vp_p_nonet_s0", False),
    ):
        loaded = vp.load_fit(name=name, cut=common.FINAL_CUT, batch=batch, network=network)
        weights[name] = weight_sizes(model=loaded)
        print("weights", name, weights[name], flush=True)
    for key, values in out.items():
        print(key, {name: round(value) for name, value in values.items()}, flush=True)
    out["weights_2025-07-01"] = weights  # ty: ignore[invalid-assignment]
    common.write_json(name="objectives.json", payload=out)


if __name__ == "__main__":
    main()
