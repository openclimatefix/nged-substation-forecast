"""Training loss in nats of the free fit and the diagnostic variants, at the tuning cut.

The capacity profile compares fits with the capacities pinned at the convex model's values; this
script records the loss of the free fit and of each variant on the same rows, with the same
definition (the loss per row times the rows, the mean of the last 20 optimiser steps), so the
write-up can compare them. All fits are the K = 4 configuration of the gate and the diagnostics.

    uv run --no-sync python ml-experiments/00105-*/e00105_losses.py
"""

import e00105_common as common
import e00105_diag as diag
import e00105_gate as gate
import numpy as np
import torch
from studies.dp_onboarding import FitCopy, fit_many
from studies.dp_pooled import DpConfig


def main() -> None:
    """Fit the free model and the variants and write `work/variant_losses.json`."""
    cache = common.load_cache()
    convex = diag.convex_capacities(cache=cache)
    batch = common.pooled_batch_for(cache=cache, sites=common.POOL, cut=common.TUNING_CUT)
    dark = DpConfig(
        variational=True,
        shared_shapes=4,
        interaction_hidden=common.INTERACTION,
        network_temperature_dark_only=True,
    )
    init = diag.capacity_init_from_convex(batch=batch, convex=convex)
    out = {}
    for name, config, capacity_init in (
        ("free", gate.config_p(), None),
        ("nonet", gate.config_p(net=False), None),
        ("a", dark, None),
        ("c", gate.config_p(), init),
        ("ac", dark, init),
    ):
        _, results = fit_many(
            batch=batch,
            copies=[FitCopy(name=name, capacity_init=capacity_init)],
            config=config,
            initial_level=torch.full((len(batch.site_labels),), 50.0),
            steps=common.FIT_STEPS,
            learning_rate=common.FIT_RATE,
            final_rate_fraction=common.FINAL_RATE_FRACTION,
        )
        out[name] = float(np.mean(results[0].losses[-20:])) * batch.n_rows
        print(f"{name}: {out[name]:.0f} nats", flush=True)
        common.write_json(name="variant_losses.json", payload=out)


if __name__ == "__main__":
    main()
