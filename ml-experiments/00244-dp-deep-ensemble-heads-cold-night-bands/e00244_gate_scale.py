"""Experiment 244, post hoc: how large must the injected level shift be before the spread rises?

The registered gate shifts arm D's forecast level by 4 C times the cold-night slope, which is a few
tenths of a unit on an input of about 0.5 (the level divided by 100). This script repeats the gate
on every cold-night row of the scored year with the shift multiplied by 1, 5, 25, and 100, for the
network without error inputs, and writes `work/gate_scale.json`.

    uv run --no-sync python e00244_gate_scale.py
"""

import json

import e00244_fit as fit_module
import e00244_model as model_module
import e00244_predict as predict_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

MULTIPLIERS = (0, 1, 5, 25, 100)


def main() -> None:
    """Spread of the 20 median corrections at each shift multiplier."""
    state = fit_module.load_state()
    rows, slopes = predict_module.build_rows()
    cold = rows.filter(
        (pl.col("head_set") == "full")
        & pl.col("cold2")
        & pl.col("month").is_in(list(predict_module.WINTER))
        & pl.col("hour_band").is_in(list(predict_module.NIGHT_BANDS))
    )
    site_idx = cold["site_index"].to_numpy()
    idx = cold["idx"].to_numpy()
    x = np.stack([state["x"][k][i] for k, i in zip(site_idx, idx, strict=True)])
    x[:, list(fit_module.ERROR_COLUMNS)] = 0.0
    level = np.array([predict_module.INJECT_C * slopes[state["labels"][k]] for k in site_idx])
    nets = []
    for seed in range(fit_module.N_SEEDS):
        net = model_module.Head(n_sites=len(state["feats"]))
        net.load_state_dict(
            torch.load(fit_module.NETS_DIR / f"full_ref_seed_{seed}.pt", weights_only=True)
        )
        nets.append(net.eval())
    out = {
        "rows": len(idx),
        "input_level_mean": float(x[:, 28].mean()),
        "input_level_sd": float(x[:, 28].std()),
        "typical_shift": float(np.abs(level / 100).mean()),
    }
    for m in MULTIPLIERS:
        xs = x.copy()
        xs[:, 28] += (m * level / 100.0).astype(np.float32)
        c = np.stack(
            [model_module.predict(net=n, x=xs, site=site_idx, device="cpu")[0] for n in nets]
        )
        out[f"x{m}"] = {
            "mean_spread": float(c.std(axis=0).mean()),
            "mean_correction": float(c.mean()),
        }
    (fit_module.WORK_DIR / "gate_scale.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
