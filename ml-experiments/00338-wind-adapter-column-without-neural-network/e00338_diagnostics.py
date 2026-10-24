"""Diagnostics for experiment 00338: does an adapter's gain track how well it fits the physics core?

For one farm and seed 0, refits the core without an adapter, with the neural adapter (x4's
column), with x1's table, with x2's boosted scale, and with x3's distilled table. Prints and
writes `work/diagnostics_<farm>.json` with each core's mean absolute error of the expected power on
the ERA5 training rows, and, on the ensemble member rows of the scored year, each column's mean
effective hub-height speed and its correlation with the neural adapter's column. Nothing here
reads a score.

    PYTHONPATH as for the fit; uv run --no-sync python \
      ml-experiments/00338-*/e00338_diagnostics.py --farm W1
"""

import argparse
import json
import os

os.environ.setdefault("E00303_TORCH_THREADS", "4")

import e00303_fit as fit303  # isort: skip  # ty: ignore[unresolved-import]
import e00326_fit as fit326  # isort: skip  # ty: ignore[unresolved-import]
import e00338_adapters as adapters
import e00338_common as common
import numpy as np
import torch


def l1(*, core: object, speed: np.ndarray, target: np.ndarray) -> float:
    """Return the core's mean absolute error of expected power at the given hub-height speeds."""
    with torch.no_grad():
        power = adapters.power_from_speed(core=core, speed=torch.as_tensor(speed))
    return float((power - torch.as_tensor(target)).abs().mean())


def main() -> None:
    """Refit the five cores for seed 0 and write the comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", required=True)
    farm = parser.parse_args().farm
    data = fit303.Hindcast(farm=farm)
    train = fit326.train_columns(data=data)
    target = data.train["target"].to_numpy()
    scored = data.scored_columns
    edges = adapters.make_edges(
        columns=train, ratio_bins=common.RATIO_BINS, temperature_bins=common.TEMPERATURE_BINS
    )
    idle = data.with_idle_layer
    none_core, _ = fit303.fit_era5_core(
        train=data.train, adapter="none", seed=0, with_idle_layer=idle
    )
    net_core, _ = fit303.fit_era5_core(
        train=data.train, adapter="net", seed=0, with_idle_layer=idle
    )
    x1 = adapters.fit_table_core(
        train_columns=train, target=target, edges=edges, with_idle_layer=idle, seed=0
    )
    scale = x1.table_scale().detach().numpy()[adapters.cell_of(columns=train, edges=edges)]
    constraints = adapters.monotone_constraints(scale=scale, columns=train)
    reference, _ = fit303.fit_era5_core(
        train=data.train, adapter="none", seed=0, with_idle_layer=idle
    )
    x2 = adapters.fit_boosted_scale(
        core=reference, train_columns=train, target=target, constraints=constraints, seed=0
    )
    x3, _ = adapters.distilled_table(
        net_core=net_core, train_columns=train, edges=edges, with_idle_layer=idle
    )
    columns = {
        "none": lambda c: adapters.hub_speed(core=none_core, columns=c),
        "neural (x4)": lambda c: adapters.hub_speed(core=net_core, columns=c),
        "x1": lambda c: adapters.hub_speed(core=x1, columns=c),
        "x2": lambda c: x2.effective_speed(columns=c),
        "x3": lambda c: adapters.hub_speed(core=x3, columns=c),
    }
    cores = {"none": none_core, "neural (x4)": net_core, "x1": x1, "x2": x2.core, "x3": x3}
    report = {}
    neural = columns["neural (x4)"](scored)
    for name, make in columns.items():
        on_train = make(train)
        on_scored = make(scored)
        report[name] = {
            "era5_l1": l1(core=cores[name], speed=on_train, target=target),
            "scored_members_mean_speed": float(on_scored.mean()),
            "corr_with_neural_column": float(np.corrcoef(on_scored, neural)[0, 1]),
            "mean_abs_difference_from_neural_m_s": float(np.abs(on_scored - neural).mean()),
        }
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / f"diagnostics_{farm}.json").write_text(json.dumps(report, indent=1))
    print(farm, json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
