"""Fit every arm on each direction's whole training year and forecast the scored year.

The settings come from `decisions.json` (chosen on each direction's own training year by
`e00119_grid.py`): the weight-decay constant c, the epoch count (the median of the blocks'
early-stopped epochs), and lambda. Each arm is fitted with seeds 11 to 15. The script writes the
seeds' quantiles for the scored year's rows, and the fitted ENC-c weights of direction F1 for the
substation cards. It refuses to run until the gate has passed or been ruled on.

    uv run python ml-experiments/00119-*/e00119_final.py [--directions F1 F2] [--arms ...]
"""

import argparse
import json
import time
from pathlib import Path

import e00119_bundle as bundles
import e00119_common as common
import e00119_cv as cv
import e00119_data as data
import e00119_model as model
import numpy as np
import torch

ARM_ORDER = ("ENC-c", "ENC-c0", "ENC-cps", "ENC-cpt", "ENC-r", "ENC-r0")


def final_path(*, direction: str, arm: str) -> Path:
    """Return the file that holds an arm's seeds' scored-year quantiles."""
    return common.WORK_DIR / "final" / f"{direction}_{arm}.npz"


def settings_for(*, decisions: dict, direction: str, arm: model.Arm) -> dict:
    """Return the c, epochs, and lambda an arm inherits from its design's grid."""
    chosen = decisions[f"{direction}.{'ENC-r' if arm.kind == 'r' else 'ENC-c'}"]
    return {"c": chosen["c"], "epochs": chosen["epochs"], "shrink": chosen["shrink"]}


def fit_arm(*, direction: cv.Direction, arm_name: str) -> str:
    """Fit one arm with five seeds and store the scored-year forecasts."""
    started = time.monotonic()
    dataset = data.load_dataset()
    rows = bundles.rows_with_sensitivities(dataset=dataset)
    tensors = bundles.make_tensors(dataset=dataset)
    arm = model.ARMS[arm_name]
    bundle = bundles.make_bundle(
        rows=rows,
        tensors=tensors,
        arm=arm,
        distant=bundles.distant_sites(dataset=dataset),
        foreign=bundles.foreign_runs(dataset=dataset),
    )
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    settings = settings_for(decisions=decisions, direction=direction.name, arm=arm)
    train_index = np.flatnonzero(cv.run_mask(rows=rows, period=direction.train))
    scored_index = np.flatnonzero(cv.run_mask(rows=rows, period=direction.scored))
    if np.intersect1d(train_index, scored_index).size:
        msg = "training and scored rows overlap"
        raise RuntimeError(msg)
    days = len(np.unique(rows["run"].to_numpy()[train_index]))
    train = bundle.subset(cv.to_device(train_index))
    scored = bundle.subset(cv.to_device(scored_index))
    forecasts = []
    for seed in cv.SEEDS:
        fitted = model.fit(
            arm=arm,
            train=train,
            validation=None,
            seed=seed,
            decay_c=settings["c"],
            n_days=days,
            epochs=settings["epochs"],
        )
        quantiles = model.quantiles_of(result=fitted, bundle=scored, shrink=settings["shrink"])
        forecasts.append(model.to_numpy(quantiles).astype(np.float32))
        if direction.name == "F1" and arm_name == "ENC-c":
            target = common.WORK_DIR / "final" / f"weights_F1_ENC-c_{seed}.pt"
            torch.save(
                {"state": fitted.model.state_dict(), "standardiser": fitted.standardiser}, target
            )
    target = final_path(direction=direction.name, arm=arm_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target.with_name(f"{target.stem}.tmp.npz"),
        index=scored_index,
        quantiles=np.stack(forecasts),
        seeds=np.array(cv.SEEDS),
    )
    target.with_name(f"{target.stem}.tmp.npz").rename(target)
    seconds = time.monotonic() - started
    return f"{direction.name} {arm_name}: {len(scored_index):,} rows, {seconds:.0f} s"


def main() -> None:
    """Fit the requested arms in the requested directions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", nargs="+", default=["F1", "F2"])
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER))
    args = parser.parse_args()
    for name in args.directions:
        for arm_name in args.arms:
            print(fit_arm(direction=cv.DIRECTIONS[name], arm_name=arm_name), flush=True)


if __name__ == "__main__":
    main()
