"""Gate tests 3 and 4 of experiment 119: reproduction of K0, causality, and the fold purge.

uv run python ml-experiments/00119-*/e00119_checks.py
"""

import json
from collections.abc import Callable

import e00119_bundle as bundles
import e00119_common as common
import e00119_cv as cv
import e00119_data as data
import e00119_model as model
import numpy as np
import polars as pl
import torch

REPRODUCTION_TOLERANCE = 1e-6
CAUSAL_RUN = 600
"""The run index (2025-12) whose forecasts must not depend on later runs."""


def fresh_result(*, arm: model.Arm, bundle: model.Bundle, random_head: bool) -> model.FitResult:
    """Return an unfitted encoder in float64, optionally with a random output layer."""
    torch.manual_seed(0)
    encoder = model.Encoder(kind=arm.kind, n_sites=len(common.SCORED_SITES)).to(model.DEVICE)
    if random_head:
        torch.nn.init.normal_(encoder.head.weight, std=0.5)
        torch.nn.init.normal_(encoder.head.bias, std=0.5)
    return model.FitResult(
        model=encoder, standardiser=model.fit_standardiser(bundle=bundle), arm=arm, best_epoch=0
    )


def reproduction(*, bundle_for: Callable[[model.Arm], model.Bundle]) -> dict[str, float]:
    """Return the largest difference between each untrained arm and K0's quantiles."""
    out = {}
    for name in ("ENC-c", "ENC-r"):
        arm = model.ARMS[name]
        bundle = bundle_for(arm)
        result = fresh_result(arm=arm, bundle=bundle, random_head=False)
        quantiles = model.quantiles_of(result=result, bundle=bundle).double()
        out[name] = float((quantiles - bundle.columns["q"].double()).abs().max())
    return out


def causality(*, bundle_for: Callable[[model.Arm], model.Bundle]) -> dict[str, float]:
    """Return the largest change in forecasts at run `CAUSAL_RUN` when later weather changes.

    Every patch value of the runs after `CAUSAL_RUN` is replaced by noise, as is every `actual`;
    the forecasts of rows from runs up to `CAUSAL_RUN` must stay exactly the same.
    """
    out = {}
    for name in ("ENC-c", "ENC-c0", "ENC-cps", "ENC-r"):
        arm = model.ARMS[name]
        bundle = bundle_for(arm)
        result = fresh_result(arm=arm, bundle=bundle, random_head=True)
        keep = (bundle.columns["run"] <= CAUSAL_RUN).nonzero().squeeze(-1)
        before = model.quantiles_of(result=result, bundle=bundle.subset(keep))
        patch = bundle.patch.clone()
        patch[:, CAUSAL_RUN + 1 :] = torch.randn_like(patch[:, CAUSAL_RUN + 1 :]) * 50.0
        noisy_columns = dict(bundle.columns)
        noisy_columns["actual"] = torch.randn_like(bundle.columns["actual"]) * 50.0
        noisy = model.Bundle(noisy_columns, patch, bundle.cell_ok, bundle.geometry)
        after = model.quantiles_of(result=result, bundle=noisy.subset(keep))
        out[name] = float((before - after).abs().max())
    return out


def purge(*, rows: pl.DataFrame) -> dict[str, int]:
    """Return, per direction, how many training rows of any block lie within the purge window."""
    out = {}
    for direction in cv.DIRECTIONS.values():
        violations = 0
        for block in cv.blocks_of(direction=direction):
            train, tune = cv.fold_indices(rows=rows, direction=direction, block=block)
            low, high = block[0] - cv.PURGE, block[1] + cv.PURGE
            run = rows["run_date"].to_numpy()[train]
            valid = rows["valid_time"].to_numpy()[train]
            bounds = [np.datetime64(b.replace(tzinfo=None), "us") for b in (low, high)]
            violations += int(((run >= bounds[0]) & (run < bounds[1])).sum())
            violations += int(((valid >= bounds[0]) & (valid < bounds[1])).sum())
            overlap = np.intersect1d(train, tune)
            violations += len(overlap)
            scored_start = np.datetime64(direction.scored[0].replace(tzinfo=None), "us")
            if direction.name == "F1":
                violations += int((rows["run_date"].to_numpy()[train] >= scored_start).sum())
        out[direction.name] = violations
    return out


def main() -> None:
    """Run the checks and write `checks.json`."""
    dataset = data.load_dataset()
    rows = bundles.rows_with_sensitivities(dataset=dataset)
    tensors = bundles.make_tensors(dataset=dataset)
    distant = bundles.distant_sites(dataset=dataset)
    foreign = bundles.foreign_runs(dataset=dataset)

    def bundle_for(arm: model.Arm) -> model.Bundle:
        return bundles.make_bundle(
            rows=rows, tensors=tensors, arm=arm, distant=distant, foreign=foreign
        )

    result = {
        "reproduction_max_abs_difference": reproduction(bundle_for=bundle_for),
        "causality_max_abs_change": causality(bundle_for=bundle_for),
        "purge_violations": purge(rows=rows),
        "k0_unsorted_rows": int(
            (np.diff(rows.select(common.QUANTILE_COLUMNS).to_numpy(), axis=1) < 0).any(axis=1).sum()
        ),
    }
    common.write_json_atomic(path=common.WORK_DIR / "checks.json", content=result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
