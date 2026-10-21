"""Choose the weight decay, the epoch count, and lambda on each direction's training year.

For ENC-c and ENC-r, in each direction, fit every block of the month-block cross-validation for
each c in {0.4, 4, 40} and keep the c with the lowest mean out-of-fold pinball loss. The median of
that c's early-stopped epochs is the final fit's epoch count, and lambda in {0, 0.5, 1} is chosen on
the out-of-fold predictions of that c. ENC-c0, ENC-cps, and ENC-cpt inherit ENC-c's settings, and
ENC-r0 inherits ENC-r's. The script reads no row of any scored year, writes `decisions.json`, and
records the time of the fits (the plan's timing check).

    uv run python ml-experiments/00119-*/e00119_grid.py
"""

import time

import e00119_bundle as bundles
import e00119_cv as cv
import e00119_data as data
import e00119_model as model
import numpy as np

GRID_SEED = 11


def choose(*, direction: cv.Direction, arm_name: str) -> dict:
    """Run the grid for one arm and direction and return the chosen settings."""
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
    actual = rows["actual"].to_numpy()
    k0 = rows.select(
        [f"q{p}" for p in ("01", "05", "10", "25", "40", "50", "60", "75", "90", "95", "99")]
    ).to_numpy()
    table: dict[str, dict] = {}
    for decay in cv.DECAY_GRID:
        started = time.monotonic()
        oof = cv.run_oof(
            rows=rows, bundle=bundle, arm=arm, direction=direction, decay_c=decay, seed=GRID_SEED
        )
        score = float(cv.pinball_numpy(actual=actual[oof.index], quantiles=oof.quantiles).mean())
        k0_score = float(cv.pinball_numpy(actual=actual[oof.index], quantiles=k0[oof.index]).mean())
        table[str(decay)] = {
            "oof_score": score,
            "k0_oof_score": k0_score,
            "best_epochs": oof.best_epochs,
            "train_loss": oof.train_loss,
            "oof_loss": oof.oof_loss,
            "fit_seconds": oof.seconds,
            "grid_seconds": time.monotonic() - started,
        }
        shrink_scores = {
            str(s): float(
                cv.pinball_numpy(
                    actual=actual[oof.index],
                    quantiles=cv.shrink_quantiles(
                        k0=k0[oof.index], quantiles=oof.quantiles, kind=arm.kind, shrink=s
                    ),
                ).mean()
            )
            for s in cv.SHRINK_GRID
        }
        table[str(decay)]["shrink_scores"] = shrink_scores
        print(
            f"{direction.name} {arm_name} c={decay}: OOF {score:.4f} vs K0 {k0_score:.4f}",
            flush=True,
        )
    best_c = min(table, key=lambda c: table[c]["oof_score"])
    epochs = int(np.median(table[best_c]["best_epochs"]))
    shrink = min(table[best_c]["shrink_scores"], key=table[best_c]["shrink_scores"].get)
    return {"table": table, "c": float(best_c), "epochs": epochs, "shrink": float(shrink)}


def main() -> None:
    """Run the grid for both designs in both directions and write `decisions.json`."""
    dataset = data.load_dataset()
    distant = bundles.distant_sites(dataset=dataset)
    cv.write_decisions(
        key="placebo_distant_site",
        value={dataset.sites[i]: dataset.sites[d] for i, d in enumerate(distant)},
    )
    cv.write_decisions(
        key="parameters",
        value={"c": model.parameter_count(kind="c"), "r": model.parameter_count(kind="r")},
    )
    for direction in cv.DIRECTIONS.values():
        for arm_name in ("ENC-c", "ENC-r"):
            result = choose(direction=direction, arm_name=arm_name)
            cv.write_decisions(key=f"{direction.name}.{arm_name}", value=result)
            print(
                f"{direction.name} {arm_name}: c={result['c']} epochs={result['epochs']} "
                f"lambda={result['shrink']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
