"""Score the stage 1 arms of experiment 00049: absolute scores, contrasts, and reproduction checks.

Reads the refit predictions `e00049_fit.py` wrote, assembles each arm over the 12 validation
months, applies the spread factor 1.15 about the median, restricts every arm and the stored
references to the cases they all hold, and writes the tables the write-up quotes under
`data/ml-experiments/00049-.../work/stage1/`.

    uv run python ml-experiments/00049-vp1-screening-reference-under-p-roll-demand/e00049_stage1.py
"""

from typing import Any, Final

import e00049_fit as fitting
import e00049_stage0 as stage0
import numpy as np
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final = stage0.WORK_DIR.parent / "stage1"
ARM_FOLDERS: Final[tuple[str, ...]] = ("c0", "no_christmas", "no_covid", "hl2y", "c0b", "f1")
STORED: Final[dict[str, str]] = {"c0": "roll12_s", "c0b": "type_id_roll12_s"}
"""The stored arm of experiment 00034 that each reference refit should reproduce."""
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "no_christmas", "c0"),
    ("C2", "no_covid", "c0"),
    ("C3", "hl2y", "c0"),
    ("F1", "f1", "c0b"),
)
REPRODUCTION_TOLERANCE: Final[float] = 0.03
EQUIVALENCE_MARGIN: Final[float] = 0.03
FINGERPRINT_THRESHOLD: Final[float] = 0.10


def read_refit_arm(*, folder: str) -> pl.DataFrame:
    """Return an arm's quantiles with the spread factor 1.15, keyed by case."""
    files = sorted((fitting.ANCHOR_PREDS_DIR / folder).glob("*.parquet"))
    if not files:
        msg = f"no refit predictions for {folder}"
        raise FileNotFoundError(msg)
    frame = pl.concat([pl.read_parquet(f) for f in files]).sort(*stage0.KEYS)
    if frame.select(*stage0.KEYS).n_unique() != frame.height:
        msg = f"{folder} holds duplicate cases"
        raise ValueError(msg)
    quantiles = stage0.spread(
        quantiles=frame.select(stage0.LEVEL_COLUMNS).to_numpy().astype(np.float64),
        factor=stage0.FIXED_FACTOR,
    )
    return frame.select(*stage0.KEYS).hstack(
        pl.DataFrame(quantiles, schema=list(stage0.LEVEL_COLUMNS))
    )


def zone(*, row: dict[str, Any], rule: str) -> str:
    """Return the state of a column-drop contrast: stays, removable, or unresolved."""
    lower, upper = float(row["lower"]), float(row["upper"])
    if lower > 0:
        return "stays"
    if lower >= -EQUIVALENCE_MARGIN and upper <= EQUIVALENCE_MARGIN:
        return "removable"
    return "unresolved" if rule == "drop" else "n/a"


def main() -> None:
    """Score every arm that has refit predictions and print the tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases, _, scored = stage0.load_all()
    stored = {
        name: pl.DataFrame(scored[name], schema=list(stage0.LEVEL_COLUMNS))
        for name in STORED.values()
    }
    refit = {}
    for folder in ARM_FOLDERS:
        try:
            refit[folder] = read_refit_arm(folder=folder)
        except FileNotFoundError:
            print(f"{folder}: no predictions yet")
    keys = cases.select(*stage0.KEYS)
    for table in refit.values():
        keys = keys.join(table.select(*stage0.KEYS), on=list(stage0.KEYS), how="inner")
    held = cases.join(keys, on=list(stage0.KEYS), how="semi")
    mask = cases.join(keys.with_columns(held=pl.lit(True)), on=list(stage0.KEYS), how="left")[
        "held"
    ].is_not_null()
    print(f"cases held by every refit arm: {held.height:,} of {cases.height:,}")
    arms: dict[str, pl.DataFrame] = {}
    for name, table in refit.items():
        joined = held.select(*stage0.KEYS).join(
            table, on=list(stage0.KEYS), how="left", validate="1:1"
        )
        arms[name] = stage0.per_case_loss(
            cases=held, quantiles=joined.select(stage0.LEVEL_COLUMNS).to_numpy().astype(np.float64)
        )
    for name, quantiles in stored.items():
        arms[name] = stage0.per_case_loss(
            cases=held, quantiles=quantiles.filter(pl.Series(mask)).to_numpy().astype(np.float64)
        )
    rows = []
    for label, treatment, reference in CONTRASTS:
        if treatment in arms and reference in arms:
            result = stage0.contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                level=stage0.DECISIVE_LEVEL,
                n=stage0.N_RESAMPLES_DECISIVE,
            )
            rule = "port" if label == "F1" else "drop"
            state = (
                ("port" if result["upper"] < -FINGERPRINT_THRESHOLD else "not worth porting")
                if label == "F1"
                else zone(row=result, rule=rule)
            )
            rows.append({"id": label, **result, "state": state})
    for reference, stored_name in STORED.items():
        if reference in arms:
            rows.append(
                {
                    "id": f"reproduction {reference}",
                    **stage0.contrast(
                        arms=arms,
                        treatment=reference,
                        reference=stored_name,
                        level=95.0,
                        n=stage0.N_RESAMPLES,
                    ),
                    "state": "",
                }
            )
    absolute = []
    for name, frame in arms.items():
        interval = bootstrap.bootstrap_absolute(
            losses=experiment_store.losses_for_bootstrap(
                arm_losses={name: frame.select(*stage0.KEYS, "qs")}
            ),
            arm=name,
            metric="qs",
        )
        absolute.append(
            {
                "arm": name,
                "qs": interval["value"],
                "lower_95": interval["lower_95"],
                "upper_95": interval["upper_95"],
                "mae": float(frame["mae"].to_numpy().mean()),
            }
        )
    pl.DataFrame(rows).write_parquet(RESULTS_DIR / "contrasts.parquet")
    pl.DataFrame(absolute).write_parquet(RESULTS_DIR / "absolute.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=250):
        print(pl.DataFrame(absolute).sort("qs"))
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
