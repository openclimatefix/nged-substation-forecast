"""The known-answer gate of experiment 201: can the network recover an injected interaction?

An interaction the convex model does not have is added to the actual power of every row: a
cold-evening effect of up to 3 points that grows as the true temperature falls below 2 C (full size
at -3 C or colder), on local half-hours 32 to 43 (16:00 to 22:00), and a Saturday-evening shift of
1 point on local half-hours 34 to 45 (17:00 to 23:00). The network (arm 3's inputs) is trained on
the injected power. The gate passes when, on the scored year, (a) the network's median moves by at
least 50% of the injected effect (the slope, through the origin, of the median shift between the
network trained on injected power and the same network trained on the real power), and (b) on rows
with no injection the network trained on injected power scores no more than 0.02 points worse than
the network trained on the real power.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00201-*/e00201_gate.py
"""

import json

import e00201_common as common
import e00201_data as data
import e00201_nn as nn_
import numpy as np
import polars as pl
import torch

ARM = 3
COLD_FULL_C = -3.0
COLD_POINTS = 3.0
SATURDAY_POINTS = 1.0
PASS_RECOVERY = 0.5
PASS_NO_HARM = 0.02
SEED = 0


def injected_effect(*, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the cold-evening effect and the Saturday-evening effect, in points, per row."""
    hh = frame["local_hh"].to_numpy()
    temp = frame["true_temp_c"].to_numpy().astype(np.float64)
    depth = np.clip((data.COLD_C - temp) / (data.COLD_C - COLD_FULL_C), 0.0, 1.0)
    cold = COLD_POINTS * depth * ((hh >= 32) & (hh <= 43))
    saturday = SATURDAY_POINTS * ((frame["dow"].to_numpy() == 5) & (hh >= 34) & (hh <= 45))
    return cold, saturday


def main() -> None:
    """Run the gate and write `work/gate.json`."""
    torch.set_num_threads(1)
    loaded = data.load_data()
    frame, split = loaded.frame, loaded.split
    cold, saturday = injected_effect(frame=frame)
    effect = cold + saturday
    actual = frame["actual"].to_numpy().astype(np.float64) + effect
    q = nn_.quantile_matrix(frame=frame)
    x = loaded.matrices[ARM]
    injected_fit = nn_.fit_predict(x=x, q=q, actual=actual, split=split, seed=SEED)
    control_fit = nn_.fit_predict(
        x=x, q=q, actual=frame["actual"].to_numpy().astype(np.float64), split=split, seed=SEED
    )
    test = split == "test"
    median = nn_.MEDIAN_INDEX
    shift = injected_fit.quantiles[:, median] - control_fit.quantiles[:, median]

    def slope(*, part: np.ndarray, rows: np.ndarray) -> float:
        return float((shift[rows] * part[rows]).sum() / (part[rows] ** 2).sum())

    base = nn_.row_scores(actual=actual, quantiles=q)
    new = nn_.row_scores(actual=actual, quantiles=injected_fit.quantiles)
    real = frame["actual"].to_numpy().astype(np.float64)
    control_gain = nn_.row_scores(actual=real, quantiles=control_fit.quantiles) - nn_.row_scores(
        actual=real, quantiles=q
    )
    clean = test & (effect == 0.0)
    injected = test & (effect > 0.0)
    out = {
        "recovery_total": slope(part=effect, rows=test),
        "recovery_cold": slope(part=cold, rows=test & (cold > 0) & (saturday == 0)),
        "recovery_saturday": slope(part=saturday, rows=test & (saturday > 0) & (cold == 0)),
        "rows_test": int(test.sum()),
        "rows_injected": int(injected.sum()),
        "rows_clean": int(clean.sum()),
        "score_change_clean_rows": float(new[clean].mean() - base[clean].mean()),
        "control_score_change_clean_rows": float(control_gain[clean].mean()),
        "score_change_clean_rows_vs_control": float(
            (
                nn_.row_scores(actual=real, quantiles=injected_fit.quantiles)
                - nn_.row_scores(actual=real, quantiles=control_fit.quantiles)
            )[clean].mean()
        ),
        "score_d_injected_rows": float(base[injected].mean()),
        "score_net_injected_rows": float(new[injected].mean()),
        "epochs_injected": injected_fit.epochs,
        "epochs_control": control_fit.epochs,
    }
    no_harm = out["score_change_clean_rows_vs_control"] < PASS_NO_HARM
    out["passes"] = bool(out["recovery_total"] >= PASS_RECOVERY and no_harm)
    (common.WORK_DIR / "gate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
