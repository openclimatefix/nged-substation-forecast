"""Gain by temperature decile for the network, F1, and F2, and where each gain sits (scored year).

Rows are the scored-year rows at the five decision substations used by `e00216_score.py`. A row's
decile is its valid time's 24-hour-averaged analysis-frame temperature among the substation's
training-side values (`e00216_temperature.py`). The gain of an arm is the control's row score minus
the arm's. Writes `work/deciles.json`.

    uv run --no-sync python ml-experiments/00216-*/e00216_deciles.py
"""

import json

import e00216_common as common
import e00216_score as score
import e00216_temperature as temperature
import numpy as np
import polars as pl

RATIO: str = "1"
N_RESAMPLES: int = 20_000
OUTER = (1, 10)


def share_with_interval(*, gain: np.ndarray, outer: np.ndarray, months: np.ndarray) -> dict:
    """Return the share of the total gain in the outer deciles and a month-resampled interval."""
    labels, index = np.unique(months, return_inverse=True)
    total = np.bincount(index, weights=gain, minlength=len(labels))
    part = np.bincount(index, weights=gain * outer, minlength=len(labels))
    rng = np.random.default_rng(216)
    draws = rng.integers(len(labels), size=(N_RESAMPLES, len(labels)))
    ratio = part[draws].sum(axis=1) / total[draws].sum(axis=1)
    return {
        "share": float(part.sum() / total.sum()),
        "lower_95": float(np.percentile(ratio, 2.5)),
        "upper_95": float(np.percentile(ratio, 97.5)),
    }


def main() -> None:
    """Compute the per-decile gains and the outer-decile shares."""
    names = ("r0", f"f1_{RATIO}", f"f2_{RATIO}")
    frames = {
        name: score.add_losses(frame=score.read_arm(arm=name, period="scored")) for name in names
    }
    keys = list(score.KEYS)
    base = (
        frames["r0"]
        .filter(pl.col("site").is_in(common.DECISION_SITES))
        .select(*keys, "lead_hours", "hour_band", "qs", "actual")
        .rename({"qs": "qs_r0"})
    )
    for name in names[1:]:
        base = base.join(
            frames[name].select(*keys, pl.col("qs").alias(f"qs_{name}")), on=keys, how="inner"
        )
    network = score.read_network(rows=frames["r0"])
    for seed, frame in network.items():
        scored = score.add_losses(frame=frame)
        base = base.join(
            scored.select(*keys, pl.col("qs").alias(f"qs_n{seed}")), on=keys, how="inner"
        )
    base = base.with_columns(
        qs_n=pl.mean_horizontal([f"qs_n{s}" for s in score.N_SEEDS]),
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )
    base = base.join(temperature.all_frames(), on=["site", "valid_time"], how="inner")
    gains = {
        "n": pl.col("qs_r0") - pl.col("qs_n"),
        "f1": pl.col("qs_r0") - pl.col(f"qs_f1_{RATIO}"),
        "f2": pl.col("qs_r0") - pl.col(f"qs_f2_{RATIO}"),
    }
    base = base.with_columns(**{f"gain_{k}": v for k, v in gains.items()})
    out: dict = {"rows": base.height, "by_decile": {}, "outer_share": {}, "by_hour_band": {}}
    for decile in range(1, 11):
        part = base.filter(pl.col("decile") == decile)
        out["by_decile"][decile] = {
            "rows": part.height,
            "mean_qs_r0": score.mean_of(part["qs_r0"]),
            **{f"gain_{k}": score.mean_of(part[f"gain_{k}"]) for k in gains},
        }
    outer = base["decile"].is_in(list(OUTER)).to_numpy().astype(float)
    months = base["month"].to_numpy()
    out["rows_share_outer"] = float(outer.mean())
    out["r0_score_share_outer"] = float(
        (base["qs_r0"].to_numpy() * outer).sum() / base["qs_r0"].sum()
    )
    for k in gains:
        out["outer_share"][k] = share_with_interval(
            gain=base[f"gain_{k}"].to_numpy(), outer=outer, months=months
        )
    for band in range(4):
        for tail, label in ((10, "hot"), (1, "cold")):
            part = base.filter((pl.col("hour_band") == band) & (pl.col("decile") == tail))
            out["by_hour_band"][f"{label}_band{band}"] = {
                "rows": part.height,
                **{f"gain_{k}": score.mean_of(part[f"gain_{k}"]) for k in gains},
            }
    common.write_json_atomic(path=common.WORK_DIR / "deciles.json", content=out)
    print(json.dumps(out, indent=1))
    base.write_parquet(common.WORK_DIR / "decile_rows.parquet")


if __name__ == "__main__":
    main()
