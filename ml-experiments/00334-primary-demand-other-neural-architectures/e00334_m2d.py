"""Post hoc: score arm M2D (M2 plus darkness and sun elevation) against M1, M2, and the default.

Tests the explanation that M2 trails M1 because it lacks the two sun-geometry inputs.

    data/ml-experiments/00334-*/run.sh ml-experiments/00334-*/e00334_m2d.py
"""

import e00311_score as base  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl

import e00334_common as common  # isort: skip
import e00334_score as score  # isort: skip


def main() -> None:
    """Join M2D to the scored rows and write its contrasts."""
    rows = score.assemble_rows()
    for arm in ("m1", "m2", "m2d"):
        fits = score.read_fits(arm=arm, cuts=common.SCORED_CUTS, seeds=common.SEEDS).rename(
            {q: f"{arm}_{q}" for q in common.QUANTILES}
        )
        rows = rows.join(fits, on=score.KEYS, how="left", validate="1:1")
    rows = rows.filter(
        pl.col("m1_q50").is_not_null()
        & pl.col("m2_q50").is_not_null()
        & pl.col("m2d_q50").is_not_null()
    )
    site = rows["site"].to_numpy()
    actual = rows["actual"].to_numpy().astype(np.float64)
    th = base.thresholds()
    edge = np.array([th[s] for s in site], dtype=np.float64)
    arms = {
        "default": score.quantile_matrix(rows=rows, prefix="default"),
        **{
            a.upper(): np.sort(score.quantile_matrix(rows=rows, prefix=a), axis=1)
            for a in ("m1", "m2", "m2d")
        },
    }
    m = {a: base.per_row(actual=actual, q=q, edge=edge) for a, q in arms.items()}
    boot = base.Blocks(
        blocks=rows["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy(),
        seed=common.EXPERIMENT_ID + 2,
    )
    cold = rows["cold_winter_night"].to_numpy()
    sunny = rows["sunny_daylight"].to_numpy()
    every = np.ones(rows.height, dtype=bool)
    out: dict = {"n_rows": rows.height, "absolute": {a: float(m[a]["qs"].mean()) for a in arms}}
    for a, b in (("M2D", "default"), ("M2D", "M2"), ("M2D", "M1"), ("M2", "M1")):
        d = m[a]["qs"] - m[b]["qs"]
        out[f"{a}_minus_{b}"] = {
            "all": boot.mean(values=d, mask=every),
            "cold winter nights": boot.mean(values=d, mask=cold),
            "sunny daylight": boot.mean(values=d, mask=sunny),
        }
    common.write_json_atomic(path=common.REPORT_DIR / "report_m2d.json", content=out)
    print(
        out["absolute"],
        {
            k: {g: round(v["est"], 3) for g, v in c.items()}
            for k, c in out.items()
            if k.startswith("M2")
        },
    )


if __name__ == "__main__":
    main()
