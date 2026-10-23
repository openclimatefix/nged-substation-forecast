"""Build the scored-year rows and arms of experiment 251.

Arm 1 is experiment 220's A+H+S. Arm 2 adds the hand-set 60% clip at D5 as a per-row mean shift (the
clipped arm D of experiment 180 minus its unclipped arm D, which equals experiment 220's arm D; the
shift includes the clip's effect on the filter state and the fit, so it is non-zero at night too),
added to every quantile at D5. Arm 2b adds the same shift to A+S (no network head). Arm 2c (post
hoc) is arm 2b at D5 and arm 1 at the other five substations. Arm 3 is arm 2 on every row without a
reading gap, so it is arm 2 here. Comparison arms are experiment 87's BG2, manual heuristic,
climatology, and weak production-style XGBoost model. Output goes to `work/arms/scored/`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00251-*/e00251_build.py
"""

from typing import Final

import numpy as np
import polars as pl

import e00251_common as common  # isort: skip

OUT = common.ARM_DIR / "scored"
STACK_ARMS: Final = {"d": "a", "dh": "a_h", "a_s": "a_s", "arm1": "a_hs"}


def clip_shift(*, rows: pl.DataFrame) -> np.ndarray:
    """Return the per-row clip shift in points, zero outside D5; assert the joins are 1:1."""
    shift = np.zeros(rows.height, dtype=np.float64)
    site = rows.filter(pl.col("site") == "D5").select("row", "run", "valid_time", "mean_d")
    read = lambda arm: pl.read_parquet(  # noqa: E731
        common.E180_ARMS / arm / "dp_preds_s2" / "d_scored_D5.parquet",
        columns=["run", "valid_time", "mean"],
    ).unique(subset=["run", "valid_time"])
    clipped = read("k60").rename({"mean": "mean_clip"})
    control = read("c").rename({"mean": "mean_ctrl"})
    joined = site.join(clipped, on=["run", "valid_time"], how="left", validate="1:1").join(
        control, on=["run", "valid_time"], how="left", validate="1:1"
    )
    assert joined["mean_clip"].null_count() == 0, "clipped arm D missing rows"
    assert joined["mean_ctrl"].null_count() == 0, "unclipped arm D missing rows"
    gap = np.abs(joined["mean_ctrl"].to_numpy() - joined["mean_d"].to_numpy()).max()
    assert gap < 1e-3, f"unclipped control is not experiment 220's arm D: {gap}"
    shift[joined["row"].to_numpy()] = (joined["mean_clip"] - joined["mean_ctrl"]).to_numpy()
    return shift


def main() -> None:
    """Write every arm and the row table with the sunny flag."""
    scored = pl.read_parquet(common.E228_SCORED / "rows.parquet")
    rows220 = pl.read_parquet(common.E220_ARMS / "rows.parquet").with_row_index("row")
    sun = pl.read_parquet(
        common.E178_ROWS, columns=["site", "power_fcst_init_time", "valid_time", "unit", "sunny"]
    )
    keys = ["site", "power_fcst_init_time", "valid_time"]
    rows = scored.join(sun, on=keys, how="left", validate="1:1")
    assert rows["sunny"].null_count() == 0, "sunny flag missing"
    shift_all = clip_shift(rows=rows220)
    index = rows["row"].to_numpy()
    shift = shift_all[index][:, None]
    arms: dict[str, np.ndarray] = {}
    for name, stem in STACK_ARMS.items():
        arms[name] = np.load(common.E220_ARMS / f"{stem}.npy").astype(np.float32)[index]
    arms["arm2"] = (arms["arm1"] + shift).astype(np.float32)
    arms["arm2b"] = (arms["a_s"] + shift).astype(np.float32)
    d5_rows = (rows["site"] == "D5").to_numpy()[:, None]
    arms["arm2c"] = np.where(d5_rows, arms["arm2b"], arms["arm1"]).astype(np.float32)
    for name in ("bg2", "climatology", "manual_heuristic"):
        arms[name] = np.load(common.E228_SCORED / f"{name}.npy")
    # experiment 228's stored comparison arms share its row order, which `scored` preserves.
    xgb = pl.read_parquet(
        common.E87_PREDICTIONS / common.STORED_87["xgboost"] / "predictions.parquet",
        columns=[*common.KEYS, *common.QUANTILES],
    ).unique(subset=common.KEYS)
    joined = rows.select(*common.KEYS, "row").join(xgb, on=common.KEYS, how="left", validate="1:1")
    q = joined.select(list(common.QUANTILES)).to_numpy() * 100.0
    ok = np.isfinite(q).all(axis=1)
    print("xgboost rows missing:", int((~ok).sum()), "of", rows.height)
    arms["xgboost"] = np.sort(q, axis=1).astype(np.float32)
    keep = ok  # the weak model is scored on rows it covers; every arm is cut to the same rows
    rows = rows.filter(pl.Series(keep))
    OUT.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(OUT / "rows.parquet")
    for name, q_arm in arms.items():
        assert q_arm.shape[0] == keep.shape[0], name
        common.write_npy_atomic(path=OUT / f"{name}.npy", array=np.sort(q_arm[keep], axis=1))
    print("rows", rows.height, "D5 shift mean abs", float(np.abs(shift).mean()))


if __name__ == "__main__":
    main()
