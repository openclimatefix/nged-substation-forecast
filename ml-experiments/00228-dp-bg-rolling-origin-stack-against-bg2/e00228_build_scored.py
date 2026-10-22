"""Build the scored-year rows (2025-07 to 2026-06) with BG2, climatology, manual heuristic, D, DH.

D is experiment 220's arm A (arm D with experiment 169's bands), DH is its A+H (the network head),
and A+H+S is the full stack. The rows are experiment 220's, joined to the stored forecasts of
experiment 87. Output goes to `work/arms/scored/`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00228-*/e00228_build_scored.py
"""

import numpy as np
import polars as pl

import e00228_common as common  # isort: skip

OUT = common.ARM_DIR / "scored"
OURS: dict[str, str] = {"d": "a", "dh": "a_h", "dhs": "a_hs"}


def main() -> None:
    """Join the stored baselines onto experiment 220's rows and write every arm."""
    rows = pl.read_parquet(common.E220_ARMS / "rows.parquet").with_row_index("row")
    keep = [*common.KEYS, "row"]
    arms = {
        ours: np.load(common.E220_ARMS / f"{theirs}.npy").astype(np.float32)
        for ours, theirs in OURS.items()
    }
    drop = np.zeros(rows.height, dtype=bool)
    stored = {}
    for name, folder in common.STORED_87.items():
        frame = pl.read_parquet(
            common.E87_PREDICTIONS / folder / "predictions.parquet",
            columns=[*common.KEYS, "actual", *common.QUANTILES],
        ).unique(subset=common.KEYS)
        joined = rows.select(keep).join(frame, on=common.KEYS, how="left", validate="1:1")
        joined = joined.sort("row")
        gap = np.abs(joined["actual"].to_numpy() * 100.0 - rows["actual"].to_numpy())
        print(name, "max actual gap (points):", float(np.nanmax(gap)))
        q = joined.select(list(common.QUANTILES)).to_numpy() * 100.0
        missing = ~np.isfinite(q).all(axis=1)
        print(name, "missing rows:", int(missing.sum()))
        drop |= missing
        stored[name] = q.astype(np.float32)
    keep_rows = ~drop
    print("rows kept", int(keep_rows.sum()), "of", rows.height)
    OUT.mkdir(parents=True, exist_ok=True)
    rows.filter(pl.Series(keep_rows)).write_parquet(OUT / "rows.parquet")
    for name, q in {**arms, **stored}.items():
        common.write_npy_atomic(
            path=OUT / f"{name}.npy", array=np.sort(q[keep_rows], axis=1).astype(np.float32)
        )


if __name__ == "__main__":
    main()
