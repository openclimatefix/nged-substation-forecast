"""Step 2 of experiment 00356: Wind farms research default 1 on every daily case.

Reuses experiment 00346's cases and member mixtures: the ERA5-trained XGBoost on the 51 members,
experiment 33's spread factor, and experiment 259's offset table fitted once on the hindcast runs
(init before 2025-06-17). Stores `d1_keys.parquet` and `d1_s<seed>.npy` (cases, 11), seeds 0 to 2.

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_d1.py
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00346_final as f346  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = f346.c303


def main() -> None:
    """Build default 1's quantiles on every daily case, one array per seed."""
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases, mix = f346.load_cases()
    in_hindcast = (cases["power_fcst_init_time"] < f346.HINDCAST_LAST_INIT).to_numpy()
    cases.select(*c303.KEY_COLUMNS).write_parquet(common.DATA_DIR / "d1_keys.parquet")
    for seed in sorted(mix):
        adjusted = c303.apply_spread(cases=cases, mixture=mix[seed])
        table, keys = f346.fit_table(cases=cases, adjusted=adjusted, select=in_hindcast)
        out = ens_final.apply_offsets(cases=cases, quantiles=adjusted, table=table, keys=keys)
        np.save(common.DATA_DIR / f"d1_s{seed}.npy", out.astype(np.float32))
        print(f"seed {seed}: {out.shape}", flush=True)
    print(pl.Series(cases["power_fcst_init_time"]).max())


if __name__ == "__main__":
    main()
