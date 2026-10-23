"""Build every arm's forecast for the six Primaries and store it as parquet.

systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00265-*/e00265_run.py
"""

import os
from concurrent.futures import ProcessPoolExecutor

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import e00265_common as common  # noqa: E402
import e00265_widen as widen  # noqa: E402


def run_site(site: str) -> str:
    """Write every arm for one site; return a one-line summary."""
    frames = {name: common.load_reference(site=name) for name in common.SITES}
    resolved = {name: widen.resolved_from_reference(reference=f) for name, f in frames.items()}
    pool = {name: r for name, r in resolved.items() if name != site}
    heavy = {name for name, f in frames.items() if widen.has_heavy_lower_tail(reference=f)}
    pool_light = {name: r for name, r in pool.items() if name not in heavy}
    reference = frames[site]
    on, off = common.SWITCH_ON_SHARE, common.SWITCH_OFF_SHARE
    arms = {"arm1": (reference, None)}
    arms["control"] = widen.widen(reference=reference)
    arms["arm2"] = widen.widen(reference=reference, pool=pool, pool_mode="fixed")
    arms["arm2c"] = widen.widen(reference=reference, pool=pool, pool_mode="asof")
    arms["arm3"] = widen.widen(reference=reference, pool=pool, pool_mode="fixed", switch=(on, off))
    arms["arm3_sens"] = widen.widen(
        reference=reference,
        pool=pool,
        pool_mode="fixed",
        switch=(common.SENSITIVITY_ON_SHARE, common.SENSITIVITY_OFF_SHARE),
    )
    # Post hoc: faster shrinkage (K = 300 rows) and a pool without the heavy-tailed Primaries.
    arms["arm2p"] = widen.widen(
        reference=reference, pool=pool_light, pool_mode="fixed", shrinkage_rows=300.0
    )
    arms["arm3p"] = widen.widen(
        reference=reference,
        pool=pool_light,
        pool_mode="fixed",
        switch=(on, off),
        shrinkage_rows=300.0,
    )
    for name, (frame, log) in arms.items():
        common.write_parquet_atomic(path=common.PREDS_DIR / name / f"{site}.parquet", frame=frame)
        if log is not None:
            common.write_parquet_atomic(
                path=common.WORK_DIR / "logs" / f"{name}_{site}.parquet", frame=log
            )
    return f"{site}: {reference.height:,} rows, arms {sorted(arms)}"


def main() -> None:
    """Run the six sites in parallel workers (4 cores)."""
    with ProcessPoolExecutor(max_workers=4) as pool:
        for line in pool.map(run_site, list(common.SITES)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
